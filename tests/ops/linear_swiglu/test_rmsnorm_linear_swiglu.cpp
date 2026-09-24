// Exactness gate for the composite RMSNorm + gate_up (SwiGLU) projection.
//
// rmsnorm_linear_swiglu() selects its own route and never exposes the choice: on the NVFP4 TMA
// FusedW4A4 prefill route the norm is that route's prologue, and on every other weight format,
// policy or token count it materializes the normalized row and runs the standalone projection. Both
// branches have the same oracle - the standalone Offset RMSNorm followed by the standalone
// linear_swiglu() on the same policy - and every comparison here is on raw output bits.
#include "core/weight.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/rmsnorm.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden       = 5120;
constexpr std::int32_t kGateUpRows   = 34816;
constexpr std::int32_t kIntermediate = kGateUpRows / 2;
constexpr float kEps                 = 1.0e-6F;
constexpr ops::LinearPolicy kA4      = ops::LinearPolicy::AllowA4;

struct Fixture {
    explicit Fixture(const quantized_weight::PackedWeight& packed) : payload(packed.payload.size()) {
        payload.copy_from_host(packed.payload.data(), packed.payload.size());
        weight = packed.device_weight(payload.data());
    }

    GuardedDeviceBuffer payload;
    Weight weight;
};

std::vector<std::uint16_t> download(const GuardedDeviceBuffer& buffer) {
    std::vector<std::uint16_t> bits(buffer.bytes() / sizeof(std::uint16_t));
    if (!bits.empty()) {
        buffer.copy_to_host(bits.data(), bits.size() * sizeof(std::uint16_t));
    }
    return bits;
}

int compare_bits(std::string_view label, const std::vector<std::uint16_t>& composite,
                 const std::vector<std::uint16_t>& reference) {
    if (composite.size() != reference.size()) {
        std::cerr << label << ": size mismatch composite=" << composite.size()
                  << " reference=" << reference.size() << '\n';
        return 1;
    }
    for (std::size_t i = 0; i < composite.size(); ++i) {
        if (composite[i] != reference[i]) {
            std::cerr << label << ": bit mismatch at element " << i << " composite=0x" << std::hex
                      << composite[i] << " reference=0x" << reference[i] << std::dec << '\n';
            return 1;
        }
    }
    return 0;
}

// Deterministic BF16 rows in (-0.5, 0.5): every row has a representable sum of squares, and the
// group absmaxes stay spread across several E4M3 scale buckets.
std::vector<std::uint16_t> host_rows(std::int32_t rows, std::uint32_t salt) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(kHidden) * rows);
    std::uint32_t state = 0x9e3779b9U ^ salt ^ (0x85ebca6bU * static_cast<std::uint32_t>(rows));
    for (auto& value : bits) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        const float scaled = static_cast<float>(state >> 8) / 8388608.0F - 0.5F;
        value              = f32_to_bf16(scaled);
    }
    return bits;
}

std::size_t composite_capacity(ops::LinearPolicy policy, std::int32_t tokens) {
    return std::max(ops::rmsnorm_linear_swiglu_workspace_capacity_bytes(QType::NVFP4, kGateUpRows,
                                                                        kHidden, policy, 1, tokens),
                    std::size_t(256));
}

// The two-kernel reference and the composite, compared bit for bit on the activation they leave for
// the down projection. `zero_row` zeroes the first residual row so the all-zero group path (scale
// byte zero, codes zero) is part of the comparison.
int run_case(Fixture& fixture, std::int32_t tokens, ops::LinearPolicy policy, bool zero_row,
             cudaStream_t stream, bool capture) {
    auto host_residual = host_rows(tokens, 0x1234U + static_cast<std::uint32_t>(tokens));
    const auto host_gain = host_rows(1, 0x5bf03U);
    if (zero_row) {
        for (std::int32_t d = 0; d < kHidden; ++d) { host_residual[d] = 0; }
    }

    GuardedDeviceBuffer residual(host_residual.size() * sizeof(std::uint16_t));
    residual.copy_from_host(host_residual.data(), host_residual.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer gain(host_gain.size() * sizeof(std::uint16_t));
    gain.copy_from_host(host_gain.data(), host_gain.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer normalized(static_cast<std::size_t>(kHidden) * tokens *
                                   sizeof(std::uint16_t));
    GuardedDeviceBuffer reference_output(static_cast<std::size_t>(kIntermediate) * tokens *
                                         sizeof(std::uint16_t));
    GuardedDeviceBuffer composite_output(static_cast<std::size_t>(kIntermediate) * tokens *
                                         sizeof(std::uint16_t));

    const Tensor x(residual.data(), DType::BF16, {kHidden, tokens});
    const Tensor norm_gain(gain.data(), DType::BF16, {kHidden, 1});
    Tensor h(normalized.data(), DType::BF16, {kHidden, tokens});
    Tensor reference(reference_output.data(), DType::BF16, {kIntermediate, tokens});
    Tensor composite(composite_output.data(), DType::BF16, {kIntermediate, tokens});

    int failures = 0;
    WorkspaceArena workspace(composite_capacity(policy, tokens));

    normalized.fill(0xff);
    reference_output.fill(0xff);
    composite_output.fill(0xff);
    ops::rmsnorm(x, norm_gain, kEps, /*unit_offset=*/true, h, stream);
    {
        auto call = workspace.scope();
        ops::linear_swiglu(h, fixture.weight, reference, policy, workspace, stream);
    }

    workspace.reset();
    const auto launch = [&] {
        ops::rmsnorm_linear_swiglu(x, norm_gain, kEps, fixture.weight, composite, policy, workspace,
                                   stream);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (capture) {
        definition.capture(stream, launch);
        graph.instantiate(definition);
        composite_output.fill(0xff);
        cuda_check(cudaDeviceSynchronize(), "finish capture input reset");
        graph.launch(stream);
    } else {
        launch();
    }
    cuda_check(cudaStreamSynchronize(stream), "finish composite case");

    const std::string label = "rmsnorm_linear_swiglu T=" + std::to_string(tokens) +
                              " policy=" + std::to_string(static_cast<int>(policy)) +
                              (zero_row ? " zero-row" : "") + (capture ? " graph" : "");
    failures += compare_bits(label, download(composite_output), download(reference_output));
    // Whichever branch the composite takes, it writes only its destination and its own scratch: the
    // caller's separate normalized buffer stays untouched.
    failures += normalized.verify_guards(label + " caller normalized buffer");
    failures += residual.verify_guards(label + " residual");
    failures += gain.verify_guards(label + " norm gain");
    failures += composite_output.verify_guards(label + " output");
    failures += fixture.payload.verify_guards(label + " weight payload");
    std::cout << (failures == 0 ? "PASS " : "FAIL ") << label << "\n";
    return failures;
}

// The fold is the whole point of the composite, so on the route it covers it must add nothing to the
// projection's own scratch. On the decode GEMV, on a token count with no quantized route at all, and
// under the policies that keep the activation at BF16, the composite has to reserve the normalized
// image as well.
int run_capacity_case() {
    int failures = 0;
    const auto base = [&](ops::LinearPolicy policy, std::int32_t tokens) {
        return ops::linear_swiglu_workspace_capacity_bytes(QType::NVFP4, kGateUpRows, kHidden,
                                                           policy, tokens, tokens);
    };
    const auto composite = [&](ops::LinearPolicy policy, std::int32_t tokens) {
        return ops::rmsnorm_linear_swiglu_workspace_capacity_bytes(QType::NVFP4, kGateUpRows,
                                                                   kHidden, policy, tokens, tokens);
    };
    for (std::int32_t tokens : {256, 512, 1024, 2048}) {
        if (composite(kA4, tokens) != base(kA4, tokens)) {
            std::cerr << "composite capacity is not the base capacity on the folding route at T="
                      << tokens << '\n';
            ++failures;
        }
    }
    for (std::int32_t tokens : {1, 2, 100, 255, 511}) {
        if (composite(kA4, tokens) <= base(kA4, tokens)) {
            std::cerr << "composite capacity omits the normalized image at T=" << tokens << '\n';
            ++failures;
        }
    }
    for (ops::LinearPolicy policy : {ops::LinearPolicy::AllowA8, ops::LinearPolicy::A16Only}) {
        for (std::int32_t tokens : {1, 16}) {
            if (composite(policy, tokens) <= base(policy, tokens)) {
                std::cerr << "composite capacity omits the normalized image under policy "
                          << static_cast<int>(policy) << " at T=" << tokens << '\n';
                ++failures;
            }
        }
    }
    std::cout << (failures == 0 ? "PASS " : "FAIL ") << "rmsnorm_linear_swiglu capacity\n";
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        quantized_weight::PatternedWeightOptions options;
        options.weight_scale_divisor = 0.125F;
        options.input_scale_divisor  = 3.5F;
        const quantized_weight::PackedWeight packed =
            quantized_weight::make_patterned_weight(QType::NVFP4, kGateUpRows, kHidden, 1803U,
                                                    options);
        Fixture fixture(packed);

        int failures = 0;
        failures += run_capacity_case();
        DeviceContext context;
        // The folding route, once under graph capture because the composite allocates nothing and
        // synchronizes nothing.
        failures += run_case(fixture, 256, kA4, /*zero_row=*/true, context.stream, /*capture=*/true);
        failures += run_case(fixture, 256, kA4, /*zero_row=*/false, nullptr, false);
        failures += run_case(fixture, 512, kA4, /*zero_row=*/false, context.stream, /*capture=*/false);
        failures += run_case(fixture, 1024, kA4, /*zero_row=*/true, nullptr, false);
        // Non-folding branches: the decode GEMV, the A16 prefill route below the quantized route's
        // token cutoff, and the policies that never quantize the activation. Each has to reproduce
        // rmsnorm() + linear_swiglu() bit for bit instead of refusing.
        failures += run_case(fixture, 1, kA4, /*zero_row=*/false, context.stream, /*capture=*/true);
        failures += run_case(fixture, 1, kA4, /*zero_row=*/true, nullptr, false);
        failures += run_case(fixture, 100, kA4, /*zero_row=*/false, nullptr, false);
        failures += run_case(fixture, 255, kA4, /*zero_row=*/false, context.stream, false);
        failures += run_case(fixture, 1, ops::LinearPolicy::A16Only, /*zero_row=*/false, nullptr,
                             false);
        failures += run_case(fixture, 16, ops::LinearPolicy::AllowA8, /*zero_row=*/false,
                             context.stream, false);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " rmsnorm_linear_swiglu exactness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "rmsnorm_linear_swiglu test failed: " << error.what() << '\n';
        return 1;
    }
}
