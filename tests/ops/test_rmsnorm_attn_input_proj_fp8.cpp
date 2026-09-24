// Exactness gate for the composite RMSNorm + FP8 attention-input projection.
//
// rmsnorm_attn_input_proj() selects its own route and never exposes the choice: on the T >= 5 FP8
// activation-quantized A8 route the norm is that route's prologue, and on every other parent, policy
// or token count it materializes the normalized row and runs the standalone projection. Both
// branches have the same oracle - the standalone Offset RMSNorm followed by the standalone
// projection - and comparisons here are on raw output bits and, where the two branches share a
// scratch layout, on the whole scratch image.
#include "core/weight.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/rmsnorm.h"

#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/input_projection_test_common.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

constexpr std::int32_t kHidden = 5120;
constexpr std::int32_t kQRows  = 6144;
constexpr std::int32_t kKvRows = 1024;
constexpr std::int32_t kRows   = 14336;
constexpr float kEps           = 1.0e-6F;

std::vector<std::uint8_t> download(const void* device, std::size_t bytes) {
    std::vector<std::uint8_t> host(bytes);
    CUDA_CHECK(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
    return host;
}

std::size_t composite_capacity(ops::LinearPolicy policy, std::int32_t tokens) {
    return std::max<std::size_t>(
        ops::rmsnorm_attn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, policy, tokens, tokens),
        1);
}

std::size_t base_capacity(ops::LinearPolicy policy, std::int32_t tokens) {
    return std::max<std::size_t>(ops::attn_input_proj_workspace_capacity_bytes(
                                     QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, policy, tokens,
                                     tokens),
                                 1);
}

// On a folding token count the composite and the manual producer/consumer pair allocate the same A8
// planes from identically sized scratch, so both the planes and the four outputs are compared whole.
int run_folded_case(DevicePackedWeight& parent, std::int32_t tokens, bool zero_row) {
    DeviceContext device;
    const Weight weight = parent.view();

    auto activation = make_bf16_activation(kHidden, tokens, 71U + static_cast<std::uint32_t>(tokens));
    if (zero_row) {
        // Column zero is the all-zero activation row: its row scale must stay exactly zero instead
        // of becoming a division by zero.
        std::fill(activation.begin(), activation.begin() + kHidden, 0.0F);
    }
    DeviceBuffer residual = to_device(bf16_bits(activation));
    DeviceBuffer gain     = to_device(bf16_bits(make_bf16_activation(kHidden, 1, 977U)));

    GuardedBf16Tensor normalized(kHidden, tokens);
    GuardedBf16Tensor query(kQRows, tokens), composite_query(kQRows, tokens);
    GuardedBf16Tensor gate(kQRows, tokens), composite_gate(kQRows, tokens);
    GuardedBf16Tensor key(kKvRows, tokens), composite_key(kKvRows, tokens);
    GuardedBf16Tensor value(kKvRows, tokens), composite_value(kKvRows, tokens);

    Tensor x(residual.p, DType::BF16, {kHidden, tokens});
    Tensor norm_gain(gain.p, DType::BF16, {kHidden});
    Tensor h  = normalized.tensor();
    Tensor q  = query.tensor();
    Tensor g  = gate.tensor();
    Tensor k  = key.tensor();
    Tensor v  = value.tensor();
    Tensor cq = composite_query.tensor();
    Tensor cg = composite_gate.tensor();
    Tensor ck = composite_key.tensor();
    Tensor cv = composite_value.tensor();

    const std::size_t capacity = composite_capacity(ops::LinearPolicy::AllowA8, tokens);
    GuardedDeviceBuffer scratch(capacity);
    GuardedDeviceBuffer composite_scratch(capacity);
    DeviceArena arena(DeviceSpan{scratch.data(), scratch.bytes()});
    DeviceArena composite_arena(DeviceSpan{composite_scratch.data(), composite_scratch.bytes()});

    // The reference producer/consumer pair, allocated through the same arena call the composite
    // makes so the two scratch images share a layout and can be compared whole.
    ops::rmsnorm(x, norm_gain, kEps, true, h, device.stream);
    {
        const auto scope  = arena.scope();
        const auto planes = ops::detail::allocate_fp8_a8_workspace(arena, tokens, kHidden);
        ops::detail::launch_fp8_a8_quantize(h, weight, planes, device.stream);
        ops::detail::fp8_attn_input_a8_mma_launch(weight, q, g, k, v, planes, tokens, device.stream);
    }
    ops::rmsnorm_attn_input_proj(x, norm_gain, kEps, weight, cq, cg, ck, cv,
                                 ops::LinearPolicy::AllowA8, composite_arena, device.stream);
    cuda_synchronize();

    int failures = 0;
    for (const GuardedBf16Tensor* guarded : {&query, &gate, &key, &value}) {
        failures += guarded->verify_guards("attn input reference");
    }
    for (const GuardedBf16Tensor* guarded : {&composite_query, &composite_gate, &composite_key,
                                             &composite_value}) {
        failures += guarded->verify_guards("attn input composite");
        failures += guarded->verify_fully_written("attn input composite");
    }

    if (download(scratch.data(), scratch.bytes()) !=
        download(composite_scratch.data(), composite_scratch.bytes())) {
        std::cerr << "composite A8 activation planes differ at T=" << tokens << "\n";
        ++failures;
    }

    const GuardedBf16Tensor* references[] = {&query, &gate, &key, &value};
    const GuardedBf16Tensor* composite[]  = {&composite_query, &composite_gate, &composite_key,
                                            &composite_value};
    const char* labels[]                  = {"q", "gate", "k", "v"};
    for (std::size_t index = 0; index < 4; ++index) {
        if (references[index]->bits() != composite[index]->bits()) {
            std::cerr << "composite attn input output " << labels[index] << " differs at T="
                      << tokens << "\n";
            ++failures;
        }
    }

    // Guard the comparison itself against a pair of runs that wrote nothing at all.
    const std::vector<double> composite_values = composite_query.values();
    if (!std::any_of(composite_values.begin(), composite_values.end(),
                     [](double value) { return value != 0.0; })) {
        std::cerr << "composite attn input produced an all-zero q at T=" << tokens << "\n";
        return failures + 1;
    }
    if (zero_row) {
        // The zero row has a zero scale, so its whole output column must be zero on both sides: an
        // empty row must not borrow a neighbour's scale.
        const std::vector<double> reference_values = query.values();
        for (std::int32_t row = 0; row < kQRows; ++row) {
            if (reference_values[row] != 0.0 || composite_values[row] != 0.0) {
                std::cerr << "zero activation row produced a non-zero output at T=" << tokens
                          << "\n";
                ++failures;
                break;
            }
        }
    }
    return failures;
}

// On a non-folding token count or policy the composite has to reproduce rmsnorm() followed by the
// standalone projection exactly, including for the token counts below the A8 quantizer's T >= 5
// cutoff and under a policy that never quantizes the activation.
int run_materialized_case(DevicePackedWeight& parent, std::int32_t tokens, ops::LinearPolicy policy) {
    DeviceContext device;
    const Weight weight = parent.view();

    const std::vector<float> activation =
        make_bf16_activation(kHidden, tokens, 131U + static_cast<std::uint32_t>(tokens));
    DeviceBuffer residual = to_device(bf16_bits(activation));
    DeviceBuffer gain     = to_device(bf16_bits(make_bf16_activation(kHidden, 1, 977U)));

    GuardedBf16Tensor normalized(kHidden, tokens);
    GuardedBf16Tensor query(kQRows, tokens), composite_query(kQRows, tokens);
    GuardedBf16Tensor gate(kQRows, tokens), composite_gate(kQRows, tokens);
    GuardedBf16Tensor key(kKvRows, tokens), composite_key(kKvRows, tokens);
    GuardedBf16Tensor value(kKvRows, tokens), composite_value(kKvRows, tokens);

    Tensor x(residual.p, DType::BF16, {kHidden, tokens});
    Tensor norm_gain(gain.p, DType::BF16, {kHidden});
    Tensor h  = normalized.tensor();
    Tensor q  = query.tensor();
    Tensor g  = gate.tensor();
    Tensor k  = key.tensor();
    Tensor v  = value.tensor();
    Tensor cq = composite_query.tensor();
    Tensor cg = composite_gate.tensor();
    Tensor ck = composite_key.tensor();
    Tensor cv = composite_value.tensor();

    GuardedDeviceBuffer scratch(base_capacity(policy, tokens));
    GuardedDeviceBuffer composite_scratch(composite_capacity(policy, tokens));
    DeviceArena arena(DeviceSpan{scratch.data(), scratch.bytes()});
    DeviceArena composite_arena(DeviceSpan{composite_scratch.data(), composite_scratch.bytes()});

    ops::rmsnorm(x, norm_gain, kEps, true, h, device.stream);
    {
        const auto scope = arena.scope();
        ops::attn_input_proj(h, weight, q, g, k, v, policy, arena, device.stream);
    }
    ops::rmsnorm_attn_input_proj(x, norm_gain, kEps, weight, cq, cg, ck, cv, policy,
                                 composite_arena, device.stream);
    cuda_synchronize();

    int failures = 0;
    for (const GuardedBf16Tensor* guarded : {&composite_query, &composite_gate, &composite_key,
                                             &composite_value}) {
        failures += guarded->verify_guards("attn input composite");
        failures += guarded->verify_fully_written("attn input composite");
    }

    const GuardedBf16Tensor* references[] = {&query, &gate, &key, &value};
    const GuardedBf16Tensor* composite[]  = {&composite_query, &composite_gate, &composite_key,
                                            &composite_value};
    const char* labels[]                  = {"q", "gate", "k", "v"};
    for (std::size_t index = 0; index < 4; ++index) {
        if (references[index]->bits() != composite[index]->bits()) {
            std::cerr << "composite attn input output " << labels[index]
                      << " differs from rmsnorm + projection at T=" << tokens << " policy="
                      << static_cast<int>(policy) << "\n";
            ++failures;
        }
    }
    const std::vector<double> composite_values = composite_query.values();
    if (!std::any_of(composite_values.begin(), composite_values.end(),
                     [](double value) { return value != 0.0; })) {
        std::cerr << "composite attn input produced an all-zero q at T=" << tokens << "\n";
        return failures + 1;
    }
    return failures;
}

// The composite allocates nothing outside its arena and synchronizes nothing, so both branches have
// to survive graph capture and still write every element on replay.
int run_capture_case(DevicePackedWeight& parent, std::int32_t tokens) {
    DeviceContext device;
    const Weight weight = parent.view();
    DeviceBuffer residual = to_device(bf16_bits(make_bf16_activation(kHidden, tokens, 419U)));
    DeviceBuffer gain     = to_device(bf16_bits(make_bf16_activation(kHidden, 1, 977U)));

    GuardedBf16Tensor normalized(kHidden, tokens);
    GuardedBf16Tensor query(kQRows, tokens), composite_query(kQRows, tokens);
    GuardedBf16Tensor gate(kQRows, tokens), composite_gate(kQRows, tokens);
    GuardedBf16Tensor key(kKvRows, tokens), composite_key(kKvRows, tokens);
    GuardedBf16Tensor value(kKvRows, tokens), composite_value(kKvRows, tokens);

    Tensor x(residual.p, DType::BF16, {kHidden, tokens});
    Tensor norm_gain(gain.p, DType::BF16, {kHidden});
    Tensor h  = normalized.tensor();
    Tensor q  = query.tensor();
    Tensor g  = gate.tensor();
    Tensor k  = key.tensor();
    Tensor v  = value.tensor();
    Tensor cq = composite_query.tensor();
    Tensor cg = composite_gate.tensor();
    Tensor ck = composite_key.tensor();
    Tensor cv = composite_value.tensor();

    GuardedDeviceBuffer scratch(base_capacity(ops::LinearPolicy::AllowA8, tokens));
    GuardedDeviceBuffer composite_scratch(composite_capacity(ops::LinearPolicy::AllowA8, tokens));
    DeviceArena arena(DeviceSpan{scratch.data(), scratch.bytes()});
    DeviceArena composite_arena(DeviceSpan{composite_scratch.data(), composite_scratch.bytes()});

    ops::rmsnorm(x, norm_gain, kEps, true, h, device.stream);
    ops::attn_input_proj(h, weight, q, g, k, v, ops::LinearPolicy::AllowA8, arena, device.stream);
    cuda_synchronize();

    const auto composite_call = [&] {
        ops::rmsnorm_attn_input_proj(x, norm_gain, kEps, weight, cq, cg, ck, cv,
                                     ops::LinearPolicy::AllowA8, composite_arena, device.stream);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    int failures = 0;
    for (int phase = 0; phase < 2; ++phase) {
        if (phase) {
            for (GuardedBf16Tensor* out :
                 {&composite_query, &composite_gate, &composite_key, &composite_value})
                out->repaint(device.stream);
            graph.launch(device.stream);
        } else {
            definition.capture(device.stream, composite_call);
            graph.instantiate(definition);
            composite_call();
        }
        cuda_synchronize(device.stream);
    }
    for (const GuardedBf16Tensor* guarded : {&composite_query, &composite_gate, &composite_key,
                                             &composite_value}) {
        failures += guarded->verify_guards("composite attn input replay");
        failures += guarded->verify_fully_written("composite attn input replay");
    }
    if (query.bits() != composite_query.bits() || gate.bits() != composite_gate.bits() ||
        key.bits() != composite_key.bits() || value.bits() != composite_value.bits()) {
        std::cerr << "composite attn input differs from rmsnorm + projection after graph replay at T="
                  << tokens << "\n";
        ++failures;
    }
    return failures;
}

// The fold is the whole point of the composite, so on a folding token count it must add nothing to
// the projection's own scratch. Below the cutoff, and under a policy that keeps the activation at
// BF16, the composite has to reserve the normalized image as well.
int run_capacity_case() {
    int failures = 0;
    for (std::int32_t tokens : {5, 8, 16, 128, 2048}) {
        if (ops::rmsnorm_attn_input_proj_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::AllowA8, tokens,
                tokens) != ops::attn_input_proj_workspace_capacity_bytes(
                               QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden,
                               ops::LinearPolicy::AllowA8, tokens, tokens)) {
            std::cerr << "composite capacity is not the base capacity on the folding route at T="
                      << tokens << "\n";
            ++failures;
        }
    }
    for (std::int32_t tokens : {1, 2, 3, 4}) {
        const std::size_t composite = ops::rmsnorm_attn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::AllowA8, tokens, tokens);
        const std::size_t base      = ops::attn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::AllowA8, tokens, tokens);
        if (composite <= base) {
            std::cerr << "composite capacity omits the normalized image at T=" << tokens << "\n";
            ++failures;
        }
    }
    for (std::int32_t tokens : {1, 5, 1024}) {
        const std::size_t composite = ops::rmsnorm_attn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::A16Only, tokens, tokens);
        const std::size_t base      = ops::attn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::A16Only, tokens, tokens);
        if (composite <= base) {
            std::cerr << "composite capacity omits the normalized image under A16Only at T="
                      << tokens << "\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    DevicePackedWeight parent(quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16,
                                                                     kRows, kHidden, 349U));
    int failures                                  = run_capacity_case();
    for (std::int32_t tokens : {5, 6, 7, 8, 16, 33, 64, 128, 256, 1024}) {
        failures += run_folded_case(parent, tokens, false);
    }
    failures += run_folded_case(parent, 8, true);
    failures += run_folded_case(parent, 1024, true);
    for (std::int32_t tokens : {1, 2, 3, 4}) {
        failures += run_materialized_case(parent, tokens, ops::LinearPolicy::AllowA8);
    }
    for (std::int32_t tokens : {1, 8, 64}) {
        failures += run_materialized_case(parent, tokens, ops::LinearPolicy::A16Only);
    }
    for (std::int32_t tokens : {3, 8}) {
        failures += run_capture_case(parent, tokens);
    }
    if (failures == 0) {
        std::cout << "rmsnorm attention input projection FP8: exact on both routes\n";
    }
    return failures == 0 ? 0 : 1;
}
