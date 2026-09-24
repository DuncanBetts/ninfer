// ninfer::ops - NVFP4 gate_up (SwiGLU) projection fused with the Offset RMSNorm that feeds it.
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Geometry                = Nvfp4N34816K5120;
constexpr int kBlock          = 256;
constexpr int kPairsPerRow    = Geometry::kInputRows / 2;
constexpr int kPairsPerThread = kPairsPerRow / kBlock;
static_assert(kPairsPerThread == 10);

// One CTA owns one token and produces its NVFP4 activation representation without the normalized
// row ever reaching memory. The norm is `rmsnorm_cta_row_bf16x2`, the same row body the D=5120
// Offset RMSNorm launcher runs, so the staged image is bit-identical to what that kernel writes to
// memory; the group scales and codes are the standalone NVFP4 quantizer's `quantize_nvfp4_k16` on
// that same image.
//
// The pair ownership of the two halves does not coincide: an NVFP4 group is 16 contiguous elements
// (8 pairs) while a thread's pairs are strided by kBlock. The row is therefore staged in shared
// memory, which is exactly the image the standalone quantizer would have re-read from the
// intermediate allocation.
template <RmsEpilogue Epilogue>
__launch_bounds__(kBlock) __global__
    void rmsnorm_nvfp4_quantize_kernel(const __nv_bfloat162* __restrict__ residual,
                                       const __nv_bfloat162* __restrict__ gain,
                                       std::uint8_t* __restrict__ codes,
                                       std::uint8_t* __restrict__ scales, std::int32_t tokens,
                                       float eps, float input_scale_divisor) {
    constexpr int kGroupsPerRow = Geometry::kGroupsPerRow;
    const int token             = static_cast<int>(blockIdx.x);
    const int thread            = static_cast<int>(threadIdx.x);
    if (token >= tokens) {
        // The tiled scale plane is addressed in whole tiles, so the launch covers the last tile's
        // padding. A padded token owns no input and no code byte, only the scale the consumer's
        // tile reads; zero it so it is defined rather than whatever the arena last held.
        for (int group = thread; group < kGroupsPerRow; group += kBlock) {
            scales[nvfp4_tiled_scale_offset<Geometry>(token, group)] = 0;
        }
        return;
    }

    const std::int64_t row_base = static_cast<std::int64_t>(token) * kPairsPerRow;
    __shared__ alignas(16) __nv_bfloat162 staged[kPairsPerRow];
    rmsnorm_cta_row_bf16x2<Epilogue, kBlock, kPairsPerThread, true, Geometry::kInputRows>(
        residual + row_base, gain, nullptr, staged, Geometry::kInputRows, eps);
    __syncthreads();

    const auto* staged_bf16 = reinterpret_cast<const __nv_bfloat16*>(staged);
    for (int group = thread; group < kGroupsPerRow; group += kBlock) {
        const Nvfp4QuantizedK16 quantized =
            quantize_nvfp4_k16(staged_bf16 + group * 16, input_scale_divisor);
        auto* code_destination =
            codes + static_cast<std::int64_t>(token) * Geometry::kCodeBytesPerRow + group * 8;
        store_vec(code_destination, make_uint2(quantized.codes_lo, quantized.codes_hi));
        scales[nvfp4_tiled_scale_offset<Geometry>(token, group)] = quantized.scale;
    }
}

} // namespace

void launch_nvfp4_linear_swiglu_fused_rmsnorm_quantize(const Tensor& residual,
                                                       const Tensor& norm_gain, float eps,
                                                       float input_scale_divisor,
                                                       Nvfp4W4a4Workspace workspace,
                                                       cudaStream_t stream) {
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("nvfp4 fused RMSNorm requires caller workspace");
    }
    rmsnorm_nvfp4_quantize_kernel<RmsEpilogue::Offset>
        <<<nvfp4_w4a4_padded_tokens(residual.ne[1]), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(residual.data),
            static_cast<const __nv_bfloat162*>(norm_gain.data), workspace.codes, workspace.scales,
            residual.ne[1], eps, input_scale_divisor);
    CUDA_CHECK(cudaGetLastError());
}

void nvfp4_linear_swiglu_fused_rmsnorm_launch(const Tensor& residual, const Tensor& norm_gain,
                                              float eps, const Weight& weight, Tensor& out,
                                              WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens        = residual.ne[1];
    auto scope                       = workspace.scope();
    const Nvfp4W4a4Workspace scratch =
        allocate_nvfp4_w4a4_workspace(workspace, tokens, Geometry::kInputRows);
    launch_nvfp4_linear_swiglu_fused_rmsnorm_quantize(residual, norm_gain, eps,
                                                      weight.input_scale_divisor, scratch, stream);
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    launch_nvfp4_linear_swiglu_w4a4_tma(
        scratch.codes, scratch.scales, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        tokens, alpha, stream);
}

} // namespace ninfer::ops::detail
