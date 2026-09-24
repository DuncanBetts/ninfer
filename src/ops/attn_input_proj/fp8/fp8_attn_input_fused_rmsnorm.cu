// ninfer::ops - FP8 attention-input projection fused with the Offset RMSNorm that feeds it.
#include "core/weight.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/linear/fp8/fp8_a8_row_quantize.cuh"
#include "ops/linear/fp8/fp8_config.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// One CTA owns one token and produces the A8 activation representation of a row that never reaches
// memory. The sum-of-squares reduction, the pair ownership, the rsqrtf form and the BF16 rounding
// are the D=5120 Offset RMSNorm launcher's; the row scale and the codes are the A8 quantizer's on
// the same pairs, so both halves are bit-identical to the two standalone kernels they replace.
//
// The normalized image is the activation-quantization semantic boundary: it is rounded to BF16
// before the absmax scale sees it, but it stays in registers, which is the whole saving.
template <class ActivationGeometry, RmsEpilogue Epilogue, int Threads = 256>
__global__ __launch_bounds__(Threads,
                             2) void fp8_attn_input_fused_rmsnorm_quantize_kernel(
    const __nv_bfloat16* __restrict__ residual, const __nv_bfloat16* __restrict__ gain, float eps,
    std::uint8_t* __restrict__ codes, float* __restrict__ scales) {
    static_assert((ActivationGeometry::kInputRows % (Threads * 2)) == 0);
    constexpr int pairs_per_thread = ActivationGeometry::kInputRows / (Threads * 2);
    constexpr int warps            = Threads / kWarpSize;
    __shared__ float warp_sums[warps];
    __shared__ float inv_shared;
    __shared__ float warp_maxima[warps];
    __shared__ float token_scale;

    const int token = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    const auto* residual_pairs = reinterpret_cast<const std::uint32_t*>(
        residual + static_cast<std::int64_t>(token) * ActivationGeometry::kInputRows);
    auto* code_pairs = reinterpret_cast<std::uint16_t*>(
        codes + static_cast<std::int64_t>(token) * ActivationGeometry::kInputRows);
    const auto* gain_pairs = reinterpret_cast<const std::uint32_t*>(gain);

    float2 values[pairs_per_thread];
    float sum = 0.0f;
#pragma unroll
    for (int item = 0; item < pairs_per_thread; ++item) {
        const float2 xf = bf16x2_bits_to_float2(residual_pairs[tid + item * Threads]);
        values[item]    = xf;
        sum += xf.x * xf.x + xf.y * xf.y;
    }

    const float block_sum = block_reduce_sum<Threads>(sum, warp_sums);
    if (tid == 0) {
        inv_shared = rsqrtf(block_sum / static_cast<float>(ActivationGeometry::kInputRows) + eps);
    }
    __syncthreads();
    const float inv = inv_shared;

#pragma unroll
    for (int item = 0; item < pairs_per_thread; ++item) {
        const float2 xf            = values[item];
        const float2 wf            = bf16x2_bits_to_float2(gain_pairs[tid + item * Threads]);
        const __nv_bfloat162 image = __floats2bfloat162_rn(
            rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, 0.0f),
            rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, 0.0f));
        values[item] = bf16x2_to_float2(image);
    }
    fp8_a8_quantize_values(values, code_pairs, warp_maxima, token_scale, scales + token);
}

} // namespace

void launch_fp8_attn_input_fused_rmsnorm_quantize(const Tensor& residual,
                                                  const Tensor& norm_weight, float eps,
                                                  Fp8A8Workspace workspace, cudaStream_t stream) {
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("fp8 fused RMSNorm requires caller workspace");
    }
    constexpr int kThreads = 256;
    fp8_attn_input_fused_rmsnorm_quantize_kernel<Fp8Activation5120Geometry, RmsEpilogue::Offset,
                                                 kThreads>
        <<<residual.ne[1], kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(residual.data),
            static_cast<const __nv_bfloat16*>(norm_weight.data), eps, workspace.codes,
            workspace.scales);
    CUDA_CHECK(cudaGetLastError());
}

void fp8_attn_input_fused_rmsnorm_launch(const Tensor& residual, const Tensor& norm_weight,
                                         float eps, const Weight& weight, Tensor& q, Tensor& gate,
                                         Tensor& k, Tensor& v, Fp8A8Workspace workspace,
                                         cudaStream_t stream) {
    launch_fp8_attn_input_fused_rmsnorm_quantize(residual, norm_weight, eps, workspace, stream);
    fp8_attn_input_a8_mma_launch(weight, q, gate, k, v, workspace, residual.ne[1], stream);
}

} // namespace ninfer::ops::detail
