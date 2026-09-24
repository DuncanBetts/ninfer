#pragma once

// ninfer::ops - FP8 A8 per-token row quantization, shared by the standalone quantizer and the
// entries that produce the activation row inside the same kernel.

#include "ops/common/warp.cuh"

#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Completes one token's A8 activation quantization from its row already promoted to FP32 pairs.
// `values` holds the caller's pair ownership (thread + item * blockDim.x) in registers,
// `code_pairs` receives one FP8x2 per input pair in the same order, and `scale_out[0]` receives the
// row scale. `warp_maxima` (blockDim.x/32 entries) and `token_scale` are scratch the caller places
// in shared memory. The scale is the row absmax divided by the E4M3 finite maximum, or zero for an
// all-zero row, and the inverse is guarded so a zero scale quantizes to zero instead of NaN.
template <int PairsPerThread>
__device__ __forceinline__ void fp8_a8_quantize_values(float2 (&values)[PairsPerThread],
                                                       std::uint16_t* code_pairs,
                                                       float* warp_maxima, float& token_scale,
                                                       float* scale_out) {
    const int tid   = static_cast<int>(threadIdx.x);
    const int lane  = tid & (kWarpSize - 1);
    const int warp  = tid / kWarpSize;
    const int warps = static_cast<int>(blockDim.x) / kWarpSize;

    float maximum = 0.0F;
#pragma unroll
    for (int item = 0; item < PairsPerThread; ++item) {
        maximum = fmaxf(maximum, fabsf(values[item].x));
        maximum = fmaxf(maximum, fabsf(values[item].y));
    }
    maximum = warp_max(maximum);
    if (lane == 0) { warp_maxima[warp] = maximum; }
    __syncthreads();
    if (warp == 0) {
        maximum = lane < warps ? warp_maxima[lane] : 0.0F;
        maximum = warp_max(maximum);
        if (lane == 0) { token_scale = maximum > 0.0F ? maximum / 448.0F : 0.0F; }
    }
    __syncthreads();

    const float scale   = token_scale;
    const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
#pragma unroll
    for (int item = 0; item < PairsPerThread; ++item) {
        const int pair      = tid + item * static_cast<int>(blockDim.x);
        const float2 scaled = make_float2(values[item].x * inverse, values[item].y * inverse);
        code_pairs[pair]    = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
    }
    if (tid == 0) { scale_out[0] = scale; }
}

} // namespace ninfer::ops::detail
