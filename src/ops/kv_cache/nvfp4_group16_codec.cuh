#pragma once

// D256 KV-cache NVFP4 codec. Unlike weight artifacts, cache rows have no matrix-level divisor:
// every represented value is exactly E2M1(code) * E4M3(scale) for its contiguous G16 group.

#include "ops/kernel/paged_kv_address.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKVCacheNvfp4HeadDim        = 256;
inline constexpr int kKVCacheNvfp4Group          = 16;
inline constexpr int kKVCacheNvfp4Groups         = 16;
inline constexpr int kKVCacheNvfp4CodeBytes      = 128;
inline constexpr float kKVCacheNvfp4MaxFinite    = 6.0F;
inline constexpr float kKVCacheNvfp4ScaleMinimum = 0x1p-9F;
inline constexpr float kKVCacheNvfp4ScaleMaximum = 448.0F;

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_nvfp4_code_index(int physical_page, int kv_head,
                                                                  int d, int page_offset) {
    return paged_kv_element_offset<kKVCacheNvfp4CodeBytes, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, d >> 1);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_nvfp4_scale_index(int physical_page, int kv_head,
                                                                   int group, int page_offset) {
    return paged_kv_element_offset<kKVCacheNvfp4Groups, Geometry::KVHeads>(physical_page, kv_head,
                                                                           page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_nvfp4_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kKVCacheNvfp4HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

struct KVCacheNvfp4QuantizedGroup16 {
    std::uint32_t codes_lo = 0;
    std::uint32_t codes_hi = 0;
    std::uint8_t scale     = 0;
};

__device__ __forceinline__ KVCacheNvfp4QuantizedGroup16
kv_cache_nvfp4_quantize_group16(const float* source) {
    float2 values[8];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = make_float2(source[2 * pair], source[2 * pair + 1]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    KVCacheNvfp4QuantizedGroup16 result{};
    if (max_abs == 0.0F) return result;

    const float raw_scale = __fdiv_rn(max_abs, kKVCacheNvfp4MaxFinite);
    const float bounded =
        fminf(kKVCacheNvfp4ScaleMaximum, fmaxf(kKVCacheNvfp4ScaleMinimum, raw_scale));
    result.scale                  = __nv_cvt_float_to_fp8(bounded, __NV_SATFINITE, __NV_E4M3);
    const float represented_scale = detail::decode_nvfp4_e4m3(result.scale);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair].x = __fdiv_rn(values[pair].x, represented_scale);
        values[pair].y = __fdiv_rn(values[pair].y, represented_scale);
    }
    detail::pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
    return result;
}

__device__ __forceinline__ int4 kv_cache_nvfp4_dequant_f16x8(const std::uint8_t* codes,
                                                             std::uint8_t scale_code) {
    // Every finite E2M1 value times a legal nonnegative E4M3 cache scale is exactly representable
    // in FP16 (at most four product fraction bits and magnitude <= 2688). Half2 multiplication is
    // therefore the exact FP16 expansion boundary, not an additional approximation.
    const std::uint32_t packed = load_vec<std::uint32_t>(codes);
    const std::uint8_t* bytes  = reinterpret_cast<const std::uint8_t*>(&packed);
    __nv_fp8_e4m3 encoded_scale;
    encoded_scale.__x    = scale_code;
    const __half scale   = static_cast<__half>(encoded_scale);
    const __half2 scale2 = __halves2half2(scale, scale);
    unsigned half_bits[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        __nv_fp4x2_e2m1 encoded;
        encoded.__x         = bytes[pair];
        const __half2 value = __hmul2(static_cast<__half2>(encoded), scale2);
        half_bits[pair]     = *reinterpret_cast<const unsigned*>(&value);
    }
    return make_int4(static_cast<int>(half_bits[0]), static_cast<int>(half_bits[1]),
                     static_cast<int>(half_bits[2]), static_cast<int>(half_bits[3]));
}

// E4M3-image of one G16 group, in element order: byte j is E4M3(E2M1(code j) * E4M3(scale)).
//
// Rounding: the half2 product is exact (as in the FP16 decoders above), so the single
// approximation is the FP16 -> E4M3 step, which is round-to-nearest-even with saturation.
// E4M3's exponent range is therefore the binding constraint, not FP16's:
//   * products below 2^-10 (half the smallest E4M3 subnormal, 2^-9) round to zero; the
//     smallest representable value is 0.5 * 2^-9 = 2^-10 and it ties to zero,
//   * products above 448 saturate, so a group scale at kKVCacheNvfp4ScaleMaximum times a
//     code above 1.0 clamps where the FP16 decoders would have kept the larger value.
// Everything between those bounds is E4M3-representable or rounded once, to at most the
// ~4 significand bits the stored value already carries.
__device__ __forceinline__ int4 kv_cache_nvfp4_dequant_e4m3x16(const std::uint8_t* codes,
                                                               std::uint8_t scale_code) {
    const int2 packed         = load_vec<int2>(codes);
    const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(&packed);
    __nv_fp8_e4m3 encoded_scale;
    encoded_scale.__x    = scale_code;
    const __half scale   = static_cast<__half>(encoded_scale);
    const __half2 scale2 = __halves2half2(scale, scale);
    unsigned words[4];
#pragma unroll
    for (int quad = 0; quad < 4; ++quad) {
        std::uint16_t halves[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            __nv_fp4x2_e2m1 encoded;
            encoded.__x         = bytes[2 * quad + i];
            const __half2 value = __hmul2(static_cast<__half2>(encoded), scale2);
            halves[i] = __nv_cvt_halfraw2_to_fp8x2(value, __NV_SATFINITE, __NV_E4M3);
        }
        words[quad] = static_cast<unsigned>(halves[0]) | (static_cast<unsigned>(halves[1]) << 16);
    }
    return make_int4(static_cast<int>(words[0]), static_cast<int>(words[1]),
                     static_cast<int>(words[2]), static_cast<int>(words[3]));
}

struct KVCacheNvfp4DequantizedF16x16 {
    int4 lo;
    int4 hi;
};

__device__ __forceinline__ KVCacheNvfp4DequantizedF16x16
kv_cache_nvfp4_dequant_f16x16(const std::uint8_t* codes, std::uint8_t scale_code) {
    const int2 packed         = load_vec<int2>(codes);
    const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(&packed);
    __nv_fp8_e4m3 encoded_scale;
    encoded_scale.__x    = scale_code;
    const __half scale   = static_cast<__half>(encoded_scale);
    const __half2 scale2 = __halves2half2(scale, scale);
    unsigned half_bits[8];
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        __nv_fp4x2_e2m1 encoded;
        encoded.__x         = bytes[pair];
        const __half2 value = __hmul2(static_cast<__half2>(encoded), scale2);
        half_bits[pair]     = *reinterpret_cast<const unsigned*>(&value);
    }
    return {
        make_int4(static_cast<int>(half_bits[0]), static_cast<int>(half_bits[1]),
                  static_cast<int>(half_bits[2]), static_cast<int>(half_bits[3])),
        make_int4(static_cast<int>(half_bits[4]), static_cast<int>(half_bits[5]),
                  static_cast<int>(half_bits[6]), static_cast<int>(half_bits[7])),
    };
}

} // namespace ninfer::ops
