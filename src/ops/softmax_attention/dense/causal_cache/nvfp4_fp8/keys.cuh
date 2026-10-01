#pragma once

#include "ops/kv_cache/nvfp4_group16_codec.cuh"

namespace ninfer::ops::detail {

// K source of the NVFP4-G16/E4M3-QK profile. A staged group is
// `E4M3(E2M1(code) * E4M3(G16 scale))`, so the stored group scale is already inside the
// operand the MMA consumes and the software column scale is exactly one.
struct Nvfp4Fp8KvTiledKeys {
    using Scale      = __half;       // unit software scale word, one per key
    using CacheScale = std::uint8_t; // cache K G16 scale plane element

    // Elements [16*group, 16*group + 16) of one key row, addressed by (page, page_offset).
    // The decoded word is written straight into the swizzled 16-byte staging granule the
    // fp8-K family fills with a cp_async: one G16 group is exactly one granule.
    template <class Geometry>
    __device__ __forceinline__ static void
    stage_group(std::uint8_t* dst, const std::uint8_t* cache_k, const CacheScale* cache_k_scale,
                std::int32_t physical_page, std::int32_t kv_head, std::int32_t page_offset,
                int group) {
        const std::int64_t code_offset = kv_cache_nvfp4_code_index<Geometry>(
            physical_page, kv_head, group * kKVCacheNvfp4Group, page_offset);
        const std::int64_t scale_offset =
            kv_cache_nvfp4_scale_index<Geometry>(physical_page, kv_head, group, page_offset);
        store_vec(dst, kv_cache_nvfp4_dequant_e4m3x16(cache_k + code_offset,
                                                      cache_k_scale[scale_offset]));
    }

    template <class Geometry>
    __device__ __forceinline__ static Scale key_scale(const CacheScale*, std::int32_t, std::int32_t,
                                                      std::int32_t) {
        return __float2half_rn(1.0F);
    }

    // Invisible keys have their codes zeroed, so their unit scale keeps them at zero.
    __device__ __forceinline__ static Scale invisible_key_scale() { return __float2half_rn(1.0F); }
};

} // namespace ninfer::ops::detail
