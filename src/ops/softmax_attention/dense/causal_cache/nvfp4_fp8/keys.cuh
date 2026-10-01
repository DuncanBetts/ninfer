#pragma once

#include "ops/kv_cache/nvfp4_group16_codec.cuh"
#include "ops/softmax_attention/common/causal_tile_io.cuh"

namespace ninfer::ops::detail {

// K source of the NVFP4-G16/E4M3-QK profile. A staged group is
// `E4M3(E2M1(code) * E4M3(G16 scale))`, so the stored group scale is already inside the
// operand the MMA consumes and the software column scale is exactly one.
//
// The cache planes hold neither the operand nor the group scale in the shape the MMA needs, so
// this policy stages the raw code and scale bytes with cp.async and decodes them into the staged
// operand after the tile lands. That keeps the synchronous global read and the odd register
// pressure off the QK critical path, and it leaves the decoded bytes bit-identical because the
// decode reads exactly the bytes the direct path read.
struct Nvfp4Fp8KvTiledKeys {
    using Scale      = __half;       // unit software scale word, one per key
    using CacheScale = std::uint8_t; // cache K G16 scale plane element

    // Raw cache bytes staged per key: 128 code bytes plus one G16 scale byte per group.
    static constexpr int kRawKeyBytes = kKVCacheNvfp4CodeBytes + kKVCacheNvfp4Groups;

    // Four threads share a key: slot `s` owns groups [4s, 4s + 4) end to end, copying two
    // 16-byte code granules plus one 4-byte scale word and later decoding exactly those groups.
    // Keeping a group's copy and its decode on one thread is what lets cp.async.wait_group alone
    // order them, so the decode needs no barrier of its own and the step keeps one barrier.
    // A key outside the visible range is staged as zeros; its decoded operand is then the same
    // zero granule the direct path stored.
    template <class Geometry>
    __device__ __forceinline__ static void
    stage_tile(std::uint8_t* raw, const std::uint8_t* cache_k, const CacheScale* cache_k_scale,
               std::int32_t physical_page, std::int32_t kv_head, std::int32_t page_offset0,
               std::int32_t tile_k0, std::int32_t max_query_abs, int key_rows, int tid,
               int threads) {
        std::uint8_t* raw_scales = raw + key_rows * kKVCacheNvfp4CodeBytes;
        const int slot           = tid & 3;
        const int stride         = threads >> 2;
        for (int key_l = tid >> 2; key_l < key_rows; key_l += stride) {
            const bool visible             = tile_k0 + key_l <= max_query_abs;
            const std::int32_t page_offset = page_offset0 + key_l;
            std::uint8_t* codes            = raw + key_l * kKVCacheNvfp4CodeBytes + slot * 32;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                std::uint8_t* dst = codes + half * 16;
                if (visible) {
                    cp_async<16, Cache::cg>(
                        dst, cache_k + kv_cache_nvfp4_code_index<Geometry>(
                                           physical_page, kv_head, slot * 64 + half * 32,
                                           page_offset));
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
            std::uint8_t* scales = raw_scales + key_l * kKVCacheNvfp4Groups + slot * 4;
            if (visible) {
                cp_async<4>(scales, cache_k_scale + kv_cache_nvfp4_scale_index<Geometry>(
                                                      physical_page, kv_head, slot * 4,
                                                      page_offset));
            } else {
                store_vec(scales, 0u);
            }
        }
    }

    // Decodes the staged raw tile into the granules the QK ldmatrix reads, in element order.
    // A pair of groups decodes from one 16-byte code word, which is exactly the granule the
    // staging copy moved, so the decode reads and writes shared memory in 16-byte words.
    __device__ __forceinline__ static void decode_tile(std::uint8_t* k_e4m3,
                                                       const std::uint8_t* raw, int key_rows,
                                                       int tid, int threads) {
        const std::uint8_t* raw_scales = raw + key_rows * kKVCacheNvfp4CodeBytes;
        const int slot                 = tid & 3;
        const int stride               = threads >> 2;
        for (int key_l = tid >> 2; key_l < key_rows; key_l += stride) {
            const std::uint8_t* codes =
                raw + key_l * kKVCacheNvfp4CodeBytes + slot * 4 * (kKVCacheNvfp4Group / 2);
            const std::uint32_t scale_word =
                load_vec<std::uint32_t>(raw_scales + key_l * kKVCacheNvfp4Groups + slot * 4);
            const std::uint8_t* scale_bytes =
                reinterpret_cast<const std::uint8_t*>(&scale_word);
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                std::uint8_t codes_pair[kKVCacheNvfp4Group];
                store_vec(codes_pair, load_vec<int4>(codes + pair * kKVCacheNvfp4Group));
                const int group = slot * 4 + 2 * pair;
                store_vec(causal_k_granule(k_e4m3, key_l, group),
                          kv_cache_nvfp4_dequant_e4m3x16(codes_pair, scale_bytes[2 * pair]));
                store_vec(causal_k_granule(k_e4m3, key_l, group + 1),
                          kv_cache_nvfp4_dequant_e4m3x16(codes_pair + kKVCacheNvfp4Group / 2,
                                                         scale_bytes[2 * pair + 1]));
            }
        }
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
