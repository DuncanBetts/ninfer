#pragma once

#include "ops/softmax_attention/dense/causal_cache/k8v4/schedule.cuh"

namespace ninfer::ops::detail {

// The E4M3-QK route reuses the k8v4 tiled geometry: the decoded K operand occupies the same
// 256 B/key E4M3 staging the fp8-K family uses while V keeps the NVFP4-G16 code plane, so
// the smem budget and the warp shape are identical (QueryTile 128 / KeyTile 64 / 256 threads
// / 8 warps). V uses the k8v4 NVFP4-G16 values policy.
using Nvfp4Fp8KvTiledInstance = K8V4KvTiledMmaSchedule<kMxfp8TiledQueryRows, 64>;

} // namespace ninfer::ops::detail
