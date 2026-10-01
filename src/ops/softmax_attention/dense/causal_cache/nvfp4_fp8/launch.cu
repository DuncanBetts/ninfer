#include "ops/softmax_attention/dense/causal_cache/nvfp4_fp8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4_fp8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4_fp8/tiled_launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/launch.h"
#include "ops/kv_cache/append/launch.h"

namespace ninfer::ops::detail {

void nvfp4_fp8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& positions, const Tensor& valid, const Tensor& rows,
                                   float scale, PagedKVBatchLayerView cache,
                                   CausalAttentionExecutionEnvelope envelope,
                                   WorkspaceArena& workspace, Tensor& out,
                                   DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto plan           = make_nvfp4_fp8_kv_causal_plan(
        q.ne[1], q.ne[2], q.ne[3], envelope, execution.multiprocessor_count);
    // The mxfp8 tiled producer carries one query row set (its partials and merge are
    // single-request), so batched and short windows stay on the NVFP4 families, which read
    // the same storage and already own the multi-batch mainloops.
    if (plan.family != Nvfp4KvFamily::Tiled) {
        nvfp4_kv_append_attention(q, k, v, positions, valid, rows, scale, cache, envelope,
                                  workspace, out, execution);
        return;
    }
    kv_cache_append_batch_launch(k, v, positions, valid, rows, cache, stream);
    const auto p = make_causal_operands(q, positions, out, scale, envelope.max_visible_keys);
    const auto view =
        make_quantized_causal_cache_view<Nvfp4KvCacheView<false>>(cache, &valid, &rows);
    nvfp4_fp8_kv_tiled_attention(p, view, plan.partition, workspace, stream);
}

void nvfp4_fp8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                                   const PagedKVLayerView& cache,
                                   CausalAttentionExecutionEnvelope envelope,
                                   WorkspaceArena& workspace, Tensor& out,
                                   DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto plan           = make_nvfp4_fp8_kv_causal_plan(
        q.ne[1], q.ne[2], 1, envelope, execution.multiprocessor_count);
    if (plan.family != Nvfp4KvFamily::Tiled) {
        nvfp4_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out,
                                  execution);
        return;
    }
    const auto view = single_row_paged_kv_batch_view(cache);
    nvfp4_fp8_kv_tiled_attention(
        make_causal_operands(q, positions, out, scale, envelope.max_visible_keys),
        make_quantized_causal_cache_view<Nvfp4KvCacheView<false>>(view), plan.partition, workspace,
        stream);
}

} // namespace ninfer::ops::detail
