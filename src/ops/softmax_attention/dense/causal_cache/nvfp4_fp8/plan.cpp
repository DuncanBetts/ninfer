#include "ops/softmax_attention/dense/causal_cache/nvfp4_fp8/plan.h"
#include "ops/softmax_attention/common/mxfp8_tiled_plan.h"
#include <algorithm>

namespace ninfer::ops::detail {

Nvfp4KvCausalPlan make_nvfp4_fp8_kv_causal_plan(int heads, int width, int batch,
                                                CausalAttentionExecutionEnvelope envelope,
                                                int multiprocessor_count) {
    // Family boundaries are the NVFP4 profile's, unchanged: width <= 8 grouped, width <= 192
    // parallel-grouped, wider tiled. Only the tiled family runs the E4M3-QK operand route --
    // the mxfp8 tiled producer is single-request, so batched windows and widths <= 192
    // (decode, MTP/DFlash verification) keep the NVFP4 FP16 kernels over the same storage.
    // The tiled family takes the mxfp8 split partition its launcher requires.
    Nvfp4KvCausalPlan plan =
        make_nvfp4_kv_causal_plan(heads, width, batch, envelope, multiprocessor_count);
    if (plan.family == Nvfp4KvFamily::Tiled)
        plan.partition = mxfp8_tiled_partition(heads, width, envelope.max_visible_keys,
                                               multiprocessor_count);
    return plan;
}

std::size_t nvfp4_fp8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                         CausalAttentionExecutionEnvelope envelope,
                                         int multiprocessor_count) {
    std::size_t maximum = 0;
    // Family selection is monotone in query width, so the same walk that accumulates the
    // grouped/parallel-grouped partial storage finds the first tiled width.
    int width = min_width;
    for (; width <= max_width; ++width) {
        const auto plan =
            make_nvfp4_fp8_kv_causal_plan(heads, width, batch, envelope, multiprocessor_count);
        if (plan.family == Nvfp4KvFamily::Tiled) break;
        const int splits = plan.partition.capacity;
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, splits, batch);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    if (width <= max_width)
        maximum = std::max(maximum, mxfp8_tiled_workspace_bytes(heads, width, max_width,
                                                                envelope.max_visible_keys,
                                                                multiprocessor_count));
    return maximum;
}

} // namespace ninfer::ops::detail
