#pragma once

#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/plan.h"

namespace ninfer::ops::detail {

// The E4M3-QK route shares NVFP4-G16 storage and therefore the NVFP4 profile's family
// boundaries: short and batched windows keep the NVFP4 grouped mainloops, which read the
// same planes and carry the multi-batch tile shapes. Only the tiled family changes its K
// operand source, so it takes the mxfp8 split partition instead of the grouped budget.
// Concretely (query width): <= 8 grouped, <= 192 parallel-grouped, wider tiled, so the E4M3
// QK operand serves single-request prefill above 192 while decode and MTP/DFlash verification
// windows keep the NVFP4 FP16 kernels over the identical storage.
Nvfp4KvCausalPlan make_nvfp4_fp8_kv_causal_plan(int heads, int width, int batch,
                                                CausalAttentionExecutionEnvelope envelope,
                                                int multiprocessor_count);
std::size_t nvfp4_fp8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                         CausalAttentionExecutionEnvelope envelope,
                                         int multiprocessor_count);

} // namespace ninfer::ops::detail
