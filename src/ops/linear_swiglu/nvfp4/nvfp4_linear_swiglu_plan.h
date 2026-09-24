#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                       std::int32_t min_tokens,
                                                                       std::int32_t max_tokens);

void nvfp4_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream);
void nvfp4_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void nvfp4_linear_swiglu_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     WorkspaceArena& workspace, cudaStream_t stream);

// True when the route that `nvfp4_linear_swiglu_dispatch` would select is the TMA FusedW4A4 route,
// which is the prefill route the input-RMSNorm fusion replaces. Callers MUST gate on this so the
// fused and unfused paths never disagree about which projection runs.
[[nodiscard]] bool nvfp4_linear_swiglu_tma_fused_route(LinearPolicy policy, std::int32_t tokens);

// The fused producer for the route above: one kernel turns a residual row into this operation's
// NVFP4 activation representation, so the normalized row never reaches memory. `input_scale_divisor`
// is the operand weight's, exactly as the standalone quantizer receives it.
void launch_nvfp4_linear_swiglu_fused_rmsnorm_quantize(const Tensor& residual,
                                                       const Tensor& norm_gain, float eps,
                                                       float input_scale_divisor,
                                                       Nvfp4W4a4Workspace workspace,
                                                       cudaStream_t stream);
void nvfp4_linear_swiglu_fused_rmsnorm_launch(const Tensor& residual, const Tensor& norm_gain,
                                              float eps, const Weight& weight, Tensor& out,
                                              WorkspaceArena& workspace, cudaStream_t stream);

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena& workspace,
                                  cudaStream_t stream);

} // namespace ninfer::ops::detail
