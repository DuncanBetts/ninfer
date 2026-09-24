#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kFp8AttnInputLastSimtT     = 5;
inline constexpr int kFp8AttnInputLastSmallMmaT = 33;

// Whether the FP8 attention-input route quantizes its activation row with the A8 quantizer.
// A16 owns T<=4; A8 takes over at T>=5. The fused RMSNorm entry is eligible under exactly this
// predicate, so the route and the fusion cannot drift apart.
[[nodiscard]] constexpr bool fp8_attn_input_a8_route(std::int32_t tokens) {
    return tokens >= 5;
}

[[nodiscard]] std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

void fp8_attn_input_a16_small_mma_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void fp8_attn_input_a16_gemm_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, cudaStream_t stream);

// Whether the attention-input entry can take the Offset RMSNorm that produces its activation as a
// prologue: the T >= 5 A8 quantizer fuses; the T <= 4 A16 routes do not. Both halves read
// fp8_attn_input_a8_route, so the eligibility predicate and the route cannot drift apart.
[[nodiscard]] bool fp8_attn_input_fused_rmsnorm_route(LinearPolicy policy, std::int32_t tokens);

void fp8_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, Fp8A8Workspace workspace, cudaStream_t stream);

// The A8 matrix-product half of the route, exposed so a fused producer can feed it without
// quantizing separately. Tile choice is identical to fp8_attn_input_a8_launch's.
void fp8_attn_input_a8_mma_launch(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                                  Tensor& v, Fp8A8Workspace workspace, std::int32_t tokens,
                                  cudaStream_t stream);

// Norms and quantizes in one kernel: emits the same A8 codes and scales that launch_fp8_a8_quantize
// produces from the Offset RMSNorm of residual, without storing the normalized rows.
void launch_fp8_attn_input_fused_rmsnorm_quantize(const Tensor& residual,
                                                  const Tensor& norm_weight, float eps,
                                                  Fp8A8Workspace workspace, cudaStream_t stream);

void fp8_attn_input_fused_rmsnorm_launch(const Tensor& residual, const Tensor& norm_weight,
                                         float eps, const Weight& weight, Tensor& q, Tensor& gate,
                                         Tensor& k, Tensor& v, Fp8A8Workspace workspace,
                                         cudaStream_t stream);

void fp8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                             Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                             cudaStream_t stream);

} // namespace ninfer::ops::detail
