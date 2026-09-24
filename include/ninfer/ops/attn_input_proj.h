#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Computes four independent linear projections for each token:
 *
 *   q[:,t]    = linear(x[:,t], query_key_weight[0:6144,:])
 *   k[:,t]    = linear(x[:,t], query_key_weight[6144:7168,:])
 *   gate[:,t] = linear(x[:,t], gate_value_weight[0:6144,:])
 *   v[:,t]    = linear(x[:,t], gate_value_weight[6144:7168,:]).
 *
 * All tensors are contiguous BF16. Shapes are x [5120,T], q/gate [6144,T], and k/v [1024,T].
 * T may be any positive value.
 * The two parent weights are RowSplit [7168,5120] with FP16 scales and group size 64:
 * query_key is Q4_G64_FP16 and gate_value is Q5_G64_FP16. The oracle exact-decodes each row and
 * evaluates every projection naively in FP64 from the represented inputs. The BF16 outputs are
 * promoted and compared directly with those ideal values; final output storage rounding belongs
 * to AttnInputProj's named A16 criterion, not the oracle. Production routes choose their private
 * accumulator and staging precision. Inputs and the four outputs must be mutually non-overlapping.
 * Current registered routes require no transient allocation. The Op has no persistent state side
 * effect.
 */
void attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     cudaStream_t stream);

/**
 * Computes the single-parent Q/K/output-gate/V projection.
 *
 * The parent stores rows in physical order query, key, output gate, value while the public output
 * argument order is q, gate, k, v. Every route writes the four independently contiguous final
 * allocations directly; no packed parent output is materialized. The NVFP4 A4 and FP8 A8
 * profiles may use caller-owned transient storage for their private quantized activation.
 *
 * Registered parent forms are:
 *
 * - Q8_G32_FP16 RowSplit `[9216,2048]`, with row counts `[4096,512,4096,512]`. `x` is
 *   BF16 `[2048,T]`, q/gate are BF16 `[4096,T]`, and k/v are BF16 `[512,T]`.
 * - BF16 Contiguous `[14336,5120]`, with row counts `[6144,1024,6144,1024]`. `x` is
 *   BF16 `[5120,T]`, q/gate are BF16 `[6144,T]`, and k/v are BF16 `[1024,T]`.
 * - NVFP4 BlockScaleK16M128x4 `[14336,5120]`, with the same logical row and tensor shapes as
 *   BF16.
 * - FP8_E4M3FN_ROW_BF16 RowScale `[14336,5120]`, with the same logical row and tensor shapes as
 *   BF16.
 *
 * `T` is the positive token extent of the Op contract. All three policies permit the BF16
 * and Q8_G32_FP16 A16 implementations. NVFP4 uses A16 under A16Only/AllowA8; AllowA4 permits the
 * resolver to select either a qualified A16 route or activation quantization to NVFP4 at every
 * positive T. FP8 accepts all policies at every positive T. AllowA8/AllowA4 permit the resolver to
 * choose a qualified A16 route or private activation quantization followed by A8 Tensor Core
 * computation. A16Only preserves the represented BF16 activation at every positive T; tile and
 * route cutoffs are private implementation choices, independent of speculative block width.
 *
 * The oracle evaluates every projection independently with naive FP64 accumulation from the
 * logical values represented by the persistent weight and BF16 activation. The final four BF16
 * stores belong to the Op's criterion for the selected activation-compute path.
 *
 * `workspace` is caller-owned call-scoped transient storage sized by
 * attn_input_proj_workspace_capacity_bytes(). It must not overlap the input, parent weight, or any
 * output. The Op does not allocate device memory internally.
 */
[[nodiscard]] std::size_t
attn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                         std::int32_t input_rows, LinearPolicy policy,
                                         std::int32_t min_tokens, std::int32_t max_tokens);

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                     WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Transient workspace needed by rmsnorm_attn_input_proj() over [min_tokens,max_tokens]. On a route
 * whose producer quantizes the normalized row it is about to consume this equals
 * attn_input_proj_workspace_capacity_bytes() for the same parent, `policy` and token count.
 * Everywhere else it adds the BF16 [input_rows,tokens] normalized image the Op has to materialize.
 */
[[nodiscard]] std::size_t
rmsnorm_attn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                                 std::int32_t input_rows, LinearPolicy policy,
                                                 std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Applies the unit-offset input RMSNorm and the single-parent Q/output-gate/K/V projection as one
 * semantic Op, so the caller never writes the normalized row. Where the parent's route at this
 * `tokens` quantizes the normalized row it is about to consume — an FP8_E4M3FN_ROW_BF16 RowSplit
 * [14336,5120] parent on its activation-quantized A8 route, which is T >= 5 — the norm is that
 * route's prologue and the normalized image never reaches memory. On any other parent, `policy` or
 * token count the Op materializes the normalized row in `workspace` and runs the standalone
 * projection, producing the arithmetic of rmsnorm() followed by attn_input_proj(). Route selection is
 * the Op's own choice and is not observable except through `workspace` capacity.
 *
 * `residual` is contiguous BF16 [D,T] and `norm_weight` is contiguous BF16 [D,1,1,1] with D equal to
 * the parent's input rows. The norm uses rsqrtf(sum(x*x)/D + eps) and multiplies each element by
 * 1 + norm_weight, rounding to BF16 before any activation quantization; eps must be finite and
 * positive. q/gate and k/v are the projection outputs described by attn_input_proj() for this parent.
 *
 * The exact-byte oracle for both branches is rmsnorm() with unit offset into BF16 storage followed by
 * the standalone projection. `workspace` is sized by
 * rmsnorm_attn_input_proj_workspace_capacity_bytes() and must not overlap any input, weight, or
 * output; the Op allocates no device memory of its own and has no persistent state side effect.
 */
void rmsnorm_attn_input_proj(const Tensor& residual, const Tensor& norm_weight, float eps,
                             const Weight& projection_weight, Tensor& q, Tensor& gate, Tensor& k,
                             Tensor& v, LinearPolicy policy, WorkspaceArena& workspace,
                             cudaStream_t stream);

/**
 * Applies the A16-only single-parent Q/K/output-gate/V projection without transient workspace.
 */
void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);

/**
 * Three-output Q8 specialization. The Q8_G32_FP16 RowSplit parent stores rows in order
 * [query 4096, key 1024, value 1024]. Registered parent forms are [6144,2048] with BF16
 * x [2048,T] for the Qwen3.6 companion and [6144,5120] with BF16 x [5120,T] for DFlash2.
 * q is contiguous BF16 [4096,T], and k/v are contiguous BF16 [1024,T]. Every route writes the
 * three independent final allocations directly; no parent output or transient workspace is
 * materialized. T may be any positive value. Q and K remain raw projection outputs: this Op does
 * not normalize or rotate either tensor.
 */
void attn_input_proj(const Tensor& x, const Weight& query_key_value_weight, Tensor& q, Tensor& k,
                     Tensor& v, cudaStream_t stream);

} // namespace ninfer::ops
