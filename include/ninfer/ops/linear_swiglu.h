#pragma once

// ninfer::ops - fused gate/up projection followed by SwiGLU.

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the transient capacity required by LinearSwiGLU for every T in the inclusive
 * [min_tokens,max_tokens] interval. The QType and dimensions are the fixed implementation profile.
 * Invalid profiles or intervals throw; a legal static-zero route returns zero.
 */
[[nodiscard]] std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype,
                                                                 std::int32_t gate_up_rows,
                                                                 std::int32_t input_rows,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens);

/**
 * Policy-bearing capacity query. Q4/Q8 use A16 under every policy. NVFP4 uses A16 under
 * A16Only/AllowA8 through T=16; AllowA4 accepts every positive T. Row-scaled FP8 accepts all
 * policies, with A8 permitted by AllowA8/AllowA4.
 * A permissive policy covers whichever qualified route the private resolver selects across the
 * requested interval.
 */
[[nodiscard]] std::size_t
linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                       std::int32_t input_rows, LinearPolicy policy,
                                       std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Op: linear_swiglu
 *
 * Math / indexing:
 *   gate_up = Linear(x, gate_up_weight); M=gate_up_rows/2;
 *   ideal[i,t] = SiLU(gate_up[i,t]) * gate_up[M+i,t].
 *
 * Logical shapes / supported domain:
 *   T may be any positive value. The registered profiles are:
 *   - Q4_G64_FP16 weight [34816,5120], x [5120,T], out [17408,T];
 *   - Q8_G32_FP16 weight [12288,2048], x [2048,T], out [6144,T];
 *   - Q8_G32_FP16 weight [34816,5120], x [5120,T], out [17408,T];
 *   - NVFP4 BlockScaleK16M128x4 weight [34816,5120], x [5120,T], out [17408,T];
 *   - FP8_E4M3FN_ROW_BF16 RowScale weight [34816,5120], x [5120,T], out [17408,T].
 *   Inputs and output are contiguous BF16. Q4/Q8 scales are FP16, NVFP4 scales are E4M3FN, and
 *   row-scaled FP8 has one BF16 multiplier per gate/up parent row. Gate rows `[0,17408)` precede
 *   their matching up rows `[17408,34816)`.
 *
 * Numeric:
 *   The oracle exact-decodes the registered weight and evaluates `ideal` naively in FP64 from the
 *   represented inputs. The BF16 output is promoted and compared directly with that result; output
 *   storage rounding belongs to LinearSwiGLU's named activation-compute criterion, not the oracle.
 *   Production routes may fuse or materialize gate/up and may choose their natural accumulator,
 *   staging, and workspace precision; those private choices are not semantic rounding boundaries.
 *   AllowA8/AllowA4 permit FP8 activation quantization; route thresholds are implementation
 * choices.
 *
 * Effects:
 *   Writes the full output; x/weight and output must not alias.
 *
 * Workspace:
 *   Caller-owned transient storage reported by linear_swiglu_workspace_capacity_bytes(),
 *   scoped to the call. Q8, NVFP4 A16, and row-scaled FP8 A16 require zero bytes; A4/A8 routes use
 *   caller-owned activation storage and may use private projection storage. There is no persistent
 *   state side effect.
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena& ws, cudaStream_t stream);

/**
 * A16-only convenience form. Q4/Q8 and row-scaled FP8 retain their complete positive-T domain.
 * NVFP4 is admitted only through T=16; larger NVFP4 extents require the policy-bearing AllowA4
 * form.
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream);

/**
 * Transient workspace needed by rmsnorm_linear_swiglu() over [min_tokens,max_tokens]. On a route
 * whose producer quantizes the normalized row it is about to consume this equals
 * linear_swiglu_workspace_capacity_bytes() for the same profile, `policy` and token count.
 * Everywhere else it adds the BF16 [input_rows,tokens] normalized image the Op has to materialize.
 */
[[nodiscard]] std::size_t
rmsnorm_linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                               std::int32_t input_rows, LinearPolicy policy,
                                               std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Applies the unit-offset input RMSNorm and the gate/up projection with SwiGLU as one semantic Op, so
 * the caller never writes the normalized row.
 *
 * Math / indexing:
 *   y = RmsNorm(residual, norm_gain, eps) with the unit-offset weight convention;
 *   out = LinearSwiGLU(y, gate_up_weight) evaluated on that same normalized image.
 *
 * On the NVFP4 [34816,5120] route whose producer quantizes the activation before projecting it, the
 * norm is that route's prologue: one kernel takes the NVFP4 codes and block scales directly from
 * `residual` and the normalized row never reaches memory. On any other weight format, `policy` or
 * token count the Op materializes the normalized row in `ws` and runs the standalone projection.
 * Which branch runs is the Op's own choice and is not observable except through `ws` capacity; both
 * branches are bit-identical to rmsnorm() followed by linear_swiglu() on the same route.
 *
 * Effects / workspace:
 *   residual/gain/out are contiguous BF16 with residual [D,T], norm_gain [D,1,1,1] where D is the
 *   weight's input rows, and out [gate_up_rows/2,T]; eps must be finite and positive. The output is
 *   written in full and must not alias an input. `ws` is sized by
 *   rmsnorm_linear_swiglu_workspace_capacity_bytes(). There is no persistent state side effect.
 */
void rmsnorm_linear_swiglu(const Tensor& residual, const Tensor& norm_gain, float eps,
                           const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                           WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops
