#pragma once

#include "models/qwen3_5/execution/parameters.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                              std::int32_t last, bool mtp = false);
// The block post-attention RMSNorm belongs to this stage. On the dense gate/up route whose producer
// quantizes the normalized row it is about to consume, that route carries the norm as its prologue
// and the normalized row never reaches memory; the MoE, MTP and every other dense route
// materialize it in the call workspace. ffn_workspace_bytes() reports either case.
// `row` is read by the norm and `residual` receives the FFN delta in place, as before.
void ffn(const Tensor& row, const Tensor& norm_gain, float eps, const FfnParameters& parameters,
         Tensor& residual, const ops::SparseMoeHints& hints, WorkspaceArena& workspace,
         cudaStream_t stream, bool mtp = false);

} // namespace ninfer::models::qwen3_5::execution
