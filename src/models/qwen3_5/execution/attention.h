#pragma once

#include "models/qwen3_5/execution/parameters.h"

namespace ninfer::models::qwen3_5::execution {

// The block input RMSNorm belongs to this stage. Where the parent's route quantizes the normalized
// row it is about to consume, that route carries the norm as its prologue and the normalized row
// never reaches memory; elsewhere the stage materializes the normalized row in the call workspace.
// attention_projection_workspace_bytes() reports the capacity for either case.
[[nodiscard]] std::size_t
attention_projection_workspace_bytes(const AttentionParameters& parameters, std::int32_t first,
                                     std::int32_t last);
void attention_projection(const Tensor& residual, const Tensor& norm_weight, float eps,
                          const AttentionParameters& parameters, Tensor& query, Tensor& gate,
                          Tensor& key, Tensor& value, WorkspaceArena& workspace,
                          cudaStream_t stream);

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query,
               cudaStream_t stream);
void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query, Tensor& key,
               cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
