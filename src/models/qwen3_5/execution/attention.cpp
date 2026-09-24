#include "models/qwen3_5/execution/attention.h"

#include "core/layout.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/rmsnorm.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

void require_rope_axes(const Tensor& positions, const RopeConfig& config) {
    if (positions.ne[1] != 3) { return; }
    for (std::size_t i = 0; i < config.pair_axes.size(); ++i) {
        if (config.pair_axes[i] != i % 3) {
            throw std::invalid_argument("text RoPE: this MRoPE axis mapping has no native route");
        }
    }
}

} // namespace

std::size_t attention_projection_workspace_bytes(const AttentionParameters& parameters,
                                                 std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("attention projection: invalid column interval");
    }
    if (const auto* single = std::get_if<LinearParameters>(&parameters.projection)) {
        const auto& weight = single->weight;
        return ops::rmsnorm_attn_input_proj_workspace_capacity_bytes(weight.qtype, weight.n,
                                                                     weight.k, single->policy,
                                                                     first, last);
    }
    // The two-parent form has no activation-quantizing producer, so its norm always materializes
    // the normalized row.
    WorkspaceLayoutBuilder layout;
    const auto& pair = std::get<ops::PairedProjectionWeights>(parameters.projection);
    (void)layout.alloc(DType::BF16, {pair.first.k, last});
    return layout.peak_bytes(1);
}

void attention_projection(const Tensor& residual, const Tensor& norm_weight, float eps,
                          const AttentionParameters& parameters, Tensor& query, Tensor& gate,
                          Tensor& key, Tensor& value, WorkspaceArena& workspace,
                          cudaStream_t stream) {
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        Tensor normalized = workspace.alloc(DType::BF16, {residual.ne[0], residual.ne[1]});
        ops::rmsnorm(residual, norm_weight, eps, true, normalized, stream);
        ops::attn_input_proj(normalized, pair->first, pair->second, query, gate, key, value,
                             stream);
        return;
    }
    const auto& single = std::get<LinearParameters>(parameters.projection);
    ops::rmsnorm_attn_input_proj(residual, norm_weight, eps, single.weight, query, gate, key, value,
                                 single.policy, workspace, stream);
}

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query,
               cudaStream_t stream) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, query, stream);
}

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query, Tensor& key,
               cudaStream_t stream) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, query, key, stream);
}

} // namespace ninfer::models::qwen3_5::execution
