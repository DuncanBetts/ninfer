#include "models/qwen3_5/execution/ffn.h"

#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/silu_mul.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                std::int32_t last, bool mtp) {
    if (first <= 0 || last < first) { throw std::invalid_argument("FFN: invalid column interval"); }
    WorkspaceLayoutBuilder layout;
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        // The expert pipeline has no route that consumes the norm's output row as it is produced, so
        // the normalized image is reserved ahead of the expert scratch.
        (void)layout.alloc(DType::BF16, {moe->routed_gate_up.k, last});
        (void)layout.alloc_bytes(ops::sparse_moe_workspace_capacity_bytes(
            moe->routed_gate_up.qtype, moe->routed_down.qtype, first, last));
        return layout.peak_bytes(1);
    }
    const auto& p    = std::get<DenseParameters>(parameters);
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    if (mtp) {
        (void)layout.alloc(DType::BF16, {gu.k, last});
        (void)layout.alloc(DType::BF16, {gu.n, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
        }
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        (void)layout.alloc(DType::BF16, {down.n, last});
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(down.qtype, down.n, down.k,
                                                                      p.down.policy, first, last));
    } else {
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::rmsnorm_linear_swiglu_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
        }
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
                down.qtype, down.n, down.k, p.down.policy, first, last));
        }
    }
    return layout.peak_bytes(1);
}

void ffn(const Tensor& row, const Tensor& norm_gain, float eps, const FfnParameters& parameters,
         Tensor& residual, const ops::SparseMoeHints& hints, WorkspaceArena& workspace,
         cudaStream_t stream, bool mtp) {
    auto scope         = workspace.scope();
    const auto columns = row.ne[1];
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        Tensor normalized = workspace.alloc(DType::BF16, {row.ne[0], columns});
        ops::rmsnorm(row, norm_gain, eps, true, normalized, stream);
        const auto storage = workspace.alloc_bytes(ops::sparse_moe_workspace_capacity_bytes(
            moe->routed_gate_up.qtype, moe->routed_down.qtype, columns, columns));
        WorkspaceArena scratch(storage);
        ops::sparse_moe(normalized, *moe, ops::SparseMoeEpilogue::AddResidual, residual, hints,
                        scratch, stream);
        return;
    }
    const auto& p    = std::get<DenseParameters>(parameters);
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    if (mtp) {
        Tensor normalized = workspace.alloc(DType::BF16, {row.ne[0], columns});
        ops::rmsnorm(row, norm_gain, eps, true, normalized, stream);
        Tensor gate_up = workspace.alloc(DType::BF16, {gu.n, columns});
        {
            auto call = workspace.scope();
            ops::linear(normalized, gu, gate_up, p.gate_up.policy, workspace, stream);
        }
        Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
        ops::silu_mul(gate_up.slice(0, 0, gu.n / 2), gate_up.slice(0, gu.n / 2, gu.n / 2),
                      activation, stream);
        Tensor delta = workspace.alloc(DType::BF16, {down.n, columns});
        ops::linear(activation, down, delta, p.down.policy, workspace, stream);
        ops::residual_add(delta, residual, stream);
        return;
    }
    Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
    {
        auto call = workspace.scope();
        ops::rmsnorm_linear_swiglu(row, norm_gain, eps, gu, activation, p.gate_up.policy, workspace,
                                   stream);
    }
    ops::linear_add(activation, down, residual, p.down.policy, workspace, stream);
}

} // namespace ninfer::models::qwen3_5::execution
