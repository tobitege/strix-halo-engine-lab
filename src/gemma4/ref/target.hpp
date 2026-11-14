#pragma once

#include "gemma4/model/model.hpp"
#include <functional>

namespace strix::gemma4 {

struct LayerWeights {
    LinearView input_norm, post_attention_norm, pre_ffn_norm, post_ffn_norm, post_ple_norm;
    LinearView q, k, v, o, q_norm, k_norm, gate, up, down, ple_gate, ple_projection, scalar;
};
struct HostWeights {
    ModelConfig config;
    LinearView embedding, ple_table, ple_model, ple_model_norm, final_norm;
    std::vector<LayerWeights> layer;
    LinearView head; // optional separate quantized head; empty preserves tied embedding
};
[[nodiscard]] HostWeights bind_host_weights(const ModelManifest& model);

class HostStream {
public:
    explicit HostStream(const ModelConfig& config);
    ~HostStream();
    // Full independent rollback copy, preserving contiguous storage and ring tags.
    [[nodiscard]] std::unique_ptr<HostStream> snapshot() const;
    [[nodiscard]] std::uint64_t retained_bytes() const;
    [[nodiscard]] std::unique_ptr<KvTxn> begin();
    [[nodiscard]] std::uint64_t frontier() const;
    // Canonical logical state, independent of ring addresses or unpublished candidates.
    [[nodiscard]] std::string digest() const;
    // Read-only view of a producing layer: committed snapshot plus, when given, that transaction's overlay.
    [[nodiscard]] KvView kv_view(std::uint32_t layer, const KvTxn* transaction = nullptr) const;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class HostTarget;
};
// Optional linear recipe of a named opt-in identity. It returns false to keep the g4-v1 recipe for
// that view; otherwise it writes rows*outputs values. parallel(count, job) runs job(begin, end) over
// disjoint output ranges on the reference workers.
using ParallelOutputs = std::function<void(std::size_t, const std::function<void(std::size_t, std::size_t)>&)>;
using LinearRecipe = std::function<bool(LinearView, const float* x, std::size_t rows, float* out, const ParallelOutputs&)>;
class HostTarget final : public Target {
public:
    // Model and table storage must outlive this reference. Workers parallelize outputs only.
    HostTarget(HostWeights weights, RopeTables tables, std::uint32_t workers = 16, LinearRecipe recipe = {});
    ~HostTarget() override;
    ForwardResult forward_rows(std::span<const RowPlan> rows) override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
struct OwnedRopeTables {
    std::shared_ptr<MappedFile> sliding, global;
    RopeTables view;
};
[[nodiscard]] OwnedRopeTables load_rope_tables(const std::filesystem::path& directory);

} // namespace strix::gemma4
