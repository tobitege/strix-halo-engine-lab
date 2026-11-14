#pragma once

#include "coordinator.hpp"
#include "gemma4/ref/target.hpp"

namespace strix::gemma4::engine {

class HostBackend final : public Backend {
public:
    HostBackend(const ModelManifest& manifest, RopeTables tables, std::uint32_t workers = 16);
    bool input_continuation_supported() const override { return manifest_.identity == arithmetic_id; }
    std::unique_ptr<StreamState> create_stream() override;
    ForwardResult forward_rows(std::span<const RowPlan> rows) override;
    [[nodiscard]] const ModelConfig& config() const override { return manifest_.config; }
    [[nodiscard]] strix::serve::Json identity() const override;
private:
    const ModelManifest& manifest_;
    RopeTables tables_;
    HostTarget target_;
    std::uint32_t workers_;
};
// Shared immutable drafter assets and target views must outlive every session.
DraftFactory host_drafts(const dflash::Weights& weights, dflash::RopeView rope,
                         LinearView embedding, LinearView head);

} // namespace strix::gemma4::engine
