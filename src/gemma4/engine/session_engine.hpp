#pragma once

#include "coordinator.hpp"
#include "strix/session/engine.hpp"
#include "strix/session/backend.hpp"
#include <mutex>

namespace strix::gemma4::engine {
// Native family plans remain device-neutral: the existing KvTxn/RowPlan types
// contain no accelerator SDK types. The executor delegates model computation
// to the loaded target; resources/allocations remain owned by that target.
struct ForwardPlan final : session::ExecutionPlan {
    ForwardPlan(session::Backend& b, std::span<const RowPlan> r) : ExecutionPlan(b), rows(r) {}
    std::span<const RowPlan> rows;
    ForwardResult result;
};
struct PrefillPlan final : session::ExecutionPlan {
    PrefillPlan(session::Backend& b, std::span<const PrefillItem> i, bool h, bool t, bool grouped)
        : ExecutionPlan(b), items(i), hidden(h), taps(t), batch(grouped) {}
    std::span<const PrefillItem> items;
    bool hidden, taps, batch;
    std::vector<PrefillRows> result;
};

class SessionEngine final : public session::Engine {
public:
    SessionEngine(Backend&, DraftFactory, std::shared_ptr<CostModel>, std::mutex&,
                  std::size_t slots, std::size_t prefill_limit, std::size_t prefill_budget);
    ~SessionEngine();
    session::KvStore& store();
    void prepare(session::KvHandle, Request);
    void start(session::KvHandle);
    std::uint64_t initial_cursor(session::KvHandle) const;
    // Borrowed on the externally serialized owner, until its next model call.
    const Result& result(session::KvHandle) const;
    strix::serve::Json lane_receipt(session::KvHandle) const;
    // Prepare an immutable prompt-purpose payload only if the physical backend
    // can capture it and this numerical boundary is reusable.
    bool cache_candidate(session::KvHandle);
    // Explicit sealed complete-turn producer, separate from legacy admission.
    bool turn_cache_candidate(session::KvHandle);
    [[nodiscard]] std::optional<std::uint64_t> turn_retained_bound(session::KvHandle) const;
    [[nodiscard]] std::optional<std::uint64_t> turn_work_bound() const;
    void commit_pending(session::KvHandle&) override;
    bool matches_decode() const;
    std::size_t chunk_rows() const;
    void prefill(std::span<const std::uint32_t>, session::KvHandle&) override;
    session::StepResult decode(session::KvHandle&, const session::SamplerState&) override;
    std::vector<session::AdvanceResult> advance(std::span<const session::AdvanceItem>) override;
    void cancel_lane(session::KvHandle&) override;
    void append_external(std::span<const std::uint32_t>, session::KvHandle&) override;
    void resume_lane(session::KvHandle&) override;
    // Borrowed until the next family call; empty once output is selected.
    std::span<const float> retained_logits(session::KvHandle) const;
    session::Capabilities caps() const override;
    strix::profile::ExecutionReceipt receipt() const override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strix::gemma4::engine
