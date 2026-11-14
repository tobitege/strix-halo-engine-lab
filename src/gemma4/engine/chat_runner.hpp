#pragma once

#include "coordinator.hpp"
#include "gemma4/serve/adapter.hpp"
#include <optional>

namespace strix::gemma4::engine {

struct ChatRequest {
    gemma4::serve::ChatInput input;
    Request generation;
    std::uint32_t thinking_budget = 0;
};
struct ChatResult {
    gemma4::serve::Output output;
    std::vector<std::uint32_t> ids;
    std::vector<strix::serve::Json> receipts;
    std::string finish;
    std::size_t prompt_tokens = 0;
    bool budget_continued = false;
};
// Callback returns false on cancellation/disconnect, including during prefill.
using ChatSink = std::function<bool(const gemma4::serve::OutputDelta&)>;
using Cancelled = std::function<bool()>;
// One chat request as coordinator jobs: phase 0, then (thinking budget reached) the phase-1 answer continuation.
// The caller submits phase_request() to a Coordinator slot, passes the slot's Result to absorb() after each step
// (new ids -> parser -> sink) and to end_phase() once the slot stops. The shared
// owner restarts the phase on the same external job/ticket when it returns true.
class ChatJob {
public:
    ChatJob(const gemma4::serve::Adapter& adapter, ChatRequest request, ChatSink sink = {},
        std::optional<gemma4::serve::PreparedPrompt> prepared = {});
    [[nodiscard]] Request phase_request();
    // false once the sink refused a delta (cancel the slot).
    bool absorb(const Result& out);
    bool end_phase(const Result& out, strix::serve::Json receipt);
    [[nodiscard]] bool disconnected() const { return disconnected_; }
    // Frozen before take() moves presentation/accounting out of this job.
    [[nodiscard]] bool complete_turn() const noexcept { return complete_turn_; }
    [[nodiscard]] ChatResult take() { return std::move(result_); }
private:
    const gemma4::serve::Adapter& adapter_;
    ChatRequest request_;
    ChatSink sink_;
    ChatResult result_;
    std::vector<std::uint32_t> prompt_, stop_ids_;
    std::optional<gemma4::serve::OutputParser> parser_;
    std::uint64_t draws_ = 0;
    std::size_t sent_ = 0;
    unsigned phase_ = 0;
    bool starts_in_reasoning_ = false, disconnected_ = false, complete_turn_ = false;
    bool control_output_ = false;
};
ChatResult run_chat(Backend& backend, const gemma4::serve::Adapter& adapter,
    ChatRequest request, DraftFactory drafts = {}, const ChatSink& sink = {}, const Cancelled& cancelled = {},
    std::shared_ptr<CostModel> costs = {});

} // namespace strix::gemma4::engine
