#include "chat_runner.hpp"

#include <stdexcept>

namespace strix::gemma4::engine {
ChatJob::ChatJob(const gemma4::serve::Adapter& adapter, ChatRequest request, ChatSink sink,
    std::optional<gemma4::serve::PreparedPrompt> prepared_prompt)
    : adapter_(adapter), request_(std::move(request)), sink_(std::move(sink)) {
    if (request_.generation.max_tokens > UINT32_MAX) throw std::invalid_argument("max_tokens exceeds chat limit");
    if (request_.thinking_budget && (!request_.input.enable_thinking || request_.thinking_budget >= request_.generation.max_tokens))
        throw std::invalid_argument("thinking_budget requires thinking and must be below max_tokens");
    const auto prepared = prepared_prompt ? std::move(*prepared_prompt) : adapter_.prepare(request_.input);
    prompt_ = prepared.ids;
    stop_ids_ = prepared.stop_ids;
    starts_in_reasoning_ = prepared.starts_in_reasoning;
    draws_ = request_.generation.sampling.draws;
}
Request ChatJob::phase_request() {
    complete_turn_ = false;control_output_ = false;
    auto job = request_.generation;
    job.prompt = prompt_; job.stop_ids = stop_ids_; job.sampling.draws = draws_;
    job.stop_ids.push_back(adapter_.tokenizer().find("<|tool_response>"));
    job.max_tokens = phase_ == 0 && request_.thinking_budget ? request_.thinking_budget : request_.generation.max_tokens-result_.ids.size();
    result_.prompt_tokens += prompt_.size();
    parser_.emplace(starts_in_reasoning_);
    sent_ = 0;
    return job;
}
bool ChatJob::absorb(const Result& out) {
    while (!disconnected_ && sent_ < out.ids.size()) {
        const auto bytes=adapter_.tokenizer().token_bytes(out.ids[sent_++]);
        // The parser exposes no open-channel fact for an empty thought. Keep a
        // conservative admission fact on this existing consumption path: only
        // ordinary bytes and whole terminal markers certify reusable output.
        // No second parser, text reconstruction, or token selection is involved.
        control_output_ |= bytes.find('<')!=std::string::npos && bytes!="<eos>" && bytes!="<turn|>";
        const auto delta = parser_->push(bytes);
        if (sink_ && !sink_(delta)) disconnected_ = true;
    }
    return !disconnected_;
}
bool ChatJob::end_phase(const Result& out, strix::serve::Json receipt) {
    const auto tail = parser_->finish();
    if (!disconnected_ && sink_ && !sink_(tail)) disconnected_ = true;
    if (out.status == Status::failed) throw std::runtime_error(out.error);
    draws_ = out.rng_draws;
    receipt.set("chat_policy",strix::serve::Json(strix::serve::Json::Object{
        {"enable_thinking",request_.input.enable_thinking},
        {"conversational_mode",request_.input.conversational_mode},
        {"thinking_budget",std::uint64_t(request_.thinking_budget)},
        {"phase",std::uint64_t(phase_)},
        {"selection",phase_ ? "budget-answer-continuation" : request_.thinking_budget ? "within-thinking-budget" : "ordinary-chat"}}));
    result_.receipts.push_back(std::move(receipt));
    result_.ids.insert(result_.ids.end(),out.ids.begin(),out.ids.end());
    const auto& parsed = parser_->output();
    result_.output.reasoning += parsed.reasoning; result_.output.content += parsed.content;
    for (const auto& call : parsed.tool_calls.as_array()) result_.output.tool_calls.push(call);
    result_.output.reasoning_complete = result_.output.reasoning_complete || parsed.reasoning_complete;
    result_.output.unfinished_call = parsed.unfinished_call; result_.output.unfinished_control = parsed.unfinished_control;
    result_.output.stopped = parsed.stopped;
    result_.finish = disconnected_ ? "cancelled" : status_name(out.status);
    if (phase_ || !request_.thinking_budget || out.status != Status::length || disconnected_) {
        complete_turn_ = !disconnected_ && (out.status == Status::stopped || out.status == Status::length) &&
            out.error.empty() && !control_output_ && !parsed.unfinished_call && !parsed.unfinished_control &&
            (!starts_in_reasoning_ || parsed.reasoning_complete) &&
            (!request_.input.enable_thinking || phase_ || parsed.reasoning_complete) &&
            (parsed.reasoning.empty() || parsed.reasoning_complete) && !parsed.tool_calls.size();
        if (result_.output.tool_calls.size() && result_.finish == "stop") result_.finish = "tool_calls";
        return false;
    }
    const auto continuation = gemma4::serve::continue_after_budget(request_.input,parsed,
        std::uint32_t(out.ids.size()),request_.thinking_budget,std::uint32_t(request_.generation.max_tokens));
    prompt_ = adapter_.tokenizer().encode(continuation.prompt);
    starts_in_reasoning_ = continuation.prompt.ends_with("<|channel>thought\n");
    result_.budget_continued = true;
    phase_ = 1;
    return true;
}
ChatResult run_chat(Backend& backend, const gemma4::serve::Adapter& adapter, ChatRequest request,
    DraftFactory drafts, const ChatSink& sink, const Cancelled& cancelled, std::shared_ptr<CostModel> costs) {
    ChatJob chat(adapter,std::move(request),sink);
    for (bool more = true; more;) {
        Coordinator coordinator(backend,drafts,costs);
        const auto slot = coordinator.submit(chat.phase_request());
        while (coordinator.result(slot).status == Status::running) {
            if ((cancelled && cancelled()) || chat.disconnected()) { coordinator.cancel(slot); break; }
            coordinator.step();
            if (!chat.absorb(coordinator.result(slot))) coordinator.cancel(slot);
        }
        more = chat.end_phase(coordinator.result(slot),coordinator.receipt(slot));
    }
    return chat.take();
}
} // namespace strix::gemma4::engine
