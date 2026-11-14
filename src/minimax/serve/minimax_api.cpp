#include "minimax/serve/minimax_api.hpp"
#include "minimax/serve/minimax_chat.hpp"
#include "minimax/serve/minimax_tokenizer.hpp"
#include "strix/serve/request_fields.hpp"
#include "strix/serve/token_memo.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace strix::minimax::serve {
namespace sv = strix::serve;
namespace {
class DelegatedTokenizer final : public sv::ApiTokenizer {
public:
    explicit DelegatedTokenizer(std::shared_ptr<const MinimaxTokenizer> tokenizer) : tokenizer_(std::move(tokenizer)) {}
    std::vector<std::uint32_t> encode(std::string_view text, bool special) const override { return tokenizer_->encode(text,special); }
    std::string decode(std::span<const std::uint32_t> ids, bool skip) const override { return tokenizer_->decode(ids,skip); }
    std::string token_bytes(std::uint32_t id) const override { return tokenizer_->token_bytes(id); }
    std::size_t vocab_size() const override { return tokenizer_->vocab_size(); }
    std::uint32_t find(std::string_view text) const override { return tokenizer_->token_id(text); }
    TokenType type(std::uint32_t id) const override {
        if (id >= tokenizer_->vocab_size() || tokenizer_->token_text(id).empty()) return TokenType::unknown;
        return tokenizer_->is_special(id) ? TokenType::control : tokenizer_->is_added(id) ? TokenType::user_defined : TokenType::normal;
    }
    std::uint32_t eos_token() const override { return tokenizer_->eos_id(); }
private:
    std::shared_ptr<const MinimaxTokenizer> tokenizer_;
};
class CallbackTokenizer final : public sv::ApiTokenizer {
public:
    explicit CallbackTokenizer(strix::minimax::serve::ApiTokenizer tokenizer) : tokenizer_(std::move(tokenizer)) {}
    std::vector<std::uint32_t> encode(std::string_view text, bool special) const override {
        if (special && tokenizer_.encode) return tokenizer_.encode(text);
        if (tokenizer_.delegated) return tokenizer_.delegated->encode(text,special);
        throw std::invalid_argument("tokenizer does not expose encoding with this special-token policy");
    }
    std::string decode(std::span<const std::uint32_t> ids, bool skip) const override {
        if (skip && tokenizer_.decode) return tokenizer_.decode(ids);
        if (tokenizer_.delegated) return tokenizer_.delegated->decode(ids,skip);
        if (!tokenizer_.decode) throw std::invalid_argument("tokenizer does not expose decoding");
        return tokenizer_.decode(ids);
    }
    std::string token_bytes(std::uint32_t id) const override {
        if (!tokenizer_.stream_bytes) throw std::invalid_argument("tokenizer does not expose stream bytes");
        return tokenizer_.stream_bytes(id);
    }
    std::size_t vocab_size() const override { return metadata().vocab_size(); }
    std::uint32_t find(std::string_view text) const override { return metadata().find(text); }
    TokenType type(std::uint32_t id) const override { return metadata().type(id); }
    std::uint32_t eos_token() const override { return tokenizer_.eos_id; }
private:
    const sv::ApiTokenizer& metadata() const {
        if (!tokenizer_.delegated) throw std::invalid_argument("tokenizer does not expose vocabulary metadata");
        return *tokenizer_.delegated;
    }
    strix::minimax::serve::ApiTokenizer tokenizer_;
};
sv::AssistantOutput parse_output(const std::string& text) {
    sv::AssistantOutput out;
    out.tool_calls = sv::Json::array();
    const auto close = text.find("</mm:think>");
    if (close == std::string::npos) { out.content=text; return out; }
    const auto open=text.find("<mm:think>");
    const auto begin=open==std::string::npos || open>close ? 0 : open+10;
    out.reasoning=text.substr(begin,close-begin);
    out.content=text.substr(close+11);
    while (!out.content.empty() && (out.content.front()=='\n' || out.content.front()==' ')) out.content.erase(out.content.begin());
    return out;
}
void validate(const sv::HttpRequest& request, const sv::Json& body) {
    if (request.path!="/v1/chat/completions" && request.path!="/v1/completions" &&
        request.path!="/v1/messages" && request.path!="/v1/responses") return;
    sv::refuse_unexecuted(body,false,false);
    const auto refuse=[](const std::string& key) { throw sv::RequestFieldError(key,key+" is not supported by this server"); };
    for (const char* key : {"ngram_draft","draft_depth_max","tool_draft","ngram_peers","thinking_budget","thinking_policy","max_thinking_tokens","state_cache"})
        if (const auto* v=body.find(key); v && !v->is_null()) refuse(key);
    if (const auto* strix=body.find("strix"); strix && strix->is_object())
        for (const auto& [key,value]:strix->as_object()) if (!value.is_null()) refuse("strix."+key);
    if (const auto* thinking=body.find("thinking"); thinking && !thinking->is_null()) {
        if (!thinking->is_object()) refuse("thinking");
        for (const auto& [key,value]:thinking->as_object()) if (key!="type" && !value.is_null()) refuse("thinking."+key);
        if (const auto* type=thinking->find("type"); type && !type->is_null() &&
            (!type->is_string() || (type->as_string()!="enabled" && type->as_string()!="disabled"))) refuse("thinking.type");
    }
    if (request.path=="/v1/responses")
        if (const auto* reasoning=body.find("reasoning"); reasoning && reasoning->is_object())
            for (const auto& [key,value]:reasoning->as_object())
                if (key!="effort" && key!="summary" && key!="generate_summary" && !value.is_null()) refuse("reasoning."+key);
    for (const char* key : {"n","best_of"})
        if (const auto* v=body.find(key); v && !v->is_null() && (!v->is_number() || sv::exact_integer(v->number_source())!=std::optional<std::uint64_t>{1}))
            throw sv::RequestFieldError(key,std::string("only ")+key+" = 1 is supported");
    for (const char* key : {"stop","stop_sequences"}) if (const auto* v=body.find(key); v && !v->is_null()) {
        bool empty=v->is_string() && v->as_string().empty();
        if (v->is_array()) { empty=true; for (const auto& part:v->as_array()) empty=empty && part.is_string() && part.as_string().empty(); }
        if (!empty) throw sv::RequestFieldError(key,"stop strings is not supported by this server");
    }
    if (const auto* v=body.find("suffix"); v && !v->is_null() && !(v->is_string() && v->as_string().empty())) refuse("suffix");
    if (const auto* v=body.find("echo"); v && !v->is_null() && !(v->is_bool() && !v->as_bool())) refuse("echo");
    if (request.path=="/v1/completions") {
        const auto* prompt=body.find("prompt");
        if (!prompt || !prompt->is_string()) throw std::invalid_argument("prompt (string) required");
        for (const char* key : {"messages","tools","tool_choice","chat_template_kwargs","reasoning_effort"})
            if (const auto* value=body.find(key); value && !value->is_null()) refuse(key);
    }
    if (const auto* kw=body.find("chat_template_kwargs"); kw && kw->is_object())
        for (const auto& [key,value]:kw->as_object()) if (key!="thinking_mode" && key!="reasoning_effort" && !value.is_null()) refuse("chat_template_kwargs."+key);
    if (const auto* thinking=body.find("thinking"); thinking && !thinking->is_null() && request.path!="/v1/messages") refuse("thinking");
    if (const auto* enable=body.find("enable_thinking"); enable && !enable->is_null()) refuse("enable_thinking");
}
} // namespace

ApiTokenizer api_tokenizer(std::shared_ptr<const MinimaxTokenizer> tokenizer) {
    if (!tokenizer) throw std::invalid_argument("API tokenizer is required");
    ApiTokenizer t;
    t.eos_id=tokenizer->eos_id();
    // Chat turns resend their history: the memo encodes only the new tail, with the ids of a full encode.
    t.encode=[tokenizer,memo=std::make_shared<sv::TokenMemo>(),lookahead=tokenizer->mark_lookahead()](std::string_view text) {
        if (!sv::TokenMemo::enabled_by_default()) return tokenizer->encode(text,true);
        return memo->encode(1U,lookahead,text,[&](std::string_view tail,std::vector<sv::EncodeMark>& marks) {
            return tokenizer->encode_marked(tail,true,marks); });
    };
    t.decode=[tokenizer](std::span<const std::uint32_t> ids) { return tokenizer->decode(ids,true); };
    t.stream_bytes=[tokenizer](std::uint32_t id) { return tokenizer->is_special(id) ? std::string() : tokenizer->token_bytes(id); };
    t.delegated=std::make_shared<DelegatedTokenizer>(std::move(tokenizer));
    return t;
}

sv::Json reply_timings(const sv::GenerationStats& st, const RequestExtra& x) {
    auto hex = [](std::uint64_t v) {
        char b[24];
        std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
        return std::string(b);
    };
    sv::Json timings = sv::Json::object();
    timings.set("queue_ms", sv::Json(st.queue_ms));
    timings.set("prefill_ms", sv::Json(st.prefill_ms));
    timings.set("decode_ms", sv::Json(st.decode_ms));
    timings.set("decode_tok_s", sv::Json(st.decode_ms > 0 ? 1e3 * static_cast<double>(st.generated_tokens > 0 ? st.generated_tokens - 1 : 0) / st.decode_ms : 0.0));
    timings.set("completion_tokens", sv::Json(static_cast<std::int64_t>(st.generated_tokens)));
    timings.set("ttft_ms", sv::Json(x.ttft_ms));
    timings.set("max_gap_ms", sv::Json(x.max_gap_ms));
    timings.set("stall_ms", sv::Json(x.stall_ms));
    timings.set("prefill_units", sv::Json(static_cast<std::int64_t>(x.units)));
    timings.set("cycles", sv::Json(static_cast<std::int64_t>(x.cycles)));
    timings.set("drafted", sv::Json(static_cast<std::int64_t>(x.drafted)));
    timings.set("accepted", sv::Json(static_cast<std::int64_t>(x.accepted)));
    timings.set("ids_fnv", sv::Json(hex(x.ids_fnv)));
    timings.set("prompt_fnv", sv::Json(hex(x.prompt_fnv)));
    timings.set("sched", sv::Json(x.sched));
    timings.set("effective_sched", sv::Json(x.fallback_reason.empty() ? x.sched : x.fallback_reason == "sched_off" ? "off" : "unlimited"));
    timings.set("service_ms", sv::Json(x.service_ms));
    timings.set("max_unit_layers", sv::Json(static_cast<std::int64_t>(x.max_unit_layers)));
    timings.set("task_id", sv::Json(static_cast<std::int64_t>(x.task_id)));
    timings.set("nested", sv::Json(std::int64_t{x.nested ? 1 : 0}));
    timings.set("parent_request", sv::Json(static_cast<std::int64_t>(x.parent_request)));
    timings.set("nest_count", sv::Json(static_cast<std::int64_t>(x.nest_count)));
    timings.set("nest_service_ms", sv::Json(x.nest_service_ms));
    timings.set("nest_switch_ms", sv::Json(x.nest_switch_ms));
    timings.set("park_wall_ms", sv::Json(x.park_wall_ms));
    timings.set("short_first", sv::Json(x.short_first));
    timings.set("fallback_reason", sv::Json(x.fallback_reason));
    timings.set("dspark_mode", sv::Json(x.dspark_mode));
    timings.set("dspark_reason", sv::Json(x.dspark_reason));
    timings.set("dspark_context", sv::Json(x.context));
    timings.set("sampling", sv::Json("greedy"));
    if (x.resident_reported) {
        timings.set("cached_tokens", sv::Json(static_cast<std::int64_t>(st.cached_tokens)));
        timings.set("prefill_tokens_computed", sv::Json(static_cast<std::int64_t>(x.prefill_computed)));
        timings.set("cache_reply_extended", sv::Json(x.reply_extended));
    }
    return timings;
}

MinimaxApi::MinimaxApi(ApiTokenizer tokenizer, sv::GenerationQueue& queue, MinimaxApiOptions options, Observer& observer) {
    auto shared_options=std::make_shared<MinimaxApiOptions>(std::move(options));
    sv::ApiConfig config;
    config.model_id=shared_options->model_id;
    config.default_max_tokens=shared_options->default_max_tokens;
    config.deployment=shared_options->deployment;
    config.default_thinking=false;
    config.sessions_enabled=shared_options->session_enabled;
    config.compact_chat_envelope=true;
    config.timed_completion_envelope=true;
    config.exact_max_tokens=true;
    config.minimal_metadata=true;
    config.status_slots=shared_options->http_slots;
    config.explicit_thinking_only=true;
    config.tokenize_text_field=true;
    config.session_error_envelope=true;
    config.stream_receipts=false;
    config.sse_keepalive_seconds=0;
    config.sse_progress_seconds=0;
    config.observe_expiry_seconds=0;
    config.stop_tokens.push_back(tokenizer.eos_id);
    config.chat_extension_fields={"chat_template_kwargs","reasoning_effort","session"};
    config.validate_request=[tokenizer](const sv::HttpRequest& request,const sv::Json& body) {
        validate(request,body);
        if (request.path=="/detokenize") return; // vocabulary metadata is checked by the shared operation
        if (!tokenizer.encode) throw std::invalid_argument("tokenizer does not expose encoding");
        if (request.path=="/tokenize") return;
        const auto* stream=body.find("stream");
        if (stream && stream->is_bool() && stream->as_bool()) {
            if (!tokenizer.stream_bytes) throw std::invalid_argument("tokenizer does not expose stream bytes");
        } else if (!tokenizer.decode) throw std::invalid_argument("tokenizer does not expose decoding");
    };
    config.prepare_chat=[tokenizer](const sv::Json& body) {
        const auto* messages=body.find("messages");
        if (!messages || !messages->is_array() || !messages->size()) throw std::invalid_argument("messages required");
        MinimaxChatOptions chat;
        if (const auto* kw=body.find("chat_template_kwargs"); kw && kw->is_object()) {
            if (const auto* v=kw->find("thinking_mode"); v && v->is_string()) chat.thinking_mode=v->as_string();
            if (const auto* v=kw->find("reasoning_effort"); v && v->is_string()) chat.reasoning_effort=v->as_string();
        }
        if (const auto* v=body.find("reasoning_effort"); v && v->is_string()) chat.reasoning_effort=v->as_string();
        if (const auto* t=body.find("enable_thinking"); t && t->is_bool()) chat.thinking_mode=t->as_bool() ? "enabled" : "disabled";
        const auto* choice=body.find("tool_choice");
        const auto* type=choice && choice->is_object() ? choice->find("type") : choice;
        const bool off=type && type->is_string() && type->as_string()=="none";
        const auto* tools=body.find("tools");
        sv::ApiChatPrompt prompt;
        try { prompt.tokens=tokenizer.encode(render_minimax_chat(*messages,tools && tools->is_array() && !off ? tools : nullptr,chat)); }
        catch (const MinimaxChatError& error) { throw std::invalid_argument(error.what()); }
        return prompt;
    };
    // Requests execute the existing fixed greedy policy. Shared validation and
    // notes retain the requested filter/seed data without changing planner input.
    config.prepare_job=[](const sv::HttpRequest&,const sv::Json&,sv::GenerationJob& job) { job.sampling={}; };
    config.output_parser=[](const std::string& text,bool,const sv::Json&) { return parse_output(text); };
    config.tools_off_marker="]<]minimax[>[<tool_call>"; // tool_choice "none": a call block is withheld, not sent as text
    config.terminal_data=[shared_options](const sv::GenerationTicket& ticket) {
        const auto extra=shared_options->take_extra ? shared_options->take_extra(ticket) : std::nullopt;
        sv::ApiTerminalData data;
        data.timings=reply_timings(ticket.stats(),extra.value_or(RequestExtra{}));
        if (extra && ticket.stats().generated_tokens>=2) data.max_gap_ms=extra->max_gap_ms;
        return data;
    };
    config.request_finished=[&observer](const sv::ApiRequestObservation& event) {
        if (!event.observed) return;
        RequestRecord record;
        record.request_id=event.request_id; record.completion_id=event.completion_id; record.route=event.route;
        record.time=event.time; record.stream=event.stream; record.prompt_tokens=event.prompt_tokens;
        const auto& stats=event.stats;
        record.cached_tokens=stats.cached_tokens; record.output_tokens=stats.generated_tokens;
        record.queue_ms=stats.queue_ms; record.prefill_ms=stats.prefill_ms; record.decode_ms=stats.decode_ms;
        record.imported_tokens=stats.imported_tokens;
        record.first_token_ms=stats.generated_tokens ? stats.first_token_ms.value_or(-1.0) : -1.0;
        record.max_gap_ms=event.terminal.max_gap_ms.value_or(-1.0);
        record.draft=stats.proposed>0; record.cycles=stats.cycles; record.proposed=stats.proposed; record.accepted=stats.accepted;
        record.stop_reason=event.error.empty() ? stats.finish : std::string(); record.error=event.error;
        record.status=event.refused ? "refused" : !event.error.empty() ? "error" : !event.delivered || stats.finish=="cancelled" ? "cancelled" : "done";
        observer.record(std::move(record));
    };
    config.status_data=[&observer](const sv::QueueSnapshot& snapshot,bool stopping,bool detail) { return observer.status(snapshot,stopping,detail); };
    config.metrics_data=[&observer,shared_options](const sv::QueueSnapshot& snapshot) {
        return observer.metrics(snapshot,shared_options->legacy_metrics ? shared_options->legacy_metrics(snapshot) : legacy_metric_lines(snapshot));
    };
    config.route_data=[shared_options](const sv::HttpRequest& request)->std::optional<sv::Json> {
        if (request.path=="/admin/sched" && shared_options->admin_sched) {
            try { return shared_options->admin_sched(request); }
            catch (const std::exception& error) { throw sv::ApiRouteError(400,error.what()); }
        }
        return std::nullopt;
    };
    service_=std::make_unique<sv::ApiService>(std::move(config),sv::ApiTokenizerBinding(std::make_shared<CallbackTokenizer>(std::move(tokenizer))),queue,shared_options->context_tokens,false);
}

void MinimaxApi::handle(const sv::HttpRequest& request, sv::HttpResponseWriter& response) { service_->handle(request,response); }

} // namespace strix::minimax::serve
