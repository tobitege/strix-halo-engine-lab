#include "minimax/serve/minimax_api.hpp"
#include "strix/serve/request_fields.hpp"
#include "strix/serve/session_api.hpp"

#include "minimax/serve/minimax_chat.hpp"
#include "minimax/serve/minimax_tokenizer.hpp"
#include "strix/serve/bpe_tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace strix::minimax::serve {
namespace {

namespace sv = strix::serve;
void send_json(sv::HttpResponseWriter& w, int status, const sv::Json& body) { w.send(status, "application/json", body.dump()); }
void send_error(sv::HttpResponseWriter& w, int status, const std::string& message, const std::string& param = {}) {
    sv::Json e = sv::Json::object();
    e.set("message", sv::Json(message));
    e.set("type", sv::Json(status == 400 ? "invalid_request_error" : "server_error"));
    if (!param.empty()) e.set("param", sv::Json(param));
    sv::Json b = sv::Json::object();
    b.set("error", std::move(e));
    send_json(w, status, b);
}

// Generation fields, checked with the shared front-end checks (strix/serve/request_fields.hpp) so the contract is the
// Qwen/DeepSeek one. This server decodes greedy only and executes no constraints, logit transforms or scoring: those
// fields are refused with a 400 naming them unless their value is neutral, as are stop strings, n / best_of other than
// 1, suffix and echo. A positive temperature runs greedy and the reply says so (timings.sampling_note, the shared
// server's note text). Returns the sampling note ("" when nothing was ignored).
std::string check_fields(const sv::Json& body) {
    sv::refuse_unexecuted(body, false, false);
    const auto set = [&](const char* key) -> const sv::Json* { const auto* v = body.find(key); return v && !v->is_null() ? v : nullptr; };
    const auto refuse = [](const std::string& param, const std::string& what) {
        throw sv::RequestFieldError(param, what + " is not supported by this server");
    };
    const auto real = [&](const char* key, double low, double high, bool open_low, bool open_high, double absent) {
        const auto* v = set(key);
        return v ? sv::finite_number(*v, key, low, high, open_low, open_high) : absent;
    };
    const double temperature = real("temperature", 0, 2, false, false, 0);
    const double top_p = real("top_p", 0, 1, true, false, 1);
    const double min_p = real("min_p", 0, 1, false, true, 0);
    const std::uint64_t top_k = set("top_k") ? sv::unsigned_integer(*set("top_k"), "top_k", UINT32_MAX) : 0;
    if (const auto* v = set("seed")) (void)sv::unsigned_integer(*v, "seed", INT64_MAX);
    for (const char* key : {"n", "best_of"})
        if (const auto* v = set(key); v && (!v->is_number() || sv::exact_integer(v->number_source()) != std::optional<std::uint64_t>{1}))
            throw sv::RequestFieldError(key, std::string("only ") + key + " = 1 is supported");
    if (const auto* v = set("stop")) {
        bool empty = v->is_string() && v->as_string().empty();
        if (v->is_array()) {
            empty = true;
            for (const auto& s : v->as_array()) empty = empty && s.is_string() && s.as_string().empty();
        }
        if (!empty) refuse("stop", "stop strings");
    }
    if (const auto* v = set("suffix"); v && !(v->is_string() && v->as_string().empty())) refuse("suffix", "suffix");
    if (const auto* v = set("echo"); v && !(v->is_bool() && !v->as_bool())) refuse("echo", "echo");
    if (temperature > 0.0) return "temperature ignored: this engine decodes greedy (exact)";
    if (top_p != 1.0 || min_p != 0.0 || top_k != 0U) return "filters ignored at temperature 0 (greedy)";
    return {};
}

// max_completion_tokens (preferred) or max_tokens: a positive integer read exactly from its JSON spelling; a value above
// the context is clamped to it (the engine ends a generation at its context limit anyway), never wrapped. A server
// without a known context (0) clamps to the largest count a job holds.
std::uint32_t requested_max_tokens(const sv::Json& body, std::uint32_t fallback, std::uint64_t context) {
    for (const char* key : {"max_completion_tokens", "max_tokens"})
        if (const auto* v = body.find(key); v && !v->is_null()) {
            const auto requested = sv::unsigned_integer(*v, key, UINT64_MAX);
            if (requested < 1U) throw sv::RequestFieldError(key, std::string(key) + " must be a positive integer");
            const auto limit = context ? std::min<std::uint64_t>(context, UINT32_MAX) : std::uint64_t{UINT32_MAX};
            return static_cast<std::uint32_t>(std::min(requested, limit));
        }
    return fallback;
}

// A boolean request option: absent or null = false; any other non-boolean is a 400.
bool option(const sv::Json& body, const char* key) {
    const auto* v = body.find(key);
    if (!v || v->is_null()) return false;
    if (!v->is_bool()) throw std::invalid_argument(std::string(key) + " must be a boolean");
    return v->as_bool();
}

// Generated text split as the template writes reasoning back: <mm:think> ... </mm:think> then the answer.
void split_reasoning(const std::string& text, std::string& reasoning, std::string& content) {
    static const std::string open = "<mm:think>", close = "</mm:think>";
    const auto c = text.find(close);
    if (c == std::string::npos) {
        content = text;
        return;
    }
    auto o = text.find(open);
    const std::size_t r0 = o == std::string::npos || o > c ? 0 : o + open.size();
    reasoning = text.substr(r0, c - r0);
    content = text.substr(c + close.size());
    while (!content.empty() && (content.front() == '\n' || content.front() == ' ')) content.erase(content.begin());
}

}  // namespace

ApiTokenizer api_tokenizer(std::shared_ptr<const MinimaxTokenizer> tokenizer) {
    ApiTokenizer t;
    t.eos_id = tokenizer->eos_id();
    t.encode = [tokenizer](std::string_view text) { return tokenizer->encode(text, true); };
    t.decode = [tokenizer](std::span<const std::uint32_t> ids) { return tokenizer->decode(ids, true); };
    t.stream_bytes = [tokenizer](std::uint32_t id) { return tokenizer->is_special(id) ? std::string() : tokenizer->token_bytes(id); };
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
    return timings;
}

MinimaxApi::MinimaxApi(ApiTokenizer tokenizer, sv::GenerationQueue& queue, MinimaxApiOptions options, Observer& observer)
    : tok_(std::move(tokenizer)), queue_(queue), options_(std::move(options)), observer_(observer) {}

void MinimaxApi::generate(const sv::HttpRequest&, sv::HttpResponseWriter& w, std::vector<std::uint32_t> prompt, const sv::Json& body,
                          bool chat, const std::string& inbound_id) {
    const std::string sampling_note = check_fields(body);
    const std::uint32_t max_tokens = requested_max_tokens(body, options_.default_max_tokens, options_.context_tokens);
    const auto timings = [&](const sv::GenerationStats& st, const std::optional<RequestExtra>& x) {
        auto t = reply_timings(st, x.value_or(RequestExtra{}));
        if (!sampling_note.empty()) t.set("sampling_note", sv::Json(sampling_note));
        return t;
    };
    const bool stream = option(body, "stream");
    const bool want_ids = option(body, "return_token_ids");
    bool usage_option = false;
    if (const auto* so = body.find("stream_options"); so && so->is_object()) usage_option = option(*so, "include_usage");
    const bool include_usage = stream && usage_option;
    const std::string id = std::string(chat ? "chatcmpl-" : "cmpl-") + std::to_string(next_id_++);
    const auto created = static_cast<std::int64_t>(std::time(nullptr));
    // a generation request that reaches submission echoes its completion id when it has no accepted inbound id (that
    // one is already in the response head, see handle)
    if (inbound_id.empty()) w.add_response_header("X-Request-Id", id);
    sv::GenerationJob job;
    job.prompt = std::move(prompt);
    if (options_.session_enabled) if (const auto* v = body.find("session")) {
        if (!v->is_string()) { w.send(400,"application/json",sv::session_error("invalid_request","session must be a string").dump()); return; }
        const auto& session_id = v->as_string();
        try { (void)sv::session_command("POST", "/v1/sessions/" + session_id + "/park", ""); }
        catch (const sv::SessionApiError&) { w.send(400,"application/json",sv::session_error("invalid_request","invalid session id").dump()); return; }
        job.session = session_id;
    }
    job.max_tokens = max_tokens;
    job.stop_tokens.push_back(tok_.eos_id);  // not = {eos}: gcc 14 -O3 reports a false -Warray-bounds on the one-element list
    const bool named_session = !job.session.empty();
    const std::size_t prompt_tokens = job.prompt.size();
    auto ticket = queue_.submit(std::move(job));
    // Armed instrumentation is decided once per request, by the queue at submission (GenerationTicket::observed: the
    // process-wide flag, one relaxed load); a refused request has no ticket and reads the flag itself. An observed
    // request gets exactly one ring record, written below once the ticket has finished and the delivery is known.
    std::optional<RequestRecord> rec;
    if (ticket ? ticket->observed() : observe_armed()) {
        rec.emplace();
        rec->request_id = inbound_id;
        rec->completion_id = id;
        rec->route = chat ? "chat.completions" : "completions";
        rec->time = created;
        rec->stream = stream;
        rec->prompt_tokens = prompt_tokens;
    }
    if (!ticket) {
        if (rec) {
            rec->status = "refused";
            rec->error = "queue full";
            observer_.record(std::move(*rec));
        }
        return send_error(w, 503, "queue full");
    }
    // Waits for the ticket's end (every path does, so the engine's statistics are final when recorded); a cancelled
    // ticket wakes wait() at once, so the loop sleeps instead of spinning until the engine finishes it.
    std::vector<std::uint32_t> ids, all_ids;
    bool delivered = true;  // every byte of the reply reached the client's socket
    auto drain = [&](auto&& on_token) {  // on_token(id) for every id but the stop token
        for (bool done = false; !done;) {
            const auto fresh = ticket->wait(std::chrono::milliseconds(100), done);
            for (const auto t : fresh) {
                if (want_ids) all_ids.push_back(t);
                if (t == tok_.eos_id) continue;
                ids.push_back(t);
                on_token(t);
            }
            if (stream && w.client_gone()) {
                delivered = false;
                ticket->cancel();
            }
            if (!done && fresh.empty() && ticket->cancelled()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    };
    // The scheduler's measurements of the finished ticket, taken once on every path (nothing is left in its table).
    auto take_extra = [&]() -> std::optional<RequestExtra> {
        return options_.take_extra ? options_.take_extra(*ticket) : std::nullopt;
    };
    // The single end of a request: the engine's final statistics, the delivery outcome, one record.
    auto finish_record = [&](const sv::GenerationStats& st, const std::string& err, const std::optional<RequestExtra>& x) {
        if (!rec) return;
        // the engine's longest emission gap, when the scheduler measured one (two or more tokens)
        if (x && st.generated_tokens >= 2) rec->max_gap_ms = x->max_gap_ms;
        rec->cached_tokens = st.cached_tokens;
        rec->output_tokens = st.generated_tokens;
        rec->queue_ms = st.queue_ms;
        rec->prefill_ms = st.prefill_ms;
        rec->decode_ms = st.decode_ms;
        rec->imported_tokens = st.imported_tokens;
        rec->first_token_ms = st.generated_tokens > 0 ? st.first_token_ms.value_or(-1.0) : -1.0;
        rec->draft = st.proposed > 0;
        rec->cycles = st.cycles;
        rec->proposed = st.proposed;
        rec->accepted = st.accepted;
        rec->stop_reason = err.empty() ? st.finish : std::string();
        rec->error = err;
        rec->status = !err.empty() ? "error" : (!delivered || st.finish == "cancelled") ? "cancelled" : "done";
        observer_.record(std::move(*rec));
    };
    auto chunk = [&](const std::string& delta, const char* finish) {
        sv::Json choice = sv::Json::object();
        choice.set("index", sv::Json(0));
        if (chat) {
            sv::Json d = sv::Json::object();
            if (!delta.empty()) d.set("content", sv::Json(delta));
            choice.set("delta", std::move(d));
        } else {
            choice.set("text", sv::Json(delta));
        }
        choice.set("finish_reason", finish ? sv::Json(finish) : sv::Json());
        sv::Json c = sv::Json::object();
        c.set("id", sv::Json(id));
        c.set("object", sv::Json(chat ? "chat.completion.chunk" : "text_completion"));
        c.set("created", sv::Json(created));
        c.set("model", sv::Json(options_.model_id));
        sv::Json arr = sv::Json::array();
        arr.as_array().push_back(std::move(choice));
        c.set("choices", std::move(arr));
        return c;
    };
    auto event = [](const sv::Json& c) { return "data: " + c.dump() + "\n\n"; };
    auto usage_json = [&](const sv::GenerationStats& st) {
        sv::Json usage = sv::Json::object();
        usage.set("prompt_tokens", sv::Json(static_cast<std::int64_t>(prompt_tokens)));
        usage.set("completion_tokens", sv::Json(static_cast<std::int64_t>(st.generated_tokens)));
        usage.set("total_tokens", sv::Json(static_cast<std::int64_t>(prompt_tokens + st.generated_tokens)));
        sv::Json details = sv::Json::object();
        details.set("cached_tokens", sv::Json(static_cast<std::int64_t>(st.cached_tokens)));
        usage.set("prompt_tokens_details", std::move(details));
        return usage;
    };
    auto finish_reason = [](const sv::GenerationStats& st) {
        return st.finish == "stop" ? "stop" : st.finish == "cancelled" ? "stop" : "length";
    };

    if (stream) {
        if (!w.begin_stream(200, "text/event-stream", {{"Cache-Control", "no-cache"}})) {
            ticket->cancel();
            delivered = false;
            drain([](std::uint32_t) {});
            finish_record(ticket->stats(), ticket->error(), take_extra());
            return;
        }
        sv::Utf8StreamDecoder utf8;
        auto write = [&](const std::string& data) {
            if (delivered && !w.write(data)) {
                delivered = false;
                ticket->cancel();
            }
        };
        drain([&](std::uint32_t t) {
            const std::string text = utf8.push(tok_.stream_bytes(t));
            if (!text.empty()) write(event(chunk(text, nullptr)));
        });
        const auto st = ticket->stats();
        const std::string err = ticket->error();
        const auto x = take_extra();
        const std::string tail = utf8.flush();
        if (!tail.empty()) write(event(chunk(tail, nullptr)));
        if (!err.empty()) {  // the headers are out: the error travels as an event
            sv::Json e = sv::Json::object();
            e.set("message", sv::Json(err));
            e.set("type", sv::Json("server_error"));
            sv::Json b = sv::Json::object();
            b.set("error", std::move(e));
            if (named_session) { const auto failure = sv::session_failure(err); write(event(sv::session_error(failure.code,err))); }
            else write(event(b));
        }
        auto last = chunk("", finish_reason(st));
        last.set("timings", timings(st, x));
        if (want_ids) last.set("strix", reply_extension(all_ids));
        write(event(last));
        if (include_usage) {
            sv::Json u = sv::Json::object();
            u.set("id", sv::Json(id));
            u.set("object", sv::Json(chat ? "chat.completion.chunk" : "text_completion"));
            u.set("created", sv::Json(created));
            u.set("model", sv::Json(options_.model_id));
            u.set("choices", sv::Json::array());
            u.set("usage", usage_json(st));
            if (want_ids) u.set("strix", reply_extension(all_ids));
            write(event(u));
        }
        write("data: [DONE]\n\n");
        // The reply is delivered when every event through [DONE] was written. The chunked-encoding terminator is
        // framing after the complete reply (a client may close once it has read [DONE]); it is skipped after a failed
        // write, so nothing is written once the client is gone.
        if (delivered) w.end_stream();
        finish_record(st, err, x);
        return;
    }

    drain([](std::uint32_t) {});
    const auto st = ticket->stats();
    const std::string err = ticket->error();
    const auto x = take_extra();
    if (!err.empty()) {
        if (named_session) { const auto failure = sv::session_failure(err); w.send(failure.status,"application/json",sv::session_error(failure.code,err).dump()); }
        else send_error(w, 400, err);
        finish_record(st, err, x);
        return;
    }
    const std::string text = tok_.decode(ids);
    sv::Json choice = sv::Json::object();
    choice.set("index", sv::Json(0));
    if (chat) {
        std::string reasoning, content;
        split_reasoning(text, reasoning, content);
        sv::Json m = sv::Json::object();
        m.set("role", sv::Json("assistant"));
        m.set("content", sv::Json(content));
        if (!reasoning.empty()) m.set("reasoning_content", sv::Json(reasoning));
        choice.set("message", std::move(m));
    } else {
        choice.set("text", sv::Json(text));
    }
    choice.set("finish_reason", sv::Json(finish_reason(st)));
    sv::Json out = sv::Json::object();
    out.set("id", sv::Json(id));
    out.set("object", sv::Json(chat ? "chat.completion" : "text_completion"));
    out.set("created", sv::Json(created));
    out.set("model", sv::Json(options_.model_id));
    sv::Json arr = sv::Json::array();
    arr.as_array().push_back(std::move(choice));
    out.set("choices", std::move(arr));
    out.set("usage", usage_json(st));
    out.set("timings", timings(st, x));
    if (want_ids) out.set("strix", reply_extension(all_ids));
    send_json(w, 200, out);
    finish_record(st, err, x);  // a non-streamed reply is one send(): its delivery is not observable here
}

void MinimaxApi::handle(const sv::HttpRequest& req, sv::HttpResponseWriter& w) {
    // An accepted inbound id is echoed on every response (the writer's response head carries it, whatever sends the
    // response); generation requests always echo one (see generate).
    std::string inbound = req.header("x-request-id");
    if (!valid_request_id(inbound)) inbound.clear();
    if (!inbound.empty()) w.add_response_header("X-Request-Id", inbound);
    try {
        if (req.method == "GET" && req.path == "/health") {
            sv::Json b = sv::Json::object();
            b.set("status", sv::Json(queue_.draining() ? "draining" : "ok"));
            sv::deployment_health(b, options_.deployment);
            return send_json(w, queue_.draining() ? 503 : 200, b);
        }
        if (req.method == "GET" && req.path == "/v1/models") {
            sv::Json m = sv::Json::object();
            m.set("id", sv::Json(options_.model_id));
            m.set("object", sv::Json("model"));
            m.set("context_length", sv::Json(static_cast<std::int64_t>(options_.context_tokens)));
            sv::Json arr = sv::Json::array();
            arr.as_array().push_back(std::move(m));
            sv::Json b = sv::Json::object();
            b.set("object", sv::Json("list"));
            b.set("data", std::move(arr));
            return send_json(w, 200, b);
        }
        // The first GET /metrics or GET /api/status?detail=1 arms the instrumentation (one-way).
        if (req.method == "GET" && req.path == "/metrics") {
            observe_arm();
            const auto q = queue_.snapshot();  // one snapshot for the legacy lines and the strix_* families
            const std::string legacy = options_.legacy_metrics ? options_.legacy_metrics(q) : legacy_metric_lines(q);
            return w.send(200, "text/plain; version=0.0.4", observer_.metrics(q, legacy));
        }
        if (req.method == "GET" && req.path == "/api/status") {
            const bool detail = query_flag(req.query, "detail");
            if (detail) observe_arm();
            auto status = observer_.status(queue_.snapshot(), queue_.stopping(), detail);
            sv::deployment_status(status, options_.deployment, queue_, options_.http_slots);
            return send_json(w, 200, status);
        }
        if (sv::handle_deployment_route(req, w, queue_, options_.deployment)) return;
        if (options_.admin_sched && req.path == "/admin/sched") return send_json(w, 200, options_.admin_sched(req));
        if (req.method != "POST") return send_error(w, 404, "no route " + req.method + " " + req.path);
        const sv::Json body = sv::parse_request_json(req.body);
        if (req.path == "/tokenize") {
            const auto* text = body.find("text");
            if (!text || !text->is_string()) return send_error(w, 400, "text required");
            sv::Json arr = sv::Json::array();
            for (auto t : tok_.encode(text->as_string())) arr.as_array().push_back(sv::Json(static_cast<std::int64_t>(t)));
            sv::Json b = sv::Json::object();
            b.set("tokens", std::move(arr));
            return send_json(w, 200, b);
        }
        if (req.path == "/v1/chat/completions") {
            const auto* messages = body.find("messages");
            if (!messages || !messages->is_array() || messages->as_array().empty()) return send_error(w, 400, "messages required");
            MinimaxChatOptions o;
            if (const auto* kw = body.find("chat_template_kwargs"); kw && kw->is_object()) {
                if (const auto* v = kw->find("thinking_mode"); v && v->is_string()) o.thinking_mode = v->as_string();
                if (const auto* v = kw->find("reasoning_effort"); v && v->is_string()) o.reasoning_effort = v->as_string();
            }
            if (const auto* v = body.find("reasoning_effort"); v && v->is_string()) o.reasoning_effort = v->as_string();
            // tool_choice "none" (string or {"type":"none"}): the tools are not offered, as in the shared server.
            const auto* choice = body.find("tool_choice");
            const auto* choice_type = choice && choice->is_object() ? choice->find("type") : choice;
            const bool tools_off = choice_type && choice_type->is_string() && choice_type->as_string() == "none";
            const auto* tools = body.find("tools");
            const std::string text = render_minimax_chat(*messages, tools && tools->is_array() && !tools_off ? tools : nullptr, o);
            return generate(req, w, tok_.encode(text), body, true, inbound);
        }
        if (req.path == "/v1/completions") {
            const auto* p = body.find("prompt");
            if (!p || !p->is_string()) return send_error(w, 400, "prompt (string) required");
            return generate(req, w, tok_.encode(p->as_string()), body, false, inbound);
        }
        return send_error(w, 404, "no route " + req.path);
    } catch (const sv::RequestFieldError& e) {
        return send_error(w, 400, e.what(), e.param);
    } catch (const std::exception& e) {
        return send_error(w, 400, e.what());
    }
}

}  // namespace strix::minimax::serve
