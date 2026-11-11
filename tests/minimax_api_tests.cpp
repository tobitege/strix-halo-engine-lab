// strix-minimax-serve HTTP API, host only: a scripted engine thread drains the GenerationQueue the way the serve's
// scheduler does (take, publish token by token, finish) and a recording response writer captures headers, bodies and
// stream events. Covers request-id echo, return_token_ids, include_usage, chunk order, every terminal path and the
// armed-only ring.
#include "minimax/serve/minimax_api.hpp"
#include "minimax/serve/minimax_observe.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/http_server.hpp"
#include "strix/serve/json.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace sv = strix::serve;
namespace mo = strix::minimax::serve;
using sv::Json;
using Headers = std::vector<std::pair<std::string, std::string>>;

int failures = 0, checks = 0;
void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::cerr << "FAIL " << what << "\n";
    }
}
const Json& at(const Json& j, std::string_view key) {
    static const Json missing;
    const auto* v = j.find(key);
    return v ? *v : missing;
}
std::string header(const Headers& h, std::string_view name) {
    for (const auto& [k, v] : h)
        if (k == name) return v;
    return {};
}
std::vector<std::int64_t> ints(const Json& arr) {
    std::vector<std::int64_t> out;
    if (arr.is_array())
        for (const auto& v : arr.as_array()) out.push_back(v.as_int());
    return out;
}

constexpr std::uint32_t kEos = 9;

mo::ApiTokenizer fake_tokenizer() {
    mo::ApiTokenizer t;
    t.eos_id = kEos;
    t.encode = [](std::string_view text) { return std::vector<std::uint32_t>{1, 2, static_cast<std::uint32_t>(text.size() % 1000U)}; };
    t.decode = [](std::span<const std::uint32_t> ids) {
        std::string s;
        for (const auto id : ids) s += "<" + std::to_string(id) + ">";
        return s;
    };
    t.stream_bytes = [](std::uint32_t id) {
        if (id == 300) return std::string("\xc3");  // the two halves of one UTF-8 character
        if (id == 301) return std::string("\xa9");
        return "<" + std::to_string(id) + ">";
    };
    return t;
}

// What the scripted engine does with the next job.
struct Script {
    Script(std::vector<std::uint32_t> t = {}, std::string e = {}, bool h = false, std::string f = "stop")
        : tokens(std::move(t)), error(std::move(e)), hold(h), finish(std::move(f)) {}
    std::vector<std::uint32_t> tokens;
    std::string error;    // finish with this error after the tokens
    bool hold;            // wait for release() before finishing
    std::string finish;   // when every token was published without a stop token
};

class FakeEngine {
public:
    explicit FakeEngine(sv::GenerationQueue& q) : q_(q), thread_([this] { run(); }) {}
    ~FakeEngine() {
        q_.shutdown();
        release();
        thread_.join();
    }
    void next(Script s) {
        const std::lock_guard<std::mutex> l(m_);
        script_ = std::move(s);
        released_ = false;
    }
    void release() {
        const std::lock_guard<std::mutex> l(m_);
        released_ = true;
        cv_.notify_all();
    }
    // Blocks until the engine has published n tokens in total (since it started).
    void wait_published(std::uint64_t n) {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [&] { return published_ >= n; });
    }
    [[nodiscard]] std::uint64_t published() {
        const std::lock_guard<std::mutex> l(m_);
        return published_;
    }
    // Blocks until the engine has finished n jobs in total (since it started).
    void wait_finished(std::uint64_t n) {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [&] { return finished_ >= n; });
    }
    [[nodiscard]] std::uint64_t finished() {
        const std::lock_guard<std::mutex> l(m_);
        return finished_;
    }
    std::string session() {
        const std::lock_guard<std::mutex> l(m_);
        return session_;
    }
    std::uint32_t max_tokens() {
        const std::lock_guard<std::mutex> l(m_);
        return max_tokens_;
    }
    std::vector<std::uint32_t> prompt() {
        const std::lock_guard<std::mutex> l(m_);
        return prompt_;
    }
    // As the serve scheduler: per-request extras inserted before the ticket finishes, taken by the API (hook).
    void keep_extras(bool on) {
        const std::lock_guard<std::mutex> l(m_);
        keep_extras_ = on;
    }
    std::optional<mo::RequestExtra> take_extra(const sv::GenerationTicket& t) {
        const std::lock_guard<std::mutex> l(m_);
        const auto it = extras_.find(&t);
        if (it == extras_.end()) return std::nullopt;
        const auto x = it->second;
        extras_.erase(it);
        return x;
    }
    [[nodiscard]] std::size_t extras_left() {
        const std::lock_guard<std::mutex> l(m_);
        return extras_.size();
    }

private:
    void run() {
        for (;;) {
            auto t = q_.take(true);
            if (!t) {
                if (q_.stopping()) break;
                continue;  // a cancellation-only batch returns control without shutting down
            }
            Script s;
            {
                const std::lock_guard<std::mutex> l(m_);
                s = script_;
                session_ = t->job.session;
                max_tokens_ = t->job.max_tokens;
                prompt_ = t->job.prompt;
            }
            sv::GenerationStats st;
            st.prompt_tokens = t->job.prompt.size();
            st.prefill_ms = 1.5;
            const auto decode_start = std::chrono::steady_clock::now();
            std::string why = s.finish;
            for (const auto id : s.tokens) {
                ++st.generated_tokens;
                const bool more = t->job.on_tokens(std::span(&id, 1));
                {
                    const std::lock_guard<std::mutex> l(m_);
                    ++published_;
                    cv_.notify_all();
                }
                if (id == kEos) {
                    why = "stop";
                    break;
                }
                if (!more) {
                    why = "cancelled";
                    break;
                }
                if (st.generated_tokens >= t->job.max_tokens) {
                    why = "length";
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (s.hold) {
                std::unique_lock<std::mutex> l(m_);
                cv_.wait(l, [&] { return released_; });
            }
            st.finish = why;
            st.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - decode_start).count();
            // as the serve scheduler: the first-token time only while armed
            if (t->job.observed && st.generated_tokens > 0)
                st.first_token_ms = std::chrono::duration<double, std::milli>(decode_start - t->submitted).count();
            {
                const std::lock_guard<std::mutex> l(m_);
                if (keep_extras_) {
                    mo::RequestExtra x;
                    x.ttft_ms = 12.5;
                    x.max_gap_ms = 3.25;
                    x.stall_ms = 1.0;
                    x.units = 2;
                    x.cycles = 4;
                    x.drafted = 6;
                    x.accepted = 3;
                    x.ids_fnv = 0xabcU;
                    x.prompt_fnv = 0x1234U;
                    x.sched = "interleave";
                    x.service_ms = 250;
                    x.max_unit_layers = 7;
                    x.short_first = true;
                    st.cycles = x.cycles;
                    st.proposed = x.drafted;
                    st.accepted = x.accepted;
                    extras_[t->ticket.get()] = x;
                }
            }
            q_.finish(*t, st, s.error);
            {
                const std::lock_guard<std::mutex> l(m_);
                ++finished_;
                cv_.notify_all();
            }
        }
    }
    sv::GenerationQueue& q_;
    std::mutex m_;
    std::condition_variable cv_;
    Script script_;
    std::string session_;
    std::uint32_t max_tokens_ = 0;
    std::vector<std::uint32_t> prompt_;
    bool released_ = false;
    bool keep_extras_ = false;
    std::map<const sv::GenerationTicket*, mo::RequestExtra> extras_;
    std::uint64_t published_ = 0, finished_ = 0;
    std::thread thread_;
};

class Writer final : public sv::HttpResponseWriter {
public:
    int status = 0;
    std::string type, body;
    Headers headers;
    bool streamed = false, ended = false, fail_begin = false;
    int gone_after = -1;  // writes accepted before the client goes away (-1 = never); the next write fails
    bool gone_now = false;  // the client is gone from the start
    bool failed = false;    // a write failed
    std::function<void(std::size_t)> before_write;  // runs before write number n (0-based)
    int end_calls = 0;
    std::vector<std::string> writes;
    std::function<void()> before_begin;  // runs inside begin_stream (e.g. wait for the engine)

    void send(int s, std::string_view content_type, std::string_view b, const Headers& h) override {
        status = s;
        type = content_type;
        body = b;
        headers = head(h);
    }
    bool begin_stream(int s, std::string_view content_type, const Headers& h) override {
        if (before_begin) before_begin();
        if (fail_begin) return false;
        status = s;
        type = content_type;
        headers = head(h);
        streamed = true;
        return true;
    }
    bool write(std::string_view data) override {
        if (before_write) before_write(writes.size());
        if (gone_now || failed || (gone_after >= 0 && static_cast<int>(writes.size()) >= gone_after)) {
            failed = true;
            return false;
        }
        writes.emplace_back(data);
        return true;
    }
    bool fail_end = false;  // the chunked terminator cannot be written (the client closed after [DONE])
    void end_stream() override {
        ended = true;
        ++end_calls;
        if (fail_end) failed = true;
    }
    [[nodiscard]] bool client_gone() const override { return gone_now || failed; }
    // as the server's connection: headers added for this request go into the response head after the call's own
    Headers added;
    void add_response_header(std::string name, std::string value) override { added.emplace_back(std::move(name), std::move(value)); }
    [[nodiscard]] Headers head(const Headers& h) const {
        Headers out = h;
        out.insert(out.end(), added.begin(), added.end());
        return out;
    }

    // the JSON of each "data: {...}" event ([DONE] as a null)
    [[nodiscard]] std::vector<Json> events() const {
        std::vector<Json> out;
        for (const auto& w : writes) {
            check(w.rfind("data: ", 0) == 0 && w.size() >= 8U && w.compare(w.size() - 2U, 2U, "\n\n") == 0, "event framing");
            const auto payload = w.substr(6, w.size() - 8U);
            out.push_back(payload == "[DONE]" ? Json() : sv::parse_json(payload));
        }
        return out;
    }
};

sv::HttpRequest post(const std::string& path, const std::string& body, const std::string& request_id = {}) {
    sv::HttpRequest r;
    r.method = "POST";
    r.path = path;
    r.target = path;
    r.version = "HTTP/1.1";
    r.body = body;
    if (!request_id.empty()) r.headers.emplace_back("x-request-id", request_id);
    return r;
}
sv::HttpRequest get(const std::string& path, const std::string& query = {}) {
    sv::HttpRequest r;
    r.method = "GET";
    r.path = path;
    r.query = query;
    r.target = query.empty() ? path : path + "?" + query;
    r.version = "HTTP/1.1";
    return r;
}

const std::string kChat = R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":8)";

struct Rig {
    // hooks: the serve scheduler's side (extras from the fake engine, extra mm_* lines, /admin/sched)
    explicit Rig(bool with_hooks = false, bool with_sessions = false) : hooks(with_hooks), sessions(with_sessions) { engine.keep_extras(with_hooks); }
    bool hooks;
    bool sessions;
    sv::GenerationQueue queue{8};
    mo::Observer observer{[] {
        mo::ObserverInfo i;
        i.model = "test-model";
        i.slots = 8;
        i.context_tokens = 8256;
        i.instance_id = "00000000000000aa";
        return i;
    }()};
    mo::MinimaxApi api{fake_tokenizer(), queue, [this] {
                           mo::MinimaxApiOptions o;
                           o.session_enabled = sessions;
                           o.model_id = "test-model";
                           o.default_max_tokens = 16;
                           o.context_tokens = 8256;
                           if (hooks) {
                               o.take_extra = [this](const sv::GenerationTicket& t) { return engine.take_extra(t); };
                               o.legacy_metrics = [](const sv::QueueSnapshot& q) {
                                   return mo::legacy_metric_lines(q) + "mm_step_seconds 1.500\nmm_prefill_units 2\nmm_accepted 7\n";
                               };
                               o.admin_sched = [](const sv::HttpRequest& r) {
                                   if (r.method == "POST" && r.body != "{}") throw std::invalid_argument("sched: off | interleave");
                                   sv::Json b = sv::Json::object();
                                   b.set("sched", sv::Json("off"));
                                   return b;
                               };
                           }
                           return o;
                       }(),
                       observer};
    FakeEngine engine{queue};

    Json status(bool detail = false) {
        Writer w;
        api.handle(get("/api/status", detail ? "detail=1" : ""), w);
        check(w.status == 200, "status 200");
        return sv::parse_json(w.body);
    }
};

void test_unarmed_non_stream() {
    mo::observe_reset_for_tests();
    Rig rig;
    rig.engine.next({{5, 6, kEos}});
    Writer w;
    rig.api.handle(post("/v1/chat/completions", kChat + R"(,"return_token_ids":true})"), w);
    const auto j = sv::parse_json(w.body);
    check(w.status == 200, "non-stream 200");
    check(ints(at(at(j, "strix"), "token_ids")) == std::vector<std::int64_t>{5, 6, 9}, "token ids incl. the stop token");
    check(at(at(j, "usage"), "completion_tokens").as_int() == 3, "completion tokens == len(token_ids)");
    check(at(at(at(j, "usage"), "prompt_tokens_details"), "cached_tokens").as_int() == 0, "cached tokens detail");
    check(at(at(j, "choices").as_array()[0], "message").find("content") != nullptr, "message content");
    check(at(at(at(j, "choices").as_array()[0], "message"), "content").as_string() == "<5><6>", "text without the stop token");
    check(header(w.headers, "X-Request-Id") == at(j, "id").as_string(), "echo = completion id without an inbound id");
    const auto s = rig.status();
    check(!at(s, "armed").as_bool(), "plain status does not arm");
    check(at(at(s, "instance"), "id").as_string() == "00000000000000aa" && at(at(s, "instance"), "started_at").as_int() > 0 &&
              at(at(rig.status(), "instance"), "started_at").as_int() == at(at(s, "instance"), "started_at").as_int(),
          "instance identity present unarmed and stable");
    check(at(s, "requests").size() == 0U, "no ring record while unarmed");
    check(at(at(s, "totals"), "requests").as_int() == 1 && at(at(s, "totals"), "generated_tokens").as_int() == 3, "totals from the queue");
    // without return_token_ids: no strix object
    rig.engine.next({{5, kEos}});
    Writer w2;
    rig.api.handle(post("/v1/chat/completions", kChat + "}", "abcdef12.1"), w2);
    check(sv::parse_json(w2.body).find("strix") == nullptr, "no strix object unless asked");
    check(header(w2.headers, "X-Request-Id") == "abcdef12.1", "inbound id echoed");
    auto count_ids = [](const Headers& h) {
        std::size_t n = 0;
        for (const auto& kv : h) n += kv.first == "X-Request-Id" ? 1U : 0U;
        return n;
    };
    check(count_ids(w.headers) == 1U && count_ids(w2.headers) == 1U && w2.added.size() == 1U, "one X-Request-Id, in the response head");
    Writer w3;
    rig.engine.next({{5, kEos}});
    rig.api.handle(post("/v1/chat/completions", kChat + "}", "bad id!"), w3);
    check(header(w3.headers, "X-Request-Id").rfind("chatcmpl-", 0) == 0, "invalid inbound id ignored");
    Writer w4;
    rig.api.handle(post("/v1/chat/completions", kChat + R"(,"return_token_ids":"yes"})", "abcdef12.2"), w4);
    check(w4.status == 400 && header(w4.headers, "X-Request-Id") == "abcdef12.2", "non-boolean option is a 400, id echoed");
    Writer w5;
    rig.api.handle(get("/health"), w5);
    check(w5.status == 200 && w5.headers.empty(), "health without an inbound id has no echo");
    Writer w6;
    rig.api.handle(post("/v1/chat/completions", "{not json", "abcdef12.3"), w6);
    check(w6.status == 400 && header(w6.headers, "X-Request-Id") == "abcdef12.3", "malformed JSON: 400 with the inbound id");
}

void test_armed_stream() {
    mo::observe_reset_for_tests();
    Rig rig;
    Writer m;
    rig.api.handle(get("/metrics"), m);
    check(m.status == 200 && m.type == "text/plain; version=0.0.4" && mo::observe_armed(), "GET /metrics arms");
    rig.engine.next({{5, 6, 7}, "", false, "length"});
    Writer w;
    rig.api.handle(post("/v1/chat/completions",
                        R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":3,"stream":true,"stream_options":{"include_usage":true},"return_token_ids":true})",
                        "0123456789abcdef.1"),
                   w);
    check(w.streamed && w.ended && w.type == "text/event-stream", "stream");
    check(header(w.headers, "X-Request-Id") == "0123456789abcdef.1", "stream echoes the inbound id");
    const auto ev = w.events();
    check(ev.size() == 6U, "3 content + final + usage + [DONE]");
    if (ev.size() == 6U) {
        for (int i = 0; i < 3; ++i) {
            const auto& c = at(ev[static_cast<std::size_t>(i)], "choices").as_array()[0];
            check(at(at(c, "delta"), "content").as_string() == "<" + std::to_string(5 + i) + ">" && at(c, "finish_reason").is_null(), "content chunk");
        }
        const auto& fin = ev[3];
        check(at(at(fin, "choices").as_array()[0], "finish_reason").as_string() == "length", "final chunk finish_reason");
        check(ints(at(at(fin, "strix"), "token_ids")) == std::vector<std::int64_t>{5, 6, 7}, "final chunk token ids");
        check(at(ev[4], "choices").size() == 0U && at(at(ev[4], "usage"), "completion_tokens").as_int() == 3, "usage chunk");
        check(ints(at(at(ev[4], "strix"), "token_ids")) == std::vector<std::int64_t>{5, 6, 7}, "usage chunk token ids");
        check(at(ev[4], "object").as_string() == "chat.completion.chunk", "usage chunk object");
        check(ev[5].is_null(), "[DONE] last");
    }
    const auto s = rig.status(true);
    const auto& list = at(s, "requests");
    check(list.size() == 1U, "armed request recorded");
    if (list.size() == 1U) {
        const auto& r = list.as_array()[0];
        check(at(r, "id").as_string() == "0123456789abcdef.1" && at(r, "request_id").as_string() == "0123456789abcdef.1", "ring id");
        check(at(r, "completion_id").as_string().rfind("chatcmpl-", 0) == 0, "completion id kept apart");
        check(at(r, "status").as_string() == "done" && at(r, "stop_reason").as_string() == "length", "status");
        check(at(r, "stream").as_bool() && at(r, "output_tokens").as_int() == 3, "stream record");
        check(at(r, "first_token_ms").is_number() && at(r, "first_token_ms").as_double() >= 0.0, "first_token_ms measured");
        check(at(r, "computed_tokens").as_int() == 3, "computed tokens");
    }
    check(at(at(at(s, "latency"), "ttft_seconds"), "count").as_int() == 1, "ttft histogram counted");
    // completions route, stream without usage
    rig.engine.next({{5, kEos}});
    Writer c;
    rig.api.handle(post("/v1/completions", R"({"prompt":"x","stream":true,"return_token_ids":true})"), c);
    const auto cev = c.events();
    check(cev.size() == 3U, "completions stream: content + final + [DONE]");
    if (cev.size() == 3U) {
        check(at(cev[1], "object").as_string() == "text_completion", "completions object");
        check(ints(at(at(cev[1], "strix"), "token_ids")) == std::vector<std::int64_t>{5, 9}, "completions final ids incl. stop");
    }
    // first published token is the stop token
    rig.engine.next({{kEos}});
    Writer e;
    rig.api.handle(post("/v1/chat/completions", kChat + "}"), e);
    const auto st = rig.status();
    const auto& r0 = at(st, "requests").as_array()[0];
    check(at(r0, "output_tokens").as_int() == 1 && at(r0, "first_token_ms").is_number(), "stop token as first token is measured");
}

void test_terminal_paths() {
    mo::observe_reset_for_tests();
    mo::observe_arm();
    Rig rig;
    // engine error, non-stream: 400 with the id, error record
    rig.engine.next({{5}, "engine failure"});
    Writer a;
    rig.api.handle(post("/v1/chat/completions", kChat + "}", "abcdefgh"), a);
    check(a.status == 400 && header(a.headers, "X-Request-Id") == "abcdefgh", "engine error: 400");
    auto s = rig.status();
    auto r = at(s, "requests").as_array()[0];
    check(at(r, "status").as_string() == "error" && at(r, "error").as_string() == "engine failure" && at(r, "stop_reason").as_string().empty(),
          "error record");
    // engine error, stream: an error event before the final chunk, no second response
    rig.engine.next({{5}, "engine failure"});
    Writer b;
    rig.api.handle(post("/v1/chat/completions", kChat + R"(,"stream":true})"), b);
    const auto ev = b.events();
    check(ev.size() == 4U && at(at(ev[1], "error"), "message").as_string() == "engine failure", "stream error event");
    check(b.status == 200 && b.body.empty(), "no second response after the stream started");
    // stream start fails: cancelled record
    rig.engine.next({{5, 6, kEos}});
    Writer c;
    c.fail_begin = true;
    rig.api.handle(post("/v1/chat/completions", kChat + R"(,"stream":true})", "abcdefgh.2"), c);
    s = rig.status();
    r = at(s, "requests").as_array()[0];
    check(at(r, "id").as_string() == "abcdefgh.2" && at(r, "status").as_string() == "cancelled", "failed stream start recorded once");
    // client gone mid-stream: cancelled, bounded time
    rig.engine.next({{5, 6, 7, 8, 10, 11, 12, 13}});
    Writer d;
    d.gone_after = 1;
    const auto t0 = std::chrono::steady_clock::now();
    rig.api.handle(post("/v1/chat/completions", kChat + R"(,"stream":true})", "abcdefgh.3"), d);
    check(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5), "client gone: returns");
    s = rig.status();
    r = at(s, "requests").as_array()[0];
    check(at(r, "id").as_string() == "abcdefgh.3" && at(r, "status").as_string() == "cancelled", "client gone: cancelled");
    std::size_t records = at(s, "requests").size();
    check(records == 4U, "one record per request (4 so far)");
    // queue refusal
    {
        Rig closed;
        closed.queue.shutdown();
        Writer q;
        closed.api.handle(post("/v1/chat/completions", kChat + "}", "abcdefgh.4"), q);
        check(q.status == 503 && header(q.headers, "X-Request-Id") == "abcdefgh.4", "queue full: 503 with the id");
        const auto cs = closed.status();
        check(at(at(cs, "requests").as_array()[0], "status").as_string() == "refused", "refused record");
        check(at(cs, "state").as_string() == "stopping", "stopping state");
    }
}

void test_arming_mid_request() {
    mo::observe_reset_for_tests();
    Rig rig;
    rig.engine.next({{5, 6, kEos}, "", true});
    Writer w;
    std::thread client([&] { rig.api.handle(post("/v1/chat/completions", kChat + "}"), w); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    Writer m;
    rig.api.handle(get("/api/status", "detail=1"), m);
    check(mo::observe_armed(), "status?detail=1 arms");
    rig.engine.release();
    client.join();
    const auto s = rig.status(true);
    check(w.status == 200, "request finished");
    check(at(s, "requests").size() == 0U, "a request submitted before arming is not recorded");
    check(at(at(at(s, "latency"), "ttft_seconds"), "count").as_int() == 0, "nor counted");
}


void test_delivery_outcomes() {
    mo::observe_reset_for_tests();
    mo::observe_arm();
    Rig rig;
    const auto base = rig.engine.published();
    // tokens published before the stream start fails: the record carries the engine's final statistics
    rig.engine.next({{5, 6, 7, kEos}});
    Writer a;
    a.fail_begin = true;
    a.before_begin = [&] { rig.engine.wait_published(base + 4); };
    rig.api.handle(post("/v1/chat/completions", kChat + R"(,"stream":true})", "deliver01"), a);
    auto r = at(rig.status(), "requests").as_array()[0];
    check(at(r, "id").as_string() == "deliver01" && at(r, "status").as_string() == "cancelled", "failed start after publication: cancelled");
    check(at(r, "output_tokens").as_int() == 4 && at(r, "first_token_ms").is_number(), "failed start: final engine statistics recorded");
    // the engine finished (confirmed before the write), then the final chunk could not be written: cancelled, full output
    {
        const auto done_before = rig.engine.finished();
        rig.engine.next({{5, 6, 7, kEos}});
        Writer b;
        b.gone_after = 3;  // three content chunks reach the client, the final chunk does not
        b.before_write = [&](std::size_t n) {
            if (n == 3U) rig.engine.wait_finished(done_before + 1);
        };
        rig.api.handle(post("/v1/chat/completions", kChat + R"(,"stream":true})", "deliver02"), b);
        r = at(rig.status(), "requests").as_array()[0];
        check(at(r, "id").as_string() == "deliver02" && at(r, "status").as_string() == "cancelled" && at(r, "stop_reason").as_string() == "stop",
              "final event failure after the engine finished: cancelled, engine stop reason kept");
        check(at(r, "output_tokens").as_int() == 4 && b.writes.size() == 3U, "final event failure: full output, no write after the failure");
        check(b.end_calls == 0, "no stream terminator after a failed write");
    }
    // a complete stream ends with exactly one terminator
    {
        rig.engine.next({{5, kEos}});
        Writer ok;
        rig.api.handle(post("/v1/chat/completions", kChat + R"(,"stream":true})", "deliver06"), ok);
        r = at(rig.status(), "requests").as_array()[0];
        check(ok.end_calls == 1 && at(r, "status").as_string() == "done", "delivered stream: one terminator, done");
    }
    // the terminator after [DONE] fails: the reply was complete, so the request is done
    {
        rig.engine.next({{5, kEos}});
        Writer late;
        late.fail_end = true;
        rig.api.handle(post("/v1/chat/completions", kChat + R"(,"stream":true})", "deliver07"), late);
        r = at(rig.status(), "requests").as_array()[0];
        check(late.end_calls == 1 && late.events().back().is_null() && at(r, "status").as_string() == "done",
              "terminator failure after [DONE]: done");
    }
    // an error before any publication: null first token
    rig.engine.next({{}, "no device"});
    Writer c;
    rig.api.handle(post("/v1/chat/completions", kChat + "}", "deliver03"), c);
    r = at(rig.status(), "requests").as_array()[0];
    check(c.status == 400 && at(r, "status").as_string() == "error" && at(r, "first_token_ms").is_null() && at(r, "output_tokens").as_int() == 0,
          "error before publication");
    // cancelled while queued: the engine is busy with a held request; the second client goes away at once
    rig.engine.next({{5, kEos}, "", true});
    const auto before = rig.engine.published();
    Writer held;
    std::thread first([&] { rig.api.handle(post("/v1/chat/completions", kChat + "}", "deliver04"), held); });
    rig.engine.wait_published(before + 2);  // the first request is in the engine, held before its finish
    Writer gone;
    gone.gone_now = true;
    std::thread releaser([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        rig.engine.release();
    });
    rig.api.handle(post("/v1/chat/completions", kChat + R"(,"stream":true})", "deliver05"), gone);
    first.join();
    releaser.join();
    const auto s = rig.status();
    bool found = false;
    for (const auto& x : at(s, "requests").as_array())
        if (at(x, "id").as_string() == "deliver05") {
            found = true;
            check(at(x, "status").as_string() == "cancelled" && at(x, "output_tokens").as_int() == 0 && at(x, "first_token_ms").is_null(),
                  "cancelled while queued: no token, null first token");
        }
    check(found, "queued cancellation recorded");
    check(at(at(s, "queue"), "cancelled").as_int() >= 1, "queue counted the cancellation");
}

void test_options_and_utf8() {
    mo::observe_reset_for_tests();
    Rig rig;
    Writer a;
    rig.api.handle(post("/v1/chat/completions", kChat + R"(,"stream_options":{"include_usage":"yes"}})"), a);
    check(a.status == 400, "include_usage validated without stream");
    for (const bool chat : {true, false})
        for (const bool usage : {true, false}) {
            rig.engine.next({{300, 301, kEos}});
            Writer w;
            std::string body = chat ? R"({"messages":[{"role":"user","content":"hi"}],"stream":true)" : R"({"prompt":"x","stream":true)";
            if (usage) body += R"(,"stream_options":{"include_usage":true})";
            body += "}";
            rig.api.handle(post(chat ? "/v1/chat/completions" : "/v1/completions", body), w);
            const auto ev = w.events();
            const std::size_t expect = usage ? 4U : 3U;  // one content chunk (the split character), final, [usage], [DONE]
            check(ev.size() == expect, std::string("event count ") + (chat ? "chat" : "completions") + (usage ? " usage" : ""));
            if (ev.size() == expect) {
                const auto& c0 = at(ev[0], "choices").as_array()[0];
                const std::string text = chat ? at(at(c0, "delta"), "content").as_string() : at(c0, "text").as_string();
                check(text == "\xc3\xa9", "split UTF-8 character delivered whole");
                check(ev[1].find("strix") == nullptr, "no ids unless asked");
                if (usage) check(at(ev[2], "choices").size() == 0U && ev[2].find("usage") != nullptr, "usage chunk");
            }
        }
}
// The serve scheduler's side: its per-request extras in the reply's timings (non-stream and the stream's final chunk)
// and in the ring (max_gap_ms), taken once on every path; its mm_* lines with HELP/TYPE; /admin/sched.
void test_scheduler_side() {
    mo::observe_reset_for_tests();
    mo::observe_arm();
    Rig rig(true);
    const std::vector<std::string> keys{"queue_ms", "prefill_ms", "decode_ms", "decode_tok_s", "completion_tokens", "ttft_ms",
                                        "max_gap_ms", "stall_ms", "prefill_units", "cycles", "drafted", "accepted", "ids_fnv",
                                        "prompt_fnv", "sched", "effective_sched", "service_ms", "max_unit_layers", "task_id", "nested", "parent_request", "nest_count", "nest_service_ms", "nest_switch_ms", "park_wall_ms", "short_first", "fallback_reason", "dspark_mode", "dspark_reason", "dspark_context", "sampling"};
    auto key_order = [](const Json& t) {
        std::vector<std::string> out;
        if (t.is_object())
            for (const auto& m : t.as_object()) out.push_back(m.first);
        return out;
    };
    rig.engine.next({{5, 6, kEos}});
    Writer a;
    rig.api.handle(post("/v1/chat/completions", kChat + "}", "sched0001"), a);
    const auto j = sv::parse_json(a.body);
    const auto& t = at(j, "timings");
    check(at(t, "dspark_mode").as_string() == "auto" && at(t, "dspark_reason").as_string() == "plain:prior_unqualified" && !at(t, "dspark_context").as_bool(), "unqualified auto timings");
    check(key_order(t) == keys, "non-stream timings: the scheduler's keys in order");
    check(at(t, "ttft_ms").as_double() == 12.5 && at(t, "max_gap_ms").as_double() == 3.25 && at(t, "prefill_units").as_int() == 2 &&
              at(t, "drafted").as_int() == 6 && at(t, "completion_tokens").as_int() == 3,
          "non-stream timings: the scheduler's values");
    check(at(t, "ids_fnv").as_string() == "0000000000000abc" && at(t, "prompt_fnv").as_string() == "0000000000001234" &&
              at(t, "sampling").as_string() == "greedy",
          "non-stream timings: digests as 16 hex");
    check(at(t, "sched").as_string() == "interleave" && at(t, "effective_sched").as_string() == "interleave" &&
          at(t, "service_ms").as_double() == 250 && at(t, "max_unit_layers").as_int() == 7 &&
          at(t, "short_first").as_bool() && at(t, "fallback_reason").as_string().empty(), "scheduler policy receipt");
    mo::RequestExtra forced;
    forced.sched = "interleave"; forced.fallback_reason = "fp32_prefill";
    check(at(mo::reply_timings({}, forced), "effective_sched").as_string() == "unlimited", "forced-unlimited receipt");
    auto r = at(rig.status(), "requests").as_array()[0];
    check(at(r, "max_gap_ms").as_double() == 3.25, "ring max_gap_ms from the scheduler");
    check(at(at(r, "draft"), "proposed").as_int() == 6 && at(r, "mtp_accepted").as_int() == 3, "ring draft counts from the stats");
    check(rig.engine.extras_left() == 0U, "extras taken (non-stream)");
    mo::RequestExtra nested;
    nested.nested = true; nested.parent_request = 41;
    const auto nested_timings = mo::reply_timings({}, nested);
    check(at(nested_timings, "nested").as_int() == 1 && at(nested_timings, "parent_request").as_int() == 41,
          "nested receipt retains parent identity");
    // stream: timings on the final chunk only
    rig.engine.next({{5, 6, kEos}});
    Writer b;
    rig.api.handle(post("/v1/chat/completions", kChat + R"(,"stream":true,"return_token_ids":true})", "sched0002"), b);
    const auto ev = b.events();
    check(ev.size() == 4U, "stream: 2 content + final + [DONE]");
    if (ev.size() == 4U) {
        check(ev[0].find("timings") == nullptr && ev[1].find("timings") == nullptr, "no timings on content chunks");
        check(key_order(at(ev[2], "timings")) == keys && at(at(ev[2], "timings"), "max_gap_ms").as_double() == 3.25, "final chunk timings");
        check(key_order(ev[2]).back() == "strix", "final chunk: timings then the token ids");
    }
    check(rig.engine.extras_left() == 0U, "extras taken (stream)");
    // one token: no gap measured
    rig.engine.next({{kEos}});
    Writer c;
    rig.api.handle(post("/v1/chat/completions", kChat + "}", "sched0003"), c);
    r = at(rig.status(), "requests").as_array()[0];
    check(at(r, "max_gap_ms").is_null(), "one token: max_gap_ms null");
    // failed stream start: the extras are still taken (nothing left behind)
    rig.engine.next({{5, kEos}});
    Writer d;
    d.fail_begin = true;
    rig.api.handle(post("/v1/chat/completions", kChat + R"(,"stream":true})", "sched0004"), d);
    check(rig.engine.extras_left() == 0U, "extras taken (failed stream start)");
    // the scheduler's mm_* lines: value lines unchanged, HELP/TYPE from the table, one snapshot
    Writer m;
    rig.api.handle(get("/metrics"), m);
    check(m.body.find("# HELP mm_step_seconds ") != std::string::npos &&
              m.body.find("# TYPE mm_step_seconds counter\nmm_step_seconds 1.500\n") != std::string::npos,
          "scheduler line with HELP/TYPE");
    check(m.body.find("# TYPE mm_prefill_units counter\nmm_prefill_units 2\n") != std::string::npos &&
              m.body.find("# TYPE mm_accepted counter\nmm_accepted 7\n") != std::string::npos,
          "scheduler counters typed");
    check(m.body.find("untyped") == std::string::npos, "every mm_* line has a table entry");
    auto value = [&](const std::string& name) {  // the value of an unlabelled sample, -1 when absent
        const auto at_line = m.body.find("\n" + name + " ");
        return at_line == std::string::npos ? -1.0 : std::stod(m.body.substr(at_line + name.size() + 2U));
    };
    check(value("mm_completed") >= 3.0 && value("mm_completed") == value("strix_requests_completed_total"),
          "queue lines and strix_* families from the same snapshot");
    check(value("strix_ttft_seconds_count") >= 3.0, "armed histogram beside the legacy lines");
    // /admin/sched through the hook; errors are 400 with the inbound id
    Writer g;
    rig.api.handle(get("/admin/sched"), g);
    check(g.status == 200 && at(sv::parse_json(g.body), "sched").as_string() == "off", "GET /admin/sched");
    Writer p;
    rig.api.handle(post("/admin/sched", R"({"sched":"bad"})", "sched0005"), p);
    check(p.status == 400 && header(p.headers, "X-Request-Id") == "sched0005", "bad POST /admin/sched: 400, id echoed");
    // without the hook there is no such route
    Rig plain;
    Writer n;
    plain.api.handle(get("/admin/sched"), n);
    check(n.status == 404, "no /admin/sched without the scheduler's hook");
}
void test_sessions() {
    for (bool enabled : {false, true}) {
        Rig rig(false, enabled);
        for (const auto& value : {std::string("123"), std::string("\"bad.id\""), std::string("\"\"")}) {
            rig.engine.next({{5, kEos}});
            Writer w;
            rig.api.handle(post("/v1/chat/completions", kChat + ",\"session\":" + value + "}"), w);
            check(w.status == (enabled ? 400 : 200), "session validation follows admission");
            if (enabled) check(at(at(sv::parse_json(w.body), "error"), "code").as_string() == "invalid_request", "invalid session error code");
            else check(rig.engine.session().empty(), "disabled session field never reaches scheduler");
        }
        rig.engine.next({{5, kEos}});
        Writer named;
        rig.api.handle(post("/v1/completions", R"({"prompt":"hi","session":"conversation","max_tokens":2})"), named);
        check(named.status == 200 && rig.engine.session() == (enabled ? "conversation" : ""), "named completion admission");
        rig.engine.next({{}, "session_budget: capture peak refused"});
        Writer budget;
        rig.api.handle(post("/v1/chat/completions", kChat + R"(,"session":"conversation"})"), budget);
        check(budget.status == (enabled ? 507 : 400), "named session budget HTTP status");
        if (enabled) check(at(at(sv::parse_json(budget.body), "error"), "code").as_string() == "session_budget", "named session budget error code");
    }
}
}  // namespace

// Generation fields: unsupported non-neutral values are refused with a 400 naming the field before any submission; a
// positive temperature runs greedy and says so; max_tokens is a positive integer read without wrapping.
void test_request_fields() {
    mo::observe_reset_for_tests();
    Rig rig;
    const auto refused = [&](const std::string& extra, const std::string& param) {
        Writer w;
        rig.api.handle(post("/v1/chat/completions", kChat + "," + extra + "}"), w);
        const auto j = sv::parse_json(w.body);
        check(w.status == 400 && at(at(j, "error"), "param").as_string() == param, "refused with its field: " + extra);
    };
    refused(R"("stop":["\n\n"])", "stop");
    refused(R"("stop":"END")", "stop");
    refused(R"("n":2)", "n");
    refused(R"("logprobs":true)", "logprobs");
    refused(R"("top_logprobs":3)", "top_logprobs");
    refused(R"("response_format":{"type":"json_object"})", "response_format");
    refused(R"("presence_penalty":0.5)", "presence_penalty");
    refused(R"("logit_bias":{"5":-100})", "logit_bias");
    refused(R"("tool_choice":"required")", "tool_choice");
    refused(R"("temperature":3)", "temperature");
    refused(R"("temperature":"hot")", "temperature");
    refused(R"("top_p":0)", "top_p");
    refused(R"("seed":-1)", "seed");
    refused(R"("max_completion_tokens":0)", "max_completion_tokens");
    // exact integers: read from the spelling, never through a double
    refused(R"("max_completion_tokens":1.00000000000000001)", "max_completion_tokens");
    refused(R"("max_completion_tokens":0.99999999999999999)", "max_completion_tokens");
    refused(R"("max_completion_tokens":9007199254740992.5)", "max_completion_tokens");
    refused(R"("top_k":4294967296)", "top_k");
    refused(R"("seed":9223372036854775808)", "seed");
    refused(R"("n":1.5)", "n");
    refused(R"("logprobs":1e400)", "logprobs");
    // the shared refusals (same checks as the Qwen/DeepSeek server)
    refused(R"("tool_choice":{"type":"function","function":{"name":"f"}})", "tool_choice");
    refused(R"("tool_choice":{"type":"auto","disable_parallel_tool_use":true})", "tool_choice.disable_parallel_tool_use");
    refused(R"("tools":[{"type":"function","function":{"name":"f","strict":true}}])", "tools[0].function.strict");
    refused(R"("prefill_disable":["x"])", "prefill_disable");
    refused(R"("frequency_penalty":0.5)", "frequency_penalty");
    refused(R"("stream":true,"stop":"END")", "stop");
    {
        Writer w;  // max_tokens sits in kChat, so the overriding field is max_completion_tokens
        rig.api.handle(post("/v1/chat/completions", R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":-1})"), w);
        check(w.status == 400 && at(at(sv::parse_json(w.body), "error"), "param").as_string() == "max_tokens", "negative max_tokens refused");
        Writer w2;
        rig.api.handle(post("/v1/chat/completions", R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":1.5})"), w2);
        check(w2.status == 400, "fractional max_tokens refused");
    }
    check(rig.engine.finished() == 0U, "refused requests never reach the engine");
    const auto accepted = [&](const std::string& extra, std::size_t n) {
        rig.engine.next({{5, kEos}});
        Writer w;
        rig.api.handle(post("/v1/chat/completions", kChat + "," + extra + "}"), w);
        rig.engine.wait_finished(n);
        check(w.status == 200, "neutral values accepted: " + extra);
        return sv::parse_json(w.body);
    };
    auto j = accepted(R"("stop":null,"n":1,"logprobs":false,"presence_penalty":0,"logit_bias":{},"response_format":{"type":"text"},"tool_choice":"auto","seed":7)", 1);
    check(at(j, "timings").find("sampling_note") == nullptr, "no note when nothing was ignored");
    j = accepted(R"("temperature":0.7,"top_p":0.9)", 2);
    check(at(at(j, "timings"), "sampling_note").as_string() == "temperature ignored: this engine decodes greedy (exact)" &&
              at(at(j, "timings"), "sampling").as_string() == "greedy", "positive temperature runs greedy and says so");
    j = accepted(R"("top_p":0.9)", 3);
    check(at(at(j, "timings"), "sampling_note").as_string() == "filters ignored at temperature 0 (greedy)", "filters note at temperature 0");
    (void)accepted(R"("max_completion_tokens":4294967296)", 4);
    check(rig.engine.max_tokens() == 8256U, "2^32 max_tokens is clamped to the context, not wrapped to 0");
    (void)accepted(R"("max_completion_tokens":5)", 5);
    check(rig.engine.max_tokens() == 5U, "max_completion_tokens wins over max_tokens");
    (void)accepted(R"("max_completion_tokens":null)", 6);
    check(rig.engine.max_tokens() == 8U, "a null max_completion_tokens falls back to max_tokens");
    (void)accepted(R"("max_completion_tokens":1e3)", 7);
    check(rig.engine.max_tokens() == 1000U, "1e3 is the integer 1000");
    (void)accepted(R"("max_completion_tokens":1.0,"n":1.0,"top_k":40)", 8);
    check(rig.engine.max_tokens() == 1U, "1.0 is the integer 1");
    // tool_choice none: the tools are not offered (same prompt as without tools)
    (void)accepted(R"("tool_choice":"none","tools":[{"type":"function","function":{"name":"f","parameters":{}}}])", 9);
    const auto without_offer = rig.engine.prompt();
    (void)accepted(R"("tools":[{"type":"function","function":{"name":"f","parameters":{}}}])", 10);
    const auto offered = rig.engine.prompt();
    (void)accepted(R"("tool_choice":{"type":"none"})", 11);
    check(without_offer == rig.engine.prompt() && offered != without_offer, "tool_choice none renders no tools");
}

void test_deployment_surface() {
    Rig rig;
    Writer absent; rig.api.handle(get("/v1/deployment"), absent);
    check(absent.status == 404, "no active deployment is 404");
    mo::MinimaxApiOptions options;
    options.deployment.deployment_json = [] { return R"({"schema":"strix-deployment-v1"})"; };
    options.deployment.deployment_id = [] { return "0123456789abcdef"; };
    options.deployment.capacity_provider = [] { return sv::ServeCapacity{8, 0, 4096, 0}; };
    mo::MinimaxApi api(fake_tokenizer(), rig.queue, options, rig.observer);
    Writer plan; api.handle(get("/v1/deployment"), plan);
    check(plan.status == 200 && plan.body == R"({"schema":"strix-deployment-v1"})", "shared plan handler");
    Writer health; api.handle(get("/health"), health);
    check(at(sv::parse_json(health.body), "deployment_id").as_string() == "0123456789abcdef", "health deployment identity");
    Writer status; api.handle(get("/api/status"), status);
    check(at(at(sv::parse_json(status.body), "capacity"), "kv_bytes_free").as_int() == 4096, "capacity callback");
    Writer drained; api.handle(post("/admin/drain", ""), drained);
    check(drained.status == 200 && at(sv::parse_json(drained.body), "drained").as_bool(), "shared drain handler");
    Writer draining; api.handle(get("/health"), draining);
    check(draining.status == 503 && !rig.queue.submit({}), "drain health and admission");
}

int main() {
    test_deployment_surface();
    test_request_fields();
    test_unarmed_non_stream();
    test_armed_stream();
    test_terminal_paths();
    test_arming_mid_request();
    test_delivery_outcomes();
    test_options_and_utf8();
    test_scheduler_side();
    test_sessions();
    std::cout << (failures ? "FAIL" : "PASS") << " strix-minimax-api-tests: " << checks - failures << "/" << checks << " checks\n";
    return failures ? 1 : 0;
}
