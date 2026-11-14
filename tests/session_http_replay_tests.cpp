#include "session_replay/legacy_reference.hpp"

#include <filesystem>
#include <fstream>
#include <thread>

using namespace session_fixture;
namespace {
sv::Json read(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary); CHECK(bool(f));
    return sv::parse_json(std::string(std::istreambuf_iterator<char>(f), {}));
}
// Only volatile identity/time fields are normalized. Every other byte (token
// text, finish reason, receipt, cache/frontier/cursor and SSE framing) survives.
void normalize(sv::Json& j) {
    if (j.is_array()) { for (auto& v : j.as_array()) normalize(v); }
    if (!j.is_object()) return;
    for (auto& [key, value] : j.as_object()) {
        if (key == "id" && value.is_string() &&
            (value.as_string().starts_with("cmpl-") || value.as_string().starts_with("chatcmpl-"))) value = "<request-id>";
        else if (key == "created" || key == "queue_ms" || key == "prefill_ms" || key == "decode_ms" || key == "total_ms" || key == "tokens_per_second" || key == "decode_tokens_per_second") value = 0;
        else normalize(value);
    }
}
std::string normalized(const Writer& w) {
    if (w.body.starts_with("data:") || w.body.starts_with(":")) {
        std::string out;
        std::size_t at = 0;
        while (at < w.body.size()) {
            auto end = w.body.find('\n', at); if (end == std::string::npos) end = w.body.size();
            auto line = w.body.substr(at, end - at);
            if (line.starts_with("data: {") || line.starts_with("data:{")) {
                const auto start = line[5] == ' ' ? 6U : 5U;
                const auto raw = line.substr(start);
                auto j = sv::parse_json(raw); CHECK(j.dump() == raw);
                normalize(j); line = line.substr(0, start) + j.dump();
            }
            out += line; if (end < w.body.size()) out += '\n'; at = end + 1;
        }
        return out;
    }
    auto j = sv::parse_json(w.body); CHECK(j.dump() == w.body); normalize(j); return j.dump();
}
sv::ApiConfig config() {
    sv::ApiConfig c; c.model_id = "session-fixture"; c.sampling_policy = "reject";
    c.stop_tokens = {0}; c.sse_keepalive_seconds = 0; c.sse_progress_seconds = 0;
    c.default_thinking = false;
    c.chat_renderer = [](const sv::ChatTemplateInput& in) {
        return in.messages.as_array().back().find("content")->as_string();
    };
    return c;
}
}
int main(int argc, char** argv) {
    try {
        CHECK(argc == 2);
        const auto dir = std::filesystem::path(argv[1]) / "tests/session_replay/fixtures";
        const auto requests = read(dir / "requests.json");
        const auto expected = read(dir / "expected.json");
        auto tokens = tokenizer();
        for (bool cache_enabled : {false, true}) {
            FakeEngine engine(1); engine.limits.temperature = false; engine.limits.seed = false;
            sv::GenerationQueue queue, legacy_queue;
            ss::Policy p; p.enabled = cache_enabled;
            sv::SessionRuntime runtime(queue, engine, engine.store, p, runtime_hooks(engine));
            auto c = config(); runtime.configure_api(c);
            sv::ApiService api(c, tokens, queue, 64, false);
            LegacyReference reference;
            reference.cache_enabled = cache_enabled;
            auto old_config = config(); old_config.engine_progress = true;
            sv::ApiService legacy(old_config, tokens, legacy_queue, 64, false);
            std::exception_ptr owner_failure;
            std::thread owner([&] { try { runtime.run(); } catch (...) { owner_failure = std::current_exception(); } });
            std::thread old_owner([&] { legacy_queue.run(reference); });
            try {
                std::size_t n = 0;
                std::uint64_t cached = 0;
                for (const auto& fixture : requests.as_array()) {
                    sv::HttpRequest req;
                    req.method = fixture.find("method")->as_string(); req.path = fixture.find("path")->as_string(); req.target = req.path;
                    req.body = fixture.find("body")->dump();
                    Writer actual, before;
                    api.handle(req, actual); legacy.handle(req, before);
                    CHECK(actual.status == expected.find("statuses")->as_array()[n].as_int());
                    CHECK(actual.status == before.status);
                    if (actual.status == 200 && cache_enabled) cached += static_cast<std::uint64_t>(expected.find("cache_tokens")->as_array()[n].as_int());
                    if (queue.snapshot().cached_tokens != cached)
                        throw std::runtime_error("cached count at " + std::to_string(n) + ": " +
                            std::to_string(queue.snapshot().cached_tokens) + " expected " + std::to_string(cached) + " response " + actual.body);
                    const auto a = normalized(actual), b = normalized(before);
                    if (a != b) throw std::runtime_error("replay mismatch at " + std::to_string(n) + "\nactual: " + a + "\nreference: " + b);
                    if (n < 3) {
                        if (n < 2) CHECK(actual.body.find(expected.find("text")->as_string()) != std::string::npos);
                        CHECK(actual.body.find(expected.find("identity")->as_string()) != std::string::npos);
                    }
                    if (n == 0) CHECK(actual.body.find("\"finish_reason\":\"stop\"") != std::string::npos);
                    if (n == 1) CHECK(actual.body.find("\"finish_reason\":\"length\"") != std::string::npos);
                    if (n == 2) CHECK(actual.ended && actual.body.ends_with("data: [DONE]\n\n"));
                    ++n;
                }
                CHECK(engine.releases == 8 && engine.initializes == 8);
                CHECK(engine.restores == (cache_enabled ? 5U : 0U));
                CHECK(engine.captures == (cache_enabled ? 7U : 0U));
                CHECK(runtime.cache_entries() == (cache_enabled ? 2U : 0U));
                // Refused policies never acquired a physical slot.
            } catch (...) {
                runtime.shutdown(); legacy_queue.shutdown(); owner.join(); old_owner.join(); throw;
            }
            runtime.shutdown(); legacy_queue.shutdown(); owner.join(); old_owner.join();
            if (owner_failure) std::rethrow_exception(owner_failure);
        }
        // Disconnect uses the same production handler and queue. It need not
        // compare time-dependent races against the frozen reference.
        {
            FakeEngine e(1); sv::GenerationQueue q; ss::Policy off; off.enabled = false;
            sv::SessionRuntime r(q, e, e.store, off, runtime_hooks(e));
            auto cfg = config(); r.configure_api(cfg); sv::ApiService service(cfg, tokens, q, 64, false);
            std::thread worker([&] { r.run(); });
            sv::HttpRequest req; req.method = "POST"; req.path = "/v1/completions";
            req.body = "{\"prompt\":\"hi\",\"max_tokens\":3,\"stream\":true}";
            Writer gone; gone.disconnect_after = 1;
            try { service.handle(req, gone); } catch (...) { r.shutdown(); worker.join(); throw; }
            r.shutdown(); worker.join(); CHECK(gone.gone && q.snapshot().running == 0);
        }
        std::cout << "production HTTP replay checks passed\n"; return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
