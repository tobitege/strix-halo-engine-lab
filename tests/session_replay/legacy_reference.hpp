#pragma once

// Host fixture and an independent pre-session GenerationQueue::run oracle.
// The reference implements the old GenerationEngine contract directly and
// never calls SessionRuntime, session::Engine, or the new scheduling policy.
#include "strix/serve/session_runtime.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string("check failed: ") + #x + " line " + std::to_string(__LINE__)); } while (false)

namespace session_fixture {
namespace sv = strix::serve;
namespace ss = strix::session;

struct Physical {
    std::vector<std::uint32_t> rows;
    std::uint64_t draws = 0, outputs = 0;
    bool operator==(const Physical&) const = default;
};
struct FakeEngine final : ss::Engine {
    std::array<Physical, 4> physical;
    std::vector<std::string> trace;
    std::vector<std::vector<std::uint32_t>> suffixes;
    std::vector<std::uint32_t> allowances;
    std::vector<std::uint32_t> script{65, 66, 0};
    unsigned captures = 0, restores = 0, releases = 0, initializes = 0, batch_calls = 0, batch_prefills = 0;
    unsigned width = 3, rolled_back = 0;
    bool fail_restore = false, bad_capture = false, fail_release = false, fail_decode = false;
    int malformed = 0;
    int malformed_slot = -1;
    bool pending_prefill = false;
    std::function<void()> during_decode;
    ss::Capabilities limits;
    ss::KvStore store;
    ss::Hooks hooks() {
        ss::Hooks h;
        h.initialize = [this](ss::KvHandle k) {
            ++initializes; physical.at(k.slot) = {}; trace.push_back("acquire:" + std::to_string(k.slot) + ":" + std::to_string(k.generation));
        };
        h.reset = [this](ss::KvHandle k) { physical.at(k.slot) = {}; };
        h.release = [this](ss::KvHandle k) {
            ++releases; trace.push_back("release:" + std::to_string(k.slot));
            if (fail_release && k.slot == 0) throw std::runtime_error("injected release failure");
            physical.at(k.slot) = {};
        };
        h.capture = [this](ss::KvHandle k, std::uint32_t) {
            ++captures;
            if (bad_capture) return ss::RankState{};
            return ss::RankState{std::make_shared<const Physical>(physical.at(k.slot)), sizeof(Physical)};
        };
        h.view = [this](ss::KvHandle k, std::uint32_t) {
            return ss::RankState{std::make_shared<const Physical>(physical.at(k.slot)), sizeof(Physical)};
        };
        h.restore = [this](ss::KvHandle k, std::uint32_t, const ss::RankState& r) {
            ++restores;
            if (fail_restore) throw std::runtime_error("injected restore failure");
            physical.at(k.slot) = *std::static_pointer_cast<const Physical>(r.owner);
            trace.push_back("restore:" + std::to_string(physical.at(k.slot).rows.size()));
        };
        h.truncate = [this](ss::KvHandle k, std::uint32_t, std::uint64_t n) { physical.at(k.slot).rows.resize(static_cast<std::size_t>(n)); };
        h.partial_restore_supported = true;
        return h;
    }
    explicit FakeEngine(unsigned slots = 2) : store(slots, 1, hooks()) {
        limits.max_context = 64; limits.max_slots = slots; limits.max_batch = slots;
        limits.prefix_cache = true; limits.batch_decode = true; limits.batch_prefill = true;
        limits.speculative = true; limits.temperature = true; limits.seed = true;
    }
    ss::Capabilities caps() const override { return limits; }
    strix::profile::ExecutionReceipt receipt() const override {
        strix::profile::ExecutionReceipt r; r.engine = "host-fixture"; r.implementation = "script-v1"; return r;
    }
    void consume(ss::KvHandle k, std::span<const std::uint32_t> ids) {
        physical.at(k.slot).rows.insert(physical.at(k.slot).rows.end(), ids.begin(), ids.end());
        store.append_consumed(k, ids);
    }
    void anchor(ss::KvHandle k) {
        if (auto p = store.pending(k)) {
            const std::array<std::uint32_t, 1> id{*p};
            consume(k, id); store.set_pending(k, {});
        }
    }
    void prefill(std::span<const std::uint32_t> ids, ss::KvHandle& k) override {
        suffixes.emplace_back(ids.begin(), ids.end()); anchor(k);
        if (pending_prefill && !ids.empty()) { consume(k, ids.first(ids.size() - 1)); store.set_pending(k, ids.back()); }
        else consume(k, ids);
        trace.push_back("prefill:" + std::to_string(k.slot) + ":" + std::to_string(ids.size()));
    }
    void prefill_batch(std::span<const ss::PrefillItem> items) override {
        ++batch_prefills;
        for (const auto& i : items) (void)store.consumed(i.state);
        for (const auto& i : items) prefill(i.tokens, i.state);
    }
    void validate(ss::KvHandle k, const ss::SamplerState& s) {
        CHECK(physical.at(k.slot).rows.size() == store.consumed(k));
        CHECK(physical.at(k.slot).draws == s.rng_draws);
    }
    ss::StepResult decode(ss::KvHandle& k, const ss::SamplerState& s) override {
        validate(k, s); allowances.push_back(s.max_tokens);
        if (during_decode) during_decode();
        if (fail_decode) throw std::runtime_error("injected decode failure");
        ss::StepResult r;
        r.before = {store.consumed(k), store.pending(k)};
        auto& p = physical.at(k.slot);
        r.rng_draws = p.draws;
        if (s.cancellation.stop_requested()) r.finish = ss::FinishReason::cancelled;
        else if (p.rows.size() + store.pending(k).has_value() == limits.max_context) r.finish = ss::FinishReason::context;
        else if (!s.max_tokens) r.finish = ss::FinishReason::length;
        else {
            // Proposals may extend beyond a stop, but only committed prefix
            // rows survive. Check the physical trace and cursor after rollback.
            const auto candidates = std::min<unsigned>(width, s.max_tokens);
            r.proposed_drafts = candidates - 1;
            for (unsigned n = 0; n < candidates; ++n) {
                anchor(k);
                const auto id = script[p.outputs % script.size()];
                ++p.outputs;
                if (s.parameters.temperature > 0) ++p.draws;
                store.set_pending(k, id); r.tokens.push_back(id);
                if (std::find(s.stop_tokens.begin(), s.stop_tokens.end(), id) != s.stop_tokens.end()) {
                    r.finish = ss::FinishReason::stop;
                    const auto committed = p;
                    // Model speculative physical rows/cursors beyond the stop,
                    // then roll them back before returning the retained IDs.
                    for (unsigned extra = n + 1; extra < candidates; ++extra) {
                        p.rows.push_back(script[p.outputs % script.size()]); ++p.outputs;
                        if (s.parameters.temperature > 0) ++p.draws;
                    }
                    trace.push_back("tentative:" + std::to_string(p.rows.size()) + ":" + std::to_string(p.draws));
                    p = committed;
                    rolled_back += candidates - n - 1;
                    break;
                }
                if (store.consumed(k) + 1 == limits.max_context) { r.finish = ss::FinishReason::context; break; }
                if (r.tokens.size() == s.max_tokens) { r.finish = ss::FinishReason::length; break; }
            }
            r.accepted_drafts = static_cast<std::uint32_t>(r.tokens.size() - 1);
            r.rng_draws = p.draws;
        }
        r.after = {store.consumed(k), store.pending(k)};
        trace.push_back("decode:" + std::to_string(k.slot) + ":" + std::to_string(p.rows.size()) + ":" + std::to_string(p.draws));
        if (malformed_slot < 0 || k.slot == static_cast<unsigned>(malformed_slot)) {
            if (malformed == 1) { r.tokens.clear(); r.finish = ss::FinishReason::none; }
            if (malformed == 2) ++r.after.consumed_tokens;
            if (malformed == 3) r.tokens.push_back(99);
            if (malformed == 4) r.finish = ss::FinishReason::none;
            if (malformed == 5) ++r.rng_draws, r.tokens.clear(), r.after = r.before;
        }
        return r;
    }
    std::vector<ss::StepResult> decode_batch(std::span<const ss::DecodeItem> items) override {
        ++batch_calls;
        for (const auto& i : items) validate(i.state, i.sampler);
        std::vector<ss::StepResult> out;
        for (const auto& i : items) out.push_back(decode(i.state, i.sampler));
        return out;
    }
};
inline sv::SessionHooks runtime_hooks(FakeEngine& e) {
    sv::SessionHooks h;
    h.identity = {"fixture-weights-v1", "fixture-profile-v1", "fixture-state-v1", 1};
    h.checkpoint = [](const sv::SessionView&) { return true; };
    h.receipt = [&e](const sv::SessionView& v, sv::GenerationStats& stats) {
        const auto& p = e.physical.at(v.state.slot);
        stats.engine_extra.emplace_back("fixture", sv::Json(sv::Json::Object{
            {"identity", "fixture-state-v1"}, {"slot", std::uint64_t(v.state.slot)},
            {"generation", v.state.generation}, {"frontier", e.store.consumed(v.state)},
            {"cursor", p.draws}, {"rows", std::uint64_t(p.rows.size())}}).dump());
    };
    return h;
}
inline sv::GenerationJob job(std::vector<std::uint32_t> prompt = {10, 11, 12}, unsigned max = 3) {
    sv::GenerationJob j; j.prompt = std::move(prompt); j.max_tokens = max; j.stop_tokens = {0}; return j;
}
struct Outcome { std::vector<std::uint32_t> ids; sv::GenerationStats stats; std::string error; };
inline Outcome drain(const std::shared_ptr<sv::GenerationTicket>& ticket) {
    CHECK(ticket);
    bool done = false;
    auto ids = ticket->wait(std::chrono::milliseconds(0), done, false);
    CHECK(done);
    return {std::move(ids), ticket->stats(), ticket->error()};
}
inline Outcome execute(sv::SessionRuntime& runtime, sv::GenerationQueue& queue, sv::GenerationJob j) {
    auto ticket = queue.submit(std::move(j));
    for (int n = 0; n < 20; ++n) {
        runtime.step();
        bool done = false;
        // Don't consume until done; progress is a separate snapshot.
        if (queue.snapshot().running == 0 && queue.snapshot().queued == 0) return drain(ticket);
        (void)done;
    }
    throw std::runtime_error("fixture owner failed to terminate");
}

// Frozen scripted GenerationEngine reference, derived from the pre-session
// queue contract: emit A, B, stop in a single committed call, cap before stop,
// and return the explicit slot/frontier/cursor receipt. No session code runs.
class LegacyReference final : public sv::GenerationEngine {
public:
    std::uint64_t generation = 0;
    bool cache_enabled = false;
    std::vector<std::vector<std::uint32_t>> retained_prompts;
    sv::GenerationStats generate(sv::GenerationJob& j) override {
        sv::GenerationStats s; s.prompt_tokens = j.prompt.size(); ++generation;
        if (cache_enabled) {
            for (const auto& prefix : retained_prompts)
                if (prefix.size() <= j.prompt.size() && prefix.size() > s.cached_tokens &&
                    std::equal(prefix.begin(), prefix.end(), j.prompt.begin())) s.cached_tokens = prefix.size();
            if (j.cache_retention != sv::CacheRetention::no_store && s.cached_tokens < j.prompt.size())
                retained_prompts.push_back(j.prompt);
        }
        if (j.on_admitted) j.on_admitted();
        if (j.on_progress) j.on_progress({sv::GenerationPhase::decode, j.prompt.size() - s.cached_tokens, s.cached_tokens, j.prompt.size()});
        std::vector<std::uint32_t> ids;
        for (auto id : {65U, 66U, 0U}) {
            if (ids.size() == j.max_tokens) break;
            ids.push_back(id);
            if (std::find(j.stop_tokens.begin(), j.stop_tokens.end(), id) != j.stop_tokens.end()) { s.finish = "stop"; break; }
        }
        s.generated_tokens = ids.size(); s.cycles = 1;
        s.proposed = ids.empty() ? 0 : ids.size() - 1; s.accepted = s.proposed;
        if (!j.on_tokens(ids)) s.finish = "cancelled";
        const auto frontier = j.prompt.size() + ids.size() - (ids.empty() ? 0 : 1);
        s.engine_extra.emplace_back("fixture", sv::Json(sv::Json::Object{
            {"identity", "fixture-state-v1"}, {"slot", std::uint64_t(0)}, {"generation", generation},
            {"frontier", std::uint64_t(frontier)}, {"cursor", std::uint64_t(0)}, {"rows", std::uint64_t(frontier)}}).dump());
        return s;
    }
    std::uint64_t context_tokens() const override { return 64; }
};
inline std::shared_ptr<const sv::BpeTokenizer> tokenizer() {
    std::vector<std::string> tokens;
    std::vector<std::int32_t> types;
    unsigned extra = 0;
    for (unsigned b = 0; b < 256; ++b) {
        unsigned cp = b;
        if (!((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174)) cp = 256 + extra++;
        std::string t;
        if (cp < 128) t.push_back(static_cast<char>(cp));
        else { t.push_back(static_cast<char>(0xc0 | (cp >> 6))); t.push_back(static_cast<char>(0x80 | (cp & 63))); }
        tokens.push_back(t); types.push_back(1);
    }
    for (const auto* special : {"<|endoftext|>", "<|im_start|>", "<|im_end|>"}) { tokens.emplace_back(special); types.push_back(3); }
    return sv::BpeTokenizer::from_parts(std::move(tokens), std::move(types), {}, "qwen35");
}
struct Writer final : sv::HttpResponseWriter {
    int status = 0;
    std::string body;
    bool gone = false, ended = false;
    std::size_t writes = 0, disconnect_after = SIZE_MAX;
    void send(int s, std::string_view, std::string_view b, const std::vector<std::pair<std::string, std::string>>&) override { status = s; body = b; }
    bool begin_stream(int s, std::string_view, const std::vector<std::pair<std::string, std::string>>&) override { status = s; return !gone; }
    bool write(std::string_view b) override { body += b; gone = ++writes >= disconnect_after; return !gone; }
    void end_stream() override { ended = true; }
    bool client_gone() const override { return gone; }
};
} // namespace session_fixture
