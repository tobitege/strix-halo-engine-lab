#include "strix/serve/session_runtime.hpp"
#include <cstdio>
#include <stdexcept>

namespace ss = strix::session;
namespace sv = strix::serve;
struct Engine final : ss::Engine {
    ss::KvStore& store;
    explicit Engine(ss::KvStore& s) : store(s) {}
    void prefill(std::span<const std::uint32_t> t, ss::KvHandle& h) override { store.append_consumed(h, t); }
    ss::StepResult decode(ss::KvHandle&, const ss::SamplerState&) override { return {}; }
    ss::Capabilities caps() const override {
        ss::Capabilities c;
        c.max_context = 16384; c.max_slots = 2; c.max_batch = 1; c.prefix_cache = true; c.owner_control = true;
        return c;
    }
    strix::profile::ExecutionReceipt receipt() const override { return {}; }
};
int main() try {
    ss::Hooks storage;
    storage.initialize = storage.reset = storage.release = [](ss::KvHandle) {};
    storage.view = storage.capture = [](ss::KvHandle, std::uint32_t) {
        return ss::RankState{std::make_shared<const int>(0), sizeof(int)};
    };
    storage.restore = [](ss::KvHandle, std::uint32_t, const ss::RankState&) {};
    storage.truncate = [](ss::KvHandle, std::uint32_t, std::uint64_t) {};
    storage.resident_prefill_supported = true;
    ss::KvStore store(2, 1, storage);
    Engine engine(store);
    sv::GenerationQueue queue;
    sv::SessionHooks hooks;
    hooks.identity = {"weights", "profile", "rows", 1};
    hooks.control_turn = [](sv::SessionRuntime&) {};
    hooks.resident_identity = [base = hooks.identity](const sv::GenerationJob& job) {
        auto id = base;
        id.profile += job.prompt.size() >= 8192 ? ":int4" : ":native";
        return id;
    };
    sv::SessionRuntime runtime(queue, engine, store, {}, hooks);
    auto turn = [&](std::size_t n, std::uint64_t expected) {
        sv::GenerationJob job; job.prompt.assign(n, 1); job.max_tokens = 1;
        const auto ticket = queue.submit(job);
        if (!runtime.reserve(sv::QueueSelection::fifo)) throw std::runtime_error("reservation failed");
        auto lease = runtime.activate(4096);
        if (lease.stats.cached_tokens != expected) throw std::runtime_error("wrong arithmetic regime reused");
        engine.prefill(std::span(job.prompt).subspan(static_cast<std::size_t>(expected)), lease.state);
        runtime.complete(lease.state, {}, n - n % 4096);
        if (!ticket->error().empty()) throw std::runtime_error("retention failed");
    };
    turn(6000, 0);
    turn(6001, 4096); // establish a live one-chunk certificate below the threshold
    turn(9000, 0);   // same prefix tokens, different effective arithmetic: cold
    turn(10000, 8192); // same arithmetic still admits reuse
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
