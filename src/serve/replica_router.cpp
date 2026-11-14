#include "strix/serve/replica_router.hpp"

#include "strix/serve/json.hpp"

#include <algorithm>
#include <stdexcept>

namespace strix::serve {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

std::uint64_t fnv(std::uint64_t h, std::string_view bytes) noexcept {
    for (const char c : bytes) { h ^= static_cast<unsigned char>(c); h *= kFnvPrime; }
    return h;
}

// Final avalanche (splitmix64) so table slots do not follow FNV's weak low bits. Never returns 0 (0 = empty slot).
std::uint64_t mix(std::uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    x ^= x >> 31U;
    return x ? x : 1;
}

std::uint64_t reported_load(const ReplicaState& s) noexcept {
    const auto load = s.capacity.slots_busy + s.capacity.queue_depth;
    return load < s.capacity.slots_busy ? ~0ULL : load;
}

std::uint64_t json_u64(const Json* value) {
    if (!value || !value->is_number()) return 0;
    const double d = value->as_double();
    return d <= 0 ? 0 : d >= 18446744073709549568.0 ? ~0ULL : static_cast<std::uint64_t>(d);
}

} // namespace

std::string_view route_reason_name(RouteReason reason) noexcept {
    switch (reason) {
    case RouteReason::session: return "session";
    case RouteReason::prefix: return "prefix";
    case RouteReason::load: return "load";
    default: return "none";
    }
}

ReplicaRouter::ReplicaRouter(std::size_t replicas, std::size_t affinity_entries, std::size_t block_bytes)
    : block_bytes_(block_bytes), inflight_(replicas, 0), sessions_(affinity_entries), chains_(affinity_entries) {
    if (replicas == 0 || replicas > kMaxReplicas) throw std::invalid_argument("replica router: 1..64 replicas");
    if (affinity_entries == 0) throw std::invalid_argument("replica router: affinity_entries must be positive");
    if (block_bytes == 0) throw std::invalid_argument("replica router: block_bytes must be positive");
}

// Chain hash of block i covers blocks 0..i, so equal hashes mean an equal prefix up to that block boundary.
// Only full blocks count: a partial tail is the part of the prompt most likely to differ.
template<class F> void ReplicaRouter::for_each_block(std::string_view prompt, F&& visit) const {
    std::uint64_t h = kFnvOffset;
    for (std::size_t at = 0; at + block_bytes_ <= prompt.size(); at += block_bytes_) {
        h = fnv(h, prompt.substr(at, block_bytes_));
        if (!visit(mix(h))) return;
    }
}

std::size_t ReplicaRouter::place(const RouteRequest& request, std::span<const ReplicaState> states) {
    return place(request, states, 0);
}

std::size_t ReplicaRouter::place(const RouteRequest& request, std::span<const ReplicaState> states,
    std::uint64_t excluded) {
    if (states.size() != inflight_.size()) throw std::invalid_argument("replica router: one state per replica");
    const std::lock_guard lock(mutex_);
    const std::size_t n = inflight_.size();
    const auto usable = [&](std::size_t r) { return states[r].healthy && !((excluded >> r) & 1U); };
    const auto load = [&](std::size_t r) { return std::max(reported_load(states[r]), inflight_[r]); };

    // Least loaded usable replica: the fallback, and the reference of the imbalance rule.
    std::size_t least = kNoReplica;
    for (std::size_t r = 0; r < n; ++r) {
        if (!usable(r)) continue;
        if (least == kNoReplica) { least = r; continue; }
        const auto a = load(r), b = load(least);
        const double ea = states[r].ewma_ms_per_token, eb = states[least].ewma_ms_per_token;
        if (a < b || (a == b && ea > 0 && eb > 0 && ea < eb)) least = r;
    }
    if (least == kNoReplica) { last_reason_ = RouteReason::none; return kNoReplica; }

    std::size_t chosen = kNoReplica;
    RouteReason reason = RouteReason::load;
    Session* session = nullptr;
    std::uint64_t session_key = 0;
    if (!request.session_key.empty()) {
        session_key = mix(fnv(kFnvOffset, request.session_key));
        session = &sessions_[session_key % sessions_.size()];
        if (session->key == session_key && session->replica_plus_one != 0) {
            const std::size_t holder = session->replica_plus_one - 1U;
            if (holder < n && usable(holder)) { chosen = holder; reason = RouteReason::session; }
        }
    }
    if (chosen == kNoReplica) {
        // Per replica: number of leading blocks recorded for it.
        std::uint64_t alive = 0;
        for (std::size_t r = 0; r < n; ++r) if (usable(r)) alive |= 1ULL << r;
        std::uint64_t best_mask = 0;
        std::size_t depth = 0;
        for_each_block(request.prompt_bytes, [&](std::uint64_t key) {
            const Chain& c = chains_[key % chains_.size()];
            alive &= c.key == key ? c.mask : 0;
            if (!alive) return false;
            best_mask = alive; ++depth;
            return true;
        });
        if (depth != 0) {
            std::size_t best = kNoReplica;
            for (std::size_t r = 0; r < n; ++r)
                if (((best_mask >> r) & 1U) && (best == kNoReplica || load(r) < load(best))) best = r;
            // A hot prefix may not pile a whole burst on one node: more than one slot of imbalance overrides affinity.
            if (best != kNoReplica && load(best) <= load(least) + 1U) { chosen = best; reason = RouteReason::prefix; }
        }
    }
    if (chosen == kNoReplica) chosen = least;

    ++inflight_[chosen];
    if (session) *session = Session{session_key, static_cast<std::uint32_t>(chosen + 1U)};
    for_each_block(request.prompt_bytes, [&](std::uint64_t key) {
        Chain& c = chains_[key % chains_.size()];
        if (c.key != key) c = Chain{key, 0};
        c.mask |= 1ULL << chosen;
        return true;
    });
    last_reason_ = reason;
    return chosen;
}

void ReplicaRouter::finished(std::size_t replica, const RouteRequest&) noexcept {
    const std::lock_guard lock(mutex_);
    if (replica < inflight_.size() && inflight_[replica] != 0) --inflight_[replica];
}

void ReplicaRouter::forget(std::size_t replica) noexcept {
    const std::lock_guard lock(mutex_);
    if (replica >= inflight_.size()) return;
    for (auto& s : sessions_) if (s.replica_plus_one == replica + 1U) s = Session{};
    for (auto& c : chains_) {
        c.mask &= ~(1ULL << replica);
        if (!c.mask) c = Chain{};
    }
}

std::uint64_t ReplicaRouter::inflight(std::size_t replica) const noexcept {
    const std::lock_guard lock(mutex_);
    return replica < inflight_.size() ? inflight_[replica] : 0;
}

RouteReason ReplicaRouter::last_reason() const noexcept {
    const std::lock_guard lock(mutex_);
    return last_reason_;
}

std::size_t ReplicaRouter::matching_blocks(std::string_view prompt_bytes, std::size_t replica) const noexcept {
    const std::lock_guard lock(mutex_);
    if (replica >= inflight_.size()) return 0;
    std::size_t depth = 0;
    for_each_block(prompt_bytes, [&](std::uint64_t key) {
        const Chain& c = chains_[key % chains_.size()];
        if (c.key != key || !((c.mask >> replica) & 1U)) return false;
        ++depth;
        return true;
    });
    return depth;
}

std::optional<ReplicaState> replica_state_from_status(std::string_view status_json) {
    Json status;
    try { status = parse_json(status_json); } catch (const std::exception&) { return std::nullopt; }
    if (!status.is_object()) return std::nullopt;
    ReplicaState out;
    const Json* capacity = status.find("capacity");
    if (capacity && capacity->is_object()) {
        out.capacity.slots_total = json_u64(capacity->find("slots_total"));
        out.capacity.slots_busy = json_u64(capacity->find("slots_busy"));
        out.capacity.queue_depth = json_u64(capacity->find("queue_depth"));
        const Json* kv = capacity->find("kv_bytes_free");
        if (kv && kv->is_number()) out.capacity.kv_bytes_free = json_u64(kv);
    }
    const Json* state = status.find("state");
    const bool refusing = state && state->is_string()
        && (state->as_string() == "failed" || state->as_string() == "draining");
    out.healthy = out.capacity.slots_total > 0 && !refusing;
    return out;
}

RouteRequest route_request_from_http(std::string_view path, std::string_view session_header,
    std::string_view body, RouteKeyStorage& storage) {
    storage.session.clear();
    storage.prompt.clear();
    constexpr std::string_view sessions = "/v1/sessions/";
    if (path.starts_with(sessions)) {
        const auto rest = path.substr(sessions.size());
        storage.session = std::string(rest.substr(0, rest.find('/')));
    }
    if (storage.session.empty()) storage.session = std::string(session_header);
    Json parsed;
    bool is_json = false;
    if (!body.empty()) {
        try { parsed = parse_json(body); is_json = parsed.is_object(); } catch (const std::exception&) {}
    }
    if (is_json) {
        if (storage.session.empty())
            for (const char* name : {"session_id", "session", "conversation_id"}) {
                const Json* v = parsed.find(name);
                if (v && v->is_string() && !v->as_string().empty()) { storage.session = v->as_string(); break; }
            }
        // Prompt order of the rendered request: tools and system come before the turns. The canonical dump makes
        // two requests that share leading messages share leading bytes, whatever the client's member order.
        for (const char* name : {"tools", "system", "instructions", "messages", "input", "prompt"}) {
            const Json* v = parsed.find(name);
            if (v && !v->is_null()) storage.prompt += v->dump();
        }
    } else {
        storage.prompt.assign(body);
    }
    RouteRequest out;
    out.session_key = storage.session;
    out.prompt_bytes = storage.prompt;
    out.estimated_tokens = storage.prompt.size() / 4U;
    return out;
}

} // namespace strix::serve
