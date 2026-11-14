#include "strix/serve/replica_router.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _MSC_VER
#include <crtdbg.h>
#include <cstdlib>
#endif

namespace {
using namespace strix::serve;

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

ReplicaState replica(std::uint64_t busy, std::uint64_t queued = 0, bool healthy = true) {
    ReplicaState s;
    s.capacity.slots_total = 8;
    s.capacity.slots_busy = busy;
    s.capacity.queue_depth = queued;
    s.healthy = healthy;
    return s;
}

// A prompt of `blocks` full 64-byte blocks: the first `shared` come from `family`, the rest from `tail`.
std::string prompt(char family, std::size_t shared, char tail, std::size_t blocks) {
    std::string out;
    for (std::size_t b = 0; b < blocks; ++b)
        out += std::string(63, b < shared ? family : tail) + static_cast<char>('0' + b % 10);
    return out;
}

RouteRequest request(std::string_view prompt_bytes, std::string_view session = {}) {
    RouteRequest r;
    r.session_key = session;
    r.prompt_bytes = prompt_bytes;
    r.estimated_tokens = prompt_bytes.size() / 4;
    return r;
}

void least_load() {
    ReplicaRouter router(2, 256, 64);
    std::vector<ReplicaState> states{replica(0), replica(0)};
    require(router.place(request("a"), states) == 0, "equal load goes to the lower index");
    require(router.last_reason() == RouteReason::load, "reason is load");
    // The first placement is not in the reported state yet: the router's own count must spread the burst.
    require(router.place(request("b"), states) == 1, "second request of a burst goes to the other replica");
    require(router.inflight(0) == 1 && router.inflight(1) == 1, "in-flight count");
    router.finished(0, request("a"));
    router.finished(1, request("b"));
    router.finished(1, request("b"));
    require(router.inflight(0) == 0 && router.inflight(1) == 0, "finished never underflows");

    states = {replica(3, 2), replica(4, 0)};
    require(router.place(request("c"), states) == 1, "busy + queued decides");
    router.finished(1, request("c"));
    states = {replica(2), replica(2)};
    states[0].ewma_ms_per_token = 20; states[1].ewma_ms_per_token = 12;
    require(router.place(request("d"), states) == 1, "equal load: the faster replica");
    router.finished(1, request("d"));
}

void session_stickiness() {
    ReplicaRouter router(2, 256, 64);
    std::vector<ReplicaState> states{replica(5), replica(0)};
    require(router.place(request("hello", "conv-1"), states) == 1, "new session goes to the least loaded replica");
    router.finished(1, request("hello", "conv-1"));
    // The holder is now much busier; a named session still returns to it (its state lives there).
    states = {replica(0), replica(7, 4)};
    require(router.place(request("hello again", "conv-1"), states) == 1, "session stays on its replica");
    require(router.last_reason() == RouteReason::session, "reason is session");
    router.finished(1, request("hello again", "conv-1"));
    require(router.place(request("other", "conv-2"), states) == 0, "another session is placed by load");
    router.finished(0, request("other", "conv-2"));

    // Holder unhealthy: the session moves and is re-bound.
    states = {replica(6), replica(0, 0, false)};
    require(router.place(request("x", "conv-1"), states) == 0, "session moves off an unhealthy replica");
    router.finished(0, request("x", "conv-1"));
    states = {replica(6), replica(0)};
    require(router.place(request("y", "conv-1"), states) == 0, "session is re-bound to the new replica");
    router.finished(0, request("y", "conv-1"));
}

void prefix_affinity() {
    ReplicaRouter router(2, 1024, 64);
    const auto base = prompt('A', 4, 'A', 4);
    std::vector<ReplicaState> states{replica(1), replica(0)};
    require(router.place(request(base), states) == 1, "first sight: least loaded");
    router.finished(1, request(base));
    require(router.matching_blocks(base, 1) == 4 && router.matching_blocks(base, 0) == 0, "chain recorded");

    // Same first three blocks, different tail: goes where the prefix is, even one slot busier.
    const auto sibling = prompt('A', 3, 'B', 6);
    states = {replica(0), replica(1)};
    require(router.place(request(sibling), states) == 1, "prefix affinity within one slot of imbalance");
    require(router.last_reason() == RouteReason::prefix, "reason is prefix");
    router.finished(1, request(sibling));

    // More than one slot busier: the imbalance rule overrides affinity, and the other replica learns the prefix.
    states = {replica(0), replica(2)};
    require(router.place(request(sibling), states) == 0, "imbalance overrides affinity");
    require(router.last_reason() == RouteReason::load, "override is a load placement");
    router.finished(0, request(sibling));
    require(router.matching_blocks(sibling, 0) == 6, "second replica now holds the chain");

    // Both hold the sibling prefix: the less loaded of the holders.
    states = {replica(1), replica(2)};
    require(router.place(request(sibling), states) == 0, "among holders, the less loaded");
    router.finished(0, request(sibling));

    // Longest chain wins: replica 1 holds 4 blocks of `base`, replica 0 only its first 3.
    states = {replica(0), replica(1)};
    require(router.place(request(base), states) == 1, "longest matching chain wins");
    router.finished(1, request(base));

    // A burst on one hot prefix cannot pile up: in-flight placements count as load.
    states = {replica(0), replica(0)};
    ReplicaRouter burst(2, 1024, 64);
    std::size_t on[2] = {0, 0};
    for (int i = 0; i < 8; ++i) ++on[burst.place(request(base), states)];
    require(on[0] + on[1] == 8 && on[0] >= 3 && on[1] >= 3, "a hot prefix burst is spread within the imbalance bound");

    // A prompt shorter than one block has no affinity.
    require(router.matching_blocks("short", 0) == 0, "partial block is not hashed");
    // An unrelated prompt matches nothing.
    require(router.matching_blocks(prompt('Z', 4, 'Z', 4), 0) == 0 && router.matching_blocks(prompt('Z', 4, 'Z', 4), 1) == 0,
        "unrelated prompt has no chain");
}

void unhealthy_and_excluded() {
    ReplicaRouter router(3, 256, 64);
    const auto base = prompt('C', 4, 'C', 4);
    std::vector<ReplicaState> states{replica(0), replica(0), replica(0)};
    require(router.place(request(base), states) == 0, "placed on 0");
    router.finished(0, request(base));
    states[0].healthy = false;
    require(router.place(request(base), states) == 1, "prefix holder unhealthy: skipped");
    router.finished(1, request(base));
    states[0].healthy = true;
    require(router.place(request(base), states, 0b011) == 2, "excluded replicas are never chosen");
    router.finished(2, request(base));
    states[0].healthy = states[1].healthy = states[2].healthy = false;
    require(router.place(request(base), states) == kNoReplica, "no healthy replica");
    require(router.last_reason() == RouteReason::none, "no placement, no reason");
    require(router.inflight(0) + router.inflight(1) + router.inflight(2) == 0, "nothing recorded without a placement");
    states[2].healthy = true;
    require(router.place(request(base), states, 0b100) == kNoReplica, "only healthy replica excluded");

    router.forget(2);
    require(router.matching_blocks(base, 2) == 0 && router.matching_blocks(base, 1) == 4, "forget drops one replica only");

    bool rejected = false;
    try { (void)router.place(request(base), std::span<const ReplicaState>(states.data(), 2)); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "state count must match");
    rejected = false;
    try { ReplicaRouter bad(0); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "zero replicas rejected");
}

void status_and_request_parsing() {
    auto s = replica_state_from_status(
        R"({"state":"ready","capacity":{"slots_total":8,"slots_busy":3,"kv_bytes_free":null,"queue_depth":2}})");
    require(s && s->healthy && s->capacity.slots_total == 8 && s->capacity.slots_busy == 3
        && s->capacity.queue_depth == 2 && !s->capacity.kv_bytes_free, "status parsed");
    s = replica_state_from_status(R"({"state":"generating","capacity":{"slots_total":1,"slots_busy":1,"kv_bytes_free":4096,"queue_depth":0}})");
    require(s && s->healthy && s->capacity.kv_bytes_free && *s->capacity.kv_bytes_free == 4096, "kv bytes parsed");
    s = replica_state_from_status(R"({"state":"draining","capacity":{"slots_total":8,"slots_busy":0,"queue_depth":0}})");
    require(s && !s->healthy, "draining replica takes no new request");
    s = replica_state_from_status(R"({"state":"failed","capacity":{"slots_total":8}})");
    require(s && !s->healthy, "failed replica is unhealthy");
    s = replica_state_from_status(R"({"state":"ready"})");
    require(s && !s->healthy, "no capacity: not routable");
    require(!replica_state_from_status("not json") && !replica_state_from_status("[1]"), "non-object status rejected");

    RouteKeyStorage a, b, c;
    const auto ra = route_request_from_http("/v1/chat/completions", "",
        R"({"model":"m","messages":[{"role":"system","content":"S"},{"role":"user","content":"one"}],"stream":true})", a);
    const auto rb = route_request_from_http("/v1/chat/completions", "",
        R"({"stream":false,"messages":[{"role":"system","content":"S"},{"role":"user","content":"one"},{"role":"assistant","content":"two"}],"model":"m"})", b);
    require(ra.session_key.empty() && !ra.prompt_bytes.empty(), "prompt bytes from messages");
    // Shared leading messages give shared leading bytes whatever the member order of the request.
    require(rb.prompt_bytes.substr(0, ra.prompt_bytes.size() - 1) == ra.prompt_bytes.substr(0, ra.prompt_bytes.size() - 1),
        "conversation prefix is a byte prefix");
    require(ra.estimated_tokens == ra.prompt_bytes.size() / 4, "token estimate");

    auto rc = route_request_from_http("/v1/sessions/alpha/generate", "hdr", R"({"prompt":"p"})", c);
    require(rc.session_key == "alpha", "session id from the path");
    rc = route_request_from_http("/v1/completions", "hdr", R"({"prompt":"p","session_id":"body"})", c);
    require(rc.session_key == "hdr", "session header before body field");
    rc = route_request_from_http("/v1/completions", "", R"({"prompt":"p","session_id":"body"})", c);
    require(rc.session_key == "body" && rc.prompt_bytes == "\"p\"", "session id from the body");
    rc = route_request_from_http("/v1/completions", "", "raw bytes", c);
    require(rc.session_key.empty() && rc.prompt_bytes == "raw bytes", "non-JSON body is hashed as is");
    rc = route_request_from_http("/v1/models", "", "", c);
    require(rc.session_key.empty() && rc.prompt_bytes.empty(), "empty request");
}

} // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    try {
        least_load();
        session_stickiness();
        prefix_affinity();
        unhealthy_and_excluded();
        status_and_request_parsing();
    } catch (const std::exception& e) {
        std::cerr << "replica_router_tests: " << e.what() << "\n";
        return 1;
    }
    std::cout << "replica_router_tests: ok\n";
    return 0;
}
