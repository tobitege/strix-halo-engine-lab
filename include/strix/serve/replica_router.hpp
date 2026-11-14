#pragma once

// Placement policy for N independent engine replicas (one full engine per node). Pure host: descriptors in, replica
// index out. No sockets, no engine types. A front end fills ReplicaState from each replica's status route and calls
// place() / finished() around every forwarded request.

#include "strix/serve/deployment.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::serve {

struct ReplicaState {
    ServeCapacity capacity;
    bool healthy = false;
    double ewma_ms_per_token = 0; // 0 = unknown
};

struct RouteRequest {
    std::string_view session_key;   // named session / conversation id; empty = none
    std::string_view prompt_bytes;  // rendered messages, for prefix affinity
    std::uint64_t estimated_tokens = 0;
};

inline constexpr std::size_t kNoReplica = static_cast<std::size_t>(-1);

// Why place() chose a replica; reported by front ends, not used by the policy.
enum class RouteReason : std::uint8_t { none, session, prefix, load };
[[nodiscard]] std::string_view route_reason_name(RouteReason reason) noexcept;

// Policy order:
//  1. a named session goes to the replica that holds it (while that replica is healthy);
//  2. the replica with the longest matching chain of prompt block hashes, unless its load exceeds the least loaded
//     healthy replica by more than one slot;
//  3. least load, then lower ewma_ms_per_token, then lower index.
// Load = max(slots_busy + queue_depth as reported, requests this router placed there and has not finished): the
// second term covers placements made between two status reads.
// Unhealthy replicas are never chosen; with none healthy place() returns kNoReplica and records nothing.
// Thread-safe. Memory is fixed at construction (two direct-mapped tables of affinity_entries slots).
class ReplicaRouter {
public:
    static constexpr std::size_t kMaxReplicas = 64;
    explicit ReplicaRouter(std::size_t replicas, std::size_t affinity_entries = 4096, std::size_t block_bytes = 1024);

    // Returns the replica index, or kNoReplica. states.size() must equal replicas().
    std::size_t place(const RouteRequest& request, std::span<const ReplicaState> states);
    // As place(), never choosing a replica whose bit is set in `excluded` (a replica that just refused a connection).
    std::size_t place(const RouteRequest& request, std::span<const ReplicaState> states, std::uint64_t excluded);
    void finished(std::size_t replica, const RouteRequest& request) noexcept;

    // A replica restarted or went away: drop the sessions and prefix chains recorded for it.
    void forget(std::size_t replica) noexcept;

    [[nodiscard]] std::size_t replicas() const noexcept { return inflight_.size(); }
    [[nodiscard]] std::uint64_t inflight(std::size_t replica) const noexcept;
    [[nodiscard]] RouteReason last_reason() const noexcept;
    // Number of leading full prompt blocks recorded for `replica` (test and status aid).
    [[nodiscard]] std::size_t matching_blocks(std::string_view prompt_bytes, std::size_t replica) const noexcept;

private:
    struct Session { std::uint64_t key = 0; std::uint32_t replica_plus_one = 0; };
    struct Chain { std::uint64_t key = 0; std::uint64_t mask = 0; };
    template<class F> void for_each_block(std::string_view prompt, F&& visit) const;

    mutable std::mutex mutex_;
    std::size_t block_bytes_;
    std::vector<std::uint64_t> inflight_;
    std::vector<Session> sessions_;
    std::vector<Chain> chains_;
    RouteReason last_reason_ = RouteReason::none;
};

// ReplicaState from the body of a replica's status route (`capacity` object of deployment_status, `state`).
// healthy = the body parses, carries a capacity with slots_total > 0 and the state is neither failed nor draining.
// nullopt when the body is not a JSON object.
[[nodiscard]] std::optional<ReplicaState> replica_state_from_status(std::string_view status_json);

// The routing view of one API request. `storage` owns the bytes the returned views point into.
struct RouteKeyStorage { std::string session, prompt; };
[[nodiscard]] RouteRequest route_request_from_http(std::string_view path, std::string_view session_header,
    std::string_view body, RouteKeyStorage& storage);

} // namespace strix::serve
