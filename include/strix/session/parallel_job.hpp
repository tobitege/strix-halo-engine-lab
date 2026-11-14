#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strix::session {

// Copies construction input, including rvalues: no retained mutable alias can
// modify the allocation. Copies share only immutable storage. No native state.
class TokenBuffer {
public:
    TokenBuffer() = default;
    explicit TokenBuffer(std::span<const std::uint32_t> tokens)
        : tokens_(std::make_shared<const std::vector<std::uint32_t>>(tokens.begin(), tokens.end())) {}
    [[nodiscard]] std::span<const std::uint32_t> tokens() const noexcept {
        return tokens_ ? std::span<const std::uint32_t>(*tokens_) : std::span<const std::uint32_t>{};
    }
private:
    std::shared_ptr<const std::vector<std::uint32_t>> tokens_;
};

// Half-open token interval [offset, offset+length), checked without overflow.
// An empty reference is valid, including the default empty buffer.
struct TokenReference {
    TokenBuffer buffer;
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
    [[nodiscard]] std::span<const std::uint32_t> tokens() const;
};

// Lengths count generated IDs, including stop IDs, against the ONE job budget.
// These are per-point bounds, never additional independent request budgets.
struct Contract {
    std::string task;
    std::uint32_t min_tokens = 0;
    std::uint32_t max_tokens = 1;
    std::vector<std::string> evidence_ids;
};
struct PointSpec {
    std::string id;
    Contract contract;
    TokenReference suffix;
};
struct DependencyEdge { std::string before, after; };
// Immutable tokenized evidence owned by a declared producer. Consumers must
// name it in their contract and be downstream of that producer in the DAG.
// Digest selection, grammar and textual audit are protocol responsibilities.
struct Digest {
    std::string id;
    std::string producer;
    TokenReference evidence;
};
struct MergeSpec {
    std::vector<std::string> point_order; // topological permutation of all IDs
    std::string instruction;
};
struct JobPolicy {
    std::uint64_t token_budget = 0; // total generated IDs across points AND merge
    std::uint32_t max_lanes = 1;   // requested concurrency, not admitted capacity
};
struct ParallelJob {
    std::string request_id;
    TokenReference prefix;
    std::vector<PointSpec> points;
    std::vector<DependencyEdge> dependencies;
    std::vector<Digest> digests;
    MergeSpec merge;
    JobPolicy policy;
};
// Owning request-wide accounting; no per-lane budget or scheduler state.
struct JobReceipt {
    std::string request_id;
    std::uint64_t budget_tokens = 0;
    std::uint64_t output_tokens = 0;
    std::vector<std::string> completed_points;
};

// Pure structural validation; throws std::invalid_argument, never changes job
// or engine state. Allocation failure propagates. Validate the exact owning
// value before admission; mutating its public fields requires revalidation.
// Detailed PLAN rendering/auditing and cost/admission policy are separate APIs.
void validate_job(const ParallelJob& job);

// Minimal consumer (link strix_session, C++20; no family/server/backend needed):
// ParallelJob job;
// job.request_id = "request";
// job.points = {{"point", {"answer", 0, 16, {}}, {}}};
// job.merge.point_order = {"point"};
// job.policy = {32, 1};
// validate_job(job);

} // namespace strix::session
