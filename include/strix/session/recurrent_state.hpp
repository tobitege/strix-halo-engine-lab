#pragma once

#include "strix/session/backend.hpp"
#include "strix/session/kv.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix::session {

// Absolute model-input frontier, never a generated-output count. pending is an
// admitted anchor at consumed, not yet evaluated. Matches Engine::Frontier.
struct LaneFrontier {
    std::uint64_t consumed = 0;
    std::optional<std::uint32_t> pending;
    bool operator==(const LaneFrontier&) const = default;
};

struct LaneIdentity {
    std::uint32_t version = 1, ranks = 0, rank = 0;
    std::string schema, topology, model, numerical, base;
    bool operator==(const LaneIdentity&) const = default;
    void validate() const {
        if (version != 1 || !ranks || rank >= ranks || schema.empty() ||
            topology.empty() || model.empty() || numerical.empty() || base.empty())
            throw std::invalid_argument("unknown lane identity");
    }
};

// These are interpretation authority, not textual identity or wire IDs. Share
// only for identical payload types, model/assets, arithmetic and rank ordering,
// exactly as Hooks::snapshot_domain/snapshot_schema. No global registry.
struct LaneTrust {
    std::shared_ptr<const void> domain, schema;
    static bool same(const std::shared_ptr<const void>& a,
                     const std::shared_ptr<const void>& b) noexcept {
        return a && b && a.get() == b.get() &&
               !a.owner_before(b) && !b.owner_before(a);
    }
    bool accepts(const LaneTrust& other) const noexcept {
        return same(domain, other.domain) && same(schema, other.schema);
    }
};

// Admission precedes metadata allocation, backend writes and pending handling.
// Empty input is a no-op, including the anchor. The caller also checks handle
// provenance, terminal/in-flight state and every rank's coupled frontier.
inline LaneFrontier validate_lane_append(LaneFrontier before,
        std::span<const std::uint32_t> input, std::uint64_t capacity,
        std::uint64_t vocabulary) {
    if (!vocabulary || before.consumed > capacity ||
        (before.pending && (before.consumed == capacity || *before.pending >= vocabulary)))
        throw std::invalid_argument("invalid lane frontier");
    const auto logical = before.consumed + (before.pending ? 1U : 0U);
    if (input.size() > capacity - logical)
        throw std::length_error("lane context capacity");
    for (auto id : input)
        if (id >= vocabulary) throw std::invalid_argument("invalid input token");
    if (input.empty()) return before;
    // This reference admission consumes the complete supplied suffix. Adapters
    // choosing a final-pending route must report that route explicitly instead.
    return {logical + input.size(), {}};
}

inline void validate_lane_checkpoint(const StateSnapshot& checkpoint, LaneFrontier target) {
    if (!checkpoint.sealed() || checkpoint.consumed != target.consumed ||
        checkpoint.pending != target.pending)
        throw std::invalid_argument("rollback needs a matching complete checkpoint");
    // KvStore performs origin/domain/schema/rank checks before erased hooks.
}

// Typed ownership utility for adapter rank payloads. Payload owns recurrence /
// convolution storage; it is NOT a complete engine snapshot. Clone must create
// independent storage, including subresources. No callback may retain writable
// aliases. Ordinary shared mutable pointers do not satisfy this contract.
template<class Payload> class LaneState {
public:
    using FactorLog = std::vector<std::uint8_t>; // adapter-defined immutable encoding
    struct Checkpoint {
        const LaneIdentity identity;
        const LaneTrust trust;
        const LaneFrontier frontier;
        const Payload payload;
    };

    static LaneState create(LaneIdentity identity, LaneTrust trust,
            LaneFrontier frontier, Payload payload, FactorLog log = {}) {
        identity.validate();
        if (!trust.domain || !trust.schema)
            throw std::invalid_argument("lane interpretation tokens required");
        auto base = std::make_shared<const Checkpoint>(Checkpoint{
            std::move(identity), std::move(trust), frontier, std::move(payload)});
        auto factors = std::make_shared<const FactorLog>(std::move(log));
        return LaneState(std::move(base), std::move(factors), frontier);
    }
    LaneState(LaneState&&) noexcept = default;
    LaneState& operator=(LaneState&&) noexcept = default;
    LaneState(const LaneState&) = delete;
    LaneState& operator=(const LaneState&) = delete;

    const std::shared_ptr<const Checkpoint>& base() const noexcept { return base_; }
    const std::shared_ptr<const FactorLog>& log() const noexcept { return log_; }
    LaneFrontier frontier() const noexcept { return frontier_; }
    bool has_private_dense() const noexcept { return bool(dense_); }

    void validate(const LaneIdentity& expected, const LaneTrust& trust) const {
        expected.validate();
        if (!base_ || !log_ || base_->identity != expected || !base_->trust.accepts(trust))
            throw std::invalid_argument("incompatible lane identity/domain/schema/base");
    }
    void require_dense_reference() const {
        if (!log_ || !log_->empty())
            throw UnsupportedOperation("factor-log evaluation is not implemented");
    }
    const Payload& read(const LaneIdentity& expected, const LaneTrust& trust) const {
        validate(expected, trust); // before interpreting any payload
        require_dense_reference();
        return dense_ ? *dense_ : base_->payload;
    }
    // Both snapshots and forks share the immutable checkpoint/log but clone any
    // dense continuation. This deliberately costs O(dense state), not O(1).
    template<class Clone> LaneState clone(const LaneIdentity& expected,
            const LaneTrust& trust, Clone&& clone_payload) const {
        validate(expected, trust);
        require_dense_reference();
        LaneState result(base_, log_, frontier_);
        if (dense_) result.dense_ = checked_clone(clone_payload, *dense_);
        return result;
    }
    // Prepare off to the side; publish only after the complete rank operation
    // succeeds. Full-lane orchestration must additionally restore/invalidate all
    // attempted ranks, including a throwing rank, through KvStore.
    template<class Clone, class Work> void update(const LaneIdentity& expected,
            const LaneTrust& trust, LaneFrontier after, Clone&& clone_payload, Work&& work) {
        const auto& current = read(expected, trust);
        auto next = checked_clone(clone_payload, current);
        std::forward<Work>(work)(*next);
        dense_.swap(next);
        frontier_ = after;
    }
    template<class Clone> void densify(const LaneIdentity& expected,
            const LaneTrust& trust, Clone&& clone_payload) {
        update(expected, trust, frontier_, std::forward<Clone>(clone_payload), [](Payload&) {});
    }

private:
    std::shared_ptr<const Checkpoint> base_;
    std::shared_ptr<const FactorLog> log_;
    std::unique_ptr<Payload> dense_;
    LaneFrontier frontier_;
    LaneState(std::shared_ptr<const Checkpoint> base, std::shared_ptr<const FactorLog> log,
              LaneFrontier frontier) : base_(std::move(base)), log_(std::move(log)), frontier_(frontier) {}
    template<class Clone> static std::unique_ptr<Payload> checked_clone(Clone& clone, const Payload& value) {
        auto result = clone(value);
        if (!result) throw std::runtime_error("null dense clone");
        return result;
    }
};

enum class LaneSerializationStatus { supported, unsupported };
struct LaneSerialization {
    LaneSerializationStatus status = LaneSerializationStatus::unsupported;
    std::vector<std::uint8_t> bytes;
    std::string reason;
};

// Session-level operations; no math, sampler, scheduling or pair wire protocol.
// All calls are serialized and quiescent, outside hot loops. Adapter owns the
// KvStore and keeps existing Engine hooks/arithmetic. Attention-only adapters
// use genuine KV page COW with no recurrent payload. Dense recurrent adapters
// embed LaneState in each complete immutable rank owner captured by the store.
class RecurrentStateOps {
public:
    virtual ~RecurrentStateOps() = default;
    // Includes recurrent/conv, attention, drafter, RNG, selection/policy cursors
    // on EVERY rank, plus consumed history/pending. Recurrence alone cannot resume.
    virtual StateSnapshot snapshot(KvHandle) const = 0;
    // Validate sealed provenance and capacity before allocation. Failure cleans
    // up/quarantines the unpublished generation; source lifetime is irrelevant.
    virtual KvHandle fork(const StateSnapshot&) = 0;
    // expected rejects stale coordination frontiers. Teacher-forced backend
    // evaluation consumes the anchor once; inputs produce no outputs/RNG draws.
    virtual void append(KvHandle, LaneFrontier expected, std::span<const std::uint32_t>) = 0;
    // Absolute target AND matching complete checkpoint; no counter-only rewind.
    virtual void rollback(KvHandle, LaneFrontier target, const StateSnapshot&) = 0;
    // Preserve logical state/frontier/cursors/base; rollback every attempted rank
    // on failure or invalidate this lane. Never publish mutable shared recurrence.
    virtual void densify(KvHandle) = 0;
    // Explicit version/schema/topology/model/numerical/base and bounded lengths.
    // Unsupported opaque/device formats return unsupported with no bytes. Never
    // emit pointer values/raw C++ memory. Bytes alone confer no restore authority.
    virtual LaneSerialization serialize(const StateSnapshot&) const = 0;
};

} // namespace strix::session
