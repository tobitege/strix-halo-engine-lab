#pragma once
// Retention policy, index and byte ledger for kept lane states (conversation / prompt prefix cache).
//
// Pure host code: no device types. The family keeps the payload (a lane snapshot) under the opaque entry id and
// asks this class what to keep, what to evict and what restores a prompt. Every decision is a deterministic
// function of the call sequence, so two ranks fed the same calls make the same decisions; a pair lets rank 0
// choose (prepare -> victims -> erase on both -> commit) and rank 1 replay.
//
// An entry is addressed by (identity, key token ids). It restores a prompt that starts with the key when the
// restore frontier (rows held by the snapshot) is below the prompt size; the longest such key wins.
#include <cstdint>
#include <list>
#include <limits>
#include <span>
#include <vector>

#include "strix/serve/prefix_directory.hpp"

namespace strix::serve {

enum class KeepKind : std::uint8_t { prompt_end = 0, reply_end = 1 };
inline constexpr std::size_t kKeepKinds = 2;

struct KeepIdentity {
    std::uint64_t model = 0;    // model + engine configuration id
    std::uint64_t variant = 0;  // family variant word (for example MTP on/off)
    bool operator==(const KeepIdentity&) const = default;
};

struct KeepCandidate {
    std::uint64_t id = 0;  // opaque, unique, nonzero; chosen by the caller (both ranks use the same id)
    KeepIdentity identity;
    KeepKind kind = KeepKind::prompt_end;
    std::vector<std::uint32_t> ids;  // key tokens
    std::uint64_t frontier = 0;      // rows the snapshot restores
    std::uint64_t payload_bytes = 0; // family payload size
};

struct KeepEntry {
    std::uint64_t id = 0;
    KeepIdentity identity;
    KeepKind kind = KeepKind::prompt_end;
    std::uint64_t frontier = 0, payload_bytes = 0, hash = 0, bytes = 0;
    std::uint32_t leases = 0;
    std::vector<std::uint32_t> ids;
};

enum class KeepEvict : std::uint8_t { lru, replaced, pressure };

struct KeepStats {
    std::uint64_t hits = 0, misses = 0, stores = 0, refused = 0, evictions = 0, pressure_evictions = 0;
    std::uint64_t bytes = 0, entries = 0, budget = 0;
    std::uint64_t hit_tokens[kKeepKinds] = {0, 0};
    std::uint64_t hits_by_kind[kKeepKinds] = {0, 0};
    std::uint64_t stores_by_kind[kKeepKinds] = {0, 0};
};

class ConversationKeep {
public:
    static constexpr std::uint64_t kEntryOverhead = 128;  // index node + key bookkeeping charged per entry
    // hash == nullptr selects prefix_block_hash; tests pass a weak hash to force collisions.
    explicit ConversationKeep(std::uint64_t budget = 0, PrefixBlockHash hash = nullptr);

    static std::uint64_t charged_bytes(std::uint64_t payload_bytes, std::size_t key_tokens) noexcept {
        constexpr auto max = std::numeric_limits<std::uint64_t>::max();
        if (payload_bytes > max - kEntryOverhead ||
            key_tokens > (max - kEntryOverhead - payload_bytes) / sizeof(std::uint32_t)) return max;
        return payload_bytes + key_tokens * sizeof(std::uint32_t) + kEntryOverhead;
    }

    // Longest kept key that is a strict prefix of `prompt` with frontier < prompt size (hash candidate, then a
    // full token compare). find() has no side effects; lookup() counts a hit / miss (and hit tokens by kind).
    const KeepEntry* find(const KeepIdentity& identity, std::span<const std::uint32_t> prompt) const;
    const KeepEntry* lookup(const KeepIdentity& identity, std::span<const std::uint32_t> prompt);
    const KeepEntry* get(std::uint64_t id) const;

    // Two-phase store. prepare() stages one candidate (refuses and counts it when the budget is 0, the payload is
    // empty, the id is 0 / kept, or it alone exceeds the budget). victims() is the ordered eviction list that
    // makes room: the equal-key entry first, then LRU; false when leases make that impossible. The caller erases
    // the victims (on every rank), then commit(true) admits the candidate; commit(false) drops it.
    bool prepare(KeepCandidate candidate);
    bool has_pending() const { return !pending_.empty(); }
    std::uint64_t pending_bytes() const { return has_pending() ? pending_.front().bytes : 0; }
    // Before allocating a payload, bound retained + staging ownership by the grant.
    bool can_stage(std::uint64_t payload_bound, std::size_t tokens) const;
    std::uint64_t replacement_id() const;  // kept entry with the pending key (0: none)
    bool victims(std::vector<std::uint64_t>& out) const;
    bool pending_fits() const;  // no equal key left and bytes + pending <= budget
    bool commit(bool accept) noexcept;   // allocation-free transfer of the prepared node

    // Single-process convenience: prepare + victims + erase + commit.
    bool store(KeepCandidate candidate);

    bool erase(std::uint64_t id, KeepEvict reason = KeepEvict::lru);
    void touch(std::uint64_t id);
    void clear();

    // Leases: a leased entry (being restored) is never a victim.
    bool acquire(std::uint64_t id);
    void release(std::uint64_t id);

    // Pressure: ordered LRU victims (never leased) freeing at least `target_bytes` (fewer when nothing more can
    // go). shrink_plan() brings bytes under a lowered budget. set_budget() only moves the ceiling.
    std::vector<std::uint64_t> reclaim_plan(std::uint64_t target_bytes) const;
    std::vector<std::uint64_t> shrink_plan() const;
    void set_budget(std::uint64_t budget) { budget_ = budget; }

    std::uint64_t budget() const { return budget_; }
    std::uint64_t bytes() const { return bytes_; }
    std::size_t size() const { return entries_.size(); }
    std::uint64_t oldest() const;  // LRU entry that is not leased (0: none)
    KeepStats stats() const;

private:
    std::uint64_t key_hash(const KeepIdentity& identity, std::span<const std::uint32_t> ids) const;
    std::uint64_t budget_ = 0, bytes_ = 0;
    PrefixBlockHash hash_;
    std::list<KeepEntry> entries_;  // front = least recently used
    std::list<KeepEntry> pending_;  // node allocated before the paired vote; commit only splices
    KeepStats counters_;
};

}  // namespace strix::serve
