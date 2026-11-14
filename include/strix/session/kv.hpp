#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace strix::session {

struct KvHandle {
  std::uint32_t slot = UINT32_MAX;
  std::uint64_t generation = 0;
  std::uint64_t store = 0;
  [[nodiscard]] bool valid() const noexcept {
    return slot != UINT32_MAX && generation != 0 && store != 0;
  }
  bool operator==(const KvHandle&) const = default;
};

class ResidentState {
public:
  [[nodiscard]] KvHandle source() const noexcept { return source_; }
  [[nodiscard]] std::span<const std::uint32_t> tokens() const noexcept { return tokens_; }
private:
  KvHandle source_;
  std::vector<std::uint32_t> tokens_;
  ResidentState(KvHandle h, std::vector<std::uint32_t> tokens) : source_(h), tokens_(std::move(tokens)) {}
  friend class KvStore;
};

struct RankState {
  std::shared_ptr<const void> owner;
  std::uint64_t bytes = 0;
};
enum class SnapshotPurpose { resume, prompt_reuse };

// Opaque owners must represent immutable snapshots and include all rank-local
// attention, recurrent, draft, RNG, policy and cursor state required to resume.
struct StateSnapshot {
  std::uint64_t consumed = 0;
  std::optional<std::uint32_t> pending;
  std::vector<std::uint32_t> tokens; // exactly [0, consumed)
  std::vector<RankState> ranks;
  std::uint64_t bytes = 0; // sum of opaque rank payload bytes
  // True only for an unmodified store-issued checkpoint. This is provenance,
  // not a claim that another store can interpret its erased payloads.
  [[nodiscard]] bool sealed() const noexcept;
  // Opaque copyable value preserves aggregate initialization of the five legacy
  // fields. Copying this value cannot authorize substituted metadata/payloads:
  // sealed() checks the complete original checkpoint, including rank order.
  class Provenance {
  public:
    Provenance() = default;
  private:
    struct Seal;
    std::shared_ptr<const Seal> seal_;
    friend class KvStore;
    friend struct StateSnapshot;
  };
  Provenance provenance;
  SnapshotPurpose purpose = SnapshotPurpose::resume;
private:
  using Seal = Provenance::Seal;
  friend class KvStore;
};

// initialize/reset/release acquire, clear and relinquish physical state for a
// specific slot generation. release must be safe after partial initialization.
// restore/truncate may throw after mutation: the store rolls back EVERY attempted
// rank, including the throwing rank, or poisons the lane. Hooks are trusted code;
// capture must freeze all payloads, including pages still owned by a live lane.
// No mutable alias may bypass their write barrier. Calls are externally serialized
// and quiescent. Hooks always receive slot identity as well as rank.
struct Hooks {
  std::function<void(KvHandle)> initialize;
  std::function<void(KvHandle)> reset;
  std::function<void(KvHandle)> release;
  std::function<RankState(KvHandle, std::uint32_t)> view;
  std::function<RankState(KvHandle, std::uint32_t)> capture;
  std::function<void(KvHandle, std::uint32_t, const RankState&)> restore;
  std::function<void(KvHandle, std::uint32_t, std::uint64_t)> truncate;
  bool partial_restore_supported = false;
  // Explicit shared-prefix contract. fork_rank shares immutable attention page
  // owners (zero prefix-byte copies), detaches only pages touched by later writes,
  // and independently restores recurrent/draft/RNG/selection/policy/cursor state.
  // An eager full-prefix copy does NOT implement this hook. Existing backend
  // page-table implementations may supply it directly; no shared kernels needed.
  bool copy_on_write_supported = false;
  std::function<void(KvHandle, std::uint32_t, const RankState&)> fork_rank;
  // Cross-store interpretation is OFF unless BOTH tokens are non-null and equal
  // by shared ownership AND address. Adapters share these trusted tokens only
  // for identical payload C++ types, schema, rank order/topology, model/assets,
  // numerical identity and continuation semantics. Equal rank counts or cache
  // token IDs alone never grant compatibility. Tokens/assets outlive snapshots.
  std::shared_ptr<const void> snapshot_domain;
  std::shared_ptr<const void> snapshot_schema;
  // Prompt-only checkpoints freeze physical consumed rows and hidden context,
  // deliberately exclude the old pending token and sampling/draft cursor, and
  // carry a distinct sealed purpose. restore_prompt builds a fresh sampler and
  // drafter from the NEW job; false declines reuse without changing the lane.
  // The restore hook validates/prepares ALL ranks before atomically installing
  // them. A throwing hook poisons the destination; no partial reuse is exposed.
  std::function<RankState(KvHandle, std::uint32_t)> capture_prompt;
  std::function<bool(KvHandle, std::span<const RankState>)> restore_prompt;
  // Explicit physical-lifecycle promise: release cancels suffix work while
  // preserving certified prefill rows; initialize does not clear those rows.
  bool resident_prefill_supported = false;
};

class KvStore {
public:
  KvStore(std::uint32_t slots, std::uint32_t ranks, Hooks hooks);
  // Hooks and their resources must outlive this destructor. It attempts release
  // of all live/quarantined generations without throwing. Backends must reclaim
  // any failed releases at their own teardown. Snapshot owners retain their own
  // resources independently, even after this store/source/cache has gone away.
  ~KvStore();
  KvStore(const KvStore&) = delete;
  KvStore& operator=(const KvStore&) = delete;
  KvStore(KvStore&&) = delete;
  KvStore& operator=(KvStore&&) = delete;

  KvHandle acquire();
  // Resident reuse is not a snapshot restore. The serialized owner certifies a
  // completed prefill frontier before release; only that free generation may be
  // reacquired, before physical overwrite, using the cache-owned token index.
  KvHandle acquire_resident(KvHandle, std::span<const std::uint32_t>);
  [[nodiscard]] std::shared_ptr<const ResidentState> release_resident(KvHandle, std::uint64_t safe_prefill_rows);
  [[nodiscard]] std::shared_ptr<const ResidentState> resident_prefix(KvHandle) const noexcept;
  [[nodiscard]] bool resident_valid(KvHandle, std::uint64_t rows) const noexcept;
  void invalidate_residents() noexcept;
  // Validate provenance/metadata before allocation or any erased-payload hook.
  // Reusable after source release. Failure releases the unpublished lane; failed
  // cleanup quarantines capacity, never returns a partial valid handle.
  [[nodiscard]] KvHandle fork(const StateSnapshot&);
  [[nodiscard]] bool supports_fork() const noexcept;
  void reset(KvHandle);
  void release(KvHandle);
  void invalidate(KvHandle) noexcept;
  // Metadata only: Engine must reserve before physical work, evaluate through
  // family hooks, then commit IDs. These methods do not evaluate model inputs.
  void reserve_consumed(KvHandle, std::uint64_t additional);
  // If pending exists, a nonempty append starts with that exact anchor, consumes
  // it once and clears pending. Empty append is a no-op, including pending.
  void append_consumed(KvHandle, std::span<const std::uint32_t>);
  void set_pending(KvHandle, std::optional<std::uint32_t>);
  [[nodiscard]] std::uint64_t consumed(KvHandle) const;
  [[nodiscard]] std::optional<std::uint32_t> pending(KvHandle) const;
  [[nodiscard]] std::vector<std::uint32_t> tokens(KvHandle) const;
  [[nodiscard]] RankState view(KvHandle, std::uint32_t rank) const;
  [[nodiscard]] StateSnapshot capture(KvHandle) const;
  [[nodiscard]] StateSnapshot capture_prompt(KvHandle) const;
  bool restore_prompt(KvHandle, const StateSnapshot&);
  void restore(KvHandle, const StateSnapshot&);
  // Retains exactly [0,prefix), discarding pending even when prefix==consumed.
  // At that boundary no physical rows or committed RNG draws are rewound.
  void truncate(KvHandle, std::uint64_t prefix);
  [[nodiscard]] std::uint32_t ranks() const noexcept { return ranks_; }

private:
  struct Slot {
    std::uint64_t generation = 0;
    bool active = false;
    bool poisoned = false;
    std::uint64_t consumed = 0;
    std::uint64_t resident_rows = 0; // inactive generation only, no token index
    std::weak_ptr<const ResidentState> resident; // cache alone retains token storage
    std::optional<std::uint32_t> pending;
    std::vector<std::uint32_t> tokens;
  };
  std::vector<Slot> slots_;
  std::uint32_t ranks_;
  std::uint64_t store_id_;
  Hooks hooks_;
  KvHandle acquire_slot(std::uint32_t);
  void check(KvHandle, bool allow_poisoned = false) const;
  Slot& checked(KvHandle, bool allow_poisoned = false);
  void validate_snapshot(const StateSnapshot&) const;
  [[nodiscard]] StateSnapshot capture_impl(KvHandle, SnapshotPurpose) const;
};

} // namespace strix::session
