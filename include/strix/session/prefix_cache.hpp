#pragma once

#include "strix/session/kv.hpp"
#include "strix/prefix_state_cache.hpp"

#include <cstdint>
#include <memory>
#include <map>
#include <functional>
#include <string>
#include <vector>

namespace strix::session {

// Adapters supply stable identity for model weights, numerical profile, state
// format/version and rank topology. Unknown/empty identity fields are refused.
struct Identity {
  std::string model;
  std::string profile;
  std::string state_format;
  std::uint32_t ranks = 0;
  [[nodiscard]] bool known() const noexcept {
    return !model.empty() && !profile.empty() && !state_format.empty() && ranks != 0;
  }
  bool operator==(const Identity&) const = default;
};

// Library metadata only; unrelated to family-owned session/session_id fields.
// model binds assets, profile binds arithmetic, state_format binds schema;
// topology names ordered rank layout and template_id binds rendering/tokenizer.
struct TurnMetadata {
  std::string conversation;
  std::string template_id;
  std::string topology;
  std::string keep; // optional unique protected name; explicit unpin before replace
  [[nodiscard]] bool known() const noexcept {
    return !conversation.empty() && !template_id.empty() && !topology.empty();
  }
};

struct PrefixSnapshot {
  Identity identity;
  StateSnapshot state;
  std::uint64_t attention_frontier = 0;
  std::uint64_t recurrent_frontier = 0;
  std::optional<TurnMetadata> turn;
};

// A free physical slot, not an immutable payload. Valid only while its store
// certifies the released generation and frontier; never exposed as StateSnapshot.
struct ResidentPrefix {
  Identity identity;
  KvHandle source;
  std::shared_ptr<const ResidentState> state;
  // Trailing rows of tokens() that are consumed reply rows rather than prompt rows (0 for a prefill record).
  std::uint64_t reply_rows = 0;
  [[nodiscard]] std::span<const std::uint32_t> tokens() const noexcept { return state->tokens(); }
};
struct ResidentHit {
  std::shared_ptr<const ResidentPrefix> entry;
  std::uint64_t rows = 0;
};

enum class EvictionPolicy { lru, fifo };
struct Policy {
  bool enabled = true;
  bool store = true;
  bool leave_one_token = false;
  std::uint64_t max_entries = 8;
  std::uint64_t max_bytes = UINT64_MAX;
  EvictionPolicy eviction = EvictionPolicy::lru;
  // Default refuses duplicate identity+token entries. Opt-in replacement freezes
  // new metadata, replaces byte accounting and refreshes BOTH insertion and use
  // order (so FIFO and LRU treat replacement as a new admission).
  bool replace_duplicates = false;
  // Default is a strict byte bound. Opt-in permits ONE sole checkpoint larger
  // than max_bytes, evicting all others; the next admission still enforces the
  // normal entry/byte limits. max_entries==0 continues to disable storage.
  bool retain_one_oversize = false;
  // Some physical cache contracts charge only resident state, excluding the
  // host token index. Defaults preserve the existing total-byte accounting.
  bool charge_token_bytes = true;
  // Explicit default-OFF. Requires finite nonzero max_bytes and LRU. The same
  // RAM/entry budget covers ordinary entries and protected completed turns.
  // Legacy sole-oversize and token-charge exemptions cannot bypass this bound.
  bool completed_turns = false;
};

// This cache accepts fully consumed checkpoints only: pending anchors are
// deliberately refused because suffix admission must consume them exactly.
// Admitted metadata is privately frozen; opaque payload owners are immutable
// snapshots by contract. Lookup ownership survives later eviction.
// Store-issued provenance is retained when present. Cache identity/token matches
// do not establish erased-payload compatibility: KvStore still validates its
// sealed origin/domain/schema and rank order before calling restore/fork hooks.
// Unsealed metadata may be indexed for legacy adapters, but KvStore refuses it.
class PrefixCache {
public:
  explicit PrefixCache(Policy policy = {}, std::shared_ptr<PrefixStateCache> owner = {});
  ~PrefixCache();
  PrefixCache(const PrefixCache&) = delete;
  PrefixCache& operator=(const PrefixCache&) = delete;
  [[nodiscard]] const std::shared_ptr<PrefixStateCache>& owner() const noexcept { return owner_; }
  bool admit(std::shared_ptr<const PrefixSnapshot>, PrefixStateCache::Reservation* = nullptr);
  bool admit_resident(const KvStore&, KvHandle, std::span<const std::uint32_t>, const Identity&,
                      std::uint64_t reply_rows = 0);
  [[nodiscard]] ResidentHit lookup_resident(const KvStore&, std::span<const std::uint32_t>,
      const Identity&, std::uint64_t alignment) const;
  void invalidate_resident(KvHandle) noexcept;
  void invalidate_residents() noexcept;
  [[nodiscard]] std::shared_ptr<const PrefixSnapshot> lookup(
      std::span<const std::uint32_t>, const Identity&) const;
  [[nodiscard]] std::shared_ptr<const PrefixSnapshot> lookup_turn(
      std::span<const std::uint32_t>, const Identity&, const TurnMetadata&) const;
  [[nodiscard]] bool turns_enabled() const noexcept;
  // Conservative declared payload bound, checked before consuming an anchor.
  [[nodiscard]] bool can_admit_turn(const Identity&, const TurnMetadata&,
      std::uint64_t tokens, std::uint64_t state_bytes) const;
  [[nodiscard]] PrefixStateCache::Reservation reserve_turn(const Identity&, const TurnMetadata&,
      std::uint64_t tokens, std::uint64_t state_bytes);
  // Kept entries live until explicit unpin or cache destruction. Unpin
  // removes protection, not the entry. Replacement cannot silently evict keeps.
  bool unpin(const std::string& keep);
  [[nodiscard]] std::uint64_t bytes() const noexcept { return owner_->retained_charge(scope_); }
  [[nodiscard]] std::size_t size() const noexcept;

private:
  [[nodiscard]] bool room(std::uint64_t, const TurnMetadata*) const;
  [[nodiscard]] std::shared_ptr<const PrefixSnapshot> find(
      std::span<const std::uint32_t>, const Identity&, const TurnMetadata*) const;
  [[nodiscard]] PrefixEntryPolicy retention(bool strict = false) const;
  Policy policy_;
  std::shared_ptr<PrefixStateCache> owner_;
  std::uint64_t scope_;
};

// No slab is allocated from a legacy unlimited sentinel. Host payload extents
// are adopted on demand, with physical credits held through the common ledger.
std::shared_ptr<PrefixStateCache> make_cache_owner(std::uint64_t budget);

// Named, opaque, immutable encoded checkpoints. Physical codecs, paired
// agreement and durable manifests stay with the family. This owner supplies
// RAM retention/replacement, protected restore peaks and deterministic LRU.
// It never admits a checkpoint larger than its configured budget.
class NamedStateCache {
public:
  using Bytes = std::vector<std::uint8_t>;
  using Entry = PrefixStateCache::NamedView;
  NamedStateCache(std::uint64_t bytes, std::uint64_t entries, std::shared_ptr<PrefixStateCache> owner = {});
  ~NamedStateCache();
  NamedStateCache(const NamedStateCache&) = delete;
  NamedStateCache& operator=(const NamedStateCache&) = delete;
  [[nodiscard]] const std::shared_ptr<PrefixStateCache>& owner() const noexcept { return owner_; }
  [[nodiscard]] std::uint64_t bytes() const noexcept;
  [[nodiscard]] std::uint64_t evictions() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept { return owner_->entry_count(policy_.scope); }
  // Borrowed nonowning directory; the core removes descriptors before eviction.
  [[nodiscard]] const std::map<std::string, Entry>& entries() const noexcept;
  [[nodiscard]] const Bytes* find(const std::string&) const noexcept;
  // Concurrent/restore readers retain this immutable lease across eviction.
  [[nodiscard]] std::shared_ptr<const PrefixCacheCheckpoint> lease(const std::string&) const;
  void touch(const std::string&);
  void erase(const std::string&);
  bool put(const std::string&, Bytes, PrefixStateCache::Reservation* = nullptr);
  bool evict_until(const std::function<bool(std::uint64_t)>& fits, const std::optional<std::string>& protected_id = {});
private:
  std::shared_ptr<PrefixStateCache> owner_;
  PrefixEntryPolicy policy_;
};

} // namespace strix::session
