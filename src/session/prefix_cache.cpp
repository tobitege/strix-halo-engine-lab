#include "strix/session/prefix_cache.hpp"
#include "strix/model_content_hash.hpp"
#include "../core/byte_budget.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strix::session {
namespace {
bool safe_checkpoint(const PrefixSnapshot& snapshot) {
  const auto& state = snapshot.state;
  if (!snapshot.identity.known() || state.pending || state.ranks.size() != snapshot.identity.ranks ||
      state.consumed != state.tokens.size() || state.consumed == 0 ||
      snapshot.attention_frontier != state.consumed || snapshot.recurrent_frontier != state.consumed)
    return false;
  std::uint64_t rank_bytes = 0;
  for (const auto& rank : state.ranks) {
    if (!rank.owner || rank.bytes > UINT64_MAX - rank_bytes) return false;
    rank_bytes += rank.bytes;
  }
  if (rank_bytes != state.bytes || state.tokens.size() >
      (UINT64_MAX - rank_bytes) / sizeof(std::uint32_t)) return false;
  return true;
}

std::size_t prefix_length(std::span<const std::uint32_t> a, std::span<const std::uint32_t> b) {
  std::size_t n = 0;
  while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
  return n;
}
bool same_turn(const TurnMetadata& a, const TurnMetadata& b) {
  return a.conversation == b.conversation && a.template_id == b.template_id && a.topology == b.topology;
}
// Charge both token/rank indices (public metadata and the provenance seal),
// list/snapshot ownership and string storage. The fixed envelope conservatively
// covers the private seal and shared-owner control metadata. Payload bytes are
// the family's complete retained allocation declaration, never KV-row estimates.
std::optional<std::uint64_t> charge(const Identity& id, const TurnMetadata* turn,
    std::uint64_t tokens, std::uint64_t payload) {
  auto n = payload;
  auto add = [&](std::uint64_t x) { if (x > UINT64_MAX - n) return false; n += x; return true; };
  if (tokens > UINT64_MAX / (2 * sizeof(std::uint32_t)) ||
      !add(tokens * 2 * sizeof(std::uint32_t)) ||
      !add(std::uint64_t(id.ranks) * 2 * sizeof(RankState)) ||
      !add(sizeof(PrefixSnapshot) + sizeof(PrefixCacheCheckpoint) + 2 * sizeof(std::uint64_t) + 8 * sizeof(void*) + 256)) return {};
  for (const auto* str : {&id.model, &id.profile, &id.state_format})
    if (!add(str->size()) || !add(1)) return {};
  if (turn) for (const auto* str : {&turn->conversation, &turn->template_id, &turn->topology, &turn->keep})
    if (!add(str->size()) || !add(1)) return {};
  return n;
}
}

bool PrefixCache::turns_enabled() const noexcept {
  return policy_.enabled && policy_.completed_turns && policy_.max_entries &&
      policy_.max_bytes && policy_.max_bytes != UINT64_MAX && policy_.eviction == EvictionPolicy::lru;
}

std::shared_ptr<PrefixStateCache> make_cache_owner(std::uint64_t budget) {
  PrefixStateCachePolicy p;
  p.host_byte_budget = std::max<std::uint64_t>(1, budget);
  p.retention = PrefixCacheRetention::lru;
  return std::make_shared<PrefixStateCache>(p, [](std::uint64_t n) {
    if (n > SIZE_MAX) throw PrefixCacheAllocationFailure();
    return PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[static_cast<std::size_t>(n ? n : 1)],
        std::default_delete<std::byte[]>()), n};
  });
}
PrefixCache::PrefixCache(Policy policy, std::shared_ptr<PrefixStateCache> owner)
    : policy_(policy), owner_(owner ? std::move(owner) : make_cache_owner(policy.max_bytes)), scope_(owner_->new_scope()) {}
PrefixCache::~PrefixCache() { owner_->remove(scope_, [](const auto&) { return true; }); }
PrefixEntryPolicy PrefixCache::retention(bool strict) const {
  return {scope_, policy_.max_bytes, policy_.max_entries, policy_.eviction == EvictionPolicy::fifo,
      !strict && !policy_.completed_turns && policy_.retain_one_oversize};
}
std::size_t PrefixCache::size() const noexcept { return owner_->entry_count(scope_); }
bool PrefixCache::room(std::uint64_t total, const TurnMetadata* turn) const {
  return owner_->retention_room(retention(true), total, turn ? "turn:" + turn->conversation : std::string{},
      turn ? turn->keep : std::string_view{});
}

bool PrefixCache::can_admit_turn(const Identity& id, const TurnMetadata& turn,
    std::uint64_t tokens, std::uint64_t payload) const {
  if (!turns_enabled() || !policy_.store || !id.known() || !turn.known() || !tokens) return false;
  const auto total = charge(id, &turn, tokens, payload);
  return total && room(*total, &turn);
}

PrefixStateCache::Reservation PrefixCache::reserve_turn(const Identity& id, const TurnMetadata& turn,
    std::uint64_t tokens, std::uint64_t payload) {
  if (!can_admit_turn(id, turn, tokens, payload)) return {};
  const auto total = charge(id, &turn, tokens, payload);
  return total ? owner_->reserve(*total, nullptr, {}, true) : PrefixStateCache::Reservation{};
}

bool PrefixCache::admit(std::shared_ptr<const PrefixSnapshot> snapshot, PrefixStateCache::Reservation* credit) {
  if (!policy_.enabled || !policy_.store || !snapshot || !safe_checkpoint(*snapshot) || !policy_.max_entries)
    return false;
  const auto* turn = snapshot->turn ? &*snapshot->turn : nullptr;
  if (turn && (!turns_enabled() || !turn->known() || !snapshot->state.sealed() ||
      snapshot->state.purpose != SnapshotPurpose::prompt_reuse)) return false;
  // A turn-enabled cache charges EVERY entry against the same strict RAM bound.
  const bool strict = policy_.completed_turns;
  const auto charged = strict ? charge(snapshot->identity, turn, snapshot->state.tokens.size(), snapshot->state.bytes) :
      std::optional<std::uint64_t>(snapshot->state.bytes + (policy_.charge_token_bytes ?
          snapshot->state.tokens.size() * sizeof(std::uint32_t) : 0));
  if (!charged) return false;
  const auto total = *charged;
  if (strict ? !room(total, turn) : (total > policy_.max_bytes && !policy_.retain_one_oversize)) return false;

  auto entry = std::make_shared<PrefixCacheCheckpoint>();
  entry->retention = retention(strict);
  entry->charge = total;
  const auto metadata_charge = charge(snapshot->identity, turn, snapshot->state.tokens.size(), 0);
  if (!metadata_charge) return false;
  entry->declared_metadata = *metadata_charge;
  entry->legacy_index_exemption = !strict;
  entry->snapshot = std::make_shared<const PrefixSnapshot>(*snapshot);
  if (turn) { entry->replacement = "turn:" + turn->conversation; entry->keep = turn->keep; }
  else {
    // Separate namespace from turns/residency; exact tokens and identity are
    // still verified by the query facade before a restore can be attempted.
    std::vector<std::byte> key;
    for (const auto* text : {&snapshot->identity.model, &snapshot->identity.profile, &snapshot->identity.state_format}) {
      const auto n = static_cast<std::uint64_t>(text->size());
      const auto bytes = std::as_bytes(std::span(&n, 1)); key.insert(key.end(), bytes.begin(), bytes.end());
      const auto body = std::as_bytes(std::span(text->data(), text->size())); key.insert(key.end(), body.begin(), body.end());
    }
    const auto ranks = std::as_bytes(std::span(&snapshot->identity.ranks, 1)); key.insert(key.end(), ranks.begin(), ranks.end());
    const auto tokens = std::as_bytes(std::span(snapshot->state.tokens)); key.insert(key.end(), tokens.begin(), tokens.end());
    ::strix::core::shared_byte_budget().acquire(key.size()); // cache hashing draws from the byte budget
    entry->replacement = "prefix:" + model_content_identity_hex(sha256_bytes(key));
  }
  return owner_->retain(std::move(entry), turn || policy_.replace_duplicates, credit);
}

bool PrefixCache::admit_resident(const KvStore& store, KvHandle source,
    std::span<const std::uint32_t> tokens, const Identity& identity, std::uint64_t reply_rows) {
  if (!policy_.enabled || !policy_.store || !policy_.max_entries || !identity.known() ||
      identity.ranks != store.ranks() || reply_rows > tokens.size() || !store.resident_valid(source, tokens.size())) return false;
  const auto total = charge(identity, nullptr, tokens.size(), 0);
  if (!total || !room(*total, nullptr)) return false;
  const auto certificate = store.resident_prefix(source);
  if (!certificate || certificate->tokens().size() != tokens.size() ||
      prefix_length(certificate->tokens(), tokens) != tokens.size()) return false;
  auto value = std::make_shared<ResidentPrefix>(ResidentPrefix{identity, source, certificate, reply_rows});
  auto entry = std::make_shared<PrefixCacheCheckpoint>();
  entry->retention = retention(true); entry->charge = *total; entry->resident = std::move(value);
  entry->replacement = "resident:" + std::to_string(source.store) + ":" + std::to_string(source.slot);
  return owner_->retain(std::move(entry), true);
}
ResidentHit PrefixCache::lookup_resident(const KvStore& store, std::span<const std::uint32_t> prompt,
    const Identity& identity, std::uint64_t alignment) const {
  ResidentHit hit;
  if (!policy_.enabled || !identity.known() || !alignment || prompt.empty()) return hit;
  std::uint64_t best_rows = 0, ordinal = 0;
  std::uint32_t best_slot = UINT32_MAX;
  // One locked query returns the complete ownership lease. A monotonically
  // increasing selection ordinal preserves the slot tie-break without packing
  // or multiplying potentially large token counts into the rank.
  auto lease = owner_->query(scope_, [&](const PrefixCacheCheckpoint& c) -> std::uint64_t {
    if (!c.resident) return 0;
    const auto& r = *c.resident;
    if (r.identity != identity || !store.resident_valid(r.source, r.tokens().size())) return 0;
    auto n = std::min<std::uint64_t>(prefix_length(r.tokens(), prompt), prompt.size() - 1);
    n -= n % alignment;
    if (n > best_rows || (n && n == best_rows && r.source.slot < best_slot)) {
      best_rows = n; best_slot = r.source.slot;
      if (ordinal == UINT64_MAX) throw std::overflow_error("resident selection exhausted");
      return ++ordinal;
    }
    return 0;
  });
  if (lease) hit = {std::shared_ptr<const ResidentPrefix>(lease, lease->resident.get()), best_rows};
  return hit;
}
void PrefixCache::invalidate_resident(KvHandle h) noexcept {
  owner_->remove(scope_, [&](const auto& c) { return c.resident && c.resident->source.store == h.store && c.resident->source.slot == h.slot; });
}
void PrefixCache::invalidate_residents() noexcept {
  owner_->remove(scope_, [](const auto& c) { return bool(c.resident); });
}
bool PrefixCache::unpin(const std::string& keep) { return owner_->unpin(scope_, keep); }

std::shared_ptr<const PrefixSnapshot> PrefixCache::lookup(
    std::span<const std::uint32_t> prompt, const Identity& identity) const {
  return find(prompt, identity, nullptr);
}

std::shared_ptr<const PrefixSnapshot> PrefixCache::lookup_turn(
    std::span<const std::uint32_t> prompt, const Identity& identity, const TurnMetadata& turn) const {
  if (!turns_enabled() || !turn.known()) return {};
  return find(prompt, identity, &turn);
}

std::shared_ptr<const PrefixSnapshot> PrefixCache::find(
    std::span<const std::uint32_t> prompt, const Identity& identity, const TurnMetadata* turn) const {
  if (!policy_.enabled || !identity.known()) return {};
  auto best = owner_->query(scope_, [&](const PrefixCacheCheckpoint& c) -> std::uint64_t {
    if (!c.snapshot) return 0;
    const auto& snapshot = *c.snapshot;
    if (!safe_checkpoint(snapshot) || snapshot.identity != identity ||
        (turn && (!snapshot.state.sealed() || snapshot.state.purpose != SnapshotPurpose::prompt_reuse)) || (turn ? (!snapshot.turn || !same_turn(*snapshot.turn, *turn)) : bool(snapshot.turn))) return 0;
    const auto n = snapshot.state.tokens.size();
    if (n > prompt.size() || (policy_.leave_one_token && n == prompt.size()) ||
        prefix_length(snapshot.state.tokens, prompt) != n) return 0;
    return n;
  });
  if (!best) return {};
  return std::shared_ptr<const PrefixSnapshot>(best, best->snapshot.get());
}

NamedStateCache::NamedStateCache(std::uint64_t bytes, std::uint64_t entries, std::shared_ptr<PrefixStateCache> owner)
    : owner_(owner ? std::move(owner) : make_cache_owner(bytes)), policy_{owner_->new_scope(), bytes, entries, false, false} {}
NamedStateCache::~NamedStateCache() { owner_->remove(policy_.scope, [](const auto&) { return true; }); }
std::uint64_t NamedStateCache::bytes() const noexcept { return owner_->retained_charge(policy_.scope); }
std::uint64_t NamedStateCache::evictions() const noexcept { return owner_->scope_evictions(policy_.scope); }
const std::map<std::string, NamedStateCache::Entry>& NamedStateCache::entries() const noexcept {
  return owner_->named_view(policy_.scope);
}
std::shared_ptr<const PrefixCacheCheckpoint> NamedStateCache::lease(const std::string& id) const {
  return owner_->query(policy_.scope, [&](const auto& c) -> std::uint64_t { return c.name == id ? 1 : 0; }, false);
}
const NamedStateCache::Bytes* NamedStateCache::find(const std::string& id) const noexcept {
  const auto c = lease(id); return c ? c->encoded.get() : nullptr;
}
void NamedStateCache::touch(const std::string& id) {
  (void)owner_->query(policy_.scope, [&](const auto& c) -> std::uint64_t { return c.name == id ? 1 : 0; });
}
void NamedStateCache::erase(const std::string& id) { owner_->remove(policy_.scope, [&](const auto& c) { return c.name == id; }); }
bool NamedStateCache::put(const std::string& id, Bytes value, PrefixStateCache::Reservation* staging) {
  if (!policy_.max_entries || value.size() > policy_.max_bytes) return false;
  auto c = std::make_shared<PrefixCacheCheckpoint>();
  c->retention = policy_; c->name = id; c->replacement = id;
  return owner_->retain_encoded(std::move(c), std::move(value), staging);
}
bool NamedStateCache::evict_until(const std::function<bool(std::uint64_t)>& fits, const std::optional<std::string>& protected_id) {
  return owner_->reclaim(policy_.scope, fits, protected_id);
}

} // namespace strix::session
