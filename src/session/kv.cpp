#include "strix/session/kv.hpp"

#include <atomic>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strix::session {
struct StateSnapshot::Provenance::Seal {
  std::uint64_t origin = 0;
  std::shared_ptr<const void> domain, schema;
  std::uint64_t consumed = 0, bytes = 0;
  SnapshotPurpose purpose = SnapshotPurpose::resume;
  std::optional<std::uint32_t> pending;
  std::vector<std::uint32_t> tokens;
  std::vector<RankState> ranks;
};
namespace {
std::atomic<std::uint64_t> next_store{1};

bool same_owner(const std::shared_ptr<const void>& a, const std::shared_ptr<const void>& b) noexcept {
  return a && b && a.get() == b.get() && !a.owner_before(b) && !b.owner_before(a);
}

std::uint64_t checked_bytes(const std::vector<RankState>& ranks) {
  std::uint64_t sum = 0;
  for (const auto& rank : ranks) {
    if (!rank.owner) throw std::invalid_argument("snapshot rank owner is null");
    if (rank.bytes > UINT64_MAX - sum) throw std::overflow_error("snapshot byte count overflow");
    sum += rank.bytes;
  }
  return sum;
}
}

bool StateSnapshot::sealed() const noexcept {
  const auto& seal_ = provenance.seal_;
  if (!seal_ || consumed != seal_->consumed || pending != seal_->pending ||
      tokens != seal_->tokens || purpose != seal_->purpose || bytes != seal_->bytes || ranks.size() != seal_->ranks.size()) return false;
  for (std::size_t r = 0; r < ranks.size(); ++r)
    if (ranks[r].bytes != seal_->ranks[r].bytes || !same_owner(ranks[r].owner, seal_->ranks[r].owner)) return false;
  return true;
}

KvStore::KvStore(std::uint32_t slots, std::uint32_t ranks, Hooks hooks)
    : slots_(slots), ranks_(ranks), store_id_(next_store.fetch_add(1)), hooks_(std::move(hooks)) {
  if (slots == 0 || ranks == 0 || store_id_ == 0 || !hooks_.initialize ||
      !hooks_.reset || !hooks_.release || !hooks_.view || !hooks_.capture ||
      !hooks_.restore || !hooks_.truncate)
    throw std::invalid_argument("KvStore requires slots, ranks and complete lifecycle/state hooks");
}

KvStore::~KvStore() {
  for (std::uint32_t i = 0; i < slots_.size(); ++i) {
    const auto& slot = slots_[i];
    if (slot.active || slot.poisoned) {
      try { hooks_.release({i, slot.generation, store_id_}); } catch (...) {}
    }
  }
}

bool KvStore::supports_fork() const noexcept {
  return hooks_.copy_on_write_supported && static_cast<bool>(hooks_.fork_rank);
}

void KvStore::validate_snapshot(const StateSnapshot& snapshot) const {
  if (!snapshot.sealed()) throw std::invalid_argument("snapshot provenance or sealed metadata mismatch");
  const auto& seal = *snapshot.provenance.seal_;
  if (seal.origin != store_id_ &&
      !(same_owner(seal.domain, hooks_.snapshot_domain) && same_owner(seal.schema, hooks_.snapshot_schema)))
    throw std::invalid_argument("foreign snapshot domain/schema");
  if (snapshot.consumed != snapshot.tokens.size() || snapshot.ranks.size() != ranks_)
    throw std::invalid_argument("snapshot frontier/rank mismatch");
  if (checked_bytes(snapshot.ranks) != snapshot.bytes) throw std::invalid_argument("snapshot byte total mismatch");
}

KvHandle KvStore::fork(const StateSnapshot& snapshot) {
  if (!supports_fork()) throw std::invalid_argument("shared-prefix fork unsupported");
  validate_snapshot(snapshot);
  if (snapshot.purpose != SnapshotPurpose::resume) throw std::invalid_argument("prompt checkpoint cannot fork a lane");
  // Allocate all logical metadata before physical allocation/mutation.
  auto new_tokens = snapshot.tokens;
  const auto handle = acquire();
  try {
    for (std::uint32_t rank = 0; rank < ranks_; ++rank) hooks_.fork_rank(handle, rank, snapshot.ranks[rank]);
  } catch (...) {
    const auto failure = std::current_exception();
    try { release(handle); } catch (...) {
      // No caller has this handle. Make it unreachable and quarantine the slot.
      slots_[handle.slot].active = false;
      slots_[handle.slot].resident_rows = 0;
      slots_[handle.slot].poisoned = true;
    }
    std::rethrow_exception(failure);
  }
  auto& slot = slots_[handle.slot];
  slot.tokens.swap(new_tokens);
  slot.consumed = snapshot.consumed;
  slot.pending = snapshot.pending;
  return handle;
}

void KvStore::check(KvHandle handle, bool allow_poisoned) const {
  if (!handle.valid() || handle.slot >= slots_.size() || handle.store != store_id_)
    throw std::invalid_argument("invalid, stale or foreign KvHandle");
  const auto& slot = slots_[handle.slot];
  if (!slot.active || slot.generation != handle.generation)
    throw std::invalid_argument("invalid, stale or foreign KvHandle");
  if (slot.poisoned && !allow_poisoned) throw std::runtime_error("KvHandle state is poisoned");
}

KvStore::Slot& KvStore::checked(KvHandle handle, bool allow_poisoned) {
  check(handle, allow_poisoned);
  return slots_[handle.slot];
}

KvHandle KvStore::acquire() {
  for (std::uint32_t i = 0; i < slots_.size(); ++i)
    if (!slots_[i].active && !slots_[i].poisoned && slots_[i].generation != UINT64_MAX)
      return acquire_slot(i);
  throw std::runtime_error("KvStore exhausted");
}
KvHandle KvStore::acquire_slot(std::uint32_t i) {
  auto& slot = slots_.at(i);
  slot.resident.reset();
  slot.resident_rows = 0; // invalidate BEFORE initialization or any overwrite
  ++slot.generation;
  const KvHandle handle{i, slot.generation, store_id_};
  try { hooks_.initialize(handle); }
  catch (...) {
    try { hooks_.release(handle); slot.poisoned = false; }
    catch (...) { slot.poisoned = true; }
    slot.active = false;
    throw;
  }
  slot.active = true; slot.poisoned = false; slot.consumed = 0;
  slot.pending.reset(); slot.tokens.clear();
  return handle;
}
bool KvStore::resident_valid(KvHandle h, std::uint64_t rows) const noexcept {
  if (!hooks_.resident_prefill_supported || !h.valid() || h.store != store_id_ || h.slot >= slots_.size() || !rows) return false;
  const auto& s = slots_[h.slot];
  return !s.active && !s.poisoned && s.generation == h.generation &&
      s.generation != UINT64_MAX && rows <= s.resident_rows && !s.resident.expired();
}
KvHandle KvStore::acquire_resident(KvHandle source, std::span<const std::uint32_t> prefix) {
  if (!resident_valid(source, prefix.size())) throw std::invalid_argument("stale resident prefix");
  const auto certificate = resident_prefix(source);
  if (!certificate || !std::equal(prefix.begin(), prefix.end(), certificate->tokens().begin()))
    throw std::invalid_argument("resident token certificate mismatch");
  std::vector<std::uint32_t> tokens(prefix.begin(), prefix.end());
  const auto h = acquire_slot(source.slot);
  auto& s = slots_[h.slot]; s.tokens.swap(tokens); s.consumed = prefix.size();
  return h;
}
std::shared_ptr<const ResidentState> KvStore::release_resident(KvHandle h, std::uint64_t rows) {
  const auto& s = checked(h);
  if (!hooks_.resident_prefill_supported || rows > s.consumed) throw std::invalid_argument("uncommitted resident frontier");
  // Freeze only the canonical, committed prefix before cancel/release hooks.
  // Allocation failure leaves the active lane untouched and certifies nothing.
  auto certificate = std::shared_ptr<const ResidentState>(new ResidentState(h,
      {s.tokens.begin(), s.tokens.begin() + static_cast<std::ptrdiff_t>(rows)}));
  // release() cancels open work and quarantines failures before publication.
  release(h);
  slots_[h.slot].resident_rows = rows;
  slots_[h.slot].resident = certificate;
  return certificate;
}
std::shared_ptr<const ResidentState> KvStore::resident_prefix(KvHandle h) const noexcept {
  if (!resident_valid(h, 1)) return {};
  return slots_[h.slot].resident.lock();
}
void KvStore::invalidate_residents() noexcept {
  for (auto& s : slots_) { s.resident_rows = 0; s.resident.reset(); }
}

void KvStore::reset(KvHandle handle) {
  auto& slot = checked(handle, true);
  slot.resident_rows = 0; slot.resident.reset();
  try {
    hooks_.reset(handle);
  } catch (...) {
    slot.poisoned = true;
    throw;
  }
  slot.consumed = 0;
  slot.pending.reset();
  slot.tokens.clear();
  slot.poisoned = false;
}

void KvStore::release(KvHandle handle) {
  auto& slot = checked(handle, true);
  slot.resident_rows = 0; slot.resident.reset();
  try {
    hooks_.release(handle);
  } catch (...) {
    slot.poisoned = true;
    throw;
  }
  slot.active = false;
  slot.poisoned = false;
  slot.consumed = 0;
  slot.pending.reset();
  slot.tokens.clear();
}

void KvStore::invalidate(KvHandle handle) noexcept {
  try {
    check(handle, true);
    slots_[handle.slot].resident_rows = 0; slots_[handle.slot].resident.reset();
    slots_[handle.slot].poisoned = true;
  } catch (...) {
  }
}

void KvStore::append_consumed(KvHandle handle, std::span<const std::uint32_t> tokens) {
  auto& slot = checked(handle);
  if (tokens.size() > UINT64_MAX - slot.consumed) throw std::overflow_error("consumed frontier overflow");
  if (tokens.size() > slot.tokens.max_size() - slot.tokens.size()) throw std::length_error("token history overflow");
  if (!tokens.empty() && slot.pending && tokens.front() != *slot.pending)
    throw std::invalid_argument("append must consume pending anchor first");
  slot.tokens.insert(slot.tokens.end(), tokens.begin(), tokens.end());
  slot.consumed += tokens.size();
  if (!tokens.empty()) slot.pending.reset();
}

void KvStore::reserve_consumed(KvHandle handle, std::uint64_t additional) {
  auto& slot = checked(handle);
  if (additional > UINT64_MAX - slot.consumed || additional > slot.tokens.max_size() - slot.tokens.size())
    throw std::length_error("token history reservation overflow");
  slot.tokens.reserve(slot.tokens.size() + static_cast<std::size_t>(additional));
}

void KvStore::set_pending(KvHandle handle, std::optional<std::uint32_t> token) {
  checked(handle).pending = token;
}

std::uint64_t KvStore::consumed(KvHandle handle) const { check(handle); return slots_[handle.slot].consumed; }
std::optional<std::uint32_t> KvStore::pending(KvHandle handle) const { check(handle); return slots_[handle.slot].pending; }
std::vector<std::uint32_t> KvStore::tokens(KvHandle handle) const { check(handle); return slots_[handle.slot].tokens; }

RankState KvStore::view(KvHandle handle, std::uint32_t rank) const {
  check(handle);
  if (rank >= ranks_) throw std::out_of_range("rank out of range");
  return hooks_.view(handle, rank);
}

StateSnapshot KvStore::capture(KvHandle handle) const { return capture_impl(handle, SnapshotPurpose::resume); }
StateSnapshot KvStore::capture_prompt(KvHandle handle) const {
  if (!hooks_.capture_prompt || !hooks_.restore_prompt) throw std::invalid_argument("prompt checkpoint unsupported");
  return capture_impl(handle, SnapshotPurpose::prompt_reuse);
}
StateSnapshot KvStore::capture_impl(KvHandle handle, SnapshotPurpose purpose) const {
  check(handle);
  const auto& slot = slots_[handle.slot];
  StateSnapshot result;
  result.consumed = slot.consumed;
  result.pending = purpose == SnapshotPurpose::resume ? slot.pending : std::nullopt;
  result.purpose = purpose;
  result.tokens = slot.tokens;
  result.ranks.reserve(ranks_);
  for (std::uint32_t rank = 0; rank < ranks_; ++rank) {
    auto state = purpose == SnapshotPurpose::resume ? hooks_.capture(handle, rank) : hooks_.capture_prompt(handle, rank);
    if (!state.owner) throw std::runtime_error("capture returned null rank owner");
    if (state.bytes > UINT64_MAX - result.bytes) throw std::overflow_error("captured byte count overflow");
    result.bytes += state.bytes;
    result.ranks.push_back(std::move(state));
  }
  auto seal = std::make_shared<StateSnapshot::Seal>();
  seal->origin = store_id_;
  seal->domain = hooks_.snapshot_domain;
  seal->schema = hooks_.snapshot_schema;
  seal->consumed = result.consumed;
  seal->pending = result.pending;
  seal->purpose = purpose;
  seal->tokens = result.tokens;
  seal->ranks = result.ranks;
  seal->bytes = result.bytes;
  result.provenance.seal_ = std::move(seal);
  return result;
}

void KvStore::restore(KvHandle handle, const StateSnapshot& snapshot) {
  auto& slot = checked(handle);
  validate_snapshot(snapshot);
  if (snapshot.purpose != SnapshotPurpose::resume) throw std::invalid_argument("prompt checkpoint requires sampler restart");
  if (snapshot.consumed > slot.tokens.max_size()) throw std::length_error("snapshot token history too large");
  if (snapshot.consumed < slot.consumed && !hooks_.partial_restore_supported)
    throw std::invalid_argument("partial restore unsupported");

  auto new_tokens = snapshot.tokens;
  std::vector<RankState> old;
  old.reserve(ranks_);
  for (std::uint32_t rank = 0; rank < ranks_; ++rank) {
    auto state = hooks_.capture(handle, rank);
    if (!state.owner) throw std::runtime_error("rollback capture returned null rank owner");
    old.push_back(std::move(state));
  }

  std::uint32_t attempted = 0;
  try {
    for (; attempted < ranks_; ++attempted) hooks_.restore(handle, attempted, snapshot.ranks[attempted]);
  } catch (...) {
    const auto failure = std::current_exception();
    bool rollback_ok = true;
    const auto count = std::min<std::uint32_t>(attempted + 1, ranks_);
    for (std::uint32_t rank = 0; rank < count; ++rank) {
      try { hooks_.restore(handle, rank, old[rank]); } catch (...) { rollback_ok = false; }
    }
    if (!rollback_ok) slot.poisoned = true;
    std::rethrow_exception(failure);
  }
  slot.tokens.swap(new_tokens);
  slot.consumed = snapshot.consumed;
  slot.pending = snapshot.pending;
}

bool KvStore::restore_prompt(KvHandle handle, const StateSnapshot& snapshot) {
  auto& slot = checked(handle);
  validate_snapshot(snapshot);
  if (snapshot.purpose != SnapshotPurpose::prompt_reuse || snapshot.pending || !hooks_.restore_prompt ||
      slot.consumed || slot.pending) throw std::invalid_argument("prompt restore requires a fresh lane and prompt checkpoint");
  auto tokens = snapshot.tokens;
  bool accepted = false;
  try { accepted = hooks_.restore_prompt(handle, snapshot.ranks); }
  catch (...) { slot.poisoned = true; throw; }
  if (!accepted) return false;
  slot.tokens.swap(tokens); slot.consumed = snapshot.consumed;
  return true;
}

void KvStore::truncate(KvHandle handle, std::uint64_t prefix) {
  auto& slot = checked(handle);
  if (prefix > slot.consumed || prefix > slot.tokens.size()) throw std::out_of_range("truncate beyond committed frontier");
  if (prefix == slot.consumed) { slot.pending.reset(); return; }
  if (!hooks_.partial_restore_supported) throw std::invalid_argument("partial restore unsupported");
  std::vector<std::uint32_t> new_tokens(slot.tokens.begin(), slot.tokens.begin() + static_cast<std::ptrdiff_t>(prefix));
  std::vector<RankState> old;
  old.reserve(ranks_);
  for (std::uint32_t rank = 0; rank < ranks_; ++rank) {
    auto state = hooks_.capture(handle, rank);
    if (!state.owner) throw std::runtime_error("rollback capture returned null rank owner");
    old.push_back(std::move(state));
  }
  std::uint32_t attempted = 0;
  try {
    for (; attempted < ranks_; ++attempted) hooks_.truncate(handle, attempted, prefix);
  } catch (...) {
    const auto failure = std::current_exception();
    bool rollback_ok = true;
    const auto count = std::min<std::uint32_t>(attempted + 1, ranks_);
    for (std::uint32_t rank = 0; rank < count; ++rank) {
      try { hooks_.restore(handle, rank, old[rank]); } catch (...) { rollback_ok = false; }
    }
    if (!rollback_ok) slot.poisoned = true;
    std::rethrow_exception(failure);
  }
  slot.tokens.swap(new_tokens);
  slot.consumed = prefix;
  slot.pending.reset();
}

} // namespace strix::session
