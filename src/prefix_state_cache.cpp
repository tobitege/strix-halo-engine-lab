#include "strix/prefix_state_cache.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/unbuffered_file.hpp"
#include "strix/session/checkpoint_io.hpp"
#include "strix/session/prefix_cache.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>

namespace strix {
namespace {

constexpr std::uint64_t segment_magic = 0x31474553'4350'4d54ULL;    // "TMPCSEG1"
constexpr std::uint64_t checkpoint_magic = 0x31504b43'4350'4d54ULL; // "TMPCCKP1"

void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

std::string hex(const ModelContentIdentity& digest) { return model_content_identity_hex(digest); }

std::filesystem::path segment_path(const std::filesystem::path& dir, const ModelContentIdentity& digest) {
    return dir / ("seg-" + hex(digest) + ".tmps");
}
std::filesystem::path checkpoint_path(const std::filesystem::path& dir, const ModelContentIdentity& digest) {
    return dir / ("ckpt-" + hex(digest) + ".tmpc");
}

// Fixed-width little-endian header fields; the engine runs on little-endian hosts only.
static_assert(std::endian::native == std::endian::little, "prefix cache files are little-endian");

struct SegmentHeader {
    std::uint64_t magic = segment_magic;
    std::uint8_t identity[32]{}, digest[32]{}, parent[32]{};
    std::uint64_t block_index = 0;
    std::uint32_t block_rows = 0, reserved = 0;
    std::uint64_t payload_bytes = 0, payload_checksum = 0;
};
struct CheckpointHeader {
    std::uint64_t magic = checkpoint_magic;
    std::uint8_t identity[32]{}, digest[32]{};
    std::uint64_t position = 0;
    std::uint32_t block_rows = 0, reserved = 0;
    std::uint64_t segment_count = 0, fixed_bytes = 0, metadata_bytes = 0;
    std::uint64_t fixed_checksum = 0, metadata_checksum = 0;
};
static_assert(sizeof(SegmentHeader) == 8 + 96 + 8 + 8 + 16);
static_assert(sizeof(CheckpointHeader) == 8 + 64 + 8 + 8 + 40);

void put(std::uint8_t (&out)[32], const ModelContentIdentity& value) { std::memcpy(out, value.bytes.data(), 32); }
ModelContentIdentity get(const std::uint8_t (&in)[32]) {
    ModelContentIdentity value;
    std::memcpy(value.bytes.data(), in, 32);
    return value;
}

class File final {
public:
    File(const std::filesystem::path& path, const char* mode) : file_(std::fopen(path.string().c_str(), mode)) {}
    // Whole-file reads: unbuffered unless STRIX_SESSION_DIRECT_IO=0.
    enum class Stream { payload };
    File(const std::filesystem::path& path, const char* mode, Stream) {
        if (!state_file_io::unbuffered_io_enabled()) { file_ = std::fopen(path.string().c_str(), mode); return; }
        reader_ = std::make_unique<state_file_io::SequentialReader>(path);
    }
    ~File() { if (file_) std::fclose(file_); }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    [[nodiscard]] bool open() const noexcept {
        return file_ != nullptr || (reader_ && reader_->open());
    }
    void read(void* data, std::uint64_t bytes) {
        if (reader_) {
            try { reader_->read(data, bytes); } catch (const std::exception&) { throw std::runtime_error("prefix cache file is truncated"); }
            return;
        }
        auto* out = static_cast<std::byte*>(data);
        while (bytes) {
            const auto chunk = static_cast<std::size_t>(std::min<std::uint64_t>(bytes, 64ULL << 20));
            if (std::fread(out, 1, chunk, file_) != chunk) throw std::runtime_error("prefix cache file is truncated");
            out += chunk;
            bytes -= chunk;
        }
    }
private:
    std::FILE* file_ = nullptr;
    std::unique_ptr<state_file_io::SequentialReader> reader_;
};

// A tagged envelope inside the existing v1 metadata section. No parallel files/store.
std::vector<std::byte> disk_metadata(const PrefixCacheCheckpoint& checkpoint) {
    if (checkpoint.producer_certificate.empty()) return checkpoint.metadata;
    const auto count = static_cast<std::uint64_t>(checkpoint.producer_certificate.size());
    std::vector<std::byte> out(sizeof(count) + count*sizeof(std::uint64_t) + checkpoint.metadata.size());
    std::memcpy(out.data(), &count, sizeof(count));
    std::memcpy(out.data()+sizeof(count), checkpoint.producer_certificate.data(), count*sizeof(std::uint64_t));
    std::memcpy(out.data()+sizeof(count)+count*sizeof(std::uint64_t), checkpoint.metadata.data(), checkpoint.metadata.size());
    return out;
}

std::uint64_t checkpoint_host_bytes(const PrefixCacheCheckpoint& checkpoint) {
    return checkpoint.fixed.bytes + checkpoint.metadata.size() + checkpoint.producer_certificate.size()*sizeof(std::uint64_t);
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

std::vector<std::uint32_t> PrefixCacheCheckpoint::tokens() const {
    std::vector<std::uint32_t> out;
    out.reserve(static_cast<std::size_t>(position));
    if(chain_retention && segments.size()==1) {
        std::vector<const PrefixCacheSegment*> path;
        for(auto p=segments.front().get();p;p=p->ancestor.get()) path.push_back(p);
        for(auto i=path.rbegin();i!=path.rend();++i) out.insert(out.end(),(*i)->tokens.begin(),(*i)->tokens.end());
        return out;
    }
    for (const auto& segment : segments) out.insert(out.end(), segment->tokens.begin(), segment->tokens.end());
    return out;
}

ModelContentIdentity prefix_cache_block_digest(const ModelContentIdentity& previous,
    std::span<const std::uint32_t> block_tokens) {
    std::vector<std::byte> buffer(32 + block_tokens.size_bytes());
    std::memcpy(buffer.data(), previous.bytes.data(), 32);
    if (!block_tokens.empty()) std::memcpy(buffer.data() + 32, block_tokens.data(), block_tokens.size_bytes());
    return sha256_bytes(buffer);
}

std::vector<ModelContentIdentity> prefix_cache_block_digests(const ModelContentIdentity& identity,
    std::span<const std::uint32_t> tokens, std::uint32_t block_rows, std::uint64_t max_blocks) {
    require(block_rows != 0 && identity.valid(), "prefix cache digests need an identity and a block size");
    const auto blocks = std::min<std::uint64_t>(max_blocks, tokens.size() / block_rows);
    std::vector<ModelContentIdentity> digests;
    digests.reserve(static_cast<std::size_t>(blocks));
    auto previous = identity;
    for (std::uint64_t k = 0; k < blocks; ++k) {
        previous = prefix_cache_block_digest(previous, tokens.subspan(static_cast<std::size_t>(k * block_rows), block_rows));
        digests.push_back(previous);
    }
    return digests;
}

std::uint64_t prefix_cache_checksum(std::span<const std::byte> bytes) noexcept {
    constexpr std::uint64_t prime = 0x9e3779b97f4a7c15ULL;
    std::uint64_t lane[4] = {0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL, 0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL};
    const auto* data = bytes.data();
    std::size_t size = bytes.size(), offset = 0;
    for (; offset + 32 <= size; offset += 32) {
        for (unsigned i = 0; i < 4; ++i) {
            std::uint64_t word;
            std::memcpy(&word, data + offset + 8 * i, 8);
            lane[i] = std::rotl((lane[i] ^ word) * prime, 29);
        }
    }
    std::uint64_t tail = 0;
    for (unsigned shift = 0; offset < size; ++offset, shift += 8)
        tail ^= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(data[offset])) << (shift % 64);
    std::uint64_t hash = static_cast<std::uint64_t>(size) * prime;
    for (const auto value : lane) hash = std::rotl((hash ^ value) * prime, 31);
    return std::rotl((hash ^ tail) * prime, 27) ^ (hash >> 17);
}

PrefixStateCache::PrefixStateCache(PrefixStateCachePolicy policy, PrefixCacheAllocator allocate)
    : policy_(std::move(policy)), allocate_(std::move(allocate)) {
    require(policy_.host_byte_budget != 0 && static_cast<bool>(allocate_),
        "prefix state cache needs a positive host byte budget and an allocator");
    if (!policy_.disk_directory.empty()) {
        session::checkpoint_io::create_directories(policy_.disk_directory,policy_.durable && !policy_.legacy);
        // Leftovers of interrupted writers (old enough not to belong to a live process).
        sweep_disk(std::chrono::minutes(10));
        update_disk_gauge();
        writer_ = std::thread([this] { writer_loop(); });
    }
}

PrefixStateCache::~PrefixStateCache() {
    if (writer_.joinable()) {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        writer_.join();
    }
}

void PrefixStateCache::reap_immutable_locked(Ledger& ledger) {
    for (auto i = ledger.certificate_owners.begin(); i != ledger.certificate_owners.end();) {
        if (i->first.expired()) { ledger.certificates -= i->second; i = ledger.certificate_owners.erase(i); }
        else ++i;
    }
    for (auto i = ledger.immutable_owners.begin(); i != ledger.immutable_owners.end();) {
        if (i->first.expired()) { ledger.immutable -= i->second; i = ledger.immutable_owners.erase(i); }
        else ++i;
    }
}
PrefixStateCache::RetentionLease::~RetentionLease() {
    value.reset();
    std::lock_guard lock(ledger->mutex);
    ledger->retention_metadata -= metadata;
    ledger->exempt -= exempt;
    reap_immutable_locked(*ledger);
}
std::uint64_t PrefixStateCache::new_scope() {
    std::lock_guard lock(mutex_);
    if (next_scope_ == UINT64_MAX) throw std::overflow_error("cache scope exhausted");
    const auto scope = ++next_scope_;
    scope_evictions_.emplace(scope, 0);
    named_views_.try_emplace(scope);
    return scope;
}
const std::map<std::string, PrefixStateCache::NamedView>& PrefixStateCache::named_view(std::uint64_t scope) const {
    std::lock_guard lock(mutex_);
    return named_views_.at(scope);
}
std::vector<std::shared_ptr<const PrefixCacheCheckpoint>> PrefixStateCache::inspect(std::uint64_t scope) const {
    std::lock_guard lock(mutex_);
    std::vector<std::shared_ptr<const PrefixCacheCheckpoint>> result;
    for (const auto& d : lru_) if (entries_.at(d).checkpoint->retention.scope == scope)
        result.push_back(entries_.at(d).checkpoint);
    return result;
}
std::shared_ptr<const PrefixCacheCheckpoint> PrefixStateCache::query(std::uint64_t scope, const EntryRank& rank, bool touch) {
    std::lock_guard lock(mutex_);
    auto best = entries_.end();
    std::uint64_t score = 0;
    for (auto i = entries_.begin(); i != entries_.end(); ++i) {
        if (i->second.checkpoint->retention.scope != scope) continue;
        const auto n = rank(*i->second.checkpoint);
        if (n > score) { score = n; best = i; }
    }
    if (best == entries_.end()) return {};
    if (touch) touch_locked(best->second);
    return best->second.checkpoint;
}
std::uint64_t PrefixStateCache::retained_charge(std::uint64_t scope) const noexcept {
    std::lock_guard lock(mutex_);
    if(const auto i=chain_stats_.find(scope);i!=chain_stats_.end()) return i->second.bytes;
    std::uint64_t total = 0;
    for (const auto& [d,e] : entries_) if (e.checkpoint->retention.scope == scope) {
        (void)d;
        if (e.checkpoint->charge > UINT64_MAX - total) return UINT64_MAX;
        total += e.checkpoint->charge;
    }
    return total;
}
std::uint64_t PrefixStateCache::scope_evictions(std::uint64_t scope) const noexcept {
    std::lock_guard lock(mutex_);
    const auto i = scope_evictions_.find(scope);
    return i == scope_evictions_.end() ? 0 : i->second;
}
std::size_t PrefixStateCache::entry_count(std::uint64_t scope) const noexcept {
    std::lock_guard lock(mutex_);
    return static_cast<std::size_t>(std::count_if(entries_.begin(), entries_.end(),
        [scope](const auto& item) { return item.second.checkpoint->retention.scope == scope; }));
}
bool PrefixStateCache::retention_room_locked(const PrefixEntryPolicy& p, std::uint64_t n,
    std::string_view replacement, std::string_view keep) const {
    if (!p.scope || !p.max_entries || (n > p.max_bytes && !p.sole_oversize)) return false;
    std::uint64_t protected_bytes = 0, protected_count = 0;
    for (const auto& [d,e] : entries_) {
        (void)d;
        const auto& c = *e.checkpoint;
        if (c.retention.scope != p.scope || !e.protected_entry) continue;
        if ((!replacement.empty() && c.replacement == replacement) || (!keep.empty() && c.keep == keep)) return false;
        if (c.charge > UINT64_MAX - protected_bytes) return false;
        protected_bytes += c.charge; ++protected_count;
    }
    return protected_count < p.max_entries && (n > p.max_bytes ? protected_count == 0 :
        protected_bytes <= p.max_bytes - n);
}
bool PrefixStateCache::retention_room(const PrefixEntryPolicy& p, std::uint64_t n,
    std::string_view replacement, std::string_view keep) const {
    std::lock_guard lock(mutex_);
    return retention_room_locked(p, n, replacement, keep);
}
bool PrefixStateCache::reclaim_one_locked(std::optional<std::uint64_t> scope, std::string_view protected_name) {
    for (const auto& d : lru_) {
        auto i = entries_.find(d);
        const auto& c = *i->second.checkpoint;
        if (c.chain_retention && !scope) continue; // this adopter has scoped victim authority
        if ((scope && c.retention.scope != *scope) || i->second.protected_entry ||
            (!protected_name.empty() && c.name == protected_name)) continue;
        // Physical slab owners cannot be reclaimed while their backing is leased.
        // Generic lookups may be evicted: their lease stays charged in the ledger.
        if (!c.retention.scope && payload_pool_ && i->second.checkpoint.use_count() != 1) continue;
        const auto owner = c.retention.scope;
        erase_locked(d, true);
        ++stats_.evicted; ++stats_.evictions_space;
        if (owner) ++scope_evictions_.at(owner);
        return true;
    }
    return false;
}
void PrefixStateCache::remove(std::uint64_t scope, const EntryPredicate& predicate) noexcept {
    std::lock_guard lock(mutex_);
    for (auto i = entries_.begin(); i != entries_.end();) {
        const auto current = i++;
        if (current->second.checkpoint->retention.scope == scope && predicate(*current->second.checkpoint))
            erase_locked(current->first);
    }
}
bool PrefixStateCache::unpin(std::uint64_t scope, const std::string& keep) {
    std::lock_guard lock(mutex_);
    for (auto& [d,e] : entries_) {
        (void)d;
        if (e.checkpoint->retention.scope == scope && e.protected_entry && e.checkpoint->keep == keep && !keep.empty()) {
            e.protected_entry = false;
            return true;
        }
    }
    return false;
}
bool PrefixStateCache::reclaim(std::uint64_t scope, const std::function<bool(std::uint64_t)>& fits,
    const std::optional<std::string>& protected_name) {
    std::lock_guard lock(mutex_);
    for (;;) {
        std::uint64_t n = 0;
        for (const auto& [d,e] : entries_) if (e.checkpoint->retention.scope == scope) {
            (void)d;
            if (e.checkpoint->charge > UINT64_MAX - n) return false;
            n += e.checkpoint->charge;
        }
        if (fits(n)) return true;
        if (!reclaim_one_locked(scope, protected_name.value_or(""))) return false;
    }
}
bool PrefixStateCache::retain(std::shared_ptr<PrefixCacheCheckpoint> value, bool replace, Reservation* credit) {
    if (!value || value->chain_retention) return false;
    value = std::make_shared<PrefixCacheCheckpoint>(*value);
    std::lock_guard lock(mutex_);
    const auto& p = value->retention;
    if (!scope_evictions_.contains(p.scope) || !named_views_.contains(p.scope) ||
        !value->segments.empty() || !value->metadata.empty() ||
        !value->producer_certificate.empty() || !retention_room_locked(p, value->charge, value->replacement, value->keep)) return false;
    auto duplicate = entries_.end();
    for (auto i = entries_.begin(); i != entries_.end(); ++i)
        if (i->second.checkpoint->retention.scope == p.scope &&
            i->second.checkpoint->replacement == value->replacement) { duplicate = i; break; }
    if (duplicate != entries_.end() && !replace) return false;
    // Freeze/allocate incoming control storage before deleting any old value.
    auto lease = std::make_shared<RetentionLease>(); lease->ledger = ledger_;
    lease->value = value;
    const auto payload = value->snapshot ? value->snapshot->state.bytes : value->encoded ? value->fixed.bytes : 0;
    if (payload > value->charge) return false;
    const auto certificate_bytes = value->resident ? sizeof(session::ResidentState) +
        value->resident->tokens().size() * sizeof(std::uint32_t) : 0;
    if (certificate_bytes > value->charge - payload) return false;
    const auto budget_metadata = value->charge - payload - certificate_bytes;
    const auto metadata = value->declared_metadata ? value->declared_metadata : budget_metadata;
    if (metadata < budget_metadata || (metadata != budget_metadata &&
        (!value->legacy_index_exemption || (!value->encoded && (!value->snapshot || value->snapshot->turn))))) return false;
    const auto exempt = metadata - budget_metadata;
    std::uint64_t needed = budget_metadata;
    decltype(ledger_->immutable_owners) staged, certificates;
    {
        std::lock_guard ledger_lock(ledger_->mutex);
        reap_immutable_locked(*ledger_);
        if (metadata > UINT64_MAX - ledger_->retention_metadata || exempt > UINT64_MAX - ledger_->exempt) return false;
        if (value->snapshot) for (const auto& r : value->snapshot->state.ranks) {
            auto old = ledger_->immutable_owners.find(r.owner);
            if (old != ledger_->immutable_owners.end()) {
                if (old->second != r.bytes) return false;
            } else {
                auto [i, added] = staged.emplace(r.owner, r.bytes);
                if (!added && i->second != r.bytes) return false;
                if (added) { if (r.bytes > UINT64_MAX - needed) return false; needed += r.bytes; }
            }
        }
        if (value->resident && !ledger_->certificate_owners.contains(value->resident->state)) {
            if (certificate_bytes > UINT64_MAX - needed) return false;
            needed += certificate_bytes;
            certificates.emplace(value->resident->state, certificate_bytes);
        }
        if (value->encoded) {
            const auto found = receipts_.find(value->fixed.data.get());
            const auto receipt = found == receipts_.end() ? nullptr : found->second.lock();
            if (!receipt || !receipt->registered) {
                if (value->encoded->capacity() > UINT64_MAX - needed) return false;
                needed += value->encoded->capacity();
            }
        }
    }
    // Preallocate the real directory and LRU nodes before changing victims.
    // Node transfer at publication cannot allocate or fail.
    auto serial = epoch_;
    do {
        if (serial == UINT64_MAX) throw std::overflow_error("cache entry epoch exhausted");
        ++serial;
        value->digest = {};
        std::memcpy(value->digest.bytes.data(), &p.scope, sizeof(p.scope));
        std::memcpy(value->digest.bytes.data() + 8, &serial, sizeof(serial));
        value->digest.bytes.back() = 0xff;
    } while (entries_.contains(value->digest));
    std::shared_ptr<const PrefixCacheCheckpoint> retained(lease, value.get());
    std::list<ModelContentIdentity> pending_lru;
    decltype(entries_) pending_directory;
    std::map<std::string, NamedView> pending_view;
    pending_lru.push_back(value->digest);
    pending_directory.emplace(value->digest, Entry{retained, pending_lru.begin(), serial, 0, {}, !value->keep.empty()});
    if (value->encoded) pending_view.emplace(value->name, NamedView{*value->encoded, serial});
    const auto credited = credit && credit->ledger_ == ledger_ ? std::min(needed, credit->bytes_) : 0;
    const auto incremental = needed - credited;
    // Never wrap service arithmetic, even for the legacy unlimited sentinel.
    if (needed > policy_.host_byte_budget && !p.sole_oversize) return false;
    if (duplicate != entries_.end()) erase_locked(duplicate->first);
    for (;;) {
        std::uint64_t n = 0, count = 0;
        for (const auto& [d,e] : entries_) if (e.checkpoint->retention.scope == p.scope) {
            (void)d;
            if (e.checkpoint->charge > UINT64_MAX - n) return false;
            n += e.checkpoint->charge; ++count;
        }
        if (count < p.max_entries && (value->charge > p.max_bytes ? count == 0 : n <= p.max_bytes - value->charge)) break;
        if (!reclaim_one_locked(p.scope)) return false;
    }
    // The legacy limit governs retained entries. A sole oversize prompt can
    // replace an evicted capture whose backing is still borrowed by the active
    // producer. Those leases remain fully charged/reported; producer transient
    // admission bounds their peak. Strict turns never receive this exemption.
    const auto budget = p.sole_oversize ? std::max(policy_.host_byte_budget, needed) : policy_.host_byte_budget;
    if (budget == policy_.host_byte_budget) evict_to_budget_locked(incremental);
    else while (retained_bytes_locked() > budget - incremental)
        if (!reclaim_one_locked()) break;
    if (retained_bytes_locked() > budget - incremental && !p.sole_oversize) return false;
    if (payload_pool_) {
        // The attached Qwen slab is already a physical commitment, including
        // its free ranges. Typed owners and adopted codec extents cannot spend
        // that same memory again: they must fit the remaining metadata reserve.
        // Existing slab-only capture admission and pool fragmentation are intact.
        std::lock_guard ledger_lock(ledger_->mutex);
        reap_immutable_locked(*ledger_);
        std::uint64_t outside = metadata_bytes_;
        const auto pending = ledger_->reserved - credited;
        for (const auto n : {ledger_->immutable, ledger_->certificates, ledger_->retention_metadata,
                 ledger_->encoded, pending, needed, exempt}) {
            if (n > metadata_allowance_ || outside > metadata_allowance_ - n) return false;
            outside += n;
        }
    }
    {
        std::lock_guard ledger_lock(ledger_->mutex);
        // Sole-oversize is a retention compatibility exception, never an
        // arithmetic exception. Evicted captures can remain externally leased;
        // include those owners and the complete metadata before publishing.
        auto immutable = ledger_->immutable;
        auto certificate = ledger_->certificates;
        for (const auto& [owner,n] : staged) {
            (void)owner;
            if (n > UINT64_MAX - immutable) return false;
            immutable += n;
        }
        for (const auto& [owner,n] : certificates) {
            (void)owner;
            if (n > UINT64_MAX - certificate) return false;
            certificate += n;
        }
        if (metadata > UINT64_MAX - ledger_->retention_metadata) return false;
        std::uint64_t complete = 0;
        for (const auto n : {ledger_->bytes, ledger_->reserved - credited,
                 metadata_bytes_, immutable, certificate, ledger_->retention_metadata + metadata}) {
            if (n > UINT64_MAX - complete) return false;
            complete += n;
        }
        if (credited) { ledger_->reserved -= credited; credit->bytes_ -= credited; }
        ledger_->immutable_owners.merge(staged);
        ledger_->certificates = certificate;
        ledger_->certificate_owners.merge(certificates);
        ledger_->immutable = immutable;
        ledger_->retention_metadata += metadata; ledger_->exempt += exempt;
        lease->metadata = metadata; lease->exempt = exempt;
    }
    entries_.merge(pending_directory);
    named_views_.at(p.scope).merge(pending_view);
    lru_.splice(lru_.end(), pending_lru);
    epoch_ = serial;
    prune_receipts_locked();
    ++stats_.inserted;
    return true;
}
PrefixStateCache::ChainStats PrefixStateCache::chain_stats(std::uint64_t scope) const {
    std::lock_guard lock(mutex_);
    auto i=chain_stats_.find(scope);
    return i==chain_stats_.end() ? ChainStats{} : i->second;
}
std::shared_ptr<const PrefixCacheSegment> PrefixStateCache::chain_segment(std::uint64_t scope, const ModelContentIdentity& d) const {
    std::lock_guard lock(mutex_);
    const auto i=chain_refs_.find(d);
    if(i==chain_refs_.end() || i->second.scope!=scope) return {};
    const auto s=segments_.find(d);
    return s==segments_.end() ? nullptr : s->second.lock();
}
std::shared_ptr<const PrefixCacheCheckpoint> PrefixStateCache::chain_checkpoint(std::uint64_t scope, const ModelContentIdentity& d, bool touch) {
    std::lock_guard lock(mutex_);
    const auto i=entries_.find(d);
    if(i==entries_.end() || !i->second.checkpoint->chain_retention || i->second.checkpoint->retention.scope!=scope) return {};
    if(touch) touch_locked(i->second);
    return i->second.checkpoint;
}
void PrefixStateCache::prune_chain_locked(const ModelContentIdentity& d, bool eviction) {
    const auto i=chain_refs_.find(d);
    if(i==chain_refs_.end() || i->second.refs || i->second.pins) return;
    auto& s=chain_stats_.at(i->second.scope);
    s.bytes-=i->second.charge; --s.blocks;
    if(eviction) ++s.evicted_blocks;
    // Remove the query index before releasing any payload owner.
    segments_.erase(d); chain_refs_.erase(i);
}
void PrefixStateCache::release_chain_locked(const PrefixCacheCheckpoint& c, bool eviction) {
    auto& s=chain_stats_.at(c.retention.scope);
    s.bytes-=c.charge; --s.snapshots;
    if(eviction) ++s.evicted_snapshots;
    for(auto p=c.segments.front();p;p=p->ancestor) {
        --chain_refs_.at(p->digest).refs;
        prune_chain_locked(p->digest,eviction);
    }
}
bool PrefixStateCache::retain_chain(std::shared_ptr<PrefixCacheCheckpoint> c,
    const std::function<void(PrefixCacheCheckpoint&, const PrefixCacheAllocator&)>& fill) {
    if(!c || c->segments.size()!=1 || !c->retention.scope || !c->charge ||
       c->snapshot || c->resident || c->encoded || !c->metadata.empty() || !c->producer_certificate.empty()) return false;
    c=std::make_shared<PrefixCacheCheckpoint>(*c);
    std::lock_guard lock(mutex_);
    const auto scope=c->retention.scope;
    if(policy_.legacy || payload_pool_ || !scope_evictions_.contains(scope)) return false;
    if(auto i=entries_.find(c->digest);i!=entries_.end()) {
        if(i->second.checkpoint->retention.scope!=scope || !i->second.checkpoint->chain_retention) return false;
        auto a=c->segments.front(), b=i->second.checkpoint->segments.front();
        while(a && b) {
            if(a->digest!=b->digest || a->parent!=b->parent || a->block_rows!=b->block_rows ||
               a->block_index!=b->block_index || a->tokens!=b->tokens) return false;
            a=a->ancestor; b=b->ancestor;
        }
        if(a || b || c->position!=i->second.checkpoint->position) return false;
        touch_locked(i->second); return true;
    }
    std::vector<std::shared_ptr<const PrefixCacheSegment>> path, fresh;
    std::set<ModelContentIdentity,DigestLess> visited;
    auto need=c->charge;
    for(auto p=c->segments.front();p;p=p->ancestor) {
        if(!visited.insert(p->digest).second || !p->charge || p->payload.bytes>p->charge) return false;
        if(p->ancestor && (p->block_index!=p->ancestor->block_index+1 || p->parent!=p->ancestor->digest)) return false;
        if(!p->ancestor && p->block_index!=0) return false;
        path.push_back(p);
        const auto i=chain_refs_.find(p->digest);
        if(i==chain_refs_.end()) {
            // A leased, erased block cannot be republished as a new allocation.
            if(p->payload.data || p->charge>UINT64_MAX-need) return false;
            need+=p->charge; fresh.push_back(p);
        } else if(i->second.scope!=scope || segments_.at(p->digest).lock()!=p) return false;
    }
    const auto limit=c->retention.max_bytes;
    if(c->fixed.bytes>c->charge || need>limit || need>policy_.host_byte_budget) return false;
    auto& gauges=chain_stats_[scope];
    // All control nodes are prepared before selecting a victim.
    decltype(chain_refs_) pending_refs;
    decltype(segments_) pending_segments;
    decltype(entries_) pending_entry;
    std::list<ModelContentIdentity> pending_lru{c->digest};
    c->chain_retention=true;
    pending_entry.emplace(c->digest,Entry{c,pending_lru.begin(),0,0,{}});
    for(const auto& p:fresh) {
        pending_refs.emplace(p->digest,ChainRef{scope,0,0,p->charge});
        pending_segments.emplace(p->digest,p);
    }
    for(const auto& p:path) if(auto i=chain_refs_.find(p->digest);i!=chain_refs_.end()) ++i->second.pins;
    const auto unpin=[&] {
        for(const auto& p:path) if(auto i=chain_refs_.find(p->digest);i!=chain_refs_.end() && i->second.pins) {
            --i->second.pins; prune_chain_locked(p->digest,true);
        }
    };
    try {
        while(gauges.bytes>limit-need || retained_bytes_locked()>policy_.host_byte_budget-need)
            if(!reclaim_one_locked(scope)) { unpin(); return false; }
        Reservation credit(ledger_,need);
        { std::lock_guard credits(ledger_->mutex); ledger_->reserved+=need; }
        std::reverse(fresh.begin(),fresh.end());
        std::size_t at=0;
        // Reserve the complete legacy charge, including control storage. Payload
        // receipts own that charge until the last immutable lease is released.
        const PrefixCacheAllocator allocator=[&](std::uint64_t bytes) {
            if(at>fresh.size()) throw std::invalid_argument("chain allocation count");
            const auto expected=at<fresh.size()?fresh[at]->payload.bytes:c->fixed.bytes;
            const auto charge=at<fresh.size()?fresh[at]->charge:c->charge;
            if(bytes!=expected) throw std::invalid_argument("chain allocation size");
            auto raw=bytes ? allocate_(bytes) : PrefixCacheBuffer{};
            if(bytes && (!raw.data || raw.bytes<bytes)) throw PrefixCacheAllocationFailure();
            auto receipt=std::make_shared<Receipt>();
            receipt->ledger=ledger_; receipt->payload=std::move(raw.data); receipt->bytes=charge;
            receipt->segment=at<fresh.size();
            // Metadata-only segments still need a lifetime/credit owner; no
            // zero-byte allocation is required of the service allocator.
            auto* address=bytes ? receipt->payload.get() : reinterpret_cast<std::byte*>(receipt.get());
            PrefixCacheBuffer result{std::shared_ptr<std::byte>(receipt,address),bytes};
            {
                std::lock_guard credits(ledger_->mutex);
                ledger_->reserved-=charge; credit.bytes_-=charge; ledger_->bytes+=charge;
                if(receipt->segment) ++ledger_->segments;
                receipt->registered=true;
            }
            ++at; return result;
        };
        fill(*c,allocator);
        if(at!=fresh.size()+1 || !c->fixed.data) throw std::invalid_argument("chain incomplete capture");
        for(const auto& p:fresh) if(!p->payload.data) throw std::invalid_argument("chain missing block capture");
        // Publication is a nonallocating node transfer. The pending entry owns
        // the same checkpoint filled by the codec through its captured handle.
        chain_refs_.merge(pending_refs); segments_.merge(pending_segments);
        for(const auto& p:fresh) { gauges.bytes+=p->charge; ++gauges.blocks; }
        for(const auto& p:path) ++chain_refs_.at(p->digest).refs;
        gauges.bytes+=c->charge; ++gauges.snapshots;
        pending_entry.begin()->second.epoch=++epoch_;
        entries_.merge(pending_entry); lru_.splice(lru_.end(),pending_lru);
        unpin(); ++stats_.inserted;
        return true;
    } catch(...) { unpin(); throw; }
}
bool PrefixStateCache::retain_encoded(std::shared_ptr<PrefixCacheCheckpoint> c, std::vector<std::uint8_t> bytes, Reservation* staging) {
    if (!c || bytes.size() > c->retention.max_bytes || !c->retention.max_entries) return false;
    // Hold the exact physical commitment before adopting the staging buffer.
    // The resolver/family still admits codec/capture peaks before producing it.
    auto own_credit = staging ? Reservation{} : reserve(bytes.capacity(), nullptr, {}, true);
    auto& credit = staging ? *staging : own_credit;
    if (credit.ledger_ != ledger_ || credit.bytes_ < bytes.capacity()) return false;
    auto encoded = std::make_shared<const std::vector<std::uint8_t>>(std::move(bytes));
    // The immutable vector is itself the codec's backing allocation. Pool leases
    // expose its bytes without duplicating storage; no raw device pointer enters.
    PrefixCacheBuffer backing{std::shared_ptr<std::byte>(encoded,
        reinterpret_cast<std::byte*>(const_cast<std::uint8_t*>(encoded->data()))), encoded->capacity()};
    auto pool = std::make_shared<PrefixPayloadPool>(std::move(backing), 1);
    c->fixed = pool->allocate(encoded->size());
    c->charge = c->fixed.bytes;
    const auto control = sizeof(PrefixCacheCheckpoint) + sizeof(PrefixPayloadPool) + sizeof(Receipt) + 256;
    if (c->name.size() > (UINT64_MAX - control) / 3) return false;
    c->declared_metadata = control + c->name.size() * 3;
    c->legacy_index_exemption = true;
    // Use the existing physical Receipt and ledger. Publication registers it;
    // its pool extent remains charged until the final checkpoint/read lease dies.
    auto receipt = std::make_shared<Receipt>();
    receipt->ledger = ledger_; receipt->payload = c->fixed.data; receipt->bytes = encoded->capacity();
    receipt->encoded_owner = encoded;
    c->fixed.data = std::shared_ptr<std::byte>(receipt, c->fixed.data.get());
    c->encoded = std::shared_ptr<const std::vector<std::uint8_t>>(receipt, encoded.get());
    {
        std::lock_guard lock(mutex_);
        prune_receipts_locked();
        receipts_[c->fixed.data.get()] = receipt;
        std::lock_guard ledger_lock(ledger_->mutex);
        ledger_->reserved -= receipt->bytes;
        credit.bytes_ -= receipt->bytes;
        ledger_->bytes += receipt->bytes;
        ledger_->encoded += receipt->bytes;
        receipt->registered = true; receipt->encoded = true;
    }
    return retain(std::move(c), true);
}

PrefixStateCache::Reservation::~Reservation() {
    if (ledger_) { std::lock_guard lock(ledger_->mutex); ledger_->reserved -= bytes_; }
}

PrefixStateCache::Reservation PrefixStateCache::reserve(std::uint64_t bytes, const PrefixCacheCheckpoint* incoming, std::string_view protected_name, bool session_retention) {
    if (policy_.legacy && !session_retention) return {};
    std::lock_guard lock(mutex_);
    std::vector<std::uint64_t> payloads;
    std::uint64_t metadata=0;
    if(payload_pool_ && incoming) {
        // Capture allocates fixed first, then each missing segment. Already leased
        // segments are charged physically by the pool and are never reclaimed here.
        if(!incoming->fixed.data && incoming->fixed.bytes) payloads.push_back(incoming->fixed.bytes);
        for(const auto& segment:incoming->segments)
            if(!segment->payload.data && segment->payload.bytes) payloads.push_back(segment->payload.bytes);
        metadata=incoming->metadata.size()+incoming->producer_certificate.size()*sizeof(std::uint64_t);
    }
    const auto physical_fits=[&] {
        return !payload_pool_ || (metadata<=metadata_allowance_ && metadata_bytes_<=metadata_allowance_-metadata &&
            payload_pool_->can_allocate(payloads));
    };
    if (bytes <= policy_.host_byte_budget) {
        evict_to_budget_locked(bytes, policy_.retention == PrefixCacheRetention::fair ? incoming : nullptr,payloads,metadata,protected_name);
        if ((!incoming || policy_.retention != PrefixCacheRetention::fair || entries_.size() < 256) &&
            retained_bytes_locked() <= policy_.host_byte_budget - bytes && physical_fits()) {
            std::lock_guard ledger_lock(ledger_->mutex);
            ledger_->reserved += bytes;
            return Reservation(ledger_, bytes);
        }
    }
    ++stats_.budget_skips;
    return {};
}

PrefixCacheBuffer PrefixStateCache::allocate(std::uint64_t bytes, Reservation& reservation, bool segment) {
    if (policy_.legacy) return allocate(bytes);
    if (reservation.ledger_ != ledger_ || bytes > reservation.bytes_)
        throw PrefixCacheAllocationFailure();
    auto buffer = allocate(bytes);
    std::lock_guard lock(mutex_);
    auto receipt = receipts_.at(buffer.data.get()).lock();
    std::lock_guard ledger_lock(ledger_->mutex);
    ledger_->reserved -= bytes;
    reservation.bytes_ -= bytes;
    ledger_->bytes += bytes;
    receipt->registered = true; receipt->segment = segment;
    if (segment) ++ledger_->segments;
    return buffer;
}

PrefixCacheBuffer PrefixStateCache::allocate(std::uint64_t bytes) const {
    if (policy_.legacy) {
        auto buffer = allocate_(bytes);
        if (bytes && (!buffer.data || buffer.bytes < bytes)) throw std::runtime_error("prefix cache allocation failed");
        buffer.bytes = bytes; return buffer;
    }
    PrefixCacheBuffer buffer;
    try { buffer = allocate_(bytes); }
    catch (const PrefixCacheFatalError&) { throw; }
    catch (const std::bad_alloc&) { throw PrefixCacheAllocationFailure(); }
    if (bytes && (!buffer.data || buffer.bytes < bytes)) throw PrefixCacheAllocationFailure();
    buffer.bytes = bytes;
    auto receipt = std::make_shared<Receipt>();
    receipt->ledger = ledger_; receipt->payload = buffer.data; receipt->bytes = bytes;
    buffer.data = std::shared_ptr<std::byte>(receipt, buffer.data.get());
    std::lock_guard lock(mutex_);
    prune_receipts_locked();
    receipts_[buffer.data.get()] = receipt;
    return buffer;
}

std::uint64_t PrefixStateCache::retained_bytes_locked() {
    const auto host_metadata = policy_.legacy ? host_bytes_locked() : metadata_bytes_;
    std::lock_guard lock(ledger_->mutex);
    reap_immutable_locked(*ledger_);
    std::uint64_t total = 0;
    // Exempt legacy index metadata is reported, but is outside legacy budget
    // arithmetic. Subtract it before adding, so saturation cannot create credit.
    for (const auto n : {ledger_->bytes, ledger_->reserved, host_metadata, ledger_->immutable,
             ledger_->retention_metadata - ledger_->exempt, ledger_->certificates}) {
        if (n > UINT64_MAX - total) return UINT64_MAX;
        total += n;
    }
    return total;
}

std::uint64_t PrefixStateCache::host_bytes_locked() {
    std::uint64_t total = 0;
    for (auto it = segments_.begin(); it != segments_.end();) {
        if (const auto segment = it->second.lock()) {
            total += segment->payload.bytes;
            ++it;
        } else {
            it = segments_.erase(it);
        }
    }
    for (auto it = session_buffers_.begin(); it != session_buffers_.end();) {
        if (it->second.first.expired()) it = session_buffers_.erase(it);
        else { total += it->second.second; ++it; }
    }
    for (const auto& [digest, entry] : entries_) {
        if (entry.checkpoint->retention.scope) continue;
        total += checkpoint_host_bytes(*entry.checkpoint);
        if (session_buffers_.count(entry.checkpoint->fixed.data.get())) total -= entry.checkpoint->fixed.bytes;
    }
    return total;
}

std::uint64_t PrefixStateCache::incremental_bytes(const PrefixCacheCheckpoint& checkpoint) {
    std::lock_guard lock(mutex_);
    return incremental_bytes_locked(checkpoint);
}

std::uint64_t PrefixStateCache::incremental_bytes_locked(const PrefixCacheCheckpoint& checkpoint) {
    std::uint64_t total = checkpoint_host_bytes(checkpoint);
    const auto registered = [&](const PrefixCacheBuffer& b) {
        const auto it = receipts_.find(b.data.get());
        const auto r = it == receipts_.end() ? nullptr : it->second.lock();
        return b.data && r && r->registered;
    };
    if (!policy_.legacy && registered(checkpoint.fixed)) total -= checkpoint.fixed.bytes;
    for (const auto& segment : checkpoint.segments) {
        const auto found = segments_.find(segment->digest);
        if (policy_.legacy ? (found == segments_.end() || found->second.expired()) : !registered(segment->payload))
            total += segment->payload.bytes;
    }
    return total;
}

void PrefixStateCache::prune_receipts_locked() const noexcept {
    for (auto i = receipts_.begin(); i != receipts_.end();)
        if (i->second.expired()) i = receipts_.erase(i); else ++i;
}

void PrefixStateCache::register_buffer_locked(const PrefixCacheBuffer& buffer, bool segment) {
    const auto it = receipts_.find(buffer.data.get());
    if (it == receipts_.end()) return; // external named envelopes use session_buffers_ accounting
    if (auto r = it->second.lock(); r && !r->registered) {
        r->registered = true; r->segment = segment;
        std::lock_guard lock(ledger_->mutex);
        ledger_->bytes += r->bytes;
        if (segment) ++ledger_->segments;
    }
}
namespace {
std::size_t group_prefix_lcp(const PrefixCacheSegment& a, const PrefixCacheSegment& b) {
    if (a.identity != b.identity || a.block_rows != b.block_rows) return 0;
    std::size_t n = 0;
    while (n < std::min(a.tokens.size(), b.tokens.size()) && a.tokens[n] == b.tokens[n]) ++n;
    return n;
}
bool group_prefix_before(const PrefixCacheSegment& a, std::string_view ak,
                         const PrefixCacheSegment& b, std::string_view bk) {
    if (a.identity != b.identity) return a.identity.bytes < b.identity.bytes;
    if (a.block_rows != b.block_rows) return a.block_rows < b.block_rows;
    if (a.tokens != b.tokens) return a.tokens < b.tokens;
    return ak < bk;
}
}
void PrefixStateCache::update_group_root_locked(const std::string& key,
    const std::shared_ptr<const PrefixCacheSegment>& root) {
    auto& activity = group_activity_.at(key);
    if (!activity.members) activity.born_epoch = epoch_;
    if (!policy_.fair_reclaim_idle_groups || policy_.retention != PrefixCacheRetention::fair) return;
    if (activity.members) {
        const auto previous = activity.root.lock();
        if (!previous || !root || previous->identity != root->identity || previous->block_rows != root->block_rows ||
            previous->tokens != root->tokens) group_prefix_order_valid_ = false;
        activity.root = root;
        return;
    }
    // Compute the link and overlaps before the final throwing allocation. Nothing
    // visible (including validity and the following LCP) changes on failure.
    auto next = group_prefix_order_.end();
    bool valid = group_prefix_order_valid_ && bool(root);
    if (valid) {
        for (auto it = group_prefix_order_.begin(); it != group_prefix_order_.end(); ++it) {
            const auto other = (*it)->root.lock();
            if (!other) { valid = false; break; }
            if (group_prefix_before(*root, key, *other, (*it)->key)) { next = it; break; }
        }
    }
    std::size_t previous_lcp = 0, next_lcp = 0;
    if (valid) {
        if (next != group_prefix_order_.begin())
            previous_lcp = group_prefix_lcp(*(*std::prev(next))->root.lock(), *root);
        if (next != group_prefix_order_.end())
            next_lcp = group_prefix_lcp(*root, *(*next)->root.lock());
    }
    const auto link = group_prefix_order_.insert(next, &activity);
    activity.root = root;
    activity.key = group_activity_.find(key)->first;
    activity.prefix_order = link;
    activity.prefix_lcp = previous_lcp;
    group_prefix_order_valid_ = valid;
    if (valid && next != group_prefix_order_.end()) (*next)->prefix_lcp = next_lcp;
}
void PrefixStateCache::mark_private_prefix_successors_locked(
    const PrefixCacheCheckpoint* incoming, std::uint64_t newest) {
    for (auto* activity : group_prefix_order_) activity->private_successor = false;
    if (!policy_.fair_reclaim_idle_groups || !group_prefix_order_valid_ || named_session_parking_) return;
    struct Leaf { std::string_view key; GroupActivity* activity; const PrefixCacheSegment* root; std::size_t lcp; std::uint64_t birth_age; };
    std::vector<Leaf> leaves;
    leaves.reserve(group_prefix_order_.size() + 1);
    for (auto* activity : group_prefix_order_) {
        const auto root = activity->root.lock();
        if (!root) return;
        leaves.push_back({activity->key, activity, root.get(), activity->prefix_lcp, newest - activity->born_epoch});
    }
    std::string incoming_key;
    if (incoming) {
        if (incoming->segments.empty()) return;
        const auto& root = *incoming->segments.front();
        incoming_key = incoming->conversation.empty() ? hex(root.digest) : incoming->conversation;
        const auto& key = incoming_key;
        if (const auto a = group_activity_.find(key); a != group_activity_.end()) {
            const auto previous = a->second.root.lock();
            if (!previous || previous->identity != root.identity || previous->block_rows != root.block_rows ||
                previous->tokens != root.tokens) return;
        } else {
            // One linear merge of the arriving group; no sorting or group-pair scan.
            const auto at = std::find_if(leaves.begin(), leaves.end(), [&](const auto& leaf) {
                return group_prefix_before(root, key, *leaf.root, leaf.key);
            });
            const auto offset = static_cast<std::size_t>(at - leaves.begin());
            const auto before = offset ? group_prefix_lcp(*leaves[offset - 1].root, root) : 0;
            if (at != leaves.end()) at->lcp = group_prefix_lcp(root, *at->root);
            leaves.insert(at, Leaf{key, nullptr, &root, before, 0});
        }
    }
    if (leaves.size() < 3) return; // No rival history with which to establish exclusivity.
    std::vector<std::size_t> adjacent(leaves.size() + 1, 0);
    for (std::size_t i = 1; i < leaves.size(); ++i) adjacent[i] = leaves[i].lcp;
    for (std::size_t i = 0; i < leaves.size(); ++i) {
        // Lexicographic neighbors give the exact maximum LCP with every other
        // retained group. Remove the proposed successor before testing privacy.
        if (i && adjacent[i] < leaves[i].root->tokens.size() && adjacent[i] < leaves[i - 1].root->tokens.size() &&
            adjacent[i] > std::max(adjacent[i + 1], std::min(adjacent[i], adjacent[i - 1])) &&
            leaves[i].birth_age > leaves[i - 1].birth_age && leaves[i].activity) leaves[i].activity->private_successor = true;
        if (i + 1 < leaves.size() && adjacent[i + 1] < leaves[i].root->tokens.size() && adjacent[i + 1] < leaves[i + 1].root->tokens.size() &&
            adjacent[i + 1] > std::max(adjacent[i], std::min(adjacent[i + 1], adjacent[i + 2])) &&
            leaves[i].birth_age > leaves[i + 1].birth_age && leaves[i].activity) leaves[i].activity->private_successor = true;
    }
}
void PrefixStateCache::associate_locked(Entry& entry, const std::string& key, bool reused) {
    if (entry.conversations.contains(key) || entry.conversations.size() >= 16) return;
    // Roll back both allocating containers before publishing members or links.
    const auto membership = entry.conversations.insert(key).first;
    try {
        const auto [group, created] = group_activity_.try_emplace(key);
        try {
            update_group_root_locked(key, entry.checkpoint->segments.empty() ? nullptr : entry.checkpoint->segments.front());
        } catch (...) {
            if (created) group_activity_.erase(group);
            throw;
        }
        ++group->second.members;
        group->second.reused = group->second.reused || reused || entry.reuse != 0;
    } catch (...) {
        entry.conversations.erase(membership);
        throw;
    }
}
void PrefixStateCache::add_locked(std::shared_ptr<const PrefixCacheCheckpoint> checkpoint) {
    const auto key = checkpoint->conversation.empty() ?
        hex(checkpoint->segments.empty() ? checkpoint->digest : checkpoint->segments.front()->digest) : checkpoint->conversation;
    // Allocate missing index nodes privately. Keep existing weak values observable
    // until all directory/LRU/association allocations have succeeded.
    decltype(segments_) staged_segments;
    for (const auto& segment : checkpoint->segments)
        if (!segments_.contains(segment->digest)) staged_segments.try_emplace(segment->digest, segment);
    lru_.push_back(checkpoint->digest);
    const auto link = std::prev(lru_.end());
    try {
        const auto [entry, inserted] = entries_.emplace(checkpoint->digest, Entry{checkpoint, link, epoch_ + 1, 0, {}});
        if (!inserted) { lru_.erase(link); return; }
        ++epoch_;
        try {
            if (!checkpoint->retention.scope) associate_locked(entry->second, key);
        } catch (...) {
            --epoch_;
            entries_.erase(entry);
            throw;
        }
    } catch (...) {
        lru_.erase(link);
        throw;
    }
    // Node transfer and weak-pointer assignment allocate nothing; DigestLess is
    // noexcept. Repeated digests retain the same last-value-wins behavior.
    segments_.merge(staged_segments);
    for (const auto& segment : checkpoint->segments) segments_.find(segment->digest)->second = segment;
    prune_receipts_locked();
    metadata_bytes_ += checkpoint->metadata.size() + checkpoint->producer_certificate.size()*sizeof(std::uint64_t);
    register_buffer_locked(checkpoint->fixed, false);
    for (const auto& segment : checkpoint->segments) {
        register_buffer_locked(segment->payload, true);
    }
}
void PrefixStateCache::erase_locked(const ModelContentIdentity& digest, bool eviction) {
    const auto it = entries_.find(digest);
    if (it == entries_.end()) return;
    if (it->second.checkpoint->chain_retention) release_chain_locked(*it->second.checkpoint, eviction);
    if (it->second.checkpoint->retention.scope && it->second.checkpoint->encoded)
        named_views_.at(it->second.checkpoint->retention.scope).erase(it->second.checkpoint->name);
    metadata_bytes_ -= it->second.checkpoint->metadata.size() + it->second.checkpoint->producer_certificate.size()*sizeof(std::uint64_t);
    for (const auto& key : it->second.conversations) {
        const auto group = group_activity_.find(key);
        if (--group->second.members == 0) {
            if (policy_.fair_reclaim_idle_groups && policy_.retention == PrefixCacheRetention::fair) {
                const auto next = std::next(group->second.prefix_order);
                if (next != group_prefix_order_.end()) (*next)->prefix_lcp =
                    std::min(group->second.prefix_lcp, (*next)->prefix_lcp);
                group_prefix_order_.erase(group->second.prefix_order);
            }
            group_activity_.erase(group);
            if (group_activity_.empty()) group_prefix_order_valid_ = true;
        } else if (policy_.fair_reclaim_idle_groups && policy_.retention == PrefixCacheRetention::fair) {
            // Keep the weak representative backed by a retained entry even when
            // callers supplied equal prefixes in distinct segment objects.
            for (const auto& [other_digest, other] : entries_) {
                if (other_digest == digest || !other.conversations.contains(key)) continue;
                if (other.checkpoint->segments.empty()) group_prefix_order_valid_ = false;
                group->second.root = other.checkpoint->segments.empty() ? nullptr : other.checkpoint->segments.front();
                break;
            }
        }
    }
    lru_.erase(it->second.lru);
    entries_.erase(it);
}
void PrefixStateCache::touch_locked(Entry& entry, bool reused) {
    if (entry.checkpoint->retention.scope && entry.checkpoint->retention.fifo) return;
    entry.epoch = ++epoch_;
    if (entry.checkpoint->retention.scope && entry.checkpoint->encoded)
        named_views_.at(entry.checkpoint->retention.scope).at(entry.checkpoint->name).touched = entry.epoch;
    if (reused) {
        ++entry.reuse;
        for (const auto& key : entry.conversations) group_activity_.at(key).reused = true;
    }
    lru_.splice(lru_.end(), lru_, entry.lru);
}

void PrefixStateCache::evict_to_budget_locked(std::uint64_t reserve, const PrefixCacheCheckpoint* incoming,
    std::span<const std::uint64_t> payloads, std::uint64_t metadata, std::string_view protected_name) {
    // Directory/association work is bounded independently of the payload size.
    auto remaining=payload_pool_?std::min<std::size_t>(entries_.size(),256):entries_.size();
    while (!lru_.empty() && remaining--) {
        const bool physical_pressure=payload_pool_ &&
            (metadata>metadata_allowance_ || metadata_bytes_>metadata_allowance_-metadata || !payload_pool_->can_allocate(payloads));
        const bool pressure = reserve > policy_.host_byte_budget ||
            retained_bytes_locked() > policy_.host_byte_budget - reserve || physical_pressure;
        const bool entry_pressure = policy_.retention == PrefixCacheRetention::fair && entries_.size() + (incoming ? 1U : 0U) > 256;
        if (!pressure && !entry_pressure) break;
        const bool mixed = std::any_of(entries_.begin(), entries_.end(), [](const auto& item) {
            return item.second.checkpoint->retention.scope != 0;
        });
        if (mixed && policy_.retention == PrefixCacheRetention::lru) {
            if (!reclaim_one_locked({}, protected_name)) break;
            continue;
        }
        auto first=lru_.begin();
        while(first!=lru_.end() && entries_.at(*first).checkpoint->chain_retention) ++first;
        if(first==lru_.end()) break;
        auto victim = entries_.find(*first);
        if(payload_pool_ && policy_.retention==PrefixCacheRetention::lru) {
            auto oldest=lru_.begin();
            while(oldest!=lru_.end() && entries_.at(*oldest).checkpoint.use_count()!=1) ++oldest;
            if(oldest==lru_.end()) break;
            victim=entries_.find(*oldest);
        }
        unsigned victim_rank = 0;
        bool victim_superseded = false;
        long double best = -1;
        if (policy_.retention == PrefixCacheRetention::fair) {
            struct Group {
                long double bytes = 0;
                std::uint64_t largest = 0;
                std::uint64_t age = UINT64_MAX;
                bool reused = false;
                const GroupActivity* activity = nullptr;
                std::map<ModelContentIdentity, long double, DigestLess> branches;
                std::map<ModelContentIdentity, std::uint64_t, DigestLess> branch_largest;
            };
            std::map<std::string, Group> groups;
            std::map<ModelContentIdentity, std::size_t, DigestLess> refs;
            std::map<ModelContentIdentity, std::map<std::string, std::size_t>, DigestLess> owners;
            // Include a would-be capture in fairness accounting before allocating its payload.
            Entry estimate;
            // One epoch is one capture or validated touch anywhere in this store,
            // across all conversations; neither wall time nor per-conversation turns.
            // Compare modular ages from a common reference, never raw epoch maxima.
            const auto newest = epoch_ + (incoming ? 1U : 0U);
            std::uint64_t newest_age = UINT64_MAX;
            bool certified_only = true;
            std::vector<std::pair<ModelContentIdentity, const Entry*>> ranking;
            ranking.reserve(entries_.size() + (incoming ? 1U : 0U));
            for (const auto& [d,e] : entries_) if (!e.checkpoint->retention.scope) ranking.emplace_back(d, &e);
            if (incoming) {
                estimate.epoch = epoch_ + 1;
                estimate.checkpoint = std::shared_ptr<const PrefixCacheCheckpoint>(incoming, [](const auto*) {});
                estimate.conversations.insert(incoming->conversation.empty() ?
                    hex(incoming->segments.empty() ? incoming->digest : incoming->segments.front()->digest) : incoming->conversation);
                ranking.emplace_back(incoming->digest, &estimate);
            }
            for (const auto& item : ranking) {
                const auto& e = *item.second;
                for (const auto& seg : e.checkpoint->segments) {
                    ++refs[seg->digest];
                    for (const auto& key : e.conversations) ++owners[seg->digest][key];
                }
            }
            std::map<ModelContentIdentity, std::map<std::string, ModelContentIdentity>, DigestLess> branches;
            for (const auto& [d,ep] : ranking) {
              const auto& e = *ep;
              certified_only = certified_only && !e.checkpoint->producer_certificate.empty() &&
                  !e.checkpoint->segments.empty();
              for (const auto& key : e.conversations) {
                const auto& c = *e.checkpoint;
                // Branch = first block after the deepest retained shared ancestor.
                // An ancestor ending at this checkpoint uses its own terminal digest.
                std::size_t shared = 0;
                while (shared<c.segments.size() && owners.at(c.segments[shared]->digest).at(key)>1) ++shared;
                const auto branch = c.segments.empty() ? d : c.segments[std::min(shared,c.segments.size()-1)]->digest;
                branches[d][key] = branch;
                auto full = checkpoint_host_bytes(c);
                auto charge = static_cast<long double>(full)/e.conversations.size();
                for (const auto& seg : c.segments) {
                    full += seg->payload.bytes;
                    charge += static_cast<long double>(seg->payload.bytes) / (owners[seg->digest].at(key) * owners[seg->digest].size());
                }
                auto& g = groups[key]; g.bytes += charge; g.largest = std::max(g.largest,full);
                g.age = std::min(g.age, newest - e.epoch);
                if (const auto activity = group_activity_.find(key); activity != group_activity_.end()) {
                    g.reused = activity->second.reused;
                    g.activity = &activity->second;
                }
                newest_age = std::min(newest_age, g.age);
                g.branches[branch] += charge;
                g.branch_largest[branch] = std::max(g.branch_largest[branch],full);
              }
            }
            if (certified_only) mark_private_prefix_successors_locked(incoming, newest);
            for (auto it = entries_.begin(); it != entries_.end(); ++it) {
                if (it->second.checkpoint->retention.scope || it->second.checkpoint.use_count() != 1) continue;
                const auto& c = *it->second.checkpoint;
                unsigned rank = 0;
                unsigned ordinary_rank = 0;
                for (const auto& key : it->second.conversations) {
                    const auto& g = groups.at(key);
                    const auto cap = std::min(policy_.host_byte_budget, std::max(g.largest,policy_.host_byte_budget/3));
                    const auto branch = branches.at(it->first).at(key);
                    const auto branch_cap = std::min(cap,std::max(g.branch_largest.at(branch),cap/g.branches.size()));
                    const bool borrowed = groups.size()>1 && g.bytes>cap;
                    // Age alone cannot establish abandonment. Require a newer group
                    // sharing a prefix private to this pair, beyond every rival's
                    // divergence. Common headers and reused groups cannot qualify.
                    const bool idle = policy_.fair_reclaim_idle_groups && !named_session_parking_ && certified_only &&
                        !g.reused && g.activity && g.activity->private_successor && g.age - newest_age > 32;
                    ordinary_rank = std::max(ordinary_rank, g.branches.at(branch)>branch_cap ? 3U : borrowed ? 2U : 0U);
                    rank = std::max(rank, g.branches.at(branch)>branch_cap ? 3U : borrowed || idle ? 2U : 0U);
                }
                const bool unleased = it->second.checkpoint.use_count()==1;
                std::uint64_t marginal = unleased ? c.metadata.size() + c.producer_certificate.size()*sizeof(std::uint64_t) : 0;
                if (unleased && c.fixed.data.use_count()==1) marginal += c.fixed.bytes;
                for (const auto& seg : c.segments)
                    if (unleased && refs[seg->digest] == 1 && seg.use_count()==1) marginal += seg->payload.bytes;
                const auto score = static_cast<long double>(marginal) * (1 + epoch_ - it->second.epoch) / (1 + it->second.reuse);
                if (rank > victim_rank || (rank == victim_rank && (score > best ||
                    (score == best && (it->second.epoch < victim->second.epoch ||
                        (it->second.epoch == victim->second.epoch && DigestLess{}(it->first,victim->first))))))) {
                    victim = it; victim_rank = rank; best = score;
                    victim_superseded = rank > ordinary_rank;
                }
            }
        }
        if (mixed && policy_.retention == PrefixCacheRetention::fair) {
            auto generic = entries_.end();
            for (const auto& d : lru_) {
                auto i = entries_.find(d);
                const auto& c = *i->second.checkpoint;
                if (c.retention.scope && !c.chain_retention && !i->second.protected_entry && (protected_name.empty() || c.name != protected_name)) {
                    generic = i; break;
                }
            }
            // Keep inherited fair/certificate/group ranking inside its domain;
            // arbitrate that candidate against generic FIFO/LRU by global age.
            if (generic != entries_.end() && (best < 0 || generic->second.epoch < victim->second.epoch)) {
                const auto d = generic->first;
                ++scope_evictions_.at(generic->second.checkpoint->retention.scope);
                erase_locked(d); ++stats_.evicted; ++stats_.evictions_space;
                continue;
            }
        }
        if (policy_.retention == PrefixCacheRetention::fair && best < 0) break;
        const auto digest = victim->first;
        erase_locked(digest);
        ++stats_.evicted;
        if (victim_superseded) ++stats_.evictions_superseded;
        else if (victim_rank == 2) ++stats_.evictions_conversation_cap;
        else if (victim_rank == 3) ++stats_.evictions_branch_cap;
        else ++stats_.evictions_space;
    }
}

bool PrefixStateCache::insert(std::shared_ptr<const PrefixCacheCheckpoint> checkpoint, Reservation* reservation) {
    require(checkpoint && !checkpoint->retention.scope && !checkpoint->snapshot && !checkpoint->resident && !checkpoint->encoded,
        "session RAM payloads cannot enter the legacy prefix disk writer");
    require(checkpoint && checkpoint->block_rows && checkpoint->position == checkpoint->segments.size() * checkpoint->block_rows,
        "prefix cache checkpoint is malformed");
    require(checkpoint->conversation.size()<=128,"prefix cache conversation key exceeds 128 bytes");
    {
        std::lock_guard lock(mutex_);
        if (const auto found = entries_.find(checkpoint->digest); found != entries_.end()) {
            if (!checkpoint->conversation.empty() && checkpoint->conversation.size()<=128 && found->second.conversations.size()<16)
                associate_locked(found->second, checkpoint->conversation);
            touch_locked(found->second);
            return true;
        }
        if (reservation && reservation->ledger_ == ledger_) {
            std::lock_guard ledger_lock(ledger_->mutex);
            ledger_->reserved -= reservation->bytes_; reservation->bytes_ = 0;
        }
        ++stats_.captures;
        const auto metadata=checkpoint->metadata.size()+checkpoint->producer_certificate.size()*sizeof(std::uint64_t);
        if(payload_pool_ && metadata>metadata_allowance_-std::min(metadata_allowance_,metadata_bytes_)) {
            ++stats_.budget_skips; return false;
        }
        const auto need = incremental_bytes_locked(*checkpoint);
        if (need > policy_.host_byte_budget) {
            ++stats_.rejected_over_budget; ++stats_.budget_skips;
            return false;
        }
        if (!policy_.legacy) {
            evict_to_budget_locked(need, policy_.retention == PrefixCacheRetention::fair ? checkpoint.get() : nullptr);
            if ((policy_.retention == PrefixCacheRetention::fair && entries_.size() >= 256) ||
                retained_bytes_locked() > policy_.host_byte_budget - need) {
                ++stats_.budget_skips; ++stats_.rejected_over_budget; return false;
            }
        } else evict_to_budget_locked(need);
        add_locked(checkpoint);
        ++stats_.inserted;
    }
    if (writer_.joinable() && (checkpoint->producer_certificate.empty() || (policy_.certified_disk && !policy_.legacy))) {
        {
            std::lock_guard lock(queue_mutex_);
            queue_.push_back(std::move(checkpoint));
        }
        queue_cv_.notify_one();
    }
    return true;
}

bool PrefixStateCache::contains(const ModelContentIdentity& digest) {
    {
        std::lock_guard lock(mutex_);
        if (entries_.count(digest)) return true;
    }
    if (policy_.disk_directory.empty()) return false;
    std::error_code error;
    return std::filesystem::exists(checkpoint_path(policy_.disk_directory, digest), error);
}

std::shared_ptr<const PrefixCacheSegment> PrefixStateCache::find_segment(const ModelContentIdentity& digest) {
    std::lock_guard lock(mutex_);
    const auto found = segments_.find(digest);
    return found == segments_.end() ? nullptr : found->second.lock();
}

namespace {
// Bumps the disk epoch when a function that publishes or removes checkpoint files starts and when it ends,
// so a lookup ticket whose disk probes overlap such a change is never trusted.
struct DiskChange {
    std::atomic<std::uint64_t>& epoch;
    explicit DiskChange(std::atomic<std::uint64_t>& e) : epoch(e) { epoch.fetch_add(1, std::memory_order_acq_rel); }
    ~DiskChange() { epoch.fetch_add(1, std::memory_order_acq_rel); }
};
}  // namespace

PrefixLookupTicket PrefixStateCache::prepare_ticket(const ModelContentIdentity& identity, std::span<const std::uint32_t> tokens,
    std::uint32_t block_rows, std::uint64_t limit, bool probe_disk) const {
    require(block_rows != 0, "prefix cache lookup needs a block size");
    PrefixLookupTicket ticket;
    ticket.identity = identity;
    ticket.block_rows = block_rows;
    ticket.tokens = tokens;
    ticket.lookup_tokens = std::min<std::uint64_t>(limit, tokens.size());
    ticket.digests = prefix_cache_block_digests(identity, tokens, block_rows, ticket.lookup_tokens / block_rows);
    if (!probe_disk || policy_.disk_directory.empty()) return ticket;
    // The epoch is read first: a publication or removal that overlaps the probes changes it.
    ticket.disk_epoch = disk_epoch_.load(std::memory_order_acquire);
    ticket.disk.assign(ticket.digests.size(), PrefixLookupTicket::disk_absent);
    for (std::size_t k = 0; k < ticket.digests.size(); ++k) {
        const auto path = checkpoint_path(policy_.disk_directory, ticket.digests[k]);
        std::error_code error;
        if (!std::filesystem::exists(path, error)) continue;
        try {
            File probe(path, "rb");
            CheckpointHeader candidate;
            if (!probe.open()) continue;
            probe.read(&candidate, sizeof(candidate));
            if (candidate.reserved <= 1U)
                ticket.disk[k] = candidate.reserved ? PrefixLookupTicket::disk_certified : PrefixLookupTicket::disk_legacy;
        } catch (const std::exception&) {
            ticket.disk[k] = PrefixLookupTicket::disk_unknown; // the lookup probes it itself and accounts the failure
        }
    }
    return ticket;
}

PrefixLookupTicket PrefixStateCache::prepare_lookup(const ModelContentIdentity& identity, std::span<const std::uint32_t> tokens,
    std::uint32_t block_rows, std::uint64_t limit) const {
    auto owned = std::make_shared<const std::vector<std::uint32_t>>(tokens.begin(), tokens.end());
    auto ticket = prepare_ticket(identity, *owned, block_rows, limit, true);
    ticket.owned = std::move(owned);
    return ticket;
}

std::shared_ptr<const PrefixCacheCheckpoint> PrefixStateCache::find_longest(const ModelContentIdentity& identity,
    std::span<const std::uint32_t> tokens, std::uint32_t block_rows, std::uint64_t limit, bool promote, std::string_view conversation,
    const std::function<bool(const PrefixCacheCheckpoint&)>& eligible) {
    // Not prepared ahead: the disk tier is probed during the walk, only for the blocks above the hit.
    return find_longest(prepare_ticket(identity, tokens, block_rows, limit, false), promote, conversation, eligible);
}

std::shared_ptr<const PrefixCacheCheckpoint> PrefixStateCache::find_longest(const PrefixLookupTicket& ticket, bool promote,
    std::string_view conversation, const std::function<bool(const PrefixCacheCheckpoint&)>& eligible) {
    require(ticket.block_rows != 0 && ticket.digests.size() == ticket.lookup_tokens / ticket.block_rows &&
            ticket.lookup_tokens <= ticket.tokens.size(), "prefix cache lookup ticket is not prepared");
    const auto& identity = ticket.identity;
    const auto block_rows = ticket.block_rows;
    const auto tokens = ticket.tokens;
    const auto& digests = ticket.digests;
    // One acquisition for the bookkeeping and the host-tier probe of every block of the prompt.
    std::vector<std::shared_ptr<const PrefixCacheCheckpoint>> host(digests.size());
    {
        std::lock_guard lock(mutex_);
        for (auto it=receipts_.begin(); it!=receipts_.end();)
            if (it->second.expired()) it=receipts_.erase(it); else ++it;
        ++stats_.lookups;
        stats_.lookup_tokens += ticket.lookup_tokens;
        for (std::size_t k = 0; k < digests.size(); ++k)
            if (const auto it = entries_.find(digests[k]); it != entries_.end()) host[k] = it->second.checkpoint;
    }
    // Prepared disk metadata is used only if no checkpoint file was published or removed since it was read.
    const bool prepared_disk = ticket.disk.size() == digests.size() &&
        ticket.disk_epoch == disk_epoch_.load(std::memory_order_acquire);
    for (auto k = digests.size(); k > 0; --k) {
        const auto& digest = digests[k - 1];
        const std::shared_ptr<const PrefixCacheCheckpoint> found = std::move(host[k - 1]);
        if (found) {
            if (eligible && !eligible(*found)) {
                std::lock_guard lock(mutex_); ++stats_.plan_misses; continue;
            }
            const auto position = static_cast<std::uint64_t>(k) * block_rows;
            const auto stored = found->tokens();
            if (found->identity == identity && found->block_rows == block_rows && found->position == position &&
                stored.size() == position && std::equal(stored.begin(), stored.end(), tokens.begin())) {
                std::lock_guard lock(mutex_);
                if (auto it=entries_.find(digest); it!=entries_.end()) {
                    touch_locked(it->second,true);
                    if (promote && !conversation.empty() && conversation.size()<=128 && it->second.conversations.size()<16)
                        associate_locked(it->second, std::string(conversation), true);
                }
                ++stats_.host_hits; stats_.hit_tokens += position; stats_.restored_tokens += position;
                return found;
            }
            { std::lock_guard lock(mutex_); erase_locked(digest); ++stats_.evicted; ++stats_.evictions_invalid; }
            continue;  // digest collision or corrupt entry: never used
        }
        if (policy_.disk_directory.empty() || (eligible && (!policy_.certified_disk || policy_.legacy))) continue;
        std::error_code error;
        const std::uint8_t known = prepared_disk ? ticket.disk[k - 1] : std::uint8_t{PrefixLookupTicket::disk_unknown};
        if (known == PrefixLookupTicket::disk_absent) continue;
        if (known == PrefixLookupTicket::disk_unknown &&
            !std::filesystem::exists(checkpoint_path(policy_.disk_directory, digest), error)) continue;
        try {
            // Certified and legacy lookups never cross. Check before payload allocation.
            if (known == PrefixLookupTicket::disk_unknown) {
                File probe(checkpoint_path(policy_.disk_directory, digest), "rb");
                CheckpointHeader candidate;
                if (!probe.open()) continue;
                probe.read(&candidate, sizeof(candidate));
                if (candidate.reserved != (eligible ? 1U : 0U)) continue;
            } else if ((known == PrefixLookupTicket::disk_certified) != static_cast<bool>(eligible)) continue;
            if (auto loaded = load_from_disk(identity, digest, tokens.first(static_cast<std::size_t>(k * block_rows)), block_rows)) {
                if (eligible && (loaded->producer_certificate.empty() || !eligible(*loaded))) {
                    std::lock_guard lock(mutex_); ++stats_.plan_misses; continue;
                }
                {
                    std::lock_guard lock(mutex_);
                    // No-store hits protect existing retained groups without creating
                    // associations or ticking the capture/touch clock.
                    const auto key = conversation.empty() ?
                        hex(loaded->segments.empty() ? loaded->digest : loaded->segments.front()->digest) : std::string(conversation);
                    if (auto group = group_activity_.find(key); group != group_activity_.end()) group->second.reused = true;
                    ++stats_.disk_hits; stats_.hit_tokens += loaded->position; stats_.restored_tokens += loaded->position;
                }
                // Into the host tier without writing it back.
                if (promote) {
                    std::lock_guard lock(mutex_);
                    const auto need = incremental_bytes_locked(*loaded);
                    if (need <= policy_.host_byte_budget && !entries_.count(loaded->digest)) {
                        // The returned handle is the directory lease, including its fixed blob.
                        auto promoted=std::const_pointer_cast<PrefixCacheCheckpoint>(loaded);
                        if (conversation.size()<=128) promoted->conversation=conversation;
                        evict_to_budget_locked(need, policy_.retention == PrefixCacheRetention::fair ? promoted.get() : nullptr);
                        if ((policy_.retention != PrefixCacheRetention::fair || entries_.size()<256) &&
                            (policy_.legacy || retained_bytes_locked() <= policy_.host_byte_budget - need)) {
                            add_locked(promoted);
                            auto& entry = entries_.at(promoted->digest);
                            entry.reuse=1;
                            for (const auto& key : entry.conversations) group_activity_.at(key).reused=true;
                        }
                    }
                }
                std::filesystem::last_write_time(checkpoint_path(policy_.disk_directory, digest),
                    std::filesystem::file_time_type::clock::now(), error);
                return loaded;
            }
        } catch (const std::exception&) {
            std::lock_guard lock(mutex_);
            ++stats_.disk_read_failures;
        }
    }
    std::lock_guard lock(mutex_);
    ++stats_.misses;
    return nullptr;
}

std::shared_ptr<const PrefixCacheSegment> PrefixStateCache::load_segment(const ModelContentIdentity& identity,
    const ModelContentIdentity& digest, std::uint32_t block_rows, Reservation* reservation) {
    if (auto retained = find_segment(digest)) return retained;
    File file(segment_path(policy_.disk_directory, digest), "rb", File::Stream::payload);
    if (!file.open()) throw std::runtime_error("prefix cache segment file is missing");
    SegmentHeader header;
    file.read(&header, sizeof(header));
    if (header.magic != segment_magic || get(header.identity) != identity || get(header.digest) != digest ||
        header.block_rows != block_rows)
        throw std::runtime_error("prefix cache segment header does not match");
    auto segment = std::make_shared<PrefixCacheSegment>();
    segment->identity = identity;
    segment->digest = digest;
    segment->parent = get(header.parent);
    segment->block_index = header.block_index;
    segment->block_rows = block_rows;
    segment->tokens.resize(block_rows);
    file.read(segment->tokens.data(), static_cast<std::uint64_t>(block_rows) * sizeof(std::uint32_t));
    if (prefix_cache_block_digest(segment->parent, segment->tokens) != digest)
        throw std::runtime_error("prefix cache segment tokens do not match their digest");
    segment->payload = reservation ? allocate(header.payload_bytes, *reservation, true) : allocate(header.payload_bytes);
    file.read(segment->payload.data.get(), header.payload_bytes);
    if (prefix_cache_checksum(segment->payload.span()) != header.payload_checksum)
        throw std::runtime_error("prefix cache segment payload checksum differs");
    std::lock_guard lock(mutex_);
    stats_.disk_read_bytes += sizeof(header) + header.payload_bytes;
    return segment;
}

std::shared_ptr<const PrefixCacheCheckpoint> PrefixStateCache::load_from_disk(const ModelContentIdentity& identity,
    const ModelContentIdentity& digest, std::span<const std::uint32_t> tokens, std::uint32_t block_rows) {
    const auto start = std::chrono::steady_clock::now();
    File file(checkpoint_path(policy_.disk_directory, digest), "rb", File::Stream::payload);
    if (!file.open()) return nullptr;
    CheckpointHeader header;
    file.read(&header, sizeof(header));
    if (header.magic != checkpoint_magic || get(header.identity) != identity || get(header.digest) != digest ||
        header.block_rows != block_rows || header.position != tokens.size() ||
        header.segment_count != header.position / block_rows || header.position % block_rows ||
        header.reserved > 1U || header.metadata_bytes > policy_.host_byte_budget)
        throw std::runtime_error("prefix cache checkpoint header does not match");
    std::vector<std::uint8_t> segment_digests(static_cast<std::size_t>(header.segment_count * 32));
    file.read(segment_digests.data(), segment_digests.size());
    // Inspect all payload sizes before the first pinned allocation. Conservative credits
    // also cover segments another reader may release between inspection and loading.
    std::uint64_t payload_bytes = header.fixed_bytes;
    if (!policy_.legacy) {
        for (std::uint64_t k = 0; k < header.segment_count; ++k) {
            ModelContentIdentity sd;
            std::memcpy(sd.bytes.data(), segment_digests.data() + k * 32, 32);
            File segment_file(segment_path(policy_.disk_directory, sd), "rb");
            if (!segment_file.open()) throw std::runtime_error("prefix cache segment file is missing");
            SegmentHeader sh; segment_file.read(&sh, sizeof(sh));
            if (sh.payload_bytes > policy_.host_byte_budget || payload_bytes > policy_.host_byte_budget - sh.payload_bytes) {
                std::lock_guard lock(mutex_); ++stats_.budget_skips; return nullptr;
            }
            payload_bytes += sh.payload_bytes;
        }
    }
    auto reservation = policy_.legacy ? Reservation{} : reserve(payload_bytes);
    if (!policy_.legacy && !reservation) return nullptr;
    auto checkpoint = std::make_shared<PrefixCacheCheckpoint>();
    checkpoint->identity = identity;
    checkpoint->digest = digest;
    checkpoint->position = header.position;
    checkpoint->block_rows = block_rows;
    checkpoint->metadata.resize(static_cast<std::size_t>(header.metadata_bytes));
    file.read(checkpoint->metadata.data(), header.metadata_bytes);
    if (prefix_cache_checksum(checkpoint->metadata) != header.metadata_checksum)
        throw std::runtime_error("prefix cache metadata checksum differs");
    if (header.reserved == 1U) {
        std::uint64_t count = 0;
        if (checkpoint->metadata.size() < sizeof(count)) throw std::runtime_error("prefix cache certificate is truncated");
        std::memcpy(&count, checkpoint->metadata.data(), sizeof(count));
        if (!count || count > (checkpoint->metadata.size()-sizeof(count))/sizeof(std::uint64_t))
            throw std::runtime_error("prefix cache certificate is malformed");
        checkpoint->producer_certificate.resize(static_cast<std::size_t>(count));
        std::memcpy(checkpoint->producer_certificate.data(), checkpoint->metadata.data()+sizeof(count), count*sizeof(std::uint64_t));
        checkpoint->metadata.erase(checkpoint->metadata.begin(),
            checkpoint->metadata.begin()+static_cast<std::ptrdiff_t>(sizeof(count)+count*sizeof(std::uint64_t)));
    }
    checkpoint->fixed = policy_.legacy ? allocate(header.fixed_bytes) : allocate(header.fixed_bytes, reservation);
    file.read(checkpoint->fixed.data.get(), header.fixed_bytes);
    if (prefix_cache_checksum(checkpoint->fixed.span()) != header.fixed_checksum)
        throw std::runtime_error("prefix cache fixed-state checksum differs");
    auto previous = identity;
    for (std::uint64_t k = 0; k < header.segment_count; ++k) {
        ModelContentIdentity segment_digest;
        std::memcpy(segment_digest.bytes.data(), segment_digests.data() + k * 32, 32);
        auto segment = load_segment(identity, segment_digest, block_rows, policy_.legacy ? nullptr : &reservation);
        // The chain must be exactly the requested tokens.
        if (segment->block_index != k || segment->parent != previous ||
            !std::equal(segment->tokens.begin(), segment->tokens.end(),
                tokens.begin() + static_cast<std::ptrdiff_t>(k * block_rows)))
            throw std::runtime_error("prefix cache checkpoint chain does not match the requested tokens");
        previous = segment_digest;
        checkpoint->segments.push_back(std::move(segment));
    }
    if (previous != digest) throw std::runtime_error("prefix cache checkpoint chain does not end at its digest");
    std::lock_guard lock(mutex_);
    ++stats_.disk_reads;
    stats_.disk_read_bytes += sizeof(header) + header.fixed_bytes + header.metadata_bytes;
    stats_.disk_read_ms += elapsed_ms(start);
    return checkpoint;
}

void PrefixStateCache::write_to_disk(const PrefixCacheCheckpoint& checkpoint) {
    const DiskChange disk_change(disk_epoch_);
    const auto start = std::chrono::steady_clock::now();
    const auto& dir = policy_.disk_directory;
    std::uint64_t written = 0;
    std::error_code error;
    for (const auto& segment : checkpoint.segments) {
        const auto path = segment_path(dir, segment->digest);
        if (std::filesystem::exists(path, error)) {
            // This segment may have been published by an earlier non-durable writer.
            if(policy_.durable && !policy_.legacy) {
                session::checkpoint_io::sync_existing(path);
            }
            continue;
        }
        SegmentHeader header;
        put(header.identity, segment->identity);
        put(header.digest, segment->digest);
        put(header.parent, segment->parent);
        header.block_index = segment->block_index;
        header.block_rows = segment->block_rows;
        header.payload_bytes = segment->payload.bytes;
        header.payload_checksum = prefix_cache_checksum(segment->payload.span());
        session::checkpoint_io::Publication file(path,
            {policy_.durable && !policy_.legacy,session::checkpoint_io::DirectIo::automatic,0666,{},false});
        file.write(std::as_bytes(std::span(&header,1)));
        file.write(std::as_bytes(std::span(segment->tokens)));
        file.write(segment->payload.span());
        file.commit();
        written += sizeof(header) + segment->payload.bytes;
    }
    const auto path = checkpoint_path(dir, checkpoint.digest);
    // Every segment name is durable before publishing the checkpoint that references it.
    session::checkpoint_io::sync_directory(dir, policy_.durable && !policy_.legacy);
    if (!std::filesystem::exists(path, error)) {
        CheckpointHeader header;
        put(header.identity, checkpoint.identity);
        put(header.digest, checkpoint.digest);
        header.position = checkpoint.position;
        header.block_rows = checkpoint.block_rows;
        header.segment_count = checkpoint.segments.size();
        header.fixed_bytes = checkpoint.fixed.bytes;
        const auto metadata = disk_metadata(checkpoint);
        header.reserved = checkpoint.producer_certificate.empty() ? 0U : 1U;
        header.metadata_bytes = metadata.size();
        header.fixed_checksum = prefix_cache_checksum(checkpoint.fixed.span());
        header.metadata_checksum = prefix_cache_checksum(metadata);
        session::checkpoint_io::Publication file(path,
            {policy_.durable && !policy_.legacy,session::checkpoint_io::DirectIo::automatic,0666});
        file.write(std::as_bytes(std::span(&header,1)));
        for (const auto& segment : checkpoint.segments) file.write(std::as_bytes(std::span(segment->digest.bytes)));
        file.write(metadata);
        file.write(checkpoint.fixed.span());
        file.commit();
        written += sizeof(header) + checkpoint.fixed.bytes + metadata.size();
    } else if(policy_.durable && !policy_.legacy) {
        session::checkpoint_io::sync_existing(path);
        session::checkpoint_io::sync_directory(dir,true);
    }
    std::lock_guard lock(mutex_);
    ++stats_.disk_writes;
    stats_.disk_write_bytes += written;
    stats_.disk_write_ms += elapsed_ms(start);
}

void PrefixStateCache::update_disk_gauge() {
    std::uint64_t bytes=0;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(policy_.disk_directory,error)) {
        if (entry.is_regular_file(error)) bytes+=entry.file_size(error);
    }
    std::lock_guard lock(mutex_); stats_.disk_bytes=bytes;
}

void PrefixStateCache::enforce_disk_budget() {
    if (!policy_.disk_byte_budget) return;
    const DiskChange disk_change(disk_epoch_);
    namespace fs = std::filesystem;
    struct Item { fs::path path; fs::file_time_type time; std::uint64_t bytes; };
    std::vector<Item> checkpoints;
    std::uint64_t total = 0;
    std::error_code error;
    for (const auto& entry : fs::directory_iterator(policy_.disk_directory, error)) {
        if (!entry.is_regular_file(error)) continue;
        const auto bytes = entry.file_size(error);
        total += bytes;
        if (entry.path().extension() == ".tmpc") checkpoints.push_back({entry.path(), entry.last_write_time(error), bytes});
    }
    if (total <= policy_.disk_byte_budget) return;
    std::sort(checkpoints.begin(), checkpoints.end(), [](const Item& a, const Item& b) { return a.time < b.time; });
    std::size_t removed = 0;
    // Approximate: a removed checkpoint frees its own file now and its unshared segments below.
    while (removed + 1 < checkpoints.size() && total > policy_.disk_byte_budget) {
        fs::remove(checkpoints[removed].path, error);
        total -= std::min(total, checkpoints[removed].bytes);
        ++removed;
        std::lock_guard lock(mutex_);
        ++stats_.disk_evictions;
    }
    if (!removed) return;
    // Delete segment files no remaining checkpoint references.
    std::unordered_set<std::string> referenced;
    for (std::size_t i = removed; i < checkpoints.size(); ++i) {
        File file(checkpoints[i].path, "rb");
        if (!file.open()) continue;
        CheckpointHeader header;
        try {
            file.read(&header, sizeof(header));
            if (header.magic != checkpoint_magic) continue;
            std::vector<std::uint8_t> digests(static_cast<std::size_t>(header.segment_count * 32));
            file.read(digests.data(), digests.size());
            for (std::uint64_t k = 0; k < header.segment_count; ++k) {
                ModelContentIdentity digest;
                std::memcpy(digest.bytes.data(), digests.data() + k * 32, 32);
                referenced.insert(segment_path(policy_.disk_directory, digest).filename().string());
            }
        } catch (const std::exception&) {
        }
    }
    for (const auto& entry : fs::directory_iterator(policy_.disk_directory, error))
        if (entry.path().extension() == ".tmps" && !referenced.count(entry.path().filename().string()))
            fs::remove(entry.path(), error);
}

void PrefixStateCache::sweep_disk(std::chrono::seconds min_age) {
    const DiskChange disk_change(disk_epoch_);
    namespace fs = std::filesystem;
    std::error_code error;
    const auto now = fs::file_time_type::clock::now();
    const auto old = [&](const fs::directory_entry& entry) {
        const auto time = entry.last_write_time(error);
        return !error && now - time > min_age;
    };
    std::unordered_set<std::string> referenced;
    std::vector<fs::path> segments;
    for (const auto& entry : fs::directory_iterator(policy_.disk_directory, error)) {
        const auto name = entry.path().filename().string();
        if (name.find(".tmp.") != std::string::npos) {
            if (old(entry)) fs::remove(entry.path(), error);
            continue;
        }
        if (entry.path().extension() == ".tmps") { segments.push_back(entry.path()); continue; }
        if (entry.path().extension() != ".tmpc") continue;
        File file(entry.path(), "rb");
        if (!file.open()) continue;
        try {
            CheckpointHeader header;
            file.read(&header, sizeof(header));
            if (header.magic != checkpoint_magic || header.segment_count > (1ULL << 24)) continue;
            std::vector<std::uint8_t> digests(static_cast<std::size_t>(header.segment_count * 32));
            file.read(digests.data(), digests.size());
            for (std::uint64_t k = 0; k < header.segment_count; ++k) {
                ModelContentIdentity digest;
                std::memcpy(digest.bytes.data(), digests.data() + k * 32, 32);
                referenced.insert(segment_path(policy_.disk_directory, digest).filename().string());
            }
        } catch (const std::exception&) {
        }
    }
    for (const auto& path : segments)
        if (!referenced.count(path.filename().string()) && old(fs::directory_entry(path, error))) fs::remove(path, error);
}

void PrefixStateCache::writer_loop() {
    for (;;) {
        std::shared_ptr<const PrefixCacheCheckpoint> next;
        {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
            if (queue_.empty()) {
                writer_busy_ = false;
                idle_cv_.notify_all();
                return;
            }
            next = std::move(queue_.front());
            queue_.pop_front();
            writer_busy_ = true;
        }
        try {
            write_to_disk(*next);
            enforce_disk_budget();
            update_disk_gauge();
        } catch (const std::exception&) {
            std::lock_guard lock(mutex_);
            ++stats_.disk_write_failures;
        }
        {
            std::lock_guard lock(queue_mutex_);
            writer_busy_ = false;
        }
        idle_cv_.notify_all();
    }
}

void PrefixStateCache::flush() {
    if (!writer_.joinable()) return;
    std::unique_lock lock(queue_mutex_);
    idle_cv_.wait(lock, [&] { return queue_.empty() && !writer_busy_; });
}

bool PrefixStateCache::admit_capture(const PrefixCacheCheckpoint& checkpoint) {
    std::lock_guard lock(mutex_);
    const auto need = incremental_bytes_locked(checkpoint);
    if (need <= policy_.host_byte_budget) {
        evict_to_budget_locked(need, policy_.retention == PrefixCacheRetention::fair ? &checkpoint : nullptr);
        const auto remaining = incremental_bytes_locked(checkpoint);
        const bool directory_fits = policy_.retention == PrefixCacheRetention::lru || entries_.size() < 256;
        if (directory_fits && remaining <= policy_.host_byte_budget &&
            retained_bytes_locked() <= policy_.host_byte_budget - remaining) return true;
    }
    ++stats_.budget_skips;
    return false;
}
void PrefixStateCache::record_capture_failure() {
    std::lock_guard lock(mutex_);
    ++stats_.capture_failures;
}

void PrefixStateCache::record_no_store_skip() {
    std::lock_guard lock(mutex_); ++stats_.no_store_skips;
}
PrefixStateCacheStats PrefixStateCache::counters() {
    std::lock_guard lock(mutex_);
    auto out = stats_;
    for(std::size_t i=0;i<certificate_declines_.size();++i)
        out.certificate_declines[i]=certificate_declines_[i].load(std::memory_order_relaxed);
    if (policy_.legacy) {
        out.host_bytes = host_bytes_locked(); out.host_segments = segments_.size();
        std::lock_guard ledger_lock(ledger_->mutex);
        out.host_bytes += ledger_->encoded;
    } else {
        std::lock_guard ledger_lock(ledger_->mutex);
        // Count the retained slab (including free ranges) and its reserved metadata once.
        out.host_bytes = payload_pool_ ? policy_.host_byte_budget : ledger_->bytes + metadata_bytes_;
        if(payload_pool_) out.pool_free_bytes=payload_pool_->free_bytes();
        out.host_segments = ledger_->segments;
    }
    {
        std::lock_guard ledger_lock(ledger_->mutex);
        reap_immutable_locked(*ledger_);
        out.immutable_bytes = ledger_->immutable;
        out.retention_metadata_bytes = ledger_->retention_metadata + ledger_->certificates;
        out.reserved_bytes = ledger_->reserved;
        out.encoded_host_bytes = ledger_->encoded;
        out.policy_exempt_bytes = ledger_->exempt;
    }
    out.service_bytes = retained_bytes_locked();
    if (out.service_bytes > policy_.host_byte_budget)
        out.legacy_live_over_budget_bytes = out.service_bytes - policy_.host_byte_budget;
    if (out.policy_exempt_bytes > UINT64_MAX - out.service_bytes) out.service_bytes = UINT64_MAX;
    else out.service_bytes += out.policy_exempt_bytes;
    if (payload_pool_) out.service_bytes = policy_.host_byte_budget;
    out.session_entries = static_cast<std::uint64_t>(std::count_if(entries_.begin(), entries_.end(),
        [](const auto& item) { return item.second.checkpoint->retention.scope != 0; }));
    out.host_checkpoints = entries_.size();
    return out;
}
PrefixStateCacheStats PrefixStateCache::stats() {
    if (!policy_.legacy || next_scope_) return counters();
    std::lock_guard lock(mutex_);
    auto out = stats_;
    for(std::size_t i=0;i<certificate_declines_.size();++i)
        out.certificate_declines[i]=certificate_declines_[i].load(std::memory_order_relaxed);
    out.host_bytes = host_bytes_locked();
    out.host_checkpoints = entries_.size();
    out.host_segments = segments_.size();
    return out;
}


bool PrefixStateCache::admit_session_working_bytes(std::uint64_t bytes) {
    std::lock_guard lock(mutex_);
    if (bytes > policy_.host_byte_budget) return false;
    evict_to_budget_locked(bytes);
    return retained_bytes_locked() <= policy_.host_byte_budget - bytes;
}
bool PrefixStateCache::park_session(const std::string& id, PrefixCacheBuffer envelope, std::uint64_t max_sessions) {
    require(!id.empty() && id.size()<=128 && std::all_of(id.begin(),id.end(),[](unsigned char c) {
        return (c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9') || c=='-' || c=='_';
    }),"invalid session id");
    require(envelope.data && envelope.bytes && max_sessions,"session parking requires a budget and envelope");
    {
        std::lock_guard receipt_lock(mutex_);
        auto r = receipts_[envelope.data.get()].lock();
        if (!r) {
            r=std::make_shared<Receipt>(); r->ledger=ledger_; r->payload=envelope.data; r->bytes=envelope.bytes;
            receipts_[envelope.data.get()]=r;
        }
        envelope.data=std::shared_ptr<std::byte>(r,envelope.data.get());
    }
    const auto digest=sha256_bytes_fast(envelope.span());
    auto checkpoint=std::make_shared<PrefixCacheCheckpoint>();
    checkpoint->conversation=id;
    checkpoint->digest=digest; checkpoint->block_rows=1; checkpoint->fixed=std::move(envelope);
    std::lock_guard lock(mutex_);
    named_session_parking_ = true;
    for(auto it=named_sessions_.begin();it!=named_sessions_.end();) {
        if(!entries_.count(it->second)) it=named_sessions_.erase(it); else ++it;
    }
    const auto need=checkpoint_host_bytes(*checkpoint);
    if(need>policy_.host_byte_budget) { ++stats_.rejected_over_budget; return false; }
    if(auto old=named_sessions_.find(id);old!=named_sessions_.end()) {
        if(auto e=entries_.find(old->second);e!=entries_.end()) { erase_locked(e->first); }
        named_sessions_.erase(old);
    }
    // Different names must not alias the same mutable lifecycle. Bind name into storage key.
    std::vector<std::byte> key;
    for(const auto c:id) key.push_back(static_cast<std::byte>(c));
    for(auto c:digest.bytes) key.push_back(static_cast<std::byte>(c));
    checkpoint->digest=sha256_bytes(key);
    while(named_sessions_.size()>=max_sessions) {
        auto victim=lru_.begin();
        while(victim!=lru_.end() && std::none_of(named_sessions_.begin(),named_sessions_.end(),
            [&](const auto& n){return n.second==*victim;})) ++victim;
        require(victim!=lru_.end(),"session LRU invariant failed");
        const auto d=*victim; erase_locked(d);
        for(auto it=named_sessions_.begin();it!=named_sessions_.end();) {
            if(it->second==d) it=named_sessions_.erase(it); else ++it;
        }
        ++stats_.evicted;
    }
    evict_to_budget_locked(need);
    for(auto it=named_sessions_.begin();it!=named_sessions_.end();) {
        if(!entries_.count(it->second)) it=named_sessions_.erase(it); else ++it;
    }
    // Outstanding handles can retain pinned segments: do not claim admission if eviction did not free enough.
    if(retained_bytes_locked()>policy_.host_byte_budget-need) { ++stats_.rejected_over_budget; return false; }
    add_locked(checkpoint);
    session_buffers_[checkpoint->fixed.data.get()] = {checkpoint->fixed.data, checkpoint->fixed.bytes};
    named_sessions_[id]=checkpoint->digest; ++stats_.inserted; return true;
}
PrefixCacheBuffer PrefixStateCache::resume_session(const std::string& id) {
    std::lock_guard lock(mutex_);
    const auto n=named_sessions_.find(id); if(n==named_sessions_.end()) return {};
    const auto e=entries_.find(n->second); if(e==entries_.end()) { named_sessions_.erase(n); return {}; }
    touch_locked(e->second,true); return e->second.checkpoint->fixed;
}
std::vector<std::string> PrefixStateCache::list_sessions() {
    std::lock_guard lock(mutex_); std::vector<std::string> out;
    for(auto it=named_sessions_.begin();it!=named_sessions_.end();) {
        if(!entries_.count(it->second)) it=named_sessions_.erase(it);
        else { out.push_back(it->first); ++it; }
    }
    return out;
}
bool PrefixStateCache::delete_session(const std::string& id) {
    std::lock_guard lock(mutex_); const auto n=named_sessions_.find(id); if(n==named_sessions_.end()) return false;
    if(auto e=entries_.find(n->second);e!=entries_.end()) { erase_locked(e->first); }
    named_sessions_.erase(n); ++stats_.evictions_explicit; return true;
}

}  // namespace strix

namespace strix {
void PrefixStateCache::record_staging(std::uint64_t cap,std::uint64_t bytes,bool active) {
    std::lock_guard lock(mutex_);
    if(cap) stats_.staging_budget_bytes=cap;
    stats_.staging_bytes=active?bytes:0;
    stats_.staging_peak_bytes=std::max(stats_.staging_peak_bytes,bytes);
}
void PrefixStateCache::record_staging_skip(bool cap) {
    std::lock_guard lock(mutex_);
    ++stats_.budget_skips;
    if(cap) ++stats_.staging_cap_skips; else ++stats_.capture_busy_skips;
}
}

namespace strix {
void PrefixStateCache::record_row_prefetch() {std::lock_guard lock(mutex_); ++stats_.row_prefetches;}
void PrefixStateCache::record_certificate_decline(PrefixCertificateDecline reason) noexcept {
    certificate_declines_[static_cast<std::size_t>(reason)].fetch_add(1,std::memory_order_relaxed);
}
void PrefixStateCache::record_snapshot_preparation() {std::lock_guard lock(mutex_); ++stats_.snapshot_preparations;}
void PrefixStateCache::record_tripwire_skip() {std::lock_guard lock(mutex_); ++stats_.capture_skips_tripwire;}
void PrefixStateCache::record_batch_config_skip() {std::lock_guard lock(mutex_); ++stats_.capture_skips_batch_config;}
void PrefixStateCache::record_cross_stream_skip() {std::lock_guard lock(mutex_); ++stats_.capture_skips_cross_stream;}
}
