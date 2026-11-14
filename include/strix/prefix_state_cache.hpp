#pragma once

#include "strix/model_runtime_binding.hpp"

#include <stdexcept>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <utility>

namespace strix {

namespace session { struct PrefixSnapshot; struct ResidentPrefix; }

// A quota is part of this store's one service budget, never another allocator.
// The sole-oversize exception is for legacy prompt snapshots only.
struct PrefixEntryPolicy {
    std::uint64_t scope = 0, max_bytes = 0, max_entries = 0;
    bool fifo = false, sole_oversize = false;
};

// Block-boundary prefix state cache (see docs/engine/serving/prefix-state-cache.md).
//
// A checkpoint is the complete session state after the first P = k*T prompt tokens of a tiled
// prefill (T = --known-prefix-prefill-rows), split into
//   - k row segments, one per T-row block: the per-position state rows of that block (KV and raw
//     index keys of every attention layer). Rows never change after they are written, so the
//     checkpoints of one prompt share the segments of their common blocks;
//   - one fixed blob: the state that is overwritten in place (recurrent matrices, convolution
//     histories, n-gram and position history);
//   - opaque metadata (session counters and arithmetic lineage).
// The store does not interpret any payload. It owns keys, token verification, the host byte
// budget with selectable retention and the optional disk tier.
//
// Keys. identity = SHA-256 of everything that changes the state a prefix produces (model content,
// state layout, runtime compatibility, prefill and decode executions and kernel plans, numerical
// profile pair, T, ...; built by the caller). digest_k = SHA-256(digest_{k-1} || tokens of block
// k as little-endian uint32), digest_0 = identity. A digest therefore names one identity and one
// exact token prefix; the tokens are stored and compared as well.
struct PrefixCacheBuffer {
    std::shared_ptr<std::byte> data;
    std::uint64_t bytes = 0;
    [[nodiscard]] std::span<std::byte> span() const noexcept { return {data.get(), static_cast<std::size_t>(bytes)}; }
};
// Allocates payload storage (pinned host memory on the GPU path). Must return bytes >= requested.
using PrefixCacheAllocator = std::function<PrefixCacheBuffer(std::uint64_t bytes)>;

struct PrefixCacheSegment {
    ModelContentIdentity identity, digest, parent;
    std::uint64_t block_index = 0;  // 0-based: rows [block_index*T, (block_index+1)*T)
    std::uint32_t block_rows = 0;
    std::vector<std::uint32_t> tokens;  // the T tokens of this block
    PrefixCacheBuffer payload;
    // Scoped chain adopters retain ancestry once, rather than a full prefix
    // vector per snapshot. Zero charge/ancestor preserve the tiled producer.
    std::shared_ptr<const PrefixCacheSegment> ancestor;
    std::uint64_t charge = 0;
    // The original segment released payload first. Keep that order with chain
    // ancestry too: snapshot, leaf payload, then unneeded ancestor payloads.
    ~PrefixCacheSegment() { payload = {}; }
};

struct PrefixCacheCheckpoint {
    ModelContentIdentity identity, digest;
    std::vector<std::uint64_t> producer_certificate; // nonempty: certified producer family
    std::string conversation; // retention only; never serialized or part of numerical identity
    std::uint64_t position = 0;  // P = segments.size() * block_rows
    std::uint32_t block_rows = 0;
    std::vector<std::shared_ptr<const PrefixCacheSegment>> segments;
    PrefixCacheBuffer fixed;
    std::vector<std::byte> metadata;
    // RAM-only extensions. The legacy disk codec never writes these fields.
    PrefixEntryPolicy retention;
    std::uint64_t charge = 0, declared_metadata = 0;
    bool legacy_index_exemption = false;
    bool chain_retention = false;
    std::string name, replacement, keep;
    std::shared_ptr<const session::PrefixSnapshot> snapshot;
    std::shared_ptr<const session::ResidentPrefix> resident;
    std::shared_ptr<const std::vector<std::uint8_t>> encoded;
    // Copy of every token of the prefix, checked on lookup.
    [[nodiscard]] std::vector<std::uint32_t> tokens() const;
};

[[nodiscard]] ModelContentIdentity prefix_cache_block_digest(const ModelContentIdentity& previous,
    std::span<const std::uint32_t> block_tokens);
// digests[k-1] names the first k blocks; at most max_blocks digests.
[[nodiscard]] std::vector<ModelContentIdentity> prefix_cache_block_digests(const ModelContentIdentity& identity,
    std::span<const std::uint32_t> tokens, std::uint32_t block_rows, std::uint64_t max_blocks);
// The part of a longest-prefix lookup that depends only on the prompt (PrefixStateCache::prepare_lookup).
// It holds no cache state and no lease: a hit is decided, verified against the stored tokens and accounted
// by find_longest(ticket). Copies share the token storage.
struct PrefixLookupTicket {
    enum Disk : std::uint8_t { disk_unknown = 0, disk_absent = 1, disk_legacy = 2, disk_certified = 3 };
    ModelContentIdentity identity;
    std::uint32_t block_rows = 0;
    std::uint64_t lookup_tokens = 0;                         // min(limit, prompt tokens)
    std::shared_ptr<const std::vector<std::uint32_t>> owned; // the prompt, when the ticket outlives the caller's span
    std::span<const std::uint32_t> tokens;                   // the prompt the digests were computed from
    std::vector<ModelContentIdentity> digests;               // digests[k-1] names the first k blocks
    std::vector<std::uint8_t> disk;                          // Disk class per block; empty: not probed
    std::uint64_t disk_epoch = 0;
};
// Cheap 64-bit payload checksum used by the disk tier (not a security digest).
[[nodiscard]] std::uint64_t prefix_cache_checksum(std::span<const std::byte> bytes) noexcept;

enum class PrefixCacheRetention { fair, lru };

struct PrefixStateCachePolicy {
    std::uint64_t host_byte_budget = 0;
    std::filesystem::path disk_directory;  // empty: no disk tier
    std::uint64_t disk_byte_budget = 0;    // 0 with a directory: unbounded
    PrefixCacheRetention retention = PrefixCacheRetention::fair;
    bool legacy = false; // Baseline preserves the original explicit-cache path.
    bool certified_disk = true; // Certificate envelope in existing v1 checkpoint metadata.
    bool durable = true; // Sync files and directory before publishing a checkpoint.
    bool fair_reclaim_idle_groups = true; // Unreused certified groups with a private-prefix successor; false = old fair.
};

// Allocators may mark non-resource failures that opportunistic capture must propagate.
class PrefixCacheFatalError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class PrefixCacheAllocationFailure : public std::runtime_error {
public:
    PrefixCacheAllocationFailure() : std::runtime_error("prefix state cache pinned host allocation failed") {}
};

// One attach-time backing allocation. Leases return aligned ranges, never pinned pages.
class PrefixPayloadPool : public std::enable_shared_from_this<PrefixPayloadPool> {
    PrefixCacheBuffer backing_;
    std::uint64_t alignment_ = 256;
    std::mutex mutex_;
    using FreeRanges = std::map<std::uint64_t,std::uint64_t>;
    FreeRanges free_;
    void release(std::uint64_t at, std::uint64_t bytes, FreeRanges::node_type node) noexcept {
        std::lock_guard lock(mutex_);
        auto next=free_.lower_bound(at);
        if(next!=free_.begin()) {
            auto prev=std::prev(next);
            if(prev->first+prev->second==at) {at=prev->first;bytes+=prev->second;free_.erase(prev);}
        }
        if(next!=free_.end() && at+bytes==next->first) {bytes+=next->second;free_.erase(next);}
        node.key()=at; node.mapped()=bytes;
        free_.insert(std::move(node)); // granted leases already own this node; no allocation
    }
public:
    explicit PrefixPayloadPool(PrefixCacheBuffer backing, std::uint64_t alignment = 256)
        : backing_(std::move(backing)), alignment_(alignment) {
        if (!alignment || (alignment & (alignment - 1))) throw std::invalid_argument("prefix pool alignment");
        free_.emplace(0,backing_.bytes);
    }
    std::uint64_t bytes() const {return backing_.bytes;}
    std::uint64_t free_bytes() {
        std::lock_guard lock(mutex_);
        std::uint64_t total=0;
        for(const auto& [at,bytes]:free_) { (void)at; total+=bytes; }
        return total;
    }
    // Simulate the same first-fit order as capture allocation, including alignment
    // and contiguous extents. This is a non-waiting admission observation, not a lease.
    bool can_allocate(std::span<const std::uint64_t> requests) {
        if(requests.empty()) return true;
        std::lock_guard lock(mutex_);
        std::vector<std::pair<std::uint64_t,std::uint64_t>> ranges(free_.begin(),free_.end());
        for(const auto bytes:requests) {
            if(!bytes) continue;
            if(bytes>UINT64_MAX-(alignment_-1)) return false;
            const auto aligned=(bytes+alignment_-1)&~(alignment_-1);
            bool found=false;
            for(auto& range:ranges) if(range.second>=aligned) {
                range.first+=aligned; range.second-=aligned; found=true; break;
            }
            if(!found) return false;
        }
        return true;
    }
    PrefixCacheBuffer allocate(std::uint64_t bytes) {
        if(!bytes) return {};
        if(bytes>UINT64_MAX-(alignment_-1)) throw PrefixCacheAllocationFailure();
        const auto aligned=(bytes+alignment_-1)&~(alignment_-1);
        struct Lease {
            std::shared_ptr<PrefixPayloadPool> pool;
            std::uint64_t at=0, bytes=0;
            FreeRanges::node_type return_node;
            ~Lease() {if(bytes) pool->release(at,bytes,std::move(return_node));}
        };
        auto lease=std::make_shared<Lease>(); lease->pool=shared_from_this();
        // Reserve destructor-path control storage before modifying the free list.
        // Failure here leaves the pool untouched and no lease has been granted.
        FreeRanges spare;
        spare.emplace(0,0);
        lease->return_node=spare.extract(spare.begin());
        std::lock_guard lock(mutex_);
        for(auto it=free_.begin();it!=free_.end();++it) if(it->second>=aligned) {
            const auto at=it->first, remaining=it->second-aligned;
            auto node=free_.extract(it);
            if(remaining) {node.key()=at+aligned;node.mapped()=remaining;free_.insert(std::move(node));}
            lease->at=at;lease->bytes=aligned;
            return {std::shared_ptr<std::byte>(lease,backing_.data.get()+at),bytes};
        }
        throw PrefixCacheAllocationFailure();
    }
};

enum class PrefixCertificateDecline : std::size_t {
    serial_async_capture_unsupported,
    serial_lookup_disabled_or_short,
    serial_certificate_missing_or_incompatible,
    serial_plan_unsupported,
    serial_no_certified_checkpoint,
    serial_producer_plan_mismatch,
    serial_cursor_continuation_unsupported,
    serial_checkpoint_preflight,
    serial_capture_unit_unsupported,
    serial_capture_boundary_unsupported,
    serial_capture_metadata_mismatch,
    policy_selection_cache_unqualified,
    prefix_lookup_disabled_or_short,
    prefix_restore_disabled,
    batch_suffix_not_solo,
    prefix_certificate_missing_or_incompatible,
    prefix_producer_plan_mismatch,
    prefix_checkpoint_preflight,
    prefix_continuation_unsupported,
    prefix_no_compatible_checkpoint,
};

inline constexpr std::array<std::string_view,20> prefix_certificate_decline_names{
    "serial-async-capture-unsupported",
    "serial-lookup-disabled-or-short",
    "serial-certificate-missing-or-incompatible",
    "serial-plan-unsupported",
    "serial-no-certified-checkpoint",
    "serial-producer-plan-mismatch",
    "serial-cursor-continuation-unsupported",
    "serial-checkpoint-preflight",
    "serial-capture-unit-unsupported",
    "serial-capture-boundary-unsupported",
    "serial-capture-metadata-mismatch",
    "policy-selection-cache-unqualified",
    "prefix-lookup-disabled-or-short",
    "prefix-restore-disabled",
    "batch-suffix-not-solo",
    "prefix-certificate-missing-or-incompatible",
    "prefix-producer-plan-mismatch",
    "prefix-checkpoint-preflight",
    "prefix-continuation-unsupported",
    "prefix-no-compatible-checkpoint",
};
struct PrefixStateCacheStats {
    // Generic RAM gauges are distinct from the historical pinned-host gauges.
    std::uint64_t immutable_bytes = 0, retention_metadata_bytes = 0, reserved_bytes = 0;
    std::uint64_t encoded_host_bytes = 0, service_bytes = 0, session_entries = 0;
    std::uint64_t policy_exempt_bytes = 0;
    std::uint64_t legacy_live_over_budget_bytes = 0;
    std::uint64_t lookups = 0, plan_misses = 0, host_hits = 0, disk_hits = 0, misses = 0;
    std::uint64_t inserted = 0, evicted = 0, rejected_over_budget = 0;
    std::uint64_t host_bytes = 0, host_checkpoints = 0, host_segments = 0;
    std::uint64_t disk_writes = 0, disk_write_bytes = 0, disk_write_failures = 0, disk_evictions = 0;
    std::uint64_t disk_bytes = 0;
    std::uint64_t disk_reads = 0, disk_read_bytes = 0, disk_read_failures = 0;
    std::uint64_t lookup_tokens = 0, hit_tokens = 0, restored_tokens = 0;
    std::uint64_t captures = 0, no_store_skips = 0, budget_skips = 0, capture_failures = 0;
    std::uint64_t evictions_space = 0, evictions_conversation_cap = 0, evictions_branch_cap = 0;
    std::uint64_t evictions_explicit = 0, evictions_invalid = 0, evictions_superseded = 0;
    std::uint64_t staging_budget_bytes=0, staging_bytes=0, staging_peak_bytes=0, capture_busy_skips=0, staging_cap_skips=0, row_prefetches=0;
    std::uint64_t capture_skips_cross_stream=0;
    std::array<std::uint64_t,prefix_certificate_decline_names.size()> certificate_declines{};
    std::uint64_t capture_skips_tripwire=0,capture_skips_batch_config=0;
    std::uint64_t snapshot_preparations=0, capture_skipped_no_slot=0, pool_bytes=0, pool_budget_bytes=0, pool_metadata_reserved=0, pool_free_bytes=0, pool_staging_bytes=0;
    double disk_read_ms = 0.0, disk_write_ms = 0.0;
};

class PrefixStateCache final {
    struct Ledger;
public:
    class Reservation {
        friend class PrefixStateCache;
        std::shared_ptr<Ledger> ledger_;
        std::uint64_t bytes_ = 0;
        Reservation(std::shared_ptr<Ledger> ledger, std::uint64_t bytes) : ledger_(std::move(ledger)), bytes_(bytes) {}
    public:
        Reservation() = default;
        Reservation(const Reservation&) = delete;
        Reservation& operator=(const Reservation&) = delete;
        Reservation(Reservation&& other) noexcept : ledger_(std::move(other.ledger_)), bytes_(std::exchange(other.bytes_, 0)) {}
        Reservation& operator=(Reservation&& other) noexcept {
            if (this != &other) { Reservation old(std::move(*this)); ledger_ = std::move(other.ledger_); bytes_ = std::exchange(other.bytes_, 0); }
            return *this;
        }
        ~Reservation();
        explicit operator bool() const noexcept { return bool(ledger_); }
    };
    // Credits include outstanding reservations and live payload leases. An incoming
    // capture preserves fair directory admission; disk reads need no directory slot.
    [[nodiscard]] Reservation reserve(std::uint64_t bytes, const PrefixCacheCheckpoint* incoming = nullptr,
        std::string_view protected_name = {}, bool session_retention = false);
    [[nodiscard]] PrefixCacheBuffer allocate(std::uint64_t bytes, Reservation&, bool segment = false);

    // Session facades carry query/compatibility policy only. All retained values,
    // replacement, protection, victim ordering and physical credits live here.
    std::uint64_t new_scope();
    struct NamedView { const std::vector<std::uint8_t>& bytes; std::uint64_t touched = 0; };
    // Borrowed compatibility directory. Every core erase removes its descriptor
    // BEFORE releasing backing. Concurrent readers use query() ownership leases.
    [[nodiscard]] const std::map<std::string, NamedView>& named_view(std::uint64_t scope) const;
    using EntryPredicate = std::function<bool(const PrefixCacheCheckpoint&)>;
    using EntryRank = std::function<std::uint64_t(const PrefixCacheCheckpoint&)>;
    [[nodiscard]] std::shared_ptr<const PrefixCacheCheckpoint> query(std::uint64_t scope,
        const EntryRank&, bool touch = true);
    [[nodiscard]] std::vector<std::shared_ptr<const PrefixCacheCheckpoint>> inspect(std::uint64_t scope) const;
    bool retain(std::shared_ptr<PrefixCacheCheckpoint>, bool replace, Reservation* = nullptr);
    bool retention_room(const PrefixEntryPolicy&, std::uint64_t, std::string_view replacement = {}, std::string_view keep = {}) const;
    void remove(std::uint64_t scope, const EntryPredicate&) noexcept;
    bool unpin(std::uint64_t scope, const std::string&);
    bool reclaim(std::uint64_t scope, const std::function<bool(std::uint64_t)>&,
        const std::optional<std::string>& protected_name = {});
    [[nodiscard]] std::uint64_t retained_charge(std::uint64_t scope) const noexcept;
    [[nodiscard]] std::size_t entry_count(std::uint64_t scope) const noexcept;
    [[nodiscard]] std::uint64_t scope_evictions(std::uint64_t scope) const noexcept;
    // Adopt a codec's existing host allocation as pool backing, without copying
    // or serializing a device owner. Capacity, including codec slack, is charged.
    bool retain_encoded(std::shared_ptr<PrefixCacheCheckpoint>, std::vector<std::uint8_t>, Reservation* = nullptr);

    // Legacy block-sharing policy: one leaf segment per checkpoint, snapshot LRU
    // first, then unneeded ancestors. The codec fills new blocks root-to-leaf,
    // then fixed, using ONLY the supplied allocator (no store reentry). Each
    // request must match its declared payload bytes. Credits include metadata,
    // staging and outstanding read leases. No durable writer is used here.
    struct ChainStats {
        std::uint64_t bytes=0, blocks=0, snapshots=0, evicted_blocks=0, evicted_snapshots=0;
    };
    bool retain_chain(std::shared_ptr<PrefixCacheCheckpoint>,
        const std::function<void(PrefixCacheCheckpoint&, const PrefixCacheAllocator&)>& fill);
    [[nodiscard]] std::shared_ptr<const PrefixCacheSegment> chain_segment(std::uint64_t scope, const ModelContentIdentity&) const;
    [[nodiscard]] std::shared_ptr<const PrefixCacheCheckpoint> chain_checkpoint(std::uint64_t scope, const ModelContentIdentity&, bool touch=false);
    [[nodiscard]] ChainStats chain_stats(std::uint64_t scope) const;

    PrefixStateCache(PrefixStateCachePolicy, PrefixCacheAllocator);
    ~PrefixStateCache();
    PrefixStateCache(const PrefixStateCache&) = delete;
    PrefixStateCache& operator=(const PrefixStateCache&) = delete;

    // Longest checkpoint for this identity whose prefix is tokens[0, P) with P = k*block_rows <=
    // limit. Host first, then disk (promote=false suppresses host publication). Null: miss.
    // The returned handle keeps its payloads alive. An eligibility predicate runs outside
    // the mutex on that lease, before hit accounting. Disk candidates must carry a certificate.
    [[nodiscard]] std::shared_ptr<const PrefixCacheCheckpoint> find_longest(const ModelContentIdentity& identity,
        std::span<const std::uint32_t> tokens, std::uint32_t block_rows, std::uint64_t limit, bool promote = true, std::string_view conversation = {},
        const std::function<bool(const PrefixCacheCheckpoint&)>& eligible = {});
    // The same lookup in two steps. prepare_lookup() does the work that depends only on the prompt: it copies
    // the tokens, hashes every block and, with a disk tier, reads which blocks have a checkpoint file and of
    // which class. It takes no cache lock and changes nothing, so it can run on any thread before the request
    // is admitted. find_longest(ticket) then decides under the cache state of that moment and returns what
    // the one-step lookup returns: it probes the host tier of every block in one lock acquisition, verifies
    // the stored tokens exactly, and loads a disk checkpoint (payload read, budget credits, certificate
    // check) itself. Prepared disk metadata is used only while this cache has not published or removed a
    // checkpoint file since it was read; otherwise the lookup probes the files as the one-step form does.
    [[nodiscard]] PrefixLookupTicket prepare_lookup(const ModelContentIdentity& identity, std::span<const std::uint32_t> tokens,
        std::uint32_t block_rows, std::uint64_t limit) const;
    [[nodiscard]] std::shared_ptr<const PrefixCacheCheckpoint> find_longest(const PrefixLookupTicket& ticket, bool promote = true,
        std::string_view conversation = {}, const std::function<bool(const PrefixCacheCheckpoint&)>& eligible = {});
    [[nodiscard]] bool contains(const ModelContentIdentity& digest);
    // A retained segment with this digest (host tier only), or null.
    [[nodiscard]] std::shared_ptr<const PrefixCacheSegment> find_segment(const ModelContentIdentity& digest);
    // Inserts into the host tier under its retention policy and queues the disk write.
    // False when the checkpoint or bounded directory cannot be admitted.
    bool insert(std::shared_ptr<const PrefixCacheCheckpoint>, Reservation* = nullptr);
    [[nodiscard]] PrefixCacheBuffer allocate(std::uint64_t bytes) const;
    // Named complete-session blobs share this store's pinned allocator, byte accounting and LRU.
    // They never enter the opportunistic content-prefix namespace or the legacy disk writer.
    // Serialized caller reserves temporary capture/read bytes before allocating on unified memory.
    bool admit_session_working_bytes(std::uint64_t bytes);
    bool park_session(const std::string& id, PrefixCacheBuffer envelope, std::uint64_t max_sessions);
    [[nodiscard]] PrefixCacheBuffer resume_session(const std::string& id);
    [[nodiscard]] std::vector<std::string> list_sessions();
    bool delete_session(const std::string& id);
    // Waits for queued disk writes.
    void flush();
    [[nodiscard]] PrefixStateCacheStats stats();
    [[nodiscard]] PrefixStateCacheStats counters();
    // The logical ledger remains admission credit, not another physical allocation.
    // A bounded metadata allowance stays inside B alongside the retained pinned slab.
    void attach_payload_pool(std::shared_ptr<PrefixPayloadPool> pool, std::uint64_t metadata_allowance) {
        if(!pool || pool->bytes()>policy_.host_byte_budget ||
            metadata_allowance!=policy_.host_byte_budget-pool->bytes())
            throw std::invalid_argument("prefix pool commitments must equal B");
        payload_pool_=std::move(pool); metadata_allowance_=metadata_allowance;
        allocate_=[pool=payload_pool_](std::uint64_t n) {return pool->allocate(n);};
        stats_.pool_bytes=payload_pool_->bytes(); stats_.pool_budget_bytes=policy_.host_byte_budget;
        stats_.pool_metadata_reserved=metadata_allowance;
    }
    std::shared_ptr<void> capture_resources;
    void set_pool_staging_bytes(std::uint64_t bytes) {stats_.pool_staging_bytes=bytes;}
    void record_no_slot() {std::lock_guard lock(mutex_); ++stats_.capture_skipped_no_slot;}
    void record_no_store_skip();
    void record_capture_failure();
    void record_snapshot_preparation();
    void record_staging(std::uint64_t cap, std::uint64_t bytes, bool active);
    void record_staging_skip(bool cap);
    void record_cross_stream_skip();
    void record_certificate_decline(PrefixCertificateDecline reason) noexcept;
    void record_tripwire_skip();
    void record_batch_config_skip();
    void record_row_prefetch();
    // Advisory admission only. Use reserve() to hold credits across allocation.
    bool admit_capture(const PrefixCacheCheckpoint&);
    [[nodiscard]] const PrefixStateCachePolicy& policy() const noexcept { return policy_; }
    // Bytes a checkpoint adds to the host tier now (segments already retained are not counted).
    [[nodiscard]] std::uint64_t incremental_bytes(const PrefixCacheCheckpoint&);

private:
    struct Entry {
        std::shared_ptr<const PrefixCacheCheckpoint> checkpoint;
        std::list<ModelContentIdentity>::iterator lru;
        std::uint64_t epoch = 0, reuse = 0;
        std::set<std::string> conversations;
        bool protected_entry = false;
    };
    struct DigestLess {
        bool operator()(const ModelContentIdentity& a, const ModelContentIdentity& b) const noexcept {
            return a.bytes < b.bytes;
        }
    };
    struct Ledger {
        std::mutex mutex;
        std::uint64_t bytes = 0, segments = 0, reserved = 0, immutable = 0, retention_metadata = 0, encoded = 0, certificates = 0, exempt = 0;
        std::map<std::weak_ptr<const void>, std::uint64_t, std::owner_less<std::weak_ptr<const void>>> immutable_owners;
        std::map<std::weak_ptr<const void>, std::uint64_t, std::owner_less<std::weak_ptr<const void>>> certificate_owners;
    };
    struct RetentionLease {
        std::shared_ptr<Ledger> ledger;
        std::shared_ptr<const PrefixCacheCheckpoint> value;
        std::uint64_t metadata = 0, exempt = 0;
        ~RetentionLease();
    };
    struct Receipt {
        std::shared_ptr<Ledger> ledger;
        std::shared_ptr<std::byte> payload;
        std::shared_ptr<const std::vector<std::uint8_t>> encoded_owner;
        std::uint64_t bytes = 0;
        bool registered = false, segment = false, encoded = false;
        ~Receipt() {
            if (registered) {
                std::lock_guard lock(ledger->mutex);
                payload.reset();
                encoded_owner.reset();
                ledger->bytes -= bytes;
                if (encoded) ledger->encoded -= bytes;
                if (segment) --ledger->segments;
            }
        }
    };
    std::shared_ptr<Ledger> ledger_ = std::make_shared<Ledger>();
    static void reap_immutable_locked(Ledger&);
    std::uint64_t next_scope_ = 0;
    std::map<std::uint64_t, std::uint64_t> scope_evictions_;
    struct ChainRef { std::uint64_t scope=0, refs=0, pins=0, charge=0; };
    std::map<ModelContentIdentity, ChainRef, DigestLess> chain_refs_;
    std::map<std::uint64_t, ChainStats> chain_stats_;
    void release_chain_locked(const PrefixCacheCheckpoint&, bool eviction);
    void prune_chain_locked(const ModelContentIdentity&, bool eviction);
    std::map<std::uint64_t, std::map<std::string, NamedView>> named_views_;
    bool retention_room_locked(const PrefixEntryPolicy&, std::uint64_t, std::string_view, std::string_view) const;
    bool reclaim_one_locked(std::optional<std::uint64_t> scope = {}, std::string_view protected_name = {});
    mutable std::map<const std::byte*, std::weak_ptr<Receipt>> receipts_;
    void prune_receipts_locked() const noexcept;
    void register_buffer_locked(const PrefixCacheBuffer&, bool segment);
    // Bounded by retained associations; forget reuse when the last member leaves.
    struct GroupActivity {
        std::size_t members = 0;
        bool reused = false;
        std::uint64_t born_epoch = 0;
        std::weak_ptr<const PrefixCacheSegment> root;
        std::string_view key;
        std::list<GroupActivity*>::iterator prefix_order;
        std::size_t prefix_lcp = 0;
        bool private_successor = false;
    };
    std::map<std::string, GroupActivity> group_activity_;
    // Exact first-block order/LCPs are maintained at group insertion/removal.
    // Weak roots add no payload leases. Ambiguous multi-root groups decline.
    std::list<GroupActivity*> group_prefix_order_;
    bool group_prefix_order_valid_ = true;
    void update_group_root_locked(const std::string&, const std::shared_ptr<const PrefixCacheSegment>&);
    void mark_private_prefix_successors_locked(const PrefixCacheCheckpoint*, std::uint64_t);
    void associate_locked(Entry&, const std::string&, bool reused = false);
    void add_locked(std::shared_ptr<const PrefixCacheCheckpoint>);
    void erase_locked(const ModelContentIdentity&, bool eviction = false);
    friend struct PrefixStateCacheTestAccess;
    // Global store activity: one insertion/duplicate capture or validated touch
    // across ALL conversations, not wall time or per-conversation turns.
    std::uint64_t epoch_ = 0, metadata_bytes_ = 0;
    std::uint64_t host_bytes_locked();
    std::uint64_t retained_bytes_locked();
    void touch_locked(Entry&, bool reused = false);
    void evict_to_budget_locked(std::uint64_t reserve, const PrefixCacheCheckpoint* incoming = nullptr,
        std::span<const std::uint64_t> payloads = {}, std::uint64_t metadata = 0,
        std::string_view protected_name = {});
    std::shared_ptr<const PrefixCacheCheckpoint> load_from_disk(const ModelContentIdentity& identity,
        const ModelContentIdentity& digest, std::span<const std::uint32_t> tokens, std::uint32_t block_rows);
    std::shared_ptr<const PrefixCacheSegment> load_segment(const ModelContentIdentity& identity,
        const ModelContentIdentity& digest, std::uint32_t block_rows, Reservation* reservation);
    void write_to_disk(const PrefixCacheCheckpoint&);
    void enforce_disk_budget();
    void update_disk_gauge();
    std::uint64_t incremental_bytes_locked(const PrefixCacheCheckpoint&);
    // Deletes temporary files and segment files no checkpoint references, older than min_age.
    void sweep_disk(std::chrono::seconds min_age);
    void writer_loop();

    bool named_session_parking_ = false; // Parking stores preserve the original fair ranking.
    std::map<std::string, ModelContentIdentity> named_sessions_;
    std::map<const std::byte*, std::pair<std::weak_ptr<std::byte>, std::uint64_t>> session_buffers_;
    PrefixStateCachePolicy policy_;
    PrefixCacheAllocator allocate_;
    std::shared_ptr<PrefixPayloadPool> payload_pool_;
    std::uint64_t metadata_allowance_=0;
    // A ticket over caller-owned tokens; probe_disk=false leaves the disk tier to the lookup.
    [[nodiscard]] PrefixLookupTicket prepare_ticket(const ModelContentIdentity& identity, std::span<const std::uint32_t> tokens,
        std::uint32_t block_rows, std::uint64_t limit, bool probe_disk) const;
    // Changes whenever this cache publishes or removes checkpoint files (start and end of the change).
    std::atomic<std::uint64_t> disk_epoch_{0};
    mutable std::mutex mutex_;
    std::map<ModelContentIdentity, Entry, DigestLess> entries_;
    std::list<ModelContentIdentity> lru_;  // front: least recently used
    std::map<ModelContentIdentity, std::weak_ptr<const PrefixCacheSegment>, DigestLess> segments_;
    PrefixStateCacheStats stats_;
    // Refusals run outside lookup locks; no allocation or additional store lock.
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
    std::array<std::atomic<std::uint64_t>,prefix_certificate_decline_names.size()> certificate_declines_{};
    friend struct PrefixStateCacheTestAccess;
    // Disk writer.
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_, idle_cv_;
    std::deque<std::shared_ptr<const PrefixCacheCheckpoint>> queue_;
    bool writer_busy_ = false, stopping_ = false;
    std::thread writer_;
};

}  // namespace strix
