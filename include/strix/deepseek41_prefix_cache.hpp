#pragma once

#include "strix/deepseek41_attention_state.hpp"
#include "strix/prefix_state_cache.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// Prefix cache of deepseek41 attention states (strix/ds41-attention-state-v1) for prompts that share a beginning:
// multi-turn chat (each turn extends the previous prompt), a common system prompt, RAG documents.
//
// A prompt is cut into blocks of `block` tokens (a multiple of 8, the prefill's window, and of every compressor ratio,
// so the compressed entries of a block are its own and the ratio-2 carry is empty at a boundary). Block k of a prompt
// is a node keyed by the hash chain key_k = FNV(key_{k-1}, tokens of block k) and holds what the block adds to the
// state: its tokens (compared on lookup, so a hash collision is a miss, never a wrong hit) and, per KV-source layer,
// its compressed KV and index-key entries. Those never change once written, so blocks are shared by every prompt with
// the same beginning. A node may also hold a snapshot: the per-sequence pieces that are not append-only (every layer's
// window ring, the ratio-2 carries) plus the greedy token after the boundary and the FNV of its logits. A snapshot at
// boundary p, the KV pieces of blocks 0..p/block - 1 and their tokens are exactly the AttentionState a prefill of those
// p tokens exports (Engram history = the tokens; the last row's top-k is left out: neither a prefill extend nor a decode
// step reads it, both select again), so the device prefill resumes from it (DevicePrefill::import_state + extend) and
// gives every row, the logits and the final state of a fresh prefill bitwise.
//
// Snapshots are taken where a prefill run ends (a run split costs the decoder cone of the exact suffix replay once
// more, see ds41_prefill.hpp): plan_prefill_pieces cuts a prompt at the resume point, at the deepest block shared with
// cached prompts (a branch point, e.g. the end of a common system prompt; once), at max_rows and at the prompt's last
// full block (optionally also one block before it, so an edited last block still resumes close to the end); every
// piece end on a block boundary is inserted. A decode that reaches a block boundary can insert its state too (a
// session snapshot, BatchForward::export_state).
//
// Memory: about 890 B per cached token (KV + index keys) + 4 B (token) and 2.9 MB per snapshot (40 rings of 136 x
// 528 B). The budget counts these bytes; on insert the least recently used snapshots go first (and the blocks only
// they needed), never anything a live request holds (the cache holds copies on the host, apart from the KV pool).
//
// Disk tier (options.dir non-empty): every inserted block and snapshot is also written (write-through, by a writer
// thread; serialization and enqueueing run on the caller) to one file each in `dir`: a 128-byte header (magic, version, the model identity,
// the layer layout hash, block size, chain key, parent key, depth, insert sequence, payload size and checksum, header
// checksum), then the payload (the layer layout table; a block: its tokens and per KV-source layer its KV and index-key
// entries; a snapshot: next token, logits FNV and per layer the v1 ring bytes and carry floats). Files are written to
// a temporary name, synced, renamed and the directory synced. The disk tier has its own budget and LRU (same logical
// clock) and is self-contained like the RAM tier (a disk snapshot keeps its blocks on disk). A lookup walks the chain
// over both tiers; a hit whose pieces are (partly) on disk reads those files (O_DIRECT where available, several
// readers, checksums verified) and is used exactly like a RAM hit (then promoted into RAM). Opening a cache on an
// existing directory rebuilds the disk index from the files: other model or layout, bad checksum, truncated, orphan
// (parent missing) -> ignored and deleted; temporaries deleted.
//
// Determinism for two ranks: lookup / state / insert / eviction depend only on the calls and the state bytes (the
// attention state is replicated on both ranks), with a logical clock, so two ranks making the same calls hold the
// same cache. Serving lets rank 0 decide (hit position and key go to rank 1 in the control block) and rank 1 checks.
// The disk tier keeps this: index updates happen at the call (not when a write completes); after a restart the ranks
// exchange inventory() and keep the common entries (retain); a read that fails on either rank makes both forget() the
// path and run the prompt as a miss (load() does not change the cache; use() does, after both ranks agree).
namespace strix::deepseek41 {

#ifdef STRIX_DS41_PREFIX_IO_TESTING
namespace prefix_io_test {
// Test closure only: copied at Writer construction, called before/after each real
// job on that writer thread. Ordinary builds contain no hook or extra state.
using WriterHook = void (*)(void*, bool before, const std::filesystem::path&, bool write) noexcept;
void set_writer_hook(WriterHook, void*) noexcept;
}
#endif

struct PrefixCacheOptions {
    std::uint32_t block = 256;            // tokens per block (multiple of 8 and of the compressor ratios)
    std::uint64_t budget_bytes = 0;       // RAM tier; 0: no limit
    std::filesystem::path dir;            // disk tier: directory of this rank's files (created); empty: no disk tier
    std::uint64_t disk_budget_bytes = 0;  // 0: no limit
    std::array<std::uint8_t, 32> model_id{};  // identity of the model (and layer count) the states belong to
    bool promote = true;                  // a hit read from disk is inserted into the RAM tier
    bool drop_page_cache = true;          // written files leave the page cache; reads use O_DIRECT
    std::filesystem::path quota_dir;      // explicit service root shared by its namespaces; empty: legacy per-tier quota
    std::uint64_t aggregate_disk_budget_bytes = 4096ULL << 20U;
    std::uint64_t queue_budget_bytes = 128ULL << 20U;  // includes the active writer's payload; 0: no limit
    std::uint64_t min_free_bytes = 0;      // filesystem floor after predicted writes (serve: 16 GiB)
    std::function<bool(bool)> disk_admission;  // optional both-rank agreement on local resource admission
    std::uint32_t io_threads = 4;         // parallel file reads (load, open)
};

struct PrefixCacheStats {
    std::uint64_t bytes = 0, blocks = 0, snapshots = 0;
    std::uint64_t lookups = 0, hits = 0, hit_tokens = 0, inserts = 0, evicted_snapshots = 0, evicted_blocks = 0, refused = 0;
    std::uint64_t capture_checks = 0, capture_bypasses = 0, capture_predicted_peak = 0, host_ceiling_bytes = 0;
    std::uint64_t export_bytes = 0, export_inrun = 0, export_run = 0, export_handoff = 0, export_session = 0;
    std::uint64_t row_layers = 0, planned_splits = 0;
    // disk tier
    std::uint64_t disk_bytes = 0, disk_blocks = 0, disk_snapshots = 0;
    std::uint64_t disk_hits = 0, disk_evicted_snapshots = 0, disk_evicted_blocks = 0, disk_refused = 0, disk_dropped = 0;
    std::uint64_t scanned_files = 0, ignored_files = 0;  // at open: files read / ignored (corrupt, other model, orphan)
    std::uint64_t written_files = 0, written_bytes = 0, write_errors = 0;
    std::uint64_t queued_bytes = 0, queue_peak_bytes = 0, disk_resource_refused = 0;
    std::uint64_t read_files = 0, read_bytes = 0, read_errors = 0;
    double write_seconds = 0.0, read_seconds = 0.0, scan_seconds = 0.0;
};

// The next token of a snapshot taken inside a prefill run (DevicePrefill::export_state_at): unknown, so lookup never
// takes such a snapshot as a whole-prompt hit (it resumes from it with rows left to prefill).
inline constexpr std::uint32_t kPrefixUnknownNextToken = 0xFFFFFFFFU;

struct PrefixLookup {
    std::uint32_t resume = 0;   // longest snapshot boundary (multiple of block, <= tokens.size()); 0: none
    std::uint64_t key = 0;      // its node's chain key (0 when resume is 0)
    std::uint32_t shared = 0;   // deepest block boundary whose blocks are cached (>= resume): a branch point
    std::uint32_t next_token = 0;  // the snapshot's greedy token after `resume`
    std::uint64_t logits_fnv = 0;  // FNV of the logits at resume - 1 (0: unknown, a session snapshot)
    bool disk = false;             // the state at `resume` needs files of the disk tier
    std::uint64_t disk_bytes = 0;  // their size
};

// A disk-tier entry as the ranks compare them after a restart (kind 1 block, 2 snapshot; digest = payload checksum).
struct PrefixDiskItem {
    std::uint64_t key = 0;
    std::uint32_t kind = 0, depth = 0;
    std::uint64_t digest = 0;
    friend bool operator==(const PrefixDiskItem&, const PrefixDiskItem&) = default;
};

// A file of the disk tier could not be read or did not verify.
class PrefixCacheReadError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct PrefixLoadInfo {
    std::uint32_t disk_files = 0;
    std::uint64_t disk_bytes = 0;
    double disk_seconds = 0.0, assemble_seconds = 0.0;
};

class PrefixCache {
public:
    explicit PrefixCache(const PrefixCacheOptions& options, std::shared_ptr<PrefixStateCache> owner = {});
    ~PrefixCache();
    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    [[nodiscard]] std::uint32_t block() const noexcept { return options_.block; }
    [[nodiscard]] const PrefixCacheOptions& options() const noexcept { return options_; }
    [[nodiscard]] bool has_disk() const noexcept { return !options_.dir.empty(); }
    // The longest resumable prefix of `tokens` (counts a lookup, and a hit when resume > 0; no LRU change). With
    // min_rows > 1 a snapshot that would leave a prefill of fewer than min_rows rows is passed over (resume is the whole
    // prompt or leaves at least min_rows): the two-rank row-split prefill needs every run to give each rank a row.
    [[nodiscard]] PrefixLookup lookup(std::span<const std::uint32_t> tokens, std::uint32_t min_rows = 1);
    // The state at boundary `position` of `tokens` (a lookup's resume), assembled from RAM and disk; the cache does not
    // change. Throws std::runtime_error when the cache does not hold it (two ranks out of step) and PrefixCacheReadError
    // when a disk file fails (missing, short, checksum, other content). topk empty and unknown; next_token from the
    // snapshot.
    [[nodiscard]] AttentionState load(std::span<const std::uint32_t> tokens, std::uint32_t position, std::uint64_t key);
    // After a load(): the snapshot made most recently used; pieces read from disk inserted into RAM (options.promote).
    void use(const AttentionState& loaded, std::uint64_t key);
    // load() + use().
    [[nodiscard]] AttentionState state(std::span<const std::uint32_t> tokens, std::uint32_t position, std::uint64_t key);
    // Drops the disk entries of the blocks of tokens[0, position) (and their snapshots, and what hangs below them):
    // after a failed read, on both ranks.
    void forget(std::span<const std::uint32_t> tokens, std::uint32_t position);
    // Inserts what `s` (an exported state at a block boundary, s.tokens its prompt, s.next_token its greedy token) adds:
    // the blocks the cache lacks and the snapshot at s.position (an existing one is only made most recently used), in
    // RAM and (write-through) on disk. kv_from (a block boundary): s's KV and index-key vectors hold only the entries
    // of positions [kv_from, position) (a session snapshot exports only the new ones); a tier that lacks a block below
    // kv_from is not inserted. Evicts least recently used snapshots to stay within each budget; false when neither tier
    // took it (it cannot fit, or the position is not a block boundary).
    bool insert(const AttentionState& s, std::uint64_t logits_fnv, std::uint32_t kv_from = 0);
    // Deepest block boundary <= position whose blocks of tokens are held by every tier (a session snapshot's kv_from).
    [[nodiscard]] std::uint32_t covered(std::span<const std::uint32_t> tokens, std::uint32_t position) const;
    void clear();      // both tiers (the files are deleted)
    void clear_ram();  // the RAM tier only (a following hit reads the disk)
    void flush();      // waits for the pending file writes and deletions
    [[nodiscard]] PrefixCacheStats stats() const;
    [[nodiscard]] const PrefixLoadInfo& last_load() const noexcept { return last_load_; }
    // Disk tier inventory (sorted) and the intersection with a peer's: entries the peer lacks (or holds with other
    // bytes) are dropped. Returns the number of dropped items.
    [[nodiscard]] std::vector<PrefixDiskItem> inventory() const;
    std::uint64_t retain(std::span<const PrefixDiskItem> peer);
    // Bytes the cache would take for one snapshot at `position` of a prompt sharing nothing (the budget unit).
    [[nodiscard]] static std::uint64_t snapshot_bytes(const AttentionState& s, std::uint32_t block = 256);

    // Complete generated conversations use the same attention-state adapter, in a separate versioned namespace.
    class Sessions;
    struct DiskEntry;
    struct Writer;

private:
    PrefixCacheOptions options_;
    std::shared_ptr<PrefixStateCache> owner_;
    std::uint64_t scope_ = 0;
    std::unordered_map<std::uint64_t, std::unique_ptr<DiskEntry>> disk_;
    std::unique_ptr<Writer> writer_;
    std::vector<AttentionLayerState> layout_;  // roles of each layer (owns_kv, owns_index, ratio), from the first insert
    std::uint64_t clock_ = 0; // durable sequence only; RAM ordering belongs to owner_
    PrefixCacheStats stats_;
    PrefixLoadInfo last_load_;
    void set_layout(const AttentionState& s);
    [[nodiscard]] ModelContentIdentity ram_id(std::uint64_t key) const;
    [[nodiscard]] std::shared_ptr<const PrefixCacheSegment> ram_node(std::uint64_t key, std::uint32_t depth, std::span<const std::uint32_t> block_tokens) const;
    [[nodiscard]] std::shared_ptr<const PrefixCacheCheckpoint> ram_snapshot(std::uint64_t key, bool touch = false) const;
    [[nodiscard]] const std::uint8_t* ram_layer(const PrefixCacheSegment&, std::size_t layer, bool index) const;
    [[nodiscard]] const DiskEntry* disk_entry(std::uint64_t key, std::uint64_t parent, std::uint32_t depth, std::span<const std::uint32_t> block_tokens) const;
    bool insert_ram(const AttentionState& s, std::uint64_t logits_fnv, std::uint32_t kv_from, const std::vector<std::uint64_t>& keys);
    bool insert_disk(const AttentionState& s, std::uint64_t logits_fnv, std::uint32_t kv_from, const std::vector<std::uint64_t>& keys);
    void evict_disk_one();
    void prune_disk(std::uint64_t key);
    void remove_disk(std::uint64_t key);       // the entry and its files
    void drop_disk_snapshot(DiskEntry& e);
    void sweep_disk();                         // orphans and blocks no snapshot needs
    void open_disk();
    [[nodiscard]] std::filesystem::path block_path(std::uint64_t key) const;
    [[nodiscard]] std::filesystem::path snapshot_path(std::uint64_t key) const;
};

// Hash chain of blocks: key of the first `blocks` blocks of `tokens` (0 for none).
[[nodiscard]] std::uint64_t prefix_key(std::span<const std::uint32_t> tokens, std::uint32_t block, std::uint32_t blocks);

// The piece ends of a prefill of `total` tokens that resumes at `resume`: the deepest shared block boundary (when it
// lies between resume and the last full block), every max_rows rows (rounded down to a block boundary when possible,
// else to 8), the last full block boundary that leaves at least min_rows rows after it (with before_tail also the
// boundary one block before it), and total. Strictly increasing, all > resume, the last = total, every piece at least
// min_rows rows when total - resume >= min_rows (a boundary that would make a shorter piece is dropped); empty when
// resume == total.
[[nodiscard]] std::vector<std::uint32_t> plan_prefill_pieces(std::uint32_t resume, std::uint32_t shared, std::uint32_t total, std::uint32_t block,
                                                             std::uint32_t max_rows, std::uint32_t min_rows = 1, bool before_tail = false);

// With in-run snapshots (DevicePrefill::set_snapshot_points): the runs end only every max_rows rows (as
// plan_prefill_pieces splits them) and at total; the boundaries plan_prefill_pieces adds for the cache (the shared
// block, the last full block, with before_tail one block before it) become snapshot points inside those runs instead
// (ascending, each strictly inside a run). The same snapshots, without the extra runs. reach (0: any): a point more than
// reach rows before its run's end splits the run there instead (an in-run point widens the decoder cone over every row
// between it and the end, a split costs one more read of the experts), so only the points near the end stay inside.
// extra (0: none): one more cache boundary (rounded down to a block), e.g. the end of a chat's system prompt + tools.
struct PrefillRunPlan {
    std::vector<std::uint32_t> ends, points;
};
[[nodiscard]] PrefillRunPlan plan_prefill_runs(std::uint32_t resume, std::uint32_t shared, std::uint32_t total, std::uint32_t block,
                                               std::uint32_t max_rows, std::uint32_t min_rows = 1, bool before_tail = false,
                                               std::uint32_t reach = 0, std::uint32_t extra = 0);

// Admission is part of planning: a declined capture has exactly the baseline endpoints and no points.
[[nodiscard]] PrefillRunPlan select_prefix_prefill_plan(std::uint32_t resume, std::uint32_t shared, std::uint32_t total, std::uint32_t block,
                                                       std::uint32_t max_rows, std::uint32_t min_rows, bool capture, bool inrun,
                                                       bool before_tail, std::uint32_t reach, std::uint32_t stable);

// The bit-relevant configuration whose states a cache keeps: everything besides the model file and the layer count that
// changes a state's bits. `text` is a canonical "key=value;" list (version, weights overlay + CB4, trellis pack, sites and
// mix, shared-expert TP, EP map, the prefill identity and the decode identity); `tag` its 16 hex digits. States of
// another namespace are never reused: give every namespace its own disk directory (<dir>/<tag>) and fold `text` into the
// model id, so a file written under another configuration, or by code without a namespace, fails the header check
// and is ignored. Files are identified by size and the SHA-256 of their first and last 16 MiB (of every file of a pack
// directory: size and first 1 MiB), as the forwards read them.
struct PrefixNamespace {
    std::string text, tag;
};
// From the environment as the forwards read it (STRIX_DS41_OVERLAY, STRIX_DS41_CB4, STRIX_DS41_TRELLIS,
// STRIX_DS41_TRELLIS_SITES, STRIX_DS41_TRELLIS_MIX, STRIX_DS41_SHEXP_TP, STRIX_DS41_EP_MAP). expert_tp_decode: the decode
// forward's identity (BatchForward::expert_tp()); the device prefill is expert parallel.
[[nodiscard]] PrefixNamespace prefix_namespace_from_env(bool expert_tp_decode);
// Identity of a model file and configuration for the disk tier: SHA-256 of the file size, its first 16 MiB (header,
// metadata and tensor table), the layer count and the namespace text (hashing the whole file would take minutes).
[[nodiscard]] std::array<std::uint8_t, 32> prefix_model_id(const std::filesystem::path& model, std::uint32_t layers, const PrefixNamespace& ns);

}  // namespace strix::deepseek41
