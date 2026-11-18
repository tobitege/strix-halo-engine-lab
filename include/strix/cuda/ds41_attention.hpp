#pragma once
// CUDA (sm_120) kernels of the deepseek41 CSA2 attention: window ring, compressor (ratio 2 with its carried group,
// ratio 1), indexer (query prep, FP4 scores, exact top-k with the candidate pool), D512 multi-query attention with
// sinks, and the attention's own F16 projections (compressor kv / gate, indexer q_b / weights). The arithmetic of each
// kernel is its host reference: ds41_attention.hpp (ds41::attn::*) and f16_rows of ds41_layer.hpp, so results are the
// host's bits (docs/models/deepseek-v4.1/cuda-attention.md). Rows of one call are consecutive positions.
//
// Two levels:
//  - kernel views (enqueue_*): device pointers only; nothing allocates or synchronizes;
//  - AttentionState: the per-layer caches of a contiguous layer range, the workspace and the RoPE tables, and
//    enqueue_layer(), which runs one layer's attention from the outputs of the Q8_0 sites (x after attn_norm, qr after
//    q_a_norm, q = attn_q_b, kv = attn_kv) to o (after the inverse RoPE, the input of attn_output_a).
#include "strix/cuda/ds41_common.hpp"
#include "strix/cuda/ds41_attention_alloc.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strix::deepseek41 {
struct AttentionState;  // strix/ds41-attention-state-v1 (deepseek41_attention_state.hpp)
struct AttentionFrame;  // its in-place serialization
}

namespace strix::ds41::cuda::attn {

using namespace ::strix::ds41::attn;  // shapes, formats, RopeRow, Pool, the host references

class ExportCapture;  // capture_state

// ---- kernel views -------------------------------------------------------------------------------------------------
// RoPE tables are indexed by position: rope[pos] = rope_row(kind, pos) (host double, rounded once); a call reads the
// rows' positions from pos[] on the device, so a captured graph replays with new positions.

// F16 x FP32 rows, declared order of f16_rows (ds41_layer.hpp): lane class c = element index mod 32 sums
// acc = acc + w * x (two roundings, no fused multiply-add) over its elements ascending, then the xor butterfly 16..1.
// Up to four jobs per launch (jobs may read different x); width % 256 == 0, 16-byte aligned weight rows.
struct F16Job {
    const std::uint16_t* weight = nullptr;  // count x width
    std::uint32_t width = 0, count = 0;
    const float* x = nullptr;               // rows x x_stride
    std::uint32_t x_stride = 0;
    float* out = nullptr;                   // rows x out_stride
    std::uint32_t out_stride = 0;
};
struct F16RowsView {
    F16Job jobs[4]{};
    std::uint32_t job_count = 0, rows = 0;
};
void enqueue_f16_rows(const F16RowsView& v, cudaStream_t stream);

struct WindowView {
    const float* kv;              // R x 512, attn_kv (before kv_norm)
    const float* norm;            // 512, attn_kv_a_norm
    const RopeRow* rope;          // table of the layer's kind, by position
    const std::uint64_t* pos;     // R
    std::uint32_t rows, capacity; // capacity >= 128 + rows, slot = pos % capacity
    std::uint8_t* ring;           // capacity x 528
};
void enqueue_window_write(const WindowView& v, cudaStream_t stream);

struct CompressView {
    std::uint32_t ratio, rows;    // ratio 2 or 1
    const float* kv;              // R x 512, compressor kv
    const float* score;           // R x 512, compressor gate (ratio 2)
    const float* norm;            // 512
    const std::uint16_t* wk;      // 128 x 512 F16, indexer attn_k
    const float* knorm;           // 128
    const RopeRow* rope;          // compressed table, by position (read at pos / ratio * ratio)
    const std::uint64_t* pos;     // R
    float* state;                 // ratio 2: 1024 floats (kv, score) of the pending group's first token, in place
    std::uint32_t* state_valid;   // ratio 2: 1 when state holds a pending token
    std::uint8_t* kv_cache;       // 288 B entries at pos / ratio
    std::uint8_t* idx_cache;      // 68 B entries at pos / ratio
    float* stash = nullptr;       // ratio 2, optional: rows x 1024 floats, every row's (kv, score) (partial commit)
};
// Writes every completed group's entries, then (ratio 2) the pending group after the last row into state. A row at an
// odd position whose group's first token is not in the call reads state (the caller keeps it valid; see the docs).
void enqueue_compress(const CompressView& v, cudaStream_t stream);
// Ratio 2: state / state_valid as a call whose last row is stash row `row` at position `pos` leaves them.
void enqueue_carry_restore(const float* stash, float* state, std::uint32_t* state_valid, std::uint64_t pos, std::uint32_t row, cudaStream_t stream);

inline constexpr unsigned kIdxPrepBytes = kIdxHeads * kIdxDim + kIdxHeads * 4 * 4 + kIdxHeads * 4;  // codes, scales, weights
struct IndexView {
    std::uint32_t rows, ratio, max_count;  // max_count >= every row's (pos + 1) / ratio
    const float* q;               // R x 32 x 128, indexer attn_q_b(qr)
    const float* w;               // R x 32, indexer proj(x)
    const RopeRow* rope;          // table of the layer's kind, by position
    const std::uint64_t* pos;     // R
    const std::uint8_t* idx_cache;
    float* scores;                // R x max_count
    void* qprep;                  // scratch R x kIdxPrepBytes
};
void enqueue_index_scores(const IndexView& v, cudaStream_t stream);

struct SelectView {
    std::uint32_t rows, ratio, max_count, topk, cand_blocks, block_size;
    Pool pool;
    const std::uint64_t* pos;     // R (count = (pos + 1) / ratio)
    const float* scores;          // R x max_count
    std::uint8_t* candidates;     // R x cand_stride: written by the source, read by users
    std::uint32_t cand_stride;    // >= ceil(max_count / block_size)
    std::uint32_t* block_keys;    // scratch R x cand_stride (candidate source)
    std::uint32_t* idx;           // R x topk, ascending
    std::uint32_t* idx_count;     // R
    std::uint32_t* survivors = nullptr;  // scratch select_survivor_words(R, max_count, topk) (rows above kSelDirect)
    bool long_context = false;          // opt in to > 512K entries (up to kSelMaxSlices slices)
};
void enqueue_select(const SelectView& v, cudaStream_t stream);

struct AttentionView {
    std::uint32_t rows, capacity, topk;
    const float* q;               // R x heads x 512, attn_q_b before RoPE
    const float* sink;            // heads
    const RopeRow* rope;          // table of the layer's kind, by position
    const std::uint64_t* pos;     // R
    const std::uint8_t* ring;
    const std::uint8_t* kv_cache; // null for a window-only layer
    const std::uint32_t* idx;     // R x topk (null for a window-only layer)
    const std::uint32_t* idx_count; // R (null for a window-only layer)
    float* partial;               // scratch R x heads x kPartialFloats
    float* o;                     // R x heads x 512, after the inverse RoPE
    std::uint32_t heads = kHeads; // the call's heads (a tensor-parallel subset), multiple of 4
};
inline constexpr unsigned kPartStride = 4 + kHeadDim;  // per chunk: m, l, 2 pad, acc[512]
inline constexpr std::size_t kPartialFloats = std::size_t{kMaxChunks} * kPartStride;  // per (row, head)
// Decode rows: 1 = one warp (two heads) per attention block instead of 2 (twice the blocks; same bits). Process-wide.
void set_attention_chunk_warps(unsigned warps);
// 4-warp chunk kernel: every tile of a chunk copied at once (4 raw buffers), and the combine loads every chunk's
// accumulator with the (m, l) pairs; the same arithmetic (same bits)
void set_attention_prefetch(bool on);
void enqueue_attention(const AttentionView& v, cudaStream_t stream);

// ---- RoPE tables ----------------------------------------------------------------------------------------------------
// Device rope_row tables of both kinds for positions [0, positions).
class RopeTables {
public:
    explicit RopeTables(std::uint32_t positions);
    ~RopeTables();
    RopeTables(const RopeTables&) = delete;
    RopeTables& operator=(const RopeTables&) = delete;
    [[nodiscard]] const RopeRow* table(RopeKind kind) const { return kind == RopeKind::window ? window_ : compressed_; }
    [[nodiscard]] std::uint32_t positions() const { return positions_; }

private:
    std::uint32_t positions_;
    RopeRow* window_ = nullptr;
    RopeRow* compressed_ = nullptr;
};

// ---- layer level ----------------------------------------------------------------------------------------------------
// Device weights of one layer's attention (the Q8_0 sites are not here).
struct LayerWeights {
    const float* kv_norm = nullptr;             // attn_kv_a_norm.weight, 512
    const float* sinks = nullptr;               // attn_sinks.weight, 64
    // KV sources
    const std::uint16_t* compressor_kv = nullptr;    // attn_compressor_kv.weight, 512 x hidden F16
    const std::uint16_t* compressor_gate = nullptr;  // attn_compressor_gate.weight (ratio 2), 512 x hidden F16
    const float* compressor_norm = nullptr;          // 512
    const std::uint16_t* index_k = nullptr;          // indexer.attn_k.weight, 128 x 512 F16
    const float* index_k_norm = nullptr;             // indexer.k_norm.weight, 128
    // index sources
    const std::uint16_t* index_q_b = nullptr;        // indexer.attn_q_b.weight, 4096 x q_lora_rank F16
    const std::uint16_t* index_proj = nullptr;       // indexer.proj.weight, 32 x hidden F16
};

// One call's rows (consecutive positions), device buffers.
struct LayerIo {
    std::uint32_t rows = 0;
    const std::uint64_t* pos = nullptr;  // R positions on the device
    std::uint64_t pos_bound = 0;         // host: >= every row's position (sizes the indexer launches)
    const float* x = nullptr;            // R x hidden, attn_norm output (compressor and indexer weights inputs)
    const float* qr = nullptr;           // R x q_lora_rank, q_a_norm output (indexer q_b input)
    const float* q = nullptr;            // R x 64 x 512, attn_q_b output
    const float* kv = nullptr;           // R x 512, attn_kv output
    float* o = nullptr;                  // R x 64 x 512, attention output after the inverse RoPE
};


// Per-layer state of layers [first, end): ring (every layer), compressed KV / index keys and the ratio-2 carry (KV
// sources), the last selection (index sources), the candidate pool (candidate source). Sources outside the range
// (a consumer whose producer runs elsewhere) get their caches here too, filled by the caller (import_*).
class AttentionState {
public:
    // plan: one entry per model layer (released_plan() or plan_from_contract()).
    AttentionState(std::vector<ds41::attn::LayerAttention> plan, const LayerAttentionOptions& options);
    ~AttentionState();
    AttentionState(const AttentionState&) = delete;
    AttentionState& operator=(const AttentionState&) = delete;

    // The layer's attention for io.rows consecutive rows: window write, compress (KV source), index scores and
    // selection (index source), attention. Every buffer of io is read on `stream` in order.
    // phase: all (default); prep = everything before the attention kernels (reads x, qr, kv, pos; never q);
    // attend = the attention kernels only (after a prep call of the same layer and io, ordered before it).
    enum class Phase { all, prep, attend };
    void enqueue_layer(std::uint32_t layer, const LayerWeights& w, const LayerIo& io, cudaStream_t stream, Phase phase = Phase::all);

    struct Caches {
        std::uint8_t* ring = nullptr;       // ring_capacity x 528
        std::uint8_t* kv_cache = nullptr;   // KV source: (max_positions / ratio + 1) x 288
        std::uint8_t* idx_cache = nullptr;  // KV source: x 68
        float* carry = nullptr;             // KV source, ratio 2: 1024 floats
        std::uint32_t* carry_valid = nullptr;
        float* carry_rows = nullptr;        // KV source, ratio 2: max_rows x 1024, the last call's rows (kv, score)
        std::uint32_t* idx = nullptr;       // index source: max_rows x 512 (the last call's selection)
        std::uint32_t* idx_count = nullptr; // index source: max_rows
        std::uint32_t row0 = 0, rows = 0;   // logical row origin and allocation size of idx/idx_count
    };
    [[nodiscard]] const Caches& caches(std::uint32_t layer) const;
    [[nodiscard]] std::uint32_t ring_capacity() const { return ring_capacity_; }
    [[nodiscard]] std::uint8_t* candidates() const { return candidates_; }  // max_rows x cand_stride
    [[nodiscard]] std::uint32_t cand_stride() const { return cand_stride_; }
    [[nodiscard]] std::uint32_t candidate_row0() const { return candidate_row0_; }
    // Logical row origin for compact decoder selections and candidates (one pass; caches remain global).
    void set_decoder_row0(std::uint32_t row0);
    [[nodiscard]] const RopeTables& rope() const { return *rope_; }
    [[nodiscard]] std::size_t device_bytes() const { return bytes_; }
    // Zeroes every cache, carry and selection (a new sequence).
    void reset(cudaStream_t stream);
    // Partial commit of the last enqueue_layer call of every layer of [first, end) (a verify window whose rows
    // [kept_row + 1, rows) are rejected): the ratio-2 carries become the one the call leaves when row kept_row (at
    // position last_pos) is its last row: that row's (kv, score) when last_pos opens a group, else none. Everything
    // else the rejected rows wrote is unreachable from the committed prefix: ring slots of positions > last_pos are
    // slots of positions <= last_pos - 136 (outside every later row's window) and are rewritten before any read;
    // compressed entries >= (last_pos + 1) / ratio are past every later row's visible count until their own row
    // rewrites them; top-k rows / candidates are per call. Enqueued on `stream`.
    void commit_rows(std::uint32_t first, std::uint32_t end, std::uint64_t last_pos, std::uint32_t kept_row, cudaStream_t stream);

    [[nodiscard]] const std::vector<ds41::attn::LayerAttention>& plan() const;
    [[nodiscard]] const LayerAttentionOptions& options() const;

    // ---- handoff in the decode format strix/ds41-attention-state-v1 (src/cuda/models/ds41/ds41_attention_export.cu) ----
    static constexpr std::uint32_t kNoLastRow = 0xFFFFFFFFU;
    // The state after positions [0, position) for layers [first, end) (inside the range; the others stay absent):
    // rings re-slotted to AttentionHost::ring_capacity (the last min(position, capacity) positions, other slots zero),
    // KV sources' kv / idx_keys entries [0, position / ratio), the ratio-2 carry (zeros when not valid), index sources'
    // top-k of row last_row of the last enqueue_layer call (kNoLastRow: topk_known = false). tokens and next_token are
    // left empty / 0 for the caller. Synchronizes `stream` (every enqueued layer must be on it or before it).
    [[nodiscard]] deepseek41::AttentionState export_state(std::uint32_t first, std::uint32_t end, std::uint32_t position,
                                                          std::uint32_t last_row, cudaStream_t stream) const;
    // Zeroes everything (reset), then loads every present layer of `state` inside the range (ring re-slotted to this
    // ring's capacity, caches, carry) and the KV / index-key caches of sources outside the range the range reads. The
    // next call continues at state.position. Throws std::invalid_argument on layer roles, sizes or a missing layer.
    // Synchronizes `stream`.
    void import_state(const deepseek41::AttentionState& state, cudaStream_t stream);
    // export_state's state captured straight into the serialized bytes (serialize_attention_state's format) in the
    // reusable buffers of `out`: one small gather + D2H of the per-layer counts (top-k counts, carry flags), then one
    // gather kernel packs every carried piece at its serialized offset into a device staging buffer and ONE D2H copies
    // it into out's pinned host buffer; the frame (header, sizes, tokens) is written on the host. Synchronizes
    // `stream`; on return the device state may be overwritten (the next prefill can run). out.seal() (any thread) then
    // adds the digest and checksum: the bytes equal serialize_attention_state(export_state(...)) with these tokens and
    // next_token.
    void capture_state(std::uint32_t first, std::uint32_t end, std::uint32_t position, std::uint32_t last_row,
                       std::span<const std::uint32_t> tokens, std::uint32_t next_token, ExportCapture& out, cudaStream_t stream) const;
    // Serialized bytes of a capture at `position` with every index source's top-k full: a bound for ExportCapture::reserve.
    [[nodiscard]] std::size_t capture_bound(std::uint32_t first, std::uint32_t end, std::uint32_t position) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::uint32_t ring_capacity_ = 0, cand_stride_ = 0;
    std::uint32_t candidate_row0_ = 0;
    std::uint8_t* candidates_ = nullptr;
    std::unique_ptr<RopeTables> rope_;
    std::size_t bytes_ = 0;
};

// Reusable buffers of AttentionState::capture_state: a pinned host buffer that holds the serialized state and a device
// staging buffer, both grown on demand (reserve() up front: no allocation, and no implicit synchronization, per capture).
// One capture at a time per object; a pipeline keeps two (one being sent while the next is captured).
class ExportCapture {
public:
    ExportCapture();
    ~ExportCapture();
    ExportCapture(const ExportCapture&) = delete;
    ExportCapture& operator=(const ExportCapture&) = delete;

    void reserve(std::size_t bytes);
    // Digest + checksum (seal_attention_frame on `threads` threads, 0: its default): after it bytes() is the serialized
    // state. sha256: also the SHA-256 of the bytes (the fast provider, strix::Sha256Fast) on one more thread, over the
    // bytes before the digest while the digest and checksum are computed, then the last 16 bytes: sha256() / sha_ms.
    // CPU only, any thread, the device may run the next prefill meanwhile. Idempotent (a later sha256 = true adds it).
    void seal(unsigned threads = 0, bool sha256 = false);
    [[nodiscard]] const std::string& sha256() const noexcept;  // hex, empty until a seal with sha256
    [[nodiscard]] bool captured() const noexcept;
    [[nodiscard]] bool sealed() const noexcept;
    [[nodiscard]] std::span<const std::uint8_t> bytes() const;  // sealed: the serialized state (in the pinned buffer)
    [[nodiscard]] std::uint64_t digest() const;                 // sealed: AttentionState::digest()
    [[nodiscard]] std::uint32_t position() const noexcept;
    [[nodiscard]] const deepseek41::AttentionFrame& frame() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;       // usable capture buffer bytes
    [[nodiscard]] std::size_t pinned_bytes() const noexcept;   // buffer plus pinned metadata
    struct Timing {
        double capture_ms = 0;   // host wall of capture_state (counts round trip + pack + D2H + frame)
        double meta_ms = 0;      // host wall of the counts gather + D2H + synchronization
        double pack_ms = 0;      // device: memset + gather kernel into the staging buffer (events)
        double d2h_ms = 0;       // device: the one D2H copy (events)
        double frame_ms = 0;     // host: header, sizes and tokens written into the pinned buffer
        double seal_ms = 0;      // host wall of seal (digest + checksum, and the SHA-256 when asked)
        double sha_ms = 0;       // host: the SHA-256 thread's own time (seal with sha256)
        std::uint32_t segments = 0, copies = 0;  // gathered pieces; cudaMemcpyAsync calls of the capture
    };
    [[nodiscard]] const Timing& timing() const noexcept;

private:
    friend class AttentionState;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::ds41::cuda::attn
