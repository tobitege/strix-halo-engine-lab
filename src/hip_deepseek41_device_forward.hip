#include "strix/env_flags.hpp"
#include "strix/file_snapshot.hpp"
#include "strix/hip_check.hpp"
#include "strix/deepseek41_device_forward.hpp"

#include "strix/deepseek41_attention_host.hpp"
#include "strix/dist/communicator.hpp"
#include "hip_ds41_staged_ready.hpp"
#include "xport/ds41_adapter.hpp"
#include "strix/hip_ds41_attention.hpp"
#include "strix/ds41_engram.hpp"
#include "strix/ds41_expert_tp.hpp"
#include "strix/ds41_moe.hpp"
#include "strix/ds41_native_fp8.hpp"
#include "strix/ds41_packed.hpp"
#include "strix/ds41_shexp_tp.hpp"
#include "strix/ds41_trellis.hpp"
#include "strix/ds41_trellis_group.hpp"
#include "strix/ds41_weight_overlay.hpp"
#include "strix/ds41_engram_tp.hpp"
#include "strix/ds41_touch_batch.hpp"
#include "strix/ds41_exact_flags.hpp"
#include "strix/hip_ds41_expert_tp.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_layer.hpp"
#include "hip_ds41_step_trace.hpp"

#include "hip_ds41_prepared_load.hpp"

#include "core/hip_range_weight_loader.hpp"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string_view>
#include <stdexcept>
#include <future>
#include <thread>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace strix::deepseek41 {
namespace {

void check(hipError_t e, const char* what) {
    ::strix::hip_check_prefixed("deepseek41 device forward: ", e, what);
}

std::uint64_t fnv(const std::vector<float>& v) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const float f : v) {
        auto b = std::bit_cast<std::uint32_t>(f);
        for (int k = 0; k < 4; ++k, b >>= 8U) {
            h ^= b & 0xffU;
            h *= 1099511628211ULL;
        }
    }
    return h;
}

template <typename T>
struct Dev {
    T* p = nullptr;
    std::size_t n = 0;
    Dev() = default;
    explicit Dev(std::size_t count) : n(count) {
        check(hipMalloc(reinterpret_cast<void**>(&p), std::max<std::size_t>(count, 1) * sizeof(T)), "hipMalloc");
    }
    ~Dev() {
        if (p != nullptr) (void)hipFree(p);
    }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    Dev(Dev&& o) noexcept : p(std::exchange(o.p, nullptr)), n(o.n) {}
    Dev& operator=(Dev&& o) noexcept {
        std::swap(p, o.p);
        std::swap(n, o.n);
        return *this;
    }
    template <typename C>
    void upload(const C& host, hipStream_t s) {
        check(hipMemcpyAsync(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice, s), "upload");
    }
};

template <typename T>
std::vector<T> download(const T* p, std::size_t n, hipStream_t s) {
    std::vector<T> out(n);
    check(hipMemcpyAsync(out.data(), p, n * sizeof(T), hipMemcpyDeviceToHost, s), "download");
    check(hipStreamSynchronize(s), "download");
    return out;
}

class Mapped {
public:
    explicit Mapped(const std::filesystem::path& path) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("deepseek41 device forward: cannot open " + path.string());
        strix::LoaderCapture::record(path, fd_);
        struct stat st {};
        if (::fstat(fd_, &st) != 0) throw std::runtime_error("deepseek41 device forward: cannot stat the model");
        size_ = static_cast<std::uint64_t>(st.st_size);
        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) throw std::runtime_error("deepseek41 device forward: mmap failed");
        base_ = static_cast<const std::uint8_t*>(p);
    }
    ~Mapped() {
        if (base_ != nullptr) ::munmap(const_cast<std::uint8_t*>(base_), size_);
        if (fd_ >= 0) ::close(fd_);
    }
    Mapped(const Mapped&) = delete;
    Mapped& operator=(const Mapped&) = delete;
    [[nodiscard]] std::span<const std::uint8_t> bytes(std::uint64_t offset, std::uint64_t size) const {
        if (offset > size_ || size > size_ - offset) throw std::out_of_range("deepseek41 device forward: tensor outside the file");
        return {base_ + offset, static_cast<std::size_t>(size)};
    }

private:
    int fd_ = -1;
    std::uint64_t size_ = 0;
    const std::uint8_t* base_ = nullptr;
};

// Routes of one window resident on the device (gate/up reads rows, down reads and writes slots).
struct DeviceRoutes {
    Dev<std::uint32_t> experts, begins, rows, slots, order, low_counts;
    std::uint32_t expert_count = 0, max_rows = 0;
    DeviceRoutes(const ds41::WindowRoutes& r, hipStream_t s)
        : experts(r.experts.size()), begins(r.begins.size()), rows(r.rows.size()), slots(r.slots.size()), order(r.order.size()),
          low_counts(r.low_counts.size()), expert_count(static_cast<std::uint32_t>(r.experts.size())), max_rows(r.max_rows) {
        experts.upload(r.experts, s);
        begins.upload(r.begins, s);
        rows.upload(r.rows, s);
        slots.upload(r.slots, s);
        order.upload(r.order, s);
        low_counts.upload(r.low_counts, s);
    }
    [[nodiscard]] ds41::RoutesView gate_up() const { return {experts.p, begins.p, rows.p, slots.p, expert_count, max_rows}; }
    [[nodiscard]] ds41::RoutesView down() const { return {experts.p, begins.p, slots.p, slots.p, expert_count, max_rows}; }
};

// out = a + b elementwise (one rounding): the canonical y = y0 + y1 of the two K-halves of wo_b.
// Reads `n` 16-byte words and keeps nothing (the cache lines stay in the MALL for the kernels that follow).
__global__ void touch_kernel(const uint4* __restrict__ p, std::uint64_t n, std::uint32_t* __restrict__ never) {
    std::uint32_t acc = 0;
    for (std::uint64_t i = blockIdx.x * std::uint64_t{blockDim.x} + threadIdx.x; i < n; i += std::uint64_t{gridDim.x} * blockDim.x) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9U && n == 0U) never[0] = acc;  // never true: keeps the loads
}

// Engram output-row TP (ds41::engram_tp()): rows [first, first + half) of every token row of ekv ([rows][n]) packed
// contiguous for the exchange, and the peer's half scattered back into its rows: plain copies, the same bits.
__global__ void engram_pack(const float* __restrict__ ekv, float* __restrict__ tx, std::uint32_t rows, std::uint32_t n, std::uint32_t half,
                            std::uint32_t first) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * half) tx[i] = ekv[std::size_t{i / half} * n + first + i % half];
}
__global__ void engram_scatter(const float* __restrict__ rx, float* __restrict__ ekv, std::uint32_t rows, std::uint32_t n, std::uint32_t half,
                               std::uint32_t first) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * half) ekv[std::size_t{i / half} * n + first + i % half] = rx[i];
}
// Up to 16 spans read by one launch (ds41::touch_batch()): span k is the 16-byte words [end[k - 1], end[k]) of the
// concatenation, each read once as by touch_kernel; a thread's index only grows, so its span index only moves forward.
struct TouchSpans {
    const uint4* p[16];
    std::uint64_t end[16];
    std::uint32_t count;
};
__global__ void touch_spans_kernel(TouchSpans s, std::uint32_t* __restrict__ never) {
    std::uint32_t acc = 0, k = 0;
    const std::uint64_t total = s.count ? s.end[s.count - 1U] : 0U;
    for (std::uint64_t i = blockIdx.x * std::uint64_t{blockDim.x} + threadIdx.x; i < total; i += std::uint64_t{gridDim.x} * blockDim.x) {
        while (i >= s.end[k]) ++k;
        const uint4 v = s.p[k][i - (k ? s.end[k - 1U] : 0U)];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9U && total == 0U) never[0] = acc;  // never true: keeps the loads
}

__global__ void add_halves(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, std::uint32_t n) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
// The same add for one column chunk of STRIX_DS41_OB_SPLIT: a and b hold [rows][cols], out the rows of `width` values
// from column col0.
__global__ void add_halves_cols(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, std::uint32_t rows,
                                std::uint32_t cols, std::uint32_t col0, std::uint32_t width) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) out[std::size_t{i / cols} * width + col0 + i % cols] = a[i] + b[i];
}
// STRIX_DS41_OB_SPLIT=K: the tensor-parallel o_b half in K column chunks, each one's exchange overlapping the next one's
// product (every output's product and add unchanged, so the same bits). 1 / unset: one product, one exchange.
std::uint32_t ob_split(std::uint32_t hidden) {
    static const std::uint32_t k = [] {
        const char* v = std::getenv("STRIX_DS41_OB_SPLIT");
        return v && *v ? static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10)) : 1U;
    }();
    if (k <= 1U) return 1U;
    if (k > 8U || hidden % k != 0U || (hidden / k) % 32U != 0U) throw std::invalid_argument("STRIX_DS41_OB_SPLIT: 2, 4, 5 or 8");
    return k;
}
// STRIX_DS41_LATE_HC_LAYERS=all (default since P17.5) | none | <comma-separated layers>: see late_mix in the forward.
bool late_hc_layer(std::uint32_t layer) {
    static const std::vector<bool> on = [] {
        std::vector<bool> r(1024, false);
        const char* v = std::getenv("STRIX_DS41_LATE_HC_LAYERS");
        if (!v || !*v) return std::vector<bool>(1024, true);
        const std::string s(v);
        if (s == "all") return std::vector<bool>(1024, true);
        if (s == "none" || s == "off") return r;
        std::size_t i = 0;
        while (i < s.size()) {
            const std::size_t j = s.find(',', i);
            const auto l = std::stoul(s.substr(i, j == std::string::npos ? std::string::npos : j - i));
            if (l >= r.size()) throw std::invalid_argument("STRIX_DS41_LATE_HC_LAYERS: layer out of range");
            r[l] = true;
            if (j == std::string::npos) break;
            i = j + 1U;
        }
        return r;
    }();
    return layer < on.size() && on[layer];
}

// STRIX_DS41_COMPRESS_RING: the ratio-2 carries of a rolled-back window, every layer in one launch (block b: layer b),
// exactly the copies and memsets of commit_window: open, the kept last row's (kv, score) and valid 1; else zeros.
struct CarryRestore {
    std::uint32_t* state;  // 2 x kHeadDim words: kv, score
    std::uint32_t* valid;
    const std::uint32_t* kv;
    const std::uint32_t* score;
};
constexpr std::uint32_t max_carry_restores = 64;  // per launch: 2 KiB of kernel arguments
struct CarryRestores {
    CarryRestore layer[max_carry_restores];
};
__global__ void restore_carries(CarryRestores a, std::uint32_t open) {
    const CarryRestore& c = a.layer[blockIdx.x];
    constexpr std::uint32_t D = ds41::attn::kHeadDim;
    for (std::uint32_t i = threadIdx.x; i < 2U * D; i += blockDim.x) c.state[i] = open != 0U ? (i < D ? c.kv[i] : c.score[i - D]) : 0U;
    if (threadIdx.x == 0U) *c.valid = open;
}

std::uint64_t row_bytes(ds41::WeightFormat f, std::uint32_t width) {
    return std::uint64_t{width} / ds41::block_elements(f) * ds41::block_bytes(f);
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// Byte ranges of the model file into device memory through the shared weight-load queue (core/weight_load_queue.hpp):
// O_DIRECT reads on bounded workers, per-ticket pinned staging retired by upload events, an own non-blocking copy
// stream. load() only queues; ready() is the one wait, at the end of the load or before the first consumer.
struct DirectLoader : strix::core::HipRangeWeightLoader {
    explicit DirectLoader(const std::filesystem::path& path) : HipRangeWeightLoader(path.string()) {
        strix::LoaderCapture::record(path, fd());
    }
};

// FNV of the live window of a ring (slot = position % capacity): positions max(0, T - window) .. T - 1 in order (older
// slots are never read again, and a prefill on the exact cone does not write them).
std::uint64_t ring_window_hash(const std::uint8_t* ring, std::uint32_t positions) {
    namespace attn = ds41::attn;
    std::uint64_t h = 14695981039346656037ULL;
    const std::uint32_t first = positions > attn::kWindow ? positions - attn::kWindow : 0U;
    for (std::uint32_t p = first; p < positions; ++p) {
        const std::uint8_t* e = ring + std::size_t{p % AttentionHost::ring_capacity} * attn::kWindowEntryBytes;
        for (std::uint32_t i = 0; i < attn::kWindowEntryBytes; ++i) h = (h ^ e[i]) * 1099511628211ULL;
    }
    return h;
}

}  // namespace

struct DeviceForward::Impl {
    DeviceForwardOptions options;
    TargetModel model;
    Plan plan;
    std::unique_ptr<Mapped> file;
    std::unique_ptr<ds41::WeightOverlay> overlay;
    std::map<std::string, std::array<std::uint32_t, 4>> cb4;  // weights packed to CB4 and their tables
    std::map<std::string, std::uint32_t> uniform;              // weights packed to U5 / U6 (bits)
    std::set<std::string> router_bf16;                         // routers held as their exact BF16 copy
    std::set<std::string> f8;                                  // F16 tensors held as their exact F8 rows
    std::set<std::string> planar;                              // packed site weights in the planar layout
    std::map<std::string, std::array<std::uint32_t, 2>> native_fp8;  // native FP8 site weights: rows, width (codes, then exponents)
    Dev<float> fp8_gate, fp8_up;                                      // native FP8 shared expert: gate and up products [R][I]
    // trellis sites (options.trellis, ds41_trellis.hpp): the pack, the library context, the prepared site of each
    // GGUF weight name this rank runs from the pack, and the bytes it reads per step (op timing)
    std::unique_ptr<ds41::TrellisPack> trellis_pack;
    std::unique_ptr<ds41::TrellisContext> trellis;
    std::vector<std::string> trellis_classes;
    std::map<std::string, const ds41::TrellisSite*> trellis_sites;
    std::map<std::string, std::uint64_t> trellis_bytes;
    Dev<float> tr_gate, tr_up, tr_act;  // trellis shared expert: gate, up and the SwiGLU activation [R][I] (F32)
    // trellis shared expert split over the ranks (options.trellis_shexp_tp, ds41-trellis-k5-v1:shexp-tp): per layer the
    // rank slices this process runs (world 2: its own; one process: both, rank order); tr_p1 = rank 1's partial (one process)
    bool shexp_tp = false;
    std::vector<std::vector<ds41::ShexpTpSlices>> trellis_shexp_tp;
    Dev<float> tr_p1;
    // trellis attention projections (ds41-trellis-k5-v1, q_b / o_a / o_b): per layer, the linears this process runs in
    // rank-slice order (TP: this rank's; one process: both ranks'), empty = today's kernel
    struct TrellisAttn { std::vector<const ds41::TrellisSite*> q_b, o_a, o_b; };
    std::vector<TrellisAttn> trellis_attn;
    std::map<int, std::vector<const ds41::TrellisSite*>> trellis_shared_halves;
    std::unique_ptr<ds41::TrellisGroupRunner> trellis_oa;  // STRIX_DS41_TRELLIS_OA_GROUP=1|2: o_a groups in one staged call
    // trellis LM head (ds41-trellis-k5-v1:head): the vocabulary slices this process computes (world 2: its own; world 1:
    // both, so one process gives the two ranks' bits) and their prepared row slices of the pack's head
    std::vector<std::pair<ds41::VocabSlice, const ds41::TrellisSite*>> trellis_head;
    // STRIX_DS41_TRELLIS_HEADOUT: a step of R > 1 rows writes the head to head_rows ([slice k][max_window_rows][count],
    // contiguous rows, so the library writes in place and R5..7 go as 8 rows) instead of logits; R1 writes logits in place.
    // head_rows_valid: the last head step went to head_rows (argmax and device_logits_rows read it from there).
    bool headout = false, head_rows_valid = false;
    [[nodiscard]] std::pair<const float*, std::size_t> logits_rows(std::uint32_t slice_first, std::uint32_t vocab) const;
    [[nodiscard]] const ds41::TrellisSite* trellis_site(const std::string& name) const {
        const auto it = trellis_sites.find(name);
        return it == trellis_sites.end() ? nullptr : it->second;
    }
    std::map<std::string, const ManifestTensor*> tensors;
    std::map<std::string, Dev<std::uint8_t>> weights;  // every tensor a device operation reads
    std::unique_ptr<ds41::EngramHasher> hasher;
    std::vector<std::uint64_t> engram_rows;
    std::vector<std::uint32_t> history;  // tokens of positions [0, position) (export_state)
    hipStream_t stream = nullptr;
    std::uint32_t position = 0;
    // attention piece state (device): per-layer rings, the KV sources' caches and ratio-2 carries (ping-pong),
    // the candidate pool and each index source's selection
    std::array<ds41::attn::LayerAttention, 40> roles{};
    struct LayerState {
        Dev<std::uint8_t> ring, kv_cache, idx_cache;
        Dev<float> carry[2];
        Dev<std::uint32_t> carry_valid[2];
        int carry_cur = 0;
        Dev<std::uint32_t> idx, idx_count;
        Dev<float> ckv, cscore;  // compressor inputs of the last step's rows (the ratio-2 carry of a rolled-back window)
    };
    std::vector<LayerState> attn;
    Dev<std::uint8_t> candidates;
    std::vector<hipEvent_t> op_events;  // op_timing: one per plan step + 1
    std::unique_ptr<ds41::StepTrace> trace;  // STRIX_DS41_STEP_TRACE (hip_ds41_step_trace.hpp), null when unset
    Dev<std::uint32_t> op_ids;           // op_timing: every layer's route ids of the step (copied on the stream)
    Dev<float> index_scores, iq, iw, attn_partial;
    Dev<std::uint32_t> select_keys;
    // stream_experts: device slots for the selected experts of one layer and their pinned staging
    Dev<std::uint8_t> slot_gate, slot_up, slot_down;
    std::uint8_t* slot_pinned = nullptr;
    std::uint8_t* receive_pinned = nullptr;  // receive_buffer(): a pipeline TAIL's message payloads
    std::size_t receive_bytes = 0;
    double stream_seconds = 0.0;  // last step: host time copying expert bytes (file -> pinned)
    Dev<std::uint8_t> index_qprep;       // index scores scratch: R x kIdxPrepBytes
    Dev<std::uint32_t> attn_tickets;     // R x kAttnGroupCount, zero between calls (the last chunk block combines)
    Dev<std::uint32_t> router_tickets;   // 1, zero between calls (STRIX_DS41_ROUTER_FAST: the last scores block selects)
    Dev<std::uint64_t> pos64;
    Dev<ds41::attn::RopeRow> rope_rows;  // [window @ pos][compressed @ pos][compressed @ even group start], W rows each
    std::uint32_t rows = 1;              // rows of the last step (a verify window: R > 1)
    Dev<float> capture[3];               // capture_main_hidden: input streams of the last three layers, [R][hc][hidden]
    Dev<float> streams[2];
    int cur = 0;
    bool identity_pre = true;  // pre_carry is the identity (before the first FFN site)
    std::array<float, ds41::hc_copies> pre_carry{};
    Dev<float> x, qr, q, kv, o, o_low, y, shared, raw, low, high, h, logits, emb, ekv, ekv_tx, ekv_rx, scores, route_w;
    Dev<float> head_rows;  // STRIX_DS41_TRELLIS_HEADOUT (see headout)
    Dev<std::uint8_t> erows, image_row;  // erows: [engram layer of the range][row][column] x 264 B, uploaded once per step
    std::uint8_t* erows_pinned = nullptr;  // pinned staging of erows (the upload is asynchronous)
    Dev<std::uint32_t> token, pos, ids;
    Dev<ds41::HcMix> mix_attn, mix_ffn;
    Dev<float> mix_partial;
    Dev<std::uint32_t> hc_counters;  // enqueue_hc_site's per-row counters, back at zero after every launch
    // world 2, STRIX_DS41_DECODE_PREFETCH=MB (0 off, default): while the o_b exchange travels, a lowest-priority
    // stream (joined after the exchange start, so the copy back never waits for it) reads ahead the weights of the next
    // steps, up to MB per row of the step (the wait grows with the rows), so they come from the MALL afterwards
    // (BatchForward's STRIX_DS41_BATCH_PREFETCH read-ahead). Not in the half-sum exchange: the overlapped shared expert
    // already fills it (16 MB there cost R1 +5 %). Only reads: the same bits.
    std::uint64_t prefetch_bytes = 0;
    hipStream_t prefetch_stream = nullptr;
    hipEvent_t prefetch_ready = nullptr;
    Dev<std::uint32_t> never;
    Dev<std::int8_t> site_codes;
    Dev<std::int8_t> o_codes;  // the attention output's A8 from its combine (STRIX_DS41_ATTN_A8=1)
    Dev<float> o_deltas;
    Dev<std::int32_t> o_sums;
    Dev<float> site_deltas;
    Dev<std::int32_t> site_sums;
    Dev<std::int8_t> codes, act_codes, sh_act_codes;
    Dev<float> deltas, act_deltas, sh_act_deltas;
    Dev<std::int32_t> sums, act_sums, sh_act_sums;
    Dev<ds41::ArgmaxCandidate> partial, cand0, cand1, best;
    std::vector<std::uint32_t> route_ids;
    // routed window of the current layer, built on the device from the router's ids (no host round trip)
    struct BuiltRoutes {
        Dev<std::uint32_t> experts, begins, rows, slots, order, low_counts, counts;
    } built;
    std::vector<float> logits_host;
    std::uint32_t expert_first = 0, expert_end = 0;  // routed experts resident on this rank
    // STRIX_DS41_EXPERT_TP=1 (two ranks): every routed expert split by its intermediate rows over the ranks
    // (ds41-expert-tp2-i1152-v1, ds41_expert_tp.hpp); expert_first/end then cover all experts.
    bool expert_tp = false;
    ds41::VocabSlice head_slice{};                   // head rows resident on this rank
    std::unique_ptr<ds41::EngramRowReader> engram_reader;
    std::vector<std::uint8_t> engram_bytes;  // Engram rows of the current token, [engram layer][column] x 264 B
    // STRIX_DS41_ENGRAM_ASYNC=1: the NVMe reads run on a host thread, one batch per Engram layer straight into its
    // pinned staging, while the plan is enqueued; each layer's upload waits only for its own rows (same bytes).
    bool engram_async = false;
    // STRIX_DS41_COMPRESS_RING=1: commit_window restores every ratio-2 carry in one kernel and does not synchronise (the
    // carries are only read by the stream's later work; host reads of them download, which synchronises).
    bool compress_ring = false;
    std::map<std::uint32_t, std::unique_ptr<DeviceRoutes>> shared_routes;  // per window rows (one shared expert)
    Dev<float> half_own, half_other;  // half sums [R][H], then (split shared expert) this rank's shared rows [R][H / 2]
    Dev<ds41::ArgmaxCandidate> cand_own, cand_other;
    // world 2, shared expert split by rows over the ranks (DeviceForwardOptions::split_shared): this rank's A8
    // activation half [R][I / 2] codes then [R][I / 64] deltas, and the peer's; the slot of the shared_expert hash.
    bool split_shared = false;
    Dev<std::uint8_t> shx_own, shx_other;
    std::size_t shared_hash_at = 0;
    // pipeline stages: the cut this TAIL receives (first > 0) or this FRONT produces (layers < the model's)
    std::uint32_t first = 0;
    ds41::pipe::BoundaryPlan cut;
    ds41::pipe::BoundaryGeometry geo;
    std::uint32_t last_position = 0;  // position of row 0 of the last step (export_boundary)
    std::vector<std::vector<std::uint32_t>> boundary_topk;  // host staging of the boundary's top-k sections
    Dev<std::uint8_t> control;        // exchange_host staging: [send | receive] x control_chunk
    static constexpr std::size_t control_chunk = 1U << 20U;  // a whole verify window's STEP frame in one exchange
    [[nodiscard]] bool distributed() const noexcept { return options.world == 2U; }
    [[nodiscard]] double comm_seconds() const { return distributed() ? options.communicator->stats().seconds : 0.0; }

    std::span<const std::uint8_t> host_tensor(const std::string& name) const {
        const auto it = tensors.find(name);
        if (it == tensors.end()) throw std::invalid_argument("deepseek41 device forward: no tensor " + name);
        if (overlay)
            if (const auto* e = overlay->find(name)) return overlay->bytes(*e);
        return file->bytes(it->second->absolute_offset, it->second->byte_size);
    }
    [[nodiscard]] const std::uint8_t* dev(const std::string& name) const { return weights.at(name).p; }
    void site_weight(ds41::Q8A8View& v, const std::string& name) const {  // Q8_0 rows, or CB4 rows of a packed overlay tensor
        v.weight = dev(name);
        if (const auto f = native_fp8.find(name); f != native_fp8.end()) {  // native FP8: codes, then the block exponents
            v.packed = ds41::packed_native_fp8;
            v.scales = v.weight + std::size_t{f->second[0]} * f->second[1];
            v.planar = false;
            return;
        }
        const auto it = cb4.find(name);
        v.packed = it != cb4.end() ? 4U : uniform.contains(name) ? uniform.at(name) : 0U;
        if (it != cb4.end()) std::copy(it->second.begin(), it->second.end(), v.table);
        v.planar = planar.contains(name);
    }
    [[nodiscard]] const float* devf(const std::string& name) const { return reinterpret_cast<const float*>(weights.at(name).p); }
    [[nodiscard]] const std::uint16_t* devh(const std::string& name) const {
        return reinterpret_cast<const std::uint16_t*>(weights.at(name).p);
    }
    [[nodiscard]] std::vector<float> buffer(Buffer b) const;
    ~Impl() {
        // Plans borrow the activation allocations, group workspaces and streams.
        trellis_oa.reset();
        trellis.reset();
    }
};

std::vector<float> DeviceForward::Impl::buffer(Buffer b) const {
    const auto& c = model.config;
    const auto mix = [&](const ds41::HcMix* p) {
        const auto m = download(p, 1, stream)[0];
        std::vector<float> v;
        v.insert(v.end(), m.pre.begin(), m.pre.end());
        v.insert(v.end(), m.post.begin(), m.post.end());
        v.insert(v.end(), m.comb.begin(), m.comb.end());
        return v;
    };
    switch (b) {
    case Buffer::stream: return download(streams[cur].p, plan.buffer_elements(b), stream);
    case Buffer::mix_attn: return mix(mix_attn.p);
    case Buffer::mix_ffn: return mix(mix_ffn.p);
    case Buffer::pre_carry: return {pre_carry.begin(), pre_carry.end()};
    case Buffer::x: return download(x.p, c.hidden_size, stream);
    case Buffer::qr: return download(qr.p, c.q_lora_rank, stream);
    case Buffer::q: return download(q.p, std::size_t{c.num_attention_heads} * c.head_dim, stream);
    case Buffer::kvrow: return download(kv.p, c.head_dim, stream);
    case Buffer::o: return download(o.p, std::size_t{c.num_attention_heads} * c.head_dim, stream);
    case Buffer::o_low: return download(o_low.p, std::size_t{c.o_groups} * c.o_lora_rank, stream);
    case Buffer::y: return download(y.p, c.hidden_size, stream);
    case Buffer::shared: return download(shared.p, c.hidden_size, stream);
    case Buffer::route_w: return download(route_w.p, c.num_experts_per_tok, stream);
    case Buffer::raw: return download(raw.p, std::size_t{c.num_experts_per_tok} * c.hidden_size, stream);
    default: throw std::logic_error("deepseek41 device forward: buffer without a device copy");
    }
}

std::vector<Op> DeviceForward::host_ops() { return {}; }  // every operation runs on the device

namespace {
// Load-time choices from the environment, so every app built on DeviceForward can switch them without its own flags: a
// requantization overlay when the caller sets none (STRIX_DS41_OVERLAY=FILE.gguf, STRIX_DS41_CB4=1 packs its sites), and
// the exact storage forms STRIX_DS41_ROUTER_BF16 / STRIX_DS41_F16_AS_F8 (0 off, 1 on; unset: the caller's option).
DeviceForwardOptions with_env_overlay(DeviceForwardOptions o) {
    if (o.weights_overlay.empty())
        if (const char* e = std::getenv("STRIX_DS41_OVERLAY"); e != nullptr && *e != 0) {
            o.weights_overlay = e;
            if (const char* c = std::getenv("STRIX_DS41_CB4"); c != nullptr && *c == '1') o.cb4_overlay = true;
        }
    const auto flag = [](const char* name, bool& value) {
        if (const char* v = std::getenv(name); v != nullptr && (*v == '0' || *v == '1')) value = *v == '1';
    };
    flag("STRIX_DS41_ROUTER_BF16", o.router_bf16);
    flag("STRIX_DS41_OP_TIMING", o.op_timing);
    flag("STRIX_DS41_F16_AS_F8", o.f16_as_f8);
    // native FP8 dense weights (default off): STRIX_DS41_NATIVE_FP8=DIR of the original checkpoint, STRIX_DS41_NATIVE_FP8_SITES=classes
    if (o.native_fp8.empty())
        if (const char* e = std::getenv("STRIX_DS41_NATIVE_FP8"); e != nullptr && *e != 0) o.native_fp8 = e;
    if (const char* e = std::getenv("STRIX_DS41_NATIVE_FP8_SITES"); e != nullptr && *e != 0) o.native_fp8_sites = e;
    // trellis dense sites (default off): STRIX_DS41_TRELLIS=pack dir or STRIX_DS41_TRELLIS_MIX=class=dir,... (mixed K
    // per class, ds41_trellis.hpp), STRIX_DS41_TRELLIS_SITES=classes
    if (o.trellis.empty())
        if (const auto spec = ds41::trellis_env_spec(); !spec.empty()) o.trellis = spec;
    if (const char* e = std::getenv("STRIX_DS41_TRELLIS_SITES"); e != nullptr && *e != 0) o.trellis_sites = e;
    flag("STRIX_DS41_SHEXP_TP", o.trellis_shexp_tp);
    return o;
}
}  // namespace

DeviceForward::DeviceForward(const std::filesystem::path& gguf, const HostForwardOptions& options)
    : DeviceForward(gguf, [&] {
          DeviceForwardOptions o;
          o.layers = options.layers;
          o.max_positions = options.max_positions;
          o.weights_overlay = options.weights_overlay;
          return o;
      }()) {}

DeviceForward::DeviceForward(const std::filesystem::path& gguf, const DeviceForwardOptions& options_in) : impl_(std::make_unique<Impl>()) {
    const DeviceForwardOptions options = with_env_overlay(options_in);
    const auto t_load = std::chrono::steady_clock::now();
    auto& m = *impl_;
    m.options = options;
    if (options.world != 1U && options.world != 2U) throw std::invalid_argument("deepseek41 device forward: world must be 1 or 2");
    if (options.world == 2U && (options.communicator == nullptr || options.rank > 1U))
        throw std::invalid_argument("deepseek41 device forward: world 2 needs a communicator and rank 0 or 1");
    m.model = validate_target(read_file(gguf));
    if (options.first_layer != 0U && (options.first_layer >= options.layers || !options.head))
        throw std::invalid_argument("deepseek41 device forward: a pipeline TAIL runs layers [first_layer, layers) and the head");
    m.first = options.first_layer;
    m.plan = build_decode_plan(m.model, options.first_layer, options.layers, options.first_layer == 0U, options.head);
    m.geo = ds41::pipe::boundary_geometry(m.model.config);
    if (options.first_layer != 0U) m.cut = ds41::pipe::plan_boundary(m.model.layers, options.first_layer);
    else if (options.layers < m.model.config.num_hidden_layers) m.cut = ds41::pipe::plan_boundary(m.model.layers, options.layers);
    m.file = std::make_unique<Mapped>(gguf);
    if (!options.weights_overlay.empty()) m.overlay = std::make_unique<ds41::WeightOverlay>(options.weights_overlay, m.model.manifest);
    for (const auto& t : m.model.manifest.tensors) m.tensors.emplace(t.name, &t);
    check(hipStreamCreateWithFlags(&m.stream, hipStreamNonBlocking), "stream");
    if (options.world == 2U) {  // default 2 MB per row (=0 off)
        const char* pf = std::getenv("STRIX_DS41_DECODE_PREFETCH");
        m.prefetch_bytes = (pf && *pf ? std::strtoull(pf, nullptr, 10) : 2U) << 20U;
        if (m.prefetch_bytes) {
            int least = 0, greatest = 0;
            check(hipDeviceGetStreamPriorityRange(&least, &greatest), "stream priorities");
            check(hipStreamCreateWithPriority(&m.prefetch_stream, hipStreamNonBlocking, least), "prefetch stream");
            check(hipEventCreateWithFlags(&m.prefetch_ready, hipEventDisableTiming), "prefetch event");
            m.never = Dev<std::uint32_t>(1);
        }
    }
    {
        const auto& cfg = m.model.config;
        const std::uint32_t per_rank = cfg.n_routed_experts / options.world;
        m.expert_first = options.rank * per_rank;
        m.expert_end = m.expert_first + per_rank;
        m.expert_tp = ds41::expert_tp::decode_enabled(options.world, cfg.moe_intermediate_size, options.stream_experts);  // default on since P16
        if (m.expert_tp) {
            if (!m.distributed() || options.stream_experts || cfg.moe_intermediate_size != ds41::expert_tp::inter)
                throw std::invalid_argument("deepseek41 device forward: STRIX_DS41_EXPERT_TP=1 needs two ranks, resident experts and I = 2304");
            m.expert_first = 0;
            m.expert_end = cfg.n_routed_experts;
        }
        m.head_slice = m.distributed() ? ds41::vocab_slice(cfg.vocab_size, 2U, options.rank) : ds41::VocabSlice{0U, cfg.vocab_size};
    }
    std::set<std::string> names;
    for (const auto& s : m.plan.steps)
        for (const auto& w : s.weights)
            if (w.find("engram_embd") == std::string::npos) names.insert(w);  // Engram rows are read per token
    DirectLoader loader(gguf);
    std::unique_ptr<DirectLoader> overlay_loader;
    if (m.overlay) overlay_loader = std::make_unique<DirectLoader>(m.overlay->path());
    const std::function<void()> loaded = [&] { loader.ready(); if (overlay_loader) overlay_loader->ready(); };
    ds41::prepared::HipLoader prepared(gguf, m.overlay ? m.overlay->path() : std::filesystem::path{}, m.model, options.rank, options.world, m.expert_tp, &options.trellis, &options.trellis_sites);
    std::unique_ptr<ds41::SafetensorsCheckpoint> checkpoint;
    std::vector<std::string> fp8_classes;
    if (!options.native_fp8.empty()) {
        checkpoint = std::make_unique<ds41::SafetensorsCheckpoint>(options.native_fp8);
        fp8_classes = ds41::parse_fp8_site_classes(options.native_fp8_sites);
    }
    for (const auto& n : names) {
        const auto* t = m.tensors.at(n);
        if (checkpoint) {  // native FP8 site weight: the checkpoint's codes and block exponents instead of the Q8_0 tensor
            const auto cls = ds41::fp8_site_class(n);
            if (cls && std::find(fp8_classes.begin(), fp8_classes.end(), *cls) != fp8_classes.end()) {
                const auto rows = static_cast<std::uint32_t>(t->dimensions[1]), width = static_cast<std::uint32_t>(t->dimensions[0]);
                const auto host = ds41::read_fp8_tensor(*checkpoint, *ds41::fp8_original_name(n), rows, width);
                Dev<std::uint8_t> d(host.bytes.size());
                check(hipMemcpy(d.p, host.bytes.data(), host.bytes.size(), hipMemcpyHostToDevice), "native FP8 weight upload");
                m.native_fp8.emplace(n, std::array<std::uint32_t, 2>{rows, width});
                m.weights.emplace(n, std::move(d));
                continue;
            }
        }
        const auto* over = m.overlay ? m.overlay->find(n) : nullptr;
        std::uint64_t offset = over != nullptr ? over->absolute_offset : t->absolute_offset, bytes = t->byte_size;
        if (options.stream_experts && n.ends_with("_exps.weight")) continue;  // read per layer from the file
        if (n.ends_with("_exps.weight") && !options.stream_experts) {
            const auto& cfg = m.model.config;
            const bool down = n.ends_with("ffn_down_exps.weight");
            const auto count = m.expert_tp ? ds41::expert_tp::part_bytes(bytes, cfg.n_routed_experts, down, cfg.hidden_size)
                                         : bytes / cfg.n_routed_experts * (m.expert_end - m.expert_first);
            // Allocation is also needed by the fallback; avoid allocating this speculative destination
            // unless a validated artifact is available (explicit or discovered).
            if (prepared.active()) {
                Dev<std::uint8_t> d(count);
                if (prepared.load(n, count, d.p, m.stream)) {
                    m.weights.emplace(n, std::move(d));
                    continue;
                }
            }
        }
        if (m.expert_tp && n.ends_with("_exps.weight")) {  // this rank's part of every expert (ds41_expert_tp.hpp)
            const auto& cfg = m.model.config;
            const bool down = n.ends_with("ffn_down_exps.weight");
            Dev<std::uint8_t> d(ds41::expert_tp::part_bytes(bytes, cfg.n_routed_experts, down, cfg.hidden_size));
            auto& from = over != nullptr ? *overlay_loader : loader;
            ds41::expert_tp::load_part([&](std::uint64_t o, std::uint64_t b, std::uint8_t* dst) { from.load(o, b, dst); }, offset, bytes,
                                       cfg.n_routed_experts, down, cfg.hidden_size, options.rank, d.p, m.stream, loaded);
            m.weights.emplace(n, std::move(d));
            continue;
        }
        if (n.ends_with("_exps.weight")) {  // [experts][rows][row bytes]: this rank's expert range only
            const std::uint64_t per = bytes / m.model.config.n_routed_experts;
            offset += per * m.expert_first;
            bytes = per * (m.expert_end - m.expert_first);
        } else if (n == "output.weight") {  // [vocab][row bytes]: this rank's vocabulary slice only
            const std::uint64_t per = bytes / m.model.config.vocab_size;
            offset += per * m.head_slice.first;
            bytes = per * m.head_slice.count;
        }
        const bool site = n.ends_with("attn_q_a.weight") || n.ends_with("attn_q_b.weight") || n.ends_with("attn_kv.weight") ||
                          n.ends_with("attn_output_a.weight") || n.ends_with("attn_output_b.weight") || n == "output.weight" ||
                          n.ends_with("_shexp.weight");
        // F8 only where it pays: engram_kv (25600 x 6144); indexer.attn_q_b is also exact but its narrow rows run no faster
        if (options.f16_as_f8 && n.ends_with("engram_kv.weight")) {
            const auto f16 = m.host_tensor(n);
            const auto packed = ds41::pack_f16_as_f8(
                std::span<const std::uint16_t>(reinterpret_cast<const std::uint16_t*>(f16.data()), f16.size() / 2U), t->dimensions[0]);
            if (packed) {
                Dev<std::uint8_t> d(packed->size());
                check(hipMemcpy(d.p, packed->data(), packed->size(), hipMemcpyHostToDevice), "F8 weight upload");
                m.f8.insert(n);
                m.weights.emplace(n, std::move(d));
                continue;
            }
        }
        if (options.router_bf16 && n.ends_with("ffn_gate_inp.weight")) {  // exact BF16 copy of an F32 router
            const auto f32 = m.host_tensor(n);
            std::vector<std::uint16_t> half(f32.size() / 4U);
            bool exact = true;
            for (std::size_t i = 0; i < half.size(); ++i) {
                std::uint32_t u;
                std::memcpy(&u, f32.data() + 4U * i, 4);
                exact = exact && (u & 0xffffU) == 0U;
                half[i] = static_cast<std::uint16_t>(u >> 16U);
            }
            if (exact) {
                Dev<std::uint8_t> d(half.size() * 2U);
                check(hipMemcpy(d.p, half.data(), half.size() * 2U, hipMemcpyHostToDevice), "BF16 router upload");
                m.router_bf16.insert(n);
                m.weights.emplace(n, std::move(d));
                continue;
            }
        }
        if (over != nullptr && options.cb4_overlay && site) {  // packed rows of this rank's range (ds41_packed)
            const auto whole = m.overlay->bytes(*over);
            if (const auto choice = ds41::choose_packing(whole)) {
                const bool planar = options.packed_planar && !n.ends_with("_shexp.weight");
                const auto packed = ds41::pack_rows(whole.subspan(offset - over->absolute_offset, bytes), t->dimensions[0], *choice,
                                                    planar ? ds41::PackedLayout::planar : ds41::PackedLayout::interleaved);
                if (planar) m.planar.insert(n);
                Dev<std::uint8_t> d(packed.size());
                check(hipMemcpy(d.p, packed.data(), packed.size(), hipMemcpyHostToDevice), "packed weight upload");
                if (choice->kind == ds41::PackedKind::cb4) {
                    std::array<std::uint32_t, 4> words{};
                    std::memcpy(words.data(), choice->table.data(), 16);
                    m.cb4.emplace(n, words);
                } else {
                    m.uniform.emplace(n, static_cast<std::uint32_t>(choice->kind));
                }
                m.weights.emplace(n, std::move(d));
                continue;
            }
        }
        Dev<std::uint8_t> d(bytes);
        (over != nullptr ? *overlay_loader : loader).load(offset, bytes, d.p);
        m.weights.emplace(n, std::move(d));
    }
    loaded();  // end-of-load publication: the one wait for every tensor queued above
    prepared.report();
    ds41::prepared::device_hashes(m.weights, options.rank, m.stream);
    // The declared identities of the steps on native FP8 weights (the shared expert only with all three tensors).
    for (auto& s : m.plan.steps) {
        if (s.weights.empty() || !m.native_fp8.contains(s.weights[0])) continue;
        if (s.op == Op::shared_expert) {
            if (!m.native_fp8.contains(s.weights[1]) || !m.native_fp8.contains(s.weights[2]))
                throw std::logic_error("deepseek41 device forward: native FP8 shared expert needs gate, up and down");
            s.identity = ds41::fp8_swiglu_identity;
        } else {
            s.identity = s.op == Op::o_a ? ds41::fp8_grouped_identity : s.op == Op::o_b ? ds41::fp8_ksplit_identity : ds41::fp8_product_identity;
        }
    }
    // trellis sites (identity ds41-trellis-k5-v1): each enabled class prepares its weights (whole or this rank's
    // slice) from the pack; the Q8_0 / packed copies above stay loaded. Site groups add their prepares here.
    if (!options.trellis.empty()) {
        m.trellis_pack = std::make_unique<ds41::TrellisPack>(options.trellis);
        m.trellis = std::make_unique<ds41::TrellisContext>();
        m.trellis_classes = ds41::parse_trellis_site_classes(options.trellis_sites);
        if (m.trellis_classes.empty())
            std::fprintf(stderr, "ds41 trellis: STRIX_DS41_TRELLIS is set but no site class is selected (STRIX_DS41_TRELLIS_SITES empty): "
                                 "no site reads the pack\n");
        const auto prepare = [&](const std::string& name, const ds41::TrellisSlice& slice = {}) {
            const auto pack_name = ds41::trellis_pack_name(name);
            if (!pack_name) throw std::logic_error("deepseek41 device forward: no trellis name for " + name);
            const auto& site = m.trellis->prepare(*m.trellis_pack, *pack_name, slice);
            m.trellis_sites[name] = &site;
            m.trellis_bytes[name] += site.device_bytes;
            return &site;
        };
        for (auto& s : m.plan.steps) {
            if (ds41::trellis_dense_full_enabled() && (s.op == Op::q_a || s.op == Op::kv) &&
                ds41::trellis_class_enabled(m.trellis_classes, s.weights[0])) {
                if (m.native_fp8.contains(s.weights[0]))
                    throw std::invalid_argument("deepseek41 device forward: dense site both native FP8 and trellis");
                const auto* site = prepare(s.weights[0]);
                s.identity = ds41::trellis_step_identity(site->bits, s.op == Op::q_a ? "q_a" : "kv");
            }
            // shared expert (replicated on both ranks, its whole gate / up / down): F32 x -> gate, up -> clamped SwiGLU (F32)
            // -> down, three trellis linears (docs/models/deepseek-v4.1/trellis-k5-v1.md, shexp)
            if (s.op == Op::shared_expert && ds41::trellis_class_enabled(m.trellis_classes, s.weights[0])) {
                if (m.native_fp8.contains(s.weights[0]))
                    throw std::invalid_argument("deepseek41 device forward: shared expert both native FP8 and trellis");
                if (options.trellis_shexp_tp) {
                    // ds41-trellis-k5-v1:shexp-tp: rank r's gate / up row slices and down column slice (world 2: its own;
                    // one process: both ranks', rank order). trellis_sites names the first, so the shexp checks see a
                    // trellis shared expert; trellis_bytes counts every slice this process reads.
                    const std::uint32_t inter = m.model.config.moe_intermediate_size;
                    if (m.trellis_shexp_tp.empty()) m.trellis_shexp_tp.assign(m.model.config.num_hidden_layers, {});
                    auto& slices = m.trellis_shexp_tp.at(static_cast<std::size_t>(s.layer));
                    for (std::uint32_t r = 0; r < ds41::shexp_tp_ranks; ++r) {
                        if (m.distributed() && r != options.rank) continue;
                        const auto gu = ds41::shexp_tp_gate_up_slice(inter, r), dn = ds41::shexp_tp_down_slice(inter, r);
                        ds41::ShexpTpSlices t;
                        t.gate = prepare(s.weights[0], gu);
                        t.up = prepare(s.weights[1], gu);
                        t.down = prepare(s.weights[2], dn);
                        slices.push_back(t);
                    }
                    for (std::size_t k = 0; k < 3U; ++k) {
                        const auto& f = slices.front();
                        m.trellis_sites[s.weights[k]] = k == 0U ? f.gate : k == 1U ? f.up : f.down;
                    }
                    s.identity = ds41::trellis_shexp_step_identity(slices.front().gate->bits, slices.front().down->bits, true);
                } else {
                    const auto* gate = prepare(s.weights[0]);
                    (void)prepare(s.weights[1]);
                    const auto* down = prepare(s.weights[2]);
                    // K4 whole-2304 and row-1152 products both use legacy S16 and H128 blocks.
                    // Keep the whole down product and its original association of sums.
                    if (ds41::trellis_dense_row_slices_enabled() && gate->bits == 4U && gate->rows == 2304U) {
                        auto& halves = m.trellis_shared_halves[s.layer];
                        for (const auto& w : {s.weights[0], s.weights[1]})
                            for (std::uint32_t first : {0U, 1152U})
                                halves.push_back(&m.trellis->prepare(*m.trellis_pack, *ds41::trellis_pack_name(w),
                                    {ds41::TrellisSlice::Kind::rows, first, 1152U}));
                    }
                    s.identity = ds41::trellis_shexp_step_identity(gate->bits, down->bits, false);
                }
            }
            // LM head: rows [slice.first, + count) of the vocabulary, F32 x (the final norm's output), no A8
            if (s.op == Op::head && ds41::trellis_class_enabled(m.trellis_classes, "output.weight")) {
                for (std::uint32_t r = 0; r < 2U; ++r) {
                    if (m.distributed() && r != m.options.rank) continue;
                    const auto sl = ds41::vocab_slice(m.model.config.vocab_size, 2U, r);
                    m.trellis_head.emplace_back(sl, prepare("output.weight", {ds41::TrellisSlice::Kind::rows, sl.first, sl.count}));
                }
                s.identity = ds41::trellis_step_identity(m.trellis_head.back().second->bits, "head");
                m.headout = ds41::trellis_headout_enabled();
            }
        }
        // attention projections (docs/models/deepseek-v4.1/trellis-k5-v1.md, q_b / o_a / o_b): rank r's q_b rows [16384 r, +16384), o_a groups
        // [4 r, 4 r + 4) (one linear each), o_b K-half columns [4096 r, +4096); one process runs both ranks' linears
        m.trellis_attn.assign(m.model.config.num_hidden_layers, {});
        {
            const auto& c = m.model.config;
            const bool tp = m.distributed() && options.tp_attention;
            const std::uint32_t first_rank = tp ? options.rank : 0U, ranks = tp ? 1U : 2U;
            const std::uint32_t q_rows = c.num_attention_heads / 2U * c.head_dim, o_half = c.o_groups * c.o_lora_rank / 2U;
            for (auto& s : m.plan.steps) {
                if (s.layer < 0 || (s.op != Op::q_b && s.op != Op::o_a && s.op != Op::o_b)) continue;
                const auto& name = s.weights[0];
                if (!ds41::trellis_class_enabled(m.trellis_classes, name)) continue;
                if (m.native_fp8.contains(name)) throw std::invalid_argument("deepseek41 device forward: " + name + " both native FP8 and trellis");
                auto& ta = m.trellis_attn[static_cast<std::size_t>(s.layer)];
                for (std::uint32_t r = first_rank; r < first_rank + ranks; ++r) {
                    const char* cls = s.op == Op::q_b ? "q_b" : s.op == Op::o_a ? "o_a" : "o_b";
                    const ds41::TrellisSite* last = nullptr;
                    if (s.op == Op::q_b) {
                        ta.q_b.push_back(last = prepare(name, {ds41::TrellisSlice::Kind::rows, r * q_rows, q_rows}));
                    } else if (s.op == Op::o_b) {
                        ta.o_b.push_back(last = prepare(name, {ds41::TrellisSlice::Kind::columns, r * o_half, o_half}));
                    } else {
                        for (std::uint32_t g = r * (c.o_groups / 2U); g < (r + 1U) * (c.o_groups / 2U); ++g) {
                            const auto pack_name = ds41::trellis_pack_name(name, static_cast<int>(g));
                            const auto& site = m.trellis->prepare(*m.trellis_pack, *pack_name);
                            m.trellis_bytes[name] += site.device_bytes;
                            ta.o_a.push_back(last = &site);
                        }
                    }
                    s.identity = ds41::trellis_step_identity(last->bits, cls);
                }
            }
        }
        m.trellis->reserve(m.stream, max_window_rows);
        if (const int mode = ds41::trellis_oa_group_mode(); mode != 0)  // default on (=0: the per-group calls below)
            for (const auto& ta : m.trellis_attn)
                if (!ta.o_a.empty()) {
                    m.trellis_oa = std::make_unique<ds41::TrellisGroupRunner>(ta.o_a, mode, m.stream);
                    std::fprintf(stderr, "ds41 trellis: %s\n", m.trellis_oa->describe().c_str());
                    break;
                }
    }
    m.shexp_tp = !m.trellis_shexp_tp.empty();
    ds41::trellis_shared_grouped_admit(m.shexp_tp, !m.trellis_shared_halves.empty(), 8U);
    {  // STRIX_DS41_SHARED_DOWN_FUSED: inventory of every shared down site + startup probe, before any capture
        std::vector<const ds41::TrellisSite*> downs;
        std::size_t shared_layers = 0;
        for (const auto& s : m.plan.steps) {
            if (s.op != Op::shared_expert) continue;
            ++shared_layers;
            if (const auto* d = m.trellis_site(s.weights[2])) downs.push_back(d);
        }
        ds41::trellis_shared_down_fused_admit(downs, shared_layers, m.shexp_tp, 8U);
    }
    if (options.trellis_shexp_tp && !m.shexp_tp)
        throw std::invalid_argument("deepseek41 device forward: STRIX_DS41_SHEXP_TP=1 needs STRIX_DS41_TRELLIS with the shexp class");
    if (m.shexp_tp && m.expert_tp)  // two per-rank partial folds into the half sums: no declared identity for the pair yet
        throw std::invalid_argument("deepseek41 device forward: STRIX_DS41_SHEXP_TP=1 with expert TP (STRIX_DS41_EXPERT_TP, default on "
                                    "since P16) is not a declared identity; set STRIX_DS41_EXPERT_TP=0 or leave SHEXP_TP unset");
    m.hasher = std::make_unique<ds41::EngramHasher>(m.model.engram);
    m.engram_rows.assign(std::size_t{m.hasher->layers()} * m.hasher->columns(), 0U);
    if (options.engram_direct_io) {
        std::vector<ds41::EngramTable> tables;
        for (const auto l : m.model.engram.layer_ids) {
            const auto* t = m.tensors.at("blk." + std::to_string(l) + ".engram_embd.weight");
            tables.push_back({t->absolute_offset, t->byte_size / 264U});
        }
        m.engram_reader = std::make_unique<ds41::EngramRowReader>(gguf, std::move(tables), ds41::engram_reader_threads(64U), true);  // 64 on the pair since P15
    }

    const auto& c = m.model.config;
    namespace attn = ds41::attn;
    m.roles = attn::released_plan();  // the host forward's AttentionHost checks these roles against the model
    std::uint32_t max_count = 1, source_ratio = 0;
    constexpr std::size_t W = max_window_rows;  // every activation buffer holds a verify window
    for (std::uint32_t l = 0; l < options.layers; ++l) {
        const auto& r = m.roles[l];
        Impl::LayerState L;
        L.ring = Dev<std::uint8_t>(std::size_t{AttentionHost::ring_capacity} * attn::kWindowEntryBytes);
        check(hipMemsetAsync(L.ring.p, 0, L.ring.n, m.stream), "ring");
        if (r.owns_kv) {
            const std::size_t entries = options.max_positions / r.ratio + 1U;
            max_count = std::max<std::uint32_t>(max_count, static_cast<std::uint32_t>(entries));
            L.kv_cache = Dev<std::uint8_t>(entries * attn::kKvEntryBytes);
            L.idx_cache = Dev<std::uint8_t>(entries * attn::kIdxEntryBytes);
            check(hipMemsetAsync(L.kv_cache.p, 0, L.kv_cache.n, m.stream), "kv cache");
            check(hipMemsetAsync(L.idx_cache.p, 0, L.idx_cache.n, m.stream), "index cache");
            L.ckv = Dev<float>(W * attn::kHeadDim);
            L.cscore = Dev<float>(W * attn::kHeadDim);
            for (int k = 0; k < 2; ++k) {
                L.carry[k] = Dev<float>(2U * attn::kHeadDim);
                L.carry_valid[k] = Dev<std::uint32_t>(1);
                check(hipMemsetAsync(L.carry[k].p, 0, L.carry[k].n * 4U, m.stream), "carry");
                check(hipMemsetAsync(L.carry_valid[k].p, 0, 4U, m.stream), "carry");
            }
        }
        if (r.owns_index) {
            L.idx = Dev<std::uint32_t>(W * attn::kTopK);
            L.idx_count = Dev<std::uint32_t>(W);
            check(hipMemsetAsync(L.idx.p, 0, W * attn::kTopK * 4U, m.stream), "top-k");
            check(hipMemsetAsync(L.idx_count.p, 0, W * 4U, m.stream), "top-k");
        }
        if (r.candidate_source) source_ratio = r.ratio;
        m.attn.push_back(std::move(L));
    }
    const std::size_t cand_bytes = source_ratio != 0U ? (options.max_positions / source_ratio + attn::kCandBlockSize) / attn::kCandBlockSize
                                                      : (max_count + attn::kCandBlockSize) / attn::kCandBlockSize;
    m.candidates = Dev<std::uint8_t>(W * cand_bytes);
    check(hipMemsetAsync(m.candidates.p, 0, W * cand_bytes, m.stream), "candidates");
    m.index_scores = Dev<float>(W * max_count);
    m.select_keys = Dev<std::uint32_t>(W * max_count);
    m.iq = Dev<float>(W * attn::kIdxHeads * attn::kIdxDim);
    m.iw = Dev<float>(W * attn::kIdxHeads);
    if (options.stream_experts) {
        if (options.world != 1U) throw std::invalid_argument("deepseek41 device forward: stream_experts is a one-node mode");
        const std::uint64_t gs = row_bytes(ds41::WeightFormat::iq2_xxs, c.hidden_size) * c.moe_intermediate_size;
        const std::uint64_t ds = row_bytes(ds41::WeightFormat::q2_k, c.moe_intermediate_size) * c.hidden_size;
        const std::uint64_t k = c.num_experts_per_tok;
        m.slot_gate = Dev<std::uint8_t>(k * gs);
        m.slot_up = Dev<std::uint8_t>(k * gs);
        m.slot_down = Dev<std::uint8_t>(k * ds);
        check(hipHostMalloc(reinterpret_cast<void**>(&m.slot_pinned), k * (2U * gs + ds), hipHostMallocDefault), "expert staging");
    }
    m.index_qprep = Dev<std::uint8_t>(std::size_t{W} * attn::hip::kIdxPrepBytes);
    m.attn_tickets = Dev<std::uint32_t>(std::size_t{W} * attn::hip::kAttnGroupCount);
    check(hipMemsetAsync(m.attn_tickets.p, 0, m.attn_tickets.n * 4U, m.stream), "attention tickets");
    m.router_tickets = Dev<std::uint32_t>(1);
    check(hipMemsetAsync(m.router_tickets.p, 0, 4U, m.stream), "router tickets");
    m.attn_partial = Dev<float>(W * attn::kHeads * attn::kMaxChunks * (2U + attn::kHeadDim));
    m.pos64 = Dev<std::uint64_t>(W);
    m.rope_rows = Dev<attn::RopeRow>(3U * W);
    const std::size_t H = c.hidden_size, hc = c.hc_mult, I = c.moe_intermediate_size, K = c.num_experts_per_tok;
    m.streams[0] = Dev<float>(W * hc * H);
    m.streams[1] = Dev<float>(W * hc * H);
    if (options.capture_main_hidden) {
        if (options.layers < 4U) throw std::invalid_argument("deepseek41 device forward: capture_main_hidden needs at least 4 layers");
        for (auto& cap : m.capture) cap = Dev<float>(W * hc * H);
    }
    m.x = Dev<float>(W * H);
    m.qr = Dev<float>(W * c.q_lora_rank);
    m.q = Dev<float>(W * c.num_attention_heads * c.head_dim);
    m.kv = Dev<float>(W * c.head_dim);
    m.o = Dev<float>(W * c.num_attention_heads * c.head_dim);
    m.o_low = Dev<float>(W * c.o_groups * c.o_lora_rank);
    m.y = Dev<float>(W * H);
    m.shared = Dev<float>(W * H);
    m.raw = Dev<float>(W * K * H);
    m.low = Dev<float>(W * H);
    m.high = Dev<float>(W * H);
    m.h = Dev<float>(W * H);
    m.logits = Dev<float>(W * c.vocab_size);  // [row][vocab]
    if (m.headout) m.head_rows = Dev<float>(m.trellis_head.size() * W * m.trellis_head.front().first.count);
    const std::size_t cols = m.model.engram.rows_per_token();
    m.erows = Dev<std::uint8_t>(std::size_t{m.model.engram.layer_ids.size()} * W * cols * 264U);
    check(hipHostMalloc(reinterpret_cast<void**>(&m.erows_pinned), m.erows.n, hipHostMallocDefault), "engram staging");
    m.emb = Dev<float>(W * cols * 256U);
    m.ekv = Dev<float>(W * H * (hc + 1U));
    if (options.world == 2U) {  // the exchanged halves of ds41::engram_tp()
        m.ekv_tx = Dev<float>(std::min<std::size_t>(W, 8U) * H * (hc + 1U) / 2U);
        m.ekv_rx = Dev<float>(std::min<std::size_t>(W, 8U) * H * (hc + 1U) / 2U);
    }
    m.scores = Dev<float>(W * c.n_routed_experts);
    m.route_w = Dev<float>(W * K);
    m.ids = Dev<std::uint32_t>(W * K);
    if (options.op_timing) {
        m.op_ids = Dev<std::uint32_t>(std::size_t{options.layers} * max_window_rows * K);
        check(hipMemset(m.op_ids.p, 0xff, m.op_ids.n * 4U), "op timing ids");
    }
    if (const int level = ds41::StepTrace::env_level(); level != 0) m.trace = std::make_unique<ds41::StepTrace>(level, options.rank);
    if (options.op_timing || (m.trace && m.trace->level() == 1)) {
        m.op_events.resize(m.plan.steps.size() + 1U);
        for (auto& e : m.op_events) check(hipEventCreate(&e), "op timing event");
    }
    m.image_row = Dev<std::uint8_t>(W);
    m.image_row.upload(std::vector<std::uint8_t>(W, 0), m.stream);
    m.token = Dev<std::uint32_t>(W);
    m.pos = Dev<std::uint32_t>(W);
    m.mix_attn = Dev<ds41::HcMix>(W);
    m.mix_ffn = Dev<ds41::HcMix>(W);
    m.mix_partial = Dev<float>(ds41::hc_mix_partial_floats(static_cast<std::uint32_t>(W), static_cast<std::uint32_t>(H)));
    m.hc_counters = Dev<std::uint32_t>(ds41::hc_site_counter_count(static_cast<std::uint32_t>(W)));
    m.hc_counters.upload(std::vector<std::uint32_t>(m.hc_counters.n, 0U), m.stream);
    const std::size_t site_width = std::max<std::size_t>({H, c.q_lora_rank, std::size_t{c.num_attention_heads} * c.head_dim,
                                                          std::size_t{c.o_groups} * c.o_lora_rank});
    m.site_codes = Dev<std::int8_t>(W * site_width);
    m.o_codes = Dev<std::int8_t>(W * c.num_attention_heads * c.head_dim);
    m.o_deltas = Dev<float>(W * c.num_attention_heads * c.head_dim / 32U);
    m.o_sums = Dev<std::int32_t>(W * c.num_attention_heads * c.head_dim / 32U);
    m.site_deltas = Dev<float>(W * site_width / 32U);
    m.site_sums = Dev<std::int32_t>(W * site_width / 32U);
    m.codes = Dev<std::int8_t>(W * H);
    m.deltas = Dev<float>(W * H / 32U);
    m.sums = Dev<std::int32_t>(W * H / 32U);
    m.act_codes = Dev<std::int8_t>(W * K * I);
    m.act_deltas = Dev<float>(W * K * I / 32U);
    m.act_sums = Dev<std::int32_t>(W * K * I / 32U);
    m.sh_act_codes = Dev<std::int8_t>(W * I);
    m.sh_act_deltas = Dev<float>(W * I / 32U);
    m.sh_act_sums = Dev<std::int32_t>(W * I / 32U);
    if (!m.native_fp8.empty()) {
        m.fp8_gate = Dev<float>(W * I);
        m.fp8_up = Dev<float>(W * I);
    }
    if (m.trellis) {
        m.tr_gate = Dev<float>(W * I);
        m.tr_up = Dev<float>(W * I);
        m.tr_act = Dev<float>(W * I);
        if (m.shexp_tp && !m.distributed()) m.tr_p1 = Dev<float>(W * H);
    }
    std::uint32_t partial_blocks = 0;
    for (std::uint32_t r = 0; r < 2U; ++r)
        partial_blocks = std::max(partial_blocks, ds41::argmax_partial_blocks(ds41::vocab_slice(c.vocab_size, 2U, r).count));
    m.partial = Dev<ds41::ArgmaxCandidate>(W * partial_blocks);
    m.cand0 = Dev<ds41::ArgmaxCandidate>(W);
    m.cand1 = Dev<ds41::ArgmaxCandidate>(W);
    m.best = Dev<ds41::ArgmaxCandidate>(W);
    m.cand_own = Dev<ds41::ArgmaxCandidate>(W);
    m.cand_other = Dev<ds41::ArgmaxCandidate>(W);
    // two ranks: the split shared expert is opt-in (options.split_shared or STRIX_DS41_SPLIT_SHARED=1). P10 measured it
    // equal to the overlapped form at R1 and 6-7 ms slower per verify window at R >= 2 (its extra exchanges).
    m.split_shared = m.distributed() && (options.split_shared || [] {
        const char* v = std::getenv("STRIX_DS41_SPLIT_SHARED");
        return v && *v == '1';
    }());
    // P13 defaults (after P12): HC_SITE, QA_KV, ENGRAM_ASYNC and COMPRESS_RING on, =0 turns each off.
    m.engram_async = strix::envcfg::flag("STRIX_DS41_ENGRAM_ASYNC", true);
    m.compress_ring = strix::envcfg::flag("STRIX_DS41_COMPRESS_RING", true);
    if (m.split_shared && (I % 64U != 0U || H % 2U != 0U)) throw std::invalid_argument("deepseek41 device forward: shared split geometry");
    if (m.split_shared)
        for (const auto& s : m.plan.steps)
            if (s.op == Op::shared_expert && m.trellis_site(s.weights[0]) != nullptr)
                throw std::invalid_argument("deepseek41 device forward: the trellis shared expert is replicated (no STRIX_DS41_SPLIT_SHARED)");
    m.half_own = Dev<float>(W * H + (m.split_shared ? W * H / 2U : 0U));
    m.half_other = Dev<float>(W * H + (m.split_shared ? W * H / 2U : 0U));
    if (m.split_shared) {
        m.shx_own = Dev<std::uint8_t>(W * I / 2U + W * I / 64U * 4U);
        m.shx_other = Dev<std::uint8_t>(W * I / 2U + W * I / 64U * 4U);
    }
    {
        const std::size_t n = W * K + 1U;
        m.built.experts = Dev<std::uint32_t>(n);
        m.built.begins = Dev<std::uint32_t>(n);
        m.built.rows = Dev<std::uint32_t>(n);
        m.built.slots = Dev<std::uint32_t>(n);
        m.built.order = Dev<std::uint32_t>(n);
        m.built.low_counts = Dev<std::uint32_t>(n);
        m.built.counts = Dev<std::uint32_t>(2);
    }
    for (std::uint32_t r = 1; r <= W; ++r)  // the shared expert as expert 0 of a one-expert window of r rows
        m.shared_routes.emplace(r, std::make_unique<DeviceRoutes>(ds41::build_window_routes(std::vector<std::uint32_t>(r, 0U), r, 1U, 1U), m.stream));
    check(hipStreamSynchronize(m.stream), "setup");
    load_seconds_ = seconds_since(t_load);
}

DeviceForward::~DeviceForward() {
    if (impl_) {
        impl_->trellis_oa.reset();
        impl_->trellis.reset();
    }
    if (impl_ && impl_->slot_pinned != nullptr) (void)hipHostFree(impl_->slot_pinned);
    if (impl_ && impl_->receive_pinned != nullptr) (void)hipHostFree(impl_->receive_pinned);
    if (impl_ && impl_->erows_pinned != nullptr) (void)hipHostFree(impl_->erows_pinned);
    if (impl_)
        for (auto e : impl_->op_events) (void)hipEventDestroy(e);
    if (impl_ && impl_->prefetch_stream != nullptr) {
        (void)hipStreamSynchronize(impl_->prefetch_stream);
        (void)hipStreamDestroy(impl_->prefetch_stream);
        (void)hipEventDestroy(impl_->prefetch_ready);
    }
    if (impl_ && impl_->stream != nullptr) (void)hipStreamDestroy(impl_->stream);
}

const float* DeviceForward::device_logits() const noexcept { return impl_->logits.p; }
std::pair<const float*, std::size_t> DeviceForward::Impl::logits_rows(std::uint32_t slice_first, std::uint32_t vocab) const {
    if (head_rows_valid)
        for (std::size_t k = 0; k < trellis_head.size(); ++k)
            if (const auto& sl = trellis_head[k].first; sl.first == slice_first) return {head_rows.p + k * max_window_rows * sl.count, sl.count};
    return {logits.p + slice_first, vocab};
}
DeviceForward::LogitsRows DeviceForward::device_logits_rows(std::uint32_t first) const {
    const auto [p, stride] = impl_->logits_rows(first, impl_->model.config.vocab_size);
    return {p, stride};
}

std::span<std::uint8_t> DeviceForward::receive_buffer(std::size_t bytes) {
    auto& m = *impl_;
    if (bytes > m.receive_bytes) {
        check(hipStreamSynchronize(m.stream), "receive buffer");  // no upload still reads the old one
        if (m.receive_pinned != nullptr) (void)hipHostFree(m.receive_pinned);
        m.receive_pinned = nullptr;
        m.receive_bytes = 0;
        const std::size_t n = std::max<std::size_t>(bytes, std::size_t{1} << 20U);
        check(hipHostMalloc(reinterpret_cast<void**>(&m.receive_pinned), n, hipHostMallocDefault), "receive buffer");
        m.receive_bytes = n;
    }
    return {m.receive_pinned, bytes};
}

std::size_t DeviceForward::cb4_tensors() const noexcept {
    return impl_->cb4.size() + impl_->uniform.size() + impl_->router_bf16.size() + impl_->f8.size();
}

std::string DeviceForward::overlay_description() const { return impl_->overlay ? impl_->overlay->describe() : std::string{}; }

std::string DeviceForward::storage_description() const {
    const auto& m = *impl_;
    std::string out = "packed sites " + std::to_string(m.cb4.size() + m.uniform.size()) + ", BF16 routers " +
                      std::to_string(m.router_bf16.size()) + ", F8 rows " + std::to_string(m.f8.size());
    if (!m.native_fp8.empty())
        out += ", native FP8 sites " + std::to_string(m.native_fp8.size()) + " (" + m.options.native_fp8.string() + ")";
    if (m.trellis)
        out += ", trellis sites " + std::to_string(m.trellis_sites.size()) + " (" + std::to_string(m.trellis->device_bytes() >> 20U) +
               " MiB, " + m.trellis_pack->describe() + ", " + m.trellis->announced_identity(m.trellis_classes, m.options.trellis_shexp_tp) + ")";
    return out;
}

double DeviceForward::last_stream_seconds() const noexcept { return impl_->stream_seconds; }

std::vector<std::pair<std::string, std::uint64_t>> DeviceForward::state_hashes() const {
    const auto& m = *impl_;
    namespace attn = ds41::attn;
    const auto fnv_bytes = [](const void* data, std::size_t n, std::uint64_t h = 14695981039346656037ULL) {
        const auto* b = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ULL;
        return h;
    };
    std::vector<std::pair<std::string, std::uint64_t>> out;
    for (std::uint32_t l = m.first; l < m.options.layers; ++l) {
        const auto& r = m.roles[l];
        const auto& L = m.attn[l];
        const std::string at = "@" + std::to_string(l);
        const auto ring = download(L.ring.p, L.ring.n, m.stream);
        out.emplace_back("ring" + at, ring_window_hash(ring.data(), m.position));
        if (r.owns_kv) {
            const std::size_t entries = m.position / r.ratio;
            const auto kv = download(L.kv_cache.p, entries * attn::kKvEntryBytes, m.stream);
            const auto ix = download(L.idx_cache.p, entries * attn::kIdxEntryBytes, m.stream);
            out.emplace_back("kv" + at, fnv_bytes(kv.data(), kv.size()));
            out.emplace_back("idx_keys" + at, fnv_bytes(ix.data(), ix.size()));
            if (r.ratio == 2U) {
                const auto carry = download(L.carry[L.carry_cur].p, L.carry[L.carry_cur].n, m.stream);
                const auto valid = download(L.carry_valid[L.carry_cur].p, 1, m.stream);
                out.emplace_back("carry" + at, valid[0] != 0U ? fnv_bytes(carry.data(), carry.size() * 4U) : 0U);
            }
        }
        if (r.owns_index) {  // after step(token) the last row is row 0 of the window
            const auto count = download(L.idx_count.p, 1, m.stream)[0];
            const auto idx = download(L.idx.p, count, m.stream);
            out.emplace_back("topk" + at, fnv_bytes(idx.data(), idx.size() * 4U, count));
        }
    }
    return out;
}

AttentionState DeviceForward::export_state() const {
    const auto& m = *impl_;
    namespace attn = ds41::attn;
    AttentionState s;
    s.position = m.position;
    s.next_token = next_token_;
    s.tokens = m.history;
    s.layers.resize(m.options.layers);  // a TAIL leaves layers [0, first) absent (empty ring)
    for (const auto& src : m.cut.kv_sources) {  // TAIL: its mirrors of the KV sources before first (no ring: still absent)
        if (m.first == 0U) break;
        const auto& L = m.attn[src.layer];
        auto& S = s.layers[src.layer];
        const std::size_t entries = m.position / src.ratio;
        S.owns_kv = true;
        S.ratio = src.ratio;
        S.kv = download(L.kv_cache.p, entries * attn::kKvEntryBytes, m.stream);
        S.idx_keys = download(L.idx_cache.p, entries * attn::kIdxEntryBytes, m.stream);
    }
    for (std::uint32_t l = m.first; l < m.options.layers; ++l) {
        const auto& r = m.roles[l];
        const auto& L = m.attn[l];
        auto& S = s.layers[l];
        S.owns_kv = r.owns_kv;
        S.owns_index = r.owns_index;
        S.ratio = r.ratio;
        S.ring = download(L.ring.p, L.ring.n, m.stream);
        if (r.owns_kv) {
            const std::size_t entries = m.position / r.ratio;
            S.kv = download(L.kv_cache.p, entries * attn::kKvEntryBytes, m.stream);
            S.idx_keys = download(L.idx_cache.p, entries * attn::kIdxEntryBytes, m.stream);
            if (r.ratio == 2U) {
                S.carry = download(L.carry[L.carry_cur].p, L.carry[L.carry_cur].n, m.stream);
                S.carry_valid = download(L.carry_valid[L.carry_cur].p, 1, m.stream)[0] != 0U;
            }
        }
        if (r.owns_index) {  // the last row is row m.rows - 1 of the last window
            const std::uint32_t last = m.rows - 1U;
            const auto count = download(L.idx_count.p + last, 1, m.stream)[0];
            S.topk = download(L.idx.p + std::size_t{last} * attn::kTopK, count, m.stream);
        }
    }
    return s;
}

void DeviceForward::import_state(const AttentionState& s) {
    auto& m = *impl_;
    namespace attn = ds41::attn;
    const auto fail = [](const std::string& what) { throw std::invalid_argument("deepseek41 device forward: import_state: " + what); };
    // A whole forward or a FRONT [0, layers) takes the first `layers` layers of a state with at least as many; a TAIL its
    // layers [first, layers) and the caches of the KV sources before first that its layers read (the mirrors).
    if (s.layers.size() < m.options.layers) fail("the state has " + std::to_string(s.layers.size()) + " layers, the forward " + std::to_string(m.options.layers));
    for (std::uint32_t l = m.first; l < m.options.layers; ++l)
        if (s.layers[l].ring.empty()) fail("layer " + std::to_string(l) + " is absent from the state");
    if (s.position >= m.options.max_positions) fail("position " + std::to_string(s.position) + " leaves no room in max_positions");
    if (s.tokens.size() != s.position) fail("token history length != position");
    if (s.position == 0U) fail("an empty state (start a fresh forward instead)");
    hipStream_t st = m.stream;
    std::vector<std::uint32_t> counts(m.options.layers, 0U);  // host sources of the top-k counts, alive until the sync
    for (const auto& src : m.cut.kv_sources) {  // TAIL: the mirrored caches of the KV sources before first
        if (m.first == 0U) break;
        const auto& r = m.roles[src.layer];
        auto& L = m.attn[src.layer];
        const auto& S = s.layers[src.layer];
        const std::size_t entries = s.position / r.ratio;
        if (!S.owns_kv || S.kv.size() != entries * attn::kKvEntryBytes || S.idx_keys.size() != entries * attn::kIdxEntryBytes)
            fail("the state lacks the caches of KV source " + std::to_string(src.layer) + " (the TAIL's layers read them)");
        check(hipMemsetAsync(L.kv_cache.p, 0, L.kv_cache.n, st), "import kv");
        check(hipMemsetAsync(L.idx_cache.p, 0, L.idx_cache.n, st), "import idx");
        check(hipMemcpyAsync(L.kv_cache.p, S.kv.data(), S.kv.size(), hipMemcpyHostToDevice, st), "import kv");
        check(hipMemcpyAsync(L.idx_cache.p, S.idx_keys.data(), S.idx_keys.size(), hipMemcpyHostToDevice, st), "import idx");
    }
    for (std::uint32_t l = m.first; l < m.options.layers; ++l) {
        const auto& r = m.roles[l];
        auto& L = m.attn[l];
        const auto& S = s.layers[l];
        if (S.owns_kv != r.owns_kv || S.owns_index != r.owns_index || (r.owns_kv && S.ratio != r.ratio)) fail("layer " + std::to_string(l) + " roles differ");
        if (S.ring.size() != L.ring.n) fail("ring size of layer " + std::to_string(l));
        check(hipMemcpyAsync(L.ring.p, S.ring.data(), L.ring.n, hipMemcpyHostToDevice, st), "import ring");
        if (r.owns_kv) {
            const std::size_t entries = s.position / r.ratio;
            if (S.kv.size() != entries * attn::kKvEntryBytes || S.idx_keys.size() != entries * attn::kIdxEntryBytes) fail("cache size of layer " + std::to_string(l));
            check(hipMemsetAsync(L.kv_cache.p, 0, L.kv_cache.n, st), "import kv");
            check(hipMemsetAsync(L.idx_cache.p, 0, L.idx_cache.n, st), "import idx");
            check(hipMemcpyAsync(L.kv_cache.p, S.kv.data(), S.kv.size(), hipMemcpyHostToDevice, st), "import kv");
            check(hipMemcpyAsync(L.idx_cache.p, S.idx_keys.data(), S.idx_keys.size(), hipMemcpyHostToDevice, st), "import idx");
            for (int k = 0; k < 2; ++k) {
                check(hipMemsetAsync(L.carry[k].p, 0, L.carry[k].n * 4U, st), "import carry");
                check(hipMemsetAsync(L.carry_valid[k].p, 0, 4U, st), "import carry");
            }
            L.carry_cur = 0;
            if (r.ratio == 2U && S.carry_valid) {
                if (S.carry.size() != L.carry[0].n) fail("carry size of layer " + std::to_string(l));
                check(hipMemcpyAsync(L.carry[0].p, S.carry.data(), S.carry.size() * 4U, hipMemcpyHostToDevice, st), "import carry");
                check(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(L.carry_valid[0].p), 1, 1, st), "import carry");
            }
        }
        if (r.owns_index) {  // the last row's selection as row 0 (state_hashes reads it; the next step overwrites it)
            if (S.topk.size() > attn::kTopK) fail("top-k size of layer " + std::to_string(l));
            const std::uint32_t count = counts[l] = S.topk_known ? static_cast<std::uint32_t>(S.topk.size()) : 0U;
            check(hipMemsetAsync(L.idx_count.p, 0, L.idx_count.n * 4U, st), "import top-k");
            check(hipMemcpyAsync(L.idx_count.p, &counts[l], 4U, hipMemcpyHostToDevice, st), "import top-k");
            if (count != 0U) check(hipMemcpyAsync(L.idx.p, S.topk.data(), std::size_t{count} * 4U, hipMemcpyHostToDevice, st), "import top-k");
        }
    }
    m.hasher->reset();
    for (const auto t : s.tokens) m.hasher->push(t, m.engram_rows);
    m.history = s.tokens;
    m.position = s.position;
    m.rows = 1;
    next_token_ = s.next_token;
    tokens_.assign(1, s.next_token);
    check(hipStreamSynchronize(st), "import state");
}

const std::vector<float>& DeviceForward::last_logits() const noexcept { return impl_->logits_host; }

const ds41::pipe::BoundaryPlan& DeviceForward::boundary_plan() const {
    if (impl_->cut.cut == 0U) throw std::logic_error("deepseek41 device forward: not a pipeline stage");
    return impl_->cut;
}

ds41::pipe::Boundary DeviceForward::export_boundary() const {
    const auto& m = *impl_;
    namespace pipe = ds41::pipe;
    namespace attn = ds41::attn;
    if (m.first != 0U || m.options.head || m.cut.cut != m.options.layers) throw std::logic_error("deepseek41 device forward: export_boundary needs a FRONT (layers < the model's, head off)");
    const std::uint32_t R = m.rows, p = m.last_position;
    if (m.position != p + R || m.history.size() < m.position) throw std::logic_error("deepseek41 device forward: export_boundary before a step");
    const std::uint32_t H = m.model.config.hidden_size, hc = m.model.config.hc_mult;
    pipe::Boundary b;
    b.position = p;
    b.rows = R;
    b.tokens.assign(m.history.end() - static_cast<std::ptrdiff_t>(R), m.history.end());
    b.stream = download(m.streams[m.cur].p, std::size_t{R} * hc * H, m.stream);
    const auto mixes = download(m.mix_ffn.p, R, m.stream);
    pipe::Section mix{pipe::SectionKind::mix_ffn, m.cut.cut - 1U, 0U, R, {}};
    mix.bytes.resize(std::size_t{R} * sizeof(ds41::HcMix));
    std::memcpy(mix.bytes.data(), mixes.data(), mix.bytes.size());
    b.sections.push_back(std::move(mix));
    for (const auto& src : m.cut.kv_sources) {
        const auto& L = m.attn[src.layer];
        const std::uint32_t first = p / src.ratio, end = (p + R) / src.ratio, n = end - first;
        pipe::Section k{pipe::SectionKind::kv_entries, src.layer, first, n, {}};
        if (n != 0U) {
            const auto kv = download(L.kv_cache.p + std::size_t{first} * attn::kKvEntryBytes, std::size_t{n} * attn::kKvEntryBytes, m.stream);
            const auto ix = download(L.idx_cache.p + std::size_t{first} * attn::kIdxEntryBytes, std::size_t{n} * attn::kIdxEntryBytes, m.stream);
            k.bytes = kv;
            k.bytes.insert(k.bytes.end(), ix.begin(), ix.end());
        }
        b.sections.push_back(std::move(k));
    }
    for (const auto& src : m.cut.topk_sources) {
        const auto& L = m.attn[src.layer];
        const auto counts = download(L.idx_count.p, R, m.stream);
        const auto ids = download(L.idx.p, std::size_t{R} * attn::kTopK, m.stream);
        pipe::Section k{pipe::SectionKind::topk, src.layer, 0U, R, {}};
        const auto put = [&](std::uint32_t v) {
            for (int i = 0; i < 4; ++i) k.bytes.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
        };
        for (std::uint32_t r = 0; r < R; ++r) {
            put(counts[r]);
            for (std::uint32_t j = 0; j < counts[r]; ++j) put(ids[std::size_t{r} * attn::kTopK + j]);
        }
        b.sections.push_back(std::move(k));
    }
    if (m.cut.candidate_source >= 0) {
        const std::uint32_t count = (p + R) / m.cut.candidate_ratio;
        const std::uint32_t stride = (count + m.geo.candidate_block - 1U) / m.geo.candidate_block;
        pipe::Section k{pipe::SectionKind::candidates, static_cast<std::uint32_t>(m.cut.candidate_source), stride, R, {}};
        if (stride != 0U) k.bytes = download(m.candidates.p, std::size_t{R} * stride, m.stream);
        b.sections.push_back(std::move(k));
    }
    return b;
}

void DeviceForward::exchange_host(std::span<const std::byte> out, std::span<std::byte> in) {
    auto& m = *impl_;
    if (!m.distributed()) throw std::logic_error("deepseek41 device forward: exchange_host needs world 2");
    if (!in.empty() && out.size() != in.size()) throw std::invalid_argument("deepseek41 device forward: exchange_host sizes differ");
    // STRIX_DS41_HOST_EXCHANGE=1: host to host through the communicator's pinned buffers (no device round trip; the same
    // messages on the wire, so the peer may use either path). Default: through the device control buffer.
    static const bool host_only = [] {
        const char* v = std::getenv("STRIX_DS41_HOST_EXCHANGE");
        return v != nullptr && std::string_view(v) == "1";
    }();
    if (host_only) {
        for (std::size_t at = 0; at < out.size(); at += Impl::control_chunk) {
            const std::size_t n = std::min(Impl::control_chunk, out.size() - at);
            m.options.communicator->exchange_host(out.subspan(at, n), in.empty() ? std::span<std::byte>{} : in.subspan(at, n),
                                                  dist::Tag{static_cast<std::uint32_t>(dist::TagKind::control), 0xC1U, at});
        }
        return;
    }
    if (m.control.p == nullptr) m.control = Dev<std::uint8_t>(2U * Impl::control_chunk);
    for (std::size_t at = 0; at < out.size(); at += Impl::control_chunk) {
        const std::size_t n = std::min(Impl::control_chunk, out.size() - at);
        check(hipMemcpyAsync(m.control.p, out.data() + at, n, hipMemcpyHostToDevice, m.stream), "control upload");
        m.options.communicator->exchange(m.stream, m.control.p, m.control.p + Impl::control_chunk, n,
                                         dist::Tag{static_cast<std::uint32_t>(dist::TagKind::control), 0xC1U, at});
        if (!in.empty())  // an empty `in`: the peer's bytes are not wanted
            check(hipMemcpyAsync(in.data() + at, m.control.p + Impl::control_chunk, n, hipMemcpyDeviceToHost, m.stream), "control download");
        check(hipStreamSynchronize(m.stream), "control");
    }
}

std::vector<float> DeviceForward::window_logits() {
    auto& m = *impl_;
    const auto& c = m.model.config;
    const std::uint32_t R = static_cast<std::uint32_t>(tokens_.size()), V = c.vocab_size;
    if (R == 0U || !m.options.head) throw std::logic_error("deepseek41 device forward: window_logits needs a step with the head");
    auto all = download(m.logits.p, std::size_t{R} * V, m.stream);
    if (m.head_rows_valid)  // trellis head with HEADOUT at R > 1: this process's slices are in head_rows ([slice][row][count])
        for (std::size_t k = 0; k < m.trellis_head.size(); ++k) {
            const auto& sl = m.trellis_head[k].first;
            const auto rows = download(m.head_rows.p + k * max_window_rows * sl.count, std::size_t{R} * sl.count, m.stream);
            for (std::uint32_t r = 0; r < R; ++r)
                std::copy_n(rows.data() + std::size_t{r} * sl.count, sl.count, all.data() + std::size_t{r} * V + sl.first);
        }
    if (!m.distributed()) return all;
    const auto& own = m.head_slice;
    const auto other = ds41::vocab_slice(V, 2U, 1U - m.options.rank);
    if (other.count != own.count) throw std::logic_error("deepseek41 device forward: unequal vocabulary slices");
    std::vector<float> mine(std::size_t{R} * own.count), theirs(std::size_t{R} * own.count);
    for (std::uint32_t r = 0; r < R; ++r)
        std::copy_n(all.data() + std::size_t{r} * V + own.first, own.count, mine.data() + std::size_t{r} * own.count);
    exchange_host(std::as_bytes(std::span<const float>(mine)), std::as_writable_bytes(std::span<float>(theirs)));
    for (std::uint32_t r = 0; r < R; ++r)
        std::copy_n(theirs.data() + std::size_t{r} * own.count, own.count, all.data() + std::size_t{r} * V + other.first);
    return all;
}

std::uint64_t DeviceForward::weight_bytes() const noexcept {
    std::uint64_t n = 0;
    for (const auto& [name, d] : impl_->weights) n += d.n;
    return n;
}

const std::vector<float>& DeviceForward::step(std::uint32_t token) {
    const std::uint32_t one[1] = {token};
    (void)step_window(one);
    return impl_->logits_host;
}

std::uint32_t DeviceForward::next_position() const noexcept { return impl_->position; }
void* DeviceForward::stream() const noexcept { return impl_->stream; }
std::array<const float*, 3> DeviceForward::captured_streams() const {
    if (!impl_->options.capture_main_hidden) throw std::logic_error("deepseek41 device forward: capture_main_hidden is off");
    return {impl_->capture[0].p, impl_->capture[1].p, impl_->capture[2].p};
}

void DeviceForward::reset() {
    auto& m = *impl_;
    for (auto& L : m.attn) {
        const auto zero = [&](void* p, std::size_t bytes) {
            if (p != nullptr && bytes != 0U) check(hipMemsetAsync(p, 0, bytes, m.stream), "reset");
        };
        zero(L.ring.p, L.ring.n);
        zero(L.kv_cache.p, L.kv_cache.n);
        zero(L.idx_cache.p, L.idx_cache.n);
        for (int k = 0; k < 2; ++k) {
            zero(L.carry[k].p, L.carry[k].n * 4U);
            zero(L.carry_valid[k].p, L.carry_valid[k].n * 4U);
        }
        L.carry_cur = 0;
        zero(L.idx.p, L.idx.n * 4U);
        zero(L.idx_count.p, L.idx_count.n * 4U);
    }
    check(hipMemsetAsync(m.candidates.p, 0, m.candidates.n, m.stream), "reset");
    check(hipStreamSynchronize(m.stream), "reset");
    m.hasher->reset();
    m.history.clear();
    m.position = 0;
    m.rows = 1;
    tokens_.clear();
}

void DeviceForward::commit_window(std::uint32_t kept) {
    auto& m = *impl_;
    const std::uint32_t R = m.rows;
    if (kept == 0U || kept > R) throw std::invalid_argument("deepseek41 device forward: commit_window keeps 1..R rows");
    if (kept == R) return;
    const std::uint32_t first = m.position - R, last = first + kept - 1U;
    // The ratio-2 compressor's pending group after row kept - 1: that row's (kv, score) when its position opens a
    // group, else none (exactly what the compress kernel leaves after a last row at that position).
    if (m.compress_ring) {
        const std::size_t D = ds41::attn::kHeadDim;
        CarryRestores a{};
        std::uint32_t n = 0;
        const auto launch = [&] {
            if (n == 0U) return;
            restore_carries<<<n, 256, 0, m.stream>>>(a, last % 2U == 0U ? 1U : 0U);
            check(hipGetLastError(), "rollback carry");
            n = 0;
        };
        for (std::uint32_t l = m.first; l < m.options.layers; ++l) {
            const auto& r = m.roles[l];
            if (!r.owns_kv || r.ratio != 2U) continue;
            auto& L = m.attn[l];
            a.layer[n++] = {reinterpret_cast<std::uint32_t*>(L.carry[L.carry_cur].p), L.carry_valid[L.carry_cur].p,
                            reinterpret_cast<const std::uint32_t*>(L.ckv.p + (kept - 1U) * D),
                            reinterpret_cast<const std::uint32_t*>(L.cscore.p + (kept - 1U) * D)};
            if (n == max_carry_restores) launch();
        }
        launch();
    }
    for (std::uint32_t l = m.first; l < m.options.layers && !m.compress_ring; ++l) {
        const auto& r = m.roles[l];
        if (!r.owns_kv || r.ratio != 2U) continue;
        auto& L = m.attn[l];
        float* state = L.carry[L.carry_cur].p;
        const std::size_t D = ds41::attn::kHeadDim;
        if (last % 2U == 0U) {
            check(hipMemcpyAsync(state, L.ckv.p + (kept - 1U) * D, D * 4U, hipMemcpyDeviceToDevice, m.stream), "rollback carry");
            check(hipMemcpyAsync(state + D, L.cscore.p + (kept - 1U) * D, D * 4U, hipMemcpyDeviceToDevice, m.stream), "rollback carry");
            check(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(L.carry_valid[L.carry_cur].p), 1, 1, m.stream), "rollback carry");
        } else {
            check(hipMemsetAsync(state, 0, 2U * D * 4U, m.stream), "rollback carry");
            check(hipMemsetAsync(L.carry_valid[L.carry_cur].p, 0, 4U, m.stream), "rollback carry");
        }
    }
    m.hasher->truncate(first + kept);
    m.history.resize(first + kept);
    m.position = first + kept;
    m.rows = kept;
    tokens_.resize(kept);
    if (!m.compress_ring) check(hipStreamSynchronize(m.stream), "rollback");
}

void DeviceForward::collect_op_timing(const std::vector<double>& exchange) {
    auto& m = *impl_;
    const auto& c = m.model.config;
    const std::uint32_t R = m.rows, K = c.num_experts_per_tok;
    const auto ids = download(m.op_ids.p, std::size_t{m.options.layers} * max_window_rows * K, m.stream);
    const bool tp = m.distributed() && m.options.tp_attention;
    const std::uint32_t per_rank = m.expert_end - m.expert_first;
    op_timing_.assign(exchange.size(), OpTiming{});
    for (std::size_t i = 0; i < exchange.size(); ++i) {
        const auto& s = m.plan.steps[i];
        auto& o = op_timing_[i];
        o.op = s.op;
        o.layer = s.layer;
        o.site = s.site;
        float ms = 0.0F;
        check(hipEventElapsedTime(&ms, m.op_events[i], m.op_events[i + 1]), "op timing");
        o.ms = ms;
        o.exchange_ms = exchange[i] * 1e3;
        std::uint32_t used = 0;  // this rank's experts the step's rows selected
        if (s.op == Op::routed_experts && s.layer >= 0) {
            std::vector<bool> seen(per_rank, false);
            for (std::uint32_t k = 0; k < R * K; ++k) {
                const auto e = ids[(std::size_t{static_cast<std::uint32_t>(s.layer)} * max_window_rows * K) + k];
                if (e >= m.expert_first && e < m.expert_end && !seen[e - m.expert_first]) {
                    seen[e - m.expert_first] = true;
                    ++used;
                }
            }
        }
        for (const auto& w : s.weights) {
            if (const auto t = m.trellis_bytes.find(w); t != m.trellis_bytes.end()) {  // trellis site: this rank's prepared bytes
                o.weight_bytes += t->second;
                continue;
            }
            const auto it = m.weights.find(w);
            if (it == m.weights.end()) continue;  // Engram rows (read per token from the file) and streamed experts
            std::uint64_t b = it->second.n;
            if (w.ends_with("_exps.weight")) b = per_rank != 0U ? b / per_rank * used : 0U;
            else if (w == "token_embd.weight") b = b / c.vocab_size * R;
            else if (tp && (w.ends_with("attn_q_b.weight") || w.ends_with("attn_output_a.weight") || w.ends_with("attn_output_b.weight")))
                b /= 2U;
            else if (m.split_shared && w.ends_with("_shexp.weight")) b /= 2U;
            o.weight_bytes += b;
        }
    }
}

const std::vector<std::uint32_t>& DeviceForward::step_window(std::span<const std::uint32_t> window) {
    if (impl_->first != 0U) throw std::logic_error("deepseek41 device forward: a pipeline TAIL steps on boundaries (step_boundary)");
    return run_step(window, nullptr);
}

const std::vector<std::uint32_t>& DeviceForward::step_boundary(const ds41::pipe::Boundary& b) {
    auto& m = *impl_;
    if (m.first == 0U) throw std::logic_error("deepseek41 device forward: step_boundary needs first_layer > 0");
    if (b.position != m.position)
        throw std::invalid_argument("deepseek41 device forward: boundary at position " + std::to_string(b.position) + ", the TAIL is at " +
                                    std::to_string(m.position));
    if (b.rows == 0U || b.rows > max_window_rows) throw std::invalid_argument("deepseek41 device forward: a step has 1..8 rows");
    ds41::pipe::validate_boundary(b, m.cut, m.geo);
    return run_step(b.tokens, &b);
}

const std::vector<std::uint32_t>& DeviceForward::run_step(std::span<const std::uint32_t> window, const ds41::pipe::Boundary* boundary) {
    auto& m = *impl_;
    const auto& c = m.model.config;
    const auto R = static_cast<std::uint32_t>(window.size());
    if (R == 0U || R > max_window_rows) throw std::invalid_argument("deepseek41 device forward: a step has 1..8 rows");
    for (const auto t : window)
        if (t >= c.vocab_size) throw std::invalid_argument("deepseek41 device forward: token outside the vocabulary");
    if (m.position + R > m.options.max_positions) throw std::out_of_range("deepseek41 device forward: context full");
    const std::uint32_t H = c.hidden_size, hc = c.hc_mult, K = c.num_experts_per_tok, I = c.moe_intermediate_size;
    const std::uint32_t V = c.vocab_size, W = max_window_rows;
    const std::uint32_t position = m.position;  // of row 0
    const bool hashing = m.options.op_hashes && R == 1U;
    m.rows = R;
    hipStream_t st = m.stream;
    ds41::StepTrace* const tr = m.trace.get();
    if (tr) tr->begin(R);
    std::vector<std::uint32_t> pos32(R);
    std::vector<std::uint64_t> pos64(R);
    for (std::uint32_t r = 0; r < R; ++r) pos32[r] = position + r, pos64[r] = position + r;
    m.token.upload(std::vector<std::uint32_t>(window.begin(), window.end()), st);
    m.pos.upload(pos32, st);
    m.pos64.upload(pos64, st);
    namespace attn = ds41::attn;
    std::vector<attn::RopeRow> rope_rows(3U * W);
    for (std::uint32_t r = 0; r < R; ++r) {
        rope_rows[r] = attn::rope_row(attn::RopeKind::window, position + r);
        rope_rows[W + r] = attn::rope_row(attn::RopeKind::compressed, position + r);
        rope_rows[2U * W + r] = attn::rope_row(attn::RopeKind::compressed, (position + r) / 2U * 2U);
    }
    m.rope_rows.upload(rope_rows, st);
    const attn::RopeRow* rope_window = m.rope_rows.p;
    const attn::RopeRow* rope_compressed = m.rope_rows.p + W;
    const attn::RopeRow* rope_group = m.rope_rows.p + 2U * W;
    const auto t_step = std::chrono::steady_clock::now();
    if (tr) tr->mark("upload_inputs");
    const auto& e = m.model.engram;
    const std::uint32_t cols = e.rows_per_token(), engram_layers = m.hasher->layers();
    std::vector<std::uint64_t> engram_rows(std::size_t{R} * engram_layers * cols);  // [row][engram layer][column]
    // Engram layers of the range [first, layers): layer_ids [engram_first, engram_in_range)
    std::uint32_t engram_first = 0, engram_in_range = 0;
    while (engram_in_range < m.hasher->layers() && e.layer_ids[engram_in_range] < m.options.layers) ++engram_in_range;
    while (engram_first < engram_in_range && e.layer_ids[engram_first] < m.first) ++engram_first;
    const std::uint32_t engram_count = engram_in_range - engram_first;
    for (std::uint32_t r = 0; r < R; ++r) {
        m.hasher->push(window[r], m.engram_rows);
        m.history.push_back(window[r]);
        std::copy(m.engram_rows.begin(), m.engram_rows.end(), engram_rows.begin() + static_cast<std::ptrdiff_t>(std::size_t{r} * m.engram_rows.size()));
    }
    if (tr) tr->mark("engram_hash");
    engram_seconds_ = 0.0;
    m.stream_seconds = 0.0;
    // STRIX_DS41_ENGRAM_ASYNC: layer li's batch is its rows in [row][column] order, read straight into its pinned
    // staging (the layout of the synchronous path); Op::engram of the layer waits for it, then uploads it. The previous
    // step ended synchronised, so no upload still reads the staging. engram_io is declared last: on unwinding it joins
    // the reads before the requests and promises they use go away.
    const bool engram_async = m.engram_async && m.engram_reader && engram_count != 0U;
    std::vector<std::vector<ds41::EngramRowRequest>> layer_requests(engram_async ? engram_in_range : 0U);
    std::vector<std::promise<void>> engram_ready(layer_requests.size());
    std::vector<std::future<void>> engram_wait;
    double engram_read_seconds = 0.0;
    std::future<void> engram_io;
    if (engram_async) {
        for (std::uint32_t li = 0; li < engram_in_range; ++li) {
            engram_wait.push_back(engram_ready[li].get_future());
            if (li < engram_first) {  // a TAIL never runs the layers before its range
                engram_ready[li].set_value();
                continue;
            }
            for (std::uint32_t r = 0; r < R; ++r)
                for (std::uint32_t k = 0; k < cols; ++k)
                    layer_requests[li].push_back({li, engram_rows[(std::size_t{r} * engram_layers + li) * cols + k]});
        }
        engram_io = std::async(std::launch::async, [&] {
            const auto t0 = std::chrono::steady_clock::now();
            std::uint32_t li = engram_first;
            try {
                for (; li < engram_in_range; ++li) {
                    std::uint8_t* out = m.erows_pinned + std::size_t{li} * R * cols * 264U;
                    m.engram_reader->read(layer_requests[li], {out, layer_requests[li].size() * 264U});
                    engram_ready[li].set_value();
                }
            } catch (...) {
                for (; li < engram_in_range; ++li) engram_ready[li].set_exception(std::current_exception());
            }
            engram_read_seconds = seconds_since(t0);
        });
    } else if (m.engram_reader) {  // every Engram layer of the range, every row, in one batch from the NVMe (O_DIRECT)
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<ds41::EngramRowRequest> requests;
        for (std::uint32_t r = 0; r < R; ++r)
            for (std::uint32_t li = 0; li < engram_layers; ++li)
                if (e.layer_ids[li] < m.options.layers && e.layer_ids[li] >= m.first)
                    for (std::uint32_t k = 0; k < cols; ++k)
                        requests.push_back({li, engram_rows[(std::size_t{r} * engram_layers + li) * cols + k]});
        m.engram_bytes.resize(requests.size() * 264U);
        m.engram_reader->read(requests, m.engram_bytes);
        engram_seconds_ = seconds_since(t0);
        if (tr) tr->mark("engram_read");
    }
    if (engram_count != 0U && !engram_async) {  // rows of every Engram layer of the range: [layer][row][column], one asynchronous upload
        for (std::uint32_t li = engram_first; li < engram_in_range; ++li)
            for (std::uint32_t r = 0; r < R; ++r) {
                std::uint8_t* out = m.erows_pinned + ((std::size_t{li} * R + r) * cols) * 264U;
                if (m.engram_reader) {  // the reader's order: [row][engram layer of the range][column]
                    std::copy_n(m.engram_bytes.data() + (std::size_t{r} * engram_count + (li - engram_first)) * cols * 264U, std::size_t{cols} * 264U,
                                out);
                } else {
                    const auto table = m.host_tensor("blk." + std::to_string(e.layer_ids[li]) + ".engram_embd.weight");
                    for (std::uint32_t k = 0; k < cols; ++k)
                        std::copy_n(table.data() + engram_rows[(std::size_t{r} * engram_layers + li) * cols + k] * 264U, 264, out + std::size_t{k} * 264U);
                }
            }
        if (tr) tr->mark("engram_pack");
        const std::size_t skip = std::size_t{engram_first} * R * cols * 264U;  // layers before the range: never read
        check(hipMemcpyAsync(m.erows.p + skip, m.erows_pinned + skip, std::size_t{engram_count} * R * cols * 264U, hipMemcpyHostToDevice, st),
              "engram rows");
        if (tr) tr->mark("engram_h2d_enqueue");
    }
    if (boundary != nullptr) {  // the TAIL's input: the stream after layer first - 1, its FFN mix and the cut's sections
        namespace pipe = ds41::pipe;
        const auto& b = *boundary;
        check(hipMemcpyAsync(m.streams[m.cur].p, b.stream_data(), b.stream_floats() * 4U, hipMemcpyHostToDevice, st), "boundary stream");
        const auto* mix = b.find(pipe::SectionKind::mix_ffn, m.first - 1U);
        static_assert(sizeof(ds41::HcMix) == (2U * ds41::hc_copies + ds41::hc_copies * ds41::hc_copies) * 4U, "HcMix is pre, post, comb");
        check(hipMemcpyAsync(m.mix_ffn.p, mix->bytes.data(), mix->bytes.size(), hipMemcpyHostToDevice, st), "boundary mix");
        m.identity_pre = false;  // layer first - 1's FFN ran: its pre collapses layer first's attention input
        for (const auto& src : m.cut.kv_sources) {
            const auto* k = b.find(pipe::SectionKind::kv_entries, src.layer);
            auto& L = m.attn[src.layer];
            const std::size_t n = k->count;
            if ((std::size_t{k->first} + n) * attn::kKvEntryBytes > L.kv_cache.n)
                throw std::out_of_range("deepseek41 device forward: boundary KV entries past the cache");
            if (n != 0U) {
                check(hipMemcpyAsync(L.kv_cache.p + std::size_t{k->first} * attn::kKvEntryBytes, k->bytes.data(), n * attn::kKvEntryBytes,
                                     hipMemcpyHostToDevice, st),
                      "boundary kv");
                check(hipMemcpyAsync(L.idx_cache.p + std::size_t{k->first} * attn::kIdxEntryBytes, k->bytes.data() + n * attn::kKvEntryBytes,
                                     n * attn::kIdxEntryBytes, hipMemcpyHostToDevice, st),
                      "boundary index keys");
            }
        }
        m.boundary_topk.assign(m.cut.topk_sources.size(), {});
        for (std::size_t i = 0; i < m.cut.topk_sources.size(); ++i) {
            const auto layer = m.cut.topk_sources[i].layer;
            const auto* k = b.find(pipe::SectionKind::topk, layer);
            auto& host = m.boundary_topk[i];  // [counts W][ids R x topk], alive until the sync below
            host.assign(W + std::size_t{R} * attn::kTopK, 0U);
            std::size_t at = 0;
            const auto word = [&] {
                std::uint32_t v;
                std::memcpy(&v, k->bytes.data() + at, 4);
                at += 4U;
                return v;
            };
            for (std::uint32_t r = 0; r < R; ++r) {
                const auto n = host[r] = word();
                for (std::uint32_t j = 0; j < n; ++j) host[W + std::size_t{r} * attn::kTopK + j] = word();
            }
            auto& L = m.attn[layer];
            check(hipMemcpyAsync(L.idx_count.p, host.data(), std::size_t{W} * 4U, hipMemcpyHostToDevice, st), "boundary top-k");
            check(hipMemcpyAsync(L.idx.p, host.data() + W, std::size_t{R} * attn::kTopK * 4U, hipMemcpyHostToDevice, st), "boundary top-k");
        }
        if (m.cut.candidate_source >= 0) {
            const auto* k = b.find(pipe::SectionKind::candidates, static_cast<std::uint32_t>(m.cut.candidate_source));
            if (k->bytes.size() > m.candidates.n) throw std::out_of_range("deepseek41 device forward: boundary candidates past the pool");
            if (!k->bytes.empty())
                check(hipMemcpyAsync(m.candidates.p, k->bytes.data(), k->bytes.size(), hipMemcpyHostToDevice, st), "boundary candidates");
        }
    }
    check(hipStreamSynchronize(st), "step inputs");  // the uploads above read host temporaries (and the pinned staging)
    if (tr) tr->mark("sync_inputs");
    hashes_.clear();
    stream_hashes_.clear();
    if (m.options.record_routes) routes_.assign(m.options.layers, {});
    timing_.assign(m.options.layers, LayerTiming{});
    std::int32_t timed_layer = -1;
    auto t_layer = std::chrono::steady_clock::now();
    double comm_at_layer = m.comm_seconds();
    const auto stream_now = [&] { return m.streams[m.cur].p; };
    auto t_head = t_step;
    const std::uint32_t capture_first = m.options.layers >= 3U ? m.options.layers - 3U : 0U;

    // The shared expert of an FFN site (its A8 input already in codes/deltas).
    // Storage format of a weight: Q8_0, or the requant overlays' packed forms (CB4 table, U5/U6), same products.
    const auto fmt_of = [&](const std::string& n) {
        return m.cb4.contains(n) ? ds41::WeightFormat::cb4
            : m.uniform.contains(n) ? (m.uniform.at(n) == 5U ? ds41::WeightFormat::u5 : ds41::WeightFormat::u6)
                                    : ds41::WeightFormat::q8_0;
    };
    // Native FP8 site product of `name` rows [first, first + count) with the A8 rows codes/deltas (width values, stride width).
    const auto fp8_site = [&](const std::string& name, std::uint32_t width, std::uint32_t first, std::uint32_t count, const std::int8_t* codes,
                              const float* deltas, float* out, std::uint32_t out_stride) {
        ds41::Q8A8View v;
        m.site_weight(v, name);
        v.width = width;
        v.first = first;
        v.count = count;
        v.per_group = count;
        v.codes = codes;
        v.deltas = deltas;
        v.rows = R;
        v.x_stride = width;
        v.out_stride = out_stride;
        v.out = out;
        ds41::enqueue_q8_0_a8(v, st);
    };
    // Native FP8 shared expert (fp8_swiglu_identity): gate and up rows [first, first + count) as products, then the clamped
    // SwiGLU and its A8 rows (count values each) into act_codes / act_deltas / act_sums.
    const auto fp8_shared_gate_up = [&](const Step& s, std::uint32_t first, std::uint32_t count, std::int8_t* act_codes, float* act_deltas,
                                        std::int32_t* act_sums) {
        fp8_site(s.weights[0], H, first, count, m.codes.p, m.deltas.p, m.fp8_gate.p, count);
        fp8_site(s.weights[1], H, first, count, m.codes.p, m.deltas.p, m.fp8_up.p, count);
        ds41::enqueue_swiglu_a8(m.fp8_gate.p, m.fp8_up.p, R, count, static_cast<float>(c.swiglu_limit), act_codes, act_deltas, act_sums, st);
    };
    // The shared expert of an FFN site (its A8 input already in codes/deltas), in the stored format of its tensors.
    const auto enqueue_shared = [&](const Step& s) {
        if (const auto* gate = m.trellis_site(s.weights[0])) {  // trellis (ds41-trellis-k5-v1:shexp): F32 x, no A8
            if (const auto it = m.trellis_shared_halves.find(s.layer); it != m.trellis_shared_halves.end()) {
                if (ds41::trellis_shared_grouped_enabled() && I == 2304U) {
                    // STRIX_DS41_SHARED_GROUPED: two SwiGLU pair launches write tr_act directly (same products and bits)
                    ds41::enqueue_trellis_shared_swiglu(it->second.data(), m.x.p, m.tr_act.p, R,
                                                        static_cast<float>(c.swiglu_limit), st);
                } else {
                    for (std::size_t k = 0; k < 4U; ++k)
                        ds41::enqueue_trellis_linear(*it->second[k], m.x.p,
                            (k < 2U ? m.tr_gate.p : m.tr_up.p) + (k % 2U) * 1152U, R, H, I, false, st);
                    ds41::enqueue_swiglu_f32(m.tr_gate.p, m.tr_up.p, R, I, static_cast<float>(c.swiglu_limit), m.tr_act.p, st);
                }
            } else {
                ds41::enqueue_trellis_linear(*gate, m.x.p, m.tr_gate.p, R, H, I, false, st);
                ds41::enqueue_trellis_linear(*m.trellis_site(s.weights[1]), m.x.p, m.tr_up.p, R, H, I, false, st);
                ds41::enqueue_swiglu_f32(m.tr_gate.p, m.tr_up.p, R, I, static_cast<float>(c.swiglu_limit), m.tr_act.p, st);
            }
            ds41::enqueue_trellis_shared_down(*m.trellis_site(s.weights[2]), m.tr_act.p, I, m.shared.p, H, R, st);
            return;
        }
        if (m.native_fp8.contains(s.weights[0])) {
            fp8_shared_gate_up(s, 0U, I, m.sh_act_codes.p, m.sh_act_deltas.p, m.sh_act_sums.p);
            fp8_site(s.weights[2], I, 0U, H, m.sh_act_codes.p, m.sh_act_deltas.p, m.shared.p, H);
            return;
        }
        const DeviceRoutes& r = *m.shared_routes.at(R);
        const auto gu_fmt = fmt_of(s.weights[0]), down_fmt = fmt_of(s.weights[2]);
        if (fmt_of(s.weights[1]) != gu_fmt || (gu_fmt == ds41::WeightFormat::cb4 && m.cb4.at(s.weights[0]) != m.cb4.at(s.weights[1])))
            throw std::logic_error("deepseek41 device forward: shared gate and up must share their CB4 table");
        const auto gate_row = row_bytes(gu_fmt, H), down_row = row_bytes(down_fmt, I);
        ds41::GateUpView gu;
        if (gu_fmt == ds41::WeightFormat::cb4) std::copy(m.cb4.at(s.weights[0]).begin(), m.cb4.at(s.weights[0]).end(), gu.table);
        gu.gate = m.dev(s.weights[0]);
        gu.up = m.dev(s.weights[1]);
        gu.expert_stride_bytes = gate_row * I;
        gu.row_bytes = gate_row;
        gu.hidden = H;
        gu.inter = I;
        gu.routes = r.gate_up();
        gu.codes = m.codes.p;
        gu.deltas = m.deltas.p;
        gu.limit = static_cast<float>(c.swiglu_limit);
        gu.act_codes = m.sh_act_codes.p;
        gu.act_deltas = m.sh_act_deltas.p;
        gu.act_sums = m.sh_act_sums.p;
        ds41::enqueue_gate_up(gu_fmt, gu, st);
        ds41::RoutesProductView dv;
        if (down_fmt == ds41::WeightFormat::cb4) std::copy(m.cb4.at(s.weights[2]).begin(), m.cb4.at(s.weights[2]).end(), dv.table);
        dv.weights = m.dev(s.weights[2]);
        dv.expert_stride_bytes = down_row * H;
        dv.row_bytes = down_row;
        dv.width = I;
        dv.outputs = H;
        dv.routes = r.down();
        dv.codes = m.sh_act_codes.p;
        dv.deltas = m.sh_act_deltas.p;
        dv.out = m.shared.p;
        ds41::enqueue_routes_product(down_fmt, dv, st);
    };
    const bool overlap_shared = strix::envcfg::flag("STRIX_DS41_EP_OVERLAP", true);
    const Step* deferred_shared = nullptr;
    std::vector<double> op_exchange;  // op_timing: host seconds in the communicator per plan step
    const bool step_events = m.options.op_timing || (tr && tr->level() == 1);  // one device event after every plan step
    if (step_events) {
        op_exchange.reserve(m.plan.steps.size());
        check(hipEventRecord(m.op_events[0], st), "op timing");
    }
    if (tr) tr->first_event(st);
    // STRIX_DS41_HC_SITE=1: each hyper-connection boundary (the previous sublayer's expansion, this site's mixes, the
    // collapse, the RMSNorm and the A8 of x) as one enqueue_hc_site, same bits. The expansion waits for the next
    // site (or the Engram injection, or the head). Off with op hashes or layer timing, which need every step's output.
    static const bool hc_site_flag = strix::envcfg::flag("STRIX_DS41_HC_SITE", true);
    // STRIX_DS41_QA_KV=1: q_a and kv in one launch over one A8 of the attention input, same bits.
    // QA_KV off by default since P15 (P13/P14: -0.15 to -0.2 ms at R1 without it, 0 at R6); =1 turns it on
    static const bool qa_kv_flag = strix::envcfg::flag("STRIX_DS41_QA_KV", false);
    // STRIX_DS41_ATTN_A8=1: the attention combine writes the A8 of its output for o_a, and o_a the A8 of its output
    // for o_b (in the product kernel when one block owns whole groups), same bits.
    static const bool attn_a8 = strix::envcfg::flag("STRIX_DS41_ATTN_A8", false);
    const bool fuse_hc = hc_site_flag && !hashing && !m.options.layer_timing;
    const bool fuse_qa_kv = qa_kv_flag && !hashing;
    struct PendingPost { bool on = false; const float* sub = nullptr; const float* residual = nullptr; const ds41::HcMix* mixes = nullptr; };
    PendingPost post;              // an expansion not yet enqueued; stream_now() is already its output
    std::int64_t capture_at = -1;  // the layer whose input is captured once the pending expansion is stored
    const Step* site_mix = nullptr;
    // STRIX_DS41_LATE_HC_LAYERS (P17.5, default all; none = off; measured R1 -0.2, R6 -0.35, batch N=2..8 -0.6..-1.5 ms): on those layers, with HC_SITE and tensor-parallel attention, the attention site
    // skips its mixes (the collapse that feeds q_a uses the previous FFN site's mixes; these are first read by
    // hc_post(attn)), and they are enqueued as a mix-only site over the same stored stream right after the o_b exchange
    // starts, so they run during it. hc_post(attn) is the latest point. Same kernels and partitions, so the same bits.
    const Step* late_mix = nullptr;
    const auto enqueue_late_mix = [&] {
        if (!late_mix) return;
        ds41::HcSiteView v;
        v.rows = R;
        v.hidden = H;
        v.stream = stream_now();
        v.fn = m.devh(late_mix->weights[0]);
        v.base = m.devf(late_mix->weights[1]);
        v.scale = m.devf(late_mix->weights[2]);
        v.iters = c.hc_sinkhorn_iters;
        v.hc_eps = static_cast<float>(c.hc_eps);
        v.mix_norm_eps = 1e-20F;
        v.mixes_out = m.mix_attn.p;
        v.partial = m.mix_partial.p;
        v.counters = m.hc_counters.p;
        ds41::enqueue_hc_site(v, st);
        late_mix = nullptr;
    };
    // Residual boundary fusions; each flag is independent (QNORM_A8 off by default, ROUTES on by default).
    static const bool qnorm_a8 = strix::envcfg::flag("STRIX_DS41_R1FUSE_QNORM_A8", false);
    const bool router_routes = ds41::r1fuse_routes();  // STRIX_DS41_R1FUSE_ROUTES: default on, =0 off
    bool q_a8 = false, routes_ready = false;
    const auto route_view = [&] {
        ds41::BuildRoutesView bv;
        bv.ids = m.ids.p;
        bv.rows = R;
        bv.top_k = K;
        bv.split = m.expert_tp ? c.n_routed_experts : m.plan.expert_split;  // expert TP: every slot in the rank's fold
        bv.first = m.expert_first;
        bv.end = m.expert_end;
        bv.experts = m.built.experts.p;
        bv.begins = m.built.begins.p;
        bv.rows_out = m.built.rows.p;
        bv.slots = m.built.slots.p;
        bv.order = m.built.order.p;
        bv.low_counts = m.built.low_counts.p;
        bv.counts = m.built.counts.p;
        return bv;
    };
    bool x_a8 = false;  // codes/deltas/sums hold the A8 of the current x
    bool kv_done = false;
    bool o_a_trellis = false;  // this layer's o_a ran from the trellis pack (no A8 of o_low for a Q8 o_b)
    const auto stored_post = [&] {  // the pending expansion is in stream_now() (or enqueued before anything after it)
        post.on = false;
        if (capture_at >= 0) {
            check(hipMemcpyAsync(m.capture[static_cast<std::size_t>(capture_at) - capture_first].p, stream_now(), std::size_t{R} * hc * H * 4U,
                                 hipMemcpyDeviceToDevice, st),
                  "capture");
            capture_at = -1;
        }
    };
    const auto flush_post = [&] {
        if (!post.on) return;
        ds41::enqueue_hc_expand(post.sub, post.residual, post.mixes, R, H, stream_now(), st);
        stored_post();
    };
    // One hc site: the pending expansion (if any), the mixes of `mix` (if any) into mixes_out, then the collapse by
    // collapse_mixes, the RMSNorm by norm_weight into out and its A8 into codes / deltas / sums.
    const auto enqueue_site = [&](const Step* mix, ds41::HcMix* mixes_out, const ds41::HcMix* collapse_mixes, const float* norm_weight,
                                  float* out, std::int8_t* codes, float* deltas, std::int32_t* sums) {
        ds41::HcSiteView v;
        v.rows = R;
        v.hidden = H;
        if (post.on) {
            v.sub = post.sub;
            v.residual = post.residual;
            v.expand_mixes = post.mixes;
        }
        v.stream = stream_now();
        if (mix) {
            v.fn = m.devh(mix->weights[0]);
            v.base = m.devf(mix->weights[1]);
            v.scale = m.devf(mix->weights[2]);
            v.iters = c.hc_sinkhorn_iters;
            v.hc_eps = static_cast<float>(c.hc_eps);
            v.mix_norm_eps = 1e-20F;
            v.mixes_out = mixes_out;
            v.partial = m.mix_partial.p;
            v.counters = m.hc_counters.p;
        }
        v.collapse_mixes = collapse_mixes;
        v.norm_weight = norm_weight;
        v.norm_eps = 1e-20F;
        v.out = out;
        v.out_stride = H;
        v.codes = codes;
        v.deltas = deltas;
        v.sums = sums;
        v.a8_stride = H;
        ds41::enqueue_hc_site(v, st);
        if (post.on) stored_post();
    };
    // STRIX_DS41_DECODE_PREFETCH: the weights of the steps after `from` (whole tensors in step order, the last one cut)
    // up to prefetch_bytes, on the prefetch stream once the main one reaches this point; routed experts (rows unknown),
    // embeddings, trellis sites (their own pack) and, TP, wo_b (K-halves of every row) are skipped.
    static const bool past_experts = strix::envcfg::flag("STRIX_DS41_PREFETCH_PAST_EXPERTS", false);
    const auto prefetch_after = [&](const Step& from) {
        if (m.prefetch_bytes == 0U) return;
        check(hipEventRecord(m.prefetch_ready, st), "prefetch event");
        check(hipStreamWaitEvent(m.prefetch_stream, m.prefetch_ready, 0), "prefetch wait");
        const bool tp = m.options.tp_attention;
        std::uint64_t left = m.prefetch_bytes * R;
        TouchSpans spans{};
        const bool batched = ds41::touch_batch();
        const auto flush = [&] {
            if (spans.count) touch_spans_kernel<<<320, 256, 0, m.prefetch_stream>>>(spans, m.never.p);
            spans.count = 0;
        };
        const auto touch = [&](const uint4* p, std::uint64_t n) {
            if (!batched) {
                touch_kernel<<<320, 256, 0, m.prefetch_stream>>>(p, n, m.never.p);
                return;
            }
            spans.p[spans.count] = p;
            spans.end[spans.count] = (spans.count ? spans.end[spans.count - 1U] : 0U) + n;
            if (++spans.count == 16U) flush();
        };
        for (auto i = static_cast<std::size_t>(&from - m.plan.steps.data()) + 1U; i < m.plan.steps.size() && left; ++i)
            for (const auto& w : m.plan.steps[i].weights) {
                if (!left) break;
                // Stop at the routed experts: their ~100+ MB stream evicts from the MALL whatever was read ahead past them;
                // and skip the shared expert when it runs later, inside the half-sum exchange (after the routed experts).
                // N8 with both read ahead in the o_b window: +3.3 % (p6). STRIX_DS41_PREFETCH_PAST_EXPERTS=1: the old scope.
                if (w.ends_with("_exps.weight") && !past_experts) { left = 0; break; }
                if (w.ends_with("_exps.weight") || w.find("engram_embd") != std::string::npos || w == "token_embd.weight") continue;
                if ((overlap_shared && !hashing) && !past_experts && w.find("_shexp.") != std::string::npos) continue;
                // trellis sites (their own pack, this process's prepared slices): the library allocation of each, when the
                // linked strix-trellis exposes it (strix_weights_device_span)
                {
                    const auto& si = m.plan.steps[i];
                    std::vector<const ds41::TrellisSite*> sites;
                    if (!m.trellis_attn.empty() && si.layer >= 0 && w == si.weights[0]) {
                        const auto& ta = m.trellis_attn[static_cast<std::size_t>(si.layer)];
                        if (si.op == Op::q_b) sites = ta.q_b;
                        else if (si.op == Op::o_a) sites = ta.o_a;
                        else if (si.op == Op::o_b) sites = ta.o_b;
                    }
                    if (sites.empty())
                        if (const auto* t = m.trellis_site(w)) sites.push_back(t);
                    if (!sites.empty()) {
                        for (const auto* t : sites) {
                            const void* data = nullptr;
                            std::uint64_t n = 0;
                            if (!left || !ds41::trellis_site_span(*t, &data, &n)) break;
                            const std::uint64_t size = std::min<std::uint64_t>(n, left);
                            if (size / 16U) touch(static_cast<const uint4*>(data), size / 16U);
                            left -= size;
                        }
                        continue;
                    }
                }
                if (tp && w.ends_with("attn_output_b.weight")) continue;
                const auto it = m.weights.find(w);
                if (it == m.weights.end()) continue;
                const auto& d = it->second;
                // TP: this rank's rows of q_b and wo_a (the upper or lower half)
                const bool half = tp && (w.ends_with("attn_q_b.weight") || w.ends_with("attn_output_a.weight"));
                const std::uint8_t* base = d.p + (half ? d.n / 2U * m.options.rank : 0U);
                const std::uint64_t size = std::min<std::uint64_t>(half ? d.n / 2U : d.n, left);
                if (size / 16U)
                    touch(reinterpret_cast<const uint4*>(base), size / 16U);
                left -= size;
            }
        flush();
        check(hipGetLastError(), "prefetch");
    };
    using SplitRawExchange = xport::Ds41RawSlotsExchange<
        std::remove_reference_t<decltype(*m.options.communicator)>, hipStream_t, dist::Tag>;
    std::optional<SplitRawExchange> split_raw_exchange;
    for (const auto& s : m.plan.steps) {
        const double comm_before = step_events || tr ? m.comm_seconds() : 0.0;
        const auto layer = static_cast<std::uint32_t>(std::max(s.layer, 0));
        const std::string p = "blk." + std::to_string(layer) + ".";
        if (m.options.layer_timing && s.layer >= 0 && s.layer != timed_layer) {  // the previous layer ended synchronised
            timed_layer = s.layer;
            t_layer = std::chrono::steady_clock::now();
            comm_at_layer = m.comm_seconds();
        }
        if (s.op == Op::head) t_head = std::chrono::steady_clock::now();
        switch (s.op) {
        case Op::embed:
            ds41::enqueue_embedding(m.devh("token_embd.weight"), m.token.p, R, H, stream_now(), st);
            m.pre_carry = ds41::hc_identity_pre();
            m.identity_pre = true;
            break;
        case Op::engram: {
            flush_post();  // the injection updates the expanded stream
            const std::size_t li = static_cast<std::size_t>(std::find(e.layer_ids.begin(), e.layer_ids.end(), layer) - e.layer_ids.begin());
            if (engram_async) {  // this layer's rows only: the later Engram layers' reads overlap the layers before them
                const auto t_wait = tr ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                engram_wait.at(li).get();
                if (tr) tr->sync("engram_wait", t_wait, static_cast<int>(layer));
                const std::size_t at = li * R * cols * 264U;
                check(hipMemcpyAsync(m.erows.p + at, m.erows_pinned + at, std::size_t{R} * cols * 264U, hipMemcpyHostToDevice, st),
                      "engram rows");
            }
            ds41::enqueue_engram_rows_bf16(m.erows.p + li * R * cols * 264U, R * cols, m.emb.p, st);  // uploaded at step start (async: just above)
            ds41::F16RowsView v;
            if (m.f8.contains(p + "engram_kv.weight")) v.weight_f8 = m.dev(p + "engram_kv.weight");
            else v.weight = m.devh(p + "engram_kv.weight");
            v.width = cols * 256U;
            v.first = 0;
            v.count = H * (hc + 1U);
            v.x = m.emb.p;
            v.rows = R;
            v.x_stride = v.width;
            v.out_stride = v.count;
            v.out = m.ekv.p;
            // ds41::engram_tp(): this rank's half of the output rows (the same per-row arithmetic), the peer's by exchange
            if (const std::uint32_t n = v.count, half = n / 2U;
                m.distributed() && ds41::engram_tp(R) && v.weight_f8 && R <= 8U && n % 2U == 0U) {
                v.first = m.options.rank * half;
                v.count = half;
                v.out = m.ekv.p + v.first;
                v.out_stride = n;
                ds41::enqueue_f16_rows(v, st);
                const std::uint32_t total = R * half, blocks = (total + 255U) / 256U;
                engram_pack<<<blocks, 256, 0, st>>>(m.ekv.p, m.ekv_tx.p, R, n, half, v.first);
                check(hipGetLastError(), "engram_pack");
                xport::ds41_exchange_raw_slots(*m.options.communicator, st, m.ekv_tx.p, m.ekv_rx.p, std::size_t{total} * 4U,
                                              dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), 0x10000U | static_cast<std::uint32_t>(layer), position}, [] {});
                engram_scatter<<<blocks, 256, 0, st>>>(m.ekv_rx.p, m.ekv.p, R, n, half, (1U - m.options.rank) * half);
                check(hipGetLastError(), "engram_scatter");
            } else {
                ds41::enqueue_f16_rows(v, st);
            }
            ds41::enqueue_engram_inject(stream_now(), m.ekv.p, m.devf(p + "engram_q_norm.weight"), m.devf(p + "engram_k_norm.weight"), R, H,
                                        1e-20F, st);
            break;
        }
        case Op::hc_mix: {
            if (fuse_hc) {  // part of the site's enqueue_hc_site at its RMSNorm
                site_mix = &s;
                break;
            }
            ds41::HcMixView v;
            v.stream = stream_now();
            v.fn = m.devh(s.weights[0]);
            v.base = m.devf(s.weights[1]);
            v.scale = m.devf(s.weights[2]);
            v.rows = R;
            v.hidden = H;
            v.iters = c.hc_sinkhorn_iters;
            v.hc_eps = static_cast<float>(c.hc_eps);
            v.norm_eps = 1e-20F;
            v.out = s.site == "attn" ? m.mix_attn.p : m.mix_ffn.p;
            v.partial = m.mix_partial.p;
            ds41::enqueue_hc_mixes(v, st);
            break;
        }
        case Op::hc_pre: {
            if (fuse_hc) break;  // part of the site's enqueue_hc_site at its RMSNorm
            const ds41::HcMix* mixes = s.site == "attn" ? (m.identity_pre ? nullptr : m.mix_ffn.p) : m.mix_attn.p;
            ds41::enqueue_hc_collapse(stream_now(), mixes, R, H, m.x.p, st);
            break;
        }
        case Op::rms_norm: {
            if (fuse_hc && (s.site == "attn" || s.site == "ffn")) {
                const bool at = s.site == "attn";
                const ds41::HcMix* collapse = at ? (m.identity_pre ? nullptr : m.mix_ffn.p) : m.mix_attn.p;
                const bool late = at && site_mix && m.distributed() && m.options.tp_attention && late_hc_layer(layer);
                enqueue_site(late ? nullptr : site_mix, at ? m.mix_attn.p : m.mix_ffn.p, collapse, m.devf(s.weights[0]), m.x.p, m.codes.p,
                             m.deltas.p, m.sums.p);
                if (late) late_mix = site_mix;
                site_mix = nullptr;
                x_a8 = true;
                break;
            }
            float* v = s.site == "q" ? m.qr.p : s.site == "kv" ? m.kv.p : m.x.p;
            const std::uint32_t n = s.site == "q" ? c.q_lora_rank : s.site == "kv" ? c.head_dim : H;
            if (s.site == "q" && qnorm_a8 && (m.trellis_attn.empty() || m.trellis_attn[layer].q_b.empty())) {
                ds41::NormView nv;
                nv.x = v; nv.x_stride = n; nv.weight = m.devf(s.weights[0]); nv.rows = R; nv.n = n;
                nv.out = v; nv.out_stride = n; nv.codes = m.site_codes.p; nv.deltas = m.site_deltas.p;
                nv.sums = m.site_sums.p; nv.a8_stride = n;
                ds41::enqueue_norm(nv, st);  // same in-place norm, also emits the next projection's A8
                q_a8 = true;
            } else ds41::enqueue_rms_norm(v, m.devf(s.weights[0]), R, n, n, 1e-20F, v, st);
            break;
        }
        case Op::q_a: case Op::q_b: case Op::kv: case Op::o_a: case Op::o_b: {
            if (s.op == Op::kv && kv_done) {  // enqueued with q_a
                kv_done = false;
                break;
            }
            if (s.op == Op::q_a || s.op == Op::kv) {
                if (const auto* site = m.trellis_site(s.weights[0])) {
                    const auto* kv_site = m.trellis_site(p + "attn_kv.weight");
                    if (s.op == Op::q_a && kv_site && ds41::trellis_qakv_pair_enabled()) {
                        ds41::enqueue_trellis_qakv(*site, *kv_site, m.x.p, m.qr.p, m.kv.p, R, st);
                        kv_done = true;
                    } else {
                        ds41::enqueue_trellis_linear(*site, m.x.p, s.op == Op::q_a ? m.qr.p : m.kv.p,
                            R, H, site->rows, false, st);
                    }
                    break;
                }
            }
            if (!m.trellis_attn.empty() && (s.op == Op::q_b || s.op == Op::o_a || s.op == Op::o_b)) {
                // ds41-trellis-k5-v1 q_b / o_a / o_b: F32 inputs, no A8; the linears of this process in rank-slice order
                const auto& ta = m.trellis_attn[layer];
                const bool tp = m.distributed() && m.options.tp_attention;
                const std::uint32_t q_rows = c.num_attention_heads / 2U * c.head_dim;
                const std::uint32_t o_width = c.num_attention_heads * c.head_dim / c.o_groups, o_half = c.o_groups * c.o_lora_rank / 2U;
                if (s.op == Op::q_b && !ta.q_b.empty()) {
                    const std::size_t n = ta.q_b.size() * q_rows;  // q [R][n]: rank slices side by side
                    for (std::size_t k = 0; k < ta.q_b.size(); ++k)
                        ds41::enqueue_trellis_linear(*ta.q_b[k], m.qr.p, m.q.p + k * q_rows, R, c.q_lora_rank, n, false, st);
                    break;
                }
                if (s.op == Op::o_a && !ta.o_a.empty()) {
                    const std::size_t in_width = ta.o_a.size() * o_width, out_width = ta.o_a.size() * c.o_lora_rank;
                    if (m.trellis_oa)  // the same library calls per row, one stage + one scatter (ds41_trellis_group.hpp)
                        m.trellis_oa->enqueue(ta.o_a, m.o.p, in_width, m.o_low.p, out_width, R, st);
                    else
                        for (std::size_t g = 0; g < ta.o_a.size(); ++g)
                            ds41::enqueue_trellis_linear(*ta.o_a[g], m.o.p + g * o_width, m.o_low.p + g * c.o_lora_rank, R, in_width, out_width, false,
                                                         st);
                    o_a_trellis = true;
                    break;
                }
                if (s.op == Op::o_b && !ta.o_b.empty()) {
                    // y = h0 + h1 (one FP32 add per value): TP the rank's half, the peer's by exchange, add_halves; one process
                    // h0 into y, then y = y + h1 (accumulate), the same bits
                    if (tp && ready_detail::trellis_ob())
                        if (const auto stage = m.options.communicator->prepare_staged_ready(st, std::size_t{R} * H * 4U); stage.data) {
                            // STRIX_DS41_TRELLIS_OB_READY (default, =0 off): the device half copied into the TX view by one kernel that publishes
                            // it, the add straight from the RX view (no SDMA copies or events); same products, same add
                            const std::uint32_t n = R * H;
                            ds41::enqueue_trellis_linear(*ta.o_b[0], m.o_low.p, m.half_own.p, R, o_half, H, false, st);
                            ready_detail::copy_publish(m.half_own.p, stage, n, st);
                            xport::ds41_exchange_raw_slots_staged<true>(*m.options.communicator, st, stage.data, std::size_t{R} * H * 4U,
                                dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, position}, [&] {
                                enqueue_late_mix();
                                prefetch_after(s);  // the read-ahead of the next sites, as on the plain trellis path
                            }, [&](const auto& peer) {
                                const auto* other = static_cast<const float*>(peer.data);
                                ready_detail::add(m.options.rank == 0U ? m.half_own.p : other, m.options.rank == 0U ? other : m.half_own.p, m.y.p, n,
                                                  peer, st);
                            });
                            break;
                        }
                    if (tp) {
                        // the late hc mix runs while the halves are exchanged, as on the Q8 path (same products, same add order);
                        // no send staging: it may be host memory, which the trellis plans do not take as output
                        const auto tag = dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, position};
                        ds41::enqueue_trellis_linear(*ta.o_b[0], m.o_low.p, m.half_own.p, R, o_half, H, false, st);
                        xport::ds41_exchange_raw_slots(*m.options.communicator, st, m.half_own.p, m.half_other.p,
                            std::size_t{R} * H * 4U, tag, [&] {
                            enqueue_late_mix();
                            prefetch_after(s);
                        });
                        const float* h0 = m.options.rank == 0U ? m.half_own.p : m.half_other.p;
                        const float* h1 = m.options.rank == 0U ? m.half_other.p : m.half_own.p;
                        const std::uint32_t n = R * H;
                        add_halves<<<(n + 255U) / 256U, 256, 0, st>>>(h0, h1, m.y.p, n);
                        check(hipGetLastError(), "add_halves");
                    } else {
                        for (std::size_t k = 0; k < 2U; ++k)
                            ds41::enqueue_trellis_linear(*ta.o_b[k], m.o_low.p + k * o_half, m.y.p, R, 2U * o_half, H, k == 1U, st);
                    }
                    break;
                }
            }
            const float* in = nullptr;
            ds41::Q8A8View v;
            m.site_weight(v, s.weights[0]);
            v.first = 0;
            v.rows = R;
            std::uint32_t in_width = 0;
            if (s.op == Op::q_a) { in = m.x.p; in_width = H; v.width = H; v.count = c.q_lora_rank; v.out = m.qr.p; }
            const bool tp = m.distributed() && m.options.tp_attention;
            const std::uint32_t tp_heads = c.num_attention_heads / 2U, tp_groups = c.o_groups / 2U;
            if (s.op == Op::q_b) {
                in = m.qr.p;
                in_width = c.q_lora_rank;
                v.width = c.q_lora_rank;
                v.count = (tp ? tp_heads : c.num_attention_heads) * c.head_dim;  // TP: rows of heads [32r, 32r + 32)
                v.first = tp ? m.options.rank * tp_heads * c.head_dim : 0U;
                v.out = m.q.p;
            }
            if (s.op == Op::kv) { in = m.x.p; in_width = H; v.width = H; v.count = c.head_dim; v.out = m.kv.p; }
            if (s.op == Op::o_a) {
                in = m.o.p;
                in_width = (tp ? tp_heads : c.num_attention_heads) * c.head_dim;  // TP: this rank's heads only
                v.width = c.num_attention_heads * c.head_dim / c.o_groups;
                v.count = (tp ? tp_groups : c.o_groups) * c.o_lora_rank;
                v.first = tp ? m.options.rank * tp_groups * c.o_lora_rank : 0U;  // TP: rows of o-groups [4r, 4r + 4)
                v.per_group = c.o_lora_rank;  // output o reads group o / o_lora_rank of the attention output
                v.out = m.o_low.p;
            }
            if (s.op == Op::o_b) {
                // wo_b as two K-halves h0 (columns 0..4095) + h1 (4096..8191), the declared order of every run. One process:
                // h0 into y, then h1 accumulated. TP: this rank's half into half_own, the peer's by exchange, y = h0 + h1.
                const std::uint32_t full = c.o_groups * c.o_lora_rank, half = full / 2U;
                in = m.o_low.p;
                in_width = tp ? half : full;
                if (!attn_a8 || o_a_trellis) ds41::enqueue_quantize_a8(in, in_width, R, m.site_codes.p, m.site_deltas.p, m.site_sums.p, st);
                v.width = half;
                v.row_width = full;
                v.count = H;
                v.per_group = H;
                v.x_stride = in_width;
                v.out_stride = H;
                if (tp) {
                    v.k_offset = m.options.rank * half;
                    v.codes = m.site_codes.p;
                    v.deltas = m.site_deltas.p;
                    if (const std::uint32_t ob_k = ob_split(H); ob_k > 1U) {
                        const std::uint32_t ob_c = H / ob_k;
                        const std::size_t ob_chunk = std::size_t{R} * ob_c;
                        const auto product = [&](std::uint32_t k) {
                            auto w = v;
                            w.first = k * ob_c;
                            w.count = ob_c;
                            w.per_group = ob_c;  // one input group (the check needs per_group == count with row_width)
                            w.out_stride = ob_c;
                            w.out = m.half_own.p + k * ob_chunk;
                            ds41::enqueue_q8_0_a8(w, st);
                        };
                        const auto add = [&](std::uint32_t k) {
                            const float* own = m.half_own.p + k * ob_chunk;
                            const float* other = m.half_other.p + k * ob_chunk;
                            const auto n = static_cast<std::uint32_t>(ob_chunk);
                            add_halves_cols<<<(n + 255U) / 256U, 256, 0, st>>>(m.options.rank == 0U ? own : other, m.options.rank == 0U ? other : own, m.y.p, R, ob_c,
                                                                              k * ob_c, H);
                            check(hipGetLastError(), "add_halves_cols");
                        };
                        product(0U);
                        for (std::uint32_t k = 0; k < ob_k; ++k) {
                            xport::ds41_exchange_raw_slots(*m.options.communicator, st, m.half_own.p + k * ob_chunk,
                                m.half_other.p + k * ob_chunk, ob_chunk * 4U, dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, position}, [&] {
                                if (k == 0U) enqueue_late_mix();
                                if (k == 0U) prefetch_after(s);
                                if (k + 1U < ob_k) product(k + 1U);
                                if (k > 0U) add(k - 1U);
                            });
                        }
                        add(ob_k - 1U);
                        break;
                    }
                    if (const auto stage = m.options.communicator->prepare_staged_ready(st, std::size_t{R} * H * 4U); stage.data) {
                        auto* own = static_cast<float*>(stage.data);
                        v.out = own;
                        ready_detail::product(v, stage, st);
                        xport::ds41_exchange_raw_slots_staged<true>(*m.options.communicator, st, stage.data, std::size_t{R} * H * 4U,
                            dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, position}, [&] {
                            enqueue_late_mix();
                            prefetch_after(s);
                        }, [&](const auto& peer) {
                            const auto* other = static_cast<const float*>(peer.data);
                            const std::uint32_t n = R * H;
                            ready_detail::add(m.options.rank == 0U ? own : other, m.options.rank == 0U ? other : own,
                                              m.y.p, n, peer, st);
                        });
                        break;
                    }
                    if (auto* staged = static_cast<float*>(m.options.communicator->send_staging(std::size_t{R} * H * 4U))) {
                        v.out = staged;
                        ds41::enqueue_q8_0_a8(v, st);
                        xport::ds41_exchange_raw_slots_staged<false>(*m.options.communicator, st, staged,
                            std::size_t{R} * H * 4U, dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, position}, [&] {
                            enqueue_late_mix();
                            prefetch_after(s);
                        }, [&](const void* raw_peer) {
                                const auto* peer = static_cast<const float*>(raw_peer);
                            const std::uint32_t n = R * H;
                            add_halves<<<(n + 255U) / 256U, 256, 0, st>>>(m.options.rank == 0U ? staged : peer, m.options.rank == 0U ? peer : staged, m.y.p, n);
                            check(hipGetLastError(), "add_halves");
                        });
                        break;
                    }
                    v.out = m.half_own.p;
                    ds41::enqueue_q8_0_a8(v, st);
                    xport::ds41_exchange_raw_slots(*m.options.communicator, st, m.half_own.p, m.half_other.p,
                        std::size_t{R} * H * 4U, dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, position}, [&] {
                        enqueue_late_mix();
                        prefetch_after(s);
                    });
                    const float* h0 = m.options.rank == 0U ? m.half_own.p : m.half_other.p;
                    const float* h1 = m.options.rank == 0U ? m.half_other.p : m.half_own.p;
                    const std::uint32_t n = R * H;
                    add_halves<<<(n + 255U) / 256U, 256, 0, st>>>(h0, h1, m.y.p, n);
                    check(hipGetLastError(), "add_halves");
                } else {
                    for (std::uint32_t part = 0; part < 2U; ++part) {
                        v.k_offset = part * half;
                        v.codes = m.site_codes.p + part * half;
                        v.deltas = m.site_deltas.p + part * half / 32U;
                        v.accumulate = part == 1U;
                        v.out = m.y.p;
                        ds41::enqueue_q8_0_a8(v, st);
                    }
                }
                break;
            }
            if (v.per_group == 0U) v.per_group = v.count;
            v.x_stride = in_width;
            v.out_stride = v.count;
            if (s.op == Op::q_a || s.op == Op::kv) {
                // the attention input's A8 in codes / deltas: from the hc site, or quantized once here for q_a and kv
                if (!x_a8 && fuse_qa_kv) {
                    ds41::enqueue_quantize_a8(in, in_width, R, m.codes.p, m.deltas.p, m.sums.p, st);
                    x_a8 = true;
                }
                if (x_a8) {
                    v.codes = m.codes.p;
                    v.deltas = m.deltas.p;
                    if (fuse_qa_kv && s.op == Op::q_a && !m.trellis_site(p + "attn_kv.weight")) {
                        ds41::Q8A8View kv;
                        m.site_weight(kv, p + "attn_kv.weight");
                        kv.first = 0;
                        kv.rows = R;
                        kv.width = H;
                        kv.count = c.head_dim;
                        kv.per_group = kv.count;
                        kv.codes = m.codes.p;
                        kv.deltas = m.deltas.p;
                        kv.x_stride = H;
                        kv.out_stride = kv.count;
                        kv.out = m.kv.p;
                        ds41::enqueue_q8_0_a8_pair(v, kv, st);
                        kv_done = true;
                    } else {
                        ds41::enqueue_q8_0_a8(v, st);
                    }
                    break;
                }
            }
            if (attn_a8 && s.op == Op::o_a) {  // its input's A8 from the attention combine; its output's A8 for o_b
                v.codes = m.o_codes.p;
                v.deltas = m.o_deltas.p;
                v.a8_codes = m.site_codes.p;
                v.a8_deltas = m.site_deltas.p;
                v.a8_sums = m.site_sums.p;
                ds41::enqueue_q8_0_a8(v, st);
                break;
            }
            if (s.op != Op::q_b || !q_a8)
                ds41::enqueue_quantize_a8(in, in_width, R, m.site_codes.p, m.site_deltas.p, m.site_sums.p, st);
            q_a8 = false;
            v.codes = m.site_codes.p;
            v.deltas = m.site_deltas.p;
            ds41::enqueue_q8_0_a8(v, st);
            break;
        }
        // ---- attention piece (ds41_attention kernels); F16 projections feeding it on the layer piece's F16 rows ----
        case Op::window_write: {
            if (attn::hip::attention_window_fold()) break;   // STRIX_DS41_ATTN_WINDOW_FOLD: the attention kernel inserts the rows
            const auto& r = m.roles[layer];
            attn::hip::WindowView v{};
            v.kv = m.kv.p;
            v.norm = m.devf(s.weights[0]);
            v.rope = r.rope == attn::RopeKind::window ? rope_window : rope_compressed;
            v.pos = m.pos64.p;
            v.rows = R;
            v.capacity = AttentionHost::ring_capacity;
            v.ring = m.attn[layer].ring.p;
            attn::hip::enqueue_window_write(v, st);
            break;
        }
        case Op::compress: {
            const auto& r = m.roles[layer];
            auto& L = m.attn[layer];
            const auto f16_proj = [&](const std::string& w, float* out) {  // compressor projection on x
                ds41::F16RowsView v;
                v.weight = m.devh(w);
                v.width = H;
                v.first = 0;
                v.count = attn::kHeadDim;
                v.x = m.x.p;
                v.rows = R;
                v.x_stride = H;
                v.out_stride = attn::kHeadDim;
                v.out = out;
                return v;
            };
            if (r.ratio == 2U)  // one launch with STRIX_DS41_F16_PAIR=1
                ds41::enqueue_f16_rows_pair(f16_proj(p + "attn_compressor_kv.weight", L.ckv.p),
                                            f16_proj(p + "attn_compressor_gate.weight", L.cscore.p), st);
            else ds41::enqueue_f16_rows(f16_proj(p + "attn_compressor_kv.weight", L.ckv.p), st);
            attn::hip::CompressView v{};
            v.ratio = r.ratio;
            v.rows = R;
            v.kv = L.ckv.p;
            v.score = L.cscore.p;
            v.norm = m.devf(p + "attn_compressor_norm.weight");
            v.wk = m.devh(p + "indexer.attn_k.weight");
            v.knorm = m.devf(p + "indexer.k_norm.weight");
            v.rope = r.ratio == 2U ? rope_group : rope_compressed;  // the group's first position
            v.pos = m.pos64.p;
            v.state_in = L.carry[L.carry_cur].p;
            v.state_valid_in = L.carry_valid[L.carry_cur].p;
            v.state_out = L.carry[1 - L.carry_cur].p;
            v.state_valid_out = L.carry_valid[1 - L.carry_cur].p;
            v.kv_cache = L.kv_cache.p;
            v.idx_cache = L.idx_cache.p;
            attn::hip::enqueue_compress(v, st);
            L.carry_cur = 1 - L.carry_cur;
            break;
        }
        case Op::index: {
            const auto& r = m.roles[layer];
            auto& L = m.attn[layer];
            const std::uint32_t ratio = m.roles[static_cast<std::size_t>(r.kv_source)].ratio;
            const std::uint32_t count = (position + R) / ratio;  // of the last row; every row's count <= this
            if (count == 0U) {  // nothing compressed yet: an empty selection, as on the host
                check(hipMemsetAsync(L.idx_count.p, 0, R * 4U, st), "empty top-k");
                break;
            }
            if ((position + 1U) / ratio == 0U)
                throw std::logic_error("deepseek41 device forward: a window must not start before the first compressed entry");
            attn::hip::SelectView sv{};
            sv.rows = R;
            sv.ratio = ratio;
            sv.max_count = count;  // also the candidate pool's row stride: source and users share the ratio
            sv.topk = attn::kTopK;
            sv.cand_blocks = attn::kCandBlocks;
            sv.block_size = attn::kCandBlockSize;
            sv.pool = r.candidate_source ? attn::Pool::candidate_source : r.candidate_user ? attn::Pool::candidate_user : attn::Pool::plain;
            sv.pos = m.pos64.p;
            sv.scores = m.index_scores.p;
            sv.candidates = m.candidates.p;
            sv.keys = m.select_keys.p;
            sv.idx = L.idx.p;
            sv.idx_count = L.idx_count.p;
            if (count <= attn::kTopK && attn::hip::index_direct()) {  // every position selected: no projections, no scores
                attn::hip::enqueue_select_direct(sv, st);
                break;
            }
            ds41::F16RowsView qv;
            if (m.f8.contains(p + "indexer.attn_q_b.weight")) qv.weight_f8 = m.dev(p + "indexer.attn_q_b.weight");
            else qv.weight = m.devh(p + "indexer.attn_q_b.weight");
            qv.width = c.q_lora_rank;
            qv.first = 0;
            qv.count = attn::kIdxHeads * attn::kIdxDim;
            qv.x = m.qr.p;
            qv.rows = R;
            qv.x_stride = c.q_lora_rank;
            qv.out_stride = qv.count;
            qv.out = m.iq.p;
            ds41::F16RowsView wv;
            wv.weight = m.devh(p + "indexer.proj.weight");
            wv.width = H;
            wv.first = 0;
            wv.count = attn::kIdxHeads;
            wv.x = m.x.p;
            wv.rows = R;
            wv.x_stride = H;
            wv.out_stride = attn::kIdxHeads;
            wv.out = m.iw.p;
            ds41::enqueue_f16_rows_pair(qv, wv, st);  // one launch with STRIX_DS41_F16_PAIR=1 (F16 q_b)
            attn::hip::IndexView iv{};
            iv.rows = R;
            iv.ratio = ratio;
            iv.max_count = count;
            iv.q = m.iq.p;
            iv.w = m.iw.p;
            iv.rope = r.rope == attn::RopeKind::window ? rope_window : rope_compressed;
            iv.pos = m.pos64.p;
            iv.idx_cache = m.attn[static_cast<std::size_t>(r.kv_source)].idx_cache.p;
            iv.scores = m.index_scores.p;
            iv.qprep = m.index_qprep.p;
            if (sv.pool == attn::Pool::candidate_user && count > attn::kCandBlocks * attn::kCandBlockSize && attn::hip::index_candidate_filter()) {
                iv.candidates = m.candidates.p;
                iv.block_size = attn::kCandBlockSize;
                sv.compact = true;
            }
            attn::hip::enqueue_index_scores(iv, st);
            attn::hip::enqueue_select(sv, st);
            break;
        }
        case Op::attention: {
            const auto& r = m.roles[layer];
            attn::hip::AttentionView v{};
            v.rows = R;
            v.capacity = AttentionHost::ring_capacity;
            v.topk = attn::kTopK;
            v.q = m.q.p;
            v.sink = m.devf(s.weights[0]);
            v.rope = r.rope == attn::RopeKind::window ? rope_window : rope_compressed;
            v.pos = m.pos64.p;
            v.ring = m.attn[layer].ring.p;
            if (attn::hip::attention_live_chunks()) {   // last row's window + its selection bound (count <= top-k, <= entries)
                const std::uint64_t last = std::uint64_t{position} + R;   // (pos + 1) of the last row
                const std::uint64_t win = last < attn::kWindow ? last : attn::kWindow;
                std::uint64_t sel = 0;
                if (r.ratio != 0U) {
                    sel = last / m.roles[static_cast<std::size_t>(r.kv_source)].ratio;
                    if (sel > attn::kTopK) sel = attn::kTopK;
                }
                v.max_visible = static_cast<std::uint32_t>(win + sel);
            }
            if (attn::hip::attention_window_fold()) {   // the window insert of this call's rows (window_write's inputs)
                v.kv = m.kv.p;
                v.kv_norm = m.devf(p + "attn_kv_a_norm.weight");
                v.ring_out = m.attn[layer].ring.p;
            }
            if (r.ratio != 0U) {
                v.kv_cache = m.attn[static_cast<std::size_t>(r.kv_source)].kv_cache.p;
                v.idx = m.attn[static_cast<std::size_t>(r.index_source)].idx.p;
                v.idx_count = m.attn[static_cast<std::size_t>(r.index_source)].idx_count.p;
            }
            v.partial = m.attn_partial.p;
            v.o = m.o.p;
            v.tickets = m.attn_tickets.p;
            if (attn_a8) {
                v.codes = m.o_codes.p;
                v.deltas = m.o_deltas.p;
                v.sums = m.o_sums.p;
            }
            if (m.distributed() && m.options.tp_attention) {  // heads [32r, 32r + 32): q, sink and o local to the rank
                v.heads = c.num_attention_heads / 2U;
                v.sink = m.devf(s.weights[0]) + m.options.rank * v.heads;
            }
            attn::hip::enqueue_attention(v, st);
            break;
        }
        case Op::hc_post: {
            if (s.site == "attn") enqueue_late_mix();  // if the o_b path did not
            const int next = 1 - m.cur;
            x_a8 = false;
            if (fuse_hc) {  // stored by the next site (or before the Engram injection, or by the head)
                post.on = true;
                post.sub = m.y.p;
                post.residual = stream_now();
                post.mixes = s.site == "attn" ? m.mix_attn.p : m.mix_ffn.p;
                m.cur = next;
                if (s.site == "ffn") {
                    m.identity_pre = false;
                    if (m.options.capture_main_hidden && layer + 1U >= capture_first && layer + 1U < m.options.layers) capture_at = layer + 1U;
                }
                break;
            }
            ds41::enqueue_hc_expand(m.y.p, stream_now(), s.site == "attn" ? m.mix_attn.p : m.mix_ffn.p, R, H, m.streams[next].p, st);
            m.cur = next;
            if (s.site == "ffn") {
                if (hashing) m.pre_carry = download(m.mix_ffn.p, 1, st)[0].pre;  // host copy only for the hash
                m.identity_pre = false;
                if (m.options.capture_main_hidden && layer + 1U >= capture_first && layer + 1U < m.options.layers)
                    check(hipMemcpyAsync(m.capture[layer + 1U - capture_first].p, m.streams[next].p, std::size_t{R} * hc * H * 4U,
                                         hipMemcpyDeviceToDevice, st),
                          "capture");  // the input of layer + 1
                if (m.options.layer_timing) {
                    check(hipStreamSynchronize(st), "layer end");
                    timing_[layer] = {seconds_since(t_layer), m.comm_seconds() - comm_at_layer};
                }
            }
            break;
        }
        case Op::router: {
            ds41::RouterView v;
            v.x = m.x.p;
            if (m.router_bf16.contains(s.weights[0])) v.weight_bf16 = m.devh(s.weights[0]);
            else v.weight = m.devf(s.weights[0]);
            v.bias = m.devf(s.weights[1]);
            v.bias_image = m.devf(s.weights[2]);
            v.image_rows = m.image_row.p;
            v.rows = R;
            v.hidden = H;
            v.experts = c.n_routed_experts;
            v.top_k = K;
            v.route_scale = static_cast<float>(c.routed_scaling_factor);
            v.scores = m.scores.p;
            v.ids = m.ids.p;
            v.weights = m.route_w.p;
            v.tickets = m.router_tickets.p;  // STRIX_DS41_ROUTER_FAST=1: the selection in the scores launch
            routes_ready = router_routes && !m.options.stream_experts && ds41::enqueue_router_routes(v, route_view(), st);
            if (!routes_ready) ds41::enqueue_router(v, st);
            if (hashing) m.route_ids = download(m.ids.p, K, st);  // host copy only for the hash
            if (m.options.record_routes) routes_[layer] = download(m.ids.p, std::size_t{R} * K, st);
            if (m.options.op_timing)
                check(hipMemcpyAsync(m.op_ids.p + std::size_t{layer} * max_window_rows * K, m.ids.p, std::size_t{R} * K * 4U,
                                     hipMemcpyDeviceToDevice, st),
                      "op timing ids");
            break;
        }
        case Op::shared_expert: {
            // A8 of the FFN input, shared by the shared expert and the routed experts (from the hc site when fused)
            if (!x_a8) ds41::enqueue_quantize_a8(m.x.p, H, R, m.codes.p, m.deltas.p, m.sums.p, st);
            // Shared expert, three forms with the same bits: split by intermediate rows over the two ranks (opt-in,
            // STRIX_DS41_SPLIT_SHARED=1), overlapped with the half-sum exchange (two ranks, default; STRIX_DS41_EP_OVERLAP=0 off), or here.
            if (m.shexp_tp) {
                // ds41-trellis-k5-v1:shexp-tp: this rank's partial into shared (one process: rank 0's there, rank 1's into
                // tr_p1); the combine adds it onto the rank's half sum before the exchange (no overlapped form)
                const auto& slices = m.trellis_shexp_tp.at(layer);
                for (std::size_t k = 0; k < slices.size(); ++k)
                    ds41::enqueue_shexp_tp_partial(slices[k], m.x.p, R, static_cast<float>(c.swiglu_limit), m.tr_gate.p, m.tr_up.p,
                                                   m.tr_act.p, k == 0U ? m.shared.p : m.tr_p1.p, st);
                break;
            }
            const DeviceRoutes& r = *m.shared_routes.at(R);
            if (m.split_shared) {
                // rank r: gate/up of intermediate rows [I/2 r, I/2 r + I/2) (whole A8 groups, the same products), then the
                // activation halves are exchanged while the routed experts run; the down rows follow in the combine.
                const std::uint32_t hI = I / 2U;
                if (m.native_fp8.contains(s.weights[0])) {
                    fp8_shared_gate_up(s, hI * m.options.rank, hI, reinterpret_cast<std::int8_t*>(m.shx_own.p),
                                       reinterpret_cast<float*>(m.shx_own.p + std::size_t{R} * hI), m.sh_act_sums.p);
                    split_raw_exchange.emplace(*m.options.communicator, st, m.shx_own.p, m.shx_other.p, std::size_t{R} * hI + std::size_t{R} * hI / 32U * 4U,
                                                  dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, position});
                    if (hashing) m.shared_hash_at = hashes_.size();
                    break;
                }
                const auto gu_fmt = fmt_of(s.weights[0]);
                const auto gate_row = row_bytes(gu_fmt, H);
                ds41::GateUpView gu;
                if (gu_fmt == ds41::WeightFormat::cb4) std::copy(m.cb4.at(s.weights[0]).begin(), m.cb4.at(s.weights[0]).end(), gu.table);
                gu.gate = m.dev(s.weights[0]) + gate_row * hI * m.options.rank;
                gu.up = m.dev(s.weights[1]) + gate_row * hI * m.options.rank;
                gu.expert_stride_bytes = gate_row * I;
                gu.row_bytes = gate_row;
                gu.hidden = H;
                gu.inter = hI;
                gu.routes = r.gate_up();
                gu.codes = m.codes.p;
                gu.deltas = m.deltas.p;
                gu.limit = static_cast<float>(c.swiglu_limit);
                gu.act_codes = reinterpret_cast<std::int8_t*>(m.shx_own.p);
                gu.act_deltas = reinterpret_cast<float*>(m.shx_own.p + std::size_t{R} * hI);
                gu.act_sums = m.sh_act_sums.p;
                ds41::enqueue_gate_up(gu_fmt, gu, st);
                split_raw_exchange.emplace(*m.options.communicator, st, m.shx_own.p, m.shx_other.p, std::size_t{R} * hI + std::size_t{R} * hI / 32U * 4U,
                                              dist::Tag{static_cast<std::uint32_t>(dist::TagKind::raw_slots), layer, position});
                if (hashing) m.shared_hash_at = hashes_.size();  // the shared output exists after the combine
                break;
            }
            // Two ranks, no op hashes: the shared expert (replicated, independent of the exchange) is enqueued between
            // the start and the finish of the half-sum exchange, so it runs while the halves travel. Same kernels and
            // buffers, same bits. STRIX_DS41_EP_OVERLAP=0 keeps it here.
            if (m.distributed() && !hashing && overlap_shared) {
                deferred_shared = &s;
                break;
            }
            enqueue_shared(s);
            break;
        }
        case Op::routed_experts: {
            if (m.options.stream_experts) {
                if (R != 1U) throw std::invalid_argument("deepseek41 device forward: stream_experts decodes one row per step");
                const auto ids = download(m.ids.p, K, st);  // also waits for the producers of the previous slot contents
                auto routes = ds41::build_window_routes(ids, 1U, K, m.plan.expert_split);
                const std::vector<std::uint32_t> experts = routes.experts;  // the union, ascending
                for (std::uint32_t j = 0; j < routes.experts.size(); ++j) routes.experts[j] = j;  // slot j holds experts[j]
                const auto gate_row = row_bytes(ds41::WeightFormat::iq2_xxs, H), down_row = row_bytes(ds41::WeightFormat::q2_k, I);
                const std::uint64_t gs = gate_row * I, ds = down_row * H;
                const auto* tg = m.tensors.at(s.weights[0]);
                const auto* tu = m.tensors.at(s.weights[1]);
                const auto* td = m.tensors.at(s.weights[2]);
                const auto t0 = std::chrono::steady_clock::now();
                std::vector<std::future<void>> copies;
                for (std::size_t j = 0; j < experts.size(); ++j)
                    copies.push_back(std::async(std::launch::async, [&, j] {
                        const std::uint64_t e = experts[j];
                        std::uint8_t* out = m.slot_pinned + j * (2U * gs + ds);
                        const auto g = m.file->bytes(tg->absolute_offset + e * gs, gs);
                        const auto u = m.file->bytes(tu->absolute_offset + e * gs, gs);
                        const auto d = m.file->bytes(td->absolute_offset + e * ds, ds);
                        std::copy(g.begin(), g.end(), out);
                        std::copy(u.begin(), u.end(), out + gs);
                        std::copy(d.begin(), d.end(), out + 2U * gs);
                    }));
                for (auto& f : copies) f.get();
                m.stream_seconds += seconds_since(t0);
                for (std::size_t j = 0; j < experts.size(); ++j) {
                    const std::uint8_t* in = m.slot_pinned + j * (2U * gs + ds);
                    check(hipMemcpyAsync(m.slot_gate.p + j * gs, in, gs, hipMemcpyHostToDevice, st), "expert gate");
                    check(hipMemcpyAsync(m.slot_up.p + j * gs, in + gs, gs, hipMemcpyHostToDevice, st), "expert up");
                    check(hipMemcpyAsync(m.slot_down.p + j * ds, in + 2U * gs, ds, hipMemcpyHostToDevice, st), "expert down");
                }
                const std::vector<std::uint32_t> counts{static_cast<std::uint32_t>(routes.experts.size()), routes.max_rows};
                m.built.experts.upload(routes.experts, st);
                m.built.begins.upload(routes.begins, st);
                m.built.rows.upload(routes.rows, st);
                m.built.slots.upload(routes.slots, st);
                m.built.order.upload(routes.order, st);
                m.built.low_counts.upload(routes.low_counts, st);
                m.built.counts.upload(counts, st);
                const ds41::RoutesView gate_routes{m.built.experts.p, m.built.begins.p, m.built.rows.p, m.built.slots.p, K, 1U, m.built.counts.p};
                const ds41::RoutesView down_routes{m.built.experts.p, m.built.begins.p, m.built.slots.p, m.built.slots.p, K, 1U, m.built.counts.p};
                ds41::GateUpView gu;
                gu.gate = m.slot_gate.p;
                gu.up = m.slot_up.p;
                gu.expert_stride_bytes = gs;
                gu.row_bytes = gate_row;
                gu.hidden = H;
                gu.inter = I;
                gu.routes = gate_routes;
                gu.codes = m.codes.p;
                gu.deltas = m.deltas.p;
                gu.limit = static_cast<float>(c.swiglu_limit);
                gu.act_codes = m.act_codes.p;
                gu.act_deltas = m.act_deltas.p;
                gu.act_sums = m.act_sums.p;
                ds41::enqueue_gate_up(ds41::WeightFormat::iq2_xxs, gu, st);
                ds41::RoutesProductView dv;
                dv.weights = m.slot_down.p;
                dv.expert_stride_bytes = ds;
                dv.row_bytes = down_row;
                dv.width = I;
                dv.outputs = H;
                dv.routes = down_routes;
                dv.codes = m.act_codes.p;
                dv.deltas = m.act_deltas.p;
                dv.out = m.raw.p;
                ds41::enqueue_routes_product(ds41::WeightFormat::q2_k, dv, st);
                check(hipStreamSynchronize(st), "streamed experts");  // the uploads read the shared staging
                break;
            }
            // routes of this rank's experts [expert_first, expert_end) from the router's device ids
            const auto bv = route_view();
            if (!routes_ready) ds41::enqueue_build_routes(bv, st);
            const ds41::RoutesView gate_routes{m.built.experts.p, m.built.begins.p, m.built.rows.p, m.built.slots.p, R * K, R, m.built.counts.p};
            const ds41::RoutesView down_routes{m.built.experts.p, m.built.begins.p, m.built.slots.p, m.built.slots.p, R * K, R, m.built.counts.p};
            // expert TP: this rank's 1152 intermediate rows of every expert and its 5-block down rows (ds41_expert_tp.hpp)
            const std::uint32_t EI = m.expert_tp ? ds41::expert_tp::half : I;
            const auto gate_row = row_bytes(ds41::WeightFormat::iq2_xxs, H);
            const std::uint64_t down_row = m.expert_tp ? ds41::expert_tp::device_down_row_bytes() : row_bytes(ds41::WeightFormat::q2_k, I);
            // The built expert ids are absolute; the resident tensors start at expert_first, so the kernels get the
            // address expert 0 would have (only ids in [expert_first, expert_end) are ever dereferenced).
            const std::uint64_t gate_stride = gate_row * EI, down_stride = down_row * H;
            ds41::GateUpView gu;
            gu.gate = m.dev(s.weights[0]) - gate_stride * m.expert_first;
            gu.up = m.dev(s.weights[1]) - gate_stride * m.expert_first;
            gu.expert_stride_bytes = gate_stride;
            gu.row_bytes = gate_row;
            gu.hidden = H;
            gu.inter = EI;
            gu.routes = gate_routes;
            gu.codes = m.codes.p;
            gu.deltas = m.deltas.p;
            gu.limit = static_cast<float>(c.swiglu_limit);
            gu.act_codes = m.act_codes.p;
            gu.act_deltas = m.act_deltas.p;
            gu.act_sums = m.act_sums.p;
            ds41::enqueue_gate_up(ds41::WeightFormat::iq2_xxs, gu, st);
            ds41::RoutesProductView dv;
            dv.weights = m.dev(s.weights[2]) - down_stride * m.expert_first;
            dv.expert_stride_bytes = down_stride;
            dv.row_bytes = down_row;
            dv.width = EI;
            dv.outputs = H;
            dv.routes = down_routes;
            dv.codes = m.act_codes.p;
            dv.deltas = m.act_deltas.p;
            dv.out = m.raw.p;
            if (m.expert_tp)  // this rank's K-half parts (packed half-block rows unless STRIX_DS41_EXPERT_TP_LAYOUT=dup)
                ds41::enqueue_routes_product_groups(ds41::WeightFormat::q2_k, dv, st, ds41::expert_tp::packed_from_env());
            else ds41::enqueue_routes_product(ds41::WeightFormat::q2_k, dv, st);
            break;
        }
        case Op::combine: {
            if (m.split_shared) {
                // the peer's activation half, the full activation, this rank's down rows [H/2 r, H/2 r + H/2) after its
                // half sum in the same exchange, then (shared + low) + high on both ranks
                const std::uint32_t hI = I / 2U, hH = H / 2U;
                split_raw_exchange->finish(st);
                split_raw_exchange.reset();
                const auto* own = m.shx_own.p;
                const auto* other = m.shx_other.p;
                const auto* first = m.options.rank == 0U ? own : other;
                const auto* second = m.options.rank == 0U ? other : own;
                const auto codes = [&](const std::uint8_t* b) { return reinterpret_cast<const std::int8_t*>(b); };
                const auto deltas = [&](const std::uint8_t* b) { return reinterpret_cast<const float*>(b + std::size_t{R} * hI); };
                ds41::enqueue_assemble_a8_halves(codes(first), deltas(first), codes(second), deltas(second), R, hI, m.sh_act_codes.p,
                                                 m.sh_act_deltas.p, st);
                const DeviceRoutes& r = *m.shared_routes.at(R);
                const std::string down_name = "blk." + std::to_string(layer) + ".ffn_down_shexp.weight";
                if (m.native_fp8.contains(down_name)) {
                    fp8_site(down_name, I, hH * m.options.rank, hH, m.sh_act_codes.p, m.sh_act_deltas.p, m.half_own.p + std::size_t{R} * H, hH);
                } else {
                    const auto down_fmt = fmt_of(down_name);
                    const auto down_row = row_bytes(down_fmt, I);
                    ds41::RoutesProductView dv;
                    if (down_fmt == ds41::WeightFormat::cb4) std::copy(m.cb4.at(down_name).begin(), m.cb4.at(down_name).end(), dv.table);
                    dv.weights = m.dev(down_name) + down_row * hH * m.options.rank;
                    dv.expert_stride_bytes = down_row * H;
                    dv.row_bytes = down_row;
                    dv.width = I;
                    dv.outputs = hH;
                    dv.routes = r.down();
                    dv.codes = m.sh_act_codes.p;
                    dv.deltas = m.sh_act_deltas.p;
                    dv.out = m.half_own.p + std::size_t{R} * H;
                    ds41::enqueue_routes_product(down_fmt, dv, st);
                }
                ds41::HalfSumView v;
                v.raw = m.raw.p;
                v.weights = m.route_w.p;
                v.order = m.built.order.p;
                v.low_counts = m.built.low_counts.p;
                v.rows = R;
                v.top_k = K;
                v.hidden = H;
                v.half = m.expert_tp ? 0U : m.options.rank;  // expert TP: the fold of all the row's slots (low_counts = top_k)
                v.out = m.half_own.p;
                ds41::enqueue_half_sum(v, st);
                m.options.communicator->exchange(st, m.half_own.p, m.half_other.p, (std::size_t{R} * H + std::size_t{R} * hH) * 4U,
                                                 dist::Tag{static_cast<std::uint32_t>(dist::TagKind::half_sum), layer, position});
                const float* low = m.options.rank == 0U ? m.half_own.p : m.half_other.p;
                const float* high = m.options.rank == 0U ? m.half_other.p : m.half_own.p;
                ds41::enqueue_canonical_sum_split(low + std::size_t{R} * H, high + std::size_t{R} * H, low, high, R, H, m.y.p,
                                                  m.shared.p, st);
                if (hashing) {
                    const std::string label = "shared_expert@" + std::to_string(layer) + ":" + std::string(to_string(Buffer::shared));
                    hashes_.insert(hashes_.begin() + static_cast<std::ptrdiff_t>(m.shared_hash_at), {label, fnv(m.buffer(Buffer::shared))});
                }
                break;
            }
            if (m.distributed()) {  // this rank's half sum, the peer's by exchange, then (shared + low) + high on both
                ds41::HalfSumView v;
                v.raw = m.raw.p;
                v.weights = m.route_w.p;
                v.order = m.built.order.p;
                v.low_counts = m.built.low_counts.p;
                v.rows = R;
                v.top_k = K;
                v.hidden = H;
                v.half = m.expert_tp ? 0U : m.options.rank;  // expert TP: the fold of all the row's slots (low_counts = top_k)
                // STRIX_DS41_MOE_READY=1: the half sum straight into the coherent TX view, the combine straight from RX
                // (lean releases, no copies or events); same sums, same order. Above the view limit: as below.
                if (!m.shexp_tp && ready_detail::moe())
                    if (const auto stage = m.options.communicator->prepare_staged_ready(st, std::size_t{R} * H * 4U); stage.data) {
                        auto* own = static_cast<float*>(stage.data);
                        v.out = own;
                        ds41::enqueue_half_sum_ready(v, stage, st);
                        xport::ds41_exchange_half_sum_ready(*m.options.communicator, st, own, std::size_t{R} * H * 4U,
                            dist::Tag{static_cast<std::uint32_t>(dist::TagKind::half_sum), layer, position},
                            [&] {
                                if (deferred_shared) {
                                    enqueue_shared(*deferred_shared);
                                    deferred_shared = nullptr;
                                }
                            },
                            [&](const auto& half_peer) {
                                const auto* other = static_cast<const float*>(half_peer.data);
                                ds41::enqueue_canonical_sum_ready(m.shared.p, m.options.rank == 0U ? own : other, m.options.rank == 0U ? other : own,
                                                                  std::size_t{R} * H, m.y.p, half_peer, st);
                            });
                        break;
                    }
                v.out = m.half_own.p;
                ds41::enqueue_half_sum(v, st);
                if (m.shexp_tp) {  // shexp-tp: S_r + P_r travels, y = (S0 + P0) + (S1 + P1)
                    ds41::enqueue_add_inplace_f32(m.half_own.p, m.shared.p, std::size_t{R} * H, st);
                    xport::ds41_exchange_half_sum(*m.options.communicator, st, m.half_own.p, m.half_other.p, std::size_t{R} * H * 4U,
                                                 dist::Tag{static_cast<std::uint32_t>(dist::TagKind::half_sum), layer, position}, [] {});
                    const float* low = m.options.rank == 0U ? m.half_own.p : m.half_other.p;
                    const float* high = m.options.rank == 0U ? m.half_other.p : m.half_own.p;
                    ds41::enqueue_add_f32(low, high, std::size_t{R} * H, m.y.p, st);
                    break;
                }
                xport::ds41_exchange_half_sum(*m.options.communicator, st, m.half_own.p, m.half_other.p, std::size_t{R} * H * 4U,
                    dist::Tag{static_cast<std::uint32_t>(dist::TagKind::half_sum), layer, position},
                    [&] {
                        if (deferred_shared) {
                            enqueue_shared(*deferred_shared);
                            deferred_shared = nullptr;
                        }
                    });
                const float* low = m.options.rank == 0U ? m.half_own.p : m.half_other.p;
                const float* high = m.options.rank == 0U ? m.half_other.p : m.half_own.p;
                ds41::enqueue_canonical_sum(m.shared.p, low, high, std::size_t{R} * H, m.y.p, st);
                break;
            }
            // two ranks emulated on one node: each rank's weighted half sum, then (shared + low) + high
            for (std::uint32_t half = 0; half < 2U; ++half) {
                ds41::HalfSumView v;
                v.raw = m.raw.p;
                v.weights = m.route_w.p;
                v.order = m.built.order.p;
                v.low_counts = m.built.low_counts.p;
                v.rows = R;
                v.top_k = K;
                v.hidden = H;
                v.half = half;
                v.out = half == 0U ? m.low.p : m.high.p;
                ds41::enqueue_half_sum(v, st);
            }
            if (m.shexp_tp) {  // shexp-tp, both ranks here: y = (S0 + P0) + (S1 + P1)
                ds41::enqueue_add_inplace_f32(m.low.p, m.shared.p, std::size_t{R} * H, st);
                ds41::enqueue_add_inplace_f32(m.high.p, m.tr_p1.p, std::size_t{R} * H, st);
                ds41::enqueue_add_f32(m.low.p, m.high.p, std::size_t{R} * H, m.y.p, st);
                break;
            }
            ds41::enqueue_canonical_sum(m.shared.p, m.low.p, m.high.p, std::size_t{R} * H, m.y.p, st);
            break;
        }
        case Op::head: {
            if (fuse_hc) {
                enqueue_site(nullptr, nullptr, m.identity_pre ? nullptr : m.mix_ffn.p, m.devf("output_norm.weight"), m.h.p, m.site_codes.p,
                             m.site_deltas.p, m.site_sums.p);
            } else {
                ds41::enqueue_hc_collapse(stream_now(), m.identity_pre ? nullptr : m.mix_ffn.p, R, H, m.h.p, st);
                ds41::enqueue_rms_norm(m.h.p, m.devf("output_norm.weight"), R, H, H, 1e-20F, m.h.p, st);
                ds41::enqueue_quantize_a8(m.h.p, H, R, m.site_codes.p, m.site_deltas.p, m.site_sums.p, st);
            }
            if (!m.trellis_head.empty() && m.headout) {  // same rows and bits, no staging (R1 in place, R > 1 to head_rows)
                m.head_rows_valid = R > 1U;
                for (std::size_t k = 0; k < m.trellis_head.size(); ++k) {
                    const auto& [sl, site] = m.trellis_head[k];
                    float* y = m.head_rows_valid ? m.head_rows.p + k * max_window_rows * sl.count : m.logits.p + sl.first;
                    ds41::enqueue_trellis_linear_rows(*site, m.h.p, y, R, H, m.head_rows_valid ? sl.count : V, max_window_rows, 5U, st);
                }
            } else if (!m.trellis_head.empty()) {  // trellis head (ds41-trellis-k5-v1:head): logits[row][slice] from F32 h
                for (const auto& [sl, site] : m.trellis_head) ds41::enqueue_trellis_linear(*site, m.h.p, m.logits.p + sl.first, R, H, V, false, st);
            } else {
                ds41::Q8A8View v;
                m.site_weight(v, "output.weight");  // this rank's rows [head_slice.first, + count)
                v.width = H;
                v.first = 0;
                v.count = m.head_slice.count;
                v.per_group = m.head_slice.count;
                v.codes = m.site_codes.p;
                v.deltas = m.site_deltas.p;
                v.rows = R;
                v.x_stride = H;
                v.out_stride = V;  // logits [row][vocab]
                v.out = m.logits.p + m.head_slice.first;
                ds41::enqueue_q8_0_a8(v, st);
            }
            if (R == 1U && m.distributed() && m.options.gather_logits) {
                const auto other = ds41::vocab_slice(c.vocab_size, 2U, 1U - m.options.rank);
                if (other.count != m.head_slice.count) throw std::logic_error("deepseek41 device forward: unequal vocabulary slices");
                m.options.communicator->exchange(st, m.logits.p + m.head_slice.first, m.logits.p + other.first, std::size_t{other.count} * 4U,
                                                 dist::Tag{static_cast<std::uint32_t>(dist::TagKind::control), 0U, position});
            }
            if (R == 1U && (hashing || !m.distributed() || m.options.gather_logits)) {
                const auto t0 = tr ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                m.logits_host = download(m.logits.p, c.vocab_size, st);
                if (tr) tr->sync("logits_d2h", t0, -1);
            } else m.logits_host.clear();
            break;
        }
        case Op::argmax: {
            // two ranks: each the argmax of its vocabulary slice, then the candidates combine
            if (m.distributed()) {  // this rank's candidates, the peer's by exchange, combined in rank order on both
                const auto& sl = m.head_slice;
                const auto [lp, ls] = m.logits_rows(sl.first, V);
                ds41::enqueue_argmax(lp, R, sl.count, static_cast<std::uint32_t>(ls), sl.first, m.partial.p, m.cand_own.p, st);
                xport::ds41_exchange_argmax(*m.options.communicator, st, m.cand_own.p, m.cand_other.p, R,
                                           dist::Tag{static_cast<std::uint32_t>(dist::TagKind::argmax), 0U, position});
                const auto* first = m.options.rank == 0U ? m.cand_own.p : m.cand_other.p;
                const auto* second = m.options.rank == 0U ? m.cand_other.p : m.cand_own.p;
                ds41::enqueue_combine_argmax(first, second, R, m.best.p, st);
            } else {
                const auto r0 = ds41::vocab_slice(c.vocab_size, 2U, 0U), r1 = ds41::vocab_slice(c.vocab_size, 2U, 1U);
                const auto [l0, s0] = m.logits_rows(r0.first, V);
                const auto [l1, s1] = m.logits_rows(r1.first, V);
                ds41::enqueue_argmax(l0, R, r0.count, static_cast<std::uint32_t>(s0), r0.first, m.partial.p, m.cand0.p, st);
                ds41::enqueue_argmax(l1, R, r1.count, static_cast<std::uint32_t>(s1), r1.first, m.partial.p, m.cand1.p, st);
                ds41::enqueue_combine_argmax(m.cand0.p, m.cand1.p, R, m.best.p, st);
            }
            const auto t_best = tr ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            const auto best = download(m.best.p, R, st);
            if (tr) tr->sync("best_d2h", t_best, -1);
            tokens_.resize(R);
            for (std::uint32_t r = 0; r < R; ++r) {
                if (best[r].finite == 0U) throw std::runtime_error("deepseek41 device forward: non-finite logits");
                tokens_[r] = best[r].index;
            }
            next_token_ = tokens_.back();
            head_seconds_ = seconds_since(t_head);
            break;
        }
        }
        if (step_events) {
            op_exchange.push_back(m.comm_seconds() - comm_before);
            check(hipEventRecord(m.op_events[op_exchange.size()], st), "op timing");
        }
        if (tr) tr->op(to_string(s.op), s.layer, s.site, m.comm_seconds() - comm_before, static_cast<int>(op_exchange.size()));
        // hashes of the step's outputs, labelled and computed as HostForward's
        if (!hashing) continue;
        if (m.split_shared && s.op == Op::shared_expert) continue;  // recorded by the combine (see there)
        for (const auto b : s.outputs) {
            std::uint64_t h = 0;
            if (b == Buffer::logits) h = fnv(m.logits_host);
            else if (b == Buffer::token) h = next_token_;
            else if (b == Buffer::route_ids) {
                for (const auto v : m.route_ids) h = h * 1099511628211ULL + v;
            } else {
                const auto v = m.buffer(b);
                h = fnv(v);
                if (m.options.buffer_tap)
                    m.options.buffer_tap(std::string(to_string(s.op)) + (s.layer >= 0 ? "@" + std::to_string(s.layer) : "") +
                                             (s.site.empty() ? "" : "/" + s.site) + ":" + std::string(to_string(b)),
                                         v);
            }
            if (s.op == Op::hc_post && s.site == "ffn" && b == Buffer::stream) stream_hashes_.push_back(h);
            hashes_.emplace_back(std::string(to_string(s.op)) + (s.layer >= 0 ? "@" + std::to_string(s.layer) : "") +
                                     (s.site.empty() ? "" : "/" + s.site) + ":" + std::string(to_string(b)),
                                 h);
        }
    }
    flush_post();  // a FRONT stage (no head): the last layer's expansion is still pending with the fused hc sites
    if (engram_async) {
        engram_io.get();
        engram_seconds_ = engram_read_seconds;
        if (tr)  // the reader thread's whole read (only its duration is meaningful in the row)
            tr->sync("engram_async_read", std::chrono::steady_clock::now() -
                                              std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(engram_read_seconds)), -1);
    }
    if (tr) tr->last_event(st);
    check(hipStreamSynchronize(st), "step");
    if (tr) tr->mark("final_sync");
    if (m.options.op_timing) collect_op_timing(op_exchange);
    if (tr) tr->mark("collect_op_timing");
    m.last_position = position;
    m.position += R;
    step_seconds_ = seconds_since(t_step);
    if (tr) tr->finish(m.op_events, step_seconds_);
    return tokens_;
}

}  // namespace strix::deepseek41
