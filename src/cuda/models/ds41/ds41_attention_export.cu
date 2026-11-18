// Handoff of the CUDA attention state in the decode format strix/ds41-attention-state-v1 (see cuda_ds41_attention.hpp
// and deepseek41_attention_state.hpp). Host-side conversion only: the caches are downloaded / uploaded as they are and
// the window ring is re-slotted between this state's capacity (kWindow + max_rows) and the decode capacity
// (AttentionHost::ring_capacity); no kernel is involved, so the bits are the kernels' bits.
#include "strix/cuda/ds41_attention.hpp"
#include "strix/deepseek41_attention_host.hpp"
#include "strix/deepseek41_attention_state.hpp"
#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

namespace strix::ds41::cuda::attn {
namespace {

template <class T> void download_into(std::vector<T>& out, const void* src, std::size_t count, cudaStream_t stream) {
    out.resize(count);
    if (count != 0U) STRIX_CUDA_CHECK(cudaMemcpyAsync(out.data(), src, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
}

// The ring positions a handoff carries: the last min(position, both capacities) (the live window is the last 128).
// They sit in at most two runs of consecutive slots of a ring of capacity cap: [s0, s0 + n0) and [0, n - n0).
struct Span {
    std::uint32_t first, n, s0, n0;
    Span(std::uint32_t position, std::uint32_t cap, std::uint32_t other_cap)
        : first(position - std::min({position, cap, other_cap})), n(position - first), s0(first % cap), n0(std::min(n, cap - s0)) {}
};
// positions [first, first + n) in order (n x 528 B) <-> their slots of a host ring of capacity cap
void to_slots(const std::uint8_t* packed, const Span& sp, std::uint8_t* ring, std::uint32_t cap) {
    for (std::uint32_t i = 0; i < sp.n; ++i)
        std::memcpy(ring + std::size_t{(sp.first + i) % cap} * kWindowEntryBytes, packed + std::size_t{i} * kWindowEntryBytes, kWindowEntryBytes);
}
void from_slots(const std::uint8_t* ring, std::uint32_t cap, const Span& sp, std::uint8_t* packed) {
    for (std::uint32_t i = 0; i < sp.n; ++i)
        std::memcpy(packed + std::size_t{i} * kWindowEntryBytes, ring + std::size_t{(sp.first + i) % cap} * kWindowEntryBytes, kWindowEntryBytes);
}

[[noreturn]] void fail(const char* op, const std::string& what) { throw std::invalid_argument(std::string("ds41 cuda attention: ") + op + ": " + what); }

}  // namespace

deepseek41::AttentionState AttentionState::export_state(std::uint32_t first, std::uint32_t end, std::uint32_t position, std::uint32_t last_row,
                                                        cudaStream_t stream) const {
    const auto& P = plan();
    const auto& o = options();
    if (first >= end || first < o.first_layer || end > o.end_layer) fail("export_state", "layers outside the range");
    if (position == 0U || position > o.max_positions) fail("export_state", "position 1..max_positions");
    if (last_row != kNoLastRow && last_row >= o.max_rows) fail("export_state", "last_row >= max_rows");
    constexpr std::uint32_t dcap = deepseek41::AttentionHost::ring_capacity;
    deepseek41::AttentionState s;
    s.position = position;
    s.layers.resize(P.size());
    std::vector<std::vector<std::uint8_t>> rings(P.size());
    std::vector<std::vector<std::uint32_t>> valid(P.size()), counts(P.size()), sel(P.size());
    // everything onto the host in one stream pass, then one synchronization
    for (std::uint32_t l = first; l < end; ++l) {
        const auto& r = P[l];
        const auto& c = caches(l);
        auto& S = s.layers[l];
        S.owns_kv = r.owns_kv;
        S.owns_index = r.owns_index;
        S.ratio = r.ratio;
        // only the carried slots (a prefill state may have a ring of 128 + max_positions entries)
        const Span sp(position, ring_capacity(), dcap);
        rings[l].resize(std::size_t{sp.n} * kWindowEntryBytes);
        if (sp.n0 != 0U)
            STRIX_CUDA_CHECK(cudaMemcpyAsync(rings[l].data(), c.ring + std::size_t{sp.s0} * kWindowEntryBytes, std::size_t{sp.n0} * kWindowEntryBytes,
                                             cudaMemcpyDeviceToHost, stream));
        if (sp.n != sp.n0)
            STRIX_CUDA_CHECK(cudaMemcpyAsync(rings[l].data() + std::size_t{sp.n0} * kWindowEntryBytes, c.ring, std::size_t{sp.n - sp.n0} * kWindowEntryBytes,
                                             cudaMemcpyDeviceToHost, stream));
        if (r.owns_kv) {
            const std::size_t entries = position / r.ratio;
            download_into(S.kv, c.kv_cache, entries * kKvEntryBytes, stream);
            download_into(S.idx_keys, c.idx_cache, entries * kIdxEntryBytes, stream);
            if (r.ratio == 2U) {
                download_into(S.carry, c.carry, 2U * kHeadDim, stream);
                download_into(valid[l], c.carry_valid, 1, stream);
            }
        }
        if (r.owns_index && last_row != kNoLastRow) {
            if (last_row < c.row0 || last_row - c.row0 >= c.rows) fail("export_state", "selection row outside allocation");
            const auto row = last_row - c.row0;
            download_into(counts[l], c.idx_count + row, 1, stream);
            download_into(sel[l], c.idx + std::size_t{row} * kTopK, kTopK, stream);
        }
    }
    STRIX_CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::uint32_t l = first; l < end; ++l) {
        const auto& r = P[l];
        auto& S = s.layers[l];
        S.ring.assign(std::size_t{dcap} * kWindowEntryBytes, 0);
        to_slots(rings[l].data(), Span(position, ring_capacity(), dcap), S.ring.data(), dcap);
        if (r.owns_kv && r.ratio == 2U) {
            S.carry_valid = valid[l][0] != 0U;
            if (!S.carry_valid) std::fill(S.carry.begin(), S.carry.end(), 0.0F);
        }
        if (r.owns_index) {
            S.topk_known = last_row != kNoLastRow;
            if (S.topk_known) {
                const std::uint32_t n = counts[l][0];
                if (n > kTopK) fail("export_state", "top-k count of layer " + std::to_string(l));
                S.topk.assign(sel[l].begin(), sel[l].begin() + n);
            }
        }
    }
    return s;
}

void AttentionState::import_state(const deepseek41::AttentionState& s, cudaStream_t stream) {
    const auto& P = plan();
    const auto& o = options();
    const std::uint32_t T = s.position;
    if (s.layers.size() != P.size()) fail("import_state", "layer count differs from the plan");
    if (T == 0U || T >= o.max_positions) fail("import_state", "position 1..max_positions - 1 (room for the next row)");
    reset(stream);
    std::vector<std::vector<std::uint8_t>> staged;  // host buffers alive until the synchronization
    std::vector<std::uint32_t> topk_counts(P.size(), 0U);
    const auto upload = [&](void* dst, const void* src, std::size_t n) {
        if (n != 0U) STRIX_CUDA_CHECK(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream));
    };
    const auto caches_of = [&](std::uint32_t l, const deepseek41::AttentionLayerState& S, bool carry) {
        const auto& r = P[l];
        const auto& c = caches(l);
        const std::size_t entries = T / r.ratio;
        if (S.kv.size() != entries * kKvEntryBytes || S.idx_keys.size() != entries * kIdxEntryBytes)
            fail("import_state", "compressed caches of layer " + std::to_string(l));
        upload(c.kv_cache, S.kv.data(), S.kv.size());
        upload(c.idx_cache, S.idx_keys.data(), S.idx_keys.size());
        if (carry && r.ratio == 2U && S.carry_valid) {
            if (S.carry.size() != 2U * kHeadDim) fail("import_state", "carry of layer " + std::to_string(l));
            upload(c.carry, S.carry.data(), S.carry.size() * sizeof(float));
            static const std::uint32_t one = 1U;
            upload(c.carry_valid, &one, sizeof(one));
        }
    };
    for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l) {
        const auto& r = P[l];
        const auto& S = s.layers[l];
        const std::string at = " of layer " + std::to_string(l);
        if (!deepseek41::attention_layer_present(S)) fail("import_state", "missing layer " + std::to_string(l));
        if (S.owns_kv != r.owns_kv || S.owns_index != r.owns_index || (r.owns_kv && S.ratio != r.ratio)) fail("import_state", "roles" + at);
        if (S.ring.size() % kWindowEntryBytes != 0U) fail("import_state", "ring size" + at);
        const auto scap = static_cast<std::uint32_t>(S.ring.size() / kWindowEntryBytes);
        if (scap < std::min(T, kWindow)) fail("import_state", "ring smaller than the window" + at);
        const Span sp(T, ring_capacity(), scap);  // slots of this ring; the rest stays zero (reset)
        auto& packed = staged.emplace_back(std::size_t{sp.n} * kWindowEntryBytes);
        from_slots(S.ring.data(), scap, Span(T, scap, ring_capacity()), packed.data());
        upload(caches(l).ring + std::size_t{sp.s0} * kWindowEntryBytes, packed.data(), std::size_t{sp.n0} * kWindowEntryBytes);
        upload(caches(l).ring, packed.data() + std::size_t{sp.n0} * kWindowEntryBytes, std::size_t{sp.n - sp.n0} * kWindowEntryBytes);
        if (r.owns_kv) caches_of(l, S, true);
        if (r.owns_index && S.topk_known) {
            if (S.topk.size() > kTopK) fail("import_state", "top-k size" + at);
            topk_counts[l] = static_cast<std::uint32_t>(S.topk.size());
            upload(caches(l).idx, S.topk.data(), S.topk.size() * sizeof(std::uint32_t));
            upload(caches(l).idx_count, &topk_counts[l], sizeof(std::uint32_t));
        }
    }
    // KV sources outside the range that the range reads (a consumer whose producer runs elsewhere)
    for (std::uint32_t l = 0; l < P.size(); ++l) {
        if (l >= o.first_layer && l < o.end_layer) continue;
        if (!P[l].owns_kv || caches(l).kv_cache == nullptr) continue;
        const auto& S = s.layers[l];
        if (S.kv.empty() && T / P[l].ratio != 0U) fail("import_state", "KV source " + std::to_string(l) + " read by the range is not in the state");
        caches_of(l, S, false);
    }
    STRIX_CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ---- capture straight into the serialized bytes (ExportCapture) ----
namespace {

// One carried piece: n bytes from device src to offset dst of the staging buffer. The table goes as a kernel parameter
// (no descriptor upload: no copy that could queue behind other transfers).
struct Segment {
    const std::uint8_t* src;
    std::uint64_t dst, n;
};
constexpr std::uint32_t kMaxSegments = 320;  // 7.7 KB of parameters (sm_70+ / CUDA 12.1+: up to 32 KB)
struct SegmentTable {
    Segment s[kMaxSegments];
};

__global__ void gather_segments(const SegmentTable table, std::uint8_t* dst) {
    const Segment s = table.s[blockIdx.y];
    const std::uint64_t stride = std::uint64_t{blockDim.x} * gridDim.x;
    for (std::uint64_t i = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x; i < s.n; i += stride) dst[s.dst + i] = s.src[i];
}

void enqueue_gather(const std::vector<Segment>& segs, std::uint8_t* dst, cudaStream_t stream) {
    for (std::size_t at = 0; at < segs.size(); at += kMaxSegments) {
        SegmentTable t{};
        const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(kMaxSegments, segs.size() - at));
        std::uint64_t most = 0;
        for (std::uint32_t i = 0; i < n; ++i) {
            t.s[i] = segs[at + i];
            most = std::max(most, t.s[i].n);
        }
        const auto blocks = static_cast<unsigned>(std::clamp<std::uint64_t>((most + 4095U) / 4096U, 1U, 64U));
        gather_segments<<<dim3(blocks, n), 256, 0, stream>>>(t, dst);
        STRIX_CUDA_CHECK(cudaGetLastError());
    }
}

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

struct ExportCapture::Impl {
    std::uint8_t* host = nullptr;  // pinned
    std::size_t host_cap = 0;
    std::uint8_t* dev = nullptr;   // staging
    std::size_t dev_cap = 0;
    std::uint32_t* host_meta = nullptr;  // pinned, kMaxSegments words
    std::uint8_t* dev_meta = nullptr;
    cudaEvent_t ev[3] = {nullptr, nullptr, nullptr};
    deepseek41::AttentionFrame frame;
    bool captured = false, sealed = false;
    std::uint64_t digest = 0;
    std::string sha;  // hex (seal with sha256)
    Timing timing;

    Impl() {
        STRIX_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host_meta), kMaxSegments * 4U, cudaHostAllocDefault));
        STRIX_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_meta), kMaxSegments * 4U));
        for (auto& e : ev) STRIX_CUDA_CHECK(cudaEventCreate(&e));
    }
    ~Impl() {
        for (auto& e : ev)
            if (e != nullptr) cudaEventDestroy(e);
        if (host != nullptr) cudaFreeHost(host);
        if (dev != nullptr) cudaFree(dev);
        if (host_meta != nullptr) cudaFreeHost(host_meta);
        if (dev_meta != nullptr) cudaFree(dev_meta);
    }
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    void reserve(std::size_t bytes) {
        bytes = (bytes + 0xFFFFU) & ~std::size_t{0xFFFFU};
        if (bytes > host_cap) {
            if (host != nullptr) STRIX_CUDA_CHECK(cudaFreeHost(host));
            host = nullptr;
            host_cap = 0;
            STRIX_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host), bytes, cudaHostAllocDefault));
            host_cap = bytes;
        }
        if (bytes > dev_cap) {
            if (dev != nullptr) STRIX_CUDA_CHECK(cudaFree(dev));
            dev = nullptr;
            dev_cap = 0;
            STRIX_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev), bytes));
            dev_cap = bytes;
        }
    }
};

ExportCapture::ExportCapture() : impl_(std::make_unique<Impl>()) {}
ExportCapture::~ExportCapture() = default;
void ExportCapture::reserve(std::size_t bytes) { impl_->reserve(bytes); }
bool ExportCapture::captured() const noexcept { return impl_->captured; }
bool ExportCapture::sealed() const noexcept { return impl_->sealed; }
std::uint32_t ExportCapture::position() const noexcept { return impl_->frame.position; }
const deepseek41::AttentionFrame& ExportCapture::frame() const noexcept { return impl_->frame; }
std::size_t ExportCapture::capacity() const noexcept { return impl_->host_cap; }
std::size_t ExportCapture::pinned_bytes() const noexcept { return impl_->host_cap + (impl_->host_meta ? kMaxSegments * 4U : 0U); }
const ExportCapture::Timing& ExportCapture::timing() const noexcept { return impl_->timing; }

void ExportCapture::seal(unsigned threads, bool sha256) {
    auto& X = *impl_;
    if (!X.captured) throw std::logic_error("ds41 export capture: seal before a capture");
    const bool want_sha = sha256 && X.sha.empty();
    if (X.sealed && !want_sha) return;
    const auto t0 = std::chrono::steady_clock::now();
    const auto& f = X.frame;
    const auto bytes = [&](std::uint64_t at, std::uint64_t n) { return std::as_bytes(std::span(X.host + at, static_cast<std::size_t>(n))); };
    if (X.sealed) {  // already sealed: the SHA-256 alone
        X.sha = model_content_identity_hex(sha256_bytes_fast(bytes(0, f.total)));
        X.timing.sha_ms = ms_since(t0);
        return;
    }
    // the SHA-256 reads [0, digest_at) while the seal writes the digest and the checksum after it
    Sha256Fast h;
    double sha_ms = 0;
    std::thread sha_thread;
    if (want_sha)
        sha_thread = std::thread([&] {
            const auto s0 = std::chrono::steady_clock::now();
            h.update(bytes(0, f.digest_at));
            sha_ms = ms_since(s0);
        });
    try {
        X.digest = deepseek41::seal_attention_frame(f, std::span(X.host, X.host_cap), threads).digest;
    } catch (...) {
        if (sha_thread.joinable()) sha_thread.join();
        throw;
    }
    if (sha_thread.joinable()) {
        sha_thread.join();
        const auto s1 = std::chrono::steady_clock::now();
        h.update(bytes(f.digest_at, f.total - f.digest_at));
        X.sha = model_content_identity_hex(h.finalize());
        X.timing.sha_ms = sha_ms + ms_since(s1);
    }
    X.sealed = true;
    X.timing.seal_ms = ms_since(t0);
}
const std::string& ExportCapture::sha256() const noexcept { return impl_->sha; }
std::span<const std::uint8_t> ExportCapture::bytes() const {
    if (!impl_->sealed) throw std::logic_error("ds41 export capture: bytes before seal");
    return {impl_->host, static_cast<std::size_t>(impl_->frame.total)};
}
std::uint64_t ExportCapture::digest() const {
    if (!impl_->sealed) throw std::logic_error("ds41 export capture: digest before seal");
    return impl_->digest;
}

namespace {
// The frame of export_state's state: flags, ratio and piece sizes per layer as export_state fills them (layers outside
// [first, end) absent, with AttentionLayerState's defaults). topk_count(l) / carry_valid(l): the device values.
template <class TopkCount, class CarryValid>
deepseek41::AttentionFrame frame_of(const std::vector<ds41::attn::LayerAttention>& P, std::uint32_t first, std::uint32_t end, std::uint32_t position,
                                    bool last_row_known, std::uint64_t token_count, TopkCount topk_count, CarryValid carry_valid) {
    constexpr std::uint32_t dcap = deepseek41::AttentionHost::ring_capacity;
    deepseek41::AttentionFrame f;
    f.position = position;
    f.token_count = token_count;
    f.layers.resize(P.size());
    for (std::uint32_t l = first; l < end; ++l) {
        const auto& r = P[l];
        auto& F = f.layers[l];
        F.owns_kv = r.owns_kv;
        F.owns_index = r.owns_index;
        F.ratio = r.ratio;
        F.ring_bytes = std::size_t{dcap} * kWindowEntryBytes;
        if (r.owns_kv) {
            const std::size_t entries = position / r.ratio;
            F.kv_bytes = entries * kKvEntryBytes;
            F.idx_bytes = entries * kIdxEntryBytes;
            if (r.ratio == 2U) {
                F.carry_floats = 2U * kHeadDim;
                F.carry_valid = carry_valid(l);
            }
        }
        if (r.owns_index) {
            F.topk_known = last_row_known;
            if (last_row_known) F.topk_count = topk_count(l);
        }
    }
    deepseek41::plan_attention_frame(f);
    return f;
}
}  // namespace

std::size_t AttentionState::capture_bound(std::uint32_t first, std::uint32_t end, std::uint32_t position) const {
    const auto f = frame_of(plan(), first, end, position, true, position, [](std::uint32_t) { return std::uint64_t{kTopK}; },
                            [](std::uint32_t) { return true; });
    return static_cast<std::size_t>(f.total);
}

void AttentionState::capture_state(std::uint32_t first, std::uint32_t end, std::uint32_t position, std::uint32_t last_row,
                                   std::span<const std::uint32_t> tokens, std::uint32_t next_token, ExportCapture& out, cudaStream_t stream) const {
    const auto t0 = std::chrono::steady_clock::now();
    const auto& P = plan();
    const auto& o = options();
    if (first >= end || first < o.first_layer || end > o.end_layer) fail("capture_state", "layers outside the range");
    if (position == 0U || position > o.max_positions) fail("capture_state", "position 1..max_positions");
    if (last_row != kNoLastRow && last_row >= o.max_rows) fail("capture_state", "last_row >= max_rows");
    constexpr std::uint32_t dcap = deepseek41::AttentionHost::ring_capacity;
    auto& X = *out.impl_;
    X.captured = X.sealed = false;
    X.sha.clear();
    X.timing = {};
    std::uint32_t copies = 0;
    // 1. the per-layer values the frame depends on: carry flags (ratio-2 sources) and the last row's top-k counts
    std::vector<std::int32_t> carry_at(P.size(), -1), count_at(P.size(), -1);
    std::vector<Segment> meta;
    for (std::uint32_t l = first; l < end; ++l) {
        const auto& r = P[l];
        const auto& c = caches(l);
        if (r.owns_kv && r.ratio == 2U) {
            carry_at[l] = static_cast<std::int32_t>(meta.size());
            meta.push_back({reinterpret_cast<const std::uint8_t*>(c.carry_valid), meta.size() * 4U, 4U});
        }
        if (r.owns_index && last_row != kNoLastRow) {
            // the selection row as export_state() reads it (compact decoder rows of a bounded pass: row0)
            if (last_row < c.row0 || last_row - c.row0 >= c.rows) fail("capture_state", "selection row outside allocation");
            count_at[l] = static_cast<std::int32_t>(meta.size());
            meta.push_back({reinterpret_cast<const std::uint8_t*>(c.idx_count + (last_row - c.row0)), meta.size() * 4U, 4U});
        }
    }
    if (meta.size() > kMaxSegments) fail("capture_state", "too many layers");
    if (!meta.empty()) {
        enqueue_gather(meta, X.dev_meta, stream);
        STRIX_CUDA_CHECK(cudaMemcpyAsync(X.host_meta, X.dev_meta, meta.size() * 4U, cudaMemcpyDeviceToHost, stream));
        ++copies;
        STRIX_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    X.timing.meta_ms = ms_since(t0);
    // 2. the frame, then every carried piece gathered at its serialized offset and one D2H
    X.frame = frame_of(
        P, first, end, position, last_row != kNoLastRow, tokens.size(),
        [&](std::uint32_t l) {
            const std::uint32_t n = X.host_meta[count_at[l]];
            if (n > kTopK) fail("capture_state", "top-k count of layer " + std::to_string(l));
            return std::uint64_t{n};
        },
        [&](std::uint32_t l) { return X.host_meta[carry_at[l]] != 0U; });
    auto& f = X.frame;
    f.next_token = next_token;
    X.reserve(static_cast<std::size_t>(f.total));
    std::vector<Segment> segs;
    const std::uint32_t cap = ring_capacity();
    for (std::uint32_t l = first; l < end; ++l) {
        const auto& r = P[l];
        const auto& c = caches(l);
        const auto& F = f.layers[l];
        // ring: the carried positions [sp.first, position) from slot p % cap of this ring to slot p % dcap of the decode
        // ring (runs where both are consecutive); the other decode slots stay zero
        const Span sp(position, cap, dcap);
        for (std::uint32_t p = sp.first; p < position;) {
            const std::uint32_t a = p % cap, b = p % dcap, n = std::min({cap - a, dcap - b, position - p});
            segs.push_back({c.ring + std::size_t{a} * kWindowEntryBytes, F.ring_at + std::uint64_t{b} * kWindowEntryBytes, std::uint64_t{n} * kWindowEntryBytes});
            p += n;
        }
        if (r.owns_kv) {
            if (F.kv_bytes != 0U) segs.push_back({c.kv_cache, F.kv_at, F.kv_bytes});
            if (F.idx_bytes != 0U) segs.push_back({c.idx_cache, F.idx_at, F.idx_bytes});
            if (F.carry_floats != 0U && F.carry_valid) segs.push_back({reinterpret_cast<const std::uint8_t*>(c.carry), F.carry_at, F.carry_floats * 4U});
        }
        if (r.owns_index && F.topk_count != 0U)
            segs.push_back({reinterpret_cast<const std::uint8_t*>(c.idx + std::size_t{last_row - c.row0} * kTopK), F.topk_at, F.topk_count * 4U});
    }
    STRIX_CUDA_CHECK(cudaEventRecord(X.ev[0], stream));
    STRIX_CUDA_CHECK(cudaMemsetAsync(X.dev, 0, f.pieces_end, stream));  // empty ring slots, an invalid carry, the frame's gaps
    enqueue_gather(segs, X.dev, stream);
    STRIX_CUDA_CHECK(cudaEventRecord(X.ev[1], stream));
    STRIX_CUDA_CHECK(cudaMemcpyAsync(X.host, X.dev, f.pieces_end, cudaMemcpyDeviceToHost, stream));
    ++copies;
    STRIX_CUDA_CHECK(cudaEventRecord(X.ev[2], stream));
    STRIX_CUDA_CHECK(cudaStreamSynchronize(stream));
    float pack = 0, d2h = 0;
    STRIX_CUDA_CHECK(cudaEventElapsedTime(&pack, X.ev[0], X.ev[1]));
    STRIX_CUDA_CHECK(cudaEventElapsedTime(&d2h, X.ev[1], X.ev[2]));
    // 3. the frame's own bytes (header, flags, sizes, tokens) around the pieces
    const auto t1 = std::chrono::steady_clock::now();
    deepseek41::write_attention_frame(f, tokens, std::span(X.host, X.host_cap));
    X.timing.frame_ms = ms_since(t1);
    X.timing.pack_ms = pack;
    X.timing.d2h_ms = d2h;
    X.timing.segments = static_cast<std::uint32_t>(segs.size());
    X.timing.copies = copies;
    X.captured = true;
    X.timing.capture_ms = ms_since(t0);
}

}  // namespace strix::ds41::cuda::attn
