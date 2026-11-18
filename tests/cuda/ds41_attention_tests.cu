// sm_120 kernels of the deepseek41 attention against the host references, bit for bit: the F16 projections, window
// ring, compressor (ratio 2 with its carried group across calls, ratio 1), index scores, selection (plain, candidate
// source and users, ties), attention (window only and window + compressed), and the layer sequence of AttentionState
// over layers 0..24 of the released plan (decode windows and prefill chunks). STRIX_DEEPSEEK41_GGUF=<file> adds
// layers with real weights (inputs from real token embeddings through the layer's Q8_0 sites); STRIX_DS41_ATTN_BENCH=1
// prints timings.
#include "strix/cuda/ds41_attention.hpp"
#include "strix/cuda/ds41_core.hpp"
#include "strix/deepseek41_attention_host.hpp"
#include "strix/deepseek41_attention_state.hpp"
#include "strix/ds41_formats.hpp"
#include "strix/ds41_layer.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/model_content_hash.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace a = strix::ds41::attn;
namespace ca = strix::ds41::cuda::attn;
namespace sd = strix::deepseek41;
using strix::ds41::cuda::DeviceBuffer;

namespace {

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL %s\n", what.c_str()); }
}
template <class T> bool same(const std::vector<T>& x, const std::vector<T>& y, std::size_t n = 0) {
    if (n == 0) { if (x.size() != y.size()) return false; n = x.size(); }
    if (x.size() < n || y.size() < n) return false;
    return std::memcmp(x.data(), y.data(), n * sizeof(T)) == 0;
}
template <class T> std::size_t first_diff(const std::vector<T>& x, const std::vector<T>& y) {
    for (std::size_t i = 0; i < std::min(x.size(), y.size()); ++i) if (std::memcmp(&x[i], &y[i], sizeof(T)) != 0) return i;
    return x.size() == y.size() ? SIZE_MAX : std::min(x.size(), y.size());
}
template <class T> DeviceBuffer<T> dev(const std::vector<T>& h) {
    DeviceBuffer<T> d(std::max<std::size_t>(h.size(), 1));
    if (!h.empty()) d.upload(h.data(), h.size());
    return d;
}
template <class T> std::vector<T> down(const DeviceBuffer<T>& d, std::size_t n = 0) { return d.download(n); }
bool bench() { const char* e = std::getenv("STRIX_DS41_ATTN_BENCH"); return e != nullptr && e[0] == '1'; }

std::mt19937 rng(1234);
std::vector<float> normal(std::size_t n, float sd) {
    std::normal_distribution<float> d(0.0F, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}
std::vector<std::uint16_t> f16(std::size_t n, float sd) {
    std::vector<std::uint16_t> v(n);
    std::normal_distribution<float> d(0.0F, sd);
    for (auto& x : v) { const __half h = __float2half(d(rng)); std::memcpy(&x, &h, 2); }
    return v;
}
std::vector<float> near_one(std::size_t n) {
    std::uniform_real_distribution<float> d(0.8F, 1.2F);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

// ---- F16 rows ---------------------------------------------------------------------------------------------------
void f16_rows_case(std::uint32_t width, std::uint32_t count, std::uint32_t rows) {
    const auto w = f16(std::size_t{count} * width, 0.05F);
    const auto x = normal(std::size_t{rows} * width, 1.0F);
    const auto w2 = f16(std::size_t{32} * width, 0.05F);
    std::vector<float> ref(std::size_t{rows} * count), ref2(std::size_t{rows} * 32);
    for (std::uint32_t r = 0; r < rows; ++r) {
        strix::ds41::f16_rows(w.data(), width, 0, count, x.data() + std::size_t{r} * width, ref.data() + std::size_t{r} * count);
        strix::ds41::f16_rows(w2.data(), width, 0, 32, x.data() + std::size_t{r} * width, ref2.data() + std::size_t{r} * 32);
    }
    auto dw = dev(w), dw2 = dev(w2);
    auto dx = dev(x);
    DeviceBuffer<float> out(ref.size()), out2(ref2.size());
    ca::F16RowsView v{};
    v.rows = rows;
    v.jobs[0] = {dw.get(), width, count, dx.get(), width, out.get(), count};
    v.jobs[1] = {dw2.get(), width, 32, dx.get(), width, out2.get(), 32};
    v.job_count = 2;
    ca::enqueue_f16_rows(v, nullptr);
    expect(same(down(out), ref) && same(down(out2), ref2),
           "f16 rows width " + std::to_string(width) + " count " + std::to_string(count) + " rows " + std::to_string(rows));
}

// ---- window + compressor over a position sequence ---------------------------------------------------------------
void caches_case(unsigned ratio, unsigned positions, const std::vector<unsigned>& pattern, unsigned capacity) {
    const auto norm = near_one(512), cnorm = near_one(512), knorm = near_one(128);
    const auto wk = f16(128 * 512, 0.04F);
    std::vector<std::uint8_t> ring(std::size_t{capacity} * a::kWindowEntryBytes, 0), kv((positions / std::max(ratio, 1U) + 1) * a::kKvEntryBytes, 0),
        idx((positions / std::max(ratio, 1U) + 1) * a::kIdxEntryBytes, 0);
    a::CompressState st;
    auto dring = dev(ring), dkv = dev(kv), didx = dev(idx);
    auto dnorm = dev(norm), dcnorm = dev(cnorm), dknorm = dev(knorm);
    auto dwk = dev(wk);
    DeviceBuffer<float> dstate(1024);
    dstate.zero();
    DeviceBuffer<std::uint32_t> dvalid(1);
    dvalid.zero();
    ca::RopeTables rope(positions + 1);
    unsigned p0 = 0, k = 0;
    bool ok = true;
    while (p0 < positions) {
        const unsigned R = std::min(pattern[k++ % pattern.size()], positions - p0);
        std::vector<std::uint64_t> pos(R);
        for (unsigned r = 0; r < R; ++r) pos[r] = p0 + r;
        const auto xkv = normal(std::size_t{R} * 512, 1.5F), ckv = normal(std::size_t{R} * 512, 1.0F), csc = normal(std::size_t{R} * 512, 2.0F);
        for (unsigned r0 = 0; r0 < R; r0 += 8) {  // host references take up to 8 rows
            const unsigned n = std::min(8U, R - r0);
            const std::span<const std::uint64_t> ps(pos.data() + r0, n);
            a::window_write({std::span<const float>(xkv.data() + std::size_t{r0} * 512, std::size_t{n} * 512), norm, ps, capacity,
                             ratio == 0 ? a::RopeKind::window : a::RopeKind::compressed}, ring);
            if (ratio != 0)
                a::compress({ratio, std::span<const float>(ckv.data() + std::size_t{r0} * 512, std::size_t{n} * 512),
                             ratio == 2 ? std::span<const float>(csc.data() + std::size_t{r0} * 512, std::size_t{n} * 512) : std::span<const float>{},
                             cnorm, wk, knorm, ps}, st, kv, idx);
        }
        auto dpos = dev(pos);
        auto dxkv = dev(xkv), dckv = dev(ckv), dcsc = dev(csc);
        ca::enqueue_window_write({dxkv.get(), dnorm.get(), rope.table(ratio == 0 ? a::RopeKind::window : a::RopeKind::compressed), dpos.get(), R,
                                  capacity, dring.get()}, nullptr);
        if (ratio != 0) {
            ca::CompressView v{};
            v.ratio = ratio;
            v.rows = R;
            v.kv = dckv.get();
            v.score = dcsc.get();
            v.norm = dcnorm.get();
            v.wk = dwk.get();
            v.knorm = dknorm.get();
            v.rope = rope.table(a::RopeKind::compressed);
            v.pos = dpos.get();
            v.state = dstate.get();
            v.state_valid = dvalid.get();
            v.kv_cache = dkv.get();
            v.idx_cache = didx.get();
            ca::enqueue_compress(v, nullptr);
        }
        STRIX_CUDA_CHECK(cudaDeviceSynchronize());
        if (!same(down(dring), ring)) { ok = false; std::fprintf(stderr, "ring differs after position %u (byte %zu)\n", p0 + R, first_diff(down(dring), ring)); break; }
        if (ratio != 0) {
            if (!same(down(dkv), kv) || !same(down(didx), idx)) {
                ok = false;
                std::fprintf(stderr, "compressed caches differ after position %u (kv byte %zu, idx byte %zu)\n", p0 + R, first_diff(down(dkv), kv),
                             first_diff(down(didx), idx));
                break;
            }
            if (ratio == 2) {
                const auto valid = down(dvalid);
                const auto state = down(dstate);
                std::vector<float> hs(1024);
                std::copy(st.kv.begin(), st.kv.end(), hs.begin());
                std::copy(st.score.begin(), st.score.end(), hs.begin() + 512);
                if ((valid[0] != 0U) != st.valid || (st.valid && !same(state, hs))) { ok = false; std::fprintf(stderr, "carry differs at %u\n", p0 + R); break; }
            }
        }
        p0 += R;
    }
    expect(ok, "window + compress ratio " + std::to_string(ratio) + " over " + std::to_string(positions) + " positions");
}

// ---- index scores + selection ------------------------------------------------------------------------------------
std::vector<std::uint8_t> random_idx_entries(std::size_t n) {
    std::vector<std::uint8_t> e(n * a::kIdxEntryBytes);
    std::uniform_int_distribution<int> byte(0, 255), sc(118, 128);
    for (std::size_t t = 0; t < n; ++t) {
        for (unsigned i = 0; i < 64; ++i) e[t * 68 + i] = static_cast<std::uint8_t>(byte(rng));
        for (unsigned i = 0; i < 4; ++i) e[t * 68 + 64 + i] = static_cast<std::uint8_t>(sc(rng));
    }
    return e;
}

struct SelectParams { unsigned topk = a::kTopK, cand_blocks = a::kCandBlocks, block_size = a::kCandBlockSize; };

// R rows ending at position last; source scores with q1 (pool `first`), then (when first is the candidate source) a
// user with q2 reading the source's candidates.
void index_case(unsigned ratio, std::uint64_t last, unsigned R, a::Pool first, SelectParams sp = {}, double ms_budget = 0) {
    const unsigned count_max = static_cast<unsigned>((last + 1) / ratio);
    const auto entries = random_idx_entries(count_max + 1);
    auto dentries = dev(entries);
    ca::RopeTables rope(static_cast<std::uint32_t>(last + 1));
    std::vector<std::uint64_t> pos(R);
    for (unsigned r = 0; r < R; ++r) pos[r] = last + 1 - R + r;
    auto dpos = dev(pos);
    const unsigned stride = (count_max + sp.block_size - 1) / sp.block_size + 1;
    std::vector<std::vector<std::uint8_t>> hcand(R, std::vector<std::uint8_t>(stride, 0));
    DeviceBuffer<std::uint8_t> dcand(std::size_t{R} * stride);
    dcand.zero();
    DeviceBuffer<std::uint32_t> dkeys(std::size_t{R} * stride), didx(std::size_t{R} * sp.topk), dcount(R);
    DeviceBuffer<float> dscores(std::size_t{R} * std::max(count_max, 1U));
    DeviceBuffer<std::uint32_t> dsurv(std::max<std::size_t>(ca::select_survivor_words(R, count_max, sp.topk), 1));
    DeviceBuffer<std::uint8_t> dprep(std::size_t{R} * ca::kIdxPrepBytes);
    const std::vector<a::Pool> pools = first == a::Pool::candidate_source ? std::vector<a::Pool>{first, a::Pool::candidate_user} : std::vector<a::Pool>{first};
    for (const auto pool : pools) {
        const auto q = normal(std::size_t{R} * 32 * 128, 1.0F), w = normal(std::size_t{R} * 32, 1.0F);
        auto dq = dev(q), dw = dev(w);
        std::vector<std::vector<float>> hs;
        for (unsigned r0 = 0; r0 < R; r0 += 8) {
            const unsigned n = std::min(8U, R - r0);
            std::vector<std::vector<float>> part;
            a::index_scores({std::span<const float>(q.data() + std::size_t{r0} * 4096, std::size_t{n} * 4096), std::span<const float>(w.data() + r0 * 32, n * 32),
                             std::span<const std::uint64_t>(pos.data() + r0, n), ratio, a::RopeKind::compressed}, entries, part);
            for (auto& s : part) hs.push_back(std::move(s));
        }
        ca::IndexView iv{R, ratio, count_max, dq.get(), dw.get(), rope.table(a::RopeKind::compressed), dpos.get(), dentries.get(), dscores.get(), dprep.get()};
        ca::SelectView sv{};
        sv.rows = R; sv.ratio = ratio; sv.max_count = count_max; sv.topk = sp.topk; sv.cand_blocks = sp.cand_blocks; sv.block_size = sp.block_size;
        sv.pool = pool; sv.pos = dpos.get(); sv.scores = dscores.get(); sv.candidates = dcand.get(); sv.cand_stride = stride;
        sv.block_keys = dkeys.get(); sv.idx = didx.get(); sv.idx_count = dcount.get(); sv.survivors = dsurv.get();
        ca::enqueue_index_scores(iv, nullptr);
        ca::enqueue_select(sv, nullptr);
        const auto gs = down(dscores);
        const auto gi = down(didx);
        const auto gc = down(dcount);
        const auto gcand = down(dcand);
        bool ok_s = true, ok_i = true, ok_c = true;
        for (unsigned r = 0; r < R; ++r) {
            const auto& s = hs[r];
            if (std::memcmp(s.data(), gs.data() + std::size_t{r} * count_max, s.size() * 4) != 0) ok_s = false;
            const auto sel = a::select_topk(s, pool, hcand[r], sp.topk, sp.cand_blocks, sp.block_size);
            if (gc[r] != sel.size() || std::memcmp(sel.data(), gi.data() + std::size_t{r} * sp.topk, sel.size() * 4) != 0) ok_i = false;
            if (pool == a::Pool::candidate_source &&
                std::memcmp(hcand[r].data(), gcand.data() + std::size_t{r} * stride, (s.size() + sp.block_size - 1) / sp.block_size) != 0)
                ok_c = false;
        }
        const std::string tag = " ratio " + std::to_string(ratio) + " count " + std::to_string(count_max) + " R " + std::to_string(R) + " pool " +
                                std::to_string(static_cast<int>(pool)) + " topk " + std::to_string(sp.topk);
        expect(ok_s, "index scores" + tag);
        expect(ok_i, "selection" + tag);
        expect(ok_c, "candidates" + tag);
        if (ms_budget > 0 && bench()) {
            strix::ds41::cuda::EventTimer t;
            const int it = 20;
            t.start(nullptr);
            for (int i = 0; i < it; ++i) ca::enqueue_index_scores(iv, nullptr);
            const float ms_s = t.stop(nullptr) / it;
            t.start(nullptr);
            for (int i = 0; i < it; ++i) ca::enqueue_select(sv, nullptr);
            const float ms_k = t.stop(nullptr) / it;
            std::printf("bench index%s: scores %.1f us, select %.1f us\n", tag.c_str(), 1e3 * ms_s, 1e3 * ms_k);
        }
    }
}

// ---- attention ---------------------------------------------------------------------------------------------------
std::vector<std::uint8_t> random_ring(unsigned capacity) {
    std::vector<std::uint8_t> e(std::size_t{capacity} * a::kWindowEntryBytes);
    std::uniform_int_distribution<int> byte(0, 255), sc(122, 130);
    for (unsigned t = 0; t < capacity; ++t) {
        for (unsigned i = 0; i < 512; ++i) { int b = byte(rng); if ((b & 0x7F) == 0x7F) b ^= 1; e[t * 528 + i] = static_cast<std::uint8_t>(b); }
        for (unsigned i = 0; i < 16; ++i) e[t * 528 + 512 + i] = static_cast<std::uint8_t>(sc(rng));
    }
    return e;
}
std::vector<std::uint8_t> random_kv(std::size_t n) {
    std::vector<std::uint8_t> e(n * a::kKvEntryBytes);
    std::uniform_int_distribution<int> byte(0, 255), sc(0x20, 0x50);
    for (std::size_t t = 0; t < n; ++t) {
        for (unsigned i = 0; i < 256; ++i) e[t * 288 + i] = static_cast<std::uint8_t>(byte(rng));
        for (unsigned i = 0; i < 32; ++i) e[t * 288 + 256 + i] = static_cast<std::uint8_t>(sc(rng));
    }
    return e;
}

void attention_case(std::uint64_t last, unsigned R, bool compressed, unsigned ratio, bool timed = false) {
    const unsigned capacity = a::kWindow + std::max(R, a::kMaxRows);
    const auto ring = random_ring(capacity);
    const unsigned count_max = compressed ? static_cast<unsigned>((last + 1) / ratio) : 0U;
    const auto kv = random_kv(count_max + 1);
    std::vector<std::uint64_t> pos(R);
    for (unsigned r = 0; r < R; ++r) pos[r] = last + 1 - R + r;
    std::vector<std::uint32_t> idx(std::size_t{R} * a::kTopK, 0), cnt(R, 0);
    if (compressed) {
        for (unsigned r = 0; r < R; ++r) {
            const unsigned count = static_cast<unsigned>((pos[r] + 1) / ratio);
            std::vector<std::uint32_t> all(count);
            for (unsigned t = 0; t < count; ++t) all[t] = t;
            std::shuffle(all.begin(), all.end(), rng);
            const unsigned take = std::min(count, a::kTopK);
            std::sort(all.begin(), all.begin() + take);
            std::copy_n(all.begin(), take, idx.begin() + std::size_t{r} * a::kTopK);
            cnt[r] = take;
        }
    }
    const auto q = normal(std::size_t{R} * 64 * 512, 1.0F), sink = normal(64, 1.0F);
    const a::RopeKind kind = compressed ? a::RopeKind::compressed : a::RopeKind::window;
    std::vector<float> o(std::size_t{R} * 64 * 512);
    for (unsigned r0 = 0; r0 < R; r0 += 8) {
        const unsigned n = std::min(8U, R - r0);
        a::AttentionArgs args{std::span<const float>(q.data() + std::size_t{r0} * 32768, std::size_t{n} * 32768), sink,
                              std::span<const std::uint64_t>(pos.data() + r0, n), capacity, kind,
                              std::span<const std::uint32_t>(idx.data() + std::size_t{r0} * a::kTopK, std::size_t{n} * a::kTopK),
                              compressed ? std::span<const std::uint32_t>(cnt.data() + r0, n) : std::span<const std::uint32_t>{}, a::kTopK};
        a::attention(args, ring, kv, std::span<float>(o.data() + std::size_t{r0} * 32768, std::size_t{n} * 32768));
    }
    ca::RopeTables rope(static_cast<std::uint32_t>(last + 1));
    auto dring = dev(ring), dkv = dev(kv);
    auto dq = dev(q), dsink = dev(sink);
    auto dpos = dev(pos);
    auto didx = dev(idx), dcnt = dev(cnt);
    DeviceBuffer<float> dpart(std::size_t{R} * 64 * ca::kPartialFloats), dout(o.size());
    ca::AttentionView v{};
    v.rows = R; v.capacity = capacity; v.topk = a::kTopK; v.q = dq.get(); v.sink = dsink.get(); v.rope = rope.table(kind); v.pos = dpos.get();
    v.ring = dring.get(); v.kv_cache = compressed ? dkv.get() : nullptr; v.idx = compressed ? didx.get() : nullptr;
    v.idx_count = compressed ? dcnt.get() : nullptr; v.partial = dpart.get(); v.o = dout.get(); v.heads = 64;
    ca::enqueue_attention(v, nullptr);
    const auto got = down(dout);
    const std::string tag = "attention last " + std::to_string(last) + " R " + std::to_string(R) + (compressed ? " compressed" : " window");
    if (!same(got, o)) {
        const auto i = first_diff(got, o);
        std::fprintf(stderr, "%s: first difference at %zu: %.9g vs %.9g\n", tag.c_str(), i, got[i], o[i]);
    }
    expect(same(got, o), tag);
    if (timed && bench()) {
        strix::ds41::cuda::EventTimer t;
        const int it = 50;
        ca::enqueue_attention(v, nullptr);
        t.start(nullptr);
        for (int i = 0; i < it; ++i) ca::enqueue_attention(v, nullptr);
        std::printf("bench %s: %.1f us\n", tag.c_str(), 1e3 * t.stop(nullptr) / it);
    }
}

// ---- layer level: AttentionState against a host replica of the same sequence ---------------------------------------
struct HostLayerWeights {
    std::vector<float> kv_norm, sinks, comp_norm, k_norm;
    std::vector<std::uint16_t> comp_kv, comp_gate, idx_k, idx_q_b, idx_proj;
};
struct DevLayerWeights {
    DeviceBuffer<float> kv_norm, sinks, comp_norm, k_norm;
    DeviceBuffer<std::uint16_t> comp_kv, comp_gate, idx_k, idx_q_b, idx_proj;
    ca::LayerWeights view() const {
        ca::LayerWeights w;
        w.kv_norm = kv_norm.get(); w.sinks = sinks.get();
        w.compressor_kv = comp_kv.get(); w.compressor_gate = comp_gate.get(); w.compressor_norm = comp_norm.get();
        w.index_k = idx_k.get(); w.index_k_norm = k_norm.get(); w.index_q_b = idx_q_b.get(); w.index_proj = idx_proj.get();
        return w;
    }
};
template <class T> void put(DeviceBuffer<T>& d, const std::vector<T>& h) { if (!h.empty()) { d.resize(h.size()); d.upload(h.data(), h.size()); } }
DevLayerWeights upload(const HostLayerWeights& h) {
    DevLayerWeights d;
    put(d.kv_norm, h.kv_norm); put(d.sinks, h.sinks); put(d.comp_norm, h.comp_norm); put(d.k_norm, h.k_norm);
    put(d.comp_kv, h.comp_kv); put(d.comp_gate, h.comp_gate); put(d.idx_k, h.idx_k); put(d.idx_q_b, h.idx_q_b); put(d.idx_proj, h.idx_proj);
    return d;
}

// The host forward's attention pieces (AttentionHost) for R rows of one call.
class HostAttention {
public:
    HostAttention(std::vector<a::LayerAttention> plan, unsigned capacity, unsigned max_positions, unsigned max_rows, unsigned H, unsigned QA)
        : plan_(std::move(plan)), cap_(capacity), H_(H), QA_(QA), stride_((max_positions + a::kCandBlockSize - 1) / a::kCandBlockSize + 1) {
        L_.resize(plan_.size());
        for (std::size_t l = 0; l < plan_.size(); ++l) {
            L_[l].ring.assign(std::size_t{cap_} * 528, 0);
            if (plan_[l].owns_kv) {
                L_[l].kv.assign((max_positions / plan_[l].ratio + 1) * 288, 0);
                L_[l].ix.assign((max_positions / plan_[l].ratio + 1) * 68, 0);
            }
        }
        cand_.assign(max_rows, std::vector<std::uint8_t>(stride_, 0));
    }
    void layer(unsigned l, const HostLayerWeights& w, const std::vector<std::uint64_t>& pos, const std::vector<float>& x, const std::vector<float>& qr,
               const std::vector<float>& q, const std::vector<float>& kvrow, std::vector<float>& o, const std::vector<bool>* checked = nullptr) {
        const auto& r = plan_[l];
        auto& S = L_[l];
        const unsigned R = static_cast<unsigned>(pos.size());
        std::vector<float> ckv(std::size_t{R} * 512), csc(std::size_t{R} * 512), iq(std::size_t{R} * 4096), iw(std::size_t{R} * 32);
        for (unsigned i = 0; i < R; ++i) {
            if (r.owns_kv) {
                strix::ds41::f16_rows(w.comp_kv.data(), H_, 0, 512, &x[std::size_t{i} * H_], &ckv[std::size_t{i} * 512]);
                if (r.ratio == 2) strix::ds41::f16_rows(w.comp_gate.data(), H_, 0, 512, &x[std::size_t{i} * H_], &csc[std::size_t{i} * 512]);
            }
            if (r.owns_index) {
                strix::ds41::f16_rows(w.idx_q_b.data(), QA_, 0, 4096, &qr[std::size_t{i} * QA_], &iq[std::size_t{i} * 4096]);
                strix::ds41::f16_rows(w.idx_proj.data(), H_, 0, 32, &x[std::size_t{i} * H_], &iw[std::size_t{i} * 32]);
            }
        }
        o.assign(std::size_t{R} * 32768, 0.0F);
        if (r.owns_index) S.topk.assign(R, {});
        for (unsigned r0 = 0; r0 < R; r0 += 8) {
            const unsigned n = std::min(8U, R - r0);
            const std::span<const std::uint64_t> ps(pos.data() + r0, n);
            a::window_write({std::span<const float>(kvrow.data() + std::size_t{r0} * 512, std::size_t{n} * 512), w.kv_norm, ps, cap_, r.rope}, S.ring);
            if (r.owns_kv)
                a::compress({r.ratio, std::span<const float>(ckv.data() + std::size_t{r0} * 512, std::size_t{n} * 512),
                             std::span<const float>(csc.data() + std::size_t{r0} * 512, std::size_t{n} * 512), w.comp_norm, w.idx_k, w.k_norm, ps},
                            S.carry, S.kv, S.ix);
        }
        for (unsigned r0 = 0; r0 < R; r0 += 8) {
            const unsigned n = std::min(8U, R - r0);
            std::span<const std::uint64_t> ps(pos.data() + r0, n);
            if (r.owns_index) {
                const auto& src = L_[static_cast<std::size_t>(r.kv_source)];
                std::vector<std::vector<float>> scores;
                a::index_scores({std::span<const float>(iq.data() + std::size_t{r0} * 4096, std::size_t{n} * 4096),
                                 std::span<const float>(iw.data() + r0 * 32, n * 32), ps, plan_[static_cast<std::size_t>(r.kv_source)].ratio, r.rope},
                                src.ix, scores);
                const auto pool = r.candidate_source ? a::Pool::candidate_source : r.candidate_user ? a::Pool::candidate_user : a::Pool::plain;
                for (unsigned i = 0; i < n; ++i) S.topk[r0 + i] = a::select_topk(scores[i], pool, cand_[r0 + i]);
            }
            // attention of the rows to check (all by default), one row per call when only some are checked
            for (unsigned i0 = 0; i0 < n;) {
                unsigned m = n - i0;
                if (checked != nullptr) {
                    if (!(*checked)[r0 + i0]) { ++i0; continue; }
                    m = 1;
                }
                const unsigned b = r0 + i0;
                std::vector<std::uint32_t> idx(std::size_t{m} * 512, 0), cnt(m, 0);
                std::span<const std::uint8_t> kvc;
                if (r.ratio != 0) {
                    const auto& sel = L_[static_cast<std::size_t>(r.index_source)].topk;
                    for (unsigned i = 0; i < m; ++i) {
                        std::copy(sel[b + i].begin(), sel[b + i].end(), idx.begin() + std::size_t{i} * 512);
                        cnt[i] = static_cast<std::uint32_t>(sel[b + i].size());
                    }
                    kvc = L_[static_cast<std::size_t>(r.kv_source)].kv;
                }
                a::AttentionArgs args{std::span<const float>(q.data() + std::size_t{b} * 32768, std::size_t{m} * 32768), w.sinks,
                                      std::span<const std::uint64_t>(pos.data() + b, m), cap_, r.rope, idx,
                                      r.ratio != 0 ? std::span<const std::uint32_t>(cnt) : std::span<const std::uint32_t>{}, a::kTopK};
                a::attention(args, S.ring, kvc, std::span<float>(o.data() + std::size_t{b} * 32768, std::size_t{m} * 32768));
                i0 += m;
            }
        }
    }
    struct State {
        std::vector<std::uint8_t> ring, kv, ix;
        a::CompressState carry;
        std::vector<std::vector<std::uint32_t>> topk;
    };
    const State& state(unsigned l) const { return L_[l]; }

private:
    std::vector<a::LayerAttention> plan_;
    unsigned cap_, H_, QA_, stride_;
    std::vector<State> L_;
    std::vector<std::vector<std::uint8_t>> cand_;
};

HostLayerWeights synthetic_weights(const a::LayerAttention& r, unsigned H, unsigned QA) {
    HostLayerWeights w;
    w.kv_norm = near_one(512);
    w.sinks = normal(64, 1.0F);
    if (r.owns_kv) {
        w.comp_kv = f16(std::size_t{512} * H, 1.0F / std::sqrt(static_cast<float>(H)));
        if (r.ratio == 2) w.comp_gate = f16(std::size_t{512} * H, 2.0F / std::sqrt(static_cast<float>(H)));
        w.comp_norm = near_one(512);
        w.idx_k = f16(128 * 512, 0.05F);
        w.k_norm = near_one(128);
    }
    if (r.owns_index) {
        w.idx_q_b = f16(std::size_t{4096} * QA, 1.0F / std::sqrt(static_cast<float>(QA)));
        w.idx_proj = f16(std::size_t{32} * H, 1.0F / std::sqrt(static_cast<float>(H)));
    }
    return w;
}

// Layers [0, end) of the released plan, synthetic weights, a call pattern (prefill chunks then decode windows).
void layer_sequence(unsigned end, unsigned H, unsigned QA, unsigned max_positions, unsigned max_rows, const std::vector<unsigned>& calls) {
    const auto plan_arr = a::released_plan();
    std::vector<a::LayerAttention> plan(plan_arr.begin(), plan_arr.end());
    ca::LayerAttentionOptions opt;
    opt.first_layer = 0; opt.end_layer = end; opt.max_positions = max_positions; opt.max_rows = max_rows; opt.window_rows = 32;
    opt.hidden = H; opt.q_lora_rank = QA;
    ca::AttentionState state(plan, opt);
    HostAttention host(plan, state.ring_capacity(), max_positions, max_rows, H, QA);
    std::vector<HostLayerWeights> hw;
    std::vector<DevLayerWeights> dw;
    for (unsigned l = 0; l < end; ++l) { hw.push_back(synthetic_weights(plan[l], H, QA)); dw.push_back(upload(hw.back())); }
    std::uint64_t p0 = 0;
    bool ok = true;
    DeviceBuffer<float> dx(std::size_t{max_rows} * H), dqr(std::size_t{max_rows} * QA), dq(std::size_t{max_rows} * 32768), dkv(std::size_t{max_rows} * 512),
        dout(std::size_t{max_rows} * 32768);
    DeviceBuffer<std::uint64_t> dpos(max_rows);
    for (std::size_t call = 0; call < calls.size() && ok; ++call) {
        const unsigned R = calls[call];
        std::vector<std::uint64_t> pos(R);
        for (unsigned i = 0; i < R; ++i) pos[i] = p0 + i;
        dpos.upload(pos.data(), R);
        for (unsigned l = 0; l < end && ok; ++l) {
            const auto x = normal(std::size_t{R} * H, 1.0F), qr = normal(std::size_t{R} * QA, 1.0F), q = normal(std::size_t{R} * 32768, 1.0F),
                       kv = normal(std::size_t{R} * 512, 1.0F);
            dx.upload(x.data(), x.size()); dqr.upload(qr.data(), qr.size()); dq.upload(q.data(), q.size()); dkv.upload(kv.data(), kv.size());
            ca::LayerIo io;
            io.rows = R; io.pos = dpos.get(); io.pos_bound = p0 + R - 1; io.x = dx.get(); io.qr = dqr.get(); io.q = dq.get(); io.kv = dkv.get(); io.o = dout.get();
            state.enqueue_layer(l, dw[l].view(), io, nullptr);
            std::vector<float> o;
            host.layer(l, hw[l], pos, x, qr, q, kv, o);
            const auto got = dout.download(o.size());
            if (!same(got, o)) {
                ok = false;
                std::fprintf(stderr, "layer %u call %zu (rows %u from %llu): o differs at %zu\n", l, call, R, static_cast<unsigned long long>(p0), first_diff(got, o));
            }
            const auto& c = state.caches(l);
            const auto& hs = host.state(l);
            std::vector<std::uint8_t> ring(hs.ring.size());
            STRIX_CUDA_CHECK(cudaMemcpy(ring.data(), c.ring, ring.size(), cudaMemcpyDeviceToHost));
            if (!same(ring, hs.ring)) { ok = false; std::fprintf(stderr, "layer %u call %zu: ring differs\n", l, call); }
            if (c.kv_cache != nullptr) {
                std::vector<std::uint8_t> kvc(hs.kv.size()), ixc(hs.ix.size());
                STRIX_CUDA_CHECK(cudaMemcpy(kvc.data(), c.kv_cache, kvc.size(), cudaMemcpyDeviceToHost));
                STRIX_CUDA_CHECK(cudaMemcpy(ixc.data(), c.idx_cache, ixc.size(), cudaMemcpyDeviceToHost));
                if (!same(kvc, hs.kv) || !same(ixc, hs.ix)) { ok = false; std::fprintf(stderr, "layer %u call %zu: compressed caches differ\n", l, call); }
            }
        }
        p0 += R;
    }
    expect(ok, "layer sequence 0.." + std::to_string(end - 1) + " over " + std::to_string(p0) + " positions");
}

// ---- real weights ------------------------------------------------------------------------------------------------
struct Gguf {
    std::string path;
    strix::gguf::FileInfo info;
    explicit Gguf(const std::string& p) : path(p), info(strix::gguf::read_file(p)) {}
    const strix::gguf::TensorInfo* find(const std::string& name) const {
        for (const auto& t : info.tensors) if (t.name == name) return &t;
        return nullptr;
    }
    const strix::gguf::TensorInfo& tensor(const std::string& name) const {
        const auto* t = find(name);
        if (t == nullptr) throw std::runtime_error("tensor not found: " + name);
        return *t;
    }
    template <class T> std::vector<T> as(const std::string& name) const {
        const auto& t = tensor(name);
        std::ifstream in(path, std::ios::binary);
        in.seekg(static_cast<std::streamoff>(t.absolute_offset));
        std::vector<T> out(t.byte_size / sizeof(T));
        in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(T)));
        if (!in) throw std::runtime_error("short read: " + name);
        return out;
    }
};

HostLayerWeights real_weights(const Gguf& f, unsigned layer, const a::LayerAttention& r) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    HostLayerWeights w;
    w.kv_norm = f.as<float>(p + "attn_kv_a_norm.weight");
    w.sinks = f.as<float>(p + "attn_sinks.weight");
    if (r.owns_kv) {
        w.comp_kv = f.as<std::uint16_t>(p + "attn_compressor_kv.weight");
        if (r.ratio == 2) w.comp_gate = f.as<std::uint16_t>(p + "attn_compressor_gate.weight");
        w.comp_norm = f.as<float>(p + "attn_compressor_norm.weight");
        w.idx_k = f.as<std::uint16_t>(p + "indexer.attn_k.weight");
        w.k_norm = f.as<float>(p + "indexer.k_norm.weight");
    }
    if (r.owns_index) {
        w.idx_q_b = f.as<std::uint16_t>(p + "indexer.attn_q_b.weight");
        w.idx_proj = f.as<std::uint16_t>(p + "indexer.proj.weight");
    }
    return w;
}

// The attention inputs of `layer` for real tokens: x = attn_norm(collapse(embedding)) with the first layer's identity
// pre (the real data flow of layer 0; for the other layers an embedding-derived stand-in of the residual stream),
// qr = q_a_norm(q_a(x)), q = q_b(qr), kv = attn_kv(x), the Q8_0 x A8 sites on the device (bitwise the host's).
struct LayerInputs { std::vector<float> x, qr, q, kv; };
LayerInputs real_inputs(const Gguf& f, unsigned layer, const std::vector<std::uint32_t>& tokens, unsigned H, unsigned QA) {
    namespace cc = strix::ds41::cuda;
    const std::string p = "blk." + std::to_string(layer) + ".";
    const unsigned T = static_cast<unsigned>(tokens.size());
    static std::vector<std::uint16_t> embd;
    if (embd.empty()) embd = f.as<std::uint16_t>("token_embd.weight");
    const auto attn_norm = f.as<float>(p + "attn_norm.weight");
    const auto qa_norm = f.as<float>(p + "attn_q_a_norm.weight");
    LayerInputs in;
    in.x.resize(std::size_t{T} * H);
    std::vector<float> stream(std::size_t{H} * 4), col(H);
    const auto pre = strix::ds41::hc_identity_pre();
    for (unsigned t = 0; t < T; ++t) {
        strix::ds41::embed_token(embd.data(), H, tokens[t], stream);
        strix::ds41::hc_collapse(stream, pre, col);
        strix::ds41::rms_norm(col, attn_norm, 1e-20F, std::span<float>(in.x.data() + std::size_t{t} * H, H));
    }
    const auto q8 = [&](const std::string& name, const std::vector<float>& x, unsigned width, unsigned count) {
        const auto w = f.as<std::uint8_t>(p + name);
        auto dw = dev(w);
        auto dx = dev(x);
        cc::A8Buffer a8;
        a8.resize(T, width);
        cc::QuantizeA8View qv;
        qv.x = dx.get(); qv.rows = T; qv.width = width; qv.out = a8.view();
        cc::enqueue_quantize_a8(qv, nullptr);
        DeviceBuffer<float> out(std::size_t{T} * count);
        cc::Q8A8View v;
        v.weight = {dw.get(), count, width}; v.count = count; v.x = a8.view(); v.out = out.get();
        cc::enqueue_q8_0_a8(v, nullptr);
        return out.download();
    };
    const auto qa = q8("attn_q_a.weight", in.x, H, QA);
    in.qr.resize(qa.size());
    for (unsigned t = 0; t < T; ++t)
        strix::ds41::rms_norm(std::span<const float>(qa.data() + std::size_t{t} * QA, QA), qa_norm, 1e-20F, std::span<float>(in.qr.data() + std::size_t{t} * QA, QA));
    in.q = q8("attn_q_b.weight", in.qr, QA, 64 * 512);
    in.kv = q8("attn_kv.weight", in.x, H, 512);
    return in;
}

// Layers 0, 1, 2 (one state) and 20 (another) with the file's weights over real tokens: a prefill in chunks, then
// decode windows; o compared bitwise on sampled rows (the last row of every call and every 29th position), caches and
// selections after every call.
void real_layers(const char* path) {
    const Gguf f(path);
    const unsigned H = static_cast<unsigned>(f.tensor("blk.0.attn_kv.weight").dimensions[0]);
    const unsigned QA = static_cast<unsigned>(f.tensor("blk.0.attn_q_b.weight").dimensions[0]);
    std::printf("real file: hidden %u, q_lora_rank %u\n", H, QA);
    const auto plan_arr = a::released_plan();
    const std::vector<a::LayerAttention> plan(plan_arr.begin(), plan_arr.end());
    const std::vector<unsigned> calls{256, 256, 256, 256, 120, 1, 1, 8, 3, 1, 8, 1};
    unsigned total = 0;
    for (const auto c : calls) total += c;
    std::vector<std::uint32_t> tokens(total);
    std::uniform_int_distribution<std::uint32_t> tok(100, 120000);
    for (auto& t : tokens) t = tok(rng);
    for (const auto& range : std::vector<std::pair<unsigned, unsigned>>{{0, 3}, {20, 21}}) {
        ca::LayerAttentionOptions opt;
        opt.first_layer = range.first; opt.end_layer = range.second; opt.max_positions = 4096; opt.max_rows = 256; opt.window_rows = 64;
        opt.hidden = H; opt.q_lora_rank = QA;
        ca::AttentionState state(plan, opt);
        HostAttention host(plan, state.ring_capacity(), opt.max_positions, opt.max_rows, H, QA);
        for (unsigned l = range.first; l < range.second; ++l) {
            const auto hw = real_weights(f, l, plan[l]);
            const auto dw = upload(hw);
            const auto in = real_inputs(f, l, tokens, H, QA);
            DeviceBuffer<float> dx(std::size_t{opt.max_rows} * H), dqr(std::size_t{opt.max_rows} * QA), dq(std::size_t{opt.max_rows} * 32768),
                dkv(std::size_t{opt.max_rows} * 512), dout(std::size_t{opt.max_rows} * 32768);
            DeviceBuffer<std::uint64_t> dpos(opt.max_rows);
            unsigned p0 = 0, rows_checked = 0;
            bool ok = true;
            std::uint64_t hash_last = 0;
            for (const unsigned R : calls) {
                std::vector<std::uint64_t> pos(R);
                for (unsigned i = 0; i < R; ++i) pos[i] = p0 + i;
                dpos.upload(pos.data(), R);
                const auto slice = [&](const std::vector<float>& v, unsigned width) {
                    return std::vector<float>(v.begin() + std::size_t{p0} * width, v.begin() + std::size_t{p0 + R} * width);
                };
                const auto x = slice(in.x, H);
                const auto qr = slice(in.qr, QA);
                const auto q = slice(in.q, 32768);
                const auto kv = slice(in.kv, 512);
                dx.upload(x.data(), x.size()); dqr.upload(qr.data(), qr.size()); dq.upload(q.data(), q.size()); dkv.upload(kv.data(), kv.size());
                ca::LayerIo io;
                io.rows = R; io.pos = dpos.get(); io.pos_bound = p0 + R - 1; io.x = dx.get(); io.qr = dqr.get(); io.q = dq.get(); io.kv = dkv.get();
                io.o = dout.get();
                // layer by layer over the whole sequence: each layer here reads only its own sources (<= itself)
                state.enqueue_layer(l, dw.view(), io, nullptr);
                std::vector<bool> checked(R, false);
                for (unsigned i = 0; i < R; ++i) checked[i] = i + 1 == R || (p0 + i) % 29 == 0;
                std::vector<float> o;
                host.layer(l, hw, pos, x, qr, q, kv, o, &checked);
                const auto got = dout.download(o.size());
                for (unsigned i = 0; i < R; ++i) {
                    if (!checked[i]) continue;
                    ++rows_checked;
                    if (std::memcmp(got.data() + std::size_t{i} * 32768, o.data() + std::size_t{i} * 32768, 32768 * 4) != 0) {
                        if (ok) std::fprintf(stderr, "layer %u position %u: o differs\n", l, p0 + i);
                        ok = false;
                    }
                }
                hash_last = strix::ds41::cuda::fnv1a(got.data() + std::size_t{R - 1} * 32768, 32768);
                const auto& c = state.caches(l);
                const auto& hs = host.state(l);
                std::vector<std::uint8_t> ring(hs.ring.size());
                STRIX_CUDA_CHECK(cudaMemcpy(ring.data(), c.ring, ring.size(), cudaMemcpyDeviceToHost));
                if (!same(ring, hs.ring)) { if (ok) std::fprintf(stderr, "layer %u: ring differs after %u\n", l, p0 + R); ok = false; }
                if (c.kv_cache != nullptr) {
                    std::vector<std::uint8_t> kvc(hs.kv.size()), ixc(hs.ix.size());
                    STRIX_CUDA_CHECK(cudaMemcpy(kvc.data(), c.kv_cache, kvc.size(), cudaMemcpyDeviceToHost));
                    STRIX_CUDA_CHECK(cudaMemcpy(ixc.data(), c.idx_cache, ixc.size(), cudaMemcpyDeviceToHost));
                    if (!same(kvc, hs.kv) || !same(ixc, hs.ix)) { if (ok) std::fprintf(stderr, "layer %u: compressed caches differ after %u\n", l, p0 + R); ok = false; }
                }
                if (c.idx != nullptr) {
                    const auto& tk = hs.topk;
                    std::vector<std::uint32_t> gi(std::size_t{R} * a::kTopK), gc(R);
                    STRIX_CUDA_CHECK(cudaMemcpy(gi.data(), c.idx, gi.size() * 4, cudaMemcpyDeviceToHost));
                    STRIX_CUDA_CHECK(cudaMemcpy(gc.data(), c.idx_count, gc.size() * 4, cudaMemcpyDeviceToHost));
                    for (unsigned i = 0; i < R; ++i)
                        if (gc[i] != tk[i].size() || std::memcmp(gi.data() + std::size_t{i} * a::kTopK, tk[i].data(), tk[i].size() * 4) != 0) {
                            if (ok) std::fprintf(stderr, "layer %u position %u: top-k differs\n", l, p0 + i);
                            ok = false;
                        }
                }
                p0 += R;
            }
            std::printf("real layer %u: %u positions, %u rows of o compared, last o hash %016llx\n", l, p0, rows_checked,
                        static_cast<unsigned long long>(hash_last));
            expect(ok, "real layer " + std::to_string(l));
        }
    }
}

// ---- handoff: export in strix/ds41-attention-state-v1, compare with the host replica, import and continue ------------
// The host replica's ring has the capacity of its largest call (kWindow + rows); the expected export re-slots it to the
// decode capacity AttentionHost::ring_capacity here, independently of export_state.
struct RowInputs { std::vector<float> x, qr, q, kv; };
RowInputs row_inputs(unsigned R, unsigned H, unsigned QA) {
    RowInputs in;
    in.x = normal(std::size_t{R} * H, 1.0F); in.qr = normal(std::size_t{R} * QA, 1.0F); in.q = normal(std::size_t{R} * 32768, 1.0F);
    in.kv = normal(std::size_t{R} * 512, 1.0F);
    return in;
}

struct DevRows {
    DeviceBuffer<float> x, qr, q, kv, out;
    DeviceBuffer<std::uint64_t> pos;
    DevRows(unsigned rows, unsigned H, unsigned QA)
        : x(std::size_t{rows} * H), qr(std::size_t{rows} * QA), q(std::size_t{rows} * 32768), kv(std::size_t{rows} * 512), out(std::size_t{rows} * 32768), pos(rows) {}
    std::vector<float> run(ca::AttentionState& state, unsigned l, const DevLayerWeights& w, const std::vector<std::uint64_t>& p, const RowInputs& in) {
        const auto R = static_cast<unsigned>(p.size());
        pos.upload(p.data(), R);
        x.upload(in.x.data(), in.x.size()); qr.upload(in.qr.data(), in.qr.size()); q.upload(in.q.data(), in.q.size()); kv.upload(in.kv.data(), in.kv.size());
        ca::LayerIo io;
        io.rows = R; io.pos = pos.get(); io.pos_bound = p.back(); io.x = x.get(); io.qr = qr.get(); io.q = q.get(); io.kv = kv.get(); io.o = out.get();
        state.enqueue_layer(l, w.view(), io, nullptr);
        return out.download(std::size_t{R} * 32768);
    }
};

// The expected export of layers [0, end) after `position` positions whose last call had last_rows rows.
sd::AttentionState host_export(const HostAttention& host, const std::vector<a::LayerAttention>& plan, unsigned end, std::uint32_t position, unsigned last_rows) {
    sd::AttentionState s;
    s.position = position;
    s.layers.resize(plan.size());
    for (unsigned l = 0; l < end; ++l) {
        const auto& r = plan[l];
        const auto& h = host.state(l);
        auto& S = s.layers[l];
        S.owns_kv = r.owns_kv; S.owns_index = r.owns_index; S.ratio = r.ratio;
        const auto cap = static_cast<std::uint32_t>(h.ring.size() / 528);
        constexpr std::uint32_t dcap = sd::AttentionHost::ring_capacity;
        S.ring.assign(std::size_t{dcap} * 528, 0);
        for (std::uint32_t p = position - std::min({position, cap, dcap}); p < position; ++p)
            std::memcpy(S.ring.data() + std::size_t{p % dcap} * 528, h.ring.data() + std::size_t{p % cap} * 528, 528);
        if (r.owns_kv) {
            const std::size_t entries = position / r.ratio;
            S.kv.assign(h.kv.begin(), h.kv.begin() + static_cast<std::ptrdiff_t>(entries * 288));
            S.idx_keys.assign(h.ix.begin(), h.ix.begin() + static_cast<std::ptrdiff_t>(entries * 68));
            if (r.ratio == 2) {
                S.carry_valid = h.carry.valid;
                S.carry.assign(1024, 0.0F);
                if (h.carry.valid) {
                    std::copy(h.carry.kv.begin(), h.carry.kv.end(), S.carry.begin());
                    std::copy(h.carry.score.begin(), h.carry.score.end(), S.carry.begin() + 512);
                }
            }
        }
        if (r.owns_index) S.topk = h.topk.at(last_rows - 1);
    }
    return s;
}

bool same_state(const sd::AttentionState& x, const sd::AttentionState& y, bool with_topk, const std::string& tag) {
    bool ok = x.position == y.position && x.layers.size() == y.layers.size();
    for (std::size_t l = 0; ok && l < x.layers.size(); ++l) {
        const auto& p = x.layers[l];
        const auto& q = y.layers[l];
        const auto bad = [&](const char* what) { std::fprintf(stderr, "%s: layer %zu: %s differs\n", tag.c_str(), l, what); ok = false; };
        if (p.owns_kv != q.owns_kv || p.owns_index != q.owns_index || p.ratio != q.ratio) bad("roles");
        if (!same(p.ring, q.ring)) bad("ring");
        if (!same(p.kv, q.kv)) bad("kv");
        if (!same(p.idx_keys, q.idx_keys)) bad("idx_keys");
        if (p.carry_valid != q.carry_valid || !same(p.carry, q.carry)) bad("carry");
        if (with_topk && (p.topk_known != q.topk_known || !same(p.topk, q.topk))) bad("topk");
    }
    if (ok && with_topk && x.hashes() != y.hashes()) { std::fprintf(stderr, "%s: hashes differ\n", tag.c_str()); ok = false; }
    if (ok && x.digest(!with_topk) != y.digest(!with_topk)) { std::fprintf(stderr, "%s: digest differs\n", tag.c_str()); ok = false; }
    return ok;
}

// Layers [0, end) of the released plan: `prefill` calls on a state of max_rows rows (the RTX prefill), export, check
// against the host, serialize round trip, import into a fresh state of import_rows rows (another ring capacity), then
// `decode` calls on both: outputs equal to the uninterrupted run and to the host, final exports equal.
// capture_state (pinned, already serialized) == serialize_attention_state(export_state(...)) byte for byte, digest too
bool capture_matches(const ca::AttentionState& st, ca::ExportCapture& cap, std::uint32_t first, std::uint32_t end, std::uint32_t position,
                     std::uint32_t last_row, const std::vector<std::uint32_t>& tokens, unsigned threads, const std::string& tag) {
    st.capture_state(first, end, position, last_row, tokens, 42, cap, nullptr);
    cap.seal(threads);
    auto ref = st.export_state(first, end, position, last_row, nullptr);
    ref.tokens = tokens;
    ref.next_token = 42;
    const auto rb = sd::serialize_attention_state(ref);
    const auto got = cap.bytes();
    const bool ok = got.size() == rb.size() && std::equal(got.begin(), got.end(), rb.begin()) && cap.digest() == ref.digest() &&
                    st.capture_bound(first, end, position) >= rb.size();
    if (!ok) std::fprintf(stderr, "%s: capture of layers [%u, %u) at %u (last row %u) differs from export + serialize\n", tag.c_str(), first, end, position, last_row);
    return ok;
}

void state_handoff(unsigned end, unsigned H, unsigned QA, unsigned max_positions, unsigned max_rows, unsigned import_rows,
                   const std::vector<unsigned>& prefill, const std::vector<unsigned>& decode) {
    const auto started = std::chrono::steady_clock::now();
    const auto plan_arr = a::released_plan();
    const std::vector<a::LayerAttention> plan(plan_arr.begin(), plan_arr.end());
    ca::LayerAttentionOptions opt;
    opt.first_layer = 0; opt.end_layer = end; opt.max_positions = max_positions; opt.max_rows = max_rows; opt.window_rows = 32;
    opt.hidden = H; opt.q_lora_rank = QA;
    ca::AttentionState run_a(plan, opt);
    HostAttention host(plan, a::kWindow + std::max(max_rows, import_rows), max_positions, std::max(max_rows, import_rows), H, QA);
    std::vector<HostLayerWeights> hw;
    std::vector<DevLayerWeights> dw;
    for (unsigned l = 0; l < end; ++l) { hw.push_back(synthetic_weights(plan[l], H, QA)); dw.push_back(upload(hw.back())); }
    DevRows rows(std::max(max_rows, import_rows), H, QA);
    std::uint32_t p0 = 0;
    unsigned last = 0;
    bool ok = true;
    std::string tag = "handoff " + std::to_string(max_rows) + "->" + std::to_string(import_rows);
    for (const unsigned R : prefill) {
        std::vector<std::uint64_t> pos(R);
        for (unsigned i = 0; i < R; ++i) pos[i] = p0 + i;
        for (unsigned l = 0; l < end && ok; ++l) {
            const auto in = row_inputs(R, H, QA);
            const auto got = rows.run(run_a, l, dw[l], pos, in);
            std::vector<float> o;
            host.layer(l, hw[l], pos, in.x, in.qr, in.q, in.kv, o);
            if (!same(got, o)) { std::fprintf(stderr, "%s: prefill layer %u at %u: o differs\n", tag.c_str(), l, p0); ok = false; }
        }
        p0 += R;
        last = R;
    }
    tag += " at " + std::to_string(p0);
    std::vector<std::uint32_t> tokens(p0);
    for (std::uint32_t i = 0; i < p0; ++i) tokens[i] = (i * 2654435761U) % 129280U;
    auto exported = run_a.export_state(0, end, p0, last - 1, nullptr);
    exported.tokens = tokens;
    exported.next_token = 42;
    if (ok) {
        auto expected = host_export(host, plan, end, p0, last);
        expected.tokens = tokens;
        expected.next_token = 42;
        ok = same_state(exported, expected, true, tag + " export vs host");
    }
    // serialize -> deserialize: same bytes, same digest
    const auto bytes = sd::serialize_attention_state(exported);
    const auto back = sd::deserialize_attention_state(bytes);
    ok = ok && same_state(back, exported, true, tag + " round trip") && sd::serialize_attention_state(back) == bytes;
    {  // the capture into the serialized bytes: whole range and a sub-range, last row known or not, 1 or default threads
        ca::ExportCapture cap;
        ok = ok && capture_matches(run_a, cap, 0, end, p0, last - 1, tokens, 0, tag);
        ok = ok && capture_matches(run_a, cap, 1, end, p0, ca::AttentionState::kNoLastRow, tokens, 1, tag);
        ok = ok && capture_matches(run_a, cap, 0, 3, p0, last - 1, tokens, 2, tag);
    }
    // import into another ring capacity and export again (topk is not imported: the next call selects again)
    ca::LayerAttentionOptions opt_b = opt;
    opt_b.max_rows = import_rows;
    ca::AttentionState run_b(plan, opt_b);
    run_b.import_state(back, nullptr);
    {
        auto again = run_b.export_state(0, end, p0, ca::AttentionState::kNoLastRow, nullptr);
        again.tokens = tokens;
        again.next_token = 42;
        ok = ok && same_state(again, exported, false, tag + " import -> export");
    }
    // decode on both
    for (const unsigned R : decode) {
        if (R > std::min(max_rows, import_rows)) throw std::invalid_argument("state_handoff: decode call larger than a state's max_rows");
        std::vector<std::uint64_t> pos(R);
        for (unsigned i = 0; i < R; ++i) pos[i] = p0 + i;
        for (unsigned l = 0; l < end && ok; ++l) {
            const auto in = row_inputs(R, H, QA);
            const auto oa = rows.run(run_a, l, dw[l], pos, in);
            const auto ob = rows.run(run_b, l, dw[l], pos, in);
            std::vector<float> o;
            host.layer(l, hw[l], pos, in.x, in.qr, in.q, in.kv, o);
            if (!same(oa, o)) { std::fprintf(stderr, "%s: decode layer %u at %u: uninterrupted o differs from the host\n", tag.c_str(), l, p0); ok = false; }
            if (!same(ob, oa)) { std::fprintf(stderr, "%s: decode layer %u at %u: imported o differs (at %zu)\n", tag.c_str(), l, p0, first_diff(ob, oa)); ok = false; }
        }
        p0 += R;
        last = R;
    }
    if (ok) {
        const auto ea = run_a.export_state(0, end, p0, last - 1, nullptr);
        const auto eb = run_b.export_state(0, end, p0, last - 1, nullptr);
        ok = same_state(eb, ea, true, tag + " after decode");
        ca::ExportCapture cap;
        std::vector<std::uint32_t> tk(p0);
        for (std::uint32_t i = 0; i < p0; ++i) tk[i] = i;
        ok = ok && capture_matches(run_a, cap, 0, end, p0, last - 1, tk, 0, tag + " after decode") &&
             capture_matches(run_b, cap, 0, end, p0, last - 1, tk, 0, tag + " after decode (imported)");
        ok = ok && same_state(ea, host_export(host, plan, end, p0, last), true, tag + " after decode vs host");
    }
    std::printf("%s: %llu state bytes, digest %016llx, %.1f s\n", tag.c_str(), static_cast<unsigned long long>(exported.bytes()),
                static_cast<unsigned long long>(exported.digest()), std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    expect(ok, tag);
}

// Export / serialize / import cost of a whole 40-layer state at long positions (caches as they are: sizes and timing).
void handoff_timing(unsigned H, unsigned QA) {
    const auto plan_arr = a::released_plan();
    const std::vector<a::LayerAttention> plan(plan_arr.begin(), plan_arr.end());
    for (const std::uint32_t P : {2048U, 8192U, 32768U}) {
        ca::LayerAttentionOptions opt;
        opt.first_layer = 0; opt.end_layer = 40; opt.max_positions = P + 64; opt.max_rows = 64; opt.window_rows = 32;
        opt.hidden = H; opt.q_lora_rank = QA;
        ca::AttentionState st(plan, opt);
        const auto t0 = std::chrono::steady_clock::now();
        auto s = st.export_state(0, 40, P, 63, nullptr);
        const auto t1 = std::chrono::steady_clock::now();
        s.tokens.assign(P, 1);
        const auto bytes = sd::serialize_attention_state(s);
        const auto t2 = std::chrono::steady_clock::now();
        st.import_state(s, nullptr);
        const auto t3 = std::chrono::steady_clock::now();
        ca::ExportCapture cap;
        cap.reserve(st.capture_bound(0, 40, P));
        st.capture_state(0, 40, P, 63, s.tokens, 0, cap, nullptr);  // warm
        const auto t4 = std::chrono::steady_clock::now();
        st.capture_state(0, 40, P, 63, s.tokens, 0, cap, nullptr);
        const auto t5 = std::chrono::steady_clock::now();
        cap.seal(0, true);
        const auto t6 = std::chrono::steady_clock::now();
        auto ref = st.export_state(0, 40, P, 63, nullptr);
        ref.tokens = s.tokens;
        const auto rb = sd::serialize_attention_state(ref);
        expect(cap.bytes().size() == rb.size() && std::equal(rb.begin(), rb.end(), cap.bytes().begin()) && cap.digest() == ref.digest(),
               "capture at " + std::to_string(P) + " == export + serialize");
        const auto t7 = std::chrono::steady_clock::now();
        const auto sha_portable = strix::model_content_identity_hex(strix::sha256_bytes(std::as_bytes(std::span(rb.data(), rb.size()))));
        const auto t8 = std::chrono::steady_clock::now();
        expect(cap.sha256() == sha_portable, "capture SHA-256 at " + std::to_string(P));
        const auto& tm = cap.timing();
        const auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
        const double rings = 40.0 * sd::AttentionHost::ring_capacity * a::kWindowEntryBytes;
        std::printf("handoff %u positions: %.2f MB attention state (%.2f MB rings + %.1f B/position), %.2f MB serialized; "
                    "export %.1f ms, serialize %.1f ms, import %.1f ms\n",
                    P, static_cast<double>(s.bytes()) / 1e6, rings / 1e6, (static_cast<double>(s.bytes()) - rings) / P,
                    static_cast<double>(bytes.size()) / 1e6, ms(t1 - t0), ms(t2 - t1), ms(t3 - t2));
        std::printf("  capture %.2f ms (counts %.2f, pack %.3f, D2H %.3f, frame %.3f; %u pieces, %u copies) + seal with SHA-256 %.2f ms (SHA thread %.2f ms; "
                    "portable SHA alone %.1f ms)\n",
                    ms(t5 - t4), tm.meta_ms, tm.pack_ms, tm.d2h_ms, tm.frame_ms, tm.segments, tm.copies, ms(t6 - t5), tm.sha_ms, ms(t8 - t7));
    }
}

// ---- timings ---------------------------------------------------------------------------------------------------------
// Per layer kind, real geometry, random weights: a prefill call of N rows from position 0, then decode calls of 1 and
// 8 rows after it, and decode at long contexts (caches as left, timing only).
void layer_bench(unsigned H, unsigned QA) {
    const auto plan_arr = a::released_plan();
    const std::vector<a::LayerAttention> plan(plan_arr.begin(), plan_arr.end());
    const unsigned N = 4096, max_pos = 65536;
    ca::LayerAttentionOptions opt;
    opt.first_layer = 0; opt.end_layer = 25; opt.max_positions = max_pos; opt.max_rows = N; opt.window_rows = 64;
    opt.hidden = H; opt.q_lora_rank = QA;
    ca::AttentionState state(plan, opt);
    std::printf("bench state: %.1f MB of device memory\n", static_cast<double>(state.device_bytes()) / 1e6);
    const unsigned layers[] = {0, 2, 3, 20, 21, 24};
    std::map<unsigned, DevLayerWeights> dw;
    for (const unsigned l : layers) dw.emplace(l, upload(synthetic_weights(plan[l], H, QA)));
    DeviceBuffer<float> dx(std::size_t{N} * H), dqr(std::size_t{N} * QA), dq(std::size_t{N} * 32768), dkv(std::size_t{N} * 512), dout(std::size_t{N} * 32768);
    {
        const auto x = normal(std::size_t{N} * H, 1.0F);
        const auto qr = normal(std::size_t{N} * QA, 1.0F);
        const auto q = normal(std::size_t{N} * 32768, 1.0F);
        const auto kv = normal(std::size_t{N} * 512, 1.0F);
        dx.upload(x.data(), x.size()); dqr.upload(qr.data(), qr.size()); dq.upload(q.data(), q.size()); dkv.upload(kv.data(), kv.size());
    }
    DeviceBuffer<std::uint64_t> dpos(N);
    strix::ds41::cuda::EventTimer timer;
    const auto run = [&](unsigned R, std::uint64_t first, int iters) {
        std::vector<std::uint64_t> pos(R);
        for (unsigned i = 0; i < R; ++i) pos[i] = first + i;
        dpos.upload(pos.data(), R);
        ca::LayerIo io;
        io.rows = R; io.pos = dpos.get(); io.pos_bound = first + R - 1; io.x = dx.get(); io.qr = dqr.get(); io.q = dq.get(); io.kv = dkv.get(); io.o = dout.get();
        std::string line;
        // graph replay: stream launches cost ~6 us each under WSL, which would hide the kernels
        strix::ds41::cuda::Stream st;
        for (const unsigned l : layers) {
            state.enqueue_layer(l, dw.at(l).view(), io, st.get());  // the sources (2, 20, 24) run before their users
            cudaGraph_t g = nullptr;
            cudaGraphExec_t ge = nullptr;
            STRIX_CUDA_CHECK(cudaStreamBeginCapture(st.get(), cudaStreamCaptureModeThreadLocal));
            for (int i = 0; i < iters; ++i) state.enqueue_layer(l, dw.at(l).view(), io, st.get());
            STRIX_CUDA_CHECK(cudaStreamEndCapture(st.get(), &g));
            STRIX_CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
            STRIX_CUDA_CHECK(cudaGraphLaunch(ge, st.get()));
            st.sync();
            timer.start(st.get());
            STRIX_CUDA_CHECK(cudaGraphLaunch(ge, st.get()));
            char b[64];
            std::snprintf(b, sizeof b, " L%u %.1f", l, 1e3 * timer.stop(st.get()) / iters);
            line += b;
            STRIX_CUDA_CHECK(cudaGraphExecDestroy(ge));
            STRIX_CUDA_CHECK(cudaGraphDestroy(g));
        }
        std::printf("bench layer us, R %u at %llu:%s\n", R, static_cast<unsigned long long>(first), line.c_str());
    };
    run(N, 0, 2);
    run(1, N, 20);
    run(8, N, 20);
    run(1, 32767, 20);
    run(8, 32760, 20);
    run(1, 65535, 10);
}

// Per-kernel timings (graph replay of `iters` launches), real geometry, R rows ending at position `last`.
float graph_us(const std::function<void(cudaStream_t)>& f, int iters) {
    strix::ds41::cuda::Stream st;
    f(st.get());
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ge = nullptr;
    STRIX_CUDA_CHECK(cudaStreamBeginCapture(st.get(), cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < iters; ++i) f(st.get());
    STRIX_CUDA_CHECK(cudaStreamEndCapture(st.get(), &g));
    STRIX_CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
    STRIX_CUDA_CHECK(cudaGraphLaunch(ge, st.get()));
    st.sync();
    float best = 1e30F;
    strix::ds41::cuda::EventTimer timer;
    for (int rep = 0; rep < 5; ++rep) {
        timer.start(st.get());
        STRIX_CUDA_CHECK(cudaGraphLaunch(ge, st.get()));
        best = std::min(best, 1e3F * timer.stop(st.get()) / static_cast<float>(iters));
    }
    STRIX_CUDA_CHECK(cudaGraphExecDestroy(ge));
    STRIX_CUDA_CHECK(cudaGraphDestroy(g));
    return best;
}

void kernel_bench(unsigned H, unsigned QA, unsigned R, std::uint64_t last) {
    const unsigned count = static_cast<unsigned>(last + 1);
    const unsigned capacity = a::kWindow + std::max(R, 8U);
    std::vector<std::uint64_t> pos(R);
    for (unsigned r = 0; r < R; ++r) pos[r] = last + 1 - R + r;
    auto dpos = dev(pos);
    ca::RopeTables rope(count);
    auto dring = dev(random_ring(capacity));
    auto dkvc = dev(random_kv(count + 1));
    auto didxc = dev(random_idx_entries(count + 1));
    auto wkv = dev(f16(std::size_t{512} * H, 0.01F));
    auto wgate = dev(f16(std::size_t{512} * H, 0.01F));
    auto wqb = dev(f16(std::size_t{4096} * QA, 0.01F));
    auto wproj = dev(f16(std::size_t{32} * H, 0.01F));
    auto wk = dev(f16(128 * 512, 0.04F));
    auto norm = dev(near_one(512));
    auto knorm = dev(near_one(128));
    auto sink = dev(normal(64, 1.0F));
    auto x = dev(normal(std::size_t{R} * H, 1.0F));
    auto qr = dev(normal(std::size_t{R} * QA, 1.0F));
    auto q = dev(normal(std::size_t{R} * 32768, 1.0F));
    auto kv = dev(normal(std::size_t{R} * 512, 1.0F));
    auto sc = dev(normal(std::size_t{R} * 512, 1.0F));
    auto iq = dev(normal(std::size_t{R} * 4096, 1.0F));
    auto iw = dev(normal(std::size_t{R} * 32, 1.0F));
    DeviceBuffer<float> ckv(std::size_t{R} * 512), cgate(std::size_t{R} * 512), oq(std::size_t{R} * 4096), ow(std::size_t{R} * 32);
    DeviceBuffer<float> state(1024), scores(std::size_t{R} * count), part(std::size_t{R} * 64 * ca::kPartialFloats), out(std::size_t{R} * 32768);
    state.zero();
    DeviceBuffer<std::uint32_t> valid(1), keys(std::size_t{R} * (count / 8 + 2)), idx(std::size_t{R} * 512), cnt(R);
    valid.zero();
    DeviceBuffer<std::uint8_t> prep(std::size_t{R} * ca::kIdxPrepBytes), cand(std::size_t{R} * (count / 8 + 2));
    cand.zero();
    ca::F16RowsView fv{};
    fv.rows = R;
    fv.jobs[0] = {wkv.get(), H, 512, x.get(), H, ckv.get(), 512};
    fv.jobs[1] = {wgate.get(), H, 512, x.get(), H, cgate.get(), 512};
    fv.jobs[2] = {wqb.get(), QA, 4096, qr.get(), QA, oq.get(), 4096};
    fv.jobs[3] = {wproj.get(), H, 32, x.get(), H, ow.get(), 32};
    fv.job_count = 4;
    ca::CompressView cv{};
    cv.ratio = 2; cv.rows = R; cv.kv = kv.get(); cv.score = sc.get(); cv.norm = norm.get(); cv.wk = wk.get(); cv.knorm = knorm.get();
    cv.rope = rope.table(a::RopeKind::compressed); cv.pos = dpos.get(); cv.state = state.get(); cv.state_valid = valid.get();
    cv.kv_cache = dkvc.get(); cv.idx_cache = didxc.get();
    ca::IndexView iv{R, 1, count, iq.get(), iw.get(), rope.table(a::RopeKind::compressed), dpos.get(), didxc.get(), scores.get(), prep.get()};
    ca::SelectView sv{};
    sv.rows = R; sv.ratio = 1; sv.max_count = count; sv.topk = 512; sv.cand_blocks = 2048; sv.block_size = 8; sv.pool = a::Pool::candidate_source;
    sv.pos = dpos.get(); sv.scores = scores.get(); sv.candidates = cand.get(); sv.cand_stride = count / 8 + 2; sv.block_keys = keys.get();
    sv.idx = idx.get(); sv.idx_count = cnt.get();
    DeviceBuffer<std::uint32_t> surv(std::max<std::size_t>(ca::select_survivor_words(R, count, 512), 1));
    sv.survivors = surv.get();
    ca::enqueue_index_scores(iv, nullptr);
    ca::enqueue_select(sv, nullptr);
    STRIX_CUDA_CHECK(cudaDeviceSynchronize());
    ca::AttentionView av{};
    av.rows = R; av.capacity = capacity; av.topk = 512; av.q = q.get(); av.sink = sink.get(); av.rope = rope.table(a::RopeKind::compressed);
    av.pos = dpos.get(); av.ring = dring.get(); av.kv_cache = dkvc.get(); av.idx = idx.get(); av.idx_count = cnt.get(); av.partial = part.get();
    av.o = out.get(); av.heads = 64;
    const int it = 20;
    const float t_f16 = graph_us([&](cudaStream_t s) { ca::enqueue_f16_rows(fv, s); }, it);
    fv.job_count = 2;
    const float t_f16kv = graph_us([&](cudaStream_t s) { ca::enqueue_f16_rows(fv, s); }, it);
    fv.job_count = 4;
    const float t_win = graph_us([&](cudaStream_t s) { ca::enqueue_window_write({kv.get(), norm.get(), rope.table(a::RopeKind::compressed), dpos.get(), R, capacity, dring.get()}, s); }, it);
    const float t_comp = graph_us([&](cudaStream_t s) { ca::enqueue_compress(cv, s); }, it);
    const float t_idx = graph_us([&](cudaStream_t s) { ca::enqueue_index_scores(iv, s); }, it);
    sv.pool = a::Pool::plain;
    const float t_sel = graph_us([&](cudaStream_t s) { ca::enqueue_select(sv, s); }, it);
    sv.pool = a::Pool::candidate_source;
    const float t_selc = graph_us([&](cudaStream_t s) { ca::enqueue_select(sv, s); }, it);
    const float t_att = graph_us([&](cudaStream_t s) { ca::enqueue_attention(av, s); }, it);
    std::printf("kernels us, R %u at %llu: f16 all %.1f (kv+gate %.1f), window %.1f, compress %.1f, index %.1f, select %.1f (cand source %.1f), attention %.1f\n",
                R, static_cast<unsigned long long>(last), t_f16, t_f16kv, t_win, t_comp, t_idx, t_sel, t_selc, t_att);
}

}  // namespace

// Direct-score tests exercise long selection without the quadratic CPU index-score oracle.
void long_selection_cases() {
    auto sentinel = dev(std::vector<std::uint32_t>{0x12345678U});
    for (const auto max_count : {524289U, 2097153U, UINT32_MAX}) {
        ca::SelectView invalid{};
        invalid.rows = 1; invalid.ratio = 1; invalid.max_count = max_count;
        invalid.topk = a::kTopK; invalid.block_size = a::kCandBlockSize;
        invalid.survivors = sentinel.get();
        invalid.long_context = max_count != 524289U;
        bool rejected = false;
        try { ca::enqueue_select(invalid, nullptr); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected && down(sentinel)[0] == 0x12345678U, "selection limit rejected before any kernel");
    }
    unsigned cases = 0;
    for (const unsigned n : {524287U, 524288U, 524289U, 1048575U, 1048576U}) {
        for (unsigned ratio : {1U, 2U}) {
            if (ratio == 2U && n > 524288U) continue;
            const auto stride = (n + a::kCandBlockSize - 1U) / a::kCandBlockSize + 1U;
            const auto position = std::uint64_t{n} * ratio - 1U;
            auto pos = dev(std::vector<std::uint64_t>{position});
            for (unsigned pattern = 0; pattern < 3U; ++pattern) {
                std::vector<float> scores(n);
                for (unsigned i = 0; i < n; ++i) {
                    const auto z = i * 1664525U + 1013904223U;
                    scores[i] = pattern == 0U ? static_cast<float>(z % 100003U) :
                        pattern == 1U ? 0.0F : -__builtin_inff();
                }
                auto dscore = dev(scores);
                for (const auto pool : {a::Pool::plain, a::Pool::candidate_source, a::Pool::candidate_user}) {
                    std::vector<std::uint8_t> mask(stride, 0);
                    if (pool == a::Pool::candidate_user)
                        for (unsigned b = 0; b < stride; ++b) mask[b] = static_cast<std::uint8_t>(b % 17U == 0U);
                    auto dmask = dev(mask);
                    auto expected = a::select_topk(scores, pool, mask);
                    DeviceBuffer<std::uint32_t> keys(stride), idx(a::kTopK), count(1),
                        survivors(ca::select_survivor_words(1, n, a::kTopK));
                    ca::SelectView v{};
                    v.rows = 1; v.ratio = ratio; v.max_count = n;
                    v.topk = a::kTopK; v.cand_blocks = a::kCandBlocks; v.block_size = a::kCandBlockSize;
                    v.pool = pool; v.pos = pos.get(); v.scores = dscore.get(); v.candidates = dmask.get();
                    v.cand_stride = stride; v.block_keys = keys.get(); v.idx = idx.get(); v.idx_count = count.get();
                    v.survivors = survivors.get();
                    v.long_context = n > 524288U;
                    ca::enqueue_select(v, nullptr);
                    expect(down(count)[0] == expected.size() && same(down(idx, expected.size()), expected), "long selection exact top-k");
                    if (pool != a::Pool::plain) expect(same(down(dmask), mask), "long selection candidate mask");
                    ++cases;
                }
            }
        }
    }
    std::printf("long selection: %u cases through 1M entries checked\n", cases);
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        const auto dev_info = strix::ds41::cuda::select_device(0);
        std::printf("device %s (sm_%d%d, %d SMs)\n", dev_info.name.c_str(), dev_info.major, dev_info.minor, dev_info.sms);
        // STRIX_DS41_ATTN_CHUNK_WARPS=1|2|4: the decode attention block form (set_attention_chunk_warps), same bits
        if (const char* e = std::getenv("STRIX_DS41_ATTN_CHUNK_WARPS")) {
            ca::set_attention_chunk_warps(static_cast<unsigned>(std::atoi(e)));
            std::printf("attention chunk warps %s\n", e);
        }
        // STRIX_DS41_ATTN_PREFETCH=1: with 4 chunk warps, every tile issued at once + the all-chunk combine, same bits
        if (const char* e = std::getenv("STRIX_DS41_ATTN_PREFETCH"); e != nullptr && e[0] == '1') {
            ca::set_attention_prefetch(true);
            std::printf("attention prefetch on\n");
        }
        if (const char* mode = std::getenv("STRIX_DS41_SELECT_LONG"); mode && mode[0] == '1') {
            if (dev_info.name.find("RTX PRO 6000") == std::string::npos) throw std::runtime_error("long selection test requires RTX PRO 6000");
            long_selection_cases();
            return failures == 0 ? 0 : 1;
        }
        if (const char* e = std::getenv("STRIX_DS41_ATTN_BENCH"); e != nullptr && e[0] == '3') {  // kernel timings only
            for (const std::uint64_t last : {4095ULL, 32767ULL})
                for (const unsigned R : {1U, 8U, 64U}) kernel_bench(5120, 1280, R, last);
            return 0;
        }
        if (const char* e = std::getenv("STRIX_DS41_ATTN_BENCH"); e != nullptr && e[0] == '2') {  // timings only
            layer_bench(5120, 1280);
            return 0;
        }
        // STRIX_DS41_ATTN_ONLY=handoff: only the state handoff checks
        const char* only = std::getenv("STRIX_DS41_ATTN_ONLY");
        if (only == nullptr || std::strcmp(only, "handoff") != 0) {
        for (const unsigned rows : {1U, 3U, 8U, 13U}) {
            f16_rows_case(1024, 512, rows);
            f16_rows_case(512, 128, rows);
        }
        f16_rows_case(4096, 512, 2);
        std::printf("f16 rows checked\n");
        caches_case(0, 300, {1, 8, 3}, a::kWindow + 8);
        caches_case(2, 333, {1, 2, 3, 4, 5, 6, 7, 8}, a::kWindow + 8);
        caches_case(2, 333, {3, 37, 1, 64, 8, 2}, a::kWindow + 64);
        caches_case(1, 300, {1, 5, 8, 2, 40}, a::kWindow + 64);
        std::printf("window and compressor checked\n");
        index_case(2, 199, 1, a::Pool::plain);
        index_case(2, 1401, 8, a::Pool::plain);
        index_case(1, 100, 3, a::Pool::candidate_source);
        index_case(1, 20007, 2, a::Pool::candidate_source);  // 2501 blocks > 2048
        index_case(1, 4000, 4, a::Pool::candidate_source, SelectParams{64, 8, 8});
        index_case(1, 32767, 1, a::Pool::plain, {}, 1);
        index_case(1, 32767, 8, a::Pool::plain, {}, 1);
        index_case(1, 131071, 1, a::Pool::candidate_source, {}, 1);
        std::printf("indexer and selection checked\n");
        attention_case(5, 1, false, 0);
        attention_case(300, 8, false, 0, true);
        attention_case(5, 3, true, 2);
        attention_case(2000, 1, true, 2, true);
        attention_case(2000, 8, true, 1, true);
        attention_case(4000, 37, true, 1, true);
        std::printf("attention checked\n");
        layer_sequence(25, 512, 256, 1024, 64, {37, 64, 1, 1, 3, 8, 2, 7, 64, 30, 1, 5});
        std::printf("layer sequence checked\n");
        }
        // (prefill rows, import rows): other ring capacities on both sides; odd and even final positions (carry valid or
        // not); a position below the window; a ring of 128 + max_positions entries as the prefill driver keeps it
        // (the host replica takes ~1 min per case; STRIX_DS41_ATTN_HANDOFF_CASE=N runs case N only)
        const char* hc = std::getenv("STRIX_DS41_ATTN_HANDOFF_CASE");
        const int handoff_case = hc != nullptr ? std::atoi(hc) : -1;
        if (handoff_case < 0 || handoff_case == 0) state_handoff(25, 512, 256, 1024, 64, 8, {37, 64, 1, 3, 40}, {1, 3, 8});
        if (handoff_case < 0 || handoff_case == 1) state_handoff(25, 512, 256, 1024, 8, 64, {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 7}, {1, 8});
        if (handoff_case < 0 || handoff_case == 2) state_handoff(25, 512, 256, 1024, 64, 64, {37, 5, 20}, {1, 8, 64});
        if (handoff_case < 0 || handoff_case == 3) state_handoff(25, 512, 256, 1024, 1024, 16, {37, 64, 64, 3}, {1, 7, 16});
        std::printf("state handoff checked\n");
        handoff_timing(512, 256);
        if (const char* path = std::getenv("STRIX_DEEPSEEK41_GGUF")) real_layers(path);
        if (bench()) {
            const char* hq = std::getenv("STRIX_DS41_ATTN_BENCH_DIMS");  // "hidden,q_lora_rank"
            unsigned H = 5120, QA = 1280;
            if (hq != nullptr) std::sscanf(hq, "%u,%u", &H, &QA);
            layer_bench(H, QA);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    if (failures != 0) { std::fprintf(stderr, "%d failures\n", failures); return 1; }
    std::printf("all ds41 cuda attention checks passed\n");
    return 0;
}
