// Fast prefill attention (cuda_ds41_fast_attention.hpp) against the exact kernels (cuda_ds41_attention.hpp) on
// synthetic caches at prefill shapes, and the times of both.
//   strix-cuda-ds41-fast-attention-tests [T=4096] [reps=2] [--device N]
// Per layer kind (win: window only; r2 / r1: ratio-2 / ratio-1 compressed entries, plain pool): device ms of the exact
// index scores / selection / attention (64-row windows) and of the fast path (FP8 index keys once, index scores and
// selection per 1024-row chunk, attention per chunk with the A8 rows only); the error of the fast o on the exact
// selection (max |d|, mean |d| / mean |o|, minimum cosine per (row, head)), the fused A8 rows against quantize_a8 of the
// fast o (bitwise), the overlap of the fast and exact top-k and the error of the whole fast path. Exits non-zero when the
// minimum cosine is below 0.999, the A8 rows differ or the mean overlap is below 0.97.
#include "strix/cuda/ds41_attention.hpp"
#include "strix/cuda/ds41_core.hpp"
#include "strix/cuda/ds41_fast_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace a = strix::ds41::attn;
namespace ca = strix::ds41::cuda::attn;
namespace cu = strix::ds41::cuda;
using cu::DeviceBuffer;

namespace {

std::mt19937 rng(77);
template <class T> DeviceBuffer<T> dev(const std::vector<T>& h) {
    DeviceBuffer<T> d(std::max<std::size_t>(h.size(), 1));
    d.upload(h.data(), h.size());
    return d;
}
std::vector<float> normal(std::size_t n, float sd) {
    std::normal_distribution<float> d(0.0F, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}
// Entries as the writers produce them: codes of a normal vector (E4M3 window / E2M1 compressed / E2M1 index keys)
// with per-block scales around the data's magnitude.
std::vector<std::uint8_t> window_entries(std::size_t n) {
    std::vector<std::uint8_t> e(n * a::kWindowEntryBytes);
    std::normal_distribution<float> d(0.0F, 1.0F);
    for (std::size_t t = 0; t < n; ++t) {
        std::uint8_t* x = e.data() + t * a::kWindowEntryBytes;
        for (unsigned b = 0; b < 16; ++b) {
            float v[32], amax = 1e-4F;
            for (auto& f : v) { f = d(rng); amax = std::max(amax, std::fabs(f)); }
            const int ex = a::log2_ceil(amax / 448.0F);
            const float s = a::pow2f(ex);
            for (unsigned i = 0; i < 32; ++i) x[32 * b + i] = a::e4m3_encode(a::clampf(v[i] / s, -448.0F, 448.0F));
            x[512 + b] = static_cast<std::uint8_t>(ex + 127);
        }
    }
    return e;
}
std::vector<std::uint8_t> kv_entries(std::size_t n) {
    std::vector<std::uint8_t> e(n * a::kKvEntryBytes);
    std::normal_distribution<float> d(0.0F, 1.0F);
    for (std::size_t t = 0; t < n; ++t) {
        std::uint8_t* x = e.data() + t * a::kKvEntryBytes;
        for (unsigned b = 0; b < 32; ++b) {
            float v[16], amax = 0.01171875F;
            for (auto& f : v) { f = d(rng); amax = std::max(amax, std::fabs(f)); }
            const std::uint8_t sc = a::e4m3_encode(amax / 6.0F);
            const float s = a::e4m3_value(sc);
            for (unsigned i = 0; i < 16; i += 2) {
                const std::uint8_t lo = a::e2m1_encode(a::clampf(v[i] / s, -6.0F, 6.0F)), hi = a::e2m1_encode(a::clampf(v[i + 1] / s, -6.0F, 6.0F));
                x[8 * b + i / 2] = static_cast<std::uint8_t>(lo | (hi << 4U));
            }
            x[256 + b] = sc;
        }
    }
    return e;
}
std::vector<std::uint8_t> idx_entries(std::size_t n) {
    std::vector<std::uint8_t> e(n * a::kIdxEntryBytes);
    std::uniform_int_distribution<int> byte(0, 255), sc(118, 128);
    for (std::size_t t = 0; t < n; ++t) {
        for (unsigned i = 0; i < 64; ++i) e[t * a::kIdxEntryBytes + i] = static_cast<std::uint8_t>(byte(rng));
        for (unsigned i = 64; i < 68; ++i) e[t * a::kIdxEntryBytes + i] = static_cast<std::uint8_t>(sc(rng));
    }
    return e;
}

struct Timer {
    cudaEvent_t a, b;
    Timer() { STRIX_CUDA_CHECK(cudaEventCreate(&a)); STRIX_CUDA_CHECK(cudaEventCreate(&b)); }
    ~Timer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start() { STRIX_CUDA_CHECK(cudaEventRecord(a, nullptr)); }
    float stop() {
        STRIX_CUDA_CHECK(cudaEventRecord(b, nullptr));
        STRIX_CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0;
        STRIX_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        return ms;
    }
};

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<std::string> pos_args;
    int device = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = std::atoi(argv[++i]);
        else pos_args.emplace_back(argv[i]);
    }
    const unsigned T = !pos_args.empty() ? static_cast<unsigned>(std::atoi(pos_args[0].c_str())) : 4096U;
    const int reps = pos_args.size() > 1 ? std::atoi(pos_args[1].c_str()) : 2;
    const unsigned W = 64, C = std::min(1024U, T), cap = a::kWindow + T, H = a::kHeads, QH = H * a::kHeadDim;
    bool ok = true;
    try {
        cu::select_device(device);
        cudaDeviceProp prop{};
        STRIX_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::printf("device %d: %s sm_%d%d, %d SMs\n", device, prop.name, prop.major, prop.minor, prop.multiProcessorCount);
        ca::RopeTables rope(T + 1);
        const auto ring = dev(window_entries(cap));
        const auto kv = dev(kv_entries(T + 1));
        const auto idxc = dev(idx_entries(T + 1));
        std::vector<std::uint64_t> pos(T);
        for (unsigned i = 0; i < T; ++i) pos[i] = i;
        const auto dpos = dev(pos);
        // q rows: a few large heads, most moderate (sd 0.5 .. 4 per head)
        std::vector<float> qh(std::size_t{C} * QH);
        {
            std::normal_distribution<float> d(0.0F, 1.0F);
            for (unsigned r = 0; r < C; ++r)
                for (unsigned h = 0; h < H; ++h) {
                    const float sd = 0.5F + 3.5F * static_cast<float>((h * 37U) % 64U) / 63.0F;
                    for (unsigned i = 0; i < a::kHeadDim; ++i) qh[(std::size_t{r} * H + h) * a::kHeadDim + i] = d(rng) * sd;
                }
        }
        const auto q = dev(qh);
        // indexer inputs of C rows (row r uses row r % C, in the exact windows as in the fast chunks)
        const auto iq = dev(normal(std::size_t{C} * 32 * 128, 1.0F));
        const auto iw = dev(normal(std::size_t{C} * 32, 1.0F));
        const auto sink = dev(normal(64, 1.0F));
        const unsigned stride = (T + a::kCandBlockSize - 1) / a::kCandBlockSize + 1;
        DeviceBuffer<float> scores(std::size_t{W} * (T + 1)), fscores(std::size_t{C} * (T + 1)), partial(std::size_t{W} * 64 * ca::kPartialFloats);
        DeviceBuffer<float> o_exact(std::size_t{C} * QH), o_fast(std::size_t{C} * QH);
        DeviceBuffer<std::uint8_t> qprep(std::size_t{W} * ca::kIdxPrepBytes), fqprep(std::size_t{C} * ca::kFastIdxPrepBytes), cand(std::size_t{C} * stride);
        DeviceBuffer<std::uint32_t> bkeys(std::size_t{C} * stride), idx(std::size_t{T} * a::kTopK), cnt(T), fidx(std::size_t{T} * a::kTopK), fcnt(T);
        DeviceBuffer<std::uint32_t> surv(std::max<std::size_t>(1, ca::select_survivor_words(C, T + 1, a::kTopK)));
        DeviceBuffer<std::uint8_t> k8(std::size_t{T + 1} * a::kIdxDim);
        DeviceBuffer<float> kscale(T + 1);
        const bool fast_index = ca::fast_index_supported(device);
        // the caches as FP16 rows (every position of the ring, every compressed entry), converted once (second call timed)
        DeviceBuffer<std::uint16_t> win16(std::size_t{T} * a::kHeadDim), kv16(std::size_t{T + 1} * a::kHeadDim);
        {
            Timer tk;
            const ca::FastKvView kvv{ring.get(), cap, 0, T, win16.get(), kv.get(), T + 1, kv16.get()};
            ca::enqueue_fast_kv(kvv, nullptr);
            tk.start();
            ca::enqueue_fast_kv(kvv, nullptr);
            std::printf("FP16 entries (%u window + %u compressed): %.3f ms\n", T, T + 1, tk.stop());
        }
        cu::A8Buffer a8f, a8r;
        a8f.resize(C, QH);
        a8r.resize(C, QH);
        cand.zero();

        struct Kind { const char* name; unsigned ratio; };
        for (const Kind k : {Kind{"win", 0}, Kind{"r2", 2}, Kind{"r1", 1}}) {
            const auto rope_t = rope.table(k.ratio ? a::RopeKind::compressed : a::RopeKind::window);
            double t_scores = 0, t_select = 0, t_exact = 0, t_fast = 0, t_fscores = 0, t_fselect = 0, t_fkeys = 0;
            double max_abs = 0, sum_abs = 0, sum_ref = 0, min_cos = 1.0, min_cos2 = 1.0, sum_abs2 = 0;
            double ov_sum = 0, ov_min = 1.0;
            std::size_t ov_rows = 0, a8_diff = 0;
            Timer tm;
            const auto select_view = [&](unsigned rows, unsigned w0, unsigned max_count, const float* sc, std::uint32_t* ix, std::uint32_t* ct) {
                ca::SelectView sv{};
                sv.rows = rows; sv.ratio = k.ratio; sv.max_count = max_count; sv.topk = a::kTopK; sv.cand_blocks = a::kCandBlocks;
                sv.block_size = a::kCandBlockSize; sv.pool = a::Pool::plain; sv.pos = dpos.get() + w0; sv.scores = sc;
                sv.candidates = cand.get(); sv.cand_stride = stride; sv.block_keys = bkeys.get();
                sv.idx = ix + std::size_t{w0} * a::kTopK; sv.idx_count = ct + w0; sv.survivors = surv.get();
                return sv;
            };
            // exact selection of every row (timed per 64-row window, as the prefill runs it)
            for (unsigned w0 = 0; w0 < T; w0 += W) {
                const unsigned n = std::min(W, T - w0);
                const unsigned max_count = k.ratio ? (w0 + n) / k.ratio : 0U;
                if (k.ratio == 0U || max_count == 0U) {
                    STRIX_CUDA_CHECK(cudaMemset(cnt.get() + w0, 0, n * 4U));
                    STRIX_CUDA_CHECK(cudaMemset(fcnt.get() + w0, 0, n * 4U));
                    continue;
                }
                ca::IndexView iv{n, k.ratio, max_count, iq.get() + std::size_t{w0 % C} * 32 * 128, iw.get() + std::size_t{w0 % C} * 32, rope_t,
                                 dpos.get() + w0, idxc.get(), scores.get(), qprep.get()};
                const ca::SelectView sv = select_view(n, w0, max_count, scores.get(), idx.get(), cnt.get());
                double bs = 1e30, bl = 1e30;
                for (int rep = 0; rep < reps; ++rep) {
                    tm.start(); ca::enqueue_index_scores(iv, nullptr); bs = std::min<double>(bs, tm.stop());
                    tm.start(); ca::enqueue_select(sv, nullptr); bl = std::min<double>(bl, tm.stop());
                }
                t_scores += bs;
                t_select += bl;
            }
            // fast index: FP8 keys once, scores and selection per chunk of C rows; overlap with the exact selection
            if (k.ratio != 0U && fast_index) {
                const ca::FastIndexKeysView kk{idxc.get(), T / k.ratio, k8.get(), kscale.get()};
                ca::enqueue_fast_index_keys(kk, nullptr);
                tm.start();
                ca::enqueue_fast_index_keys(kk, nullptr);
                t_fkeys = tm.stop();
                for (unsigned c0 = 0; c0 < T; c0 += C) {
                    const unsigned n = std::min(C, T - c0), max_count = (c0 + n) / k.ratio;
                    if (max_count == 0U) continue;
                    ca::FastIndexView fv{n, k.ratio, max_count, iq.get(), iw.get(), rope_t, dpos.get() + c0, k8.get(), kscale.get(), fscores.get(), fqprep.get()};
                    const ca::SelectView sv = select_view(n, c0, max_count, fscores.get(), fidx.get(), fcnt.get());
                    double bs = 1e30, bl = 1e30;
                    for (int rep = 0; rep < reps; ++rep) {
                        tm.start(); ca::enqueue_fast_index_scores(fv, nullptr); bs = std::min<double>(bs, tm.stop());
                        tm.start(); ca::enqueue_select(sv, nullptr); bl = std::min<double>(bl, tm.stop());
                    }
                    t_fscores += bs;
                    t_fselect += bl;
                }
                const auto ei = idx.download(std::size_t{T} * a::kTopK), fi = fidx.download(std::size_t{T} * a::kTopK);
                const auto ec = cnt.download(T), fc = fcnt.download(T);
                for (unsigned r = 0; r < T; ++r) {
                    if (ec[r] == 0U) continue;
                    unsigned i = 0, j = 0, both = 0;
                    const std::uint32_t* x = ei.data() + std::size_t{r} * a::kTopK;
                    const std::uint32_t* y = fi.data() + std::size_t{r} * a::kTopK;
                    while (i < ec[r] && j < fc[r]) {
                        if (x[i] == y[j]) { ++both; ++i; ++j; } else if (x[i] < y[j]) ++i; else ++j;
                    }
                    const double ov = static_cast<double>(both) / std::max(ec[r], fc[r]);
                    ov_sum += ov;
                    ov_min = std::min(ov_min, ov);
                    ++ov_rows;
                }
            } else if (k.ratio != 0U) {
                STRIX_CUDA_CHECK(cudaMemcpy(fidx.get(), idx.get(), std::size_t{T} * a::kTopK * 4U, cudaMemcpyDeviceToDevice));
                STRIX_CUDA_CHECK(cudaMemcpy(fcnt.get(), cnt.get(), std::size_t{T} * 4U, cudaMemcpyDeviceToDevice));
            }
            // attention per chunk of C rows: exact in 64-row windows, fast in one call
            for (unsigned c0 = 0; c0 < T; c0 += C) {
                const unsigned n = std::min(C, T - c0);
                ca::AttentionView av{};
                av.capacity = cap; av.topk = a::kTopK; av.sink = sink.get(); av.rope = rope_t; av.ring = ring.get();
                av.partial = partial.get(); av.heads = H;
                if (k.ratio) { av.kv_cache = kv.get(); }
                ca::FastAttentionView fv{};
                fv.rows = n; fv.topk = a::kTopK; fv.q = q.get(); fv.sink = sink.get(); fv.rope = rope_t;
                fv.pos = dpos.get() + c0; fv.win16 = win16.get(); fv.win_first = 0;
                if (k.ratio) { fv.kv16 = kv16.get(); fv.idx = idx.get() + std::size_t{c0} * a::kTopK; fv.idx_count = cnt.get() + c0; }
                fv.a8 = a8f.view(); fv.a8.rows = n; fv.heads = H;
                double be = 1e30, bf = 1e30;
                for (int rep = 0; rep < reps; ++rep) {
                    tm.start();
                    for (unsigned w0 = 0; w0 < n; w0 += W) {
                        av.rows = std::min(W, n - w0);
                        av.q = q.get() + std::size_t{w0} * QH;
                        av.pos = dpos.get() + c0 + w0;
                        if (k.ratio) { av.idx = idx.get() + std::size_t{c0 + w0} * a::kTopK; av.idx_count = cnt.get() + c0 + w0; }
                        av.o = o_exact.get() + std::size_t{w0} * QH;
                        ca::enqueue_attention(av, nullptr);
                    }
                    be = std::min<double>(be, tm.stop());
                    fv.o = nullptr;
                    tm.start();
                    ca::enqueue_fast_attention(fv, nullptr);  // timed as the prefill runs it: the A8 rows only
                    bf = std::min<double>(bf, tm.stop());
                }
                t_exact += be;
                t_fast += bf;
                fv.o = o_fast.get();
                ca::enqueue_fast_attention(fv, nullptr);
                // accuracy (every row of the chunk) and the fused A8 against quantize_a8 of the fast o
                cu::enqueue_quantize_a8({o_fast.get(), n, QH, 0, a8r.view()}, nullptr);
                STRIX_CUDA_CHECK(cudaDeviceSynchronize());
                const auto e = o_exact.download(std::size_t{n} * QH);
                const auto cmp = [&](const std::vector<float>& f, double& mx, double& sabs, double& sref, double& mc) {
                    for (unsigned r = 0; r < n; ++r)
                        for (unsigned h = 0; h < H; ++h) {
                            double dot = 0, ne = 0, nf = 0;
                            for (unsigned i = 0; i < a::kHeadDim; ++i) {
                                const std::size_t at = (std::size_t{r} * H + h) * a::kHeadDim + i;
                                const double x = e[at], y = f[at];
                                dot += x * y; ne += x * x; nf += y * y;
                                mx = std::max(mx, std::fabs(x - y));
                                sabs += std::fabs(x - y);
                                sref += std::fabs(x);
                            }
                            const double cs = ne > 0 && nf > 0 ? dot / std::sqrt(ne * nf) : (ne == nf ? 1.0 : 0.0);
                            mc = std::min(mc, cs);
                        }
                };
                cmp(o_fast.download(std::size_t{n} * QH), max_abs, sum_abs, sum_ref, min_cos);
                const std::size_t nc = std::size_t{n} * QH, ng = nc / 32;
                const auto c1 = a8f.codes.download(nc), c2 = a8r.codes.download(nc);
                const auto d1 = a8f.deltas.download(ng), d2 = a8r.deltas.download(ng);
                const auto s1 = a8f.sums.download(ng), s2 = a8r.sums.download(ng);
                for (std::size_t i = 0; i < c1.size(); ++i) a8_diff += c1[i] != c2[i];
                for (std::size_t i = 0; i < d1.size(); ++i) a8_diff += (std::memcmp(&d1[i], &d2[i], 4) != 0) + (s1[i] != s2[i]);
                // the whole fast path: fast selection + fast attention
                if (k.ratio) {
                    fv.idx = fidx.get() + std::size_t{c0} * a::kTopK;
                    fv.idx_count = fcnt.get() + c0;
                    ca::enqueue_fast_attention(fv, nullptr);
                    double mx = 0, sref = 0;
                    cmp(o_fast.download(std::size_t{n} * QH), mx, sum_abs2, sref, min_cos2);
                }
            }
            std::printf("T %5u %-4s exact: scores %7.2f select %7.2f attention %8.2f ms | fast: keys %.2f scores %7.2f select %7.2f "
                        "attention %7.2f ms | attention x%.1f, index x%.1f\n",
                        T, k.name, t_scores, t_select, t_exact, t_fkeys, t_fscores, t_fselect, t_fast, t_exact / t_fast,
                        t_fscores + t_fselect > 0 ? (t_scores + t_select) / (t_fscores + t_fselect + t_fkeys) : 0.0);
            std::printf("           fast attention on the exact selection: o err max %.3e mean rel %.3e min cos %.6f | a8 diffs %zu\n", max_abs,
                        sum_abs / std::max(sum_ref, 1e-30), min_cos, a8_diff);
            if (k.ratio)
                std::printf("           fast selection: top-k overlap mean %.4f min %.4f over %zu rows | fast index + attention: mean rel %.3e min cos %.6f\n",
                            ov_rows ? ov_sum / ov_rows : 1.0, ov_min, ov_rows, sum_abs2 / std::max(sum_ref, 1e-30), min_cos2);
            if (min_cos < 0.999 || a8_diff != 0 || !std::isfinite(max_abs) || (ov_rows && ov_sum / ov_rows < 0.97)) ok = false;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
