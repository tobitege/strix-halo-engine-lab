// FAST WMMA indexer score core (src/attn_core/index_score_wmma.hpp): gate + timing against the exact kernels of
// one engine (built twice: IDXW_MM = MiniMax MSA prefill index, IDXW_GLM = GLM DSA prefill indexer).
// Per shape: exact (STRIX_IDX_WMMA=0) and FAST (=1, GLM also =f16) on the same inputs; checks the written set is
// identical (sentinels), reports score rel-L2 / max error, top-k recall (|A n B| / k, mean |A u B|) and the kernel
// TF/s (median of 5). Inputs are clustered so the top-k is not a pure near-tie lottery.
// Usage: strix-hip-idx-wmma-*-tests [quick]
#include "attn_core/index_score_wmma.hpp"
#if defined(IDXW_MM)
#include "minimax/attn/msa_fused.hpp"
#include "strix/minimax_msa.hpp"
#elif defined(IDXW_GLM)
#include "glm/kernels/glm_dsa.hpp"
#include "glm/prefill/glm_prefill.hpp"
#include "glm/common/glm_math.hpp"
#else
#error "define IDXW_MM or IDXW_GLM"
#endif

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
int failures = 0;
void hok(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::fprintf(stderr, "HIP %s: %s\n", what, hipGetErrorString(e)); std::exit(2); }
}
void check(bool c, const std::string& what) {
    if (!c) { ++failures; std::printf("FAIL %s\n", what.c_str()); }
}
template <class T> T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    hok(hipMalloc(&d, std::max<std::size_t>(h.size(), 1) * sizeof(T)), "malloc");
    if (!h.empty()) hok(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    return d;
}
template <class T> std::vector<T> host(const T* d, std::size_t n) {
    std::vector<T> h(n);
    hok(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "download");
    return h;
}
[[maybe_unused]] std::uint16_t bf16(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U);
}
// Clustered data: 16 unit directions; every key / pooled key is noise + a * direction(block); each query vector is
// noise + b * direction(random cluster).
struct Clusters {
    std::vector<std::vector<float>> dir;
    std::mt19937 rng{1234};
    explicit Clusters(unsigned dim) : dir(16, std::vector<float>(dim)) {
        std::normal_distribution<float> n(0.0F, 1.0F);
        for (auto& d : dir) {
            double s = 0;
            for (auto& x : d) { x = n(rng); s += double{x} * x; }
            for (auto& x : d) x = static_cast<float>(x / std::sqrt(s / dim));
        }
    }
    void vec(float* out, unsigned dim, unsigned cluster, float a, float noise) {
        std::normal_distribution<float> n(0.0F, 1.0F);
        for (unsigned i = 0; i < dim; ++i) out[i] = noise * n(rng) + a * dir[cluster % 16][i];
    }
};
float elapsed_ms(hipEvent_t a, hipEvent_t b) {
    float ms = 0;
    hok(hipEventElapsedTime(&ms, a, b), "elapsed");
    return ms;
}
template <class F> float median_ms(F&& f, unsigned reps = 5) {
    hipEvent_t a, b;
    hok(hipEventCreate(&a), "ev");
    hok(hipEventCreate(&b), "ev");
    f();  // warm
    std::vector<float> t;
    for (unsigned i = 0; i < reps; ++i) {
        hok(hipEventRecord(a, nullptr), "rec");
        f();
        hok(hipEventRecord(b, nullptr), "rec");
        hok(hipEventSynchronize(b), "sync");
        t.push_back(elapsed_ms(a, b));
    }
    std::sort(t.begin(), t.end());
    hok(hipEventDestroy(a), "ev");
    hok(hipEventDestroy(b), "ev");
    return t[t.size() / 2];
}
void set_mode(const char* m) { setenv("STRIX_IDX_WMMA", m, 1); }

#if defined(IDXW_MM)
namespace msa = strix::minimax::msa;
namespace mf = strix::minimax::msa::fused;
// One chunk of `rows` rows at positions pos0.. (all sparse: pos0 >= 2048), cache filled up to pos0 + rows.
void mm_case(unsigned pos0, unsigned rows, bool timing) {
    const unsigned total = pos0 + rows, max_blocks = (total + msa::kBlock - 1) / msa::kBlock, H = 2;
    Clusters cl(msa::kIdxDim);
    std::uniform_int_distribution<unsigned> pick(0, 15);
    std::vector<std::uint16_t> ik(std::size_t{max_blocks} * msa::kBlock * msa::kIdxDim, 0);
    std::vector<float> tmp(msa::kIdxDim);
    for (unsigned t = 0; t < total; ++t) {
        cl.vec(tmp.data(), msa::kIdxDim, (t / msa::kBlock) * 7U, 0.35F, 1.0F);
        for (unsigned i = 0; i < msa::kIdxDim; ++i) ik[std::size_t{t} * msa::kIdxDim + i] = bf16(tmp[i]);
    }
    std::vector<float> iq(std::size_t{rows} * H * msa::kIdxDim), w(msa::kIdxDim);
    for (unsigned q = 0; q < rows * H; ++q) cl.vec(iq.data() + std::size_t{q} * msa::kIdxDim, msa::kIdxDim, pick(cl.rng), 1.5F, 1.0F);
    for (auto& x : w) x = 0.05F * static_cast<float>(pick(cl.rng)) / 15.0F;
    std::vector<msa::RopeRow> rope(rows);
    std::vector<std::uint64_t> pos(rows);
    for (unsigned r = 0; r < rows; ++r) { pos[r] = pos0 + r; rope[r] = msa::rope_row(pos[r], 5000000.0); }
    std::uint16_t* ikc = dev(ik);
    std::uint16_t** ikcp = dev(std::vector<std::uint16_t*>{ikc});
    float* diq = dev(iq);
    float* dw = dev(w);
    msa::RopeRow* drope = dev(rope);
    std::uint64_t* dpos = dev(pos);
    const std::size_t nsc = std::size_t{rows} * H * max_blocks;
    float* scores = dev(std::vector<float>(nsc));
    float* dummy = dev(std::vector<float>(std::size_t{rows} * 32 * msa::kMaxChunks * 130 + 16));
    std::uint32_t* udummy = dev(std::vector<std::uint32_t>(std::size_t{rows} * 2 * 64 + 16));
    mf::DecodeView v{};
    v.seqs = 1; v.rows_per_seq = std::min(rows, 16U); v.q_heads = 32; v.kv_heads = 2; v.idx_heads = 2;
    v.max_blocks = max_blocks;
    v.q = dummy; v.k = dummy; v.v = dummy; v.iq = diq; v.ik = dummy; v.q_w = dw; v.k_w = dw; v.iq_w = dw; v.ik_w = dw;
    v.rope = drope; v.pos = dpos; v.k_cache = ikcp; v.v_cache = ikcp; v.ik_cache = ikcp; v.scores = scores;
    v.idx = udummy; v.idx_count = udummy; v.partial = dummy; v.ticket = udummy; v.o = dummy;
    const unsigned windows = (rows + 15) / 16;
    const mf::PrefillView p{v, rows, pos0, windows, 0, true, 64U};
    static const char* const modes[4] = {"0", "bf16", "bf16x2", "exact"};
    std::vector<float> res[4];
    float ms[4]{};
    for (unsigned m = 0; m < 4; ++m) {
        set_mode(modes[m]);
        hok(hipMemset(scores, 0xa5, nsc * sizeof(float)), "sentinel");
        mf::enqueue_prefill_index(p, nullptr);
        hok(hipDeviceSynchronize(), "run");
        res[m] = host(scores, nsc);
        if (timing) ms[m] = median_ms([&] { mf::enqueue_prefill_index(p, nullptr); });
    }
    set_mode("0");
    for (unsigned fm = 1; fm < 4; ++fm) {
        strix::attn_core::idx::Audit audit;
        double num = 0, den = 0, maxerr = 0;
        bool same_set = true;
        std::uint32_t sentinel;
        std::memset(&sentinel, 0xa5, 4);
        double flops = 0;
        for (unsigned r = 0; r < rows; ++r) {
            const unsigned blocks = static_cast<unsigned>((pos[r] + msa::kBlock) / msa::kBlock);
            flops += 2.0 * H * msa::kIdxDim * static_cast<double>(pos[r] + 1);
            for (unsigned h = 0; h < H; ++h) {
                const float* a = res[0].data() + (std::size_t{r} * H + h) * max_blocks;
                const float* b = res[fm].data() + (std::size_t{r} * H + h) * max_blocks;
                for (unsigned j = 0; j < max_blocks; ++j) {
                    std::uint32_t ua, ub;
                    std::memcpy(&ua, a + j, 4);
                    std::memcpy(&ub, b + j, 4);
                    if ((ua == sentinel) != (ub == sentinel) || ((ua == sentinel) != (j >= blocks))) same_set = false;
                    if (j < blocks) {
                        const double d = double{a[j]} - b[j];
                        num += d * d; den += double{a[j]} * a[j];
                        maxerr = std::max(maxerr, std::fabs(d));
                    }
                }
                audit.add(a, b, max_blocks, msa::kTopK - 1, [&](std::size_t j) { return j + 1 < blocks; });
            }
        }
        check(same_set, std::string("minimax written set ") + modes[fm] + " pos0=" + std::to_string(pos0));
        // "exact": every (row, head) selects the exact set (top kTopK - 1 of the blocks before the newest one).
        if (fm == 3) check(audit.exact_rows == audit.rows, std::string("minimax exact selection pos0=") + std::to_string(pos0));
        const double rel = std::sqrt(num / den);
        check(rel < (fm >= 2 ? 2e-5 : 2e-2), std::string("minimax score rel-L2 ") + modes[fm] + " pos0=" + std::to_string(pos0));
        std::printf("minimax %s pos0=%u rows=%u: rel-L2 %.2e max|d| %.3e; %s\n", modes[fm], pos0, rows, rel, maxerr, audit.line("minimax").c_str());
        if (timing)
            std::printf("minimax %s pos0=%u rows=%u: exact %.3f ms (%.2f TF/s), wmma %.3f ms (%.2f TF/s), speedup %.2fx\n", modes[fm], pos0, rows,
                        ms[0], flops / ms[0] / 1e9, ms[fm], flops / ms[fm] / 1e9, ms[0] / ms[fm]);
    }
    for (void* q : {static_cast<void*>(ikc), static_cast<void*>(ikcp), static_cast<void*>(diq), static_cast<void*>(dw),
                    static_cast<void*>(drope), static_cast<void*>(dpos), static_cast<void*>(scores), static_cast<void*>(dummy),
                    static_cast<void*>(udummy)})
        hok(hipFree(q), "free");
}
#else
namespace P = strix::glm::pf;
namespace D = strix::glm::dev;
using strix::glm::kIdxHeads;
using strix::glm::kIdxDim;
using strix::glm::kKpool;
using strix::glm::kIdxPools;
float key_value(std::uint32_t k) {  // inverse of ordered_key
    const std::uint32_t u = (k & 0x80000000U) ? (k & 0x7fffffffU) : ~k;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
// staged: the bounded call gets its pools one window at a time (the earlier windows are not addressable later).
void glm_case(unsigned t0, unsigned C, bool timing, bool bounded, bool staged = false) {
    const unsigned Pmax = (t0 + C) / kKpool, kcap = bounded ? P::kIndexerKeyStride : Pmax;
    Clusters cl(kIdxDim);
    std::uniform_int_distribution<unsigned> pick(0, 15);
    std::vector<float> pooled(std::size_t{Pmax + 1} * kIdxDim), qI(std::size_t{C} * kIdxHeads * kIdxDim), wh(std::size_t{C} * kIdxHeads);
    for (unsigned p = 0; p < Pmax; ++p) cl.vec(pooled.data() + std::size_t{p} * kIdxDim, kIdxDim, (p / 32U) * 5U, 0.4F, 1.0F);
    for (unsigned q = 0; q < C * kIdxHeads; ++q) cl.vec(qI.data() + std::size_t{q} * kIdxDim, kIdxDim, pick(cl.rng), 1.5F, 1.0F);
    std::normal_distribution<float> n(0.0F, 1.0F);
    for (auto& x : wh) x = n(cl.rng);
    D::DsaSlot slot{};
    slot.pooled = dev(pooled);
    slot.cap = (Pmax + 1) * kKpool;
    float* dq = dev(qI);
    float* dwh = dev(wh);
    std::uint32_t* keys = dev(std::vector<std::uint32_t>(std::size_t{C} * kcap));
    std::uint32_t* tok = dev(std::vector<std::uint32_t>(std::size_t{C} * P::kMaxTokens));
    std::uint32_t* cnt = dev(std::vector<std::uint32_t>(C));
    const auto run = [&] {
        hok(hipMemcpy(dwh, wh.data(), wh.size() * sizeof(float), hipMemcpyHostToDevice), "wh");
        if (bounded && staged)
            P::indexer_select_rows_bounded(slot, t0, dq, dwh, keys, tok, cnt, C, nullptr, 0, [&](unsigned begin, unsigned) {
                return static_cast<const float*>(slot.pooled) + std::size_t{begin} * kIdxDim;
            });
        else if (bounded) P::indexer_select_rows_bounded(slot, t0, dq, dwh, keys, tok, cnt, C, nullptr);
        else P::indexer_select_rows_versions(slot, t0, dq, dwh, keys, kcap, tok, cnt, C, nullptr, 3, 2);
    };
    const char* modes[4] = {"0", "1", "f16", "exact"};
    std::vector<std::uint32_t> k[4], t[4], c[4];
    float ms[4]{};
    for (unsigned m = 0; m < 4; ++m) {
        set_mode(modes[m]);
        hok(hipMemset(keys, 0xa5, std::size_t{C} * kcap * 4), "sentinel");
        run();
        hok(hipDeviceSynchronize(), "run");
        k[m] = host(keys, std::size_t{C} * kcap);
        t[m] = host(tok, std::size_t{C} * P::kMaxTokens);
        c[m] = host(cnt, C);
        if (timing) {
            hipEvent_t ev[3];
            for (auto& e : ev) hok(hipEventCreate(&e), "ev");
            std::vector<float> s;
            for (unsigned rep = 0; rep < 6; ++rep) {
                P::set_idx_profile(ev);
                run();
                P::set_idx_profile(nullptr);
                hok(hipEventSynchronize(ev[2]), "ev");
                if (rep) s.push_back(elapsed_ms(ev[0], ev[1]));
            }
            std::sort(s.begin(), s.end());
            ms[m] = s[s.size() / 2];
            for (auto& e : ev) hok(hipEventDestroy(e), "ev");
        }
    }
    set_mode("0");
    double flops = 0;
    for (unsigned r = 0; r < C; ++r) {
        const unsigned Pr = (t0 + r + 1) / kKpool;
        if (Pr > kIdxPools) flops += 2.0 * kIdxHeads * kIdxDim * Pr;
    }
    {  // exact: the token lists of the v3 path, entry for entry (WMMA scores + exact rescoring of the cut-off)
        std::size_t bad = 0;
        for (unsigned r = 0; r < C; ++r) {
            bad += c[0][r] != c[3][r] ? 1U : 0U;
            for (unsigned i = 0; i < c[0][r] && i < P::kMaxTokens; ++i)
                bad += t[0][std::size_t{r} * P::kMaxTokens + i] != t[3][std::size_t{r} * P::kMaxTokens + i] ? 1U : 0U;
        }
        check(bad == 0, std::string("glm exact mode: token lists equal to v3, t0=") + std::to_string(t0) + ", differing entries " +
                            std::to_string(bad));
        std::printf("glm %s t0=%u C=%u op=exact: token lists identical to v3 (%u rows)\n", staged ? "staged" : bounded ? "bounded" : "single", t0, C, C);
    }
    for (unsigned m = 1; m < 3; ++m) {
        strix::attn_core::idx::Audit audit;
        double num = 0, den = 0, tok_recall = 0, rows = 0;
        bool counts = true;
        for (unsigned r = 0; r < C; ++r) {
            const unsigned Pr = (t0 + r + 1) / kKpool;
            counts = counts && c[0][r] == c[m][r];
            if (Pr <= kIdxPools) continue;
            if (!bounded) {
                for (unsigned p = 0; p < Pr; ++p) {
                    const double a = key_value(k[0][std::size_t{r} * kcap + p]), b = key_value(k[m][std::size_t{r} * kcap + p]);
                    num += (a - b) * (a - b); den += a * a;
                }
                audit.add(k[0].data() + std::size_t{r} * kcap, k[m].data() + std::size_t{r} * kcap, Pr, kIdxPools,
                          [](std::size_t) { return true; });
            }
            std::vector<std::uint32_t> A(t[0].begin() + std::ptrdiff_t{r} * P::kMaxTokens, t[0].begin() + std::ptrdiff_t{r} * P::kMaxTokens + c[0][r]);
            std::vector<std::uint32_t> B(t[m].begin() + std::ptrdiff_t{r} * P::kMaxTokens, t[m].begin() + std::ptrdiff_t{r} * P::kMaxTokens + c[m][r]);
            std::sort(A.begin(), A.end());
            std::sort(B.begin(), B.end());
            std::vector<std::uint32_t> both;
            std::set_intersection(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(both));
            tok_recall += A.empty() ? 1.0 : static_cast<double>(both.size()) / static_cast<double>(A.size());
            rows += 1;
        }
        check(counts, std::string("glm token counts ") + modes[m] + " t0=" + std::to_string(t0));
        const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
        if (!bounded) check(rel < 5e-2, std::string("glm score rel-L2 ") + modes[m] + " t0=" + std::to_string(t0));
        std::printf("glm %s t0=%u C=%u op=%s: score rel-L2 %.2e; token recall %.5f over %.0f rows; %s\n", bounded ? "bounded" : "single",
                    t0, C, modes[m], rel, rows > 0 ? tok_recall / rows : 1.0, rows, audit.line("glm").c_str());
    }
    if (timing)
        std::printf("glm %s t0=%u C=%u scores: exact v3 %.3f ms (%.2f TF/s), wmma bf16 %.3f ms (%.2f TF/s, %.2fx), f16 %.3f ms (%.2fx), exact %.3f ms (%.2fx)\n",
                    staged ? "staged" : bounded ? "bounded" : "single", t0, C, ms[0], flops / ms[0] / 1e9, ms[1], flops / ms[1] / 1e9, ms[0] / ms[1], ms[2], ms[0] / ms[2],
                    ms[3], ms[0] / ms[3]);
    for (void* q : {static_cast<void*>(slot.pooled), static_cast<void*>(dq), static_cast<void*>(dwh), static_cast<void*>(keys),
                    static_cast<void*>(tok), static_cast<void*>(cnt)})
        hok(hipFree(q), "free");
}
#endif
}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "quick") == 0;
#if defined(IDXW_MM)
    mm_case(2048, 100, false);   // short tail tile, partial rows
    mm_case(2100, 333, false);   // unaligned start
    mm_case(4096, 4096, true);   // 8K prompt, second chunk
    if (!quick) mm_case(28672, 4096, true);  // 32K prompt, last chunk
#else
    glm_case(2000, 300, false, false);   // rows below / above the 512-pool threshold
    glm_case(4096, 2048, true, false);
    glm_case(20000, 1000, false, true);  // windowed (pools > 4096)
    glm_case(20000, 1000, false, true, true);  // windowed, pools staged per window
    if (!quick) glm_case(28672, 2048, true, true);
    if (!quick) glm_case(28672, 2048, true, true, true);
#endif
    std::printf("%s\n", failures ? "IDX-WMMA FAIL" : "IDX-WMMA PASS");
    return failures ? 1 : 0;
}
