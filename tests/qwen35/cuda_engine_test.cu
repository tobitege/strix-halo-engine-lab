// CudaEngine bitwise tests on a small synthetic qwen35 model (device generated weights):
//  1. serial decode == forward_rows(R) + commit(k) chains (random R and k, also k = 0 and k < R), logits and hidden, bf16 and q8_0
//  2. CUDA graph on == graph off
//  3. MTP: rows in one call == rows one at a time
// Home: specific (qwen35 engine test).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "strix/qwen35/cuda_engine.hpp"
#include "strix/cuda_core/dense_big.hpp"

namespace q = strix::qwen35;
static int g_fail = 0;
#define CHECK(c, ...)                         \
    do {                                      \
        if (!(c)) {                           \
            std::printf("FAIL: " __VA_ARGS__); \
            std::printf("\n");                \
            ++g_fail;                         \
        }                                     \
    } while (0)

static q::Config small_cfg() {
    q::Config c;
    c.layers = 8;
    c.hidden = 512;
    c.intermediate = 1024;
    c.vocab = 2048;
    c.heads = 6;
    c.kv_heads = 2;
    c.head_dim = 256;
    c.rotary_dim = 64;
    c.gdn_key_heads = 2;
    c.gdn_value_heads = 6;
    c.mtp_layers = 1;
    return c;
}

struct Rec {
    std::vector<float> logits, hidden;  // [pos][V], [pos][H]
};

static void run_serial(q::CudaEngine& E, const std::vector<std::uint32_t>& tok, Rec& r) {
    const auto& c = E.config();
    E.reset();
    r.logits.assign(tok.size() * c.vocab, 0);
    r.hidden.assign(tok.size() * c.hidden, 0);
    for (std::size_t i = 0; i < tok.size(); ++i) {
        const auto o = E.forward_rows(&tok[i], 1, q::LogitRows::all);
        E.read_logits(o, 0, &r.logits[i * c.vocab]);
        E.read_hidden(o.hidden_post, 1, &r.hidden[i * c.hidden]);
        E.commit(1);
    }
}

static void run_chain(q::CudaEngine& E, const std::vector<std::uint32_t>& tok, Rec& r, std::mt19937& rng, bool mark) {
    const auto& c = E.config();
    E.reset();
    r.logits.assign(tok.size() * c.vocab, 0);
    r.hidden.assign(tok.size() * c.hidden, 0);
    std::size_t pos = 0;
    int zero_k = 0, partial = 0, full = 0;
    std::vector<float> lg(c.vocab), hd(32 * c.hidden);
    while (pos < tok.size()) {
        const std::uint32_t R = std::min<std::size_t>(1 + rng() % 32, tok.size() - pos);
        const auto o = E.forward_rows(&tok[pos], R, q::LogitRows::all);
        E.read_hidden(o.hidden_post, R, hd.data());
        for (std::uint32_t i = 0; i < R; ++i) {
            E.read_logits(o, i, lg.data());
            std::copy(lg.begin(), lg.end(), r.logits.begin() + (pos + i) * c.vocab);
            std::copy(hd.begin() + i * c.hidden, hd.begin() + (i + 1) * c.hidden, r.hidden.begin() + (pos + i) * c.hidden);
        }
        std::uint32_t k = static_cast<std::uint32_t>(rng() % (R + 1));
        if (rng() % 4 == 0) k = R;
        if (k == 0 && rng() % 3 != 0) k = 1;
        zero_k += k == 0;
        partial += k > 0 && k < R;
        full += k == R;
        E.commit(k);
        pos += k;
        // the next window re-feeds the rejected rows: record only rows that will not be overwritten later is unnecessary,
        // later windows overwrite the same positions with identical bits when the engine is exact.
    }
    if (mark) std::printf("  chain: %d full, %d partial, %d zero commits\n", full, partial, zero_k);
}

static bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

int main() {
    // variants: 0 bf16, 1 q8_0, 2 nvfp4 onfly (all 2-D incl. lm_head, ssm beta/alpha, multi launches), 3 nvfp4 onfly lm_head bf16
    for (int var = 0; var < 4; ++var) {
        const auto pack = var == 1 ? q::PackFormat::q8_0 : q::PackFormat::bf16;
        std::printf("variant %d: %s%s\n", var, var == 0 ? "bf16" : var == 1 ? "q8_0" : "nvfp4 onfly", var == 3 ? " lm_head bf16" : "");
        q::EngineOptions opt;
        opt.max_positions = 512;
        opt.quiet = true;
        q::Config cfg = small_cfg();
        if (var >= 2) {
            opt.turbo_weights = q::TurboWeights::onfly;
            opt.turbo_lm_head = var == 2;
            cfg.gdn_value_heads = 16;  // ssm beta/alpha out = 16: NVFP4-eligible, exercises the 4-job multi launch
        }
        auto E = q::CudaEngine::synthetic(cfg, pack, opt, 7);
        const std::uint32_t invalid_id = cfg.vocab;
        bool rejected = false;
        try { (void)E->forward_rows(&invalid_id, 1); } catch (const std::invalid_argument&) { rejected = true; }
        CHECK(rejected, "engine rejects token outside vocabulary");
        if (var >= 2) CHECK(E->nvfp4_tensors() > 40, "nvfp4 tensors converted");
        std::mt19937 rng(99);
        std::vector<std::uint32_t> tok(160);
        for (auto& t : tok) t = rng() % 2048;
        Rec ser, ch, chg;
        run_serial(*E, tok, ser);
        bool finite = true;
        for (float f : ser.logits) finite = finite && std::isfinite(f);
        CHECK(finite, "serial logits finite");
        run_chain(*E, tok, ch, rng, true);
        CHECK(same(ser.logits, ch.logits), "chain logits differ from serial");
        CHECK(same(ser.hidden, ch.hidden), "chain hidden differs from serial");
        E->set_graphs(false);
        run_chain(*E, tok, chg, rng, false);
        CHECK(same(ser.logits, chg.logits), "graph-off chain logits differ from serial");
        E->set_graphs(true);
        {  // Every pack, including mixed eligible/ineligible BF16 and Q8 projections.
            opt.multi_gemv = false;
            opt.use_graphs = false;
            std::size_t free_before = 0, total_memory = 0;
            cudaMemGetInfo(&free_before, &total_memory);
            auto E2 = q::CudaEngine::synthetic(cfg, pack, opt, 7);
            Rec s2;
            run_serial(*E2, tok, s2);
            CHECK(same(ser.logits, s2.logits), "multi-launch off differs");
            CHECK(E2->launches_last_forward() == 0, "factory graphs-off option ignored");
            E2->configure_slots(2);
            E2.reset();
            std::size_t free_after = 0;
            cudaMemGetInfo(&free_after, &total_memory);
            CHECK(free_after + (1U << 20) >= free_before, "engine resources leaked on destruction");
        }
        // 4. multi-session slots: every row bitwise equal to the session run alone (legacy single-session serial decode as reference);
        //    sessions at different lengths, 1-row lanes mixed with 3-row windows, auto-commit and partial accept, rotating order.
        {
            const auto& c = E->config();
            const std::uint32_t NS = 4, N = 44;
            E->configure_slots(NS);
            const q::SlotRows invalid_group{0, 1};
            rejected = false;
            try { (void)E->forward_slots(&invalid_group, 1, &invalid_id); } catch (const std::invalid_argument&) { rejected = true; }
            CHECK(rejected, "slot rejects token outside vocabulary");
            std::vector<std::vector<std::uint32_t>> T(NS, std::vector<std::uint32_t>(N));
            std::vector<Rec> ref(NS);
            for (auto& v : T) for (auto& t : v) t = rng() % 2048;
            for (std::uint32_t i = 0; i < NS; ++i) run_serial(*E, T[i], ref[i]);
            const std::uint32_t pre[NS] = {0, 5, 13, 22};
            for (std::uint32_t i = 0; i < NS; ++i) {
                E->reset_slot(i);
                for (std::uint32_t done = 0; done < pre[i];) {
                    const std::uint32_t n = std::min<std::uint32_t>(7, pre[i] - done);
                    const q::SlotRows g{i, n};
                    E->forward_slots(&g, 1, &T[i][done], q::LogitRows::none);
                    done += n;
                }
                CHECK(E->slot_committed(i) == pre[i], "slot prefill length");
            }
            std::vector<float> lg(c.vocab), hd(c.hidden);
            bool all_same = true;
            std::uint32_t rows_checked = 0;
            for (std::uint32_t step = 0; step < 400; ++step) {
                std::vector<q::SlotRows> g;
                std::vector<std::uint32_t> tk;
                const bool windowed = step % 3 == 0;
                for (std::uint32_t k = 0; k < NS; ++k) {
                    const std::uint32_t sl = (k + step) % NS;
                    const std::uint32_t len = E->slot_committed(sl);
                    if (len >= N) continue;
                    if (step % 5 == 4 && k == 1) continue;  // a slot skipping steps
                    const std::uint32_t r = windowed && sl % 2 == 1 ? std::min<std::uint32_t>(3, N - len) : 1;
                    g.push_back({sl, r});
                    for (std::uint32_t j = 0; j < r; ++j) tk.push_back(T[sl][len + j]);
                }
                if (g.empty()) continue;
                const auto o = E->forward_slots(g.data(), static_cast<std::uint32_t>(g.size()), tk.data(), q::LogitRows::all, !windowed);
                std::uint32_t row = 0;
                std::vector<std::uint32_t> acc;
                for (std::size_t gi = 0; gi < g.size(); ++gi) {
                    const std::uint32_t len = E->slot_committed(g[gi].slot) - (windowed ? 0 : g[gi].rows);
                    for (std::uint32_t j = 0; j < g[gi].rows; ++j, ++row) {
                        E->read_logits(o, row, lg.data());
                        E->read_hidden(o.hidden_post + static_cast<std::size_t>(row) * c.hidden, 1, hd.data());
                        const std::size_t pos = len + j;
                        all_same = all_same && std::memcmp(lg.data(), &ref[g[gi].slot].logits[pos * c.vocab], c.vocab * 4) == 0 &&
                                   std::memcmp(hd.data(), &ref[g[gi].slot].hidden[pos * c.hidden], c.hidden * 4) == 0;
                        ++rows_checked;
                    }
                    acc.push_back(g[gi].rows == 1 ? 1 : (step / 3) % (g[gi].rows + 1));
                }
                if (windowed) E->commit_slots(acc.data());
            }
            CHECK(all_same, "slot batch rows differ from the single-session reference");
            CHECK(rows_checked > 120, "slot test rows checked (%u)", rows_checked);
            for (std::uint32_t i = 0; i < NS; ++i) CHECK(E->slot_committed(i) == N, "slot %u reached its length", i);
            // copy_slot: a copy continues bitwise like the original
            E->copy_slot(3, 1);
            CHECK(E->slot_committed(3) == N, "copy_slot length");
            const q::SlotRows continuation[] = {{1, 1}, {3, 1}};
            const std::uint32_t next[] = {T[1][0], T[1][0]};
            const auto continued = E->forward_slots(continuation, 2, next, q::LogitRows::all);
            std::vector<float> original(c.vocab), copied(c.vocab);
            E->read_logits(continued, 0, original.data());
            E->read_logits(continued, 1, copied.data());
            CHECK(same(original, copied), "copied slot continuation logits");
        }
        // large-M prefill: exact policy == windowed serial decode (logits of the last prompt token and of 8 continuation tokens,
        // which exercises the committed KV / GDN delta / conv state); flash policy is gated (finite, close, same top-1 mostly).
        for (int pol = 0; pol < 6; ++pol) {
            q::EngineOptions po = opt;
            po.multi_gemv = true;
            po.prefill_chunk = 128;  // three full chunks and an arbitrary three-row tail
            po.prefill_arith = pol < 4 ? q::PrefillArith::exact : pol == 4 ? q::PrefillArith::flash : q::PrefillArith::fast8;
            if (pol == 5 && !strix::cuda_core::dense_big_fast8_supported()) continue;
            using Arithmetic = strix::cuda_core::DenseArith;
            const Arithmetic policies[] = {Arithmetic::mma3, Arithmetic::mma2, Arithmetic::mma1, Arithmetic::fp32};
            po.dense_arithmetic = pol < 4 ? policies[pol] : opt.dense_arithmetic;
            const auto saved_default = strix::cuda_core::dense_arith();
            strix::cuda_core::set_dense_arith(po.dense_arithmetic == Arithmetic::mma3 ? Arithmetic::fp32 : Arithmetic::mma3);
            q::Config pcfg = cfg;
            pcfg.gdn_value_heads = 16;  // beta/alpha out = 16: serves the large-M path (out % 16 == 0)
            auto P = q::CudaEngine::synthetic(pcfg, pack, po, 7);
            CHECK(P->prefill_is_big() == (po.dense_arithmetic != Arithmetic::fp32), "variant %d: large-M admission follows engine arithmetic", var);
            const std::size_t n = 387, extra = 8;
            std::vector<std::uint32_t> t2(n);
            for (auto& id : t2) id = rng() % 2048;
            for (std::size_t i = 0; i < extra; ++i) t2.push_back(rng() % 2048);
            auto Ref = q::CudaEngine::synthetic(pcfg, pack, po, 7);
            Rec sr;
            run_serial(*Ref, t2, sr);
            P->reset();
            const auto o = P->prefill(t2.data(), static_cast<std::uint32_t>(n), q::LogitRows::last);
            const auto& c2 = P->config();
            std::vector<float> lp(c2.vocab);
            P->read_logits(o, 0, lp.data());
            std::vector<float> lref(sr.logits.begin() + (n - 1) * c2.vocab, sr.logits.begin() + n * c2.vocab);
            std::vector<float> cont(extra * c2.vocab), cref(sr.logits.begin() + n * c2.vocab, sr.logits.end());
            for (std::size_t i = 0; i < extra; ++i) {
                const auto oo = P->forward_rows(&t2[n + i], 1, q::LogitRows::all);
                P->read_logits(oo, 0, &cont[i * c2.vocab]);
                P->commit(1);
            }
            CHECK(P->committed() == n + extra, "committed length");
            strix::cuda_core::set_dense_arith(saved_default);
            if (pol < 4) {
                CHECK(same(lp, lref), "variant %d: prefill(exact) last-token logits differ from serial", var);
                CHECK(same(cont, cref), "variant %d: prefill(exact) continuation logits differ from serial", var);
            } else {
                double mx = 0, ref = 0;
                std::size_t top_ok = 0;
                bool fin = true;
                auto arg = [&](const std::vector<float>& v, std::size_t off) { return std::max_element(v.begin() + off, v.begin() + off + c2.vocab) - (v.begin() + off); };
                for (std::size_t i = 0; i < cont.size(); ++i) {
                    fin = fin && std::isfinite(cont[i]);
                    mx = std::max(mx, static_cast<double>(std::fabs(cont[i] - cref[i])));
                    ref = std::max(ref, static_cast<double>(std::fabs(cref[i])));
                }
                for (std::size_t i = 0; i < extra; ++i) top_ok += arg(cont, i * c2.vocab) == arg(cref, i * c2.vocab);
                std::printf("  %s prefill vs serial: max|dlogit| %.4g (max|logit| %.4g), top-1 agree %zu/%zu\n", pol == 4 ? "flash" : "fast8", mx, ref, top_ok, extra);
                // fast8 = e4m3 x e4m3 on nvfp4 weights: ~3.2 % uniform relative RMS per GEMM (dense_big_test: per-row/per-output spread 2.5-4.2 %, i.e.
                // precision, not a defect); on the flat tiny synthetic logits that reads as ~0.14 * max|logit|. The real gate is the
                // KL / top-1 kit at 8K (P-RESULT). Here: finite, within 0.25 * max|logit|, and most top-1 agree.
                const double tol = (pol == 5 && var >= 2) ? 0.25 : 0.1;
                CHECK(fin && mx < tol * ref + 1e-3, "variant %d: flash/fast8 prefill finite and close", var);
                if (pol == 5 && var >= 2) CHECK(top_ok * 4 >= extra * 3 - 1, "variant %d: fast8 top-1 mostly agrees", var);
            }
        }
        // MTP: one call with rows == the same rows one at a time
        if (E->has_mtp()) {
            const auto& c = E->config();
            E->reset();
            std::vector<std::uint32_t> t8(tok.begin(), tok.begin() + 9);
            const auto o = E->forward_rows(t8.data(), 8, q::LogitRows::none);
            std::vector<float> hid(8 * c.hidden);
            E->read_hidden(o.hidden_post, 8, hid.data());
            float* dh = nullptr;
            cudaMalloc(reinterpret_cast<void**>(&dh), hid.size() * 4);
            cudaMemcpy(dh, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice);
            std::vector<std::uint32_t> nxt(t8.begin() + 1, t8.begin() + 9);
            rejected = false;
            try { (void)E->mtp_forward(&invalid_id, dh, 1, 0); } catch (const std::invalid_argument&) { rejected = true; }
            CHECK(rejected, "MTP rejects token outside vocabulary");
            const auto m8 = E->mtp_forward(nxt.data(), dh, 8, 0);
            std::vector<float> l8(8 * c.vocab), l1(8 * c.vocab);
            for (int r = 0; r < 8; ++r) E->read_logits(q::RowsOut{m8.logits}, r, &l8[r * c.vocab]);
            for (int r = 0; r < 8; ++r) {
                const auto m1 = E->mtp_forward(&nxt[r], dh + r * c.hidden, 1, r);
                E->read_logits(q::RowsOut{m1.logits}, 0, &l1[r * c.vocab]);
            }
            CHECK(same(l8, l1), "mtp 8 rows != 8 single rows");
            cudaFree(dh);
        }
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
