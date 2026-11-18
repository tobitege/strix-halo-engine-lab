#include "../src/core/dense_linear.hpp"
// strix-cuda-qwen35-spec: exact block drafting (copy / native MTP chain / hybrid blocks, up to 32 verify rows) on the qwen35
// CUDA engine. Verifies output == serial decode token for token, reports accepted tokens per step (mean/p50/p90), tok/s
// average and best 1 s window, and the engine cost table verify_ms(R) / MTP ms / lm_head share.
// Home: specific (CLI binding of the qwen35 engine); drafting logic in src/spec (shared), glue in qwen35/cuda_spec.hpp.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/device/cuda_enumerator.hpp"
#include "strix/cuda_core/dense.hpp"
#include "strix/qwen35/cuda_engine.hpp"
#include "strix/qwen35/cuda_spec.hpp"

namespace q = strix::qwen35;
namespace sp = strix::spec;
using Clock = std::chrono::steady_clock;

namespace {
double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }

std::uint64_t mix(std::uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return x;
}

// Test chain: proposes the serial continuation with controlled accuracy (exercises long / partial accepts on the engine).
struct OracleChain final : sp::ChainDrafter {
    const std::vector<std::uint32_t>& ref;
    double acc;
    std::uint32_t stream_len = 1;
    OracleChain(const std::vector<std::uint32_t>& r, double a) : ref(r), acc(a) {}
    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
        out.clear();
        for (std::uint32_t j = 0; j < k; ++j) {
            const std::uint64_t pos = std::uint64_t(stream_len) + prefix.size() + j;
            const bool ok = double(mix(pos * 977 + 5) % 1000) / 1000.0 < acc;
            out.push_back(pos < ref.size() && ok ? ref[pos] : 3);
        }
    }
    void accept(std::span<const std::uint32_t> e) override { stream_len += std::uint32_t(e.size()); }
};

std::vector<std::uint32_t> parse_ids(const std::string& text) {
    std::vector<std::uint32_t> v;
    std::string t;
    for (const char c : text + ",") {
        if (c != ',' && c != ' ' && c != '\n' && c != '\r' && c != '\t') t += c;
        else if (!t.empty()) { v.push_back(strix::core::parse_u32(t)); t.clear(); }
    }
    return v;
}

double quantile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min<std::size_t>(v.size() - 1, std::size_t(p * double(v.size())))];
}
}  // namespace

int main(int argc, char** argv) {
    strix::device::bind_cuda_main(argc, argv);
    try {
        std::string model, synth, ids_arg, ids_file, sources = "copy,chain,hybrid", chain_hidden = "post", label = "run";
        std::uint32_t n_gen = 256, prompt_len = 64, layers_override = 0, max_pos = 0, max_block = 32, chain_k = 7;
        double oracle = -1.0;
        std::uint32_t trace_n = 0;
        bool cost_table = false, check = true, graphs = true, seed_costs = true, turbo_lm_head = true;
        q::TurboWeights turbo = q::TurboWeights::off;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                return argv[++i];
            };
            if (a == "--model") model = next();
            else if (a == "--synthetic-full") synth = next();
            else if (a == "--layers") layers_override = strix::core::parse_u32(next());
            else if (a == "--prompt-ids") ids_arg = next();
            else if (a == "--prompt-file") ids_file = next();
            else if (a == "--prompt-len") prompt_len = strix::core::parse_u32(next());
            else if (a == "--n-gen") n_gen = strix::core::parse_u32(next());
            else if (a == "--sources") sources = next();
            else if (a == "--max-block") max_block = strix::core::parse_u32(next());
            else if (a == "--chain-k") chain_k = strix::core::parse_u32(next());
            else if (a == "--oracle") oracle = std::stod(next());
            else if (a == "--cost-table") cost_table = true;
            else if (a == "--trace") trace_n = strix::core::parse_u32(next());
            else if (a == "--check") check = next() != "0";
            else if (a == "--graphs") graphs = next() != "0";
            else if (a == "--seed-costs") seed_costs = next() != "0";
            else if (a == "--chain-hidden") chain_hidden = next();
            else if (a == "--max-pos") max_pos = strix::core::parse_u32(next());
            else if (a == "--label") label = next();
            else if (a == "--turbo-weights") {
                const std::string v = next();
                if (v != "off" && v != "native" && v != "onfly" && v != "converted") throw std::runtime_error("--turbo-weights expects off|native|onfly|converted");
                turbo = v == "converted" ? q::TurboWeights::converted : v == "onfly" ? q::TurboWeights::onfly : v == "native" ? q::TurboWeights::native : q::TurboWeights::off;
            } else if (a == "--turbo-lm-head") {
                const std::string v = next();
                if (v != "nvfp4" && v != "bf16") throw std::runtime_error("--turbo-lm-head expects nvfp4|bf16");
                turbo_lm_head = v == "nvfp4";
            } else if (a == "--dense-arith") {
                const std::string v = next();
                namespace cc = strix::cuda_core;
                cc::set_dense_arith(v == "fp32" ? cc::DenseArith::fp32 : v == "mma2" ? cc::DenseArith::mma2
                                    : v == "mma1" ? cc::DenseArith::mma1 : cc::DenseArith::mma3);
            }
            else throw std::runtime_error("unknown argument " + a);
        }
        if (model.empty() == synth.empty())
            throw std::runtime_error("usage: (--model F.gguf | --synthetic-full bf16|q8_0 [--layers N]) [--prompt-ids a,b | --prompt-file F | --prompt-len N] "
                                     "[--n-gen N] [--sources copy,chain,hybrid|none] [--oracle ACC] [--cost-table] [--check 0|1] [--max-block 1..32]");
        std::vector<std::uint32_t> prompt;
        if (!ids_arg.empty()) prompt = parse_ids(ids_arg);
        if (!ids_file.empty()) {
            std::ifstream f(ids_file);
            std::stringstream ss;
            ss << f.rdbuf();
            prompt = parse_ids(ss.str());
        }
        q::EngineOptions opt;
        opt.use_graphs = graphs;
        opt.turbo_weights = turbo;
        opt.turbo_lm_head = turbo_lm_head;
        opt.max_positions = std::max<std::uint32_t>(max_pos, std::uint32_t(std::max<std::size_t>(prompt.size(), prompt_len) + n_gen + 128 +
                                                                          (cost_table ? 4096 : 0)));
        std::unique_ptr<q::CudaEngine> E;
        if (!model.empty()) E = q::CudaEngine::from_gguf(model, opt);
        else {
            q::Config cfg;
            if (layers_override) cfg.layers = layers_override;
            E = q::CudaEngine::synthetic(cfg, synth == "q8_0" ? q::PackFormat::q8_0 : q::PackFormat::bf16, opt, 0x5eedULL);
        }
        const auto& cfg = E->config();
        std::printf("identity: %s\n", E->identity().c_str());
        if (prompt.empty()) {
            std::uint64_t s = 99;
            for (std::uint32_t i = 0; i < prompt_len; ++i) prompt.push_back(std::uint32_t(mix(s + i) % cfg.vocab));
        }
        q::SpecEngine S(*E, chain_hidden == "pre");
        const auto hid = cfg.hidden;
        (void)hid;

        auto prefill = [&](std::uint32_t& anchor) {
            E->reset();
            S.reset();
            for (std::size_t p = 0; p < prompt.size(); p += 32) {
                const auto rows = std::uint32_t(std::min<std::size_t>(32, prompt.size() - p));
                const bool last = p + rows == prompt.size();
                S.prefill(prompt.data() + p, rows, last ? 0 : prompt[p + rows], last, &anchor);
            }
        };
        // Warm every row graph (commit 0 discards: no state change) so capture cost is not in any wall.
        auto warm = [&]() {
            std::vector<std::uint32_t> t(32, 1);
            for (std::uint32_t r = 1; r <= 32; ++r) {
                E->forward_rows(t.data(), r, q::LogitRows::all);
                E->commit(0);
            }
            E->sync();
        };

        if (cost_table) {
            std::uint32_t anchor = 0;
            prefill(anchor);
            warm();
            std::vector<std::uint32_t> t(32, anchor);
            std::printf("cost table (%s, ctx %zu, graphs %d)\n", label.c_str(), prompt.size(), graphs ? 1 : 0);
            std::printf("rows verify_ms(commit full) verify_ms(commit half) tok_per_s_if_all_accepted\n");
            for (const std::uint32_t r : {1U, 2U, 3U, 4U, 6U, 8U, 12U, 16U, 24U, 32U}) {
                double full = 0, half = 0;
                const int reps = 12;
                for (int rep = 0; rep < 2 * reps; ++rep) {
                    const bool h = rep & 1;
                    const double t0 = now_ms();
                    E->forward_rows(t.data(), r, q::LogitRows::all);
                    E->commit(h ? std::max(1U, r / 2) : r);
                    E->sync();
                    (h ? half : full) += now_ms() - t0;
                }
                full /= reps;
                half /= reps;
                std::printf("%u %.3f %.3f %.1f\n", r, full, half, 1000.0 * r / full);
            }
            double none = 0, last = 0;
            for (int rep = 0; rep < 24; ++rep) {
                const bool l = rep & 1;
                const double t0 = now_ms();
                E->forward_rows(t.data(), 1, l ? q::LogitRows::last : q::LogitRows::none);
                E->commit(0);
                E->sync();
                (l ? last : none) += now_ms() - t0;
            }
            std::printf("lm_head share at R=1: none %.3f ms, last %.3f ms -> lm_head %.3f ms (%.1f %%)\n", none / 12, last / 12,
                        (last - none) / 12, 100.0 * (last - none) / last);
            if (E->has_mtp()) {
                const auto o = E->forward_rows(t.data(), 1, q::LogitRows::last);
                E->commit(1);
                E->sync();
                std::vector<double> ms(33, 0.0);
                for (const std::uint32_t r : {1U, 2U, 4U, 8U, 16U, 32U}) {
                    double sum = 0;
                    for (int rep = 0; rep < 12; ++rep) {
                        std::vector<std::uint32_t> nx(32, anchor);
                        const double t0 = now_ms();
                        E->mtp_forward(nx.data(), o.hidden_post, r == 1 ? 1 : r, 0);  // hidden rows: row 0 reused (cost only)
                        E->sync();
                        sum += now_ms() - t0;
                    }
                    std::printf("mtp_forward rows %u: %.3f ms\n", r, sum / 12);
                }
            }
            std::printf("weights %.2f GB/pass, state %.0f MB/pass\n", double(E->weight_bytes()) / 1e9, double(E->state_bytes_per_pass()) / 1e6);
            return 0;
        }

        // ---- serial reference (greedy)
        std::uint32_t anchor = 0;
        std::vector<std::uint32_t> ref;
        double serial_ms = 0;
        if (check || oracle >= 0.0) {
            prefill(anchor);
            warm();
            ref.push_back(anchor);
            std::uint32_t cur = anchor;
            const double t0 = now_ms();
            while (ref.size() < n_gen) {
                const auto o = E->forward_rows(&cur, 1, q::LogitRows::last);
                E->commit(1);
                E->read_top(o, 1, &cur);
                ref.push_back(cur);
            }
            serial_ms = now_ms() - t0;
            std::printf("serial: %u tokens in %.1f ms = %.2f tok/s (%.3f ms/token)\n", n_gen, serial_ms, 1000.0 * (n_gen - 1) / serial_ms,
                        serial_ms / (n_gen - 1));
        }

        // ---- speculative run
        prefill(anchor);
        warm();
        sp::BlockConfig bc;
        bc.max_block = max_block;
        bc.chain_max_k = chain_k;
        bc.copy = sources.find("copy") != std::string::npos;
        bc.chain = sources.find("chain") != std::string::npos;
        bc.hybrid = sources.find("hybrid") != std::string::npos;
        sp::ChainDrafter* mtp_chain = S.has_chain() ? &S.make_chain() : nullptr;
        OracleChain oracle_chain(ref, oracle);
        sp::ChainDrafter* chain = nullptr;
        if (oracle >= 0.0) chain = &oracle_chain;
        else chain = mtp_chain;
        if (sources == "none") { bc.copy = bc.chain = bc.hybrid = false; }
        sp::BlockDrafter drafter(bc, chain);
        if (seed_costs) {  // engine's measured verify wall per window shape (commit full), 3 reps each
            std::vector<std::uint32_t> t(32, anchor);
            for (const std::uint32_t r : {1U, 2U, 4U, 8U, 16U, 32U}) {
                double sum = 0;
                for (int rep = 0; rep < 3; ++rep) {
                    const double t0 = now_ms();
                    E->forward_rows(t.data(), r, q::LogitRows::all);
                    E->commit(0);
                    E->sync();
                    sum += now_ms() - t0;
                }
                drafter.seed_verify_ms(r, sum / 3);
            }
        }
        drafter.commit(prompt);
        std::vector<std::uint32_t> out{anchor};
        drafter.commit(std::span<const std::uint32_t>(out.data(), 1));
        sp::BlockSpeculator spec(drafter, S);
        spec.start(anchor);
        std::vector<double> tick_ms{0.0};  // wall at each step end (relative), tokens cumulative in cum
        std::vector<std::uint32_t> cum{1};
        std::vector<double> per_step;      // emitted per step
        std::uint64_t by_source[3] = {0, 0, 0};
        const double t0 = now_ms();
        while (out.size() < n_gen) {
            sp::StepParams p;
            p.remaining = n_gen - std::uint32_t(out.size());
            p.max_rows = 32;
            p.context = E->committed();
            const auto r = spec.step(out, p);
            if (r.emitted == 0) break;
            if (per_step.size() < trace_n) {
                const auto& A = drafter.arbiter().last_arms();
                const auto& Sc = drafter.arbiter().last_arm_scores();
                std::printf("T%zu src%u rows%u prop%u acc%u em%u ver%.2f dr%.2f | ser %.4f cpy k%u e%.2f ms%.2f %.4f | mdl k%u e%.2f ms%.2f %.4f probe%d | cc1 %.2f cc2 %.2f\n",
                            per_step.size(), unsigned(r.source), r.rows, r.proposed, r.accepted, r.emitted, r.verify_ms, r.draft_ms,
                            Sc[0], A[1].k, A[1].expected_tokens, A[1].expected_ms, Sc[1], A[2].k, A[2].expected_tokens, A[2].expected_ms,
                            Sc[2], int(A[2].probe), drafter.chain_cost_ms(1), drafter.chain_cost_ms(2));
            }
            ++by_source[std::size_t(r.source)];
            per_step.push_back(double(r.emitted));
            tick_ms.push_back(now_ms() - t0);
            cum.push_back(std::uint32_t(out.size()));
        }
        const double wall = now_ms() - t0;
        std::size_t bad = 0;
        if (!ref.empty())
            for (std::size_t i = 0; i < out.size() && i < ref.size(); ++i)
                if (out[i] != ref[i]) { ++bad; if (bad == 1) std::fprintf(stderr, "MISMATCH at token %zu: spec %u serial %u\n", i, out[i], ref[i]); }
        // best 1 s window over step ends
        double best = 0;
        for (std::size_t i = 0; i < tick_ms.size(); ++i) {
            std::size_t j = i;
            while (j + 1 < tick_ms.size() && tick_ms[j] - tick_ms[i] < 1000.0) ++j;
            const double dt = tick_ms[j] - tick_ms[i];
            if (dt >= 900.0) best = std::max(best, 1000.0 * double(cum[j] - cum[i]) / dt);
        }
        const double mean_e = per_step.empty() ? 0.0 : double(out.size() - 1) / double(per_step.size());
        const auto& st = drafter.stats();
        std::printf("spec[%s]: %zu tokens, %zu steps (serial %llu copy %llu chain %llu), E mean %.2f p50 %.0f p90 %.0f, avg %.2f tok/s (%.3f ms/step), "
                    "best-1s %.2f tok/s, hybrid blocks %llu (ext %llu/%llu), mtp calls %llu rows %llu\n",
                    label.c_str(), out.size(), per_step.size(), (unsigned long long)by_source[0], (unsigned long long)by_source[1],
                    (unsigned long long)by_source[2], mean_e, quantile(per_step, 0.5), quantile(per_step, 0.9),
                    1000.0 * double(out.size() - 1) / wall, wall / double(std::max<std::size_t>(per_step.size(), 1)), best,
                    (unsigned long long)drafter.hybrid_blocks, (unsigned long long)drafter.hybrid_ext_accepted,
                    (unsigned long long)drafter.hybrid_ext_proposed, (unsigned long long)S.chain_calls(), (unsigned long long)S.chain_rows());
        std::printf("stats %s\n", st.json().c_str());
        if (!ref.empty()) {
            std::printf("EXACT: %s (%zu tokens compared, %zu mismatches)%s\n", bad == 0 ? "PASS" : "FAIL", std::min(out.size(), ref.size()), bad,
                        serial_ms > 0 ? "" : "");
            if (serial_ms > 0) std::printf("speedup vs serial: %.2fx\n", serial_ms / double(n_gen - 1) / (wall / double(out.size() - 1)));
        }
        return bad == 0 ? 0 : 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-cuda-qwen35-spec: %s\n", e.what());
        return 1;
    }
}
