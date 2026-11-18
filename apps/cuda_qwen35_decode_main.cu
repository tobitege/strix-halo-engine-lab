#include "../src/core/dense_linear.hpp"
// strix-cuda-qwen35-decode: qwen35 (bf16 / q8_0) on one CUDA device through strix::qwen35::CudaEngine. Exact greedy decode,
// C1 timing, real GGUF or on-device synthetic full-size weights (--synthetic-full bf16|q8_0, no disk).
// Home: specific (CLI of the qwen35 family); engine in src/qwen35/cuda, arithmetic in src/cuda_core.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "strix/cuda_core/dense.hpp"
#include "strix/device/cuda_enumerator.hpp"
#include "strix/qwen35/cuda_engine.hpp"
#include "strix/qwen35/prefill_controls.hpp"

namespace q = strix::qwen35;

#define CK(x)                                                                                                  \
    do {                                                                                                       \
        const cudaError_t e__ = (x);                                                                           \
        if (e__ != cudaSuccess) {                                                                              \
            throw std::runtime_error(std::string(#x) + " failed: " + cudaGetErrorString(e__) + " at line " + \
                                     std::to_string(__LINE__));                                                \
        }                                                                                                      \
    } while (0)

namespace {

// Pure streaming-read ceiling of the device in the current thermal/power state (float4 loads, grid-stride, one partial sum per block).
__global__ void bw_read_kernel(const float4* p, std::size_t n, float* out) {
    float acc = 0.0f;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float4 v = __ldcs(p + i);
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 123456.0f) out[0] = acc;
}
double bw_probe_tbs() {
    const std::size_t bytes = 4ull << 30;
    float4* p = nullptr;
    float* o = nullptr;
    CK(cudaMalloc(reinterpret_cast<void**>(&p), bytes));
    CK(cudaMalloc(reinterpret_cast<void**>(&o), 4));
    CK(cudaMemset(p, 1, bytes));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    double best = 0;
    for (int i = 0; i < 6; ++i) {
        CK(cudaEventRecord(a));
        bw_read_kernel<<<188 * 8, 256>>>(p, bytes / 16, o);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, a, b));
        best = std::max(best, static_cast<double>(bytes) / (ms * 1e-3) / 1e12);
    }
    CK(cudaFree(p));
    CK(cudaFree(o));
    return best;
}

struct ClockSampler {
    std::atomic<bool> stop{false};
    std::vector<int> mhz;
    std::thread th;
    void start() {
        th = std::thread([this] {
            while (!stop.load()) {
                #ifdef _WIN32
                FILE* p = _popen("nvidia-smi --query-gpu=clocks.sm -i 0 --format=csv,noheader,nounits", "r");
#else
                FILE* p = popen("nvidia-smi --query-gpu=clocks.sm -i 0 --format=csv,noheader,nounits", "r");
#endif
                if (p) {
                    char b[64] = {};
                    if (std::fgets(b, sizeof b, p)) mhz.push_back(std::atoi(b));
#ifdef _WIN32
                    _pclose(p);
#else
                    pclose(p);
#endif
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }
    void end() {
        stop = true;
        if (th.joinable()) th.join();
    }
};

}  // namespace

// Multi-session decode bench: N slots at context `ctx`, 1 row per slot per step (greedy), steps timed on the host including the token
// read-back (what a server pays). Reports aggregate and per-request tok/s (mean and best 1 s window), and the SM clock.
struct SlotRun {
    double total_s = 0, mean_step_ms = 0, per_req_mean = 0, per_req_peak = 0, agg_mean = 0, agg_peak = 0;
    int clk_min = 0, clk_avg = 0;
    std::vector<std::uint32_t> first_tokens;  // slot 0's generated ids (exactness cross-check between arms)
};
static SlotRun slot_decode(q::CudaEngine& E, std::uint32_t n_slots, std::uint32_t steps, std::vector<std::uint32_t>& cur) {
    SlotRun r;
    std::vector<q::SlotRows> g(n_slots);
    for (std::uint32_t i = 0; i < n_slots; ++i) g[i] = {i, 1};
    std::vector<std::uint32_t> top(n_slots);
    ClockSampler clk;
    clk.start();
    using Clock = std::chrono::steady_clock;
    std::vector<double> t_end(steps);
    const auto t0 = Clock::now();
    for (std::uint32_t i = 0; i < steps; ++i) {
        const auto o = E.forward_slots(g.data(), n_slots, cur.data(), q::LogitRows::all, true);
        E.read_top(o, n_slots, top.data());
        for (std::uint32_t k = 0; k < n_slots; ++k) cur[k] = top[k];
        r.first_tokens.push_back(top[0]);
        t_end[i] = std::chrono::duration<double>(Clock::now() - t0).count();
    }
    clk.end();
    r.total_s = t_end[steps - 1];
    r.mean_step_ms = r.total_s * 1e3 / steps;
    r.per_req_mean = steps / r.total_s;
    r.agg_mean = r.per_req_mean * n_slots;
    // best window of >= 1 s (or the whole run when shorter): steps completed / elapsed between window boundaries
    double best = r.per_req_mean;
    std::uint32_t lo = 0;
    for (std::uint32_t hi = 0; hi < steps; ++hi) {
        while (lo + 1 < hi && t_end[hi] - t_end[lo + 1] >= 1.0) ++lo;
        if (t_end[hi] - t_end[lo] >= 1.0) best = std::max(best, (hi - lo) / (t_end[hi] - t_end[lo]));
    }
    r.per_req_peak = best;
    r.agg_peak = best * n_slots;
    if (!clk.mhz.empty()) {
        r.clk_min = *std::min_element(clk.mhz.begin(), clk.mhz.end());
        long sum = 0;
        for (int m : clk.mhz) sum += m;
        r.clk_avg = static_cast<int>(sum / static_cast<long>(clk.mhz.size()));
    }
    return r;
}

int main(int argc, char** argv) {
    strix::device::bind_cuda_main(argc, argv);
    try {
        std::string model, dump, ids_arg, synth, dump_mtp, force_file, save_pack;
        bool mtp_pre = false;
        std::uint32_t n_gen = 32, rows_cap = 16, max_pos = 4096, prompt_len = 0, ctx_fill = 0, layers_override = 0, decode_rows = 1, slots = 0, ab_pairs = 3;
        std::string arms_arg;
        bool slots_verify = false, graphs = true, multi = true, bwp = false, pdl = false, print_cfg = false;
        q::EngineOptions topt;  // TURBO levers (defaults = exact)
        std::string arith_name = "mma3";
        std::string flash_pipe, flash_tail, pf_attn8, dump_cont;
        std::string pf_arith, pf_dump;            // --prefill-arith exact|flash|fast|fast2: large-M prefill (absent: windows of --rows)
        std::uint32_t pf_chunk = 2048, pf_repeat = 1, pf_slots = 2, pf_wave_min = 0;
        std::uint32_t pf_inflight = q::EngineOptions{}.prefill_inflight, pf_ramp = 0;  // bounded submission / ramp
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                return argv[++i];
            };
            if (a == "--model") model = next();
            else if (a == "--synthetic-full") synth = next();
            else if (a == "--prompt-ids") ids_arg = next();
            else if (a == "--prompt-len") prompt_len = strix::core::parse_u32(next());
            else if (a == "--ctx-fill") ctx_fill = strix::core::parse_u32(next());
            else if (a == "--n-gen") n_gen = strix::core::parse_u32(next());
            else if (a == "--rows") rows_cap = strix::core::parse_u32(next());
            else if (a == "--decode-rows") decode_rows = strix::core::parse_u32(next());
            else if (a == "--max-pos") max_pos = strix::core::parse_u32(next());
            else if (a == "--prefill-arith") pf_arith = next();
            else if (a == "--flash-pipe") flash_pipe = next();         // pad2|swz3 (bitwise equal; scheduling only)
            else if (a == "--flash-tail") flash_tail = next();         // split|off (bitwise equal; scheduling only; default split)
            else if (a == "--prefill-attn8") pf_attn8 = next();        // on|off: e4m3 flash attention (FAST, identity -a8)
            else if (a == "--prefill-chunk") pf_chunk = strix::core::parse_u32(next());
            else if (a == "--prefill-wave-min") pf_wave_min = strix::core::parse_u32(next());  // 0 = off; wave-aligned steps
            else if (a == "--prefill-dump") pf_dump = next();  // last-token logits of the LAST prefill (float32 [vocab])
            else if (a == "--prefill-slots") pf_slots = strix::core::parse_u32(next());
            else if (a == "--prefill-inflight") pf_inflight = strix::core::parse_u32(next());  // chunks queued ahead of the GPU, 0 = unbounded
            else if (a == "--prefill-ramp") pf_ramp = strix::core::parse_u32(next());          // first chunk rows, doubling; 0 = off
            else if (a == "--prefill-repeat") pf_repeat = strix::core::parse_u32(next());
            else if (a == "--layers") layers_override = strix::core::parse_u32(next());
            else if (a == "--graphs") graphs = next() != "0";
            else if (a == "--slots") slots = strix::core::parse_u32(next());  // multi-session bench: N concurrent sessions
            else if (a == "--slots-verify") slots_verify = true;  // exactness: batched slots == each slot decoded alone (logit hashes)
            else if (a == "--arms") arms_arg = next();
            else if (a == "--ab-pairs") ab_pairs = strix::core::parse_u32(next());
            else if (a == "--bw-probe") bwp = true;
            else if (a == "--pdl") pdl = next() != "0";
            else if (a == "--dense-arith") {
                const std::string v = next();
                arith_name = v == "fp32" || v == "mma2" || v == "mma1" ? v : "mma3";
                topt.dense_arithmetic = v == "fp32" ? strix::cuda_core::DenseArith::fp32 : v == "mma2" ? strix::cuda_core::DenseArith::mma2
                                                  : v == "mma1" ? strix::cuda_core::DenseArith::mma1 : strix::cuda_core::DenseArith::mma3;
            }
            else if (a == "--multi" || a == "--turbo-multi-launch") multi = next() != "0";
            else if (a == "--turbo-weights") {
                const std::string v = next();
                if (v == "native") topt.turbo_weights = q::TurboWeights::native;
                else if (v == "onfly") topt.turbo_weights = q::TurboWeights::onfly;
                else if (v == "off") topt.turbo_weights = q::TurboWeights::off;
                else if (v == "converted") topt.turbo_weights = q::TurboWeights::converted;  // --model is a pack from --turbo-save-pack
                else throw std::runtime_error("--turbo-weights expects off|native|onfly|converted");
            } else if (a == "--turbo-bits") {
                topt.turbo_bits = strix::core::parse_u32(next());
                if (topt.turbo_bits != 4) throw std::runtime_error("--turbo-bits: only 4 is implemented");
            } else if (a == "--turbo-lm-head") {
                const std::string v = next();
                if (v != "nvfp4" && v != "bf16") throw std::runtime_error("--turbo-lm-head expects nvfp4|bf16");
                topt.turbo_lm_head = v == "nvfp4";
            } else if (a == "--turbo-save-pack") save_pack = next();  // write the onfly NVFP4 weights of --model (BF16 gguf) as a converted pack, then exit
            else if (a == "--print-effective-config") print_cfg = true;
            else if (a == "--dump-logits") dump = next();
            else if (a == "--dump-cont") dump_cont = next();  // with --prefill-arith: big prefill, then dump only its last row + decode rows
            else if (a == "--force-gen-file") force_file = next();  // teacher forcing: decode feeds these tokens (comma list), not its own argmax
            else if (a == "--dump-mtp") dump_mtp = next();
            else if (a == "--mtp-hidden") mtp_pre = next() == "pre";
            else throw std::runtime_error("unknown argument " + a);
        }
        if (pf_repeat == 0) throw std::invalid_argument("--prefill-repeat must be positive");
        if (pf_slots < 1 || pf_slots > 2) throw std::invalid_argument("--prefill-slots expects 1..2");
        if (!save_pack.empty()) {
            if (model.empty()) throw std::runtime_error("--turbo-save-pack needs --model <bf16.gguf>");
            const auto t0 = std::chrono::steady_clock::now();
            const std::uint64_t n = q::save_converted_pack(model, save_pack, topt);
            std::printf("pack: %s %llu bytes in %.1f s\n", save_pack.c_str(), static_cast<unsigned long long>(n),
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            return 0;
        }
        if (bwp) {
            std::printf("bw-probe: %.3f TB/s streaming read (4 GiB)\n", bw_probe_tbs());
            return 0;
        }
        if (!print_cfg && ((model.empty() == synth.empty()) || rows_cap == 0 || rows_cap > 32 || decode_rows == 0 || decode_rows > 32))
            throw std::runtime_error(
                "usage: (--model F.gguf | --synthetic-full bf16|q8_0 [--layers N]) (--prompt-ids 1,2,3 | --prompt-len N) [--ctx-fill N] "
                "[--dense-arith mma3|mma2|mma1|fp32] [--n-gen N] [--rows 1..32] [--decode-rows 1..32 (timing of an R-row window, committed fully)] [--max-pos N] "
                "[--graphs 0|1] [--dump-logits F]");
        std::vector<std::uint32_t> prompt;
        if (!ids_arg.empty()) {
            if (ids_arg[0] == '@') {  // --prompt-ids @FILE: comma-separated ids in a file (long prompts exceed the command line)
                std::ifstream pf(ids_arg.substr(1));
                if (!pf) throw std::runtime_error("cannot open " + ids_arg.substr(1));
                ids_arg.assign(std::istreambuf_iterator<char>(pf), std::istreambuf_iterator<char>());
            }
            std::stringstream ss(ids_arg);
            std::string t;
            while (std::getline(ss, t, ',')) prompt.push_back(strix::core::parse_u32(t));
        }
        q::EngineOptions opt = topt;
        if (!pf_arith.empty()) {
            if (pf_arith == "windows") opt.prefill_big = false;  // reference: prefill() runs forward_rows windows of 32 + commit
            else if (pf_arith == "exact") opt.prefill_arith = q::PrefillArith::exact;
            else if (pf_arith == "flash") opt.prefill_arith = q::PrefillArith::flash;
            else if (pf_arith == "fast") opt.prefill_arith = q::PrefillArith::fast;
            else if (pf_arith == "fast2") opt.prefill_arith = q::PrefillArith::fast2;
            else if (pf_arith == "fast8") opt.prefill_arith = q::PrefillArith::fast8;
            else throw std::runtime_error("--prefill-arith expects windows|exact|flash|fast|fast2|fast8");
        }
        opt.prefill_chunk = pf_chunk;
        opt.prefill_wave_min_tokens = pf_wave_min;
        opt.prefill_slots = pf_slots;
        opt.prefill_inflight = pf_inflight;
        opt.prefill_ramp = pf_ramp;
        q::resolve_flash_controls(opt, flash_pipe, flash_tail, pf_attn8);
        opt.use_graphs = graphs;
        opt.multi_gemv = multi;
        opt.pdl = pdl;
        if (print_cfg) {
            q::print_effective_config(opt, arith_name.c_str(), stdout);
            return 0;
        }
        std::unique_ptr<q::CudaEngine> E;
        const std::uint64_t total_ctx = ctx_fill + (slots ? static_cast<std::uint64_t>(n_gen) * (ab_pairs + 1) * 2 : 0) + prompt.size() + prompt_len + static_cast<std::uint64_t>(n_gen) * decode_rows + 64;
        opt.max_positions = static_cast<std::uint32_t>(std::max<std::uint64_t>(max_pos, total_ctx));
        if (!model.empty()) {
            E = q::CudaEngine::from_gguf(model, opt);
        } else {
            q::Config cfg;
            if (layers_override) cfg.layers = layers_override;
            E = q::CudaEngine::synthetic(cfg, synth == "q8_0" ? q::PackFormat::q8_0 : q::PackFormat::bf16, opt, 0x5eedULL);
        }
        const auto& cfg = E->config();
        q::print_effective_config(E->options(), arith_name.c_str(), stderr);
        std::fprintf(stderr, "  nvfp4 tensors: %u, weight bytes per pass %.2f GB\n", E->nvfp4_tensors(), static_cast<double>(E->weight_bytes()) / 1e9);
        std::fprintf(stderr, "qwen35: layers %u hidden %u inter %u vocab %u heads %u/%u hd %u gdn %u/%u x %u mtp %d\n", cfg.layers,
                     cfg.hidden, cfg.intermediate, cfg.vocab, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.gdn_key_heads,
                     cfg.gdn_value_heads, cfg.gdn_value_dim, E->has_mtp() ? 1 : 0);
        std::uint64_t lcg = 12345;
        auto rnd_tok = [&]() {
            lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
            return static_cast<std::uint32_t>((lcg >> 33) % cfg.vocab);
        };
        if (slots) {
            // prefill slot 0 with ctx_fill random tokens (rows_cap rows per forward), clone it to the other slots, then ABAB:
            // A = all slots batched, B = slot 0 alone (the sequential-serving rate; N sequential requests aggregate at B).
            E->configure_slots(slots + (slots_verify ? 1 : 0));
            std::vector<std::uint32_t> ctxt(ctx_fill);
            for (auto& t : ctxt) t = rnd_tok();
            const auto tp = std::chrono::steady_clock::now();
            for (std::uint32_t p = 0; p < ctx_fill; p += rows_cap) {
                const q::SlotRows g{0, std::min<std::uint32_t>(rows_cap, ctx_fill - p)};
                E->forward_slots(&g, 1, ctxt.data() + p, q::LogitRows::none, true);
            }
            E->sync();
            std::fprintf(stderr, "slot prefill %u tokens in %.2f s\n", ctx_fill, std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count());
            const std::uint32_t pos_after_prefill = E->slot_committed(0);
            if (slots_verify) {
                // slot `slots` is the pristine copy; every slot starts from it with its own first token
                E->copy_slot(slots, 0);
                auto fnv = [](const std::vector<float>& v) {
                    std::uint64_t h = 1469598103934665603ull;
                    const auto* b = reinterpret_cast<const unsigned char*>(v.data());
                    for (std::size_t i = 0; i < v.size() * 4; ++i) h = (h ^ b[i]) * 1099511628211ull;
                    return h;
                };
                std::vector<float> lg(cfg.vocab);
                const std::uint32_t steps = n_gen;
                std::vector<std::vector<std::uint64_t>> hb(slots), hs(slots);
                std::vector<std::uint32_t> cur(slots);
                for (std::uint32_t k = 0; k < slots; ++k) { E->copy_slot(k, slots); cur[k] = (k * 7919 + 13) % cfg.vocab; }
                std::vector<q::SlotRows> g(slots);
                for (std::uint32_t k = 0; k < slots; ++k) g[k] = {k, 1};
                std::vector<std::uint32_t> top(slots);
                for (std::uint32_t i = 0; i < steps; ++i) {
                    const auto o = E->forward_slots(g.data(), slots, cur.data(), q::LogitRows::all, true);
                    E->read_top(o, slots, top.data());
                    for (std::uint32_t k = 0; k < slots; ++k) { E->read_logits(o, k, lg.data()); hb[k].push_back(fnv(lg)); cur[k] = top[k]; }
                }
                std::uint32_t bad = 0;
                for (std::uint32_t k = 0; k < slots; ++k) {
                    E->copy_slot(k, slots);
                    std::uint32_t t = (k * 7919 + 13) % cfg.vocab;
                    const q::SlotRows one{k, 1};
                    for (std::uint32_t i = 0; i < steps; ++i) {
                        const auto o = E->forward_slots(&one, 1, &t, q::LogitRows::all, true);
                        E->read_top(o, 1, &t);
                        E->read_logits(o, 0, lg.data());
                        hs[k].push_back(fnv(lg));
                        if (hs[k].back() != hb[k][i]) ++bad;
                    }
                }
                std::printf("slots-verify: %u slots x %u steps, ctx %u: %u of %u logit rows differ batched vs solo (%s)\n", slots, steps, pos_after_prefill, bad,
                            slots * steps, bad ? "FAIL" : "BITWISE EQUAL");
                return bad ? 1 : 0;
            }
            for (std::uint32_t i = 1; i < slots; ++i) E->copy_slot(i, 0);
            auto run_arm = [&](std::uint32_t n_active, const char* label) {
                // every arm starts from the same cloned state: restore slot i from a spare is not available, so arms continue
                // each slot's own history (positions advance); contexts differ by <= n_gen per arm and are reported.
                std::vector<std::uint32_t> cur(n_active);
                for (std::uint32_t k = 0; k < n_active; ++k) cur[k] = (k * 7919 + 13) % cfg.vocab;
                (void)slot_decode(*E, n_active, 8, cur);  // warm (graph capture)
                const SlotRun r = slot_decode(*E, n_active, n_gen, cur);
                std::printf("%s slots %u ctx0 %u ctx_end %u: step %.3f ms | per-request mean %.2f peak(1s) %.2f tok/s | aggregate mean %.1f peak %.1f tok/s | sm clock avg %d min %d MHz\n",
                            label, n_active, pos_after_prefill, E->slot_committed(0), r.mean_step_ms, r.per_req_mean, r.per_req_peak, r.agg_mean,
                            r.agg_peak, r.clk_avg, r.clk_min);
                std::fflush(stdout);
            };
            std::vector<std::uint32_t> arms;  // --arms 8,1,4,1 : per pair, in this order (default: slots,1)
            {
                std::stringstream as(arms_arg.empty() ? std::to_string(slots) + ",1" : arms_arg);
                std::string t;
                while (std::getline(as, t, ',')) arms.push_back(strix::core::parse_u32(t));
            }
            for (std::uint32_t pair = 0; pair < ab_pairs; ++pair)
                for (const auto n : arms) run_arm(std::min(n, slots), n == 1 ? "solo   " : "batched");
            return 0;
        }
        if (prompt.empty()) {
            for (std::uint32_t i = 0; i < prompt_len + ctx_fill; ++i) prompt.push_back(rnd_tok());
            if (prompt.empty()) throw std::runtime_error("give --prompt-ids or --prompt-len");
        }

        std::FILE* df = dump.empty() ? nullptr : std::fopen(dump.c_str(), "wb");
        if (!dump.empty() && !df) throw std::runtime_error("cannot open " + dump);
        std::FILE* df_open = nullptr;
        if (!dump_cont.empty()) {  // continuation dump: row 0 = last prefill row, row j = decode step j (same layout as --dump-logits' tail)
            if (pf_arith.empty() || df) throw std::runtime_error("--dump-cont needs --prefill-arith and no --dump-logits");
            df_open = std::fopen(dump_cont.c_str(), "wb");
            if (!df_open) throw std::runtime_error("cannot open " + dump_cont);
        }
        std::vector<float> host_logits(static_cast<std::size_t>(cfg.vocab));
        std::vector<std::uint32_t> top1;
        std::vector<float> hid_all;
        std::uint32_t next_tok = 0;
        if (!pf_arith.empty() && !df) {  // large-M prefill: pf_repeat prefills per process, FIRST = cold, LAST = warm
            for (std::uint32_t rep = 0; rep < pf_repeat; ++rep) {
                E->reset();
                const auto t0 = std::chrono::steady_clock::now();
                const auto o = E->prefill(prompt.data(), static_cast<std::uint32_t>(prompt.size()), q::LogitRows::last);
                E->sync();
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                E->read_top(o, 1, &next_tok);
                if (df_open && rep + 1 == pf_repeat) {
                    E->read_logits(o, 0, host_logits.data());
                    std::fwrite(host_logits.data(), 4, cfg.vocab, df_open);
                }
                if (!pf_dump.empty() && rep + 1 == pf_repeat) {
                    std::vector<float> lg(cfg.vocab);
                    E->read_logits(o, 0, lg.data());
                    if (std::FILE* f = std::fopen(pf_dump.c_str(), "wb")) {
                        std::fwrite(lg.data(), 4, lg.size(), f);
                        std::fclose(f);
                    }
                }
                std::fprintf(stderr, "prefill[%u/%u] %s %zu tokens in %.3f s = %.1f tok/s (%s) top1 %u\n", rep + 1, pf_repeat, pf_arith.c_str(),
                             prompt.size(), sec, static_cast<double>(prompt.size()) / sec, rep == 0 ? "FIRST/cold" : rep + 1 == pf_repeat ? "LAST/warm" : "mid", next_tok);
            }
        }
        const auto tp0 = std::chrono::steady_clock::now();
        for (std::size_t p = (!pf_arith.empty() && !df) ? prompt.size() : 0; p < prompt.size(); p += rows_cap) {
            const std::uint32_t rows = static_cast<std::uint32_t>(std::min<std::size_t>(rows_cap, prompt.size() - p));
            const bool last = p + rows == prompt.size();
            const auto want = df ? q::LogitRows::all : (last ? q::LogitRows::last : q::LogitRows::none);
            const auto o = E->forward_rows(prompt.data() + p, rows, want);
            if (!dump_mtp.empty()) {
                hid_all.resize(hid_all.size() + static_cast<std::size_t>(rows) * cfg.hidden);
                E->read_hidden(mtp_pre ? o.hidden_pre : o.hidden_post, rows, hid_all.data() + hid_all.size() - static_cast<std::size_t>(rows) * cfg.hidden);
            }
            E->commit(rows);
            if (df) {
                std::uint32_t tp[32];
                E->read_top(o, rows, tp);
                for (std::uint32_t r = 0; r < rows; ++r) {
                    E->read_logits(o, r, host_logits.data());
                    std::fwrite(host_logits.data(), 4, cfg.vocab, df);
                    top1.push_back(tp[r]);
                }
                next_tok = tp[rows - 1];
            } else if (last) {
                E->read_top(o, 1, &next_tok);
            }
        }
        E->sync();
        const double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
        if (df) {
            df_open = df;
            std::printf("prompt top1:");
            for (auto t : top1) std::printf(" %u", t);
            std::printf("\n");
        }
        std::fprintf(stderr, "prefill %zu tokens in %.2f s (rows %u)\n", prompt.size(), prefill_s, rows_cap);

        // greedy decode; decode_rows > 1 times an R-row window (tokens = the greedy token repeated, all committed)
        std::vector<std::uint32_t> gen;
        std::vector<double> step_ms;
        ClockSampler clk;
        clk.start();
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        const std::uint32_t ctx0 = E->committed();
        std::vector<std::uint32_t> forced;
        if (!force_file.empty()) {
            std::ifstream ff(force_file);
            std::string all((std::istreambuf_iterator<char>(ff)), std::istreambuf_iterator<char>());
            std::stringstream fs(all);
            std::string t;
            while (std::getline(fs, t, ',')) if (!t.empty()) forced.push_back(strix::core::parse_u32(t));
            strix::core::validate_token_ids(forced.data(), forced.size(), E->config().vocab);
        }
        for (std::uint32_t i = 0; i < n_gen; ++i) {
            if (i < forced.size()) next_tok = forced[i];
            gen.push_back(next_tok);
            if (i + 1 == n_gen) break;
            std::uint32_t toks[32];
            for (std::uint32_t r = 0; r < decode_rows; ++r) toks[r] = next_tok;
            CK(cudaEventRecord(e0, E->stream()));
            const auto o = E->forward_rows(toks, decode_rows, decode_rows == 1 ? q::LogitRows::last : q::LogitRows::all);
            E->commit(decode_rows);
            CK(cudaEventRecord(e1, E->stream()));
            if (!dump_mtp.empty() && decode_rows == 1) {
                hid_all.resize(hid_all.size() + cfg.hidden);
                E->read_hidden(mtp_pre ? o.hidden_pre : o.hidden_post, 1, hid_all.data() + hid_all.size() - cfg.hidden);
            }
            std::uint32_t tp[32];
            E->read_top(o, decode_rows == 1 ? 1 : decode_rows, tp);
            if (df_open && decode_rows == 1) {  // continuation logits follow the prompt rows in the dump (position prompt+i)
                E->read_logits(o, 0, host_logits.data());
                std::fwrite(host_logits.data(), 4, cfg.vocab, df_open);
            }
            next_tok = tp[decode_rows == 1 ? 0 : decode_rows - 1];
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            step_ms.push_back(ms);
        }
        clk.end();
        if (df_open) std::fclose(df_open);
        if (!dump_mtp.empty()) {
            // MTP row i: position i, token i+1 of (prompt + generated), trunk hidden i (post-final-norm unless --mtp-hidden pre)
            std::vector<std::uint32_t> seq = prompt;
            seq.insert(seq.end(), gen.begin(), gen.end());
            const std::size_t npos = hid_all.size() / cfg.hidden;
            float* dh = nullptr;
            CK(cudaMalloc(reinterpret_cast<void**>(&dh), hid_all.size() * 4));
            CK(cudaMemcpy(dh, hid_all.data(), hid_all.size() * 4, cudaMemcpyHostToDevice));
            std::FILE* mf = std::fopen(dump_mtp.c_str(), "wb");
            if (!mf) throw std::runtime_error("cannot open " + dump_mtp);
            for (std::size_t p0 = 0; p0 < npos; p0 += 32) {
                const std::uint32_t n = static_cast<std::uint32_t>(std::min<std::size_t>(32, npos - p0));
                const auto mo = E->mtp_forward(seq.data() + p0 + 1, dh + p0 * cfg.hidden, n, static_cast<std::uint32_t>(p0));
                for (std::uint32_t r = 0; r < n; ++r) {
                    E->read_logits(q::RowsOut{mo.logits}, r, host_logits.data());
                    std::fwrite(host_logits.data(), 4, cfg.vocab, mf);
                }
            }
            std::fclose(mf);
            std::fprintf(stderr, "mtp logits for %zu positions written\n", npos);
        }
        std::printf("generated:");
        for (auto t : gen) std::printf(" %u", t);
        std::printf("\n");
        if (!step_ms.empty()) {
            std::vector<double> s2(step_ms.begin() + (step_ms.size() > 2 ? 2 : 0), step_ms.end());
            double sum = 0;
            for (double m : s2) sum += m;
            const double mean = sum / s2.size();
            std::sort(s2.begin(), s2.end());
            const double bytes = static_cast<double>(E->weight_bytes()) + static_cast<double>(E->state_bytes_per_pass());
            std::printf("decode: rows %u ctx %u graphs %d nodes %llu: %zu steps, mean %.3f ms (median %.3f min %.3f), %.2f tok/s, "
                        "%.1f GB/s achieved (%.3f GB/pass excl. KV) vs 1792 = %.1f%%\n",
                        decode_rows, ctx0, graphs ? 1 : 0, static_cast<unsigned long long>(E->launches_last_forward()), s2.size(), mean,
                        s2[s2.size() / 2], s2.front(), 1000.0 * decode_rows / mean, bytes / (mean * 1e-3) / 1e9, bytes / 1e9,
                        100.0 * bytes / (mean * 1e-3) / 1e9 / 1792.0);
        }
        if (!clk.mhz.empty()) {
            int lo = 1 << 30, hi = 0;
            double sm = 0;
            for (int m : clk.mhz) {
                lo = std::min(lo, m);
                hi = std::max(hi, m);
                sm += m;
            }
            std::printf("sm clock MHz: min %d avg %.0f max %d (%zu samples)\n", lo, sm / clk.mhz.size(), hi, clk.mhz.size());
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-cuda-qwen35-decode: %s\n", e.what());
        return 1;
    }
}
