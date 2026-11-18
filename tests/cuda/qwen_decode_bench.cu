// Decode timing of the CUDA Qwen engine (one model load; no logits leave the device).
//   qwen_decode_bench <tables dir> <rope table> <max positions> <tokens file> <prompt N> <generate G>
// Prefills rows [0, N-1), then reports: serial greedy steps (T = 1), verify windows of T = 2..STRIX_QWEN_TMAX (default 8, up to 16) rows over the serial
// tokens (ms per window), and MTP speculative decode (depth 7, min-p STRIX_QWEN_MIN_P or 0.7) when a sidecar is set.
#include "qwen_decode_test_plan.hpp"
#include "strix/cuda/qwen_prefill.hpp"
#include "strix/env_flags.hpp"
#include "strix/residency/residency_cli.hpp"

#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace qp = strix::cuda::qwen_prefill;
using Clock = std::chrono::steady_clock;

static std::vector<std::uint32_t> tokens_of(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "missing %s\n", path.c_str()); std::exit(2); }
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (auto& ch : s) if (ch == ',' || ch == '[' || ch == ']') ch = ' ';
    std::istringstream in(s);
    std::vector<std::uint32_t> t;
    std::uint32_t v;
    while (in >> v) t.push_back(v);
    return t;
}

static int bench_main(int argc, char** argv) {
    if (argc < 7) { std::fprintf(stderr, "usage: qwen_decode_bench tables rope max_positions tokens N G [--greedy-only|--prefill-only]\n"); return 1; }
    // the scope options and the residency flags share the tail after the six positionals; scope options are taken out
    // here, everything else goes to the residency consumer (which refuses anything it does not know)
    bool prefill_only = false, greedy_only = false;
    std::vector<std::string> bench_args(argv, argv + argc);
    for (auto it = bench_args.begin() + 7; it != bench_args.end();) {
        if (*it == "--prefill-only") { prefill_only = true; it = bench_args.erase(it); }
        else if (*it == "--greedy-only") { greedy_only = true; it = bench_args.erase(it); }
        else ++it;
    }
    if (prefill_only && greedy_only) { std::fprintf(stderr, "--prefill-only and --greedy-only are exclusive\n"); return 1; }
    qp::Config c;
    c.model_gguf = std::getenv("STRIX_QWEN_GGUF") ? std::getenv("STRIX_QWEN_GGUF") : "Qwen3.8-Flash-Next-UD-Q4_K_XL-merged.gguf";
    if (const char* s = std::getenv("STRIX_QWEN_MTP")) c.mtp_sidecar_gguf = s;
    if (const char* v = std::getenv("STRIX_QWEN_DRAFT_VOCAB")) c.draft_vocab = v;
    if (const char* v = strix::envcfg::text("STRIX_QWEN_EXPERT_TRACE")) c.expert_trace = v;  // EXTR01 router trace
    c.tables_dir = argv[1];
    c.rope_table = argv[2];
    c.max_positions = static_cast<std::uint32_t>(std::atoi(argv[3]));
    c.release_by_layer = false;
    c.decode = true;
    // tokens: one file or several joined by '+', all run on one model load; N = 0 takes each whole prompt
    std::vector<std::string> files;
    for (std::string list = argv[4]; !list.empty();) {
        const auto cut = list.find('+');
        files.push_back(list.substr(0, cut));
        list = cut == std::string::npos ? std::string() : list.substr(cut + 1);
    }
    const auto n_arg = static_cast<std::uint64_t>(std::atoll(argv[5]));
    const auto g = static_cast<std::uint32_t>(std::atoi(argv[6]));
    // optional flags after the six positionals: --expert-residency auto|full|tiered, --vram-budget-gib N, --print-effective-config
    // (the same consumer and environment fold as the server: strix/residency/residency_cli.hpp)
    strix::residency::CliOptions residency_cli;
    try {
        const std::vector<std::string>& args = bench_args;
        const auto stop = strix::residency::consume_cli_tail(args, 7, residency_cli);
        if (stop != args.size()) { std::fprintf(stderr, "unknown option %s\n", args[stop].c_str()); return 1; }
        c.residency = strix::residency::finish_cli(residency_cli);
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
    if (residency_cli.print_effective_config) {
        const auto plan = qp::plan_residency(c);
        std::printf("{\"residency\":%s}\n", plan.json.c_str());
        return plan.decision.ok ? 0 : 3;
    }
    qp::Prefill p(c);
    std::printf("loaded %s\n", p.describe().c_str());
    std::printf("residency %s\n", p.residency_json().c_str());
    const qp::Sink drop = [](const qp::Piece&) {};
    // STRIX_QWEN_PROFILE_T="8,16": the profiler range covers only those windows' loops (one range each; nsys
    // --capture-range=cudaProfilerApi --capture-range-end=repeat); unset: everything after the prefill
    std::vector<std::uint32_t> prof_t;
    if (const char* v = std::getenv("STRIX_QWEN_PROFILE_T")) {
        std::stringstream ss(v);
        for (std::string x; std::getline(ss, x, ',');) if (!x.empty()) prof_t.push_back(static_cast<std::uint32_t>(std::atoi(x.c_str())));
    }
    const auto tmax = strix::cuda::decode_test::max_width();
    for (const auto& file : files) {
    const auto prompt = tokens_of(file);
    const std::uint64_t n = n_arg == 0 ? prompt.size() : n_arg;
    if (n < 2 || n > prompt.size() || !strix::cuda::decode_test::benchmark_rows_sufficient(g, tmax)) {
        std::fprintf(stderr, "bad N/G: require valid N and G >= TMAX (%u)\n", tmax); return 1;
    }
    std::printf("prompt %s n=%llu g=%u\n", file.c_str(), static_cast<unsigned long long>(n), g);
    std::vector<std::uint32_t> seq(prompt.begin(), prompt.begin() + static_cast<std::ptrdiff_t>(n));
    if (prefill_only) cudaProfilerStart();
    p.run(seq, n - 1, drop);
    if (prefill_only) {
        cudaProfilerStop();
        std::printf("profile scope: cold prefill only, %llu input tokens, no emitted tokens\n", static_cast<unsigned long long>(n - 1));
        return 0;
    }
    if (prof_t.empty()) cudaProfilerStart();  // nsys --capture-range=cudaProfilerApi: decode only, not the upload
    std::vector<double> ms;
    for (std::uint32_t i = 0; i < g; ++i) {
        const auto t0 = Clock::now();
        const auto r = p.step(&seq[n - 1 + i], 1, n - 1 + i);
        ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        seq.push_back(r.argmax[0]);
    }
    const auto med = [](const std::vector<double>& v) { const std::size_t k = v.size(); return k % 2 ? v[k / 2] : 0.5 * (v[k / 2 - 1] + v[k / 2]); };
    {   // identity receipt: FNV-1a of the greedy tokens of the serial decode (compare routes / residency modes)
        std::uint64_t h = 1469598103934665603ULL;
        for (std::size_t i = n; i < seq.size(); ++i) { h ^= seq[i]; h *= 1099511628211ULL; }
        std::printf("tokens fnv %016llx (%zu)\n", static_cast<unsigned long long>(h), seq.size() - n);
    }
    std::sort(ms.begin(), ms.end());
    std::printf("serial T=1: median %.2f ms (%.1f tok/s), min %.2f\n", med(ms), 1000.0 / med(ms), ms[0]);
    if (greedy_only) {
        cudaProfilerStop();
        std::printf("profile scope: greedy C1 only, %u emitted tokens\n", g);
        return 0;
    }
    // STRIX_QWEN_TMAX (default 8, up to 16): the widest window; 9..16 = one session's copy window (prefill-shaped families)
    for (std::uint32_t t = 2; t <= tmax; ++t) {
        std::vector<std::uint32_t> pre(seq.begin(), seq.begin() + static_cast<std::ptrdiff_t>(n));
        p.run(pre, n - 1, drop);
        std::vector<double> w;
        const bool prof = std::find(prof_t.begin(), prof_t.end(), t) != prof_t.end();
        if (prof) { cudaDeviceSynchronize(); cudaProfilerStart(); }
        for (std::uint32_t i = 0; i + t <= g; i += t) {
            const auto t0 = Clock::now();
            p.step(&seq[n - 1 + i], t, n - 1 + i);
            w.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        }
        if (prof) { cudaDeviceSynchronize(); cudaProfilerStop(); }
        std::sort(w.begin(), w.end());
        std::printf("window T=%u: median %.2f ms\n", t, med(w));
    }
    if (!c.mtp_sidecar_gguf.empty()) {
        const float min_p = std::getenv("STRIX_QWEN_MIN_P") ? static_cast<float>(std::atof(std::getenv("STRIX_QWEN_MIN_P"))) : 0.7F;
        std::vector<std::uint32_t> pre(seq.begin(), seq.begin() + static_cast<std::ptrdiff_t>(n));
        p.run(pre, n - 1, drop);
        std::uint32_t pending = seq[n - 1], produced = 0, windows = 0;
        std::uint64_t pos = n - 1;
        double draft_ms = 0, verify_ms = 0;
        const auto ts = Clock::now();
        while (produced < g) {
            const auto t0 = Clock::now();
            const auto d = p.draft(pending, 7, min_p);
            const auto t1 = Clock::now();
            std::vector<std::uint32_t> rows{pending};
            rows.insert(rows.end(), d.tokens.begin(), d.tokens.end());
            const auto r = p.verify(rows.data(), static_cast<std::uint32_t>(rows.size()), pos);
            std::uint32_t keep = 1;
            while (keep < rows.size() && rows[keep] == r.argmax[keep - 1]) ++keep;
            p.commit(keep);
            draft_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            verify_ms += std::chrono::duration<double, std::milli>(Clock::now() - t1).count();
            pos += keep;
            produced += keep;
            pending = r.argmax[keep - 1];
            ++windows;
        }
        const double sec = std::chrono::duration<double>(Clock::now() - ts).count();
        std::printf("spec depth 7: %.1f tok/s, %.2f tokens/window, draft %.2f ms/window, verify+commit %.2f ms/window\n",
                    produced / sec, static_cast<double>(produced) / windows, draft_ms / windows, verify_ms / windows);
    }
    std::printf("receipt %s\n", p.residency_receipt_json().c_str());  // tiered: hit rate, promotions, host time; full: nothing new
    }
    cudaProfilerStop();
    return 0;
}

int main(int argc, char** argv) {
    try {
        return bench_main(argc, argv);
    } catch (const std::exception& e) {  // an engine refusal (device gate, residency decision, file) with its reason
        std::fprintf(stderr, "qwen_decode_bench: %s\n", e.what());
        return 2;
    }
}
