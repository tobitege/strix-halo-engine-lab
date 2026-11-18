// strix-hip-qwen35-decode: Qwen3.8-27B (qwen35 dense) decode on one HIP device over strix::qwen35::HipEngine.
// Prompt goes through windows of up to --rows tokens (forward_commit), then greedy decode (one row per step).
// Home: specific = app glue; engine in include/strix/qwen35/hip_engine.hpp, kernels shared.
#include <algorithm>
#include <fstream>
#include <iterator>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "strix/qwen35/hip_engine.hpp"
#include "../src/core/dense_linear.hpp"
#include "../src/core/dense_linear_iu4.hpp"

namespace {
std::string sysfs_clock() {
    for (int c = 0; c < 4; ++c) {
        std::ifstream f("/sys/class/drm/card" + std::to_string(c) + "/device/pp_dpm_sclk");
        std::string line, best;
        while (std::getline(f, line))
            if (line.find('*') != std::string::npos) best = line;
        if (!best.empty()) return best;
    }
    return "0: 0Mhz *";
}
int sysfs_busy() {
    for (int c = 0; c < 4; ++c) {
        std::ifstream f("/sys/class/drm/card" + std::to_string(c) + "/device/gpu_busy_percent");
        int v;
        if (f >> v) return v;
    }
    return -1;
}
std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> r;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ','))
        if (!t.empty()) {
            const auto id = strix::core::parse_u32(t);
            if (id > 0x7FFFFFFFU) throw std::invalid_argument("token ID exceeds signed index range");
            r.push_back(static_cast<int>(id));
        }
    return r;
}
}  // namespace

int main(int argc, char** argv) try {
    using namespace strix::qwen35;
    std::string model, dump, ids_arg = "1";
    unsigned wide = 0;
    unsigned gen = 16, rows = 1, maxpos = 0, repeat = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--prompt-ids") ids_arg = next();
        else if (a == "--gen") gen = strix::core::parse_u32(next());
        else if (a == "--rows") rows = strix::core::parse_u32(next());
        else if (a == "--wide") wide = strix::core::parse_u32(next());  // prompt windows of N rows, last-row logits only (N > 16 = FAST wide prefill)
        else if (a == "--dump-logits") dump = next();
        else if (a == "--iu4-reference") strix::core::dense_iu4::set_reference(true);  // exactness gate: dequantised-pack reference form
        else if (a == "--max-pos") maxpos = strix::core::parse_u32(next());
        else if (a == "--repeat") repeat = std::max(1U, strix::core::parse_u32(next()));  // prefills per process (reset between): first = cold, last = warm
        else throw std::invalid_argument("unknown flag " + a);
    }
    if (model.empty() || rows < 1U || rows > HipEngine::max_rows())
        throw std::invalid_argument("usage: --model GGUF [--prompt-ids 1,2,3] [--gen N] [--rows 1..16] [--dump-logits FILE] [--iu4-reference] [--max-pos N] [--repeat N]; --prompt-ids @FILE reads the list from FILE");
    if (!ids_arg.empty() && ids_arg[0] == '@') {  // long prompts exceed the per-argument limit of the OS
        std::ifstream f(ids_arg.substr(1));
        if (!f) throw std::invalid_argument("cannot read " + ids_arg.substr(1));
        ids_arg.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        while (!ids_arg.empty() && std::isspace(static_cast<unsigned char>(ids_arg.back()))) ids_arg.pop_back();
    }
    const auto prompt = parse_ids(ids_arg);
    if (prompt.empty()) throw std::invalid_argument("empty prompt");
    strix::core::require_capacity(prompt.size(), gen, 0xFFFFFFFFULL);
    const auto required = prompt.size() + gen;
    if (!maxpos) maxpos = static_cast<unsigned>(required);
    strix::core::require_capacity(0, required, maxpos);

    HipEngineOptions opt;
    opt.max_positions = maxpos;
    opt.prefill_rows = wide;
    const auto t_load0 = std::chrono::steady_clock::now();
    auto E = HipEngine::from_gguf(model, opt);
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();
    std::printf("load time %.1f s\n", load_s);
    const unsigned V = E->vocab();
    std::vector<float> dumped;
    std::uint32_t top[HipEngine::max_rows()];
    auto forward = [&](unsigned n, const int* ids) {
        std::uint32_t t[HipEngine::max_rows()];
        for (unsigned i = 0; i < n; ++i) t[i] = std::uint32_t(ids[i]);
        const auto o = E->forward_commit(t, n, dump.empty() ? LogitRows::last : LogitRows::all);
        E->read_top(o, dump.empty() ? 1U : n, top);
        if (!dump.empty()) {
            const auto at = dumped.size();
            dumped.resize(at + std::size_t(n) * V);
            E->read_logits(o, n, dumped.data() + at);
        }
    };
    unsigned last_n = 0;
    double pf_ms = 0.0;
    for (unsigned pass = 0; pass < repeat; ++pass) {
    if (pass) { E->reset(); dumped.clear(); }
    const auto t_pf0 = std::chrono::steady_clock::now();
    if (wide) {
        for (std::size_t at = 0; at < prompt.size();) {
            const unsigned n = unsigned(std::min<std::size_t>(wide, prompt.size() - at));
            std::vector<std::uint32_t> t(n);
            for (unsigned i = 0; i < n; ++i) t[i] = std::uint32_t(prompt[at + i]);
            const bool last = at + n == prompt.size();
            const auto o = E->forward_commit(t.data(), n, last ? LogitRows::last : LogitRows::none);
            if (last) {
                E->read_top(o, 1, top);
                if (!dump.empty()) { dumped.resize(V); E->read_logits(o, 1, dumped.data()); }
            }
            at += n;
        }
        last_n = 1;
    } else
    for (std::size_t at = 0; at < prompt.size();) {
        last_n = unsigned(std::min<std::size_t>(rows, prompt.size() - at));
        forward(last_n, prompt.data() + at);
        at += last_n;
    }
    pf_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pf0).count();
    if (repeat > 1U)
        std::printf("prefill %u/%u (%s): %zu tokens %.1f ms = %.1f tok/s\n", pass + 1U, repeat,
                    pass == 0U ? "cold" : pass + 1U == repeat ? "warm" : "mid", prompt.size(), pf_ms, double(prompt.size()) * 1000.0 / pf_ms);
    }
    std::printf("prompt %zu tokens: %.1f ms\n", prompt.size(), pf_ms);
    std::vector<int> generated{int(dump.empty() ? top[0] : top[last_n - 1U])};

    std::atomic<bool> run{true};
    std::atomic<long> clk_sum{0}, clk_n{0}, busy_sum{0};
    std::string last_clk;
    std::thread sampler([&] {
        while (run) {
            last_clk = sysfs_clock();
            clk_sum += std::atol(last_clk.c_str() + last_clk.find(' ') + 1);
            ++clk_n;
            busy_sum += std::max(0, sysfs_busy());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    const auto t0 = std::chrono::steady_clock::now();
    for (unsigned s = 1; s < gen; ++s) {
        const int id = generated.back();
        forward(1, &id);
        generated.push_back(int(top[0]));
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    run = false;
    sampler.join();
    std::printf("generated:");
    for (int t : generated) std::printf(" %d", t);
    std::printf("\n");
    if (gen > 1U) {
        const double per = ms / double(gen - 1U);
        const double bytes = double(E->weight_bytes()) + double(E->state_bytes_per_pass()) + double(E->kv_bytes_per_position()) * double(E->committed());
        std::printf("decode: %.2f ms/token, %.2f tok/s, %.1f GB/s (%.0f %% of 243), bytes/pass %.2f GB\n", per, 1000.0 / per,
                    bytes / per / 1e6, bytes / per / 1e6 / 243.0 * 100.0, bytes / 1e9);
        std::printf("gpu: sclk avg %ld MHz (last: %s), busy avg %ld %%\n", clk_n ? clk_sum / clk_n : 0L, last_clk.c_str(),
                    clk_n ? busy_sum / clk_n : 0L);
    }
    if (!dump.empty()) {
        std::ofstream f(dump, std::ios::binary);
        f.write(reinterpret_cast<const char*>(dumped.data()), std::streamsize(dumped.size() * 4U));
        std::printf("dumped %zu positions x %u to %s\n", dumped.size() / V, V, dump.c_str());
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
