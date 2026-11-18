// Teacher-forced logit dump of the CUDA Qwen engine, the input of tools/strix-gate/fast_logit_gate.py.
//
//   qwen_tf_logits --model GGUF --tables DIR --rope-table FILE [--mtp-sidecar GGUF] [--profile exact|fast] [lever options]
//                  --prompt NAME=tokens.u32 [--prompt ...] --out DIR [--rows 64] [--prefix q] [--runs 1]
//
// Engine and profile options go through the SAME parser and resolver as strix-cuda-qwen-serve (qwen_options.hpp, serve
// role): what is gated here is the configuration the server runs. Per prompt of N tokens: prefill run() over tokens
// [0, N - R), then step() windows of <= 8 rows over tokens [N - R, N) teacher-forced (each window is committed), and
//   DIR/PREFIX-NAME-tf-r0.f32 = [R][vocab] little-endian FP32, row j = logits after consuming token N - R + j.
// --runs 2 repeats prefill + windows in one process: the first run is cold (module load, allocation), the second warm;
// the prefill ms of every run is printed and the dumps must match bit for bit or the arm fails.
#include "strix/cuda/qwen_options.hpp"
#include "strix/cuda/qwen_prefill.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qp = strix::cuda::qwen_prefill;
namespace qo = strix::cuda::qwen_options;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {
std::vector<std::uint32_t> read_tokens(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot read " + path);
    const auto bytes = static_cast<std::size_t>(f.tellg());
    if (bytes % 4U || bytes == 0) throw std::runtime_error(path + ": not a little-endian uint32 token file");
    std::vector<std::uint32_t> out(bytes / 4U);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    return out;
}
double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
}  // namespace

int main(int argc, char** argv) try {
    qo::Request request;
    std::vector<std::pair<std::string, std::string>> prompts;
    std::string out_dir, prefix = "q";
    std::uint32_t rows = 64, runs = 1;
    for (int i = 1; i < argc; ++i) {
        if (qo::take(argc, argv, i, request)) continue;
        const std::string a = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return argv[++i];
        };
        if (a == "--prompt") {
            const auto v = next();
            const auto eq = v.find('=');
            if (eq == std::string::npos || eq == 0) throw std::invalid_argument("--prompt NAME=tokens.u32");
            prompts.emplace_back(v.substr(0, eq), v.substr(eq + 1));
        } else if (a == "--rows") rows = static_cast<std::uint32_t>(qo::number(a, next()));
        else if (a == "--runs") runs = static_cast<std::uint32_t>(qo::number(a, next()));
        else if (a == "--out") out_dir = next();
        else if (a == "--prefix") prefix = next();
        else throw std::invalid_argument("unknown option " + a);
    }
    // Serve role: decode on, no state export: the configuration the server runs.
    const auto resolved = qo::resolve(request, qo::Entry::serve);
    if (request.print_effective_config) { std::printf("%s\n", qo::effective_config_json(resolved).c_str()); return 0; }
    if (prompts.empty() || out_dir.empty() || rows == 0 || runs == 0) throw std::invalid_argument("--prompt NAME=FILE and --out DIR are required; --rows, --runs >= 1");
    for (const auto& n : resolved.resolution.notices()) std::fprintf(stderr, "%s\n", n.c_str());
    std::printf("arithmetic %s\n", resolved.contract.c_str());
    fs::create_directories(out_dir);
    std::vector<std::vector<std::uint32_t>> tokens;
    for (const auto& [name, path] : prompts) {
        tokens.push_back(read_tokens(path));
        const auto n = tokens.back().size();
        if (n <= rows + 1U || n > resolved.config.max_positions)
            throw std::invalid_argument(name + ": needs rows + 2 <= tokens <= --max-positions (" + std::to_string(n) + " tokens)");
    }
    std::printf("loading...\n");
    std::fflush(stdout);
    qp::Prefill engine(resolved.config);
    // A receipt is published only after every prompt and repetition succeeds.
    using J = strix::serve::Json;
    auto executions = J::array();
    std::printf("loaded %s\n", engine.describe().c_str());
    const qp::Sink drop = [](const qp::Piece&) {};
    for (std::size_t p = 0; p < prompts.size(); ++p) {
        const auto& name = prompts[p].first;
        const auto& seq = tokens[p];
        const std::uint64_t n = seq.size(), head = n - rows;
        std::vector<float> first;
        for (std::uint32_t run = 0; run < runs; ++run) {
            std::vector<float> dump;
            const auto t0 = Clock::now();
            const auto timing = engine.run(seq, head, drop);
            const auto t1 = Clock::now();
            for (std::uint64_t at = head; at < n;) {
                const auto window = static_cast<std::uint32_t>(std::min<std::uint64_t>(8, n - at));
                const auto r = engine.step(&seq[at], window, at, true);
                if (!r.ok || r.logits.size() % window) throw std::runtime_error(name + ": step failed " + r.error);
                dump.insert(dump.end(), r.logits.begin(), r.logits.end());
                at += window;
            }
            const auto t2 = Clock::now();
            std::printf("%s run %u (%s): prefill %.1f ms (gpu %.1f ms, %llu rows), teacher-forced %u rows %.1f ms\n", name.c_str(), run,
                        run == 0 ? "cold" : "warm", ms(t0, t1), timing.total_s * 1e3, static_cast<unsigned long long>(timing.rows), rows, ms(t1, t2));
            std::fflush(stdout);
            if (run == 0) first = std::move(dump);
            else {
                const auto common = std::min(first.size(), dump.size());
                if (dump.size() != first.size() ||
                    (common != 0 && std::memcmp(first.data(), dump.data(), common * sizeof(float)))) {
                    std::size_t at = 0;
                    while (at < common && !std::memcmp(&first[at], &dump[at], sizeof(float))) ++at;
                    throw std::runtime_error(name + " run " + std::to_string(run) +
                        " differs from run 0 at element " + std::to_string(at) +
                        " (sizes " + std::to_string(first.size()) + ", " + std::to_string(dump.size()) + ")");
                }
                std::printf("%s run %u dump identical to run 0\n", name.c_str(), run);
            }
            auto execution = J::object();
            execution.set("prompt", J(name));
            execution.set("run", J(static_cast<std::uint64_t>(run)));
            execution.set("rows", J(timing.rows));
            execution.set("forms", qo::forms_json(timing.executed_forms, timing.executed_moe_route_bf16));
            executions.push(std::move(execution));
        }
        const auto path = (fs::path(out_dir) / (prefix + "-" + name + "-tf-r0.f32")).string();
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(first.data()), static_cast<std::streamsize>(first.size() * sizeof(float)));
        if (!f) throw std::runtime_error("cannot write " + path);
        std::printf("%s: wrote %s (%u rows x %zu)\n", name.c_str(), path.c_str(), rows, first.size() / rows);
    }
    auto receipt = strix::serve::parse_json(qo::effective_config_json(resolved));
    receipt.set("executions", std::move(executions));
    std::ofstream config_file(fs::path(out_dir) / "effective-config.json");
    config_file << receipt.dump() << '\n';
    config_file.close();
    if (!config_file) throw std::runtime_error("cannot write effective configuration");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "qwen_tf_logits: %s\n", e.what());
    return 1;
}
