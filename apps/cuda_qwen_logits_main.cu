// Diagnostic teacher forcing over the production CUDA engine; no sampling or draft tokens.
#include "strix/cuda/qwen_prefill.hpp"
#include "strix/device/cuda_enumerator.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace qp = strix::cuda::qwen_prefill;
static std::vector<std::uint32_t> ids(const char* path) {
    std::ifstream f(path);
    std::vector<std::uint32_t> result;
    std::uint64_t value;
    while (f >> value) {
        if (value >= 248320) throw std::invalid_argument("token outside vocabulary");
        result.push_back(static_cast<std::uint32_t>(value));
    }
    if (!f.eof() || result.size() < 2) throw std::invalid_argument("invalid token IDs file");
    return result;
}
template<class T> static void put(std::ofstream& f, T value) {
    f.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
int main(int argc, char** argv) {
    try {
        strix::device::bind_cuda_main(argc, argv);
        if (argc != 9 && argc != 10) throw std::invalid_argument("usage: strix-cuda-qwen-logits MODEL TABLES ROPE IDS POSITIONS CAP OUT BLOCK_ROWS [STEP_ROWS=1] (text uint IDs/positions; positions predict IDS[p+1])");
        const auto input = ids(argv[4]);
        std::ifstream pf(argv[5]);
        std::vector<std::uint64_t> positions;
        std::uint64_t pos;
        while (pf >> pos) {
            if (pos >= input.size() - 1 || (!positions.empty() && pos <= positions.back()))
                throw std::invalid_argument("positions must ascend and have a next-token label");
            positions.push_back(pos);
        }
        if (!pf.eof() || positions.empty()) throw std::invalid_argument("invalid positions");
        // Prefill::verify requires a nonempty prefix; the frozen TURBO selections begin at >=64.
        if (positions.front() == 0) throw std::invalid_argument("position zero has no prefilled state; select positions >=1");
        const auto cap = std::stoull(argv[6]), block = std::stoull(argv[8]);
        const auto step_rows = argc == 10 ? std::stoull(argv[9]) : 1ULL;
        if (cap < input.size() || cap > UINT32_MAX || !block || block > UINT32_MAX)
            throw std::invalid_argument("invalid capacity/block rows");
        if (!step_rows || step_rows > 8) throw std::invalid_argument("teacher STEP_ROWS must be 1..8");
        if (std::filesystem::exists(argv[7])) throw std::invalid_argument("output already exists");
        qp::Config config;
        config.model_gguf = argv[1]; config.tables_dir = argv[2]; config.rope_table = argv[3];
        config.max_positions = static_cast<std::uint32_t>(cap);
        config.block_schedule = {static_cast<std::uint32_t>(block)};
        config.decode = true; config.export_state = false; config.release_by_layer = false;
        qp::Prefill engine(config);
        std::cerr << "teacher engine: " << engine.describe() << '\n';
        const auto identity = engine.numerical_identity();
        std::cerr << "teacher route: prefill=" << identity.prefill << " decode=" << identity.decode
                  << " storage=" << identity.storage << " hash=" << identity.hash << '\n';
        std::cerr << "teacher batching: step_rows=" << step_rows << " prefix_rows=" << positions.front()
                  << " context_capacity=" << cap << " block_rows=" << block << '\n';
        std::ofstream output(argv[7], std::ios::binary);
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output.write("TMLLOG02", 8);
        put<std::uint64_t>(output, 248320); put<std::uint64_t>(output, positions.size());
        put<std::uint64_t>(output, input.size());
        output.write(reinterpret_cast<const char*>(input.data()), input.size() * sizeof(input[0]));
        output.write(reinterpret_cast<const char*>(positions.data()), positions.size() * sizeof(positions[0]));
        // Prefix rows before the first selection need no LM head. Every later input is still consumed.
        if (positions.front()) engine.run(input, positions.front(), [](const qp::Piece&) {});
        std::size_t selected = 0;
        for (std::uint64_t p = positions.front(); p <= positions.back();) {
            const auto count = static_cast<std::uint32_t>(std::min<std::uint64_t>(step_rows, positions.back() - p + 1));
            const bool save = positions[selected] < p + count;
            const auto row = engine.step(&input[p], count, p, save);
            if (!save) { p += count; continue; }
            if (row.logits.size() != std::uint64_t{248320} * count) throw std::runtime_error("unexpected logit extent");
            for (float v : row.logits) if (!std::isfinite(v)) throw std::runtime_error("nonfinite logits");
            while (selected < positions.size() && positions[selected] < p + count) {
                const auto position = positions[selected];
                put(output, input[position]); put(output, position);
                const auto* logits = row.logits.data() + (position - p) * 248320;
                output.write(reinterpret_cast<const char*>(logits), 248320 * sizeof(float));
                ++selected;
            }
            p += count;
        }
        output.write("TMLEOF01", 8); put<std::uint64_t>(output, positions.size());
        output.close();
        std::cout << "TEACHER LOGITS PASS rows=" << positions.size() << " conditioned=" << positions.back()+1
                  << " step_rows=" << step_rows << " scope=teacher-decode-full-vocabulary\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
