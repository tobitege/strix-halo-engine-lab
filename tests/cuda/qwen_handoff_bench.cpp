// CPU-only hand-off throughput bench: the CUDA service's Sender (src/cuda/models/qwen/qwen_handoff.cpp) streams a synthetic state at
// position P to a node's import routes (e.g. `strix-rtx-state-import-tests --serve-file PORT FILE`), exactly as the service
// does: begin, every row component in blocks of `block` rows (enqueued as they would finish), the fixed blob, commit.
//
//   qwen_handoff_bench <import url> <position> [codec identity|bp-v1] [connections] [chunk MiB] [block rows] [repeat]
#include "qwen_handoff.hpp"
#include "qwen_http.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace qh = strix::qwen_handoff;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: qwen_handoff_bench url position [codec] [connections] [chunk_mib] [block] [repeat]\n"); return 1; }
    const std::string url = argv[1];
    const std::uint64_t position = std::strtoull(argv[2], nullptr, 10);
    qh::Options o;
    o.codec = argc > 3 ? argv[3] : "identity";
    o.connections = argc > 4 ? static_cast<unsigned>(std::atoi(argv[4])) : 1U;
    o.chunk_bytes = (argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 32ULL) << 20U;
    const std::uint64_t block = argc > 6 ? std::strtoull(argv[6], nullptr, 10) : 8192ULL;
    const int repeat = argc > 7 ? std::atoi(argv[7]) : 1;
    strix::qwen_http::startup();
    // Synthetic content: BF16-like words (random low byte, narrow high byte) so the codec ratio is not trivial.
    std::mt19937_64 rng(7);
    for (int rep = 0; rep < repeat; ++rep) {
        qh::Sender sender(url, "", 600, o);
        const auto& l = sender.layout();
        std::vector<std::vector<std::uint8_t>> rows(l.rows.size());
        for (std::size_t c = 0; c < l.rows.size(); ++c) {
            const auto n = position > l.rows[c].shift ? position - l.rows[c].shift : 0;
            rows[c].resize(n * l.rows[c].bytes_per_row);
            for (std::size_t i = 0; i < rows[c].size(); ++i) rows[c][i] = static_cast<std::uint8_t>((i & 1) ? 0x3c + (rng() & 3) : rng());
        }
        std::vector<std::uint8_t> fixed(l.fixed_bytes);
        for (auto& b : fixed) b = static_cast<std::uint8_t>(rng());
        std::vector<std::uint32_t> tokens(position, 7U);
        const auto t0 = Clock::now();
        sender.begin(tokens, position, 1, 2, 3);
        std::uint64_t bytes = 0;
        for (std::uint64_t b0 = 0; b0 < position; b0 += block) {
            for (std::size_t c = 0; c < l.rows.size(); ++c) {
                const auto n = rows[c].size() / l.rows[c].bytes_per_row;
                if (b0 >= n) continue;
                const auto cnt = std::min<std::uint64_t>(block, n - b0);
                qh::Piece p;
                p.component = l.rows[c].id;
                p.first = b0;
                p.count = cnt;
                p.data = rows[c].data() + b0 * l.rows[c].bytes_per_row;
                p.bytes = cnt * l.rows[c].bytes_per_row;
                sender.enqueue(p);
                bytes += p.bytes;
            }
        }
        for (const auto& f : l.fixed) {
            qh::Piece p;
            p.fixed = true;
            p.component = f.id;
            p.first = 0;
            p.data = fixed.data() + f.offset;
            p.bytes = f.bytes;
            sender.enqueue(p);
            bytes += p.bytes;
        }
        const auto reply = sender.commit();
        const double s = std::chrono::duration<double>(Clock::now() - t0).count();
        const auto& t = sender.timing();
        std::printf("{\"position\":%llu,\"codec\":\"%s\",\"connections\":%u,\"chunk_mib\":%llu,\"bytes\":%llu,\"wire_bytes\":%llu,"
                    "\"seconds\":%.3f,\"MBps\":%.1f,\"wire_MBps\":%.1f,\"send_s\":%.3f,\"encode_s\":%.3f,\"chunks\":%llu,\"commit\":%s}\n",
                    static_cast<unsigned long long>(position), t.encoding.c_str(), o.connections,
                    static_cast<unsigned long long>(o.chunk_bytes >> 20U), static_cast<unsigned long long>(bytes),
                    static_cast<unsigned long long>(t.wire_bytes), s, bytes / s / 1e6, t.wire_bytes / s / 1e6, t.send_s, t.encode_s,
                    static_cast<unsigned long long>(t.chunks), reply.size() < 400 ? reply.c_str() : "\"(long)\"");
        std::fflush(stdout);
    }
    return 0;
}
