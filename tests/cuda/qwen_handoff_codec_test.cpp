// Producer-side test of the hand-off chunk codec (host only, no GPU): src/cuda/models/qwen/qwen_handoff.cpp + src/bp_codec.cpp.
//
//   qwen_handoff_codec_test                         loopback: every chunk body the sender builds, bp-v1 encoded, decodes to
//                                                   the plain body byte for byte (random, edge-case and BF16/FP32-like pieces)
//   qwen_handoff_codec_test --sample STATE.rtxs [T] the same on the pieces of a real state file (as the service enqueues
//                                                   them), plus ratio and packer MB/s (encode + SHA-256) on T threads
//   qwen_handoff_codec_test --send URL STATE.rtxs [--codec bp-v1|identity] [--connections N] [--threads T] [--chunk-mib M]
//                                                   streams the file through qwen_handoff::Sender to a node's import routes
//                                                   (a CPU test server: strix-rtx-state-import-tests --serve-file PORT FILE)
//                                                   and checks GET <origin>/take (the parked payload equals the file's)
// Build (x64 Native Tools): cl /O2 /std:c++20 /EHsc /utf-8 /Iinclude /Isrc\cuda tests\cuda\qwen_handoff_codec_test.cpp
//   src\cuda\models\qwen\qwen_handoff.cpp src\bp_codec.cpp src\serve\json.cpp src\model_content_hash.cpp ws2_32.lib
#include "qwen_handoff.hpp"
#include "qwen_http.hpp"
#include "strix/bp_codec.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/serve/json.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace qh = strix::qwen_handoff;
namespace bp = strix::bp_codec;
using Clock = std::chrono::steady_clock;

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

void check_chunk(const std::vector<qh::Piece>& pieces, const std::vector<std::uint32_t>& widths, bp::WorkerPool* pool, const std::string& what) {
    const auto plain = qh::build_chunk_body(pieces, widths, false, nullptr);
    const auto frame = qh::build_chunk_body(pieces, widths, true, pool);
    require(bp::decoded_size(frame) == plain.size(), what + ": decoded size");
    std::vector<std::uint8_t> out(plain.size());
    bp::decode(frame, out, pool);
    require(out == plain, what + ": decoded chunk differs from the plain body");
}

void loopback() {
    bp::WorkerPool pool(3);
    std::mt19937_64 rng(1);
    std::vector<std::vector<std::uint8_t>> keep;
    const auto bytes = [&](std::size_t n, int kind) {
        std::vector<std::uint8_t> v(n);
        std::normal_distribution<float> d(0.0F, 0.5F);
        for (std::size_t i = 0; i < n; ++i) {
            if (kind == 0) v[i] = static_cast<std::uint8_t>(rng());
            else if (kind == 1) v[i] = 0;
            else {
                const float f = d(rng);
                std::uint32_t u;
                std::memcpy(&u, &f, 4);
                v[i] = static_cast<std::uint8_t>(kind == 2 ? (i % 2 ? u >> 24 : u >> 16) : (u >> (8 * (i % 4))));
            }
        }
        keep.push_back(std::move(v));
        return keep.back().data();
    };
    // Single pieces of every kind and odd size, with every width.
    for (const std::size_t n : {1U, 2U, 3U, 7U, 255U, 4097U, 1000003U})
        for (int kind = 0; kind < 4; ++kind)
            for (const std::uint32_t w : {1U, 2U, 4U}) {
                const qh::Piece p{kind % 2 == 1, 3, 17, 1, bytes(n, kind), n};
                check_chunk({p}, {w}, &pool, "n" + std::to_string(n) + " kind" + std::to_string(kind) + " w" + std::to_string(w));
            }
    // A multi-segment chunk: BF16 rows of several components, a fixed FP32 range at a high offset, host words.
    std::vector<qh::Piece> pieces;
    std::vector<std::uint32_t> widths;
    for (std::uint32_t c = 0; c < 6; ++c) { pieces.push_back({false, c, 512, 64, bytes(64 * 2048, 2), 64 * 2048}); widths.push_back(2); }
    pieces.push_back({true, 0, (std::uint64_t{5} << 32) + 1024, 0, bytes(3 << 20, 3), 3 << 20});
    widths.push_back(4);
    pieces.push_back({true, 4, 0, 0, bytes(8, 0), 8});
    widths.push_back(4);
    check_chunk(pieces, widths, &pool, "multi-segment chunk");
    check_chunk(pieces, widths, nullptr, "multi-segment chunk without a pool");
    std::cout << "loopback codec chunks: exact\n";
}

struct State {
    std::string header_json;
    std::vector<std::uint8_t> payload;
    strix::serve::Json header;
};

State read_state(const std::string& path) {
    State s;
    std::ifstream in(path, std::ios::binary);
    require(static_cast<bool>(in), "cannot open " + path);
    char magic[8];
    std::uint64_t n = 0;
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(&n), 8);
    require(std::memcmp(magic, "STRXRTX1", 8) == 0, "not an rtx state file");
    s.header_json.resize(static_cast<std::size_t>(n));
    in.read(s.header_json.data(), static_cast<std::streamsize>(n));
    s.payload.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    s.header = strix::serve::parse_json(s.header_json);
    return s;
}

// The pieces of a state as the service enqueues them: per block of `block` rows, every row component's rows of
// that block; then each fixed component whole.
struct Plan {
    std::vector<qh::Piece> pieces;
    std::vector<std::string> names;
};
Plan pieces_of(const State& s, std::uint64_t block = 1024) {
    Plan plan;
    const auto position = static_cast<std::uint64_t>(s.header.find("position")->as_int());
    struct Row { std::string name; std::uint64_t bpr, rows, offset; };
    std::vector<Row> rows;
    std::uint64_t at = 0;
    for (const auto& r : s.header.find("rows")->as_array()) {
        const auto bpr = static_cast<std::uint64_t>(r.find("bytes_per_row")->as_int());
        const auto n = position - static_cast<std::uint64_t>(r.find("shift")->as_int());
        rows.push_back({r.find("id")->as_string(), bpr, n, at});
        at += bpr * n;
    }
    const std::uint64_t rows_end = at;
    for (std::uint64_t b = 0; b < position; b += block)
        for (std::uint32_t c = 0; c < rows.size(); ++c) {
            if (b >= rows[c].rows) continue;
            const auto n = std::min(block, rows[c].rows - b);
            plan.pieces.push_back({false, c, b, n, s.payload.data() + rows[c].offset + b * rows[c].bpr, n * rows[c].bpr});
            plan.names.push_back(rows[c].name);
        }
    std::uint32_t f = 0;
    for (const auto& x : s.header.find("fixed")->as_array()) {
        const auto off = static_cast<std::uint64_t>(x.find("offset")->as_int()), n = static_cast<std::uint64_t>(x.find("bytes")->as_int());
        plan.pieces.push_back({true, f++, 0, 0, s.payload.data() + rows_end + off, n});
        plan.names.push_back(x.find("id")->as_string());
    }
    return plan;
}

int sample(const std::string& path, unsigned threads) {
    const auto s = read_state(path);
    const auto plan = pieces_of(s);
    bp::WorkerPool pool(threads > 0 ? threads - 1 : 0);
    // Chunks of <= 32 MiB of whole pieces (big pieces split as Sender::enqueue does: fixed by 32 MiB ranges).
    std::vector<std::vector<qh::Piece>> chunks(1);
    std::vector<std::vector<std::uint32_t>> widths(1);
    std::uint64_t in_chunk = 0;
    const std::uint64_t limit = std::uint64_t{32} << 20;
    for (std::size_t i = 0; i < plan.pieces.size(); ++i) {
        const auto& piece = plan.pieces[i];
        for (std::uint64_t at = 0; at < piece.bytes; at += limit) {
            qh::Piece p = piece;
            if (piece.bytes > limit) { p.first = piece.first + at; p.data = piece.data + at; p.bytes = std::min(limit, piece.bytes - at); }
            if (in_chunk && in_chunk + p.bytes > limit) { chunks.emplace_back(); widths.emplace_back(); in_chunk = 0; }
            chunks.back().push_back(p);
            widths.back().push_back(qh::element_width(plan.names[i]));
            in_chunk += p.bytes;
            if (piece.bytes <= limit) break;
        }
    }
    std::uint64_t plain = 0, wire = 0;
    double pack_s = 0, decode_s = 0;
    for (std::size_t c = 0; c < chunks.size(); ++c) {
        const auto reference = qh::build_chunk_body(chunks[c], widths[c], false, nullptr);
        const auto t0 = Clock::now();
        const auto frame = qh::build_chunk_body(chunks[c], widths[c], true, &pool);
        (void)strix::sha256_bytes_fast(std::as_bytes(std::span<const std::uint8_t>(frame)));
        const auto t1 = Clock::now();
        std::vector<std::uint8_t> out(static_cast<std::size_t>(bp::decoded_size(frame)));
        const auto t2 = Clock::now();
        bp::decode(frame, out, &pool);
        const auto t3 = Clock::now();
        require(out == reference, "chunk " + std::to_string(c) + " of the sample does not decode to its plain body");
        pack_s += secs(t0, t1);
        decode_s += secs(t2, t3);
        plain += reference.size();
        wire += frame.size();
    }
    std::printf("{\"sample\":\"%s\",\"chunks\":%zu,\"plain_bytes\":%llu,\"wire_bytes\":%llu,\"ratio\":%.4f,\"threads\":%u,"
                "\"pack_encode_sha_MBps\":%.0f,\"decode_MBps\":%.0f,\"exact\":true}\n",
        path.c_str(), chunks.size(), static_cast<unsigned long long>(plain), static_cast<unsigned long long>(wire),
        static_cast<double>(wire) / static_cast<double>(plain), threads, static_cast<double>(plain) / pack_s / 1e6,
        static_cast<double>(plain) / decode_s / 1e6);
    return 0;
}

int send(const std::string& url, const std::string& path, const qh::Options& options) {
    const auto s = read_state(path);
    const auto plan = pieces_of(s);
    std::vector<std::uint32_t> tokens;
    for (const auto& t : s.header.find("tokens")->as_array()) tokens.push_back(static_cast<std::uint32_t>(t.as_int()));
    const auto t0 = Clock::now();
    qh::Sender sender(url, "", 600, options);
    std::vector<std::uint32_t> ids;
    for (std::size_t i = 0; i < plan.pieces.size(); ++i)
        ids.push_back(plan.pieces[i].fixed ? sender.fixed_id(plan.names[i]) : sender.row_id(plan.names[i]));
    sender.begin(tokens, tokens.size(), 1, 2, 3);
    const auto t1 = Clock::now();
    for (std::size_t i = 0; i < plan.pieces.size(); ++i) {
        auto p = plan.pieces[i];
        p.component = ids[i];
        sender.enqueue(p);
    }
    const auto reply = sender.commit();
    const auto t2 = Clock::now();
    const auto u = strix::qwen_http::parse_url(url);
    strix::qwen_http::Client client("http://" + u.host + ":" + u.port, 60);
    const auto [status, take] = client.get("/take");
    const auto& t = sender.timing();
    std::printf("{\"url\":\"%s\",\"encoding\":\"%s\",\"connections\":%u,\"encode_threads\":%u,\"chunks\":%llu,\"bytes\":%llu,\"wire_bytes\":%llu,"
                "\"ratio\":%.4f,\"setup_s\":%.3f,\"stream_s\":%.3f,\"send_s\":%.3f,\"encode_s\":%.3f,\"commit_s\":%.3f,\"payload_MBps\":%.1f,"
                "\"wire_MBps\":%.1f,\"take\":%s,\"commit\":%s}\n",
        url.c_str(), t.encoding.c_str(), options.connections, options.encode_threads, static_cast<unsigned long long>(t.chunks),
        static_cast<unsigned long long>(t.bytes), static_cast<unsigned long long>(t.wire_bytes),
        static_cast<double>(t.wire_bytes) / static_cast<double>(t.bytes), secs(t0, t1), secs(t1, t2), t.send_s, t.encode_s, t.commit_s,
        static_cast<double>(t.bytes) / secs(t1, t2) / 1e6, static_cast<double>(t.wire_bytes) / secs(t1, t2) / 1e6, take.c_str(), reply.c_str());
    require(status == 200, "the node's parked payload differs from the file: " + take);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        strix::qwen_http::startup();
        if (argc >= 3 && std::string(argv[1]) == "--sample") return sample(argv[2], argc >= 4 ? static_cast<unsigned>(std::stoul(argv[3])) : 4U);
        if (argc >= 4 && std::string(argv[1]) == "--send") {
            qh::Options o;
            o.codec = "bp-v1";
            o.require_codec = true;
            for (int i = 4; i + 1 < argc; i += 2) {
                const std::string a = argv[i], v = argv[i + 1];
                if (a == "--codec") { o.codec = v; o.require_codec = v != "identity"; }
                else if (a == "--connections") o.connections = static_cast<unsigned>(std::stoul(v));
                else if (a == "--threads") o.encode_threads = static_cast<unsigned>(std::stoul(v));
                else if (a == "--chunk-mib") o.chunk_bytes = std::stoull(v) << 20;
                else throw std::invalid_argument("unknown flag " + a);
            }
            return send(argv[2], argv[3], o);
        }
        loopback();
        std::cout << "qwen hand-off codec tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
