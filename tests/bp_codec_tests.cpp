// CPU tests of the bp-v1 hand-off chunk codec (include/strix/bp_codec.hpp): exact round trips on random, structured
// and edge-case data, deterministic output with and without a worker pool, malformed frames rejected.
//   strix-bp-codec-tests                              unit tests
//   strix-bp-codec-tests --sample STATE.rtxs [T]      also: ratio and MB/s per component class of a real state file
//                                                     (1 thread per class; whole payload in 32 MiB chunks on T threads)
// Builds with CMake (CTest strix-bp-codec-tests) or alone: cl /O2 /std:c++20 /EHsc /Iinclude tests\bp_codec_tests.cpp
// src\bp_codec.cpp src\serve\json.cpp
#include "strix/bp_codec.hpp"
#include "strix/serve/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace bp = strix::bp_codec;

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

using Bytes = std::vector<std::uint8_t>;
using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

Bytes concat(const std::vector<bp::Region>& regions) {
    Bytes out;
    for (const auto& r : regions) out.insert(out.end(), r.data, r.data + r.bytes);
    return out;
}

// Encodes with and without the pool (identical frames), decodes both ways, compares. Returns the frame size.
std::size_t round_trip(const std::vector<bp::Region>& regions, bp::WorkerPool& pool, const std::string& what, bp::EncodeOptions options = {}) {
    const Bytes plain = concat(regions);
    const auto frame = bp::encode(regions, nullptr, options);
    const auto pooled = bp::encode(regions, &pool, options);
    require(frame == pooled, what + ": pooled encoding differs");
    require(bp::decoded_size(frame) == plain.size(), what + ": decoded size");
    Bytes a(plain.size() + 1, 0xEE), b(plain.size() + 1, 0x11);
    bp::decode(frame, std::span<std::uint8_t>(a.data(), plain.size()), nullptr);
    bp::decode(frame, std::span<std::uint8_t>(b.data(), plain.size()), &pool);
    require(std::equal(plain.begin(), plain.end(), a.begin()) && a.back() == 0xEE, what + ": decode differs");
    require(std::equal(plain.begin(), plain.end(), b.begin()) && b.back() == 0x11, what + ": pooled decode differs");
    return frame.size();
}
std::size_t round_trip(const Bytes& data, std::uint32_t width, bp::WorkerPool& pool, const std::string& what, bp::EncodeOptions options = {}) {
    return round_trip({{data.data(), data.size(), width}}, pool, what, options);
}

std::uint16_t bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<std::uint16_t>(u >> 16U);
}
Bytes bf16_normal(std::size_t n, std::uint32_t seed, float scale) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0F, scale);
    Bytes out(n * 2);
    for (std::size_t i = 0; i < n; ++i) { const auto v = bf16(d(rng)); std::memcpy(out.data() + 2 * i, &v, 2); }
    return out;
}
Bytes random_bytes(std::size_t n, std::uint32_t seed) {
    std::mt19937_64 rng(seed);
    Bytes out(n);
    for (auto& b : out) b = static_cast<std::uint8_t>(rng());
    return out;
}

void test_xxh64() {
    require(bp::xxh64(nullptr, 0) == 0xEF46DB3751D8E999ULL, "xxh64('')");
    const std::uint8_t abc[] = {'a', 'b', 'c'};
    require(bp::xxh64(abc, 3) == 0x44BC2CF5AD770999ULL, "xxh64('abc')");
    // 39 bytes: the 32-byte lanes, one 4-byte word and three tail bytes (vector of the python-xxhash README).
    const std::string text = "Nobody inspects the spammish repetition";
    require(bp::xxh64(reinterpret_cast<const std::uint8_t*>(text.data()), text.size()) == 0xFBCEA83C8A378BF1ULL, "xxh64(39 bytes)");
}

void test_edges(bp::WorkerPool& pool) {
    require(round_trip(std::vector<bp::Region>{}, pool, "no region") == 16, "empty frame is the header");
    require(round_trip(Bytes{}, 2, pool, "empty region") == 16, "empty region adds no block");
    for (const std::uint32_t w : {1U, 2U, 4U, 3U}) {
        for (const std::size_t n : {1U, 2U, 3U, 5U, 7U, 8U, 143U, 145U, 1023U, 4099U}) {
            const auto what = "w" + std::to_string(w) + " n" + std::to_string(n);
            (void)round_trip(random_bytes(n, static_cast<std::uint32_t>(n * 7 + w)), w, pool, what + " random");
            (void)round_trip(Bytes(n, 0), w, pool, what + " zero");
            Bytes two(n);
            for (std::size_t i = 0; i < n; ++i) two[i] = (i * 2654435761U >> 7U) & 1U ? 0x3F : 0xBF;
            (void)round_trip(two, w, pool, what + " two symbols");
        }
    }
    // All-zero and constant data shrink to the headers; incompressible data grows by the headers only.
    const std::size_t big = (std::size_t{3} << 20) + 3;
    require(round_trip(Bytes(big, 0), 2, pool, "zero 3 MiB") < 200, "all-zero data is a constant per plane");
    require(round_trip(Bytes(big, 0xAB), 4, pool, "constant 3 MiB") < 200, "constant data");
    const auto noise = random_bytes(big, 5);
    for (const std::uint32_t w : {1U, 2U, 4U}) {
        const auto n = round_trip(noise, w, pool, "noise w" + std::to_string(w));
        require(n <= big + 16 + 4 * (16 + 1 + w + 3), "incompressible data stays raw");
    }
}

void test_structured(bp::WorkerPool& pool) {
    // BF16 activations: the high byte plane is coded, the low byte plane is raw.
    const auto kv = bf16_normal((std::size_t{1} << 20) + 1, 11, 0.7F);
    const auto n = round_trip(kv, 2, pool, "bf16 normal");
    std::cout << "  bf16 normal(0.7) ratio " << static_cast<double>(n) / static_cast<double>(kv.size()) << "\n";
    require(static_cast<double>(n) < 0.80 * static_cast<double>(kv.size()), "bf16 data compresses");
    // FP32 values with a narrow exponent range.
    Bytes f32((std::size_t{1} << 20) * 4);
    std::mt19937 rng(3);
    std::normal_distribution<float> d(0.0F, 0.01F);
    for (std::size_t i = 0; i < f32.size() / 4; ++i) { const float v = d(rng); std::memcpy(f32.data() + 4 * i, &v, 4); }
    const auto m = round_trip(f32, 4, pool, "fp32 normal");
    require(static_cast<double>(m) < 0.95 * static_cast<double>(f32.size()), "fp32 data compresses");
    // Skewed counts (Fibonacci-like) force the length limit of 11.
    Bytes skew;
    std::uint64_t a = 1, b = 1;
    for (unsigned s = 0; s < 30; ++s) { skew.insert(skew.end(), static_cast<std::size_t>(a), static_cast<std::uint8_t>(s)); const auto c = a + b; a = b; b = c; }
    std::shuffle(skew.begin(), skew.end(), std::mt19937(9));
    (void)round_trip(skew, 1, pool, "fibonacci counts (length limit)");
    // 256 distinct symbols, small blocks (many blocks, odd sizes), several regions with mixed widths.
    const auto small = bf16_normal(70001, 4, 3.0F);
    bp::EncodeOptions tiny;
    tiny.block_bytes = 1001;
    (void)round_trip(small, 2, pool, "small blocks", tiny);
    const Bytes header = {0, 0, 0, 0, 3, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0};
    (void)round_trip({{header.data(), header.size(), 1}, {kv.data(), 8192, 2}, {header.data(), header.size(), 1}, {f32.data(), 40000, 4},
                         {kv.data(), 0, 2}, {kv.data() + 1, 777, 2}},
        pool, "mixed regions");
}

void test_malformed(bp::WorkerPool& pool) {
    const auto kv = bf16_normal(50000, 21, 1.0F);
    const Bytes header(24, 7);
    const std::vector<bp::Region> regions = {{header.data(), header.size(), 1}, {kv.data(), kv.size(), 2}};
    const auto frame = bp::encode(regions, &pool);
    const Bytes plain = concat(regions);
    Bytes out(plain.size());
    // Every truncation is rejected.
    for (std::size_t n = 0; n < frame.size(); n += (n < 64 ? 1 : 97)) {
        bool threw = false;
        try {
            const std::span<const std::uint8_t> cut(frame.data(), n);
            if (bp::decoded_size(cut) == out.size()) bp::decode(cut, out, &pool);
        } catch (const std::invalid_argument&) { threw = true; }
        require(threw, "truncated frame accepted at " + std::to_string(n));
    }
    // A flipped bit anywhere is rejected, or (padding bits) decodes to the very same bytes.
    std::mt19937 rng(77);
    unsigned rejected = 0;
    for (unsigned trial = 0; trial < 3000; ++trial) {
        Bytes bad = frame;
        const auto at = rng() % bad.size();
        bad[at] = static_cast<std::uint8_t>(bad[at] ^ (1U << (rng() % 8U)));
        try {
            const auto n = bp::decoded_size(bad);
            if (n != out.size()) { ++rejected; continue; }
            bp::decode(bad, out, trial % 2 ? &pool : nullptr);
            require(out == plain, "a corrupted frame decoded to different bytes without an error (byte " + std::to_string(at) + ")");
        } catch (const std::invalid_argument&) { ++rejected; }
    }
    require(rejected > 2900, "corruptions are rejected (" + std::to_string(rejected) + " of 3000)");
    bool threw = false;
    try { Bytes small(plain.size() - 1); bp::decode(frame, small, nullptr); } catch (const std::invalid_argument&) { threw = true; }
    require(threw, "output size is checked");
}

// ---- real state file: ratio and throughput per component class ----

struct Component { std::string name; std::size_t offset = 0, bytes = 0; std::uint32_t width = 2; };

std::string class_of(const std::string& name) {
    if (name.rfind("kv.L", 0) == 0) return "kv (bf16)";
    if (name.rfind("index.L", 0) == 0) return "index (bf16)";
    if (name.rfind("mtp.", 0) == 0) return "mtp rows+held (bf16)";
    if (name == "recurrent_matrix") return "recurrent_matrix (fp32)";
    if (name.find("convolution") != std::string::npos) return "conv histories (bf16)";
    return "host words";
}
// The element width the producer uses (src/cuda/models/qwen/qwen_handoff.cpp, element_width).
std::uint32_t width_of(const std::string& name) {
    return (name == "recurrent_matrix" || name == "ngram_history" || name == "position_history") ? 4U : 2U;
}

int bench_sample(const std::string& path, unsigned threads) {
    std::ifstream in(path, std::ios::binary);
    require(static_cast<bool>(in), "cannot open " + path);
    char magic[8];
    std::uint64_t n = 0;
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(&n), 8);
    std::string header(static_cast<std::size_t>(n), '\0');
    in.read(header.data(), static_cast<std::streamsize>(n));
    const auto h = strix::serve::parse_json(header);
    Bytes payload((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto position = static_cast<std::size_t>(h.find("position")->as_int());
    std::vector<Component> comps;
    std::size_t at = 0;
    for (const auto& r : h.find("rows")->as_array()) {
        const auto bpr = static_cast<std::size_t>(r.find("bytes_per_row")->as_int());
        const auto shift = static_cast<std::size_t>(r.find("shift")->as_int());
        const auto name = r.find("id")->as_string();
        comps.push_back({name, at, bpr * (position - shift), width_of(name)});
        at += comps.back().bytes;
    }
    const std::size_t rows_end = at;
    for (const auto& f : h.find("fixed")->as_array()) {
        const auto name = f.find("id")->as_string();
        comps.push_back({name, rows_end + static_cast<std::size_t>(f.find("offset")->as_int()), static_cast<std::size_t>(f.find("bytes")->as_int()),
            width_of(name)});
    }
    std::cout << "{\"sample\":\"" << path << "\",\"position\":" << position << ",\"payload_bytes\":" << payload.size() << ",\"classes\":[\n";
    struct Acc { std::size_t plain = 0, encoded = 0; double enc_s = 0, dec_s = 0; };
    std::map<std::string, Acc> classes;
    for (const auto& c : comps) {
        const std::vector<bp::Region> region = {{payload.data() + c.offset, c.bytes, c.width}};
        auto& a = classes[class_of(c.name)];
        const int reps = c.bytes < (std::size_t{8} << 20) ? 5 : 2;
        Bytes frame;
        double best_e = 1e30, best_d = 1e30;
        Bytes out(c.bytes);
        for (int r = 0; r < reps; ++r) {
            const auto t0 = Clock::now();
            frame = bp::encode(region, nullptr);
            const auto t1 = Clock::now();
            bp::decode(frame, out, nullptr);
            const auto t2 = Clock::now();
            best_e = std::min(best_e, secs(t0, t1));
            best_d = std::min(best_d, secs(t1, t2));
        }
        require(std::equal(out.begin(), out.end(), payload.begin() + static_cast<std::ptrdiff_t>(c.offset)), c.name + ": round trip differs");
        a.plain += c.bytes; a.encoded += frame.size(); a.enc_s += best_e; a.dec_s += best_d;
    }
    bool first = true;
    for (const auto& [name, a] : classes) {
        std::printf("%s {\"class\":\"%s\",\"bytes\":%zu,\"ratio\":%.4f,\"encode_MBps_1t\":%.0f,\"decode_MBps_1t\":%.0f}", first ? " " : ",\n ",
            name.c_str(), a.plain, static_cast<double>(a.encoded) / static_cast<double>(a.plain), static_cast<double>(a.plain) / a.enc_s / 1e6,
            static_cast<double>(a.plain) / a.dec_s / 1e6);
        first = false;
    }
    // The whole payload as the sender ships it: 32 MiB chunks of pieces, each a 24-byte segment header + data.
    std::vector<std::vector<bp::Region>> chunks(1);
    std::size_t in_chunk = 0;
    const Bytes seg_header(24, 0);
    for (const auto& c : comps) {
        for (std::size_t off = 0; off < c.bytes;) {
            const std::size_t piece = std::min(c.bytes - off, std::size_t{8} << 20);
            if (in_chunk && in_chunk + piece > (std::size_t{32} << 20)) { chunks.emplace_back(); in_chunk = 0; }
            chunks.back().push_back({seg_header.data(), seg_header.size(), 1});
            chunks.back().push_back({payload.data() + c.offset + off, piece, c.width});
            in_chunk += piece + 24;
            off += piece;
        }
    }
    std::printf("\n],\"whole_payload\":[\n");
    for (const unsigned t : {1U, threads}) {
        bp::WorkerPool pool(t > 0 ? t - 1 : 0);
        double enc_s = 0, dec_s = 0;
        std::size_t total_plain = 0, total_encoded = 0;
        for (const auto& regions : chunks) {
            const auto t0 = Clock::now();
            const auto frame = bp::encode(regions, &pool);
            const auto t1 = Clock::now();
            Bytes out(static_cast<std::size_t>(bp::decoded_size(frame)));
            const auto t2 = Clock::now();
            bp::decode(frame, out, &pool);
            const auto t3 = Clock::now();
            require(out == concat(regions), "chunk round trip differs");
            enc_s += secs(t0, t1);
            dec_s += secs(t2, t3);
            total_plain += out.size();
            total_encoded += frame.size();
        }
        std::printf("%s {\"threads\":%u,\"chunks\":%zu,\"ratio\":%.4f,\"encode_MBps\":%.0f,\"decode_MBps\":%.0f}", t == 1U ? " " : ",\n ", t,
            chunks.size(), static_cast<double>(total_encoded) / static_cast<double>(total_plain), static_cast<double>(total_plain) / enc_s / 1e6,
            static_cast<double>(total_plain) / dec_s / 1e6);
        if (threads <= 1U) break;
    }
    std::printf("\n]}\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 3 && std::string(argv[1]) == "--sample")
            return bench_sample(argv[2], argc >= 4 ? static_cast<unsigned>(std::stoul(argv[3])) : 4U);
        bp::WorkerPool pool(3);
        test_xxh64();
        test_edges(pool);
        test_structured(pool);
        test_malformed(pool);
        std::cout << "bp-v1 codec tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
