// strix-xfer-probe: link utilisation of the shared state hand-off transport (strix::handoff), no GPU.
//   receiver: strix-xfer-probe listen PORT [--bind ADDR] [--decode-threads N] [--sessions N]
//   sender:   strix-xfer-probe send HOST:PORT[@BIND] [--sizes 64K,1M,16M] [--streams 1,2,4] [--codecs identity,bp-v1]
//             [--total 256M] [--data random|bf16|fp32|zeros|file:PATH] [--width 1|2|4] [--encode-threads N]
//             [--line-mbps 1000] [--repeat N] [--encode-batch N] [--quit]
//             [--schedule FILE [--data4 SRC]]  pieces released at their ready time (scripts/bench/xfer/schedule.py)
//   plan:     strix-xfer-probe plan --schedule FILE [--threads N]  (shared cost model per link class, no network)
//   codec:    strix-xfer-probe codec [--data ...] [--width W] [--total 256M] [--threads 1,8]  (CPU throughput only)
// The sender runs every (size, streams, codec) combination as one session of --total bytes cut into --sizes pieces,
// and prints one JSON line per session: plain / wire MB/s over the session wall (connect + handshake + all pieces +
// reply), % of the line rate, sender and receiver process CPU seconds per plain GB. "file:" data is replicated to
// --total bytes (a real state dump measures the codec ratio on real bytes). The receiver answers each session with
// its own counters, so one sender line carries both ends.
#include "strix/bp_codec.hpp"
#include "strix/handoff.hpp"
#include "strix/handoff_policy.hpp"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/resource.h>
#endif

namespace {

namespace handoff = strix::handoff;
namespace bp_codec = strix::bp_codec;
using Clock = std::chrono::steady_clock;
unsigned g_encode_batch = handoff::SenderOptions{}.encode_batch;  // --encode-batch

double cpu_seconds() {
#ifdef _WIN32
    FILETIME c, e, k, u;
    if (!GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) return 0;
    auto t = [](const FILETIME& f) { return (static_cast<double>(f.dwHighDateTime) * 4294967296.0 + f.dwLowDateTime) * 1e-7; };
    return t(k) + t(u);
#else
    rusage r{};
    getrusage(RUSAGE_SELF, &r);
    return static_cast<double>(r.ru_utime.tv_sec + r.ru_stime.tv_sec) + 1e-6 * static_cast<double>(r.ru_utime.tv_usec + r.ru_stime.tv_usec);
#endif
}

std::uint64_t parse_bytes(const std::string& s) {
    if (s.empty()) throw std::invalid_argument("empty size");
    char* end = nullptr;
    errno = 0;
    const long double v = std::strtold(s.c_str(), &end);
    if (end == s.c_str() || errno == ERANGE || !std::isfinite(v) || v < 0)
        throw std::invalid_argument("bad size: " + s);
    std::uint64_t mul = 1;
    const std::string suffix(end);
    if (suffix == "K" || suffix == "k") mul = 1ULL << 10;
    else if (suffix == "M" || suffix == "m") mul = 1ULL << 20;
    else if (suffix == "G" || suffix == "g") mul = 1ULL << 30;
    else if (!suffix.empty()) throw std::invalid_argument("bad size suffix: " + s);
    const long double scaled = v * static_cast<long double>(mul);
    // Exclusive power-of-two bounds remain exact even when long double is binary64 (MSVC).
    // Probe sizes must contain at least one byte; fractional bytes are truncated.
    if (!std::isfinite(scaled) || scaled < 1 || scaled >= std::ldexp(1.0L, 64) ||
        scaled >= std::ldexp(1.0L, std::numeric_limits<std::size_t>::digits))
        throw std::invalid_argument("size must be positive and representable: " + s);
    const auto bytes = static_cast<std::uint64_t>(scaled);
    if (bytes > std::vector<std::uint8_t>{}.max_size()) throw std::invalid_argument("size exceeds allocation limit: " + s);
    return bytes;
}

unsigned parse_threads(const std::string& s) {
    std::size_t used = 0;
    const auto n = std::stoull(s, &used);
    if (s.empty() || used != s.size() || n > 256) throw std::invalid_argument("threads must be 0..256");
    return static_cast<unsigned>(n);
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');)
        if (!t.empty()) out.push_back(t);
    return out;
}

// Synthetic data. bf16 / fp32: normal(0, 1) values (sign and low mantissa random, exponent concentrated), the shape
// of attention K/V and recurrent state; random: incompressible upper bound; file: a real dump replicated to `total`.
std::vector<std::uint8_t> make_data(const std::string& kind, std::uint64_t total, std::uint32_t& width) {
    std::vector<std::uint8_t> d(total);
    std::uint64_t x = 0x9E3779B97F4A7C15ULL;
    auto next = [&x] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    if (kind == "random") {
        for (std::uint64_t i = 0; i + 8 <= total; i += 8) { const auto v = next(); std::memcpy(&d[i], &v, 8); }
        if (!width) width = 1;
    } else if (kind == "zeros") {
        if (!width) width = 1;
    } else if (kind == "bf16" || kind == "fp32") {
        const bool bf = kind == "bf16";
        if (!width) width = bf ? 2 : 4;
        const std::uint64_t n = total / (bf ? 2 : 4);
        for (std::uint64_t i = 0; i < n; ++i) {
            const double u1 = (static_cast<double>(next() >> 11) + 1.0) * 0x1.0p-53, u2 = static_cast<double>(next() >> 11) * 0x1.0p-53;
            const float f = static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
            std::uint32_t b;
            std::memcpy(&b, &f, 4);
            if (bf) { const auto h = static_cast<std::uint16_t>(b >> 16); std::memcpy(&d[2 * i], &h, 2); }
            else std::memcpy(&d[4 * i], &b, 4);
        }
    } else if (kind.rfind("file:", 0) == 0) {
        std::ifstream f(kind.substr(5), std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + kind.substr(5));
        std::vector<std::uint8_t> src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (src.empty()) throw std::runtime_error("empty file");
        for (std::uint64_t i = 0; i < total; i += src.size())
            std::memcpy(&d[i], src.data(), std::min<std::uint64_t>(src.size(), total - i));
        if (!width) width = 1;
    } else {
        throw std::invalid_argument("unknown --data " + kind);
    }
    return d;
}

handoff::Codec parse_codec(const std::string& s) {
    if (s == "identity" || s == "off") return handoff::Codec::identity;
    if (s == "bp-v1" || s == "bp_v1") return handoff::Codec::bp_v1;
    throw std::invalid_argument("unknown codec " + s);
}

int run_listen(int argc, char** argv) {
    if (argc < 3) throw std::invalid_argument("listen PORT");
    const auto port = static_cast<std::uint16_t>(std::stoi(argv[2]));
    std::string bind;
    unsigned threads = 4;
    long sessions = -1;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--bind") bind = argv[i + 1];
        else if (a == "--decode-threads") threads = static_cast<unsigned>(std::stoul(argv[i + 1]));
        else if (a == "--sessions") sessions = std::stol(argv[i + 1]);
        else throw std::invalid_argument("unknown option " + a);
    }
    handoff::Listener listener(port, bind);
    std::cerr << "xfer-probe listening on " << listener.port() << std::endl;
    for (long n = 0; sessions < 0 || n < sessions; ++n) {
        handoff::ReceiverOptions ro;
        ro.decode_threads = threads;
        auto session = listener.accept(ro);
        const double c0 = cpu_seconds();
        const auto t0 = Clock::now();
        handoff::Piece piece;
        while (session->next(piece)) {}
        const double wall = std::chrono::duration<double>(Clock::now() - t0).count();
        const auto st = session->stats();
        char buf[512];
        std::snprintf(buf, sizeof buf,
            "{\"wall_s\":%.4f,\"cpu_s\":%.4f,\"decode_s\":%.4f,\"wait_s\":%.4f,\"first_piece_s\":%.4f,\"end_s\":%.4f,\"wire_bytes\":%llu}",
            wall, cpu_seconds() - c0, st.decode_seconds, st.wait_seconds, st.first_piece, st.end, static_cast<unsigned long long>(st.wire_bytes));
        const std::string text = buf;
        session->reply(0, std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
        if (session->hello().model == "xfer-probe/quit") break;
    }
    return 0;
}

// --schedule FILE: lines "READY_MS BYTES WIDTH" (one piece each, in order; '#' comments). Piece bytes come from the
// --data source (width 1/2) or the --data4 source (width 4), read cyclically. The session opens first (the connection
// is warm before the prefill starts, as in the services); t0 = prefill start; piece i is submitted at t0 + READY_MS.
// tail_s = finish (receiver checked the manifest and replied) - last ready: the hand-off seconds the prefill sees.
int run_schedule(const handoff::Endpoint& to, const std::string& file, const std::string& data, const std::string& data4,
    const std::vector<std::string>& streams, const std::vector<std::string>& codecs, unsigned enc_threads, double line_mbps,
    unsigned repeat, std::uint64_t& session_id, bool quit) {
    struct Item { double ready_s; std::uint64_t bytes; std::uint32_t width; };
    std::vector<Item> items;
    std::ifstream in(file);
    if (!in) throw std::runtime_error("cannot open schedule " + file);
    std::uint64_t need2 = 0, need4 = 0;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        double ms = 0;
        std::uint64_t b = 0;
        std::uint32_t w = 2;
        if (!(ls >> ms >> b >> w)) throw std::invalid_argument("bad schedule line: " + line);
        items.push_back({ms / 1e3, b, w});
        (w == 4 ? need4 : need2) = std::max(w == 4 ? need4 : need2, b);
    }
    if (items.empty()) throw std::invalid_argument("empty schedule");
    std::uint32_t w2 = 2, w4 = 4;
    const std::uint64_t pool = std::uint64_t{256} << 20;
    const auto src2 = make_data(data, std::max(pool, need2), w2), src4 = make_data(data4, std::max(pool, need4), w4);
    for (unsigned rep = 0; rep < repeat; ++rep)
        for (const auto& ns : streams)
            for (const auto& cs : codecs) {
                handoff::SenderOptions so;
                so.codec = parse_codec(cs);
                so.streams = static_cast<unsigned>(std::stoul(ns));
                so.encode_threads = enc_threads;
                so.encode_batch = g_encode_batch;
                handoff::Hello hello;
                hello.session = ++session_id;
                hello.model = "xfer-probe/v1";
                handoff::Sender sender(to, hello, so);
                const double c0 = cpu_seconds();
                const auto t0 = Clock::now();
                std::uint64_t off2 = 0, off4 = 0;
                double last_ready = 0;
                for (std::size_t i = 0; i < items.size(); ++i) {
                    const auto& it = items[i];
                    std::this_thread::sleep_until(t0 + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(it.ready_s)));
                    const auto& src = it.width == 4 ? src4 : src2;
                    auto& off = it.width == 4 ? off4 : off2;
                    if (off + it.bytes > src.size()) off = 0;
                    const bp_codec::Region r{src.data() + off, static_cast<std::size_t>(it.bytes), it.width};
                    off += it.bytes;
                    sender.submit(handoff::PieceHeader{1, static_cast<std::uint32_t>(i), 0, 0, it.bytes}, std::span<const bp_codec::Region>(&r, 1));
                    last_ready = std::max(last_ready, it.ready_s);
                }
                const auto reply = sender.finish();
                const double end = std::chrono::duration<double>(Clock::now() - t0).count();
                const auto st = sender.stats();
                const double span = end - std::min(items.front().ready_s, end);
                std::printf("{\"schedule\":\"%s\",\"pieces\":%zu,\"streams\":%s,\"codec\":\"%s\",\"plain_bytes\":%llu,\"wire_bytes\":%llu,\"ratio\":%.4f,"
                            "\"last_ready_s\":%.4f,\"end_s\":%.4f,\"tail_s\":%.4f,\"link_util_pct\":%.1f,\"send_cpu_s\":%.3f,\"encode_s\":%.4f,\"recv\":%s}\n",
                    file.c_str(), items.size(), ns.c_str(), st.codec == handoff::Codec::bp_v1 ? "bp-v1" : "identity",
                    static_cast<unsigned long long>(st.plain_bytes), static_cast<unsigned long long>(st.wire_bytes),
                    static_cast<double>(st.plain_bytes) / static_cast<double>(std::max<std::uint64_t>(1, st.wire_bytes)), last_ready, end, end - last_ready,
                    100.0 * static_cast<double>(st.wire_bytes) * 8.0 / 1e6 / line_mbps / std::max(1e-9, span), cpu_seconds() - c0, st.encode_seconds,
                    reply.payload.empty() ? "{}" : std::string(reply.payload.begin(), reply.payload.end()).c_str());
                std::fflush(stdout);
            }
    if (quit) {
        handoff::Hello hello;
        hello.session = ++session_id;
        hello.model = "xfer-probe/quit";
        handoff::Sender sender(to, hello, handoff::SenderOptions{});
        (void)sender.finish();
    }
    return 0;
}

int run_send(int argc, char** argv) {
    if (argc < 3) throw std::invalid_argument("send HOST:PORT");
    const auto to = handoff::parse_path_map(argv[2]).at(0);
    std::vector<std::string> sizes{"1M"}, streams{"1"}, codecs{"identity"};
    std::string data = "bf16";
    std::uint64_t total = 256ULL << 20;
    std::uint32_t width = 0;
    unsigned enc_threads = 8, repeat = 1;
    double line_mbps = 1000;
    bool quit = false;
    std::string schedule, data4 = "fp32";
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--quit") { quit = true; continue; }
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
        const std::string v = argv[++i];
        if (a == "--sizes") sizes = split(v);
        else if (a == "--streams") streams = split(v);
        else if (a == "--codecs") codecs = split(v);
        else if (a == "--total") total = parse_bytes(v);
        else if (a == "--data") data = v;
        else if (a == "--width") width = static_cast<std::uint32_t>(std::stoul(v));
        else if (a == "--encode-threads") enc_threads = static_cast<unsigned>(std::stoul(v));
        else if (a == "--line-mbps") line_mbps = std::stod(v);
        else if (a == "--repeat") repeat = static_cast<unsigned>(std::stoul(v));
        else if (a == "--schedule") schedule = v;
        else if (a == "--data4") data4 = v;
        else if (a == "--encode-batch") g_encode_batch = static_cast<unsigned>(std::stoul(v));
        else throw std::invalid_argument("unknown option " + a);
    }
    std::uint64_t session_id = static_cast<std::uint64_t>(Clock::now().time_since_epoch().count());
    if (!schedule.empty()) return run_schedule(to, schedule, data, data4, streams, codecs, enc_threads, line_mbps, repeat, session_id, quit);
    const auto bytes = make_data(data, total, width);
    for (unsigned rep = 0; rep < repeat; ++rep)
        for (const auto& sz : sizes)
            for (const auto& ns : streams)
                for (const auto& cs : codecs) {
                    const std::uint64_t piece = std::max<std::uint64_t>(1, parse_bytes(sz));
                    handoff::SenderOptions so;
                    so.codec = parse_codec(cs);
                    so.streams = static_cast<unsigned>(std::stoul(ns));
                    so.encode_threads = enc_threads;
                so.encode_batch = g_encode_batch;
                    handoff::Hello hello;
                    hello.session = ++session_id;
                    hello.model = "xfer-probe/v1";
                    const double c0 = cpu_seconds();
                    const auto t0 = Clock::now();
                    handoff::Sender sender(to, hello, so);
                    std::uint64_t n = 0;
                    for (std::uint64_t off = 0; off < total; off += piece, ++n) {
                        const bp_codec::Region r{bytes.data() + off, static_cast<std::size_t>(std::min(piece, total - off)), width};
                        sender.submit(handoff::PieceHeader{1, static_cast<std::uint32_t>(n), 0, off, r.bytes}, std::span<const bp_codec::Region>(&r, 1));
                    }
                    const auto reply = sender.finish();
                    const double wall = std::chrono::duration<double>(Clock::now() - t0).count();
                    const double cpu = cpu_seconds() - c0;
                    const auto st = sender.stats();
                    const double plain_mbs = static_cast<double>(st.plain_bytes) / wall / 1e6, wire_mbs = static_cast<double>(st.wire_bytes) / wall / 1e6;
                    const double gb = static_cast<double>(st.plain_bytes) / 1e9;
                    std::printf("{\"data\":\"%s\",\"piece\":%llu,\"pieces\":%llu,\"streams\":%s,\"codec\":\"%s\",\"plain_bytes\":%llu,\"wire_bytes\":%llu,"
                                "\"ratio\":%.4f,\"wall_s\":%.4f,\"plain_MBps\":%.1f,\"wire_MBps\":%.1f,\"line_pct\":%.1f,\"first_byte_s\":%.4f,"
                                "\"send_cpu_s_per_GB\":%.3f,\"encode_s\":%.4f,\"recv\":%s}\n",
                        data.c_str(), static_cast<unsigned long long>(piece), static_cast<unsigned long long>(n), ns.c_str(),
                        st.codec == handoff::Codec::bp_v1 ? "bp-v1" : "identity", static_cast<unsigned long long>(st.plain_bytes),
                        static_cast<unsigned long long>(st.wire_bytes), static_cast<double>(st.plain_bytes) / static_cast<double>(std::max<std::uint64_t>(1, st.wire_bytes)),
                        wall, plain_mbs, wire_mbs, 100.0 * wire_mbs * 8.0 / line_mbps, st.first_byte, cpu / gb, st.encode_seconds,
                        reply.payload.empty() ? "{}" : std::string(reply.payload.begin(), reply.payload.end()).c_str());
                    std::fflush(stdout);
                }
    if (quit) {
        handoff::Hello hello;
        hello.session = ++session_id;
        hello.model = "xfer-probe/quit";
        handoff::Sender sender(to, hello, handoff::SenderOptions{});
        (void)sender.finish();
    }
    return 0;
}

// plan --schedule FILE [--threads N]: the shared cost model (strix/handoff_policy.hpp) on a schedule, per link class:
// identity on one stream, today's family default (bp-v1, one stream, 8 threads, serial encoder) and the choice auto
// would make, with the predicted tail after the last ready piece and the link's busy share.
int run_plan(int argc, char** argv) {
    std::string file;
    unsigned threads = 16;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--schedule") file = v;
        else if (a == "--threads") threads = parse_threads(v);
        else throw std::invalid_argument("unknown option " + a);
    }
    std::ifstream in(file);
    if (!in) throw std::runtime_error("cannot open schedule " + file);
    std::vector<handoff::PieceLoad> pieces;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        double ms = 0, b = 0;
        unsigned w = 2;
        if (!(ls >> ms >> b >> w)) throw std::invalid_argument("bad schedule line: " + line);
        pieces.push_back({ms / 1e3, b, w == 4 ? handoff::DataClass::fp32 : w == 2 ? handoff::DataClass::bf16 : handoff::DataClass::raw});
    }
    if (pieces.empty()) throw std::invalid_argument("empty schedule");
    const handoff::CodecModel cm;
    double last = 0, plain = 0;
    for (const auto& p : pieces) { last = std::max(last, p.ready_s); plain += p.plain_bytes; }
    std::printf("# %s: %zu pieces, %.1f MB, last ready %.3f s\n", file.c_str(), pieces.size(), plain / 1e6, last);
    for (const auto& link : handoff::link_classes) {
        const auto id = handoff::predict(pieces, link, cm, handoff::Choice{handoff::Codec::identity, 1, 0, 1});
        const auto today = handoff::predict(pieces, link, cm, handoff::Choice{handoff::Codec::bp_v1, 1, 8, 1});
        const auto c = handoff::choose(pieces, link, cm, threads);
        const auto best = handoff::predict(pieces, link, cm, c);
        std::printf("{\"link\":\"%s\",\"identity_tail_s\":%.3f,\"bp8_tail_s\":%.3f,\"auto\":\"%s s=%u t=%u b=%u\",\"auto_tail_s\":%.3f,"
                    "\"auto_wire_MB\":%.1f,\"auto_link_busy_pct\":%.1f}\n",
            std::string(link.name).c_str(), id.tail_s, today.tail_s, handoff::codec_name(c.codec), c.streams, c.encode_threads, c.encode_batch,
            best.tail_s, best.wire_bytes / 1e6, 100.0 * best.utilisation(best.end_s - pieces.front().ready_s));
    }
    return 0;
}

int run_codec(int argc, char** argv) {
    std::string data = "bf16";
    std::uint64_t total = 256ULL << 20;
    std::uint32_t width = 0;
    std::vector<std::string> threads{"1", "8"};
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--data") data = v;
        else if (a == "--total") total = parse_bytes(v);
        else if (a == "--width") width = static_cast<std::uint32_t>(std::stoul(v));
        else if (a == "--threads") threads = split(v);
        else throw std::invalid_argument("unknown option " + a);
    }
    const auto bytes = make_data(data, total, width);
    const bp_codec::Region r{bytes.data(), bytes.size(), width};
    for (const auto& t : threads) {
        bp_codec::WorkerPool pool(static_cast<unsigned>(std::stoul(t)));
        bp_codec::EncodeStats es;
        auto t0 = Clock::now();
        const auto frame = bp_codec::encode(std::span<const bp_codec::Region>(&r, 1), &pool, {}, &es);
        const double enc = std::chrono::duration<double>(Clock::now() - t0).count();
        std::vector<std::uint8_t> out(bytes.size());
        t0 = Clock::now();
        bp_codec::decode(frame, out, &pool);
        const double dec = std::chrono::duration<double>(Clock::now() - t0).count();
        t0 = Clock::now();
        const auto h = bp_codec::xxh64(bytes.data(), bytes.size());
        const double hs = std::chrono::duration<double>(Clock::now() - t0).count();
        std::printf("{\"data\":\"%s\",\"width\":%u,\"threads\":%s,\"ratio\":%.4f,\"encode_MBps\":%.0f,\"decode_MBps\":%.0f,\"xxh64_1t_MBps\":%.0f,"
                    "\"huffman_planes\":%llu,\"raw_planes\":%llu,\"const_planes\":%llu,\"ok\":%d,\"h\":%llu}\n",
            data.c_str(), width, t.c_str(), static_cast<double>(bytes.size()) / static_cast<double>(frame.size()), static_cast<double>(bytes.size()) / enc / 1e6,
            static_cast<double>(bytes.size()) / dec / 1e6, static_cast<double>(bytes.size()) / hs / 1e6, static_cast<unsigned long long>(es.huffman_planes),
            static_cast<unsigned long long>(es.raw_planes), static_cast<unsigned long long>(es.constant_planes), out == bytes ? 1 : 0,
            static_cast<unsigned long long>(h & 0xFFFF));
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "listen") return run_listen(argc, argv);
        if (mode == "send") return run_send(argc, argv);
        if (mode == "codec") return run_codec(argc, argv);
        if (mode == "plan") return run_plan(argc, argv);
        std::cerr << "usage: strix-xfer-probe listen PORT | send HOST:PORT [options] | codec [options]\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "strix-xfer-probe: " << e.what() << "\n";
        return 1;
    }
}
