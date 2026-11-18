// strix::handoff + strix::handoff policy host tests (loopback TCP, no GPU).
//   1. a session delivers every piece byte for byte, in seq order, for identity and bp-v1, with the encoder one piece
//      at a time and batched (SenderOptions::encode_batch), on 1 and 3 streams; batched frames are the same bytes
//      on the wire (per-sequence header and encoded-payload bytes) as serial ones.
//   2. the cost model: predict() arithmetic, choose() picks the cheapest choice within 1 % of the earliest end.
//   3. the resolver: family defaults stay unless asked, auto needs a declared link, explicit values and their
//      validation, effective-config lines and envcfg records.
#include "strix/handoff.hpp"
#include "strix/handoff_policy.hpp"

#include <atomic>
#include <future>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <crtdbg.h>
#include <stdlib.h>
#endif

namespace {

namespace hf = strix::handoff;
int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

void set_env(const char* k, const char* v) {
#ifdef _WIN32
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1);
    else unsetenv(k);
#endif
}

std::vector<std::uint8_t> bf16_like(std::size_t n, std::uint64_t seed) {
    std::vector<std::uint8_t> d(n);
    std::uint64_t x = seed | 1;
    for (std::size_t i = 0; i < n; ++i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        d[i] = (i & 1) ? static_cast<std::uint8_t>(0x3C + (x & 3)) : static_cast<std::uint8_t>(x);  // BF16: few high bytes
    }
    return d;
}

struct Witness {
    std::uint64_t wire = 0;
    std::size_t largest_batch = 0, peak_reservation = 0;
    bool budget_ok = true;
    std::vector<std::vector<std::uint8_t>> frames;
};

Witness session(hf::Codec codec, unsigned streams, unsigned batch, const std::vector<std::vector<std::uint8_t>>& pieces,
                std::size_t budget = std::size_t{512} << 20) {
    hf::Listener listener(0, "127.0.0.1");
    std::uint64_t wire = 0;
    std::atomic<bool> ok{true};
    Witness witness;
    witness.frames.resize(pieces.size());
    std::promise<void> queued;
    auto ready = queued.get_future().share();
    bool gate_open = false;
    std::thread rx([&] {
        try {
            auto s = listener.accept();
            hf::Piece p;
            std::size_t n = 0;
            while (s->next(p)) {
                if (n >= pieces.size() || p.seq != n || p.plain != pieces[n] || p.header.index != n) ok = false;
                ++n;
            }
            if (n != pieces.size()) ok = false;
            wire = s->stats().wire_bytes;
            s->reply(7);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "receiver: %s\n", e.what());
            ok = false;
        }
    });
    try {
        hf::SenderOptions so;
        so.codec = codec;
        so.streams = streams;
        so.encode_threads = 4;
        so.encode_batch = batch;
        so.max_queued_wire_bytes = budget;
        so.socket_buffer = 4096;
        so.before_encode = [&] { ready.wait(); };
        so.observe_batch = [&](std::size_t count, std::size_t reserved) {
            witness.largest_batch = std::max(witness.largest_batch, count);
            witness.peak_reservation = std::max(witness.peak_reservation, reserved);
            if (reserved > budget && count != 1) witness.budget_ok = false;
        };
        so.observe_frame = [&](std::uint32_t seq, std::span<const std::uint8_t> header, std::span<const std::uint8_t> payload) {
            auto& frame = witness.frames.at(seq);
            frame.assign(header.begin(), header.end());
            frame.insert(frame.end(), payload.begin(), payload.end());
        };
        hf::Hello hello;
        hello.session = 42;
        hello.model = "test/handoff-v1";
        hf::Sender tx(hf::Endpoint{"127.0.0.1", listener.port(), ""}, hello, so);
        struct OpenGate {
            std::promise<void>& promise;
            bool& open;
            ~OpenGate() { if (!open) promise.set_value(); }
        } guard{queued, gate_open};
        for (std::size_t i = 0; i < pieces.size(); ++i) {
            const strix::bp_codec::Region r{pieces[i].data(), pieces[i].size(), 2};
            tx.submit(hf::PieceHeader{3, static_cast<std::uint32_t>(i), 0, i, 1}, std::span<const strix::bp_codec::Region>(&r, 1));
        }
        queued.set_value();
        gate_open = true;
        const auto reply = tx.finish();
        if (reply.status != 7) ok = false;
        if (tx.stats().codec != codec) ok = false;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sender: %s\n", e.what());
        ok = false;
    }
    rx.join();
    witness.wire = ok ? wire : 0;
    return witness;
}

void test_sessions() {
    std::vector<std::vector<std::uint8_t>> pieces;
    const std::size_t sizes[] = {0, 1, 3, 65536, 1 << 20, (1 << 20) + 7, 3 << 20, 4096, 5, 2 << 20};
    for (std::size_t i = 0; i < 40; ++i) pieces.push_back(bf16_like(sizes[i % 10], 1000 + i));
    for (const unsigned streams : {1U, 3U}) {
        const auto id1 = session(hf::Codec::identity, streams, 1, pieces), id8 = session(hf::Codec::identity, streams, 8, pieces);
        const auto bp1 = session(hf::Codec::bp_v1, streams, 1, pieces), bp8 = session(hf::Codec::bp_v1, streams, 8, pieces);
        check(id1.wire > 0 && id1.wire == id8.wire, "identity: delivered, encode_batch has no effect");
        check(bp1.wire > 0 && bp1.wire == bp8.wire && bp1.frames == bp8.frames,
              "bp-v1: every frame header and encoded payload matches serial bytes");
        check(bp1.largest_batch == 1 && bp8.largest_batch > 1, "deterministically exercised multi-piece encoding");
        check(bp1.wire > 0 && bp1.wire < id1.wire, "bp-v1 compresses BF16-like rows");
    }
}

void test_encoding_budget() {
    std::vector<std::vector<std::uint8_t>> pieces(16, bf16_like(256 << 10, 73));
    // Make both byte planes incompressible; initial queue is held behind the gate.
    for (auto& piece : pieces)
        for (std::size_t i = 0; i < piece.size(); ++i) piece[i] = static_cast<std::uint8_t>((i * 131 + i / 257) & 255);
    const auto bounded = session(hf::Codec::bp_v1, 1, 16, pieces, 600 << 10);
    check(bounded.wire > 0 && bounded.budget_ok && bounded.peak_reservation > 0 &&
          bounded.peak_reservation <= (600U << 10) && bounded.largest_batch == 1,
          "encoding reservations limit batches before allocating outputs");
    const auto oversized = session(hf::Codec::bp_v1, 1, 16, pieces, 1);
    check(oversized.wire > 0 && oversized.budget_ok && oversized.peak_reservation > 1 && oversized.largest_batch == 1,
          "one oversized encoding makes progress alone");
    check(bounded.frames == oversized.frames, "budget changes preserve all frame bytes");
}

void test_model() {
    const hf::LinkClass* g1 = hf::find_link("1gbe");
    check(g1 && !hf::find_link("nope"), "link table lookup");
    const hf::CodecModel cm;
    std::vector<hf::PieceLoad> one{{0.0, 117.7e6, hf::DataClass::raw}};
    auto p = hf::predict(one, *g1, cm, hf::Choice{hf::Codec::identity, 1, 0, 1});
    check(std::fabs(p.end_s - (1.0 + g1->per_message_s + hf::kSessionFinishS)) < 1e-9, "identity: one 117.7 MB piece takes 1 s + per-message cost on 1gbe");
    // a piece ready late: the tail is its own wire time
    std::vector<hf::PieceLoad> two{{0.0, 1e6, hf::DataClass::raw}, {5.0, 117.7e6, hf::DataClass::raw}};
    p = hf::predict(two, *g1, cm, hf::Choice{hf::Codec::identity, 1, 0, 1});
    check(std::fabs(p.tail_s - (1.0 + g1->per_message_s + hf::kSessionFinishS)) < 1e-9, "tail = wire of the last-ready piece");
    // bp-v1 on BF16 rows on 1gbe: wire bytes / 1.468, encode far faster than the link
    std::vector<hf::PieceLoad> rows(64, hf::PieceLoad{0.0, 16.0 * (1 << 20), hf::DataClass::bf16});
    const auto bp = hf::predict(rows, *g1, cm, hf::Choice{hf::Codec::bp_v1, 1, 8, 1});
    const auto id = hf::predict(rows, *g1, cm, hf::Choice{hf::Codec::identity, 1, 0, 1});
    check(bp.end_s < id.end_s / 1.4, "bp-v1 wins on 1gbe for BF16 rows");
    const auto c = hf::choose(rows, *g1, cm, 16);
    check(c.codec == hf::Codec::bp_v1 && c.streams == 1 && c.encode_threads <= 2, "1gbe: bp-v1, one stream, a small pool");
    const hf::LinkClass cross{"cross", 100, 0, 68.5, "test"};
    // At default encoder rates bp-v1/two streams wins outright: the audit's
    // original example was not a tie. A calibrated 102 MB/s encoder makes
    // bp-v1/one stream and identity/two streams both eligible within 1%.
    auto tie_model = cm;
    tie_model.encode_MBps_1t = tie_model.encode_MBps_max = 102;
    const auto tied = hf::choose(rows, cross, tie_model, 16);
    check(tied.codec == hf::Codec::identity && tied.streams == 2, "codec-first tie across stream counts");
    for (const unsigned limit : {257U, 2147483648U, std::numeric_limits<unsigned>::max()}) {
        bool rejected = false;
        try { (void)hf::choose(rows, *g1, cm, limit); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "oversized thread limit rejected without enumeration");
    }
    check(hf::choose(rows, *g1, cm, 0).encode_threads == 0, "zero worker cap uses caller encoding");
    (void)hf::choose(rows, *g1, cm, 256);
    // small pieces on a fast link: batching lifts the one-block-per-piece ceiling
    std::vector<hf::PieceLoad> small(512, hf::PieceLoad{0.0, double(1 << 20), hf::DataClass::bf16});
    const hf::LinkClass* oc = hf::find_link("oculink");
    const auto serial = hf::predict(small, *oc, cm, hf::Choice{hf::Codec::bp_v1, 1, 8, 1});
    const auto batched = hf::predict(small, *oc, cm, hf::Choice{hf::Codec::bp_v1, 1, 8, 16});
    check(batched.end_s < serial.end_s / 1.5, "encode_batch: 1 MiB pieces no longer encode on one thread");
}

void test_resolver() {
    for (const char* k : {"STRIX_HANDOFF_LINK", "STRIX_HANDOFF_CODEC", "STRIX_HANDOFF_STREAMS", "STRIX_HANDOFF_ENCODE_THREADS", "STRIX_HANDOFF_ENCODE_BATCH"})
        set_env(k, nullptr);
    const hf::Choice family{hf::Codec::bp_v1, 2, 8, 1};
    auto r = hf::resolve(family, {});
    check(r.choice.codec == family.codec && r.choice.streams == 2 && r.choice.encode_threads == 8 && r.choice.encode_batch == 1 && !r.link,
          "nothing asked: the family defaults");
    check(r.lines.size() == 5 && r.lines[1] == "handoff codec bp-v1 (default)" && r.lines[0] == "handoff link undeclared (default)", "default lines");
    bool threw = false;
    try { (void)hf::resolve(family, hf::Request{"", "auto", "", "", ""}); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "auto without a declared link is refused");
    r = hf::resolve(family, hf::Request{"1gbe", "auto", "auto", "auto", "auto"});
    check(r.link && r.link->name == "1gbe" && r.choice.codec == hf::Codec::bp_v1 && r.choice.streams == 1, "auto on 1gbe");
    check(r.lines[1].find("auto: 1gbe predicts") != std::string::npos, "auto line names the prediction");
    set_env("STRIX_HANDOFF_CODEC", "identity");
    set_env("STRIX_HANDOFF_STREAMS", "4");
    r = hf::resolve(family, {});
    check(r.choice.codec == hf::Codec::identity && r.choice.streams == 4 && r.lines[1] == "handoff codec identity (env)", "env values");
    r = hf::resolve(family, hf::Request{"", "bp-v1", "", "", ""});
    check(r.choice.codec == hf::Codec::bp_v1 && r.lines[1] == "handoff codec bp-v1 (cli)", "the CLI overrides the environment");
    set_env("STRIX_HANDOFF_CODEC", nullptr);
    set_env("STRIX_HANDOFF_STREAMS", nullptr);
    r = hf::resolve(family, hf::Request{"250", "", "", "", ""});
    check(r.link && r.link->name == "custom" && std::fabs(r.link->goodput_MBps - 250) < 1e-9, "numeric link = MB/s");
    auto copied = r;
    auto moved = std::move(r);
    r = hf::resolve(family, hf::Request{"500", "", "", "", ""});
    check(copied.link && moved.link && copied.link->goodput_MBps == 250 && moved.link->goodput_MBps == 250,
          "numeric link copy/move owns its value after source replacement");
    copied = moved;
    moved = hf::resolve(family, hf::Request{"750", "", "", "", ""});
    check(copied.link->goodput_MBps == 250, "numeric link copy assignment is independent");
    r = hf::resolve(family, hf::Request{"loopback", "bp-v1", "3", "auto", "1"});
    check(r.choice.codec == hf::Codec::bp_v1 && r.choice.streams == 3 && r.choice.encode_threads >= 4 && r.choice.encode_batch == 1,
          "partial auto optimizes the fixed codec, streams and batch");
    std::vector<hf::PieceLoad> steady(64, {0, 16.0 * (1 << 20), hf::DataClass::bf16});
    const auto prediction = hf::predict(steady, *r.link, {}, r.choice);
    char predicted[80];
    std::snprintf(predicted, sizeof predicted, "predicts %.3f s, %.0f MB on the wire", prediction.end_s, prediction.wire_bytes / 1e6);
    check(r.lines[3].find(predicted) != std::string::npos, "diagnostic predicts the applied choice");
    std::vector<hf::PieceLoad> small(512, {0, double(1 << 20), hf::DataClass::bf16});
    r = hf::resolve(family, hf::Request{"loopback", "bp-v1", "3", "auto", "1"}, small);
    check(r.choice.encode_threads == 1, "explicit serial batch constrains the small-piece model");
    r = hf::resolve(family, hf::Request{"loopback", "bp-v1", "3", "auto", "16"}, small);
    check(r.choice.encode_threads >= 4, "parallel batch admits more workers");
    r = hf::resolve(family, hf::Request{"loopback", "", "", "auto", ""});
    check(r.choice.codec == family.codec && r.choice.streams == family.streams && r.choice.encode_batch == family.encode_batch,
          "omitted non-auto fields remain family defaults during search");
    set_env("STRIX_HANDOFF_LINK", "1gbe");
    set_env("STRIX_HANDOFF_STREAMS", "auto");
    threw = false;
    try { (void)hf::resolve(family, {}, {}, 16, {false, false}); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "HTTP adapter refuses environment auto streams");
    r = hf::resolve(family, hf::Request{"", "", "2", "", ""}, {}, 16, {false, false});
    check(r.choice.streams == 2, "explicit HTTP connections override environment auto");
    set_env("STRIX_HANDOFF_STREAMS", nullptr);
    set_env("STRIX_HANDOFF_LINK", nullptr);
    for (const hf::Request& bad : {hf::Request{"", "zstd", "", "", ""}, hf::Request{"", "", "0", "", ""}, hf::Request{"", "", "65", "", ""},
                                   hf::Request{"fast", "", "", "", ""}, hf::Request{"", "", "", "x", ""}, hf::Request{"", "", "", "", "0"}}) {
        threw = false;
        try { (void)hf::resolve(family, bad); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "invalid values are refused");
    }
    check(strix::envcfg::describe().find("env STRIX_HANDOFF_CODEC") != std::string::npos, "reads are recorded for --print-effective-config");
    const auto so = hf::sender_options(hf::Choice{hf::Codec::identity, 3, 0, 4});
    check(so.codec == hf::Codec::identity && so.streams == 3 && so.encode_threads == 0 && so.encode_batch == 4, "sender_options");
}

}  // namespace

int main() {
#ifdef _WIN32
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    test_model();
    test_resolver();
    test_sessions();
    test_encoding_budget();
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("handoff tests: PASS");
    return 0;
}
