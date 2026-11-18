#pragma once
// strix::handoff policy: ONE resolver for the prompt-state hand-off options of every family (codec, TCP streams,
// encode threads) and the cost model it decides with. Host only, header only; no sockets here (strix/handoff.hpp).
//
// Estimated cost model (docs/xfer.md states its limits and required calibration):
//   link     goodput G (payload MB/s after TCP/IP framing), per-message cost a (s), per-stream cap S (one TCP
//            connection's ceiling, CPU bound on fast links): wire(piece) = a + wire_bytes / min(G, streams x S)
//   codec    bp-v1 (byte planes + Huffman, strix/bp_codec.hpp): ratio by data class, measured on a real Qwen state
//            (BF16 rows 1.468, FP32 recurrent matrix 1.171; random 1.0); encode rate min(threads x E1, Emax).
//            These sample constants are not universal: batching assumes ready pieces, receiver decode is omitted,
//            and nominal/non-TCP link classes require host/workload calibration before production promotion.
//   pipeline pieces leave in submit order through two stages, encoder then wire; a piece starts encoding when it is
//            ready (its layer / block is done) and the encoder is free. predict() returns when the last byte lands.
// A codec is lossless, streams and threads only schedule: no option here changes a bit of the imported state, so
// `auto` may pick any of them; the family defaults stay what each family ships until a receipt promotes `auto`.
//
// Switches (CLI values override; every read is recorded for --print-effective-config):
//   STRIX_HANDOFF_LINK            1gbe | 2.5gbe | 10gbe | usb4 | oculink | loopback | <goodput MB/s>; unset = undeclared
//   STRIX_HANDOFF_CODEC           identity | bp-v1 | auto            (auto needs a declared link)
//   STRIX_HANDOFF_STREAMS         1..64 | auto
//   STRIX_HANDOFF_ENCODE_THREADS  0..256 | auto
//   STRIX_HANDOFF_ENCODE_BATCH    1..256 | auto   (bp-v1 pieces encoded side by side; 1 = one at a time)
#include "strix/env_flags.hpp"
#include "strix/handoff.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::handoff {

struct LinkClass {
    std::string_view name;
    double goodput_MBps = 0;     // payload MB/s at line rate
    double per_message_s = 0;    // fixed cost per piece (frame header, syscall, ACK-free TCP: ~RTT/2 amortised)
    double per_stream_MBps = 0;  // one TCP connection's ceiling (sender + receiver CPU)
    std::string_view source;     // receipt or "nominal"
};

// Measured: 1gbe (prefill host -> node, 2026-11-15), loopback (node, 2026-11-15: one stream CPU ceiling). Nominal: 2.5/10 GbE at
// the same 94.1 % TCP/IPv4 MTU-1500 efficiency (1448 / 1538 payload bytes per wire byte); usb4 = the node pair's
// measured host path (4 rings, 35-36 Gb/s at >= 16 MB pieces, 2026-10-31) for a relay; oculink = PCIe 4.0 x4.
inline constexpr std::array<LinkClass, 6> link_classes{{
    {"1gbe", 117.7, 12e-6, 118.0, "measured 2026-11-15 PC->node: 118.0 MB/s one stream at 16 MiB pieces, +11.5 us per piece"},
    {"2.5gbe", 294.2, 12e-6, 7000.0, "nominal 2500 Mb/s x 0.941; per-piece cost as 1gbe; per-stream cap = loopback"},
    {"10gbe", 1176.8, 12e-6, 7000.0, "nominal 10 Gb/s x 0.941; per-piece cost as 1gbe; per-stream cap = loopback"},
    {"usb4", 4400.0, 60e-6, 4400.0, "measured 2026-10-31 host path >= 16 MB (relay, not TCP)"},
    {"oculink", 6500.0, 12e-6, 7000.0, "nominal PCIe 4.0 x4 DMA; per-stream cap = loopback"},
    {"loopback", 19000.0, 12e-6, 7000.0, "measured 2026-11-15 node loopback: 7.0 GB/s one stream at 16 MiB, 18.9 GB/s 8 streams"},
}};

[[nodiscard]] inline const LinkClass* find_link(std::string_view name) {
    for (const auto& l : link_classes)
        if (l.name == name) return &l;
    return nullptr;
}

enum class DataClass : std::uint8_t { bf16, fp32, raw };

struct CodecModel {
    double encode_MBps_1t = 2160;   // bp-v1 one thread (node 2,164 / PC 1,997 MB/s on real BF16 rows)
    double encode_MBps_max = 4800;  // saturation (PC 8 threads 4,795; node 8 threads 5,274)
    double decode_MBps = 11000;     // receiver pool (PC 10,984 / node 17,286 at 8 threads): never the limit below 10 GbE
    double ratio_bf16 = 1.468, ratio_fp32 = 1.171;
    [[nodiscard]] double ratio(DataClass c) const { return c == DataClass::bf16 ? ratio_bf16 : c == DataClass::fp32 ? ratio_fp32 : 1.0; }
    [[nodiscard]] double encode_MBps(unsigned threads) const {
        return std::min(std::max(1U, threads) * encode_MBps_1t, encode_MBps_max);
    }
};

struct PieceLoad {
    double ready_s = 0;  // when the producer hands the piece over (seconds from the start of the prefill)
    double plain_bytes = 0;
    DataClass data = DataClass::bf16;
};

struct Choice {
    Codec codec = Codec::bp_v1;
    unsigned streams = 1;
    unsigned encode_threads = 8;
    unsigned encode_batch = 16;  // SenderOptions::encode_batch
};

// END + manifest check + REPLY after the last byte: measured tails exceeded the wire-only model by 8-11 ms at 1 GbE
// (DS41 32K shape: 0.014 vs 0.006 s streamed, 0.284 vs 0.274 s after the prefill).
inline constexpr double kSessionFinishS = 0.010;

struct Prediction {
    double end_s = 0;          // last byte on the receiver
    double tail_s = 0;         // end_s - last ready
    double wire_bytes = 0, plain_bytes = 0;
    double link_busy_s = 0;    // seconds the wire stage was sending
    double encode_busy_s = 0;  // encoder seconds (wall of the pool)
    [[nodiscard]] double utilisation(double span_s) const { return span_s > 0 ? link_busy_s / span_s : 0; }
};

[[nodiscard]] inline double stream_goodput(const LinkClass& link, unsigned streams) {
    return std::min(link.goodput_MBps, std::max(1U, streams) * link.per_stream_MBps);
}

[[nodiscard]] inline Prediction predict(std::span<const PieceLoad> pieces, const LinkClass& link, const CodecModel& codec, const Choice& c) {
    Prediction p;
    const double g = stream_goodput(link, c.streams) * 1e6;
    double enc_free = 0, wire_free = 0, last_ready = 0;
    for (const auto& pc : pieces) {
        last_ready = std::max(last_ready, pc.ready_s);
        double ready = pc.ready_s, wire = pc.plain_bytes;
        if (c.codec == Codec::bp_v1) {
            // one piece encodes on at most one thread per 1 MiB codec block unless pieces are batched (2026-11-15: a 1 MiB
            // piece ran at 1.30 GB/s on 8 threads, a 16 MiB piece at 4.05 GB/s)
            const double blocks = std::ceil(pc.plain_bytes / double(1 << 20));
            const unsigned busy = c.encode_batch > 1 ? c.encode_threads : static_cast<unsigned>(std::min<double>(c.encode_threads, std::max(1.0, blocks)));
            const double t = pc.plain_bytes / (codec.encode_MBps(busy) * 1e6);
            enc_free = std::max(enc_free, pc.ready_s) + t;
            p.encode_busy_s += t;
            ready = enc_free;
            wire = pc.plain_bytes / codec.ratio(pc.data);
        }
        const double t = link.per_message_s + wire / g;
        wire_free = std::max(wire_free, ready) + t;
        p.link_busy_s += t;
        p.wire_bytes += wire;
        p.plain_bytes += pc.plain_bytes;
    }
    p.end_s = wire_free + kSessionFinishS;
    p.tail_s = p.end_s - last_ready;
    return p;
}

// The cheapest choice within 1 % of the earliest end: identity before bp-v1, fewer streams, fewer threads.
struct AutoFields {
    bool codec = true, streams = true, threads = true, batch = true;
};

[[nodiscard]] inline Choice choose(std::span<const PieceLoad> pieces, const LinkClass& link, const CodecModel& codec,
                                   unsigned max_threads, Choice fixed = {}, AutoFields automatic = {}) {
    if (max_threads > 256) throw std::invalid_argument("handoff: thread limit must be 0..256");
    std::vector<Choice> cands;
    // Enumerate in the advertised cost order, constraining every non-auto field.
    for (const auto kind : {Codec::identity, Codec::bp_v1}) {
        if (!automatic.codec && kind != fixed.codec) continue;
        const auto streams = automatic.streams ? std::vector<unsigned>{1, 2, 4, 8} : std::vector<unsigned>{fixed.streams};
        for (const unsigned s : streams) {
            std::vector<unsigned> threads;
            if (!automatic.threads) threads.push_back(fixed.encode_threads);
            else if (kind == Codec::identity || max_threads == 0) threads.push_back(0);
            else for (unsigned t = 1;;) {
                threads.push_back(t);
                if (t > max_threads / 2) break;
                t *= 2;
            }
            for (const unsigned t : threads) {
                const auto batches = automatic.batch && kind == Codec::bp_v1 && t > 1
                    ? std::vector<unsigned>{1, 16} : std::vector<unsigned>{automatic.batch ? 1U : fixed.encode_batch};
                for (const unsigned b : batches) cands.push_back({kind, s, t, b});
            }
        }
    }
    double best = 1e300;
    for (const auto& c : cands) best = std::min(best, predict(pieces, link, codec, c).end_s);
    for (const auto& c : cands)
        if (predict(pieces, link, codec, c).end_s <= best * 1.01 + 1e-9) return c;
    return cands.front();
}

// ---- the resolver -------------------------------------------------------------------------------------------
struct Request {  // CLI values ("" = not given: the environment, then the family default, decides)
    std::string link, codec, streams, encode_threads, encode_batch;
};

struct Resolved {
    Choice choice;
    std::optional<LinkClass> link;  // owned value, including numeric links; empty = undeclared
    std::vector<std::string> lines;   // "handoff <key> <value> (<origin>)" for --print-effective-config
};

[[nodiscard]] inline const char* codec_name(Codec c) { return c == Codec::bp_v1 ? "bp-v1" : "identity"; }

// `family` = what the family ships today; `pieces` = the request's load for auto (may be empty: a steady 1 GiB of
// BF16 rows is assumed); `max_threads` bounds auto's encoder pool (hardware threads of the sender).
struct Constraints {
    bool auto_streams = true;
    bool encode_batch = true;
};

[[nodiscard]] inline Resolved resolve(const Choice& family, Request req, std::span<const PieceLoad> pieces = {},
                                    unsigned max_threads = 16, Constraints constraints = {}) {
    auto pick = [](std::string& cli, const char* env) -> std::string {
        const char* raw = envcfg::text(env);
        return !cli.empty() ? cli : raw ? std::string(raw) : std::string();
    };
    const std::string link = pick(req.link, "STRIX_HANDOFF_LINK"), codec = pick(req.codec, "STRIX_HANDOFF_CODEC"),
                      streams = pick(req.streams, "STRIX_HANDOFF_STREAMS"), threads = pick(req.encode_threads, "STRIX_HANDOFF_ENCODE_THREADS"),
                      batch = pick(req.encode_batch, "STRIX_HANDOFF_ENCODE_BATCH");
    Resolved r;
    r.choice = family;
    if (!constraints.auto_streams && streams == "auto")
        throw std::invalid_argument("handoff: auto streams are not supported by this adapter (HTTP POSTs wait for replies)");
    if (!constraints.encode_batch && !batch.empty())
        throw std::invalid_argument("handoff: encode_batch is not supported by this adapter");
    if (!link.empty()) {
        if (const auto* named = find_link(link)) r.link = *named;
        if (!r.link) {
            std::size_t used = 0;
            double v = 0;
            try { v = std::stod(link, &used); } catch (...) { used = 0; }
            if (used != link.size() || !(v > 0) || !std::isfinite(v)) throw std::invalid_argument("STRIX_HANDOFF_LINK: unknown link '" + link + "'");
            r.link = LinkClass{"custom", v, 60e-6, std::min(v, 1400.0), "declared MB/s"};
        }
    }
    const bool any_auto = codec == "auto" || streams == "auto" || threads == "auto" || batch == "auto";
    if (any_auto && !r.link) throw std::invalid_argument("handoff auto needs a declared link (--handoff-link / STRIX_HANDOFF_LINK)");
    if (codec == "identity" || codec == "off") r.choice.codec = Codec::identity;
    else if (codec == "bp-v1") r.choice.codec = Codec::bp_v1;
    else if (!codec.empty() && codec != "auto") throw std::invalid_argument("STRIX_HANDOFF_CODEC: expected identity, bp-v1 or auto, got '" + codec + "'");
    auto count = [](const std::string& v, unsigned lo, unsigned hi, const char* what) {
        std::size_t used = 0;
        unsigned long n = 0;
        try { n = std::stoul(v, &used); } catch (...) { used = 0; }
        if (used != v.size() || n < lo || n > hi) throw std::invalid_argument(std::string(what) + ": expected " + std::to_string(lo) + ".." + std::to_string(hi) + " or auto, got '" + v + "'");
        return static_cast<unsigned>(n);
    };
    if (!streams.empty() && streams != "auto") r.choice.streams = count(streams, 1, kMaxStreams, "STRIX_HANDOFF_STREAMS");
    if (!threads.empty() && threads != "auto") r.choice.encode_threads = count(threads, 0, 256, "STRIX_HANDOFF_ENCODE_THREADS");
    if (!batch.empty() && batch != "auto") r.choice.encode_batch = count(batch, 1, 256, "STRIX_HANDOFF_ENCODE_BATCH");
    std::string why;
    if (any_auto) {
        std::vector<PieceLoad> steady;
        if (pieces.empty()) {
            for (int i = 0; i < 64; ++i) steady.push_back({0.0, 16.0 * (1 << 20), DataClass::bf16});
            pieces = steady;
        }
        r.choice = choose(pieces, *r.link, CodecModel{}, max_threads, r.choice,
                          {codec == "auto", streams == "auto", threads == "auto", batch == "auto"});
        const auto pr = predict(pieces, *r.link, CodecModel{}, r.choice);
        char buf[160];
        std::snprintf(buf, sizeof buf, "auto: %s predicts %.3f s, %.0f MB on the wire", std::string(r.link->name).c_str(), pr.end_s, pr.wire_bytes / 1e6);
        why = buf;
    }
    auto origin = [&](const std::string& v, const std::string& cli) { return v.empty() ? std::string("default") : v == "auto" ? why : !cli.empty() ? "cli" : "env"; };
    r.lines.push_back("handoff link " + (r.link ? std::string(r.link->name) : std::string("undeclared")) + (link.empty() ? " (default)" : " (" + origin(link, req.link) + ")"));
    r.lines.push_back(std::string("handoff codec ") + codec_name(r.choice.codec) + " (" + origin(codec, req.codec) + ")");
    r.lines.push_back("handoff streams " + std::to_string(r.choice.streams) + " (" + origin(streams, req.streams) + ")");
    r.lines.push_back("handoff encode_threads " + std::to_string(r.choice.encode_threads) + " (" + origin(threads, req.encode_threads) + ")");
    r.lines.push_back("handoff encode_batch " + std::to_string(r.choice.encode_batch) + " (" + origin(batch, req.encode_batch) + ")");
    return r;
}

[[nodiscard]] inline SenderOptions sender_options(const Choice& c, SenderOptions base = {}) {
    base.codec = c.codec;
    base.streams = c.streams;
    base.encode_threads = c.encode_threads;
    base.encode_batch = c.encode_batch;
    return base;
}

}  // namespace strix::handoff
