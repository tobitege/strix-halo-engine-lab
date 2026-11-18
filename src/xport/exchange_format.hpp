#pragma once
// TP2 exchange format (shared by every pair engine): the element format of the partial rows that cross the link in
// the TP2 all-reduce exchanges of prefill (the partial hidden after attention and after the FFN / MoE).
//
//   f32  (default, every engine): the partials are moved as IEEE-754 binary32, bytes never changed. Bitwise identical
//        to the build without this trait.
//   f16  (FAST only): each rank rounds its own FP32 partial to binary16 (round to nearest even) on the GPU before the
//        send, and BOTH ranks sum the two binary16 partials widened back to binary32 in the same fixed rank order
//        (h + (f32(p0h) + f32(p1h))). The two ranks therefore keep bitwise identical hidden states; the logits change
//        (bit-changing, named FAST policy, quality gate). Wire, TX / RX slot and consumer read bytes are halved.
//
// Only an explicit FAST opt-in may select f16 (resolve() refuses it where the engine forbids FAST: fail closed); no
// profile selects it by default. Both ranks must
// resolve the same value: the value is a bit-changing row of the engine's product profile (exchange_format_lever()),
// so the existing rank agreement of the profile fingerprint compares it at load, and the exchange headers carry the
// message size, so a disagreement that slipped through fails at the first exchange.
// Engines bind it with exchange_format_lever() (profile row), resolve() (one resolver) and the device helpers of
// xport/exchange_format_hip.hpp (pack kernel body, widening loads).
#include "strix/engine_profile.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::xport {

enum class ExchangeFormat : std::uint32_t { f32 = 0, f16 = 1 };

inline constexpr const char* kExchangeFormatEnv = "STRIX_TP2_EXCHANGE_FMT";

inline constexpr const char* exchange_format_name(ExchangeFormat f) { return f == ExchangeFormat::f16 ? "f16" : "f32"; }
inline constexpr std::size_t exchange_elem_bytes(ExchangeFormat f) { return f == ExchangeFormat::f16 ? 2 : 4; }
inline constexpr std::size_t exchange_payload_bytes(ExchangeFormat f, std::size_t rows, std::size_t width) {
    return rows * width * exchange_elem_bytes(f);
}

// "f32" | "f16" (null / empty: f32). Anything else throws (bf16 is not offered: not measured).
inline ExchangeFormat parse_exchange_format(const char* raw) {
    if (raw == nullptr || *raw == 0) return ExchangeFormat::f32;
    const std::string_view v(raw);
    if (v == "f32") return ExchangeFormat::f32;
    if (v == "f16") return ExchangeFormat::f16;
    throw std::invalid_argument(std::string(kExchangeFormatEnv) + " must be f32 or f16 (got '" + std::string(v) + "')");
}

struct ExchangeFormatChoice {
    ExchangeFormat format = ExchangeFormat::f32;
    std::string reason;  // effective-config / receipt text
};

// The one resolution of an engine: `raw` = the engine's lookup of kExchangeFormatEnv, `fast_policy` = the engine's
// named FAST prefill policy is effective, `policy` = its name (reason text). f16 without the FAST policy throws.
inline ExchangeFormatChoice resolve_exchange_format(const char* raw, bool fast_policy, std::string_view policy) {
    ExchangeFormatChoice c;
    c.format = parse_exchange_format(raw);
    if (c.format == ExchangeFormat::f16 && !fast_policy)
        throw std::invalid_argument(std::string(kExchangeFormatEnv) + "=f16 changes logits: it needs the FAST prefill policy (" +
                                    std::string(policy) + ")");
    c.reason = c.format == ExchangeFormat::f32 ? (raw == nullptr || *raw == 0 ? "default (exact FP32 partials)" : "explicit f32")
                                               : "explicit f16 under " + std::string(policy) + " (bit-changing, gate required)";
    return c;
}

// The product-profile row an engine adds to its table: bit-changing, builtin f32, left out of the fingerprint at f32
// (the identity of every engine is unchanged while the row is at its builtin value).
inline constexpr profile::Lever exchange_format_lever() {
    return {"tp2_exchange_fmt", kExchangeFormatEnv, "", {"", "", "", ""}, "f32", profile::Kind::bit_changing,
            {true, false, true, false}, {true, true, true, false}, "", "f32",
            "TP2 prefill exchange element format: f32 exact; f16 FAST only (rounded partials, fixed rank-order sum)", false, {}, true};
}

// One receipt line per prefill: format, messages, bytes per message, and the mean wire ms per message of the
// exchange module (host progress thread: header to trailer), measured by the caller.
inline void print_exchange_receipt(unsigned rank, ExchangeFormat f, std::uint64_t messages, std::uint64_t bytes, double wire_s) {
    std::printf("rank %u: tp2 exchange fmt %s: %llu messages, %.1f MB each, wire %.2f ms each (%.3f s total, %.2f GB/s)\n", rank,
                exchange_format_name(f), static_cast<unsigned long long>(messages),
                messages ? static_cast<double>(bytes) / static_cast<double>(messages) / 1e6 : 0.0,
                messages ? wire_s * 1e3 / static_cast<double>(messages) : 0.0, wire_s,
                wire_s > 0 ? static_cast<double>(bytes) / wire_s / 1e9 : 0.0);
}

}  // namespace strix::xport
