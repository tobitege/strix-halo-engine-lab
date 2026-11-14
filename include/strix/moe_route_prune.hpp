#pragma once
// RoutePrune (FAST, default off): host declaration of the shared route-pruning policy (device side:
// src/moe_core/route_prune.hpp). Governed by the profile table row "moe_route_prune"
// (STRIX_QPF2K_MOE_ROUTE_PRUNE / --prefill-moe-route-prune). Spec grammar:
//   ""  | "off" | "0"      no pruning (the default; identity bytes unchanged)
//   rel<f>[n]              keep assignments whose weight is >= f * the row's weight sum      (0 < f < 1)
//   topp<f>[n]             keep the best assignments until their mass reaches f * the sum     (0 < f < 1)
//   cap<k>[n]              keep the k best assignments of each row                           (1 <= k <= 31)
// A trailing "n" renormalises the kept weights to the row's original sum. The best assignment is always kept.
#include "strix/env_flags.hpp"

#include <charconv>
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace strix::moe_core {

struct RoutePrune {
    enum Mode : unsigned { off = 0U, relative = 1U, top_p = 2U, cap = 3U };
    unsigned mode = off;
    float value = 0.0F;
    unsigned renorm = 0U;
    constexpr bool active() const { return mode != off; }
};

inline RoutePrune parse_route_prune(std::string_view text) {
    if (text.empty() || text == "off" || text == "0") return {};
    RoutePrune p{};
    const auto take = [&](std::string_view head, unsigned mode) {
        if (text.substr(0, head.size()) != head) return false;
        text.remove_prefix(head.size());
        p.mode = mode;
        return true;
    };
    if (!take("rel", RoutePrune::relative) && !take("topp", RoutePrune::top_p) && !take("cap", RoutePrune::cap))
        throw std::invalid_argument("MoE route prune spec must be off, rel<f>[n], topp<f>[n] or cap<k>[n]");
    if (!text.empty() && text.back() == 'n') { p.renorm = 1U; text.remove_suffix(1); }
    // Decimal, locale-independent input only: no whitespace, controls, hex or nonfinite spellings.
    if (text.empty() || text.find_first_not_of("0123456789.eE+-") != std::string_view::npos)
        throw std::invalid_argument("MoE route prune spec has an invalid value");
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), p.value);
    const bool whole = parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
    const bool range = p.mode == RoutePrune::cap ? p.value >= 1.0F && p.value <= 31.0F && p.value == static_cast<float>(static_cast<unsigned>(p.value))
                                                 : p.value > 0.0F && p.value < 1.0F;
    if (!whole || !range)
        throw std::invalid_argument("MoE route prune spec has an invalid value");
    return p;
}

// Canonical, JSON-safe identity tag, derived from the actual float policy (not input spelling).
inline std::string route_prune_tag(RoutePrune p) {
    if (!p.active()) return {};
    char number[64];
    const auto result = std::to_chars(number, number + sizeof(number), p.value);
    if (result.ec != std::errc{}) throw std::runtime_error("cannot format MoE route prune value");
    std::string tag = p.mode == RoutePrune::cap ? "cap" : p.mode == RoutePrune::relative ? "rel" : "topp";
    tag.append(number, result.ptr);
    if (p.renorm) tag += 'n';
    return tag;
}
inline std::string route_prune_tag(std::string_view text) { return route_prune_tag(parse_route_prune(text)); }

// Process-wide request (restart to change it), as the other prefill drafts.
inline const std::string& route_prune_active_tag() {
    static const std::string tag = [] {
        const char* value = strix::envcfg::text("STRIX_QPF2K_MOE_ROUTE_PRUNE");
        return route_prune_tag(value ? value : "");
    }();
    return tag;
}
inline RoutePrune route_prune_active() { return parse_route_prune(route_prune_active_tag()); }

// Host-only diagnostic sink. A failed record poisons the sink permanently; never append
// behind a partial record. The lock covers the whole record and flush for all callers
// sharing this writer. Separate processes must use separate census destinations.
struct RouteCensusFileIo {
    static std::FILE* open(const char* path) { return std::fopen(path, "ab"); }
    static std::size_t write(const void* data, std::size_t size, std::size_t count, std::FILE* file) {
        return std::fwrite(data, size, count, file);
    }
    static int flush(std::FILE* file) { return std::fflush(file); }
    static void close(std::FILE* file) { std::fclose(file); }
};
template<class Io = RouteCensusFileIo>
class RouteCensusWriter {
public:
    explicit RouteCensusWriter(std::string path) : path_(std::move(path)) {}
    ~RouteCensusWriter() { if (file_) Io::close(file_); }
    void append(std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts,
                const float* weights, const std::uint32_t* ids) {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (failed_) throw std::runtime_error("route census '" + path_ + "': disabled after record failure");
        if (!file_) file_ = Io::open(path_.c_str());
        if (!file_) fail("cannot open file");
        const std::uint32_t head[4]{0x52434e53U, rows, top_k, experts};
        const auto count = std::size_t{rows} * top_k;
        if (Io::write(head, sizeof(head[0]), 4U, file_) != 4U ||
            Io::write(weights, sizeof(float), count, file_) != count ||
            Io::write(ids, sizeof(std::uint32_t), count, file_) != count)
            fail("incomplete record write");
        if (Io::flush(file_) != 0) fail("record flush failed");
    }
private:
    [[noreturn]] void fail(const char* reason) {
        failed_ = true;
        throw std::runtime_error("route census '" + path_ + "': " + reason);
    }
    std::string path_;
    std::FILE* file_ = nullptr;
    bool failed_ = false;
    std::mutex mutex_;
};

}  // namespace strix::moe_core
