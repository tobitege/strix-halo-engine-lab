#pragma once

#include "strix/moe_route_bf16.hpp"

#include <cstdint>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::cuda::qwen_route {
// Experimental storage boundary only. No READY entry or production default.
inline constexpr std::uint64_t elements_per_row = 10ULL * 2560ULL;
inline constexpr std::uint32_t max_csr_rows = (65535U * 32U) / 10U;

struct Policy {
    bool requested = false;
    bool enabled = false;
    std::string_view reason = "off";
};
inline Policy resolve(std::optional<bool> option, const char* request,
                      const char* baseline, const char* defaults, const char* exact) {
    const bool wanted = option.value_or(request && request[0] && request[0] != '0');
    if (!wanted) return {};
    if (baseline && std::string_view(baseline) == "1") return {true, false, "baseline"};
    if (defaults && defaults[0] == '0') return {true, false, "qpf2k-defaults-off"};
    if (exact && std::string_view(exact) == "1") return {true, false, "exact-master"};
    return {true, true, "explicit-unqualified-cuda-a8-storage"};
}
inline Policy environment(std::optional<bool> option = {}) {
    return resolve(option, std::getenv("STRIX_QPF2K_MOE_ROUTE_BF16"),
                   std::getenv("STRIX_BASELINE"), std::getenv("STRIX_QPF2K_DEFAULTS"),
                   std::getenv("STRIX_PREFILL_EXACT_FP16"));
}

inline constexpr std::uint64_t hash_part(std::uint64_t h, std::string_view text) {
    // NUL-framed, fixed-order names: no concatenation ambiguity.
    for (const char c : text) { h ^= static_cast<unsigned char>(c); h *= 0x100000001b3ULL; }
    h ^= 0U; h *= 0x100000001b3ULL;
    return h;
}
struct Identity {
    std::string_view prefill, decode, storage;
    std::uint64_t hash;
};
inline constexpr Identity identity(bool prefill, bool decode) {
    const auto p = prefill ? "cuda-qwen-a8-prefill+moeroute-bf16-rne-v1" : "cuda-qwen-a8-prefill-v1";
    const auto d = decode ? "cuda-qwen-a8-decode+moeroute-bf16-rne-v1" : "cuda-qwen-a8-decode-v1";
    const auto s = prefill == decode
        ? (prefill ? "cuda-qwen-a8-rawdot-bf16-rne-v1" : "cuda-qwen-a8-rawdot-f32-v1")
        : "cuda-qwen-a8-rawdot-phase-mixed-v1";
    auto h = hash_part(0xcbf29ce484222325ULL, "cuda-qwen-target-pair-v1");
    h = hash_part(hash_part(hash_part(h, p), d), s);
    return {p, d, s, h};
}
enum class Composition {
    graphs, cohort, multisegment, mtp_sidecar, prefix_cache,
    untagged_state_import, node_handoff, overlap, other_weight_pack
};
inline constexpr std::string_view composition_name(Composition c) {
    switch (c) {
    case Composition::graphs: return "graphs";
    case Composition::cohort: return "cohort";
    case Composition::multisegment: return "multisegment";
    case Composition::mtp_sidecar: return "MTP-sidecar";
    case Composition::prefix_cache: return "prefix-cache";
    case Composition::untagged_state_import: return "untagged-state-import";
    case Composition::node_handoff: return "node-handoff/export-state";
    case Composition::overlap: return "prefill-overlap";
    case Composition::other_weight_pack: return "non-A8-weight-pack";
    }
    return "unknown-composition";
}
inline void require_composition(bool bf16, Composition c) {
    if (bf16) throw std::invalid_argument("MoE route BF16 candidate declined: " +
        std::string(composition_name(c)) + " lacks same-target qualification");
}

inline constexpr std::uint64_t row_bytes(bool bf16) { return elements_per_row * (bf16 ? 2U : 4U); }
inline std::uint64_t checked_extent(std::uint64_t rows, std::uint64_t stride, std::uint64_t width) {
    if (rows == 0 || stride == 0 || width == 0 || stride < width) throw std::invalid_argument("MoE rawdot rows/stride");
    if (rows - 1U > (std::numeric_limits<std::uint64_t>::max() - width) / stride)
        throw std::overflow_error("MoE rawdot extent overflow");
    return (rows - 1U) * stride + width;
}
struct Layout {
    std::uint64_t stride, bytes;
};
inline Layout layout(std::uint32_t rows, bool bf16) {
    if (rows == 0 || rows > max_csr_rows) throw std::invalid_argument("MoE rawdot CSR rows");
    const auto stride = row_bytes(bf16);
    return {stride, checked_extent(rows, stride, stride)};
}
inline void validate_view(std::uintptr_t base, std::uint64_t stride, std::uint64_t capacity,
                          std::uint32_t rows, bool bf16) {
    if (rows == 0 || rows > max_csr_rows) throw std::invalid_argument("MoE rawdot CSR rows");
    if (!base || base % 16U || stride % 16U) throw std::invalid_argument("MoE rawdot alignment");
    const auto bytes = checked_extent(rows, stride, row_bytes(bf16));
    if (bytes > capacity) throw std::invalid_argument("MoE rawdot capacity");
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - base)
        throw std::overflow_error("MoE rawdot address overflow");
}

inline std::uint64_t new_engine_binding() {
    static std::atomic<std::uint64_t> next{1};
    auto value = next.load();
    do {
        if (value == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("canonical engine binding exhausted");
    } while (!next.compare_exchange_weak(value, value + 1U));
    return value;
}

inline void validate_snapshot(std::uint64_t target, std::uint64_t expected_target,
                              std::uint64_t binding, std::uint64_t expected_binding,
                              std::uint64_t length, std::uint64_t max_positions,
                              std::uint64_t rows, std::uint64_t row_width,
                              std::uint64_t fixed, std::uint64_t expected_fixed,
                              std::uint64_t held, std::uint64_t expected_held) {
    if (target != expected_target) throw std::invalid_argument("canonical snapshot target identity mismatch");
    if (binding != expected_binding) throw std::invalid_argument("canonical snapshot engine binding mismatch");
    if (!length || length >= max_positions) throw std::invalid_argument("canonical snapshot length");
    const auto bytes = checked_extent(length, row_width, row_width);
    if (rows != bytes || fixed != expected_fixed || held != expected_held)
        throw std::invalid_argument("canonical snapshot payload extent");
}
} // namespace strix::cuda::qwen_route
