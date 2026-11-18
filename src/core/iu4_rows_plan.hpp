#pragma once

#include <stdexcept>
#include <atomic>
#include <cstdint>
#include <string_view>
#include "strix/env_flags.hpp"

// Shared small-row IU4 staging plan. The automatic table is measured on gfx1151;
// force measures every legal group for qualification.
namespace strix::core::iu4_rows {

struct Iu4Plan { unsigned S, GC; };
enum class RowReduction { legacy, canonical };
inline RowReduction parse_row_reduction(std::string_view value) {
    if (value.empty() || value == "legacy") return RowReduction::legacy;
    if (value == "canonical") return RowReduction::canonical;
    throw std::invalid_argument("STRIX_IU4_ROW_REDUCTION must be legacy or canonical");
}
constexpr std::string_view row_reduction_name(RowReduction policy) {
    return policy == RowReduction::canonical ? "canonical" : "legacy";
}
constexpr std::string_view row_reduction_contract(RowReduction policy) {
    return policy == RowReduction::canonical ? "iu4-g64g96-gc32-v1" : "iu4-row-buckets-v1";
}
struct RowReductionResolution {
    RowReduction requested = RowReduction::legacy;
    RowReduction effective = RowReduction::legacy;
    bool active = false;
};
inline RowReductionResolution resolve_row_reduction(bool active, std::string_view value) {
    const auto requested = parse_row_reduction(value);
    return {requested, active ? requested : RowReduction::legacy, active};
}
constexpr bool supports_groups(unsigned groups) {
    return groups == 32U || groups == 40U || groups == 64U || groups == 96U;
}
enum class SingleDecodePolicy { split, automatic, force };
inline SingleDecodePolicy parse_single_decode_policy(std::string_view value) {
    if (value == "0") return SingleDecodePolicy::split;
    if (value.empty() || value == "1" || value == "auto") return SingleDecodePolicy::automatic;
    if (value == "force") return SingleDecodePolicy::force;
    throw std::invalid_argument("STRIX_IU4_R8_SINGLE_DECODE must be 0, 1, auto, or force");
}
constexpr std::string_view single_decode_policy_name(SingleDecodePolicy policy) {
    return policy == SingleDecodePolicy::split ? "split" : policy == SingleDecodePolicy::force ? "force" : "auto";
}
struct SingleDecodeResolution {
    SingleDecodePolicy requested = SingleDecodePolicy::automatic;
    SingleDecodePolicy effective = SingleDecodePolicy::split;
    std::string_view reason = "native-weights";
};
inline SingleDecodeResolution resolve_single_decode_policy(bool active, std::string_view value) {
    const auto requested = parse_single_decode_policy(value);
    const auto effective = active ? requested : SingleDecodePolicy::split;
    return {requested, effective, !active ? "native-weights" : effective == SingleDecodePolicy::split ? "explicit-off" :
        effective == SingleDecodePolicy::force ? "diagnostic-force" : "measured-g96-r7-r8-n2560"};
}

struct ExecutionPolicy {
    SingleDecodePolicy single_decode = SingleDecodePolicy::automatic;
    RowReduction reduction = RowReduction::legacy;
};
inline std::atomic<std::uint64_t> single_decode_launches{0};
namespace detail {
inline ExecutionPolicy& execution_policy_storage(const ExecutionPolicy* resolved = nullptr) {
    // Standalone callers resolve once; engines install their resolved startup
    // policy before launching any workers. No environment bookkeeping at dispatch.
    static ExecutionPolicy policy = [resolved] {
        if (resolved) return *resolved;
        const char* single = envcfg::text("STRIX_IU4_R8_SINGLE_DECODE");
        const char* reduction = envcfg::text("STRIX_IU4_ROW_REDUCTION");
        return ExecutionPolicy{parse_single_decode_policy(single ? single : ""),
                               parse_row_reduction(reduction ? reduction : "")};
    }();
    return policy;
}
}  // namespace detail
inline const ExecutionPolicy& execution_policy() { return detail::execution_policy_storage(); }
// Startup only, before workers exist. Diagnostics may reinstall only between
// synchronized runs. The plan is immutable for the lifetime of an execution.
inline void configure_execution_policy(ExecutionPolicy policy) {
    detail::execution_policy_storage(&policy) = policy;
}
constexpr Iu4Plan iu4_plan(unsigned rows, unsigned groups, RowReduction reduction = RowReduction::legacy) {
    if (groups == 32U) return {4U, 32U};
    if (groups == 40U) return {4U, 40U};
    if (reduction == RowReduction::canonical) return {8U, 32U};
    if (groups == 64U) return rows >= 6U ? Iu4Plan{8U, 32U} : Iu4Plan{8U, 64U};
    return rows >= 4U ? Iu4Plan{8U, 32U} : Iu4Plan{8U, 96U};
}
constexpr bool staged_single_decode_admitted(unsigned rows, unsigned groups) {
    if (rows < 7U || rows > 8U) return false;
    const auto p = iu4_plan(rows, groups);
    constexpr unsigned lds_bytes = 64U * 1024U;
    const unsigned old_bytes = rows * p.GC * 128U;
    const unsigned new_bytes = old_bytes + rows * 32U * p.S * 8U;
    return old_bytes <= lds_bytes && new_bytes <= lds_bytes &&
           lds_bytes / old_bytes == lds_bytes / new_bytes;
}
// Cold and warm, direct and producer-A8 activations all win at this shape;
// keeping the LDS block count alone selected the losing G40 regime instead.
constexpr bool measured_single_decode_admitted(unsigned rows, unsigned groups, unsigned outputs) {
    return (rows == 7U || rows == 8U) && groups == 96U && outputs == 2560U;
}
constexpr bool measured_single_decode_device(std::string_view arch, unsigned wave_size) {
    return wave_size == 32U && (arch == "gfx1151" || arch.starts_with("gfx1151:"));
}
// force is diagnostic: measure every legal group before qualifying a crossover.
constexpr bool single_decode_applies(unsigned rows, unsigned groups, SingleDecodePolicy policy, unsigned outputs,
                                    bool device_qualified) {
    if (rows < 7U || rows > 8U || !supports_groups(groups)) return false;
    return policy == SingleDecodePolicy::force ||
           (policy == SingleDecodePolicy::automatic && device_qualified && measured_single_decode_admitted(rows, groups, outputs));
}
constexpr unsigned packed_loads(unsigned rows, unsigned groups, SingleDecodePolicy policy, unsigned outputs, bool device_qualified) {
    return rows >= 7U && !single_decode_applies(rows, groups, policy, outputs, device_qualified) ? 2U : 1U;
}

}  // namespace strix::core::iu4_rows
