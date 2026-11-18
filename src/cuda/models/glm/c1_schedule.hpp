#pragma once
// C1 grouped MoE schedule (cutlass_path.inl), host and device. Experts run in groups of
// 16 (double-buffered decode windows, parity g % 2, reuse guarded by retired[g - 2]).
// With compaction (decode's skip_empty_expert_groups) the device lists the experts that
// hold routed rows first, ascending, then the empty ones (a permutation of all experts),
// and the host launches only the groups that can hold a routed expert: at most
// rows * routes experts carry rows, so ceil(min(experts, rows * routes) / 16) groups,
// numbered 0.. as before (parity and the guard alternate over executed groups). Each
// expert's decode, GEMM problems and epilogue are its own; the merge sums every route in
// a fixed per-row order. Which group slot holds an expert changes no value.
#include <cstdint>

#if defined(__CUDACC__)
#define C1_HD __host__ __device__
#else
#define C1_HD
#endif

namespace strix::glm::rtxpf {
inline constexpr unsigned c1_group_experts = 16;

C1_HD inline unsigned c1_groups(unsigned experts, unsigned rows, unsigned routes, bool compact) {
    const unsigned span = compact && rows * routes < experts ? rows * routes : experts;
    return (span + c1_group_experts - 1) / c1_group_experts;
}
// act = the experts with counts > 0 (ascending), then those with 0 (ascending).
C1_HD inline void c1_compact(const std::uint32_t* counts, std::uint32_t* act, unsigned experts) {
    unsigned k = 0;
    for (unsigned j = 0; j < experts; ++j)
        if (counts[j]) act[k++] = j;
    for (unsigned j = 0; j < experts; ++j)
        if (!counts[j]) act[k++] = j;
}
// The expert of group slot s (no list: the slot itself).
C1_HD inline unsigned c1_expert(const std::uint32_t* act, unsigned s) { return act ? act[s] : s; }
}  // namespace strix::glm::rtxpf
