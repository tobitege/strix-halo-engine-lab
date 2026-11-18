#pragma once
// Wave-aligned step rows: choose how many rows one prefill step covers so that a kernel whose grid is
// ceil(rows / tile_rows) * ctas_per_tile CTAs fills whole waves of sms * ctas_per_sm resident CTAs. With a long prefix
// every CTA of a causal attention step does nearly the same work, so a partial last wave idles the rest of the GPU for a
// full CTA duration (27B flash at 4096 rows on 188 SMs: 768 CTAs = 4.09 waves -> 5, 81.7 %; 3968 rows: 744 = 3.96 -> 4).
// Shared host logic, model- and backend-agnostic (any tiled attention / GEMM grid on CUDA SMs or HIP CUs). Step rows
// only partition the prompt, so row-independent kernels keep their bits (27B: logits sha equal at 3968 / 4096 / 6016).
// Bound: CUDA Qwen3.8-27B prefill (qwen35 engine, prefill_wave_min_tokens). Same pattern: CUDA Flash-Next QSA grids,
// HIP gfx1151 flash prefill (40 CUs) - follow-ups named in wave4/next1112/RTXLONG-RESULT.md.
#include <cstdint>
#include <stdexcept>

namespace strix::wave_fit {

struct Grid {
    std::uint32_t tile_rows = 0, ctas_per_tile = 0, sms = 0, ctas_per_sm = 0;
};

[[nodiscard]] inline double efficiency(const Grid& g, std::uint32_t rows) {
    if (g.tile_rows == 0 || g.ctas_per_tile == 0 || g.sms == 0 || g.ctas_per_sm == 0 || rows == 0)
        throw std::invalid_argument("wave_fit: empty grid");
    const std::uint64_t ctas = std::uint64_t{(rows + g.tile_rows - 1) / g.tile_rows} * g.ctas_per_tile;
    const std::uint64_t wave = std::uint64_t{g.sms} * g.ctas_per_sm;
    const std::uint64_t waves = (ctas + wave - 1) / wave;
    return static_cast<double>(ctas) / static_cast<double>(waves * wave);
}

// The largest multiple of `granule` in [min_rows, max_rows] whose wave efficiency is >= min_eff; when none reaches it,
// the most efficient one (ties: larger). max_rows itself is returned when it already reaches min_eff.
[[nodiscard]] inline std::uint32_t rows(const Grid& g, std::uint32_t max_rows, std::uint32_t min_rows, std::uint32_t granule,
                                        double min_eff = 0.97) {
    if (granule == 0 || min_rows == 0 || min_rows > max_rows) throw std::invalid_argument("wave_fit: bad range");
    std::uint32_t best = max_rows;
    double best_eff = efficiency(g, max_rows);
    if (best_eff >= min_eff) return max_rows;
    for (std::uint32_t r = (max_rows - 1) / granule * granule; r >= min_rows && r > 0; r -= granule) {
        const double e = efficiency(g, r);
        if (e >= min_eff) return r;
        if (e > best_eff) { best_eff = e; best = r; }
        if (r < granule) break;
    }
    return best;
}

}  // namespace strix::wave_fit
