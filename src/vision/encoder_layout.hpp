#pragma once
// Buffer geometry of the HIP vision encoder, shared by plan() (host) and the runner so the reported bytes are the
// allocated bytes. Host header: no HIP types.
#include "strix/vision/encoder.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace strix::vision::layout {

struct ProjectorGeometry {
    unsigned rows = 0, mh = 0, mw = 0, tokens = 0, in = 0;
};
ProjectorGeometry merge_geometry(unsigned grid_h, unsigned grid_w, unsigned ratio);
ProjectorGeometry projector_geometry(const EncoderOutput&, unsigned grid_h, unsigned grid_w, const MergeProjector&,
                                     const float* rows, bool resident = false);

inline constexpr unsigned kGemmIn = 128;  // f16_dense_tall: in % 128 == 0
inline constexpr unsigned kGemmOut = 16;  // 16 outputs per column tile
inline constexpr std::size_t kAlign = 256;

constexpr unsigned round_up(unsigned v, unsigned m) { return (v + m - 1U) / m * m; }
constexpr std::size_t aligned(std::size_t bytes) { return (bytes + kAlign - 1U) / kAlign * kAlign; }
// Head dim the dense attention core runs (0 = unsupported): 72 pads to 80 with zero components.
constexpr unsigned head_pad(unsigned d) { return d == 64U ? 64U : d == 72U || d == 80U ? 80U : 0U; }
// Logical key positions backed for one tile of n rows (mask::bidir_capacity).
constexpr unsigned key_slots(unsigned n) { return (2U * n + 15U) & ~15U; }
constexpr std::size_t panel_bytes(std::size_t out, std::size_t in) {
    return aligned(std::size_t{round_up(static_cast<unsigned>(out), kGemmOut)} * round_up(static_cast<unsigned>(in), kGemmIn) * 2U) +
           aligned(std::size_t{round_up(static_cast<unsigned>(out), kGemmOut)} * 4U);
}

struct Geometry {
    unsigned rows = 0, slots = 0;
    unsigned w = 0, wp = 0, w16 = 0, heads = 0, d = 0, dp = 0, hw = 0, q16 = 0, p = 0, pp = 0;
    unsigned m1 = 0, m116 = 0, m = 0, mp = 0, chunk = 0;
    bool compact = false, split = false, learned = false, rope = false;
    unsigned attn_terms = 1;  // EncoderOptions::attention_terms
    // bytes
    std::size_t patches = 0, pos = 0, trig = 0, x = 0, nrm = 0, qkv = 0, q = 0, ao = 0, cmp = 0, o = 0, k = 0, vt = 0;
    std::size_t klo = 0, vtlo = 0;  // two-term attention: lo terms of k, vt
    std::size_t lo = 0, ylo = 0, mid = 0, act = 0, o2 = 0, mlo = 0, mylo = 0;
    std::size_t fixed() const { return patches + pos + 2U * trig + x + nrm + qkv + q + ao + cmp + o + k + vt + klo + vtlo + lo + ylo; }
    std::size_t mlp() const { return mid + act + o2 + mlo + mylo; }
};

// `slots` = sum of key_slots(n) over the attention tiles.
inline Geometry geometry(const EncoderConfig& c, unsigned rows, unsigned slots, const EncoderOptions& opt) {
    Geometry g;
    g.rows = rows;
    g.slots = slots;
    g.w = c.width;
    g.wp = round_up(c.width, kGemmIn);
    g.w16 = round_up(c.width, kGemmOut);
    g.heads = c.heads;
    g.d = c.head_dim();
    g.dp = head_pad(g.d);
    g.hw = g.heads * g.dp;
    g.q16 = round_up(3U * c.width, kGemmOut);
    g.p = c.patch_dim();
    g.pp = round_up(g.p, kGemmIn);
    g.m1 = c.mlp_out();
    g.m116 = round_up(g.m1, kGemmOut);
    g.m = c.mlp_width;
    g.mp = round_up(g.m, kGemmIn);
    g.compact = g.dp != g.d || g.hw != g.wp;  // attention output is not already a [rows][wp] GEMM input
    g.split = opt.split_activations;
    g.learned = c.has_learned();
    g.rope = c.has_rope();
    const std::size_t n = rows;
    g.patches = aligned(n * g.pp * 4U);
    g.pos = aligned(n * g.w * 4U);  // also holds an adapter addend when the config has no learned table
    g.trig = g.rope ? aligned(n * (g.d / 2U) * 4U) : 0U;
    g.x = g.nrm = aligned(n * g.wp * 4U);
    g.qkv = aligned(n * g.q16 * 4U);
    g.q = g.ao = aligned(n * g.hw * 4U);
    g.cmp = g.compact ? aligned(n * g.wp * 4U) : 0U;
    g.o = aligned(n * g.w16 * 4U);
    g.k = g.vt = aligned(std::size_t{slots} * g.hw * 2U);
    g.attn_terms = opt.attention_terms;
    if (g.attn_terms == 2U) g.klo = g.vtlo = g.k;
    if (g.split) {
        g.lo = aligned(n * std::max(g.pp, g.wp) * 4U);
        g.ylo = aligned(n * std::max(g.q16, g.w16) * 4U);
    }
    const std::size_t per_row = (std::size_t{g.m116} + g.mp + g.w16 + (g.split ? std::max(g.wp, g.mp) + std::max(g.m116, g.w16) : 0U)) * 4U;
    std::size_t cap = per_row ? opt.mlp_workspace_bytes / per_row : n;
    cap = std::max<std::size_t>(cap / 128U * 128U, 128U);
    // Balanced chunks (scheduling only: every row's sums are the same in any chunk): the fewest chunks under the cap,
    // all of one size rounded up to the GEMM's 128-row CTA, so no chunk is a small tail that leaves the GPU idle
    // (1550 rows under a 1280-row cap ran a 270-row chunk at a third of the big chunk's rate).
    const std::size_t parts = (n + cap - 1U) / cap;
    const std::size_t even = parts ? (n + parts - 1U) / parts : 0U;
    const std::size_t execution_chunk = (even + 127U) / 128U * 128U;  // <= cap (a multiple of 128)
    g.chunk = static_cast<unsigned>(std::min<std::size_t>(execution_chunk, n));
    // Allocation must cover every smaller admitted image, even when balancing reduces this execution's chunk.
    const std::size_t chunk = std::min(cap, (n + 127U) / 128U * 128U);
    g.mid = aligned(chunk * g.m116 * 4U);
    g.act = aligned(chunk * g.mp * 4U);
    g.o2 = aligned(chunk * g.w16 * 4U);
    if (g.split) {
        g.mlo = aligned(chunk * std::max(g.wp, g.mp) * 4U);
        g.mylo = aligned(chunk * std::max(g.m116, g.w16) * 4U);
    }
    return g;
}

inline std::size_t weight_bytes(const EncoderConfig& c) {
    std::size_t b = panel_bytes(c.width, c.patch_dim());
    b += std::size_t{c.layers} * (panel_bytes(3U * c.width, c.width) + panel_bytes(c.width, c.width) + panel_bytes(c.mlp_out(), c.width) +
                                  panel_bytes(c.width, c.mlp_width) + 4U * aligned(std::size_t{c.width} * 4U));
    if (c.final_norm) b += 2U * aligned(std::size_t{c.width} * 4U);
    return b;
}

// fn(i0, i1) over [0, n) on up to hardware_concurrency threads (one thread under 2^18 units of work); for
// elementwise passes, whose values cannot depend on the split. Defined in reference.cpp.
void parallel_ranges(std::size_t n, std::size_t work_per_item, const std::function<void(std::size_t, std::size_t)>& fn);

}  // namespace strix::vision::layout
