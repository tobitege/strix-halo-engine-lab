// GLM-5.3-Flash prefill MoE: grouped WMMA launches over all routed experts of a chunk (identity: glm_pf_moe.hpp).
// Port of the MiniMax v4 grouped core (3e1897a1, src/minimax/moe/mmx_prefill_core.hpp item_core_v3 Tiled,
// mmx_prefill.hip gather_transform_tiled / gate_up_finish_tiled / down_combine) to the GLM shapes with ONE K split.
//
// Library contract mirrored (strix-trellis e7f801d, strix_gemm_pinned(splits = 1)):
//   input   t = RN16(H128(RN16(materialize(x * suh))))    transforms.hpp:8-26 input_transform, hadamard.hpp:9-37
//   core    per output: acc = +0; acc = WMMA_f32_16x16x16_f16(A_k, B_k, acc) for k-tiles 0 .. K/16 - 1 ascending,
//           A_k = the transformed f16 row, B_k = codebook_value(state, mul1) of the trellis tile (gemm.hip:15-77
//           wmma_core; wmma_tiled.hip tiled_core and wmma_packed decoder_core claim the same bits per output)
//   finish  v = (+0 + p); y = H128(v) * svh (* alpha = 1, no bias, beta = 0)   transforms.hpp:28-51 output_transform
//           (wmma_tiled.hip fused epilogue: the same per-group arithmetic)
// Every helper keeps that operation order; only addressing, scheduling and launch structure differ.
// Compile with -ffp-contract=off -fno-fast-math -fno-gpu-flush-denormals-to-zero.
#include "glm/prefill/glm_pf_moe.hpp"

#include "glm/prefill/glm_moe_traits.hpp"
#include "moe_core/one_split_prefill.hpp"

#include <hip/hip_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::glm::pf {

namespace {

using Traits = MoeTraits;
using Core = strix::moe_core::OneSplitPrefill<Traits>;
using Item = Traits::Item;
using GroupDev = Traits::GroupDev;
using Tables = Traits::Tables;
constexpr unsigned kE = Traits::kE, kTop = Traits::kTop, kH = Traits::kH, kI = Traits::kI;
constexpr unsigned kHB = Traits::kHB, kIB = Traits::kIB;
constexpr unsigned kG3Blocks = Traits::kG3Blocks;

template <int Bits, int MT, int K, int N>
__global__ __launch_bounds__(128) void grouped_kernel(const Item* items, unsigned count) {
    unsigned lo = 0, hi = count;  // last item with cta_begin <= blockIdx.x
    while (hi - lo > 1) {
        const unsigned mid = (lo + hi) / 2;
        if (items[mid].cta_begin <= blockIdx.x) lo = mid;
        else hi = mid;
    }
    const Item it = items[lo];
    constexpr unsigned gx = N / 128;
    const unsigned local = blockIdx.x - it.cta_begin;
    // Prefetch ring of 4 k tiles; MT 8 at 4 spills (256 VGPR + scratch): 2 there for Bits <= 4, 1 above (Bits 5/6
    // spill even at 2). Same bits for every ring size.
    constexpr int kPF = MT == 8 ? (Bits <= 4 ? 2 : 1) : 4;
    Core::item_core<Bits, MT, K, N, kPF>(it, local % gx, local / gx);
}

// Row-tile class of a group with `rows` rows: MT in {1, 2, 4, 8} (index 0..3).
unsigned mt_class(unsigned rows) { return rows <= 16 ? 0U : rows <= 32 ? 1U : rows <= 64 ? 2U : 3U; }


__global__ __launch_bounds__(256) void gather_transform_kernel(const float* x, Tables t) {
    Core::gather_transform_kernel(x, t);
}

__global__ __launch_bounds__(256) void gather_transform_v3_kernel(const float* x, Tables t) {
    Core::gather_transform_v3_kernel(x, t);
}

__global__ __launch_bounds__(256) void finish_kernel(Tables t) {
    Core::finish_kernel(t);
}

__global__ __launch_bounds__(256) void combine_kernel(Tables t, const float* shared, float* out, float* eo) {
    Core::combine_kernel(t, shared, out, eo);
}

// ---------------------------------------------------------------- host side
void launched(const char* what) {
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("glm prefill moe ") + what + ": " + hipGetErrorString(e));
}
void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("glm prefill moe ") + what + ": " + hipGetErrorString(e));
}

constexpr std::size_t align256(std::size_t v) { return (v + 255) & ~std::size_t{255}; }
bool bits_ok(unsigned b) { return Traits::accepts_bits(b); }

struct Plan {
    std::vector<GroupDev> groups;
    std::vector<std::uint32_t> pair_row, pair_group, tok_pair;
    std::vector<float> tok_w;
    std::vector<Item> items;  // gate/up classes 0..3, then down classes 0..3
    unsigned gu_count[4] = {}, dn_count[4] = {}, gu_ctas[4] = {}, dn_ctas[4] = {};
    unsigned tpairs = 0;
    std::vector<std::uint32_t> tile_group;  // [tpairs / 16]
    std::size_t off_tiles = 0;
    std::size_t off_groups = 0, off_pair_row = 0, off_pair_group = 0, off_tok_pair = 0, off_tok_w = 0, off_items = 0,
                off_xt = 0, off_at = 0, off_pg = 0, off_pu = 0, off_pd = 0, bytes = 0;
};

bool exl3_ok(const MoeExl3& m) {
    return m.trellis != nullptr && m.suh != nullptr && m.svh != nullptr && (reinterpret_cast<std::uintptr_t>(m.trellis) & 3U) == 0 &&
           (reinterpret_cast<std::uintptr_t>(m.suh) & 1U) == 0 && (reinterpret_cast<std::uintptr_t>(m.svh) & 1U) == 0;
}

Plan make_plan(const PrefillMoeView& v) {
    if (v.rows == 0) throw std::invalid_argument("glm prefill moe: rows must be > 0");
    if (v.x == nullptr || v.ids == nullptr || v.weights == nullptr || v.experts == nullptr || v.shared == nullptr || v.out == nullptr)
        throw std::invalid_argument("glm prefill moe: null pointer in view");
    if (!bits_ok(v.bits_gu) || !bits_ok(v.bits_dn)) throw std::invalid_argument("glm prefill moe: bits must be 3..6");
    const unsigned T = v.rows;
    if (T > (1U << 24)) throw std::invalid_argument("glm prefill moe: too many rows");
    Plan p;
    std::vector<std::vector<std::uint32_t>> erows(kE);
    std::vector<std::uint32_t> slot(std::size_t{T} * kTop);  // index of route (t, k) inside its expert's row list
    for (unsigned t = 0; t < T; ++t)
        for (unsigned k = 0; k < kTop; ++k) {
            const std::uint32_t e = v.ids[std::size_t{t} * kTop + k];
            if (e >= kE) throw std::invalid_argument("glm prefill moe: expert id out of range");
            slot[std::size_t{t} * kTop + k] = static_cast<std::uint32_t>(erows[e].size());
            erows[e].push_back(t);  // ascending token order
        }
    std::vector<std::uint32_t> group_of(kE, ~0U);
    for (unsigned e = 0; e < kE; ++e) {
        if (erows[e].empty()) continue;
        const MoeExpert& w = v.experts[e];
        if (!exl3_ok(w.gate) || !exl3_ok(w.up) || !exl3_ok(w.down))
            throw std::invalid_argument("glm prefill moe: routed expert " + std::to_string(e) + " has null or misaligned weights");
        GroupDev g{};
        g.w = w;
        g.n = static_cast<unsigned>(erows[e].size());
        g.pair_begin = static_cast<unsigned>(p.pair_row.size());
        g.tbegin = p.tpairs;
        p.tpairs += (g.n + 15U) / 16U * 16U;
        p.tile_group.insert(p.tile_group.end(), (g.n + 15U) / 16U, static_cast<std::uint32_t>(p.groups.size()));
        group_of[e] = static_cast<std::uint32_t>(p.groups.size());
        for (const std::uint32_t r : erows[e]) {
            p.pair_row.push_back(r);
            p.pair_group.push_back(group_of[e]);
        }
        p.groups.push_back(g);
    }
    p.tok_pair.resize(std::size_t{T} * kTop);
    p.tok_w.resize(std::size_t{T} * kTop);
    for (std::size_t i = 0; i < p.tok_pair.size(); ++i) {
        p.tok_pair[i] = p.groups[group_of[v.ids[i]]].pair_begin + slot[i];
        p.tok_w[i] = v.weights[i];
    }
    const std::size_t P = p.pair_row.size();
    // Workspace layout (every region 256-byte aligned).
    std::size_t off = 0;
    auto take = [&](std::size_t bytes) {
        const std::size_t at = off;
        off += align256(bytes);
        return at;
    };
    p.off_groups = take(p.groups.size() * sizeof(GroupDev));
    p.off_pair_row = take(P * 4);
    p.off_pair_group = take(P * 4);
    p.off_tok_pair = take(p.tok_pair.size() * 4);
    p.off_tok_w = take(p.tok_w.size() * 4);
    p.off_items = take(3 * p.groups.size() * sizeof(Item));
    p.off_tiles = take(p.tile_group.size() * 4);
    p.off_xt = take(2 * std::size_t{p.tpairs} * kH * 2);
    p.off_at = take(std::size_t{p.tpairs} * kI * 2);
    p.off_pg = take(P * kI * 4);
    p.off_pu = take(P * kI * 4);
    p.off_pd = take(P * kH * 4);
    p.bytes = off;
    return p;
}

void fill_items(Plan& p, unsigned char* ws) {
    auto* xt = reinterpret_cast<_Float16*>(ws + p.off_xt);
    auto* at = reinterpret_cast<_Float16*>(ws + p.off_at);
    auto* pg = reinterpret_cast<float*>(ws + p.off_pg);
    auto* pu = reinterpret_cast<float*>(ws + p.off_pu);
    auto* pd = reinterpret_cast<float*>(ws + p.off_pd);
    std::vector<Item> gu[4], dn[4];
    for (const GroupDev& g : p.groups) {
        const unsigned cls = mt_class(g.n), tm = 16U << cls;
        const unsigned gy = (g.n + tm - 1) / tm;
        for (unsigned m = 0; m < 2; ++m) {
            Item it{};
            it.trellis = m == 0 ? g.w.gate.trellis : g.w.up.trellis;
            it.x = xt + (static_cast<std::size_t>(m) * p.tpairs + g.tbegin) * kH;
            it.part = (m == 0 ? pg : pu) + static_cast<std::size_t>(g.pair_begin) * kI;
            it.rows = g.n;
            it.gy = gy;
            it.cta_begin = p.gu_ctas[cls];
            p.gu_ctas[cls] += (kI / 128) * gy;
            gu[cls].push_back(it);
        }
        Item it{};
        it.trellis = g.w.down.trellis;
        it.x = at + static_cast<std::size_t>(g.tbegin) * kI;
        it.part = pd + static_cast<std::size_t>(g.pair_begin) * kH;
        it.rows = g.n;
        it.gy = gy;
        it.cta_begin = p.dn_ctas[cls];
        p.dn_ctas[cls] += (kH / 128) * gy;
        dn[cls].push_back(it);
    }
    for (unsigned c = 0; c < 4; ++c) {
        p.gu_count[c] = static_cast<unsigned>(gu[c].size());
        p.items.insert(p.items.end(), gu[c].begin(), gu[c].end());
    }
    for (unsigned c = 0; c < 4; ++c) {
        p.dn_count[c] = static_cast<unsigned>(dn[c].size());
        p.items.insert(p.items.end(), dn[c].begin(), dn[c].end());
    }
}

bool gather_v3() {  // default gather v3 (same bits; measurement: -12/-13 % at T 2048/4096); STRIX_GLM_PF_MOE_GATHER=1: v1
    static const bool v3 = [] {                     // (v2 removed: 2.1x slower, k13)
        const char* e = std::getenv("STRIX_GLM_PF_MOE_GATHER");
        return !(e != nullptr && std::string(e) == "1");
    }();
    return v3;
}
template <int Bits, int K, int N>
void launch_class(const Item* items_d, unsigned n, unsigned ctas, unsigned cls, hipStream_t stream) {
    if (n == 0) return;
    switch (cls) {
        case 0: hipLaunchKernelGGL((grouped_kernel<Bits, 1, K, N>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
        case 1: hipLaunchKernelGGL((grouped_kernel<Bits, 2, K, N>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
        case 2: hipLaunchKernelGGL((grouped_kernel<Bits, 4, K, N>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
        default: hipLaunchKernelGGL((grouped_kernel<Bits, 8, K, N>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
    }
}
template <int K, int N>
void launch_stage(unsigned bits, const Item* items_d, const unsigned (&count)[4], const unsigned (&ctas)[4], hipStream_t stream) {
    std::size_t at = 0;
    for (unsigned c = 0; c < 4; ++c) {
        switch (bits) {
            case 3: launch_class<3, K, N>(items_d + at, count[c], ctas[c], c, stream); break;
            case 4: launch_class<4, K, N>(items_d + at, count[c], ctas[c], c, stream); break;
            case 5: launch_class<5, K, N>(items_d + at, count[c], ctas[c], c, stream); break;
            case 6: launch_class<6, K, N>(items_d + at, count[c], ctas[c], c, stream); break;
            default: throw std::invalid_argument("glm prefill moe: bits must be 3..6");
        }
        at += count[c];
    }
}

PrefillMoeMarker g_marker = nullptr;
void* g_marker_user = nullptr;
void mark(const char* label, hipStream_t stream) {
    if (g_marker != nullptr) g_marker(label, stream, g_marker_user);
}

}  // namespace

void set_prefill_moe_marker(PrefillMoeMarker marker, void* user) {
    g_marker = marker;
    g_marker_user = user;
}

std::size_t prefill_moe_workspace_bytes(const PrefillMoeView& v) { return make_plan(v).bytes; }

std::size_t prefill_moe_workspace_bound(std::uint32_t rows) {
    const std::size_t P = std::size_t{kTop} * rows, G = std::min<std::size_t>(kE, P);
    const std::size_t PX = P + 15 * G;  // every group padded to 16 rows
    std::size_t b = align256(G * sizeof(GroupDev)) + 2 * align256(P * 4) + 2 * align256(P * 4) + align256(3 * G * sizeof(Item));
    b += align256(PX / 16 * 4);  // tile -> group table (gather v3)
    b += align256(2 * PX * kH * 2) + align256(PX * kI * 2);
    b += 2 * align256(P * kI * 4) + align256(P * kH * 4);
    return b;
}

void enqueue_prefill_moe_glm(const PrefillMoeView& v, void* workspace, std::size_t workspace_bytes, hipStream_t stream) {
    Plan p = make_plan(v);
    if (workspace == nullptr || (reinterpret_cast<std::uintptr_t>(workspace) & 255U) != 0 || workspace_bytes < p.bytes)
        throw std::invalid_argument("glm prefill moe: workspace null, misaligned or too small");
    auto* ws = static_cast<unsigned char*>(workspace);
    fill_items(p, ws);
    ok(hipMemcpyAsync(ws + p.off_groups, p.groups.data(), p.groups.size() * sizeof(GroupDev), hipMemcpyHostToDevice, stream), "groups");
    ok(hipMemcpyAsync(ws + p.off_pair_row, p.pair_row.data(), p.pair_row.size() * 4, hipMemcpyHostToDevice, stream), "pairs");
    ok(hipMemcpyAsync(ws + p.off_pair_group, p.pair_group.data(), p.pair_group.size() * 4, hipMemcpyHostToDevice, stream), "pairs");
    ok(hipMemcpyAsync(ws + p.off_tok_pair, p.tok_pair.data(), p.tok_pair.size() * 4, hipMemcpyHostToDevice, stream), "tokens");
    ok(hipMemcpyAsync(ws + p.off_tok_w, p.tok_w.data(), p.tok_w.size() * 4, hipMemcpyHostToDevice, stream), "tokens");
    ok(hipMemcpyAsync(ws + p.off_items, p.items.data(), p.items.size() * sizeof(Item), hipMemcpyHostToDevice, stream), "items");
    ok(hipMemcpyAsync(ws + p.off_tiles, p.tile_group.data(), p.tile_group.size() * 4, hipMemcpyHostToDevice, stream), "tiles");
    ok(hipStreamSynchronize(stream), "tables");  // the host vectors die with this call
    Tables t{};
    t.groups = reinterpret_cast<const GroupDev*>(ws + p.off_groups);
    t.tile_group = reinterpret_cast<const std::uint32_t*>(ws + p.off_tiles);
    t.pair_row = reinterpret_cast<const std::uint32_t*>(ws + p.off_pair_row);
    t.pair_group = reinterpret_cast<const std::uint32_t*>(ws + p.off_pair_group);
    t.tok_pair = reinterpret_cast<const std::uint32_t*>(ws + p.off_tok_pair);
    t.tok_w = reinterpret_cast<const float*>(ws + p.off_tok_w);
    t.xt = reinterpret_cast<_Float16*>(ws + p.off_xt);
    t.at = reinterpret_cast<_Float16*>(ws + p.off_at);
    t.part_g = reinterpret_cast<float*>(ws + p.off_pg);
    t.part_u = reinterpret_cast<float*>(ws + p.off_pu);
    t.part_d = reinterpret_cast<float*>(ws + p.off_pd);
    t.pairs = static_cast<unsigned>(p.pair_row.size());
    t.rows = v.rows;
    t.tpairs = p.tpairs;
    const auto* items_d = reinterpret_cast<const Item*>(ws + p.off_items);
    std::size_t n_gu = 0;
    for (unsigned c = 0; c < 4; ++c) n_gu += p.gu_count[c];
    mark("start", stream);
    if (gather_v3())
        hipLaunchKernelGGL(gather_transform_v3_kernel, dim3(t.tpairs / 16, 2, kHB / kG3Blocks), dim3(256), 0, stream, v.x, t);
    else
        hipLaunchKernelGGL(gather_transform_kernel, dim3((t.pairs * 2 * kHB + 7) / 8), dim3(256), 0, stream, v.x, t);
    launched("gather transform");
    mark("gather", stream);
    launch_stage<kH, kI>(v.bits_gu, items_d, p.gu_count, p.gu_ctas, stream);
    launched("gate/up");
    mark("gate_up", stream);
    hipLaunchKernelGGL(finish_kernel, dim3((t.pairs * kIB + 7) / 8), dim3(256), 0, stream, t);
    launched("finish");
    mark("finish", stream);
    launch_stage<kI, kH>(v.bits_dn, items_d + n_gu, p.dn_count, p.dn_ctas, stream);
    launched("down");
    mark("down", stream);
    hipLaunchKernelGGL(combine_kernel, dim3((v.rows * kHB + 7) / 8), dim3(256), 0, stream, t, v.shared, v.out, v.eo_debug);
    launched("combine");
    mark("combine", stream);
}

MoeExl3 moe_exl3_from_weights(strix_weights w, unsigned N, unsigned K, unsigned bits) {
    const void* data = nullptr;
    std::uint64_t bytes = 0;
    if (strix_weights_device_span(w, &data, &bytes) != STRIX_SUCCESS || data == nullptr)
        throw std::invalid_argument("glm prefill moe: strix_weights_device_span failed");
    if (N % 128 != 0 || K % 128 != 0 || bits < 1 || bits > 8) throw std::invalid_argument("glm prefill moe: bad matrix shape");
    constexpr std::size_t A = STRIX_TRELLIS_WORKSPACE_ALIGNMENT;  // api.hip prepare_regions alignment
    auto al = [](std::size_t v) { return (v + A - 1) / A * A; };
    const std::size_t tb = std::size_t{N} * K / 8 * bits;
    const std::size_t suh_off = al(tb), svh_off = al(suh_off + std::size_t{K} * 2), total = al(svh_off + std::size_t{N} * 2);
    if (bytes != total)
        throw std::invalid_argument("glm prefill moe: weight span " + std::to_string(bytes) + " bytes, expected " + std::to_string(total) +
                                    " (no bias, layout trellis | suh | svh)");
    const auto* base = static_cast<const unsigned char*>(data);
    MoeExl3 m;
    m.trellis = reinterpret_cast<const std::uint32_t*>(base);
    m.suh = reinterpret_cast<const std::uint16_t*>(base + suh_off);
    m.svh = reinterpret_cast<const std::uint16_t*>(base + svh_off);
    return m;
}

}  // namespace strix::glm::pf
