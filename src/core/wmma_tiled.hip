// Register-blocked, double-buffered WMMA prefill core for gfx1151.
// TrellisF16 numerics are those of wmma_core (gemm.hip): every output accumulates the
// split's K16 tiles in ascending order with v_wmma_f32_16x16x16_f16 from +0,
// on the same decoded FP16 Q and the same FP16 input. Only the decomposition
// changes: a block owns BM x BN outputs, decodes each Q tile once per block
// into LDS ([n][k], two K16 tiles per stage) and every wave reuses A and B
// fragments across its MT x NT accumulators.
// Explicit PackedIU4 FAST v1 instead rescales signed I32 sums at every K64
// boundary in ascending group order. It never enters the default resolver.
#include "wmma_tiled.hpp"
#include "trellis_codec.hpp"
#include "hadamard.hpp"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "tiled_core.hpp"
namespace strix::core {
namespace wmma_tiled {
namespace {

// Row blocks per rasterisation group (block -> tile mapping only; bits unchanged).
// Default per tile (measured: 2 is best for the 128- and 256-row tiles at large K);
// STRIX_TRELLIS_WMMA_RASTER overrides it for measurements.
int raster_rows(int tile_default) {
    static const int value = [] {
        const char* v = std::getenv("STRIX_TRELLIS_WMMA_RASTER");
        int r = v ? std::atoi(v) : 0;
        return r >= 1 && r <= 64 ? r : 0;
    }();
    return value ? value : tile_default;
}
using Config = LinearTile;
// Index = tile id accepted by launch_wmma_tiled(tile=...).
constexpr auto& configs = linear_tiles;
constexpr int config_count = int(sizeof(configs) / sizeof(configs[0]));

template<int Bits, int CB, int BM, int BN, int WM, int WN>
hipError_t launch_one(const DeviceWeights& w, const _Float16* input, float* partial, int rows, int splits,
                      hipStream_t stream, const WmmaFusedOutput* fused, int raster) {
    using G = Geometry<BM, BN, WM, WN>;
    dim3 grid(unsigned(((rows + BM - 1) / BM) * (w.output / BN)), 1, unsigned(splits));   // N % BN == 0 (launcher)
    if (fused && fused->output) {
        if constexpr (BN % 128 == 0)
            tiled_core<TiledContract<TrellisF16<Bits, CB>>, BM, BN, WM, WN, true><<<grid, G::threads, 0, stream>>>(w.trellis, input, partial,
                w.input, w.output, rows, splits, w.svh, w.bias, fused->output, fused->alpha, fused->beta, raster);
        else return hipErrorNotSupported;
    } else {
        tiled_core<TiledContract<TrellisF16<Bits, CB>>, BM, BN, WM, WN, false><<<grid, G::threads, 0, stream>>>(w.trellis, input, partial,
            w.input, w.output, rows, splits, nullptr, nullptr, nullptr, 1.f, 0.f, raster);
    }
    return hipGetLastError();
}
#ifdef STRIX_TRELLIS_WMMA_PROBE
template<int Bits, int BM, int BN, int WM, int WN>
hipError_t launch_probe_one(const DeviceWeights& w, const _Float16* input, float* partial, int rows, int splits,
                            hipStream_t stream, int probe, int raster) {
    using G = Geometry<BM, BN, WM, WN>;
    dim3 grid(unsigned(((rows + BM - 1) / BM) * (w.output / BN)), 1, unsigned(splits));
#define PROBE_CASE(P) case P: tiled_core<TiledContract<TrellisF16<Bits, 2>>, BM, BN, WM, WN, false, P><<<grid, G::threads, 0, stream>>>(w.trellis, input,         partial, w.input, w.output, rows, splits, nullptr, nullptr, nullptr, 1.f, 0.f, raster); break
    switch (probe) { PROBE_CASE(1); PROBE_CASE(2); PROBE_CASE(3); default: return hipErrorNotSupported; }
#undef PROBE_CASE
    return hipGetLastError();
}
template<int Bits>
hipError_t launch_probe_bits(const DeviceWeights& w, const _Float16* input, float* partial, int rows, int splits,
                             hipStream_t stream, int tile, int probe) {
    switch (tile) {
#define TILE_CASE(I) case I: return launch_probe_one<Bits, configs[I].bm, configs[I].bn, configs[I].wm, configs[I].wn>(w, input, partial, rows, splits, stream, probe, raster_rows(configs[I].raster))
    TILE_CASE(0); TILE_CASE(1); TILE_CASE(2); TILE_CASE(3); TILE_CASE(4); TILE_CASE(5); TILE_CASE(6); TILE_CASE(7);
#undef TILE_CASE
    }
    return hipErrorNotSupported;
}
#endif
template<int Bits, int CB>
hipError_t launch_bits(const DeviceWeights& w, const _Float16* input, float* partial, int rows, int splits,
                       hipStream_t stream, const WmmaFusedOutput* fused, int tile, int raster) {
    switch (tile) {
#define TILE_CASE(I) case I: return launch_one<Bits, CB, configs[I].bm, configs[I].bn, configs[I].wm, configs[I].wn>(w, input, partial, rows, splits, stream, fused, raster ? raster : raster_rows(configs[I].raster))
    TILE_CASE(0); TILE_CASE(1); TILE_CASE(2); TILE_CASE(3); TILE_CASE(4); TILE_CASE(5); TILE_CASE(6); TILE_CASE(7);
#undef TILE_CASE
    }
    return hipErrorNotSupported;
}
} // namespace

// Measured policy (7 EAQ tensors x M128-4096, cb2 K5/K6, every tile; README per-rank shapes
// confirmed in the final ABA). Within 1.2 % of the best tile on average, worst 1.19x, and faster
// than wmma_core in every measured cell. -1 = keep wmma_core (fewer than 128 rows: unmeasured).
int policy_tile(const DeviceWeights& w, int rows, int splits) {
    (void)splits;   // the split count is fixed by the numerics; every tile honours it
    if (rows < 128 || w.codebook != 2 || w.bits < 4 || w.bits > 6) return -1;
    if (rows <= 128) return w.output % 256 == 0 && w.output < 16384 ? 6 : 0;
    if (rows <= 1024) return 4;
    return 0;
}
template<int Tile, bool Audit = false, bool FullRows = false, int Group = 64> hipError_t query_iu4_resources(IU4Resources* resources) {
    constexpr auto c = iu4_tiles[Tile];
    using G = Geometry<c.bm, c.bn, c.wm, c.wn, PackedIU4>;
    auto kernel = tiled_core<TiledContract<PackedIU4>, c.bm, c.bn, c.wm, c.wn, false, 0, Audit, FullRows, Group>;
    hipFuncAttributes attr{};
    hipError_t status = hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel));
    if (status != hipSuccess) return status;
    int blocks = 0;
    status = hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, G::threads, 0);
    if (status != hipSuccess) return status;
    *resources = {attr.numRegs, blocks * G::waves, attr.localSizeBytes, attr.sharedSizeBytes};
    return hipSuccess;
}
template<class Operand, int Tile, bool Audit = false, bool FullRows = false, int Group = 64>
hipError_t launch_plain(const uint32_t* packed, const _Float16* input,
    const IU4Operands& quantised, float* result, int inputs, int outputs, int rows,
    hipStream_t stream, int32_t* group_sums) {
    constexpr auto c = iu4_tiles[Tile];
    using G = Geometry<c.bm, c.bn, c.wm, c.wn, Operand>;
    static_assert(G::waves >= 4 && G::waves <= 8);
    const auto blocks = uint64_t((uint64_t(rows) + uint64_t(c.bm) - 1U) / uint64_t(c.bm)) * uint64_t(outputs / c.bn);
    if (blocks > uint64_t(INT32_MAX)) return hipErrorInvalidValue;
    if constexpr (std::is_same_v<Operand, PackedIU4>) {
        // Cache successful qualification per device/plan; diagnostic registers
        // and occupancy must pass K3 before a dense launch can execute.
        static std::atomic<uint64_t> qualified{0};
        int device = 0;
        hipError_t status = hipGetDevice(&device);
        if (status != hipSuccess) return status;
        const uint64_t bit = device >= 0 && device < 64 ? uint64_t(1) << unsigned(device) : 0;
        if (!bit || !(qualified.load(std::memory_order_relaxed) & bit)) {
            IU4Resources resources;
            status = query_iu4_resources<Tile, Audit, FullRows, Group>(&resources);
            if (status != hipSuccess) return status;
            if (resources.scratch_bytes || resources.active_waves < 4)
                return hipErrorNotSupported;
            if (bit) qualified.fetch_or(bit, std::memory_order_relaxed);
        }
    }
    const dim3 grid{unsigned(blocks)};
    tiled_core<TiledContract<Operand>, c.bm, c.bn, c.wm, c.wn, false, 0, Audit, FullRows, Group>
        <<<grid, G::threads, 0, stream>>>(packed, input, result, inputs, outputs, rows, 1,
        nullptr, nullptr, nullptr, 1.f, 0.f, raster_rows(c.raster), quantised.a_q,
        quantised.a_scale, quantised.w_scale, group_sums);
    return hipGetLastError();
}
template<class Operand, bool Audit = false, bool FullRows = false, int Group = 64>
hipError_t launch_plain_plan(const uint32_t* packed, const _Float16* input,
    const IU4Operands& quantised, float* result, int inputs, int outputs, int rows,
    hipStream_t stream, int tile, int32_t* group_sums) {
    switch (tile) {
        case 0: return launch_plain<Operand,0,Audit,FullRows,Group>(packed,input,quantised,result,inputs,outputs,rows,stream,group_sums);
        case 1: return launch_plain<Operand,1,Audit,FullRows,Group>(packed,input,quantised,result,inputs,outputs,rows,stream,group_sums);
        case 2: return launch_plain<Operand,2,Audit,FullRows,Group>(packed,input,quantised,result,inputs,outputs,rows,stream,group_sums);
        default: return hipErrorInvalidValue;
    }
}
template<int Group>
hipError_t launch_iu4_group(const IU4Operands& operands, float* output,
    int inputs, int outputs, int rows, hipStream_t stream, int tile, int32_t* group_sums) {
    // Full-row specializations remove the row clamps. Tail kernels clamp loads;
    // qualification covers the actual variant.
    const auto* w = reinterpret_cast<const uint32_t*>(operands.w_q);
    if (rows % iu4_tiles[tile].bm == 0) {
        if (group_sums) return launch_plain_plan<PackedIU4, true, true, Group>(w, nullptr, operands, output, inputs, outputs, rows, stream, tile, group_sums);
        return launch_plain_plan<PackedIU4, false, true, Group>(w, nullptr, operands, output, inputs, outputs, rows, stream, tile, nullptr);
    }
    if (group_sums) return launch_plain_plan<PackedIU4, true, false, Group>(w, nullptr, operands, output, inputs, outputs, rows, stream, tile, group_sums);
    return launch_plain_plan<PackedIU4, false, false, Group>(w, nullptr, operands, output, inputs, outputs, rows, stream, tile, nullptr);
}
template<int Group> hipError_t query_iu4_group(int tile, IU4Resources* resources, bool full_rows) {
    switch (tile * 2 + int(full_rows)) {
        case 0: return query_iu4_resources<0, false, false, Group>(resources);
        case 1: return query_iu4_resources<0, false, true, Group>(resources);
        case 2: return query_iu4_resources<1, false, false, Group>(resources);
        case 3: return query_iu4_resources<1, false, true, Group>(resources);
        case 4: return query_iu4_resources<2, false, false, Group>(resources);
        case 5: return query_iu4_resources<2, false, true, Group>(resources);
        default: return hipErrorInvalidValue;
    }
}
// Grouped IU4 (MoE). g64 only (the FAST_I4 contract); plan = tile, epilogue raw or transform.
template<int Tile, bool Transform> auto grouped_kernel() {
    constexpr auto c = iu4_tiles[Tile];
    return tiled_core<TiledContract<PackedIU4>, c.bm, c.bn, c.wm, c.wn, Transform, 0, false, false, 64, true>;
}
template<int Tile, bool Transform> hipError_t query_grouped(IU4Resources* resources) {
    constexpr auto c = iu4_tiles[Tile];
    using G = Geometry<c.bm, c.bn, c.wm, c.wn, PackedIU4>;
    auto kernel = grouped_kernel<Tile, Transform>();
    hipFuncAttributes attr{};
    hipError_t status = hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel));
    if (status != hipSuccess) return status;
    int blocks = 0;
    status = hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, G::threads, 0);
    if (status != hipSuccess) return status;
    *resources = {attr.numRegs, blocks * G::waves, attr.localSizeBytes, attr.sharedSizeBytes};
    return hipSuccess;
}
template<int Tile, bool Transform> hipError_t launch_grouped(const IU4GroupedOperands& o,
    const IU4GroupTile* tiles, const uint32_t* ntiles, uint32_t max_tiles, float* output,
    int inputs, int outputs, hipStream_t stream) {
    constexpr auto c = iu4_tiles[Tile];
    using G = Geometry<c.bm, c.bn, c.wm, c.wn, PackedIU4>;
    static std::atomic<uint64_t> qualified{0};
    int device = 0;
    hipError_t status = hipGetDevice(&device);
    if (status != hipSuccess) return status;
    const uint64_t bit = device >= 0 && device < 64 ? uint64_t(1) << unsigned(device) : 0;
    if (!bit || !(qualified.load(std::memory_order_relaxed) & bit)) {
        IU4Resources resources;
        status = query_grouped<Tile, Transform>(&resources);
        if (status != hipSuccess) return status;
        if (resources.scratch_bytes || resources.active_waves < 4) return hipErrorNotSupported;
        if (bit) qualified.fetch_or(bit, std::memory_order_relaxed);
    }
    const uint64_t blocks = uint64_t(max_tiles) * uint64_t(outputs / c.bn);
    if (!blocks) return hipSuccess;
    if (blocks > uint64_t(INT32_MAX)) return hipErrorInvalidValue;
    GroupedArgs gp;
    gp.tiles = tiles; gp.ntiles = ntiles;
    gp.w_expert_bytes = o.w_expert_bytes; gp.w_scale_expert = o.w_scale_expert;
    gp.col_scale = o.col_scale; gp.map = o.map;
    gp.a_mul = o.a_mul; gp.a_add = o.a_add; gp.o_mul = o.o_mul; gp.o_add = o.o_add;
    grouped_kernel<Tile, Transform>()<<<dim3{unsigned(blocks)}, G::threads, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(o.w_q), nullptr, Transform ? nullptr : output, inputs, outputs, 0, 1,
        nullptr, nullptr, Transform ? output : nullptr, 1.f, 0.f, 1, o.a_q, o.a_scale, o.w_scale, nullptr, gp, NoAdapter{}, NoChain{});
    return hipGetLastError();
}
// EXL3 -> IU4 g64 pack (the engines' iu4_convert_kernel with the shared requantisation).
// Grid (ceil((K/64) * (N/16) / 8), experts), 256 threads; wave = one 16-column tile x one K64
// group; lane l holds columns l/4 and l/4 + 8 and k pairs 2(l&3), 8 + 2(l&3) of the group's
// four K16 tiles. s = FP16(max|w| / 7) (1 when all zero, floored at 2^-24); q = iu4_requant(w, 1/s).
template<int Bits>
__global__ __launch_bounds__(256) void exl3_iu4_convert_kernel(const uint32_t* const* trellis, int inputs, int outputs,
    uint8_t* wq, _Float16* ws, size_t wq_expert, size_t ws_expert, int scale_rule) {
    const bool full = scale_rule == iu4_scale_rule_fast_full;
    const int lane = int(threadIdx.x & 31U), item = int(blockIdx.x) * 8 + int(threadIdx.x >> 5U), ntiles = outputs / 16;
    const int expert = int(blockIdx.y);
    const uint32_t* src = trellis[expert];
    if (src == nullptr || item >= (inputs / 64) * ntiles) return;
    const int g = item / ntiles, nt = item % ntiles, groups = inputs / 64;
    float v[4][8];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        RawLane<Bits> raw;
        raw.load(src + (size_t(g * 4 + u) * size_t(ntiles) + size_t(nt)) * size_t(Bits * 8));
        const Stream<Bits> st(raw);
        const uint32_t p[4] = {decode_pair<2>(st.template state<0>(), st.template state<1>()),
                               decode_pair<2>(st.template state<2>(), st.template state<3>()),
                               decode_pair<2>(st.template state<4>(), st.template state<5>()),
                               decode_pair<2>(st.template state<6>(), st.template state<7>())};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __half2 h = __builtin_bit_cast(__half2, p[j]);
            v[u][2 * j] = __low2float(h);
            v[u][2 * j + 1] = __high2float(h);
        }
    }
    const int q = lane & 3;
    uint8_t* base = wq ? wq + size_t(expert) * wq_expert : nullptr;
    _Float16* sbase = ws + size_t(expert) * ws_expert;
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int n = nt * 16 + lane / 4 + 8 * half;
        float amax = 0.0f;
        #pragma unroll
        for (int u = 0; u < 4; ++u)
            #pragma unroll
            for (int j = 0; j < 4; ++j) amax = fmaxf(amax, fabsf(v[u][4 * half + j]));
        amax = fmaxf(amax, __shfl_xor(amax, 1));
        amax = fmaxf(amax, __shfl_xor(amax, 2));
        _Float16 s = _Float16(amax == 0.0f ? 1.0f : fmaxf(amax / 7.0f, 0x1p-24f));
        if (full) {  // fast full range: the group's positive and negative extremes (the quad's four lanes hold its 64 values)
            float pos = 0.0f, neg = 0.0f;
            #pragma unroll
            for (int u = 0; u < 4; ++u)
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    pos = fmaxf(pos, v[u][4 * half + j]);
                    neg = fmaxf(neg, -v[u][4 * half + j]);
                }
            pos = fmaxf(pos, __shfl_xor(pos, 1));
            pos = fmaxf(pos, __shfl_xor(pos, 2));
            neg = fmaxf(neg, __shfl_xor(neg, 1));
            neg = fmaxf(neg, __shfl_xor(neg, 2));
            s = _Float16(iu4_fast_full_scale(pos, neg));
        }
        const float inv = 1.0f / float(s);
        if (q == 0) sbase[size_t(n) * size_t(groups) + size_t(g)] = s;
        if (!base) continue;
        #pragma unroll
        for (int u = 0; u < 4; ++u)
            #pragma unroll
            for (int pp = 0; pp < 2; ++pp) {
                const float w0 = v[u][4 * half + 2 * pp], w1 = v[u][4 * half + 2 * pp + 1];
                const int lo = full ? iu4_requant_full(w0, inv) : iu4_requant(w0, inv),
                          hi = full ? iu4_requant_full(w1, inv) : iu4_requant(w1, inv);
                const int k = (g * 4 + u) * 16 + 8 * pp + 2 * q;
                base[iu4_weight_byte(k, n, outputs)] = uint8_t((uint32_t(lo) & 15U) | ((uint32_t(hi) & 15U) << 4U));
            }
    }
}
// One CTA: per-expert tile counts, Hillis-Steele inclusive scan in LDS, then each expert's tiles.
__global__ __launch_bounds__(1024) void grouped_tiles_kernel(const IU4Group* groups, int experts, int bm,
    IU4GroupTile* tiles, uint32_t* ntiles, uint32_t max_tiles) {
    __shared__ uint32_t scan[2][1024];
    __shared__ uint32_t carry;
    const int tid = int(threadIdx.x);
    if (tid == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < experts; base += 1024) {
        const int e = base + tid;
        const uint32_t count = e < experts ? groups[e].count : 0U;
        const uint32_t n = (count + uint32_t(bm) - 1U) / uint32_t(bm);
        int cur = 0;
        scan[0][tid] = n;
        __syncthreads();
        for (int d = 1; d < 1024; d *= 2) {
            const uint32_t v = scan[cur][tid] + (tid >= d ? scan[cur][tid - d] : 0U);
            scan[cur ^ 1][tid] = v;
            cur ^= 1;
            __syncthreads();
        }
        const uint32_t first_tile = carry + scan[cur][tid] - n;
        if (e < experts) {
            const IU4Group g = groups[e];
            for (uint32_t t = 0; t < n && first_tile + t < max_tiles; ++t) {
                const uint32_t r = t * uint32_t(bm);
                tiles[first_tile + t] = {g.expert, g.first + r, g.count - r < uint32_t(bm) ? g.count - r : uint32_t(bm)};
            }
        }
        __syncthreads();
        if (tid == 1023) carry += scan[cur][1023];
        __syncthreads();
    }
    if (tid == 0) *ntiles = carry < max_tiles ? carry : max_tiles;
}
// One wave per row/group: each lane owns Group/32 adjacent K values. The caller
// performs had128 first; no expert copies, rotation or synchronization here.
template<int Group>
__global__ __launch_bounds__(256) void quantise_activations(
    const _Float16* input, uint8_t* packed, float* scales, int inputs, int rows) {
    constexpr int per_lane = Group / 32;
    const size_t index = size_t(blockIdx.x) * 8U + size_t(threadIdx.x / 32U);
    const int lane = int(threadIdx.x & 31U);
    if (index >= size_t(rows) * size_t(inputs / Group)) return;
    float x[per_lane];
    float maximum = 0;
    #pragma unroll
    for (int j = 0; j < per_lane; ++j) {
        x[j] = float(input[index * size_t(Group) + size_t(lane * per_lane + j)]);
        maximum = fmaxf(maximum, fabsf(x[j]));
    }
    #pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) maximum = fmaxf(maximum, __shfl_xor(maximum, delta));
    const float scale = maximum == 0 ? 1.f : maximum / 7.f;
    #pragma unroll
    for (int j = 0; j < per_lane; j += 2) {
        const int a = max(-7, min(7, int(nearbyintf(x[j] / scale))));
        const int b = max(-7, min(7, int(nearbyintf(x[j + 1] / scale))));
        packed[index * size_t(Group / 2) + size_t((lane * per_lane + j) / 2)] = uint8_t((unsigned(a) & 15U) | ((unsigned(b) & 15U) << 4U));
    }
    if (lane == 0) scales[index] = scale;
}
} // namespace wmma_tiled

hipError_t query_wmma_iu4_resources(int tile, IU4Resources* resources, bool full_rows, int group) {
    if (!resources) return hipErrorInvalidValue;
    switch (group) {
        case 64: return wmma_tiled::query_iu4_group<64>(tile, resources, full_rows);
        case 128: return wmma_tiled::query_iu4_group<128>(tile, resources, full_rows);
        case 256: return wmma_tiled::query_iu4_group<256>(tile, resources, full_rows);
        default: return hipErrorInvalidValue;
    }
}

hipError_t launch_wmma_tiled_iu4(const IU4Operands& operands, float* output,
    int inputs, int outputs, int rows, hipStream_t stream, int tile, int32_t* group_sums) {
    if (tile == -1) tile = iu4_policy_tile(rows);
    const int group = operands.group;
    if (!iu4_geometry_ok(inputs, outputs, rows, tile) || !iu4_group_ok(group) || inputs % group ||
        !operands.a_q || !operands.w_q || !operands.a_scale || !operands.w_scale || !output ||
        (uintptr_t(operands.a_q) & 15U) || (uintptr_t(operands.w_q) & 15U) ||
        (uintptr_t(operands.a_scale) & 3U) || (uintptr_t(operands.w_scale) & 1U) ||
        (uintptr_t(output) & 3U) || (uintptr_t(group_sums) & 3U)) return hipErrorInvalidValue;
    // The core addresses packed A/W with 32-bit byte offsets.
    if (uint64_t(rows) * uint64_t(inputs) / 2U > UINT32_MAX || uint64_t(outputs) * uint64_t(inputs) / 2U > UINT32_MAX)
        return hipErrorNotSupported;
    switch (group) {
        case 64: return wmma_tiled::launch_iu4_group<64>(operands, output, inputs, outputs, rows, stream, tile, group_sums);
        case 128: return wmma_tiled::launch_iu4_group<128>(operands, output, inputs, outputs, rows, stream, tile, group_sums);
        default: return wmma_tiled::launch_iu4_group<256>(operands, output, inputs, outputs, rows, stream, tile, group_sums);
    }
}
size_t iu4_grouped_max_tiles(size_t map_rows, int experts, int tile) {
    if (tile < 0 || tile > 2 || experts < 0) return 0;
    const size_t bm = size_t(iu4_tiles[tile].bm);
    return (map_rows + bm - 1U) / bm + size_t(experts);
}
hipError_t build_iu4_grouped_tiles(const IU4Group* groups, int experts, int tile,
    IU4GroupTile* tiles, uint32_t* ntiles, uint32_t max_tiles, hipStream_t stream) {
    if (tile < 0 || tile > 2 || experts <= 0 || !groups || !tiles || !ntiles) return hipErrorInvalidValue;
    wmma_tiled::grouped_tiles_kernel<<<1, 1024, 0, stream>>>(groups, experts, iu4_tiles[tile].bm, tiles, ntiles, max_tiles);
    return hipGetLastError();
}
namespace {
hipError_t query_grouped_bits(int tile, bool transform, IU4Resources* resources) {
    switch (tile * 2 + int(transform)) {
        case 0: return wmma_tiled::query_grouped<0, false>(resources);
        case 1: return wmma_tiled::query_grouped<0, true>(resources);
        case 2: return wmma_tiled::query_grouped<1, false>(resources);
        case 3: return wmma_tiled::query_grouped<1, true>(resources);
        case 4: return wmma_tiled::query_grouped<2, false>(resources);
        case 5: return wmma_tiled::query_grouped<2, true>(resources);
        default: return hipErrorInvalidValue;
    }
}
hipError_t launch_grouped_bits(const IU4GroupedOperands& o, const IU4GroupTile* tiles,
    const uint32_t* ntiles, uint32_t max_tiles, float* output, int inputs, int outputs, hipStream_t stream, int tile) {
    const bool transform = o.col_scale != nullptr;
    switch (tile * 2 + int(transform)) {
        case 0: return wmma_tiled::launch_grouped<0, false>(o, tiles, ntiles, max_tiles, output, inputs, outputs, stream);
        case 1: return wmma_tiled::launch_grouped<0, true>(o, tiles, ntiles, max_tiles, output, inputs, outputs, stream);
        case 2: return wmma_tiled::launch_grouped<1, false>(o, tiles, ntiles, max_tiles, output, inputs, outputs, stream);
        case 3: return wmma_tiled::launch_grouped<1, true>(o, tiles, ntiles, max_tiles, output, inputs, outputs, stream);
        case 4: return wmma_tiled::launch_grouped<2, false>(o, tiles, ntiles, max_tiles, output, inputs, outputs, stream);
        default: return wmma_tiled::launch_grouped<2, true>(o, tiles, ntiles, max_tiles, output, inputs, outputs, stream);
    }
}
} // namespace
hipError_t query_wmma_iu4_grouped_resources(int tile, bool transform, IU4Resources* resources, int group) {
    if (!resources || group != 64 || tile < 0 || tile > 2) return hipErrorInvalidValue;
    return query_grouped_bits(tile, transform, resources);
}
hipError_t launch_wmma_tiled_iu4_grouped(const IU4GroupedOperands& o,
    const IU4GroupTile* tiles, const uint32_t* ntiles, uint32_t max_tiles, float* output,
    int inputs, int outputs, hipStream_t stream, int tile) {
    if (!iu4_geometry_ok(inputs, outputs, 1, tile) || o.group != 64 || !tiles || !ntiles ||
        !o.a_q || !o.w_q || !o.a_scale || !o.w_scale || !output || !o.a_mul ||
        (uintptr_t(o.a_q) & 15U) || (uintptr_t(o.w_q) & 15U) || (o.w_expert_bytes & 15U) ||
        o.w_expert_bytes < size_t(inputs) * size_t(outputs) / 2U ||
        o.w_scale_expert < size_t(outputs) * size_t(inputs / 64) ||
        (uintptr_t(o.a_scale) & 3U) || (uintptr_t(o.w_scale) & 1U) || (uintptr_t(output) & 3U))
        return hipErrorInvalidValue;
    return launch_grouped_bits(o, tiles, ntiles, max_tiles, output, inputs, outputs, stream, tile);
}
hipError_t convert_exl3_iu4(const uint32_t* const* trellis, int bits, int experts, int inputs, int outputs,
    uint8_t* wq, _Float16* ws, size_t wq_expert, size_t ws_expert, hipStream_t stream, int scale_rule) {
    if (!trellis || bits < 2 || bits > 8 || experts <= 0 || inputs <= 0 || inputs % 64 || outputs <= 0 || outputs % 16 || !ws ||
        !iu4_scale_rule_valid(scale_rule))
        return hipErrorInvalidValue;
    const size_t items = size_t(inputs / 64) * size_t(outputs / 16);
    const dim3 grid(unsigned((items + 7U) / 8U), unsigned(experts));
    switch (bits) {  // every EXL3 width an engine stores (MiniMax: routed experts 3-bit, shared expert 5-bit)
        case 2: wmma_tiled::exl3_iu4_convert_kernel<2><<<grid, 256, 0, stream>>>(trellis, inputs, outputs, wq, ws, wq_expert, ws_expert, scale_rule); break;
        case 3: wmma_tiled::exl3_iu4_convert_kernel<3><<<grid, 256, 0, stream>>>(trellis, inputs, outputs, wq, ws, wq_expert, ws_expert, scale_rule); break;
        case 4: wmma_tiled::exl3_iu4_convert_kernel<4><<<grid, 256, 0, stream>>>(trellis, inputs, outputs, wq, ws, wq_expert, ws_expert, scale_rule); break;
        case 5: wmma_tiled::exl3_iu4_convert_kernel<5><<<grid, 256, 0, stream>>>(trellis, inputs, outputs, wq, ws, wq_expert, ws_expert, scale_rule); break;
        case 6: wmma_tiled::exl3_iu4_convert_kernel<6><<<grid, 256, 0, stream>>>(trellis, inputs, outputs, wq, ws, wq_expert, ws_expert, scale_rule); break;
        case 7: wmma_tiled::exl3_iu4_convert_kernel<7><<<grid, 256, 0, stream>>>(trellis, inputs, outputs, wq, ws, wq_expert, ws_expert, scale_rule); break;
        default: wmma_tiled::exl3_iu4_convert_kernel<8><<<grid, 256, 0, stream>>>(trellis, inputs, outputs, wq, ws, wq_expert, ws_expert, scale_rule); break;
    }
    return hipGetLastError();
}
hipError_t launch_wmma_tiled_f16_plan(const _Float16* a, const _Float16* w,
    float* output, int inputs, int outputs, int rows, hipStream_t stream, int tile) {
    if (tile == -1) tile = iu4_policy_tile(rows);
    if (!iu4_geometry_ok(inputs, outputs, rows, tile) || !a || !w || !output ||
        (uintptr_t(a) & 15U) || (uintptr_t(w) & 15U) || (uintptr_t(output) & 3U)) return hipErrorInvalidValue;
    return wmma_tiled::launch_plain_plan<PackedF16>(reinterpret_cast<const uint32_t*>(w),
        a, {}, output, inputs, outputs, rows, stream, tile, nullptr);
}
hipError_t quantise_wmma_iu4_activations(const _Float16* input, uint8_t* packed,
    float* scales, int inputs, int rows, hipStream_t stream, int group) {
    if (!iu4_group_ok(group) || inputs <= 0 || inputs % group || rows <= 0 || !input || !packed || !scales ||
        (uintptr_t(input) & 1U) || (uintptr_t(packed) & 15U) || (uintptr_t(scales) & 3U)) return hipErrorInvalidValue;
    const size_t groups = size_t(rows) * size_t(inputs / group);
    if ((groups + 7U) / 8U > size_t(INT32_MAX)) return hipErrorInvalidValue;
    const unsigned blocks = unsigned((groups + 7U) / 8U);
    switch (group) {
        case 64: wmma_tiled::quantise_activations<64><<<blocks, 256, 0, stream>>>(input, packed, scales, inputs, rows); break;
        case 128: wmma_tiled::quantise_activations<128><<<blocks, 256, 0, stream>>>(input, packed, scales, inputs, rows); break;
        default: wmma_tiled::quantise_activations<256><<<blocks, 256, 0, stream>>>(input, packed, scales, inputs, rows); break;
    }
    return hipGetLastError();
}

bool wmma_tiled_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRIX_TRELLIS_WMMA_TILED");
        return !(v && std::strcmp(v, "0") == 0);
    }();
    return enabled;
}
int wmma_tiled_fused_mode() {
    static const int mode = [] {
        // Default "auto": fused only when the grid has more blocks than CUs (one-wave grids keep the separate
        // transform). Measured over 12 process pairs on two devices: no fused cell significantly (95 %) and materially
        // (> 1 %) slower than the separate transform, pooled or per node.
        // "0" (or "off"/"false") restores the separate transform everywhere, "1" fuses whenever possible.
        // Same bits in every mode.
        const char* v = std::getenv("STRIX_TRELLIS_WMMA_TILED_FUSED");
        if (v && (std::strcmp(v, "0") == 0 || std::strcmp(v, "off") == 0 || std::strcmp(v, "false") == 0)) return 0;
        if (v && std::strcmp(v, "1") == 0) return 1;
        return 2;
    }();
    return mode;
}
bool wmma_tiled_fused_output_enabled() { return wmma_tiled_fused_mode() != 0; }
static int fused_min_blocks() {
    // Default: more blocks than CUs (WGPs) on the current device. With one resident block per CU the fused
    // epilogue's serial tail is exposed (fused neutral up to 20 blocks on gfx1151, faster from 24-27 up).
    // Assumes one resident block per CU, true for the 8-wave policy tiles 0/4/6; cached per device id.
    static const int forced = [] {
        const char* v = std::getenv("STRIX_TRELLIS_WMMA_TILED_FUSED_MIN_BLOCKS");
        const int x = v ? std::atoi(v) : 0;
        return x > 0 ? x : 0;
    }();
    if (forced) return forced;
    static std::atomic<int> per_device[64] = {};   // CU count + 1 per device id, 0 = not queried yet
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) return 32;
    int n = per_device[dev].load(std::memory_order_relaxed);
    if (n == 0) {
        int cus = 0;
        n = hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, dev) == hipSuccess && cus > 0 ? cus + 1 : 32;
        per_device[dev].store(n, std::memory_order_relaxed);
    }
    return n;
}
static bool tiled_geometry_ok(const DeviceWeights& w, int rows, int splits) {
    return w.trellis && rows > 0 && splits > 0 && w.input > 0 && w.output > 0 &&
        w.input % 128 == 0 && w.output % 128 == 0 && w.bits >= 1 && w.bits <= 8 &&
        w.codebook >= 0 && w.codebook <= 2;
}
bool wmma_tiled_fuses_output(const DeviceWeights& w, int rows, int splits, int tile) {
    if (!tiled_geometry_ok(w, rows, splits) || splits != 1 || !w.svh) return false;
    if (tile < 0) tile = wmma_tiled::policy_tile(w, rows, splits);
    if (tile < 0 || tile >= wmma_tiled::config_count) return false;
    const auto& c = wmma_tiled::configs[tile];
    return c.bn % 128 == 0 && w.output % c.bn == 0;
}
int wmma_tiled_raster_rows(int tile) {
    return wmma_tiled::raster_rows(wmma_tiled::configs[tile].raster);
}
bool wmma_tiled_fuse_selected(const DeviceWeights& w, int rows, int splits, int tile) {
    const int mode = wmma_tiled_fused_mode();
    if (mode == 0) return false;
    if (tile < 0) tile = wmma_tiled::policy_tile(w, rows, splits);
    if (!wmma_tiled_fuses_output(w, rows, splits, tile)) return false;
    if (mode == 1) return true;
    const auto& c = wmma_tiled::configs[tile];
    const long blocks = long((rows + c.bm - 1) / c.bm) * (w.output / c.bn) * splits;
    return blocks >= fused_min_blocks();
}
#ifdef STRIX_TRELLIS_WMMA_PROBE
hipError_t launch_wmma_tiled_probe(const DeviceWeights& w, const _Float16* input, float* partial, int rows,
    int splits, hipStream_t stream, int tile, int probe) {
    if (!tiled_geometry_ok(w, rows, splits) || w.codebook != 2 || tile < 0 || tile >= wmma_tiled::config_count ||
        w.output % wmma_tiled::configs[tile].bn) return hipErrorNotSupported;
    if (w.bits == 4) return wmma_tiled::launch_probe_bits<4>(w, input, partial, rows, splits, stream, tile, probe);
    if (w.bits == 5) return wmma_tiled::launch_probe_bits<5>(w, input, partial, rows, splits, stream, tile, probe);
    if (w.bits == 6) return wmma_tiled::launch_probe_bits<6>(w, input, partial, rows, splits, stream, tile, probe);
    return hipErrorNotSupported;
}
#endif
hipError_t launch_wmma_tiled(const DeviceWeights& w, const _Float16* input, float* partial, int rows,
    int splits, hipStream_t stream, const WmmaFusedOutput* fused, int tile, int raster) {
    if (!tiled_geometry_ok(w, rows, splits) || !input || (uintptr_t(input) & 15u)) return hipErrorNotSupported;
    if (tile < 0) tile = wmma_tiled::policy_tile(w, rows, splits);
    if (tile < 0 || tile >= wmma_tiled::config_count) return hipErrorNotSupported;
    if (w.output % wmma_tiled::configs[tile].bn) return hipErrorNotSupported;   // whole N blocks only
    if (fused && fused->output) {
        if (!wmma_tiled_fuses_output(w, rows, splits, tile)) return hipErrorNotSupported;
    } else if (!partial) return hipErrorNotSupported;
#define BITS_CASE(B) case B: switch (w.codebook) { \
        case 0: return wmma_tiled::launch_bits<B, 0>(w, input, partial, rows, splits, stream, fused, tile, raster); \
        case 1: return wmma_tiled::launch_bits<B, 1>(w, input, partial, rows, splits, stream, fused, tile, raster); \
        case 2: return wmma_tiled::launch_bits<B, 2>(w, input, partial, rows, splits, stream, fused, tile, raster); } break
    switch (w.bits) {
        BITS_CASE(1); BITS_CASE(2); BITS_CASE(3); BITS_CASE(4);
        BITS_CASE(5); BITS_CASE(6); BITS_CASE(7); BITS_CASE(8);
    }
#undef BITS_CASE
    return hipErrorNotSupported;
}
} // namespace strix::core
