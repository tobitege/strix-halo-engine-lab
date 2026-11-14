#pragma once
// The library WMMA prefill core (moved verbatim from wmma_tiled.hip, header-only so engines
// instantiate it with their own adapters instead of keeping private copies).
//   Contract: Operand policy + fold (0 = one WMMA chain per K split from +0; F > 0 =
//             chain F K16 tiles from +0, fold each chain into the FP32 total with v_add_f32,
//             ascending: the GLM_FAST_WX identity, glm-pf-wmma-x).
//   Adapter:  NoAdapter = library operands; otherwise the model plug-in (row gather, tile
//             loop, post-H128 epilogue), see NoAdapter below.
#include "wmma_tiled.hpp"
#include "trellis_codec.hpp"
#include "hadamard.hpp"
#include <type_traits>
#include "integer_operand.hpp"
#include "integer_grouped_schedule.hpp"
#include "load_cohort.hpp"

namespace strix::core {
namespace wmma_tiled {
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Half8 = _Float16 __attribute__((ext_vector_type(8)));
// A fragment is loaded as 8 dwords (2 x ds_load_b128) and bit-cast to Half16 only at
// the WMMA: loading <16 x half> directly makes gfx11 true16 codegen split and re-pack
// every dword (v_lshrrev_b32 + v_mov_b16 per dword). Bits are identical either way.
typedef uint32_t Frag __attribute__((ext_vector_type(8), aligned(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
using U4 = uint32_t __attribute__((ext_vector_type(4)));
using U2 = uint32_t __attribute__((ext_vector_type(2)));

// The three ring words of one lane, loaded early and combined at decode time,
// so the loads stay in flight across a stage of WMMA: the shared codec's Lane
// reader (trellis_codec.hpp LaneWords / TileStates) behind the names used here.
template<int Bits> struct RawLane : trellis::LaneWords<Bits> {
    __device__ __forceinline__ RawLane() : trellis::LaneWords<Bits>(0, 0, 0) {}
    __device__ __forceinline__ void load(const uint32_t* data) {
        static_cast<trellis::LaneWords<Bits>&>(*this) =
            trellis::LaneWords<Bits>::issue_at(data, trellis::LaneGeom<Bits>(threadIdx.x & 31U));
    }
};
template<int Bits> struct Stream : trellis::TileStates<Bits> {
    __device__ __forceinline__ explicit Stream(const RawLane<Bits>& r)
        : trellis::TileStates<Bits>(trellis::align_words<Bits>(r, trellis::LaneGeom<Bits>(threadIdx.x & 31U).align)) {}
};
// Two consecutive K values of one column; low half = first state.
template<int CB> __device__ __forceinline__ uint32_t decode_pair(uint32_t first, uint32_t second) {
    if constexpr (CB == 2) {
        return __builtin_bit_cast(uint32_t, trellis::Mul1::pair(first, second));
    } else {
        _Float16 x = _Float16(trellis::codebook_value(first, CB)), y = _Float16(trellis::codebook_value(second, CB));
        return uint32_t(__builtin_bit_cast(uint16_t, x)) | (uint32_t(__builtin_bit_cast(uint16_t, y)) << 16);
    }
}
// The same pair from states as the lane reader extracted them: mul1 reads only the low 16 bits, so no mask.
template<int CB> __device__ __forceinline__ uint32_t decode_pair(trellis::State16 first, trellis::State16 second) {
    if constexpr (CB == 2) return __builtin_bit_cast(uint32_t, trellis::mul1_pair(first, second));
    else return decode_pair<CB>(uint32_t(first), uint32_t(second));
}

template<int BM, int BN, int WM, int WN, class Operand = PackedF16> struct Geometry {
    static constexpr int KT = Operand::stage_tiles, BK = 16 * KT;
    static constexpr bool iu4 = std::is_same_v<Operand, PackedIU4>;
    // IU4 row swizzle (below) avoids padding; F16 preserves the original layout.
    static constexpr int row_bytes = iu4 ? BK / 2 : (BK + 8) * 2;
    static constexpr int waves_m = BM / WM, waves_n = BN / WN, waves = waves_m * waves_n;
    static constexpr int threads = [] {
        if constexpr (is_integer_operand<Operand>::value) {
            if constexpr (Operand::Plan::structure == IntegerStructure::dense_tall) return 256;
            else if constexpr (Operand::Plan::structure == IntegerStructure::dense_wide) return 128;
            else return waves * 32 * 8;
        } else return waves * 32;
    }();
    static constexpr int MT = WM / 16, NT = WN / 16;
    static constexpr int chunks_per_row = BK * Operand::element_bits / 128;
    static constexpr int a_chunks = BM * chunks_per_row;     // 16-byte chunks per stage
    static constexpr int a_per_thread = (a_chunks + threads - 1) / threads;
    static constexpr int q_tiles = (BN / 16) * KT;            // packed tiles per stage
    static constexpr int q_per_wave = (q_tiles + waves - 1) / waves;
    // IU4 stages both K64 groups' FP32 scales per stage: [2][BM] activation (rows
    // permuted so a lane's eight rows are contiguous) and [2][BN] weight scales.
    static constexpr int scale_bytes = iu4 ? (BM + BN) * 2 * 4 : 0;
    static constexpr int stage_bytes = is_integer_operand<Operand>::value ? 4 * BM * BN * 4 : (BM + BN) * row_bytes + scale_bytes;
    static constexpr int lds_bytes = [] {
        if constexpr (is_integer_operand<Operand>::value) {
            if constexpr (Operand::Plan::structure == IntegerStructure::grouped_chunk) return int(Operand::grouped_bytes);
            if constexpr (Operand::Plan::structure == IntegerStructure::dense_tall || Operand::Plan::structure == IntegerStructure::dense_wide) return 1;
        }
        return 2 * stage_bytes;
    }();
    static_assert(BM % WM == 0 && BN % WN == 0 && WM % 16 == 0 && WN % 16 == 0);
};

// Grouped IU4 (MoE): the CTA's tile, expert-offset operands and row maps (wmma_tiled.hpp).
struct GroupedArgs {
    const IU4GroupTile* tiles = nullptr;
    const uint32_t* ntiles = nullptr;
    size_t w_expert_bytes = 0, w_scale_expert = 0;
    const uint16_t* const* col_scale = nullptr;
    const uint32_t* map = nullptr;
    uint32_t a_mul = 1, a_add = 0, o_mul = 1, o_add = 0;
};
// Accumulation fold of a contract: 0 (TiledContract) = one WMMA chain per K split.
template<class C, class = void> struct contract_fold { static constexpr int value = 0; };
template<class C> struct contract_fold<C, std::void_t<decltype(C::fold)>> { static constexpr int value = C::fold; };
// GLM_FAST_WX (glm-pf-wmma-x): mul1 trellis F16 operands; per (row, output) one WMMA chain
// over each K64 (four K16 tiles, ascending) from +0, folded into an FP32 total that starts at
// +0 with v_add_f32 in ascending K; no K splits; H128 + svh epilogue. A tile/plan change keeps
// these bits (same per-element sequence).
template<int Bits> struct WmmaXContract {
    using Operand = TrellisF16<Bits, 2>;
    static constexpr int fold = 4;
    static constexpr unsigned version = 1;
};
// Model adapter (plug-in surface). NoAdapter = the library operands (input rows, output
// transform). An active adapter supplies, as device members of a trivially copyable struct:
//   struct Tile { int rows, first_col; const uint32_t* packed; ... };  (adapter context)
//   kTiles = true:  bool next(Tile&, unsigned iter, int bn) - the CTA's iter-th tile (uniform;
//                   false ends the CTA). blockIdx.z belongs to the adapter (no K splits).
//   kTiles = false: void base(Tile&) - library raster over `rows`, packed from the kernel arg.
//   const _Float16* a_row(const Tile&, int r)  - FP16 row r < rows of the tile (gather)
//   Tile also carries int split, splits (the tile's K partition, library boundaries).
//   void partial(const Tile&, int r, int col, float v)  - unfused: the raw FP32 split partial
//   void store(const Tile&, int r, int col0, float (&v)[4], int lane)
//        - 128 outputs col0 + lane + 32 j after hadamard128 (FP32 total, no svh): svh scale,
//          activation, combine... Fused epilogue only.
// kPassRow: global-A row offset added per chain pass (paired gate/up reads an interleaved
// pre-quantised image, row = map*a_mul + a_add + pass*kPassRow). NoChain: 0 (unchanged).
struct NoChain {
    static constexpr bool kActive = false;
    static constexpr int kPasses = 1;
    static constexpr uint32_t kPassRow = 0;
    static constexpr bool kSaveRegs = false;
};
struct NoAdapter {
    static constexpr bool kActive = false, kTiles = false;
    struct Tile {};
};
// Probe (benchmark builds only, timing decomposition, wrong outputs):
// 1 = no Q decode/LDS store, 2 = no WMMA, 3 = WMMA + fragment reads only.
template<class Contract> constexpr unsigned core_waves_per_eu() {
    if constexpr (is_integer_operand<typename Contract::Operand>::value) return Contract::Operand::Plan::waves_per_eu;
    else return 0;
}
template<class Contract, int BM, int BN, int WM, int WN, bool Fused, int Probe = 0,
    bool IU4Audit = false, bool IU4FullRows = false, int IU4Group = 64, bool IU4Grouped = false,
    class Adapter = NoAdapter, class Chain = NoChain>
__global__ __launch_bounds__((Geometry<BM, BN, WM, WN, typename Contract::Operand>::threads))
__attribute__((amdgpu_waves_per_eu(core_waves_per_eu<Contract>()))) void tiled_core(
    const uint32_t* __restrict__ packed, const _Float16* __restrict__ input,
    float* __restrict__ partial, int inputs, int outputs, int rows, int splits,
    const uint16_t* __restrict__ scales, const float* __restrict__ bias,
    float* __restrict__ output, float alpha, float beta, int raster_m,
    const uint8_t* a_q = nullptr, const float* a_scale = nullptr,
    const _Float16* w_scale = nullptr, int32_t* group_sums = nullptr, GroupedArgs gp = {}, Adapter ad = {}, Chain chain = {}) {
    using Operand = typename Contract::Operand;
    using G = Geometry<BM, BN, WM, WN, Operand>;
    constexpr bool Integer = is_integer_operand<Operand>::value;
    constexpr bool IU4 = G::iu4;
    constexpr bool IU4Pack = IU4;
    constexpr bool DenseF16 = std::is_same_v<Operand, PackedF16>;
    // The trellis-only register stream is never instantiated for plain operands.
    constexpr int Bits = [] { if constexpr (Integer || IU4Pack || DenseF16) return 4; else return Operand::bits; }();
    constexpr int CB = [] { if constexpr (Integer || IU4Pack || DenseF16) return 2; else return Operand::codebook; }();
    constexpr int KT = G::KT, STRIDE = G::row_bytes / 2;
    constexpr int MT = G::MT, NT = G::NT;
    constexpr auto structure = [] {
        if constexpr (Integer) return Operand::Plan::structure;
        else return IntegerStructure::per_group;
    }();
    // Dense plans own one CTA tile and handle their own row/output tails. Enter
    // them directly, as their schedule requires no adapter iteration or raster.
    if constexpr (structure == IntegerStructure::dense_tall) {
#include "integer_tall_mainloop.inc"
    } else if constexpr (structure == IntegerStructure::dense_wide) {
#include "integer_wide_mainloop.inc"
    } else {
    __shared__ __attribute__((aligned(16))) uint8_t lds[G::lds_bytes];
    const int thread = int(threadIdx.x), lane = thread & 31;
    const int wave = __builtin_amdgcn_readfirstlane(thread / 32);
    const int spatial_wave = Integer ? wave % G::waves : wave;
    const int wave_m = spatial_wave / G::waves_n, wave_n = spatial_wave % G::waves_n;
    constexpr int Fold = contract_fold<Contract>::value;
    constexpr bool AD = Adapter::kActive;
    static_assert(!AD || (Integer || (!IU4 && !DenseF16 && Probe == 0 && (!Fused || BN % 128 == 0))), "adapters: trellis F16 operands");
    static_assert(Fold == 0 || (!IU4 && !DenseF16));
    static_assert(!Chain::kActive || (IU4Grouped && Fused && IU4Group == 64 && !AD));
    // Gate values stay in each lane's registers across the paired gate/up mainloops.
    constexpr int chain_groups = (G::waves_m * 16 * (BN / 128) + G::waves - 1) / G::waves;
    float chain_saved[Chain::kSaveRegs ? MT * chain_groups * 4 : 1];
    const auto* chain_packed = packed;
    const auto* chain_wscale = w_scale;
    for (int chain_pass = 0; chain_pass < Chain::kPasses; ++chain_pass) {
    if constexpr (Chain::kActive) { packed = chain_packed; w_scale = chain_wscale; }
    // Adapter tile loop: one pass for library launches; kTiles adapters iterate their CTA's tiles.
    for (unsigned iter = 0;; ++iter) {
    [[maybe_unused]] typename Adapter::Tile at{};
    if constexpr (Adapter::kTiles) {
        if (!ad.next(at, iter, BN)) return;   // uniform
        rows = at.rows; packed = at.packed; splits = at.splits;
    } else if (iter) { if constexpr (Chain::kActive) break; else return; }
    // Grouped rasterisation: consecutive blocks cover raster_m row blocks of one column
    // block, then the next column block, so resident blocks share A rows and Q slabs.
    // Only the block -> tile mapping changes; every tile is computed exactly as before.
    int first_row = 0, first_col = 0;
    [[maybe_unused]] uint32_t map_first = 0;
    static_assert(!IU4Grouped || (G::iu4 && !IU4Audit && !IU4FullRows && Probe == 0));
    if constexpr (IU4Grouped) {
        // Tile-major, column blocks of one tile consecutive: resident CTAs share the tile's A rows
        // and successive tiles of one expert reuse its weight panel. Uniform early exit.
        const uint32_t nb = uint32_t(outputs / BN), ti = blockIdx.x / nb;
        if (ti >= *gp.ntiles) return;
        const IU4GroupTile t = gp.tiles[ti];
        const uint32_t expert = uint32_t(__builtin_amdgcn_readfirstlane(int(t.expert)));
        map_first = uint32_t(__builtin_amdgcn_readfirstlane(int(t.first)));
        rows = __builtin_amdgcn_readfirstlane(int(t.rows));
        first_col = int(blockIdx.x % nb) * BN;
        packed += size_t(expert) * (gp.w_expert_bytes / 4U);
        w_scale += size_t(expert) * gp.w_scale_expert;
        if constexpr (Fused) scales = gp.col_scale[expert];
        if constexpr (Chain::kActive) chain.bind(expert, chain_pass, packed, w_scale, scales);
        (void)raster_m;
    } else if constexpr (!Adapter::kTiles) {
    const int blocks_m = (rows + BM - 1) / BM;
    const int pid = int(blockIdx.x), per_group = raster_m * (outputs / BN);
    const int group = pid / per_group, first_m = group * raster_m;
    const int group_m = blocks_m - first_m < raster_m ? blocks_m - first_m : raster_m;
    const int local = pid - group * per_group;
    first_row = (first_m + local % group_m) * BM; first_col = (local / group_m) * BN;
    }
    if constexpr (Adapter::kTiles) { first_row = 0; first_col = at.first_col; }
    else if constexpr (AD) ad.base(at);
    // Grouped: source / destination row of tile row r (r < rows).
    [[maybe_unused]] auto map_row = [&](int r, uint32_t mul, uint32_t add) -> uint32_t {
        const uint32_t p = map_first + uint32_t(r);
        return (gp.map ? gp.map[p] : p) * mul + add;
    };
    if constexpr (Integer) {
#include "integer_mainloop.inc"
    } else {
    const int tiles_n = outputs / 16, tiles_k = inputs / 16;
    // Same split bounds as wmma_core. Pinned in SGPRs so LLVM cannot rematerialise the
    // 64-bit division inside the stage loop.
    // Adapters own blockIdx.z; a tile-loop adapter names the tile's K split (Tile::split of Tile::splits).
    const int64_t split_z = [&]() -> int64_t { if constexpr (Adapter::kTiles) return int64_t(at.split); else return AD ? 0 : int64_t(blockIdx.z); }();
    int begin = int(int64_t(tiles_k) * split_z / splits);
    int end = int(int64_t(tiles_k) * (split_z + 1) / splits);
    asm volatile("" : "+s"(begin), "+s"(end));
    const int stages = (end - begin + KT - 1) / KT;

    U4 a_reg[G::a_per_thread];
    using WeightStage = std::conditional_t<IU4Pack || DenseF16, U4, RawLane<Bits>>;
    constexpr int b_chunks = BN * G::chunks_per_row;
    constexpr int b_per_thread = (b_chunks + G::threads - 1) / G::threads;
    WeightStage q_reg[IU4Pack || DenseF16 ? b_per_thread : G::q_per_wave];
    auto row_offset = [](int r, int byte) {
        if constexpr (IU4) return iu4_lds_offset(r, byte);
        else return r * G::row_bytes + byte;
    };

    uint32_t probe_sink = 0;
    // IU4 groups: GK K16 tiles per scale group; a stage has SLOTS flush segments of SEG
    // tiles. Each stage stages one FP32 scale per segment in LDS (g >= BK: same group).
    constexpr int GK = IU4Group / 16, SEG = GK < KT ? GK : KT, SLOTS = KT / SEG;
    static_assert(!IU4 || (IU4Group == 64 || IU4Group == 128 || IU4Group == 256));
    constexpr int sa_per_thread = IU4 ? (SLOTS * BM + G::threads - 1) / G::threads : 1;
    constexpr int sw_per_thread = IU4 ? (SLOTS * BN + G::threads - 1) / G::threads : 1;
    // sw_reg keeps the F16 weight scale as loaded: converting in fetch() would consume the load before the
    // stage's WMMA and stall every wave on memory once per stage; store() converts (same FP32 value).
    float sa_reg[sa_per_thread] = {};
    _Float16 sw_reg[sw_per_thread] = {};
    // Grouped: gathered A rows, resolved once (clamped to the tile's last row).
    [[maybe_unused]] uint32_t a_src[IU4Grouped ? G::a_per_thread : 1], sa_src[IU4Grouped ? sa_per_thread : 1];
    if constexpr (IU4Grouped) {
        const uint32_t a_add = gp.a_add + uint32_t(chain_pass) * Chain::kPassRow;
        #pragma unroll
        for (int i = 0; i < G::a_per_thread; ++i) {
            const int r = (thread + i * G::threads) / G::chunks_per_row;
            a_src[i] = map_row(r < rows ? r : rows - 1, gp.a_mul, a_add);
        }
        #pragma unroll
        for (int i = 0; i < sa_per_thread; ++i) {
            const int r = (thread + i * G::threads) / SLOTS;
            sa_src[i] = map_row(r < rows ? r : rows - 1, gp.a_mul, a_add);
        }
    }
    // Adapter rows: the gathered row pointers, resolved once per tile.
    [[maybe_unused]] const _Float16* a_ptr[AD ? G::a_per_thread : 1] = {};
    if constexpr (AD) {
        #pragma unroll
        for (int i = 0; i < G::a_per_thread; ++i) {
            const int chunk = thread + i * G::threads, row = first_row + chunk / G::chunks_per_row;
            if (chunk < G::a_chunks && row < rows) a_ptr[i] = ad.a_row(at, row);
        }
    }
    // Raw loads of one stage. cohort_a: activations (+ IU4 row scales); cohort_b: weights (+ IU4 column
    // scales). No destination register is read here; store() is each cohort's first use. Which cohorts
    // are issued before the stage's matrix work is the contract's load schedule (load_cohort.hpp).
    using Schedule = typename pipeline::contract_schedule<Contract>::type;
    static_assert(Schedule::lookahead == 1 && Schedule::lds_stages * G::stage_bytes == G::lds_bytes,
        "one register stage of lookahead, two LDS stage buffers");
    auto fetch = [&](int stage, auto cohort_set) {
        using Cohorts = decltype(cohort_set);
        if constexpr (Probe == 3) return;
        const int k0 = begin + stage * KT;
        if constexpr (IU4) {
            // 32-bit offsets from uniform bases; no EXEC guards. Rows past the end and the
            // missing group of a K tail stage load clamped (valid) bytes that are never stored.
            static_assert(G::a_chunks % G::threads == 0 && b_chunks % G::threads == 0);
            const int groups = inputs / IU4Group;
            const auto* wq = reinterpret_cast<const uint8_t*>(packed);
            if constexpr (pipeline::has<Cohorts>(pipeline::cohort_a))
            #pragma unroll
            for (int i = 0; i < G::a_per_thread; ++i) {
                const int chunk = thread + i * G::threads;
                const int r = chunk / G::chunks_per_row, part = chunk % G::chunks_per_row;
                int row = first_row + r;
                if constexpr (!IU4FullRows) row = row < rows ? row : rows - 1;
                if constexpr (IU4Grouped) row = int(a_src[i]);
                int kk = k0 * 16 + part * 32;
                kk = kk < inputs ? kk : inputs - 32;
                a_reg[i] = *reinterpret_cast<const U4*>(a_q + (uint32_t(row) * uint32_t(inputs / 2) + uint32_t(kk / 2)));
            }

            if constexpr (pipeline::has<Cohorts>(pipeline::cohort_b))
            #pragma unroll
            for (int i = 0; i < b_per_thread; ++i) {
                const int chunk = thread + i * G::threads;
                const int c = chunk / G::chunks_per_row, part = chunk % G::chunks_per_row;
                const int col = first_col + c;
                const uint32_t column = uint32_t(col / 16) * 128U + uint32_t(col % 16) * 8U;
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int kt = k0 + part * 2 + j;
                    kt = kt < tiles_k ? kt : tiles_k - 1;
                    const U2 pair = *reinterpret_cast<const U2*>(wq + (uint32_t(kt) * uint32_t(tiles_n) * 128U + column));
                    q_reg[i][2 * j] = pair[0]; q_reg[i][2 * j + 1] = pair[1];
                }
            }
            
            if constexpr (pipeline::has<Cohorts>(pipeline::cohort_a))
            #pragma unroll
            for (int i = 0; i < sa_per_thread; ++i) {
                const int e = thread + i * G::threads;
                if ((SLOTS * BM) % G::threads == 0 || e < SLOTS * BM) {
                    int row = first_row + e / SLOTS;
                    if constexpr (!IU4FullRows) row = row < rows ? row : rows - 1;
                    if constexpr (IU4Grouped) row = int(sa_src[i]);
                    int g = (k0 + (e % SLOTS) * SEG) / GK;
                    g = g < groups ? g : groups - 1;
                    sa_reg[i] = a_scale[uint32_t(row) * uint32_t(groups) + uint32_t(g)];
                }
            }
            if constexpr (pipeline::has<Cohorts>(pipeline::cohort_b))
            #pragma unroll
            for (int i = 0; i < sw_per_thread; ++i) {
                const int e = thread + i * G::threads;
                if ((SLOTS * BN) % G::threads == 0 || e < SLOTS * BN) {
                    int g = (k0 + (e % SLOTS) * SEG) / GK;
                    g = g < groups ? g : groups - 1;
                    sw_reg[i] = w_scale[uint32_t(first_col + e / SLOTS) * uint32_t(groups) + uint32_t(g)];
                }
            }
        } else {
        if constexpr (pipeline::has<Cohorts>(pipeline::cohort_a))
        #pragma unroll
        for (int i = 0; i < G::a_per_thread; ++i) {
            const int chunk = thread + i * G::threads;
            const int r = chunk / G::chunks_per_row, part = chunk % G::chunks_per_row;
            const int row = first_row + r;
            const bool valid = chunk < G::a_chunks && row < rows && k0 + part / 2 < end;
            if constexpr (AD) a_reg[i] = valid ? *reinterpret_cast<const U4*>(a_ptr[i] + size_t(k0) * 16U + size_t(part) * 8U) : U4{0, 0, 0, 0};
            else
            a_reg[i] = valid ? *reinterpret_cast<const U4*>(input + size_t(row) * size_t(inputs) + size_t(k0) * 16U + size_t(part) * 8U) : U4{0, 0, 0, 0};
        }
        if constexpr (DenseF16) {
            if constexpr (pipeline::has<Cohorts>(pipeline::cohort_b))
            #pragma unroll
            for (int i = 0; i < b_per_thread; ++i) {
                const int chunk = thread + i * G::threads;
                const int c = chunk / G::chunks_per_row, part = chunk % G::chunks_per_row;
                U4 values = {};
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    constexpr int values_per_word = 2;
                    const int kk = k0 * 16 + (part * 4 + j) * values_per_word;
                    if (chunk < b_chunks && kk < inputs) {
                        const size_t offset = ((size_t(kk / 16) * size_t(tiles_n) + size_t((first_col + c) / 16)) * 16U +
                            size_t((first_col + c) % 16)) * size_t(16 / values_per_word) + size_t((kk % 16) / values_per_word);
                        values[j] = packed[offset];
                    }
                }
                q_reg[i] = values;
            }
        } else {
        if constexpr (pipeline::has<Cohorts>(pipeline::cohort_b))
        #pragma unroll
        for (int i = 0; i < G::q_per_wave; ++i) {
            const int t = wave + i * G::waves;            // wave-uniform
            const int kt = t / (BN / 16), nt = t % (BN / 16);
            if (t < G::q_tiles && k0 + kt < end && first_col / 16 + nt < tiles_n)
                q_reg[i].load(packed + (size_t(k0 + kt) * size_t(tiles_n) + size_t(first_col / 16 + nt)) * size_t(Bits * 8));
        }
        }
        }
    };
    // Publication of one stage into LDS buffer `buffer`: the first use of the cohorts' registers.
    auto store = [&](int stage, int buffer, auto cohort_set) {
        using Cohorts = decltype(cohort_set);
        if constexpr (Probe == 3) return;
        const int k0 = begin + stage * KT;
        auto* ab = lds + buffer * G::stage_bytes;
        _Float16* as = reinterpret_cast<_Float16*>(ab);
        _Float16* bs = as + BM * STRIDE;
        if constexpr (pipeline::has<Cohorts>(pipeline::cohort_a))
        #pragma unroll
        for (int i = 0; i < G::a_per_thread; ++i) {
            const int chunk = thread + i * G::threads;
            if (chunk < G::a_chunks) {
                const int r = chunk / G::chunks_per_row, part = chunk % G::chunks_per_row;
                if constexpr (IU4) {
                    *reinterpret_cast<U2*>(ab + row_offset(r, part * 16)) = __builtin_shufflevector(a_reg[i], a_reg[i], 0, 1);
                    *reinterpret_cast<U2*>(ab + row_offset(r, part * 16 + 8)) = __builtin_shufflevector(a_reg[i], a_reg[i], 2, 3);
                } else *reinterpret_cast<U4*>(ab + row_offset(r, part * 16)) = a_reg[i];
            }
        }
        if constexpr (IU4 || DenseF16) {
            if constexpr (pipeline::has<Cohorts>(pipeline::cohort_b))
            #pragma unroll
            for (int i = 0; i < b_per_thread; ++i) {
                const int chunk = thread + i * G::threads;
                if (chunk < b_chunks) {
                    const int c = chunk / G::chunks_per_row, part = chunk % G::chunks_per_row;
                    if constexpr (IU4) {
                        auto* base = ab + BM * G::row_bytes;
                        *reinterpret_cast<U2*>(base + row_offset(c, part * 16)) = __builtin_shufflevector(q_reg[i], q_reg[i], 0, 1);
                        *reinterpret_cast<U2*>(base + row_offset(c, part * 16 + 8)) = __builtin_shufflevector(q_reg[i], q_reg[i], 2, 3);
                    } else *reinterpret_cast<U4*>(ab + BM * G::row_bytes + row_offset(c, part * 16)) = q_reg[i];
                }
            }
            
            if constexpr (IU4) {
                float* sl = reinterpret_cast<float*>(ab + (BM + BN) * G::row_bytes);
                if constexpr (pipeline::has<Cohorts>(pipeline::cohort_a))
                #pragma unroll
                for (int i = 0; i < sa_per_thread; ++i) {
                    const int e = thread + i * G::threads;
                    if ((SLOTS * BM) % G::threads == 0 || e < SLOTS * BM) {
                        // Row r of a 16-row tile at (r%2)*8 + (r%16)/2: lane half h reads rows 2j+h as 8 floats.
                        const int r = e / SLOTS;
                        sl[(e % SLOTS) * BM + (r & ~15) + (r & 1) * 8 + ((r & 15) >> 1)] = sa_reg[i];
                    }
                }
                if constexpr (pipeline::has<Cohorts>(pipeline::cohort_b))
                #pragma unroll
                for (int i = 0; i < sw_per_thread; ++i) {
                    const int e = thread + i * G::threads;
                    if ((SLOTS * BN) % G::threads == 0 || e < SLOTS * BN) sl[SLOTS * BM + (e % SLOTS) * BN + e / SLOTS] = float(sw_reg[i]);
                }
            }
        } else {
        if constexpr (pipeline::has<Cohorts>(pipeline::cohort_b))
        #pragma unroll
        for (int i = 0; i < G::q_per_wave; ++i) {
            const int t = wave + i * G::waves;
            const int kt = t / (BN / 16), nt = t % (BN / 16);
            if constexpr (Probe == 1) { probe_sink ^= q_reg[i].a ^ q_reg[i].b ^ q_reg[i].c; continue; }
            if (t < G::q_tiles && k0 + kt < end && first_col / 16 + nt < tiles_n) {
                Stream<Bits> s(q_reg[i]);
                const int r0 = 2 * (lane % 4), c = lane / 4;
                _Float16* base = bs + (nt * 16 + c) * STRIDE + kt * 16 + r0;
                *reinterpret_cast<uint32_t*>(base) = decode_pair<CB>(s.template state<0>(), s.template state<1>());
                *reinterpret_cast<uint32_t*>(base + 8) = decode_pair<CB>(s.template state<2>(), s.template state<3>());
                *reinterpret_cast<uint32_t*>(base + 8 * STRIDE) = decode_pair<CB>(s.template state<4>(), s.template state<5>());
                *reinterpret_cast<uint32_t*>(base + 8 * STRIDE + 8) = decode_pair<CB>(s.template state<6>(), s.template state<7>());
            }
        }
        }
    };

    if constexpr (Probe != 0) {
        for (int j = thread; j < 2 * G::stage_bytes; j += G::threads) lds[j] = 0;
        __syncthreads();
    }
    Float8 acc[MT][NT] = {};
    // Fold > 0: the current chain (from +0); acc is the folded FP32 total.
    [[maybe_unused]] Float8 fold_acc[Fold ? MT : 1][Fold ? NT : 1] = {};
    // IU4: fresh I32 per scale group; persists across stages only when g > BK.
    using IU4Sum = int32_t __attribute__((ext_vector_type(8)));
    [[maybe_unused]] IU4Sum sum[IU4 ? MT : 1][IU4 ? NT : 1] = {};
    if (stages > 0) {
        fetch(0, pipeline::Set<pipeline::cohort_all>{});
        store(0, 0, pipeline::Set<pipeline::cohort_all>{});
        __syncthreads();
    }
    for (int stage = 0; stage < stages; ++stage) {
        const int buffer = stage & 1;
        if constexpr (Schedule::early != 0U) { if (stage + 1 < stages) fetch(stage + 1, pipeline::Set<Schedule::early>{}); }
        const auto* ab = lds + buffer * G::stage_bytes;
        const _Float16* as = reinterpret_cast<const _Float16*>(ab);
        const _Float16* bs = as + BM * STRIDE;
        const int k0 = begin + stage * KT;
        if constexpr (IU4) {
            using Nibbles = int32_t __attribute__((ext_vector_type(2)));
            using F4 = float __attribute__((ext_vector_type(4)));
            // Per K64 group: every wave tile accumulates fresh I32 over four K16 steps
            // (A/B fragments shared across the MT x NT tiles, independent chains), then one
            // flush per element in ascending g: float(sum) * (sA * sW), separate mul/add.
            const auto* sl = reinterpret_cast<const float*>(ab + (BM + BN) * G::row_bytes);
            const int swz = ((lane & 15) >> 1) * 8;     // iu4_lds_offset key for rows == lane (mod 16)
            const auto* a_frag = ab + (wave_m * WM + (lane & 15)) * G::row_bytes;
            const auto* b_frag = ab + (BM + wave_n * WN + (lane & 15)) * G::row_bytes;
            static_assert(KT % 4 == 0 && WM % 16 == 0 && WN % 16 == 0);
            // Grouped tiles are ragged (a group's last tile holds its remaining rows): a wave whose WM rows all
            // lie past the tile's rows skips its WMMA and flush (wave-uniform), leaving the SIMD to resident
            // waves. It still stages its share of the operands. Live rows keep their arithmetic; the skipped
            // accumulators only cover rows that are never stored.
            if (!IU4Grouped || wave_m * WM < rows) {
            // Same at 16-row granularity inside a live wave: tile mt computes only while it holds a live row.
            const auto mt_live = [&](int mt) { return !IU4Grouped || mt == 0 || wave_m * WM + mt * 16 < rows; };
            #pragma unroll
            for (int seg = 0; seg < SLOTS; ++seg) {
                const int t0 = k0 + seg * SEG;          // first K16 tile of the segment
                if (t0 >= tiles_k) break;               // K tail (uniform)
                #pragma unroll
                for (int j = 0; j < SEG; ++j) {
                    const int byte = ((seg * SEG + j) * 8) ^ swz;
                    Nibbles left[MT], right[NT];
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt) left[mt] = *reinterpret_cast<const Nibbles*>(a_frag + mt * 16 * G::row_bytes + byte);
                    #pragma unroll
                    for (int nt = 0; nt < NT; ++nt) right[nt] = *reinterpret_cast<const Nibbles*>(b_frag + nt * 16 * G::row_bytes + byte);
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        if (!mt_live(mt)) continue;
                        #pragma unroll
                        for (int nt = 0; nt < NT; ++nt)
                            sum[mt][nt] = integer_wmma<4>(left[mt], right[nt], sum[mt][nt]);
                    }
                }
                // g > BK: flush only after the group's last stage (inputs % g == 0).
                if (GK <= KT || (t0 + SEG) % GK == 0) {
                const int g = t0 / GK;
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    if (!mt_live(mt)) continue;
                    const float* sap = sl + seg * BM + wave_m * WM + mt * 16 + (lane >> 4) * 8;
                    const F4 lo = *reinterpret_cast<const F4*>(sap), hi = *reinterpret_cast<const F4*>(sap + 4);
                    const float sa[8] = {lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
                    #pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        const float sw = sl[SLOTS * BM + seg * BN + wave_n * WN + nt * 16 + (lane & 15)];
                        #pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            // Explicit separate FP32 multiply and add; -ffp-contract=off.
                            acc[mt][nt][e] += float(sum[mt][nt][e]) * (sa[e] * sw);
                            if constexpr (IU4Audit) {   // diagnostic specialization only
                                // One lane base pointer; per-element offsets are wave-uniform.
                                const int lane_row = first_row + wave_m * WM + lane / 16;
                                int32_t* sums = group_sums + size_t(g) * size_t(rows) * size_t(outputs) +
                                    size_t(uint32_t(lane_row) * uint32_t(outputs) + uint32_t(first_col + wave_n * WN + lane % 16));
                                if (IU4FullRows || lane_row + mt * 16 + 2 * e < rows)
                                    sums[(mt * 16 + 2 * e) * outputs + nt * 16] = sum[mt][nt][e];
                            }
                        }
                        sum[mt][nt] = IU4Sum{};
                    }
                }
                }
            }
            }
        } else {
        #pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            if (k0 + kt < end) {                          // uniform; never multiply zeros into acc
                Frag right[NT];
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const _Float16* p = bs + (wave_n * WN + nt * 16 + lane % 16) * STRIDE + kt * 16;
                    right[nt] = *reinterpret_cast<const Frag*>(p);
                }
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const _Float16* p = as + (wave_m * WM + mt * 16 + lane % 16) * STRIDE + kt * 16;
                    const Frag left = *reinterpret_cast<const Frag*>(p);
                    #pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        if constexpr (Probe == 2) {   // consume both whole fragments, no WMMA
                            Half16 x = __builtin_bit_cast(Half16, left) * __builtin_bit_cast(Half16, right[nt]);
                            acc[mt][nt][0] += float(x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] +
                                                    x[8] + x[9] + x[10] + x[11] + x[12] + x[13] + x[14] + x[15]);
                        }
                        else if constexpr (Fold > 0) fold_acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            __builtin_bit_cast(Half16, left), __builtin_bit_cast(Half16, right[nt]), fold_acc[mt][nt]);
                        else acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            __builtin_bit_cast(Half16, left), __builtin_bit_cast(Half16, right[nt]), acc[mt][nt]);
                    }
                }
                if constexpr (Fold > 0) {
                    // Chain boundary every Fold K16 tiles from the split start, and at its end (uniform).
                    if ((k0 + kt + 1 - begin) % Fold == 0 || k0 + kt + 1 == end) {
                        #pragma unroll
                        for (int mt = 0; mt < MT; ++mt)
                            #pragma unroll
                            for (int nt = 0; nt < NT; ++nt) { acc[mt][nt] = acc[mt][nt] + fold_acc[mt][nt]; fold_acc[mt][nt] = Float8{}; }
                    }
                }
            }
        }
        }
        if (stage + 1 < stages) {
            if constexpr (Schedule::late != 0U) fetch(stage + 1, pipeline::Set<Schedule::late>{});
            store(stage + 1, buffer ^ 1, pipeline::Set<pipeline::cohort_all>{});
        }
        __syncthreads();
    }

    if constexpr (Probe == 1) acc[0][0][0] += float(probe_sink & 1u);
    if constexpr (!Fused) {
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int col = first_col + wave_n * WN + nt * 16 + lane % 16;
                if (col < outputs) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int row = first_row + wave_m * WM + mt * 16 + 2 * j + lane / 16;
                        if constexpr (IU4Grouped) {
                            if (row < rows) partial[size_t(map_row(row, gp.o_mul, gp.o_add)) * size_t(outputs) + size_t(col)] = acc[mt][nt][j];
                        } else if constexpr (AD) {
                            if (row < rows) ad.partial(at, row, col, acc[mt][nt][j]);   // raw split partial
                        } else if (row < rows) partial[(size_t(blockIdx.z) * size_t(rows) + size_t(row)) * size_t(outputs) + size_t(col)] = acc[mt][nt][j];
                    }
                }
            }
    } else {
        // splits == 1, BN % 128 == 0 and outputs % BN == 0 (checked by the launcher).
        // Per 16-row slice of every wave row, stage FP32 [waves_m*16][BN] in LDS
        // and run output_transform's per-group arithmetic in its exact order.
        constexpr int SLICE = G::waves_m * 16, FSTRIDE = BN + 4;
        static_assert(SLICE * FSTRIDE * 4 <= 2 * G::stage_bytes, "epilogue staging exceeds LDS");
        float* f = reinterpret_cast<float*>(lds);
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    f[(wave_m * 16 + 2 * j + lane / 16) * FSTRIDE + wave_n * WN + nt * 16 + lane % 16] = acc[mt][nt][j];
            __syncthreads();
            constexpr int groups = SLICE * (BN / 128);
            // One epilogue group; gi = its step in this wave (compile-time in the unrolled chain loop,
            // so the paired chain's saved gate registers never need a dynamic index / scratch).
            auto epilogue_group = [&](const int g, [[maybe_unused]] const int gi) {
                const int slice_row = g / (BN / 128), col_group = g % (BN / 128);
                const int row = first_row + (slice_row / 16) * WM + mt * 16 + slice_row % 16;
                if (row < rows) {
                    const int begin_col = first_col + col_group * 128;
                    size_t out_row = size_t(row);
                    if constexpr (IU4Grouped) out_row = size_t(map_row(row, gp.o_mul, gp.o_add));
                    float values[4] = {};
                    if constexpr (AD) {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) values[j] = f[slice_row * FSTRIDE + col_group * 128 + lane + 32 * j];
                        hadamard128(values);
                        ad.store(at, row, begin_col, values, lane);
                        return;
                    }
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) values[j] += f[slice_row * FSTRIDE + col_group * 128 + lane + 32 * j];
                    hadamard128(values);
                    if constexpr (Chain::kActive) {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j)
                            values[j] *= __half2float(__ushort_as_half(scales[begin_col + lane + 32*j]));
                        chain.store(map_first + uint32_t(row), uint32_t(begin_col), chain_pass,
                            mt * chain_groups + gi, values, chain_saved);
                        return;
                    }
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int channel = begin_col + lane + 32 * j;
                        const size_t index = out_row * size_t(outputs) + size_t(channel);
                        float value = values[j] * __half2float(__ushort_as_half(scales[channel]));
                        if (bias) value += bias[channel];
                        value *= alpha;
                        if (beta != 0) value += beta * output[index];
                        output[index] = value;
                    }
                }
            };
            if constexpr (Chain::kActive) {
                constexpr int gsteps = (groups + G::waves - 1) / G::waves;
                #pragma unroll
                for (int gi = 0; gi < gsteps; ++gi)
                    if (wave + gi * G::waves < groups) epilogue_group(wave + gi * G::waves, gi);
            } else {
                for (int g = wave; g < groups; g += G::waves) epilogue_group(g, 0);
            }
            __syncthreads();
        }
    }
    } // non-integer operand
    // The grouped plan owns exactly one routed tile, like its dense siblings.
    if constexpr (structure == IntegerStructure::grouped_chunk) return;
    }   // adapter tile loop
    }   // paired chain passes
    }   // general tiled plans
}
} // namespace wmma_tiled
} // namespace strix::core
