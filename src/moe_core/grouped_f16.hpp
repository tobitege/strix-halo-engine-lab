#pragma once
#include <cstdint>
// Device-only fragments. Types supplies the engine's existing vector types.
namespace strix::moe_core {
// Global-address issue and the K pipeline shared by the grouped F16 bodies (projection and down). An engine supplies
// its own vector types and adopts the scheduler through GroupedF16ActivationIssue; it keeps no private copy.
struct GroupedF16Issue {
    // LDS-only barrier: the staged tiles are complete (lgkmcnt) while the next step's global loads stay in flight.
    static __device__ __forceinline__ void barrier(){__asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");}
    // 16-byte global (not flat) load: flat loads also count in lgkmcnt and would end at barrier().
    template<class Word4> static __device__ __forceinline__ Word4 load16(const void* p) {
        typedef unsigned U32x4 __attribute__((ext_vector_type(4)));
        const U32x4 v=*reinterpret_cast<const __attribute__((address_space(1))) U32x4*>(reinterpret_cast<std::uintptr_t>(p));
        return Word4{v.x,v.y,v.z,v.w};
    }
    template<class Word2> static __device__ __forceinline__ Word2 load8(const void* p) {
        typedef unsigned U32x2 __attribute__((ext_vector_type(2)));
        const U32x2 v=*reinterpret_cast<const __attribute__((address_space(1))) U32x2*>(reinterpret_cast<std::uintptr_t>(p));
        return Word2{v.x,v.y};
    }
    // The K loop: two register stages (or one with OneStage), LDS-only barriers. fetch(step, regs) issues the global
    // loads of a step, stage(step, regs, buffer) converts and writes the LDS tiles, step_fn(buffer) consumes them.
    template<bool OneStage,class Regs,class Fetch,class Stage,class Step>
    static __device__ __forceinline__ void pipeline(unsigned steps,const Fetch& fetch,const Stage& stage,const Step& step_fn) {
        if constexpr(OneStage) {
            Regs r;
            fetch(0U,r);
            for(unsigned step=0;step<steps;++step) {
                if(step)barrier();
                stage(step,r,0U);
                barrier();
                if(step+1U<steps)fetch(step+1U,r);
                step_fn(0U);
            }
        } else {
            Regs r0,r1;
            fetch(0U,r0);fetch(1U,r1); // steps is even
#pragma unroll 1
            for(unsigned step=0;step<steps;step+=2U) {
                if(step)barrier();
                stage(step,r0,0U);
                barrier();
                if(step+2U<steps)fetch(step+2U,r0);
                step_fn(0U);
                barrier();
                stage(step+1U,r1,0U);
                barrier();
                if(step+3U<steps)fetch(step+3U,r1);
                step_fn(0U);
            }
        }
    }
};
// Activation issue policy of a register stage. Uniform: every lane issues the loads of every part; the fetch is one
// straight block without exec-mask regions or skip branches, and the register stage needs no zero initialisation.
// Contract: `src` of a part outside the batch must already point at readable memory (the bodies clamp it to the first
// row of the batch) and whatever is staged for such a part must stay unobservable (no store, no certificate reads it).
// Masked: a part outside the batch issues nothing and keeps `dst` (the zero-initialised stage).
template<bool Uniform> struct GroupedF16ActivationIssue {
    static constexpr bool uniform=Uniform;
    template<class Word4,unsigned N> static __device__ __forceinline__ void issue(bool live,const Word4* src,Word4 (&dst)[N]) {
        if constexpr(Uniform) {
            (void)live;
#pragma unroll
            for(unsigned q=0;q<N;++q)dst[q]=GroupedF16Issue::load16<Word4>(src+q);
        } else if(live) {
#pragma unroll
            for(unsigned q=0;q<N;++q)dst[q]=GroupedF16Issue::load16<Word4>(src+q);
        }
    }
};
template<class Types> struct GroupedF16Fragments {
    using FragmentWord8=typename Types::Word8;
    using FragmentHalf16=typename Types::Half16;
    using FragmentFloat8=typename Types::Float8;
    using FragmentWord4=typename Types::Word4;
template<bool Contiguous> static __device__ __forceinline__ FragmentWord8 f16_frag(const _Float16* row,unsigned lo_half,unsigned hi_half) {
    if constexpr(Contiguous) { (void)hi_half; return *reinterpret_cast<const FragmentWord8*>(row+lo_half); }
    else {
        const FragmentWord4 lo=*reinterpret_cast<const FragmentWord4*>(row+lo_half),hi=*reinterpret_cast<const FragmentWord4*>(row+hi_half);
        return FragmentWord8{lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w};
    }
}
static __device__ __forceinline__ FragmentFloat8 f16_wmma(FragmentWord8 a,FragmentWord8 b,FragmentFloat8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(FragmentHalf16,a),__builtin_bit_cast(FragmentHalf16,b),c);
}
};
} // namespace strix::moe_core

// ---- Folded F16 grouped core (FP16 trellis weights x FP16 rows on v_wmma_f32_16x16x16_f16) -------------------------
// One CTA = 8 waves x 128 outputs x 16 * MT rows; K in stages of Fold::kUnits 16-wide k-units (K64 folds): the rows are
// staged in LDS once per CTA and stage, each wave transposes its decoded tile through its own LDS rows into a B
// fragment, one WMMA chain from +0 per (row fragment, output tile, stage), stages added in ascending order into an
// FP32 total, H128 epilogue through the same LDS. Traits: Dec (trellis decode: LaneGeom, TileStates, load_u32,
// mul1_pair, hadamard128, uniform, uni), Fold (stage width), CP (output tiles per wave), Input (kFused, stage, offset),
// Output (store of a finished row). Engines bind the traits; the device body lives here once.
#include <cstddef>
#include <cstdint>
namespace strix::moe_core::folded_f16 {
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
using U4 = std::uint32_t __attribute__((ext_vector_type(4)));
struct FoldK64 { static constexpr unsigned kUnits = 4; };

constexpr unsigned kRowsMax = 64;  // MT = 4 row tiles; the epilogue always works in 64-row halves
// Pair pitch 36 dwords (as exact_x): b128 reads stay aligned and rows rotate banks.
template<unsigned NR, unsigned ActivationPitch = 36>
union alignas(16) StorageR {
    struct { std::uint32_t w[128][36], a[NR][ActivationPitch]; } stage;
    float total[kRowsMax][128];
};
static_assert(sizeof(StorageR<64>) == 32768 && sizeof(StorageR<128>) == 36864);
static_assert(sizeof(StorageR<64, 68>) == 35840 && sizeof(StorageR<128, 68>) == 53248);

__device__ __forceinline__ Half16 frag(const std::uint32_t* p) {
    const U4 lo = *reinterpret_cast<const U4*>(p), hi = *reinterpret_cast<const U4*>(p + 4);
    struct { U4 l, h; } v{lo, hi};
    return __builtin_bit_cast(Half16, v);
}

// Rows: input(r) = transformed FP16 row, output(r) = FP32 destination (exact_x interface). nr <= 16 * MT.
template<int Bits, unsigned MT, unsigned CP, class Dec, class Fold, class Mat, class Rows, class Lds, class Output, class Input>
__device__ __forceinline__ void core(Mat m, Rows rows, unsigned nr, Lds& lds, Output output, Input input, bool all_fragments = false) {
    static_assert(Fold::kUnits == 4U, "the staged LDS rows hold four 16-wide k-units (K64 folds)");
    static_assert(MT == 1 || MT == 2 || MT == 4 || MT == 8);
    static_assert(CP == 1 || (CP == 2 && MT == 8));
    // CP 2: wave w owns output tiles 2 (w & 3), 2 (w & 3) + 1 and row tiles 4 (w >> 2) .. 4 (w >> 2) + 3;
    // acc[4 s + t] = (output tile 2 (w & 3) + s, row tile 4 (w >> 2) + t). CP 1: acc[t] = (output tile w, row tile t).
    constexpr unsigned NR = 16U * MT, Q = NR * 8U / 256U;  // 16-byte activation chunks per thread per stage
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid / 32U, c = lane & 15U;
    const unsigned col0 = blockIdx.x * 128U;
    const unsigned units = m.input / 16U;
    const typename Dec::template LaneGeom<Bits> geo(lane);
    const auto* packed = reinterpret_cast<const unsigned char*>(Dec::uniform(m.trellis));
    // Chunk q of a stage: row q / 8, 16-byte column q % 8. Rows past nr read row index 0 (never published).
    const _Float16* src[Q > 0 ? Q : 1];
    unsigned qrow[Q > 0 ? Q : 1], qcol[Q > 0 ? Q : 1];
#pragma unroll
    for (unsigned j = 0; j < Q; ++j) {
        const unsigned q = tid + j * 256U;
        qrow[j] = q / 8U;
        qcol[j] = q % 8U;
        if constexpr (!Input::kFused) src[j] = rows.input(qrow[j] < nr ? qrow[j] : 0U);
    }
    // MT 1: 128 chunks, the upper half of the CTA stages nothing.
    const bool stager = Q > 0 || tid < NR * 8U;
    const unsigned q1 = tid, r1 = q1 / 8U, c1 = q1 % 8U;
    const _Float16* src1 = (Q == 0 && stager && !Input::kFused) ? rows.input(r1 < nr ? r1 : 0U) : nullptr;

    std::uint32_t raw[4][3];
    U4 xa[Q > 0 ? Q : 1];
    const auto fetch = [&](unsigned base, unsigned nk) {
#pragma unroll
        for (unsigned kk = 0; kk < 4U; ++kk) if (kk < nk) {
            const auto* tile = packed + (std::size_t{base + kk} * (m.output / 16U) + col0 / 16U + wave) * Bits * 32U;
            raw[kk][0] = Dec::load_u32(tile, geo.oa);
            raw[kk][1] = Dec::load_u32(tile, geo.ob);
            raw[kk][2] = Dec::load_u32(tile, geo.oc);
        }
        if constexpr (!Input::kFused && Q > 0) {
#pragma unroll
            for (unsigned j = 0; j < Q; ++j) {
                xa[j] = U4{0, 0, 0, 0};
                if (qcol[j] < nk * 2U) xa[j] = *reinterpret_cast<const U4*>(src[j] + base * 16U + qcol[j] * 8U);
            }
        } else if constexpr (!Input::kFused) {
            xa[0] = U4{0, 0, 0, 0};
            if (stager && c1 < nk * 2U) xa[0] = *reinterpret_cast<const U4*>(src1 + base * 16U + c1 * 8U);
        }
    };
    // Row fragment t (rows 16 t .. 16 t + 15) is live while it holds a row below nr. A ragged tile runs on the
    // smallest MT covering it, so up to MT / 2 - 1 fragments (CP 2: up to three in the upper waves) hold no row: their
    // WMMA chains are skipped (wave-uniform count, read once). Their accumulators stay +0 and are never published;
    // live fragments keep every operand and the order of their chain, so every stored value keeps its bits.
    const unsigned live = all_fragments ? MT : Dec::uni((nr + 15U) / 16U);
    Float8 acc[MT] = {};
    fetch(0, units < 4U ? units : 4U);
    for (unsigned base = 0; base < units; base += 4U) {
        const unsigned nk = units - base < 4U ? units - base : 4U;
#pragma unroll
        for (unsigned kk = 0; kk < 4U; ++kk) if (kk < nk) {
            typename Dec::template TileStates<Bits> st;
            st.high = __builtin_amdgcn_alignbit(raw[kk][0], raw[kk][1], geo.align);
            st.low = __builtin_amdgcn_alignbit(raw[kk][1], raw[kk][2], geo.align);
            st.tail = raw[kk][2] << (32U - geo.align);
            const unsigned o = wave * 16U + lane / 4U, p = kk * 8U + (lane & 3U);
            lds.stage.w[o][p] = __builtin_bit_cast(std::uint32_t, Dec::mul1_pair(st.template state<0>(), st.template state<1>()));
            lds.stage.w[o][p + 4U] = __builtin_bit_cast(std::uint32_t, Dec::mul1_pair(st.template state<2>(), st.template state<3>()));
            lds.stage.w[o + 8U][p] = __builtin_bit_cast(std::uint32_t, Dec::mul1_pair(st.template state<4>(), st.template state<5>()));
            lds.stage.w[o + 8U][p + 4U] = __builtin_bit_cast(std::uint32_t, Dec::mul1_pair(st.template state<6>(), st.template state<7>()));
        }
        if constexpr (Input::kFused) {
            input.template stage<NR>(m, nr, base, lds);
        } else if constexpr (Q > 0) {
#pragma unroll
            for (unsigned j = 0; j < Q; ++j) *reinterpret_cast<U4*>(&lds.stage.a[qrow[j]][qcol[j] * 4U]) = xa[j];
        } else if (stager) {
            *reinterpret_cast<U4*>(&lds.stage.a[r1][c1 * 4U]) = xa[0];
        }
        __syncthreads();
        if (base + 4U < units) {
            const unsigned next = base + 4U;
            fetch(next, units - next < 4U ? units - next : 4U);
        }
        asm volatile("" ::: "memory");
        const unsigned aoff = Input::kFused ? (base & 4U) * 8U : 0U;
        // The WMMA accumulator add is not round-to-nearest (synthetic drift grew linearly with K when one chain ran
        // over all of K): chain only the stage's k-units from +0, then fold the stage into the total with v_add_f32.
        Float8 part[MT] = {};
        // NT row fragments per wave; the first lim of them are live (wave-uniform scalar).
        constexpr unsigned NT = CP == 1 ? MT : 4U;
        const unsigned ot = CP == 1 ? wave : 2U * (wave & 3U), rt = CP == 1 ? 0U : 4U * (wave >> 2U);
        const unsigned lim = Dec::uni(live > rt ? live - rt : 0U);
        const auto full = [&] {
            for (unsigned kk = 0; kk < nk; ++kk) {
                if constexpr (CP == 1) {
                    const Half16 b = frag(&lds.stage.w[wave * 16U + c][kk * 8U]);
#pragma unroll
                    for (unsigned t = 0; t < MT; ++t)
                        part[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(frag(&lds.stage.a[t * 16U + c][aoff + kk * 8U]), b, part[t]);
                } else {
                    const Half16 b0 = frag(&lds.stage.w[ot * 16U + c][kk * 8U]), b1 = frag(&lds.stage.w[(ot + 1U) * 16U + c][kk * 8U]);
#pragma unroll
                    for (unsigned t = 0; t < 4U; ++t) {
                        const Half16 a = frag(&lds.stage.a[(rt + t) * 16U + c][aoff + kk * 8U]);
                        part[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b0, part[t]);
                        part[4U + t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b1, part[4U + t]);
                    }
                }
            }
        };
        const auto masked = [&] {  // the same loop, each fragment behind its predicate
            for (unsigned kk = 0; kk < nk; ++kk) {
                if constexpr (CP == 1) {
                    const Half16 b = frag(&lds.stage.w[wave * 16U + c][kk * 8U]);
#pragma unroll
                    for (unsigned t = 0; t < MT; ++t) if (t < lim)
                        part[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(frag(&lds.stage.a[t * 16U + c][aoff + kk * 8U]), b, part[t]);
                } else {
                    const Half16 b0 = frag(&lds.stage.w[ot * 16U + c][kk * 8U]), b1 = frag(&lds.stage.w[(ot + 1U) * 16U + c][kk * 8U]);
#pragma unroll
                    for (unsigned t = 0; t < 4U; ++t) if (t < lim) {
                        const Half16 a = frag(&lds.stage.a[(rt + t) * 16U + c][aoff + kk * 8U]);
                        part[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b0, part[t]);
                        part[4U + t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b1, part[4U + t]);
                    }
                }
            }
        };
        // Full tiles keep the branch-free loop; only a wave with a dead fragment takes the predicated one.
        if (lim >= NT) full(); else masked();
#pragma unroll
        for (unsigned t = 0; t < MT; ++t) acc[t] = acc[t] + part[t];  // a dead fragment adds +0 to +0
        __syncthreads();
    }
    // acc[t][j] -> row index (16 t + 2 j + lane / 16), output wave * 16 + lane % 16; 64-row halves through lds.total.
    constexpr unsigned TH = MT < 4U ? MT : 4U;
#pragma unroll
    for (unsigned h = 0; h < MT / TH; ++h) {
        if constexpr (CP == 1) {
#pragma unroll
            for (unsigned t = 0; t < TH; ++t)
#pragma unroll
                for (unsigned j = 0; j < 8U; ++j) lds.total[t * 16U + 2U * j + lane / 16U][wave * 16U + c] = acc[h * TH + t][j];
        } else if ((wave >> 2U) == h) {  // the waves holding this half's row tiles
#pragma unroll
            for (unsigned s = 0; s < 2U; ++s)
#pragma unroll
                for (unsigned t = 0; t < 4U; ++t)
#pragma unroll
                    for (unsigned j = 0; j < 8U; ++j)
                        lds.total[t * 16U + 2U * j + lane / 16U][(2U * (wave & 3U) + s) * 16U + c] = acc[4U * s + t][j];
        }
        __syncthreads();
        const unsigned h0 = h * kRowsMax, h1 = nr < h0 + kRowsMax ? nr : h0 + kRowsMax;
        for (unsigned r = h0 + wave; r < h1; r += 8U) {
            float v[4];
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) v[j] = lds.total[r - h0][lane + 32U * j];
            Dec::hadamard128(v, lane);
            output(rows, m, r, col0, v, lane);
        }
        __syncthreads();
    }
}

// A row span [0, n) in tiles of 64 (or 128) rows, the last one on the smallest MT covering it (uniform per CTA).
template<int Bits, unsigned CP, class Dec, class Fold, class Mat, class RowsAt, class Lds, class Output, class Input>
__device__ __forceinline__ void span(Mat m, RowsAt at, unsigned n, Lds& lds, Output output, Input input, bool all_fragments = false) {
    constexpr unsigned Tile = sizeof(lds.stage.a) / sizeof(lds.stage.a[0]);
    for (unsigned r = 0; r < n; r += Tile) {
        const unsigned k = n - r < Tile ? n - r : Tile;
        if constexpr (Tile == 128U) {
            if (k > 64U) { core<Bits, 8, CP, Dec, Fold>(m, at(r, k), k, lds, output, input.offset(r), all_fragments); continue; }
        }
        if (k > 32U) core<Bits, 4, 1, Dec, Fold>(m, at(r, k), k, lds, output, input.offset(r), all_fragments);
        else if (k > 16U) core<Bits, 2, 1, Dec, Fold>(m, at(r, k), k, lds, output, input.offset(r), all_fragments);
        else core<Bits, 1, 1, Dec, Fold>(m, at(r, k), k, lds, output, input.offset(r), all_fragments);
    }
}
}  // namespace strix::moe_core::folded_f16
