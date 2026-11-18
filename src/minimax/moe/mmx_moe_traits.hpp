#pragma once

#include <type_traits>

#include "../../moe_core/exl3_trellis.hpp"
#include "../../moe_core/decode_schedule.hpp"
#include "mmx_moe.hpp"

namespace strix::minimax::moe::opt {

template <int Bits> using WeightDecode = strix::moe_core::Exl3Trellis<Bits>;
// The registered operand source the decode policy is bound to (hip/weight/trellis_sources.hpp); same codec and input transform.
template <int Bits> using WeightSource = strix::weight::hip::LegacyMul1Source<static_cast<unsigned>(Bits)>;
static_assert(std::is_base_of_v<WeightSource<3>, WeightDecode<3>>);

// Host dispatch selects a policy on the existing grouped PAIR/PIPE kernels.
template <bool Packed> struct MinimaxPrefillTraits {
    template <int Bits> using Decode = std::conditional_t<Packed, strix::moe_core::Mul1PairDecode<Bits>,
                                                         strix::moe_core::ScalarDecode<Bits>>;
};

// C1's Args, LDS vector and device operations stay private to the translation
// unit. This contract supplies the shared Decode family with MiniMax geometry;
// inherited operations preserve the existing EXL3 K chains and publication ABI.
template <class ArgsT, class WideV4T, class DeviceOps>
struct MinimaxTraits : DeviceOps {
    using Args = ArgsT;
    using Dispatch = strix::minimax::moe::opt::Dispatch;
    using ExpertSlices = strix::minimax::moe::opt::ExpertSlices;
    using Exl3 = strix::minimax::moe::opt::Exl3;
    using WideV4 = WideV4T;
    using SlotInfo = Dispatch;  // Only the GLM overloads use a separate slot record.
    static constexpr auto kHidden = strix::minimax::moe::opt::kHidden;
    static constexpr auto kInter = strix::minimax::moe::opt::kInter;
    static constexpr auto kSharedSlot = strix::minimax::moe::opt::kSharedSlot;
    static constexpr auto kTopE = strix::minimax::moe::opt::kTopE;
    static constexpr unsigned kGateSplits = strix::moe_core::vector_splits(kHidden, kInter);
    static constexpr unsigned kDownSplits = strix::moe_core::vector_splits(kInter, kHidden);
    static constexpr unsigned kSplitCh = 384;
    using Schedule = strix::moe_core::DecodeSchedule<kGateSplits, kDownSplits, kSplitCh, 4U>;
    static_assert(kGateSplits == 16 && kDownSplits == 4);
    static_assert(kGateSplits * kSplitCh == kHidden && kDownSplits * kSplitCh == kInter);
};

struct ExpertLayout {
    // Expert-major packed streams; gate/up work items are consecutive, separate matrices.
    static constexpr unsigned kGateUpInterleave = 2;
    static constexpr unsigned kRowBlock = 16;
    template <class Geometry, class Columns>
    static __host__ __device__ constexpr std::size_t packed_stride(Columns columns) {
        return static_cast<std::size_t>(columns) * Geometry::kWords;
    }
};

struct InputTransform : WeightDecode<3>::InputTransform {
    // RN16(H128(RN16(RN32(x*suh)))); the bit width does not affect this transform.
    using WeightDecode<3>::InputTransform::input_block;
};

struct Epilogue {
    static constexpr float kAlpha = 1.702F, kLimit = 7.0F;
static __device__ __forceinline__ float swiglu(float g, float u) {  // tests/minimax_gate_common.hpp swiglu_kernel
    const float gg = g < kLimit ? g : kLimit;
    const float uu = u < -kLimit ? -kLimit : (u > kLimit ? kLimit : u);
    return gg * (1.0F / (1.0F + expf(-kAlpha * gg))) * (uu + 1.0F);
}

static __device__ __forceinline__ void finish_block(const float* part, unsigned splits, unsigned rows, unsigned row, unsigned output,
                                             unsigned b, const std::uint16_t* svh, unsigned lane, float (&v)[4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = 0.0F;
    for (unsigned s = 0; s < splits; ++s)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            v[j] += part[(static_cast<std::size_t>(s) * rows + row) * output + b * 128 + lane + 32U * static_cast<unsigned>(j)];
    strix::moe_core::hadamard128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = v[j] * strix::moe_core::half_bits(svh[b * 128 + lane + 32U * static_cast<unsigned>(j)]);
#if defined(STRIX_WFMT_CAPTURE) && STRIX_WFMT_CAPTURE
    for (int j = 0; j < 4; ++j) strix::wfmt_trace::trellis_trace(strix::wfmt_trace::kTrellisFinish, v[j]);
#endif
}

    static __device__ __forceinline__ float router_accumulate(float w, float v, float acc) {
        return __builtin_fmaf(w, v, acc);
    }
    static __device__ __forceinline__ float shared_add(float acc, float v) { return acc + v; }
};

}  // namespace strix::minimax::moe::opt
