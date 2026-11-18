#pragma once

#include "../ggml_sources.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace strix::cuda::weight {

struct Sm120Dense {
    struct SignedGroup {
        std::uint32_t codes[8];
        float scale;
    };
};
struct CudaQwenImmaFourChain {};
struct CudaF16BlockScaled {};
struct CudaDs41FastA8 {};

// These are wire reads only. Arithmetic/reduction and cp.async scheduling live
// in the existing cores. In particular, no dequantized temporary is introduced.
struct Q8Block {
    static constexpr unsigned kBlockElements = 32, kBlockBytes = 34;
    STRIX_CUDA_WEIGHT_DEVICE static float scale_bits(unsigned short bits) {
        return __half2float(__ushort_as_half(bits));
    }
    STRIX_CUDA_WEIGHT_DEVICE static float scale(const std::uint8_t* block) {
        return scale_bits(*reinterpret_cast<const unsigned short*>(block));
    }
    STRIX_CUDA_WEIGHT_DEVICE static int code(const std::uint8_t* block, unsigned k) {
        return static_cast<std::int8_t>(block[2U + k]);
    }
    STRIX_CUDA_WEIGHT_DEVICE static unsigned word(const unsigned short* codes, unsigned i) {
        return static_cast<unsigned>(codes[2U * i]) | (static_cast<unsigned>(codes[2U * i + 1U]) << 16U);
    }
    STRIX_CUDA_WEIGHT_DEVICE static unsigned byte_word(const std::uint8_t* block, unsigned word_index) {
        const auto* p = block + 2U + 4U * word_index;
        return static_cast<unsigned>(p[0]) | (static_cast<unsigned>(p[1]) << 8U) |
               (static_cast<unsigned>(p[2]) << 16U) | (static_cast<unsigned>(p[3]) << 24U);
    }
    STRIX_CUDA_WEIGHT_DEVICE static int pair_code(unsigned short pair, unsigned which) {
        return static_cast<std::int8_t>(which ? pair >> 8U : pair & 0xffU);
    }
    // Shared raw136 strip, block b at byte 34*b. The even/odd realignment
    // and scale extraction are exactly the incumbent tuned-loader sequence.
    STRIX_CUDA_WEIGHT_DEVICE static void raw136(const unsigned* row, unsigned b, unsigned lane,
                                               unsigned (&codes)[2]) {
        const bool even = (b & 1U) == 0U;
        const unsigned qword = even ? 17U * b / 2U : (17U * b + 1U) / 2U;
        const unsigned* w = row + qword + lane;
        if (even) { codes[0] = __byte_perm(w[0], w[1], 0x5432); codes[1] = __byte_perm(w[4], w[5], 0x5432); }
        else { codes[0] = w[0]; codes[1] = w[4]; }
    }
    STRIX_CUDA_WEIGHT_DEVICE static float raw136_scale(const unsigned* row, unsigned b) {
        const bool even = (b & 1U) == 0U;
        const unsigned sword = even ? 17U * b / 2U : (17U * b - 1U) / 2U;
        const unsigned sw = row[sword];
        const unsigned short h = static_cast<unsigned short>(even ? (sw & 0xffffU) : (sw >> 16U));
        return scale_bits(h);
    }
};

template <>
struct OperandSource<Q8_0Tag, SignedA8Tag, Sm120Dense, CudaQwenImmaFourChain> : Q8Block {
    using View = GgmlView;
    using Group = Sm120Dense::SignedGroup;
    static constexpr unsigned group_k = 32, alignment = 2, shared_bytes = 0;
    static constexpr bool has_input_transform = false, has_output_transform = false;
    using Fragment = Group;
    static ::strix::weight::Status validate(const ::strix::weight::WeightView& w, const LaunchShape& shape) {
        auto valid = OperandSource<Q8_0Tag, NativeScalarTag, Sm120Dense, CudaQwenImmaFourChain>::validate(w, shape);
        if (!valid) return valid;
        if (shape.k % 64 || w.payload.range.offset % 4)
            return {::strix::weight::Error::unsupported_shape, "CUDA raw Q8 core requires paired groups and dword rows"};
        return {::strix::weight::Error::ok, {}};
    }
    STRIX_CUDA_WEIGHT_DEVICE static Group signed_group(const View& v, std::uint64_t row, std::uint64_t group) {
        Group result;
        const auto* block = v.blocks + row * v.row_stride_bytes + group * kBlockBytes;
        result.scale = scale(block);
        const auto* c = reinterpret_cast<const unsigned short*>(block + 2);
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) result.codes[i] = word(c, i);
        return result;
    }
};
using Q8ImmaSource = OperandSource<Q8_0Tag, SignedA8Tag, Sm120Dense, CudaQwenImmaFourChain>;

// DS41 stages the same Q8 source into its existing tile16 image. Its relaxed
// accumulation stays separate from the exact Qwen numerical policy.
template <>
struct OperandSource<Q8_0Tag, SignedA8Tag, Sm120Dense, CudaDs41FastA8> : Q8Block {
    struct View { const uint4* codes; const unsigned short* scales; std::uint32_t groups, tiles; };
    static constexpr unsigned group_k = 32, alignment = 16, shared_bytes = 0;
    using Fragment = std::uint32_t[2];
    static constexpr bool has_input_transform = false, has_output_transform = false;
    static ::strix::weight::Status validate(const ::strix::weight::WeightView& w, const LaunchShape& shape) {
        return OperandSource<Q8_0Tag, NativeScalarTag, Sm120Dense, CudaDs41FastA8>::validate(w, shape);
    }
};
using Q8Ds41TileSource = OperandSource<Q8_0Tag, SignedA8Tag, Sm120Dense, CudaDs41FastA8>;

// Existing mma::Tile WeightFmt interface, with the original Q4_0 register
// layout and exact unpack. This source is bound by an alias in mma_tile.cuh.
struct Q40F16Source {
    static constexpr unsigned kBlockBytes = 18;
    struct Regs { unsigned short w[9]; };
    STRIX_CUDA_WEIGHT_DEVICE static void load(Regs& r, const std::uint8_t* blk) {
#pragma unroll
        for (int i = 0; i < 9; ++i) r.w[i] = *reinterpret_cast<const unsigned short*>(blk + 2 * i);
    }
    STRIX_CUDA_WEIGHT_DEVICE static unsigned unpack_pair(unsigned u) {
        const unsigned v = 0x64006400U | (u & 15U) | (((u >> 8U) & 15U) << 16U);
        const __half2 r = __hsub2(*reinterpret_cast<const __half2*>(&v), __half2half2(__ushort_as_half(0x6408)));
        return *reinterpret_cast<const unsigned*>(&r);
    }
    STRIX_CUDA_WEIGHT_DEVICE static void store(const Regs& r, __half* dw_half, float* scale_dst, unsigned kh) {
        uint4 pv[2];
        unsigned* ww = reinterpret_cast<unsigned*>(pv);
        const unsigned sh = kh * 4U;
#pragma unroll
        for (int i = 0; i < 8; ++i) ww[i] = unpack_pair(static_cast<unsigned>(r.w[1 + i]) >> sh);
        uint4* dw = reinterpret_cast<uint4*>(dw_half);
        dw[0] = pv[0];
        dw[1] = pv[1];
        if (!kh) *scale_dst = Q8Block::scale_bits(r.w[0]);
    }
};

template <>
struct OperandSource<GgmlTag<2>, F16TileTag, Sm120Dense, CudaF16BlockScaled> : Q40F16Source {
    using View = GgmlView;
    using Fragment = Regs;
    static constexpr unsigned group_k = 32, alignment = 2, shared_bytes = 0;
    static constexpr bool has_input_transform = false, has_output_transform = false;
    static ::strix::weight::Status validate(const ::strix::weight::WeightView& w, const LaunchShape& shape) {
        return OperandSource<GgmlTag<2>, NativeScalarTag, Sm120Dense, CudaF16BlockScaled>::validate(w, shape);
    }
};
using Q40TileSource = OperandSource<GgmlTag<2>, F16TileTag, Sm120Dense, CudaF16BlockScaled>;

} // namespace strix::cuda::weight
