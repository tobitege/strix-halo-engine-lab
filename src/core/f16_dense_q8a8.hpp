#pragma once
// Named policy dense-q8a8-f16-v1 of the shared FP16 dense core (f16_dense_tall.hpp): Q8_0 weights x A8 activations in
// ONE FP16 WMMA chain per output, y[r][o] = sum_k a16[r][k] * w16[o][k] (FP32 accumulation, k ascending, from +0).
//
// Class: FAST (not the bits of the integer Q8_0 x A8 products: those scale an exact integer sum per 32-k group).
// Arithmetic contract:
//   w16[o][k] = fp16_rne(float(code_w) * float(d))   d = the FP16 scale of the weight's Q8_0 block (product exact in FP32)
//   a16[r][k] = fp16_rne(float(code_a) * delta)       delta = the FP32 step of the row's 32-k group
// Weights are decoded by the CTA's stagers once per (CTA, stage of 64 k) into LDS: a Q8_0 byte is read once per
// 16 * Waves rows (34 bytes per 32 weights, about half the bytes of an FP16 mirror) and no FP16 copy of the weights
// exists. Activations are converted in registers (lane = row). Any engine with Q8_0 rows and A8 codes binds it by
// filling View; nothing here names a model.
// Limits (callers check with supported()): in % 128 == 0, x_stride % 32 == 0, in <= x_stride.
#include "f16_dense_tall.hpp"
#include "../attn_core/wmma_types.hpp"  // the operand-image epilogue rounds with the attention cores' to_operand
#include "../hip/weight/ggml_sources.hpp"

namespace strix::core::f16_dense_tall::q8a8 {
inline constexpr char kPolicyName[] = "dense-q8a8-f16-v1";

struct View {
    const std::uint8_t* weight;  // Q8_0 rows: per 32 weights {FP16 scale, 32 x int8}; output o at weight + o * row_bytes
    const std::int8_t* codes;    // A8 codes: row r at codes + r * x_stride
    const float* deltas;         // deltas[r * (x_stride / 32) + k / 32]
    float* out;                  // out[r * out_stride + o]
    std::uint32_t row_bytes;     // bytes between consecutive weight rows
    std::uint32_t k_bytes;       // byte offset of the first used block inside a weight row (a K part)
    std::uint32_t in, count, rows, x_stride, out_stride;
    std::uint32_t tile0;         // first column tile of the launch (set by enqueue)
    std::uint32_t accumulate;    // out += sum
    // Batched form (grid z): `batch` independent products of the same shape in ONE launch; product z reads
    // weight + z * batch_weight_bytes, codes + z * batch_in, deltas + z * batch_in / 32 and writes out + z * batch_out.
    // Each product's bits are those of its own call (scheduling only). 0 or 1 = one product (the fields are not read).
    // Caller guarantees disjoint output regions across products (including every row, also when accumulating).
    // supported() checks shape admission only; it does not check output overlap or allocation extents.
    std::uint32_t batch = 0U, batch_in = 0U, batch_out = 0U;
    std::uint64_t batch_weight_bytes = 0U;
};

// Forms (bits of the kernel template; every form gives the contract's bits):
//   kWPacked : the stagers decode weights two at a time in packed FP16 (w16_pair) instead of one FP32 product each.
inline constexpr unsigned kWPacked = 1U;

// Compatibility names forward to the architecture's existing operand operations.
namespace detail { using namespace ::strix::weight::hip::gfx1151; }


template <unsigned Waves, unsigned Form = 0U> struct Cols {
    using WeightSource = ::strix::weight::hip::Q8F16Source;
    static constexpr unsigned kWaves = Waves;
    View v;
    __device__ __forceinline__ ColumnTile tile(unsigned i) const {
        const unsigned o = 16U * (v.tile0 + i);
        return {nullptr, v.out + o, v.out_stride, o < v.count};
    }
    // 64 weights = two Q8_0 blocks (68 bytes) of the stager's column; columns past count read the last one (never stored)
    __device__ __forceinline__ void stage(unsigned tile, unsigned col, unsigned unit, U4* nb) const {
        const unsigned o0 = 16U * (v.tile0 + tile) + col, o = o0 < v.count ? o0 : v.count - 1U;
        const auto* p = v.weight + static_cast<std::size_t>(o) * v.row_bytes + v.k_bytes + (unit / 2U) * 34U;
        WeightSource::template stage<Form>(p,nb);
    }
    __device__ __forceinline__ Half16 fragment(unsigned row, unsigned unit) const {
        return WeightSource::activation_fragment(v.codes,v.deltas,v.x_stride,row,unit);
    }
};

// Stores only outputs below count (the last column tile may be partial); accumulate adds to what is there.
struct Store {
    std::uint32_t count, tile0, accumulate;
    struct Tile {
        float* y;
        unsigned stride, c;
        bool live;
    };
    __device__ __forceinline__ Tile open(unsigned i, const ColumnTile& ct, unsigned c) const {
        return {f16_dense_tall::detail::uniform(ct.y), f16_dense_tall::detail::uni(ct.stride), c, 16U * (tile0 + i) + c < count};
    }
    __device__ __forceinline__ void put(const Tile& t, unsigned r, float value) const {
        if (!t.live) return;
        float* dst = t.y + static_cast<std::size_t>(r) * t.stride + t.c;
        *dst = accumulate ? *dst + value : value;
    }
    __device__ __forceinline__ void close() const {}
    template <class F> __device__ __forceinline__ void run(F&& f) { f(*this); }
};

// Operand-image epilogue: the producer side of attn_core's query_op hook (attn_core/sparse_flash_wmma.hpp). Each
// output's FP32 sum, after the engine's pair transform E::pair(e, r, o, x0, x1) on the column pair (o, o + 1) (o even;
// e.g. RoPE), is stored as tile::to_operand<E::kOp>(x) into image[r * out_stride + o] instead of out (out is not
// written). A consumer that applied the same transform and rounding to the FP32 outputs gets the same bits: the sum
// is the same register value either way and to_operand is the consumer's rounding function. Halves the output bytes.
// Policy E: using Args; static constexpr tile::Operand kOp; static __device__ void pair(const Args&, unsigned r,
// unsigned o, float& x0, float& x1). Limits (supported_image()): one K part, no accumulate, count even.
template <class E> struct StoreImage {
    std::uint32_t count, tile0;
    std::uint16_t* image;
    typename E::Args e;
    struct Tile {
        std::uint16_t* y;
        unsigned stride, o;
        bool live;
    };
    __device__ __forceinline__ Tile open(unsigned i, const ColumnTile& ct, unsigned c) const {
        const unsigned o = 16U * (tile0 + i) + c;
        return {f16_dense_tall::detail::uniform(image + 16U * (tile0 + i)), f16_dense_tall::detail::uni(ct.stride), o, o < count};
    }
    // lanes (c, c ^ 1) hold columns (o, o ^ 1) of the same row and call put together (the row depends on lane / 16)
    __device__ __forceinline__ void put(const Tile& t, unsigned r, float value) const {
        const float partner = __shfl_xor(value, 1, 32);
        if (!t.live || (t.o & 1U) != 0U) return;
        float x0 = value, x1 = partner;
        E::pair(e, r, t.o, x0, x1);
        *reinterpret_cast<std::uint32_t*>(t.y + static_cast<std::size_t>(r) * t.stride + (t.o & 15U)) =
            static_cast<std::uint32_t>(attn_core::tile::to_operand<E::kOp>(x0)) |
            (static_cast<std::uint32_t>(attn_core::tile::to_operand<E::kOp>(x1)) << 16U);
    }
    __device__ __forceinline__ void close() const {}
    template <class F> __device__ __forceinline__ void run(F&& f) { f(*this); }
};

template <unsigned TC, unsigned Waves, unsigned Form> __global__ __launch_bounds__(32 * Waves) void kernel(View v) {
    __shared__ Storage<TC> lds;
    if (v.batch > 1U) {  // product blockIdx.z of a batched launch (CTA-uniform)
        const unsigned z = blockIdx.z;
        v.weight += static_cast<std::size_t>(z) * v.batch_weight_bytes;
        v.codes += static_cast<std::size_t>(z) * v.batch_in;
        v.deltas += static_cast<std::size_t>(z) * (v.batch_in / 32U);
        v.out += static_cast<std::size_t>(z) * v.batch_out;
    }
    core<TC>(nullptr, v.in, v.rows, Cols<Waves, Form>{v}, lds, Store{v.count, v.tile0, v.accumulate});
}
template <unsigned TC, unsigned Waves, class E>
__global__ __launch_bounds__(32 * Waves) void kernel_image(View v, std::uint16_t* image, typename E::Args e) {
    __shared__ Storage<TC> lds;
    core<TC>(nullptr, v.in, v.rows, Cols<Waves>{v}, lds, StoreImage<E>{v.count, v.tile0, image, e});
}

inline bool supported(const View& v) {
    return ::strix::weight::hip::Q8F16Source::supported(v) &&
           (v.batch <= 1U || (v.batch <= 65535U && v.batch_in % 32U == 0U));
}

// Rows per weight read = 16 * waves (the same bits either way). waves 0 = the rule: STRIX_F16_Q8A8_WAVES (8 or 16) when
// set, else the measured one (gfx1151, ms per call 128-row / 256-row CTAs): in 1024: 512 rows 0.186 / 0.165, 1024 rows
// equal, 2048-4096 rows 1.48 / 1.72 and 3.48 / 3.91; in 8192: 2048 rows 1.70 / 1.30. So 256-row CTAs for wide inputs
// (in >= 8192) and for 512-1023 rows, 128-row CTAs elsewhere.
inline unsigned waves_for(unsigned rows, unsigned in, unsigned waves = 0U) {
    static const unsigned env = static_cast<unsigned>(envcfg::integer("STRIX_F16_Q8A8_WAVES", 0));
    if (!waves) waves = env;
    return waves == 8U || waves == 16U ? waves : rows >= 512U && (in >= 8192U || rows < 1024U) ? 16U : 8U;
}

// Form of a call (kWPacked or 0; the same bits either way). kFormRule = STRIX_F16_Q8A8_WPACK (default on; 0 = the scalar
// FP32 decode of every weight, the previous form).
inline constexpr unsigned kFormRule = ~0U;
inline unsigned form_for(unsigned form = kFormRule) {
    static const unsigned rule = envcfg::flag("STRIX_F16_Q8A8_WPACK", true) ? kWPacked : 0U;
    return (form == kFormRule ? rule : form) & kWPacked;
}

template <unsigned Waves, unsigned Form> inline void launch(View v, hipStream_t stream) {
    const unsigned tiles = (v.count + 15U) / 16U, gx = (v.rows + 16U * Waves - 1U) / (16U * Waves);
    const Plan pl = plan(tiles, v.rows, true);
    const unsigned gz = v.batch > 1U ? v.batch : 1U;  // batched form: one product per grid z
    v.tile0 = 0U;
    if (pl.full) hipLaunchKernelGGL((kernel<16, Waves, Form>), dim3(gx, pl.full, gz), dim3(32U * Waves), 0, stream, v);
    v.tile0 = 16U * pl.full;
    if (pl.tc == 1U) hipLaunchKernelGGL((kernel<1, Waves, Form>), dim3(gx, pl.gy, gz), dim3(32U * Waves), 0, stream, v);
    if (pl.tc == 2U) hipLaunchKernelGGL((kernel<2, Waves, Form>), dim3(gx, pl.gy, gz), dim3(32U * Waves), 0, stream, v);
    if (pl.tc == 4U) hipLaunchKernelGGL((kernel<4, Waves, Form>), dim3(gx, pl.gy, gz), dim3(32U * Waves), 0, stream, v);
}
template <unsigned Waves> inline void launch_form(const View& v, unsigned form, hipStream_t stream) {
    if (form == kWPacked) launch<Waves, kWPacked>(v, stream);
    else launch<Waves, 0U>(v, stream);
}

// Caller checks supported(v) and the launch error (hipGetLastError).
inline void enqueue(const View& v, hipStream_t stream, unsigned waves = 0U, unsigned form = kFormRule) {
    const unsigned f = form_for(form);
    if (waves_for(v.rows, v.in, waves) == 16U) launch_form<16>(v, f, stream);
    else launch_form<8>(v, f, stream);
}

// The operand-image form (StoreImage): same grid, same chains, same CTA rule as enqueue.
inline bool supported_image(const View& v, const std::uint16_t* image) {
    return supported(v) && image != nullptr && v.accumulate == 0U && v.count % 2U == 0U && v.out_stride % 2U == 0U &&
           reinterpret_cast<std::uintptr_t>(image) % 4U == 0U;
}
template <unsigned Waves, class E> inline void launch_image(View v, std::uint16_t* image, typename E::Args e, hipStream_t stream) {
    const unsigned tiles = (v.count + 15U) / 16U, gx = (v.rows + 16U * Waves - 1U) / (16U * Waves);
    const Plan pl = plan(tiles, v.rows, true);
    v.tile0 = 0U;
    if (pl.full) hipLaunchKernelGGL((kernel_image<16, Waves, E>), dim3(gx, pl.full), dim3(32U * Waves), 0, stream, v, image, e);
    v.tile0 = 16U * pl.full;
    if (pl.tc == 1U) hipLaunchKernelGGL((kernel_image<1, Waves, E>), dim3(gx, pl.gy), dim3(32U * Waves), 0, stream, v, image, e);
    if (pl.tc == 2U) hipLaunchKernelGGL((kernel_image<2, Waves, E>), dim3(gx, pl.gy), dim3(32U * Waves), 0, stream, v, image, e);
    if (pl.tc == 4U) hipLaunchKernelGGL((kernel_image<4, Waves, E>), dim3(gx, pl.gy), dim3(32U * Waves), 0, stream, v, image, e);
}
// Caller checks supported_image(v, image) and the launch error.
template <class E> inline void enqueue_image(const View& v, std::uint16_t* image, typename E::Args e, hipStream_t stream, unsigned waves = 0U) {
    if (waves_for(v.rows, v.in, waves) == 16U) launch_image<16, E>(v, image, e, stream);
    else launch_image<8, E>(v, image, e, stream);
}
// Independent FP32 chains per K part, added in unfused order by Store.
// Caller validates each equal-width part; the stream orders the output additions.
inline void enqueue_parts(View v, unsigned parts, hipStream_t stream, unsigned form = kFormRule) {
    for (unsigned part = 0; part < parts; ++part) {
        enqueue(v, stream, 0U, form);
        if (part + 1U == parts) break;
        v.k_bytes += v.in / 32U * 34U;
        v.codes += v.in;
        v.deltas += v.in / 32U;
        v.accumulate = 1U;
    }
}
}  // namespace strix::core::f16_dense_tall::q8a8
