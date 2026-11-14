#include "glm/prefill/glm_pf_exact.hpp"
#include "glm/prefill/glm_pf_dense_policy.hpp"
#include "glm/moe/glm_trellis_device.hpp"
#include "glm/prefill/glm_pf_exact_x.hpp"
#include "glm/prefill/glm_pf_wmma_x.hpp"
#include "strix/route_receipt.hpp"
#include <strix_trellis.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace strix::glm::pf::exact {
namespace {
namespace d = strix::glm::moe::dev;
using Mat = engine::Exl3Raw;
std::size_t align256(std::size_t n) { return (n + 255U) & ~std::size_t{255}; }
bool aligned(const void* p, std::uintptr_t a) { return p && (reinterpret_cast<std::uintptr_t>(p) & (a - 1U)) == 0; }
void ok(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(std::string("glm pf exact: ") + hipGetErrorString(e));
}
// Linear's codebook tag is authoritative, including prepared-only linears.
// Prepared spans use the pinned v03 no-bias file layout.
bool parts(const engine::Linear& l, Mat& m) {
    m = l.raw;
    if (!m.input || !m.output || m.input % 128U || m.output % 128U ||
        m.bits < 2 || m.bits > 8 || m.codebook != 2 ||
        m.input > static_cast<unsigned>(INT32_MAX) || m.output > static_cast<unsigned>(INT32_MAX)) return false;
    const std::size_t limit = std::numeric_limits<std::size_t>::max();
    if (std::size_t{m.input} / 8U > (limit - 1024U) / m.output / m.bits) return false;
    if (l.prepared) {
        const void* base = nullptr;
        std::uint64_t bytes = 0;
        if (strix_weights_device_span(static_cast<strix_weights>(l.prepared), &base, &bytes) != STRIX_SUCCESS || !base) return false;
        const std::size_t tr = std::size_t{m.input} / 8U * m.output * m.bits;
        const std::size_t so = align256(tr), vo = align256(so + std::size_t{m.input} * 2U);
        if (bytes != align256(vo + std::size_t{m.output} * 2U)) return false;
        const auto* b = static_cast<const unsigned char*>(base);
        m.trellis = reinterpret_cast<const std::uint32_t*>(b);
        m.suh = reinterpret_cast<const std::uint16_t*>(b + so);
        m.svh = reinterpret_cast<const std::uint16_t*>(b + vo);
    }
    return aligned(m.trellis, 4) && aligned(m.suh, 2) && aligned(m.svh, 2);
}
// Original-input fused routes partition H128 before converting to K16.
// This matters for e.g. K5 5120->2304. Fused-both uses the same graph.
bool h128_partition(const Mat& m) {
    return (m.bits == 6 && m.input == 5120 && m.output == 1280) ||
        (m.bits == 5 && ((m.input == 4096 && m.output == 1024) ||
         (m.input == 8192 && m.output == 5120) || (m.input == 5120 && m.output == 2304) ||
         (m.input == 2304 && m.output == 5120)));
}
__global__ void input_kernel(const float* x, std::size_t xs, Mat m, unsigned rows, _Float16* t) {
    const std::size_t job = std::size_t{blockIdx.x} * 4U + threadIdx.x / 32U;
    const unsigned blocks = m.input / 128U;
    if (job >= std::size_t{rows} * blocks) return;
    d::transform_block(x + (job / blocks) * xs, m.suh, t + (job / blocks) * m.input,
                       static_cast<unsigned>(job % blocks) * 128U, threadIdx.x & 31U);
}
// Eight waves, one 16-column tile each. Static row indices keep the two
// chains per row in registers. Ascending split totals live in LDS.
template<int Bits, unsigned TM>
__global__ __launch_bounds__(256) void core_kernel(Mat m, const _Float16* t, float* y,
                                                  std::size_t ys, unsigned rows, unsigned splits, bool hp) {
    __shared__ std::uint32_t act[TM][64];
    __shared__ float total[TM][128];
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x / 32U;
    const unsigned row0 = blockIdx.y * TM, col0 = blockIdx.x * 128U;
    const unsigned nr = rows - row0 < TM ? rows - row0 : TM;
    const unsigned coltile = col0 / 16U + wave, kt = m.input / 16U;
    for (unsigned q = threadIdx.x; q < TM * 128U; q += 256U) total[q / 128U][q % 128U] = 0.0F;
    __syncthreads();
    const d::LaneGeom<Bits> geo(lane);
    const auto* weights = reinterpret_cast<const unsigned char*>(d::uniform(m.trellis));
    for (unsigned sp = 0; sp < splits; ++sp) {
        const unsigned units = hp ? m.input / 128U : kt;
        const unsigned begin = static_cast<unsigned>(std::uint64_t{units} * sp / splits) * (hp ? 8U : 1U);
        const unsigned end = static_cast<unsigned>(std::uint64_t{units} * (sp + 1U) / splits) * (hp ? 8U : 1U);
        float sums[TM][2];
#pragma unroll
        for (unsigned r = 0; r < TM; ++r) sums[r][0] = sums[r][1] = 0.0F;
        for (unsigned base = begin; base < end; base += 8U) {
            for (unsigned q = threadIdx.x; q < TM * 64U; q += 256U) {
                const unsigned r = q / 64U, p = q % 64U;
                std::uint32_t v = 0;
                if (r < nr && base + p / 8U < end)
                    __builtin_memcpy(&v, t + std::size_t{row0 + r} * m.input + base * 16U + p * 2U, 4);
                act[r][p] = v;
            }
            __syncthreads();
            for (unsigned kk = 0; kk < 8U && base + kk < end; ++kk) {
                const std::size_t off = (std::size_t{base + kk} * (m.output / 16U) + coltile) * Bits * 32U;
                const d::TileStates<Bits> st(weights + off, geo);
                const d::Half2 w0 = d::mul1_pair(st.template state<0>(), st.template state<1>());
                const d::Half2 w1 = d::mul1_pair(st.template state<2>(), st.template state<3>());
                const d::Half2 w2 = d::mul1_pair(st.template state<4>(), st.template state<5>());
                const d::Half2 w3 = d::mul1_pair(st.template state<6>(), st.template state<7>());
#pragma unroll
                for (unsigned r = 0; r < TM; ++r) {
                    // Padded rows are initialized and isolated, never published.
                    const d::Half2 a0 = __builtin_bit_cast(d::Half2, act[r][kk * 8U + (lane & 3U)]);
                    const d::Half2 a1 = __builtin_bit_cast(d::Half2, act[r][kk * 8U + (lane & 3U) + 4U]);
                    sums[r][0] = __builtin_amdgcn_fdot2(w0, a0, sums[r][0], false);
                    sums[r][0] = __builtin_amdgcn_fdot2(w1, a1, sums[r][0], false);
                    sums[r][1] = __builtin_amdgcn_fdot2(w2, a0, sums[r][1], false);
                    sums[r][1] = __builtin_amdgcn_fdot2(w3, a1, sums[r][1], false);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (unsigned r = 0; r < TM; ++r) {
            const float v0 = d::reduce4(sums[r][0]), v1 = d::reduce4(sums[r][1]);
            if ((lane & 3U) == 0) {
                total[r][wave * 16U + lane / 4U] += v0;
                total[r][wave * 16U + lane / 4U + 8U] += v1;
            }
        }
    }
    __syncthreads();
    for (unsigned r = wave; r < nr; r += 8U) {
        float v[4];
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) v[j] = total[r][lane + 32U * j];
        d::hadamard128(v, lane);
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            const unsigned c = col0 + lane + 32U * j;
            y[std::size_t{row0 + r} * ys + c] = v[j] * d::half_bits(m.svh[c]);
        }
    }
}
struct XRows {
    const _Float16* t;
    float* y;
    std::size_t ys;
    unsigned width, row0;
    __device__ const _Float16* input(unsigned r) const { return t + std::size_t{row0 + r} * width; }
    __device__ float* output(unsigned r) const { return y + std::size_t{row0 + r} * ys; }
};
template<int B, bool Prefetch = false>
__global__ __launch_bounds__(256) void x_kernel(Mat m, const _Float16* t, float* y,
                                               std::size_t ys, unsigned rows, unsigned splits, bool hp) {
    __shared__ exact_x::Storage lds;
    const unsigned row0 = blockIdx.y * 32U;
    const unsigned nr = rows - row0 < 32U ? rows - row0 : 32U;
    exact_x::core<B, Prefetch>(m, XRows{t, y, ys, m.input, row0}, nr, splits, hp, lds);
}
template<int B, unsigned R>
__global__ __launch_bounds__(256) void x3_kernel(Mat m, const _Float16* t, float* y,
                                                std::size_t ys, unsigned rows, unsigned splits, bool hp) {
    __shared__ exact_x::StorageS<R> lds;
    const unsigned row0 = blockIdx.y * 8U * R;
    const unsigned nr = rows - row0 < 8U * R ? rows - row0 : 8U * R;
    exact_x::core_s<B, R>(m, XRows{t, y, ys, m.input, row0}, nr, splits, hp, lds);
}
// FAST glm-pf-wmma-x (STRIX_GLM_PF_WMMA=1): 64-row blocks on the WMMA core.
template<int B, unsigned Rows = 64, unsigned CP = 1, bool FusedInput = false>
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(6))) void w_kernel(Mat m, const _Float16* t, float* y, std::size_t ys, unsigned rows, const float* raw = nullptr, std::size_t xs = 0, unsigned all_frag = 0U) {
    __shared__ wmma_x::StorageR<Rows, FusedInput ? 68U : 36U> lds;
    const unsigned row0 = blockIdx.y * Rows;
    const unsigned nr = rows - row0 < Rows ? rows - row0 : Rows;
    auto at = [&](unsigned r, unsigned) { return XRows{t, y, ys, m.input, row0 + r}; };
    if constexpr (FusedInput) wmma_x::span<B, CP>(m, at, nr, lds, aux::Store{}, aux::H128Input{raw, xs, row0}, all_frag != 0U);
    else wmma_x::span<B, CP>(m, at, nr, lds, aux::Store{}, aux::StagedInput{}, all_frag != 0U);
}
template<int B, bool Fused>
void launch_w(Mat m, const _Float16* t, float* y, std::size_t ys, unsigned rows, hipStream_t s,
              const float* raw, std::size_t xs) {
    const unsigned af = wmma_x::all_fragments();
    if (wmma_x::tile_rows() == 128U && wmma_x::cols_per_wave() == 2U)
        w_kernel<B, 128, 2, Fused><<<dim3(m.output / 128U, (rows - 1U) / 128U + 1U), 256, 0, s>>>(m, t, y, ys, rows, raw, xs, af);
    else if (wmma_x::tile_rows() == 128U)
        w_kernel<B, 128, 1, Fused><<<dim3(m.output / 128U, (rows - 1U) / 128U + 1U), 256, 0, s>>>(m, t, y, ys, rows, raw, xs, af);
    else w_kernel<B, 64, 1, Fused><<<dim3(m.output / 128U, (rows - 1U) / 64U + 1U), 256, 0, s>>>(m, t, y, ys, rows, raw, xs, af);
}
template<int B, unsigned TM>
void launch(Mat m, const _Float16* t, float* y, std::size_t ys, unsigned rows, hipStream_t s, std::size_t* scratch) {
    if constexpr (TM == 5) {
        if (scratch) {
            hipFuncAttributes a{};
            ok(hipFuncGetAttributes(&a, reinterpret_cast<const void*>(&w_kernel<B>)));
            *scratch = a.localSizeBytes;
            return;
        }
        launch_w<B, false>(m, t, y, ys, rows, s, nullptr, 0);
        ok(hipGetLastError());
        return;
    } else if constexpr (TM == 3 || TM == 4) {
        constexpr unsigned R = TM == 3 ? 4U : 8U;
        if (scratch) {
            hipFuncAttributes a{};
            ok(hipFuncGetAttributes(&a, reinterpret_cast<const void*>(&x3_kernel<B, R>)));
            *scratch = a.localSizeBytes;
            return;
        }
        x3_kernel<B, R><<<dim3(m.output / 128U, (rows - 1U) / (8U * R) + 1U), 256, 0, s>>>(
            m, t, y, ys, rows, vector_splits(m.input, m.output, m.bits), h128_partition(m));
        ok(hipGetLastError());
        return;
    } else if constexpr (TM == 0 || TM == 1) {
        if (scratch) {
            hipFuncAttributes a{};
            ok(hipFuncGetAttributes(&a, reinterpret_cast<const void*>(&x_kernel<B, TM == 1>)));
            *scratch = a.localSizeBytes;
            return;
        }
        x_kernel<B, TM == 1><<<dim3(m.output / 128U, (rows - 1U) / 32U + 1U), 256, 0, s>>>(
            m, t, y, ys, rows, vector_splits(m.input, m.output, m.bits), h128_partition(m));
        ok(hipGetLastError());
        return;
    } else {
        if (scratch) {
            hipFuncAttributes a{};
            ok(hipFuncGetAttributes(&a, reinterpret_cast<const void*>(&core_kernel<B, TM>)));
            *scratch = a.localSizeBytes;
            return;
        }
        core_kernel<B, TM><<<dim3(m.output / 128U, (rows - 1U) / TM + 1U), 256, 0, s>>>(
            m, t, y, ys, rows, vector_splits(m.input, m.output, m.bits), h128_partition(m));
        ok(hipGetLastError());
    }
}
template<unsigned TM>
void dispatch(Mat m, const _Float16* t, float* y, std::size_t ys, unsigned rows, hipStream_t s, std::size_t* scratch) {
#define CASE(B) case B: launch<B, TM>(m, t, y, ys, rows, s, scratch); break
    switch (m.bits) {
    CASE(2); CASE(3); CASE(4); CASE(5); CASE(6); CASE(7); CASE(8);
    default: throw std::invalid_argument("glm pf exact: bits");
    }
#undef CASE
}
void dispatch_tm(unsigned tm, Mat m, const _Float16* t, float* y, std::size_t ys, unsigned rows, hipStream_t s,
                 std::size_t* scratch = nullptr) {
    switch (tm) {
    case 1: dispatch<1>(m, t, y, ys, rows, s, scratch); break;
    case 0: dispatch<0>(m, t, y, ys, rows, s, scratch); break;
    case 3: dispatch<3>(m, t, y, ys, rows, s, scratch); break;
    case 4: dispatch<4>(m, t, y, ys, rows, s, scratch); break;
    case 5: dispatch<5>(m, t, y, ys, rows, s, scratch); break;
    case 16: dispatch<16>(m, t, y, ys, rows, s, scratch); break;
    case 32: dispatch<32>(m, t, y, ys, rows, s, scratch); break;
    case 64: dispatch<64>(m, t, y, ys, rows, s, scratch); break;
    default: throw std::invalid_argument("STRIX_GLM_PF_EXACT_TM must be 16, 32 or 64");
    }
}
}  // namespace
unsigned x_auto_rows() {
    // glm_pf_dense_policy.hpp: STRIX_GLM_PF_EXACT_X_MIN_ROWS, else 64 rows (32 with the FAST WMMA identity).
    return dense_policy::auto_min_rows(envcfg::text);
}
struct RouteSelection {
    Route route;
    ::strix::route::Reason bypass;
};
static RouteSelection select_route(unsigned rows) {
    using R = ::strix::route::Reason;
    static_assert(static_cast<unsigned>(Route::tm) == dense_policy::kTm && static_cast<unsigned>(Route::x) == dense_policy::kX &&
                  static_cast<unsigned>(Route::x2) == dense_policy::kX2 && static_cast<unsigned>(Route::x3) == dense_policy::kX3 &&
                  static_cast<unsigned>(Route::x3r8) == dense_policy::kX3r8);
    const auto pick = dense_policy::choose(rows, envcfg::text);
    return {static_cast<Route>(pick.route),
            pick.route == dense_policy::kX3r8 ? R::shape : pick.automatic ? R::short_call : R::wrong_route};
}
Route route(unsigned rows) { return select_route(rows).route; }
unsigned vector_splits(unsigned input, unsigned output, unsigned bits) {
    const unsigned s = bits == 5 && input == 4096 && output == 1024 ? 16U :
        (output <= 1024 ? 32U : (output <= 4096 && input >= 4096 ? 16U : 4U));
    return std::min(s, input / 16U);
}
bool supported(const engine::Linear& l) { Mat m; return parts(l, m); }
std::size_t workspace_bytes(unsigned rows, unsigned input) {
    if (std::size_t{rows} > (std::numeric_limits<std::size_t>::max() - 255U) / 2U / std::max(1U, input))
        throw std::overflow_error("glm pf exact: workspace size");
    return align256(std::size_t{rows} * input * 2U);
}
unsigned tile_rows() {
    const char* p = std::getenv("STRIX_GLM_PF_EXACT_TM");
    // default 16 (measurement: TM16 1.8-2.1x vs strix_gemv 8-row blocks, ~200 GB/s weight stream; TM32/64 slower)
    if (!p || std::strcmp(p, "16") == 0) return 16;
    if (std::strcmp(p, "32") == 0) return 32;
    if (std::strcmp(p, "64") == 0) return 64;
    throw std::invalid_argument("STRIX_GLM_PF_EXACT_TM must be 16, 32 or 64");
}
void linear_rows(const engine::Linear& l, const float* x, std::size_t xs, float* y, std::size_t ys,
                 unsigned rows, void* ws, std::size_t bytes, hipStream_t s) {
    Mat m;
    if (!parts(l, m)) throw std::invalid_argument("glm pf exact: unsupported linear");
    if (!rows || !aligned(x, 4) || !aligned(y, 4) || xs < m.input || ys < m.output ||
        !aligned(ws, 256) || bytes < workspace_bytes(rows, m.input) ||
        std::size_t{rows} * (m.input / 128U) > std::size_t{UINT32_MAX} * 4U ||
        std::size_t{rows} > std::numeric_limits<std::size_t>::max() / sizeof(float) / std::max(xs, ys))
        throw std::invalid_argument("glm pf exact: rows, strides or workspace");
    const auto selection = select_route(rows);
    const Route selected = selection.route;
    const unsigned tm = tile_rows();
    auto* t = static_cast<_Float16*>(ws);
    const auto grid = static_cast<unsigned>((std::size_t{rows} * (m.input / 128U) + 3U) / 4U);
    const unsigned core = selected == Route::x3r8 && wmma_x::enabled() && m.input % 64U == 0 ? 5U : selected == Route::x3r8 ? 4U : selected == Route::x3 ? 3U :
        selected == Route::x2 ? 1U : (selected == Route::x ? 0U : tm);
    if (core == 5U) wmma_x::dense_calls.fetch_add(1, std::memory_order_relaxed);
    // Shared route receipt (count only): the family of this call, and the reason a requested WMMA identity was bypassed.
    if (core == 5U) ::strix::route::launched(::strix::route::Product::dense, ::strix::route::Family::f16_wmma, rows);
    else {
        ::strix::route::launched(::strix::route::Product::dense, ::strix::route::Family::exact, rows);
        if (::strix::route::enabled() && wmma_x::enabled())
            ::strix::route::bypass(::strix::route::Product::dense, ::strix::route::Family::f16_wmma,
                                   selection.bypass, rows);
    }
    if (core == 5U && aux::input_admitted(m.output) && aux::enabled("STRIX_GLM_PF_AUX_INPUT")) {
#define CASE(B) case B: launch_w<B, true>(m, t, y, ys, rows, s, x, xs); break
        switch (m.bits) {
        CASE(2); CASE(3); CASE(4); CASE(5); CASE(6); CASE(7); CASE(8);
        default: throw std::invalid_argument("glm pf exact: bits");
        }
#undef CASE
        ok(hipGetLastError());
        return;
    }
    input_kernel<<<grid, 128, 0, s>>>(x, xs, m, rows, t);
    ok(hipGetLastError());
    dispatch_tm(core, m, t, y, ys, rows, s);
}
hipFuncAttributes x_resources(const engine::Linear& l, unsigned variant) {
    Mat m;
    if (!parts(l, m)) throw std::invalid_argument("glm pf exact: unsupported linear");
    hipFuncAttributes a{};
    if (variant < 1 || variant > 4) throw std::invalid_argument("glm pf exact: X resources variant must be 1..4");
#define CASE(B) case B: if (variant == 4) { ok(hipFuncGetAttributes(&a, reinterpret_cast<const void*>(&x3_kernel<B, 8>))); } \
                       else if (variant == 3) { ok(hipFuncGetAttributes(&a, reinterpret_cast<const void*>(&x3_kernel<B, 4>))); } \
                       else if (variant == 2) { ok(hipFuncGetAttributes(&a, reinterpret_cast<const void*>(&x_kernel<B, true>))); } \
                       else { ok(hipFuncGetAttributes(&a, reinterpret_cast<const void*>(&x_kernel<B>))); } break
    switch (m.bits) {
    CASE(2); CASE(3); CASE(4); CASE(5); CASE(6); CASE(7); CASE(8);
    }
#undef CASE
    return a;
}
std::size_t scratch_bytes(const engine::Linear& l, unsigned tm) {
    Mat m;
    if (!parts(l, m)) throw std::invalid_argument("glm pf exact: unsupported linear");
    std::size_t bytes = 0;
    dispatch_tm(tm, m, nullptr, nullptr, 0, 1, nullptr, &bytes);
    return bytes;
}
}  // namespace strix::glm::pf::exact
