// EXL3 / trellis HIP migration fixture. It launches the incumbent device helpers of src/moe_core/exl3_trellis.hpp (input
// transform in its three forms, the split core stream_column, the LDS-free ring depths) and the MiniMax / GLM prefill
// epilogues finish_block over seeded synthetic experts with heterogeneous suh / svh / trellis words (three experts, each its
// own sidecars), same-TU included like the other kernel fixtures. Under STRIX_WFMT_CAPTURE it records the weight bytes,
// activations, the transformed input, the raw core outputs, the finished outputs and the per-thread trace of the four
// stages (trellis_partial_trace.hpp hooks); the Gemma EXL3 public device entry points (input, core routes, output, linear,
// many-row, prefill) are captured by their outputs. Without it the self-checks still run: the three input-transform forms,
// every chunk partition and a host oracle agree bit for bit, the core is independent of rows-per-wave and ring depth and
// matches a double-precision dot of the decoded coefficients, the MiniMax split finish equals the GLM single-row finish and a
// host H128 + svh, different experts give different results, and Gemma's transport tiles are never addressed by the nominal
// row_bytes. STRIX_WFMT_TRELLIS_REPORT=1 reports a mismatch and keeps going (diagnosis); the default throws.
#include "moe_core/exl3_trellis.hpp"
#include "minimax/moe/mmx_moe_traits.hpp"
#if defined(STRIX_WFMT_GLM) && STRIX_WFMT_GLM
#include "glm/prefill/glm_moe_traits.hpp"
#endif
#include "core/trellis_codec.hpp"
#include "strix/env_flags.hpp"
#include "hip/weight/registry.hpp"
#include "trellis_checks.hpp"
#if defined(STRIX_WFMT_GEMMA) && STRIX_WFMT_GEMMA
#include "gemma4/exl3/device.hpp"
#endif
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
// Quiet transfers (before the capture header redirects hipMemcpy): nothing here is recorded implicitly.
void raw_upload(void* dst, const void* src, std::size_t n) { check(hipMemcpy(dst, src, n, hipMemcpyHostToDevice)); }
void raw_download(void* dst, const void* src, std::size_t n) { check(hipMemcpy(dst, src, n, hipMemcpyDeviceToHost)); }
}  // namespace
#ifdef STRIX_WFMT_CAPTURE
#include "../../scripts/bench/wfmt/hip_transfer_capture.hpp"
#endif

namespace {
std::string g_arithmetic = "hip-exl3-mul1-legacy@1";
void context(const std::string& name, std::initializer_list<std::uint64_t> shape, const char* format) {
#ifdef STRIX_WFMT_CAPTURE
    strix::weight::capture::set_context(name.c_str(), g_arithmetic.c_str(), shape, format);
#else
    (void)name; (void)shape; (void)format;
#endif
}
void emit(const char* section, const void* data, std::size_t bytes) {
#ifdef STRIX_WFMT_CAPTURE
    strix::weight::capture::emit(section, data, bytes);
#else
    (void)section; (void)data; (void)bytes;
#endif
}
template <class T> void emit_vec(const char* section, const std::vector<T>& v) { emit(section, v.data(), v.size() * sizeof(T)); }

bool g_report = false;
unsigned g_cases = 0, g_failures = 0;
unsigned g_glm_checks = 0, g_gemma_cases = 0;
strix::wfmt::TrellisCoverage g_coverage{
#if defined(STRIX_WFMT_GLM) && STRIX_WFMT_GLM
    true,
#else
    false,
#endif
#if defined(STRIX_WFMT_GEMMA) && STRIX_WFMT_GEMMA
    true,
#else
    false,
#endif
    false, false};
void expect(bool ok, const std::string& what) {
    if (ok) return;
    ++g_failures;
    std::cerr << "MISMATCH " << what << '\n';
    if (!g_report) throw std::runtime_error("trellis fixture: " + what);
}
template <class T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && (a.empty() || !std::memcmp(a.data(), b.data(), a.size() * sizeof(T)));
}

struct Buffer {
    void* p = nullptr;
    std::size_t bytes = 0;
    explicit Buffer(std::size_t n) : bytes(n) { check(hipMalloc(&p, n)); check(hipMemset(p, 0, n)); }
    ~Buffer() { (void)hipFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template <class T> T* as() const { return static_cast<T*>(p); }
    template <class T> void put(const std::vector<T>& v) const { raw_upload(p, v.data(), v.size() * sizeof(T)); }
    void clear() const { check(hipMemset(p, 0, bytes)); }
    template <class T> std::vector<T> read(std::size_t n) const {
        std::vector<T> v(n);
        raw_download(v.data(), p, n * sizeof(T));
        return v;
    }
};

struct Rng {
    std::uint64_t s;
    explicit Rng(const std::string& name) : s(UINT64_C(14695981039346656037)) {
        for (const char c : name) s = (s ^ static_cast<unsigned char>(c)) * UINT64_C(1099511628211);
        if (!s) s = 1;
    }
    std::uint32_t next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return static_cast<std::uint32_t>((s * UINT64_C(2685821657736338717)) >> 32); }
};

// ---- host halves / float helpers (round to nearest even, finite range only)
float half_to_float(std::uint16_t h) {
    const unsigned sign = h >> 15U, exp = (h >> 10U) & 31U, man = h & 1023U;
    float v;
    if (exp == 0U) v = std::ldexp(static_cast<float>(man), -24);
    else v = std::ldexp(static_cast<float>(1024U + man), static_cast<int>(exp) - 25);
    return sign ? -v : v;
}
std::uint16_t float_to_half(float f) {
    std::uint32_t b;
    std::memcpy(&b, &f, 4);
    const std::uint32_t sign = (b >> 16U) & 0x8000U, mag = b & 0x7fffffffU;
    if (mag >= 0x7f800000U) return static_cast<std::uint16_t>(sign | 0x7c00U | (mag > 0x7f800000U ? 0x200U : 0U));
    if (mag >= 0x477ff000U) return static_cast<std::uint16_t>(sign | 0x7c00U);  // rounds to infinity
    if (mag <= 0x33000000U) return static_cast<std::uint16_t>(sign);            // <= 2^-25 rounds to zero
    const unsigned e = mag >> 23U;
    const std::uint32_t m = mag & 0x7fffffU;
    std::uint32_t q;
    if (e < 113U) {  // half subnormal
        const unsigned shift = 126U - e;
        const std::uint32_t full = m | 0x800000U;
        q = full >> shift;
        const std::uint32_t rem = full & ((1U << shift) - 1U), half = 1U << (shift - 1U);
        if (rem > half || (rem == half && (q & 1U))) ++q;
    } else {
        q = ((e - 112U) << 10U) | (m >> 13U);
        const std::uint32_t rem = m & 0x1fffU;
        if (rem > 0x1000U || (rem == 0x1000U && (q & 1U))) ++q;
    }
    return static_cast<std::uint16_t>(sign | q);
}
// Ascending-stride H128 in FP32 (strides 1..64), then RN32(* 0x1.6a09e6p-4): the shared hadamard128 statement order.
void host_h128(float* v) {
    for (unsigned stride = 1; stride < 128; stride *= 2)
        for (unsigned base = 0; base < 128; base += 2 * stride)
            for (unsigned j = 0; j < stride; ++j) {
                const float a = v[base + j], b = v[base + stride + j];
                v[base + j] = a + b;
                v[base + stride + j] = a - b;
            }
    for (unsigned i = 0; i < 128; ++i) v[i] = v[i] * 0x1.6a09e6p-4f;
}

// ---- synthetic experts: independent suh / svh / trellis words each
constexpr unsigned kK = 256, kN = 256, kExperts = 3, kRowsMax = 257;
struct Expert {
    std::vector<std::uint32_t> trellis;
    std::vector<std::uint16_t> suh, svh;
};
std::uint16_t scale_half(Rng& r) {
    const std::uint32_t t = r.next();
    const std::uint32_t sign = (t & 1U) << 15U, pick = (t >> 1U) % 32U;
    if (pick == 0U) return static_cast<std::uint16_t>(sign);                                   // +-0
    if (pick == 1U) return static_cast<std::uint16_t>(sign | ((t >> 8U) % 1023U + 1U));        // subnormal
    const std::uint32_t exp = 9U + (t >> 12U) % 12U;                                            // 2^-6 .. 2^5
    return static_cast<std::uint16_t>(sign | (exp << 10U) | ((t >> 16U) & 1023U));
}
Expert make_expert(unsigned bits, unsigned index) {
    Rng r("trellis-expert-" + std::to_string(bits) + "-" + std::to_string(index));
    Expert e;
    e.trellis.resize(static_cast<std::size_t>(kK / 16U) * (kN / 16U) * bits * 8U);
    for (auto& w : e.trellis) w = r.next();
    for (unsigned i = 0; i < kK; ++i) e.suh.push_back(scale_half(r));
    for (unsigned i = 0; i < kN; ++i) e.svh.push_back(scale_half(r));
    return e;
}
std::vector<float> make_activations(const std::string& name, std::size_t n) {
    Rng r(name);
    std::vector<float> x(n);
    for (auto& a : x) {
        const std::uint32_t t = r.next();
        a = (t % 16U == 0U) ? ((t & 16U) ? -0.0F : 0.0F)
                            : std::ldexp(static_cast<float>(static_cast<int>(r.next() % 2000001U) - 1000000), static_cast<int>(r.next() % 8U) - 22);
    }
    return x;
}

struct ExpertDev {
    const std::uint32_t* trellis;
    const std::uint16_t* suh;
    const std::uint16_t* svh;
};

// Diagnostic trace of the four stages (hooks compiled only under STRIX_WFMT_CAPTURE); armed per launch.
struct Trace {
#ifdef STRIX_WFMT_CAPTURE
    std::unique_ptr<Buffer> values, counts;
    unsigned threads = 0;
    explicit Trace(unsigned t) : values(std::make_unique<Buffer>(static_cast<std::size_t>(t) * strix::wfmt_trace::kTrellisStreams * strix::wfmt_trace::kTrellisSlots * 4U)),
        counts(std::make_unique<Buffer>(static_cast<std::size_t>(t) * strix::wfmt_trace::kTrellisStreams * 4U)), threads(t) {
        auto* v = values->as<unsigned>(); auto* c = counts->as<unsigned>();
        check(hipMemcpyToSymbol(HIP_SYMBOL(strix::wfmt_trace::trellis_values), &v, sizeof v));
        check(hipMemcpyToSymbol(HIP_SYMBOL(strix::wfmt_trace::trellis_counts), &c, sizeof c));
    }
    ~Trace() {
        unsigned* v = nullptr; unsigned* c = nullptr;
        (void)hipMemcpyToSymbol(HIP_SYMBOL(strix::wfmt_trace::trellis_values), &v, sizeof v);
        (void)hipMemcpyToSymbol(HIP_SYMBOL(strix::wfmt_trace::trellis_counts), &c, sizeof c);
    }
    // Compact words: for each thread with records, [thread, count x 4, values of stream 0.., 1.., 2.., 3..].
    std::vector<std::uint32_t> packed() const {
        constexpr unsigned S = strix::wfmt_trace::kTrellisStreams, N = strix::wfmt_trace::kTrellisSlots;
        const auto used = counts->read<unsigned>(static_cast<std::size_t>(threads) * S);
        const auto all = values->read<std::uint32_t>(static_cast<std::size_t>(threads) * S * N);
        std::vector<std::uint32_t> packed;
        for (unsigned t = 0; t < threads; ++t) {
            unsigned total = 0;
            for (unsigned s = 0; s < S; ++s) total += used[t * S + s];
            if (!total) continue;
            packed.push_back(t);
            for (unsigned s = 0; s < S; ++s) {
                if (used[t * S + s] > N) throw std::runtime_error("diagnostic trace capacity");
                packed.push_back(used[t * S + s]);
            }
            for (unsigned s = 0; s < S; ++s)
                packed.insert(packed.end(), all.begin() + (static_cast<std::size_t>(t) * S + s) * N,
                              all.begin() + (static_cast<std::size_t>(t) * S + s) * N + used[t * S + s]);
        }
        if (packed.empty()) throw std::runtime_error("missing diagnostic trace");
        return packed;
    }
#else
    explicit Trace(unsigned) {}
    std::vector<std::uint32_t> packed() const { return {}; }
#endif
};
template <class F> void launch(bool traced, unsigned threads, const char* section, F&& body) {
    if (!traced) { body(); check(hipDeviceSynchronize()); return; }
    Trace trace(threads);
    body();
    check(hipDeviceSynchronize());
    const auto words = trace.packed();
    if (!words.empty()) emit(section, words.data(), words.size() * 4U);
}

// ---- kernels over the incumbent helpers
constexpr unsigned kThreads = 128;  // four waves of 32
template <int Bits> using Codec = strix::moe_core::Exl3Trellis<Bits>;

// Variant 0: Exl3Trellis::input_block (transform_block); 1: transform_block_at; 2: transform_values + a store here.
template <int Bits, int Variant>
__global__ void input_probe(const ExpertDev* experts, const std::uint32_t* sel, const float* x, _Float16* out, unsigned row0, unsigned rows) {
    const unsigned lane = threadIdx.x & 31U;
    const unsigned job = blockIdx.x * 4U + (threadIdx.x >> 5U);
    constexpr unsigned blocks = kK / 128U;
    if (job >= rows * blocks) return;
    const unsigned row = row0 + job / blocks, begin = job % blocks * 128U;
    const std::uint16_t* suh = experts[sel[row]].suh;
    const float* xr = x + static_cast<std::size_t>(row) * kK;
    _Float16* o = out + static_cast<std::size_t>(row) * kK;
    if constexpr (Variant == 0) {
        Codec<Bits>::input_block(xr, suh, o, begin, lane);
    } else if constexpr (Variant == 1) {
        strix::moe_core::transform_block_at(xr, suh, o, begin, begin, lane);
    } else {
        float v[4];
        strix::moe_core::transform_values(xr, suh, begin, lane, v);
#pragma unroll
        for (int j = 0; j < 4; ++j) o[begin + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
    }
}

// One wave per output tile column: splits [s0, s1) of the expert's matrix over rows [row0, row0 + nrows) of xt.
template <int Bits, int R, int Depth>
__global__ void core_probe(const ExpertDev* experts, unsigned expert, const _Float16* xt, unsigned row0, unsigned nrows,
                           unsigned splits, unsigned s0, unsigned s1, float* part) {
    const unsigned lane = threadIdx.x & 31U;
    const unsigned col = blockIdx.x * 4U + (threadIdx.x >> 5U);
    if (col >= kN / 16U) return;
    float acc[R][2];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = 0.0F;
    strix::moe_core::stream_column<Bits, R, Depth>(experts[expert].trellis, kN, col, kK / 16U, splits, s0, s1,
                                                   xt + static_cast<std::size_t>(row0) * kK, kK, nrows, lane, acc);
    if ((lane & 3U) == 0U)
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (static_cast<unsigned>(r) < nrows)
#pragma unroll
                for (int h = 0; h < 2; ++h)
                    part[static_cast<std::size_t>(r + row0) * kN + col * 16U + lane / 4U + 8U * static_cast<unsigned>(h)] = acc[r][h];
}

// Variant 0: MiniMax Epilogue::finish_block over S split partials; 1: GLM MoeTraits::finish_block over one partial row.
template <int Variant>
__global__ void finish_probe(const ExpertDev* experts, const std::uint32_t* sel, const float* part, unsigned splits, unsigned rows,
                             float* out) {
    const unsigned lane = threadIdx.x & 31U;
    const unsigned job = blockIdx.x * 4U + (threadIdx.x >> 5U);
    constexpr unsigned blocks = kN / 128U;
    if (job >= rows * blocks) return;
    const unsigned row = job / blocks, b = job % blocks;
    const std::uint16_t* svh = experts[sel[row]].svh;
    float v[4];
    if constexpr (Variant == 0) strix::minimax::moe::opt::Epilogue::finish_block(part, splits, rows, row, kN, b, svh, lane, v);
    else {
#if defined(STRIX_WFMT_GLM) && STRIX_WFMT_GLM
        strix::glm::pf::MoeTraits::finish_block(part + static_cast<std::size_t>(row) * kN, b, svh, lane, v);
#else
        // Same statements as the GLM epilogue (v = 0 + row; H128; * svh), used when the GLM headers are not built.
        float acc[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = 0.0F + part[static_cast<std::size_t>(row) * kN + b * 128U + lane + 32U * static_cast<unsigned>(j)];
        strix::moe_core::hadamard128(acc, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = acc[j] * strix::moe_core::half_bits(svh[b * 128U + lane + 32U * static_cast<unsigned>(j)]);
#endif
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) out[static_cast<std::size_t>(row) * kN + b * 128U + lane + 32U * static_cast<unsigned>(j)] = v[j];
}

// ---- per-bits device state
template <int Bits> struct Device {
    std::vector<Expert> host;
    std::vector<std::unique_ptr<Buffer>> trellis, suh, svh;
    std::unique_ptr<Buffer> experts, sel, x, xt, raw, part, out;
    std::vector<float> activations;
    Device() {
        std::vector<ExpertDev> dev;
        for (unsigned e = 0; e < kExperts; ++e) {
            host.push_back(make_expert(Bits, e));
            trellis.push_back(std::make_unique<Buffer>(host[e].trellis.size() * 4U)); trellis[e]->put(host[e].trellis);
            suh.push_back(std::make_unique<Buffer>(kK * 2U)); suh[e]->put(host[e].suh);
            svh.push_back(std::make_unique<Buffer>(kN * 2U)); svh[e]->put(host[e].svh);
            dev.push_back({trellis[e]->as<std::uint32_t>(), suh[e]->as<std::uint16_t>(), svh[e]->as<std::uint16_t>()});
        }
        experts = std::make_unique<Buffer>(dev.size() * sizeof(ExpertDev)); experts->put(dev);
        sel = std::make_unique<Buffer>(kRowsMax * 4U);
        x = std::make_unique<Buffer>(static_cast<std::size_t>(kRowsMax) * kK * 4U);
        xt = std::make_unique<Buffer>(static_cast<std::size_t>(kRowsMax) * kK * 2U);
        raw = std::make_unique<Buffer>(static_cast<std::size_t>(kRowsMax) * kN * 4U);
        part = std::make_unique<Buffer>(static_cast<std::size_t>(16) * kRowsMax * kN * 4U);
        out = std::make_unique<Buffer>(static_cast<std::size_t>(kRowsMax) * kN * 4U);
        activations = make_activations("trellis-x-" + std::to_string(Bits), static_cast<std::size_t>(kRowsMax) * kK);
        x->put(activations);
    }
    void emit_sources(const char* label) const {
        for (unsigned e = 0; e < kExperts; ++e) {
            context(std::string(label) + ".expert" + std::to_string(e), {kK, kN, static_cast<std::uint64_t>(Bits)}, "exl3.mul1.canonical@1");
            emit_vec("source", host[e].trellis);
            std::vector<std::uint16_t> sidecars = host[e].suh;
            sidecars.insert(sidecars.end(), host[e].svh.begin(), host[e].svh.end());
            emit_vec("sidecars", sidecars);
        }
    }
};

unsigned grid_for(unsigned jobs) { return (jobs + 3U) / 4U; }

template <int Bits> std::vector<std::uint16_t> run_input(Device<Bits>& d, int variant, const std::vector<std::uint32_t>& sel,
                                                         const std::vector<std::pair<unsigned, unsigned>>& parts, bool traced, const char* section) {
    d.sel->put(sel);
    check(hipMemset(d.xt->p, 0xff, d.xt->bytes));
    for (const auto& [row0, rows] : parts) {
        const unsigned jobs = rows * (kK / 128U), grid = grid_for(jobs);
        launch(traced, grid * kThreads, section, [&] {
            if (variant == 0) input_probe<Bits, 0><<<grid, kThreads>>>(d.experts->template as<ExpertDev>(), d.sel->template as<std::uint32_t>(), d.x->template as<float>(), d.xt->template as<_Float16>(), row0, rows);
            else if (variant == 1) input_probe<Bits, 1><<<grid, kThreads>>>(d.experts->template as<ExpertDev>(), d.sel->template as<std::uint32_t>(), d.x->template as<float>(), d.xt->template as<_Float16>(), row0, rows);
            else input_probe<Bits, 2><<<grid, kThreads>>>(d.experts->template as<ExpertDev>(), d.sel->template as<std::uint32_t>(), d.x->template as<float>(), d.xt->template as<_Float16>(), row0, rows);
        });
    }
    return d.xt->template read<std::uint16_t>(static_cast<std::size_t>(parts.back().first + parts.back().second) * kK);
}

template <int Bits> std::vector<std::uint16_t> host_input(const Device<Bits>& d, const std::vector<std::uint32_t>& sel, unsigned rows) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(rows) * kK);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned b = 0; b < kK / 128U; ++b) {
            float v[128];
            for (unsigned i = 0; i < 128U; ++i) {
                const unsigned c = b * 128U + i;
                const float scaled = d.activations[static_cast<std::size_t>(r) * kK + c] * half_to_float(d.host[sel[r]].suh[c]);
                v[i] = half_to_float(float_to_half(scaled));
            }
            host_h128(v);
            for (unsigned i = 0; i < 128U; ++i) out[static_cast<std::size_t>(r) * kK + b * 128U + i] = float_to_half(v[i]);
        }
    return out;
}

template <int Bits> void input_cases() {
    Device<Bits> d;
    const std::string label = "trellis.b" + std::to_string(Bits);
    d.emit_sources(label.c_str());
    for (const unsigned rows : {1U, 2U, 7U, 8U, 9U, 31U, 32U, 33U, 63U, 64U, 65U, 127U, 128U, 129U, 257U}) {
        std::vector<std::uint32_t> sel(kRowsMax);
        for (unsigned r = 0; r < kRowsMax; ++r) sel[r] = (r * 5U + r / 3U) % kExperts;
        context(label + ".input.rows" + std::to_string(rows), {rows, kK, static_cast<std::uint64_t>(Bits)}, "exl3.input@1");
        emit("state", sel.data(), rows * 4U);
        emit("activation_codes", d.activations.data(), static_cast<std::size_t>(rows) * kK * 4U);
        const auto full = run_input<Bits>(d, 0, sel, {{0U, rows}}, rows == 33U, "partials");
        emit_vec("prepared", full);
        const auto at = run_input<Bits>(d, 1, sel, {{0U, rows}}, false, "partials");
        const auto vals = run_input<Bits>(d, 2, sel, {{0U, rows}}, false, "partials");
        expect(same(full, at) && same(full, vals), label + " input forms rows " + std::to_string(rows));
        expect(same(full, host_input<Bits>(d, sel, rows)), label + " input host oracle rows " + std::to_string(rows));
        for (const unsigned part : {1U, 7U, 16U, 32U, 64U, 128U}) {
            std::vector<std::pair<unsigned, unsigned>> parts;
            for (unsigned r = 0; r < rows; r += part) parts.push_back({r, std::min(part, rows - r)});
            expect(same(full, run_input<Bits>(d, 0, sel, parts, false, "partials")), label + " input chunks " + std::to_string(part) + " rows " + std::to_string(rows));
        }
        ++g_cases;
    }
    // Two experts with different suh give different transformed inputs for the same activation row.
    std::vector<std::uint32_t> a(kRowsMax, 0U), b(kRowsMax, 1U), c(kRowsMax, 2U);
    std::vector<float> one(static_cast<std::size_t>(kRowsMax) * kK);
    for (unsigned r = 0; r < kRowsMax; ++r) std::copy(d.activations.begin(), d.activations.begin() + kK, one.begin() + static_cast<std::size_t>(r) * kK);
    d.x->put(one);
    const auto ra = run_input<Bits>(d, 0, a, {{0U, 1U}}, false, "partials"), rb = run_input<Bits>(d, 0, b, {{0U, 1U}}, false, "partials"),
               rc = run_input<Bits>(d, 0, c, {{0U, 1U}}, false, "partials");
    expect(!same(ra, rb) && !same(rb, rc) && !same(ra, rc), label + " distinct suh give identical transformed inputs");
    d.x->put(d.activations);
}

// Core + finish: rows grouped by expert (counts per expert), per-split partials, MiniMax / GLM finish, host oracle.
template <int Bits, int R, int Depth>
void core_pass(Device<Bits>& d, const std::vector<unsigned>& counts, unsigned splits, bool traced) {
    unsigned row0 = 0;
    for (unsigned e = 0; e < kExperts; ++e) {
        for (unsigned begin = 0; begin < counts[e]; begin += R) {
            const unsigned n = std::min<unsigned>(R, counts[e] - begin);
            for (unsigned s = 0; s < splits; ++s) {
                const unsigned grid = (kN / 16U + 3U) / 4U;
                launch(traced && e == 0U && begin == 0U && s == 0U, grid * kThreads, "partials", [&] {
                    core_probe<Bits, R, Depth><<<grid, kThreads>>>(d.experts->template as<ExpertDev>(), e, d.xt->template as<_Float16>(), row0 + begin, n,
                                                                    splits, s, s + 1U, d.part->template as<float>() + static_cast<std::size_t>(s) * kRowsMax * kN);
                });
            }
        }
        row0 += counts[e];
    }
}

template <int Bits> void core_cases() {
    Device<Bits> d;
    const std::string label = "trellis.b" + std::to_string(Bits);
    struct Pattern { const char* name; std::array<unsigned, 3> weights; };
    for (const unsigned total : {1U, 8U, 9U, 33U})
        for (const Pattern pat : {Pattern{"all1", {0, 1, 0}}, Pattern{"empty1", {1, 0, 2}}, Pattern{"head1", {1, 4, 0}}}) {
            const unsigned sum = pat.weights[0] + pat.weights[1] + pat.weights[2];
            std::vector<unsigned> counts(3);
            unsigned given = 0, last = 0;
            for (unsigned e = 0; e < 3U; ++e) { counts[e] = total * pat.weights[e] / sum; given += counts[e]; if (pat.weights[e]) last = e; }
            counts[last] += total - given;
            std::vector<std::uint32_t> sel;
            for (unsigned e = 0; e < 3U; ++e) sel.insert(sel.end(), counts[e], e);
            sel.resize(kRowsMax, 0U);
            std::vector<std::pair<unsigned, unsigned>> all{{0U, total}};
            const std::string name = label + ".core.rows" + std::to_string(total) + "." + pat.name;
            context(name, {total, kK, kN, static_cast<std::uint64_t>(Bits)}, "exl3.core@1");
            emit("state", sel.data(), total * 4U);
            const auto xt = run_input<Bits>(d, 0, sel, all, false, "partials");
            emit_vec("prepared", xt);
            std::vector<std::vector<float>> results;
            for (const unsigned splits : {1U, 2U, 4U, 16U}) {
                const auto full = [&](auto&& pass) {
                    check(hipMemset(d.part->p, 0, d.part->bytes));
                    pass();
                    return d.part->template read<float>(static_cast<std::size_t>(splits) * kRowsMax * kN);
                };
                const auto r8 = full([&] { core_pass<Bits, 8, 0>(d, counts, splits, splits == 4U); });
                const auto r1 = full([&] { core_pass<Bits, 1, 0>(d, counts, splits, false); });
                const auto r4 = full([&] { core_pass<Bits, 4, 0>(d, counts, splits, false); });
                const auto d2 = full([&] { core_pass<Bits, 8, 2>(d, counts, splits, false); });
                const auto d4 = full([&] { core_pass<Bits, 4, 4>(d, counts, splits, false); });
                // Only the written rows are comparable (the buffer is zero elsewhere in every arm).
                expect(same(r8, r1) && same(r8, r4), name + " core rows-per-wave splits " + std::to_string(splits));
                expect(same(r8, d2) && same(r8, d4), name + " core ring depth splits " + std::to_string(splits));
                std::vector<float> packed(static_cast<std::size_t>(splits) * total * kN);
                for (unsigned s = 0; s < splits; ++s)
                    std::copy(r8.begin() + static_cast<std::size_t>(s) * kRowsMax * kN, r8.begin() + (static_cast<std::size_t>(s) * kRowsMax + total) * kN,
                              packed.begin() + static_cast<std::size_t>(s) * total * kN);
                emit_vec("partials", packed);
                if (splits == 4U || splits == 1U) {
                    // finish: per-split partials -> MiniMax finish; full sums -> GLM finish; both against the host.
                    d.sel->put(sel);
                    const unsigned grid = grid_for(total * (kN / 128U));
                    // MiniMax layout is [split][row][output], rows = total rows (packed above).
                    Buffer pk(packed.size() * 4U); pk.put(packed);
                    launch(splits == 4U, grid * kThreads, "partials", [&] {
                        finish_probe<0><<<grid, kThreads>>>(d.experts->template as<ExpertDev>(), d.sel->template as<std::uint32_t>(), pk.as<float>(), splits, total, d.out->template as<float>());
                    });
                    const auto mm = d.out->template read<float>(static_cast<std::size_t>(total) * kN);
                    std::vector<float> sums(static_cast<std::size_t>(total) * kN, 0.0F);
                    for (unsigned s = 0; s < splits; ++s)
                        for (std::size_t i = 0; i < sums.size(); ++i) sums[i] += packed[static_cast<std::size_t>(s) * total * kN + i];
#if defined(STRIX_WFMT_GLM) && STRIX_WFMT_GLM
                    if (g_coverage.run_glm()) {
                        Buffer sb(sums.size() * 4U); sb.put(sums);
                        launch(false, 0, "partials", [&] {
                            finish_probe<1><<<grid, kThreads>>>(d.experts->template as<ExpertDev>(), d.sel->template as<std::uint32_t>(), sb.as<float>(), 1U, total, d.out->template as<float>());
                        });
                        const auto glm = d.out->template read<float>(static_cast<std::size_t>(total) * kN);
                        expect(same(mm, glm), name + " MiniMax split finish vs GLM finish splits " + std::to_string(splits));
                        ++g_glm_checks;
                    }
#endif
                    std::vector<float> ref(mm.size());
                    for (unsigned r = 0; r < total; ++r)
                        for (unsigned b = 0; b < kN / 128U; ++b) {
                            float v[128];
                            for (unsigned i = 0; i < 128U; ++i) v[i] = sums[static_cast<std::size_t>(r) * kN + b * 128U + i];
                            host_h128(v);
                            for (unsigned i = 0; i < 128U; ++i)
                                ref[static_cast<std::size_t>(r) * kN + b * 128U + i] = v[i] * half_to_float(d.host[sel[r]].svh[b * 128U + i]);
                        }
                    expect(same(mm, ref), name + " finish host oracle splits " + std::to_string(splits));
                    emit_vec("output", mm);
                    if (splits == 4U) results.push_back(mm);
                    // Coefficient oracle: the raw sums track a double-precision dot of the decoded coefficients (right expert,
                    // right tile addressing); ordering differences stay far below the tolerance.
                    if (splits == 1U)
                        for (unsigned r = 0; r < total; ++r)
                            for (unsigned n = 0; n < kN; ++n) {
                                double ref_dot = 0.0, mag = 0.0;
                                for (unsigned k = 0; k < kK; ++k) {
                                    const double w = half_to_float(strix::core::trellis::reference_bits<strix::core::trellis::Mul1>(d.host[sel[r]].trellis.data(), Bits, kN, k, n));
                                    const double a = half_to_float(xt[static_cast<std::size_t>(r) * kK + k]);
                                    ref_dot += w * a; mag += std::fabs(w * a);
                                }
                                const double got = sums[static_cast<std::size_t>(r) * kN + n];
                                if (!strix::wfmt::coefficient_matches(got, ref_dot, mag)) { expect(false, name + " core vs coefficient oracle at " + std::to_string(r) + "," + std::to_string(n)); break; }
                            }
                }
            }
            ++g_cases;
        }
    // Heterogeneous sidecars: one activation row through each expert gives three different finished rows.
    {
        std::vector<float> one(static_cast<std::size_t>(kRowsMax) * kK);
        for (unsigned r = 0; r < kRowsMax; ++r) std::copy(d.activations.begin(), d.activations.begin() + kK, one.begin() + static_cast<std::size_t>(r) * kK);
        d.x->put(one);
        std::vector<std::uint32_t> sel{0U, 1U, 2U};
        sel.resize(kRowsMax, 0U);
        const auto xt = run_input<Bits>(d, 0, sel, {{0U, 3U}}, false, "partials");
        expect(!std::equal(xt.begin(), xt.begin() + kK, xt.begin() + kK), label + " experts 0/1 same transformed row");
        check(hipMemset(d.part->p, 0, d.part->bytes));
        core_pass<Bits, 1, 0>(d, {1U, 1U, 1U}, 1U, false);
        const auto raw = d.part->template read<float>(static_cast<std::size_t>(3) * kN);
        expect(!std::equal(raw.begin(), raw.begin() + kN, raw.begin() + kN), label + " experts 0/1 same core row");
        d.x->put(d.activations);
    }
}

// ---- Gemma EXL3: public device entry points, identity from the process (STRIX_G4_EXL3_IDENTITY)
#if defined(STRIX_WFMT_GEMMA) && STRIX_WFMT_GEMMA
namespace g4 = strix::gemma4;
namespace ex = g4::exl3;
std::vector<std::uint8_t> gemma_pack(unsigned inputs, unsigned outputs, unsigned k, unsigned seed) {
    Rng r("gemma-pack-" + std::to_string(inputs) + "-" + std::to_string(outputs) + "-" + std::to_string(k) + "-" + std::to_string(seed));
    const std::size_t bytes = 2ULL * (inputs + outputs) + static_cast<std::size_t>(inputs / 16U) * (outputs / 16U) * 32U * k;
    std::vector<std::uint8_t> data(bytes);
    for (unsigned i = 0; i < inputs + outputs; ++i) {
        const auto h = scale_half(r);
        data[2ULL * i] = static_cast<std::uint8_t>(h); data[2ULL * i + 1] = static_cast<std::uint8_t>(h >> 8U);
    }
    for (std::size_t i = 2ULL * (inputs + outputs); i < bytes; ++i) data[i] = static_cast<std::uint8_t>(r.next());
    return data;
}
void gemma_cases() {
    const std::string identity = ex::active_identity();
    g_arithmetic = identity + "@1";
    for (const unsigned k : {4U, 5U, 6U, 8U})
        for (const unsigned inputs : {256U, 384U})
            for (const unsigned outputs : {128U, 256U}) {
                std::array<std::vector<std::uint8_t>, 2> packs{gemma_pack(inputs, outputs, k, 0U), gemma_pack(inputs, outputs, k, 1U)};
                std::array<std::unique_ptr<Buffer>, 2> dev;
                std::array<g4::LinearView, 2> view{};
                for (unsigned e = 0; e < 2U; ++e) {
                    dev[e] = std::make_unique<Buffer>(packs[e].size()); dev[e]->put(packs[e]);
                    view[e] = {dev[e]->as<std::uint8_t>(), outputs, inputs, static_cast<std::size_t>(inputs) * k / 8U, g4::WeightFormat::exl3_mul1};
                    g4::LinearView host = view[e]; host.data = packs[e].data();
                    expect(ex::geometry(host) && packs[e].size() == g4::weight_storage_bytes(host), "gemma transport geometry K" + std::to_string(k));
                    // Nominal row_bytes is not a stride: the tile of output column block 1 sits after suh | svh and one tile, not at row_bytes.
                    const std::size_t tile1 = 2ULL * (inputs + outputs) + 32ULL * k;
                    // (for inputs == 256 one tile happens to be row_bytes long, so the naive address coincides there)
                    if (inputs != 256U)
                        expect(tile1 != 2ULL * (inputs + outputs) + view[e].row_bytes, "gemma nominal row stride must not address tiles");
                    expect(g4::weight_storage_bytes(host) != static_cast<std::size_t>(outputs) * view[e].row_bytes, "gemma storage is not outputs * row_bytes");
                    g4::LinearView bad = host; bad.row_bytes += 1U;
                    expect(!ex::geometry(bad), "gemma geometry accepted a nominal row stride mismatch");
                }
                const std::string label = identity + ".k" + std::to_string(k) + ".i" + std::to_string(inputs) + ".o" + std::to_string(outputs);
                for (unsigned e = 0; e < 2U; ++e) {
                    context(label + ".expert" + std::to_string(e), {inputs, outputs, k}, "gemma.exl3.transport@1");
                    emit_vec("source", packs[e]);
                }
                const std::size_t maxrows = 17U;
                const auto x = make_activations("gemma-x-" + label, maxrows * inputs);
                Buffer dx(x.size() * 4U); dx.put(x);
                Buffer scratch(std::max<std::size_t>(ex::device_scratch_bytes, ex::rows_scratch_bytes(view[0], maxrows))), fail(8);
                Buffer coreo(maxrows * outputs * 4U), lin(maxrows * outputs * 4U), prep(maxrows * inputs * 2U + 4096U);
                std::array<std::vector<float>, 2> first;
                for (unsigned e = 0; e < 2U; ++e)
                    for (const unsigned rows : {1U, 2U, 7U, 8U, 9U, 17U}) {
                        context(label + ".expert" + std::to_string(e) + ".rows" + std::to_string(rows), {rows, inputs, outputs, k}, "gemma.exl3.linear@1");
                        emit("activation_codes", x.data(), static_cast<std::size_t>(rows) * inputs * 4U);
                        lin.clear(); fail.clear();
                        if (rows <= 8U) {
                            ex::device_linear(view[e], dx.as<float>(), rows, lin.as<float>(), scratch.p, fail.as<std::uint32_t>());
                        } else if (!ex::device_linear_rows(view[e], dx.as<float>(), rows, lin.as<float>(), scratch.p, scratch.bytes, fail.as<std::uint32_t>())) {
                            for (unsigned r = 0; r < rows; r += 8U)
                                ex::device_linear(view[e], dx.as<float>() + static_cast<std::size_t>(r) * inputs, std::min(8U, rows - r),
                                                  lin.as<float>() + static_cast<std::size_t>(r) * outputs, scratch.p, fail.as<std::uint32_t>());
                        }
                        check(hipDeviceSynchronize());
                        expect(fail.read<std::uint32_t>(1)[0] == 0U, "gemma linear failure bits " + label);
                        const auto out = lin.read<float>(static_cast<std::size_t>(rows) * outputs);
                        emit_vec("output", out);
                        if (rows == 8U) first[e] = out;
                        // chunk identity: the many-row route equals consecutive slices of at most eight rows
                        if (rows > 8U) {
                            std::vector<float> chunks(out.size());
                            Buffer chunk_out(out.size() * 4U);
                            for (unsigned r = 0; r < rows; r += 8U)
                                ex::device_linear(view[e], dx.as<float>() + static_cast<std::size_t>(r) * inputs, std::min(8U, rows - r),
                                                  chunk_out.as<float>() + static_cast<std::size_t>(r) * outputs, scratch.p, fail.as<std::uint32_t>());
                            check(hipDeviceSynchronize());
                            expect(same(chunk_out.read<float>(out.size()), out), "gemma many-row vs eight-row slices " + label + " rows " + std::to_string(rows));
                        }
                        if (rows <= 8U) {
                            // shared OFF and the staged decomposition input -> core -> output must equal device_linear
                            Buffer alt(out.size() * 4U);
                            ex::device_linear(view[e], dx.as<float>(), rows, alt.as<float>(), scratch.p, fail.as<std::uint32_t>(), false);
                            check(hipDeviceSynchronize());
                            expect(same(alt.read<float>(out.size()), out), "gemma shared OFF " + label);
                            ex::device_input(view[e], dx.as<float>(), rows, prep.p, fail.as<std::uint32_t>());
                            check(hipDeviceSynchronize());
                            emit("prepared", prep.read<std::uint16_t>(static_cast<std::size_t>(rows) * inputs).data(), static_cast<std::size_t>(rows) * inputs * 2U);
                            std::vector<float> by_route;
                            for (const auto route : {ex::CoreRoute::automatic, ex::CoreRoute::legacy, ex::CoreRoute::lane8, ex::CoreRoute::staged8}) {
                                coreo.clear();
                                ex::device_core(view[e], prep.p, rows, coreo.as<float>(), fail.as<std::uint32_t>(), route);
                                check(hipDeviceSynchronize());
                                const auto raw = coreo.read<float>(static_cast<std::size_t>(rows) * outputs);
                                if (route == ex::CoreRoute::automatic) { emit_vec("partials", raw); by_route = raw; }
                                else expect(same(raw, by_route), "gemma core routes " + label);
                            }
                            ex::device_output(view[e], coreo.as<float>(), rows, fail.as<std::uint32_t>());
                            check(hipDeviceSynchronize());
                            // v2/v3 device_linear splits through scratch partials, which the core-only entry does not take
                            if (ex::core_version() == 1U) expect(same(coreo.read<float>(out.size()), out), "gemma input/core/output decomposition " + label);
                        }
                        ++g_cases;
                    }
                expect(!same(first[0], first[1]), "gemma two experts gave identical outputs " + label);
                // WMMA prefill contract, captured by output (rows 8 and 17).
                for (const unsigned rows : {8U, 17U}) {
                    g_arithmetic = std::string(ex::prefill_identity) + "@1";
                    context(label + ".prefill.rows" + std::to_string(rows), {rows, inputs, outputs, k}, "gemma.exl3.prefill@1");
                    Buffer maximum(8);
                    lin.clear(); fail.clear(); maximum.clear();
                    ex::device_prefill(view[0], dx.as<float>(), rows, lin.as<float>(), scratch.p, fail.as<std::uint32_t>(), maximum.as<std::uint32_t>());
                    check(hipDeviceSynchronize());
                    expect(fail.read<std::uint32_t>(1)[0] == 0U, "gemma prefill failure bits " + label);
                    emit_vec("output", lin.read<float>(static_cast<std::size_t>(rows) * outputs));
                    g_arithmetic = identity + "@1";
                    ++g_cases;
                }
            }
}
#endif

// One registry row per EXL3 numeric version, found by its exact arithmetic; Gemma is never bound to the generic mul1 arithmetic.
void registry_cases() {
    namespace wh = strix::weight::hip;
    expect(wh::find_source({"exl3.mul1.tiles16", 1}, "trellis-tile", "trellis-mul1-legacy") != nullptr, "mul1 source unregistered");
    for (const auto name : {"g4-exl3-v1", "g4-exl3-v2", "g4-exl3-v3", "g4-exl3-wmma-pf-v1"}) {
        const auto* r = wh::find_source({"exl3.gemma.transport", 1}, "trellis-tile", name);
        expect(r && r->input_transform && r->output_transform && r->alignment == 2U, std::string("Gemma source unregistered ") + name);
    }
    expect(!wh::find_source({"exl3.gemma.transport", 1}, "trellis-tile", "trellis-mul1-legacy"), "Gemma bound to the generic mul1 arithmetic");
    expect(!wh::find_source({"exl3.gemma.transport", 1}, "trellis-tile"), "ambiguous Gemma lookup accepted");
    ++g_cases;
}
}  // namespace

int main(int argc, char** argv) try {
    if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "--require-engines"))
        throw std::runtime_error("usage: [--require-engines]");
    g_coverage.require_engines = argc == 2;
    g_coverage.gemma_only = strix::envcfg::integer("STRIX_WFMT_TRELLIS_GEMMA_ONLY", 0) != 0;
    std::cout << "GLM finish: " << (g_coverage.glm_available ? "available" : "SKIP (not built)")
              << "; Gemma incumbent: " << (g_coverage.gemma_available ? "available" : "SKIP (not built)") << '\n';
    if (const char* error = g_coverage.error()) throw std::runtime_error(error);
    if (g_coverage.gemma_only && g_coverage.glm_available) std::cout << "GLM finish: SKIP (Gemma-only requested)\n";
    registry_cases();
    g_report = strix::envcfg::integer("STRIX_WFMT_TRELLIS_REPORT", 0) != 0;
    // STRIX_WFMT_TRELLIS_GEMMA_ONLY=1: only the Gemma cases (one process per STRIX_G4_EXL3_IDENTITY).
    if (!g_coverage.gemma_only) {
        input_cases<3>();
        input_cases<4>();
        core_cases<3>();
        core_cases<4>();
    }
#if defined(STRIX_WFMT_GEMMA) && STRIX_WFMT_GEMMA
    if (g_coverage.run_gemma()) {
        const unsigned before = g_cases;
        gemma_cases();
        g_gemma_cases = g_cases - before;
    }
#endif
    std::cout << "Engine execution counts: GLM finish " << g_glm_checks << ", Gemma incumbent " << g_gemma_cases << '\n';
    std::cout << "Gemma transport-view source: SKIP (fixture exercises incumbent engine only)\n";
    if (g_coverage.require_engines && (!g_glm_checks || !g_gemma_cases))
        throw std::runtime_error("required engine checks did not execute");
    std::cout << "HIP EXL3/trellis: " << g_cases << " cases, " << g_failures << " mismatches\n";
    return g_failures ? 1 : 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
