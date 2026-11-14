// qwen-gdn-fp16-chunk revisions 2/3: the chunked route from raw projection rows (conv + L2 + chunk prep + staged scan +
// gated-norm epilogue) against an FP64 token recurrence (raw core, normed output, final state, conv history), run-to-run
// determinism, aligned split == monolithic, every tail length, resets / tiny decays, both state layouts, and --time
// (T 8192, 48 value heads, 16 key heads: prep / scan / total).
#include <hip/hip_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "row_fixture_stride.hpp"

// Production definitions the route uses (copied from hip_gdn_launcher.hip).
enum class GdnRecurrentStorage : std::uint32_t { key_value = 0, value_key = 1 };
struct GdnPrefillRows { float* qkv; float* z; float* beta; float* gate; float* core; std::uint64_t stride; };
template <GdnRecurrentStorage Storage>
__device__ __forceinline__ std::uint64_t gdn_window_state_index(std::uint32_t value_head, std::uint32_t lane,
    std::uint32_t value_index, std::uint32_t head_dim) {
    return (static_cast<std::uint64_t>(value_head) * head_dim +
        (Storage == GdnRecurrentStorage::value_key ? value_index : lane)) * head_dim +
        (Storage == GdnRecurrentStorage::value_key ? lane : value_index);
}
enum class SmallWeightKind : std::uint32_t { f32 = 0, f16 = 1, bf16 = 2 };
__device__ float bf16_to_float(std::uint16_t bits) { return __uint_as_float(static_cast<std::uint32_t>(bits) << 16U); }
__device__ std::uint16_t float_to_bf16(float value) {
    const auto bits = __float_as_uint(value);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}
__device__ float round_bf16(float value) { return bf16_to_float(float_to_bf16(value)); }
__device__ float load_small(const void* values, SmallWeightKind kind, std::uint64_t index) {
    switch (kind) {
    case SmallWeightKind::f32: return static_cast<const float*>(values)[index];
    case SmallWeightKind::f16: return static_cast<float>(static_cast<const _Float16*>(values)[index]);
    default: return bf16_to_float(static_cast<const std::uint16_t*>(values)[index]);
    }
}
constexpr std::uint32_t kGdnWindowKernel = 4;
__device__ __forceinline__ float gdn_window_add(float lower, float upper) {
    float result;
    asm("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(lower), "v"(upper));
    return result;
}
template <unsigned Distance>
__device__ __forceinline__ float gdn_window_down(float value) {
    if constexpr (Distance == 16U) {
        const auto bits = __float_as_uint(value);
        return __uint_as_float(__builtin_amdgcn_permlanex16(bits, bits, 0x76543210U, 0xfedcba98U, false, false));
    } else {
        const auto bits = __float_as_int(value);
        return __int_as_float(__builtin_amdgcn_update_dpp(bits, bits, 0x100 + static_cast<int>(Distance), 0xf, 0xf, false));
    }
}
__device__ __forceinline__ float gdn_window_key_sum(const float (&p)[4]) {
    float sum = gdn_window_add(gdn_window_add(p[0], p[2]), gdn_window_add(p[1], p[3]));
    sum = gdn_window_add(sum, gdn_window_down<16U>(sum));
    sum = gdn_window_add(sum, gdn_window_down<8U>(sum));
    sum = gdn_window_add(sum, gdn_window_down<4U>(sum));
    sum = gdn_window_add(sum, gdn_window_down<2U>(sum));
    sum = gdn_window_add(sum, gdn_window_down<1U>(sum));
    return __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(sum)));
}

#include "../src/core/wave_uniform.hpp"
#include "../src/hip_gdn_fp16_chunk.inc"

// Gated RMSNorm row epilogue with the production expression, FP32 output to rows.core.
template <bool Wave> struct TestNormT {
    static constexpr bool kRows = true, kWaveRows = Wave;
    GdnPrefillRows rows;
    const float* norm;
    float epsilon;
    __device__ void weights(unsigned lane, float (&w)[4]) const {
        for (unsigned p = 0; p < 4U; ++p) w[p] = norm[p * 32U + lane];
    }
    __device__ void operator()(std::uint64_t row, unsigned head, const float (&value)[4], const float (&z)[4],
        const float (&w)[4]) const {
        const unsigned lane = threadIdx.x % 32U;
        float square[4];
        for (unsigned p = 0; p < 4U; ++p) square[p] = value[p] * value[p];
        const float inv = rsqrtf(gdn_window_key_sum(square) / 128.0F + epsilon);
        for (unsigned p = 0; p < 4U; ++p) {
            const float result = value[p] * inv * w[p] * (1.0F / (1.0F + expf(-z[p])));
            if constexpr (Wave) strix::core::wave_row(rows.core, row * rows.stride + head * 128U + p * 32U)[lane] = result;
            else rows.core[row * rows.stride + head * 128U + p * 32U + lane] = result;
        }
    }
};
// The production row policy (scalar wave rows); TestNormT<false> keeps per-lane rows for the --time comparison.
using TestNorm = TestNormT<true>;

namespace {
constexpr std::uint32_t kD = 128U;
constexpr float kEps = 1e-6F, kNormEps = 1e-6F;

void ck(hipError_t error, const char* what) {
    if (error != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(error));
}
float bf16(float x) {
    auto bits = std::bit_cast<std::uint32_t>(x);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return std::bit_cast<float>(bits & 0xffff0000U);
}
std::uint16_t bf16_bits(float x) { return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(bf16(x)) >> 16U); }
float from_bits(std::uint16_t b) { return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16U); }

// Row layout: q (kh x 128) | k | v (vh x 128) | z (vh x 128) | beta | gate | core (vh x 128); stride % 4 == 0.
struct Layout {
    std::uint32_t kh = 0, vh = 0;
    std::uint64_t kw = 0, vw = 0, conv = 0, z = 0, beta = 0, gate = 0, core = 0, stride = 0;
};
Layout layout_of(std::uint32_t kh, std::uint32_t vh) {
    Layout g;
    g.kh = kh; g.vh = vh; g.kw = std::uint64_t{kh} * kD; g.vw = std::uint64_t{vh} * kD;
    g.conv = 2U * g.kw + g.vw; g.z = g.conv; g.beta = g.z + g.vw; g.gate = g.beta + vh; g.core = (g.gate + vh + 3U) / 4U * 4U;
    g.stride = g.core + g.vw;
    const auto bytes = fixture_row_stride_bytes(g.stride * 4U);
    if (bytes % 16U) throw std::invalid_argument("fixture row stride must keep 16-byte rows");
    g.stride = bytes / 4U;
    return g;
}
std::size_t state_index(bool vk, std::uint32_t h, std::uint32_t k, std::uint32_t v) {
    return vk ? (std::size_t{h} * kD + v) * kD + k : (std::size_t{h} * kD + k) * kD + v;
}

struct Problem {
    Layout g;
    std::uint32_t count = 0;
    bool vk = false;
    std::vector<float> rows, state, weights, norm;
    std::vector<std::uint16_t> history;
};
// Raw projection values ~ N(0, s); beta = BF16 sigmoid; gate = log decay in [log 0.2, 0) (resets: -inf, tiny: -30).
Problem make_problem(std::uint32_t count, std::uint32_t kh, std::uint32_t vh, bool vk, std::uint32_t seed,
    const std::vector<std::uint32_t>& resets = {}, const std::vector<std::uint32_t>& tiny = {}) {
    Problem p;
    p.g = layout_of(kh, vh);
    p.count = count;
    p.vk = vk;
    p.rows.assign(std::size_t{std::max(count, 1U)} * p.g.stride, 0.0F);
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_real_distribution<float> unit(0.0F, 1.0F), uni(-1.0F, 1.0F);
    for (std::uint32_t r = 0; r < count; ++r) {
        float* row = &p.rows[std::size_t{r} * p.g.stride];
        // The engine projection stores BF16-rounded values (the BF16-row route relies on it).
        for (std::uint64_t i = 0; i < p.g.conv; ++i) row[i] = bf16(0.7F * normal(rng));
        for (std::uint64_t i = 0; i < p.g.vw; ++i) row[p.g.z + i] = bf16(1.5F * normal(rng));
        for (std::uint32_t h = 0; h < vh; ++h) {
            row[p.g.beta + h] = bf16(1.0F / (1.0F + std::exp(-2.0F * normal(rng))));
            row[p.g.gate + h] = std::log(0.2F + 0.8F * unit(rng));
        }
        for (std::uint64_t i = 0; i < p.g.vw; ++i) row[p.g.core + i] = -1234.5F;
    }
    for (const auto r : resets)
        if (r < count) for (std::uint32_t h = 0; h < vh; ++h) p.rows[std::size_t{r} * p.g.stride + p.g.gate + h] = -INFINITY;
    for (const auto r : tiny)
        if (r < count) for (std::uint32_t h = 0; h < vh; ++h) p.rows[std::size_t{r} * p.g.stride + p.g.gate + h] = -30.0F;
    p.weights.resize(p.g.conv * kGdnWindowKernel);
    for (auto& w : p.weights) w = 0.5F * uni(rng);
    p.history.resize(p.g.conv * kGdnWindowKernel);
    for (auto& h : p.history) h = bf16_bits(0.7F * normal(rng));
    p.norm.resize(kD);
    for (auto& w : p.norm) w = 1.0F + 0.2F * uni(rng);
    p.state.resize(std::size_t{vh} * kD * kD);
    for (auto& s : p.state) s = 0.05F * uni(rng);
    return p;
}

// FP64 reference: production conv (FP32 expression as on the device, BF16 inputs / output), FP64 L2, token recurrence
// S_t = d S + beta (v - d S k) k^T, o = S_t q, gated norm. Returns raw core, normed rows, final state, new history.
struct Expect {
    std::vector<double> core, norm, state;
    std::vector<std::uint16_t> history;
};
Expect expect(const Problem& p) {
    const auto& g = p.g;
    const std::uint32_t T = p.count;
    std::vector<float> cv(std::size_t{T} * g.conv);
    for (std::uint64_t c = 0; c < g.conv; ++c)
        for (std::uint32_t t = 0; t < T; ++t) {
            float sum = 0.0F;
            for (std::uint32_t j = 0; j < kGdnWindowKernel; ++j) {
                const auto u = static_cast<std::int64_t>(t) - 3 + j;
                const float x = u < 0 ? from_bits(p.history[c * 4U + 1U + static_cast<unsigned>(u + 3)])
                                      : bf16(p.rows[static_cast<std::size_t>(u) * g.stride + c]);
                sum += x * p.weights[c * 4U + j];
            }
            cv[std::size_t{t} * g.conv + c] = bf16(sum / (1.0F + std::exp(-sum)));
        }
    Expect e;
    e.core.assign(std::size_t{T} * g.vw, 0.0);
    e.norm.assign(std::size_t{T} * g.vw, 0.0);
    e.state.resize(p.state.size());
    for (std::uint32_t h = 0; h < g.vh; ++h) {
        const std::uint32_t kh = h % g.kh;
        std::vector<double> S(kD * kD);  // S[v][k]
        for (std::uint32_t k = 0; k < kD; ++k)
            for (std::uint32_t v = 0; v < kD; ++v) S[v * kD + k] = p.state[state_index(p.vk, h, k, v)];
        for (std::uint32_t t = 0; t < T; ++t) {
            const float* c = &cv[std::size_t{t} * g.conv];
            const float* row = &p.rows[std::size_t{t} * g.stride];
            double q[kD], k[kD], v[kD], nq = 0.0, nk = 0.0;
            for (std::uint32_t i = 0; i < kD; ++i) {
                q[i] = c[kh * kD + i]; k[i] = c[g.kw + kh * kD + i]; v[i] = c[2U * g.kw + h * kD + i];
                nq += q[i] * q[i]; nk += k[i] * k[i];
            }
            const double sq = 1.0 / std::sqrt(nq + kEps) / std::sqrt(double(kD)), sk = 1.0 / std::sqrt(nk + kEps);
            for (std::uint32_t i = 0; i < kD; ++i) { q[i] *= sq; k[i] *= sk; }
            const double d = std::exp(double(row[g.gate + h])), beta = row[g.beta + h];
            for (auto& s : S) s *= d;
            for (std::uint32_t vi = 0; vi < kD; ++vi) {
                double sk2 = 0.0;
                for (std::uint32_t ki = 0; ki < kD; ++ki) sk2 += S[vi * kD + ki] * k[ki];
                const double delta = beta * (v[vi] - sk2);
                for (std::uint32_t ki = 0; ki < kD; ++ki) S[vi * kD + ki] += delta * k[ki];
            }
            double o[kD], ss = 0.0;
            for (std::uint32_t vi = 0; vi < kD; ++vi) {
                double acc = 0.0;
                for (std::uint32_t ki = 0; ki < kD; ++ki) acc += S[vi * kD + ki] * q[ki];
                o[vi] = acc;
                ss += acc * acc;
            }
            const double inv = 1.0 / std::sqrt(ss / kD + kNormEps);
            for (std::uint32_t vi = 0; vi < kD; ++vi) {
                e.core[std::size_t{t} * g.vw + h * kD + vi] = o[vi];
                const double z = row[g.z + h * kD + vi];
                e.norm[std::size_t{t} * g.vw + h * kD + vi] = o[vi] * inv * p.norm[vi] / (1.0 + std::exp(-z));
            }
        }
        for (std::uint32_t k = 0; k < kD; ++k)
            for (std::uint32_t v = 0; v < kD; ++v) e.state[state_index(p.vk, h, k, v)] = S[v * kD + k];
    }
    e.history = p.history;
    for (std::uint64_t c = 0; c < g.conv; ++c) {
        const std::uint16_t old[3] = {p.history[c * 4U + 1U], p.history[c * 4U + 2U], p.history[c * 4U + 3U]};
        for (unsigned i = 0; i < 4U; ++i) {
            const auto r = static_cast<std::int64_t>(T) - 4 + i;
            e.history[c * 4U + i] = r >= 0 ? bf16_bits(p.rows[static_cast<std::size_t>(r) * g.stride + c]) : old[r + 3];
        }
    }
    return e;
}

struct Device {
    float* rows = nullptr; float* state = nullptr; float* weights = nullptr; float* norm = nullptr;
    std::uint16_t* history = nullptr;
    GdnPrefillRows view{};
    explicit Device(const Problem& p) {
        ck(hipMalloc(&rows, p.rows.size() * 4U), "rows"); ck(hipMalloc(&state, p.state.size() * 4U), "state");
        ck(hipMalloc(&weights, p.weights.size() * 4U), "weights"); ck(hipMalloc(&norm, p.norm.size() * 4U), "norm");
        ck(hipMalloc(&history, p.history.size() * 2U), "history");
        upload(p);
        view = GdnPrefillRows{rows, rows + p.g.z, rows + p.g.beta, rows + p.g.gate, rows + p.g.core, p.g.stride};
    }
    void upload(const Problem& p) {
        ck(hipMemcpy(rows, p.rows.data(), p.rows.size() * 4U, hipMemcpyHostToDevice), "up rows");
        ck(hipMemcpy(state, p.state.data(), p.state.size() * 4U, hipMemcpyHostToDevice), "up state");
        ck(hipMemcpy(weights, p.weights.data(), p.weights.size() * 4U, hipMemcpyHostToDevice), "up weights");
        ck(hipMemcpy(norm, p.norm.data(), p.norm.size() * 4U, hipMemcpyHostToDevice), "up norm");
        ck(hipMemcpy(history, p.history.data(), p.history.size() * 2U, hipMemcpyHostToDevice), "up history");
    }
    ~Device() { (void)hipFree(x16); (void)hipFree(rows); (void)hipFree(state); (void)hipFree(weights); (void)hipFree(norm); (void)hipFree(history); }
    // One call over rows [first, first + count) (the view advanced by `first` rows).
    std::uint16_t* x16 = nullptr;
    // BF16 rows q | k | v | z (the engine's GdnInputTiled16Site layout) for the BF16-row route.
    void make_x16(const Problem& p) {
        const std::uint64_t w = p.g.conv + p.g.vw;
        std::vector<std::uint16_t> h(std::size_t{std::max(p.count, 1U)} * w);
        for (std::uint32_t r = 0; r < p.count; ++r)
            for (std::uint64_t i = 0; i < w; ++i) h[r * w + i] = bf16_bits(p.rows[std::size_t{r} * p.g.stride + i]);
        ck(hipMalloc(&x16, h.size() * 2U), "x16");
        ck(hipMemcpy(x16, h.data(), h.size() * 2U, hipMemcpyHostToDevice), "up x16");
    }
    void run(const Problem& p, std::uint32_t first, std::uint32_t count, bool normed) {
        GdnPrefillRows v = view;
        const std::uint64_t xw = p.g.conv + p.g.vw;
        const std::uint16_t* x = x16 ? x16 + std::uint64_t{first} * xw : nullptr;
        const std::uint64_t off = std::uint64_t{first} * p.g.stride;
        v.qkv += off; v.z += off; v.beta += off; v.gate += off; v.core += off;
        const auto go = [&]<GdnRecurrentStorage S>() {
            if (normed)
                launch_gdn_fp16_chunk_conv_fast<S>(v, count, state, history, weights, SmallWeightKind::f32, p.g.kh, p.g.vh, kEps,
                    nullptr, TestNorm{v, norm, kNormEps}, x, xw, p.g.conv);
            else launch_gdn_fp16_chunk_conv_fast<S>(v, count, state, history, weights, SmallWeightKind::f32, p.g.kh, p.g.vh, kEps,
                    nullptr, gdn_fp16_chunk::RawCore{}, x, xw, p.g.conv);
        };
        if (p.vk) go.template operator()<GdnRecurrentStorage::value_key>();
        else go.template operator()<GdnRecurrentStorage::key_value>();
        ck(hipDeviceSynchronize(), "run");
    }
    struct Out { std::vector<float> rows, state; std::vector<std::uint16_t> history; };
    // FNV-1a over every byte a case reads back (rows, state, history), in case order: one line to compare two builds.
    static inline std::uint64_t digest = 0xcbf29ce484222325ULL;
    static void fold(const void* data, std::size_t bytes) {
        const auto* b = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < bytes; ++i) digest = (digest ^ b[i]) * 0x100000001b3ULL;
    }
    Out read(const Problem& p) const {
        Out o{std::vector<float>(p.rows.size()), std::vector<float>(p.state.size()), std::vector<std::uint16_t>(p.history.size())};
        ck(hipMemcpy(o.rows.data(), rows, o.rows.size() * 4U, hipMemcpyDeviceToHost), "down rows");
        ck(hipMemcpy(o.state.data(), state, o.state.size() * 4U, hipMemcpyDeviceToHost), "down state");
        ck(hipMemcpy(o.history.data(), history, o.history.size() * 2U, hipMemcpyDeviceToHost), "down history");
        fold(o.rows.data(), o.rows.size() * 4U); fold(o.state.data(), o.state.size() * 4U);
        fold(o.history.data(), o.history.size() * 2U);
        return o;
    }
};

int failures = 0;
void expect_true(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::printf("FAIL %s\n", what.c_str()); }
}
struct Err { double maxrel = 0.0, rel_l2 = 0.0; bool finite = true; };
Err compare(const std::vector<double>& ref, const float* got, std::size_t n, std::size_t pitch_ref, std::size_t pitch_got,
    std::size_t rows, std::size_t width, std::size_t offset_ref = 0U) {
    double maxref = 0.0, maxerr = 0.0, e2 = 0.0, r2 = 0.0;
    bool finite = true;
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t i = 0; i < width; ++i) {
            const double a = ref[offset_ref + r * pitch_ref + i], b = got[r * pitch_got + i];
            finite &= std::isfinite(b);
            maxref = std::max(maxref, std::fabs(a));
            maxerr = std::max(maxerr, std::fabs(a - b));
            e2 += (a - b) * (a - b); r2 += a * a;
        }
    (void)n;
    return Err{maxerr / std::max(1.0, maxref), std::sqrt(e2 / std::max(r2, 1e-300)), finite};
}

// Screen: raw core / normed rows / state relL2 and max-abs relative error against FP64.
constexpr double kRelL2 = 3e-3, kMaxRel = 2e-2;
Err worst_out, worst_state, worst_head;
void collect(Err& worst, const Err& e) {
    worst.rel_l2 = std::max(worst.rel_l2, e.rel_l2);
    worst.maxrel = std::max(worst.maxrel, e.maxrel);
    worst.finite &= e.finite;
}
void case_screen(const std::string& name, const Problem& p) {
    const Expect e = expect(p);
    for (const bool normed : {false, true}) {
        Device d(p);
        d.run(p, 0U, p.count, normed);
        const auto o = d.read(p);
        const auto& ref = normed ? e.norm : e.core;
        const Err out = compare(ref, o.rows.data() + p.g.core, 0, p.g.vw, p.g.stride, p.count, p.g.vw);
        const Err st = compare(e.state, o.state.data(), 0, 0, 0, 1, o.state.size());
        collect(worst_out, out); collect(worst_state, st);
        Err head_error;
        for (std::uint32_t h = 0; h < p.g.vh; ++h) {
            const std::size_t off = std::size_t{h} * kD * kD;
            const Err he = compare(e.state, o.state.data() + off, 0, 0, 0, 1, kD * kD, off);
            collect(head_error, he);
            expect_true(he.finite && he.rel_l2 <= kRelL2 && he.maxrel <= kMaxRel, name + " per-head state screen");
        }
        collect(worst_head, head_error);
        std::printf("%-34s %s out relL2 %.2e max %.2e | state relL2 %.2e max %.2e\n", name.c_str(), normed ? "norm" : "core",
            out.rel_l2, out.maxrel, st.rel_l2, st.maxrel);
        std::printf("  worst head state relL2 %.2e max %.2e\n", head_error.rel_l2, head_error.maxrel);
        expect_true(out.finite && st.finite, name + " finite");
        expect_true(out.rel_l2 <= kRelL2 && out.maxrel <= kMaxRel, name + " output screen");
        expect_true(st.rel_l2 <= kRelL2 && st.maxrel <= kMaxRel, name + " state screen");
        expect_true(o.history == e.history, name + " conv history");
        // rows past count keep their sentinel / inputs untouched
        bool inputs = true;
        for (std::uint32_t r = 0; r < p.count; ++r)
            for (std::uint64_t i = 0; i < p.g.core; ++i)
                inputs &= std::bit_cast<std::uint32_t>(o.rows[r * p.g.stride + i]) == std::bit_cast<std::uint32_t>(p.rows[r * p.g.stride + i]);
        expect_true(inputs, name + " inputs untouched");
    }
}
// Determinism and aligned split invariance (bitwise).
void case_bitwise(const std::string& name, const Problem& p, std::uint32_t split) {
    Device a(p), b(p), c(p);
    a.run(p, 0U, p.count, true);
    b.run(p, 0U, p.count, true);
    c.run(p, 0U, split, true);
    c.run(p, split, p.count - split, true);
    const auto oa = a.read(p), ob = b.read(p), oc = c.read(p);
    const auto same = [](const auto& x, const auto& y) { return std::memcmp(x.data(), y.data(), x.size() * sizeof(x[0])) == 0; };
    expect_true(same(oa.rows, ob.rows) && same(oa.state, ob.state) && same(oa.history, ob.history), name + " run-to-run");
    expect_true(same(oa.rows, oc.rows) && same(oa.state, oc.state) && same(oa.history, oc.history), name + " aligned split");
    std::printf("%-34s determinism + split at %u checked\n", name.c_str(), split);
}

// The BF16-row route equals the FP32-row route bit for bit (same values, BF16 loads).
void case_b16(const std::string& name, const Problem& p, std::uint32_t split) {
    Device a(p), b(p);
    b.make_x16(p);
    a.run(p, 0U, p.count, true);
    b.run(p, 0U, split, true);
    b.run(p, split, p.count - split, true);
    const auto oa = a.read(p), ob = b.read(p);
    const auto same = [](const auto& x, const auto& y) { return std::memcmp(x.data(), y.data(), x.size() * sizeof(x[0])) == 0; };
    expect_true(same(oa.rows, ob.rows) && same(oa.state, ob.state) && same(oa.history, ob.history), name + " BF16 rows == FP32 rows");
    std::printf("%-34s BF16-row route == FP32-row route checked\n", name.c_str());
}

// Geometry changes must preserve convolution operands and metadata exactly; only T/P may change in rev 3.
void case_prep(const std::string& name, const Problem& p, bool b16, bool fallback = false) {
    Device d(p);
    if (b16) d.make_x16(p);
    using namespace gdn_fp16_chunk;
    const auto bytes = scratch_bytes(p.count, p.g.kh, p.g.vh);
    void* old_base = nullptr;
    void* new_base = nullptr;
    ck(hipMalloc(&old_base, bytes), "old prep"); ck(hipMalloc(&new_base, bytes), "new prep");
    const auto old = scratch_view(old_base, p.count, p.g.kh, p.g.vh);
    const auto next = scratch_view(new_base, p.count, p.g.kh, p.g.vh);
    const Conv conv{d.history, d.weights, SmallWeightKind::f32, d.x16, p.g.conv + p.g.vw, p.g.conv};
    const unsigned n = (p.count + kC - 1U) / kC;
    const auto reference = [&]<bool B16>() {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(prep_kernel<B16>), dim3(p.g.kh, n), dim3(256), 0, nullptr,
            d.view, p.count, conv, p.g.kh, p.g.vh, kEps, old);
    };
    if (b16) reference.template operator()<true>(); else reference.template operator()<false>();
    enqueue_gdn_fp16_chunk_prep(next, d.view, p.count, conv, p.g.kh, p.g.vh, kEps, nullptr);
    ck(hipDeviceSynchronize(), "prep comparison");
    const auto same = [&](const void* a, const void* b, std::size_t size) {
        std::vector<unsigned char> x(size), y(size);
        ck(hipMemcpy(x.data(), a, size, hipMemcpyDeviceToHost), "old prep down");
        ck(hipMemcpy(y.data(), b, size, hipMemcpyDeviceToHost), "new prep down");
        return x == y;
    };
    const std::size_t rows_p = std::size_t{n} * kC;
    expect_true(same(old.q16, next.q16, 2U * p.g.kh * rows_p * gdn_fp16_chunk::kD), name + " Q16 exact");
    expect_true(same(old.k16, next.k16, 2U * p.g.kh * rows_p * gdn_fp16_chunk::kD), name + " K16 exact");
    expect_true(same(old.v16, next.v16, 2U * p.g.vh * rows_p * gdn_fp16_chunk::kD), name + " v16 exact");
    expect_true(same(old.ef, next.ef, 8U * p.g.vh * rows_p), name + " decay metadata exact");
    if (fallback || revision() == 2U)
        expect_true(same(old.tp16, next.tp16, 4U * p.g.vh * rows_p * kC), name + " fallback/rev2 T/P exact");
    ck(hipFree(old_base), "old prep free"); ck(hipFree(new_base), "new prep free");
    std::printf("%-34s conv operands + metadata exact\n", name.c_str());
}

int time_main() {
    const Problem p = make_problem(8192U, 16U, 48U, true, 7U);
    Device d(p);
    const auto conv = gdn_fp16_chunk::Conv{d.history, d.weights, SmallWeightKind::f32, nullptr, 0U, 0U};
    void* base = gdn_fp16_chunk::persistent_scratch(nullptr, gdn_fp16_chunk::scratch_bytes(p.count, 16U, 48U));
    const auto scratch = gdn_fp16_chunk::scratch_view(base, p.count, 16U, 48U);
    const TestNorm norm{d.view, d.norm, kNormEps};
    hipEvent_t e0, e1;
    ck(hipEventCreate(&e0), "event"); ck(hipEventCreate(&e1), "event");
    const auto median = [&](auto&& body) {
        std::vector<float> ms;
        for (int i = 0; i < 23; ++i) {
            ck(hipEventRecord(e0, nullptr), "record");
            body();
            ck(hipEventRecord(e1, nullptr), "record");
            ck(hipEventSynchronize(e1), "sync");
            float t = 0.0F;
            ck(hipEventElapsedTime(&t, e0, e1), "elapsed");
            if (i >= 3) ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        return ms[ms.size() / 2U];
    };
    const float prep = median([&] { enqueue_gdn_fp16_chunk_prep(scratch, d.view, p.count, conv, 16U, 48U, kEps, nullptr); });
    const float scan = median([&] {
        enqueue_gdn_fp16_chunk_scan<GdnRecurrentStorage::value_key>(scratch, d.view, p.count, d.state, conv, 16U, 48U, nullptr, norm);
    });
    const float raw = median([&] {
        enqueue_gdn_fp16_chunk_scan<GdnRecurrentStorage::value_key>(scratch, d.view, p.count, d.state, conv, 16U, 48U, nullptr);
    });
    const float total = median([&] {
        launch_gdn_fp16_chunk_conv_fast<GdnRecurrentStorage::value_key>(d.view, p.count, d.state, d.history, d.weights,
            SmallWeightKind::f32, 16U, 48U, kEps, nullptr, norm);
    });
    d.make_x16(p);
    auto conv16 = conv;
    conv16.x16 = d.x16; conv16.x16_stride = p.g.conv + p.g.vw; conv16.z_offset = p.g.conv;
    const float prep16 = median([&] { enqueue_gdn_fp16_chunk_prep(scratch, d.view, p.count, conv16, 16U, 48U, kEps, nullptr); });
    const float scan16 = median([&] {
        enqueue_gdn_fp16_chunk_scan<GdnRecurrentStorage::value_key>(scratch, d.view, p.count, d.state, conv16, 16U, 48U, nullptr, norm);
    });
    const TestNormT<false> lane_norm{d.view, d.norm, kNormEps};
    const float lane16 = median([&] {
        enqueue_gdn_fp16_chunk_scan<GdnRecurrentStorage::value_key>(scratch, d.view, p.count, d.state, conv16, 16U, 48U, nullptr, lane_norm);
    });
    const float lane = median([&] {
        enqueue_gdn_fp16_chunk_scan<GdnRecurrentStorage::value_key>(scratch, d.view, p.count, d.state, conv, 16U, 48U, nullptr, lane_norm);
    });
    std::printf("time per-lane rows policy: scan+norm %.3f ms | BF16 rows scan+norm %.3f\n", lane, lane16);
    std::printf("time T8192 48/16 (median of 20): prep %.3f ms | scan+norm %.3f ms | scan raw %.3f ms | total %.3f ms | "
        "BF16 rows: prep %.3f scan+norm %.3f\n", prep, scan, raw, total, prep16, scan16);
    // Payload hash (every row's columns up to the tight width, and the state): equal for every stride.
    const std::uint64_t tight = p.g.core + p.g.vw;
    std::vector<float> out(p.rows.size()), st(p.state.size());
    ck(hipMemcpy(out.data(), d.rows, out.size() * 4U, hipMemcpyDeviceToHost), "down rows");
    ck(hipMemcpy(st.data(), d.state, st.size() * 4U, hipMemcpyDeviceToHost), "down state");
    std::uint64_t h = 1469598103934665603ULL;
    const auto mix = [&](const float* v, std::uint64_t n) {
        for (std::uint64_t i = 0; i < n; ++i) { h ^= std::bit_cast<std::uint32_t>(v[i]); h *= 1099511628211ULL; }
    };
    for (std::uint32_t r = 0; r < p.count; ++r) mix(&out[std::size_t{r} * p.g.stride], tight);
    mix(st.data(), st.size());
    std::printf("rows stride_bytes=%llu tight_bytes=%llu hash=%016llx\n", static_cast<unsigned long long>(p.g.stride * 4U),
        static_cast<unsigned long long>(tight * 4U), static_cast<unsigned long long>(h));
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        std::printf("no HIP device: skipped\n");
        return 77;
    }
    try {
        std::printf("%s (STRIX_QPF2K_GDN_FP16_CHUNK=%s)\n", gdn_fp16_chunk::identity(),
            std::getenv("STRIX_QPF2K_GDN_FP16_CHUNK") ? std::getenv("STRIX_QPF2K_GDN_FP16_CHUNK") : "unset");
        if (argc > 1 && std::string(argv[1]) == "--time") return time_main();
        for (const std::uint32_t t : {1U, 3U, 31U, 32U, 33U, 64U, 100U, 257U})
            case_screen("len " + std::to_string(t) + " 4/12 vk", make_problem(t, 4U, 12U, true, 11U + t));
        case_screen("len 200 4/12 kv", make_problem(200U, 4U, 12U, false, 5U));
        case_screen("len 160 2/2 vk", make_problem(160U, 2U, 2U, true, 6U));
        case_screen("len 130 4/16 vk", make_problem(130U, 4U, 16U, true, 8U));
        case_screen("resets 4/12", make_problem(160U, 4U, 12U, true, 9U, {0U, 31U, 32U, 45U, 100U}, {}));
        case_screen("tiny decays 4/12", make_problem(160U, 4U, 12U, true, 10U, {}, {5U, 40U, 41U, 99U}));
        case_screen("len 1000 16/48 vk", make_problem(1000U, 16U, 48U, true, 12U));
        for (std::uint32_t tail = 1U; tail < 32U; ++tail)
            case_screen("tail " + std::to_string(tail) + " 2/6", make_problem(64U + tail, 2U, 6U, true, 100U + tail));
        case_bitwise("bitwise 4/12 vk", make_problem(200U, 4U, 12U, true, 21U), 96U);
        case_bitwise("bitwise 16/48 kv", make_problem(300U, 16U, 48U, false, 22U), 128U);
        case_b16("b16 16/48 vk", make_problem(333U, 16U, 48U, true, 23U), 160U);
        case_b16("b16 4/12 kv", make_problem(97U, 4U, 12U, false, 24U), 64U);
        for (const bool vk : {false, true}) {
            const std::string layout = vk ? " vk" : " kv";
            for (const float scale : {0x1p-12F, 0x1p12F}) {
                auto p = make_problem(97U, 2U, 6U, vk, 81U);
                for (auto& value : p.state) value *= scale;
                const auto name = "scaled " + std::to_string(scale) + layout;
                case_screen(name, p);
                case_bitwise(name, p, 32U);
                case_b16(name, p, 64U);
            }
            auto slow = make_problem(129U, 2U, 6U, vk, 82U);
            for (std::uint32_t r = 0; r < slow.count; ++r)
                for (std::uint32_t h = 0; h < slow.g.vh; ++h)
                    slow.rows[std::size_t{r} * slow.g.stride + slow.g.gate + h] = -1e-7F;
            case_screen("near-unit decay" + layout, slow);
            for (const auto split : {32U, 64U, 96U, 128U})
                case_bitwise("boundary" + layout, slow, split);
            const auto resets = make_problem(129U, 2U, 6U, vk, 83U, {31U, 32U, 63U, 64U}, {0U, 30U, 65U, 128U});
            case_screen("reset/tiny boundary" + layout, resets);
            case_bitwise("reset/tiny boundary" + layout, resets, 64U);
            case_b16("reset/tiny boundary" + layout, resets, 32U);
            for (std::uint32_t tail = 1U; tail < 32U; ++tail)
                case_screen("small tail " + std::to_string(tail) + layout, make_problem(32U + tail, 1U, 1U, vk, 200U + tail));
        }
        auto fallback = make_problem(64U, 1U, 4U, true, 94U);
        for (std::uint32_t r = 0; r < fallback.count; ++r)
            for (std::uint32_t h = 0; h < fallback.g.vh; ++h)
                fallback.rows[std::size_t{r} * fallback.g.stride + fallback.g.gate + h] = -30.0F;
        case_prep("tiny-decay fallback", fallback, false, true);
        case_prep("tiny-decay fallback b16", fallback, true, true);
        for (const auto vh : {1U, 2U, 3U, 4U})
            for (const auto t : {1U, 15U, 16U, 17U, 31U, 32U, 33U, 65U})
                for (const bool b16 : {false, true})
                    case_prep("prep " + std::to_string(t) + "/" + std::to_string(vh) + (b16 ? " b16" : " f32"),
                        make_problem(t, 1U, vh, true, 90U + t), b16);
    } catch (const std::exception& error) {
        std::printf("error: %s\n", error.what());
        return 1;
    }
    std::printf("screen maxima: output relL2 %.6e max %.6e | state relL2 %.6e max %.6e | head relL2 %.6e max %.6e\n",
        worst_out.rel_l2, worst_out.maxrel, worst_state.rel_l2, worst_state.maxrel, worst_head.rel_l2, worst_head.maxrel);
    std::printf("readback digest %016llx\n", static_cast<unsigned long long>(Device::digest));
    std::printf(failures ? "FAILED %d\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
