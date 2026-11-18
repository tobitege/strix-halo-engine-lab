// Tests of src/cuda/models/qwen/qwen_moe.cu (CUDA routed-MoE prefill, A8 profile) against a literal host transcription of the HIP
// kernels: prefill_prologue/quantize_row (src/hip_moe_q4k_a8_probe.hip:41-73, 159-173), build_prefill_csr
// (src/hip_moe_q4k_a8_known_rows.hip:403-452, the serial builder the parallel one reproduces), project_prefill_wmma with
// gate_up_chains (:791-909), down_prefill_wmma with down_chains (:957-1032), known_moe_prefill_merge/commit_kernel
// (src/hip_moe_experts_launcher.hip:2660-2708). Every buffer the pipeline may write is pre-filled with the same sentinel on
// both sides and compared byte for byte in full (so stray writes are detected too).
//   qwen_moe_tests <tables dir> [bench|fastmma|fastmma-check|fastmma-bench]
// Optional: STRIX_QWEN_GGUF (real expert weights, default path of qwen_gguf_host.hpp), qrtx-build\fx\d64.dump.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "strix/cuda/qwen_moe.hpp"
#include "strix/cuda/qwen_route_bf16.hpp"
#include "strix/cuda/amd_math_tables.hpp"
#include "qwen_fixture.hpp"
#include "qwen_gguf_host.hpp"

#pragma fp_contract(off)

namespace mo = strix::cuda::qwen_moe;
namespace am = strix::cuda::amdmath;
using u8 = std::uint8_t; using u16 = std::uint16_t; using u32 = std::uint32_t; using i8 = std::int8_t; using u64 = std::uint64_t;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

static int g_fail = 0;
static unsigned long long g_checked = 0;
static void expect(bool ok, const std::string& what) {
    if (!ok) { ++g_fail; std::printf("FAIL %s\n", what.c_str()); }
}

static float f_of(u32 b) { float f; std::memcpy(&f, &b, 4); return f; }
static u32 b_of(float f) { u32 b; std::memcpy(&b, &f, 4); return b; }
static float half_to_float(u16 h) {  // exact
    const u32 s = (h >> 15) & 1U, e = (h >> 10) & 31U, m = h & 1023U;
    if (e == 0) { const float v = std::ldexp(static_cast<float>(m), -24); return s ? -v : v; }
    if (e == 31) return f_of((s << 31) | 0x7f800000U | (m << 13));
    return f_of((s << 31) | ((e + 112U) << 23) | (m << 13));
}
static float canon(float x) { return x != x ? f_of(0x7fc00000U) : x; }
static bool finite(float x) { return (b_of(x) & 0x7f800000U) != 0x7f800000U; }

// ---------------------------------------------------------------------------------------------------------------------
// Host gfx1151 expf: the same algorithm as amd_expf (include/strix/cuda/amd_math.cuh, verified exhaustively against gfx1151),
// with the tables read on the host.
struct HostExp {
    std::vector<u32> bits; std::vector<u32> hard_in; std::vector<float> hard_out;
    explicit HostExp(const std::string& dir) {
        bits = read(dir + "/tbl_exp_pm.bin");
        const auto raw = read(dir + "/hard_v_exp_f32.bin");
        const size_t n = raw.size() / 2;
        std::vector<std::pair<u32, u32>> v;
        for (size_t i = 0; i < n; ++i) { const u32 mag = raw[i] & 0x7fffffffU; if (mag >= (102U << 23) && mag < (127U << 23)) v.push_back({raw[i], raw[n + i]}); }
        std::sort(v.begin(), v.end());
        for (auto& p : v) { if (!hard_in.empty() && hard_in.back() == p.first) continue; hard_in.push_back(p.first); hard_out.push_back(f_of(p.second)); }
    }
    static std::vector<u32> read(const std::string& p) {
        FILE* f = std::fopen(p.c_str(), "rb"); if (!f) { std::printf("missing %s\n", p.c_str()); std::exit(2); }
        _fseeki64(f, 0, SEEK_END); const long long n = _ftelli64(f); _fseeki64(f, 0, SEEK_SET);
        std::vector<u32> v(static_cast<size_t>(n / 4)); if (std::fread(v.data(), 4, v.size(), f) != v.size()) std::exit(2); std::fclose(f); return v;
    }
    float v_exp(float x) const {
        const u32 b = b_of(x), mag = b & 0x7fffffffU;
        const double r = std::exp2(static_cast<double>(x));
        const float f = static_cast<float>(r);
        if (mag < (102U << 23)) return f;
        const auto it = std::lower_bound(hard_in.begin(), hard_in.end(), b);
        if (it != hard_in.end() && *it == b) return hard_out[static_cast<size_t>(it - hard_in.begin())];
        const u32 index = (b >> 31) * (25U << 23) + (mag - (102U << 23));
        if ((bits[index >> 5] >> (index & 31U)) & 1U) {
            const bool up = r > static_cast<double>(f);
            const bool away = (b_of(f) >> 31) ? !up : up;
            return f_of(away ? b_of(f) + 1U : b_of(f) - 1U);
        }
        return f;
    }
    float expf_(float x) const {
        const float kLog2e = f_of(0x3fb8aa3bU), kLog2eLo = f_of(0x32a5705fU);
        const float ph = x * kLog2e;
        const float e = std::nearbyint(ph);
        float pl = std::fmaf(x, kLog2e, -ph);
        pl = std::fmaf(x, kLog2eLo, pl);
        const float a = (ph - e) + pl;
        float y;
        if (std::fabs(a) < 1.0F) y = std::scalbn(v_exp(a), static_cast<int>(e));
        else y = a != a ? a : 0.0F;
        if (f_of(0xc2ce8ed0U) > x) y = 0.0F;
        if (f_of(0x42b17218U) < x) y = f_of(0x7f800000U);
        return y;
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// Host state: one flat image per view, rows x stride bytes (the device uses the same strides).
enum View { V_input, V_routes, V_rw, V_codes, V_deltas, V_sums, V_status, V_inter, V_acc, V_dcodes, V_ddeltas, V_dsums, V_raw, V_out, V_count };
static const char* kViewName[V_count] = {"input", "routes", "route_weights", "codes", "deltas", "sums", "status", "intermediates",
                                         "accumulation", "down_codes", "down_deltas", "down_sums", "rawdots", "output"};
static const size_t kViewBytes[V_count] = {mo::hidden * 2, 40, 40, mo::hidden, mo::hidden / 32 * 4, mo::hidden / 32 * 4, 4,
                                           mo::top_k * 2 * mo::ffn * 4, mo::hidden * 4, mo::top_k * mo::ffn, mo::top_k * mo::ffn / 32 * 4,
                                           mo::top_k * mo::ffn / 32 * 4, mo::top_k * mo::hidden * 4, mo::hidden * 2};
static size_t stride_of(int v, bool bf16 = false) {
    const auto width = bf16 && v == V_raw ? kViewBytes[v] / 2U : kViewBytes[v];
    return (width + 255) / 256 * 256 + 256 + 16 * static_cast<size_t>(v);
}  // distinct padded strides

struct HostCsr { std::vector<u32> summary, offsets, touched, rows, slots, bexp, bbeg, tcount, texp, tbeg; };
struct State {
    u32 T = 0;
    bool route_bf16 = false;
    std::vector<u8> view[V_count];
    HostCsr csr;
    template <class X> X* at(int v, u32 r) { return reinterpret_cast<X*>(view[v].data() + static_cast<size_t>(r) * stride_of(v, route_bf16)); }
};
static constexpr u8 kSentinel = 0xA7;
static void init_state(State& s, u32 T, bool bf16 = false) {
    s.T = T;
    s.route_bf16 = bf16;
    for (int v = 0; v < V_count; ++v) s.view[v].assign(static_cast<size_t>(T) * stride_of(v, bf16), kSentinel);
    const auto sz = mo::csr_sizes(T);
    auto fill = [](std::vector<u32>& x, size_t n) { x.assign(n, 0xA7A7A7A7U); };
    fill(s.csr.summary, 32); fill(s.csr.offsets, mo::experts + 1); fill(s.csr.touched, mo::experts);
    fill(s.csr.rows, sz.assignments); fill(s.csr.slots, sz.assignments);
    fill(s.csr.bexp, sz.max_batches); fill(s.csr.bbeg, sz.max_batches + 1);
    fill(s.csr.tcount, 1); fill(s.csr.texp, sz.max_tiles); fill(s.csr.tbeg, sz.max_tiles + 1);
}

struct HostWeights { u32 gate_type, down_type, experts_alloc; std::vector<u8> gate, up, down; };
static u32 gate_block_bytes(u32 t) { return t == mo::type_q5_k ? 176 : 144; }
static u32 down_block_bytes(u32 t) { return t == mo::type_q5_1 ? 24 : 34; }

// ---------------------------------------------------------------------------------------------------------------------
// Literal host reference.
static void ref_prologue(State& s, u32 clear_words) {
    for (u32 row = 0; row < s.T; ++row) {
        unsigned route_bits = 0;
        float* clear = s.at<float>(V_acc, row);
        for (u32 i = 0; i < clear_words; ++i) clear[i] = 0.0F;
        for (u32 k = 0; k < mo::top_k; ++k)
            if (s.at<u32>(V_routes, row)[k] >= mo::experts || !finite(s.at<float>(V_rw, row)[k])) route_bits |= mo::invalid_route;
        unsigned bad = 0;
        if (route_bits == 0) {  // quantize_row: initial_status != 0 -> return 0
            std::vector<i8> lc(mo::hidden); std::vector<float> ld(80); std::vector<int> ls(80);
            const u16* in = s.at<u16>(V_input, row);
            for (u32 group = 0; group < mo::hidden / 32; ++group) {
                float x[32], a[32];
                for (u32 lane = 0; lane < 32; ++lane) {
                    x[lane] = f_of(static_cast<u32>(in[group * 32 + lane]) << 16);
                    if (!finite(x[lane])) { bad |= mo::nonfinite_activation; x[lane] = 0.0F; }
                    a[lane] = std::fabs(x[lane]);
                }
                for (int stride = 16; stride != 0; stride /= 2) { float n[32]; for (int l = 0; l < 32; ++l) n[l] = std::fmax(a[l], a[l ^ stride]); std::memcpy(a, n, sizeof a); }
                int sum[32];
                for (u32 lane = 0; lane < 32; ++lane) {
                    float delta = a[lane] == 0.0F ? 1.0F : a[lane] / 127.0F;
                    if (!(delta > 0.0F) || !finite(delta)) { bad |= mo::invalid_delta; delta = 1.0F; }
                    const float rounded = std::nearbyint(x[lane] / delta);
                    const int code = static_cast<int>(std::fmin(127.0F, std::fmax(-127.0F, rounded)));
                    lc[group * 32 + lane] = static_cast<i8>(code); sum[lane] = code;
                    if (lane == 0) ld[group] = delta;
                }
                for (int stride = 16; stride != 0; stride /= 2) { int n[32]; for (int l = 0; l < 32; ++l) n[l] = sum[l] + sum[l ^ stride]; std::memcpy(sum, n, sizeof sum); }
                ls[group] = sum[0];
            }
            if (bad == 0) {
                std::memcpy(s.at<i8>(V_codes, row), lc.data(), mo::hidden);
                std::memcpy(s.at<float>(V_deltas, row), ld.data(), 320); std::memcpy(s.at<int>(V_sums, row), ls.data(), 320);
            }
        }
        *s.at<u32>(V_status, row) = route_bits | bad;
    }
}

// build_prefill_csr (serial, known_rows.hip:403-452) with cap 16, plus the CUDA port's 64-assignment tiles.
static void ref_csr(State& s) {
    auto& c = s.csr;
    const u32 rows = s.T, count = rows * mo::top_k;
    unsigned failure = 0, status = 0, duplicates = 0, maximum = 0, hist[16] = {};
    std::vector<u32> expert_of(count);
    for (u32 a = 0; a < count; ++a) {
        const u32 e = s.at<u32>(V_routes, a / 10)[a % 10];
        if (e >= mo::experts || !finite(s.at<float>(V_rw, a / 10)[a % 10])) failure |= mo::invalid_route;
        expert_of[a] = e < mo::experts ? e : 0;
    }
    for (u32 r = 0; r < rows; ++r) status |= *s.at<u32>(V_status, r);
    std::vector<u32> n_of(512, 0);
    if (!failure) { for (u32 a = 0; a < count; ++a) ++n_of[expert_of[a]]; for (u32 e = 0; e < 512; ++e) if (n_of[e] > 65535) failure |= mo::invalid_route; }
    if (failure) {
        for (u32 t = 0; t < 32; ++t) c.summary[t] = t == 0 ? failure | status : t == 3 ? count : t == 6 ? rows : t == 7 ? 1 : 0;
        c.tcount[0] = 0;
        return;
    }
    u32 off = 0, touched = 0, batch = 0, tile = 0;
    for (u32 e = 0; e < 512; ++e) {
        const u32 n = n_of[e], nb = (n + 15) / 16, nt = (n + 63) / 64;
        c.offsets[e] = off;
        if (n) {
            c.touched[touched] = e; maximum = std::max(maximum, n);
            for (u32 b = 0; b < nb; ++b) { c.bexp[batch + b] = e; c.bbeg[batch + b] = off + b * 16; ++hist[8 + std::min(8U, std::min(16U, n - b * 16)) - 1]; }
            for (u32 b = 0; b < nt; ++b) { c.texp[tile + b] = e; c.tbeg[tile + b] = off + b * 64; }
            u32 k = 0, unique = 0, last = UINT32_MAX;
            for (u32 a = 0; a < count && k < n; ++a) if (expert_of[a] == e) {
                const u32 r = a / 10; c.rows[off + k] = r; c.slots[off + k] = a % 10; ++k; unique += r != last; last = r;
            }
            ++hist[std::min(8U, unique) - 1]; duplicates += n - unique;
        }
        off += n; touched += n != 0; batch += nb; tile += nt;
    }
    c.offsets[512] = off;
    c.bbeg[batch] = count; c.tbeg[tile] = count; c.tcount[0] = tile;
    for (u32 t = 0; t < 32; ++t)
        c.summary[t] = t == 0 ? status : t == 1 ? touched : t == 2 ? batch : t == 3 ? count : t == 4 ? duplicates : t == 5 ? maximum
            : t == 6 ? rows : t == 7 ? 1 : t >= 8 && t < 24 ? hist[t - 8] : 0;
}

static unsigned factor(const u8* hd, unsigned group, bool minimum) {
    u32 y, z, w; std::memcpy(&y, hd + 4, 4); std::memcpy(&z, hd + 8, 4); std::memcpy(&w, hd + 12, 4);
    const unsigned shift = 8U * (group & 3U);
    const unsigned a = (y >> shift) & 255U, b = (z >> shift) & 255U, c = (w >> shift) & 255U;
    return group < 4U ? (minimum ? b : a) & 63U : minimum ? (c >> 4U) | ((b >> 6U) << 4U) : (c & 15U) | ((a >> 6U) << 4U);
}

// gate_up_chains for one (assignment row codes, matrix row): K blocks ascending, pairs p in order, fold.
static float ref_gate_up_value(const u8* wrow, bool q5k, const i8* codes, const float* deltas, const int* sums) {
    const u32 WB = q5k ? 176 : 144, QS = q5k ? 48 : 16;
    float acc[4] = {0, 0, 0, 0};
    for (u32 b = 0; b < 10; ++b) {
        const u8* blk = wrow + b * WB;
        u16 dh, dmh; std::memcpy(&dh, blk, 2); std::memcpy(&dmh, blk + 2, 2);
        const float d = half_to_float(dh), dm = half_to_float(dmh);
        for (u32 p = 0; p < 4; ++p) {
            const u32 g0 = 2 * p, g1 = g0 + 1;
            int r0 = 0, r1 = 0;
            for (u32 k = 0; k < 32; ++k) {
                const u8 byte = blk[QS + 32 * p + k];
                int q0 = byte & 15, q1 = byte >> 4;
                if (q5k) { const u8 h = blk[16 + k]; q0 |= ((h >> g0) & 1) << 4; q1 |= ((h >> g1) & 1) << 4; }
                r0 += codes[256 * b + 64 * p + k] * q0; r1 += codes[256 * b + 64 * p + 32 + k] * q1;
            }
            const float sc0 = static_cast<float>(factor(blk, g0, false)), sc1 = static_cast<float>(factor(blk, g1, false));
            const float mn0 = static_cast<float>(factor(blk, g0, true)), mn1 = static_cast<float>(factor(blk, g1, true));
            const float s0 = static_cast<float>(sums[8 * b + g0]), s1 = static_cast<float>(sums[8 * b + g1]);
            const float e0 = deltas[8 * b + g0], e1 = deltas[8 * b + g1];
            const float i0 = sc0 * static_cast<float>(r0), i1 = sc1 * static_cast<float>(r1);
            const float j0 = mn0 * s0, j1 = mn1 * s1;
            const float u0 = d * i0, v0 = dm * j0, t0 = u0 - v0;
            const float u1 = d * i1, v1 = dm * j1, t1 = u1 - v1;
            const float c0 = e0 * t0, c1 = e1 * t1;
            const float pair = c0 + c1;
            acc[p] = acc[p] + pair;
        }
    }
    const float s01 = acc[0] + acc[1], s23 = acc[2] + acc[3];
    return s01 + s23;
}

// project_prefill_wmma per HIP batch (cap 16): gate/up of every assignment, then the job loop (one 32-output group).
static void ref_project(State& s, const HostWeights& w, const HostExp& hx) {
    auto& c = s.csr;
    if ((c.summary[0] & ~mo::projection_nonfinite) != 0) return;
    const bool q5k = w.gate_type == mo::type_q5_k;
    const u32 WB = gate_block_bytes(w.gate_type);
    for (u32 batch = 0; batch < c.summary[2]; ++batch) {
        const u32 begin = c.bbeg[batch], count = c.bbeg[batch + 1] - begin, expert = c.bexp[batch];
        for (u32 tile = 0; tile < mo::ffn / 128; ++tile) {  // grid y: 128 outputs of each matrix
            for (u32 a = 0; a < count; ++a) {
                const u32 r = c.rows[begin + a], slot = c.slots[begin + a];
                float g[128], u[128];
                for (u32 o = 0; o < 128; ++o) {
                    const size_t off = (static_cast<size_t>(expert) * mo::ffn + tile * 128 + o) * 10 * WB;
                    g[o] = ref_gate_up_value(w.gate.data() + off, q5k, s.at<i8>(V_codes, r), s.at<float>(V_deltas, r), s.at<int>(V_sums, r));
                    u[o] = ref_gate_up_value(w.up.data() + off, q5k, s.at<i8>(V_codes, r), s.at<float>(V_deltas, r), s.at<int>(V_sums, r));
                }
                for (u32 quad = 0; quad < 4; ++quad) {
                    float* output = s.at<float>(V_inter, r);
                    const u32 tile_index = tile * 4 + quad;
                    float hv[32]; bool bad = false;
                    for (u32 lane = 0; lane < 32; ++lane) {
                        const float gv = g[32 * quad + lane], uv = u[32 * quad + lane];
                        hv[lane] = (gv / (1.0F + hx.expf_(-gv))) * uv;
                        output[slot * 2 * mo::ffn + tile_index * 32 + lane + mo::ffn] = canon(uv);
                        bad |= !finite(gv) || !finite(uv) || !finite(hv[lane]);
                    }
                    if (bad) {
                        for (u32 lane = 0; lane < 32; ++lane) output[slot * 2 * mo::ffn + tile_index * 32 + lane] = canon(g[32 * quad + lane]);
                        *s.at<u32>(V_status, r) |= mo::projection_nonfinite; c.summary[0] |= mo::projection_nonfinite;
                        continue;
                    }
                    float x[32];
                    for (u32 l = 0; l < 32; ++l) x[l] = std::fabs(hv[l]);
                    for (int stride = 16; stride != 0; stride /= 2) { float n[32]; for (int l = 0; l < 32; ++l) n[l] = std::fmax(x[l], x[l ^ stride]); std::memcpy(x, n, sizeof x); }
                    int sum = 0;
                    const bool tiny = x[0] > 0.0F && x[0] < 0x1p-119F;
                    const float delta = x[0] == 0.0F || tiny ? 1.0F : x[0] / 127.0F;
                    for (u32 lane = 0; lane < 32; ++lane) {
                        const float rounded = tiny ? 0.0F : std::nearbyint(hv[lane] / delta);
                        const int code = static_cast<int>(std::fmin(127.0F, std::fmax(-127.0F, rounded)));
                        sum += code;
                        output[slot * 2 * mo::ffn + tile_index * 32 + lane] = hv[lane];
                        s.at<i8>(V_dcodes, r)[slot * mo::ffn + tile_index * 32 + lane] = static_cast<i8>(code);
                    }
                    s.at<float>(V_ddeltas, r)[slot * 20 + tile_index] = delta; s.at<int>(V_dsums, r)[slot * 20 + tile_index] = sum;
                }
            }
        }
    }
}

// Independent integer reference of the specified storage rounding.
static u16 route_reference(u32 bits) {
    u32 hi = bits >> 16U, lo = bits & 65535U;
    if ((bits & 0x7fffffffU) > 0x7f800000U) return static_cast<u16>(hi | 64U);
    if (lo > 32768U || (lo == 32768U && (hi & 1U))) ++hi;
    return static_cast<u16>(hi);
}
static float ref_raw(const State& s, u32 row, u32 index) {
    const auto* p = s.view[V_raw].data() + static_cast<size_t>(row) * stride_of(V_raw, s.route_bf16);
    if (s.route_bf16) {
        u16 h; std::memcpy(&h, p + static_cast<size_t>(index) * 2U, 2);
        return f_of(static_cast<u32>(h) << 16U);
    }
    float f; std::memcpy(&f, p + static_cast<size_t>(index) * 4U, 4); return f;
}
static void ref_down(State& s, const HostWeights& w) {
    auto& c = s.csr;
    if (c.summary[0]) return;
    const bool q51 = w.down_type == mo::type_q5_1;
    const u32 WB = down_block_bytes(w.down_type);
    for (u32 batch = 0; batch < c.summary[2]; ++batch) {
        const u32 begin = c.bbeg[batch], count = c.bbeg[batch + 1] - begin, expert = c.bexp[batch];
        for (u32 a = 0; a < count; ++a) {
            const u32 r = c.rows[begin + a], slot = c.slots[begin + a];
            const i8* codes = s.at<i8>(V_dcodes, r) + slot * mo::ffn;
            const float* deltas = s.at<float>(V_ddeltas, r) + slot * 20;
            const int* sums = s.at<int>(V_dsums, r) + slot * 20;
            for (u32 o = 0; o < mo::hidden; ++o) {
                const u8* row = w.down.data() + (static_cast<size_t>(expert) * mo::hidden + o) * 20 * WB;
                float acc[4] = {0, 0, 0, 0};
                for (u32 group = 0; group < 20; ++group) {
                    const u8* blk = row + group * WB;
                    u16 dh; std::memcpy(&dh, blk, 2);
                    const float d = half_to_float(dh);
                    int rr = 0;
                    float m = 0.0F;
                    if (q51) {
                        u16 mh; std::memcpy(&mh, blk + 2, 2); m = half_to_float(mh);
                        u32 qh; std::memcpy(&qh, blk + 4, 4);
                        for (u32 k = 0; k < 32; ++k) {
                            const u8 byte = blk[8 + (k & 15)];
                            const int q = (k < 16 ? (byte & 15) : (byte >> 4)) | static_cast<int>(((qh >> k) & 1U) << 4);
                            rr += codes[32 * group + k] * q;
                        }
                    } else {
                        for (u32 k = 0; k < 32; ++k) rr += codes[32 * group + k] * static_cast<i8>(blk[2 + k]);
                    }
                    const float first_term = d * static_cast<float>(rr);
                    float combined = first_term;
                    if (q51) { const float correction = m * static_cast<float>(sums[group]); combined = first_term + correction; }
                    const float contribution = deltas[group] * combined;
                    acc[group % 4] = acc[group % 4] + contribution;
                }
                const float s01 = acc[0] + acc[1], s23 = acc[2] + acc[3];
                if (s.route_bf16) s.at<u16>(V_raw, r)[slot * mo::hidden + o] = route_reference(b_of(s01 + s23));
                else s.at<float>(V_raw, r)[slot * mo::hidden + o] = s01 + s23;
            }
        }
    }
}

static void ref_merge_commit(State& s, u32* record) {
    auto& header = s.csr.summary;
    const u32 header_at_launch = header[0];  // the HIP merge reads header[0] racily after another row sets bit 8 (see test 3)
    for (u32 row = 0; row < s.T; ++row) {
        u32* status = s.at<u32>(V_status, row);
        if (header_at_launch || *status) continue;
        bool route_bad = false;
        for (u32 k = 0; k < 10; ++k) route_bad |= s.at<u32>(V_routes, row)[k] >= mo::experts;
        if (route_bad) continue;
        for (u32 out = 0; out < mo::hidden; ++out) {
            float current = 0.0F;
            for (u32 k = 0; k < 10; ++k) current = std::fmaf(s.at<float>(V_rw, row)[k], ref_raw(s, row, k * mo::hidden + out), current);
            s.at<float>(V_acc, row)[out] = canon(current);
            if ((b_of(current) & 0x7fffffffU) >= 0x7f7f8000U) { *status |= 8; header[0] |= 8; }
        }
    }
    const u32 failure = header[0];
    u32 first_bad = s.T;
    for (u32 r = 0; r < s.T; ++r) if (*s.at<u32>(V_status, r)) { first_bad = r; break; }
    if (failure & 7U) { for (u32 wd : {1U, 2U, 4U, 5U}) header[wd] = 0; for (u32 wd = 8; wd < 24; ++wd) header[wd] = 0; }
    if (first_bad < s.T && record[0] == 0) { record[1] = *s.at<u32>(V_status, first_bad); record[0] = 2; }
    if (failure) return;
    for (u32 row = 0; row < s.T; ++row)
        for (u32 out = 0; out < mo::hidden; ++out) {
            const u32 bits = b_of(s.at<float>(V_acc, row)[out]);
            s.at<u16>(V_out, row)[out] = static_cast<u16>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
        }
}

// ---------------------------------------------------------------------------------------------------------------------
// Device mirror.
struct Device {
    u8* view[V_count] = {};
    u32 *summary = nullptr, *offsets = nullptr, *touched = nullptr, *rows = nullptr, *slots = nullptr, *bexp = nullptr, *bbeg = nullptr,
        *tcount = nullptr, *texp = nullptr, *tbeg = nullptr, *record = nullptr;
    u8 *gate = nullptr, *up = nullptr, *down = nullptr;
    mo::Rows rows_view{};
    mo::Weights weights{};
};
template <class X> static X* dalloc(size_t n) { X* p = nullptr; CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(X))); return p; }
static void upload_state(Device& d, const State& s, bool with_inter) {
    for (int v = 0; v < V_count; ++v) {
        if (!d.view[v]) d.view[v] = dalloc<u8>(s.view[v].size());
        CK(cudaMemcpy(d.view[v], s.view[v].data(), s.view[v].size(), cudaMemcpyHostToDevice));
    }
    const auto sz = mo::csr_sizes(s.T);
    auto up = [](u32*& p, const std::vector<u32>& h) { if (!p) p = dalloc<u32>(h.size()); CK(cudaMemcpy(p, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); };
    up(d.summary, s.csr.summary); up(d.offsets, s.csr.offsets); up(d.touched, s.csr.touched); up(d.rows, s.csr.rows); up(d.slots, s.csr.slots);
    up(d.bexp, s.csr.bexp); up(d.bbeg, s.csr.bbeg); up(d.tcount, s.csr.tcount); up(d.texp, s.csr.texp); up(d.tbeg, s.csr.tbeg);
    std::vector<u32> rec(2, 0); up(d.record, rec);
    (void)sz;
    auto& v = d.rows_view;
    auto S = [&](int k) { return static_cast<std::uint64_t>(stride_of(k, s.route_bf16)); };
    v.input = {reinterpret_cast<const u16*>(d.view[V_input]), S(V_input)};
    v.routes = {reinterpret_cast<const u32*>(d.view[V_routes]), S(V_routes)};
    v.route_weights = {reinterpret_cast<const float*>(d.view[V_rw]), S(V_rw)};
    v.codes = {reinterpret_cast<i8*>(d.view[V_codes]), S(V_codes)};
    v.deltas = {reinterpret_cast<float*>(d.view[V_deltas]), S(V_deltas)};
    v.sums = {reinterpret_cast<int*>(d.view[V_sums]), S(V_sums)};
    v.status = {reinterpret_cast<u32*>(d.view[V_status]), S(V_status)};
    v.intermediates = {with_inter ? reinterpret_cast<float*>(d.view[V_inter]) : nullptr, S(V_inter)};
    v.accumulation = {reinterpret_cast<float*>(d.view[V_acc]), S(V_acc)};
    v.down_codes = {reinterpret_cast<i8*>(d.view[V_dcodes]), S(V_dcodes)};
    v.down_deltas = {reinterpret_cast<float*>(d.view[V_ddeltas]), S(V_ddeltas)};
    v.down_sums = {reinterpret_cast<int*>(d.view[V_dsums]), S(V_dsums)};
    v.rawdots = {reinterpret_cast<float*>(d.view[V_raw]), S(V_raw)};
    v.route_bf16 = s.route_bf16;
    v.rawdot_capacity = s.view[V_raw].size();
    v.output = {reinterpret_cast<u16*>(d.view[V_out]), S(V_out)};
    v.csr = {d.summary, d.offsets, d.touched, d.rows, d.slots, d.bexp, d.bbeg, d.tcount, d.texp, d.tbeg};
}
static void free_state_buffers(Device& d) {
    for (auto& p : d.view) { if (p) cudaFree(p); p = nullptr; }
    for (u32** p : {&d.summary, &d.offsets, &d.touched, &d.rows, &d.slots, &d.bexp, &d.bbeg, &d.tcount, &d.texp, &d.tbeg, &d.record}) {
        if (*p) cudaFree(*p); *p = nullptr;
    }
}
static void free_device(Device& d) {
    free_state_buffers(d);
    for (u8* p : {d.gate, d.up, d.down}) if (p) cudaFree(p);
    d = Device{};
}
static void upload_weights(Device& d, const HostWeights& w) {
    d.gate = dalloc<u8>(w.gate.size()); d.up = dalloc<u8>(w.up.size()); d.down = dalloc<u8>(w.down.size());
    CK(cudaMemcpy(d.gate, w.gate.data(), w.gate.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.up, w.up.data(), w.up.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.down, w.down.data(), w.down.size(), cudaMemcpyHostToDevice));
    d.weights = {w.gate_type, d.gate, d.up, w.down_type, d.down};
}

// Compare every view and CSR array; element granularity for the report.
static void compare(const std::string& name, const Device& d, const State& s, bool with_inter, int skip = -1) {
    for (int v = 0; v < V_count; ++v) {
        if ((v == V_inter && !with_inter) || v == skip) continue;
        std::vector<u8> got(s.view[v].size());
        CK(cudaMemcpy(got.data(), d.view[v], got.size(), cudaMemcpyDeviceToHost));
        size_t bad = 0, first = SIZE_MAX;
        for (size_t i = 0; i < got.size(); ++i) if (got[i] != s.view[v][i]) { if (!bad) first = i; ++bad; }
        g_checked += got.size();
        if (bad) {
            const size_t row = first / stride_of(v, s.route_bf16), off = first % stride_of(v, s.route_bf16);
            u32 e = 0, g = 0; const size_t al = first & ~size_t(3);
            std::memcpy(&e, s.view[v].data() + al, std::min<size_t>(4, got.size() - al)); std::memcpy(&g, got.data() + al, std::min<size_t>(4, got.size() - al));
            std::printf("  %s/%s: %zu bytes differ, first row %zu byte %zu: expected word %08x (%g) got %08x (%g)\n", name.c_str(), kViewName[v],
                        bad, row, off, e, f_of(e), g, f_of(g));
        }
        expect(bad == 0, name + " view " + kViewName[v]);
    }
    const std::pair<const char*, std::pair<u32*, const std::vector<u32>*>> arrays[] = {
        {"summary", {d.summary, &s.csr.summary}}, {"offsets", {d.offsets, &s.csr.offsets}}, {"touched", {d.touched, &s.csr.touched}},
        {"csr_rows", {d.rows, &s.csr.rows}}, {"csr_slots", {d.slots, &s.csr.slots}}, {"batch_experts", {d.bexp, &s.csr.bexp}},
        {"batch_begins", {d.bbeg, &s.csr.bbeg}}, {"tile_count", {d.tcount, &s.csr.tcount}}, {"tile_experts", {d.texp, &s.csr.texp}},
        {"tile_begins", {d.tbeg, &s.csr.tbeg}}};
    for (const auto& a : arrays) {
        std::vector<u32> got(a.second.second->size());
        CK(cudaMemcpy(got.data(), a.second.first, got.size() * 4, cudaMemcpyDeviceToHost));
        size_t bad = 0, first = 0;
        for (size_t i = 0; i < got.size(); ++i) if (got[i] != (*a.second.second)[i]) { if (!bad) first = i; ++bad; }
        g_checked += got.size() * 4;
        if (bad) std::printf("  %s/%s: %zu words differ, first [%zu] expected %u got %u\n", name.c_str(), a.first, bad, first, (*a.second.second)[first], got[first]);
        expect(bad == 0, name + " csr " + a.first);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Inputs.
static u16 bf16_of(float f) { const u32 b = b_of(f); return static_cast<u16>((b + 0x7fffU + ((b >> 16) & 1U)) >> 16); }
static u16 half_of(float f) {  // RNE float -> half for moderate values
    const u32 b = b_of(f), s = (b >> 31) << 15;
    const float a = std::fabs(f);
    if (a == 0.0F) return static_cast<u16>(s);
    if (a < 0x1p-14F) return static_cast<u16>(s | static_cast<u32>(std::nearbyint(a * 0x1p24F)));
    int e; const float m = std::frexp(a, &e);  // a = m * 2^e, m in [0.5, 1)
    u32 mant = static_cast<u32>(std::nearbyint((m * 2.0F - 1.0F) * 1024.0F));
    int he = e - 1 + 15;
    if (mant == 1024) { mant = 0; ++he; }
    return static_cast<u16>(s | (static_cast<u32>(he) << 10) | mant);
}

static HostWeights synthetic_weights(u32 gate_type, u32 down_type, u32 experts_alloc, std::mt19937& rng) {
    HostWeights w{gate_type, down_type, experts_alloc, {}, {}, {}};
    const u32 GB = gate_block_bytes(gate_type), DB = down_block_bytes(down_type);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> uni(0.0F, 1.0F);
    auto scale = [&](float lo_exp, float hi_exp) {
        const float r = uni(rng);
        if (r < 0.01F) return 0.0F;
        if (r < 0.02F) return 0x1p-20F * static_cast<float>(1 + byte(rng));  // FP16 subnormal range
        float v = std::exp2(lo_exp + (hi_exp - lo_exp) * uni(rng));
        return uni(rng) < 0.1F ? -v : v;
    };
    auto fill_gate = [&](std::vector<u8>& m) {
        m.resize(static_cast<size_t>(experts_alloc) * mo::ffn * 10 * GB);
        for (auto& x : m) x = static_cast<u8>(byte(rng));
        for (size_t blk = 0; blk < m.size() / GB; ++blk) {
            const u16 d = half_of(scale(-12, -4)), dm = half_of(scale(-12, -5));
            std::memcpy(m.data() + blk * GB, &d, 2); std::memcpy(m.data() + blk * GB + 2, &dm, 2);
        }
    };
    fill_gate(w.gate); fill_gate(w.up);
    w.down.resize(static_cast<size_t>(experts_alloc) * mo::hidden * 20 * DB);
    for (auto& x : w.down) x = static_cast<u8>(byte(rng));
    for (size_t blk = 0; blk < w.down.size() / DB; ++blk) {
        const u16 d = half_of(scale(-12, -5)); std::memcpy(w.down.data() + blk * DB, &d, 2);
        if (down_type == mo::type_q5_1) { const u16 m = half_of(scale(-10, -4) * -1.0F); std::memcpy(w.down.data() + blk * DB + 2, &m, 2); }
    }
    return w;
}

// Inputs and routes. kind bits: 1 skewed routes, 2 duplicate experts within rows, 4 special rows (zero, tiny, subnormal).
static void synthetic_inputs(State& s, u32 experts_alloc, unsigned kind, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::uniform_real_distribution<float> uni(0.0F, 1.0F);
    std::uniform_int_distribution<u32> ex(0, experts_alloc - 1);
    for (u32 r = 0; r < s.T; ++r) {
        u16* in = s.at<u16>(V_input, r);
        const float sc = std::exp2(-6.0F + 10.0F * uni(rng));
        for (u32 i = 0; i < mo::hidden; ++i) in[i] = bf16_of(nd(rng) * sc);
        if (kind & 4) {
            if (r % 7 == 3) for (u32 i = 0; i < mo::hidden; ++i) in[i] = 0;                                  // zero row
            if (r % 7 == 5) for (u32 i = 0; i < mo::hidden; ++i) in[i] = bf16_of(nd(rng) * 1e-25F);         // tiny row -> tiny h
            if (r % 7 == 6) for (u32 i = 64; i < 128; ++i) in[i] = static_cast<u16>((i & 1 ? 0x8000 : 0) | (i % 120 + 1));  // subnormal groups
            if (r % 5 == 1) for (u32 i = 0; i < 32; ++i) in[i] = static_cast<u16>(i & 1 ? 0x8000 : 0);      // a +-0 group
        }
        u32* routes = s.at<u32>(V_routes, r); float* rw = s.at<float>(V_rw, r);
        for (u32 k = 0; k < 10; ++k) {
            u32 e = ex(rng);
            if ((kind & 1) && uni(rng) < 0.6F) e = ex(rng) % 3;  // hot experts 0..2
            if ((kind & 2) && k > 0 && uni(rng) < 0.25F) e = routes[k - 1];
            routes[k] = e;
            rw[k] = uni(rng) * (uni(rng) < 0.05F ? -1.0F : 1.0F);
        }
        if ((kind & 2) && r % 9 == 4) for (u32 k = 0; k < 10; ++k) routes[k] = 1;  // one row: 10 slots of one expert
    }
}

// Full pipeline on both sides and full comparison. hook(State&) may edit inputs before the run.
static void run_pipeline(const std::string& name, State& s, const HostWeights& w, Device& d, const am::Tables& tables, const HostExp& hx,
                         bool with_inter, u32 clear_words = mo::hidden) {
    upload_state(d, s, with_inter);
    mo::enqueue(d.rows_view, s.T, clear_words, d.weights, tables, d.record, nullptr);
    CK(cudaDeviceSynchronize());
    u32 record[2] = {0, 0};
    const auto t0 = std::chrono::steady_clock::now();
    ref_prologue(s, clear_words); ref_csr(s); ref_project(s, w, hx); ref_down(s, w); ref_merge_commit(s, record);
    const double host_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    compare(name, d, s, with_inter);
    u32 grec[2]; CK(cudaMemcpy(grec, d.record, 8, cudaMemcpyDeviceToHost));
    expect(grec[0] == record[0] && grec[1] == record[1], name + " record");
    g_checked += 8;
    std::printf("%s: T=%u summary0=%u touched=%u batches=%u tiles=%u record={%u,%u} host %.1fs %s\n", name.c_str(), s.T, s.csr.summary[0],
                s.csr.summary[1], s.csr.summary[2], s.csr.tcount[0], record[0], record[1], host_s, g_fail ? "(failures so far)" : "ok");
}

// Partial tensor read: experts [0, n) of an expert-major tensor.
static std::vector<u8> read_experts(const strix::qwen_gguf::Model& m, const std::string& name, u32 n, u32& type) {
    const auto& t = m.tensor(name);
    type = static_cast<u32>(t.ggml_type);
    const u64 per = t.byte_size / 512;
    std::vector<u8> v(static_cast<size_t>(per) * n);
    const char* e = std::getenv("STRIX_QWEN_GGUF");
    const std::string path = e && *e ? e : "Qwen3.8-Flash-Next-UD-Q4_K_XL-merged.gguf";
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::printf("cannot open %s\n", path.c_str()); std::exit(2); }
    _fseeki64(f, static_cast<long long>(t.absolute_offset), SEEK_SET);
    if (std::fread(v.data(), 1, v.size(), f) != v.size()) { std::printf("short read %s\n", name.c_str()); std::exit(2); }
    std::fclose(f);
    return v;
}

// ---------------------------------------------------------------------------------------------------------------------
// Fixture replay of the moe_experts step: reads {input BF16, routes, route weights} per row, write {routed BF16 output}.
static void fixture_replay(const am::Tables& tables) {
    const std::string path = "qrtx-build\\fx\\d64.dump";
    if (!std::filesystem::exists(path)) { std::printf("fixture %s absent: replay skipped\n", path.c_str()); return; }
    strix::qwen_fixture::Dump dump(path);
    strix::qwen_gguf::Model model;
    const u32 op_moe_experts = 10;  // SerialOp::moe_experts (include/strix/execution_plan.hpp:12-29)
    std::vector<u32> steps;
    for (const auto& r : dump.records()) if (r.op == op_moe_experts && r.row == 0 && !r.is_read && r.index == 0) steps.push_back(r.step);
    std::printf("fixture: %zu moe_experts steps\n", steps.size());
    unsigned steps_ok = 0;
    for (const u32 step : steps) {
        const auto* w0 = dump.find(step, false, 0, 0);
        const u32 T = w0->rows, layer = w0->layer;
        const auto input = dump.load_rows(step, true, 0), routes = dump.load_rows(step, true, 1), rws = dump.load_rows(step, true, 2);
        const auto expected = dump.load_rows(step, false, 0);
        if (input.size() != size_t(T) * 5120 || routes.size() != size_t(T) * 40 || rws.size() != size_t(T) * 40 || expected.size() != size_t(T) * 5120) {
            std::printf("  step %u layer %u: unexpected buffer sizes (%zu %zu %zu %zu)\n", step, layer, input.size(), routes.size(), rws.size(), expected.size());
            ++g_fail; continue;
        }
        // Compact relabelling of the experts used (a pure renaming: no (row, slot) value depends on the expert id order).
        std::vector<u32> used; std::vector<int> map(512, -1);
        std::vector<u32> rt(size_t(T) * 10); std::memcpy(rt.data(), routes.data(), routes.size());
        for (auto& e : rt) { if (e >= 512) continue; if (map[e] < 0) { map[e] = static_cast<int>(used.size()); used.push_back(e); } }
        HostWeights w{};
        const std::string pre = "blk." + std::to_string(layer) + ".";
        u32 gt = 0, dt = 0;
        const auto& gti = model.tensor(pre + "ffn_gate_exps.weight"); const auto& dti = model.tensor(pre + "ffn_down_exps.weight");
        gt = static_cast<u32>(gti.ggml_type); dt = static_cast<u32>(dti.ggml_type);
        const u64 gper = gti.byte_size / 512, dper = dti.byte_size / 512;
        w.gate_type = gt; w.down_type = dt; w.experts_alloc = static_cast<u32>(used.size());
        w.gate.resize(gper * used.size()); w.up.resize(gper * used.size()); w.down.resize(dper * used.size());
        const char* ge = std::getenv("STRIX_QWEN_GGUF");
        const std::string gpath = ge && *ge ? ge : "Qwen3.8-Flash-Next-UD-Q4_K_XL-merged.gguf";
        FILE* f = std::fopen(gpath.c_str(), "rb");
        for (size_t i = 0; i < used.size(); ++i) {
            for (const auto& [tn, dst, per] : {std::tuple{pre + "ffn_gate_exps.weight", &w.gate, gper}, std::tuple{pre + "ffn_up_exps.weight", &w.up, gper},
                                               std::tuple{pre + "ffn_down_exps.weight", &w.down, dper}}) {
                _fseeki64(f, static_cast<long long>(model.tensor(tn).absolute_offset + used[i] * per), SEEK_SET);
                if (std::fread(dst->data() + i * per, 1, per, f) != per) { std::printf("short read\n"); std::exit(2); }
            }
        }
        std::fclose(f);
        State s; init_state(s, T);
        for (u32 r = 0; r < T; ++r) {
            std::memcpy(s.at<u8>(V_input, r), input.data() + size_t(r) * 5120, 5120);
            for (u32 k = 0; k < 10; ++k) { const u32 e = rt[size_t(r) * 10 + k]; s.at<u32>(V_routes, r)[k] = e < 512 ? static_cast<u32>(map[e]) : e; }
            std::memcpy(s.at<u8>(V_rw, r), rws.data() + size_t(r) * 40, 40);
        }
        Device d; upload_weights(d, w); upload_state(d, s, false);
        mo::enqueue(d.rows_view, T, mo::hidden, d.weights, tables, d.record, nullptr);
        CK(cudaDeviceSynchronize());
        std::vector<u8> got(s.view[V_out].size());
        CK(cudaMemcpy(got.data(), d.view[V_out], got.size(), cudaMemcpyDeviceToHost));
        size_t bad = 0; long long first = -1;
        for (u32 r = 0; r < T; ++r) for (u32 i = 0; i < 2560; ++i) {
            u16 g, e; std::memcpy(&g, got.data() + size_t(r) * stride_of(V_out) + 2 * i, 2); std::memcpy(&e, expected.data() + size_t(r) * 5120 + 2 * i, 2);
            if (g != e) { if (first < 0) { first = r * 2560LL + i; std::printf("  step %u layer %u: first diff row %u elem %u expected %04x got %04x\n", step, layer, r, i, e, g); } ++bad; }
        }
        g_checked += size_t(T) * 5120;
        if (!bad) ++steps_ok;
        expect(bad == 0, "fixture step " + std::to_string(step) + " layer " + std::to_string(layer));
        std::printf("  step %u layer %u (gate %u down %u, %zu experts): %zu / %u BF16 differ\n", step, layer, gt, dt, used.size(), bad, T * 2560);
        free_device(d);
    }
    std::printf("fixture replay: %u / %zu moe_experts steps bit-exact\n", steps_ok, steps.size());
}

// ---------------------------------------------------------------------------------------------------------------------
// CUDA 13 has no unsynchronised __shfl_xor: same spelling as qwen_moe.cu.
__device__ __forceinline__ unsigned test_shfl_xor(unsigned v, int delta, int width) { return __shfl_xor_sync(0xffffffffU, v, delta, width); }
#define __shfl_xor test_shfl_xor
namespace route_rounding {
#include "../../src/hip_moe_route_bf16.inc"
#undef __shfl_xor
__global__ void probe(const u32* inputs, float* storage, u32* returned, u32* widened, u32 count) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    returned[i] = __float_as_uint(moe_route_store<true>(storage,i,__uint_as_float(inputs[i])));
    widened[i] = __float_as_uint(moe_route_load<true>(storage,i));
}
}
static void route_rounding_device_test() {
    std::vector<u32> bits;
    for (u32 upper=0; upper<65536U; ++upper)
        for (const auto lower : {0U,1U,0x7ffeU,0x7fffU,0x8000U,0x8001U,0xfffeU,0xffffU})
            bits.push_back(upper*65536U+lower);
    auto* in=dalloc<u32>(bits.size());
    auto* returned=dalloc<u32>(bits.size());
    auto* widened=dalloc<u32>(bits.size());
    const auto storage_bytes=bits.size()*2U+64U;
    auto* storage=dalloc<u8>(storage_bytes);
    CK(cudaMemcpy(in,bits.data(),bits.size()*4U,cudaMemcpyHostToDevice));
    CK(cudaMemset(storage,kSentinel,storage_bytes));
    route_rounding::probe<<<static_cast<u32>((bits.size()+255U)/256U),256>>>(in,
        reinterpret_cast<float*>(storage+16),returned,widened,static_cast<u32>(bits.size()));
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<u8> actual(storage_bytes);
    std::vector<u32> returned_host(bits.size()), widened_host(bits.size());
    CK(cudaMemcpy(actual.data(),storage,storage_bytes,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(returned_host.data(),returned,bits.size()*4U,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(widened_host.data(),widened,bits.size()*4U,cudaMemcpyDeviceToHost));
    size_t bad=0, first=bits.size();
    for (size_t i=0; i<bits.size(); ++i) {
        u16 stored; std::memcpy(&stored,actual.data()+16+2U*i,2);
        const auto expected=route_reference(bits[i]);
        if (!(stored==expected && returned_host[i]==(static_cast<u32>(expected)<<16U) &&
            widened_host[i]==returned_host[i])) { ++bad; if (first==bits.size()) first=i; }
        g_checked+=10U;
    }
    if (bad) std::printf("RNE device differences %zu first input %08x\n",bad,bits[first]);
    expect(bad==0,"device RNE boundaries");
    for (size_t i=0;i<actual.size();++i)
        if (i<16U || i>=16U+bits.size()*2U) expect(actual[i]==kSentinel,"device RNE guards");
    cudaFree(in);cudaFree(returned);cudaFree(widened);cudaFree(storage);
}

// ---------------------------------------------------------------------------------------------------------------------
// moe_mma FAST form (single FP32 chain, qwen-cuda-moe-i8fma-v1) against the exact kernels on the same device inputs.
//   project: hv / u intermediates and the down codes; down: rawdots from the exact kernel's down codes (isolates the down GEMM),
//   and end to end (FAST project then FAST down). Errors are reported as relative L2 plus max abs, binned by position inside the
//   CSR tile (0-15, 16-31, 32-47, 48-63), by output quad inside 64 (o % 64 / 16) and by slot, so a clustered (edge/last block/
//   partial tile) error shows as one bin standing out. The exact kernels are compared bytewise by the existing pipeline tests.
struct ErrAcc {
    double e2 = 0, r2 = 0, max_abs = 0; size_t n = 0;
    void add(double ref, double got) {
        const double e = got - ref; e2 += e * e; r2 += ref * ref; max_abs = std::max(max_abs, std::fabs(e)); ++n;
    }
    double rel() const { return r2 > 0 ? std::sqrt(e2 / r2) : (e2 > 0 ? 1.0 : 0.0); }
};
struct ErrBins {
    ErrAcc all, pos[4], quad[4], slot[10];
    void add(double ref, double got, u32 p, u32 q4, u32 sl) { all.add(ref, got); pos[p].add(ref, got); quad[q4].add(ref, got); slot[sl].add(ref, got); }
    double worst_bin() const {
        double w = 0;
        for (const auto& b : pos) w = std::max(w, b.rel());
        for (const auto& b : quad) w = std::max(w, b.rel());
        for (const auto& b : slot) if (b.n) w = std::max(w, b.rel());
        return w;
    }
    void print(const char* what) const {
        std::printf("  %-14s n=%zu relL2=%.3e maxabs=%.3e | pos %.2e %.2e %.2e %.2e | outq %.2e %.2e %.2e %.2e | worst bin %.2e (%.1fx)\n", what, all.n, all.rel(),
                    all.max_abs, pos[0].rel(), pos[1].rel(), pos[2].rel(), pos[3].rel(), quad[0].rel(), quad[1].rel(), quad[2].rel(), quad[3].rel(),
                    worst_bin(), all.rel() > 0 ? worst_bin() / all.rel() : 0.0);
    }
};
static std::vector<u8> snap(const Device& d, const State& s, int v) {
    std::vector<u8> h(s.view[v].size());
    CK(cudaMemcpy(h.data(), d.view[v], h.size(), cudaMemcpyDeviceToHost));
    return h;
}
static void fast_case(const std::string& name, u32 T, unsigned kind, const HostWeights& w, u32 NE, std::mt19937& rng, const am::Tables& tables) {
    State s; init_state(s, T); synthetic_inputs(s, NE, kind, rng);
    Device d; upload_weights(d, w); upload_state(d, s, true);
    auto& rv = d.rows_view;
    mo::enqueue_prologue(rv, T, mo::hidden, nullptr); mo::enqueue_csr(rv, T, nullptr);
    mo::enqueue_project(rv, T, d.weights, tables, nullptr, false); CK(cudaDeviceSynchronize());
    const auto inter_e = snap(d, s, V_inter), dc_e = snap(d, s, V_dcodes);
    mo::enqueue_down(rv, T, d.weights, nullptr, false); CK(cudaDeviceSynchronize());
    const auto raw_e = snap(d, s, V_raw);
    CK(cudaMemset(d.view[V_raw], 0xA7, s.view[V_raw].size()));
    mo::enqueue_down(rv, T, d.weights, nullptr, true); CK(cudaDeviceSynchronize());
    const auto raw_f1 = snap(d, s, V_raw);
    mo::enqueue_project(rv, T, d.weights, tables, nullptr, true); CK(cudaDeviceSynchronize());
    const auto inter_f = snap(d, s, V_inter), dc_f = snap(d, s, V_dcodes);
    CK(cudaMemset(d.view[V_raw], 0xA7, s.view[V_raw].size()));
    mo::enqueue_down(rv, T, d.weights, nullptr, true); CK(cudaDeviceSynchronize());
    const auto raw_f2 = snap(d, s, V_raw);
    // tile positions of every (row, slot)
    const auto sz = mo::csr_sizes(T);
    std::vector<u32> tcount(1), tbeg(sz.max_tiles + 1), rows(sz.assignments), slots(sz.assignments), summary(32);
    CK(cudaMemcpy(tcount.data(), d.tcount, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(tbeg.data(), d.tbeg, tbeg.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(rows.data(), d.rows, rows.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(slots.data(), d.slots, slots.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(summary.data(), d.summary, 128, cudaMemcpyDeviceToHost));
    std::vector<u8> pos(static_cast<size_t>(T) * 10, 0);
    u32 partial = 0, tiny = 0;
    for (u32 t = 0; t < tcount[0]; ++t) {
        const u32 cnt = tbeg[t + 1] - tbeg[t];
        if (cnt < 64) ++partial;
        if (cnt <= 8) ++tiny;
        for (u32 i = tbeg[t]; i < tbeg[t + 1]; ++i) pos[static_cast<size_t>(rows[i]) * 10 + slots[i]] = static_cast<u8>(i - tbeg[t]);
    }
    ErrBins hv, uu, r1, r2; size_t code_diff = 0, code_n = 0;
    for (u32 r = 0; r < T; ++r) for (u32 k = 0; k < 10; ++k) {
        const u32 p = pos[static_cast<size_t>(r) * 10 + k] >> 4;
        const float* ie = reinterpret_cast<const float*>(inter_e.data() + static_cast<size_t>(r) * stride_of(V_inter)) + k * 2 * mo::ffn;
        const float* inf = reinterpret_cast<const float*>(inter_f.data() + static_cast<size_t>(r) * stride_of(V_inter)) + k * 2 * mo::ffn;
        const i8* ce = reinterpret_cast<const i8*>(dc_e.data() + static_cast<size_t>(r) * stride_of(V_dcodes)) + k * mo::ffn;
        const i8* cf = reinterpret_cast<const i8*>(dc_f.data() + static_cast<size_t>(r) * stride_of(V_dcodes)) + k * mo::ffn;
        for (u32 f = 0; f < mo::ffn; ++f) {
            hv.add(ie[f], inf[f], p, (f % 64) >> 4, k); uu.add(ie[mo::ffn + f], inf[mo::ffn + f], p, (f % 64) >> 4, k);
            code_diff += ce[f] != cf[f]; ++code_n;
        }
        const float* a = reinterpret_cast<const float*>(raw_e.data() + static_cast<size_t>(r) * stride_of(V_raw)) + k * mo::hidden;
        const float* b1 = reinterpret_cast<const float*>(raw_f1.data() + static_cast<size_t>(r) * stride_of(V_raw)) + k * mo::hidden;
        const float* b2 = reinterpret_cast<const float*>(raw_f2.data() + static_cast<size_t>(r) * stride_of(V_raw)) + k * mo::hidden;
        for (u32 o = 0; o < mo::hidden; ++o) { r1.add(a[o], b1[o], p, (o % 64) >> 4, k); r2.add(a[o], b2[o], p, (o % 64) >> 4, k); }
    }
    std::printf("fast %s: T=%u tiles=%u (partial %u, <=8 rows %u) summary0=%u down-code flips %.4f%%\n", name.c_str(), T, tcount[0], partial, tiny, summary[0],
                100.0 * static_cast<double>(code_diff) / static_cast<double>(std::max<size_t>(code_n, 1)));
    hv.print("project hv"); uu.print("project u"); r1.print("down (same A)"); r2.print("down end2end");
    expect(summary[0] == 0, name + " summary");
    expect(hv.all.rel() < 1e-5 && uu.all.rel() < 1e-5, name + " project relL2");
    expect(r1.all.rel() < 1e-5, name + " down relL2");
    expect(r2.all.rel() < 3e-2, name + " end-to-end relL2");
    expect(hv.worst_bin() < 4.0 * hv.all.rel() + 1e-12 && uu.worst_bin() < 4.0 * uu.all.rel() + 1e-12 && r1.worst_bin() < 4.0 * r1.all.rel() + 1e-12,
           name + " error uniform across tile position / output quad / slot");
    g_checked += hv.all.n * 2 + r1.all.n * 2;
    free_device(d);
}
static void fast_tests(const am::Tables& tables) {
    std::mt19937 rng(99);
    for (const auto [gt, dt] : {std::pair{mo::type_q4_k, mo::type_q5_1}, std::pair{mo::type_q5_k, mo::type_q8_0},
                                std::pair{mo::type_q4_k, mo::type_q8_0}, std::pair{mo::type_q5_k, mo::type_q5_1}}) {
        const u32 NE = 24;
        HostWeights w = synthetic_weights(gt, dt, NE, rng);
        const std::string tag = "syn gate" + std::to_string(gt) + "/down" + std::to_string(dt);
        fast_case(tag + " T=1000 skew+dup", 1000, 1 | 2 | 4, w, NE, rng, tables);
        fast_case(tag + " T=37", 37, 0, w, NE, rng, tables);   // few rows per expert: tiny partial tiles
        fast_case(tag + " T=11", 11, 2, w, NE, rng, tables);
    }
    // Real weights: first NE experts of the first layer of every (gate, down) type pair of the model.
    strix::qwen_gguf::Model model;
    std::vector<std::pair<u32, u32>> seen;
    for (u32 layer = 0; layer < 96 && seen.size() < 4; ++layer) {
        const std::string pre = "blk." + std::to_string(layer) + ".";
        u32 gt = 0, dt = 0;
        try { gt = static_cast<u32>(model.tensor(pre + "ffn_gate_exps.weight").ggml_type); dt = static_cast<u32>(model.tensor(pre + "ffn_down_exps.weight").ggml_type); }
        catch (...) { continue; }
        if (!(gt == mo::type_q4_k || gt == mo::type_q5_k) || !(dt == mo::type_q5_1 || dt == mo::type_q8_0)) continue;
        if (std::find(seen.begin(), seen.end(), std::pair{gt, dt}) != seen.end()) continue;
        seen.push_back({gt, dt});
        const u32 NE = 48;
        HostWeights w; w.gate_type = gt; w.down_type = dt; w.experts_alloc = NE;
        u32 t1, t2, t3;
        w.gate = read_experts(model, pre + "ffn_gate_exps.weight", NE, t1); w.up = read_experts(model, pre + "ffn_up_exps.weight", NE, t2);
        w.down = read_experts(model, pre + "ffn_down_exps.weight", NE, t3);
        const std::string tag = "real blk" + std::to_string(layer) + " gate" + std::to_string(gt) + "/down" + std::to_string(dt);
        fast_case(tag + " T=2000", 2000, 1 | 4, w, NE, rng, tables);
        fast_case(tag + " T=100", 100, 0, w, NE, rng, tables);
    }
    expect(!seen.empty(), "real-weight layers found");
    std::printf("real-weight type pairs tested: %zu\n", seen.size());
}
// 8192 prompt rows, 512 experts (the model's routing spread: ~160 assignments / expert), exact vs FAST per kernel.
static void fast_bench(const am::Tables& tables) {
    std::mt19937 rng(7);
    for (const auto [gt, dt] : {std::pair{mo::type_q4_k, mo::type_q5_1}, std::pair{mo::type_q5_k, mo::type_q8_0}}) {
        const u32 NE = 512, T = 8192;
        HostWeights w = synthetic_weights(gt, dt, NE, rng);
        State s; init_state(s, T); synthetic_inputs(s, NE, 0, rng);
        s.view[V_inter].clear(); s.view[V_inter].shrink_to_fit(); s.view[V_inter].assign(16, 0);
        Device d; upload_weights(d, w); upload_state(d, s, false);
        auto& rv = d.rows_view;
        mo::enqueue_prologue(rv, T, mo::hidden, nullptr); mo::enqueue_csr(rv, T, nullptr); CK(cudaDeviceSynchronize());
        u32 tc = 0; CK(cudaMemcpy(&tc, d.tcount, 4, cudaMemcpyDeviceToHost));
        std::printf("bench gate %u down %u T=%u NE=%u: tiles %u (64-row tiles, fill %.1f%%)\n", gt, dt, T, NE, tc, 100.0 * T * 10 / (64.0 * tc));
        cudaEvent_t e[5]; for (auto& x : e) CK(cudaEventCreate(&x));
        float best[4] = {1e9F, 1e9F, 1e9F, 1e9F};
        for (int it = 0; it < 6; ++it) {
            CK(cudaEventRecord(e[0]));
            mo::enqueue_project(rv, T, d.weights, tables, nullptr, false); CK(cudaEventRecord(e[1]));
            mo::enqueue_project(rv, T, d.weights, tables, nullptr, true); CK(cudaEventRecord(e[2]));
            mo::enqueue_down(rv, T, d.weights, nullptr, false); CK(cudaEventRecord(e[3]));
            mo::enqueue_down(rv, T, d.weights, nullptr, true); CK(cudaEventRecord(e[4]));
            CK(cudaEventSynchronize(e[4]));
            for (int k = 0; k < 4; ++k) { float ms; cudaEventElapsedTime(&ms, e[k], e[k + 1]); if (it >= 2) best[k] = std::min(best[k], ms); }
        }
        std::printf("bench per layer: project exact %.3f ms fast %.3f ms (x%.2f) | down exact %.3f ms fast %.3f ms (x%.2f) | x40 layers: exact %.1f ms fast %.1f ms\n",
                    best[0], best[1], best[0] / best[1], best[2], best[3], best[2] / best[3], 40 * (best[0] + best[2]), 40 * (best[1] + best[3]));
        std::fflush(stdout);
        (void)std::system("nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader");
        free_device(d);
    }
}

// Dump of the FAST project + down outputs (all state views) for byte comparison between tile-variant builds.
static void fast_dump_case(FILE* out, u32 T, unsigned kind, const HostWeights& w, u32 NE, std::mt19937& rng, const am::Tables& tables) {
    State s; init_state(s, T); synthetic_inputs(s, NE, kind, rng);
    Device d; upload_weights(d, w); upload_state(d, s, true);
    auto& rv = d.rows_view;
    mo::enqueue_prologue(rv, T, mo::hidden, nullptr); mo::enqueue_csr(rv, T, nullptr);
    mo::enqueue_project(rv, T, d.weights, tables, nullptr, true);
    mo::enqueue_down(rv, T, d.weights, nullptr, true); CK(cudaDeviceSynchronize());
    for (const int v : {V_inter, V_dcodes, V_ddeltas, V_dsums, V_raw}) { const auto h = snap(d, s, v); std::fwrite(h.data(), 1, h.size(), out); }
    free_device(d);
}
static void fast_dump(const am::Tables& tables, const char* path) {
    FILE* out = std::fopen(path, "wb");
    if (!out) { std::printf("cannot open %s\n", path); std::exit(2); }
    std::mt19937 rng(4242);
    for (const auto [gt, dt] : {std::pair{mo::type_q4_k, mo::type_q5_1}, std::pair{mo::type_q5_k, mo::type_q8_0}}) {
        HostWeights w = synthetic_weights(gt, dt, 24, rng);
        fast_dump_case(out, 1000, 1 | 2 | 4, w, 24, rng, tables);
        fast_dump_case(out, 37, 0, w, 24, rng, tables);
    }
    strix::qwen_gguf::Model model;
    for (u32 layer = 0; layer < 96; ++layer) {
        const std::string pre = "blk." + std::to_string(layer) + ".";
        u32 gt = 0, dt = 0;
        try { gt = static_cast<u32>(model.tensor(pre + "ffn_gate_exps.weight").ggml_type); dt = static_cast<u32>(model.tensor(pre + "ffn_down_exps.weight").ggml_type); }
        catch (...) { continue; }
        if (!(gt == mo::type_q4_k || gt == mo::type_q5_k) || !(dt == mo::type_q5_1 || dt == mo::type_q8_0)) continue;
        HostWeights w; w.gate_type = gt; w.down_type = dt; w.experts_alloc = 48;
        u32 t1, t2, t3;
        w.gate = read_experts(model, pre + "ffn_gate_exps.weight", 48, t1); w.up = read_experts(model, pre + "ffn_up_exps.weight", 48, t2);
        w.down = read_experts(model, pre + "ffn_down_exps.weight", 48, t3);
        fast_dump_case(out, 2000, 1 | 4, w, 48, rng, tables);
        std::printf("dumped real blk%u gate%u down%u\n", layer, gt, dt);
        break;
    }
    std::fclose(out);
}
#ifdef STRIX_WFMT_CAPTURE
#include "qwen_moe_wfmt_capture.inc"
#endif
int main(int argc, char** argv) {
#ifdef STRIX_WFMT_CAPTURE
    if (argc == 5 && std::string_view(argv[1]) == "--wfmt-backend-capture" &&
        std::string_view(argv[3]) == "--tables") {
        CK(cudaSetDevice(0));
        am::Tables tables = am::load_tables(argv[4]);
        wfmt_capture(tables,argv[2]);
        am::free_tables(tables);
        return 0;
    }
#endif
    if (argc < 2) { std::printf("usage: qwen_moe_tests <tables dir> [bench|fixture]\n"); return 2; }
    const std::string mode = argc > 2 ? argv[2] : "";
    CK(cudaSetDevice(0));
    am::Tables tables = am::load_tables(argv[1]);
#ifdef STRIX_WFMT_CAPTURE
    if (mode.rfind("wfmt-capture=", 0) == 0) { wfmt_capture(tables,mode.c_str()+13); am::free_tables(tables); return 0; }
#endif
    if (mode == "fixture") { fixture_replay(tables); std::printf("%s: %d failures, %llu bytes compared\n", g_fail ? "FAIL" : "PASS", g_fail, g_checked); return g_fail ? 1 : 0; }
    if (mode == "fastmma" || mode == "fastmma-check") {
        fast_tests(tables);
        if (mode == "fastmma") fast_bench(tables);
        std::printf("%s: %d failures, %llu values compared\n", g_fail ? "FAIL" : "PASS", g_fail, g_checked);
        return g_fail ? 1 : 0;
    }
    if (mode == "fastmma-bench") { fast_bench(tables); return 0; }
    if (mode.rfind("fastdump=", 0) == 0) { fast_dump(tables, mode.c_str() + 9); return 0; }
    if (mode == "bench") {
        // T = 8192, 128 synthetic experts (VRAM < 2 GB), uniform routes, intermediates off.
        std::mt19937 rng(7);
        for (const auto [gt, dt] : {std::pair{mo::type_q4_k, mo::type_q5_1}, std::pair{mo::type_q5_k, mo::type_q8_0}}) {
            const u32 NE = 128, T = 8192;
            HostWeights w = synthetic_weights(gt, dt, NE, rng);
            State s; init_state(s, T); synthetic_inputs(s, NE, 0, rng);
            s.view[V_inter].clear(); s.view[V_inter].shrink_to_fit(); s.view[V_inter].assign(16, 0);
            Device d; upload_weights(d, w); upload_state(d, s, false);
            cudaEvent_t e0, e1, e2, e3, e4, e5; for (auto* e : {&e0, &e1, &e2, &e3, &e4, &e5}) CK(cudaEventCreate(e));
            for (int it = 0; it < 4; ++it) {
                CK(cudaEventRecord(e0));
                mo::enqueue_prologue(d.rows_view, T, mo::hidden, nullptr); mo::enqueue_csr(d.rows_view, T, nullptr); CK(cudaEventRecord(e1));
                mo::enqueue_project(d.rows_view, T, d.weights, tables, nullptr); CK(cudaEventRecord(e2));
                mo::enqueue_down(d.rows_view, T, d.weights, nullptr); CK(cudaEventRecord(e3));
                mo::enqueue_merge(d.rows_view, T, nullptr); mo::enqueue_commit(d.rows_view, T, d.record, nullptr); CK(cudaEventRecord(e4));
                CK(cudaEventSynchronize(e4));
                float a, b, c, dd, all; cudaEventElapsedTime(&a, e0, e1); cudaEventElapsedTime(&b, e1, e2); cudaEventElapsedTime(&c, e2, e3);
                cudaEventElapsedTime(&dd, e3, e4); cudaEventElapsedTime(&all, e0, e4);
                u32 s0; CK(cudaMemcpy(&s0, d.summary, 4, cudaMemcpyDeviceToHost));
                std::printf("bench gate %u down %u T=%u: prologue+csr %.3f ms, gate/up %.3f ms, down %.3f ms, merge+commit %.3f ms, total %.3f ms (summary0 %u)\n",
                            gt, dt, T, a, b, c, dd, all, s0);
            }
            free_device(d);
        }
        return 0;
    }
    HostExp hx(argv[1]);
    if (mode == "route-bf16") {
        route_rounding_device_test();
        std::mt19937 candidate_rng(2206);
        const int selected_pair = argc > 3 ? std::stoi(argv[3]) : -1;
        if (selected_pair < -1 || selected_pair > 3) throw std::invalid_argument("route BF16 pair index 0..3");
        int pair_index = 0;
        for (const auto [gt, dt] : {std::pair{mo::type_q4_k, mo::type_q5_1},
                std::pair{mo::type_q4_k, mo::type_q8_0}, std::pair{mo::type_q5_k, mo::type_q5_1},
                std::pair{mo::type_q5_k, mo::type_q8_0}}) {
            const int current_pair = pair_index++;
            if (selected_pair >= 0 && current_pair != selected_pair) continue;
            HostWeights w = synthetic_weights(gt, dt, 24, candidate_rng);
            Device d; upload_weights(d, w);
            // All window sizes, transition to tiled, CSR tails and a multi-tile expert.
            for (const u32 T : {1U,2U,3U,4U,5U,6U,7U,8U,9U,15U,16U,17U,40U,56U,65U}) {
                for (const bool bf16 : {false, true}) {
                    State s; init_state(s, T, bf16); synthetic_inputs(s, 24, 7, candidate_rng);
                    run_pipeline(std::string(bf16 ? "BF16" : "OFF") + " g" + std::to_string(gt) +
                        " d" + std::to_string(dt), s, w, d, tables, hx, true);
                    free_state_buffers(d);
                }
            }
            for (const unsigned kind : {0U, 1U, 2U, 3U, 4U}) {
                State s; init_state(s, 16, true); synthetic_inputs(s, 24, 3, candidate_rng);
                if (kind == 0U) s.at<u16>(V_input, 5)[77] = 0x7fc1U;
                if (kind == 1U) s.at<u16>(V_input, 5)[77] = 0xff80U;
                if (kind == 2U) s.at<u32>(V_routes, 5)[4] = 600U;
                if (kind == 3U) s.at<u32>(V_rw, 5)[4] = 0x7fc00000U;
                if (kind == 4U) for (u32 i=0;i<mo::hidden;++i)
                    s.at<u16>(V_input,5)[i]=bf16_of(i&1U ? -3e38F : 3e38F);
                run_pipeline("BF16 guard " + std::to_string(kind), s, w, d, tables, hx, false);
                free_state_buffers(d);
            }
            free_device(d);
        }
        std::printf("ROUTE BF16 OPERATOR %s: %d failures, %llu bytes compared\n",
                    g_fail ? "FAIL" : "PASS", g_fail, g_checked);
        am::free_tables(tables);
        return g_fail ? 1 : 0;
    }
    std::mt19937 rng(12345);
    // 1. Synthetic weights, the four (gate, down) type combinations of the model, skew + duplicates + special rows.
    const u32 NE = 24;
    for (const auto [gt, dt] : {std::pair{mo::type_q4_k, mo::type_q5_1}, std::pair{mo::type_q4_k, mo::type_q8_0},
                                 std::pair{mo::type_q5_k, mo::type_q8_0}, std::pair{mo::type_q5_k, mo::type_q5_1}}) {
        HostWeights w = synthetic_weights(gt, dt, NE, rng);
        Device d; upload_weights(d, w);
        for (const auto [T, kind] : {std::pair{16U, 7U}, std::pair{40U, 3U}, std::pair{56U, 7U}}) {
            State s; init_state(s, T); synthetic_inputs(s, NE, kind, rng);
            run_pipeline("synthetic g" + std::to_string(gt) + " d" + std::to_string(dt) + " T" + std::to_string(T) + " k" + std::to_string(kind),
                         s, w, d, tables, hx, true);
            free_state_buffers(d);
        }
        // 2. Fail-closed edges (on the Q4K/Q5_1 and Q5K/Q8_0 combinations).
        if ((gt == mo::type_q4_k && dt == mo::type_q5_1) || (gt == mo::type_q5_k && dt == mo::type_q8_0)) {
            const std::string tag = " g" + std::to_string(gt) + " d" + std::to_string(dt);
            auto edge = [&](const std::string& name, auto&& edit, bool inter) {
                State s; init_state(s, 32); synthetic_inputs(s, NE, 3, rng); edit(s);
                run_pipeline(name + tag, s, w, d, tables, hx, inter);
                free_state_buffers(d);
            };
            edge("edge NaN input row 5", [](State& s) { s.at<u16>(V_input, 5)[77] = 0x7fc1; }, true);
            edge("edge Inf input row 9", [](State& s) { s.at<u16>(V_input, 9)[2559] = 0xff80; }, true);
            edge("edge invalid route row 3", [](State& s) { s.at<u32>(V_routes, 3)[4] = 600; }, true);
            edge("edge NaN route weight row 30", [](State& s) { s.at<u32>(V_rw, 30)[9] = 0x7fc00000U; }, true);
            edge("edge projection overflow row 7", [](State& s) { for (u32 i = 0; i < 2560; ++i) s.at<u16>(V_input, 7)[i] = bf16_of(i & 1 ? -3e38F : 3e38F); }, true);
            edge("edge clean 32 rows, no intermediates", [](State&) {}, false);
        }
        free_device(d);
    }
    // 3. merge/commit overflow rule (|sum| >= 0x7f7f8000): crafted raw dots, standalone merge + commit.
    {
        const u32 T = 16;
        State s; init_state(s, T);
        std::mt19937 r2(5); std::uniform_real_distribution<float> uni(-1.0F, 1.0F);
        for (u32 r = 0; r < T; ++r) {
            *s.at<u32>(V_status, r) = 0;
            for (u32 k = 0; k < 10; ++k) { s.at<u32>(V_routes, r)[k] = k; s.at<float>(V_rw, r)[k] = uni(r2); }
            for (u32 i = 0; i < 10 * 2560; ++i) s.at<float>(V_raw, r)[i] = uni(r2) * 1e3F;
        }
        for (u32 i = 0; i < 32; ++i) s.csr.summary[i] = 0;
        // Row 11: slot sums land at/above the BF16 overflow threshold 0x7f7f8000 in one element, just below in another.
        for (u32 k = 0; k < 10; ++k) { s.at<float>(V_rw, 11)[k] = k == 0 ? 1.0F : 0.0F; }
        s.at<float>(V_raw, 11)[0 * 2560 + 100] = f_of(0x7f7f8000U);
        s.at<float>(V_raw, 11)[0 * 2560 + 101] = f_of(0x7f7f7fffU);
        for (int pass = 0; pass < 2; ++pass) {
            State t = s;
            if (pass == 1) t.at<float>(V_raw, 11)[100] = f_of(0x7f7f7fffU);  // no overflow: full commit
            Device d; upload_state(d, t, false);
            mo::enqueue_merge(d.rows_view, T, nullptr); mo::enqueue_commit(d.rows_view, T, d.record, nullptr);
            CK(cudaDeviceSynchronize());
            u32 record[2] = {0, 0}; ref_merge_commit(t, record);
            // After an overflow the HIP merge's CTAs of later rows race on header[0] (skip or not): the accumulation (a temporary,
            // never committed on failure) is then not deterministic in HIP either, so it is excluded in that pass.
            compare(pass ? "merge/commit near-threshold" : "merge/commit overflow", d, t, false, pass ? -1 : V_acc);
            u32 grec[2]; CK(cudaMemcpy(grec, d.record, 8, cudaMemcpyDeviceToHost));
            expect(grec[0] == record[0] && grec[1] == record[1], "merge record");
            std::printf("merge/commit pass %d: header0=%u record={%u,%u}\n", pass, t.csr.summary[0], record[0], record[1]);
            free_device(d);
        }
    }
    // 4. Real expert weights: blk.0 (Q4_K gate/up, Q5_1 down) and blk.2 (Q5_K, Q8_0), experts 0..63.
    {
        bool have = true;
        try { strix::qwen_gguf::Model probe; (void)probe; } catch (const std::exception& e) { std::printf("GGUF unavailable (%s): real-weight tests skipped\n", e.what()); have = false; }
        if (have) {
            strix::qwen_gguf::Model model;
            for (const u32 layer : {0U, 2U, 4U}) {
                const u32 NR = 64;
                HostWeights w{};
                u32 t1, t2, t3;
                const std::string pre = "blk." + std::to_string(layer) + ".";
                w.gate = read_experts(model, pre + "ffn_gate_exps.weight", NR, t1);
                w.up = read_experts(model, pre + "ffn_up_exps.weight", NR, t2);
                w.down = read_experts(model, pre + "ffn_down_exps.weight", NR, t3);
                w.gate_type = t1; w.down_type = t3; w.experts_alloc = NR;
                expect(t1 == t2, "gate/up types");
                Device d; upload_weights(d, w);
                State s; init_state(s, 32); synthetic_inputs(s, NR, 3, rng);
                run_pipeline("real blk." + std::to_string(layer) + " gate " + std::to_string(t1) + " down " + std::to_string(t3) + " T32", s, w, d, tables, hx, true);
                free_device(d);
            }
        }
    }
    fixture_replay(tables);
    std::printf("%s: %d failures, %llu bytes compared\n", g_fail ? "FAIL" : "PASS", g_fail, g_checked);
    am::free_tables(tables);
    return g_fail ? 1 : 0;
}
