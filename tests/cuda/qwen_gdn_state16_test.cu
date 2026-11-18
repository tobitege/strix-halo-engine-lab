// CUDA GDN recurrence with a 16-bit stored recurrent state (STRIX_RECURRENT_STATE=f16|bf16), src/cuda/models/qwen/qwen_gdn.cu.
//   qwen_gdn_state16_test [--tables DIR]
// (a) device Codec<f16> / Codec<bf16> store, load and round_in_register against the host integer functions, bit for bit;
// (b) f16 / bf16: 8 rows in ONE round-each-row launch == 8 single-row launches (core and state bits), == split(window) +
//     commit replay of the kept rows, == a host reference with the rounding points of the contract; prefill mode (narrow
//     once at the store) == its host reference; every lane mapping G gives the same bits;
// (c) f32 launches (default Tuning and an explicit f32 Tuning) == the host reference of the incumbent arithmetic, bitwise.
// Real dims, synthetic inputs. The math tables only feed expf(gate) (identical on every arm); a missing table dir falls back
// to zeroed tables, which keeps every comparison valid.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#if defined(_MSC_VER)
#include <crtdbg.h>
#endif

#include "strix/cuda/qwen_gdn.hpp"
#include "strix/cuda/amd_math_tables.hpp"
#include "../../src/attn_core/recurrent_state.hpp"

namespace am = strix::cuda::amdmath;
namespace g = strix::cuda::qwen_gdn;
namespace rs = strix::attn_core::rstate;
using strix::RecurrentStateType;
using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

static int g_pass = 0, g_fail = 0;
static am::Tables g_tables;
static void verdict(bool ok, const std::string& what, const std::string& detail = {}) {
    if (ok) { ++g_pass; std::printf("ok   %s\n", what.c_str()); }
    else { ++g_fail; std::printf("FAIL %s %s\n", what.c_str(), detail.c_str()); }
}

constexpr u32 CW = g::kConvWidth, VW = g::kValueWidth, VH = g::kValueHeads, KH = g::kKeyHeads, D = g::kHeadDim;
constexpr u64 kStride = CW + VW + VH + VH + VW;
constexpr u64 kOffZ = CW, kOffBeta = CW + VW, kOffGate = kOffBeta + VH, kOffCore = kOffGate + VH;

struct Rng {
    u64 s;
    explicit Rng(u64 seed) : s(seed * 0x9e3779b97f4a7c15ULL + 1) {}
    u64 next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    float uni() { return static_cast<float>((next() >> 40) * (1.0 / 16777216.0)); }
    float normal() {
        const float u1 = std::max(uni(), 1e-7F), u2 = uni();
        return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
    }
};

// ------------------------------------------------------------------------------------------------ (a) codec
template <RecurrentStateType T> __global__ void codec_kernel(const u32* bits, u32 n, u16* stored, u32* rounded) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = rs::f32_from_bits(bits[i]);
    stored[i] = rs::Codec<T>::store(x);
    rounded[i] = rs::f32_bits(rs::round_in_register<T>(x));
}
template <RecurrentStateType T> __global__ void load_kernel(u32* out) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 65536U) out[i] = rs::f32_bits(rs::Codec<T>::load(static_cast<u16>(i)));
}
static bool is_nan_bits(u32 b) { return (b & 0x7fffffffU) > 0x7f800000U; }

template <RecurrentStateType T> static void test_codec(const char* name, const std::vector<u32>& pats) {
    const u32 n = static_cast<u32>(pats.size());
    u32 *dbits = nullptr, *drounded = nullptr, *dload = nullptr; u16* dstored = nullptr;
    CK(cudaMalloc(&dbits, n * 4ULL)); CK(cudaMalloc(&drounded, n * 4ULL)); CK(cudaMalloc(&dstored, n * 2ULL));
    CK(cudaMalloc(&dload, 65536 * 4ULL));
    CK(cudaMemcpy(dbits, pats.data(), n * 4ULL, cudaMemcpyHostToDevice));
    codec_kernel<T><<<(n + 255) / 256, 256>>>(dbits, n, dstored, drounded);
    load_kernel<T><<<256, 256>>>(dload);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<u16> stored(n); std::vector<u32> rounded(n), loaded(65536);
    CK(cudaMemcpy(stored.data(), dstored, n * 2ULL, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(rounded.data(), drounded, n * 4ULL, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(loaded.data(), dload, 65536 * 4ULL, cudaMemcpyDeviceToHost));
    u64 bad_store = 0, bad_round = 0, bad_load = 0; u32 first_store = 0, first_round = 0;
    for (u32 i = 0; i < n; ++i) {
        const float x = rs::f32_from_bits(pats[i]);
        const u16 want = T == RecurrentStateType::f16 ? rs::f16_from_f32_host(x) : rs::bf16_from_f32(x);
        if (stored[i] != want) { if (!bad_store++) first_store = pats[i]; }
        const u32 wr = rs::f32_bits(rs::round_trip(T, x));
        if (rounded[i] != wr && !(is_nan_bits(rounded[i]) && is_nan_bits(wr))) { if (!bad_round++) first_round = pats[i]; }
    }
    for (u32 h = 0; h < 65536U; ++h) {
        const u32 want = rs::f32_bits(T == RecurrentStateType::f16 ? rs::f32_from_f16_host(static_cast<u16>(h)) :
                                                                      rs::f32_from_bf16(static_cast<u16>(h)));
        if (loaded[h] != want && !(is_nan_bits(loaded[h]) && is_nan_bits(want))) ++bad_load;
    }
    char d[160];
    std::snprintf(d, sizeof(d), "(store %llu bad first %08x, round %llu bad first %08x, load %llu bad)",
        static_cast<unsigned long long>(bad_store), first_store, static_cast<unsigned long long>(bad_round), first_round,
        static_cast<unsigned long long>(bad_load));
    verdict(!bad_store && !bad_round && !bad_load,
        std::string("codec ") + name + " device == host over " + std::to_string(n) + " patterns + 65536 loads", d);
    CK(cudaFree(dbits)); CK(cudaFree(drounded)); CK(cudaFree(dstored)); CK(cudaFree(dload));
}

// ------------------------------------------------------------------------------------------------ recurrence fixtures
__global__ void decay_oracle(const float* gate, float* out, u32 n, am::Tables t) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = am::amd_expf(t, gate[i]);
}
static std::vector<float> expf_dev(const std::vector<float>& gate) {
    float *dg = nullptr, *dy = nullptr; const u32 n = static_cast<u32>(gate.size());
    CK(cudaMalloc(&dg, n * 4ULL)); CK(cudaMalloc(&dy, n * 4ULL));
    CK(cudaMemcpy(dg, gate.data(), n * 4ULL, cudaMemcpyHostToDevice));
    decay_oracle<<<(n + 255) / 256, 256>>>(dg, dy, n, g_tables);
    CK(cudaGetLastError());
    std::vector<float> y(n); CK(cudaMemcpy(y.data(), dy, n * 4ULL, cudaMemcpyDeviceToHost));
    CK(cudaFree(dg)); CK(cudaFree(dy));
    return y;
}

struct Inputs { std::vector<float> qkv, beta, gate, state; u32 count = 0; };
static Inputs make_inputs(u32 count, u64 seed) {
    Inputs in; in.count = count; Rng r(seed);
    in.qkv.assign(static_cast<size_t>(count) * CW, 0.0F);
    for (u32 row = 0; row < count; ++row)
        for (u32 c = 0; c < CW; ++c) {
            const float scale = c < 2U * g::kKeyWidth ? 0.09F : 1.0F;  // q / k ~ unit vectors, v ~ N(0,1)
            in.qkv[static_cast<u64>(row) * CW + c] = r.normal() * scale;
        }
    in.beta.resize(static_cast<size_t>(count) * VH); in.gate.resize(in.beta.size());
    for (auto& b : in.beta) b = 0.2F + 0.7F * r.uni();
    for (auto& x : in.gate) x = -(0.02F + 0.4F * r.uni());
    in.state.resize(g::kMatrixFloats);
    for (auto& x : in.state) x = r.normal() * 0.3F;
    return in;
}

struct Workspace {
    float* base = nullptr; u32 count = 0;
    explicit Workspace(u32 n) : count(n) { CK(cudaMalloc(&base, std::max<u64>(n, 1) * kStride * 4)); CK(cudaMemset(base, 0, std::max<u64>(n, 1) * kStride * 4)); }
    ~Workspace() { cudaFree(base); }
    g::Rows rows(u32 first = 0) const {
        float* b = base + static_cast<u64>(first) * kStride;
        return {b, b + kOffZ, b + kOffBeta, b + kOffGate, b + kOffCore, kStride};
    }
    void put(u64 off, u32 width, const std::vector<float>& v) const {
        CK(cudaMemcpy2D(base + off, kStride * 4, v.data(), width * 4ULL, width * 4ULL, count, cudaMemcpyHostToDevice));
    }
    std::vector<float> get(u64 off, u32 width) const {
        std::vector<float> v(static_cast<size_t>(count) * width);
        CK(cudaMemcpy2D(v.data(), width * 4ULL, base + off, kStride * 4, width * 4ULL, count, cudaMemcpyDeviceToHost));
        return v;
    }
};

// state <-> bytes of the stored type
static std::vector<u8> encode(const std::vector<float>& s, RecurrentStateType t) {
    std::vector<u8> b(strix::recurrent_state_bytes(s.size(), t));
    for (size_t i = 0; i < s.size(); ++i) {
        if (t == RecurrentStateType::f32) std::memcpy(&b[i * 4], &s[i], 4);
        else { const u16 h = t == RecurrentStateType::f16 ? rs::f16_from_f32_host(s[i]) : rs::bf16_from_f32(s[i]); std::memcpy(&b[i * 2], &h, 2); }
    }
    return b;
}
static u8* upload(const std::vector<u8>& b) { u8* d = nullptr; CK(cudaMalloc(&d, b.size())); CK(cudaMemcpy(d, b.data(), b.size(), cudaMemcpyHostToDevice)); return d; }
static std::vector<u8> download(const u8* d, size_t n) { std::vector<u8> b(n); CK(cudaMemcpy(b.data(), d, n, cudaMemcpyDeviceToHost)); return b; }

// Host reference: the incumbent arithmetic of the kernel (G = 4 tree), optional state narrowing after every row.
static float tree4(const float* p128) {
    float lanes[4];
    for (int sub = 0; sub < 4; ++sub) {
        float r[32];
        for (int j = 0; j < 32; ++j) r[j] = p128[sub + 4 * j];
        for (int half = 16; half; half /= 2)
            for (int j = 0; j < half; ++j) r[j] = r[j] + r[j + half];
        lanes[sub] = r[0];
    }
    for (int m = 2; m; m /= 2) {
        float n[4];
        for (int sub = 0; sub < 4; ++sub) n[sub] = lanes[sub] + lanes[sub ^ m];
        std::memcpy(lanes, n, sizeof(n));
    }
    return lanes[0];
}
static void ref_rec(const Inputs& in, u32 first, u32 count, std::vector<float>& state, std::vector<float>& core, RecurrentStateType t,
    bool per_row, const std::vector<float>& decay) {
    core.assign(static_cast<size_t>(count) * VW, 0.0F);
    std::vector<float> S(D), p(D);
    for (u32 vh = 0; vh < VH; ++vh)
        for (u32 col = 0; col < D; ++col) {
            for (u32 k = 0; k < D; ++k) S[k] = state[(static_cast<u64>(vh) * D + k) * D + col];
            const u32 kh = vh % KH;
            for (u32 row = 0; row < count; ++row) {
                const u32 gr = first + row;
                const float* r = in.qkv.data() + static_cast<u64>(gr) * CW;
                const float* key = r + KH * D + kh * D;
                const float* query = r + kh * D;
                const float v = r[2 * KH * D + vh * D + col];
                const float dcy = decay[gr * VH + vh], b = in.beta[gr * VH + vh];
                for (u32 k = 0; k < D; ++k) { S[k] = S[k] * dcy; p[k] = S[k] * key[k]; }
                const float pred = tree4(p.data());
                const float delta = (v - pred) * b;
                for (u32 k = 0; k < D; ++k) { S[k] = std::fmaf(key[k], delta, S[k]); p[k] = S[k] * query[k]; }
                core[static_cast<u64>(row) * VW + vh * D + col] = tree4(p.data());
                if (per_row) for (u32 k = 0; k < D; ++k) S[k] = rs::round_trip(t, S[k]);
            }
            for (u32 k = 0; k < D; ++k) state[(static_cast<u64>(vh) * D + k) * D + col] = rs::round_trip(t, S[k]);  // the store
        }
}

struct Run { std::vector<float> core; std::vector<u8> state; };
static bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}
static std::string label(const char* what, RecurrentStateType t, const char* extra = "") {
    return std::string(what) + " " + strix::recurrent_state_type_name(t) + extra;
}

// One launch over rows [first, first+n) of ws: state_in -> state_out (device byte buffers of type t).
static void launch(const Workspace& ws, u32 first, u32 n, u8* in, u8* out, void* scratch, RecurrentStateType t, bool round, u32 lanes = 4) {
    g::Tuning tuning{lanes, t, round};
    g::launch_recurrence_split(ws.rows(first), n, reinterpret_cast<const float*>(in), reinterpret_cast<float*>(out), scratch, g_tables,
        nullptr, tuning);
    CK(cudaGetLastError());
}

static void test_type(RecurrentStateType t, const Inputs& in, const std::vector<float>& decay) {
    const u32 T = in.count;
    const size_t bytes = strix::recurrent_state_bytes(g::kMatrixFloats, t);
    Workspace ws(T);
    ws.put(0, CW, in.qkv); ws.put(kOffBeta, VH, in.beta); ws.put(kOffGate, VH, in.gate);
    void* scratch = nullptr; CK(cudaMalloc(&scratch, g::decay_scratch_bytes(T)));
    const auto s0 = in.state;
    // the contract's starting point is a STORED state: narrow the host copy first
    std::vector<float> start = s0;
    for (auto& x : start) x = rs::round_trip(t, x);
    const auto start_bytes = encode(start, t);

    // host references
    std::vector<float> ref_state = start, ref_core;
    ref_rec(in, 0, T, ref_state, ref_core, t, true, decay);
    const auto ref_bytes = encode(ref_state, t);

    // one launch of T rows, round each row
    u8 *a = upload(start_bytes), *b = upload(start_bytes);
    launch(ws, 0, T, a, b, scratch, t, true);
    CK(cudaDeviceSynchronize());
    Run one{ws.get(kOffCore, VW), download(b, bytes)};
    verdict(same_bits(one.core, ref_core) && one.state == ref_bytes, label("window launch RoundEachRow == host reference", t));

    // T single-row launches, in place
    {
        Workspace ws2(T);
        ws2.put(0, CW, in.qkv); ws2.put(kOffBeta, VH, in.beta); ws2.put(kOffGate, VH, in.gate);
        u8* c = upload(start_bytes);
        for (u32 r = 0; r < T; ++r) launch(ws2, r, 1, c, c, scratch, t, true);
        CK(cudaDeviceSynchronize());
        Run serial{ws2.get(kOffCore, VW), download(c, bytes)};
        verdict(same_bits(serial.core, one.core) && serial.state == one.state, label("8 rows RoundEachRow == 8 single-row launches", t));
        CK(cudaFree(c));
    }
    // split window (S0 kept) + commit replay of keep rows == serial of keep rows; full window == serial of T rows
    for (u32 keep : {1U, 5U, T}) {
        u8* s0d = upload(start_bytes); u8* win = upload(start_bytes); u8* other = upload(start_bytes);
        launch(ws, 0, T, s0d, win, scratch, t, true);                  // window: S0 stays in s0d
        launch(ws, 0, keep, s0d, other, scratch, t, true);             // commit replay from S0
        CK(cudaDeviceSynchronize());
        std::vector<float> st = start, core_k;
        ref_rec(in, 0, keep, st, core_k, t, true, decay);
        const bool s0_intact = download(s0d, bytes) == start_bytes;
        const bool replay_ok = download(other, bytes) == encode(st, t);
        const bool full_ok = download(win, bytes) == one.state;
        verdict(s0_intact && replay_ok && full_ok, label("split window + commit replay == serial", t, (" keep=" + std::to_string(keep)).c_str()),
            std::string("s0=") + (s0_intact ? "1" : "0") + " replay=" + (replay_ok ? "1" : "0") + " full=" + (full_ok ? "1" : "0"));
        CK(cudaFree(s0d)); CK(cudaFree(win)); CK(cudaFree(other));
    }
    // prefill mode: FP32 inside the call, narrowed once at the store
    {
        std::vector<float> st = start, core_p;
        ref_rec(in, 0, T, st, core_p, t, false, decay);
        u8 *pi = upload(start_bytes), *po = upload(start_bytes);
        launch(ws, 0, T, pi, po, scratch, t, false);
        CK(cudaDeviceSynchronize());
        verdict(same_bits(ws.get(kOffCore, VW), core_p) && download(po, bytes) == encode(st, t),
            label("prefill launch (narrow at store) == host reference", t));
        CK(cudaFree(pi)); CK(cudaFree(po));
    }
    // lane mapping independence
    for (u32 lanes : {2U, 8U, 16U}) {
        u8* li = upload(start_bytes); u8* lo = upload(start_bytes);
        launch(ws, 0, T, li, lo, scratch, t, true, lanes);
        CK(cudaDeviceSynchronize());
        verdict(same_bits(ws.get(kOffCore, VW), one.core) && download(lo, bytes) == one.state,
            label("RoundEachRow", t, (" G=" + std::to_string(lanes) + " == G=4").c_str()));
        CK(cudaFree(li)); CK(cudaFree(lo));
    }
    CK(cudaFree(a)); CK(cudaFree(b)); CK(cudaFree(scratch));
}

// (c) f32 incumbent
static void test_f32(const Inputs& in, const std::vector<float>& decay) {
    const u32 T = in.count;
    Workspace ws(T);
    ws.put(0, CW, in.qkv); ws.put(kOffBeta, VH, in.beta); ws.put(kOffGate, VH, in.gate);
    void* scratch = nullptr; CK(cudaMalloc(&scratch, g::decay_scratch_bytes(T)));
    std::vector<float> st = in.state, core;
    ref_rec(in, 0, T, st, core, RecurrentStateType::f32, false, decay);
    const auto want = encode(st, RecurrentStateType::f32);
    // default Tuning (what every existing caller passes), in place
    {
        u8* d = upload(encode(in.state, RecurrentStateType::f32));
        g::launch_recurrence(ws.rows(), T, reinterpret_cast<float*>(d), scratch, g_tables, nullptr);
        CK(cudaDeviceSynchronize());
        verdict(same_bits(ws.get(kOffCore, VW), core) && download(d, want.size()) == want, "f32 launch_recurrence (default Tuning) == incumbent host reference");
        CK(cudaFree(d));
    }
    // explicit f32 Tuning with and without round_each_row: same code, same bits; lanes 2/8/16 too
    for (bool round : {false, true})
        for (u32 lanes : {2U, 4U, 8U, 16U}) {
            u8 *i = upload(encode(in.state, RecurrentStateType::f32)), *o = upload(encode(in.state, RecurrentStateType::f32));
            launch(ws, 0, T, i, o, scratch, RecurrentStateType::f32, round, lanes);
            CK(cudaDeviceSynchronize());
            verdict(same_bits(ws.get(kOffCore, VW), core) && download(o, want.size()) == want,
                std::string("f32 split launch G=") + std::to_string(lanes) + (round ? " round_each_row" : "") + " == incumbent host reference");
            CK(cudaFree(i)); CK(cudaFree(o));
        }
    CK(cudaFree(scratch));
}

static am::Tables zero_tables() {
    am::Tables t{};
    const auto zalloc = [](size_t bytes) { void* p = nullptr; CK(cudaMalloc(&p, bytes)); CK(cudaMemset(p, 0, bytes)); return p; };
    t.exp_bits = static_cast<const u32*>(zalloc(2ULL * 25 * (1ULL << 23) / 8));
    t.rsq_bits = static_cast<const u32*>(zalloc((1ULL << 24) / 8));
    t.rcp_bits = static_cast<const u32*>(zalloc((1ULL << 23) / 8));
    t.exp2_tab = static_cast<const float2*>(zalloc(129 * sizeof(float2)));
    return t;
}

int main(int argc, char** argv) {
#if defined(_MSC_VER)
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, 0); _CrtSetReportMode(_CRT_ERROR, 0);
#endif
    std::string tables = std::getenv("STRIX_QWEN_TABLES") ? std::getenv("STRIX_QWEN_TABLES") : "qrtx-build\\tables";
    for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == "--tables") tables = argv[++i];
    try { g_tables = am::load_tables(tables); std::printf("math tables: %s\n", tables.c_str()); }
    catch (const std::exception& e) { std::printf("math tables unavailable (%s): zeroed tables, comparisons stay valid\n", e.what()); g_tables = zero_tables(); }

    // (a) codec patterns: random 32-bit patterns, every high half with the rounding-tie / boundary low halves, specials
    std::vector<u32> pats;
    Rng r(7);
    for (u32 i = 0; i < (1U << 20); ++i) pats.push_back(static_cast<u32>(r.next() >> 32));
    for (u32 hi = 0; hi < 65536U; ++hi)
        for (u32 lo : {0x0000U, 0x7fffU, 0x8000U, 0x8001U, 0xffffU, 0x0fffU, 0x1000U, 0x1001U, 0x1fffU, 0x2000U, 0x3000U}) pats.push_back(hi << 16 | lo);
    for (u32 s : {0x00000000U, 0x80000000U, 0x7f800000U, 0xff800000U, 0x7fc00000U, 0xffc00000U, 0x7f800001U, 0xff8fffffU, 0x7fffffffU,
                  0x477fefffU, 0x477ff000U, 0x33000000U, 0x33000001U, 0x387fffffU, 0x38800000U, 0x38000000U, 0x7f7fffffU, 0x00000001U})
        pats.push_back(s);
    test_codec<RecurrentStateType::f16>("f16", pats);
    test_codec<RecurrentStateType::bf16>("bf16", pats);

    const auto in = make_inputs(8, 11);
    const auto decay = expf_dev(in.gate);
    test_type(RecurrentStateType::f16, in, decay);
    test_type(RecurrentStateType::bf16, in, decay);
    test_f32(in, decay);
    std::printf("qwen_gdn_state16_test: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
