// Host routed-expert executor: bitwise tests (Executor at 1/3/16 threads, scalar and SIMD tables) against reference_run, and
// reference_run against an independent oracle ported verbatim from tests/cuda/qwen_moe_tests.cu (HostExp, ref_gate_up_value,
// ref_down and the h quantizer of ref_project), which is the arithmetic accepted bit-exact against the GPU.
// Tables: STRIX_EXP_TABLES=<dir with tbl_exp_pm.bin, hard_v_exp_f32.bin> uses the real gfx1151 probe tables; otherwise
// synthetic tables of the same layout are generated in the temp directory (same code path, not the real bits).
#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS 1
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "cpu/core/routed_expert.hpp"

namespace ex = strix::cpu::expert;
using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using i8 = std::int8_t;
using i32 = std::int32_t;

static int g_fail = 0;
static unsigned g_cases = 0;
static void expect(bool ok, const std::string& what) {
    if (!ok) { ++g_fail; std::printf("FAIL %s\n", what.c_str()); }
}

// ------------------------------------------------------------------------------------------- oracle (verbatim port)
static float f_of(u32 b) { float f; std::memcpy(&f, &b, 4); return f; }
static u32 b_of(float f) { u32 b; std::memcpy(&b, &f, 4); return b; }
static float half_to_float(u16 h) {
    const u32 s = (h >> 15) & 1U, e = (h >> 10) & 31U, m = h & 1023U;
    if (e == 0) { const float v = std::ldexp(static_cast<float>(m), -24); return s ? -v : v; }
    if (e == 31) return f_of((s << 31) | 0x7f800000U | (m << 13));
    return f_of((s << 31) | ((e + 112U) << 23) | (m << 13));
}
static bool finite(float x) { return (b_of(x) & 0x7f800000U) != 0x7f800000U; }

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
        std::fseek(f, 0, SEEK_END); const long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
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
        if (index >> 5 < bits.size() && ((bits[index >> 5] >> (index & 31U)) & 1U)) {
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

static unsigned factor(const u8* hd, unsigned group, bool minimum) {
    u32 y, z, w; std::memcpy(&y, hd + 4, 4); std::memcpy(&z, hd + 8, 4); std::memcpy(&w, hd + 12, 4);
    const unsigned shift = 8U * (group & 3U);
    const unsigned a = (y >> shift) & 255U, b = (z >> shift) & 255U, c = (w >> shift) & 255U;
    return group < 4U ? (minimum ? b : a) & 63U : minimum ? (c >> 4U) | ((b >> 6U) << 4U) : (c & 15U) | ((a >> 6U) << 4U);
}

// ref_gate_up_value; the 10 super-blocks became `blocks`.
static float ref_gate_up_value(const u8* wrow, u32 blocks, bool q5k, const i8* codes, const float* deltas, const int* sums) {
    const u32 WB = q5k ? 176 : 144, QS = q5k ? 48 : 16;
    float acc[4] = {0, 0, 0, 0};
    for (u32 b = 0; b < blocks; ++b) {
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

// ref_down for one (codes, deltas, sums, output row); 20 groups became `groups`.
static float ref_down_value(const u8* row, u32 groups, bool q51, const i8* codes, const float* deltas, const int* sums) {
    const u32 WB = q51 ? 24 : 34;
    float acc[4] = {0, 0, 0, 0};
    for (u32 group = 0; group < groups; ++group) {
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
    return s01 + s23;
}

// ref_project's quad epilogue (SwiGLU + quantizer) for 32 values; false = non-finite.
static bool ref_quad(const HostExp& hx, const float* g, const float* u, i8* codes, float* delta, int* sum) {
    float hv[32]; bool bad = false;
    for (u32 lane = 0; lane < 32; ++lane) {
        hv[lane] = (g[lane] / (1.0F + hx.expf_(-g[lane]))) * u[lane];
        bad |= !finite(g[lane]) || !finite(u[lane]) || !finite(hv[lane]);
    }
    if (bad) return false;
    float x[32];
    for (u32 l = 0; l < 32; ++l) x[l] = std::fabs(hv[l]);
    for (int stride = 16; stride != 0; stride /= 2) { float n[32]; for (int l = 0; l < 32; ++l) n[l] = std::fmax(x[l], x[l ^ stride]); std::memcpy(x, n, sizeof x); }
    int s = 0;
    const bool tiny = x[0] > 0.0F && x[0] < 0x1p-119F;
    const float dl = x[0] == 0.0F || tiny ? 1.0F : x[0] / 127.0F;
    for (u32 lane = 0; lane < 32; ++lane) {
        const float rounded = tiny ? 0.0F : std::nearbyint(hv[lane] / dl);
        const int code = static_cast<int>(std::fmin(127.0F, std::fmax(-127.0F, rounded)));
        s += code; codes[lane] = static_cast<i8>(code);
    }
    *delta = dl; *sum = s;
    return true;
}

// ------------------------------------------------------------------------------------------- fixtures
struct Rng {
    std::mt19937_64 g;
    explicit Rng(std::uint64_t s) : g(s) {}
    u32 u(u32 n) { return static_cast<u32>(g() % n); }
    u32 bits() { return static_cast<u32>(g()); }
};
static u16 rand_half(Rng& r) {  // finite, moderate (exponent 6..14) fp16, either sign
    return static_cast<u16>((r.u(2) << 15) | ((6U + r.u(9)) << 10) | r.u(1024));
}

struct Layer {
    std::vector<u8> gate, up, down;
    ex::LayerWeights w;
};
static Layer make_layer(Rng& r, ex::Shape sh, ex::GateFormat gf, ex::DownFormat df, u32 experts) {
    Layer L;
    const u32 gwb = gf == ex::GateFormat::q5_k ? 176 : 144, dwb = df == ex::DownFormat::q5_1 ? 24 : 34;
    const size_t gbytes = static_cast<size_t>(experts) * sh.ffn * (sh.hidden / 256) * gwb;
    const size_t dbytes = static_cast<size_t>(experts) * sh.hidden * (sh.ffn / 32) * dwb;
    L.gate.resize(gbytes); L.up.resize(gbytes); L.down.resize(dbytes);
    auto fill = [&](std::vector<u8>& v) {
        for (size_t i = 0; i < v.size(); i += 8) { const std::uint64_t x = r.g(); std::memcpy(&v[i], &x, std::min<size_t>(8, v.size() - i)); }
    };
    fill(L.gate); fill(L.up); fill(L.down);  // scales/mins/quants: every bit pattern
    for (auto* m : {&L.gate, &L.up})
        for (size_t o = 0; o < m->size(); o += gwb) {
            const u16 d = rand_half(r), dm = rand_half(r);
            std::memcpy(m->data() + o, &d, 2); std::memcpy(m->data() + o + 2, &dm, 2);
        }
    for (size_t o = 0; o < L.down.size(); o += dwb) {
        const u16 d = rand_half(r); std::memcpy(L.down.data() + o, &d, 2);
        if (df == ex::DownFormat::q5_1) { const u16 m = rand_half(r); std::memcpy(L.down.data() + o + 2, &m, 2); }
    }
    L.w.gate = L.gate.data(); L.w.up = L.up.data(); L.w.down = L.down.data();
    L.w.gate_format = gf; L.w.down_format = df; L.w.shape = sh;
    return L;
}

struct RowData {
    std::vector<i8> codes; std::vector<float> deltas; std::vector<i32> sums;
};
static RowData make_row(Rng& r, u32 hidden, bool extreme_codes) {
    RowData d;
    d.codes.resize(hidden); d.deltas.resize(hidden / 32); d.sums.assign(hidden / 32, 0);
    for (u32 i = 0; i < hidden; ++i) {
        int c = static_cast<int>(r.u(255)) - 127;
        if (extreme_codes) c = r.u(2) ? 127 : -127;
        d.codes[i] = static_cast<i8>(c);
        d.sums[i / 32] += c;
    }
    for (auto& e : d.deltas) e = (1.0F + static_cast<float>(r.u(1000)) / 100.0F) * 0x1p-8F;
    return d;
}

struct Case {
    std::vector<RowData> rowdata;
    std::vector<ex::RowA8> rows;
    std::vector<ex::Assignment> as;
};
static Case make_case(Rng& r, u32 hidden, u32 experts, u32 n, bool extreme) {
    Case c;
    const u32 R = n >= 40 ? 4 : std::min<u32>(n, 6);
    for (u32 i = 0; i < R; ++i) c.rowdata.push_back(make_row(r, hidden, extreme && (i & 1)));
    for (auto& d : c.rowdata) c.rows.push_back({d.codes.data(), d.deltas.data(), d.sums.data()});
    std::vector<u32> slots(R, 0);
    for (u32 a = 0; a < n; ++a) {
        const u32 row = a % R;
        u32 e = n >= 40 ? (a % 7 == 0 ? 1U : 0U) : r.u(std::min<u32>(experts, 3));
        if (n > R && row == 0 && slots[0] == 1) e = c.as[0].expert;  // one row uses an expert in two slots
        c.as.push_back({row, slots[row]++, e});
    }
    return c;
}

constexpr u32 kSlots = 10;
struct Out {
    std::vector<float> raw; std::vector<u32> status;
};
static Out blank(u32 rows, u32 hidden) {
    Out o;
    o.raw.assign(static_cast<size_t>(rows) * kSlots * hidden, f_of(0x7fc01234U));  // sentinel NaN payload: untouched stays
    o.status.assign(rows, 0);
    return o;
}
static ex::Job make_job(const Layer& L, Case& c, Out& o) {
    ex::Job j;
    j.weights = L.w; j.rows = c.rows.data(); j.assignments = c.as.data(); j.count = static_cast<u32>(c.as.size());
    j.rawdots = o.raw.data(); j.rawdot_row_stride = static_cast<std::uint64_t>(kSlots) * L.w.shape.hidden; j.status = o.status.data();
    return j;
}

// independent oracle run
static void oracle_run(const HostExp& hx, const Layer& L, const Case& c, Out& o) {
    const ex::Shape sh = L.w.shape;
    const u32 blocks = sh.hidden / 256, groups = sh.ffn / 32;
    const bool q5k = L.w.gate_format == ex::GateFormat::q5_k, q51 = L.w.down_format == ex::DownFormat::q5_1;
    const u32 gwb = q5k ? 176 : 144, dwb = q51 ? 24 : 34;
    struct H { std::vector<i8> codes; std::vector<float> deltas; std::vector<int> sums; };
    std::vector<H> hs(c.as.size());
    bool bad = false;
    for (size_t a = 0; a < c.as.size(); ++a) {
        const auto& as = c.as[a]; const auto& row = c.rows[as.row];
        hs[a].codes.resize(sh.ffn); hs[a].deltas.resize(groups); hs[a].sums.resize(groups);
        std::vector<float> g(sh.ffn), u(sh.ffn);
        for (u32 f = 0; f < sh.ffn; ++f) {
            const size_t off = (static_cast<size_t>(as.expert) * sh.ffn + f) * blocks * gwb;
            g[f] = ref_gate_up_value(L.gate.data() + off, blocks, q5k, row.codes, row.deltas, row.sums);
            u[f] = ref_gate_up_value(L.up.data() + off, blocks, q5k, row.codes, row.deltas, row.sums);
        }
        for (u32 q = 0; q < groups; ++q)
            if (!ref_quad(hx, &g[q * 32], &u[q * 32], &hs[a].codes[q * 32], &hs[a].deltas[q], &hs[a].sums[q])) { bad = true; o.status[as.row] |= 1U; }
    }
    if (bad) return;
    for (size_t a = 0; a < c.as.size(); ++a) {
        const auto& as = c.as[a];
        for (u32 oo = 0; oo < sh.hidden; ++oo) {
            const u8* row = L.down.data() + (static_cast<size_t>(as.expert) * sh.hidden + oo) * groups * dwb;
            o.raw[static_cast<size_t>(as.row) * kSlots * sh.hidden + static_cast<size_t>(as.slot) * sh.hidden + oo] =
                ref_down_value(row, groups, q51, hs[a].codes.data(), hs[a].deltas.data(), hs[a].sums.data());
        }
    }
}

static bool same(const Out& a, const Out& b) {
    return a.raw.size() == b.raw.size() && std::memcmp(a.raw.data(), b.raw.data(), a.raw.size() * 4) == 0 && a.status == b.status;
}

static void write_tables(const std::string& dir, Rng& r) {
    std::filesystem::create_directories(dir);
    {
        FILE* f = std::fopen((dir + "/tbl_exp_pm.bin").c_str(), "wb");
        std::vector<u32> w((static_cast<size_t>(50U) << 23) / 32U + 1, 0);
        for (auto& x : w) x = r.bits() & r.bits() & r.bits() & r.bits();  // ~1/16 of the entries flip the rounding
        std::fwrite(w.data(), 4, w.size(), f); std::fclose(f);
    }
    {
        const u32 n = 3000;
        std::vector<u32> raw(2 * n);
        for (u32 i = 0; i < n; ++i) {
            raw[i] = (r.u(2) << 31) | ((102U << 23) + r.u(25U << 23));
            raw[n + i] = 0x3f800000U + r.u(4096);
        }
        FILE* f = std::fopen((dir + "/hard_v_exp_f32.bin").c_str(), "wb");
        std::fwrite(raw.data(), 4, raw.size(), f); std::fclose(f);
    }
}

int main() {
    Rng rng(0x5eedULL);
    std::string dir;
    bool synthetic = false;
    if (const char* e = std::getenv("STRIX_EXP_TABLES")) dir = e;
    else {
        dir = (std::filesystem::temp_directory_path() / "strix_expert_test_tables").string();
        write_tables(dir, rng);
        synthetic = true;
    }
    const HostExp hx(dir);
    const ex::Gfx1151Exp exp(dir);

    struct Sz { ex::Shape sh; std::vector<u32> counts; u32 experts; };
    const std::vector<Sz> sizes = {{{512, 256}, {1, 2, 3, 5, 8, 13, 16, 40}, 6}, {{2560, 640}, {1, 5, 16, 40}, 4}};
    const unsigned thread_counts[] = {1, 3, 16};
    for (const auto& sz : sizes)
        for (int gf = 0; gf < 2; ++gf)
            for (int df = 0; df < 2; ++df) {
                const Layer L = make_layer(rng, sz.sh, static_cast<ex::GateFormat>(gf), static_cast<ex::DownFormat>(df), sz.experts);
                for (u32 n : sz.counts) {
                    Case c = make_case(rng, sz.sh.hidden, sz.experts, n, false);
                    const std::string tag = std::string(gf ? "q5k" : "q4k") + (df ? "/q8_0 " : "/q5_1 ") + std::to_string(sz.sh.hidden) + "x" +
                                            std::to_string(sz.sh.ffn) + " n=" + std::to_string(n);
                    const u32 rows = static_cast<u32>(c.rows.size());
                    Out ref = blank(rows, sz.sh.hidden), orc = blank(rows, sz.sh.hidden);
                    { ex::Job j = make_job(L, c, ref); ex::reference_run(j, exp); }
                    oracle_run(hx, L, c, orc);
                    expect(same(ref, orc), "reference_run vs verbatim oracle: " + tag);
                    expect(ref.status == std::vector<u32>(rows, 0), "finite case has no status: " + tag);
                    ++g_cases;
                    for (unsigned t : thread_counts)
                        for (int simd = 0; simd < 2; ++simd) {
                            ex::Executor E(t, exp, simd != 0);
                            Out got = blank(rows, sz.sh.hidden);
                            ex::Job j = make_job(L, c, got);
                            E.run(j);
                            expect(same(ref, got), "executor " + std::string(E.isa()) + " threads=" + std::to_string(t) + ": " + tag);
                            Out again = blank(rows, sz.sh.hidden);  // a second job on the same pool
                            ex::Job j2 = make_job(L, c, again);
                            E.run(j2);
                            expect(same(ref, again), "executor rerun " + std::string(E.isa()) + " threads=" + std::to_string(t) + ": " + tag);
                            ++g_cases;
                        }
                }
                // extreme codes (+-127: maximum integer dots) and the non-finite case (huge delta on row 1)
                {
                    Case c = make_case(rng, sz.sh.hidden, sz.experts, 8, true);
                    const u32 rows = static_cast<u32>(c.rows.size());
                    Out ref = blank(rows, sz.sh.hidden);
                    { ex::Job j = make_job(L, c, ref); ex::reference_run(j, exp); }
                    for (unsigned t : thread_counts) {
                        ex::Executor E(t, exp, true);
                        Out got = blank(rows, sz.sh.hidden);
                        ex::Job j = make_job(L, c, got); E.run(j);
                        expect(same(ref, got), "extreme codes threads=" + std::to_string(t));
                    }
                    c.rowdata[1].deltas[3] = 3.0e38F;
                    Out bref = blank(rows, sz.sh.hidden), borc = blank(rows, sz.sh.hidden);
                    { ex::Job j = make_job(L, c, bref); ex::reference_run(j, exp); }
                    oracle_run(hx, L, c, borc);
                    expect(same(bref, borc), "non-finite: reference vs oracle");
                    expect((bref.status[1] & ex::kNonfinite) != 0, "non-finite sets status of the offending row");
                    expect(bref.status[0] == 0, "non-finite leaves other rows clean");
                    for (unsigned t : thread_counts)
                        for (int simd = 0; simd < 2; ++simd) {
                            ex::Executor E(t, exp, simd != 0);
                            Out got = blank(rows, sz.sh.hidden);
                            ex::Job j = make_job(L, c, got); E.run(j);
                            expect(same(bref, got), "non-finite executor " + std::string(E.isa()) + " threads=" + std::to_string(t));
                        }
                    ++g_cases;
                }
            }
    if (synthetic) { std::error_code ec; std::filesystem::remove_all(dir, ec); }
    ex::Executor probe(1, exp);
    std::printf("strix-cpu-routed-expert-tests: %s (%u cases, tables %s, simd isa %s)\n", g_fail ? "FAILED" : "PASS", g_cases,
                synthetic ? "synthetic" : "real", probe.isa());
    return g_fail ? 1 : 0;
}
