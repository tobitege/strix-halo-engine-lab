// Host routed-expert executor: scalar oracle + persistent worker pool (see routed_expert.hpp for the contract).
#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS 1
#endif
#include "cpu/core/routed_expert.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "cpu/arch/avx2/routed_expert_avx2.hpp"
#include "cpu/core/backend.hpp"
#include "cpu/core/routed_expert_kernels.hpp"

#if defined(_MSC_VER)
#pragma fp_contract(off)
#endif
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#define STRIX_PAUSE() _mm_pause()
#elif defined(__x86_64__) || defined(__i386__)
#define STRIX_PAUSE() __builtin_ia32_pause()
#else
#define STRIX_PAUSE() std::this_thread::yield()
#endif

namespace strix::cpu::expert {
namespace {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using i8 = std::int8_t;
using i32 = std::int32_t;

float f_of(u32 b) noexcept { float f; std::memcpy(&f, &b, 4); return f; }
u32 b_of(float f) noexcept { u32 b; std::memcpy(&b, &f, 4); return b; }
bool finite(float x) noexcept { return (b_of(x) & 0x7f800000U) != 0x7f800000U; }

constexpr u32 kGateItemOutputs = 32;  // phase 1 item = one 32-output group (= one h quantization group)
constexpr u32 kDownItemOutputs = 16;  // phase 2 item = this many hidden outputs of one expert

std::vector<u32> read_u32_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("routed_expert: missing table " + path);
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<u32> v(static_cast<std::size_t>(n) / 4);
    const bool ok = std::fread(v.data(), 4, v.size(), f) == v.size();
    std::fclose(f);
    if (!ok) throw std::runtime_error("routed_expert: short read " + path);
    return v;
}

// ---- scalar row kernels (the oracle; arithmetic = project_rows_kernel / down_rows_kernel) -------------------------------
unsigned factor(const u8* hd, unsigned group, bool minimum) noexcept {
    u32 y, z, w;
    std::memcpy(&y, hd + 4, 4); std::memcpy(&z, hd + 8, 4); std::memcpy(&w, hd + 12, 4);
    const unsigned shift = 8U * (group & 3U);
    const unsigned a = (y >> shift) & 255U, b = (z >> shift) & 255U, c = (w >> shift) & 255U;
    return group < 4U ? (minimum ? b : a) & 63U : minimum ? (c >> 4U) | ((b >> 6U) << 4U) : (c & 15U) | ((a >> 6U) << 4U);
}

template <bool Q5K>
void gate_row_scalar(const u8* wrow, u32 blocks, const RowA8* const* rows, u32 na, float* out) {
    constexpr u32 WB = Q5K ? 176 : 144, QS = Q5K ? 48 : 16;
    for (u32 a = 0; a < na; ++a) {
        const i8* codes = rows[a]->codes;
        const float* deltas = rows[a]->deltas;
        const i32* sums = rows[a]->sums;
        float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        for (u32 b = 0; b < blocks; ++b) {
            const u8* blk = wrow + static_cast<std::size_t>(b) * WB;
            u16 dh, dmh;
            std::memcpy(&dh, blk, 2); std::memcpy(&dmh, blk + 2, 2);
            const float d = detail::half_to_float(dh), dm = detail::half_to_float(dmh);
            float x[8];
            for (u32 j = 0; j < 8; ++j) {
                const u32 p = j >> 1, hi = j & 1;
                int dot = 0;
                for (u32 k = 0; k < 32; ++k) {
                    const u8 byte = blk[QS + 32 * p + k];
                    int q = hi ? (byte >> 4) : (byte & 15);
                    if (Q5K) q |= ((blk[16 + k] >> j) & 1) << 4;
                    dot += codes[256 * b + 32 * j + k] * q;
                }
                const float D = d * static_cast<float>(factor(blk, j, false));
                const float DM = dm * static_cast<float>(factor(blk, j, true));
                const float u = D * static_cast<float>(dot);
                const float v = DM * static_cast<float>(sums[8 * b + j]);
                const float t = u - v;
                x[j] = deltas[8 * b + j] * t;
            }
            for (u32 p = 0; p < 4; ++p) acc[p] = acc[p] + (x[2 * p] + x[2 * p + 1]);
        }
        const float s01 = acc[0] + acc[1], s23 = acc[2] + acc[3];
        out[a] = s01 + s23;
    }
}

template <bool Q51>
void down_row_scalar(const u8* wrow, u32 groups, const i8* const* codes, const float* const* deltas, const i32* const* sums,
                     u32 na, float* out) {
    constexpr u32 WB = Q51 ? 24 : 34;
    for (u32 a = 0; a < na; ++a) {
        float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        for (u32 g = 0; g < groups; ++g) {
            const u8* blk = wrow + static_cast<std::size_t>(g) * WB;
            u16 dh;
            std::memcpy(&dh, blk, 2);
            const float d = detail::half_to_float(dh);
            const i8* c = codes[a] + 32 * g;
            int dot = 0;
            float m = 0.0F;
            if (Q51) {
                u16 mh; u32 qh;
                std::memcpy(&mh, blk + 2, 2); std::memcpy(&qh, blk + 4, 4);
                m = detail::half_to_float(mh);
                for (u32 k = 0; k < 32; ++k) {
                    const u8 byte = blk[8 + (k & 15)];
                    const int q = (k < 16 ? (byte & 15) : (byte >> 4)) | static_cast<int>(((qh >> k) & 1U) << 4);
                    dot += c[k] * q;
                }
            } else {
                for (u32 k = 0; k < 32; ++k) dot += c[k] * static_cast<i8>(blk[2 + k]);
            }
            const float first = d * static_cast<float>(dot);
            float combined = first;
            if (Q51) combined = first + (m * static_cast<float>(sums[a][g]));
            acc[g % 4] = acc[g % 4] + (deltas[a][g] * combined);
        }
        const float s01 = acc[0] + acc[1], s23 = acc[2] + acc[3];
        out[a] = s01 + s23;
    }
}

// SwiGLU + quantize_a8 over one 32-output group. Returns false (writes nothing but the status decision) on non-finite.
bool swiglu_quantize(const Gfx1151Exp& exp, const float* g, const float* u, i8* codes, float* delta, i32* sum) {
    float hv[32];
    bool bad = false;
    for (u32 l = 0; l < 32; ++l) {
        const float ex = exp.expf(-g[l]);
        hv[l] = (g[l] / (1.0F + ex)) * u[l];
        bad |= !finite(g[l]) || !finite(u[l]) || !finite(hv[l]);
    }
    if (bad) return false;
    float x = 0.0F;
    for (u32 l = 0; l < 32; ++l) x = std::fmax(x, std::fabs(hv[l]));
    const bool tiny = x > 0.0F && x < 0x1p-119F;
    const float dl = x == 0.0F || tiny ? 1.0F : x / 127.0F;
    int s = 0;
    for (u32 l = 0; l < 32; ++l) {
        const float rounded = tiny ? 0.0F : std::nearbyint(hv[l] / dl);
        const int code = static_cast<int>(std::fmin(127.0F, std::fmax(-127.0F, rounded)));
        s += code;
        codes[l] = static_cast<i8>(code);
    }
    *delta = dl;
    *sum = s;
    return true;
}

void or_status(u32* status, u32 bits) {
    std::atomic_ref<u32>(*status).fetch_or(bits, std::memory_order_relaxed);
}

std::size_t gate_row_offset(const LayerWeights& w, u32 expert, u32 o) {
    const u32 blocks = w.shape.hidden / 256;
    return (static_cast<std::size_t>(expert) * w.shape.ffn + o) * blocks * detail::kGateBlockBytes[static_cast<int>(w.gate_format)];
}
std::size_t down_row_offset(const LayerWeights& w, u32 expert, u32 o) {
    const u32 groups = w.shape.ffn / 32;
    return (static_cast<std::size_t>(expert) * w.shape.hidden + o) * groups * detail::kDownBlockBytes[static_cast<int>(w.down_format)];
}

}  // namespace

// ---- detail ----------------------------------------------------------------------------------------------------------
namespace detail {
float half_to_float(u16 h) noexcept {
    const u32 s = (h >> 15) & 1U, e = (h >> 10) & 31U, m = h & 1023U;
    if (e == 0) { const float v = std::ldexp(static_cast<float>(m), -24); return s ? -v : v; }
    if (e == 31) return f_of((s << 31) | 0x7f800000U | (m << 13));
    return f_of((s << 31) | ((e + 112U) << 23) | (m << 13));
}
const Kernels& scalar_kernels() noexcept {
    static const Kernels k = [] {
        Kernels t;
        t.name = "scalar";
        t.gate[0] = gate_row_scalar<false>; t.gate[1] = gate_row_scalar<true>;
        t.down[0] = down_row_scalar<true>;  t.down[1] = down_row_scalar<false>;
        return t;
    }();
    return k;
}
}  // namespace detail

// ---- Gfx1151Exp (host twin of amd_expf, tests/cuda/qwen_moe_tests.cu HostExp) -----------------------------------------
struct Gfx1151Exp::Data {
    std::vector<u32> bits, hard_in;
    std::vector<float> hard_out;

    float v_exp(float x) const noexcept {
        const u32 b = b_of(x), mag = b & 0x7fffffffU;
        const double r = std::exp2(static_cast<double>(x));
        const float f = static_cast<float>(r);
        if (mag < (102U << 23)) return f;
        const auto it = std::lower_bound(hard_in.begin(), hard_in.end(), b);
        if (it != hard_in.end() && *it == b) return hard_out[static_cast<std::size_t>(it - hard_in.begin())];
        const u32 index = (b >> 31) * (25U << 23) + (mag - (102U << 23));
        // a table shorter than the full probe table (synthetic test tables) reads as "no correction" past its end
        if ((index >> 5) < bits.size() && ((bits[index >> 5] >> (index & 31U)) & 1U)) {
            const bool up = r > static_cast<double>(f);
            const bool away = (b_of(f) >> 31) ? !up : up;
            return f_of(away ? b_of(f) + 1U : b_of(f) - 1U);
        }
        return f;
    }
};

Gfx1151Exp::Gfx1151Exp(const std::string& dir) : d_(std::make_unique<Data>()) {
    d_->bits = read_u32_file(dir + "/tbl_exp_pm.bin");
    const auto raw = read_u32_file(dir + "/hard_v_exp_f32.bin");
    const std::size_t n = raw.size() / 2;
    std::vector<std::pair<u32, u32>> v;
    for (std::size_t i = 0; i < n; ++i) {
        const u32 mag = raw[i] & 0x7fffffffU;
        if (mag >= (102U << 23) && mag < (127U << 23)) v.push_back({raw[i], raw[n + i]});
    }
    std::sort(v.begin(), v.end());
    for (auto& p : v) {
        if (!d_->hard_in.empty() && d_->hard_in.back() == p.first) continue;
        d_->hard_in.push_back(p.first);
        d_->hard_out.push_back(f_of(p.second));
    }
}
Gfx1151Exp::~Gfx1151Exp() = default;

float Gfx1151Exp::expf(float x) const noexcept {
    const float kLog2e = f_of(0x3fb8aa3bU), kLog2eLo = f_of(0x32a5705fU);
    const float ph = x * kLog2e;
    const float e = std::nearbyint(ph);
    float pl = std::fmaf(x, kLog2e, -ph);
    pl = std::fmaf(x, kLog2eLo, pl);
    const float a = (ph - e) + pl;
    float y;
    if (std::fabs(a) < 1.0F) y = std::scalbn(d_->v_exp(a), static_cast<int>(e));
    else y = a != a ? a : 0.0F;
    if (f_of(0xc2ce8ed0U) > x) y = 0.0F;
    if (f_of(0x42b17218U) < x) y = f_of(0x7f800000U);
    return y;
}

// ---- reference_run ---------------------------------------------------------------------------------------------------
void reference_run(const Job& job, const Gfx1151Exp& exp) {
    const LayerWeights& w = job.weights;
    const u32 hidden = w.shape.hidden, ffn = w.shape.ffn, blocks = hidden / 256, groups = ffn / 32;
    const detail::Kernels& k = detail::scalar_kernels();
    std::vector<i8> hc(static_cast<std::size_t>(job.count) * ffn);
    std::vector<float> hd(static_cast<std::size_t>(job.count) * groups);
    std::vector<i32> hs(static_cast<std::size_t>(job.count) * groups);
    bool bad = false;
    for (u32 a = 0; a < job.count; ++a) {
        const Assignment& as = job.assignments[a];
        const RowA8* row = &job.rows[as.row];
        for (u32 grp = 0; grp < groups; ++grp) {
            float g[32], u[32];
            for (u32 l = 0; l < 32; ++l) {
                const u32 o = grp * 32 + l;
                k.gate[static_cast<int>(w.gate_format)](w.gate + gate_row_offset(w, as.expert, o), blocks, &row, 1, &g[l]);
                k.gate[static_cast<int>(w.gate_format)](w.up + gate_row_offset(w, as.expert, o), blocks, &row, 1, &u[l]);
            }
            if (!swiglu_quantize(exp, g, u, hc.data() + static_cast<std::size_t>(a) * ffn + grp * 32,
                                 hd.data() + static_cast<std::size_t>(a) * groups + grp,
                                 hs.data() + static_cast<std::size_t>(a) * groups + grp)) {
                bad = true;
                job.status[as.row] |= kNonfinite;
            }
        }
    }
    if (bad) return;
    for (u32 a = 0; a < job.count; ++a) {
        const Assignment& as = job.assignments[a];
        const i8* c = hc.data() + static_cast<std::size_t>(a) * ffn;
        const float* d = hd.data() + static_cast<std::size_t>(a) * groups;
        const i32* s = hs.data() + static_cast<std::size_t>(a) * groups;
        float* out = job.rawdots + static_cast<std::size_t>(as.row) * job.rawdot_row_stride + static_cast<std::size_t>(as.slot) * hidden;
        for (u32 o = 0; o < hidden; ++o)
            k.down[static_cast<int>(w.down_format)](w.down + down_row_offset(w, as.expert, o), groups, &c, &d, &s, 1, &out[o]);
    }
}

// ---- Executor --------------------------------------------------------------------------------------------------------
#if defined(_MSC_VER)
#pragma warning(disable : 4324)  // cache-line aligned counters pad the struct on purpose
#endif
struct Executor::Impl {
    unsigned nthreads = 1;
    const Gfx1151Exp& exp;
    const detail::Kernels* kern = nullptr;
    std::vector<std::thread> workers;

    std::mutex m;
    std::condition_variable cv_wake, cv_idle;
    std::atomic<std::uint64_t> gen{0};
    bool stop = false, job_active = false;
    unsigned inside = 0;

    // prepared job (read-only while participants run)
    const Job* job = nullptr;
    u32 hidden = 0, ffn = 0, blocks = 0, groups = 0, n_experts = 0, n1 = 0, n2 = 0;
    std::vector<u32> order, ex_id, ex_begin;                 // order: assignment indices grouped by expert
    std::vector<const RowA8*> row_ptr;                       // by order position
    std::vector<const i8*> c_ptr;
    std::vector<const float*> d_ptr;
    std::vector<const i32*> s_ptr;
    std::vector<i8> hcodes;
    std::vector<float> hdelta;
    std::vector<i32> hsum;
    alignas(64) std::atomic<u32> next1{0};
    alignas(64) std::atomic<u32> done1{0};
    alignas(64) std::atomic<u32> next2{0};
    alignas(64) std::atomic<u32> done2{0};
    alignas(64) std::atomic<bool> bad{false};

    Impl(unsigned t, const Gfx1151Exp& e, bool simd) : nthreads(t ? t : 1), exp(e) {
        kern = &detail::scalar_kernels();
        if (simd && caps().avx2) {
            if (const detail::Kernels* k = arch::avx2::expert_kernels()) kern = k;
        }
        for (unsigned i = 1; i < nthreads; ++i) workers.emplace_back([this] { worker(); });
    }
    ~Impl() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop = true;
        }
        cv_wake.notify_all();
        for (auto& t : workers) t.join();
    }

    void worker() {
        std::uint64_t seen = 0;
        for (;;) {
            bool woke = false;
            for (int spin = 0; spin < 2000; ++spin) {
                if (gen.load(std::memory_order_acquire) != seen) { woke = true; break; }
                STRIX_PAUSE();
            }
            std::unique_lock<std::mutex> lk(m);
            if (!woke) cv_wake.wait(lk, [&] { return stop || gen.load(std::memory_order_acquire) != seen; });
            if (stop) return;
            seen = gen.load(std::memory_order_acquire);
            if (!job_active) continue;
            ++inside;
            lk.unlock();
            participate();
            lk.lock();
            if (--inside == 0) cv_idle.notify_all();
        }
    }

    void prepare(const Job& j) {
        job = &j;
        const LayerWeights& w = j.weights;
        hidden = w.shape.hidden; ffn = w.shape.ffn; blocks = hidden / 256; groups = ffn / 32;
        const u32 n = j.count;
        order.resize(n); row_ptr.resize(n); c_ptr.resize(n); d_ptr.resize(n); s_ptr.resize(n);
        hcodes.resize(static_cast<std::size_t>(n) * ffn);
        hdelta.resize(static_cast<std::size_t>(n) * groups);
        hsum.resize(static_cast<std::size_t>(n) * groups);
        ex_id.clear(); ex_begin.clear();
        // group by expert, stable (first-seen expert order, assignment order inside an expert)
        std::vector<u32>& idx = order;
        for (u32 a = 0; a < n; ++a) idx[a] = a;
        std::stable_sort(idx.begin(), idx.end(), [&](u32 x, u32 y) { return j.assignments[x].expert < j.assignments[y].expert; });
        for (u32 p = 0; p < n; ++p) {
            const Assignment& as = j.assignments[idx[p]];
            if (p == 0 || as.expert != j.assignments[idx[p - 1]].expert) { ex_id.push_back(as.expert); ex_begin.push_back(p); }
            row_ptr[p] = &j.rows[as.row];
            c_ptr[p] = hcodes.data() + static_cast<std::size_t>(p) * ffn;
            d_ptr[p] = hdelta.data() + static_cast<std::size_t>(p) * groups;
            s_ptr[p] = hsum.data() + static_cast<std::size_t>(p) * groups;
        }
        ex_begin.push_back(n);
        n_experts = static_cast<u32>(ex_id.size());
        n1 = n_experts * groups;
        n2 = n_experts * (hidden / kDownItemOutputs);
        next1.store(0, std::memory_order_relaxed); done1.store(0, std::memory_order_relaxed);
        next2.store(0, std::memory_order_relaxed); done2.store(0, std::memory_order_relaxed);
        bad.store(false, std::memory_order_relaxed);
    }

    void gate_item(u32 item) {
        const LayerWeights& w = job->weights;
        const u32 e = item / groups, grp = item % groups, expert = ex_id[e];
        const u32 begin = ex_begin[e], count = ex_begin[e + 1] - begin;
        const int gf = static_cast<int>(w.gate_format);
        float g[detail::kMaxAssign][kGateItemOutputs], u[detail::kMaxAssign][kGateItemOutputs];
        for (u32 c0 = 0; c0 < count; c0 += detail::kMaxAssign) {
            const u32 na = std::min(detail::kMaxAssign, count - c0), first = begin + c0;
            for (u32 l = 0; l < kGateItemOutputs; ++l) {
                const u32 o = grp * kGateItemOutputs + l;
                float tg[detail::kMaxAssign], tu[detail::kMaxAssign];
                kern->gate[gf](w.gate + gate_row_offset(w, expert, o), blocks, &row_ptr[first], na, tg);
                kern->gate[gf](w.up + gate_row_offset(w, expert, o), blocks, &row_ptr[first], na, tu);
                for (u32 a = 0; a < na; ++a) { g[a][l] = tg[a]; u[a][l] = tu[a]; }
            }
            for (u32 a = 0; a < na; ++a) {
                const u32 p = first + a;
                if (!swiglu_quantize(exp, g[a], u[a], hcodes.data() + static_cast<std::size_t>(p) * ffn + grp * 32,
                                     &hdelta[static_cast<std::size_t>(p) * groups + grp], &hsum[static_cast<std::size_t>(p) * groups + grp])) {
                    bad.store(true, std::memory_order_relaxed);
                    or_status(job->status + job->assignments[order[p]].row, kNonfinite);
                }
            }
        }
    }

    void down_item(u32 item) {
        const LayerWeights& w = job->weights;
        const u32 per = hidden / kDownItemOutputs;
        const u32 e = item / per, o0 = (item % per) * kDownItemOutputs, expert = ex_id[e];
        const u32 begin = ex_begin[e], count = ex_begin[e + 1] - begin;
        const int df = static_cast<int>(w.down_format);
        for (u32 c0 = 0; c0 < count; c0 += detail::kMaxAssign) {
            const u32 na = std::min(detail::kMaxAssign, count - c0), first = begin + c0;
            for (u32 l = 0; l < kDownItemOutputs; ++l) {
                const u32 o = o0 + l;
                float t[detail::kMaxAssign];
                kern->down[df](w.down + down_row_offset(w, expert, o), groups, &c_ptr[first], &d_ptr[first], &s_ptr[first], na, t);
                for (u32 a = 0; a < na; ++a) {
                    const Assignment& as = job->assignments[order[first + a]];
                    job->rawdots[static_cast<std::size_t>(as.row) * job->rawdot_row_stride + static_cast<std::size_t>(as.slot) * hidden + o] = t[a];
                }
            }
        }
    }

    void participate() {
        for (;;) {
            const u32 i = next1.fetch_add(1, std::memory_order_relaxed);
            if (i >= n1) break;
            gate_item(i);
            done1.fetch_add(1, std::memory_order_release);
        }
        unsigned spins = 0;
        while (done1.load(std::memory_order_acquire) < n1) {
            STRIX_PAUSE();
            if (++spins > 20000) { std::this_thread::yield(); spins = 0; }
        }
        if (bad.load(std::memory_order_relaxed)) return;
        for (;;) {
            const u32 i = next2.fetch_add(1, std::memory_order_relaxed);
            if (i >= n2) break;
            down_item(i);
            done2.fetch_add(1, std::memory_order_release);
        }
    }

    void run(const Job& j) {
        if (j.count == 0) return;
        prepare(j);
        {
            std::lock_guard<std::mutex> lk(m);
            job_active = true;
            gen.fetch_add(1, std::memory_order_release);
        }
        if (!workers.empty()) cv_wake.notify_all();
        participate();
        unsigned spins = 0;
        while (!bad.load(std::memory_order_relaxed) && done2.load(std::memory_order_acquire) < n2) {
            STRIX_PAUSE();
            if (++spins > 20000) { std::this_thread::yield(); spins = 0; }
        }
        std::unique_lock<std::mutex> lk(m);
        job_active = false;
        cv_idle.wait(lk, [&] { return inside == 0; });
    }
};

Executor::Executor(unsigned threads, const Gfx1151Exp& exp, bool allow_simd) : impl_(std::make_unique<Impl>(threads, exp, allow_simd)) {}
Executor::~Executor() = default;
void Executor::run(const Job& job) { impl_->run(job); }
unsigned Executor::threads() const noexcept { return impl_->nthreads; }
const char* Executor::isa() const noexcept { return impl_->kern->name; }

}  // namespace strix::cpu::expert
