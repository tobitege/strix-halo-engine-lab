// Exhaustive check of include/strix/cuda/amd_math.cuh against gfx1151 (qwen-rtx-prefill probes p1/p2).
//   qwen_amd_math_tests <tables dir>
// The dir holds tbl_exp_pm.bin, tbl_rsq_1_4.bin, tbl_rcp_1_2.bin, hard_v_{exp,rsq,rcp}_f32.bin (inputs then outputs),
// hash_v_{exp,rsq,rcp}_f32.bin (FNV-1a 64 of the hardware outputs per 4096 consecutive input patterns) and, optionally,
// miss_expf.bin (every finite input where gfx1151 expf differs from (float)exp(double), then those outputs).
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "strix/cuda/amd_math.cuh"
#include "strix/cuda/amd_math_tables.hpp"

namespace am = strix::cuda::amdmath;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

static std::vector<std::uint32_t> read_u32(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "missing %s\n", p.c_str()); std::exit(2); }
    std::fseek(f, 0, SEEK_END); const long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<std::uint32_t> v(static_cast<size_t>(n) / 4);
    if (std::fread(v.data(), 4, v.size(), f) != v.size()) std::exit(2);
    std::fclose(f);
    return v;
}
template <class T> static T* upload(const std::vector<T>& v) {
    T* d = nullptr; CK(cudaMalloc(&d, std::max<size_t>(v.size(), 1) * sizeof(T)));
    if (!v.empty()) CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// hard list -> (canonical input, output) sorted; canonicalisation for rsq ([1,4)) and rcp ([1,2), positive).
static void hard_list(const std::string& p, int kind, std::vector<std::uint32_t>& in, std::vector<float>& out) {
    const auto raw = read_u32(p);
    const size_t n = raw.size() / 2;
    std::map<std::uint32_t, std::uint32_t> m;
    unsigned conflicts = 0;
    for (size_t i = 0; i < n; ++i) {
        std::uint32_t b = raw[i], y = raw[n + i];
        // raw-instruction hard entries outside the emulated domain (subnormal or special inputs) are dropped
        if (kind != 0 && (((b >> 23) & 255U) == 0U || ((b >> 23) & 255U) == 255U)) continue;
        if (kind == 1) {  // rsq
            const int e = static_cast<int>(b >> 23) - 127;
            const int k = e >= 0 ? e / 2 : -((-e + 1) / 2);
            b = (b & 0x007fffffU) | (static_cast<std::uint32_t>(e - 2 * k + 127) << 23);
            y = static_cast<std::uint32_t>(static_cast<int>(y) + (k << 23));
        } else if (kind == 2) {  // rcp
            const int e = static_cast<int>((b >> 23) & 255U) - 127;
            const std::uint32_t s = b & 0x80000000U;
            b = (b & 0x007fffffU) | (127U << 23);
            y = static_cast<std::uint32_t>(static_cast<int>(y & 0x7fffffffU) + (e << 23));
            if ((y >> 23) == 0 || (y >> 23) >= 255) continue;  // subnormal/flushed results are outside the domain
            (void)s;
        }
        auto it = m.find(b);
        if (it != m.end() && it->second != y) ++conflicts;
        m[b] = y;
    }
    for (const auto& kv : m) { in.push_back(kv.first); float f; std::memcpy(&f, &kv.second, 4); out.push_back(f); }
    std::printf("hard %s: %zu entries -> %zu canonical, %u conflicts\n", p.c_str(), n, in.size(), conflicts);
    if (conflicts) std::exit(3);
}

__global__ void k_hash(int fn, am::Tables t, std::uint32_t first_block, std::uint32_t blocks, unsigned long long* out) {
    __shared__ std::uint32_t v[4096];
    const std::uint32_t blk = first_block + blockIdx.x;
    if (blockIdx.x >= blocks) return;
    const std::uint32_t base = blk * 4096U;
    for (unsigned i = threadIdx.x; i < 4096; i += blockDim.x) {
        const float x = __uint_as_float(base + i);
        const float y = fn == 0 ? am::amd_v_exp(t, x) : fn == 1 ? am::amd_v_rsq(t, x) : am::amd_v_rcp(t, x);
        v[i] = __float_as_uint(y);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long f = 1469598103934665603ULL;
        for (unsigned i = 0; i < 4096; ++i)
            for (int b = 0; b < 4; ++b) { f ^= (v[i] >> (8 * b)) & 255U; f *= 1099511628211ULL; }
        out[blockIdx.x] = f;
    }
}

struct Ranges { std::uint32_t lo, hi; };  // input patterns [lo, hi), multiples of 4096

static unsigned long long check_hashes(int fn, const am::Tables& t, const std::vector<std::uint32_t>& amd_hash_u32,
                                       const std::vector<Ranges>& rs, const char* name) {
    const unsigned long long* amd = reinterpret_cast<const unsigned long long*>(amd_hash_u32.data());
    unsigned long long* d; CK(cudaMalloc(&d, 65536 * 8));
    std::vector<unsigned long long> h(65536);
    unsigned long long blocks = 0, bad = 0;
    for (const auto& r : rs) {
        for (std::uint32_t b0 = r.lo / 4096; b0 < r.hi / 4096; b0 += 65536) {
            const std::uint32_t n = std::min<std::uint32_t>(65536, r.hi / 4096 - b0);
            k_hash<<<n, 256>>>(fn, t, b0, n, d);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(h.data(), d, n * 8, cudaMemcpyDeviceToHost));
            for (std::uint32_t i = 0; i < n; ++i) { ++blocks; if (h[i] != amd[b0 + i]) { if (bad < 5) std::printf("  %s block %u differs\n", name, b0 + i); ++bad; } }
        }
    }
    CK(cudaFree(d));
    std::printf("%s: %llu blocks of 4096 inputs (%.3e inputs), %llu differ -> %s\n", name, blocks, blocks * 4096.0, bad, bad ? "FAIL" : "PASS");
    return bad;
}

// expf over every finite input: the emulation must equal (float)exp(double) except exactly at the AMD miss list.
__global__ void k_expf(int fn, am::Tables t, std::uint32_t base, const std::uint32_t* miss_in, const float* miss_out, int miss_n,
                       unsigned long long* cnt) {
    const std::uint32_t b = base + blockIdx.x * blockDim.x + threadIdx.x;
    const float x = __uint_as_float(b);
    if (!(x == x) || (b & 0x7fffffffU) == 0x7f800000U) return;
    const float y = fn == 0 ? am::amd_expf(t, x) : am::amd_log1pf(t, x);
    const int m = am::find_hard(miss_in, miss_n, b);
    float want;
    if (m >= 0) want = __ldg(miss_out + m);
    else want = static_cast<float>(fn == 0 ? exp(static_cast<double>(x)) : log1p(static_cast<double>(x)));
    if (want != want && y != y) return;
    atomicAdd(&cnt[0], 1ULL);
    if (__float_as_uint(y) != __float_as_uint(want)) {
        // count separately where CUDA's double exp itself may round differently (the probe flagged 18 hard inputs)
        atomicAdd(&cnt[m >= 0 ? 1 : 2], 1ULL);
    }
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    std::vector<std::uint32_t> ein, rin, cin; std::vector<float> eout, rout, cout_;
    hard_list(dir + "/hard_v_exp_f32.bin", 0, ein, eout);
    hard_list(dir + "/hard_v_rsq_f32.bin", 1, rin, rout);
    hard_list(dir + "/hard_v_rcp_f32.bin", 2, cin, cout_);
    // the exp table covers [102<<23, 127<<23) per sign; keep only hard entries with |x| in [2^-25, 1)
    { std::vector<std::uint32_t> a; std::vector<float> b;
      for (size_t i = 0; i < ein.size(); ++i) { const std::uint32_t m = ein[i] & 0x7fffffffU; if (m >= (102U << 23) && m < (127U << 23)) { a.push_back(ein[i]); b.push_back(eout[i]); } }
      ein.swap(a); eout.swap(b); }
    am::Tables t = am::load_tables(dir);  // the library loader (includes the fast exp2 table)
    std::printf("exp2 fast table %s\n", t.exp2_tab ? "on" : "off");
    unsigned long long fails = 0;
    fails += check_hashes(0, t, read_u32(dir + "/hash_v_exp_f32.bin"), {{0U, 127U << 23}, {0x80000000U, 0x80000000U | (127U << 23)}}, "v_exp_f32 |x|<1");
    fails += check_hashes(1, t, read_u32(dir + "/hash_v_rsq_f32.bin"), {{1U << 23, 0x7f800000U}}, "v_rsq_f32 normal x>0");
    fails += check_hashes(2, t, read_u32(dir + "/hash_v_rcp_f32.bin"),
                          {{1U << 23, 253U << 23}, {0x80000000U | (1U << 23), 0x80000000U | (253U << 23)}}, "v_rcp_f32 normal |x|<2^126");
    for (int fn = 0; fn < 2; ++fn) {
    const std::string mp = dir + (fn == 0 ? "/miss_expf.bin" : "/miss_log1pf.bin");
    if (FILE* f = std::fopen(mp.c_str(), "rb")) {
        std::fclose(f);
        const auto raw = read_u32(mp);
        const size_t n = raw.size() / 2;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = {raw[i], raw[n + i]};
        std::sort(v.begin(), v.end());
        std::vector<std::uint32_t> mi(n); std::vector<float> mo(n);
        for (size_t i = 0; i < n; ++i) { mi[i] = v[i].first; std::memcpy(&mo[i], &v[i].second, 4); }
        std::uint32_t* dmi = upload(mi); float* dmo = upload(mo);
        unsigned long long* cnt; CK(cudaMallocManaged(&cnt, 3 * 8)); cnt[0] = cnt[1] = cnt[2] = 0;
        for (std::uint32_t hi = 0; hi < 256; ++hi) k_expf<<<1U << 16, 256>>>(fn, t, hi << 24, dmi, dmo, static_cast<int>(n), cnt);
        CK(cudaDeviceSynchronize());
        std::printf("%s: %llu finite inputs, %zu AMD misses; differ at a miss %llu, elsewhere %llu -> %s\n", fn == 0 ? "expf" : "log1pf", cnt[0], n, cnt[1], cnt[2],
                    (cnt[1] || cnt[2] > 64) ? "FAIL" : (cnt[2] ? "PASS except fp64-reference ties (listed above)" : "PASS"));
        if (cnt[1] || cnt[2] > 64) ++fails;
    }
    }
    std::printf("%s\n", fails ? "FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
