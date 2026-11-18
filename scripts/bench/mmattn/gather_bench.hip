// gather_bench: store-pattern cost of the MiniMax prefill MoE gather + input transform
// (src/minimax/moe/mmx_prefill.hip gather_transform_token_kernel, 4.53 ms per 4096-row call in the 8K pair trace).
//
// The kernel writes, per token, 10 transformed fp16 rows (4 routed pairs + the shared pair, gate and up) into the
// tile-major xt ([R/16][ch/16][R%16][ch%16]): every store instruction writes two 32-byte row segments, and the other
// 15 rows of each tile come from tokens routed to the same expert, ~32 tokens apart. This bench rebuilds that
// routing (random top-4 of 128 experts, groups padded to 16 rows) and times the same per-wave work with:
//   0 incumbent: wave = (token, 128-block), loops pairs k then matrix m (gate, up), tile-major stores
//   1 m-outer:   the grid runs every token for gate first, then every token for up (half the open tile bands)
//   2 floor:     the same values written row-major (contiguous 64-byte lane groups), the layout byte floor
//   3 tile32:    tile-major with 32-channel tiles ([R/16][ch/32][R%16][ch%32], 64-byte row segments)
// usage: gather_bench [tokens=4096] [reps=20]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
constexpr unsigned kHidden = 6144, kHB = kHidden / 128U, kExperts = 128, kTopE = 4, kWaves = 8;

void ck(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}

struct Tab {
    unsigned rows, tpairs;
    const std::uint32_t* tok_pair;    // [T][kTopE] routed pair ids
    const std::uint32_t* pair_group;  // [P]
    const std::uint32_t* g_tbegin;    // [G] padded row start
    const std::uint32_t* g_pbegin;    // [G] first pair
    const _Float16* suh;              // [G][2][kHidden]
    unsigned shared_group;
    _Float16* xt;                     // [2][tpairs][kHidden] (layout per mode)
};

__device__ __forceinline__ std::size_t at16(unsigned R, unsigned ch) {
    return ((static_cast<std::size_t>(R >> 4U) * (kHidden >> 4U) + (ch >> 4U)) << 8U) + ((R & 15U) << 4U) + (ch & 15U);
}
__device__ __forceinline__ std::size_t at32(unsigned R, unsigned ch) {
    return ((static_cast<std::size_t>(R >> 4U) * (kHidden >> 5U) + (ch >> 5U)) << 9U) + ((R & 15U) << 5U) + (ch & 31U);
}

// 128-point Hadamard butterfly over v[4] x 32 lanes (element = lane + 32 j): the work shape of InputTransform::h128.
__device__ __forceinline__ void h128(float (&v)[4], unsigned lane) {
#pragma unroll
    for (unsigned d = 1; d < 32; d <<= 1U) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor(v[j], static_cast<int>(d), 32);
            v[j] = (lane & d) ? o - v[j] : v[j] + o;
        }
    }
    const float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
    v[0] = (a0 + a2) * 0.08838834764F;
    v[1] = (a1 + a3) * 0.08838834764F;
    v[2] = (a0 - a2) * 0.08838834764F;
    v[3] = (a1 - a3) * 0.08838834764F;
}

template <int Mode>
__global__ __launch_bounds__(256) void gather_kernel(const float* y, Tab t) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    unsigned q = blockIdx.x * kWaves + wave, m0 = 0, m1 = 2;
    if constexpr (Mode == 1) {  // m-outer: first half of the grid = gate, second = up
        const unsigned half = t.rows * kHB;
        if (q >= 2 * half) return;
        m0 = q >= half ? 1U : 0U;
        m1 = m0 + 1U;
        q -= m0 * half;
    }
    if (q >= t.rows * kHB) return;
    const unsigned tok = q / kHB, b = q % kHB;
    const float* x = y + static_cast<std::size_t>(tok) * kHidden;
    float xv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) xv[j] = x[b * 128U + lane + 32U * static_cast<unsigned>(j)];
#pragma unroll 1
    for (unsigned k = 0; k <= kTopE; ++k) {
        const unsigned p = k < kTopE ? t.tok_pair[tok * kTopE + k] : t.g_pbegin[t.shared_group] + tok;
        const unsigned g = t.pair_group[p];
        const unsigned R = t.g_tbegin[g] + (p - t.g_pbegin[g]);
        for (unsigned m = m0; m < m1; ++m) {
            const _Float16* suh = t.suh + (static_cast<std::size_t>(g) * 2U + m) * kHidden;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned c = b * 128U + lane + 32U * static_cast<unsigned>(j);
                v[j] = static_cast<float>(static_cast<_Float16>(xv[j] * static_cast<float>(suh[c])));
            }
            h128(v, lane);
            _Float16* out = t.xt + static_cast<std::size_t>(m) * t.tpairs * kHidden;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned c = b * 128U + lane + 32U * static_cast<unsigned>(j);
                const std::size_t o = Mode == 2 ? static_cast<std::size_t>(R) * kHidden + c : Mode == 3 ? at32(R, c) : at16(R, c);
                out[o] = static_cast<_Float16>(v[j]);
            }
        }
    }
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
    const unsigned T = argc > 1 ? static_cast<unsigned>(std::atoi(argv[1])) : 4096U;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    // routing: random distinct top-4 per token (LCG), groups = experts + shared, rows sorted by token, padded to 16
    std::vector<std::vector<unsigned>> grp(kExperts + 1);
    std::vector<std::uint32_t> tok_exp(std::size_t{T} * kTopE);
    std::uint64_t s = 0x9E3779B97F4A7C15ULL;
    auto rnd = [&]() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<unsigned>(s >> 33); };
    for (unsigned tok = 0; tok < T; ++tok) {
        unsigned e[kTopE];
        for (unsigned k = 0; k < kTopE; ++k) {
            bool dup;
            do {
                e[k] = rnd() % kExperts;
                dup = false;
                for (unsigned i = 0; i < k; ++i) dup |= e[i] == e[k];
            } while (dup);
        }
        std::sort(e, e + kTopE);
        for (unsigned k = 0; k < kTopE; ++k) { tok_exp[tok * kTopE + k] = e[k]; grp[e[k]].push_back(tok); }
        grp[kExperts].push_back(tok);
    }
    const unsigned G = kExperts + 1;
    std::vector<std::uint32_t> tbegin(G), pbegin(G), pair_group, tok_pair(std::size_t{T} * kTopE);
    unsigned rowc = 0, pc = 0;
    for (unsigned g = 0; g < G; ++g) {
        tbegin[g] = rowc;
        pbegin[g] = pc;
        for (unsigned i = 0; i < grp[g].size(); ++i) pair_group.push_back(g);
        pc += static_cast<unsigned>(grp[g].size());
        rowc += (static_cast<unsigned>(grp[g].size()) + 15U) & ~15U;
    }
    for (unsigned g = 0; g < kExperts; ++g)
        for (unsigned i = 0; i < grp[g].size(); ++i) {
            const unsigned tok = grp[g][i];
            for (unsigned k = 0; k < kTopE; ++k)
                if (tok_exp[tok * kTopE + k] == g) tok_pair[tok * kTopE + k] = pbegin[g] + i;
        }
    const unsigned tpairs = rowc;
    auto up = [](const auto& v) {
        void* d;
        ck(hipMalloc(&d, v.size() * sizeof(v[0])), "malloc");
        ck(hipMemcpy(d, v.data(), v.size() * sizeof(v[0]), hipMemcpyHostToDevice), "cp");
        return d;
    };
    std::vector<_Float16> suh(std::size_t{G} * 2U * kHidden);
    for (auto& v : suh) v = static_cast<_Float16>((rnd() & 1U) ? 1.0F : -1.0F);
    std::vector<float> yh(std::size_t{T} * kHidden);
    for (auto& v : yh) v = static_cast<float>(static_cast<int>(rnd() % 2001U) - 1000) * 1e-3F;
    Tab t{T, tpairs, static_cast<std::uint32_t*>(up(tok_pair)), static_cast<std::uint32_t*>(up(pair_group)),
          static_cast<std::uint32_t*>(up(tbegin)), static_cast<std::uint32_t*>(up(pbegin)), static_cast<_Float16*>(up(suh)),
          kExperts, nullptr};
    const float* y = static_cast<float*>(up(yh));
    const std::size_t xt_bytes = 2U * std::size_t{tpairs} * kHidden * sizeof(_Float16);
    ck(hipMalloc(reinterpret_cast<void**>(&t.xt), xt_bytes), "xt");
    hipEvent_t e0, e1;
    ck(hipEventCreate(&e0), "ev");
    ck(hipEventCreate(&e1), "ev");
    const double wr_mb = 2.0 * static_cast<double>(T) * (kTopE + 1) * kHidden * 2.0 / 1e6, rd_mb = static_cast<double>(T) * kHidden * 4.0 / 1e6;
    std::printf("gather tokens %u pairs %u padded rows %u: writes %.1f MB, y reads %.1f MB\n", T, pc, tpairs, wr_mb, rd_mb);
    const char* names[4] = {"incumbent", "m-outer", "row-major floor", "tile32"};
    std::vector<double> ms[4];
    for (int i = 0; i < reps + 2; ++i)
        for (int mode = 0; mode < 4; ++mode) {
            const unsigned waves = T * kHB * (mode == 1 ? 2U : 1U);
            const dim3 grid((waves + kWaves - 1) / kWaves);
            ck(hipEventRecord(e0, nullptr), "rec");
            if (mode == 0) hipLaunchKernelGGL(gather_kernel<0>, grid, dim3(256), 0, nullptr, y, t);
            if (mode == 1) hipLaunchKernelGGL(gather_kernel<1>, grid, dim3(256), 0, nullptr, y, t);
            if (mode == 2) hipLaunchKernelGGL(gather_kernel<2>, grid, dim3(256), 0, nullptr, y, t);
            if (mode == 3) hipLaunchKernelGGL(gather_kernel<3>, grid, dim3(256), 0, nullptr, y, t);
            ck(hipEventRecord(e1, nullptr), "rec");
            ck(hipEventSynchronize(e1), "sync");
            float v = 0.0F;
            ck(hipEventElapsedTime(&v, e0, e1), "el");
            if (i >= 2) ms[mode].push_back(v);
        }
    for (int mode = 0; mode < 4; ++mode) {
        const double m = median(ms[mode]);
        std::printf("mode %d %-16s %.3f ms (min %.3f max %.3f) %.0f GB/s effective\n", mode, names[mode], m,
                    *std::min_element(ms[mode].begin(), ms[mode].end()), *std::max_element(ms[mode].begin(), ms[mode].end()),
                    (wr_mb + rd_mb * (mode == 1 ? 2.0 : 1.0)) / m);
    }
    return 0;
}
