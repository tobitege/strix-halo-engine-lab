// QSA index block scores: block_score_tree_kernel (default) vs the warp-shuffle block_score_kernel, byte for byte, on random
// and adversarial inputs (signed zeros, infinities, NaN, overflow, subnormals), over the whole score buffer (unwritten entries
// keep a sentinel, so the write set is compared too). --bench: median time of each form at long positions.
//   qwen_qsa_score_tree_test [--bench]
#include "strix/cuda/qwen_qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace q = strix::cuda::qwen_qsa;

#define CK(x)                                                                                 \
    do {                                                                                      \
        cudaError_t e_ = (x);                                                                 \
        if (e_ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::exit(2);                                                                     \
        }                                                                                     \
    } while (0)

static std::uint32_t mix(std::uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return static_cast<std::uint32_t>(x);
}
static float value(std::uint64_t i, std::uint32_t seed, bool adversarial) {
    const std::uint32_t h = mix(i * 0x9E3779B97F4A7C15ULL + seed);
    float v = (static_cast<float>(h & 0xFFFFFF) / 8388608.0F - 1.0F) * 1.5F;
    if (adversarial) {
        switch ((h >> 24) % 64U) {
            case 0: v = 0.0F; break;
            case 1: v = -0.0F; break;
            case 2: v = std::numeric_limits<float>::infinity(); break;
            case 3: v = -std::numeric_limits<float>::infinity(); break;
            case 4: v = std::numeric_limits<float>::quiet_NaN(); break;
            case 5: v = 3.0e38F; break;
            case 6: v = 1.0e-39F; break;
            case 7: v = -1.0e-41F; break;
            default: break;
        }
    }
    return v;
}

struct Case {
    std::uint32_t rows;
    std::uint64_t p0;
    bool adversarial;
};

static double run(const Case& c, bool timing, std::size_t& bad) {
    const std::uint32_t max_blocks = static_cast<std::uint32_t>((c.p0 + c.rows) / q::kRatio) + 1U;
    const std::size_t qn = static_cast<std::size_t>(c.rows) * q::kIndexQWidth, kn = static_cast<std::size_t>(max_blocks) * q::kIndexDim;
    const std::size_t sn = static_cast<std::size_t>(c.rows) * max_blocks;
    std::vector<float> hq(qn), hk(kn);
    for (std::size_t i = 0; i < qn; ++i) hq[i] = value(i, 101U, c.adversarial && i % 7 == 0);
    for (std::size_t i = 0; i < kn; ++i) hk[i] = value(i, 202U, c.adversarial && i % 5 == 0);
    float *dq = nullptr, *dk = nullptr, *ds = nullptr;
    CK(cudaMalloc(&dq, qn * 4));
    CK(cudaMalloc(&dk, kn * 4));
    CK(cudaMalloc(&ds, sn * 4));
    CK(cudaMemcpy(dq, hq.data(), qn * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dk, hk.data(), kn * 4, cudaMemcpyHostToDevice));
    std::vector<float> out[2];
    double ms_form[2] = {0, 0};
    const q::ScoreForm forms[2] = {q::ScoreForm::shuffle, q::ScoreForm::tree};
    for (int f = 0; f < 2; ++f) {
        CK(cudaMemset(ds, 0x7F, sn * 4));
        q::block_scores_form(forms[f], dq, c.rows, c.p0, dk, ds, max_blocks, nullptr);
        CK(cudaDeviceSynchronize());
        out[f].resize(sn);
        CK(cudaMemcpy(out[f].data(), ds, sn * 4, cudaMemcpyDeviceToHost));
    }
    // 16-B alignment admission: the tree form with query rows / pooled keys one float off a 16-B boundary must give the
    // shuffle scorer's bytes (scalar fallback), not a misaligned float4 load
    std::size_t misaligned = 0;
    {
        float *mq = nullptr, *mk = nullptr;
        CK(cudaMalloc(&mq, (qn + 1U) * 4));
        CK(cudaMalloc(&mk, (kn + 1U) * 4));
        CK(cudaMemcpy(mq + 1, hq.data(), qn * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(mk + 1, hk.data(), kn * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(ds, 0x7F, sn * 4));
        q::block_scores_form(q::ScoreForm::tree, mq + 1, c.rows, c.p0, mk + 1, ds, max_blocks, nullptr);
        CK(cudaDeviceSynchronize());
        std::vector<float> mo(sn);
        CK(cudaMemcpy(mo.data(), ds, sn * 4, cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < sn; ++i) misaligned += std::memcmp(&out[0][i], &mo[i], 4) != 0;
        CK(cudaFree(mq));
        CK(cudaFree(mk));
    }
    bad += misaligned;
    if (timing) {
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        for (int rep = 0; rep < 2; ++rep)  // ABAB: shuffle, tree, shuffle, tree; keep each form's last median
            for (int f = 0; f < 2; ++f) {
                std::vector<float> ms(5);
                for (float& m : ms) {
                    CK(cudaEventRecord(e0));
                    q::block_scores_form(forms[f], dq, c.rows, c.p0, dk, ds, max_blocks, nullptr);
                    CK(cudaEventRecord(e1));
                    CK(cudaEventSynchronize(e1));
                    CK(cudaEventElapsedTime(&m, e0, e1));
                }
                std::sort(ms.begin(), ms.end());
                ms_form[f] = ms[2];
            }
    }
    std::size_t diff = 0, written = 0;
    for (std::size_t i = 0; i < sn; ++i) {
        diff += std::memcmp(&out[0][i], &out[1][i], 4) != 0;
        std::uint32_t b;
        std::memcpy(&b, &out[0][i], 4);
        written += b != 0x7F7F7F7FU;
    }
    bad += diff;
    double pairs = 0;
    for (std::uint32_t r = 0; r < c.rows; ++r) {
        const std::uint64_t complete = (c.p0 + r + 1U) / q::kRatio;
        if (complete > q::kTopK) pairs += static_cast<double>(complete);
    }
    std::printf("rows=%u p0=%llu adv=%d written=%zu mismatch=%zu misaligned_mismatch=%zu", c.rows, static_cast<unsigned long long>(c.p0),
                c.adversarial ? 1 : 0, written, diff, misaligned);
    if (timing)
        std::printf(" shuffle_ms=%.3f tree_ms=%.3f ratio=%.3f pairs/ns shuffle %.2f tree %.2f", ms_form[0], ms_form[1],
                    ms_form[1] / ms_form[0], pairs / (ms_form[0] * 1e6), pairs / (ms_form[1] * 1e6));
    std::printf("\n");
    std::fflush(stdout);
    CK(cudaFree(dq));
    CK(cudaFree(dk));
    CK(cudaFree(ds));
    return ms_form[1];
}

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    std::size_t bad = 0;
    const Case checks[] = {{1024, 1024, false}, {1024, 2000, true}, {77, 5000, true}, {1, 9000, false}, {1000, 30001, true},
                           {130, 2047, false}, {64, 2049, true}, {1024, 65536 - 1024, true}};
    for (const Case& c : checks) run(c, false, bad);
    if (bench) {
        const Case b[] = {{1024, 32768 - 1024, false}, {1024, 131072 - 1024, false}, {1024, 262144 - 1024, false}};
        for (const Case& c : b) run(c, true, bad);
    }
    std::printf("qwen_qsa_score_tree_test: %s (%zu mismatching floats)\n", bad == 0 ? "IDENTICAL" : "MISMATCH", bad);
    return bad == 0 ? 0 : 1;
}
