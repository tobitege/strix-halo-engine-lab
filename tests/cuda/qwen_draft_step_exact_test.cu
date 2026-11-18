// Byte-identity + timing test: MTP draft/commit step decode kernels, old vs new (rows 1..8).
//   qwen_draft_step_exact_test --tables DIR
// Includes the two .cu files directly to reach their anonymous-namespace kernels. Build: tests/cuda/build_qwen_test.bat
// (sources: this file + src/cuda/core/amd_math_tables.cu; gguf_reader not needed).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "strix/cuda/amd_math_tables.hpp"
#include "../../src/cuda/models/qwen/qwen_plemtp.cu"
#include "../../src/cuda/models/qwen/qwen_mtp_draft.cu"

namespace pm = strix::cuda::qwen_plemtp;
namespace md = strix::cuda::qwen_mtp_draft;
namespace am = strix::cuda::amdmath;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

static int g_pass = 0, g_fail = 0;
static std::mt19937_64 rng(12345);
static void report(bool ok, const std::string& name) { (ok ? g_pass : g_fail)++; if (!ok) std::printf("FAIL %s\n", name.c_str()); }
static float nrm(float sd) { static std::normal_distribution<float> d(0.0F, 1.0F); return d(rng) * sd; }
static std::uint16_t bf(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16) & 1U)) >> 16); }
static std::uint16_t half_bits(float f) { const __half h = __float2half(f); std::uint16_t b; std::memcpy(&b, &h, 2); return b; }

template <class T> static T* dalloc(const std::vector<T>& h) {
    T* d = nullptr; CK(cudaMalloc(&d, h.size() * sizeof(T))); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d;
}
template <class T> static T* dzero(std::size_t n) { T* d = nullptr; CK(cudaMalloc(&d, n * sizeof(T))); CK(cudaMemset(d, 0xA5, n * sizeof(T))); return d; }
template <class T> static std::vector<T> fetch(const T* d, std::size_t n) { std::vector<T> h(n); CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }
template <class T> static bool same(const std::vector<T>& a, const std::vector<T>& b) { return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(T)); }

static float time_us(const std::function<void()>& f) {
    for (int i = 0; i < 10; ++i) f();
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for (int i = 0; i < 50; ++i) f();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b)); CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return ms * 1000.0F / 50.0F;
}
static void timing(const char* name, unsigned rows, const std::function<void()>& o, const std::function<void()>& n) {
    const float to = time_us(o), tn = time_us(n);
    std::printf("TIME %-22s rows %u  old %8.2f us  new %8.2f us\n", name, rows, to, tn);
}

static std::vector<std::uint8_t> q8_blocks(std::size_t blocks, float scale_sd) {
    std::vector<std::uint8_t> v(blocks * 34U);
    for (std::size_t b = 0; b < blocks; ++b) {
        const std::uint16_t s = half_bits(std::fabs(nrm(scale_sd)) + 1e-4F);
        v[b * 34] = static_cast<std::uint8_t>(s & 255U); v[b * 34 + 1] = static_cast<std::uint8_t>(s >> 8U);
        for (int j = 0; j < 32; ++j) v[b * 34 + 2 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(std::clamp(static_cast<int>(nrm(40.0F)), -128, 127)));
    }
    return v;
}

static void test_step(const am::Tables& tb) {
    using namespace pm;
    // weights
    std::vector<std::uint8_t> eh(static_cast<std::size_t>(kHidden) * kMtpEhRowQ4);
    for (std::size_t b = 0; b < eh.size() / 18U; ++b) {
        const std::uint16_t s = half_bits(std::fabs(nrm(0.01F)) + 1e-5F);
        eh[b * 18] = static_cast<std::uint8_t>(s & 255U); eh[b * 18 + 1] = static_cast<std::uint8_t>(s >> 8U);
        for (int j = 0; j < 16; ++j) eh[b * 18 + 2 + j] = static_cast<std::uint8_t>(rng() & 255U);
    }
    auto up = q8_blocks(static_cast<std::size_t>(kHc) * kHidden * (kMtpRank / 32U), 0.02F);
    auto kq = q8_blocks(static_cast<std::size_t>(kMtpKvWidth) * (kHidden / 32U), 0.01F);
    auto vq = q8_blocks(static_cast<std::size_t>(kMtpKvWidth) * (kHidden / 32U), 0.01F);
    std::vector<std::uint16_t> ik(static_cast<std::size_t>(kMtpIndexDim) * kHidden);
    for (auto& x : ik) x = bf(nrm(0.05F));
    std::vector<float> enorm(kHidden), hnorm(kExpanded);
    for (auto& x : enorm) x = std::fabs(nrm(1.0F)) + 0.2F;
    for (auto& x : hnorm) x = std::fabs(nrm(1.0F)) + 0.2F;
    auto* d_eh = dalloc(eh); auto* d_up = dalloc(up); auto* d_kq = dalloc(kq); auto* d_vq = dalloc(vq); auto* d_ik = dalloc(ik);
    auto* d_en = dalloc(enorm); auto* d_hn = dalloc(hnorm);
    for (unsigned variant = 0; variant < 2; ++variant) {
        MtpRounding rd; if (variant) { rd.normalized_bf16 = false; rd.projection_bf16 = false; rd.gemma_offset = true; }
        for (unsigned rows = 1; rows <= 8; ++rows) {
            const std::string tag = "rows " + std::to_string(rows) + " variant " + std::to_string(variant);
            std::vector<std::uint16_t> emb(rows * kHidden), hid(static_cast<std::size_t>(rows) * kExpanded);
            for (auto& x : emb) x = bf(nrm(1.5F));
            for (auto& x : hid) x = bf(nrm(4.0F));
            auto* d_emb = dalloc(emb); auto* d_hid = dalloc(hid);
            // 2. input norms
            auto* n_old = dzero<float>(static_cast<std::size_t>(rows) * (kHidden + kExpanded));
            auto* n_new = dzero<float>(static_cast<std::size_t>(rows) * (kHidden + kExpanded));
            mtp_input_norms_kernel<<<dim3(2, rows), 256>>>(d_emb, d_hid, d_en, d_hn, n_old, rd);
            mtp_input_norms_v2_kernel<<<dim3(2, rows), 256>>>(d_emb, d_hid, d_en, d_hn, n_new, rd);
            CK(cudaDeviceSynchronize());
            const auto no = fetch(n_old, static_cast<std::size_t>(rows) * (kHidden + kExpanded)), nn = fetch(n_new, no.size());
            report(same(no, nn), "input_norms " + tag);
            // 3. fused projection (use normalized from old)
            auto* f_old = dzero<std::uint16_t>(static_cast<std::size_t>(rows) * kExpanded);
            auto* f_new = dzero<std::uint16_t>(static_cast<std::size_t>(rows) * kExpanded);
            mtp_fused_projection_kernel<<<dim3(kHidden / 8U, (rows + kFuseRows - 1U) / kFuseRows), 256>>>(n_old, d_eh, rows, f_old, rd);
            if (rows == 1) mtp_fused_projection_v2_kernel<1><<<dim3(kHidden / 8U, 1), 256>>>(n_old, d_eh, rows, f_new, rd);
            else mtp_fused_projection_v2_kernel<4><<<dim3(kHidden / 8U, (rows + 3U) / 4U), 256>>>(n_old, d_eh, rows, f_new, rd);
            CK(cudaDeviceSynchronize());
            const auto fo = fetch(f_old, static_cast<std::size_t>(rows) * kExpanded), fn = fetch(f_new, fo.size());
            report(same(fo, fn), "fused_projection " + tag);
            // 6. hyper output (hc_normalized random, low random)
            std::vector<float> hcn(static_cast<std::size_t>(rows) * kExpanded), low(static_cast<std::size_t>(rows) * kMtpRank);
            for (auto& x : hcn) x = nrm(1.0F);
            for (auto& x : low) x = nrm(1.5F);
            auto* d_hcn = dalloc(hcn); auto* d_low = dalloc(low);
            auto* h_old = dzero<std::uint16_t>(static_cast<std::size_t>(rows) * kHidden);
            auto* h_new = dzero<std::uint16_t>(static_cast<std::size_t>(rows) * kHidden);
            mtp_hyper_output_kernel<<<dim3(kHidden / 32U, rows), 128>>>(d_hcn, d_low, d_up, h_old, tb);
            mtp_hyper_output_v2_kernel<<<dim3(kHidden / 32U, rows), 128>>>(d_hcn, d_low, d_up, h_new, tb);
            CK(cudaDeviceSynchronize());
            const auto ho = fetch(h_old, static_cast<std::size_t>(rows) * kHidden), hn = fetch(h_new, ho.size());
            report(same(ho, hn), "hyper_output " + tag);
            // 7. cache projection
            const std::size_t ko = static_cast<std::size_t>(rows) * kMtpIndexDim, kv = static_cast<std::size_t>(rows) * kMtpKvWidth;
            auto* io = dzero<float>(ko); auto* ino = dzero<float>(ko); auto* kk = dzero<float>(kv); auto* kn = dzero<float>(kv);
            auto* vo = dzero<float>(kv); auto* vn = dzero<float>(kv);
            const dim3 g(kMtpIndexDim + 2U * kMtpKvWidth, (rows + kLinRows - 1U) / kLinRows);
            mtp_cache_projection_kernel<<<g, 256>>>(h_old, rows, d_ik, d_kq, d_vq, io, kk, vo);
            mtp_cache_projection_v2_kernel<<<g, 256>>>(h_old, rows, d_ik, d_kq, d_vq, ino, kn, vn);
            CK(cudaDeviceSynchronize());
            report(same(fetch(io, ko), fetch(ino, ko)) && same(fetch(kk, kv), fetch(kn, kv)) && same(fetch(vo, kv), fetch(vn, kv)), "cache_projection " + tag);
            if (variant == 0 && (rows == 1 || rows == 4 || rows == 8)) {
                timing("input_norms", rows, [&] { mtp_input_norms_kernel<<<dim3(2, rows), 256>>>(d_emb, d_hid, d_en, d_hn, n_old, rd); },
                    [&] { mtp_input_norms_v2_kernel<<<dim3(2, rows), 256>>>(d_emb, d_hid, d_en, d_hn, n_new, rd); });
                timing("fused_projection", rows, [&] { mtp_fused_projection_kernel<<<dim3(kHidden / 8U, (rows + kFuseRows - 1U) / kFuseRows), 256>>>(n_old, d_eh, rows, f_old, rd); },
                    [&] { if (rows == 1) mtp_fused_projection_v2_kernel<1><<<dim3(kHidden / 8U, 1), 256>>>(n_old, d_eh, rows, f_new, rd);
                          else mtp_fused_projection_v2_kernel<4><<<dim3(kHidden / 8U, (rows + 3U) / 4U), 256>>>(n_old, d_eh, rows, f_new, rd); });
                timing("hyper_output", rows, [&] { mtp_hyper_output_kernel<<<dim3(kHidden / 32U, rows), 128>>>(d_hcn, d_low, d_up, h_old, tb); },
                    [&] { mtp_hyper_output_v2_kernel<<<dim3(kHidden / 32U, rows), 128>>>(d_hcn, d_low, d_up, h_new, tb); });
                timing("cache_projection", rows, [&] { mtp_cache_projection_kernel<<<g, 256>>>(h_old, rows, d_ik, d_kq, d_vq, io, kk, vo); },
                    [&] { mtp_cache_projection_v2_kernel<<<g, 256>>>(h_old, rows, d_ik, d_kq, d_vq, ino, kn, vn); });
            }
            for (void* p : {static_cast<void*>(d_emb), static_cast<void*>(d_hid), static_cast<void*>(n_old), static_cast<void*>(n_new), static_cast<void*>(f_old),
                            static_cast<void*>(f_new), static_cast<void*>(d_hcn), static_cast<void*>(d_low), static_cast<void*>(h_old), static_cast<void*>(h_new),
                            static_cast<void*>(io), static_cast<void*>(ino), static_cast<void*>(kk), static_cast<void*>(kn), static_cast<void*>(vo), static_cast<void*>(vn)}) CK(cudaFree(p));
        }
    }
}

static void test_draft() {
    namespace d = strix::cuda::qwen_mtp_draft;
    constexpr unsigned experts = 16;
    const std::size_t ge = static_cast<std::size_t>(experts) * d::kFfn * (d::kHidden / 32U);
    const std::size_t de = static_cast<std::size_t>(experts) * d::kHidden * (d::kFfn / 32U);
    for (int trial = 0; trial < 3; ++trial) {
        auto gate = q8_blocks(ge, 0.01F), upw = q8_blocks(ge, 0.01F), down = q8_blocks(de, 0.01F);
        std::vector<std::uint16_t> hid(d::kHidden);
        for (auto& x : hid) x = bf(nrm(trial == 2 ? 8.0F : 1.5F));
        std::vector<std::uint32_t> routes(d::kTopK);
        std::vector<unsigned> perm(experts); for (unsigned i = 0; i < experts; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (unsigned i = 0; i < d::kTopK; ++i) routes[i] = perm[i];
        std::vector<float> rw(d::kTopK); for (auto& x : rw) x = std::fabs(nrm(0.3F));
        auto* dg = dalloc(gate); auto* du = dalloc(upw); auto* dd = dalloc(down); auto* dh = dalloc(hid); auto* dr = dalloc(routes); auto* dw = dalloc(rw);
        auto* so = dzero<float>(d::kTopK * d::kFfn); auto* sn = dzero<float>(d::kTopK * d::kFfn);
        auto* oo = dzero<std::uint16_t>(d::kHidden); auto* on = dzero<std::uint16_t>(d::kHidden);
        d::gate_up_kernel<<<dim3(d::kFfn / 8U, d::kTopK), 256>>>(dh, dr, dg, du, so);
        d::gate_up_v2_kernel<<<dim3(d::kFfn / 4U, d::kTopK), 128>>>(dh, dr, dg, du, sn);
        CK(cudaDeviceSynchronize());
        report(same(fetch(so, d::kTopK * d::kFfn), fetch(sn, d::kTopK * d::kFfn)), "gate_up trial " + std::to_string(trial));
        d::down_kernel<<<d::kHidden / 8U, 256>>>(dr, dw, dd, so, oo);
        d::down_v2_kernel<<<d::kHidden / 4U, 128>>>(dr, dw, dd, so, on);
        CK(cudaDeviceSynchronize());
        report(same(fetch(oo, d::kHidden), fetch(on, d::kHidden)), "down trial " + std::to_string(trial));
        if (trial == 0) {
            timing("draft gate_up", 1, [&] { d::gate_up_kernel<<<dim3(d::kFfn / 8U, d::kTopK), 256>>>(dh, dr, dg, du, so); },
                [&] { d::gate_up_v2_kernel<<<dim3(d::kFfn / 4U, d::kTopK), 128>>>(dh, dr, dg, du, sn); });
            timing("draft down", 1, [&] { d::down_kernel<<<d::kHidden / 8U, 256>>>(dr, dw, dd, so, oo); },
                [&] { d::down_v2_kernel<<<d::kHidden / 4U, 128>>>(dr, dw, dd, so, on); });
        }
        for (void* p : {static_cast<void*>(dg), static_cast<void*>(du), static_cast<void*>(dd), static_cast<void*>(dh), static_cast<void*>(dr), static_cast<void*>(dw),
                        static_cast<void*>(so), static_cast<void*>(sn), static_cast<void*>(oo), static_cast<void*>(on)}) CK(cudaFree(p));
    }
    // select
    for (std::uint32_t count : {1U, 7U, 1000U, 16384U, 32768U, 33000U, 73001U, 150000U}) {
        for (int mode = 0; mode < 3; ++mode) {
            std::vector<float> lg(count); std::vector<std::uint32_t> ids(count);
            for (std::uint32_t i = 0; i < count; ++i) { lg[i] = nrm(mode == 1 ? 6.0F : 2.0F); ids[i] = static_cast<std::uint32_t>(rng() % 150000U); }
            if (mode == 2 && count > 4) { const float mx = *std::max_element(lg.begin(), lg.end()); lg[count - 1] = mx; lg[count / 2] = mx; lg[1] = mx; }
            auto* dl = dalloc(lg); auto* di = dalloc(ids);
            auto* oo = dzero<std::uint32_t>(2); auto* on = dzero<std::uint32_t>(2); auto* to = dzero<std::uint32_t>(1); auto* tn = dzero<std::uint32_t>(1);
            d::select_kernel<<<1, 1024>>>(dl, count, di, oo, to);
            d::select_v2_kernel<<<1, 1024>>>(dl, count, di, on, tn);
            CK(cudaDeviceSynchronize());
            report(same(fetch(oo, 2), fetch(on, 2)) && same(fetch(to, 1), fetch(tn, 1)), "select count " + std::to_string(count) + " mode " + std::to_string(mode));
            if (mode == 0 && (count == 33000U || count == 73001U || count == 150000U))
                timing(("select n=" + std::to_string(count)).c_str(), 1, [&] { d::select_kernel<<<1, 1024>>>(dl, count, di, oo, to); },
                    [&] { d::select_v2_kernel<<<1, 1024>>>(dl, count, di, on, tn); });
            for (void* p : {static_cast<void*>(dl), static_cast<void*>(di), static_cast<void*>(oo), static_cast<void*>(on), static_cast<void*>(to), static_cast<void*>(tn)}) CK(cudaFree(p));
        }
    }
}

int main(int argc, char** argv) {
    std::string tables = "D:\\tmp\\qrtxpf-build\\tables";
    for (int i = 1; i + 1 < argc; i += 2) if (std::string(argv[i]) == "--tables") tables = argv[i + 1];
    CK(cudaSetDevice(0));
    int clk = 0; CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
    std::printf("device clock attr %d kHz\n", clk);
    const am::Tables tb = am::load_tables(tables);
    test_step(tb);
    test_draft();
    std::printf("PASS %d FAIL %d\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
