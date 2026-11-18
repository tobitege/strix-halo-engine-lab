// Replay of the HIP indexed_attention step (op 6) of the d64 fixture for all 12 QSA layers: dense A8 q/k/v projection
// (dense family, qwen_dense.cu) -> qwen_qsa layer_core -> dense A8 o projection, compared byte for byte with the step's
// write records and with the final KV / raw-index state (positions 0..63) of d64-state.
// Build: build_qwen_test.bat <wt> qwen_qsa_replay <out> <wt>\src\cuda\models\qwen\qwen_qsa.cu <dense wt>\src\cuda\models\qwen\qwen_dense.cu -I<dense wt>\include
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "qwen_fixture.hpp"
#include "qwen_gguf_host.hpp"
#include "strix/cuda/amd_math_tables.hpp"
#include "strix/cuda/qwen_dense.hpp"
#include "strix/cuda/qwen_qsa.hpp"

namespace q = strix::cuda::qwen_qsa;
namespace qd = strix::cuda::qwen_dense;
namespace am = strix::cuda::amdmath;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

template <class T> static T* up(const void* data, size_t bytes) {
    T* d = nullptr; CK(cudaMalloc(&d, bytes ? bytes : 1)); if (bytes) CK(cudaMemcpy(d, data, bytes, cudaMemcpyHostToDevice)); return d;
}
static std::vector<std::uint8_t> read_file(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "missing %s\n", p.c_str()); std::exit(2); }
    _fseeki64(f, 0, SEEK_END); const auto n = _ftelli64(f); _fseeki64(f, 0, SEEK_SET);
    std::vector<std::uint8_t> v(static_cast<size_t>(n));
    if (std::fread(v.data(), 1, v.size(), f) != v.size()) std::exit(2);
    std::fclose(f);
    return v;
}
// first differing element (as u16) of two byte ranges
static bool cmp16(const char* what, unsigned layer, const std::uint8_t* got, const std::uint8_t* want, size_t bytes, size_t row_bytes) {
    for (size_t i = 0; i + 1 < bytes; i += 2) {
        std::uint16_t a, b; std::memcpy(&a, got + i, 2); std::memcpy(&b, want + i, 2);
        if (a != b) {
            size_t diff = 0;
            for (size_t j = i; j + 1 < bytes; j += 2) diff += std::memcmp(got + j, want + j, 2) != 0;
            std::printf("FAIL layer %u %s: %zu/%zu BF16 differ, first row %zu element %zu got 0x%04x want 0x%04x\n", layer, what, diff,
                bytes / 2, i / row_bytes, (i % row_bytes) / 2, a, b);
            return false;
        }
    }
    std::printf("ok   layer %u %s (%zu bytes)\n", layer, what, bytes);
    return true;
}

int main() {
    const std::string fx = "qrtx-build\\fx\\";
    strix::qwen_fixture::Dump dump(fx + "d64.dump");
    const auto kv_final = read_file(fx + "d64-state\\attention-kv.raw");
    const auto idx_final = read_file(fx + "d64-state\\sparse-index.raw");
    const std::uint64_t capacity = 8192;  // manifest context_capacity (layer slice)
    if (kv_final.size() != 12 * capacity * 2048 || idx_final.size() != 12 * capacity * 256) { std::printf("state size mismatch\n"); return 2; }
    auto cs = read_file(fx + "rope_cs_262144.bin");
    auto tables = am::load_tables("qrtx-build\\tables");
    strix::qwen_gguf::Model m;
    const unsigned rows = 64;
    auto* dcs = up<float2>(cs.data(), 64ULL * 32 * 8);  // positions 0..63 suffice
    float *qg, *k, *v, *iq, *ik, *att;
    CK(cudaMalloc(&qg, rows * 12288ULL * 4)); CK(cudaMalloc(&k, rows * 512ULL * 4)); CK(cudaMalloc(&v, rows * 512ULL * 4));
    CK(cudaMalloc(&iq, rows * 512ULL * 4)); CK(cudaMalloc(&ik, rows * 128ULL * 4)); CK(cudaMalloc(&att, rows * 6144ULL * 4));
    std::uint16_t *kvs, *raws, *out;
    CK(cudaMalloc(&kvs, rows * 2048ULL)); CK(cudaMalloc(&raws, rows * 256ULL)); CK(cudaMalloc(&out, rows * 5120ULL));
    std::int8_t* codes; float* deltas;
    CK(cudaMalloc(&codes, rows * 6144ULL)); CK(cudaMalloc(&deltas, (6144 / 32) * rows * 4ULL));
    auto ws = q::allocate_workspace(rows, rows);
    int fails = 0, passes = 0;
    for (const auto& rec : dump.records()) {
        if (rec.is_read || rec.op != 6 || rec.index != 0 || rec.row != 0) continue;
        const unsigned layer = rec.layer, step = rec.step, ord = layer / 4;
        const auto in = dump.load_rows(step, true, 0), want = dump.load_rows(step, false, 0);
        if (in.size() != rows * 5120ULL || want.size() != rows * 5120ULL) { std::printf("layer %u: unexpected record sizes\n", layer); ++fails; continue; }
        const std::string b = "blk." + std::to_string(layer) + ".";
        const auto wq8 = m.bytes(b + "attn_q.weight"), wk8 = m.bytes(b + "attn_k.weight"), wv8 = m.bytes(b + "attn_v.weight"),
                   wo8 = m.bytes(b + "attn_output.weight"), iqw = m.bytes(b + "indexer.q_proj.weight"), ikw = m.bytes(b + "indexer.k_proj.weight"),
                   qn = m.bytes(b + "attn_q_norm.weight"), kn = m.bytes(b + "attn_k_norm.weight"), iqn = m.bytes(b + "indexer.q_norm.weight"),
                   ikn = m.bytes(b + "indexer.k_norm.weight");
        auto* dwq = up<void>(wq8.data(), wq8.size()); auto* dwk = up<void>(wk8.data(), wk8.size()); auto* dwv = up<void>(wv8.data(), wv8.size());
        auto* dwo = up<void>(wo8.data(), wo8.size());
        auto* diqw = up<std::uint16_t>(iqw.data(), iqw.size()); auto* dikw = up<std::uint16_t>(ikw.data(), ikw.size());
        auto* dqn = up<float>(qn.data(), qn.size()); auto* dkn = up<float>(kn.data(), kn.size());
        auto* diqn = up<float>(iqn.data(), iqn.size()); auto* dikn = up<float>(ikn.data(), ikn.size());
        auto* hid = up<std::uint16_t>(in.data(), in.size());
        CK(cudaMemset(kvs, 0, rows * 2048ULL)); CK(cudaMemset(raws, 0, rows * 256ULL));
        // q/k/v: one A8 site, store round_bf16 into FP32 (HIP QsaInputRowsSite::store, :3494-3498)
        qd::Site in_site{};
        in_site.store = qd::Store::f32_round_bf16; in_site.width = 2560; in_site.segments = 3;
        in_site.segment[0] = {dwq, 12288, qg, 12288}; in_site.segment[1] = {dwk, 512, k, 512}; in_site.segment[2] = {dwv, 512, v, 512};
        CK(qd::quantize_project(nullptr, in_site, rows, qd::Input::bf16, hid, 2560, codes, deltas, rows));
        q::LayerWeights w{diqw, dikw, dqn, dkn, diqn, dikn};
        q::LayerState st{kvs, raws};
        q::RowBuffers rb{hid, qg, k, v, iq, ik, att};
        q::layer_core(rb, rows, 0, w, st, dcs, tables, ws, nullptr);
        // o: A8 site over the FP32 (BF16-valued) attention output, BF16 store (QsaOutputRowsSite, :3512)
        qd::Site out_site{};
        out_site.store = qd::Store::bf16; out_site.width = 6144; out_site.segments = 1;
        out_site.segment[0] = {dwo, 2560, out, 2560};
        CK(qd::quantize_project(nullptr, out_site, rows, qd::Input::f32, att, 6144, codes, deltas, rows));
        CK(cudaDeviceSynchronize());
        std::vector<std::uint8_t> got(rows * 5120ULL), gkv(rows * 2048ULL), graw(rows * 256ULL);
        CK(cudaMemcpy(got.data(), out, got.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gkv.data(), kvs, gkv.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(graw.data(), raws, graw.size(), cudaMemcpyDeviceToHost));
        bool ok = cmp16("step output (5120 B rows)", layer, got.data(), want.data(), got.size(), 5120);
        ok = cmp16("KV state pos 0..63", layer, gkv.data(), kv_final.data() + ord * capacity * 2048, gkv.size(), 2048) && ok;
        ok = cmp16("raw index state pos 0..63", layer, graw.data(), idx_final.data() + ord * capacity * 256, graw.size(), 256) && ok;
        (ok ? passes : fails)++;
        for (void* p : {dwq, dwk, dwv, dwo, static_cast<void*>(diqw), static_cast<void*>(dikw), static_cast<void*>(dqn), static_cast<void*>(dkn),
                        static_cast<void*>(diqn), static_cast<void*>(dikn), static_cast<void*>(hid)}) cudaFree(p);
    }
    std::printf("qwen_qsa_replay: %d layers bit-exact, %d differ\n", passes, fails);
    return fails || passes != 12 ? 1 : 0;
}
