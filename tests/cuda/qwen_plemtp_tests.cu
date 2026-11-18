// Bit-exact tests of the CUDA Qwen PLE step and MTP cache rows (src/cuda/models/qwen/qwen_plemtp.cu) against literal host
// references of the HIP kernels and, for the host PLE code, against the engine's own C++ (src/ple_lookup.cpp).
//   qwen_plemtp_tests [--tables DIR] [--gguf PATH] [--fixture DIR]
// Build: tests\cuda\build_qwen_test.bat <wt> qwen_plemtp_tests <out> <wt>\src\cuda\models\qwen\qwen_plemtp.cu <wt>\src\ple_lookup.cpp
//        <wt>\src\model_contract.cpp <wt>\src\iq4_nl.cpp <wt>\src\model_runtime_binding.cpp
//        <wt>\tests\cuda\qwen_ple_lookup_link_stubs.cpp
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

#include "strix/cuda/qwen_plemtp.hpp"
#include "strix/iq4_nl.hpp"
#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/ple_lookup.hpp"
#include "strix/cuda/amd_math_tables.hpp"
#include "qwen_fixture.hpp"
#include "qwen_gguf_host.hpp"

namespace pm = strix::cuda::qwen_plemtp;
namespace am = strix::cuda::amdmath;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

static int g_pass = 0, g_fail = 0;
static void report(bool ok, const std::string& name, const std::string& detail = {}) {
    (ok ? g_pass : g_fail)++;
    std::printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(), detail.empty() ? "" : ": ", detail.c_str());
}
static std::uint32_t fbits(float f) { std::uint32_t b; std::memcpy(&b, &f, 4); return b; }
static float bitsf(std::uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
static float bf(std::uint16_t b) { return bitsf(static_cast<std::uint32_t>(b) << 16U); }
static std::uint16_t f2bf(float v) { const auto b = fbits(v); return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U); }
// single IEEE operations on the host (explicit, no contraction)
static float fadd(float a, float b) { volatile float r = a + b; return r; }
static float fmul(float a, float b) { volatile float r = a * b; return r; }
static float fdiv(float a, float b) { volatile float r = a / b; return r; }
static float ffma(float a, float b, float c) { return std::fmaf(a, b, c); }

template <class T> static T* dev(const std::vector<T>& v) {
    T* d = nullptr; CK(cudaMalloc(&d, std::max<size_t>(v.size(), 1) * sizeof(T)));
    if (!v.empty()) CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> static std::vector<T> host(const T* d, size_t n) {
    std::vector<T> v(n); CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
}
// first differing element (bitwise), or -1
template <class T> static long long first_diff(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return 0;
    for (size_t i = 0; i < a.size(); ++i) if (std::memcmp(&a[i], &b[i], sizeof(T))) return static_cast<long long>(i);
    return -1;
}
template <class T> static void compare(const std::string& name, const std::vector<T>& got, const std::vector<T>& want) {
    const auto i = first_diff(got, want);
    if (i < 0) { report(true, name, std::to_string(got.size()) + " values bit-exact"); return; }
    char buf[256];
    std::uint64_t g = 0, w = 0;
    if (static_cast<size_t>(i) < got.size()) std::memcpy(&g, &got[i], sizeof(T));
    if (static_cast<size_t>(i) < want.size()) std::memcpy(&w, &want[i], sizeof(T));
    size_t n = 0;
    for (size_t k = 0; k < std::min(got.size(), want.size()); ++k) n += std::memcmp(&got[k], &want[k], sizeof(T)) != 0;
    std::snprintf(buf, sizeof(buf), "size %zu/%zu, %zu differ, first at %lld got 0x%llx want 0x%llx", got.size(), want.size(), n, i,
        static_cast<unsigned long long>(g), static_cast<unsigned long long>(w));
    report(false, name, buf);
}

// ---------------------------------------------------------------------------------------------------------------
// Device oracle for the gfx1151 transcendentals used by the host references (amd_* are verified separately).
static am::Tables g_tables{};
__global__ void oracle_kernel(int op, const float* in, float* out, int n, am::Tables t) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = op == 0 ? am::amd_rsqrtf(t, in[i]) : am::amd_expf(t, in[i]);
}
static std::vector<float> oracle(int op, const std::vector<float>& in) {
    if (in.empty()) return {};
    float* d = dev(in); float* o = nullptr; CK(cudaMalloc(&o, in.size() * 4));
    oracle_kernel<<<static_cast<unsigned>((in.size() + 255) / 256), 256>>>(op, d, o, static_cast<int>(in.size()), g_tables);
    CK(cudaGetLastError());
    auto r = host(o, in.size()); CK(cudaFree(d)); CK(cudaFree(o)); return r;
}

// ---------------------------------------------------------------------------------------------------------------
// PLE hashing: the port vs the engine's build_ple_hash_contract / hash_ple_token.
static void test_ple_hash() {
    const auto engine = strix::build_ple_hash_contract(strix::ModelContract{});
    const auto port = pm::ple_hash_contract();
    bool same = engine.minimum_table_rows == port.minimum_table_rows && engine.head_count == 16 && engine.ngram_size == 3 &&
        engine.eos_token_id == pm::kPleEos;
    for (int i = 0; i < 3; ++i) same = same && engine.multipliers[i] == port.multipliers[i];
    for (int h = 0; h < 16; ++h) same = same && engine.head_offsets[h] == port.head_offsets[h] && engine.head_vocab_sizes[h] == port.head_vocab_sizes[h];
    report(same && port.minimum_table_rows <= pm::kPleTableRows, "ple hash contract == engine",
        "rows " + std::to_string(port.minimum_table_rows));
    std::mt19937_64 rng(42);
    std::uint64_t rows_checked = 0, mismatches = 0;
    for (int seq = 0; seq < 2000; ++seq) {
        const std::uint32_t n = 1 + static_cast<std::uint32_t>(rng() % 200);
        std::vector<std::uint32_t> tokens(n);
        const int mode = seq % 4;   // 0 random, 1 EOS-heavy, 2 tiny vocab (repeats), 3 boundary ids
        for (auto& t : tokens) {
            const auto r = rng();
            if (mode == 1 && r % 5 == 0) t = pm::kPleEos;
            else if (mode == 2) t = static_cast<std::uint32_t>(r % 7);
            else if (mode == 3) { const std::uint32_t v[] = {0, 1, 248043, 248044, 248045, 248319}; t = v[r % 6]; }
            else t = static_cast<std::uint32_t>(r % 248320);
        }
        // split the sequence into blocks at random positions (history carried as the session word)
        const std::uint64_t position0 = (seq % 3 == 0) ? 0 : 1 + rng() % 1000;
        std::uint32_t history[2] = {static_cast<std::uint32_t>(rng() % 248320), (seq % 5 == 0) ? pm::kPleEos : static_cast<std::uint32_t>(rng() % 248320)};
        std::uint32_t eprev[2] = {history[0], history[1]};
        if (position0 == 0) eprev[0] = eprev[1] = pm::kPleEos;
        std::vector<std::uint64_t> ids(static_cast<size_t>(n) * 16);
        std::uint32_t at = 0;
        while (at < n) {
            const std::uint32_t len = std::min<std::uint32_t>(n - at, 1 + static_cast<std::uint32_t>(rng() % 70));
            pm::ple_hash_rows(port, tokens.data() + at, len, position0 + at, history, ids.data() + static_cast<size_t>(at) * 16);
            at += len;
        }
        for (std::uint32_t r = 0; r < n; ++r) {
            const auto h = strix::hash_ple_token(engine, tokens[r], std::span<const std::uint32_t>(eprev, 2));
            for (int k = 0; k < 16; ++k) mismatches += h.row_ids[k] != ids[static_cast<size_t>(r) * 16 + k];
            eprev[0] = h.next_context[0]; eprev[1] = h.next_context[1];
            ++rows_checked;
        }
        mismatches += eprev[0] != history[0] || eprev[1] != history[1];
    }
    report(mismatches == 0, "ple hash rows == engine hash_ple_token",
        std::to_string(rows_checked) + " rows of 2000 sequences, " + std::to_string(mismatches) + " mismatches");
}

// ---------------------------------------------------------------------------------------------------------------
// IQ4_NL gather: port (host threads and device kernel over mapped memory) vs the engine's FileMappedPleLookupBackend.
static std::vector<std::uint16_t> engine_gather(const std::vector<std::uint8_t>& table, std::uint64_t table_rows,
    const std::vector<std::uint64_t>& ids, std::uint32_t rows) {
    strix::ManifestTensor m;
    m.name = "per_layer_token_embd.weight"; m.dimensions = {160, table_rows}; m.ggml_type = 20;
    m.byte_size = table_rows * 90; m.role = strix::TensorRole::ngram_table;
    strix::RuntimeTensorBinding b;
    b.storage = strix::RuntimeTensorStorage::file_mapped; b.address = reinterpret_cast<std::uintptr_t>(table.data()); b.bytes = table.size();
    strix::FileMappedPleLookupBackend backend;
    std::vector<std::uint16_t> out(static_cast<size_t>(rows) * 2560);
    for (std::uint32_t r = 0; r < rows; ++r) {
        strix::PleLookupRequest q;
        q.row_ids = std::span<const std::uint64_t>(ids.data() + static_cast<size_t>(r) * 16, 16);
        q.row_bytes = 320; q.table_manifest = &m; q.table_runtime = &b;
        const auto h = backend.start(q);
        backend.finish(h, std::span<std::byte>(reinterpret_cast<std::byte*>(out.data() + static_cast<size_t>(r) * 2560), 5120));
    }
    return out;
}
static void gather_case(const std::string& name, const std::vector<std::uint8_t>& table, std::uint64_t table_rows,
    const std::vector<std::uint64_t>& ids, std::uint32_t rows) {
    const auto want = engine_gather(table, table_rows, ids, rows);
    std::vector<std::uint16_t> got(want.size());
    pm::ple_gather_iq4nl_host(table.data(), table_rows, ids.data(), rows, got.data(), 2560, 8);
    compare(name + " host threads == engine", got, want);
    // device kernel reading the table through mapped pinned host memory
    void* pinned = nullptr;
    CK(cudaHostAlloc(&pinned, table.size(), cudaHostAllocMapped));
    std::memcpy(pinned, table.data(), table.size());
    void* view = nullptr; CK(cudaHostGetDevicePointer(&view, pinned, 0));
    auto* d_ids = dev(ids);
    std::uint16_t* d_out = nullptr; CK(cudaMalloc(&d_out, want.size() * 2));
    std::uint32_t* d_status = dev(std::vector<std::uint32_t>{0});
    pm::launch_ple_gather_iq4nl(static_cast<const std::uint8_t*>(view), table_rows, d_ids, rows, d_out, 2560, d_status, nullptr);
    CK(cudaDeviceSynchronize());
    compare(name + " device mapped gather == engine", host(d_out, want.size()), want);
    report(host(d_status, 1)[0] == 0, name + " device gather status clean");
    CK(cudaFree(d_ids)); CK(cudaFree(d_out)); CK(cudaFree(d_status)); CK(cudaFreeHost(pinned));
}
static void test_ple_gather(const std::string& gguf_path) {
    // synthetic table: random codes, random finite FP16 scales including subnormals, zero, negative, extreme
    std::mt19937_64 rng(7);
    const std::uint64_t table_rows = 5000;
    std::vector<std::uint8_t> table(table_rows * 90);
    for (auto& b : table) b = static_cast<std::uint8_t>(rng());
    for (std::uint64_t r = 0; r < table_rows; ++r)
        for (int k = 0; k < 5; ++k) {
            std::uint16_t s = static_cast<std::uint16_t>(rng());
            if ((s & 0x7c00U) == 0x7c00U) s &= 0xbfffU;   // keep finite
            const int pick = static_cast<int>(rng() % 16);
            if (pick == 0) s = 0x0000; else if (pick == 1) s = 0x8000; else if (pick == 2) s = 0x0001; else if (pick == 3) s = 0x7bff;
            std::memcpy(&table[r * 90 + k * 18], &s, 2);
        }
    const std::uint32_t rows = 96;
    std::vector<std::uint64_t> ids(static_cast<size_t>(rows) * 16);
    for (auto& i : ids) i = rng() % table_rows;
    gather_case("ple gather synthetic", table, table_rows, ids, rows);
    // real rows of per_layer_token_embd (a few hashed rows of real tokens, read row by row from the GGUF on C:)
    try {
        strix::qwen_gguf::Model model(gguf_path);
        const auto& t = model.tensor("per_layer_token_embd.weight");
        const auto contract = pm::ple_hash_contract();
        const std::uint32_t tokens[] = {248044, 9707, 11, 1879, 0, 248319, 314, 314, 314, 248044, 77, 12345, 220, 13, 198, 151643};
        const std::uint32_t n = 16;
        std::uint32_t history[2] = {0, 0};
        std::vector<std::uint64_t> real_ids(static_cast<size_t>(n) * 16);
        pm::ple_hash_rows(contract, tokens, n, 0, history, real_ids.data());
        // compact table of only the referenced rows, ids remapped
        std::vector<std::uint8_t> small(static_cast<size_t>(n) * 16 * 90);
        std::vector<std::uint64_t> local(real_ids.size());
        FILE* f = std::fopen(gguf_path.empty() ? "Qwen3.8-Flash-Next-UD-Q4_K_XL-merged.gguf" : gguf_path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open gguf");
        for (size_t i = 0; i < real_ids.size(); ++i) {
            _fseeki64(f, static_cast<long long>(t.absolute_offset + real_ids[i] * 90), SEEK_SET);
            if (std::fread(small.data() + i * 90, 1, 90, f) != 90) throw std::runtime_error("short table read");
            local[i] = i;
        }
        std::fclose(f);
        gather_case("ple gather real GGUF rows", small, real_ids.size(), local, n);
    } catch (const std::exception& e) { report(false, "ple gather real GGUF rows", e.what()); }
}

// ---------------------------------------------------------------------------------------------------------------
// Q8_0 projection: literal host reference of project_wave8_q8_bf16_to_fp32_rows_kernel (hip_ple_launcher.hip:496-538).
static std::vector<float> ref_project(const std::vector<std::uint16_t>& x, std::uint32_t rows, std::uint32_t stride,
    const std::vector<std::uint8_t>& w, std::uint32_t width, std::uint32_t outputs) {
    std::vector<float> out(static_cast<size_t>(rows) * outputs);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t o = 0; o < outputs; ++o) {
            float partial[32][8] = {};
            const std::uint8_t* row = w.data() + static_cast<size_t>(o) * (width / 32) * 34;
            for (std::uint32_t chunk = 0; chunk < width; chunk += 256)
                for (std::uint32_t lane = 0; lane < 32; ++lane) {
                    const std::uint32_t feature = chunk + 8 * lane;
                    const std::uint8_t* blk = row + (feature / 32) * 34;
                    std::uint16_t sb; std::memcpy(&sb, blk, 2);
                    const float d = strix::iq4_nl::fp16_bits_to_float(sb);
                    for (std::uint32_t l = 0; l < 8; ++l) {
                        const float weight = fmul(d, static_cast<float>(static_cast<std::int8_t>(blk[2 + feature % 32 + l])));
                        partial[lane][l] = ffma(bf(x[static_cast<size_t>(r) * stride + feature + l]), weight, partial[lane][l]);
                    }
                }
            for (std::uint32_t dist = 16; dist > 0; dist /= 2)
                for (std::uint32_t lane = 0; lane < dist; ++lane)
                    for (std::uint32_t l = 0; l < 8; ++l) partial[lane][l] = fadd(partial[lane][l], partial[lane + dist][l]);
            float* p = partial[0];
            p[0] = fadd(p[0], p[4]); p[1] = fadd(p[1], p[5]); p[2] = fadd(p[2], p[6]); p[3] = fadd(p[3], p[7]);
            p[0] = fadd(p[0], p[2]); p[1] = fadd(p[1], p[3]);
            p[0] = fadd(p[0], p[1]);
            out[static_cast<size_t>(r) * outputs + o] = p[0];
        }
    return out;
}
static void project_case(const std::string& name, const std::vector<std::uint16_t>& x, std::uint32_t rows,
    const std::vector<std::uint8_t>& w, std::uint32_t width, std::uint32_t outputs, std::uint32_t ref_outputs) {
    // reference on the first ref_outputs outputs (host cost), device on all
    auto* dx = dev(x); auto* dw = dev(w);
    float* dout = nullptr; CK(cudaMalloc(&dout, static_cast<size_t>(rows) * outputs * 4));
    pm::launch_ple_project(dx, width, rows, dw, width, outputs, dout, outputs, nullptr);
    CK(cudaDeviceSynchronize());
    const auto all = host(dout, static_cast<size_t>(rows) * outputs);
    std::vector<std::uint8_t> wsub(w.begin(), w.begin() + static_cast<size_t>(ref_outputs) * (width / 32) * 34);
    const auto want = ref_project(x, rows, width, wsub, width, ref_outputs);
    std::vector<float> got(static_cast<size_t>(rows) * ref_outputs);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t o = 0; o < ref_outputs; ++o) got[static_cast<size_t>(r) * ref_outputs + o] = all[static_cast<size_t>(r) * outputs + o];
    compare(name, got, want);
    CK(cudaFree(dx)); CK(cudaFree(dw)); CK(cudaFree(dout));
}
static std::vector<std::uint8_t> random_q8(std::mt19937_64& rng, std::uint32_t outputs, std::uint32_t width, bool extremes) {
    std::vector<std::uint8_t> w(static_cast<size_t>(outputs) * (width / 32) * 34);
    for (size_t b = 0; b < w.size() / 34; ++b) {
        std::uint16_t s = static_cast<std::uint16_t>(0x2000 + rng() % 0x1800);   // ~2^-7 .. 2^-1
        if (extremes) { const int k = static_cast<int>(rng() % 8); if (k == 0) s = 0; else if (k == 1) s = 0x0001; else if (k == 2) s = 0x7bff; else if (k == 3) s |= 0x8000; }
        std::memcpy(&w[b * 34], &s, 2);
        for (int i = 0; i < 32; ++i) w[b * 34 + 2 + i] = static_cast<std::uint8_t>(extremes && rng() % 9 == 0 ? (rng() % 2 ? 0x81 : 0x7f) : rng());
    }
    return w;
}
static std::vector<std::uint16_t> random_bf16(std::mt19937_64& rng, size_t n, bool extremes) {
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<std::uint16_t> v(n);
    for (auto& e : v) {
        float f = nd(rng);
        if (extremes) { const int k = static_cast<int>(rng() % 10); if (k == 0) f = 0.0F; else if (k == 1) f = -0.0F; else if (k == 2) f *= 1e30F; else if (k == 3) f *= 1e-38F; }
        e = f2bf(f);
    }
    return v;
}
static void test_ple_project(const std::string& gguf_path) {
    std::mt19937_64 rng(11);
    {
        const std::uint32_t rows = 13, width = 2560, outputs = 300;
        project_case("ple projection synthetic (13 rows, 300 outputs)", random_bf16(rng, static_cast<size_t>(rows) * width, false), rows,
            random_q8(rng, outputs, width, false), width, outputs, outputs);
        project_case("ple projection extremes (zeros, -0, huge, tiny, subnormal scales)",
            random_bf16(rng, static_cast<size_t>(rows) * width, true), rows, random_q8(rng, outputs, width, true), width, outputs, outputs);
        std::vector<std::uint16_t> negz(static_cast<size_t>(4) * width, 0x8000);
        project_case("ple projection all -0 inputs", negz, 4, random_q8(rng, 64, width, false), width, 64, 64);
    }
    try {
        strix::qwen_gguf::Model model(gguf_path);
        const auto key = model.bytes("blk.1.ple_key.weight"), value = model.bytes("blk.1.ple_value.weight");
        const std::uint32_t rows = 64;
        const auto x = random_bf16(rng, static_cast<size_t>(rows) * 2560, false);
        project_case("ple key projection real weights (64 rows, ref on 512 of 10240)", x, rows, key, 2560, 10240, 512);
        project_case("ple value projection real weights (64 rows, ref on 512 of 2560)", x, rows, value, 2560, 2560, 512);
    } catch (const std::exception& e) { report(false, "ple projection real weights", e.what()); }
}

// ---------------------------------------------------------------------------------------------------------------
// Rows-form tail: literal host reference of ple_rows_gate_norm / conv_residual / conv_state (hip_ple_launcher.hip:882-1021).
struct TailData {
    std::uint32_t rows;
    std::vector<float> keys, values, nk, nq, nc, conv;
    std::vector<std::uint16_t> input, state;
};
static float host_tree256(std::vector<float> s) {
    for (std::uint32_t stride = 128; stride > 0; stride >>= 1)
        for (std::uint32_t l = 0; l < stride; ++l) s[l] = fadd(s[l], s[l + stride]);
    return s[0];
}
static void ref_tail(const TailData& d, float eps, std::vector<std::uint16_t>& out, std::vector<std::uint16_t>& state,
    std::vector<float>& keys, std::vector<float>& gates) {
    const std::uint32_t R = d.rows, H = 2560;
    keys = d.keys; state = d.state; gates.assign(static_cast<size_t>(R) * 4, 0.0F);
    out.assign(static_cast<size_t>(R) * 10240, 0);
    const float hf = 2560.0F;
    // phase 1: sums of squares
    std::vector<float> rk(R * 4), rq(R * 4);
    for (std::uint32_t r = 0; r < R; ++r)
        for (std::uint32_t s = 0; s < 4; ++s) {
            std::vector<float> a(256, 0.0F), b(256, 0.0F);
            for (std::uint32_t lane = 0; lane < 256; ++lane)
                for (std::uint32_t f = lane; f < H; f += 256) {
                    const float k = keys[static_cast<size_t>(r) * 10240 + s * H + f], q = bf(d.input[static_cast<size_t>(r) * 10240 + s * H + f]);
                    a[lane] = ffma(k, k, a[lane]); b[lane] = ffma(q, q, b[lane]);
                }
            rk[r * 4 + s] = fadd(fdiv(host_tree256(a), hf), eps); rq[r * 4 + s] = fadd(fdiv(host_tree256(b), hf), eps);
        }
    const auto ik = oracle(0, rk), iq = oracle(0, rq);
    std::vector<float> negsr(R * 4);
    for (std::uint32_t r = 0; r < R; ++r)
        for (std::uint32_t s = 0; s < 4; ++s) {
            std::vector<float> a(256, 0.0F);
            for (std::uint32_t lane = 0; lane < 256; ++lane)
                for (std::uint32_t f = lane; f < H; f += 256) {
                    const size_t i = static_cast<size_t>(s) * H + f;
                    const float nk = fmul(fmul(keys[static_cast<size_t>(r) * 10240 + i], ik[r * 4 + s]), d.nk[i]);
                    const float nq = fmul(fmul(bf(d.input[static_cast<size_t>(r) * 10240 + i]), iq[r * 4 + s]), d.nq[i]);
                    a[lane] = ffma(nk, nq, a[lane]);
                }
            const float score = fdiv(host_tree256(a), std::sqrt(hf));   // sqrtf correctly rounded
            const float mag = std::sqrt(std::max(std::fabs(score), 1e-6F));
            const float sr = score > 0.0F ? mag : score < 0.0F ? -mag : 0.0F;
            negsr[r * 4 + s] = -sr;
        }
    const auto ex = oracle(1, negsr);
    std::vector<float> rms(R * 4);
    for (std::uint32_t r = 0; r < R; ++r)
        for (std::uint32_t s = 0; s < 4; ++s) {
            const float g = fdiv(1.0F, fadd(1.0F, ex[r * 4 + s]));
            gates[r * 4 + s] = g;
            std::vector<float> a(256, 0.0F);
            for (std::uint32_t lane = 0; lane < 256; ++lane)
                for (std::uint32_t f = lane; f < H; f += 256) { const float gv = fmul(d.values[static_cast<size_t>(r) * H + f], g); a[lane] = ffma(gv, gv, a[lane]); }
            rms[r * 4 + s] = fadd(fdiv(host_tree256(a), hf), eps);
        }
    const auto ir = oracle(0, rms);
    for (std::uint32_t r = 0; r < R; ++r)
        for (std::uint32_t s = 0; s < 4; ++s)
            for (std::uint32_t f = 0; f < H; ++f) {
                const float gv = fmul(d.values[static_cast<size_t>(r) * H + f], gates[r * 4 + s]);
                keys[static_cast<size_t>(r) * 10240 + s * H + f] = fmul(fmul(gv, ir[r * 4 + s]), d.nc[s * H + f]);
            }
    // conv residual
    std::vector<float> convs(static_cast<size_t>(R) * 10240), negc(convs.size());
    for (std::uint32_t r = 0; r < R; ++r)
        for (std::uint32_t c = 0; c < 10240; ++c) {
            float conv = 0.0F;
            for (std::uint32_t tap = 0; tap < 4; ++tap) {
                const std::uint32_t lag = (3 - tap) * 3;
                float x;
                if (lag == 0) x = keys[static_cast<size_t>(r) * 10240 + c];
                else if (r >= lag) x = bf(f2bf(keys[static_cast<size_t>(r - lag) * 10240 + c]));
                else x = bf(d.state[static_cast<size_t>(c) * 9 + 9 + r - lag]);
                conv = ffma(x, d.conv[static_cast<size_t>(c) * 4 + tap], conv);
            }
            convs[static_cast<size_t>(r) * 10240 + c] = conv; negc[static_cast<size_t>(r) * 10240 + c] = -conv;
        }
    const auto ec = oracle(1, negc);
    for (std::uint32_t r = 0; r < R; ++r)
        for (std::uint32_t c = 0; c < 10240; ++c) {
            const size_t i = static_cast<size_t>(r) * 10240 + c;
            const float gated = fmul(d.values[static_cast<size_t>(r) * H + c % H], gates[r * 4 + c / H]);
            const float silu = fdiv(convs[i], fadd(1.0F, ec[i]));
            out[i] = f2bf(fadd(fadd(bf(d.input[i]), gated), silu));
        }
    for (std::uint32_t c = 0; c < 10240; ++c)
        for (std::uint32_t i = 0; i < 9; ++i) {
            const std::uint32_t src = R + i;
            state[static_cast<size_t>(c) * 9 + i] = src < 9 ? state[static_cast<size_t>(c) * 9 + src] : f2bf(keys[static_cast<size_t>(src - 9) * 10240 + c]);
        }
}
static void tail_case(const std::string& name, const TailData& d, bool in_place) {
    std::vector<std::uint16_t> want_out, want_state; std::vector<float> want_keys, want_gates;
    ref_tail(d, 1e-6F, want_out, want_state, want_keys, want_gates);
    pm::PleTailArgs a{};
    float* keys = dev(d.keys); float* values = dev(d.values);
    std::uint16_t* input = dev(d.input); std::uint16_t* state = dev(d.state);
    std::uint16_t* output = input;
    if (!in_place) CK(cudaMalloc(&output, d.input.size() * 2));
    float *nk = dev(d.nk), *nq = dev(d.nq), *nc = dev(d.nc), *conv = dev(d.conv), *gates = nullptr;
    CK(cudaMalloc(&gates, static_cast<size_t>(d.rows) * 16));
    a.keys = keys; a.key_stride = 10240; a.values = values; a.value_stride = 2560; a.input = input; a.input_stride = 10240;
    a.output = output; a.output_stride = 10240; a.conv_state = state; a.norm_key = nk; a.norm_query = nq; a.norm_conv = nc; a.conv = conv;
    a.gates = gates; a.rows = d.rows; a.epsilon = 1e-6F;
    pm::launch_ple_tail(a, g_tables, nullptr);
    CK(cudaDeviceSynchronize());
    compare(name + " gates", host(gates, static_cast<size_t>(d.rows) * 4), want_gates);
    compare(name + " normalized", host(keys, d.keys.size()), want_keys);
    compare(name + " output", host(output, d.input.size()), want_out);
    compare(name + " conv state", host(state, d.state.size()), want_state);
    for (void* p : {static_cast<void*>(keys), static_cast<void*>(values), static_cast<void*>(input), static_cast<void*>(state),
             static_cast<void*>(nk), static_cast<void*>(nq), static_cast<void*>(nc), static_cast<void*>(conv), static_cast<void*>(gates)}) CK(cudaFree(p));
    if (!in_place) CK(cudaFree(output));
}
static std::vector<float> f32_tensor(strix::qwen_gguf::Model& m, const std::string& name) {
    const auto b = m.bytes(name); std::vector<float> v(b.size() / 4); std::memcpy(v.data(), b.data(), b.size()); return v;
}
static void test_ple_tail(const std::string& gguf_path) {
    std::mt19937_64 rng(5);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    TailData d;
    bool real = false;
    try {
        strix::qwen_gguf::Model model(gguf_path);
        d.nk = f32_tensor(model, "blk.1.ple_norm_key.weight"); d.nq = f32_tensor(model, "blk.1.ple_norm_query.weight");
        d.nc = f32_tensor(model, "blk.1.ple_norm_conv.weight"); d.conv = f32_tensor(model, "blk.1.ple_conv1d.weight");
        real = true;
    } catch (const std::exception& e) { report(false, "ple tail real norm/conv weights", e.what()); }
    if (!real) {
        d.nk.resize(10240); d.nq.resize(10240); d.nc.resize(10240); d.conv.resize(40960);
        for (auto* v : {&d.nk, &d.nq, &d.nc, &d.conv}) for (auto& e : *v) e = nd(rng);
    }
    for (std::uint32_t rows : {64U, 5U, 9U, 10U}) {
        d.rows = rows;
        d.keys.resize(static_cast<size_t>(rows) * 10240); d.values.resize(static_cast<size_t>(rows) * 2560);
        for (auto& e : d.keys) e = nd(rng) * 3.0F;
        for (auto& e : d.values) e = nd(rng) * 0.5F;
        d.input = random_bf16(rng, static_cast<size_t>(rows) * 10240, false);
        d.state = random_bf16(rng, static_cast<size_t>(10240) * 9, false);
        // edge rows: an all-zero key/query stream (score 0, rsqrt(eps)), a huge-key stream, tiny values
        if (rows >= 5) {
            for (std::uint32_t f = 0; f < 2560; ++f) { d.keys[f] = 0.0F; d.input[f] = 0; }
            for (std::uint32_t f = 0; f < 2560; ++f) d.keys[10240 + 2560 + f] *= 1e18F;
            for (std::uint32_t f = 0; f < 2560; ++f) d.values[2 * 2560 + f] *= 1e-30F;
            for (std::uint32_t f = 0; f < 2560; ++f) d.input[3 * 10240 + f] = f2bf(-bf(d.input[3 * 10240 + f]) * 0.0F);
        }
        tail_case("ple tail " + std::string(real ? "real weights " : "synthetic ") + std::to_string(rows) + " rows", d, rows != 9);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------
// MTP cache rows: literal host references of every HIP kernel of one primed row (see src/cuda/models/qwen/qwen_plemtp.cu).
static std::uint16_t f2bf_quiet(float x) {
    const auto b = fbits(x);
    if ((b & 0x7f800000U) == 0x7f800000U && (b & 0x007fffffU)) return static_cast<std::uint16_t>((b >> 16U) | 0x0040U);
    return f2bf(x);
}
static float rbf(float v) { return bf(f2bf(v)); }
static float half_at(const std::uint8_t* p) { return strix::iq4_nl::fp16_bits_to_float(static_cast<std::uint16_t>(p[0] | (p[1] << 8U))); }
static std::uint16_t host_f2h(float f) {   // RNE float -> fp16 (normal / subnormal / overflow)
    const std::uint32_t b = fbits(f), sign = (b >> 16U) & 0x8000U;
    const float a = std::fabs(f);
    if (a >= 65520.0F) return static_cast<std::uint16_t>(sign | 0x7c00U);
    if (a < 6.103515625e-05F) {   // subnormal: units of 2^-24
        const float q = std::nearbyint(a * 16777216.0F);   // exact scaling, RNE
        return static_cast<std::uint16_t>(sign | static_cast<std::uint32_t>(q));
    }
    const std::uint32_t m = fbits(a);
    std::uint32_t e = (m >> 23U) - 112U, mant = m & 0x7fffffU;
    std::uint32_t h = (e << 10U) | (mant >> 13U);
    const std::uint32_t rem = mant & 0x1fffU;
    if (rem > 0x1000U || (rem == 0x1000U && (h & 1U))) ++h;
    return static_cast<std::uint16_t>(sign | h);
}
static std::vector<std::uint8_t> ref_q8_to_q4(const std::vector<std::uint8_t>& q8) {
    const size_t blocks = q8.size() / 34;
    std::vector<std::uint8_t> q4(blocks * 18);
    for (size_t b = 0; b < blocks; ++b) {
        const std::uint8_t* s = &q8[b * 34];
        const float scale = half_at(s);
        float v[32], mx = 0.0F, amax = 0.0F;
        for (int i = 0; i < 32; ++i) { v[i] = fmul(scale, static_cast<float>(static_cast<std::int8_t>(s[2 + i]))); if (std::fabs(v[i]) > amax) { amax = std::fabs(v[i]); mx = v[i]; } }
        const float d = fmul(mx, -0.125F), inv = d != 0.0F ? fdiv(1.0F, d) : 0.0F;
        const std::uint16_t hd = host_f2h(d);
        q4[b * 18] = static_cast<std::uint8_t>(hd & 0xff); q4[b * 18 + 1] = static_cast<std::uint8_t>(hd >> 8);
        for (int j = 0; j < 16; ++j) {
            const auto lo = static_cast<unsigned>(std::min(15.0F, ffma(inv, v[j], 8.5F)));
            const auto hi = static_cast<unsigned>(std::min(15.0F, ffma(inv, v[j + 16], 8.5F)));
            q4[b * 18 + 2 + j] = static_cast<std::uint8_t>(lo | (hi << 4));
        }
    }
    return q4;
}
static float host_tree(std::vector<float> s) {   // halving tree over a power-of-two count
    for (size_t stride = s.size() / 2; stride > 0; stride >>= 1)
        for (size_t l = 0; l < stride; ++l) s[l] = fadd(s[l], s[l + stride]);
    return s[0];
}
struct MtpHost {
    std::vector<std::uint8_t> embd, eh_q4, hc_down, hc_up, k, v;   // embd: compact Q8 table
    std::vector<float> enorm, hnorm, hc_norm, k_norm, cs;
    std::vector<std::uint16_t> index_k;
    std::uint32_t vocab = 0; std::uint64_t rope_positions = 0;
};
struct MtpStages {
    std::vector<std::uint16_t> embedding, fused, hidden0, kv, raw;
    std::vector<float> normalized, hc_normalized, low, index_key, key, value;
};
static MtpStages ref_mtp(const MtpHost& w, const pm::MtpRounding& rd, const std::vector<std::uint16_t>& hidden,
    const std::vector<std::uint32_t>& tokens, std::uint32_t rows, std::uint64_t position0) {
    const std::uint32_t H = 2560, W = 10240;
    MtpStages s;
    s.embedding.resize(static_cast<size_t>(rows) * H);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t f = 0; f < H; ++f) {
            const std::uint8_t* blk = &w.embd[(static_cast<size_t>(tokens[r]) * (H / 32) + f / 32) * 34];
            s.embedding[static_cast<size_t>(r) * H + f] = f2bf(fmul(half_at(blk), static_cast<float>(static_cast<std::int8_t>(blk[2 + f % 32]))));
        }
    // input norms
    s.normalized.resize(static_cast<size_t>(rows) * (H + W));
    for (std::uint32_t r = 0; r < rows; ++r)
        for (int part = 0; part < 2; ++part) {
            const std::uint32_t n = part ? W : H;
            const std::uint16_t* in = part ? &hidden[static_cast<size_t>(r) * W] : &s.embedding[static_cast<size_t>(r) * H];
            const float* g = part ? w.hnorm.data() : w.enorm.data();
            std::vector<float> lanes(256, 0.0F);
            for (std::uint32_t l = 0; l < 256; ++l) for (std::uint32_t i = l; i < n; i += 256) { const float x = bf(in[i]); lanes[l] = ffma(x, x, lanes[l]); }
            const float inv = fdiv(1.0F, std::sqrt(fadd(fdiv(host_tree(lanes), static_cast<float>(n)), rd.epsilon)));
            float* out = &s.normalized[static_cast<size_t>(r) * (H + W) + (part ? H : 0)];
            for (std::uint32_t i = 0; i < n; ++i) {
                const float gain = rd.gemma_offset ? fadd(1.0F, g[i]) : g[i];
                const float v = fmul(fmul(bf(in[i]), inv), gain);
                out[i] = rd.normalized_bf16 ? bf(f2bf_quiet(v)) : v;
            }
        }
    // fused Q4 projection
    s.fused.resize(static_cast<size_t>(rows) * W);
    const auto code4 = [](const std::uint8_t* blk, unsigned j) { return static_cast<float>(static_cast<int>((blk[2 + (j & 15)] >> (j >= 16 ? 4 : 0)) & 15) - 8); };
    for (std::uint32_t r = 0; r < rows; ++r) {
        const float* x = &s.normalized[static_cast<size_t>(r) * (H + W)];
        for (std::uint32_t o = 0; o < H; ++o) {
            const std::uint8_t* packed = &w.eh_q4[static_cast<size_t>(o) * (2 * H / 32) * 18];
            std::vector<float> e(32, 0.0F);
            std::vector<std::vector<float>> sk(4, std::vector<float>(32, 0.0F));
            for (std::uint32_t lane = 0; lane < 32; ++lane)
                for (std::uint32_t base = 8 * lane; base < H; base += 256) {
                    const std::uint8_t* eb = packed + (base / 32) * 18; const std::uint8_t* wb = packed + ((H + base) / 32) * 18;
                    for (std::uint32_t i = 0; i < 8; ++i) {
                        const unsigned j = base % 32 + i;
                        e[lane] = ffma(fmul(half_at(eb), code4(eb, j)), x[base + i], e[lane]);
                        const float c = fmul(half_at(wb), code4(wb, j));
                        for (int st = 0; st < 4; ++st) sk[st][lane] = ffma(c, x[H + st * H + base + i], sk[st][lane]);
                    }
                }
            const float er = host_tree(e), proj = rd.projection_bf16 ? bf(f2bf_quiet(er)) : er;
            for (int st = 0; st < 4; ++st) {
                const float t = host_tree(sk[st]), sv = rd.projection_bf16 ? bf(f2bf_quiet(t)) : t;
                s.fused[static_cast<size_t>(r) * W + st * H + o] = f2bf_quiet(fadd(proj, sv));
            }
        }
    }
    // HC norm
    std::vector<float> rin(static_cast<size_t>(rows) * 4);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t st = 0; st < 4; ++st) {
            std::vector<float> lanes(256, 0.0F);
            for (std::uint32_t l = 0; l < 256; ++l) for (std::uint32_t f = l; f < H; f += 256) { const float v = bf(s.fused[static_cast<size_t>(r) * W + st * H + f]); lanes[l] = ffma(v, v, lanes[l]); }
            rin[r * 4 + st] = fadd(fdiv(host_tree(lanes), static_cast<float>(H)), rd.epsilon);
        }
    const auto inv = oracle(0, rin);
    s.hc_normalized.resize(static_cast<size_t>(rows) * W);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t st = 0; st < 4; ++st)
            for (std::uint32_t f = 0; f < H; ++f) {
                const size_t i = static_cast<size_t>(r) * W + st * H + f;
                s.hc_normalized[i] = fmul(fmul(bf(s.fused[i]), inv[r * 4 + st]), w.hc_norm[st * H + f]);
            }
    // split-K low rank
    std::vector<float> scaled(static_cast<size_t>(rows) * 320), negs(scaled.size());
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t o = 0; o < 320; ++o) {
            float total = 0.0F;
            for (std::uint32_t tile = 0; tile < 16; ++tile) {
                std::vector<float> lanes(32, 0.0F);
                for (std::uint32_t lane = 0; lane < 32; ++lane)
                    for (std::uint32_t sidx = 0; sidx < 5; ++sidx)
                        for (std::uint32_t j = 0; j < 4; ++j) {
                            const std::uint32_t f = tile * 640 + sidx * 128 + lane * 4 + j;
                            const std::uint8_t* blk = &w.hc_down[(static_cast<size_t>(o) * 320 + f / 32) * 34];
                            const float c = fmul(half_at(blk), static_cast<float>(static_cast<std::int8_t>(blk[2 + f % 32])));
                            lanes[lane] = ffma(s.hc_normalized[static_cast<size_t>(r) * W + f], c, lanes[lane]);
                        }
                total = fadd(total, host_tree(lanes));
            }
            const float sc = fdiv(total, 4.0F);
            scaled[r * 320 + o] = sc; negs[r * 320 + o] = -sc;
        }
    const auto es = oracle(1, negs);
    s.low.resize(scaled.size());
    for (size_t i = 0; i < scaled.size(); ++i) s.low[i] = fmul(scaled[i], fdiv(1.0F, fadd(1.0F, es[i])));
    // lane-chain output
    std::vector<float> gates(static_cast<size_t>(rows) * W), negg(gates.size());
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t st = 0; st < 4; ++st)
            for (std::uint32_t f = 0; f < H; ++f) {
                const std::uint8_t* row = &w.hc_up[(static_cast<size_t>(st) * H + f) * 10 * 34];
                float gate = 0.0F;
                for (std::uint32_t b = 0; b < 10; ++b)
                    for (std::uint32_t k = 0; k < 32; ++k)
                        gate = ffma(s.low[static_cast<size_t>(r) * 320 + b * 32 + k], fmul(half_at(row + b * 34), static_cast<float>(static_cast<std::int8_t>(row[b * 34 + 2 + k]))), gate);
                gates[static_cast<size_t>(r) * W + st * H + f] = gate; negg[static_cast<size_t>(r) * W + st * H + f] = -gate;
            }
    const auto eg = oracle(1, negg);
    s.hidden0.resize(static_cast<size_t>(rows) * H);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t f = 0; f < H; ++f) {
            float mixed = 0.0F;
            for (std::uint32_t st = 0; st < 4; ++st) {
                const size_t i = static_cast<size_t>(r) * W + st * H + f;
                mixed = ffma(s.hc_normalized[i], fdiv(1.0F, fadd(1.0F, eg[i])), mixed);
            }
            s.hidden0[static_cast<size_t>(r) * H + f] = f2bf(fdiv(mixed, 4.0F));
        }
    // cache projections
    s.index_key.resize(static_cast<size_t>(rows) * 128); s.key.resize(static_cast<size_t>(rows) * 512); s.value.resize(static_cast<size_t>(rows) * 512);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t out = 0; out < 1152; ++out) {
            const int kind = out < 128 ? 0 : out < 640 ? 1 : 2;
            const std::uint32_t o = kind == 0 ? out : kind == 1 ? out - 128 : out - 640;
            std::vector<float> lanes(256, 0.0F);
            for (std::uint32_t l = 0; l < 256; ++l)
                for (std::uint32_t f = l; f < H; f += 256) {
                    float wv;
                    if (kind == 0) wv = bf(w.index_k[static_cast<size_t>(o) * H + f]);
                    else { const std::uint8_t* blk = &(kind == 1 ? w.k : w.v)[(static_cast<size_t>(o) * 80 + f / 32) * 34]; wv = fmul(half_at(blk), static_cast<float>(static_cast<std::int8_t>(blk[2 + f % 32]))); }
                    lanes[l] = ffma(bf(s.hidden0[static_cast<size_t>(r) * H + f]), wv, lanes[l]);
                }
            const float v = rbf(host_tree(lanes));
            (kind == 0 ? s.index_key[static_cast<size_t>(r) * 128 + o] : kind == 1 ? s.key[static_cast<size_t>(r) * 512 + o] : s.value[static_cast<size_t>(r) * 512 + o]) = v;
        }
    // K norm + RoPE
    std::vector<float> means(static_cast<size_t>(rows) * 2);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t h = 0; h < 2; ++h) {
            std::vector<float> red(256);
            for (std::uint32_t f = 0; f < 256; ++f) { const float x = s.key[static_cast<size_t>(r) * 512 + h * 256 + f]; red[f] = fmul(x, x); }
            means[r * 2 + h] = fadd(fdiv(host_tree(red), 256.0F), rd.epsilon);
        }
    const auto kinv = oracle(0, means);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t h = 0; h < 2; ++h) {
            float* v = &s.key[static_cast<size_t>(r) * 512 + h * 256];
            float src[256];
            for (std::uint32_t f = 0; f < 256; ++f) src[f] = rbf(fmul(fmul(v[f], kinv[r * 2 + h]), w.k_norm[f]));
            const std::uint64_t p = position0 + r;
            for (std::uint32_t f = 0; f < 256; ++f) {
                float res = src[f];
                if (f < 64) {
                    const std::uint32_t pair = f % 32;
                    const float c = w.cs[(p * 32 + pair) * 2], sn = w.cs[(p * 32 + pair) * 2 + 1];
                    const float ac = fmul(src[f], c);
                    res = f < 32 ? ffma(-src[f + 32], sn, ac) : ffma(src[f - 32], sn, ac);
                }
                v[f] = rbf(res);
            }
        }
    // stores (all rows finite in the tests; the validator is exercised separately)
    s.kv.resize(static_cast<size_t>(rows) * 1024); s.raw.resize(static_cast<size_t>(rows) * 128);
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t i = 0; i < 512; ++i) {
            s.kv[static_cast<size_t>(r) * 1024 + i] = f2bf(s.key[static_cast<size_t>(r) * 512 + i]);
            s.kv[static_cast<size_t>(r) * 1024 + 512 + i] = f2bf(s.value[static_cast<size_t>(r) * 512 + i]);
        }
        for (std::uint32_t i = 0; i < 128; ++i) s.raw[static_cast<size_t>(r) * 128 + i] = f2bf(s.index_key[static_cast<size_t>(r) * 128 + i]);
    }
    return s;
}
static void mtp_case(const std::string& name, const MtpHost& w, const pm::MtpRounding& rd, std::uint32_t rows, std::uint64_t position0,
    std::mt19937_64& rng, float hidden_scale) {
    std::normal_distribution<float> nd(0.0F, hidden_scale);
    std::vector<std::uint16_t> hidden(static_cast<size_t>(rows) * 10240);
    for (auto& e : hidden) e = f2bf(nd(rng));
    std::vector<std::uint32_t> tokens(rows);
    for (auto& t : tokens) t = static_cast<std::uint32_t>(rng() % w.vocab);
    const auto want = ref_mtp(w, rd, hidden, tokens, rows, position0);
    // device
    pm::MtpWeights dw{};
    auto* embd = dev(w.embd); auto* eh = dev(w.eh_q4); auto* down = dev(w.hc_down); auto* up = dev(w.hc_up); auto* k = dev(w.k); auto* v = dev(w.v);
    auto* enorm = dev(w.enorm); auto* hnorm = dev(w.hnorm); auto* hcn = dev(w.hc_norm); auto* kn = dev(w.k_norm); auto* cs = dev(w.cs);
    auto* ik = dev(w.index_k);
    dw.token_embd = embd; dw.vocab = w.vocab; dw.enorm = enorm; dw.hnorm = hnorm; dw.eh_q4 = eh; dw.hc_norm = hcn; dw.hc_down = down;
    dw.hc_up = up; dw.index_k = ik; dw.k = k; dw.v = v; dw.k_norm = kn; dw.rope_cs = cs; dw.rope_positions = w.rope_positions;
    auto* dh = dev(hidden); auto* dt = dev(tokens);
    const std::uint64_t cap = position0 + rows + 2;
    std::uint16_t *kv = nullptr, *raw = nullptr;
    CK(cudaMalloc(&kv, cap * 1024 * 2)); CK(cudaMalloc(&raw, cap * 128 * 2));
    CK(cudaMemset(kv, 0x5a, cap * 1024 * 2)); CK(cudaMemset(raw, 0x5a, cap * 128 * 2));
    void* ws = nullptr; CK(cudaMalloc(&ws, pm::mtp_workspace_bytes(rows)));
    std::uint32_t* status = dev(std::vector<std::uint32_t>{0});
    pm::launch_mtp_cache_rows(dw, rd, dh, dt, rows, position0, kv, raw, ws, status, g_tables, nullptr);
    CK(cudaDeviceSynchronize());
    const auto vw = pm::mtp_workspace_views(ws, rows);
    compare(name + " embedding", host(vw.embedding, want.embedding.size()), want.embedding);
    compare(name + " input norms", host(vw.normalized, want.normalized.size()), want.normalized);
    compare(name + " fused Q4 projection", host(vw.fused, want.fused.size()), want.fused);
    compare(name + " HC normalized", host(vw.hc_normalized, want.hc_normalized.size()), want.hc_normalized);
    compare(name + " HC low rank", host(vw.low, want.low.size()), want.low);
    compare(name + " hidden0", host(vw.hidden0, want.hidden0.size()), want.hidden0);
    compare(name + " index key", host(vw.index_key, want.index_key.size()), want.index_key);
    compare(name + " K (norm + RoPE)", host(vw.key, want.key.size()), want.key);
    compare(name + " V", host(vw.value, want.value.size()), want.value);
    const auto kv_all = host(kv, cap * 1024), raw_all = host(raw, cap * 128);
    std::vector<std::uint16_t> kv_rows(kv_all.begin() + position0 * 1024, kv_all.begin() + (position0 + rows) * 1024);
    std::vector<std::uint16_t> raw_rows(raw_all.begin() + position0 * 128, raw_all.begin() + (position0 + rows) * 128);
    compare(name + " KV rows", kv_rows, want.kv);
    compare(name + " raw index rows", raw_rows, want.raw);
    bool untouched = true;
    for (std::uint64_t i = 0; i < cap * 1024; ++i) if ((i < position0 * 1024 || i >= (position0 + rows) * 1024) && kv_all[i] != 0x5a5a) untouched = false;
    report(untouched && host(status, 1)[0] == 0, name + " only rows [p0, p0 + rows) written, status clean");
    for (void* p : {static_cast<void*>(embd), static_cast<void*>(eh), static_cast<void*>(down), static_cast<void*>(up), static_cast<void*>(k),
             static_cast<void*>(v), static_cast<void*>(enorm), static_cast<void*>(hnorm), static_cast<void*>(hcn), static_cast<void*>(kn),
             static_cast<void*>(cs), static_cast<void*>(ik), static_cast<void*>(dh), static_cast<void*>(dt), static_cast<void*>(kv),
             static_cast<void*>(raw), ws, static_cast<void*>(status)}) CK(cudaFree(p));
}
static std::vector<float> rope_table(std::uint64_t positions) {   // stand-in cos/sin values (the node table comes later)
    std::vector<float> cs(positions * 64);
    for (std::uint64_t p = 0; p < positions; ++p)
        for (std::uint32_t i = 0; i < 32; ++i) {
            const double a = static_cast<double>(p) * std::pow(1e7, -2.0 * i / 64.0);
            cs[(p * 32 + i) * 2] = static_cast<float>(std::cos(a)); cs[(p * 32 + i) * 2 + 1] = static_cast<float>(std::sin(a));
        }
    return cs;
}
static void test_q8_to_q4(const std::vector<std::uint8_t>& q8, const std::string& name) {
    const auto want = ref_q8_to_q4(q8);
    auto* d8 = dev(q8); std::uint8_t* d4 = nullptr; CK(cudaMalloc(&d4, want.size()));
    pm::launch_mtp_q8_to_q4(d8, q8.size() / 34, d4, nullptr);
    CK(cudaDeviceSynchronize());
    const auto got = host(d4, want.size());
    compare(name, got, want);
    for (size_t b = 0, shown = 0; b < want.size() / 18 && shown < 3; ++b)
        if (std::memcmp(&got[b * 18], &want[b * 18], 18)) {
            ++shown;
            std::printf("  block %zu q8 scale %02x%02x got", b, q8[b * 34 + 1], q8[b * 34]);
            for (int i = 0; i < 18; ++i) std::printf(" %02x", got[b * 18 + i]);
            std::printf("\n  want");
            for (int i = 0; i < 18; ++i) std::printf(" %02x", want[b * 18 + i]);
            std::printf("\n");
        }
    CK(cudaFree(d8)); CK(cudaFree(d4));
}
static void test_mtp(const std::string& gguf_path, const std::string& sidecar_path) {
    std::mt19937_64 rng(99);
    // Q8 -> Q4 edge blocks: zeros, negative extreme, ties, subnormal and maximal scales
    {
        auto q8 = random_q8(rng, 64, 2560, true);
        for (int i = 0; i < 32; ++i) q8[2 + i] = 0;                                    // all-zero block
        for (int i = 0; i < 32; ++i) q8[34 + 2 + i] = static_cast<std::uint8_t>(i == 5 ? 0x81 : (i == 9 ? 127 : 3));   // |-127| vs 127 tie: first wins
        test_q8_to_q4(q8, "mtp q8->q4 synthetic + edge blocks");
    }
    pm::MtpRounding prod{};
    // synthetic weights
    MtpHost w;
    {
        w.vocab = 64;
        w.embd = random_q8(rng, w.vocab, 2560, false);
        w.eh_q4 = ref_q8_to_q4(random_q8(rng, 2560, 5120, false));
        w.hc_down = random_q8(rng, 320, 10240, false); w.hc_up = random_q8(rng, 10240, 320, false);
        w.k = random_q8(rng, 512, 2560, false); w.v = random_q8(rng, 512, 2560, false);
        w.index_k = random_bf16(rng, static_cast<size_t>(128) * 2560, false);
        std::normal_distribution<float> nd(1.0F, 0.3F);
        w.enorm.resize(2560); w.hnorm.resize(10240); w.hc_norm.resize(10240); w.k_norm.resize(256);
        for (auto* p : {&w.enorm, &w.hnorm, &w.hc_norm, &w.k_norm}) for (auto& e : *p) e = nd(rng);
        w.rope_positions = 4096; w.cs = rope_table(w.rope_positions);
        mtp_case("mtp synthetic 9 rows @0", w, prod, 9, 0, rng, 1.0F);
        pm::MtpRounding alt{}; alt.normalized_bf16 = false; alt.projection_bf16 = false; alt.gemma_offset = true;
        mtp_case("mtp synthetic fp32-rounding gemma 6 rows @1000", w, alt, 6, 1000, rng, 3.0F);
    }
    // real distillG sidecar + target token_embd rows
    try {
        strix::qwen_gguf::Model side(sidecar_path), model(gguf_path);
        MtpHost r;
        const auto f32 = [&](const std::string& n) { const auto b = side.bytes(n); std::vector<float> v(b.size() / 4); std::memcpy(v.data(), b.data(), b.size()); return v; };
        r.enorm = f32("blk.48.nextn.enorm.weight"); r.hnorm = f32("blk.48.nextn.hnorm.weight"); r.hc_norm = f32("blk.48.hc_attn_norm.weight");
        r.k_norm = f32("blk.48.attn_k_norm.weight");
        const auto eh8 = side.bytes("blk.48.nextn.eh_proj.weight");
        test_q8_to_q4(eh8, "mtp q8->q4 real eh_proj (distillG)");
        r.eh_q4 = ref_q8_to_q4(eh8);
        r.hc_down = side.bytes("blk.48.hc_attn_down.weight"); r.hc_up = side.bytes("blk.48.hc_attn_up.weight");
        r.k = side.bytes("blk.48.attn_k.weight"); r.v = side.bytes("blk.48.attn_v.weight");
        const auto ikb = side.bytes("blk.48.indexer.k_proj.weight"); r.index_k.resize(ikb.size() / 2); std::memcpy(r.index_k.data(), ikb.data(), ikb.size());
        // 48 real token_embd rows (the kernel indexes the compact table)
        const auto& te = model.tensor("token_embd.weight");
        r.vocab = 48; r.embd.resize(static_cast<size_t>(r.vocab) * 80 * 34);
        FILE* f = std::fopen(gguf_path.empty() ? "Qwen3.8-Flash-Next-UD-Q4_K_XL-merged.gguf" : gguf_path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open gguf");
        for (std::uint32_t i = 0; i < r.vocab; ++i) {
            const std::uint64_t tok = (static_cast<std::uint64_t>(i) * 5171 + 11) % 248320;
            _fseeki64(f, static_cast<long long>(te.absolute_offset + tok * 80 * 34), SEEK_SET);
            if (std::fread(&r.embd[static_cast<size_t>(i) * 80 * 34], 1, 80 * 34, f) != 80 * 34) throw std::runtime_error("short embd read");
        }
        std::fclose(f);
        {   // the node's RoPE table (HIP angle expression, cosf/sinf on gfx1151) when present, else a stand-in
            FILE* rf = std::fopen("qrtx-build\\fx\\rope_cs_262144.bin", "rb");
            r.rope_positions = 8192; r.cs.assign(static_cast<size_t>(r.rope_positions) * 64, 0.0F);
            const bool node = rf && std::fread(r.cs.data(), 4, r.cs.size(), rf) == r.cs.size();
            if (rf) std::fclose(rf);
            if (!node) r.cs = rope_table(r.rope_positions);
            std::printf("mtp real: RoPE table %s\n", node ? "fx\\rope_cs_262144.bin (node)" : "host stand-in");
        }
        mtp_case("mtp real distillG 16 rows @0", r, prod, 16, 0, rng, 0.8F);
        mtp_case("mtp real distillG 7 rows @5000", r, prod, 7, 5000, rng, 2.0F);
    } catch (const std::exception& e) { report(false, "mtp real distillG weights", e.what()); }
}
// ---------------------------------------------------------------------------------------------------------------
// d64 fixture replay of the PLE step (SerialOp::ple = 3): inputs = the step's read records (BF16 wide rows), tokens from
// <fx>\d64.tokens (u32) or recovered from the token_embedding step's writes (SerialOp 0) by matching token_embd rows;
// expected = the step's write records and the final PLE convolution / n-gram state of <fx>\d64-state.
static std::vector<std::uint8_t> read_file(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return {};
    _fseeki64(f, 0, SEEK_END); const auto n = _ftelli64(f); _fseeki64(f, 0, SEEK_SET);
    std::vector<std::uint8_t> v(static_cast<size_t>(n));
    if (n && std::fread(v.data(), 1, v.size(), f) != v.size()) v.clear();
    std::fclose(f);
    return v;
}
// FAST form (Config::forms.mtp_mma): fused projection on FP16 tensor cores vs the exact kernel on the same random Q4 weights
// and FP32 activations. Not bit-exact by design: reports max abs, rel L2 and the spread of per-row / per-column-tile rel L2
// (uniform = no edge / last-tile / row bug), standalone timing exact vs mma with the SM clock.
static void test_mtp_mma_form() {
    const std::uint32_t K5 = 2560 * 2 / 32, W = 12800, O = 10240;
    std::mt19937_64 rng(1234);
    std::vector<std::uint8_t> q4(static_cast<size_t>(2560) * K5 * 18);
    for (size_t b = 0; b < static_cast<size_t>(2560) * K5; ++b) {
        const float d = (static_cast<float>(rng() % 2000) / 2000.0F - 0.5F) * 0.05F;
        const auto h = host_f2h(d);
        q4[b * 18] = static_cast<std::uint8_t>(h & 255U); q4[b * 18 + 1] = static_cast<std::uint8_t>(h >> 8U);
        for (int j = 0; j < 16; ++j) q4[b * 18 + 2 + j] = static_cast<std::uint8_t>(rng());
    }
    auto* dq = dev(q4);
    const std::uint32_t maxrows = 8192;
    std::vector<float> x(static_cast<size_t>(maxrows) * W);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    for (std::uint32_t r = 0; r < maxrows; ++r)
        for (std::uint32_t c = 0; c < W; ++c) {
            float v = nd(rng) * (1.0F + 0.02F * static_cast<float>(c % 97));   // gains up to ~3
            if ((rng() & 0x3ffU) == 0) v *= 20.0F;                              // outliers
            x[static_cast<size_t>(r) * W + c] = v;
        }
    auto* dx = dev(x);
    std::uint16_t *de = nullptr, *dm = nullptr;
    CK(cudaMalloc(&de, static_cast<size_t>(maxrows) * O * 2)); CK(cudaMalloc(&dm, static_cast<size_t>(maxrows) * O * 2));
    pm::MtpRounding ex{}, mm{}; mm.mma = true;
    for (std::uint32_t rows : {1U, 15U, 16U, 17U, 129U, 1000U, 8192U}) {
        CK(cudaMemset(de, 0, static_cast<size_t>(rows) * O * 2)); CK(cudaMemset(dm, 0, static_cast<size_t>(rows) * O * 2));
        pm::launch_mtp_fused_projection(dx, dq, rows, de, ex, nullptr);
        pm::launch_mtp_fused_projection(dx, dq, rows, dm, mm, nullptr);
        CK(cudaDeviceSynchronize());
        const auto a = host(de, static_cast<size_t>(rows) * O), b = host(dm, static_cast<size_t>(rows) * O);
        double dd = 0, rr = 0, mx = 0, mxref = 0;
        std::vector<double> rd(rows, 0.0), rn(rows, 0.0), cd(O / 128, 0.0), cn(O / 128, 0.0);
        for (std::uint32_t r = 0; r < rows; ++r)
            for (std::uint32_t o = 0; o < O; ++o) {
                const double u = bf(a[static_cast<size_t>(r) * O + o]), v = bf(b[static_cast<size_t>(r) * O + o]);
                const double e = v - u;
                dd += e * e; rr += u * u; mx = std::max(mx, std::fabs(e)); mxref = std::max(mxref, std::fabs(u));
                rd[r] += e * e; rn[r] += u * u; cd[o / 128] += e * e; cn[o / 128] += u * u;
            }
        double rmin = 1e9, rmax = 0, cmin = 1e9, cmax = 0;
        for (std::uint32_t r = 0; r < rows; ++r) { const double q = std::sqrt(rd[r] / std::max(rn[r], 1e-30)); rmin = std::min(rmin, q); rmax = std::max(rmax, q); }
        for (size_t c = 0; c < cd.size(); ++c) { const double q = std::sqrt(cd[c] / std::max(cn[c], 1e-30)); cmin = std::min(cmin, q); cmax = std::max(cmax, q); }
        const double rel = std::sqrt(dd / std::max(rr, 1e-30));
        char buf[300];
        std::snprintf(buf, sizeof(buf), "rows %u: maxabs %.4g (max|ref| %.4g) relL2 %.3e; per-row relL2 [%.2e, %.2e] per-128-col-tile [%.2e, %.2e]",
            rows, mx, mxref, rel, rmin, rmax, cmin, cmax);
        report(rel < 2e-3 && rmax < 5.0 * std::max(rel, 1e-9) + 1e-4 && cmax < 3.0 * std::max(rel, 1e-9) + 1e-4, "mtp_mma vs exact", buf);
    }
    for (std::uint32_t rows : {8192U, 1024U}) {
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int form = 0; form < 2; ++form) {
            const pm::MtpRounding& rd = form ? mm : ex;
            for (int i = 0; i < 3; ++i) pm::launch_mtp_fused_projection(dx, dq, rows, de, rd, nullptr);
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(e0));
            for (int i = 0; i < 20; ++i) pm::launch_mtp_fused_projection(dx, dq, rows, de, rd, nullptr);
            CK(cudaEventRecord(e1));
            for (int i = 0; i < 100; ++i) pm::launch_mtp_fused_projection(dx, dq, rows, de, rd, nullptr);  // keep the clock up
            std::printf("  SM clock under load: "); std::fflush(stdout);
            std::system("nvidia-smi -i 0 --query-gpu=clocks.sm --format=csv,noheader");
            CK(cudaDeviceSynchronize());
            float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
            std::printf("  timing rows %u %s: %.3f ms/call (20 reps) = %.1f TFLOPS\n", rows, form ? "mma  " : "exact", ms / 20.0F,
                537.0 * rows / 8192.0 / (ms / 20.0F));
        }
    }
}

static void test_fixture(const std::string& dir, const std::string& gguf_path) {
    const std::string path = dir + "\\d64.dump";
    FILE* probe = std::fopen(path.c_str(), "rb");
    if (!probe) { std::printf("SKIP d64 fixture (%s not present)\n", path.c_str()); return; }
    std::fclose(probe);
    try {
        strix::qwen_fixture::Dump dump(path);
        constexpr std::uint32_t kOpPle = 3, kOpEmbedding = 0;
        std::uint32_t ple_step = UINT32_MAX, rows = 0;
        for (const auto& r : dump.records()) if (r.op == kOpPle) { ple_step = r.step; rows = std::max(rows, r.row + 1); }
        if (ple_step == UINT32_MAX) { report(false, "d64 fixture", "no PLE step records"); return; }
        // the step's read and write buffer indices of 10240 BF16 (20480 B)
        std::int64_t in_index = -1, out_index = -1;
        for (const auto& r : dump.records())
            if (r.step == ple_step && r.row == 0 && r.bytes == 20480) (r.is_read ? in_index : out_index) = r.index;
        std::printf("d64: PLE step %u, %u rows, read index %lld, write index %lld, position0 %llu\n", ple_step, rows,
            static_cast<long long>(in_index), static_cast<long long>(out_index),
            static_cast<unsigned long long>(dump.find(ple_step, true, static_cast<std::uint32_t>(in_index), 0)->position));
        for (const auto& r : dump.records()) if (r.step == ple_step && r.row == 0)
            std::printf("  record %s index %u bytes %llu kernel %s\n", r.is_read ? "read " : "write", r.index, static_cast<unsigned long long>(r.bytes), r.kernel_id.c_str());
        const auto in_bytes = dump.load_rows(ple_step, true, static_cast<std::uint32_t>(in_index));
        const auto out_bytes = dump.load_rows(ple_step, false, static_cast<std::uint32_t>(out_index));
        std::vector<std::uint16_t> input(in_bytes.size() / 2), want(out_bytes.size() / 2);
        std::memcpy(input.data(), in_bytes.data(), in_bytes.size()); std::memcpy(want.data(), out_bytes.data(), out_bytes.size());
        const std::uint64_t position0 = dump.find(ple_step, true, static_cast<std::uint32_t>(in_index), 0)->position;
        strix::qwen_gguf::Model model(gguf_path);
        // tokens
        std::vector<std::uint32_t> tokens(rows);
        const auto tok_file = read_file(dir + "\\d64.tokens");
        if (tok_file.size() >= rows * 4ULL) std::memcpy(tokens.data(), tok_file.data(), rows * 4ULL);
        else {
            std::uint32_t emb_step = UINT32_MAX;
            for (const auto& r : dump.records()) if (r.op == kOpEmbedding && !r.is_read && r.bytes == 5120) emb_step = r.step;
            if (emb_step == UINT32_MAX) throw std::runtime_error("no tokens file and no embedding step");
            std::uint32_t emb_index = 0;
            for (const auto& r : dump.records()) if (r.step == emb_step && !r.is_read && r.row == 0 && r.bytes == 5120) emb_index = r.index;
            const auto embd = model.bytes("token_embd.weight");
            for (std::uint32_t row = 0; row < rows; ++row) {
                const auto e = dump.load(*dump.find(emb_step, false, emb_index, row));
                std::uint16_t first[32]; std::memcpy(first, e.data(), 64);
                bool found = false;
                for (std::uint32_t t = 0; t < 248320 && !found; ++t) {
                    const std::uint8_t* blk = &embd[static_cast<size_t>(t) * 80 * 34];
                    bool ok = true;
                    for (int i = 0; i < 32 && ok; ++i) ok = f2bf(fmul(half_at(blk), static_cast<float>(static_cast<std::int8_t>(blk[2 + i])))) == first[i];
                    if (!ok) continue;
                    for (int i = 0; i < 2560 && ok; ++i) {
                        const std::uint8_t* b = &embd[(static_cast<size_t>(t) * 80 + i / 32) * 34];
                        std::uint16_t g; std::memcpy(&g, &e[static_cast<size_t>(i) * 2], 2);
                        ok = f2bf(fmul(half_at(b), static_cast<float>(static_cast<std::int8_t>(b[2 + i % 32])))) == g;
                    }
                    if (ok) { tokens[row] = t; found = true; }
                }
                if (!found) throw std::runtime_error("row " + std::to_string(row) + ": embedding matches no token");
            }
        }
        std::printf("d64 tokens:"); for (auto t : tokens) std::printf(" %u", t); std::printf("\n");
        // host hash + gather (rows read from the GGUF on C:)
        const auto contract = pm::ple_hash_contract();
        std::uint32_t history[2] = {0, 0};
        std::vector<std::uint64_t> ids(static_cast<size_t>(rows) * 16);
        pm::ple_hash_rows(contract, tokens.data(), rows, position0, history, ids.data());
        const auto& tt = model.tensor("per_layer_token_embd.weight");
        std::vector<std::uint8_t> small(ids.size() * 90);
        std::vector<std::uint64_t> local(ids.size());
        FILE* f = std::fopen(gguf_path.empty() ? "Qwen3.8-Flash-Next-UD-Q4_K_XL-merged.gguf" : gguf_path.c_str(), "rb");
        for (size_t i = 0; i < ids.size(); ++i) {
            _fseeki64(f, static_cast<long long>(tt.absolute_offset + ids[i] * 90), SEEK_SET);
            if (std::fread(&small[i * 90], 1, 90, f) != 90) throw std::runtime_error("short table read");
            local[i] = i;
        }
        std::fclose(f);
        std::vector<std::uint16_t> gathered(static_cast<size_t>(rows) * 2560);
        pm::ple_gather_iq4nl_host(small.data(), ids.size(), local.data(), rows, gathered.data(), 2560, 8);
        // device projection + tail from a fresh (zero) convolution state
        const auto key = model.bytes("blk.1.ple_key.weight"), value = model.bytes("blk.1.ple_value.weight");
        auto *dg = dev(gathered); auto* dk = dev(key); auto* dv = dev(value);
        float *keys = nullptr, *values = nullptr, *gates = nullptr;
        CK(cudaMalloc(&keys, static_cast<size_t>(rows) * 10240 * 4)); CK(cudaMalloc(&values, static_cast<size_t>(rows) * 2560 * 4));
        CK(cudaMalloc(&gates, static_cast<size_t>(rows) * 16));
        pm::launch_ple_project(dg, 2560, rows, dk, 2560, 10240, keys, 10240, nullptr);
        pm::launch_ple_project(dg, 2560, rows, dv, 2560, 2560, values, 2560, nullptr);
        std::vector<float> nk = f32_tensor(model, "blk.1.ple_norm_key.weight"), nq = f32_tensor(model, "blk.1.ple_norm_query.weight"),
            nc = f32_tensor(model, "blk.1.ple_norm_conv.weight"), conv = f32_tensor(model, "blk.1.ple_conv1d.weight");
        auto *dnk = dev(nk), *dnq = dev(nq), *dnc = dev(nc), *dconv = dev(conv);
        auto* din = dev(input);
        std::uint16_t* dstate = dev(std::vector<std::uint16_t>(static_cast<size_t>(10240) * 9, 0));
        pm::PleTailArgs a{};
        a.keys = keys; a.key_stride = 10240; a.values = values; a.value_stride = 2560; a.input = din; a.input_stride = 10240;
        a.output = din; a.output_stride = 10240; a.conv_state = dstate; a.norm_key = dnk; a.norm_query = dnq; a.norm_conv = dnc; a.conv = dconv;
        a.gates = gates; a.rows = rows; a.epsilon = 1e-6F;
        pm::launch_ple_tail(a, g_tables, nullptr);
        CK(cudaDeviceSynchronize());
        compare("d64 PLE step output (all rows) == HIP", host(din, input.size()), want);
        const auto state_file = read_file(dir + "\\d64-state\\ple-convolution.raw");
        if (state_file.size() >= 184320) {
            std::vector<std::uint16_t> st(92160); std::memcpy(st.data(), state_file.data(), 184320);
            compare("d64 PLE convolution state == HIP final state", host(dstate, st.size()), st);
        } else std::printf("SKIP d64 PLE convolution state (file missing or %zu B)\n", state_file.size());
        const auto ngram_file = read_file(dir + "\\d64-state\\ngram-history.raw");
        if (ngram_file.size() >= 8) {
            std::uint32_t h[2]; std::memcpy(h, ngram_file.data(), 8);
            report(h[0] == history[0] && h[1] == history[1], "d64 n-gram history == HIP final state",
                std::to_string(history[0]) + "," + std::to_string(history[1]) + " vs " + std::to_string(h[0]) + "," + std::to_string(h[1]));
        } else report(false, "d64 n-gram history file", "missing");
        const auto position_file = read_file(dir + "\\d64-state\\position-history.raw");
        if (position_file.size() >= 8) {
            std::uint64_t p; std::memcpy(&p, position_file.data(), 8);
            report(p == position0 + rows - 1, "d64 position word == last row position", std::to_string(p));
        }
        for (void* p : {static_cast<void*>(dg), static_cast<void*>(dk), static_cast<void*>(dv), static_cast<void*>(keys), static_cast<void*>(values),
                 static_cast<void*>(gates), static_cast<void*>(dnk), static_cast<void*>(dnq), static_cast<void*>(dnc), static_cast<void*>(dconv),
                 static_cast<void*>(din), static_cast<void*>(dstate)}) CK(cudaFree(p));
    } catch (const std::exception& e) { report(false, "d64 fixture replay", e.what()); }
}
int main(int argc, char** argv) {
    std::string tables = "qrtx-build\\tables", gguf, fixture = "qrtx-build\\fx",
        sidecar = "mtp-Qwen3.8-Flash-Next-shared-distillG-Q8_0.gguf";
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--tables") tables = argv[i + 1]; else if (k == "--gguf") gguf = argv[i + 1]; else if (k == "--fixture") fixture = argv[i + 1]; else if (k == "--sidecar") sidecar = argv[i + 1];
    }
    CK(cudaSetDevice(0));
    if (argc > 1 && std::string(argv[1]) == "--mma-only") {
        test_mtp_mma_form();
        std::printf("qwen_plemtp_tests (mma only): %d passed, %d failed\n", g_pass, g_fail);
        return g_fail ? 1 : 0;
    }
    test_ple_hash();
    test_ple_gather(gguf);
    test_ple_project(gguf);
    g_tables = am::load_tables(tables);
    test_ple_tail(gguf);
    test_mtp(gguf, sidecar);
    test_mtp_mma_form();
    test_fixture(fixture, gguf);
    am::free_tables(g_tables);
    std::printf("qwen_plemtp_tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
