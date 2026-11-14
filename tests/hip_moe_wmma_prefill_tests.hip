// Routed MoE prefill: the iu8 WMMA kernels (form 1) against the dot4 kernels (form 0) through
// the complete known_rows::enqueue_prefill entry (prologue, CSR, gate/up + down quantizer, down).
// Every written byte (prologue codes, intermediates, down image, raw slot dots, statuses, the
// summary word) must be identical. Default mode: the bitwise cases. "bench": timing per form.
#include "strix/hip_moe_q4k_a8_known_rows.hpp"
#include "strix/hip_ggml_weight_formats.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/moe_route_bf16.hpp"
#include "strix/hip_qwen_moe_prepared.hpp"
#include "qwen_moe_prepared_fixture.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <bit>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <fstream>
#include <limits>
#include <utility>

namespace {
namespace a8 = strix::moe_q4k_a8;
namespace kr = strix::moe_q4k_a8::known_rows;
void check(hipError_t e, const char* why) {
    if (e != hipSuccess) throw std::runtime_error(std::string(why) + ": " + hipGetErrorString(e));
}
std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL; x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL; x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}
__device__ std::uint64_t device_mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL; x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL; x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}
// fp16 scale bits: ordinary scales; flavor 1 also zero, subnormal, negative and max ones;
// flavor 2 also infinities and NaNs (rows fail); flavor 3: those in the down weights only.
__device__ unsigned short scale_bits(std::uint64_t r, unsigned flavor) {
    const unsigned pick = static_cast<unsigned>(r & 4095U);
    if ((flavor == 1U || flavor == 2U) && pick < 48U) {
        const unsigned short special[12]{0x0000U, 0x8000U, 0x0001U, 0x03ffU, 0x8200U, 0x7bffU, 0xfbffU, 0x0400U, 0x7a00U, 0x7c00U, 0xfc00U, 0x7e00U};
        return special[pick % (flavor == 2U ? 12U : 9U)];
    }
    // 2^-12 .. 2^-6 with random mantissa, sign mostly positive; flavor 4: one binade (2^-9, as production scales),
    // so whole waves take the short dequantization of MoE tune bit 6.
    // (flavor 4 also sets the lowest mantissa bit: the scale pairs of a block then share their lowest bit, as the
    // production blocks do closely enough that every one passes the bit-6 check.)
    const unsigned exponent = flavor == 4U ? 6U : 3U + static_cast<unsigned>((r >> 12U) % 7U);
    const unsigned mantissa = static_cast<unsigned>((r >> 16U) & 1023U) | (flavor == 4U ? 1U : 0U);
    const unsigned sign = ((r >> 30U) & 15U) == 0U ? 0x8000U : 0U;
    return static_cast<unsigned short>(sign | exponent << 10U | mantissa);
}
// Bytes and weights of one block. kind: 12 Q4_K, 13 Q5_K, 7 Q5_1, 8 Q8_0, GSQ 42 Q2_0, 22 IQ2_S, 18 IQ3_XXS, 21 IQ3_S,
// 23 IQ4_XS, 20 IQ4_NL.
__host__ __device__ constexpr bool kind_iq(unsigned kind) { return kind == 22U || kind == 18U || kind == 21U || kind == 23U || kind == 20U; }
__host__ __device__ constexpr unsigned kind_block_bytes(unsigned kind) {
    return kind == 12U ? 144U : kind == 13U ? 176U : kind == 7U ? 24U : kind == 42U ? 18U : kind == 22U ? 82U : kind == 18U ? 98U :
        kind == 21U ? 110U : kind == 23U ? 136U : kind == 20U ? 18U : 34U;
}
__host__ __device__ constexpr unsigned kind_block_elements(unsigned kind) {
    return kind == 12U || kind == 13U || kind == 22U || kind == 18U || kind == 21U || kind == 23U ? 256U : kind == 42U ? 64U : 32U;
}
__host__ __device__ constexpr std::uint64_t kind_row_bytes(unsigned kind, unsigned width) {
    return static_cast<std::uint64_t>(width / kind_block_elements(kind)) * kind_block_bytes(kind);
}
// Random packed blocks with format-correct fp16 headers.
__global__ void fill_blocks(unsigned char* bytes, std::uint64_t blocks, unsigned kind, std::uint64_t seed, unsigned flavor) {
    const unsigned block_bytes = kind_block_bytes(kind);
    for (std::uint64_t b = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x; b < blocks;
         b += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        unsigned char* p = bytes + b * block_bytes;
        std::uint64_t r = device_mix(seed ^ (b * 0x2545f4914f6cdd1dULL));
        for (unsigned i = 0; i < block_bytes; ++i) {
            if ((i & 7U) == 0U) r = device_mix(r);
            p[i] = static_cast<unsigned char>(r >> (8U * (i & 7U)));
        }
        const auto put = [&](unsigned at, unsigned short v) { p[at] = static_cast<unsigned char>(v); p[at + 1U] = static_cast<unsigned char>(v >> 8U); };
        put(0U, scale_bits(device_mix(r + 1U), flavor == 6U ? 0U : flavor));
        if (kind != 8U && kind != 42U && !kind_iq(kind)) put(2U, scale_bits(device_mix(r + 2U), flavor));
        if (kind == 42U && flavor == 6U)  // one code per 32-weight half block: q = 0, 1, 2 or 3 throughout
            for (unsigned i = 0; i < 16U; ++i) p[2U + i] = static_cast<unsigned char>(0x55U * ((b + i / 8U + (r & 1U)) % 4U));
    }
}
struct Buffer {
    void* base = nullptr; std::uint64_t stride = 0, bytes = 0;
    Buffer() = default;
    Buffer(std::uint64_t rows, std::uint64_t row_bytes) : stride((row_bytes + 255U) / 256U * 256U), bytes(rows * stride) {
        check(hipMalloc(&base, bytes), "allocate fixture buffer");
    }
    Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept { *this = std::move(o); }
    Buffer& operator=(Buffer&& o) noexcept { std::swap(base, o.base); std::swap(stride, o.stride); std::swap(bytes, o.bytes); return *this; }
    ~Buffer() { if (base) (void)hipFree(base); }
    template <class T> kr::StridedRows<T> rows() const { return {static_cast<T*>(base), stride}; }
    std::vector<unsigned char> read() const {
        std::vector<unsigned char> out(bytes); check(hipMemcpy(out.data(), base, bytes, hipMemcpyDeviceToHost), "read buffer"); return out;
    }
    void poison(hipStream_t s) const { check(hipMemsetAsync(base, 0xa5, bytes, s), "poison buffer"); }
};
// group > 0: row t routes slot s to expert s + 10 * (t / group), so every expert of a full group of rows has exactly
// `group` assignments (experts >= 10 * ceil(rows / group)). flavor 6 (GSQ Q2_0): extreme / cancelling codes (code bytes
// 0x00, 0x55, 0xaa, 0xff) and inputs of one magnitude (A8 codes +-127).
struct Case {
    const char* name; unsigned experts, rows, gate_type, down_type, flavor; bool duplicates, skew, nan_input = false;
    unsigned group = 0;
    // qpf2k/moe: skew exponent (expert = experts * u^power) and optional real weights: the GGUF file and the absolute
    // byte offsets of one layer's ffn_gate_exps / ffn_up_exps / ffn_down_exps tensors (512 experts).
    double power = 3.0; const char* gguf = nullptr; std::uint64_t gate_at = 0, up_at = 0, down_at = 0;
    // crafted: experts get exact counts at the tile boundaries (0, 1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256, 257,
    // 500 for experts 0..14; the rest share the remaining assignments), filled slot-major so no row repeats an expert;
    // bad_route: one route of row T/3 is out of range (the CSR fails closed).
    bool crafted = false, bad_route = false, full128 = false;  // full128: every expert count a multiple of 128 (including zero)
};
void read_file(const char* path, std::uint64_t at, std::uint64_t bytes, void* device) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("open ") + path);
    std::vector<char> host(bytes);
    f.seekg(static_cast<std::streamoff>(at));
    f.read(host.data(), static_cast<std::streamsize>(bytes));
    if (!f) throw std::runtime_error("short read of GGUF tensor");
    check(hipMemcpy(device, host.data(), bytes, hipMemcpyHostToDevice), "upload real weights");
}
struct Fixture {
    static constexpr unsigned H = 2560U, I = 640U, K = 10U;
    Case c; hipStream_t stream = nullptr;
    a8::Request request;
    Buffer gate, up, down, input, routes, weights, codes, deltas, sums, status, output, clear, dcodes, ddeltas, dsums, rawdots, csr, admission;
    kr::PrefillRows view;
    kr::DirectBOperands direct;
    explicit Fixture(const Case& cs) : c(cs), admission(1, 4U) {
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create stream");
        request.enabled = true; request.input = a8::InputKind::bf16;
        request.shape = {H, I, c.experts, K, c.rows}; request.gate_type = request.up_type = c.gate_type;
        const unsigned gb = kind_block_bytes(c.gate_type), db = kind_block_bytes(c.down_type);
        const std::uint64_t gate_blocks = static_cast<std::uint64_t>(c.experts) * I * (H / kind_block_elements(c.gate_type));
        const std::uint64_t down_blocks = static_cast<std::uint64_t>(c.experts) * H * (I / kind_block_elements(c.down_type));
        gate = Buffer(1, gate_blocks * gb); up = Buffer(1, gate_blocks * gb); down = Buffer(1, down_blocks * db);
        const std::uint64_t seed = mix(c.experts * 131U + c.rows * 7U + c.gate_type * 3U + c.down_type + c.flavor * 99U);
        if (c.gguf) {
            read_file(c.gguf, c.gate_at, gate_blocks * gb, gate.base); read_file(c.gguf, c.up_at, gate_blocks * gb, up.base);
            read_file(c.gguf, c.down_at, down_blocks * db, down.base);
        } else {
        hipLaunchKernelGGL(fill_blocks, dim3(1024), dim3(256), 0, stream, static_cast<unsigned char*>(gate.base), gate_blocks, c.gate_type, seed ^ 1U, c.flavor == 3U || c.flavor == 5U ? 0U : c.flavor);
        hipLaunchKernelGGL(fill_blocks, dim3(1024), dim3(256), 0, stream, static_cast<unsigned char*>(up.base), gate_blocks, c.gate_type, seed ^ 2U,
            c.flavor == 3U || c.flavor == 5U ? 0U : c.flavor);
        // flavor 3: non-finite scales in the down weights only; 5: edge scales in the down weights only.
        hipLaunchKernelGGL(fill_blocks, dim3(1024), dim3(256), 0, stream, static_cast<unsigned char*>(down.base), down_blocks, c.down_type, seed ^ 3U,
            c.flavor == 3U ? 2U : c.flavor == 5U ? 1U : c.flavor);
        check(hipGetLastError(), "fill weights");
        }
        const unsigned T = c.rows;
        input = Buffer(T, 2U * H); routes = Buffer(T, 4U * K); weights = Buffer(T, 4U * K);
        codes = Buffer(T, H); deltas = Buffer(T, 4U * (H / 32U)); sums = Buffer(T, 4U * (H / 32U)); status = Buffer(T, 4U);
        output = Buffer(T, 4U * K * 2U * I); clear = Buffer(T, 4U * H);
        dcodes = Buffer(T, K * I); ddeltas = Buffer(T, 4U * K * (I / 32U)); dsums = Buffer(T, 4U * K * (I / 32U)); rawdots = Buffer(T, 4U * K * H);
        csr = Buffer(1, strix::q51_known_rows::scratch_layout(T, c.experts, K, true).bytes);
        // Host rows: BF16 activations (normal values, outlier channels, edge: tiny/huge/zero blocks).
        std::vector<std::uint16_t> x(static_cast<std::size_t>(input.stride / 2U) * T, 0U);
        std::vector<std::uint32_t> rt(static_cast<std::size_t>(routes.stride / 4U) * T, 0U);
        std::vector<float> rw(static_cast<std::size_t>(weights.stride / 4U) * T, 0.0F);
        std::uint64_t r = seed;
        std::vector<unsigned> crafted_seq;
        if (c.crafted) {
            const unsigned fixed[15]{0U, 1U, 15U, 16U, 17U, 63U, 64U, 65U, 127U, 128U, 129U, 255U, 256U, 257U, 500U};
            unsigned used = 0;
            if (c.full128) {
                const unsigned batches = T * K / 128U, per = batches / c.experts, extra = batches % c.experts;
                if (T * K % 128U) throw std::invalid_argument("full128 needs whole 128-row batches");
                for (unsigned e = 0; e < c.experts && crafted_seq.size() < static_cast<std::size_t>(T) * K; ++e)
                    for (unsigned k = 0; k < 128U * (per + (e < extra ? 1U : 0U)); ++k) crafted_seq.push_back(e);
                used = static_cast<unsigned>(crafted_seq.size());
            } else
            for (unsigned e = 0; e < 15U && e < c.experts; ++e) { for (unsigned k = 0; k < fixed[e] && k < T; ++k) crafted_seq.push_back(e); used += std::min(fixed[e], T); }
            const unsigned rest = c.experts > 15U ? c.experts - 15U : 1U;
            for (unsigned k = 0; crafted_seq.size() < static_cast<std::size_t>(T) * K; ++k) crafted_seq.push_back(15U + (k * rest) / (T * K - used + 1U));
            (void)used;
        }
        const auto bf16 = [](float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U); };
        for (unsigned t = 0; t < T; ++t) {
            for (unsigned k = 0; k < H; ++k) {
                r = mix(r);
                float v = static_cast<float>(static_cast<std::int64_t>(r % 2001U) - 1000) / 500.0F;
                if (k % 97U == 5U) v *= 40.0F;
                if (c.flavor == 6U) {  // 32-value groups all +1, all -1 or alternating (A8 codes +-127)
                    const unsigned kind = (k / 32U + t) % 3U;
                    v = kind == 0U ? 1.0F : kind == 1U ? -1.0F : (k % 2U ? -1.0F : 1.0F);
                }
                if (c.flavor == 1U || c.flavor == 2U) {
                    const unsigned block = (k / 32U + t) % 23U;
                    if (block == 1U) v = 0.0F;
                    else if (block == 2U) v *= 1e-30F;
                    else if (block == 3U) v *= c.flavor == 2U ? 1e30F : 1e5F;
                    else if (block == 4U && k % 32U != 7U) v = 0.0F;
                    else if (block == 5U) v *= 1e-36F; // subnormal deltas
                    else if (block == 6U) v = k % 2U ? 0.5F : -0.5F; // x/delta ties
                }
                x[static_cast<std::size_t>(t) * (input.stride / 2U) + k] = bf16(v);
            }
            if (c.nan_input && t == T / 2U) x[static_cast<std::size_t>(t) * (input.stride / 2U) + 77U] = 0x7fc0U;
            for (unsigned s = 0; s < K; ++s) {
                unsigned e;
                if (c.crafted) {
                    e = std::min(crafted_seq[static_cast<std::size_t>(s) * T + t], c.experts - 1U);
                    if (c.bad_route && t == T / 3U && s == 4U) e = c.experts;
                    rt[static_cast<std::size_t>(t) * (routes.stride / 4U) + s] = e;
                    r = mix(r);
                    rw[static_cast<std::size_t>(t) * (weights.stride / 4U) + s] = 0.05F + static_cast<float>((r >> 20U) % 1000U) / 1000.0F;
                    continue;
                }
                for (;;) {
                    r = mix(r);
                    if (c.group) e = s + 10U * (t / c.group);
                    else if (c.skew) { const double u = static_cast<double>(r >> 11U) / 9007199254740992.0; e = static_cast<unsigned>(c.experts * std::pow(u, c.power)); }
                    else e = static_cast<unsigned>(r % c.experts);
                    e = std::min(e, c.experts - 1U);
                    bool seen = false;
                    for (unsigned q = 0; q < s; ++q) seen |= rt[static_cast<std::size_t>(t) * (routes.stride / 4U) + q] == e;
                    if (!seen || (c.duplicates && (r >> 40U) % 7U == 0U)) break;
                }
                rt[static_cast<std::size_t>(t) * (routes.stride / 4U) + s] = e;
                rw[static_cast<std::size_t>(t) * (weights.stride / 4U) + s] = 0.05F + static_cast<float>((r >> 20U) % 1000U) / 1000.0F;
            }
        }
        check(hipMemcpy(input.base, x.data(), input.bytes, hipMemcpyHostToDevice), "upload input");
        check(hipMemcpy(routes.base, rt.data(), routes.bytes, hipMemcpyHostToDevice), "upload routes");
        check(hipMemcpy(weights.base, rw.data(), weights.bytes, hipMemcpyHostToDevice), "upload route weights");
        view.input = input.rows<const void>(); view.routes = routes.rows<const unsigned>(); view.route_weights = weights.rows<const float>();
        view.codes = codes.rows<signed char>(); view.deltas = deltas.rows<float>(); view.sums = sums.rows<int>(); view.status = status.rows<unsigned>();
        view.output = output.rows<float>(); view.clear = clear.rows<float>();
        view.down_codes = dcodes.rows<signed char>(); view.down_deltas = ddeltas.rows<float>(); view.down_sums = dsums.rows<int>();
        view.rawdots = rawdots.rows<float>();
        const auto layout = strix::q51_known_rows::scratch_layout(T, c.experts, K, true);
        const auto ptr = [&](std::uint64_t o) { return reinterpret_cast<unsigned*>(static_cast<unsigned char*>(csr.base) + o); };
        view.csr = {ptr(layout.header), ptr(layout.offsets), ptr(layout.touched), ptr(layout.rows), ptr(layout.slots), ptr(layout.batch_experts),
            ptr(layout.batch_begins)};
        check(hipStreamSynchronize(stream), "fixture setup");
    }
    ~Fixture() { if (stream) (void)hipStreamDestroy(stream); }
    std::vector<const Buffer*> written() const { return {&codes, &deltas, &sums, &status, &output, &clear, &dcodes, &ddeltas, &dsums, &rawdots}; }
    void enqueue(int form, unsigned rows = 0) {
        kr::enqueue_prefill(request, view, rows ? rows : c.rows, H, static_cast<const unsigned char*>(gate.base),
            static_cast<const unsigned char*>(up.base), c.down_type, static_cast<const unsigned char*>(down.base), stream, form, direct);
    }
    // Every written byte of one run over other weights (same rows, routes and views).
    std::vector<std::vector<unsigned char>> run_as(int form, unsigned gate_type, unsigned down_type, const Buffer& g, const Buffer& u,
        const Buffer& d) {
        for (const auto* b : written()) b->poison(stream);
        csr.poison(stream);
        a8::Request other = request; other.gate_type = other.up_type = gate_type;
        kr::enqueue_prefill(other, view, c.rows, H, static_cast<const unsigned char*>(g.base), static_cast<const unsigned char*>(u.base),
            down_type, static_cast<const unsigned char*>(d.base), stream, form);
        check(hipStreamSynchronize(stream), "run prefill (other weights)");
        std::vector<std::vector<unsigned char>> out;
        for (const auto* b : written()) out.push_back(b->read());
        std::uint32_t summary = 0;
        check(hipMemcpy(&summary, view.csr.summary, 4U, hipMemcpyDeviceToHost), "read summary");
        out.push_back(std::vector<unsigned char>(reinterpret_cast<unsigned char*>(&summary), reinterpret_cast<unsigned char*>(&summary) + 4));
        return out;
    }
    // The first `rows` prompt rows only (a shorter prefill block over the same row views).
    std::vector<unsigned char> run_rawdots(int form, unsigned rows) {
        for (const auto* b : written()) b->poison(stream);
        csr.poison(stream);
        enqueue(form, rows);
        check(hipStreamSynchronize(stream), "run prefill prefix");
        auto out = rawdots.read();
        out.resize(static_cast<std::size_t>(rows) * rawdots.stride);
        return out;
    }
    std::vector<std::vector<unsigned char>> run(int form) {
        for (const auto* b : written()) b->poison(stream);
        csr.poison(stream);
        enqueue(form);
        check(hipStreamSynchronize(stream), "run prefill");
        std::vector<std::vector<unsigned char>> out;
        for (const auto* b : written()) out.push_back(b->read());
        std::uint32_t summary = 0;
        check(hipMemcpy(&summary, view.csr.summary, 4U, hipMemcpyDeviceToHost), "read summary");
        out.push_back(std::vector<unsigned char>(reinterpret_cast<unsigned char*>(&summary), reinterpret_cast<unsigned char*>(&summary) + 4));
        return out;
    }
};
#include "hip_moe_direct_b_fixture.inc"
// qpf2k/moe: order-free 64-bit digest of a device buffer (sum over words of mix(index, word)); equal digests of
// two runs = the same bytes (up to a 2^-64 collision). Reads nothing back but 8 bytes.
__global__ void digest_kernel(const unsigned* words, std::uint64_t n, unsigned long long* out) {
    unsigned long long acc = 0;
    for (std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        acc += device_mix((i << 32U) ^ words[i] ^ 0x5bd1e995ULL * (i >> 32U));
    for (int stride = 16; stride != 0; stride /= 2) acc += __shfl_xor(acc, stride, 32);
    if ((threadIdx.x & 31U) == 0U) atomicAdd(out, acc);
}
std::vector<unsigned long long> digests(Fixture& f) {
    unsigned long long* d = nullptr;
    const auto bufs = f.written();
    check(hipMalloc(&d, 8U * (bufs.size() + 1U)), "digest alloc");
    check(hipMemsetAsync(d, 0, 8U * (bufs.size() + 1U), f.stream), "digest clear");
    for (std::size_t k = 0; k < bufs.size(); ++k)
        hipLaunchKernelGGL(digest_kernel, dim3(1024), dim3(256), 0, f.stream, static_cast<const unsigned*>(bufs[k]->base), bufs[k]->bytes / 4U, d + k);
    hipLaunchKernelGGL(digest_kernel, dim3(1), dim3(32), 0, f.stream, static_cast<const unsigned*>(static_cast<const void*>(f.view.csr.summary)), std::uint64_t{32}, d + bufs.size());
    std::vector<unsigned long long> out(bufs.size() + 1U);
    check(hipMemcpyAsync(out.data(), d, 8U * out.size(), hipMemcpyDeviceToHost, f.stream), "digest read");
    check(hipStreamSynchronize(f.stream), "digest");
    (void)hipFree(d);
    return out;
}
// The engine's ordered merge (known_moe_prefill_merge_kernel) over the fixture's views: guards, slot-order v_fmac chain with
// the route weight as the SGPR operand, accumulation (PrefillRows::clear) and the non-finite flag.
__global__ void reference_merge_kernel(kr::PrefillRows v, unsigned* header, unsigned hidden, unsigned experts, unsigned top_k) {
    const unsigned row = blockIdx.y;
    auto* status = reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(v.status.base) + static_cast<std::uint64_t>(row) * v.status.stride);
    if (header[0] || *status) return;
    const unsigned out = blockIdx.x * blockDim.x + threadIdx.x;
    if (out >= hidden) return;
    const auto* routes = reinterpret_cast<const unsigned*>(reinterpret_cast<const unsigned char*>(v.routes.base) + static_cast<std::uint64_t>(row) * v.routes.stride);
    for (unsigned slot = 0; slot < top_k; ++slot) if (routes[slot] >= experts) return;
    const auto* raw = reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(v.rawdots.base) + static_cast<std::uint64_t>(row) * v.rawdots.stride);
    const auto* w = reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(v.route_weights.base) + static_cast<std::uint64_t>(row) * v.route_weights.stride);
    float dots[10];
#pragma unroll
    for (unsigned slot = 0; slot < 10U; ++slot) if (slot < top_k) dots[slot] = raw[static_cast<std::uint64_t>(slot) * hidden + out];
    float current = 0.0F;
#pragma unroll
    for (unsigned slot = 0; slot < 10U; ++slot) if (slot < top_k) {
        const float weight = w[slot];
        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(current) : "s"(weight), "v"(dots[slot]));
    }
    auto* acc = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(v.clear.base) + static_cast<std::uint64_t>(row) * v.clear.stride);
    acc[out] = current;
    if ((__float_as_uint(current) & 0x7fffffffU) >= 0x7f7f8000U) { atomicOr(status, 8U); atomicOr(header, 8U); }
}
// qpf2k/moe-mi: reference_merge_kernel when the prologue left the accumulation unzeroed (ROUTE bit 8), mirroring the engine's
// known_moe_prefill_merge_kernel_z: a zero store on each path where the base merge returns without a store.
__global__ void reference_merge_kernel_z(kr::PrefillRows v, unsigned* header, unsigned hidden, unsigned experts, unsigned top_k) {
    const unsigned row = blockIdx.y;
    auto* status = reinterpret_cast<unsigned*>(reinterpret_cast<unsigned char*>(v.status.base) + static_cast<std::uint64_t>(row) * v.status.stride);
    auto* acc = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(v.clear.base) + static_cast<std::uint64_t>(row) * v.clear.stride);
    const unsigned out = blockIdx.x * blockDim.x + threadIdx.x;
    if (header[0] || *status) { if (out < hidden) acc[out] = 0.0F; return; }
    if (out >= hidden) return;
    const auto* routes = reinterpret_cast<const unsigned*>(reinterpret_cast<const unsigned char*>(v.routes.base) + static_cast<std::uint64_t>(row) * v.routes.stride);
    for (unsigned slot = 0; slot < top_k; ++slot) if (routes[slot] >= experts) { acc[out] = 0.0F; return; }
    const auto* raw = reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(v.rawdots.base) + static_cast<std::uint64_t>(row) * v.rawdots.stride);
    const auto* w = reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(v.route_weights.base) + static_cast<std::uint64_t>(row) * v.route_weights.stride);
    float dots[10];
#pragma unroll
    for (unsigned slot = 0; slot < 10U; ++slot) if (slot < top_k) dots[slot] = raw[static_cast<std::uint64_t>(slot) * hidden + out];
    float current = 0.0F;
#pragma unroll
    for (unsigned slot = 0; slot < 10U; ++slot) if (slot < top_k) {
        const float weight = w[slot];
        asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(current) : "s"(weight), "v"(dots[slot]));
    }
    acc[out] = current;
    if ((__float_as_uint(current) & 0x7fffffffU) >= 0x7f7f8000U) { atomicOr(status, 8U); atomicOr(header, 8U); }
}
// MoE merge+commit: the engine's known_moe_prefill_commit_kernel over the fixture's views (BF16 output rows out16), and the
// merge+commit candidate built from the engine's own body (src/hip_moe_prefill_commit.inc).
template <class T> __device__ T* fixture_row(kr::StridedRows<T> v, unsigned r) {
    return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(v.base) + static_cast<std::uint64_t>(r) * v.stride);
}
__global__ void reference_commit_kernel(kr::PrefillRows v, unsigned* header, unsigned row_count, unsigned hidden,
    kr::StridedRows<std::uint16_t> out16, unsigned* record) {
    const unsigned failure = header[0];
    if (blockIdx.x == 0U && blockIdx.y == 0U) {
        __shared__ unsigned first_bad;
        if (threadIdx.x == 0U) first_bad = row_count;
        __syncthreads();
        for (unsigned r = threadIdx.x; r < row_count; r += blockDim.x)
            if (*fixture_row(v.status, r)) { atomicMin(&first_bad, r); break; }
        __syncthreads();
        if (threadIdx.x == 0U) {
            if (failure & 7U) for (const auto word : {1U, 2U, 4U, 5U}) header[word] = 0U;
            if (failure & 7U) for (unsigned word = 8U; word < 24U; ++word) header[word] = 0U;
            if (first_bad < row_count)
                strix::merge_deferred_moe_first_error(*reinterpret_cast<strix::DeferredMoeFirstError*>(record),
                    *fixture_row(v.status, first_bad), strix::DeferredMoeDomain::q4k_a8);
        }
    }
    if (failure) return;
    const unsigned out = blockIdx.x * blockDim.x + threadIdx.x;
    if (out < hidden) {
        const auto bits = __float_as_uint(fixture_row(v.clear, blockIdx.y)[out]);
        fixture_row(out16, blockIdx.y)[out] = static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
    }
}
using strix::DeferredMoeFirstError;
using strix::DeferredMoeDomain;
using strix::merge_deferred_moe_first_error;
#include "../src/hip_moe_route_bf16.inc"
#include "../src/hip_moe_prefill_commit.inc"
// Adapter to the engine's shared merge_z / logical-commit bodies.
struct FixtureTailView {
    kr::StridedRows<const unsigned> routes;
    kr::StridedRows<const float> route_weights;
    kr::StridedRows<unsigned> status;
    kr::StridedRows<float> rawdots, accumulation;
    kr::StridedRows<std::uint16_t> output;
};
__device__ FixtureTailView fixture_tail_view(kr::PrefillRows v, kr::StridedRows<std::uint16_t> output) {
    return {v.routes, v.route_weights, v.status, v.rawdots, v.clear, output};
}
template <bool Acc, bool RouteBF16 = false, bool RouteI8 = false>
__global__ void candidate_merge_commit_kernel(kr::PrefillRows v, unsigned* header, unsigned hidden, unsigned experts, unsigned top_k,
    kr::StridedRows<std::uint16_t> out16) {
    moe_prefill_merge_z_body<true, Acc, RouteBF16, RouteI8>(fixture_tail_view(v, out16), header, hidden, experts, top_k,
        [](auto v, unsigned r) { return fixture_row(v, r); }, v.unsafe);
}
template <bool Acc, bool RouteBF16 = false, bool RouteI8 = false>
__global__ void candidate_merge_commit_vec4_kernel(kr::PrefillRows v, unsigned* header, unsigned hidden, unsigned experts, unsigned top_k,
    kr::StridedRows<std::uint16_t> out16) {
    moe_prefill_merge_z_body_vec4<Acc, RouteBF16, RouteI8>(fixture_tail_view(v, out16), header, hidden, experts, top_k,
        [](auto v, unsigned r) { return fixture_row(v, r); }, v.unsafe);
}
dim3 vec4_grid(unsigned rows) { return dim3((Fixture::H / 4U + 127U) / 128U, rows); }
// As the host: the vec4 body for widths divisible by four, the scalar MC body otherwise (the fixture's H-1 case).
template <bool Acc>
void candidate_merge_commit_vec4(hipStream_t stream, const kr::PrefillRows& v, unsigned* header, unsigned hidden, unsigned experts,
    unsigned top_k, kr::StridedRows<std::uint16_t> out16, unsigned rows) {
    if (hidden % 4U) hipLaunchKernelGGL(candidate_merge_commit_kernel<Acc>, dim3((Fixture::H + 255U) / 256U, rows), dim3(256), 0, stream,
        v, header, hidden, experts, top_k, out16);
    else hipLaunchKernelGGL(candidate_merge_commit_vec4_kernel<Acc>, vec4_grid(rows), dim3(128), 0, stream, v, header, hidden, experts, top_k, out16);
}
__global__ void candidate_finish_kernel(kr::PrefillRows v, unsigned* header, unsigned row_count, unsigned hidden,
    kr::StridedRows<std::uint16_t> out16, unsigned* record) {
    moe_prefill_finish_body<32>(fixture_tail_view(v, out16), header, row_count, hidden, record, v.unsafe,
        [](auto v, unsigned r) { return fixture_row(v, r); }, [](std::uint16_t* p, float value) {
            const auto bits=__float_as_uint(value);
            *p=static_cast<std::uint16_t>((bits+0x7fffU+((bits>>16U)&1U))>>16U);
        });
}
// Post-down fixture mutations must recompute the private proof before selecting a tail.
// This is test-only: production checks these same values in the CSR count and down epilogue.
__global__ void fixture_certificate(kr::PrefillRows v, unsigned rows, unsigned hidden, unsigned top_k) {
    const unsigned t=blockIdx.x*blockDim.x+threadIdx.x;
    bool bad=false;
    for (unsigned i=t;i<rows*top_k;i+=gridDim.x*blockDim.x)
        bad|=(__float_as_uint(fixture_row(v.route_weights,i/top_k)[i%top_k])&0x7fffffffU)>0x3f800000U;
    for (unsigned i=t;i<rows*top_k*hidden;i+=gridDim.x*blockDim.x)
        bad|=(__float_as_uint(fixture_row(v.rawdots,i/(top_k*hidden))[i%(top_k*hidden)])&0x7fffffffU)>=0x7b800000U;
    const unsigned mask=static_cast<unsigned>(__ballot(bad));
    if(mask&&(threadIdx.x&31U)==0U)atomicOr(v.unsafe,1U);
}
const char* const names[] = {"codes", "deltas", "sums", "status", "intermediates", "clear", "down_codes", "down_deltas", "down_sums", "rawdots", "summary"};
bool extreme_coverage(const Fixture& f);
bool compare(const Case& c, unsigned repeats = 1) {
    Fixture f(c);
    if (c.gate_type == 42U && (kr::prefill_kernel_form(f.request, c.down_type, 0) != kr::PrefillKernelForm::dot4 ||
                               kr::prefill_kernel_form(f.request, c.down_type, 1) != kr::PrefillKernelForm::iu8_wmma)) {
        std::printf("%-34s forms 0/1 are not dot4/iu8\n", c.name);
        return false;
    }
    const auto ref = f.run(0), got = f.run(1);
    for (unsigned r = 1; r < repeats; ++r)
        if (f.run(1) != got) { std::printf("%-34s form 1 repeat %u differs from its first run\n", c.name, r); return false; }
    // The parallel CSR builder (form 1) against the serial one (form 2, same batch cap): the whole
    // CSR allocation, summary words included, and every output byte.
    const auto csr_parallel = f.csr.read();
    const auto serial = f.run(2);
    const auto csr_serial = f.csr.read();
    bool ok = csr_parallel == csr_serial && serial == got;
    if (!ok) std::printf("  parallel CSR builder differs from the serial one (csr %s, outputs %s)\n",
        csr_parallel == csr_serial ? "same" : "differ", serial == got ? "same" : "differ");
    std::size_t nan_words = 0, nan_payload = 0, nonfinite = 0;
    for (std::size_t v = 0; v < ref.size(); ++v) {
        // FP32 views: a NaN's sign/payload is not part of the contract (both kernels write the raw
        // non-finite g of a failed (row, slot) only as a diagnostic); NaN positions must match.
        const bool fp32 = v == 1U || v == 4U || v == 5U || v == 7U || v == 9U;
        std::size_t diff = 0, first = ref[v].size();
        for (std::size_t k = 0; k < ref[v].size(); k += fp32 ? 4U : 1U) {
            if (fp32) {
                std::uint32_t a, b; std::memcpy(&a, ref[v].data() + k, 4); std::memcpy(&b, got[v].data() + k, 4);
                const bool an = (a & 0x7fffffffU) > 0x7f800000U, bn = (b & 0x7fffffffU) > 0x7f800000U;
                nan_words += an; nonfinite += (a & 0x7f800000U) == 0x7f800000U && a != 0xa5a5a5a5U;
                if (a == b) continue;
                if (an && bn) { ++nan_payload; continue; }
            } else if (ref[v][k] == got[v][k]) continue;
            ++diff; first = std::min(first, k);
        }
        if (!diff) continue;
        ok = false;
        std::printf("  %s: %zu %s differ, first at byte %zu (ref %02x got %02x)\n", names[v], diff, fp32 ? "words" : "bytes", first,
            ref[v][first], got[v][first]);
    }
    std::uint32_t summary; std::memcpy(&summary, ref.back().data(), 4);
    std::size_t bad_rows = 0;
    for (unsigned t = 0; t < c.rows; ++t) { std::uint32_t s; std::memcpy(&s, ref[3].data() + t * f.status.stride, 4); bad_rows += s != 0U; }
    if (c.gate_type == 42U && c.flavor == 6U) ok = extreme_coverage(f) && ok;
    if (c.gate_type == 42U && (c.flavor == 0U || c.flavor == 6U) && !c.nan_input && (summary != 0U || bad_rows != 0U)) {
        std::printf("  unexpected failed rows on a clean fixture\n"); ok = false;
    }
    std::printf("%-34s %s (summary %#x, rows with status %zu/%u, non-finite words %zu, NaN %zu, NaN sign/payload differences %zu)\n",
        c.name, ok ? "PASS" : "FAIL", summary, bad_rows, c.rows, nonfinite, nan_words, nan_payload);
    return ok;
}
// FP32 reference of the routed MoE with unquantized activations: exact FP32 weight values, BF16
// input as FP32, sequential FP32 sums, h = silu(g) * u in FP32 (accuracy mode only).
__device__ float half_at(const unsigned char* p) { return __half2float(__ushort_as_half(static_cast<unsigned short>(p[0] | p[1] << 8))); }
__device__ float weight_at(const unsigned char* row, unsigned k, unsigned kind) {
    if (kind == 12U || kind == 13U) {
        const unsigned char* b = row + (k / 256U) * (kind == 12U ? 144U : 176U);
        const unsigned j = (k % 256U) / 32U, i = k % 32U;
        const unsigned char* sc = b + 4;
        unsigned s, m;
        if (j < 4U) { s = sc[j] & 63U; m = sc[j + 4U] & 63U; }
        else { s = (sc[j + 4U] & 15U) | (static_cast<unsigned>(sc[j - 4U] >> 6U) << 4U); m = static_cast<unsigned>(sc[j + 4U] >> 4U) | (static_cast<unsigned>(sc[j] >> 6U) << 4U); }
        const unsigned char* qs = b + (kind == 12U ? 16U : 48U);
        const unsigned byte = qs[32U * (j / 2U) + i];
        unsigned q = j % 2U ? byte >> 4U : byte & 15U;
        if (kind == 13U) q |= ((b[16U + i] >> j) & 1U) << 4U;
        return half_at(b) * static_cast<float>(s) * static_cast<float>(q) - half_at(b + 2) * static_cast<float>(m);
    }
    if (kind == 7U) {
        const unsigned char* b = row + (k / 32U) * 24U;
        const unsigned i = k % 32U;
        const unsigned qh = static_cast<unsigned>(b[4]) | static_cast<unsigned>(b[5]) << 8U | static_cast<unsigned>(b[6]) << 16U | static_cast<unsigned>(b[7]) << 24U;
        const unsigned q = (i < 16U ? b[8U + i] & 15U : b[8U + i - 16U] >> 4U) | ((qh >> i) & 1U) << 4U;
        return half_at(b) * static_cast<float>(q) + half_at(b + 2);
    }
    if (kind_iq(kind)) {
        const unsigned char* b = row + (k / kind_block_elements(kind)) * kind_block_bytes(kind);
        const unsigned e = k % kind_block_elements(kind);
        namespace hg = strix::hip_ggml;
        return kind == 22U ? hg::block_element<hg::type_iq2_s>(b, e) : kind == 18U ? hg::block_element<hg::type_iq3_xxs>(b, e) :
            kind == 21U ? hg::block_element<hg::type_iq3_s>(b, e) : kind == 23U ? hg::block_element<hg::type_iq4_xs>(b, e) :
            hg::block_element<hg::type_iq4_nl>(b, e);
    }
    if (kind == 42U) {
        const unsigned char* b = row + (k / 64U) * 18U;
        const unsigned i = k % 64U;
        return half_at(b) * static_cast<float>(static_cast<int>((b[2U + i / 4U] >> (2U * (i % 4U))) & 3U) - 1);
    }
    const unsigned char* b = row + (k / 32U) * 34U;
    return half_at(b) * static_cast<float>(static_cast<signed char>(b[2U + k % 32U]));
}
// Acc = double: the float64 oracle (exact FP32 weight values and BF16 inputs, FP64 sums, h and SiLU in FP64).
template <class Acc>
__global__ void reference_gate_up(const unsigned char* gate, const unsigned char* up, unsigned kind, kr::StridedRows<const void> input,
    kr::StridedRows<const unsigned> routes, unsigned T, unsigned H, unsigned I, unsigned K, Acc* h) {
    const unsigned o = blockIdx.x * blockDim.x + threadIdx.x, a = blockIdx.y;
    if (o >= I || a >= T * K) return;
    const unsigned r = a / K, s = a % K;
    const auto* x = reinterpret_cast<const unsigned short*>(static_cast<const unsigned char*>(input.base) + r * input.stride);
    const unsigned e = *(reinterpret_cast<const unsigned*>(reinterpret_cast<const unsigned char*>(routes.base) + r * routes.stride) + s);
    const std::uint64_t row_bytes = kind_row_bytes(kind, H);
    const unsigned char* gr = gate + (static_cast<std::uint64_t>(e) * I + o) * row_bytes;
    const unsigned char* ur = up + (static_cast<std::uint64_t>(e) * I + o) * row_bytes;
    Acc g = 0, u = 0;
    for (unsigned k = 0; k < H; ++k) {
        const Acc v = __uint_as_float(static_cast<unsigned>(x[k]) << 16U);
        g = fma(v, static_cast<Acc>(weight_at(gr, k, kind)), g); u = fma(v, static_cast<Acc>(weight_at(ur, k, kind)), u);
    }
    h[static_cast<std::uint64_t>(a) * I + o] = (g / (Acc(1) + exp(-g))) * u;
}
template <class Acc>
__global__ void reference_down(const unsigned char* down, unsigned kind, kr::StridedRows<const unsigned> routes, const Acc* h,
    unsigned T, unsigned H, unsigned I, unsigned K, Acc* raw) {
    const unsigned n = blockIdx.x * blockDim.x + threadIdx.x, a = blockIdx.y;
    if (n >= H || a >= T * K) return;
    const unsigned r = a / K, s = a % K;
    const unsigned e = *(reinterpret_cast<const unsigned*>(reinterpret_cast<const unsigned char*>(routes.base) + r * routes.stride) + s);
    const unsigned char* row = down + (static_cast<std::uint64_t>(e) * H + n) * kind_row_bytes(kind, I);
    Acc acc = 0;
    for (unsigned k = 0; k < I; ++k) acc = fma(h[static_cast<std::uint64_t>(a) * I + k], static_cast<Acc>(weight_at(row, k, kind)), acc);
    raw[static_cast<std::uint64_t>(a) * H + n] = acc;
}
// Error of the raw slot dots and of the route-weighted per-row sums of forms 1 (A8) and 3 (FP16)
// against the reference; form 3 must be deterministic and fail no row.
// Acc = double (GSQ mode): the float64 oracle; form 1 is then the GSQ dot4 form (GSQ types have no iu8 WMMA form).
template <class Acc = float>
bool accuracy(const Case& c) {
    Fixture f(c);
    const unsigned T = c.rows, H = Fixture::H, I = Fixture::I, K = Fixture::K;
    Acc *h = nullptr, *raw = nullptr;
    check(hipMalloc(&h, sizeof(Acc) * T * K * I), "reference h"); check(hipMalloc(&raw, sizeof(Acc) * T * K * H), "reference raw");
    hipLaunchKernelGGL(reference_gate_up<Acc>, dim3((I + 127U) / 128U, T * K), dim3(128), 0, f.stream, static_cast<const unsigned char*>(f.gate.base),
        static_cast<const unsigned char*>(f.up.base), c.gate_type, f.view.input, f.view.routes, T, H, I, K, h);
    hipLaunchKernelGGL(reference_down<Acc>, dim3((H + 127U) / 128U, T * K), dim3(128), 0, f.stream, static_cast<const unsigned char*>(f.down.base),
        c.down_type, f.view.routes, h, T, H, I, K, raw);
    check(hipStreamSynchronize(f.stream), "reference");
    std::vector<Acc> ref(static_cast<std::size_t>(T) * K * H);
    check(hipMemcpy(ref.data(), raw, ref.size() * sizeof(Acc), hipMemcpyDeviceToHost), "read reference");
    (void)hipFree(h); (void)hipFree(raw);
    std::vector<float> rw(f.weights.bytes / 4U);
    check(hipMemcpy(rw.data(), f.weights.base, f.weights.bytes, hipMemcpyDeviceToHost), "read route weights");
    bool ok = true;
    std::vector<unsigned char> first3, first1;
    for (const int form : {1, 3, 3}) {
        const auto out = f.run(form);
        const auto& rd = out[9];
        if (form == 1) first1 = rd;
        std::uint32_t summary; std::memcpy(&summary, out.back().data(), 4);
        if (form == 3 && first3.empty()) first3 = rd;
        else if (form == 3 && rd != first3) { std::printf("  form 3 is not deterministic\n"); ok = false; }
        double e2 = 0, r2 = 0, emax = 0, m2 = 0, mr2 = 0;
        for (unsigned t = 0; t < T; ++t) {
            const float* got = reinterpret_cast<const float*>(rd.data() + static_cast<std::size_t>(t) * f.rawdots.stride);
            const float* w = rw.data() + static_cast<std::size_t>(t) * (f.weights.stride / 4U);
            for (unsigned n = 0; n < H; ++n) {
                double mg = 0, mr = 0;
                for (unsigned s = 0; s < K; ++s) {
                    const double x = ref[(static_cast<std::size_t>(t) * K + s) * H + n], y = got[s * H + n];
                    e2 += (y - x) * (y - x); r2 += x * x; emax = std::max(emax, std::abs(y - x));
                    mg += w[s] * y; mr += w[s] * x;
                }
                m2 += (mg - mr) * (mg - mr); mr2 += mr * mr;
            }
        }
        const double rms = std::sqrt(r2 / (static_cast<double>(T) * K * H));
        std::printf("%-34s form %d: summary %#x, raw dots rel RMS error %.3e, max |error| %.3e (%.2f RMS), merged rel RMS error %.3e\n",
            c.name, form, summary, std::sqrt(e2 / r2), emax, emax / rms, std::sqrt(m2 / mr2));
        if (summary) ok = false;
        if constexpr (std::is_same_v<Acc, double>) {
            // FP16 activations (~2^-12 relative per value) bound the FP16 form well below 2e-3; A8 codes (delta / 2 per
            // value) bound the dot4 form below 3e-2. Non-finite errors fail.
            const double limit = form == 3 ? 2e-3 : 3e-2, raw_rel = std::sqrt(e2 / r2), merged = std::sqrt(m2 / mr2);
            if (!(raw_rel <= limit) || !(merged <= limit) || !std::isfinite(emax)) {
                std::printf("  form %d error above the limit %.0e\n", form, limit);
                ok = false;
            }
        }
    }
    {
        // Form 3 against form 1 directly (GSQ: FP16 WMMA vs the dot4 A8 form).
        double d2 = 0, x2 = 0;
        for (unsigned t = 0; t < T; ++t) for (unsigned n = 0; n < K * H; ++n) {
            float a, b;
            std::memcpy(&a, first1.data() + static_cast<std::size_t>(t) * f.rawdots.stride + 4U * n, 4);
            std::memcpy(&b, first3.data() + static_cast<std::size_t>(t) * f.rawdots.stride + 4U * n, 4);
            d2 += (static_cast<double>(b) - a) * (static_cast<double>(b) - a); x2 += static_cast<double>(a) * a;
        }
        std::printf("%-34s form 3 vs form 1: raw dots rel RMS difference %.3e\n", c.name, std::sqrt(d2 / x2));
    }
    // A shorter block (other CSR batches, other batch neighbours) gives each of its rows the same bytes.
    if (c.rows >= 32U) {
        const unsigned half = c.rows / 2U / 8U * 8U;
        auto prefix = first3;
        prefix.resize(static_cast<std::size_t>(half) * f.rawdots.stride);
        const bool same = f.run_rawdots(3, half) == prefix;
        std::printf("%-34s form 3: first %u rows as their own block %s\n", c.name, half, same ? "byte-identical" : "DIFFER");
        ok = ok && same;
    }
    return ok;
}
// GSQ equivalence fixtures. Q2_0 gate/up rows whose four blocks per 256 weights share one positive scale hold the
// same values as Q4_K blocks with sc = m = 1 and dmin = d (d*q - d = d*(q - 1), and q = 1 gives +0 both ways); every
// Q2_0 down block holds the values of two Q8_0 blocks with codes q - 1 (d * float(q - 1) both ways, any scale).
__device__ unsigned q20_code(const unsigned char* block, unsigned k) { return (block[2U + k / 4U] >> (2U * (k % 4U))) & 3U; }
__global__ void q20_share_scales(unsigned char* q20, std::uint64_t superblocks) {
    for (std::uint64_t s = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x; s < superblocks;
         s += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        unsigned char* b = q20 + s * 72U;
        const unsigned char lo = b[0], hi = static_cast<unsigned char>(b[1] & 0x7fU);
        for (unsigned j = 0; j < 4U; ++j) { b[18U * j] = lo; b[18U * j + 1U] = hi; }
    }
}
__global__ void q20_to_q4k(const unsigned char* q20, unsigned char* q4k, std::uint64_t superblocks) {
    for (std::uint64_t s = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x; s < superblocks;
         s += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const unsigned char* b = q20 + s * 72U;
        unsigned char* o = q4k + s * 144U;
        o[0] = b[0]; o[1] = b[1]; o[2] = b[0]; o[3] = b[1];
        for (unsigned j = 0; j < 8U; ++j) o[4U + j] = 1U;
        for (unsigned j = 0; j < 4U; ++j) o[12U + j] = 0x11U;
        for (unsigned p = 0; p < 4U; ++p)
            for (unsigned i = 0; i < 32U; ++i)
                o[16U + 32U * p + i] = static_cast<unsigned char>(q20_code(b + 18U * p, i) | q20_code(b + 18U * p, 32U + i) << 4U);
    }
}
__global__ void q20_to_q80(const unsigned char* q20, unsigned char* q80, std::uint64_t blocks) {
    for (std::uint64_t b = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x; b < blocks;
         b += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const unsigned char* src = q20 + b * 18U;
        for (unsigned h = 0; h < 2U; ++h) {
            unsigned char* o = q80 + (2U * b + h) * 34U;
            o[0] = src[0]; o[1] = src[1];
            for (unsigned i = 0; i < 32U; ++i) o[2U + i] = static_cast<unsigned char>(static_cast<int>(q20_code(src, 32U * h + i)) - 1);
        }
    }
}
// Every written byte of two runs (FP32 views: NaN positions must match, not their sign/payload).
bool same_outputs(const std::vector<std::vector<unsigned char>>& ref, const std::vector<std::vector<unsigned char>>& got, const char* what) {
    bool ok = true;
    for (std::size_t v = 0; v < ref.size(); ++v) {
        const bool fp32 = v == 1U || v == 4U || v == 5U || v == 7U || v == 9U;
        std::size_t diff = 0, first = ref[v].size();
        for (std::size_t k = 0; k < ref[v].size(); k += fp32 ? 4U : 1U) {
            if (fp32) {
                std::uint32_t a, b; std::memcpy(&a, ref[v].data() + k, 4); std::memcpy(&b, got[v].data() + k, 4);
                if (a == b) continue;
                if ((a & 0x7fffffffU) > 0x7f800000U && (b & 0x7fffffffU) > 0x7f800000U) continue;
            } else if (ref[v][k] == got[v][k]) continue;
            ++diff; first = std::min(first, k);
        }
        if (!diff) continue;
        ok = false;
        std::printf("  %s: %s: %zu %s differ, first at byte %zu\n", what, names[v], diff, fp32 ? "words" : "bytes", first);
    }
    return ok;
}
// Largest CSR batch of the last run (assignments of one expert per CTA).
unsigned largest_batch(const Fixture& f) {
    const auto layout = strix::q51_known_rows::scratch_layout(f.c.rows, f.c.experts, Fixture::K, true);
    const auto bytes = f.csr.read();
    const auto word = [&](std::uint64_t at) { std::uint32_t v; std::memcpy(&v, bytes.data() + at, 4); return v; };
    const unsigned batches = word(layout.header + 8U);
    unsigned largest = 0;
    for (unsigned b = 0; b < batches; ++b) largest = std::max(largest, word(layout.batch_begins + 4U * (b + 1U)) - word(layout.batch_begins + 4U * b));
    return largest;
}
// Flavor 6 reaches the extreme integers of the Q2_0 A8 group: I = +-12192 (codes +-127 against q = 3), S = +-4064 and
// I - S = 0 with S != 0 (q = 1); counted over rows 0..7, every slot, outputs 0..63 of the gate weights.
bool extreme_coverage(const Fixture& f) {
    const auto codes = f.codes.read(), routes = f.routes.read(), gate = f.gate.read();
    unsigned long long hit_i = 0, hit_neg_i = 0, hit_s = 0, cancel = 0;
    const std::uint64_t row_bytes = kind_row_bytes(42U, Fixture::H);
    for (unsigned t = 0; t < 8U; ++t) for (unsigned slot = 0; slot < Fixture::K; ++slot) {
        std::uint32_t e; std::memcpy(&e, routes.data() + t * f.routes.stride + 4U * slot, 4);
        for (unsigned o = 0; o < 64U; ++o) {
            const unsigned char* w = gate.data() + (static_cast<std::uint64_t>(e) * Fixture::I + o) * row_bytes;
            for (unsigned g = 0; g < Fixture::H / 32U; ++g) {
                int dot = 0, sum = 0;
                for (unsigned k = 0; k < 32U; ++k) {
                    const unsigned idx = 32U * g + k;
                    const int a = static_cast<signed char>(codes[t * f.codes.stride + idx]);
                    const int q = (w[(idx / 64U) * 18U + 2U + (idx % 64U) / 4U] >> (2U * (idx % 4U))) & 3;
                    dot += a * q; sum += a;
                }
                hit_i += dot == 12192; hit_neg_i += dot == -12192; hit_s += sum == 4064 || sum == -4064; cancel += sum != 0 && dot == sum;
            }
        }
    }
    std::printf("  extreme integers: I=+12192 %llu, I=-12192 %llu, |S|=4064 %llu, I=S!=0 %llu\n", hit_i, hit_neg_i, hit_s, cancel);
    return hit_i && hit_neg_i && hit_s && cancel;
}
bool gsq_equivalence(const Case& c, std::initializer_list<int> forms) {
    Fixture f(c);
    const std::uint64_t superblocks = static_cast<std::uint64_t>(c.experts) * Fixture::I * (Fixture::H / 256U);
    const std::uint64_t down_blocks = static_cast<std::uint64_t>(c.experts) * Fixture::H * (Fixture::I / 64U);
    hipLaunchKernelGGL(q20_share_scales, dim3(1024), dim3(256), 0, f.stream, static_cast<unsigned char*>(f.gate.base), superblocks);
    hipLaunchKernelGGL(q20_share_scales, dim3(1024), dim3(256), 0, f.stream, static_cast<unsigned char*>(f.up.base), superblocks);
    Buffer g4(1, superblocks * 144U), u4(1, superblocks * 144U), d8(1, down_blocks * 68U);
    hipLaunchKernelGGL(q20_to_q4k, dim3(1024), dim3(256), 0, f.stream, static_cast<const unsigned char*>(f.gate.base), static_cast<unsigned char*>(g4.base), superblocks);
    hipLaunchKernelGGL(q20_to_q4k, dim3(1024), dim3(256), 0, f.stream, static_cast<const unsigned char*>(f.up.base), static_cast<unsigned char*>(u4.base), superblocks);
    hipLaunchKernelGGL(q20_to_q80, dim3(1024), dim3(256), 0, f.stream, static_cast<const unsigned char*>(f.down.base), static_cast<unsigned char*>(d8.base), down_blocks);
    check(hipGetLastError(), "equivalence images"); check(hipStreamSynchronize(f.stream), "equivalence images");
    bool ok = true;
    const bool clean = (c.flavor == 0U || c.flavor == 6U) && !c.nan_input;
    a8::Request production = f.request; production.gate_type = production.up_type = 12U;
    for (const int form : forms) {
        // The library's own selection: FP16 kernels for both runs.
        if (kr::prefill_kernel_form(f.request, 42U, form) != kr::PrefillKernelForm::f16_wmma ||
            kr::prefill_kernel_form(production, 8U, form) != kr::PrefillKernelForm::f16_wmma) {
            std::printf("%-40s form %2d: not the FP16 kernels\n", c.name, form); ok = false; continue;
        }
        const auto ref = f.run_as(form, 12U, 8U, g4, u4, d8), got = f.run(form);
        std::uint32_t summary; std::memcpy(&summary, ref.back().data(), 4);
        const bool same = same_outputs(ref, got, c.name);
        // CSR batches: up to 128 assignments when the effective tune keeps bit 1, else 64; forced groups fill them.
        const unsigned tune = kr::prefill_effective_tune(f.request, c.rows, 42U, form), cap = (tune & 2U) ? 128U : 64U;
        const unsigned largest = largest_batch(f), expected = c.group ? std::min(c.group, cap) : largest;
        bool status_ok = true;
        if (clean) {
            const auto& st = got[3];
            for (unsigned t = 0; t < c.rows; ++t) { std::uint32_t v; std::memcpy(&v, st.data() + t * f.status.stride, 4); status_ok = status_ok && v == 0U; }
            status_ok = status_ok && summary == 0U;
        }
        std::printf("%-40s form %2d: GSQ Q2_0 FP16 == production Q4_K/Q8_0 FP16 %s (summary %#x, tune %u, largest batch %u%s)\n", c.name, form,
            same ? "PASS" : "FAIL", summary, tune, largest, status_ok ? "" : ", UNEXPECTED FAILED ROWS");
        if (largest > cap || largest != expected) { std::printf("  largest batch %u, expected %u (cap %u)\n", largest, expected, cap); ok = false; }
        ok = ok && same && status_ok;
    }
    if (c.flavor == 6U) { (void)f.run(1); ok = extreme_coverage(f) && ok; }  // the A8 run writes the activation codes
    return ok;
}
void bench(const Case& c, int reps) {
    Fixture f(c);
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "event"); check(hipEventCreate(&stop), "event");
    // BENCH_FORMS: digits of the forms to time (default 013: dot4, iu8 WMMA, FP16 WMMA), or a comma-separated
    // list (forms 4..66: FP16 schedules) timed in BENCH_ROUNDS alternating rounds.
    const char* forms = std::getenv("BENCH_FORMS");
    std::vector<int> list;
    const std::string fs = forms ? forms : "013";
    if (fs.find(',') == std::string::npos) for (const char ch : fs) list.push_back(ch - '0');
    else for (std::size_t at = 0; at <= fs.size();) {
        const std::size_t end = std::min(fs.find(',', at), fs.size());
        list.push_back(std::atoi(fs.substr(at, end - at).c_str())); at = end + 1U;
    }
    const int rounds = std::getenv("BENCH_ROUNDS") ? std::atoi(std::getenv("BENCH_ROUNDS")) : 1;
    for (int round = 0; round < rounds; ++round) for (const int form : list) {
        f.enqueue(form);
        check(hipStreamSynchronize(f.stream), "warm");
        std::vector<float> ms;
        for (int i = 0; i < reps; ++i) {
            check(hipEventRecord(start, f.stream), "record");
            f.enqueue(form);
            check(hipEventRecord(stop, f.stream), "record");
            check(hipEventSynchronize(stop), "sync");
            float t = 0; check(hipEventElapsedTime(&t, start, stop), "elapsed"); ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        const double macs = static_cast<double>(c.rows) * Fixture::K * 3.0 * Fixture::H * Fixture::I;
        const auto kind = kr::prefill_kernel_form(f.request, c.down_type, form);
        const char* backend = kind == kr::PrefillKernelForm::f16_wmma ? "f16 wmma" : kind == kr::PrefillKernelForm::iu8_wmma ? "iu8 wmma" : "dot4";
        std::printf("%-30s form %d (%s): median %.3f ms (min %.3f) per %u rows, %.2f effective TOPS\n", c.name, form, backend,
            ms[ms.size() / 2], ms.front(), c.rows, 2.0 * macs / (ms[ms.size() / 2] * 1e9));
    }
    (void)hipEventDestroy(start); (void)hipEventDestroy(stop);
}
#include "hip_moe_route_bf16_fixture.inc"
#include "hip_moe_tiers_fixture.inc"
} // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "test";
        if(mode=="direct-b"||mode=="direct-b-timing") {
            direct_b_fixture(mode=="direct-b-timing",argc>2?std::stoi(argv[2]):15);return 0;
        }
        if (mode == "prepared-layout") {
            namespace qp = strix::qwen_moe_prepared;
            namespace fixture = qwen_moe_prepared_fixture;
            for (auto format : {qp::Format::q4_k, qp::Format::q5_k, qp::Format::q5_1}) {
                const auto spec = fixture::spec(format);
                const auto source = fixture::source();
                const auto layout = qp::describe(spec);
                const auto file = fixture::input(static_cast<std::size_t>(source.file_offset + layout.tensor_bytes));
                const qp::Read read = [&](std::uint64_t offset, std::span<std::byte> dst) {
                    if (offset > file.size() || dst.size() > file.size() - offset)
                        throw std::runtime_error("prepared fixture read outside source");
                    std::copy_n(file.begin() + static_cast<std::ptrdiff_t>(offset), dst.size(), dst.begin());
                };
                const auto tensor = qp::prepare(spec, source, file.size(), read, qp::allocate_hip);
                std::vector<std::byte> result(static_cast<std::size_t>(layout.tensor_bytes));
                check(hipMemcpy(result.data(), reinterpret_cast<const void*>(tensor.allocation->address()), result.size(), hipMemcpyDeviceToHost), "prepared layout download");
                fixture::tensor_oracle(spec, std::span(file).subspan(static_cast<std::size_t>(source.file_offset)), result);
                std::printf("prepared-layout format=%u version=%u bytes=%llu inverse/header/code coordinates PASS\n",
                    static_cast<unsigned>(format), layout.version, static_cast<unsigned long long>(layout.tensor_bytes));
            }
            return 0;
        }
        const bool tiers_mode=mode=="tiers"||mode=="tiersbench"||mode=="tiers-coexist";
        unsigned pool_rows = mode == "pool16k" || tiers_mode || mode == "tile64" || mode == "tile64time" ? 16384U : 8192U;
        if ((mode == "bench1" || mode == "bench2") && argc > 3)
            pool_rows = static_cast<unsigned>(std::stoul(argv[3]));
        if ((mode == "hdrbench" || mode == "hdrmc" || mode == "mcbench") && argc > 4)
            pool_rows = static_cast<unsigned>(std::stoul(argv[4]));
        if (tiers_mode) kr::set_prefill_tiers(1);
        if (mode == "tile64" || mode == "tile64time") {kr::set_prefill_tiers(0);kr::set_prefill_down_lds(0);}
        kr::reserve_prefill_pools(pool_rows);
        if (tiers_mode) return tiers_tests(argc, argv);
        if (mode == "route-bf16") { route_bf16_tests(); return 0; }
        if (mode == "route-i8") { route_i8_tests(); return 0; }
        if (mode == "schedule") {
            unsigned cases = 0;
            for (unsigned g : {12U, 13U}) for (unsigned d : {7U, 8U})
                for (unsigned flavor : {0U, 1U, 2U, 3U, 4U}) {
                    Fixture f({"FP16 schedule", 40U, 48U, g, d, flavor, true, true, flavor == 2U});
                    const auto base = f.run(3);
                    for (int form = 4; form <= 66; ++form) {
                        if (f.run(form) != base) throw std::runtime_error("FP16 schedule changed written bytes");
                        ++cases;
                    }
                }
            // Full CSR batches, multiple output tiles and a short final batch: 16 experts (85 assignments each, one
            // 64 + 21 or one 128-assignment batch), 12 experts (skewed: several above 128) and 10 experts (136
            // assignments each on average: the batch-major forms take 128-assignment batches).
            for (unsigned experts : {16U, 12U, 10U}) for (unsigned g : {12U, 13U}) for (unsigned d : {7U, 8U}) for (unsigned flavor : {0U, 4U}) {
                Fixture f({"FP16 schedule full batches", experts, 136U, g, d, flavor, true, true, false});
                const auto base = f.run(3);
                for (int form = 4; form <= 66; ++form) {
                    if (f.run(form) != base) throw std::runtime_error("FP16 full-batch schedule changed written bytes");
                    ++cases;
                }
            }
            std::printf("FP16 schedules: %u byte-identical cases PASS\n", cases);
            return 0;
        }
        if (mode == "core2" || mode == "bench2" || mode == "g3" || mode == "p1" || mode == "p1da16" || mode == "route" || mode == "da16" || mode == "small" || mode == "pool16k") {
            // qpf2k/moe. Real-layer specs: "<gguf>:<gate_at>:<up_at>:<down_at>:<gate type>:<down type>" (absolute offsets).
            struct Spec { std::string path; std::uint64_t g, u, d; unsigned gt, dt; };
            const auto parse = [](const std::string& x) {
                Spec r{}; std::size_t at = 0; std::vector<std::string> f;
                for (;;) { const std::size_t e = x.find(':', at); f.push_back(x.substr(at, e == std::string::npos ? std::string::npos : e - at)); if (e == std::string::npos) break; at = e + 1U; }
                if (f.size() != 6U) throw std::invalid_argument("layer spec needs 6 fields");
                r.path = f[0]; r.g = std::stoull(f[1]); r.u = std::stoull(f[2]); r.d = std::stoull(f[3]);
                r.gt = static_cast<unsigned>(std::stoul(f[4])); r.dt = static_cast<unsigned>(std::stoul(f[5]));
                return r;
            };
            // BENCH_MERGE=1 (bench2): every call includes the ordered merge (reference merge kernel, or the merge check after a
            // fused down), so a fused-merge core is timed against base down + merge.
            static const bool with_merge = std::getenv("BENCH_MERGE") && std::getenv("BENCH_MERGE")[0] == '1';
            const auto merge_only = [](Fixture& f, unsigned rows) {
                if (!with_merge) return;
                if (kr::prefill_fused_merge()) kr::enqueue_prefill_merge_check(f.view, rows, Fixture::H, f.c.experts, Fixture::K, f.stream);
                else if (kr::prefill_clear_skipped()) hipLaunchKernelGGL(reference_merge_kernel_z, dim3((Fixture::H + 255U) / 256U, rows), dim3(256), 0, f.stream, f.view, f.view.csr.summary, Fixture::H, f.c.experts, Fixture::K);
                else hipLaunchKernelGGL(reference_merge_kernel, dim3((Fixture::H + 255U) / 256U, rows), dim3(256), 0, f.stream, f.view, f.view.csr.summary, Fixture::H, f.c.experts, Fixture::K);
            };
            const auto call = [&](Fixture& f, unsigned rows) {
                f.enqueue(3, rows);
                merge_only(f, rows);
            };
            // pre_merge (p1 gate): summary[0] after the MoE call and before the merge (the merge itself may set bit 8).
            unsigned da16_missing = 0; // da16 mode: DA16 runs that fell back to the base path (a gate failure)
            // route mode (qpf2k/moe-mi): csr_bytes = the whole CSR allocation after the MoE call, before the merge.
            const auto run_digest = [&](Fixture& f, int core, unsigned rows, unsigned* pre_merge = nullptr, int route = 0,
                                        std::vector<unsigned char>* csr_bytes = nullptr) {
                kr::set_prefill_core2(core);
                kr::set_prefill_route((route & 3) | ((route & 32) ? 4 : 0) | ((route & 64) ? 8 : 0)); // bits 0-1, 5 (ROUTE 4), 6 (ROUTE 8)
                kr::set_prefill_da16((route >> 2) & 1); // route bit 2: STRIX_QPF2K_MOE_DA16
                { static const int small_rows[4]{0, 32, 64, 128}; kr::set_prefill_small(small_rows[(route >> 3) & 3]); } // bits 3-4: SMALL
                for (const auto* b : f.written()) b->poison(f.stream);
                f.csr.poison(f.stream);
                if (csr_bytes) kr::prefill_test_poison_a16(f.stream, rows); // route mode: the prologue's FP16 image must be rewritten
                if (route & 4) kr::prefill_test_poison_da16(f.stream, rows); // da16 mode: the DA16 image must be rewritten
                const auto poisoned_generations = kr::prefill_test_image_generations();
                f.enqueue(3, rows);
                if (kr::prefill_test_image_generations() != poisoned_generations)
                    throw std::runtime_error("poisoned prefill allocation changed through submission");
                if ((route & 4) && !kr::prefill_da16_ran()) ++da16_missing; // the DA16 path must actually run
                if ((route & 24) && !(route & 4) && !kr::prefill_small_ran()) ++da16_missing; // SMALL must actually run
                if (pre_merge || csr_bytes) {
                    check(hipStreamSynchronize(f.stream), "core2 run (pre-merge)");
                    if (pre_merge) check(hipMemcpy(pre_merge, f.view.csr.summary, sizeof(unsigned), hipMemcpyDeviceToHost), "pre-merge summary word");
                    if (csr_bytes) *csr_bytes = f.csr.read();
                }
                merge_only(f, rows);
                check(hipStreamSynchronize(f.stream), "core2 run");
                return digests(f);
            };
            if (mode == "pool16k") {
                Fixture f({"immutable T16K poisoned images", 512U, 16384U, 12U, 7U, 4U, false, true, false, 0U, 1.5});
                for (const unsigned rows : {16384U, 8192U, 16384U, 1320U}) {
                    unsigned pre_base = 0, pre_got = 0;
                    std::vector<unsigned char> csr_base, csr_got;
                    const auto base = run_digest(f, 0, rows, &pre_base, 0, &csr_base);
                    const auto got = run_digest(f, 0, rows, &pre_got, 7, &csr_got);
                    if (got != base || pre_got != pre_base || csr_got != csr_base || da16_missing)
                        throw std::runtime_error("T16K poisoned A16/DA16 bytes or executed path differ");
                }
                return 0;
            }
            if (mode == "bench2") {
                // bench2 <reps> <T> <gate type> <down type> <power (0 = uniform)> <flavor> [layer spec]; BENCH_CORES = list.
                if (argc < 8) throw std::invalid_argument("bench2 <reps> <T> <g> <d> <power> <flavor> [spec]");
                const int reps = std::atoi(argv[2]);
                const unsigned T = static_cast<unsigned>(std::atoi(argv[3]));
                const double power = std::atof(argv[6]);
                Case c{"bench2", 512U, T, static_cast<unsigned>(std::atoi(argv[4])), static_cast<unsigned>(std::atoi(argv[5])),
                    static_cast<unsigned>(std::atoi(argv[7])), false, power > 0.0, false, 0U, power};
                // BENCH_FULL128=1: every expert count a multiple of 128 (one tile-count copy everywhere; t1b settling test)
                if (std::getenv("BENCH_FULL128") && std::getenv("BENCH_FULL128")[0] == '1') { c.crafted = true; c.full128 = true; }
                Spec sp{};
                if (argc > 8) { sp = parse(argv[8]); c.gguf = sp.path.c_str(); c.gate_at = sp.g; c.up_at = sp.u; c.down_at = sp.d; c.gate_type = sp.gt; c.down_type = sp.dt; }
                Fixture f(c);
                const std::string cs = std::getenv("BENCH_CORES") ? std::getenv("BENCH_CORES") : "0,1,0";
                std::vector<int> cores;
                for (std::size_t at = 0; at <= cs.size();) { const std::size_t e = std::min(cs.find(',', at), cs.size()); cores.push_back(std::atoi(cs.substr(at, e - at).c_str())); at = e + 1U; }
                const auto ref = run_digest(f, 0, T);
                hipEvent_t start, stop;
                check(hipEventCreate(&start), "event"); check(hipEventCreate(&stop), "event");
                // BENCH_ROUTES (qpf2k/moe-mi): STRIX_QPF2K_MOE_ROUTE value per arm (same length as BENCH_CORES; default 0 each).
                std::vector<int> routes;
                if (const char* rv = std::getenv("BENCH_ROUTES")) {
                    const std::string rstr = rv;
                    for (std::size_t at = 0; at <= rstr.size();) { const std::size_t e = std::min(rstr.find(',', at), rstr.size()); routes.push_back(std::atoi(rstr.substr(at, e - at).c_str())); at = e + 1U; }
                    if (routes.size() != cores.size()) throw std::invalid_argument("BENCH_ROUTES needs one value per BENCH_CORES arm");
                }
                for (std::size_t arm = 0; arm < cores.size(); ++arm) {
                    const int core = cores[arm], route = routes.empty() ? 0 : routes[arm];
                    // BENCH_NOCHECK=1: attribution probes (G3 cores >= 2048) write wrong bits by design; time them anyway.
                    static const bool nocheck = std::getenv("BENCH_NOCHECK") && std::getenv("BENCH_NOCHECK")[0] == '1';
                    const bool same = run_digest(f, core, T, nullptr, route) == ref;
                    if (!routes.empty()) std::printf("bench2 arm %zu: route %d\n", arm, route);
                    if (!same && !nocheck) { std::printf("bench2 T%u core %d: bits DIFFER from core 0 -> abort\n", T, core); return 1; }
                    std::vector<float> ms;
                    for (int i = 0; i < reps; ++i) {
                        check(hipEventRecord(start, f.stream), "record"); call(f, T); check(hipEventRecord(stop, f.stream), "record");
                        check(hipEventSynchronize(stop), "sync");
                        float t = 0; check(hipEventElapsedTime(&t, start, stop), "elapsed"); ms.push_back(t);
                    }
                    std::sort(ms.begin(), ms.end());
                    const double flop = 2.0 * static_cast<double>(T) * Fixture::K * 3.0 * Fixture::H * Fixture::I;
                    std::printf("bench2 T%u g%u d%u power %.2f flavor %u %s core %d: median %.3f ms (min %.3f max %.3f) %.2f TOPS bits %s\n",
                        T, c.gate_type, c.down_type, power, c.flavor, c.full128 ? (c.gguf ? "real-full128" : "synthetic-full128") : c.gguf ? "real" : "synthetic", core, static_cast<double>(ms[ms.size() / 2]),
                        static_cast<double>(ms.front()), static_cast<double>(ms.back()), flop / (static_cast<double>(ms[ms.size() / 2]) * 1e9), same ? "IDENTICAL" : "DIFFER");
                    static const bool p1stats = std::getenv("STRIX_QPF2K_MOE_P1STATS") && std::getenv("STRIX_QPF2K_MOE_P1STATS")[0] == '1';
                    if (core == 257 && p1stats) {
                        const auto st = kr::prefill_p1_stats();
                        int khz = 0; check(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0), "wall clock rate");
                        const double ticks = static_cast<double>(st[4]) + 4294967296.0 * static_cast<double>(st[5]);
                        std::printf("bench2 T%u core 257 p1 stats (last call): down items ran %u skipped %u, waited %u (%.1f %%), polls %u, total wait %.3f ms (sum over items), mean %.1f us per waiting item\n",
                            T, st[2], st[3], st[0], st[2] ? 100.0 * st[0] / st[2] : 0.0, st[1], khz ? ticks / khz : -1.0, (khz && st[0]) ? 1000.0 * ticks / khz / st[0] : 0.0);
                    }
                }
                (void)hipEventDestroy(start); (void)hipEventDestroy(stop);
                kr::set_prefill_core2(-1);
                return 0;
            }
            unsigned cases = 0, failed = 0, scratch = 0, early = 0;
            // Returns the base path's pre-merge summary[0].
            const auto compare_cores = [&](Fixture& f, unsigned rows, const std::string& what) -> unsigned {
                // p1 mode (qpf2k P1, design 5): when BOTH paths fail closed in the MoE call with a projection failure only
                // (pre-merge summary[0] == 8 on both; route/prologue bits 1/2/4 are set before the launch, so both paths skip every
                // down and the raw dots are compared; a merge-only failure is not pre-merge) the raw dots are scratch the merge never
                // reads (the base down skipped every batch; P1 down items that started earlier wrote theirs): not compared. Every
                // status and all 32 summary words are always compared.
                unsigned pre_base = 0;
                if (mode == "route" || mode == "da16" || mode == "small") {
                    // qpf2k/moe-mi STRIX_QPF2K_MOE_ROUTE bits 1/2/3 vs 0 on the base core: the whole CSR allocation (every array,
                    // the 32 summary words, words left unwritten keep the poison), every written view and summary after the merge.
                    std::vector<unsigned char> csr_base;
                    const auto rbase = run_digest(f, 0, rows, &pre_base, 0, &csr_base);
                    // da16 mode: value 4 (DA16 alone) and 7 (DA16 + route bits 1 and 2).
                    for (const int route : mode == "da16" ? std::vector<int>{4, 7} : mode == "small" ? std::vector<int>{8, 16, 24, 27} : std::vector<int>{1, 2, 3, 32, 34, 64, 98}) {
                        unsigned pre_got = 0;
                        std::vector<unsigned char> csr_got;
                        const auto got = run_digest(f, 0, rows, &pre_got, route, &csr_got);
                        ++cases;
                        bool differ = pre_got != pre_base || csr_got != csr_base;
                        if (differ) std::printf("  DIFFER %s rows %u route %d: pre-merge summary[0] %u vs %u, csr %s\n", what.c_str(), rows, route,
                            pre_got, pre_base, csr_got == csr_base ? "same" : "differ");
                        for (std::size_t k = 0; k < got.size(); ++k) if (got[k] != rbase[k]) {
                            differ = true;
                            std::printf("  DIFFER %s rows %u route %d: %s\n", what.c_str(), rows, route, k < 10U ? names[k] : "summary");
                        }
                        failed += differ ? 1U : 0U;
                    }
                    kr::set_prefill_route(-1);
                    kr::set_prefill_da16(-1);
                    kr::set_prefill_small(-1);
                    return pre_base;
                }
                const auto base = run_digest(f, 0, rows, &pre_base);
                // g3 mode: the qpf2k G3 core (BM 256 both, BM 128 both, G3 gate/up + base down, base gate/up + G3 down; + 32: depth 2).
                static const std::vector<int> core2_list{1, 17, 65, 129, 145, 321, 449}, g3_list{1025, 1027, 1029, 1041, 1057, 1059, 1061, 1073}, p1_list{257};
                for (const int core : mode == "g3" ? g3_list : (mode == "p1" || mode == "p1da16") ? p1_list : core2_list) {
                    unsigned pre_got = 0;
                    const auto got = run_digest(f, core, rows, &pre_got, mode == "p1da16" ? 4 : 0); // p1da16: P1 + DA16 (route bit 2)
                    ++cases;
                    const bool skip_raw = (mode == "p1" || mode == "p1da16") && pre_base == 8U && pre_got == 8U;
                    scratch += skip_raw ? 1U : 0U;
                    if (skip_raw && got[9] != base[9]) ++early; // P1 down work that ran before the failure (witness)
                    bool differ = pre_got != pre_base;
                    if (differ) std::printf("  DIFFER %s rows %u core %d: pre-merge summary[0] %u vs %u\n", what.c_str(), rows, core, pre_got, pre_base);
                    for (std::size_t k = 0; k < got.size(); ++k) if (got[k] != base[k] && !(skip_raw && k == 9U)) {
                        differ = true;
                        std::printf("  DIFFER %s rows %u core %d: %s\n", what.c_str(), rows, core, k < 10U ? names[k] : "summary");
                    }
                    failed += differ ? 1U : 0U;
                }
                return pre_base;
            };
            std::vector<std::string> keep2; keep2.reserve(256);
            for (unsigned g : {12U, 13U}) for (unsigned d : {7U, 8U}) {
                const std::string fmt = " g" + std::to_string(g) + " d" + std::to_string(d);
                for (unsigned flavor : {0U, 1U, 2U, 3U, 4U}) {
                    keep2.push_back("E40 T48 dup skew" + fmt + " flavor " + std::to_string(flavor));
                    Fixture f({keep2.back().c_str(), 40U, 48U, g, d, flavor, true, true, flavor == 2U});
                    for (unsigned rows : {48U, 24U, 16U}) compare_cores(f, rows, keep2.back());
                }
                for (unsigned experts : {16U, 12U, 10U}) for (unsigned flavor : {0U, 4U}) {
                    keep2.push_back("E" + std::to_string(experts) + " T136 dup skew" + fmt + " flavor " + std::to_string(flavor));
                    Fixture f({keep2.back().c_str(), experts, 136U, g, d, flavor, true, true, false});
                    compare_cores(f, 136U, keep2.back());
                }
                keep2.push_back("E512 T2048 uniform" + fmt + " flavor 1");
                { Fixture f({keep2.back().c_str(), 512U, 2048U, g, d, 1U, false, false, false}); compare_cores(f, 2048U, keep2.back()); }
                keep2.push_back("E512 T8192 skew^1.5" + fmt + " flavor 4");
                { Fixture f({keep2.back().c_str(), 512U, 8192U, g, d, 4U, false, true, false, 0U, 1.5});
                  for (unsigned rows : {8192U, 4920U, 4912U, 4096U, 1320U}) compare_cores(f, rows, keep2.back()); }
            }
            // Exact expert counts at every tile boundary (T 1024, 512 experts; T 520 for expert counts up to the row count),
            // all format pairs, plus an invalid route (fails closed in the CSR on both paths).
            for (unsigned g : {12U, 13U}) for (unsigned d : {7U, 8U}) for (unsigned T : {1024U, 520U}) for (bool bad : {false, true}) {
                keep2.push_back("crafted T" + std::to_string(T) + " g" + std::to_string(g) + " d" + std::to_string(d) + (bad ? " bad route" : ""));
                Case c{keep2.back().c_str(), 512U, T, g, d, 4U, false, false, false};
                c.crafted = true; c.bad_route = bad;
                Fixture f(c);
                compare_cores(f, T, keep2.back());
            }
            std::printf("core2 synthetic: %u comparisons, %u differ\n", cases, failed);
            for (int a = 2; a < argc; ++a) {
                const Spec sp = parse(argv[a]);
                for (const double power : {0.0, 1.5, 3.0}) {
                    keep2.push_back("REAL @" + std::to_string(sp.g) + " g" + std::to_string(sp.gt) + " d" + std::to_string(sp.dt) +
                        (power > 0.0 ? " skew^" + std::to_string(power) : std::string(" uniform")));
                    Case c{keep2.back().c_str(), 512U, 8192U, sp.gt, sp.dt, 0U, false, power > 0.0, false, 0U, power, sp.path.c_str(), sp.g, sp.u, sp.d};
                    Fixture f(c);
                    for (unsigned rows : {8192U, 4920U, 4096U, 1320U, 136U, 24U, 16U}) compare_cores(f, rows, keep2.back());
                    std::printf("%s: done (%u comparisons so far, %u differ)\n", keep2.back().c_str(), cases, failed);
                }
                if (mode == "p1" || mode == "p1da16" || mode == "route" || mode == "da16" || mode == "small") {
                    // One non-finite input row (row 4096 = T/2): its batches fail while other batches' down items may already have
                    // run; then 4096 / 1320 rows (without that row) succeed on the same allocations and counter pool.
                    keep2.push_back("REAL @" + std::to_string(sp.g) + " skew^1.5 one NaN row");
                    Case c{keep2.back().c_str(), 512U, 8192U, sp.gt, sp.dt, 0U, false, true, true, 0U, 1.5, sp.path.c_str(), sp.g, sp.u, sp.d};
                    Fixture f(c);
                    // Expected pre-merge summary[0]: 8 (projection failure) with row 4096 included, 0 without, 8 again at the end.
                    const unsigned expect[5]{8U, 8U, 0U, 0U, 8U};
                    unsigned at = 0;
                    for (unsigned rows : {8192U, 4920U, 4096U, 1320U, 8192U}) {
                        const unsigned s = compare_cores(f, rows, keep2.back());
                        if (s != expect[at]) { ++failed; std::printf("  UNEXPECTED %s rows %u: pre-merge summary[0] %u, expected %u\n", keep2.back().c_str(), rows, s, expect[at]); }
                        ++at;
                    }
                    std::printf("%s: done (%u comparisons so far, %u differ)\n", keep2.back().c_str(), cases, failed);
                }
            }
            kr::set_prefill_core2(-1);
            if (mode == "p1" || mode == "p1da16") std::printf("p1: %u comparisons with a projection-only failure on both paths (raw-dot scratch not compared), %u of them with P1 down work before the failure\n", scratch, early);
            if (mode == "da16" || mode == "small" || mode == "p1da16") { std::printf("da16: %u runs without the DA16 path\n", da16_missing); failed += da16_missing; }
            std::printf("%s: %u comparisons, %u differ -> %s\n", mode.c_str(), cases, failed, failed ? "FAIL" : "PASS");
            return failed ? 1 : 0;
        }
        if (mode == "mc") {
            // B uses unchanged producers/tail; C requests the production certificate chain and uses
            // the shared engine merge/finish only when the per-call producer witness is true.
            // Every reproducible failure class is exact, including patterned old output. Racy bases
            // are separate, never passes; the suite cannot qualify with missing reproducible coverage.
            unsigned cases = 0, failed = 0, racy = 0, rejected = 0, certified = 0, admitted = 0, unproven = 0, cert_pre = 0, cert_merge = 0, unproven_success = 0, n_success = 0, n_pre = 0;
            // kind 0: slot 0 weight 1.0, dot = target bits, slots 1..9 dot +0 (sum = target, -0 becomes +0); kind 1: route of
            // slot 4 set out of range after prefill; kind 2: weight 1+ulp before producers; kinds 3/4:
            // all ten large dots with aligned or cancelling weights (including zero/subnormal).
            struct Inject { unsigned row = ~0U, element = 0; std::uint32_t target = 0; unsigned kind = 0; };
            struct Run { std::vector<unsigned long long> d; std::vector<std::vector<unsigned char>> bytes; unsigned pre = 0, post = 0, unsafe = 0; bool certificate = false; };
            const char* const tail_names[] = {"status", "csr", "out16", "accumulation", "record"};
            const auto run = [&](Fixture& f, Buffer& out16, Buffer& record, unsigned rows, int arm, const Inject& inj, std::uint64_t record_init) {
                Run r;
                const unsigned hidden=inj.kind==5U?Fixture::H-1U:Fixture::H;
                kr::set_prefill_core2(0); kr::set_prefill_route(14); kr::set_prefill_da16(1); kr::set_prefill_small(0);
                for (const auto* b : f.written()) b->poison(f.stream);
                f.csr.poison(f.stream);
                std::vector<unsigned char> old_output(out16.bytes);
                for (std::size_t i=0;i<old_output.size();++i) old_output[i]=static_cast<unsigned char>((i*37U+(i>>8U)*11U+19U)&255U);
                check(hipMemcpyAsync(out16.base, old_output.data(), out16.bytes, hipMemcpyHostToDevice, f.stream), "patterned old output");
                f.view.unsafe=arm?static_cast<unsigned*>(f.admission.base):nullptr;
                check(hipMemcpyAsync(record.base, &record_init, 8U, hipMemcpyHostToDevice, f.stream), "record init");
                std::array<float,Fixture::K> saved_weights{}, injected_weights{};
                auto* weight0=static_cast<unsigned char*>(f.weights.base)+(inj.row==~0U?0U:inj.row)*f.weights.stride;
                if (inj.row!=~0U && inj.kind!=1U) {
                    check(hipMemcpy(saved_weights.data(),weight0,4U*Fixture::K,hipMemcpyDeviceToHost),"weight save");
                    injected_weights=saved_weights;
                    auto& weights=injected_weights;
                    weights[0]=inj.kind==2U?std::nextafter(1.0F,2.0F):1.0F;
                    if (inj.kind>=3U) for (unsigned slot=0;slot<Fixture::K;++slot) {
                        const std::uint32_t bits=inj.kind==3U?0x3f800000U:
                            slot==8U?0U:slot==9U?1U:(slot&1U)?0xbf800000U:0x3f800000U;
                        std::memcpy(&weights[slot],&bits,4U);
                    }
                    check(hipMemcpyAsync(weight0,weights.data(),4U*Fixture::K,hipMemcpyHostToDevice,f.stream),"weight before producers");
                }
                f.enqueue(3, rows);
                r.certificate=kr::prefill_certificate_produced();
                if (!kr::prefill_clear_skipped()) throw std::runtime_error("mc: ROUTE 14 did not skip the clear");
                check(hipStreamSynchronize(f.stream), "mc prefill");
                if (inj.row != ~0U && inj.kind != 1U) {
                    auto* row = static_cast<unsigned char*>(f.rawdots.base) + inj.row * f.rawdots.stride;
                    for (unsigned s = 0; s < Fixture::K; ++s) {
                        const std::uint32_t v = s == 0U || inj.kind>=3U ? inj.target : 0U;
                        check(hipMemcpy(row + 4U * (static_cast<std::uint64_t>(s) * Fixture::H + inj.element), &v, 4U, hipMemcpyHostToDevice), "inject");
                    }
                }
                std::uint32_t saved_route = 0;
                auto* route4 = static_cast<unsigned char*>(f.routes.base) + (inj.row == ~0U ? 0U : inj.row) * f.routes.stride + 16U;
                if (inj.row != ~0U && inj.kind == 1U) {
                    check(hipMemcpy(&saved_route, route4, 4U, hipMemcpyDeviceToHost), "route save");
                    const std::uint32_t bad = f.c.experts;
                    check(hipMemcpy(route4, &bad, 4U, hipMemcpyHostToDevice), "route inject");
                }
                if (r.certificate && inj.kind==2U) {
                    unsigned upstream=0U;
                    check(hipMemcpy(&upstream,f.view.unsafe,4U,hipMemcpyDeviceToHost),"weight producer certificate");
                    if (!upstream) throw std::runtime_error("mc: CSR producer missed weight above one");
                }
                if (r.certificate && inj.row != ~0U) {
                    // Mutation invalidated the earlier proof. Re-run it without clearing earlier producer failures.
                    hipLaunchKernelGGL(fixture_certificate, dim3(32), dim3(256), 0, f.stream, f.view, rows, Fixture::H, Fixture::K);
                    check(hipStreamSynchronize(f.stream), "recomputed fixture certificate");
                }
                if (r.certificate) check(hipMemcpy(&r.unsafe, f.view.unsafe, 4U, hipMemcpyDeviceToHost), "fixture admission");
                check(hipMemcpy(&r.pre, f.view.csr.summary, 4U, hipMemcpyDeviceToHost), "pre-merge summary");
                const dim3 grid((Fixture::H + 255U) / 256U, rows);
                const kr::StridedRows<std::uint16_t> o{static_cast<std::uint16_t*>(out16.base), out16.stride};
                auto* rec = static_cast<unsigned*>(record.base);
                if (arm == 0 || !r.certificate) {
                    hipLaunchKernelGGL(reference_merge_kernel_z, grid, dim3(256), 0, f.stream, f.view, f.view.csr.summary, hidden, f.c.experts, Fixture::K);
                    hipLaunchKernelGGL(reference_commit_kernel, grid, dim3(256), 0, f.stream, f.view, f.view.csr.summary, rows, hidden, o, rec);
                } else {
                    if (arm == 4) candidate_merge_commit_vec4<true>(f.stream, f.view, f.view.csr.summary, hidden, f.c.experts, Fixture::K, o, rows);
                    else if (arm == 3) candidate_merge_commit_vec4<false>(f.stream, f.view, f.view.csr.summary, hidden, f.c.experts, Fixture::K, o, rows);
                    else if (arm == 2) hipLaunchKernelGGL(candidate_merge_commit_kernel<true>, grid, dim3(256), 0, f.stream, f.view, f.view.csr.summary, hidden, f.c.experts, Fixture::K, o);
                    else hipLaunchKernelGGL(candidate_merge_commit_kernel<false>, grid, dim3(256), 0, f.stream, f.view, f.view.csr.summary, hidden, f.c.experts, Fixture::K, o);
                    hipLaunchKernelGGL(candidate_finish_kernel, dim3(32), dim3(256), 0, f.stream, f.view, f.view.csr.summary, rows, hidden, o, rec);
                }
                check(hipGetLastError(), "mc merge/commit");
                check(hipStreamSynchronize(f.stream), "mc run");
                if (inj.row != ~0U && inj.kind == 1U) check(hipMemcpy(route4, &saved_route, 4U, hipMemcpyHostToDevice), "route restore");
                check(hipMemcpy(&r.post, f.view.csr.summary, 4U, hipMemcpyDeviceToHost), "post-merge summary");
                if (inj.row!=~0U && inj.kind!=1U) check(hipMemcpy(weight0,saved_weights.data(),4U*Fixture::K,hipMemcpyHostToDevice),"weight restore");
                r.d = digests(f);
                r.bytes = {f.status.read(), f.csr.read(), out16.read(), f.clear.read(), record.read()};
                // Every row has patterned padding; boundary helper also leaves its final logical lane untouched.
                for (unsigned row=0;row<f.c.rows;++row) for (std::uint64_t i=2ULL*hidden;i<out16.stride;++i)
                    if (r.bytes[2][row*out16.stride+i]!=old_output[row*out16.stride+i])
                        throw std::runtime_error("mc: output canary changed");
                if (r.post && r.bytes[2]!=old_output) throw std::runtime_error("mc: rejected call changed patterned old output");
                return r;
            };
            const auto diff = [](const std::vector<unsigned char>& a, const std::vector<unsigned char>& b, std::size_t& first) {
                std::size_t n = 0; first = ~std::size_t{0};
                for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) if (a[i] != b[i]) { if (!n) first = i; ++n; }
                return n + (a.size() != b.size() ? 1U : 0U);
            };
            static std::vector<std::string> keep4; keep4.reserve(256);
            const auto cmp = [&](Fixture& f, unsigned rows, const std::string& what, const Inject& inj = {}) {
                Buffer out16(f.c.rows, 2U * Fixture::H + 32U), record(1, 8U);
                for (const std::uint64_t record_init : {0ULL, 0x40ULL << 32U | 2ULL}) {
                    const auto base = run(f, out16, record, rows, 0, inj, record_init);
                    const auto again = run(f, out16, record, rows, 0, inj, record_init);
                    const bool merge_rejected = base.pre == 0U && base.post != 0U;
                    const char* cls = base.pre ? "PRE" : merge_rejected ? "MERGE" : "SUCCESS";
                    if (again.d != base.d || again.bytes != base.bytes || again.post != base.post) {
                        ++racy; std::printf("  RACY base %s %s rows %u\n", cls, what.c_str(), rows);
                        continue; // excluded from exact comparisons, and prevents a qualification PASS below
                    }
                    rejected += merge_rejected ? 1U : 0U; n_success += base.post == 0U ? 1U : 0U; n_pre += base.pre != 0U ? 1U : 0U;
                    if (base.post == 0U) {  // the output must have been stored: not the poison image
                        // At least one active output word must replace the patterned old image.
                        bool stored=false;
                        for (unsigned row=0;row<rows;++row) for (unsigned i=0;i<2U*Fixture::H;++i) {
                            const auto pos=row*out16.stride+i;
                            stored|=base.bytes[2][pos]!=static_cast<unsigned char>((pos*37U+(pos>>8U)*11U+19U)&255U);
                        }
                        if (!stored) throw std::runtime_error("mc: reference output not stored");
                    }
                    // Arms 3/4 (vec4 merge) must also equal arms 1/2 byte for byte, accumulation included.
                    std::vector<std::vector<unsigned char>> scalar_bytes[2];
                    for (const int arm : {1, 2, 3, 4}) {
                        const auto got = run(f, out16, record, rows, arm, inj, record_init);
                        if (arm <= 2) scalar_bytes[arm - 1] = got.bytes;
                        else if (got.bytes != scalar_bytes[arm - 3]) {
                            std::printf("  DIFFER %s %s rows %u arm %d: vec4 bytes != scalar arm %d\n", cls, what.c_str(), rows, arm, arm - 2);
                            ++failed;
                        }
                        ++cases;
                        bool bad = got.pre != base.pre || got.post != base.post;
                        if (got.certificate) {
                            ++certified; if (got.unsafe) ++unproven; else if (!got.post) ++admitted;
                            cert_pre+=got.pre!=0U; cert_merge+=got.pre==0U&&got.post!=0U;
                            unproven_success+=got.unsafe!=0U&&got.post==0U;
                        }
                        if (what=="mc reset sequence safe" && (!got.certificate || got.unsafe))
                            throw std::runtime_error("mc: unsafe word did not reset on the next eligible call");
                        if (inj.row!=~0U && inj.kind==2U && got.certificate && !got.unsafe)
                            throw std::runtime_error("mc: weight above one was admitted");
                        for (std::size_t k = 0; k < 10U; ++k) if (k != 3U && k != 5U && got.d[k] != base.d[k]) {  // upstream views
                            bad = true; std::printf("  DIFFER %s %s rows %u arm %d: upstream %s\n", cls, what.c_str(), rows, arm, names[k]);
                        }
                        for (std::size_t k = 0; k < got.bytes.size(); ++k) {
                            if (k == 3U && (arm == 1 || arm == 3) && got.certificate && !got.unsafe && !got.pre) continue;  // mode 1 does not write the accumulation
                            std::size_t first = 0;
                            const auto n = diff(got.bytes[k], base.bytes[k], first);
                            if (!n) continue;
                            std::printf("  %s %s %s rows %u arm %d record0 %llx: %s %zu bytes differ, first at %zu\n", "DIFFER",
                                cls, what.c_str(), rows, arm, static_cast<unsigned long long>(record_init), tail_names[k], n, first);
                            bad = true;
                        }
                        failed += bad ? 1U : 0U;
                    }
                }
            };
            for (unsigned g : {12U, 13U}) for (unsigned d : {7U, 8U}) for (unsigned flavor : {0U, 1U, 2U, 3U, 4U}) {
                keep4.push_back("mc E40 T48 g" + std::to_string(g) + " d" + std::to_string(d) + " flavor " + std::to_string(flavor));
                Fixture f({keep4.back().c_str(), 40U, 48U, g, d, flavor, true, true, flavor == 2U});
                cmp(f, 48U, keep4.back());
                if (flavor == 0U) cmp(f, 16U, keep4.back());
            }
            {
                keep4.push_back("mc E40 T48 group 16");
                Case c{keep4.back().c_str(), 40U, 48U, 12U, 7U, 4U, false, false, false, 16U};
                Fixture f(c); cmp(f, 48U, keep4.back());
            }
            for (unsigned g : {12U, 13U}) for (bool bad : {false, true}) {
                keep4.push_back(std::string("mc crafted T520 g") + std::to_string(g) + (bad ? " bad route" : ""));
                Case c{keep4.back().c_str(), 512U, 520U, g, 7U, 4U, false, false, false}; c.crafted = true; c.bad_route = bad;
                Fixture f(c); cmp(f, 520U, keep4.back());
            }
            {
                keep4.push_back("mc crafted full128 T1024");
                Case c{keep4.back().c_str(), 64U, 1024U, 12U, 8U, 4U, false, false, false}; c.crafted = true; c.full128 = true;
                Fixture f(c); cmp(f, 1024U, keep4.back());
            }
            for (unsigned g : {12U, 13U}) for (unsigned d : {7U, 8U}) {
                keep4.push_back("mc E512 T2048 g" + std::to_string(g) + " d" + std::to_string(d));
                Fixture f({keep4.back().c_str(), 512U, 2048U, g, d, 4U, false, true, false, 0U, 1.5});
                cmp(f, 2048U, keep4.back());
                Case c{keep4.back().c_str(), 512U, 1024U, g, d, 4U, false, false, false}; c.crafted = true;
                Fixture fc(c); cmp(fc, 1024U, keep4.back() + " crafted");
            }
            {
                // Injected merge inputs: admitted edges are byte-identical; rejected sums give the base statuses/summary/record.
                Fixture f({"mc inject E16 T48", 16U, 48U, 12U, 7U, 4U, true, true, false});
                std::vector<unsigned char> bounded(f.weights.bytes,0U);
                for (unsigned r=0;r<f.c.rows;++r) for(unsigned slot=0;slot<Fixture::K;++slot) {
                    const float w=0.5F;
                    std::memcpy(bounded.data()+r*f.weights.stride+4U*slot,&w,4U);
                }
                check(hipMemcpy(f.weights.base,bounded.data(),f.weights.bytes,hipMemcpyHostToDevice),"bounded admission weights");
                const std::uint32_t targets[] = {0x7b7fffffU, 0xfb7fffffU, 0x7b800000U, 0xfb800000U, 0x7f7f7fffU, 0xff7f7fffU, 0x7f7f8000U, 0xff7f8000U, 0x80000000U, 0x00000001U, 0x80000001U,
                    0x00800000U, 0x7f800000U, 0xff800000U, 0x7fc12345U, 0xffa00001U, 0x7f7effffU};
                for (const auto t : targets) for (const unsigned row : {0U, 29U, 47U}) for (const unsigned el : {0U, 255U, 2559U}) {
                    char b[96]; std::snprintf(b, sizeof b, "mc inject %08x row %u el %u", t, row, el);
                    keep4.push_back(b);
                    cmp(f, 48U, keep4.back(), Inject{row, el, t});
                }
                cmp(f,48U,"mc helper boundary hidden2559",Inject{~0U,0U,0U,5U});
                cmp(f,48U,"mc admitted all ten aligned",Inject{0U,0U,0x7b7fffffU,3U});
                cmp(f,48U,"mc admitted cancellation/zero/subnormal weights",Inject{0U,2559U,0x7b7fffffU,4U});
                cmp(f,48U,"mc unproven large cancellation",Inject{0U,255U,0x7b800000U,4U});
                // Repeated calls share one fixture/word: unproven -> safe -> unproven, including changed row counts.
                cmp(f, 48U, "mc unsafe sequence first", Inject{0U,0U,0x7b800000U});
                cmp(f, 40U, "mc reset sequence safe");
                cmp(f, 48U, "mc unsafe sequence last", Inject{0U,0U,0x3f800000U,2U});
                // Separate streams/fixtures own separate admission words. Leave A unsafe while B runs safely.
                Fixture other({"mc independent owner",16U,48U,12U,7U,4U,true,true,false});
                check(hipMemcpy(other.weights.base,bounded.data(),other.weights.bytes,hipMemcpyHostToDevice),"second owner weights");
                Buffer out_a(48U,2U*Fixture::H+32U),out_b(48U,2U*Fixture::H+32U),rec_a(1,8U),rec_b(1,8U);
                const auto unsafe_a=run(f,out_a,rec_a,48U,2,Inject{0U,0U,0x3f800000U,2U},0ULL);
                const auto safe_a=run(f,out_a,rec_a,40U,2,Inject{},0ULL);
                const auto unsafe_again=run(f,out_a,rec_a,48U,2,Inject{0U,0U,0x3f800000U,2U},0ULL);
                const auto safe_b=run(other,out_b,rec_b,48U,2,Inject{},0ULL);
                unsigned retained=0U; check(hipMemcpy(&retained,f.admission.base,4U,hipMemcpyDeviceToHost),"owner isolation");
                if (!unsafe_a.unsafe||safe_a.unsafe||!unsafe_again.unsafe||safe_b.unsafe||!retained)
                    throw std::runtime_error("mc: admission reset/owner isolation failed");
                for (const unsigned row : {0U, 29U, 47U}) {  // the merge's own invalid-route branch (a SUCCESS call: zeros committed)
                    keep4.push_back("mc inject route row " + std::to_string(row));
                    cmp(f, 48U, keep4.back(), Inject{row, 0U, 0U, 1U});
                }
            }
            for (unsigned g : {12U,13U}) for (unsigned d : {7U,8U}) {
                Fixture f({"mc dense certificate formats",16U,48U,g,d,4U,true,true,false});
                cmp(f,48U,"mc dense certificate formats");
            }
            for (unsigned failure : {0U,1U,2U}) {
                Case c{"mc certificate pre-rejection",16U,48U,12U,7U,failure==2U?3U:4U,true,true,failure==1U};
                Fixture f(c);
                if (failure==0U) {
                    const unsigned bad=16U;  // == experts: out of range
                    check(hipMemcpy(f.routes.base,&bad,4U,hipMemcpyHostToDevice),"route before producers");
                }
                cmp(f,48U,"mc certified route/input/projection rejection");
            }
            for (int a = 2; a < argc; ++a) {
                const std::string x = argv[a];
                std::vector<std::string> fl; std::size_t at = 0;
                for (;;) { const std::size_t e = x.find(':', at); fl.push_back(x.substr(at, e == std::string::npos ? std::string::npos : e - at)); if (e == std::string::npos) break; at = e + 1U; }
                keep4.push_back("mc REAL " + fl[1]);
                Case c{keep4.back().c_str(), 512U, 8192U, static_cast<unsigned>(std::stoul(fl[4])), static_cast<unsigned>(std::stoul(fl[5])), 0U, false, true, false, 0U, 1.5,
                    nullptr, std::stoull(fl[1]), std::stoull(fl[2]), std::stoull(fl[3])};
                c.gguf = fl[0].c_str();
                Fixture f(c);
                for (unsigned rows : {8192U, 1320U}) cmp(f, rows, keep4.back());
            }
            kr::set_prefill_core2(-1); kr::set_prefill_route(-1); kr::set_prefill_da16(-1); kr::set_prefill_small(-1);
            std::printf("mc: %u exact comparisons (%u SUCCESS, %u PRE, %u MERGE), %u differ, %u RACY excluded; "
                "%u certificate calls (%u admitted, %u unproven) -> %s\n", cases,n_success,n_pre,rejected,failed,racy,certified,admitted,unproven,
                failed||racy||!n_success||!n_pre||!rejected||!admitted||!unproven_success||!cert_pre||!cert_merge ? "FAIL/UNQUALIFIED" : "PASS");
            return failed||racy||!n_success||!n_pre||!rejected||!admitted||!unproven_success||!cert_pre||!cert_merge ? 1 : 0;
        }
        if (mode == "xor64" || mode == "xor64bench" || mode == "da16issue" || mode == "da16issuebench") {
            // da16issue / da16issuebench: the same comparisons and timing with XOR rows on and the uniform DA16 activation
            // issue (STRIX_QPF2K_MOE_DA16_ISSUE) off vs on; the DA16 image is poisoned before every compared run, and the
            // bench also reports the down phase alone (DA16 phase events).
            const bool issue_mode = mode == "da16issue" || mode == "da16issuebench";
            const auto item = [issue_mode](int on) { if (issue_mode) { kr::set_prefill_xor64(1); kr::set_prefill_da16_issue(on); } else kr::set_prefill_xor64(on); };
            // MoE XOR rows (STRIX_QPF2K_MOE_XOR64): default chain (form 3, ROUTE 14, DA16 1, HDR 1) with the item off vs on.
            // xor64 [specs...]: every written view and the CSR words byte-identical, and the on run must take tune 88 exactly
            // when the off run takes tune 80 with >= 3072 rows (synthetic Q4_K/Q5_K x Q5_1/Q8_0 at T 2048..4864 with skew,
            // crafted and full-128 routes, then real layers at T 8192/4864/4096/3072/2048). xor64bench <reps> <spec> <rows>:
            // complete routed call, ABBA hipEvent medians.
            const auto parse = [](const std::string& x) {
                std::vector<std::string> fl; std::size_t at = 0;
                for (;;) { const std::size_t e = x.find(':', at); fl.push_back(x.substr(at, e == std::string::npos ? std::string::npos : e - at)); if (e == std::string::npos) break; at = e + 1U; }
                if (fl.size() != 6U) throw std::invalid_argument("layer spec needs 6 fields");
                return fl;
            };
            kr::set_prefill_core2(0); kr::set_prefill_route(14); kr::set_prefill_da16(1); kr::set_prefill_hdr(1);
            const auto reset = [] { kr::set_prefill_da16_issue(-1); kr::set_prefill_xor64(-1); kr::set_prefill_hdr(-1); kr::set_prefill_core2(-1); kr::set_prefill_route(-1); kr::set_prefill_da16(-1); };
            if (mode == "xor64bench" || mode == "da16issuebench") {
                if (argc < 5) throw std::invalid_argument("xor64bench <reps> <layer spec> <rows>");
                const int reps = std::atoi(argv[2]);
                if (reps <= 0) throw std::invalid_argument("xor64bench: reps must be positive");
                const auto fl = parse(argv[3]);
                const unsigned rows = static_cast<unsigned>(std::atoi(argv[4]));
                Case c{"xor64bench", 512U, rows, static_cast<unsigned>(std::stoul(fl[4])), static_cast<unsigned>(std::stoul(fl[5])), 0U, false, true, false, 0U, 1.5,
                    nullptr, std::stoull(fl[1]), std::stoull(fl[2]), std::stoull(fl[3])};
                c.gguf = fl[0].c_str();
                if (issue_mode && std::getenv("BENCH_FULL128") && std::getenv("BENCH_FULL128")[0] == '1') { c.crafted = true; c.full128 = true; }
                Fixture f(c);
                hipEvent_t e0, e1;
                check(hipEventCreate(&e0), "event"); check(hipEventCreate(&e1), "event");
                unsigned tune_on = 0;
                std::array<hipEvent_t, 3> phases{};
                for (auto& e : phases) check(hipEventCreate(&e), "event");
                if (issue_mode) kr::prefill_test_da16_events(&phases);
                std::vector<double> down[2], proj[2];
                const auto timed = [&](int on) {
                    item(on);
                    check(hipEventRecord(e0, f.stream), "record");
                    f.enqueue(3, rows);
                    check(hipEventRecord(e1, f.stream), "record");
                    check(hipEventSynchronize(e1), "sync");
                    if (!kr::prefill_da16_ran()) throw std::runtime_error("xor64bench: the DA16 path did not run");
                    if (on) tune_on = kr::prefill_last_tune();
                    float ms = 0.0F; check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
                    if (issue_mode) { float d = 0.0F; check(hipEventElapsedTime(&d, phases[1], phases[2]), "down elapsed"); down[on ? 1 : 0].push_back(d); check(hipEventElapsedTime(&d, phases[0], phases[1]), "gate/up elapsed"); proj[on ? 1 : 0].push_back(d); }
                    return static_cast<double>(ms);
                };
                timed(0); timed(1);
                down[0].clear(); down[1].clear(); proj[0].clear(); proj[1].clear();
                std::vector<double> a, b;
                for (int r = 0; r < reps; ++r) { a.push_back(timed(0)); b.push_back(timed(1)); b.push_back(timed(1)); a.push_back(timed(0)); }
                const auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
                std::printf("xor64bench rows %u reps %d: off %.4f ms, on %.4f ms (tune %u), delta %.4f ms (%+.2f %%)\n",
                    rows, reps, med(a), med(b), tune_on, med(b) - med(a), 100.0 * (med(b) - med(a)) / med(a));
                if (issue_mode) {
                    kr::prefill_test_da16_events(nullptr);
                    std::printf("da16issuebench rows %u gate/up phase: masked %.4f ms, uniform %.4f ms; down min: masked %.4f ms, uniform %.4f ms\n",
                        rows, med(proj[0]), med(proj[1]), *std::min_element(down[0].begin(), down[0].end()), *std::min_element(down[1].begin(), down[1].end()));
                    std::printf("da16issuebench rows %u down phase: masked %.4f ms, uniform %.4f ms, delta %.4f ms (%+.2f %%)\n",
                        rows, med(down[0]), med(down[1]), med(down[1]) - med(down[0]), 100.0 * (med(down[1]) - med(down[0])) / med(down[0]));
                }
                reset();
                return 0;
            }
            unsigned cases = 0, failed = 0, switched = 0, certified = 0, unproven = 0;
            // Every comparison runs HDR 0 and 1, each without the certificate, with it armed (view.unsafe bound: certificate
            // prologue/count/down kernels) and with it forced unproven (first route weight 1 + ulp).
            const auto cmp = [&](Fixture& f, unsigned rows, const std::string& what) {
                const std::vector<unsigned char> routes = f.weights.read();
                for (int hdr : {0, 1}) for (int cert : {0, 1, 2}) {
                    kr::set_prefill_hdr(hdr);
                    unsigned tune = 0; bool produced = false;
                    const auto run = [&](int on) {
                        item(on);
                        check(hipMemcpy(f.weights.base, routes.data(), routes.size(), hipMemcpyHostToDevice), "xor64 routes");
                        if (cert == 2) { const float one_plus = std::nextafter(1.0F, 2.0F); check(hipMemcpy(f.weights.base, &one_plus, 4U, hipMemcpyHostToDevice), "xor64 unproven route"); }
                        const unsigned zero = 0U; check(hipMemcpy(f.admission.base, &zero, 4U, hipMemcpyHostToDevice), "xor64 admission reset");
                        f.view.unsafe = cert ? static_cast<unsigned*>(f.admission.base) : nullptr;
                        for (const auto* b : f.written()) b->poison(f.stream);
                        f.csr.poison(f.stream);
                        if (issue_mode) kr::prefill_test_poison_da16(f.stream, rows);
                        f.enqueue(3, rows);
                        produced = kr::prefill_certificate_produced();
                        check(hipStreamSynchronize(f.stream), "xor64 run");
                        if (!kr::prefill_da16_ran()) throw std::runtime_error("xor64: the DA16 path did not run (" + what + ")");
                        tune = kr::prefill_last_tune();
                        std::vector<std::vector<unsigned char>> out;
                        for (const auto* b : f.written()) out.push_back(b->read());
                        out.push_back(f.csr.read());
                        out.push_back(f.admission.read());
                        return out;
                    };
                    const std::string tag = what + " hdr " + std::to_string(hdr) + " cert " + std::to_string(cert);
                    const auto base = run(0);
                    const unsigned tune_off = tune; const bool produced_off = produced;
                    if (run(0) != base) { ++failed; std::printf("  RACY base %s rows %u\n", tag.c_str(), rows); continue; }
                    const auto got = run(1);
                    const unsigned want = !issue_mode && tune_off == 80U && rows >= 3072U ? 88U : tune_off;
                    ++cases;
                    if (tune != want) { ++failed; std::printf("  TUNE %s rows %u: off %u on %u (want %u)\n", tag.c_str(), rows, tune_off, tune, want); }
                    if (produced != produced_off || (cert && !produced)) { ++failed; std::printf("  CERTIFICATE %s rows %u: off %d on %d\n", tag.c_str(), rows, produced_off ? 1 : 0, produced ? 1 : 0); }
                    if (tune == 88U) ++switched;
                    if (cert && produced) { ++certified; unsigned u = 0; std::memcpy(&u, got.back().data(), 4U); unproven += u != 0U ? 1U : 0U; }
                    for (std::size_t k = 0; k < got.size(); ++k) if (got[k] != base[k]) {
                        std::size_t n = 0, first = ~std::size_t{0};
                        for (std::size_t i = 0; i < got[k].size() && i < base[k].size(); ++i) if (got[k][i] != base[k][i]) { if (!n) first = i; ++n; }
                        ++failed; std::printf("  DIFFER %s rows %u: %s %zu bytes, first at %zu\n", tag.c_str(), rows, k < 10U ? names[k] : k == 10U ? "csr" : "unsafe", n, first);
                    }
                }
                check(hipMemcpy(f.weights.base, routes.data(), routes.size(), hipMemcpyHostToDevice), "xor64 routes restore");
                f.view.unsafe = nullptr;
            };
            static std::vector<std::string> keep7; keep7.reserve(128);
            // BENCH_IDENT (da16issue only, for short tickets): "real" = only the real layers, "gd" (e.g. 127, 138) = only that
            // synthetic gate/down type pair; BENCH_IDENT_ROWS = one row count for the real layers.
            const std::string ident = issue_mode && std::getenv("BENCH_IDENT") ? std::getenv("BENCH_IDENT") : "";
            const unsigned ident_rows = issue_mode && std::getenv("BENCH_IDENT_ROWS") ? static_cast<unsigned>(std::atoi(std::getenv("BENCH_IDENT_ROWS"))) : 0U;
            for (unsigned g : {12U, 13U}) for (unsigned d : {7U, 8U}) {
                if (!ident.empty() && ident != std::to_string(g) + std::to_string(d)) continue;
                keep7.push_back("xor64 E512 T4864 g" + std::to_string(g) + " d" + std::to_string(d));
                Fixture f({keep7.back().c_str(), 512U, 4864U, g, d, 4U, false, true, false, 0U, 1.5});
                for (unsigned rows : {4864U, 3072U, 2048U}) cmp(f, rows, keep7.back());
                Case c{keep7.back().c_str(), 512U, 4096U, g, d, 4U, false, false, false}; c.crafted = true;
                Fixture fc(c); cmp(fc, 4096U, keep7.back() + " crafted");
                Case cf{keep7.back().c_str(), 128U, 3072U, g, d, 4U, false, false, false}; cf.crafted = true; cf.full128 = true;
                Fixture ff(cf); cmp(ff, 3072U, keep7.back() + " full128 E128");
                keep7.push_back("xor64 E512 T3072 nonfinite g" + std::to_string(g) + " d" + std::to_string(d));
                Fixture fn({keep7.back().c_str(), 512U, 3072U, g, d, 4U, false, true, true, 0U, 1.5});
                cmp(fn, 3072U, keep7.back());
            }
            for (int a = 2; a < argc; ++a) {
                if (!ident.empty() && ident != "real") break;
                const auto fl = parse(argv[a]);
                keep7.push_back("xor64 REAL " + fl[1]);
                Case c{keep7.back().c_str(), 512U, 8192U, static_cast<unsigned>(std::stoul(fl[4])), static_cast<unsigned>(std::stoul(fl[5])), 0U, false, true, false, 0U, 1.5,
                    nullptr, std::stoull(fl[1]), std::stoull(fl[2]), std::stoull(fl[3])};
                c.gguf = fl[0].c_str();
                Fixture f(c);
                for (unsigned rows : {8192U, 4864U, 4096U, 3072U, 2048U}) if (!ident_rows || rows == ident_rows) cmp(f, rows, keep7.back());
            }
            reset();
            std::printf("%s: %u comparisons (%u took tune 88, %u certified, %u of them unproven), %u differ/racy/wrong tune/certificate -> %s\n",
                issue_mode ? "da16issue" : "xor64", cases, switched, certified, unproven, failed, failed || !cases || (!issue_mode && !switched) || !certified ? "FAIL" : "PASS");
            return failed || !cases || (!issue_mode && !switched) || !certified ? 1 : 0;
        }
        if (mode == "tile64" || mode == "tile64time") {
            // MoE TILE64 (STRIX_QPF2K_MOE_TILE64): default chain (form 3, ROUTE 14, DA16 1) with today's DA16 tile vs the 64x64
            // per-wave register tile (full 128-assignment batches; partial batches keep today's tile in a filtered launch).
            // tile64 [specs...]: byte gate. Every case runs HDR 0/1 x storage FP32 (no certificate), CTRL (certificate, FP32
            // rawdots) and RB16 (certificate, BF16 route dots), and each of TILE64 gate/up only, down only and both against
            // today's path. Gate/up bytes (status, intermediates, down_sums, DA16 FP16 image), down bytes (rawdots) and the
            // rest (A8 views, clear, CSR, certificate word) are compared separately by device digests. The fill buckets
            // (rows per expert) and full/partial batches of every case are counted; each bucket must be covered.
            // tile64time <pairs> <spec> [rows...]: 2 warmups, then <pairs> interleaved off/on pairs per T and storage mode,
            // hipEvent ms per call of the gate/up and the down kernels (medians).
            const auto parse = [](const std::string& x) {
                std::vector<std::string> fl; std::size_t at = 0;
                for (;;) { const std::size_t e = x.find(':', at); fl.push_back(x.substr(at, e == std::string::npos ? std::string::npos : e - at)); if (e == std::string::npos) break; at = e + 1U; }
                if (fl.size() != 6U) throw std::invalid_argument("layer spec needs 6 fields");
                return fl;
            };
            const auto real_case = [](const char* name, const std::vector<std::string>& fl, unsigned rows) {
                Case c{name, 512U, rows, static_cast<unsigned>(std::stoul(fl[4])), static_cast<unsigned>(std::stoul(fl[5])), 0U, false, true, false, 0U, 1.5,
                    nullptr, std::stoull(fl[1]), std::stoull(fl[2]), std::stoull(fl[3])};
                c.gguf = fl[0].c_str();
                return c;
            };
            kr::set_prefill_core2(0); kr::set_prefill_route(14); kr::set_prefill_da16(1); kr::set_prefill_hdr(1);
            const auto reset = [] { kr::set_prefill_tile64(-1); kr::set_prefill_hdr(-1); kr::set_prefill_core2(-1); kr::set_prefill_route(-1);
                kr::set_prefill_da16(-1); kr::set_prefill_test_events(nullptr); };
            const char* const storage_names[3] = {"FP32", "CTRL", "RB16"};
            // Storage mode 0 FP32: no certificate; 1 CTRL: certificate bound, FP32 rawdots; 2 RB16: certificate, BF16 route dots.
            const auto set_storage = [](Fixture& f, int storage) {
                const unsigned zero = 0U; check(hipMemcpy(f.admission.base, &zero, 4U, hipMemcpyHostToDevice), "tile64 admission reset");
                f.view.unsafe = storage ? static_cast<unsigned*>(f.admission.base) : nullptr;
                f.view.route_bf16 = storage == 2;
            };
            if (mode == "tile64time") {
                if (argc < 4) throw std::invalid_argument("tile64time <pairs> <layer spec> [rows...]");
                const int pairs = std::atoi(argv[2]);
                if (pairs <= 0) throw std::invalid_argument("tile64time: pairs must be positive");
                const auto fl = parse(argv[3]);
                std::vector<unsigned> ts;
                for (int a = 4; a < argc; ++a) ts.push_back(static_cast<unsigned>(std::atoi(argv[a])));
                if (ts.empty()) ts = {8192U, 16384U};
                Fixture f(real_case("tile64time", fl, *std::max_element(ts.begin(), ts.end())));
                hipEvent_t ev[3], c0, c1;
                for (auto& e : ev) check(hipEventCreate(&e), "event");
                check(hipEventCreate(&c0), "event"); check(hipEventCreate(&c1), "event");
                kr::set_prefill_test_events(ev);
                const auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
                for (const unsigned rows : ts) for (int storage = 0; storage < 3; ++storage) {
                    set_storage(f, storage);
                    bool tile64_seen = true;
                    const auto timed = [&](int on, std::vector<double>* gu, std::vector<double>* dn, std::vector<double>* call) {
                        kr::set_prefill_tile64(on);
                        check(hipEventRecord(c0, f.stream), "record");
                        f.enqueue(3, rows);
                        check(hipEventRecord(c1, f.stream), "record");
                        check(hipEventSynchronize(c1), "sync");
                        if (!kr::prefill_da16_ran()) throw std::runtime_error("tile64time: the DA16 path did not run");
                        if (on) tile64_seen = tile64_seen && kr::prefill_tile64_ran() && kr::prefill_last_tune() == 82U;
                        float a = 0.0F, b = 0.0F, c = 0.0F;
                        check(hipEventElapsedTime(&a, ev[0], ev[1]), "elapsed"); check(hipEventElapsedTime(&b, ev[1], ev[2]), "elapsed");
                        check(hipEventElapsedTime(&c, c0, c1), "elapsed");
                        if (gu) { gu->push_back(a); dn->push_back(b); call->push_back(c); }
                    };
                    for (int w = 0; w < 2; ++w) { timed(0, nullptr, nullptr, nullptr); timed(1, nullptr, nullptr, nullptr); }
                    std::vector<double> gu0, dn0, call0, gu1, dn1, call1;
                    for (int p = 0; p < pairs; ++p) {
                        if (p % 2 == 0) { timed(0, &gu0, &dn0, &call0); timed(1, &gu1, &dn1, &call1); }
                        else { timed(1, &gu1, &dn1, &call1); timed(0, &gu0, &dn0, &call0); }
                    }
                    const std::string tag = "tile64time T" + std::to_string(rows) + " " + storage_names[storage] + " g" + fl[4] + " d" + fl[5];
                    std::printf("%s pairs %d warmups 2: gate/up off %.4f on %.4f ms (%+.2f %%); down off %.4f on %.4f ms (%+.2f %%); call off %.4f on %.4f ms%s\n",
                        tag.c_str(), pairs, med(gu0), med(gu1), 100.0 * (med(gu1) - med(gu0)) / med(gu0), med(dn0), med(dn1),
                        100.0 * (med(dn1) - med(dn0)) / med(dn0), med(call0), med(call1), tile64_seen ? "" : " (TILE64 NOT TAKEN)");
                    const auto list = [](const char* what, const std::vector<double>& v) { std::printf("  %s:", what); for (const double x : v) std::printf(" %.4f", x); std::printf("\n"); };
                    list("gate/up off", gu0); list("gate/up on", gu1); list("down off", dn0); list("down on", dn1);
                }
                f.view.unsafe = nullptr; f.view.route_bf16 = false;
                reset();
                return 0;
            }
            // Digests of one run: [0] gate/up group, [1] down group, [2] the rest; each a vector of per-buffer digests.
            const auto run_digests = [](Fixture& f, unsigned rows) {
                const auto bufs = f.written();
                const std::size_t n = bufs.size() + 2U;
                unsigned long long* d = nullptr;
                check(hipMalloc(&d, 8U * n), "tile64 digest alloc");
                check(hipMemsetAsync(d, 0, 8U * n, f.stream), "tile64 digest clear");
                for (std::size_t k = 0; k < bufs.size(); ++k)
                    hipLaunchKernelGGL(digest_kernel, dim3(1024), dim3(256), 0, f.stream, static_cast<const unsigned*>(bufs[k]->base), bufs[k]->bytes / 4U, d + k);
                hipLaunchKernelGGL(digest_kernel, dim3(256), dim3(256), 0, f.stream, static_cast<const unsigned*>(f.csr.base), f.csr.bytes / 4U, d + bufs.size());
                const std::uint64_t image_words = static_cast<std::uint64_t>(rows) * Fixture::K * Fixture::I * 2U / 4U;
                hipLaunchKernelGGL(digest_kernel, dim3(1024), dim3(256), 0, f.stream, static_cast<const unsigned*>(kr::prefill_test_da16_image()), image_words, d + bufs.size() + 1U);
                std::vector<unsigned long long> all(n);
                check(hipMemcpyAsync(all.data(), d, 8U * n, hipMemcpyDeviceToHost, f.stream), "tile64 digest read");
                check(hipStreamSynchronize(f.stream), "tile64 digest");
                (void)hipFree(d);
                unsigned admission = 0; check(hipMemcpy(&admission, f.admission.base, 4U, hipMemcpyDeviceToHost), "tile64 admission");
                // written(): codes deltas sums status intermediates clear down_codes down_deltas down_sums rawdots; then csr, image.
                std::array<std::vector<unsigned long long>, 3> g;
                g[0] = {all[3], all[4], all[8], all[bufs.size() + 1U]};
                g[1] = {all[9]};
                g[2] = {all[0], all[1], all[2], all[5], all[6], all[7], all[bufs.size()], admission};
                return g;
            };
            const char* const group_names[3] = {"gate/up (status, intermediates, down_sums, DA16 image)", "down (rawdots)", "rest (A8 views, clear, CSR, certificate)"};
            unsigned cases = 0, failed = 0, tile64_runs = 0, certified = 0, unproven = 0;
            std::array<unsigned, 7> bucket{};  // experts per fill bucket over the compared cases
            unsigned full_batches = 0, partial_batches = 0;
            const unsigned edges[7] = {16U, 32U, 64U, 128U, 256U, 512U, ~0U};
            const auto count_buckets = [&](Fixture& f, unsigned rows) {
                const auto rt = f.routes.read();
                std::vector<unsigned> per(f.c.experts, 0U);
                for (unsigned t = 0; t < rows; ++t) for (unsigned s = 0; s < Fixture::K; ++s) {
                    unsigned e = 0; std::memcpy(&e, rt.data() + static_cast<std::size_t>(t) * f.routes.stride + 4U * s, 4U);
                    if (e < f.c.experts) ++per[e];
                }
                for (const unsigned n : per) {
                    if (!n) continue;
                    unsigned b = 0; while (n > edges[b]) ++b;
                    ++bucket[b]; full_batches += n / 128U; partial_batches += n % 128U ? 1U : 0U;
                }
            };
            const auto cmp = [&](Fixture& f, unsigned rows, const std::string& what) {
                const std::vector<unsigned char> routes = f.weights.read();
                count_buckets(f, rows);
                for (int hdr : {0, 1}) for (int storage = 0; storage < 3; ++storage) {
                    kr::set_prefill_hdr(hdr);
                    unsigned tune = 0; bool produced = false, ran = false;
                    const auto run = [&](int on) {
                        kr::set_prefill_tile64(on);
                        check(hipMemcpy(f.weights.base, routes.data(), routes.size(), hipMemcpyHostToDevice), "tile64 routes");
                        set_storage(f, storage);
                        for (const auto* b : f.written()) b->poison(f.stream);
                        f.csr.poison(f.stream);
                        kr::prefill_test_poison_da16(f.stream, rows);
                        f.enqueue(3, rows);
                        produced = kr::prefill_certificate_produced();
                        ran = kr::prefill_tile64_ran();
                        check(hipStreamSynchronize(f.stream), "tile64 run");
                        if (!kr::prefill_da16_ran()) throw std::runtime_error("tile64: the DA16 path did not run (" + what + ")");
                        tune = kr::prefill_last_tune();
                        return run_digests(f, rows);
                    };
                    const std::string tag = what + " rows " + std::to_string(rows) + " hdr " + std::to_string(hdr) + " " + storage_names[storage];
                    const auto base = run(0);
                    const bool produced_off = produced;
                    if (ran) { ++failed; std::printf("  SWITCH %s: TILE64 ran with the override 0\n", tag.c_str()); }
                    if (run(0) != base) { ++failed; std::printf("  RACY base %s\n", tag.c_str()); continue; }
                    if (tune != 82U) { ++failed; std::printf("  TUNE %s: %u (the case must take the 128-row tune 82)\n", tag.c_str(), tune); continue; }
                    // 1 = both kernels, 2 = TILE64 gate/up with today's down, 3 = today's gate/up with the TILE64 down.
                    for (int on : {2, 3, 1}) {
                        const auto got = run(on);
                        ++cases;
                        const bool want_ran = on != 2 || f.c.gate_type != 13U;
                        if (ran != want_ran) { ++failed; std::printf("  SWITCH %s on %d: TILE64 ran %d, want %d\n", tag.c_str(), on, ran ? 1 : 0, want_ran ? 1 : 0); }
                        tile64_runs += ran ? 1U : 0U;
                        if (produced != produced_off || (storage && !produced)) { ++failed; std::printf("  CERTIFICATE %s on %d: off %d on %d\n", tag.c_str(), on, produced_off ? 1 : 0, produced ? 1 : 0); }
                        if (storage && produced) { ++certified; unproven += got[2].back() != 0U ? 1U : 0U; }
                        for (std::size_t k = 0; k < 3U; ++k) if (got[k] != base[k]) {
                            ++failed;
                            std::printf("  DIFFER %s on %d (%s): %s\n", tag.c_str(), on, on == 2 ? "TILE64 gate/up" : on == 3 ? "TILE64 down" : "TILE64 both", group_names[k]);
                        }
                    }
                }
                check(hipMemcpy(f.weights.base, routes.data(), routes.size(), hipMemcpyHostToDevice), "tile64 routes restore");
                f.view.unsafe = nullptr; f.view.route_bf16 = false;
            };
            static std::vector<std::string> keep8; keep8.reserve(128);
            for (unsigned g : {12U, 13U}) for (unsigned d : {7U, 8U}) {
                const std::string gd = " g" + std::to_string(g) + " d" + std::to_string(d);
                // Skewed routing (power 1.5) at T8192 and the T5120 prefix: every bucket, full and partial batches.
                keep8.push_back("tile64 E512 T8192 skew" + gd);
                { Fixture f({keep8.back().c_str(), 512U, 8192U, g, d, 4U, false, true, false, 0U, 1.5}); for (unsigned rows : {8192U, 5120U}) cmp(f, rows, keep8.back()); }
                // Crafted counts 0,1,15,16,17,63,64,65,127,128,129,255,256,257,500 (+ the rest: >512 with 32 experts).
                keep8.push_back("tile64 crafted E32 T2048" + gd);
                { Case c{keep8.back().c_str(), 32U, 2048U, g, d, 4U, false, false, false}; c.crafted = true; Fixture f(c); cmp(f, 2048U, keep8.back()); }
                keep8.push_back("tile64 crafted E64 T1024" + gd);
                { Case c{keep8.back().c_str(), 64U, 1024U, g, d, 4U, false, false, false}; c.crafted = true; Fixture f(c); cmp(f, 1024U, keep8.back()); }
                // Only full 128-assignment batches (no remainder CTA work).
                keep8.push_back("tile64 full128 E64 T1024" + gd);
                { Case c{keep8.back().c_str(), 64U, 1024U, g, d, 4U, false, false, false}; c.crafted = true; c.full128 = true; Fixture f(c); cmp(f, 1024U, keep8.back()); }
            }
            for (unsigned d : {7U, 8U}) {
                // Edge weight scales (flavor 2: non-finite products, certificate unproven) and a NaN input row.
                keep8.push_back("tile64 E512 T5120 nonfinite g12 d" + std::to_string(d));
                Fixture f({keep8.back().c_str(), 512U, 5120U, 12U, d, 2U, false, true, true, 0U, 1.5});
                cmp(f, 5120U, keep8.back());
            }
            for (int a = 2; a < argc; ++a) {
                const auto fl = parse(argv[a]);
                keep8.push_back("tile64 REAL " + fl[1] + " g" + fl[4] + " d" + fl[5]);
                Fixture f(real_case(keep8.back().c_str(), fl, 16384U));
                for (unsigned rows : {16384U, 8192U, 5120U}) cmp(f, rows, keep8.back());
            }
            reset();
            std::printf("tile64 buckets (experts per rows/expert bucket):");
            const char* const bucket_names[7] = {"1-16", "17-32", "33-64", "65-128", "129-256", "257-512", ">512"};
            bool covered = partial_batches != 0U && full_batches != 0U;
            for (unsigned b = 0; b < 7U; ++b) { std::printf(" %s=%u", bucket_names[b], bucket[b]); covered = covered && bucket[b] != 0U; }
            std::printf("; full batches %u, partial batches %u\n", full_batches, partial_batches);
            const bool pass = !failed && cases && tile64_runs && certified && covered;
            std::printf("tile64: %u comparisons (%u with a TILE64 kernel, %u certified, %u of them unproven), %u differ/racy/wrong tune/switch/certificate -> %s\n",
                cases, tile64_runs, certified, unproven, failed, pass ? "PASS" : "FAIL");
            return pass ? 0 : 1;
        }
        if (mode == "hdr" || mode == "hdrbench") {
            // MoE header reuse (STRIX_QPF2K_MOE_HDR): default chain (form 3, ROUTE 14, DA16 1) with the base gate/up kernel vs the
            // header-reuse kernel. hdr [specs...]: every written view and the summary word byte-identical (exact compares),
            // synthetic Q4_K/Q5_K x Q5_1/Q8_0 flavors 0..4, skew/crafted routes, T48..2048 (tunes 80 and 82), then real layers
            // at T8192/T4096/T1320. hdrbench <reps> <spec> [rows]: complete routed call (prologue..down), ABBA hipEvent medians.
            const auto parse = [](const std::string& x) {
                std::vector<std::string> fl; std::size_t at = 0;
                for (;;) { const std::size_t e = x.find(':', at); fl.push_back(x.substr(at, e == std::string::npos ? std::string::npos : e - at)); if (e == std::string::npos) break; at = e + 1U; }
                if (fl.size() != 6U) throw std::invalid_argument("layer spec needs 6 fields");
                return fl;
            };
            kr::set_prefill_core2(0); kr::set_prefill_route(14); kr::set_prefill_da16(1);
            if (mode == "hdrbench") {
                if (argc < 4) throw std::invalid_argument("hdrbench <reps> <layer spec> [rows]");
                const int reps = std::atoi(argv[2]);
                if (reps <= 0) throw std::invalid_argument("hdrbench: reps must be positive");
                const auto fl = parse(argv[3]);
                const unsigned rows = argc > 4 ? static_cast<unsigned>(std::atoi(argv[4])) : 8192U;
                Case c{"hdrbench", 512U, rows, static_cast<unsigned>(std::stoul(fl[4])), static_cast<unsigned>(std::stoul(fl[5])), 0U, false, true, false, 0U, 1.5,
                    nullptr, std::stoull(fl[1]), std::stoull(fl[2]), std::stoull(fl[3])};
                c.gguf = fl[0].c_str();
                Fixture f(c);
                hipEvent_t e0, e1;
                check(hipEventCreate(&e0), "event"); check(hipEventCreate(&e1), "event");
                const auto timed = [&](int hdr) {
                    kr::set_prefill_hdr(hdr);
                    check(hipEventRecord(e0, f.stream), "record");
                    f.enqueue(3, rows);
                    check(hipEventRecord(e1, f.stream), "record");
                    check(hipEventSynchronize(e1), "sync");
                    float ms = 0.0F; check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
                    return static_cast<double>(ms);
                };
                timed(0); timed(1);
                std::vector<double> a, b;
                for (int r = 0; r < reps; ++r) { a.push_back(timed(0)); b.push_back(timed(1)); b.push_back(timed(1)); a.push_back(timed(0)); }
                const auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
                double round_med = 0.0; { std::vector<double> d; for (std::size_t r = 0; r + 1U < a.size(); r += 2U) d.push_back((a[r] + a[r + 1U] - b[r] - b[r + 1U]) / 2.0); round_med = med(d); }
                std::printf("hdrbench rows %u reps %d: base %.4f ms, header reuse %.4f ms, delta %.4f ms (%.2f %%), per-round median saving %.4f ms\n",
                    rows, reps, med(a), med(b), med(b) - med(a), 100.0 * (med(b) - med(a)) / med(a), round_med);
                std::printf("  base:"); for (const double v : a) std::printf(" %.4f", v);
                std::printf("\n  hdr:"); for (const double v : b) std::printf(" %.4f", v);
                std::printf("\n");
                kr::set_prefill_hdr(-1); kr::set_prefill_core2(-1); kr::set_prefill_route(-1); kr::set_prefill_da16(-1);
                return 0;
            }
            unsigned cases = 0, failed = 0, hdr_ran = 0;
            const auto cmp = [&](Fixture& f, unsigned rows, const std::string& what) {
                const auto run = [&](int hdr) {
                    kr::set_prefill_hdr(hdr);
                    for (const auto* b : f.written()) b->poison(f.stream);
                    f.csr.poison(f.stream);
                    f.enqueue(3, rows);
                    check(hipStreamSynchronize(f.stream), "hdr run");
                    if (!kr::prefill_da16_ran()) throw std::runtime_error("hdr: the DA16 path did not run (" + what + ")");
                    std::vector<std::vector<unsigned char>> out;
                    for (const auto* b : f.written()) out.push_back(b->read());
                    out.push_back(f.csr.read());
                    return out;
                };
                const auto base = run(0);
                if (run(0) != base) { ++failed; std::printf("  RACY base %s rows %u\n", what.c_str(), rows); return; }
                const auto got = run(1);
                ++cases; ++hdr_ran;
                for (std::size_t k = 0; k < got.size(); ++k) if (got[k] != base[k]) {
                    std::size_t n = 0, first = ~std::size_t{0};
                    for (std::size_t i = 0; i < got[k].size() && i < base[k].size(); ++i) if (got[k][i] != base[k][i]) { if (!n) first = i; ++n; }
                    ++failed; std::printf("  DIFFER %s rows %u: %s %zu bytes, first at %zu\n", what.c_str(), rows, k < 10U ? names[k] : "csr", n, first);
                }
            };
            static std::vector<std::string> keep5; keep5.reserve(128);
            for (unsigned g : {12U, 13U}) for (unsigned d : {7U, 8U}) for (unsigned flavor : {0U, 1U, 2U, 3U, 4U}) {
                keep5.push_back("hdr E40 T48 g" + std::to_string(g) + " d" + std::to_string(d) + " flavor " + std::to_string(flavor));
                Fixture f({keep5.back().c_str(), 40U, 48U, g, d, flavor, true, true, flavor == 2U});
                cmp(f, 48U, keep5.back());
            }
            for (unsigned g : {12U, 13U}) for (unsigned d : {7U, 8U}) {
                keep5.push_back("hdr E512 T2048 g" + std::to_string(g) + " d" + std::to_string(d));
                Fixture f({keep5.back().c_str(), 512U, 2048U, g, d, 4U, false, true, false, 0U, 1.5});
                cmp(f, 2048U, keep5.back());
                Case c{keep5.back().c_str(), 512U, 1024U, g, d, 4U, false, false, false}; c.crafted = true;
                Fixture fc(c); cmp(fc, 1024U, keep5.back() + " crafted");
                Case cf{keep5.back().c_str(), 64U, 1024U, g, d, 4U, false, false, false}; cf.crafted = true; cf.full128 = true;
                Fixture ff(cf); cmp(ff, 1024U, keep5.back() + " full128");
            }
            for (int a = 2; a < argc; ++a) {
                const auto fl = parse(argv[a]);
                keep5.push_back("hdr REAL " + fl[1]);
                Case c{keep5.back().c_str(), 512U, 8192U, static_cast<unsigned>(std::stoul(fl[4])), static_cast<unsigned>(std::stoul(fl[5])), 0U, false, true, false, 0U, 1.5,
                    nullptr, std::stoull(fl[1]), std::stoull(fl[2]), std::stoull(fl[3])};
                c.gguf = fl[0].c_str();
                Fixture f(c);
                for (unsigned rows : {8192U, 4096U, 1320U}) cmp(f, rows, keep5.back());
            }
            kr::set_prefill_hdr(-1); kr::set_prefill_core2(-1); kr::set_prefill_route(-1); kr::set_prefill_da16(-1);
            std::printf("hdr: %u comparisons (%u with the header-reuse kernel), %u differ/racy -> %s\n", cases, hdr_ran, failed,
                failed || !cases ? "FAIL" : "PASS");
            return failed || !cases ? 1 : 0;
        }
        if (mode == "hdrmc") {
            // MoE header reuse through the real tail: X = HDR 0, Y = HDR 1, both with the certificate producers and the F02
            // merge+commit/finish selection (C arm of mcbench), X/Y/Y/X per round, hipEvent time of the complete transaction.
            if (argc < 4) throw std::invalid_argument("hdrmc <reps> <layer spec> [rows]");
            const int reps = std::atoi(argv[2]);
            if (reps<=0) throw std::invalid_argument("hdrmc: reps must be positive");
            const std::string x = argv[3];
            std::vector<std::string> fl; std::size_t at = 0;
            for (;;) { const std::size_t e = x.find(':', at); fl.push_back(x.substr(at, e == std::string::npos ? std::string::npos : e - at)); if (e == std::string::npos) break; at = e + 1U; }
            const unsigned rows = argc > 4 ? static_cast<unsigned>(std::atoi(argv[4])) : 8192U;
            Case c{"hdrmc", 512U, rows, static_cast<unsigned>(std::stoul(fl[4])), static_cast<unsigned>(std::stoul(fl[5])), 0U, false, true, false, 0U, 1.5,
                nullptr, std::stoull(fl[1]), std::stoull(fl[2]), std::stoull(fl[3])};
            c.gguf = fl[0].c_str();
            Fixture f(c);
            // The fixture's random route generator permits weights above one. Use normalized routes
            // for the admitted measurement; measure a 1+ulp route separately below.
            std::vector<unsigned char> route_weights=f.weights.read();
            for (unsigned r=0;r<rows;++r) {
                float total=0.0F;
                for (unsigned slot=0;slot<Fixture::K;++slot) { float w; std::memcpy(&w,route_weights.data()+r*f.weights.stride+4U*slot,4U); total+=w; }
                for (unsigned slot=0;slot<Fixture::K;++slot) {
                    float w; auto* dst=route_weights.data()+r*f.weights.stride+4U*slot;
                    std::memcpy(&w,dst,4U); w/=total; std::memcpy(dst,&w,4U);
                }
            }
            check(hipMemcpy(f.weights.base,route_weights.data(),f.weights.bytes,hipMemcpyHostToDevice),"normalized benchmark routes");
            Buffer out16(rows, 2U * Fixture::H), record(1, 8U);
            kr::set_prefill_core2(0); kr::set_prefill_route(14); kr::set_prefill_da16(1); kr::set_prefill_small(0);
            const dim3 grid((Fixture::H + 255U) / 256U, rows);
            const kr::StridedRows<std::uint16_t> o{static_cast<std::uint16_t*>(out16.base), out16.stride};
            auto* rec = static_cast<unsigned*>(record.base);
            const auto tail = [&](bool cand) {
                if (!cand || !kr::prefill_certificate_produced()) {
                    hipLaunchKernelGGL(reference_merge_kernel_z, grid, dim3(256), 0, f.stream, f.view, f.view.csr.summary, Fixture::H, f.c.experts, Fixture::K);
                    hipLaunchKernelGGL(reference_commit_kernel, grid, dim3(256), 0, f.stream, f.view, f.view.csr.summary, rows, Fixture::H, o, rec);
                } else {
                    hipLaunchKernelGGL(candidate_merge_commit_kernel<false>, grid, dim3(256), 0, f.stream, f.view, f.view.csr.summary, Fixture::H, f.c.experts, Fixture::K, o);
                    hipLaunchKernelGGL(candidate_finish_kernel, dim3(32), dim3(256), 0, f.stream, f.view, f.view.csr.summary, rows, Fixture::H, o, rec);
                }
            };
            hipEvent_t e0, e1;
            check(hipEventCreate(&e0), "event"); check(hipEventCreate(&e1), "event");
            const auto timed = [&](int hdr) {
                kr::set_prefill_hdr(hdr);
                f.view.unsafe=static_cast<unsigned*>(f.admission.base);
                check(hipEventRecord(e0, f.stream), "record");
                f.enqueue(3, rows);
                tail(true);
                check(hipEventRecord(e1, f.stream), "record");
                check(hipEventSynchronize(e1), "sync");
                float ms = 0.0F; check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
                return static_cast<double>(ms);
            };
            timed(0); timed(1);
            const bool produced=kr::prefill_certificate_produced();
            unsigned unsafe=0U;
            check(hipStreamSynchronize(f.stream),"witness");
            if (produced) check(hipMemcpy(&unsafe,f.view.unsafe,4U,hipMemcpyDeviceToHost),"admission");
            std::vector<double> t0, t1, d;
            for (int r = 0; r < reps; ++r) {
                const double x1 = timed(0), y1 = timed(1), y2 = timed(1), x2 = timed(0);
                t0.push_back(x1); t0.push_back(x2); t1.push_back(y1); t1.push_back(y2); d.push_back((x1 + x2 - y1 - y2) / 2.0);
            }
            const auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
            unsigned positive = 0; for (const double v : d) positive += v > 0.0 ? 1U : 0U;
            std::printf("hdrmc rows %u reps %d certificate=%u unsafe=%u: HDR0 %.4f ms, HDR1 %.4f ms, per-round median saving %.4f ms (%.2f %%), rounds > 0: %u/%d\n",
                rows, reps, produced ? 1U : 0U, unsafe, med(t0), med(t1), med(d), 100.0 * med(d) / med(t0), positive, reps);
            std::printf("  hdr0:"); for (const double v : t0) std::printf(" %.4f", v);
            std::printf("\n  hdr1:"); for (const double v : t1) std::printf(" %.4f", v);
            std::printf("\n");
            check(hipEventDestroy(e0),"destroy event"); check(hipEventDestroy(e1),"destroy event");
            kr::set_prefill_hdr(-1); kr::set_prefill_core2(-1); kr::set_prefill_route(-1); kr::set_prefill_da16(-1); kr::set_prefill_small(-1);
            return 0;
        }
        if (mode == "mcbench") {
            // B/P/C/C/P/B: B base; P certificate producers + base tail; C selected merge/finish.
            // Sparse tune-80 calls deliberately use B in every arm, matching production selection.
            if (argc < 4) throw std::invalid_argument("mcbench <reps> <layer spec> [rows]");
            const int reps = std::atoi(argv[2]);
            if (reps<=0) throw std::invalid_argument("mcbench: reps must be positive");
            const std::string x = argv[3];
            std::vector<std::string> fl; std::size_t at = 0;
            for (;;) { const std::size_t e = x.find(':', at); fl.push_back(x.substr(at, e == std::string::npos ? std::string::npos : e - at)); if (e == std::string::npos) break; at = e + 1U; }
            const unsigned rows = argc > 4 ? static_cast<unsigned>(std::atoi(argv[4])) : 8192U;
            Case c{"mcbench", 512U, rows, static_cast<unsigned>(std::stoul(fl[4])), static_cast<unsigned>(std::stoul(fl[5])), 0U, false, true, false, 0U, 1.5,
                nullptr, std::stoull(fl[1]), std::stoull(fl[2]), std::stoull(fl[3])};
            c.gguf = fl[0].c_str();
            Fixture f(c);
            // The fixture's random route generator permits weights above one. Use normalized routes
            // for the admitted measurement; measure a 1+ulp route separately below.
            std::vector<unsigned char> route_weights=f.weights.read();
            for (unsigned r=0;r<rows;++r) {
                float total=0.0F;
                for (unsigned slot=0;slot<Fixture::K;++slot) { float w; std::memcpy(&w,route_weights.data()+r*f.weights.stride+4U*slot,4U); total+=w; }
                for (unsigned slot=0;slot<Fixture::K;++slot) {
                    float w; auto* dst=route_weights.data()+r*f.weights.stride+4U*slot;
                    std::memcpy(&w,dst,4U); w/=total; std::memcpy(dst,&w,4U);
                }
            }
            check(hipMemcpy(f.weights.base,route_weights.data(),f.weights.bytes,hipMemcpyHostToDevice),"normalized benchmark routes");
            Buffer out16(rows, 2U * Fixture::H), record(1, 8U);
            kr::set_prefill_core2(0); kr::set_prefill_route(14); kr::set_prefill_da16(1); kr::set_prefill_small(0);
            const dim3 grid((Fixture::H + 255U) / 256U, rows);
            const kr::StridedRows<std::uint16_t> o{static_cast<std::uint16_t*>(out16.base), out16.stride};
            auto* rec = static_cast<unsigned*>(record.base);
            // cand: 0 base merge + commit, 1 MC merge, 2 MC merge vec4.
            const auto tail = [&](int cand) {
                if (!cand || !kr::prefill_certificate_produced()) {
                    hipLaunchKernelGGL(reference_merge_kernel_z, grid, dim3(256), 0, f.stream, f.view, f.view.csr.summary, Fixture::H, f.c.experts, Fixture::K);
                    hipLaunchKernelGGL(reference_commit_kernel, grid, dim3(256), 0, f.stream, f.view, f.view.csr.summary, rows, Fixture::H, o, rec);
                } else {
                    if (cand == 2) candidate_merge_commit_vec4<false>(f.stream, f.view, f.view.csr.summary, Fixture::H, f.c.experts, Fixture::K, o, rows);
                    else hipLaunchKernelGGL(candidate_merge_commit_kernel<false>, grid, dim3(256), 0, f.stream, f.view, f.view.csr.summary, Fixture::H, f.c.experts, Fixture::K, o);
                    hipLaunchKernelGGL(candidate_finish_kernel, dim3(32), dim3(256), 0, f.stream, f.view, f.view.csr.summary, rows, Fixture::H, o, rec);
                }
            };
            hipEvent_t e0, e1;
            check(hipEventCreate(&e0), "event"); check(hipEventCreate(&e1), "event");
            const auto timed = [&](bool call, unsigned arm) {
                f.view.unsafe=(!call||arm)?static_cast<unsigned*>(f.admission.base):nullptr;
                check(hipEventRecord(e0, f.stream), "record");
                if (call) f.enqueue(3, rows);
                tail(arm==2U ? 1 : arm==3U ? 2 : 0);
                check(hipEventRecord(e1, f.stream), "record");
                check(hipEventSynchronize(e1), "sync");
                float ms = 0.0F; check(hipEventElapsedTime(&ms, e0, e1), "elapsed");
                return static_cast<double>(ms);
            };
            for (bool force_unsafe : {false,true}) {
                const float one_plus=std::nextafter(1.0F,2.0F);
                check(hipMemcpy(f.weights.base,route_weights.data(),f.weights.bytes,hipMemcpyHostToDevice),"restore benchmark routes");
                if (force_unsafe) check(hipMemcpy(f.weights.base,&one_plus,4U,hipMemcpyHostToDevice),"unproven benchmark route");
                f.view.unsafe=static_cast<unsigned*>(f.admission.base);
                f.enqueue(3, rows);
                const bool produced=kr::prefill_certificate_produced();
                check(hipStreamSynchronize(f.stream),"benchmark producer witness");
                unsigned unsafe=0U;
                if (produced) check(hipMemcpy(&unsafe,f.view.unsafe,4U,hipMemcpyDeviceToHost),"benchmark admission");
                std::printf("mcbench %s selection: certificate=%u unsafe=%u (sparse/unsupported calls use base tail)\n",
                    force_unsafe?"unproven":"normal",produced?1U:0U,unsafe);
                if (produced && unsafe!=static_cast<unsigned>(force_unsafe)) throw std::runtime_error("mcbench: unexpected admission selection");
                tail(false); tail(true); check(hipStreamSynchronize(f.stream), "warm");
                unsigned summary = 0; check(hipMemcpy(&summary, f.view.csr.summary, 4U, hipMemcpyDeviceToHost), "summary");
                if (summary) throw std::runtime_error("mcbench: the real-layer call failed (summary nonzero)");
                for (const bool call : {false, true}) {
                    std::vector<double> tb,tp,tc,tv;
                    for (int r=0;r<reps;++r) {
                        // B/P/C/V/V/C/P/B balances first-order drift. Tail arms reuse the warm certificate.
                        tb.push_back(timed(call,0U)); tp.push_back(timed(call,1U)); tc.push_back(timed(call,2U)); tv.push_back(timed(call,3U));
                        tv.push_back(timed(call,3U)); tc.push_back(timed(call,2U)); tp.push_back(timed(call,1U)); tb.push_back(timed(call,0U));
                    }
                    const auto med=[](std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];};
                    std::printf("mcbench %s rows %u reps %d: B %.4f ms, P %.4f ms, C %.4f ms, V %.4f ms, P-B %.4f ms, B-C %.4f ms, C-V %.4f ms\n",
                        call?"call":"tail",rows,reps,med(tb),med(tp),med(tc),med(tv),med(tp)-med(tb),med(tb)-med(tc),med(tc)-med(tv));
                    for (const auto* v : {&tb,&tp,&tc,&tv}) { for (double ms:*v) std::printf(" %.4f",ms); std::printf("\n"); }
                }
            }
            check(hipEventDestroy(e0),"destroy event"); check(hipEventDestroy(e1),"destroy event");
            kr::set_prefill_core2(-1); kr::set_prefill_route(-1); kr::set_prefill_da16(-1); kr::set_prefill_small(-1);
            return 0;
        }
        if (mode == "fuse") {
            // qpf2k/moe core2 bit 9: base (core 0 + reference ordered merge) vs fused (core 513/641 + merge check): every written
            // buffer (accumulation = clear, statuses, raw dots, ...) and the 32 summary words. Args: real layer specs.
            unsigned cases = 0, failed = 0;
            const auto run = [](Fixture& f, int core, unsigned rows) {
                kr::set_prefill_core2(core);
                for (const auto* b : f.written()) b->poison(f.stream);
                f.csr.poison(f.stream);
                f.enqueue(3, rows);
                if (core && kr::prefill_fused_merge()) kr::enqueue_prefill_merge_check(f.view, rows, Fixture::H, f.c.experts, Fixture::K, f.stream);
                else hipLaunchKernelGGL(reference_merge_kernel, dim3((Fixture::H + 255U) / 256U, rows), dim3(256), 0, f.stream, f.view, f.view.csr.summary, Fixture::H, f.c.experts, Fixture::K);
                check(hipGetLastError(), "merge");
                check(hipStreamSynchronize(f.stream), "fuse run");
                const bool fused = kr::prefill_fused_merge();
                auto d = digests(f);
                d.push_back(fused ? 1ULL : 0ULL);
                return d;
            };
            const auto cmp = [&](Fixture& f, unsigned rows, const std::string& what) {
                auto base = run(f, 0, rows);
                // The serial merge races header reads against non-finite ORs: a base that is not reproducible is skipped.
                if (run(f, 0, rows) != base) { std::printf("  RACY base %s rows %u: skipped\n", what.c_str(), rows); return; }
                for (const int core : {513, 641}) {
                    auto got = run(f, core, rows);
                    ++cases;
                    const bool fused = got.back() == 1ULL;
                    got.back() = 0ULL;
                    if (!fused) { ++failed; std::printf("  NOT FUSED %s rows %u core %d\n", what.c_str(), rows, core); continue; }
                    if (got != base) { ++failed; for (std::size_t k = 0; k + 1 < got.size(); ++k) if (got[k] != base[k])
                        std::printf("  DIFFER %s rows %u core %d: %s\n", what.c_str(), rows, core, k < 10U ? names[k] : "summary"); }
                }
            };
            static std::vector<std::string> keep3; keep3.reserve(128);
            for (unsigned g : {12U, 13U}) for (unsigned d : {7U, 8U}) for (unsigned flavor : {0U, 1U, 2U, 3U, 4U}) {
                keep3.push_back("fuse E40 T48 g" + std::to_string(g) + " d" + std::to_string(d) + " flavor " + std::to_string(flavor));
                Fixture f({keep3.back().c_str(), 40U, 48U, g, d, flavor, true, true, flavor == 2U});
                cmp(f, 48U, keep3.back());
            }
            for (unsigned g : {12U, 13U}) for (bool bad : {false, true}) {
                keep3.push_back(std::string("fuse crafted T520 g") + std::to_string(g) + (bad ? " bad route" : ""));
                Case c{keep3.back().c_str(), 512U, 520U, g, 7U, 4U, false, false, false}; c.crafted = true; c.bad_route = bad;
                Fixture f(c); cmp(f, 520U, keep3.back());
            }
            for (unsigned g : {12U, 13U}) for (unsigned d : {7U, 8U}) {
                keep3.push_back("fuse E512 T2048 g" + std::to_string(g) + " d" + std::to_string(d));
                Fixture f({keep3.back().c_str(), 512U, 2048U, g, d, 4U, false, true, false, 0U, 1.5});
                cmp(f, 2048U, keep3.back());
                Case c{keep3.back().c_str(), 512U, 1024U, g, d, 4U, false, false, false}; c.crafted = true;
                Fixture fc(c); cmp(fc, 1024U, keep3.back() + " crafted");
            }
            for (int a = 2; a < argc; ++a) {
                const std::string x = argv[a];
                std::vector<std::string> fl; std::size_t at = 0;
                for (;;) { const std::size_t e = x.find(':', at); fl.push_back(x.substr(at, e == std::string::npos ? std::string::npos : e - at)); if (e == std::string::npos) break; at = e + 1U; }
                keep3.push_back("fuse REAL " + fl[1]);
                Case c{keep3.back().c_str(), 512U, 8192U, static_cast<unsigned>(std::stoul(fl[4])), static_cast<unsigned>(std::stoul(fl[5])), 0U, false, true, false, 0U, 1.5,
                    nullptr, std::stoull(fl[1]), std::stoull(fl[2]), std::stoull(fl[3])};
                c.gguf = fl[0].c_str();
                Fixture f(c);
                for (unsigned rows : {8192U, 4096U, 1320U}) cmp(f, rows, keep3.back());
            }
            kr::set_prefill_core2(-1);
            std::printf("fuse: %u comparisons, %u differ -> %s\n", cases, failed, failed ? "FAIL" : "PASS");
            return failed ? 1 : 0;
        }
        if (mode == "fastdq") {
            // MoE tune bit 6: short exact FP16 dequantization == base conversion wherever its check admits it.
            const unsigned blocks = argc > 2 ? static_cast<unsigned>(std::atoi(argv[2])) : 4096U;
            const auto c = kr::fast_dequant_selftest(0x5eed5eedULL, blocks);
            std::printf("fastdq K-quant tested %llu admitted %llu mismatched words %llu | Q5_1 %llu %llu %llu | Q8_0 %llu %llu %llu\n",
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]);
            const bool ok = c[2] == 0U && c[5] == 0U && c[1] > 0U && c[4] > 0U;  // Q8_0 (c[6..8]) is informational
            std::printf("fastdq %s\n", ok ? "PASS" : "FAIL");
            return ok ? 0 : 1;
        }
        if (mode == "bench1") {
            // bench1 <reps> <T> <gate type> <down type> <skew 0/1> [flavor]
            if (argc < 7) throw std::invalid_argument("bench1 <reps> <T> <gate type> <down type> <skew>");
            const unsigned T = static_cast<unsigned>(std::atoi(argv[3])), g = static_cast<unsigned>(std::atoi(argv[4]));
            const unsigned d = static_cast<unsigned>(std::atoi(argv[5]));
            const std::string name = "E512 T" + std::to_string(T) + (g == 12U ? " q4k" : g == 42U ? " q20" : " q5k") +
                (d == 7U ? "/q51" : d == 42U ? "/q20" : "/q80") +
                (argv[6][0] == '1' ? " skew" : "");
            // Optional argv[7]: weight-scale flavor (4 = one binade, as the production Qwen scales: MoE tune bit 6 admits
            // every block of them; 0 = scales over seven binades).
            const unsigned flavor = argc > 7 ? static_cast<unsigned>(std::atoi(argv[7])) : 0U;
            bench({name.c_str(), 512U, T, g, d, flavor, false, argv[6][0] == '1', false}, std::atoi(argv[2]));
            return 0;
        }
        if (mode == "f16") {
            // FP16 WMMA (form 3) and iu8 WMMA (form 1) against the FP32 reference.
            unsigned failed = 0;
            static std::vector<std::string> labels;
            labels.reserve(16);
            for (const unsigned g : {12U, 13U}) for (const unsigned d : {7U, 8U}) {
                labels.push_back(std::string("E512 T256 ") + (g == 12U ? "q4k" : "q5k") + (d == 7U ? "/q51" : "/q80"));
                failed += accuracy({labels.back().c_str(), 512U, 256U, g, d, 0U, false, false, false}) ? 0U : 1U;
            }
            failed += accuracy({"E64 T128 dup q4k/q51", 64U, 128U, 12U, 7U, 0U, true, false, false}) ? 0U : 1U;
            failed += accuracy({"E40 T48 skew q5k/q80", 40U, 48U, 13U, 8U, 0U, false, true, false}) ? 0U : 1U;
            failed += accuracy({"E512 T1024 q4k/q51 skew", 512U, 1024U, 12U, 7U, 0U, false, true, false}) ? 0U : 1U;
            std::printf("f16: %u failed\n", failed);
            return failed ? 1 : 0;
        }
        if (mode == "gsq") {
            // GSQ Q2_0 experts in the fast prefill profile: (1) byte-equivalence with the production FP16 kernels on
            // weights both formats hold, over FP16 schedule forms; (2) accuracy of the dot4 (form 1) and FP16 (form 3)
            // forms against the float64 oracle, determinism and batch independence; (3) every schedule form writes the
            // bytes of form 3, also on edge and non-finite scales.
            unsigned failed = 0;
            static std::vector<std::string> labels;
            labels.reserve(96);
            // (D) Decoders: every FP16 scale pattern in an even and an odd block, every code byte per position.
            for (const unsigned seed : {0U, 1U}) {
                const auto d = kr::gsq_q20_decode_selftest(seed, 1024U);
                std::printf("gsq Q2_0 decode seed %u: FP16 words %llu, differing %llu (NaN references %llu); iu8 code bytes %llu, differing %llu\n",
                    seed, d[0], d[1], d[2], d[3], d[4]);
                failed += d[1] != 0U || d[4] != 0U || d[0] != 128ULL * 65536U || d[3] != 64ULL * 65536U ? 1U : 0U;
            }
            // (0) A8 profile: the iu8 WMMA Q2_0 form (forms 1, 2) writes the bytes of the dot4 form (form 0), NaN sign/payload
            // aside; forced per-expert counts 1..16 (CSR batches of 16), extreme codes, repeated runs.
            for (unsigned g = 1; g <= 16U; ++g) {
                labels.push_back("E512 T48 group " + std::to_string(g) + " q20/q20");
                failed += compare({labels.back().c_str(), 512U, 48U, 42U, 42U, 0U, false, false, false, g}, 2U) ? 0U : 1U;
            }
            labels.push_back("E512 T256 extreme q20/q20"); failed += compare({labels.back().c_str(), 512U, 256U, 42U, 42U, 6U, false, false, false}, 3U) ? 0U : 1U;
            labels.push_back("E40 T48 skew extreme q20/q20"); failed += compare({labels.back().c_str(), 40U, 48U, 42U, 42U, 6U, false, true, false}, 3U) ? 0U : 1U;
            for (const unsigned flavor : {0U, 1U, 2U, 3U}) {
                const std::string fmt = std::string("q20/q20") + (flavor == 1U ? " edge" : flavor == 2U ? " nonfinite" : flavor == 3U ? " nonfinite-down" : "");
                labels.push_back("E512 T256 " + fmt); failed += compare({labels.back().c_str(), 512U, 256U, 42U, 42U, flavor, false, false, false}) ? 0U : 1U;
                labels.push_back("E64 T128 dup " + fmt); failed += compare({labels.back().c_str(), 64U, 128U, 42U, 42U, flavor, true, false, false}) ? 0U : 1U;
                labels.push_back("E40 T48 skew " + fmt); failed += compare({labels.back().c_str(), 40U, 48U, 42U, 42U, flavor, false, true, false}) ? 0U : 1U;
            }
            labels.push_back("E512 T2048 q20/q20 skew"); failed += compare({labels.back().c_str(), 512U, 2048U, 42U, 42U, 0U, false, true, false}) ? 0U : 1U;
            labels.push_back("E10 T24 dup q20/q20"); failed += compare({labels.back().c_str(), 10U, 24U, 42U, 42U, 0U, true, true, false}) ? 0U : 1U;
            labels.push_back("E512 T64 q20/q20 nan input"); failed += compare({labels.back().c_str(), 512U, 64U, 42U, 42U, 0U, false, false, true}) ? 0U : 1U;
            const std::initializer_list<int> forms{3, 4, 5, 19, 21, 35, 55, 66};
            const auto equiv = [&](const char* name, unsigned experts, unsigned rows, unsigned flavor, bool dup, bool skew, bool nan) {
                labels.push_back(name);
                failed += gsq_equivalence({labels.back().c_str(), experts, rows, 42U, 42U, flavor, dup, skew, nan}, forms) ? 0U : 1U;
            };
            equiv("equiv E512 T256 q20", 512U, 256U, 0U, false, false, false);
            equiv("equiv E64 T128 dup q20", 64U, 128U, 0U, true, false, false);
            equiv("equiv E40 T48 skew q20", 40U, 48U, 0U, false, true, false);
            equiv("equiv E16 T136 full batches q20", 16U, 136U, 0U, true, true, false);
            equiv("equiv E10 T136 full batches q20", 10U, 136U, 0U, true, true, false);
            equiv("equiv E512 T1024 skew q20", 512U, 1024U, 0U, false, true, false);
            equiv("equiv E512 T2048 q20", 512U, 2048U, 0U, false, false, false);
            equiv("equiv E16 T16 q20", 16U, 16U, 0U, false, false, false);
            equiv("equiv E512 T256 edge-down q20", 512U, 256U, 5U, false, false, false);
            equiv("equiv E512 T256 nonfinite-down q20", 512U, 256U, 3U, false, false, false);
            equiv("equiv E512 T64 nan input q20", 512U, 64U, 0U, false, false, true);
            equiv("equiv E512 T256 extreme q20", 512U, 256U, 6U, false, false, false);
            // Forced counts at the FP16 batch boundaries (64; 128 with tune bit 1, which needs >= 96 assignments per
            // expert of the request, hence experts = 10 * groups).
            const auto forced = [&](unsigned group, unsigned rows) {
                labels.push_back("equiv group " + std::to_string(group) + " T" + std::to_string(rows) + " q20");
                const unsigned experts = 10U * ((rows + group - 1U) / group);
                failed += gsq_equivalence({labels.back().c_str(), experts, rows, 42U, 42U, 0U, false, false, false, group}, forms) ? 0U : 1U;
            };
            forced(63U, 128U); forced(64U, 128U); forced(65U, 136U); forced(127U, 512U); forced(128U, 512U); forced(129U, 512U);
            const auto acc = [&](const char* name, unsigned experts, unsigned rows, bool dup, bool skew) {
                labels.push_back(name);
                failed += accuracy<double>({labels.back().c_str(), experts, rows, 42U, 42U, 0U, dup, skew, false}) ? 0U : 1U;
            };
            acc("E512 T256 q20/q20", 512U, 256U, false, false);
            acc("E64 T128 dup q20/q20", 64U, 128U, true, false);
            acc("E40 T48 skew q20/q20", 40U, 48U, false, true);
            acc("E512 T1024 skew q20/q20", 512U, 1024U, false, true);
            acc("E512 T2048 q20/q20", 512U, 2048U, false, false);
            // (I) IQ types: decoders against the format's element values; FP16 forms against the float64 oracle (the dot4
            // form as the A8 baseline), determinism and batch independence; schedules.
            for (const unsigned type : {22U, 18U, 21U, 23U, 20U}) for (const unsigned seed : {0U, 1U}) {
                const auto d = kr::gsq_iq_decode_selftest(type, seed, 65536U + 4096U);
                // 69,640 blocks (whole rows): 17,827,840 values for the 256-weight types, 2,228,480 for IQ4_NL; every grid
                // index (512 / 256 / 1024; IQ4: 16 nibbles) and every sign pattern (256; IQ3_XXS 128 7-bit) seen.
                const unsigned long long words = type == 20U ? 2228480ULL : 17827840ULL;
                const unsigned long long gridn = type == 21U ? 512U : type == 18U ? 256U : type == 22U ? 1024U : 16U;
                const unsigned long long signn = type == 18U ? 128U : type == 20U || type == 23U ? 0U : 256U;
                std::printf("gsq IQ type %u decode seed %u: FP16 words %llu, differing %llu (NaN references %llu), grid %llu/%llu, signs %llu/%llu\n",
                    type, seed, d[0], d[1], d[2], d[3], gridn, d[4], signn);
                failed += d[1] != 0U || d[0] != words || d[3] != gridn || d[4] != signn ? 1U : 0U;
            }
            {   // Geometry admission: rows that are not whole dwords keep the A8 forms (H 2304 = 9 IQ3_S blocks, 990 bytes).
                a8::Request r; r.enabled = true; r.input = a8::InputKind::bf16; r.gate_type = r.up_type = 21U;
                r.shape = {2304U, 640U, 512U, 10U, 16U};
                const bool guard = kr::prefill_kernel_form(r, 20U, 3) != kr::PrefillKernelForm::f16_wmma;
                r.shape.hidden = 2560U;
                const bool admit = kr::prefill_kernel_form(r, 20U, 3) == kr::PrefillKernelForm::f16_wmma;
                std::printf("gsq geometry: H 2304 IQ3_S %s, H 2560 %s\n", guard ? "kept on A8" : "FP16 (WRONG)", admit ? "FP16" : "not FP16 (WRONG)");
                failed += guard && admit ? 0U : 1U;
            }
            for (const unsigned g : {22U, 18U, 21U, 23U}) for (const unsigned d : {20U, 42U}) {
                labels.push_back("E512 T256 iq" + std::to_string(g) + "/" + std::to_string(d));
                failed += accuracy<double>({labels.back().c_str(), 512U, 256U, g, d, 0U, false, false, false}) ? 0U : 1U;
            }
            labels.push_back("E512 T1024 skew iq21/iq20"); failed += accuracy<double>({labels.back().c_str(), 512U, 1024U, 21U, 20U, 0U, false, true, false}) ? 0U : 1U;
            unsigned cases = 0;
            // TPC 2 (128-assignment batches: forced groups of 128 at T 512, 40 experts) against TPC 1 forms on the same
            // fixture: both B layouts write the same bytes; backend and batch sizes asserted.
            for (const unsigned g : {22U, 18U, 21U, 23U, 42U}) for (const unsigned d : {20U, 42U}) {
                Fixture f({"GSQ FP16 TPC2 vs TPC1", 40U, 512U, g, d, 0U, false, false, false, 128U});
                if (kr::prefill_kernel_form(f.request, d, 3) != kr::PrefillKernelForm::f16_wmma ||
                    !(kr::prefill_effective_tune(f.request, 512U, d, 3) & 2U)) { std::printf("  type %u/%u: not FP16 with bit 1\n", g, d); ++failed; continue; }
                const auto base = f.run(3);
                const unsigned largest = largest_batch(f);
                std::uint32_t summary; std::memcpy(&summary, base.back().data(), 4);
                bool ok = largest == 128U && summary == 0U;
                for (const int form : {4, 19, 35, 55}) ok = same_outputs(base, f.run(form), "GSQ FP16 TPC2 vs TPC1") && ok;
                std::printf("GSQ FP16 TPC2 (batch %u) vs TPC1 forms, gate %u down %u: %s (summary %#x)\n", largest, g, d, ok ? "PASS" : "FAIL", summary);
                failed += ok ? 0U : 1U;
            }
            for (const unsigned g : {22U, 18U, 21U, 23U}) {
                Fixture f({"GSQ IQ FP16 schedule", 40U, 48U, g, g == 23U ? 42U : 20U, 0U, true, true, false});
                const auto base = f.run(3);
                for (int form = 4; form <= 66; ++form) {
                    if (!same_outputs(base, f.run(form), "GSQ IQ FP16 schedule")) { ++failed; std::printf("  form %d differs (type %u)\n", form, g); }
                    ++cases;
                }
            }
            for (unsigned flavor : {0U, 1U, 2U, 3U}) {
                Fixture f({"GSQ FP16 schedule", 40U, 48U, 42U, 42U, flavor, true, true, flavor == 2U});
                const auto base = f.run(3);
                for (int form = 4; form <= 66; ++form) {
                    if (!same_outputs(base, f.run(form), "GSQ FP16 schedule")) { ++failed; std::printf("  form %d differs (flavor %u)\n", form, flavor); }
                    ++cases;
                }
            }
            for (unsigned experts : {16U, 10U}) {
                Fixture f({"GSQ FP16 schedule full batches", experts, 136U, 42U, 42U, 0U, true, true, false});
                const auto base = f.run(3);
                for (int form = 4; form <= 66; ++form) {
                    if (!same_outputs(base, f.run(form), "GSQ FP16 schedule full batches")) { ++failed; std::printf("  form %d differs\n", form); }
                    ++cases;
                }
            }
            std::printf("GSQ FP16 schedules: %u cases\ngsq: %u failed\n", cases, failed);
            return failed ? 1 : 0;
        }
        if (mode == "bench") {
            const int reps = argc > 2 ? std::atoi(argv[2]) : 10;
            for (unsigned T : {256U, 512U, 1024U, 2048U})
                for (const unsigned g : {12U}) for (const unsigned d : {7U}) {
                    const std::string name = "E512 T" + std::to_string(T) + (g == 12U ? " q4k" : " q5k") + (d == 7U ? "/q51" : "/q80");
                    bench({name.c_str(), 512U, T, g, d, 0U, false, false, false}, reps);
                    const std::string skew = name + " skew";
                    bench({skew.c_str(), 512U, T, g, d, 0U, false, true, false}, reps);
                }
            for (const unsigned g : {13U}) for (const unsigned d : {8U}) {
                const std::string name = std::string("E512 T1024 ") + (g == 12U ? "q4k" : "q5k") + (d == 7U ? "/q51" : "/q80");
                bench({name.c_str(), 512U, 1024U, g, d, 0U, false, false, false}, reps);
            }
            return 0;
        }
        std::vector<Case> cases;
        static std::vector<std::string> keep;
        keep.reserve(64);
        for (const unsigned g : {12U, 13U})
            for (const unsigned d : {7U, 8U})
                for (const unsigned flavor : {0U, 1U, 2U, 3U}) {
                    const std::string fmt = std::string(g == 12U ? "q4k" : "q5k") + (d == 7U ? "/q51" : "/q80") +
                        (flavor == 1U ? " edge" : flavor == 2U ? " nonfinite" : flavor == 3U ? " nonfinite-down" : "");
                    keep.push_back("E512 T256 " + fmt); cases.push_back({keep.back().c_str(), 512U, 256U, g, d, flavor, false, false, false});
                    keep.push_back("E64 T128 dup " + fmt); cases.push_back({keep.back().c_str(), 64U, 128U, g, d, flavor, true, false, false});
                    keep.push_back("E40 T48 skew " + fmt); cases.push_back({keep.back().c_str(), 40U, 48U, g, d, flavor, false, true, false});
                }
        keep.push_back("E512 T1024 q4k/q51 skew"); cases.push_back({keep.back().c_str(), 512U, 1024U, 12U, 7U, 0U, false, true, false});
        keep.push_back("E512 T2048 q4k/q51"); cases.push_back({keep.back().c_str(), 512U, 2048U, 12U, 7U, 0U, false, false, false});
        keep.push_back("E512 T2048 q5k/q80 edge skew"); cases.push_back({keep.back().c_str(), 512U, 2048U, 13U, 8U, 1U, false, true, false});
        keep.push_back("E16 T16 q4k/q51 edge"); cases.push_back({keep.back().c_str(), 16U, 16U, 12U, 7U, 1U, false, false, false});
        keep.push_back("E10 T24 dup q5k/q80"); cases.push_back({keep.back().c_str(), 10U, 24U, 13U, 8U, 0U, true, true, false});
        keep.push_back("E512 T64 q4k/q51 nan input"); cases.push_back({keep.back().c_str(), 512U, 64U, 12U, 7U, 0U, false, false, true});
        unsigned failed = 0;
        for (const auto& c : cases) failed += compare(c) ? 0U : 1U;
        std::printf("%zu cases, %u failed\n", cases.size(), failed);
        return failed ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
