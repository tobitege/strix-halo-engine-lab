#include "strix/hip_gdn_launcher.hpp"
#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/hip_shared_expert_launcher.hpp"
#include "strix/q6p_registry.hpp"
#include "strix/env_flags.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// q8_rows engine sites against their R1 kernels, T = 1..8 and 16/32/64: bitwise payloads
// (real and edge shapes, edge values), Q6P32 copies against their Q8_0 tensors (exact and
// dense A8 forms), A8 error against the exact path, and with STRIX_Q8_ROWS_BENCH=1
// warm/cold timings at the model shapes, plus weight GB/s of T = 8/16/32/64 at the
// prefill shapes. STRIX_Q8_ROWS_WIDE_ONLY=1 runs only the T = 16/32/64 checks (and, with
// STRIX_Q8_ROWS_BENCH_ONLY=1, only the benchmark).
namespace {
using namespace strix;
void require(bool value, const std::string& message) { if (!value) throw std::runtime_error(message); }
void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
}
struct Buffer {
    std::uintptr_t address = 0; std::uint64_t bytes = 0;
    explicit Buffer(std::uint64_t size) : bytes(size) {
        void* p = nullptr; check(hipMalloc(&p, size), "allocate fixture buffer"); address = reinterpret_cast<std::uintptr_t>(p);
    }
    ~Buffer() { if (hipFree(reinterpret_cast<void*>(address)) != hipSuccess) std::_Exit(3); }
    Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<std::uint8_t>& data) const {
        require(data.size() == bytes, "fixture upload size"); check(hipMemcpy(reinterpret_cast<void*>(address), data.data(), bytes, hipMemcpyHostToDevice), "upload");
    }
    std::vector<std::uint8_t> download() const {
        std::vector<std::uint8_t> data(bytes); check(hipMemcpy(data.data(), reinterpret_cast<void*>(address), bytes, hipMemcpyDeviceToHost), "download");
        return data;
    }
    void poison(hipStream_t stream) const { check(hipMemsetAsync(reinterpret_cast<void*>(address), 0xa5, bytes, stream), "poison"); }
};

enum class Site { gdn_input, gdn_output, qsa_input, qsa_output, shared_gate_up, shared_down };
constexpr std::array<Site, 6> sites{Site::gdn_input, Site::gdn_output, Site::qsa_input, Site::qsa_output,
    Site::shared_gate_up, Site::shared_down};
const char* name(Site s) {
    switch (s) {
    case Site::gdn_input: return "GDN qkv+z (BF16->FP32)";
    case Site::gdn_output: return "GDN out (FP32->BF16)";
    case Site::qsa_input: return "QSA q (BF16->FP32)";
    case Site::qsa_output: return "QSA out (FP32->BF16)";
    case Site::shared_gate_up: return "shared gate+up (BF16->FP32)";
    default: return "shared down (FP32->BF16, gated)";
    }
}
bool bf16_input(Site s) { return s == Site::gdn_input || s == Site::qsa_input || s == Site::shared_gate_up; }
// Model shapes (width, outputs); gate+up and qkv+z are each one launch of both matrices.
std::pair<std::uint32_t, std::uint32_t> model_shape(Site s) {
    switch (s) {
    case Site::gdn_input: return {2560U, 16384U};
    case Site::gdn_output: return {6144U, 2560U};
    case Site::qsa_input: return {2560U, 13952U};
    case Site::qsa_output: return {6144U, 2560U};
    case Site::shared_gate_up: return {2560U, 1280U};
    default: return {640U, 2560U};
    }
}
enum class Mode { rows, rows_a8, row_loop, shared_a8_reference, shared_a8_narrow };

std::uint16_t bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}
// Flavors: 0 smooth finite, 1 edge scales/codes/BF16 bits (subnormal, signed
// zero, extremes), 2 alternating +-127/-128 codes (31/-32 for six-bit codes),
// 3 sparse NaN/Inf activations.
std::vector<std::uint8_t> weights_bytes(std::uint32_t width, std::uint32_t outputs, unsigned flavor, bool six_bit = false) {
    const auto blocks = static_cast<std::size_t>(width / 32U) * outputs;
    std::vector<std::uint8_t> result(blocks * 34U);
    constexpr std::array<std::uint16_t, 8> scales{0x0000, 0x8000, 0x0001, 0x8001, 0x0400, 0x8400, 0x2000, 0xa000};
    for (std::size_t block = 0; block < blocks; ++block) {
        const auto scale = flavor == 1U ? scales[block % scales.size()] : static_cast<std::uint16_t>(0x2400U + (block % 3U) * 0x400U);
        result[block * 34U] = static_cast<std::uint8_t>(scale & 0xffU);
        result[block * 34U + 1U] = static_cast<std::uint8_t>(scale >> 8U);
        for (std::size_t lane = 0; lane < 32U; ++lane) {
            const int code = flavor == 2U ? (lane % 2U ? (six_bit ? -32 : -128) : (six_bit ? 31 : 127)) :
                six_bit && flavor == 1U ? static_cast<int>((block * 13U + lane * 5U) % 64U) - 32 :
                static_cast<int>((block * 7U + lane * 3U) % 29U) - 14;
            result[block * 34U + 2U + lane] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
        }
    }
    return result;
}
std::vector<std::uint8_t> input_bytes(std::uint32_t width, unsigned row, bool bf, unsigned flavor) {
    std::vector<std::uint8_t> result(static_cast<std::size_t>(width) * (bf ? 2U : 4U));
    constexpr std::array<std::uint16_t, 8> edges{0, 0x8000, 1, 0x8001, 0x0080, 0x8080, 0x3e80, 0xbe80};
    for (std::uint32_t i = 0; i < width; ++i) {
        float value = static_cast<float>(static_cast<int>((i * 5U + row * 11U) % 31U) - 15) * 0.015625F * static_cast<float>(row + 1U);
        std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        if (flavor == 1U) bits = static_cast<std::uint32_t>(edges[(i + row) % edges.size()]) << 16U;
        if (flavor == 2U) bits = std::bit_cast<std::uint32_t>(i % 2U ? -0.125F : 0.125F);
        if (flavor == 3U && i % 97U == row) bits = (i / 97U) % 3U == 0U ? 0x7f800000U : (i / 97U) % 3U == 1U ? 0xff800000U : 0x7fc01234U + row;
        if (!bf && flavor == 0U) bits ^= (i * 2654435761U + row) & 0x7fffU;  // FP32 activations use the low mantissa
        if (bf) { const auto h = static_cast<std::uint16_t>(flavor ? bits >> 16U : bf16(std::bit_cast<float>(bits))); std::memcpy(&result[2U * i], &h, 2U); }
        else std::memcpy(&result[4U * i], &bits, 4U);
    }
    return result;
}

struct Fixture {
    Site site; unsigned rows; std::uint32_t width, outputs;
    Buffer weights;
    std::vector<std::unique_ptr<Buffer>> input, output, gate;
    Fixture(Site s, unsigned count, std::uint32_t w, std::uint32_t o)
        : site(s), rows(count), width(w), outputs(o), weights(static_cast<std::uint64_t>(w / 32U) * o * 34U) {
        const bool bf = bf16_input(s);
        for (unsigned r = 0; r < rows; ++r) {
            input.push_back(std::make_unique<Buffer>(static_cast<std::uint64_t>(w) * (bf ? 2U : 4U)));
            output.push_back(std::make_unique<Buffer>(static_cast<std::uint64_t>(o) * (bf ? 4U : 2U)));
            if (s == Site::shared_down) gate.push_back(std::make_unique<Buffer>(4U));
        }
    }
    void prepare(unsigned flavor, bool six_bit = false) const {
        weights.upload(weights_bytes(width, outputs, flavor, six_bit));
        for (unsigned r = 0; r < rows; ++r) {
            input[r]->upload(input_bytes(width, r, bf16_input(site), flavor));
            if (site == Site::shared_down) {
                std::vector<std::uint8_t> g(4); const float value = 0.75F - 0.125F * static_cast<float>(r);
                std::memcpy(g.data(), &value, 4U); gate[r]->upload(g);
            }
        }
    }
    void enqueue(Mode mode, hipStream_t stream) const {
        if (site == Site::gdn_input || site == Site::gdn_output) {
            std::vector<GdnKnownWindowProjectionView> x, y;
            for (unsigned r = 0; r < rows; ++r) { x.push_back({input[r]->address, input[r]->bytes}); y.push_back({output[r]->address, output[r]->bytes}); }
            gdn_known_window_rows_projection_probe(site == Site::gdn_input ? GdnKnownWindowProjectionDirection::bf16_to_fp32 :
                GdnKnownWindowProjectionDirection::fp32_to_bf16, x, {weights.address, weights.bytes}, y, width, outputs, stream,
                mode == Mode::rows ? GdnKnownWindowRowsProbeMode::rows : mode == Mode::rows_a8 ? GdnKnownWindowRowsProbeMode::rows_a8 :
                GdnKnownWindowRowsProbeMode::row_loop);
        } else if (site == Site::qsa_input || site == Site::qsa_output) {
            std::vector<HipQsaInputProjectionView> x, y;
            for (unsigned r = 0; r < rows; ++r) { x.push_back({input[r]->address, input[r]->bytes}); y.push_back({output[r]->address, output[r]->bytes}); }
            launch_hip_qsa_rows_projection_probe(site == Site::qsa_input ? HipQsaRowsProjection::input : HipQsaRowsProjection::output,
                mode == Mode::rows ? HipQsaRowsProbeMode::rows : mode == Mode::rows_a8 ? HipQsaRowsProbeMode::rows_a8 : HipQsaRowsProbeMode::row_loop,
                x, {weights.address, weights.bytes}, y, width, outputs, stream);
        } else {
            std::vector<HipSharedQ8ProjectionView> x, y, g;
            for (unsigned r = 0; r < rows; ++r) {
                x.push_back({input[r]->address, input[r]->bytes}); y.push_back({output[r]->address, output[r]->bytes});
                if (site == Site::shared_down) g.push_back({gate[r]->address, gate[r]->bytes});
            }
            launch_hip_shared_rows_projection_probe(site == Site::shared_down ? HipSharedQ8ProjectionRole::down : HipSharedQ8ProjectionRole::gate_up,
                mode == Mode::rows ? HipSharedRowsProbeMode::rows : mode == Mode::rows_a8 ? HipSharedRowsProbeMode::rows_a8 :
                mode == Mode::shared_a8_reference ? HipSharedRowsProbeMode::rows_a8_reference :
                mode == Mode::shared_a8_narrow ? HipSharedRowsProbeMode::rows_a8_narrow : HipSharedRowsProbeMode::row_loop,
                x, {weights.address, weights.bytes}, g, y, width, outputs, stream);
        }
    }
    std::vector<std::vector<std::uint8_t>> run(Mode mode, hipStream_t stream) const {
        for (const auto& y : output) y->poison(stream);
        enqueue(mode, stream);
        check(hipStreamSynchronize(stream), "complete projection");
        std::vector<std::vector<std::uint8_t>> result;
        for (const auto& y : output) result.push_back(y->download());
        return result;
    }
    float value(const std::vector<std::uint8_t>& bytes, std::uint32_t i) const {
        if (bf16_input(site)) { float v; std::memcpy(&v, &bytes[4U * i], 4U); return v; }
        std::uint16_t h; std::memcpy(&h, &bytes[2U * i], 2U); return std::bit_cast<float>(static_cast<std::uint32_t>(h) << 16U);
    }
};

void exact_case(hipStream_t stream, Site site, unsigned rows, std::uint32_t width, std::uint32_t outputs, unsigned flavors,
    std::uint32_t& cases) {
    Fixture f(site, rows, width, outputs);
    for (unsigned flavor = 0; flavor < flavors; ++flavor) {
        f.prepare(flavor);
        const auto expected = f.run(Mode::row_loop, stream);
        for (unsigned repeat = 0; repeat < 2U; ++repeat) {
            const auto actual = f.run(Mode::rows, stream);
            for (unsigned r = 0; r < rows; ++r)
                for (std::size_t i = 0; i < expected[r].size(); i += 2U)
                    if (std::memcmp(&expected[r][i], &actual[r][i], 2U)) {
                        const unsigned size = bf16_input(site) ? 4U : 2U;
                        const auto at = i / size * size;
                        std::uint32_t e = 0, a = 0;
                        std::memcpy(&e, &expected[r][at], size); std::memcpy(&a, &actual[r][at], size);
                        throw std::runtime_error(std::string(name(site)) + " rows=" + std::to_string(rows) + " width=" +
                            std::to_string(width) + " outputs=" + std::to_string(outputs) + " flavor=" + std::to_string(flavor) +
                            ": row " + std::to_string(r) + " output " + std::to_string(at / size) + " R1 bits " + std::to_string(e) +
                            " q8_rows bits " + std::to_string(a));
                    }
        }
        if (!flavor && rows > 1U) require(expected[0] != expected[1], "degenerate row fixture");
        ++cases;
    }
}
// Six-bit Q8_0 weights through their registered Q6P32 copy: the same bits as the
// Q8_0 rows kernel. The Q8_0 tensor is poisoned before the Q6P32 run, so an
// unregistered fallback cannot pass.
// Q6P32 copy registered and the Q8_0 tensor poisoned: every launch that reads the copy (a registry hit)
// must give the Q8_0 bits of its mode (exact or dense A8); launches that keep the Q8_0 tensor (exact VOPD
// forms) count in `kept`.
void q6_case(hipStream_t stream, Site site, unsigned rows, std::uint32_t width, std::uint32_t outputs, unsigned flavors,
    std::uint32_t& cases, Mode mode = Mode::rows, std::uint32_t* kept = nullptr) {
    Fixture f(site, rows, width, outputs);
    Buffer q6(q6p::q6p32_bytes(outputs, width));
    for (unsigned flavor = 0; flavor < flavors; ++flavor) {
        f.prepare(flavor, true);
        const auto expected = f.run(mode, stream);
        const auto q8 = weights_bytes(width, outputs, flavor, true);
        std::vector<std::uint8_t> packed(q6.bytes), back(q8.size());
        require(q6p::to_q6p32(q8.data(), outputs, width, packed.data()), "six-bit fixture codes outside Q6P32");
        q6p::to_q8_0(packed.data(), outputs, width, back.data());
        require(back == q8, "Q6P32 host round trip");
        q6.upload(packed);
        f.weights.poison(stream);
        q6p::register_tensor(reinterpret_cast<const void*>(f.weights.address),
            {reinterpret_cast<const unsigned char*>(q6.address), outputs, width, q6p::Kind::q6p32});
        const auto before = q6p::hits();
        const auto actual = f.run(mode, stream);
        const bool read = q6p::hits() != before;
        q6p::clear();
        if (!read) {
            require(kept != nullptr, std::string(name(site)) + " rows=" + std::to_string(rows) + ": Q6P32 copy not read");
            ++*kept;
            continue;
        }
        for (unsigned r = 0; r < rows; ++r)
            require(expected[r] == actual[r], std::string(name(site)) + (mode == Mode::rows_a8 ? " dense A8" : "") + " Q6P32 rows=" + std::to_string(rows) + " width=" +
                std::to_string(width) + " outputs=" + std::to_string(outputs) + " flavor=" + std::to_string(flavor) +
                ": row " + std::to_string(r) + " differs from Q8_0");
        ++cases;
    }
}
// A8 max |a8 - exact| relative to max |exact| over finite outputs (flavor 0).
double a8_error(hipStream_t stream, Site site, unsigned rows, std::uint32_t width, std::uint32_t outputs, double& absolute) {
    Fixture f(site, rows, width, outputs);
    f.prepare(0);
    const auto exact = f.run(Mode::rows, stream), a8 = f.run(Mode::rows_a8, stream);
    double scale = 0.0; absolute = 0.0;
    for (unsigned r = 0; r < rows; ++r)
        for (std::uint32_t i = 0; i < outputs; ++i) {
            const double e = f.value(exact[r], i), q = f.value(a8[r], i);
            require(std::isfinite(e) && std::isfinite(q), "A8 fixture produced a non-finite output");
            scale = std::max(scale, std::abs(e)); absolute = std::max(absolute, std::abs(q - e));
        }
    return absolute / scale;
}

__global__ void flush_cache(const uint4* data, std::size_t count, unsigned* sink) {
    uint4 acc{};
    for (std::size_t i = blockIdx.x * 256U + threadIdx.x; i < count; i += static_cast<std::size_t>(gridDim.x) * 256U) {
        const auto v = data[i]; acc.x ^= v.x; acc.y ^= v.w;
    }
    if (acc.x == 0x9e3779b9U && acc.y == 1U) *sink = acc.x;
}
float median_us(const Fixture& f, Mode mode, bool cold, const Buffer& flush, hipStream_t stream) {
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "event"); check(hipEventCreate(&stop), "event");
    std::vector<float> samples;
    for (unsigned it = 0; it < 28U; ++it) {
        if (cold) {
            hipLaunchKernelGGL(flush_cache, dim3(1024), dim3(256), 0, stream, reinterpret_cast<const uint4*>(flush.address),
                flush.bytes / 16U, reinterpret_cast<unsigned*>(flush.address));
            check(hipStreamSynchronize(stream), "flush caches");
        }
        check(hipEventRecord(start, stream), "record"); f.enqueue(mode, stream); check(hipEventRecord(stop, stream), "record");
        check(hipEventSynchronize(stop), "sync");
        float ms = 0; check(hipEventElapsedTime(&ms, start, stop), "elapsed");
        if (it >= 4U) samples.push_back(ms * 1000.0F);
    }
    check(hipEventDestroy(start), "event"); check(hipEventDestroy(stop), "event");
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}
void shared_down_r8_cases(hipStream_t stream) {
    unsigned cases = 0;
    // Includes partial output waves, unusual Q8 scales/codes, nonfinite inputs,
    // and the production shape. Compare full BF16 payloads, never a tolerance.
    for (const unsigned outputs : {1U, 7U, 9U, 33U, 2560U, 2563U}) {
        Fixture f(Site::shared_down, 8U, 640U, outputs);
        for (unsigned flavor = 0; flavor < 4U; ++flavor) {
            f.prepare(flavor);
            const auto reference = f.run(Mode::shared_a8_reference, stream);
            for (unsigned repeat = 0; repeat < 3U; ++repeat)
                require(f.run(Mode::shared_a8_narrow, stream) == reference, "shared down R8 A8 organization changed output bits");
            ++cases;
        }
    }
    std::cout << "shared down R8 A8 narrow bitwise PASS " << cases << " cases x3 repeats\n";
    if (std::getenv("STRIX_Q8_ROWS_BENCH")) {
        Buffer flush(256ULL << 20U);
        check(hipMemset(reinterpret_cast<void*>(flush.address), 1, flush.bytes), "init flush");
        Fixture f(Site::shared_down, 8U, 640U, 2560U); f.prepare(0);
        for (const bool cold : {false, true})
            for (unsigned pair = 0; pair < 3U; ++pair) {
                const auto a = pair & 1U ? Mode::shared_a8_narrow : Mode::shared_a8_reference;
                const auto b = pair & 1U ? Mode::shared_a8_reference : Mode::shared_a8_narrow;
                const auto first = median_us(f, a, cold, flush, stream), second = median_us(f, b, cold, flush, stream);
                std::cout << "shared-down-r8 " << (cold ? "cold" : "warm") << " pair=" << pair
                          << " reference_us=" << (pair & 1U ? second : first)
                          << " narrow_us=" << (pair & 1U ? first : second) << '\n';
            }
    }
}
} // namespace

// Prefill shapes of the wide (T > 8) windows: GDN qkv+z 12288 + 6144, QSA q 12288 + 512 + 512.
std::pair<std::uint32_t, std::uint32_t> prefill_shape(Site s) {
    switch (s) {
    case Site::gdn_input: return {2560U, 18432U};
    case Site::qsa_input: return {2560U, 13312U};
    default: return model_shape(s);
    }
}
void wide_cases(hipStream_t stream, std::uint32_t& cases) {
    for (const auto site : sites)
        for (const unsigned rows : {16U, 32U, 64U}) {
            const auto [width, outputs] = prefill_shape(site);
            exact_case(stream, site, rows, width, outputs, rows == 64U ? 2U : 4U, cases);
            // Both model widths, output counts that are not multiples of the per-wave outputs or
            // of a block's output groups, and partial-chunk / sub-wave widths.
            for (const std::uint32_t w : {2560U, 6144U}) for (const std::uint32_t o : {1U, 7U, 9U, 17U, 1001U})
                exact_case(stream, site, rows, w, o, 2U, cases);
            for (const std::uint32_t w : {32U, 96U, 288U, 640U}) for (const std::uint32_t o : {3U, 33U})
                exact_case(stream, site, rows, w, o, 4U, cases);
        }
}
void wide_bench(hipStream_t stream) {
    Buffer flush(256ULL << 20U);
    check(hipMemset(reinterpret_cast<void*>(flush.address), 1, flush.bytes), "init flush");
    // STRIX_Q8_ROWS_BENCH_ROWS=64 restricts the row counts, STRIX_Q8_ROWS_BENCH_COLD=1 skips warm runs.
    std::vector<unsigned> counts{8U, 16U, 32U, 64U};
    if (const char* v = std::getenv("STRIX_Q8_ROWS_BENCH_ROWS")) counts = {static_cast<unsigned>(std::atoi(v))};
    const bool cold_only = std::getenv("STRIX_Q8_ROWS_BENCH_COLD") != nullptr;
    for (const bool cold : {false, true})
        for (const auto site : sites) {
            if (cold_only && !cold) continue;
            const auto [width, outputs] = prefill_shape(site);
            const double weight_bytes = static_cast<double>(width / 32U) * outputs * 34.0;
            Fixture warmup(site, 1U, width, outputs); warmup.prepare(0);
            static_cast<void>(median_us(warmup, Mode::rows, cold, flush, stream));
            std::cout << (cold ? "cold " : "warm ") << name(site) << " " << width << "->" << outputs << " ("
                      << weight_bytes / 1e6 << " MB of Q8 weights)\n";
            for (const unsigned rows : counts) {
                Fixture f(site, rows, width, outputs); f.prepare(0);
                const auto us = median_us(f, Mode::rows, cold, flush, stream);
                std::cout << "  T=" << rows << " " << us << " us  " << weight_bytes / (us * 1e3) << " GB/s weights  "
                          << us / static_cast<float>(rows) << " us/row\n";
            }
        }
}

int main() try {
    hipStream_t stream = nullptr;
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create stream");
    std::uint32_t cases = 0;
    shared_down_r8_cases(stream);
    if (std::getenv("STRIX_SHARED_DOWN_R8_ONLY")) {
        check(hipStreamDestroy(stream), "destroy stream");
        return 0;
    }
    // STRIX_Q8_ROWS_WIDE_ONLY=1: only the T = 16/32/64 checks (and their benchmark).
    if (std::getenv("STRIX_Q8_ROWS_WIDE_ONLY")) {
        if (!std::getenv("STRIX_Q8_ROWS_BENCH_ONLY")) {
            wide_cases(stream, cases);
            std::cout << "q8_rows T=16/32/64 payloads equal the R1 kernels bitwise: " << cases << " cases\n";
        }
        if (std::getenv("STRIX_Q8_ROWS_BENCH")) wide_bench(stream);
        check(hipStreamDestroy(stream), "destroy stream");
        return 0;
    }
    // STRIX_Q8_ROWS_GEMV_FIXTURE=1: small-row decode fixture only. Per GDN shape and row count: the
    // FNV-1a hash of the R1 row loop and of the rows launch (equal = bitwise), warm and cold median
    // time, and the bytes of weights and of one copy of the rows.
    if (strix::envcfg::flag("STRIX_Q8_ROWS_GEMV_FIXTURE", false)) {
        Buffer flush(256ULL << 20U);
        check(hipMemset(reinterpret_cast<void*>(flush.address), 1, flush.bytes), "init flush");
        const auto hash = [](const std::vector<std::vector<std::uint8_t>>& rows) {
            std::uint64_t h = 1469598103934665603ULL;
            for (const auto& row : rows) for (const auto byte : row) h = (h ^ byte) * 1099511628211ULL;
            return h;
        };
        for (const auto site : {Site::gdn_input, Site::gdn_output}) {
            const auto [width, outputs] = model_shape(site);
            Fixture warmup(site, 1U, width, outputs); warmup.prepare(0);
            static_cast<void>(median_us(warmup, Mode::rows, false, flush, stream));
            for (const unsigned rows : {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 16U}) {
                Fixture f(site, rows, width, outputs); f.prepare(0);
                const auto reference = hash(f.run(Mode::row_loop, stream)), actual = hash(f.run(Mode::rows, stream));
                const auto warm = median_us(f, Mode::rows, false, flush, stream), cold = median_us(f, Mode::rows, true, flush, stream);
                std::cout << "{\"site\":\"" << name(site) << "\",\"inputs\":" << width << ",\"outputs\":" << outputs << ",\"rows\":" << rows
                          << ",\"hash_r1\":\"" << std::hex << reference << "\",\"hash\":\"" << actual << std::dec << "\",\"warm_us\":" << warm
                          << ",\"cold_us\":" << cold << ",\"weight_bytes_sweep\":" << static_cast<std::uint64_t>(width / 32U) * outputs * 34U
                          << ",\"activation_bytes_copy\":" << static_cast<std::uint64_t>(rows) * width * (bf16_input(site) ? 2U : 4U) << "}" << std::endl;
            }
        }
        check(hipStreamDestroy(stream), "destroy stream");
        return 0;
    }
    // STRIX_Q8_ROWS_DECODE_BENCH=1: only cold/warm timings of the T = 1/2/4/8 window shapes.
    if (std::getenv("STRIX_Q8_ROWS_DECODE_BENCH")) {
        Buffer flush(256ULL << 20U);
        check(hipMemset(reinterpret_cast<void*>(flush.address), 1, flush.bytes), "init flush");
        for (const bool cold : {false, true})
            for (const auto site : sites) {
                const auto [width, outputs] = model_shape(site);
                Fixture warmup(site, 1U, width, outputs); warmup.prepare(0);
                static_cast<void>(median_us(warmup, Mode::rows, cold, flush, stream));
                std::cout << (cold ? "cold " : "warm ") << name(site) << " " << width << "->" << outputs << ":";
                for (const unsigned rows : {1U, 2U, 4U, 8U}) {
                    Fixture f(site, rows, width, outputs); f.prepare(0);
                    std::cout << "  T" << rows << " " << median_us(f, Mode::rows, cold, flush, stream);
                }
                std::cout << " us\n";
            }
        check(hipStreamDestroy(stream), "destroy stream");
        return 0;
    }
    for (const auto site : sites)
        for (const unsigned rows : {1U, 2U, 4U, 8U}) {
            const auto [width, outputs] = model_shape(site);
            exact_case(stream, site, rows, width, outputs, 4U, cases);
            // Tails: partial last chunk / fewer leaves than a wave, odd output groups.
            for (const std::uint32_t w : {32U, 96U, 288U, 640U}) for (const std::uint32_t o : {1U, 7U, 9U, 17U, 33U})
                exact_case(stream, site, rows, w, o, 4U, cases);
            // BF16 rows past the LDS budget (direct), and staged with a partial last chunk and group tail.
            if (bf16_input(site) && rows >= 4U) {
                exact_case(stream, site, rows, 6144U, 9U, 2U, cases);
                exact_case(stream, site, rows, 2400U, 5003U, 2U, cases);
            }
        }
    // The deferred GDN window also runs 3/5/6/7 rows (padded reduction trees).
    for (const auto site : {Site::gdn_input, Site::gdn_output})
        for (const unsigned rows : {3U, 5U, 6U, 7U}) {
            const auto [width, outputs] = model_shape(site);
            exact_case(stream, site, rows, width, outputs, 4U, cases);
            for (const std::uint32_t w : {96U, 640U}) for (const std::uint32_t o : {7U, 33U}) exact_case(stream, site, rows, w, o, 2U, cases);
        }
    std::cout << "q8_rows exact payloads equal the R1 kernels bitwise: " << cases << " cases\n";
    std::uint32_t wide = 0;
    wide_cases(stream, wide);
    std::cout << "q8_rows T=16/32/64 payloads equal the R1 kernels bitwise: " << wide << " cases\n";
    std::uint32_t q6_cases = 0, q6_kept = 0, q6_a8 = 0;
    for (const auto site : sites)
        for (const unsigned rows : {1U, 2U, 3U, 4U, 8U}) {
            if (rows == 3U && site != Site::gdn_input && site != Site::gdn_output) continue;
            const auto [width, outputs] = model_shape(site);
            q6_case(stream, site, rows, width, outputs, 4U, q6_cases, Mode::rows, &q6_kept);
            for (const std::uint32_t w : {32U, 96U, 288U, 640U}) for (const std::uint32_t o : {1U, 7U, 33U})
                q6_case(stream, site, rows, w, o, 3U, q6_cases, Mode::rows, &q6_kept);
            if (bf16_input(site) && rows >= 4U) q6_case(stream, site, rows, 2400U, 5003U, 2U, q6_cases, Mode::rows, &q6_kept);
            if (rows == 3U) continue;
            // Dense A8 (widths of whole 64-feature block pairs): LDS-staged and slot forms.
            q6_case(stream, site, rows, width, outputs, 3U, q6_a8, Mode::rows_a8);
            for (const std::uint32_t w : {64U, 320U, 640U}) for (const std::uint32_t o : {1U, 7U, 33U})
                q6_case(stream, site, rows, w, o, 2U, q6_a8, Mode::rows_a8);
            if (bf16_input(site)) q6_case(stream, site, rows, 2432U, 5003U, 2U, q6_a8, Mode::rows_a8);
        }
    std::cout << "q8_rows Q6P32 payloads equal their Q8_0 tensors bitwise: " << q6_cases << " exact cases (" << q6_kept
              << " launches kept the Q8_0 tensor), " << q6_a8 << " dense A8 cases\n";
    for (const auto site : sites)
        for (const unsigned rows : {1U, 2U, 4U, 8U}) {
            const auto [width, outputs] = model_shape(site);
            double absolute = 0.0;
            const auto relative = a8_error(stream, site, rows, width, outputs, absolute);
            std::cout << "A8 " << name(site) << " T=" << rows << ": max abs err " << absolute << " (" << relative * 100.0
                      << "% of max |out|)\n";
            require(relative < 0.05, "A8 error exceeds 5% of the output range");
        }
    if (std::getenv("STRIX_Q8_ROWS_BENCH")) {
        Buffer flush(256ULL << 20U);
        check(hipMemset(reinterpret_cast<void*>(flush.address), 1, flush.bytes), "init flush");
        for (const bool cold : {false, true})
            for (const auto site : sites) {
                const auto [width, outputs] = model_shape(site);
                Fixture warmup(site, 1U, width, outputs); warmup.prepare(0);
                static_cast<void>(median_us(warmup, Mode::row_loop, cold, flush, stream));
                std::cout << (cold ? "cold " : "warm ") << name(site) << " " << width << "->" << outputs << " (us, median of 24)\n";
                for (const unsigned rows : {1U, 2U, 4U, 8U}) {
                    Fixture f(site, rows, width, outputs); f.prepare(0);
                    std::cout << "  T=" << rows << " R1 x T " << median_us(f, Mode::row_loop, cold, flush, stream)
                              << "  rows " << median_us(f, Mode::rows, cold, flush, stream)
                              << "  rows-A8 " << median_us(f, Mode::rows_a8, cold, flush, stream);
                    Buffer q6(q6p::q6p32_bytes(outputs, width));
                    std::vector<std::uint8_t> packed(q6.bytes);
                    f.prepare(0, true);
                    require(q6p::to_q6p32(weights_bytes(width, outputs, 0, true).data(), outputs, width, packed.data()), "bench Q6P32");
                    q6.upload(packed);
                    q6p::register_tensor(reinterpret_cast<const void*>(f.weights.address),
                        {reinterpret_cast<const unsigned char*>(q6.address), outputs, width, q6p::Kind::q6p32});
                    std::cout << "  rows-Q6P32 " << median_us(f, Mode::rows, cold, flush, stream) << '\n';
                    q6p::clear();
                }
            }
        wide_bench(stream);
    }
    check(hipStreamDestroy(stream), "destroy stream");
    return 0;
} catch (const std::exception& error) {
    std::cerr << "q8_rows projection fixture failed: " << error.what() << '\n';
    return 1;
}
