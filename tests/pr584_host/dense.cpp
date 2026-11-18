#include "strix/ds41_fast_profile.hpp"
#include "strix/ds41_prefill_precision.hpp"
#include "strix/lever_cache.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <vector>

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
using hipStream_t = void*;
namespace strix::weight::hip {
struct Q8F16Source {
#include "weight_supported.inc"
};
}
namespace strix::core::f16_dense_tall::q8a8 {
struct View {
    const std::uint8_t* weight{};
    const std::int8_t* codes{};
    const float* deltas{};
    float* out{};
    unsigned row_bytes{}, k_bytes{}, in{}, count{}, rows{}, x_stride{}, out_stride{}, accumulate{};
    std::uint32_t batch{}, batch_in{}, batch_out{};
    std::uint64_t batch_weight_bytes{};
};
std::vector<View> calls;
#include "dense_supported.inc"
inline constexpr unsigned kFormRule = ~0U;  // the policy's form argument (weight decode form; the same bits either way)
void enqueue(const View& v, hipStream_t, unsigned = 0U, unsigned = kFormRule) {
    calls.push_back(v);
    float sum = 0.0F;
    for (unsigned k = 0; k < v.in; ++k) sum = std::fma(static_cast<float>(v.codes[k]), v.deltas[k / 32U], sum);
    *v.out = v.accumulate ? *v.out + sum : sum;
}
#include "parts.inc"
}
namespace fast_profile = strix::ds41::fast_profile;
struct Q8A8DensePrefillView {
    const std::uint8_t* weight{};
    const std::int8_t* codes{};
    const float* deltas{};
    float* out{};
    unsigned row_width{}, width{}, k_offset{}, k_parts{}, count{}, rows{}, x_stride{}, out_stride{};
    bool accumulate{};
    bool exact{};
    std::int8_t* a8_codes{};
    float* a8_deltas{};
    std::int32_t* a8_sums{};
    unsigned a8_stride{};
};
void* as_stream(void* stream) { return stream; }
int hipGetLastError() { return 0; }
void hip_check(int, const char*) {}
#include "dense.inc"
using strix::ds41::PrefillPrecision;
using strix::ds41::PrefillPrecisionScope;
using strix::ds41::effective_fast;
int fast_dense_mode = 1;
#include "fast_dense_on.inc"
#include "dense_validation.inc"
#include "dense_groups.inc"

// Execute the production wrapper and admission logic; only the device enqueue is a recording double.
// These tests prove route selection and no-launch rejection, not device arithmetic or A8 fallback.
void grouped_test(std::string_view test) {
    namespace q = strix::core::f16_dense_tall::q8a8;
    alignas(16) std::array<std::int8_t, 256> codes{};
    std::array<float, 8> deltas{};
    std::array<std::uint8_t, 272> weights{};
    std::array<float, 2> out{-9.0F, -9.0F};
    Q8A8DensePrefillView v{};
    v.weight = weights.data(); v.codes = codes.data(); v.deltas = deltas.data(); v.out = out.data();
    v.width = v.row_width = 128; v.count = v.k_parts = 1; v.rows = 1024; v.x_stride = 256; v.out_stride = 2;
    require(valid_q8_0_a8_dense_prefill_view(v), "valid grouped fixture rejected");
    const auto reject = [&](const Q8A8DensePrefillView& bad) {
        q::calls.clear(); out.fill(-9.0F);
        require(!enqueue_q8_0_a8_fast_dense_groups(bad, 2, 136, nullptr), "invalid/disabled grouped view accepted");
        require(q::calls.empty() && out[0] == -9.0F && out[1] == -9.0F, "decline enqueued or changed output");
    };
    if (test == "zero-parts") { v.k_parts = 0; reject(v); }
    else if (test == "k-offset") { v.k_offset = 128; reject(v); }
    else if (test == "overflow") { v.row_width = 0xffffffe0U; v.k_offset = 0xffffff80U; reject(v); }
    else if (test == "batch-off") { reject(v); }
    else if (test == "invalid") {
        auto bad = v; bad.k_parts = 2; reject(bad);
        bad = v; bad.row_width = 0; bad.k_offset = 32; reject(bad);
        bad = v; bad.row_width = 129; reject(bad);
        bad = v; bad.k_offset = 1; reject(bad);
        bad = v; bad.codes += 1; reject(bad);
        bad = v; bad.a8_codes = codes.data(); reject(bad);
        bad = v; bad.rows = 0; reject(bad);
        bad = v; bad.weight = nullptr; reject(bad);
        bad = v; bad.x_stride = 128; reject(bad);
        bad = v; bad.out_stride = 1; reject(bad);
        bad = v; bad.accumulate = true; reject(bad);
    } else {
        require(test == "controls", "unknown grouped test");
        fast_dense_mode = 0; reject(v);
        fast_dense_mode = 1;
        v.exact = true; reject(v); v.exact = false;
        { const PrefillPrecisionScope exact(PrefillPrecision{false, false}); reject(v); }
        v.rows = 127; reject(v); v.rows = 1024;
        fast_dense_mode = 0;
        const PrefillPrecisionScope fast(PrefillPrecision{true, false});
        require(enqueue_q8_0_a8_fast_dense_groups(v, 2, 136, nullptr), "per-execution FAST failed to admit groups");
        require(q::calls.size() == 1, "grouped route did not enqueue exactly once");
        const auto& call = q::calls.front();
        require(call.batch == 2 && call.batch_in == 128 && call.batch_out == 1 && call.batch_weight_bytes == 136 &&
                call.row_bytes == 136 && call.k_bytes == 0 && call.x_stride == 256 && call.out_stride == 2,
                "grouped binding changed offsets/strides");
    }
}
int main(int argc, char** argv) {
    try {
#ifdef _WIN32
        require(_putenv_s("STRIX_DS41_FAST_DENSE_MIN_ROWS", "128") == 0, "set threshold");
        require(_putenv_s("STRIX_DS41_FAST_DENSE_BATCH", argc > 1 && std::string_view(argv[1]) == "batch-off" ? "0" : "1") == 0,
                "set batching");
#else
        require(setenv("STRIX_DS41_FAST_DENSE_MIN_ROWS", "128", 1) == 0, "set threshold");
        require(setenv("STRIX_DS41_FAST_DENSE_BATCH", argc > 1 && std::string_view(argv[1]) == "batch-off" ? "0" : "1", 1) == 0,
                "set batching");
#endif
        if (argc > 1) { grouped_test(argv[1]); return 0; }
        namespace q = strix::core::f16_dense_tall::q8a8;
        alignas(16) std::array<std::int8_t, 256> codes{};
        std::array<std::uint8_t, 544> weights{};
        std::array<float, 8> deltas{};
        codes[0] = codes[128] = codes[160] = 1;
        deltas[0] = 16777216.0F; deltas[4] = 1.0F; deltas[5] = -16777216.0F;
        float out = -9.0F;
        Q8A8DensePrefillView v{};
        v.weight = weights.data(); v.codes = codes.data(); v.deltas = deltas.data(); v.out = &out;
        v.width = 128; v.row_width = 512; v.k_offset = 128; v.k_parts = 2;
        v.rows = 4096; v.count = v.out_stride = 1; v.x_stride = 256;
        require(enqueue_fast_dense(v, nullptr), "FAST parts rejected");
        require(q::calls.size() == 2 && out == 1.0F, "part chain was fused across boundary");
        const auto first = q::calls[0], second = q::calls[1];
        require(first.in == 128 && second.in == 128 && first.accumulate == 0 && second.accumulate == 1,
                "independent chain or output order lost");
        require(first.k_bytes == 136 && second.k_bytes == 272 && first.row_bytes == 544 && second.row_bytes == 544 &&
                second.codes == codes.data() + 128 && second.deltas == deltas.data() + 4, "part offsets/strides wrong");
        q::calls.clear(); out = -9.0F;
        v.k_parts = 1;
        require(enqueue_fast_dense(v, nullptr), "first unfused call rejected");
        v.k_offset += v.width; v.codes += v.width; v.deltas += v.width / 32U; v.accumulate = true;
        require(enqueue_fast_dense(v, nullptr) && out == 1.0F, "unfused arithmetic differs");
        q::calls.clear(); v.width = 64; v.k_parts = 2;
        require(!enqueue_fast_dense(v, nullptr) && q::calls.empty(), "unsupported part partially launched");
        v.width = 128; v.k_parts = 1; v.accumulate = false; v.rows = 1;
        require(q8_0_a8_fast_core_supported(v), "shared core must admit short tails");
        require(enqueue_q8_0_a8_fast_core(v, nullptr) && q::calls.size() == 1, "shared core wrapper did not launch");
        const auto shared = q::calls.front();
        require(shared.weight == v.weight && shared.codes == v.codes && shared.deltas == v.deltas && shared.out == v.out &&
                shared.row_bytes == 544 && shared.k_bytes == 272 && shared.in == 128 && shared.rows == 1 &&
                shared.count == v.count && shared.x_stride == v.x_stride && shared.out_stride == v.out_stride &&
                shared.accumulate == 0, "shared view conversion lost fields");
        q::calls.clear();
        v.k_parts = 2;
        require(!enqueue_q8_0_a8_fast_core(v, nullptr), "shared core admitted split parts");
        v.k_parts = 1; v.accumulate = true;
        require(!enqueue_q8_0_a8_fast_core(v, nullptr), "shared core admitted accumulation");
        v.accumulate = false; v.a8_codes = codes.data();
        require(!enqueue_q8_0_a8_fast_core(v, nullptr) && q::calls.empty(), "shared core admitted A8 output");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
