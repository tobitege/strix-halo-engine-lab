#pragma once
#include "strix/hip_hyper_mix_launcher.hpp"
#include <hip/hip_runtime_api.h>
#include <cstdint>
namespace strix::testing {
// Test-only seam for the prefill wide HC kernels (projection + output of one HC step over `rows`
// rows; hidden 2560, 4 streams, rank 320). Row r of each view sits at base + r * stride bytes:
// normalized = 10240 normalized FP32 values followed by the 320 low values the projection writes;
// injection = 4 FP32 values (when `inject` is set); output = hidden values in the config's
// activation storage. down: Q8_0 [rank][10240]; inject: F32 [4][10240]; up: Q8_0 [4 * hidden][rank].
// The caller owns every extent through completion and drains the stream.
struct WideHyperRowsTestViews {
    float* normalized = nullptr;
    std::uint64_t normalized_stride = 0;
    float* injection = nullptr;
    std::uint64_t injection_stride = 0;
    void* output = nullptr;
    std::uint64_t output_stride = 0;
    std::uint32_t rows = 0;
    const void* down = nullptr;
    const void* inject = nullptr;
    const void* up = nullptr;
    // Optional FP16 copy of the normalized rows (row r at normalized16 + r * 10240 halves) for projection kernel 2.
    const void* normalized16 = nullptr;
    std::uint32_t normalized16_pitch = 0;  // halves; 0 = 10240
    bool normalized16_only = false;        // the output's stream mix reads the FP16 rows too
    unsigned output3_band = UINT32_MAX;     // output 7 only: 0..8, 12/16/24/32/48/64 override the read-once env; UINT32_MAX uses it
    bool output3_zero_low_rank_stride = false;  // bench-only output 7 / band 0 / projection skipped; outputs discarded
    unsigned low_band = UINT32_MAX;         // STRIX_QPF2K_HC_LOWBAND override 0..3 (projection 4, output 7, band 16); UINT32_MAX uses it
    // Exact recompute seam (projection 4, output 7, 14 or 15 only). All raw rows and
    // weights stay live and disjoint from every written extent until completion.
    const void* recompute_input = nullptr;
    std::uint64_t recompute_stride = 0;
    const float* recompute_inverse = nullptr;
    const float* recompute_norm = nullptr;
    // Optional producer receipt, published only after a successful projection launch.
    // Required for output-only calls: retain the view/stream and its live allocations,
    // and do not overwrite the consumed image. Split calls require exclusive scratch
    // use until stream completion. Any intervening projection invalidates
    // the receipt (including another view or recompute specialization).
    std::uint64_t* low_generation = nullptr;
};
// Output form 15: experimental OUT16 inverse reuse, wide-rows only; without inverses restores form14.
// Output form 14: qpf2k OUT16 (HCW2 arithmetic), paired with projection 4 or output-only 255.
void enqueue_wide_hyper_rows_test(const HipHyperMixLauncherConfig& config, const WideHyperRowsTestViews& views,
    unsigned projection_kernel, unsigned output_kernel, hipStream_t stream);
// The transposed low table the last lane-chain-output test call used (320 x 8192 FP32 values, layout per that call).
const float* wide_hyper_rows_test_low_t();
// The FP16 low copy (low16, 8192 x 320 halves) the WMMA projections write.
const void* wide_hyper_rows_test_low16();
// STRIX_QPF2K_HC folded inject (+ MoE combine) + HC norm over `rows` uniformly strided BF16 rows (hidden 2560, 4 streams,
// FP32 norm weights [4][2560]); combine when routed is set. reference = only the production prefill norm kernel
// rows_prepare_hyper_norm_kernel<bf16, HyperTileRows> on the streams (rows a multiple of 8; the caller runs the production
// combine / inject kernels first through enqueue_elementwise_prefill_rows_test), else the prefill fold kernel.
struct Qpf2kInjectNormTestViews {
    const void* hyper = nullptr; std::uint64_t hyper_stride = 0;
    void* streams = nullptr; std::uint64_t streams_stride = 0;
    const void* hidden = nullptr; std::uint64_t hidden_stride = 0;
    const void* routed = nullptr; std::uint64_t routed_stride = 0;
    const float* gates = nullptr; std::uint64_t gates_stride = 0;
    void* combined = nullptr; std::uint64_t combined_stride = 0;
    float* normalized = nullptr; std::uint64_t normalized_stride = 0;
    const float* norm = nullptr;
    std::uint32_t rows = 0;
    float* recompute_inverse = nullptr; // null = image; otherwise [rows][4]
    bool norm_only = false;            // norm existing streams, without inject/combine
};
void enqueue_qpf2k_inject_norm_test(const HipHyperMixLauncherConfig& config, const Qpf2kInjectNormTestViews& views, bool reference,
    hipStream_t stream);
// Test seam (STRIX_QPF2K_HC fixtures): the prefill inject (combine = false: first = streams, second = hidden, weights =
// gates) or MoE-combine (first = shared, second = routed) kernel exactly as the full-T launchers enqueue it, over `rows`
// (a multiple of 8) BF16 rows one stride apart.
void enqueue_elementwise_prefill_rows_test(bool combine, const void* first, std::uint64_t first_stride, const void* second,
    std::uint64_t second_stride, const float* weights, std::uint64_t weights_stride, void* output, std::uint64_t output_stride,
    std::uint32_t rows, std::uint32_t hidden_size, std::uint32_t hc_count, hipStream_t stream);
// STRIX_QPF2K_HC output coefficients of Q8_0 up weights [4 * hidden][320]: packed = the pack kernel's float4 table
// [320][hidden]; reference = the same table from the lane-chain output kernel's expression scale * (float(code ^ 0x80) - 128).
void enqueue_qpf2k_coef_test(const void* up, std::uint32_t hidden, void* packed, void* reference, hipStream_t stream);
}
