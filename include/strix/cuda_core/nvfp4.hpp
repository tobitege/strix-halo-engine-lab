#pragma once
// TURBO weights, NVFP4 on sm_120 (Home: shared, backend core CUDA; arch specialisation by capability: needs sm_120a
// cvt.e2m1x2 / mma.sync f16, see src/cuda_core/dense/nvfp4.cu).
//
// Format = standard NVFP4: values are e2m1 (4 bit: s e1 e0 m; magnitudes 0 .5 1 1.5 2 3 4 6), one e4m3 scale per 16
// consecutive INPUT features, one FP32 global scale per tensor. Dequantised weight = e2m1 * e4m3 * gscale.
// RTN converter: gscale = amax(tensor) / (6 * 448) (so the e4m3 scale never saturates; 1 if the tensor is all zero),
// scale_g = e4m3_rn(amax_g / (6 * gscale)), q = e2m1_rn_even(w / (decode(scale_g) * gscale)), all FP32 IEEE.
//
// PHYSICAL (tile) LAYOUT of a [out][in] tensor (requires out % 16 == 0, in % 128 == 0):
//   record(m, c) = 1152 bytes for output tile m (16 output rows 16m..16m+15) and K chunk c (128 inputs 128c..128c+127);
//   records are ordered m-major: byte offset (m * (in/128) + c) * 1152, 128 B aligned.
//   data   [0,1024): [h 0..1][lane 0..31][16 B]. Lane = 4 g + t (g = lane >> 2 = row in tile modulo 8, t = lane & 3) of
//          output row 8h + g. The 16 B = 4 u32 words; word i, byte b holds group j = 2i + (b >> 1) (inputs 128c + 16j ..),
//          pair p = b & 1: p 0 = inputs 16j+2t, 16j+2t+1; p 1 = inputs 16j+2t+8, 16j+2t+9 (the mma m16n8k16 A fragment of
//          lane (g,t)); low nibble = even input, high nibble = odd input. One warp reads a record as 2 fully coalesced
//          512 B vector loads.
//   scales [1024,1152): byte 16 g + 8 h + j = e4m3 scale of (row 8h+g, group j); lane (g,t) reads one 16 B vector.
// Bytes per weight: 0.5 + 1/16 = 0.5625 (4.5 bit): 27B trunk+head = 14.4 GB per pass.
//
// Decode GEMV (rows 1..32): mma.sync m16n8k16 f16 x f16 -> f32. A = dequantised weight (e2m1 * e4m3 in f16, EXACT), B =
// activation rounded to f16 (RN). Each output sums its 16-wide k steps in ascending order inside a warp, then the split-K
// warp partials in ascending order (the split count depends only on (out, in), never on rows), then * gscale, then the
// optional residual add: rows 1..32 are bitwise identical to rows 1. Activations must satisfy |x| < 65504.
#include "strix/cuda_core/types.hpp"
#include <vector>

namespace strix::cuda_core {

// One device/stream owner. Growth retains old addresses used by captured graphs.
// Destroy only after stream completion and destruction of all referring graphs.
class Nvfp4Workspace {
    std::vector<UniqueHandle<void*, cudaFree>> allocations_;
    std::size_t bytes_ = 0;
public:
    cudaError_t reserve(std::size_t bytes) {
        if (bytes <= bytes_) return cudaSuccess;
        UniqueHandle<void*, cudaFree> allocation;
        const auto error = cudaMalloc(allocation.put(), bytes);
        if (error != cudaSuccess) return error;
        allocations_.push_back(std::move(allocation));
        bytes_ = bytes;
        return cudaSuccess;
    }
    float* data() const noexcept { return allocations_.empty() ? nullptr : static_cast<float*>(allocations_.back().get()); }
};

constexpr std::uint32_t kNvfp4MaxRows = 32;
constexpr std::uint32_t kNvfp4RecordBytes = 1152;

[[nodiscard]] constexpr bool nvfp4_shape_ok(std::uint32_t out, std::uint32_t in) { return out % 16 == 0 && in % 128 == 0; }
[[nodiscard]] constexpr std::size_t nvfp4_bytes(std::uint32_t out, std::uint32_t in) {
    return static_cast<std::size_t>(out / 16) * (in / 128) * kNvfp4RecordBytes;
}

// BF16 [out][in] (row-major, device) -> tile layout. Two launches (amax, convert); keeps `gscale` on the host (one sync).
cudaError_t nvfp4_quantize_bf16(const void* bf16, std::uint32_t out, std::uint32_t in, void* dst, float* gscale,
                                cudaStream_t stream);
// Test hook: same conversion from FP32 [out][in].
cudaError_t nvfp4_quantize_f32(const float* src, std::uint32_t out, std::uint32_t in, void* dst, float* gscale,
                               cudaStream_t stream);

// y = gscale * (W x) (+ residual), W = nvfp4 DenseWeight (w.gscale), x FP32 [rows][in], y FP32 [rows][out], rows 1..32.
// Uses an internal grow-only device workspace (not safe for concurrent streams; call nvfp4_reserve_workspace first when
// capturing a graph).
cudaError_t nvfp4_dense_rows(const DenseWeight& w, const float* x, float* y, std::uint32_t rows, const float* residual,
                             cudaStream_t stream);
// Up to 4 GEMVs that read the SAME activations x (e.g. qkv/z/beta/alpha, or q/k/v) in one launch: same arithmetic and
// bits as separate nvfp4_dense_rows calls (per-job split-K factor), fewer launches and one shared ramp/tail.
struct Nvfp4Job {
    DenseWeight w;
    float* y;
    const float* residual;
};
cudaError_t nvfp4_dense_rows_multi(const Nvfp4Job* jobs, std::uint32_t n, const float* x, std::uint32_t rows, cudaStream_t stream);
// y = silu(gate x) * (up x), same format/shape.
cudaError_t nvfp4_gate_up_silu_rows(const DenseWeight& gate, const DenseWeight& up, const float* x, float* y,
                                    std::uint32_t rows, cudaStream_t stream);
cudaError_t nvfp4_reserve_workspace(std::size_t bytes, cudaStream_t stream = nullptr);
// Current device; caller owns the stream and must destroy graphs and synchronize first.
void nvfp4_release_workspace(cudaStream_t stream);
// Tuning/test hook: target number of blocks used to pick the split-K factor (default 1000; 0 = default).
void nvfp4_set_split_target(std::uint32_t warps);
// Split-K factor used for (out, in) (diagnostics / tests).
std::uint32_t nvfp4_split_k(std::uint32_t out, std::uint32_t in);

// Large-M prefill support (src/cuda_core/dense/dense_big.cu): expands a tile-layout tensor to f16 [out][in] (e2m1 * e4m3, EXACT,
// WITHOUT gscale). Inside every 32-input block the elements are stored in the mma k-fragment order (position 8q + 4s + 2p + e holds
// input 16s + 2q + 8p + e), so one 8-byte shared-memory read gives a thread's four k of a k16 step.
cudaError_t nvfp4_expand_f16(const DenseWeight& w, void* dst_f16, cudaStream_t stream);
// FAST/TURBO prefill: e4m3 [out][in] of (e2m1 * e4m3 / 8) (rounded; WITHOUT gscale), position 32t + 4j + 2p + e of a 128-input block holds
// input 16j + 8p + 2t + e (the activation quantiser of dense_big uses the same permutation).
cudaError_t nvfp4_expand_e4m3(const DenseWeight& w, void* dst, cudaStream_t stream);

// Host reference: dequantised float (e2m1 * e4m3 in FP32, WITHOUT gscale) of element (o, k) read from a tile-layout buffer.
float nvfp4_host_dequant(const unsigned char* tiles, std::uint32_t in, std::uint32_t o, std::uint32_t k);
// Host RTN reference of the converter for one 16-group (writes 16 codes e2m1 and returns the e4m3 byte); gs = global scale.
unsigned char nvfp4_host_quantize_group(const float* w16, float gs, unsigned char* codes16);
float nvfp4_host_e4m3(unsigned char b);

}  // namespace strix::cuda_core
