#pragma once
#include "strix/cuda_core/norms.hpp"
#include "strix/cuda_core/types.hpp"

namespace strix::cuda_core {

// Gated DeltaNet decode (one layer) with DEFERRED COMMIT. Value heads map to key heads TILED: v head t uses key head t % key_heads.
struct GdnDims {
    std::uint32_t key_heads = 16, key_dim = 128, value_heads = 48, value_dim = 128, conv_kernel = 4;  // conv 4, dims 128 only
    [[nodiscard]] __host__ __device__ std::uint32_t qk_width() const noexcept { return key_heads * key_dim; }
    [[nodiscard]] __host__ __device__ std::uint32_t v_width() const noexcept { return value_heads * value_dim; }
    [[nodiscard]] __host__ __device__ std::uint32_t conv_channels() const noexcept { return 2 * qk_width() + v_width(); }
};

struct GdnLayer {
    const float* conv_w = nullptr;     // [channels][4], tap 3 multiplies the newest input
    const float* ssm_a = nullptr;      // [value_heads] stored = -exp(A_log)
    const float* dt_bias = nullptr;    // [value_heads]
    const float* norm_gain = nullptr;  // [value_dim] plain gain (not 1+w)
};

// Recurrent state. delta_base holds TWO sides [2][value_heads][key_dim][value_dim]; *parity (device scalar, 0/1) names the
// committed side. A window reads the committed side and writes the other; conv is the committed conv state [channels][3]
// (last three raw inputs, oldest first), only changed by gdn_commit.
struct GdnState {
    float* conv = nullptr;
    float* delta_base = nullptr;
};

// Per-layer window buffers kept until commit (the apply pass needs them): raw = qkv inputs [rows][channels]; qkv_c = conv+silu
// +l2norm output [rows][channels]; a_raw/b_raw [rows][value_heads] (the projections' outputs).
struct GdnWindow {
    const float* raw = nullptr;
    float* qkv_c = nullptr;
    const float* a_raw = nullptr;
    const float* b_raw = nullptr;
};

// Window forward for `rows` (1..32): conv+silu, l2norm of q (scaled 1/sqrt(key_dim)) and k, then the delta rule with rows
// processed sequentially from the committed state into the shadow side, o = S^T q, gated RMSNorm (swish) with z.
// beta = sigmoid(b); g = ssm_a * softplus(a + dt_bias) (softplus: x > 20 -> x, else log1p(exp(x))); S *= exp(g);
// d = (v - S^T k) * beta; S += k (x) d; o = S^T q (k ascending, per v column sequential). out: [rows][value_width].
cudaError_t gdn_window(const GdnDims& d, const GdnLayer& layer, const GdnState& state, const GdnWindow& win, const float* z,
                       float* out, std::uint32_t rows, float eps, const std::uint32_t* parity_dev, cudaStream_t stream);

constexpr std::uint32_t kMaxGdnLayers = 64;
struct GdnCommitLayer {
    GdnLayer layer;
    GdnState state;
    GdnWindow win;
};
// Accepts the first k of `window_rows` rows of the last window for the n layers: updates the conv states from the raw inputs;
// k == window_rows flips the parity (the shadow side becomes committed); 0 < k < window_rows first re-applies rows 0..k-1 to the
// committed side into the shadow side (same arithmetic as the window forward, no output), then flips. k == 0 is a no-op.
cudaError_t gdn_commit(const GdnDims& d, const GdnCommitLayer* layers, std::uint32_t n, std::uint32_t window_rows,
                       std::uint32_t k, std::uint32_t* parity_dev, cudaStream_t stream);

// ---- Multi-session decode: several sessions (state slots) in one batch. A GROUP is a run of consecutive rows of ONE state slot
// (a decode lane carries 1 row, a verify window several). Device tables [groups]: row0 (first row of the group in the batch buffers),
// nrows, state_slot (index of the session state). parity is [state slots]. Per-layer state buffers hold [slot][...] blocks:
// conv_stride floats per slot for conv ([channels][3]), delta_stride floats per slot for delta_base ([2][heads][128][128]).
// Arithmetic per group is exactly gdn_window / gdn_commit on that slot's state (rows sequential inside the group).
struct GdnSlotTable {
    const std::uint32_t* row0 = nullptr;
    const std::uint32_t* nrows = nullptr;
    const std::uint32_t* state_slot = nullptr;
    std::uint32_t* parity = nullptr;
    std::size_t conv_stride = 0, delta_stride = 0;
};
cudaError_t gdn_window_slots(const GdnDims& d, const GdnLayer& layer, const GdnState& state, const GdnWindow& win, const float* z,
                             float* out, std::uint32_t groups, const GdnSlotTable& t, float eps, cudaStream_t stream);
// accept: device [groups] rows accepted per group (null = all rows of every group). partial: some group has 0 < accept < nrows
// (re-applies the accepted rows to the committed side first). Flips the parity of groups with accept > 0.
cudaError_t gdn_commit_slots(const GdnDims& d, const GdnCommitLayer* layers, std::uint32_t n, std::uint32_t groups,
                             const GdnSlotTable& t, const std::uint32_t* accept, bool partial, cudaStream_t stream);

// PREFILL chunk of any `rows` (>= 1): bitwise the same state/outputs as running the rows through gdn_window windows of <= 32 and
// committing them (conv: row-parallel kernel with the same per-row arithmetic; delta rule: the same sequential kernel over all
// rows; conv state advanced to the chunk end). The delta result lands on the shadow side: call gdn_prefill_flip once after ALL
// layers of the chunk ran (every layer reads the committed side). win.raw/qkv_c/a_raw/b_raw hold [rows][...] buffers.
cudaError_t gdn_prefill_layer(const GdnDims& d, const GdnLayer& layer, const GdnState& state, const GdnWindow& win, const float* z,
                              float* out, std::uint32_t rows, float eps, const std::uint32_t* parity_dev, cudaStream_t stream);
cudaError_t gdn_prefill_flip(std::uint32_t* parity_dev, cudaStream_t stream);

}  // namespace strix::cuda_core
