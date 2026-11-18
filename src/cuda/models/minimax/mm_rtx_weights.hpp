#pragma once
// Per-layer weight blobs of the RTX PRO prefill: every tensor a layer needs, both TP halves, in
// ONE contiguous buffer (256-byte aligned pieces), so a layer is resident (one device buffer), or streamed (pinned host
// blob -> device slot with one async copy). bind() turns a blob placed at a device address into kernel pointers.
#include "mm_rtx_pack.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strix::minimax::rtxpf::host {

inline constexpr unsigned kLayers = 60, kFirstMoe = 3, kExperts = 128;

struct MatRef {
    std::size_t trellis = 0, suh = 0, svh = 0;  // byte offsets in the blob
    std::uint32_t input = 0, output = 0, bits = 0;
};
struct MlpRef {
    MatRef gate, up, down;
};
struct RankRef {
    MatRef q, k, v, o;
    std::size_t wiq = 0;  // FP32 [6144][256], this rank's index heads, transposed
    MlpRef dense, shared;
    std::vector<MlpRef> experts;
};
struct LayerBlob {
    std::vector<std::uint8_t> bytes;
    std::array<RankRef, 2> r;
    std::size_t wik = 0, q_w = 0, k_w = 0, iq_w = 0, ik_w = 0, w_post = 0, w_next = 0, router = 0, bias = 0;
    bool moe = false;
};
LayerBlob build_layer(Pack& pack, unsigned layer, bool residual_only = false);

// Kernel view of a blob placed at `base` (device).
struct DevMat {
    const std::uint32_t* trellis = nullptr;
    const std::uint16_t *suh = nullptr, *svh = nullptr;
    std::uint32_t input = 0, output = 0, bits = 0;
};
struct DevMlp {
    DevMat gate, up, down;
};
struct DevRank {
    DevMat q, k, v, o;
    const float* wiq = nullptr;
    DevMlp dense, shared;
    std::vector<DevMlp> experts;
};
struct DevLayer {
    std::array<DevRank, 2> r;
    const float *wik = nullptr, *q_w = nullptr, *k_w = nullptr, *iq_w = nullptr, *ik_w = nullptr, *w_post = nullptr, *w_next = nullptr;
    const float *router = nullptr, *bias = nullptr;
    bool moe = false;
};
DevLayer bind(const LayerBlob& blob, const std::uint8_t* base);

}  // namespace strix::minimax::rtxpf::host
