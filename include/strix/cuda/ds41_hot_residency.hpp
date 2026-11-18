#pragma once

// deepseek41 CUDA parts-host residency from a hot-expert placement (ds41_hot_placement.hpp), with optional dynamic
// expert slots refilled after misses.
//
// Per layer the device holds one compact buffer of gate / up / down expert slices: slots [0, S) are the placement's
// static rtx-pro experts in the placement's slot order (ascending id), slots [S, S + D) are dynamic. ResidentLayer
// (cuda_ds41_parts_host_driver.hpp) is the id -> slot table of the layer, on the host and on the device; the routed
// expert kernels read slot s at s * expert bytes, so a dynamic expert gives the bits of the same expert resident from
// the start (the slot only changes the address).
//
// Dynamic slots (decode, one token at a time): after a layer's answers are in (PartsHost::run_layer returned), the
// experts of the token that were remote and are in the pinned host source are copied into dynamic slots of that
// layer on a low-priority copy stream (least recently used slot first, never a slot the current token used), then
// the layer's id -> slot table is copied; the host table changes at once. The next token's layer waits for that
// copy (before_layer) before its routes read the table. A per-token budget bounds the copies (PCIe and HBM traffic).
// Every sequence starts with empty dynamic slots (reset_dynamic).

#include "strix/cuda/ds41_parts_host_driver.hpp"
#include "strix/ds41_hot_placement.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strix::ds41::cuda {

// Positional reader of the routed expert slices of a GGUF file (blk.L.ffn_{gate,up,down}_exps.weight, IQ2_XXS /
// IQ2_XXS / Q2_K, expert e at e * expert bytes of each tensor).
class GgufExpertReader {
public:
    explicit GgufExpertReader(const std::string& path);
    ~GgufExpertReader();
    GgufExpertReader(const GgufExpertReader&) = delete;
    GgufExpertReader& operator=(const GgufExpertReader&) = delete;
    [[nodiscard]] const MoeShape& shape() const noexcept { return shape_; }
    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }
    [[nodiscard]] bool has_layer(std::uint32_t layer) const noexcept;
    // part 0 gate, 1 up, 2 down; `bytes` of the slice (gate_expert_bytes or down_expert_bytes) into dst.
    void read(std::uint32_t layer, std::uint32_t expert, int part, std::uint8_t* dst) const;
    // whole tensor of a layer (all experts) into dst
    void read_tensor(std::uint32_t layer, int part, std::uint8_t* dst) const;
    // Closes the file (reads throw afterwards). The loader's reads are buffered: an open cached handle on the model
    // file is not kept for the decode's unbuffered reads of the same file.
    void close() noexcept;
    // absolute file offset of a layer's stacked expert tensor (part 0 gate, 1 up, 2 down)
    [[nodiscard]] std::uint64_t tensor_offset(std::uint32_t layer, int part) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    MoeShape shape_;
    std::uint32_t layers_ = 0;
};

struct HotResidencyOptions {
    MoeShape shape;
    std::vector<std::uint32_t> layers;               // layers to hold (each needs a placement layer and a GGUF layer)
    std::vector<std::uint32_t> dynamic_slots;        // [placement layers] dynamic slots per layer (empty: none)
    // [placement layers] experts whose weights stay in pinned host memory as the dynamic source (must not be
    // static rtx-pro experts of the layer). Empty: no source (dynamic slots never fill).
    std::vector<std::vector<std::uint32_t>> source;
    std::uint32_t copies_per_token = 64;             // dynamic copies enqueued per token at most
    bool load_weights = true;                        // false: allocate only (timing runs that do not check bits)
    unsigned load_threads = 8;                       // layers loaded in parallel
};

struct HotRefillStats {
    std::uint32_t remote = 0;      // remote (layer, expert) of the token at the layers seen so far
    std::uint32_t dynamic_hits = 0;
    std::uint32_t copies = 0;      // enqueued
    std::uint32_t not_in_source = 0, over_budget = 0, no_slot = 0;
};

class HotResidency {
public:
    HotResidency(const HotPlacement& placement, const GgufExpertReader& reader, const HotResidencyOptions& options);
    ~HotResidency();
    HotResidency(const HotResidency&) = delete;
    HotResidency& operator=(const HotResidency&) = delete;

    [[nodiscard]] const ResidentLayer& layer(std::uint32_t layer) const;
    [[nodiscard]] std::uint32_t static_slots(std::uint32_t layer) const;
    [[nodiscard]] std::uint32_t dynamic_slots(std::uint32_t layer) const;
    [[nodiscard]] std::uint64_t device_bytes() const noexcept;
    [[nodiscard]] std::uint64_t source_bytes() const noexcept;

    // Empties every dynamic slot (a new sequence); waits for pending copies.
    void reset_dynamic();
    // Starts a token: resets the per-token copy budget and the stats.
    void begin_token();
    void set_copies_per_token(std::uint32_t n) noexcept { copies_per_token_ = n; }
    [[nodiscard]] const HotRefillStats& token_stats() const noexcept { return stats_; }
    // `stream` waits for the dynamic copies of `layer` enqueued by the previous token (no-op if none).
    void before_layer(std::uint32_t layer, cudaStream_t stream);
    // After PartsHost::run_layer of `layer` (R1 window, ids in router order on the host): marks the dynamic slots the
    // token used, and enqueues copies of its remote experts that are in the source. `layer_done` is recorded on the
    // layer stream after the layer's last use of the tables and weights (the combine); the copies wait for it.
    void after_layer(std::uint32_t layer, const std::uint32_t* host_ids, std::uint32_t top_k, cudaEvent_t layer_done);
    // The same for steps that run as graphs: the copies wait on the refill stream until the mapped word
    // done_flag_dev >= sequence (written by the layer's site after its combine), and the layer's refill generation is
    // published in refill_done (stream write) and required (host write): the next token's site waits for it.
    void after_layer_mapped(std::uint32_t layer, const std::uint32_t* host_ids, std::uint32_t top_k, const std::uint32_t* done_flag_dev,
                            std::uint32_t sequence);
    // Mapped words per layer (device pointers): required generation (host-written), refill generation done (stream-written).
    [[nodiscard]] const std::uint32_t* required_dev() const noexcept { return required_dev_; }
    [[nodiscard]] const std::uint32_t* refill_done_dev() const noexcept { return refill_done_dev_; }

private:
    struct Layer;
    std::vector<std::unique_ptr<Layer>> layers_;  // [placement layers], null if not held
    MoeShape shape_;
    std::uint32_t copies_per_token_ = 0;
    std::uint64_t token_ = 0;
    cudaStream_t refill_ = nullptr;
    HotRefillStats stats_;
    std::uint32_t copies_left_ = 0;
    std::uint64_t device_bytes_ = 0, source_bytes_ = 0;
    std::vector<void*> source_chunks_;
    std::uint32_t* required_ = nullptr;  // mapped pinned [layers]
    std::uint32_t* refill_done_ = nullptr;
    std::uint32_t* required_dev_ = nullptr;
    std::uint32_t* refill_done_dev_ = nullptr;
    std::vector<std::uint32_t> gen_;
    void refill(std::uint32_t layer, const std::uint32_t* host_ids, std::uint32_t top_k, cudaEvent_t layer_done, const std::uint32_t* done_flag_dev,
                std::uint32_t sequence);
};

}  // namespace strix::ds41::cuda
