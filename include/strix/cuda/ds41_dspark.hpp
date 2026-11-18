#pragma once

// deepseek41 DSpark drafter on CUDA (sm_89 / sm_120): the device side of ds41_dspark.hpp for a single-GPU replica.
// Every result equals the host reference (ds41_dspark.cpp, identity ds41-dspark-v1) bit for bit
// (tests/cuda/ds41_dspark_tests.cu). It is the kernel set of hip_ds41_dspark.hip (32-lane waves = CUDA warps) on the
// CUDA layer core where the declared operation is the same (A8, Q8_0 x A8 products incl. the grouped attn_output_a,
// the router, the Q8_0 shared expert, the window routes and the combine), plus MXFP4 routed-expert GEMVs.
//
// On one GPU the drafter reads the target's resident tensors (token_embd F16, output Q8_0) when given their device
// addresses; nothing of the target is copied then. The drafter's numerics never change output tokens: every drafted
// token is verified by the target.

#include "strix/cuda/ds41_common.hpp"
#include "strix/cuda/ds41_core.hpp"
#include "strix/ds41_dspark.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace strix::ds41::cuda {

struct CudaDrafterOptions {
    // Device addresses of the target's tensors (null: upload the copies in the Weights views).
    const std::uint16_t* token_embd = nullptr;  // F16 [129280][5120]
    const std::uint8_t* output = nullptr;       // Q8_0 rows [129280][5120] (GGUF bytes)
    // The target's head in the tiled tensor-core layout (Q8TiledBuffer, same bits as the row kernel): when set, the
    // drafter's head reads it (core rows) and gathers its extra vocabulary rows from it; `output` is then unused and no
    // raw copy of the head exists on the device (measured saving: -0.655 GiB on the PRO).
    Q8Tiled output_tiled{};
    bool tile_head = false;                     // own copy: tile it and drop the raw rows (tests the tiled path alone)
    std::uint32_t max_append_rows = 128;        // rows of one append call
    // Expert pruning (default off): per stage, the routed experts kept on the device (ids < 128, distinct, >= top_k);
    // an empty list keeps all. Pruned experts are never selected (selection bias -1e30) and never uploaded, so the
    // drafter's device footprint shrinks; drafts change, the target's verification keeps the output exact.
    std::vector<std::vector<std::uint32_t>> keep;
};

struct CudaDraftTimes {
    std::uint64_t head_weight_bytes = 0;  // calculated active vocabulary weight payload (not physical transactions)
    float device_ms = 0.0F;  // first launch to the last op of the draft step (events)
    double wall_ms = 0.0;    // draft() including the token download
};

struct CudaDrafterState {
    std::uint64_t start=0, next=0;
    dspark::Vocab vocabulary;
    std::vector<std::vector<std::uint8_t>> rings;
};

class CudaDrafter {
public:
    CudaDrafter(const dspark::Weights& weights, const CudaDrafterOptions& options = {});
    ~CudaDrafter();
    CudaDrafter(const CudaDrafter&) = delete;
    CudaDrafter& operator=(const CudaDrafter&) = delete;

    // main_x of committed target rows (main_hidden [rows][15360]), downloaded (tests).
    [[nodiscard]] std::vector<float> main_rows(std::span<const float> main_hidden, std::uint32_t rows);
    // Appends main_x rows at the next positions (tests), or main_hidden rows (main_rows + append on the device).
    void append(std::span<const float> main_x, std::uint32_t rows);
    void append_hidden(std::span<const float> main_hidden, std::uint32_t rows);
    // The same from rows in device memory (e.g. the target step's captured inputs of layers 37-39 reduced by
    // CudaStepIo::draft_hidden). The drafter's stream waits for `producer` (an event, no host synchronisation); the rows
    // are consumed when the call returns (the caller may overwrite them).
    void append_hidden_device(const float* main_hidden, std::uint32_t rows, cudaStream_t producer);
    // One draft step from `token` at position next(): five drafted tokens and their confidences.
    [[nodiscard]] dspark::Draft draft(std::uint32_t token, bool keep_logits = false, CudaDraftTimes* times = nullptr);
    [[nodiscard]] std::uint64_t next() const noexcept;
    // Draft vocabulary (default: all ids); extra ids gather their head rows from the output tensor.
    void set_vocabulary(const dspark::Vocab& vocabulary);
    [[nodiscard]] dspark::Vocab vocabulary() const;
    // Adds context ids (prompt and committed tokens) not yet in the vocabulary; returns how many.
    std::size_t add_vocabulary(std::span<const std::uint32_t> ids);
    // Forgets every appended row (new sequence); the next row is at `position` (a sequence continued from an imported
    // state starts with an empty ring there: draft steps attend only to the rows appended since).
    void reset(std::uint64_t position = 0);
    // Quiescent committed boundary only. Restore validates all shapes before device writes.
    [[nodiscard]] CudaDrafterState export_state() const;
    void import_state(const CudaDrafterState& state);
    // Diagnostics (pruned-drafter study): counts of the routed experts each draft step selects, per stage
    // ([stage][expert], every row of the block x top_k); off by default, part of the draft graph when on.
    void set_route_stats(bool on);
    [[nodiscard]] std::vector<std::uint32_t> route_stats() const;  // downloads (after the draft stream's work)
    void clear_route_stats();
    [[nodiscard]] std::vector<std::uint8_t> ring(std::uint32_t stage) const;
    [[nodiscard]] std::size_t device_bytes() const noexcept;
    [[nodiscard]] cudaStream_t stream() const noexcept;

private:
    struct Impl;
    static void enqueue_draft_step(Impl& m, bool keep_logits, dspark::Draft& out);
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::ds41::cuda
