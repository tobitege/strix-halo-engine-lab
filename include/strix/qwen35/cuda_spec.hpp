#pragma once
// Binds the qwen35 CUDA engine to the shared block drafting layer (strix/spec/block_drafter.hpp).
// Home: specific (engine API binding: CudaEngine rows / commit / MTP); the drafting logic and the MTP chain glue are shared
// (strix::spec::MtpChain in strix/spec/mtp_chain.hpp; this file supplies only the device calls).
// Header-only (CUDA runtime calls); weight-format agnostic: everything goes through CudaEngine, so BF16 / Q8_0 / TURBO
// packs need no change here.
//
// SpecEngine owns the glue state of one stream:
//   verify(tokens)  -> forward_rows(R, all logits) + host argmax of every row           (strix::spec::VerifyTarget)
//   commit(k)       -> engine commit(k) (+ host sync so the step wall is complete)
//   make_chain()    -> native MTP drafter (strix::spec::ChainDrafter = shared MtpChain)
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "strix/qwen35/cuda_engine.hpp"
#include "strix/spec/block_drafter.hpp"
#include "strix/spec/mtp_chain.hpp"

namespace strix::qwen35 {

struct CudaChainDev {
    static float* alloc(CudaEngine&, std::size_t n) {
        float* p = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(float)) != cudaSuccess) throw std::runtime_error("spec: cudaMalloc");
        return p;
    }
    static void release(CudaEngine&, float* p) { cudaFree(p); }
    static void copy(CudaEngine& e, float* dst, const float* src, std::size_t n) {
        cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice, e.stream());
    }
};

class SpecEngine final : public strix::spec::VerifyTarget {
public:
    using Chain = strix::spec::MtpChain<CudaEngine, CudaChainDev>;
    explicit SpecEngine(CudaEngine& e, bool chain_pre_hidden = false) : e_(e) {
        if (e.has_mtp()) chain_ = std::make_unique<Chain>(e, chain_pre_hidden, 32U);
    }
    SpecEngine(const SpecEngine&) = delete;
    SpecEngine& operator=(const SpecEngine&) = delete;

    CudaEngine& engine() { return e_; }
    [[nodiscard]] bool has_chain() const { return chain_ != nullptr; }

    // ---- VerifyTarget
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        const auto rows = static_cast<std::uint32_t>(tokens.size());
        const auto o = e_.forward_rows(tokens.data(), rows, LogitRows::all);
        greedy.resize(rows);
        e_.read_top(o, rows, greedy.data());  // synchronizes
        if (chain_) chain_->observe_verify(o, rows);
    }
    void commit(std::uint32_t k) override {
        e_.commit(k);
        e_.sync();
    }
    // Prefill chunk (not a drafting step): rows at positions committed().. ; next_first = the token after the chunk when
    // it is not the last one. The last chunk returns the greedy token after the prompt (the anchor) in *top_out.
    // Feeds the MTP ring (row i's successor = tokens[i+1], the last row's = next_first / the anchor).
    void prefill(const std::uint32_t* tokens, std::uint32_t rows, std::uint32_t next_first, bool last, std::uint32_t* top_out) {
        const auto o = e_.forward_rows(tokens, rows, last ? LogitRows::last : LogitRows::none);
        if (last) e_.read_top(o, 1, top_out);
        if (chain_) chain_->observe_prefill(o, rows, tokens, last ? *top_out : next_first);
        e_.commit(rows);
        e_.sync();
    }
    Chain& make_chain() { return *chain_; }
    [[nodiscard]] std::uint64_t chain_calls() const { return chain_ ? chain_->calls() : 0; }
    [[nodiscard]] std::uint64_t chain_rows() const { return chain_ ? chain_->rows() : 0; }
    // New sequence: the engine is reset by the caller; forget the MTP positions.
    void reset() {
        if (chain_) chain_->reset();
    }

private:
    CudaEngine& e_;
    std::unique_ptr<Chain> chain_;
};

}  // namespace strix::qwen35
