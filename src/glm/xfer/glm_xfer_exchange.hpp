#pragma once
// The GLM exchange module behind the engine's Exchange seam (design notes
// section 2). Drop-in for the engine's CommExchange with the same results bit for bit:
//   reduce(p):  send_copy(p) -> the stream waits for the peer's partial (no CPU in the chain) -> p = p0 + p1 (R17,
//               rank 0's first) in place;
//   argmax:     one kernel packs (value bits, id) per row into the TX slot and publishes it; after the exchange a merge
//               kernel applies R18 (larger value, ties -> lower id) on both ranks; then drain() (the step's CPU sync,
//               which also seals the deferred CRC of the last reduction).
// All calls must come on one stream (the module's rule); the Exchanger is created on the first call with that stream.
// The faster forms (producer writing the TX slot in place, in-kernel wait inside the post-exchange kernel, the C X
// bracket in the wire window) need the boundary hook proposed in design notes section 2; this class is the seam-compatible form.
#include "glm_xfer.hpp"
#include "strix/glm/engine.hpp"

#include <array>
#include <atomic>
#include <memory>

namespace strix::glm::xfer {

// Prefill-only owner policy. The caller retires the ordinary exchange owner
// before entry, and drains this one before returning to head/decode.
class SeqparExchange {
public:
    SeqparExchange(dist::Transport& transport, unsigned rank, unsigned max_rows, hipStream_t stream,
                   unsigned compute_streams = 2);
    ~SeqparExchange();
    static std::size_t workspace_bytes(unsigned max_rows);
    void materialize_last(float* residual, unsigned rows);
    hipStream_t compute(unsigned lane) const { return side_.compute.at(lane == 1 && side_.streams == 1 ? 0 : lane); }
    float* normalized() const { return side_.normalized_dev; }
    // Column scratch and attention recurrence remain ordered A then B with
    // either one shared compute queue or the legacy per-half compute queues.
    void column_begin(unsigned lane);
    void column_end(unsigned lane);
    void join();  // both compute tails -> original engine stream
    void start_reduce_scatter(float* partial, SeqparRows tile, unsigned lane, hipStream_t producer);
    const float* finish_reduce_scatter(unsigned lane, hipStream_t consumer);
    void release_reduce_scatter(unsigned lane, hipStream_t consumer);
    void start_gather(float* normalized, SeqparRows tile, unsigned lane, hipStream_t producer);
    void finish(unsigned lane, hipStream_t consumer);
    void complete() { channel_.drain(); }
private:
    struct Side {
        hipStream_t stream = nullptr;
        std::array<hipStream_t, 2> compute{};
        unsigned streams;
        std::array<hipEvent_t, 2> ready{}, done{};
        std::array<hipEvent_t, 64> consumed{};
        hipEvent_t column = nullptr;
        float* normalized_host = nullptr;
        float* normalized_dev = nullptr;
        Side(unsigned max_rows, unsigned compute_streams);
        ~Side();
    } side_;  // destroyed after Channel has stopped its progress owner
    unsigned rank_, max_rows_;
    hipStream_t stream_;
    std::uint64_t site_ = 0;
    std::array<bool, 2> pending_{};
    std::array<bool, 2> reduction_{}, holding_{};
    std::array<unsigned, 2> slot_{};
    std::array<const float*, 2> peer_{};
    std::array<bool, 64> slot_live_{}, slot_recorded_{};
    bool column_recorded_ = false;
    void produce(unsigned lane, hipStream_t producer);
    void produced(unsigned lane);
    SeqparChannel channel_;
};

class XferExchange final : public engine::Exchange {
public:
    // `transport` is borrowed (the communicator's, between its own uses) or owned; options as Options::from_env().
    XferExchange(dist::Transport& transport, std::uint32_t rank, unsigned max_rows);
    XferExchange(std::unique_ptr<dist::Transport> transport, std::uint32_t rank, unsigned max_rows);
    ~XferExchange() override;
    void reduce(float* p, unsigned rows, hipStream_t s) override;
    void argmax(float* value, std::uint32_t* id, unsigned rows, hipStream_t s) override;
    void merge(MergeRow* rows, unsigned n, MergeCtl& ctl, hipStream_t s) override;
    // The fused boundary (design notes section 3), for an engine hook in place of `reduce(part)` + `dev::hc_expand(X, mix, part,
    // Xout, flag)`: send_copy(part), [the C X bracket while the message travels], then R17 + R7 expand in one kernel that
    // waits for the peer in-kernel (or after a stream wait). Same bits as reduce + hc_expand (bench gates). Form from
    // STRIX_GLMX_BOUNDARY = kw (default) | kw-br | br | plain. `part` is not modified.
    void boundary(const float* part, const float* X, const float* mix, float* Xout, unsigned* flag, unsigned rows, hipStream_t s);
    // The engine seam's hook (Exchange::fused_boundary, added by the integrator): always the fused form here.
    bool fused_boundary(const float* part, const float* X, const float* mix, float* Xout, unsigned* flag, unsigned rows,
                        hipStream_t s) override {
        boundary(part, X, mix, Xout, flag, rows, s);
        return true;
    }
    bool fused_boundary_rows(const float* part, const float* X, const float* mix, float* Xout, unsigned* flags,
                             unsigned rows, unsigned flag_stride, hipStream_t s) override;
    // The next reduction's TX slot for a producer that writes and publishes it (boundary_rows then only posts it).
    ProducerSlot producer_slot(unsigned rows, hipStream_t s) override;
    // Completion point for callers that end without the head (prefill chunks, gate_residual, G3 dumps): waits for every
    // posted exchange, seals the last deferred CRC and throws if any exchange failed. Results of the call are trusted
    // only after this (argmax() calls it itself). The engine's seam has no such hook yet: proposed as
    // Exchange::complete(s), called before residual / state results are returned.
    void complete();
    void complete(hipStream_t) override { complete(); }
    [[nodiscard]] const Exchanger* exchanger() const noexcept { return xf_.get(); }
    // Attention shares this owner with Xfer TP. With communicator TP, callers
    // must retire that owner before entry and complete() + sync before exit.
    Exchanger& attention_channel(hipStream_t s) { return ensure(s); }
    void reserve_attention_bytes(std::size_t bytes) {
        if (xf_ != nullptr) throw std::logic_error("glm xfer: capacity already fixed");
        attention_bytes_ = bytes;
    }

    // Exchange row fold (xport/row_fold.hpp; STRIX_GLMX_FOLD=rows gives the module its fold slots): process-wide
    // switch for in-process A/B runs. Same bits either way.
    static void set_fold_enabled(bool on) { fold_enabled_.store(on, std::memory_order_relaxed); }
private:
    void boundary_rows(const float* part, const float* X, const float* mix, float* Xout, unsigned* flags,
                       unsigned rows, unsigned flag_stride, hipStream_t s);
    std::array<std::uint8_t, spec::kMergeBytes> merge_tx_{}, merge_rx_{};
    Exchanger& ensure(hipStream_t s);
    std::unique_ptr<dist::Transport> owned_;
    dist::Transport* transport_ = nullptr;
    std::uint32_t rank_ = 0;
    unsigned max_rows_ = 0;
    std::size_t attention_bytes_ = 0;
    std::unique_ptr<Exchanger> xf_;
    std::uint64_t step_ = 0, layer_ = 0;
    const float* produced_ = nullptr;  // the TX slot handed out by producer_slot() for the next boundary
    float* bracket_ = nullptr;  // [max_rows][4][4096], allocated on first boundary()
    bool kwait_ = true, use_bracket_ = false;
    static inline std::atomic<bool> fold_enabled_{true};
};

}  // namespace strix::glm::xfer
