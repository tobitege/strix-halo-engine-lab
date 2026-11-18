#pragma once

// Native-MTP ChainDrafter glue shared by every backend that has an engine with an MTP layer (CUDA CudaEngine, HIP HipEngine).
// Home: shared (host spec layer): the ring / carry / catch-up logic is device-agnostic; a backend supplies only
//   Engine::Rows / Engine::MtpOut   output structs (device pointers: top, hidden_pre, hidden_post)
//   Engine::config().hidden, Engine::read_top(Rows, n, host) (synchronizes), Engine::mtp_forward(next_tokens, hidden_dev, rows, pos0, want_top)
//   Dev::alloc(engine, floats) / Dev::release(engine, ptr) / Dev::copy(engine, dst, src, floats)   (async device->device on the engine stream)
//
// LAZY catch-up: the hidden rows of every committed position are kept in a device ring (with their successor tokens) and
// written into the MTP KV only when a draft needs them, so copy-only stretches pay no MTP cost. Chain steps feed the MTP
// layer's own hidden back ('post' = after shared_head_norm, 'pre' = block output). Exactness never depends on the drafts
// (BlockSpeculator verifies every id); only acceptance does.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <span>
#include <vector>

#include "strix/spec/block_drafter.hpp"

namespace strix::spec {

template <class Engine, class Dev>
class MtpChain final : public ChainDrafter {
public:
    using Rows = typename Engine::Rows;
    using MtpOut = typename Engine::MtpOut;

    // mtp_rows = widest mtp_forward call (engine limit); ring = positions buffered before a forced flush.
    MtpChain(Engine& e, bool pre_hidden, std::uint32_t mtp_rows, std::uint32_t ring = 512)
        : e_(e), pre_(pre_hidden), hid_(e.config().hidden), rows_max_(std::min<std::uint32_t>(mtp_rows, 32U)), ring_n_(ring) {
        ring_ = Dev::alloc(e_, std::size_t(ring_n_) * hid_);
        carry_ = Dev::alloc(e_, hid_);
        next_.reserve(ring_n_);
    }
    ~MtpChain() override {
        if (ring_) Dev::release(e_, ring_);
        if (carry_) Dev::release(e_, carry_);
    }
    MtpChain(const MtpChain&) = delete;
    MtpChain& operator=(const MtpChain&) = delete;

    // New sequence (the engine is reset by the caller): forget the MTP positions.
    void reset() {
        next_.clear();
        mtp_pos_ = 0;
        d1_ = 0;
        last_rows_ = 0;
    }
    // The last verify pass (rows of [anchor, drafts...]); accept() needs its hidden rows.
    void observe_verify(const Rows& o, std::uint32_t rows) {
        last_ = o;
        last_rows_ = rows;
    }
    // A prefill window of `rows` committed rows: row i's successor = tokens[i + 1]; the last row's = next_after.
    void observe_prefill(const Rows& o, std::uint32_t rows, const std::uint32_t* tokens, std::uint32_t next_after) {
        succ_.resize(rows);
        for (std::uint32_t i = 0; i + 1 < rows; ++i) succ_[i] = tokens[i + 1];
        succ_[rows - 1] = next_after;
        push_hidden(o, rows, succ_.data());
    }

    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
        const auto t0 = std::chrono::steady_clock::now();
        out.clear();
        if (!next_.empty()) flush();
        // carry_ = MTP hidden after the last committed row; d1_ = its prediction (the token after the anchor's successor)
        std::uint32_t pos = mtp_pos_;  // speculative rows from here on are overwritten by the next flush
        std::uint32_t tok = d1_;       // chain token for the next step when there is no prefix
        for (std::size_t i = 0; i < prefix.size(); ++i)  // feed the copy tokens: only the last step's prediction is used
            tok = step(prefix[i], pos++, i + 1 == prefix.size());
        for (std::uint32_t j = 0; j < k; ++j) {
            out.push_back(tok);
            if (j + 1 < k) tok = step(tok, pos++, true);
        }
        draft_wall_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++drafts_;
    }
    void accept(std::span<const std::uint32_t> emitted) override {
        if (last_rows_ == 0) return;
        // rows 0..emitted-1 of the last verify are committed; emitted[i] follows row i
        push_hidden(last_, static_cast<std::uint32_t>(emitted.size()), emitted.data());
    }

    [[nodiscard]] std::uint64_t calls() const { return calls_; }    // mtp_forward calls (flush chunks + chain steps)
    [[nodiscard]] std::uint64_t rows() const { return rows_; }
    [[nodiscard]] std::uint64_t drafts() const { return drafts_; }  // draft() invocations
    [[nodiscard]] double draft_wall_ms() const { return draft_wall_ms_; }

private:
    void push_hidden(const Rows& o, std::uint32_t rows, const std::uint32_t* next_tokens) {
        const float* src = pre_ ? o.hidden_pre : o.hidden_post;
        std::uint32_t done = 0;
        while (done < rows) {
            if (next_.size() == ring_n_) flush();
            const std::uint32_t n = std::min<std::uint32_t>(rows - done, ring_n_ - static_cast<std::uint32_t>(next_.size()));
            Dev::copy(e_, ring_ + next_.size() * hid_, src + std::size_t(done) * hid_, std::size_t(n) * hid_);
            next_.insert(next_.end(), next_tokens + done, next_tokens + done + n);
            done += n;
        }
    }
    // Write the ring into the MTP KV (rows_max_ per call, causal inside each call). Leaves the last row's top / hidden in
    // d1_ / carry_.
    void flush() {
        std::uint32_t off = 0;
        const auto total = static_cast<std::uint32_t>(next_.size());
        while (off < total) {
            const std::uint32_t n = std::min<std::uint32_t>(rows_max_, total - off);
            const bool final_chunk = off + n == total;  // only the last row's draft is needed: earlier chunks skip lm_head
            const auto m = e_.mtp_forward(next_.data() + off, ring_ + std::size_t(off) * hid_, n, mtp_pos_, final_chunk);
            ++calls_;
            rows_ += n;
            mtp_pos_ += n;
            off += n;
            if (final_chunk) {
                Rows ro{};
                ro.top = m.top;
                e_.read_top(ro, n, top_tmp_);
                d1_ = top_tmp_[n - 1];
                Dev::copy(e_, carry_, (pre_ ? m.hidden_pre : m.hidden_post) + std::size_t(n - 1) * hid_, hid_);
            }
        }
        next_.clear();
    }
    std::uint32_t step(std::uint32_t input_token, std::uint32_t pos, bool want_top) {
        const auto m = e_.mtp_forward(&input_token, carry_, 1, pos, want_top);
        ++calls_;
        ++rows_;
        std::uint32_t t = 0;
        if (want_top) {
            Rows ro{};
            ro.top = m.top;
            e_.read_top(ro, 1, &t);
        }
        Dev::copy(e_, carry_, pre_ ? m.hidden_pre : m.hidden_post, hid_);
        return t;
    }

    Engine& e_;
    bool pre_;
    std::uint32_t hid_, rows_max_, ring_n_;
    Rows last_{};
    std::uint32_t last_rows_ = 0;
    float* ring_ = nullptr;
    float* carry_ = nullptr;
    std::vector<std::uint32_t> next_, succ_;
    std::uint32_t mtp_pos_ = 0, d1_ = 0;
    std::uint32_t top_tmp_[32]{};
    std::uint64_t calls_ = 0, rows_ = 0, drafts_ = 0;
    double draft_wall_ms_ = 0.0;
};

}  // namespace strix::spec
