// Per-layer state and the layer-level sequence of the deepseek41 CUDA attention (see cuda_ds41_attention.hpp).
#include "strix/cuda/ds41_attention.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace strix::ds41::cuda::attn {
namespace {


}  // namespace

struct AttentionState::Impl {
    std::vector<ds41::attn::LayerAttention> plan;
    LayerAttentionOptions options;
    std::vector<Caches> caches;
    std::vector<void*> allocations;
    // workspace
    float *ckv = nullptr, *cscore = nullptr, *iq = nullptr, *iw = nullptr, *partial = nullptr, *scores = nullptr;
    void* qprep = nullptr;
    std::uint32_t* block_keys = nullptr;
    std::uint32_t* survivors = nullptr;
    std::size_t bytes = 0;

    template <class T> T* alloc(std::size_t count) {
        void* p = nullptr;
        const std::size_t b = std::max<std::size_t>(count, 1) * sizeof(T);
        STRIX_CUDA_CHECK(cudaMalloc(&p, b));
        STRIX_CUDA_CHECK(cudaMemset(p, 0, b));
        allocations.push_back(p);
        bytes += b;
        return static_cast<T*>(p);
    }
};

AttentionState::AttentionState(std::vector<ds41::attn::LayerAttention> plan, const LayerAttentionOptions& options) : impl_(std::make_unique<Impl>()) {
    auto& I = *impl_;
    I.plan = std::move(plan);
    I.options = options;
    const auto& o = options;
    if (o.first_layer >= o.end_layer || o.end_layer > I.plan.size() || o.max_positions == 0U || o.max_rows == 0U || o.window_rows == 0U ||
        o.hidden == 0U || o.hidden % 256U != 0U || o.q_lora_rank == 0U || o.q_lora_rank % 256U != 0U ||
        o.decoder_rows > o.max_rows || o.candidate_rows > o.max_rows)
        throw std::invalid_argument("ds41 cuda attention state: options (layers, positions, rows, hidden, q_lora_rank)");
    const auto ledger = attention_allocations(I.plan, o);
    ring_capacity_ = ledger.ring_rows;
    rope_ = std::make_unique<RopeTables>(o.max_positions);
    // layers needing caches: the range and every source it reads
    std::vector<bool> need(I.plan.size(), false);
    for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l) {
        need[l] = true;
        const auto& r = I.plan[l];
        if (r.ratio != 0U) {
            need.at(static_cast<std::size_t>(r.kv_source)) = true;
            need.at(static_cast<std::size_t>(r.index_source)) = true;
        }
    }
    I.caches.assign(I.plan.size(), Caches{});
    std::uint32_t cand_ratio = 0;
    for (std::size_t l = 0; l < I.plan.size(); ++l) {
        if (!need[l]) continue;
        const auto& r = I.plan[l];
        auto& c = I.caches[l];
        if (l >= o.first_layer && l < o.end_layer) c.ring = I.alloc<std::uint8_t>(ledger.bytes("ring", static_cast<int>(l)) / sizeof(std::uint8_t));
        if (r.owns_kv) {
            c.kv_cache = I.alloc<std::uint8_t>(ledger.bytes("kv_cache", static_cast<int>(l)) / sizeof(std::uint8_t));
            c.idx_cache = I.alloc<std::uint8_t>(ledger.bytes("idx_cache", static_cast<int>(l)) / sizeof(std::uint8_t));
            if (r.ratio == 2U) {
                c.carry = I.alloc<float>(ledger.bytes("carry", static_cast<int>(l)) / sizeof(float));
                c.carry_valid = I.alloc<std::uint32_t>(ledger.bytes("carry_valid", static_cast<int>(l)) / sizeof(std::uint32_t));
                if (l >= o.first_layer && l < o.end_layer && o.allocate_workspace) c.carry_rows = I.alloc<float>(ledger.bytes("carry_rows", static_cast<int>(l)) / sizeof(float));
            }
        }
        if (r.owns_index) {
            c.rows = l >= 20U && o.decoder_rows != 0U ? o.decoder_rows : o.max_rows;
            c.idx = I.alloc<std::uint32_t>(ledger.bytes("idx", static_cast<int>(l)) / sizeof(std::uint32_t));
            c.idx_count = I.alloc<std::uint32_t>(ledger.bytes("idx_count", static_cast<int>(l)) / sizeof(std::uint32_t));
        }
        if (r.candidate_source || r.candidate_user) cand_ratio = std::max(cand_ratio, 1U);
    }
    cand_stride_ = ledger.cand_stride;
    if (cand_ratio != 0U) candidates_ = I.alloc<std::uint8_t>(ledger.bytes("candidates"));
    if (o.allocate_workspace) {
    I.ckv = I.alloc<float>(ledger.bytes("ckv") / sizeof(float));
    I.cscore = I.alloc<float>(ledger.bytes("cscore") / sizeof(float));
    I.iq = I.alloc<float>(ledger.bytes("iq") / sizeof(float));
    I.iw = I.alloc<float>(ledger.bytes("iw") / sizeof(float));
    I.qprep = I.alloc<std::uint8_t>(ledger.bytes("qprep") / sizeof(std::uint8_t));
    I.scores = I.alloc<float>(ledger.bytes("scores") / sizeof(float));
    I.block_keys = I.alloc<std::uint32_t>(ledger.bytes("block_keys") / sizeof(std::uint32_t));
    I.survivors = I.alloc<std::uint32_t>(ledger.bytes("survivors") / sizeof(std::uint32_t));
    I.partial = I.alloc<float>(ledger.bytes("partial") / sizeof(float));
    }
    bytes_ = ledger.total();
    if (bytes_ != I.bytes + ledger.bytes("rope_window") + ledger.bytes("rope_compressed"))
        throw std::logic_error("ds41 cuda attention: allocation ledger mismatch");
}

AttentionState::~AttentionState() {
    if (impl_)
        for (void* p : impl_->allocations) cudaFree(p);
}

const AttentionState::Caches& AttentionState::caches(std::uint32_t layer) const { return impl_->caches.at(layer); }
const std::vector<ds41::attn::LayerAttention>& AttentionState::plan() const { return impl_->plan; }
const LayerAttentionOptions& AttentionState::options() const { return impl_->options; }

void AttentionState::set_decoder_row0(std::uint32_t row0) {
    auto& I = *impl_;
    if (row0 > I.options.max_rows || (row0 != 0U && I.options.decoder_rows == 0U))
        throw std::invalid_argument("ds41 cuda attention: compact row origin");
    candidate_row0_ = I.options.candidate_rows != 0U ? row0 : 0U;
    for (std::size_t l = 20; l < I.caches.size(); ++l)
        if (I.caches[l].idx != nullptr) I.caches[l].row0 = row0;
}

void AttentionState::reset(cudaStream_t stream) {
    const auto& I = *impl_;
    const auto& o = I.options;
    for (std::size_t l = 0; l < I.plan.size(); ++l) {
        const auto& c = I.caches[l];
        const auto& r = I.plan[l];
        if (c.ring != nullptr) STRIX_CUDA_CHECK(cudaMemsetAsync(c.ring, 0, std::size_t{ring_capacity_} * kWindowEntryBytes, stream));
        if (c.kv_cache != nullptr) {
            const std::size_t entries = o.max_positions / r.ratio + 1U;
            STRIX_CUDA_CHECK(cudaMemsetAsync(c.kv_cache, 0, entries * kKvEntryBytes, stream));
            STRIX_CUDA_CHECK(cudaMemsetAsync(c.idx_cache, 0, entries * kIdxEntryBytes, stream));
        }
        if (c.carry != nullptr) {
            STRIX_CUDA_CHECK(cudaMemsetAsync(c.carry, 0, 2 * kHeadDim * sizeof(float), stream));
            STRIX_CUDA_CHECK(cudaMemsetAsync(c.carry_valid, 0, sizeof(std::uint32_t), stream));
        }
        if (c.idx != nullptr) {
            STRIX_CUDA_CHECK(cudaMemsetAsync(c.idx, 0, std::size_t{c.rows} * kTopK * 4U, stream));
            STRIX_CUDA_CHECK(cudaMemsetAsync(c.idx_count, 0, std::size_t{c.rows} * 4U, stream));
        }
    }
    if (candidates_ != nullptr)
        STRIX_CUDA_CHECK(cudaMemsetAsync(candidates_, 0, std::size_t{o.candidate_rows ? o.candidate_rows : o.max_rows} * cand_stride_, stream));
}

void AttentionState::commit_rows(std::uint32_t first, std::uint32_t end, std::uint64_t last_pos, std::uint32_t kept_row, cudaStream_t stream) {
    const auto& I = *impl_;
    if (kept_row >= I.options.max_rows) throw std::invalid_argument("ds41 cuda attention: commit row");
    for (std::uint32_t l = std::max(first, I.options.first_layer); l < std::min<std::uint32_t>(end, I.options.end_layer); ++l) {
        const auto& c = I.caches[l];
        if (I.plan[l].owns_kv && I.plan[l].ratio == 2U && c.carry_rows != nullptr) enqueue_carry_restore(c.carry_rows, c.carry, c.carry_valid, last_pos, kept_row, stream);
    }
}

void AttentionState::enqueue_layer(std::uint32_t layer, const LayerWeights& w, const LayerIo& io, cudaStream_t stream, Phase phase) {
    auto& I = *impl_;
    const auto& o = I.options;
    if (!o.allocate_workspace || o.decoder_rows != 0U || o.candidate_rows != 0U)
        throw std::logic_error("ds41 cuda attention: enqueue_layer needs its full workspace and row layout");
    if (layer < o.first_layer || layer >= o.end_layer) throw std::invalid_argument("ds41 cuda attention: layer outside the range");
    if (io.rows == 0U || io.rows > o.max_rows || io.pos_bound >= o.max_positions || io.pos_bound + 1U < io.rows)
        throw std::invalid_argument("ds41 cuda attention: rows 1..max_rows, positions below max_positions");
    const auto& r = I.plan[layer];
    const auto& c = I.caches[layer];
    const RopeRow* rope = rope_->table(r.rope);
    const bool prep = phase != Phase::attend, attend = phase != Phase::prep;
    // the attention's own F16 projections, one launch
    if (prep) {
        F16RowsView v{};
        v.rows = io.rows;
        const auto job = [&](const std::uint16_t* wt, std::uint32_t width, std::uint32_t count, const float* x, float* out) {
            if (wt == nullptr) throw std::invalid_argument("ds41 cuda attention: missing F16 weight");
            v.jobs[v.job_count++] = F16Job{wt, width, count, x, width, out, count};
        };
        if (r.owns_kv) {
            job(w.compressor_kv, o.hidden, kHeadDim, io.x, I.ckv);
            if (r.ratio == 2U) job(w.compressor_gate, o.hidden, kHeadDim, io.x, I.cscore);
        }
        if (r.owns_index) {
            job(w.index_q_b, o.q_lora_rank, kIdxHeads * kIdxDim, io.qr, I.iq);
            job(w.index_proj, o.hidden, kIdxHeads, io.x, I.iw);
        }
        if (v.job_count != 0U) enqueue_f16_rows(v, stream);
    }
    if (prep) enqueue_window_write(WindowView{io.kv, w.kv_norm, rope, io.pos, io.rows, ring_capacity_, c.ring}, stream);
    if (prep && r.owns_kv) {
        CompressView v{};
        v.ratio = r.ratio;
        v.rows = io.rows;
        v.kv = I.ckv;
        v.score = I.cscore;
        v.norm = w.compressor_norm;
        v.wk = w.index_k;
        v.knorm = w.index_k_norm;
        v.rope = rope_->table(RopeKind::compressed);
        v.pos = io.pos;
        v.state = c.carry;
        v.state_valid = c.carry_valid;
        v.stash = c.carry_rows;
        v.kv_cache = c.kv_cache;
        v.idx_cache = c.idx_cache;
        enqueue_compress(v, stream);
    }
    const std::uint32_t W = std::min(o.window_rows, o.max_rows);
    for (std::uint32_t w0 = 0; w0 < io.rows; w0 += W) {
        const std::uint32_t n = std::min(W, io.rows - w0);
        const std::uint64_t bound = io.pos_bound - (io.rows - w0 - n);  // >= the window's last position (rows are consecutive)
        if (prep && r.owns_index) {
            const auto& src = I.plan[static_cast<std::size_t>(r.kv_source)];
            const auto& sc = I.caches[static_cast<std::size_t>(r.kv_source)];
            const std::uint32_t max_count = static_cast<std::uint32_t>((bound + 1U) / src.ratio);
            IndexView iv{};
            iv.rows = n;
            iv.ratio = src.ratio;
            iv.max_count = max_count;
            iv.q = I.iq + std::size_t{w0} * kIdxHeads * kIdxDim;
            iv.w = I.iw + std::size_t{w0} * kIdxHeads;
            iv.rope = rope;
            iv.pos = io.pos + w0;
            iv.idx_cache = sc.idx_cache;
            iv.scores = I.scores;
            iv.qprep = I.qprep;
            enqueue_index_scores(iv, stream);
            SelectView sv{};
            sv.rows = n;
            sv.ratio = src.ratio;
            sv.max_count = max_count;
            sv.topk = kTopK;
            sv.cand_blocks = kCandBlocks;
            sv.block_size = kCandBlockSize;
            sv.pool = r.candidate_source ? Pool::candidate_source : r.candidate_user ? Pool::candidate_user : Pool::plain;
            sv.pos = io.pos + w0;
            sv.scores = I.scores;
            sv.candidates = candidates_ + std::size_t{w0} * cand_stride_;
            sv.cand_stride = cand_stride_;
            sv.block_keys = I.block_keys;
            sv.survivors = I.survivors;
            sv.idx = c.idx + std::size_t{w0} * kTopK;
            sv.idx_count = c.idx_count + w0;
            if (sv.pool != Pool::plain && candidates_ == nullptr) throw std::logic_error("ds41 cuda attention: no candidate pool");
            enqueue_select(sv, stream);
        }
        if (!attend) continue;
        AttentionView av{};
        av.rows = n;
        av.capacity = ring_capacity_;
        av.topk = kTopK;
        av.q = io.q + std::size_t{w0} * kHeads * kHeadDim;
        av.sink = w.sinks;
        av.rope = rope;
        av.pos = io.pos + w0;
        av.ring = c.ring;
        if (r.ratio != 0U) {
            const auto& ic = I.caches[static_cast<std::size_t>(r.index_source)];
            av.kv_cache = I.caches[static_cast<std::size_t>(r.kv_source)].kv_cache;
            av.idx = ic.idx + std::size_t{w0} * kTopK;
            av.idx_count = ic.idx_count + w0;
        }
        av.partial = I.partial;
        av.o = io.o + std::size_t{w0} * kHeads * kHeadDim;
        av.heads = kHeads;
        enqueue_attention(av, stream);
    }
}

}  // namespace strix::ds41::cuda::attn
