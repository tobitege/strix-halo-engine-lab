// deepseek41 CUDA layer core: device weights of a layer range and the core part of every plan op (cuda_ds41_layer_core.hpp).

#include "strix/cuda/ds41_layer_core.hpp"

#include "ds41_cuda_common.cuh"

#include <cmath>
#include <cstring>

namespace strix::ds41::cuda {
namespace {

using detail::require;

// Concatenated Q8_0 tensors (row-wise) repacked into the tiled form.
void tile_tensors(TensorFile& f, const std::vector<std::string>& names, std::uint32_t width, Q8TiledBuffer& out, cudaStream_t s) {
    std::size_t total = 0;
    for (const auto& n : names) total += f.tensor(n).byte_size;
    DeviceBuffer<std::uint8_t> cat(total);
    std::size_t at = 0;
    for (const auto& n : names) {
        STRIX_CUDA_CHECK(cudaMemcpy(cat.get() + at, f.device(n), f.tensor(n).byte_size, cudaMemcpyDeviceToDevice));
        at += f.tensor(n).byte_size;
        f.release(n);
    }
    out.repack(cat.get(), static_cast<std::uint32_t>(total / q8_0_row_bytes(width)), width, s);
    STRIX_CUDA_CHECK(cudaStreamSynchronize(s));
}

}  // namespace

LayerCoreGeometry LayerCoreGeometry::from(const deepseek41::Config& c) {
    LayerCoreGeometry g;
    g.hidden = c.hidden_size;
    g.q_lora = c.q_lora_rank;
    g.q_out = c.num_attention_heads * c.head_dim;
    g.head_dim = c.head_dim;
    g.o_groups = c.o_groups;
    g.o_lora = c.o_lora_rank;
    g.experts = c.n_routed_experts;
    g.top_k = c.num_experts_per_tok;
    g.inter = c.moe_intermediate_size;
    g.vocab = c.vocab_size;
    g.hc_iters = c.hc_sinkhorn_iters;
    g.engram_cols = (c.engram_max_ngram_size - 1U) * c.engram_n_heads;
    g.hc_eps = static_cast<float>(c.hc_eps);
    g.route_scale = static_cast<float>(c.routed_scaling_factor);
    g.swiglu_limit = static_cast<float>(c.swiglu_limit);
    return g;
}

LayerCore::LayerCore(TensorFile& f, std::uint32_t first, std::uint32_t end, bool embed, bool head, std::uint32_t max_rows,
                     cudaStream_t s)
    : g_(LayerCoreGeometry::from(f.config())), first_(first), end_(end), rows_(max_rows), max_rows_(max_rows) {
    require(first < end && max_rows > 0, "layer core range");
    const auto cfg = f.config();
    for (std::uint32_t l = first; l < end; ++l) {
        auto w = std::make_unique<LayerCoreWeights>();
        const std::string p = "blk." + std::to_string(l) + ".";
        w->layer = l;
        w->attn_norm = f.device_f32(p + "attn_norm.weight");
        w->q_a_norm = f.device_f32(p + "attn_q_a_norm.weight");
        w->ffn_norm = f.device_f32(p + "ffn_norm.weight");
        w->hc_attn_fn = f.device_f16(p + "hc_attn_fn.weight");
        w->hc_attn_base = f.device_f32(p + "hc_attn_base.weight");
        w->hc_attn_scale = f.device_f32(p + "hc_attn_scale.weight");
        w->hc_ffn_fn = f.device_f16(p + "hc_ffn_fn.weight");
        w->hc_ffn_base = f.device_f32(p + "hc_ffn_base.weight");
        w->hc_ffn_scale = f.device_f32(p + "hc_ffn_scale.weight");
        w->router = f.device_f32(p + "ffn_gate_inp.weight");
        w->bias = f.device_f32(p + "exp_probs_b.bias");
        w->bias_image = f.device_f32(p + "exp_probs_b_vl.bias");
        for (const auto e : cfg.engram_layer_ids)
            if (e == l) w->engram = true;
        if (w->engram) {
            w->engram_kv = f.device_f16(p + "engram_kv.weight");
            w->engram_q_norm = f.device_f32(p + "engram_q_norm.weight");
            w->engram_k_norm = f.device_f32(p + "engram_k_norm.weight");
        }
        tile_tensors(f, {p + "attn_q_a.weight", p + "attn_kv.weight"}, g_.hidden, w->qa_kv, s);
        tile_tensors(f, {p + "attn_q_b.weight"}, g_.q_lora, w->q_b, s);
        tile_tensors(f, {p + "attn_output_a.weight"}, g_.q_out / g_.o_groups, w->o_a, s);
        tile_tensors(f, {p + "attn_output_b.weight"}, g_.o_groups * g_.o_lora, w->o_b, s);
        tile_tensors(f, {p + "ffn_gate_shexp.weight", p + "ffn_up_shexp.weight"}, g_.hidden, w->gate_up, s);
        tile_tensors(f, {p + "ffn_down_shexp.weight"}, g_.inter, w->down, s);
        layers_.push_back(std::move(w));
    }
    if (embed) embed_table_ = f.device_f16("token_embd.weight");
    if (head) {
        output_norm_ = f.device_f32("output_norm.weight");
        tile_tensors(f, {"output.weight"}, g_.hidden, output_, s);
        has_head_ = true;
    }
    // buffers
    const std::size_t R = max_rows, H = g_.hidden;
    b_.rows = max_rows;
    b_.stream.resize(R * 4U * H);
    b_.pre_carry.resize(R * 4U);
    b_.mix_attn.resize(R * 24U);
    b_.mix_ffn.resize(R * 24U);
    b_.x.resize(R * H);
    b_.xcol.resize(R * H);
    b_.qkv_stride = g_.q_lora + g_.head_dim;
    b_.kv_offset = g_.q_lora;
    b_.qkv.resize(R * b_.qkv_stride);
    b_.qr.resize(R * g_.q_lora);
    b_.q.resize(R * g_.q_out);
    b_.o.resize(R * g_.q_out);
    b_.o_low.resize(R * g_.o_groups * g_.o_lora);
    b_.y.resize(R * H);
    b_.shared.resize(R * H);
    b_.route_w.resize(R * g_.top_k);
    b_.route_ids.resize(R * g_.top_k);
    b_.scores.resize(R * g_.experts);
    b_.tokens.resize(R);
    b_.engram_rows.resize(R * g_.engram_cols * 264U);
    b_.xa.resize(max_rows, g_.hidden);
    b_.qa.resize(max_rows, g_.q_lora);
    b_.oa.resize(max_rows, g_.q_out);
    b_.la.resize(max_rows, g_.o_groups * g_.o_lora);
    std::size_t split = 0;
    for (const std::uint32_t count : {b_.qkv_stride, g_.q_out, g_.o_groups * g_.o_lora, g_.hidden, 2U * g_.inter})
        split = std::max(split, q8_0_a8_split_scratch_floats(count, max_rows, 4));
    b_.split_scratch.resize(split);
    b_.workspace.resize(std::max({hc_mix_workspace_bytes(max_rows, g_.hidden), shared_expert_workspace_bytes(max_rows, g_.inter),
                                  engram_workspace_bytes(max_rows, g_.engram_cols, g_.hidden)}));
    if (head) {
        b_.logits.resize(R * g_.vocab);
        b_.candidates.resize(R);
    }
}

void LayerCore::set_rows(std::uint32_t rows) {
    require(rows > 0 && rows <= max_rows_, "layer core rows");
    rows_ = rows;
}

std::string LayerCore::label(const char* op, std::uint32_t layer, const char* site, const char* buffer) const {
    std::string s = op;
    if (layer != 0xffffffffU) s += "@" + std::to_string(layer);
    if (site && *site) s += std::string("/") + site;
    return s + ":" + buffer;
}

void LayerCore::tap(const std::string& lbl, const float* p, std::uint32_t count, std::uint32_t stride) const {
    if (tap_) tap_(lbl, p, rows_, count, stride ? stride : count);
}

void LayerCore::embed(const std::uint32_t* host_tokens, cudaStream_t s) {
    require(embed_table_ != nullptr, "layer core without the embedding");
    STRIX_CUDA_CHECK(cudaMemcpyAsync(b_.tokens.get(), host_tokens, rows_ * 4U, cudaMemcpyHostToDevice, s));
    enqueue_embed({{embed_table_, g_.vocab, g_.hidden}, b_.tokens.get(), rows_, b_.stream.get(), b_.pre_carry.get()}, s);
    tap(label("embed", 0xffffffffU, "", "stream"), b_.stream.get(), 4U * g_.hidden);
    tap(label("embed", 0xffffffffU, "", "pre_carry"), b_.pre_carry.get(), 4U);
}

void LayerCore::attn_input(std::uint32_t layer, const std::uint8_t* host_rows, cudaStream_t s) {
    const auto& w = weights(layer);
    if (w.engram) {
        require(host_rows != nullptr, "Engram layer without rows");
        STRIX_CUDA_CHECK(
            cudaMemcpyAsync(b_.engram_rows.get(), host_rows, std::size_t{rows_} * g_.engram_cols * 264U, cudaMemcpyHostToDevice, s));
        EngramView e;
        e.row_bytes = b_.engram_rows.get();
        e.rows = rows_;
        e.cols = g_.engram_cols;
        e.hidden = g_.hidden;
        e.kv = {w.engram_kv, 5U * g_.hidden, g_.engram_cols * 256U};
        e.q_norm = w.engram_q_norm;
        e.k_norm = w.engram_k_norm;
        e.stream = b_.stream.get();
        e.workspace = b_.workspace.get();
        enqueue_engram(e, s);
        tap(label("engram", layer, "", "stream"), b_.stream.get(), 4U * g_.hidden);
    }
    HcMixView m;
    m.stream = b_.stream.get();
    m.rows = rows_;
    m.hidden = g_.hidden;
    m.fn = w.hc_attn_fn;
    m.base = w.hc_attn_base;
    m.scale = w.hc_attn_scale;
    m.iters = g_.hc_iters;
    m.hc_eps = g_.hc_eps;
    m.mixes = b_.mix_attn.get();
    m.partial = reinterpret_cast<float*>(b_.workspace.get());
    if (!fuse_hc_) enqueue_hc_mixes(m, s);
    NormView n;
    n.hc_stream = b_.stream.get();
    n.pre = b_.pre_carry.get();
    n.pre_stride = 4;
    n.weight = w.attn_norm;
    n.rows = rows_;
    n.n = g_.hidden;
    n.out = b_.x.get();
    n.collapsed = tap_ ? b_.xcol.get() : nullptr;
    n.a8 = b_.xa.view();
    n.a8.rows = rows_;
    if (fuse_hc_) enqueue_hc_mixes_norm(m, n, s);
    else enqueue_norm(n, s);
    tap(label("hc_mix", layer, "attn", "mix_attn"), b_.mix_attn.get(), 24U);
    tap(label("hc_pre", layer, "attn", "x"), b_.xcol.get(), g_.hidden);
    tap(label("rms_norm", layer, "attn", "x"), b_.x.get(), g_.hidden);
}

void LayerCore::attn_q(std::uint32_t layer, cudaStream_t s) {
    const auto& w = weights(layer);
    Q8A8View v;
    v.tiled = w.qa_kv.view();
    v.count = b_.qkv_stride;
    v.x = b_.xa.view();
    v.x.rows = rows_;
    v.out = b_.qkv.get();
    v.split_scratch = b_.split_scratch.get();
    v.split_scratch_floats = b_.split_scratch.size();
    enqueue_q8_0_a8(v, s);
    tap(label("q_a", layer, "", "qr"), b_.qkv.get(), g_.q_lora, b_.qkv_stride);
    NormView n;
    n.x = b_.qkv.get();
    n.x_stride = b_.qkv_stride;
    n.weight = w.q_a_norm;
    n.rows = rows_;
    n.n = g_.q_lora;
    n.out = b_.qr.get();
    n.a8 = b_.qa.view();
    n.a8.rows = rows_;
    enqueue_norm(n, s);
    tap(label("rms_norm", layer, "q", "qr"), b_.qr.get(), g_.q_lora);
    Q8A8View qb;
    qb.tiled = w.q_b.view();
    qb.count = g_.q_out;
    qb.x = b_.qa.view();
    qb.x.rows = rows_;
    qb.out = b_.q.get();
    qb.split_scratch = b_.split_scratch.get();
    qb.split_scratch_floats = b_.split_scratch.size();
    enqueue_q8_0_a8(qb, s);
    tap(label("q_b", layer, "", "q"), b_.q.get(), g_.q_out);
    tap(label("kv", layer, "", "kvrow"), b_.qkv.get() + b_.kv_offset, g_.head_dim, b_.qkv_stride);
}

void LayerCore::attn_output(std::uint32_t layer, cudaStream_t s) {
    const auto& w = weights(layer);
    const std::uint32_t ol = g_.o_groups * g_.o_lora, half = ol / 2U;
    auto oa = b_.oa.view();
    oa.rows = rows_;
    enqueue_quantize_a8({b_.o.get(), rows_, g_.q_out, 0, oa}, s);
    Q8A8View a;
    a.tiled = w.o_a.view();
    a.count = ol;
    a.per_group = g_.o_lora;
    a.width = g_.q_out / g_.o_groups;
    a.x = oa;
    a.out = b_.o_low.get();
    a.split_scratch = b_.split_scratch.get();
    a.split_scratch_floats = b_.split_scratch.size();
    enqueue_q8_0_a8(a, s);
    tap(label("o_a", layer, "", "o_low"), b_.o_low.get(), ol);
    auto la = b_.la.view();
    la.rows = rows_;
    enqueue_quantize_a8({b_.o_low.get(), rows_, ol, 0, la}, s);
    for (std::uint32_t h = 0; h < 2U; ++h) {
        Q8A8View v;
        v.tiled = w.o_b.view();
        v.count = g_.hidden;
        v.k_offset = h * half;
        v.width = half;
        v.x = {la.codes + h * half, la.deltas + h * half / 32U, la.sums + h * half / 32U, rows_, half, ol};
        v.out = b_.y.get();
        v.accumulate = h > 0;
        v.split_scratch = b_.split_scratch.get();
        v.split_scratch_floats = b_.split_scratch.size();
        enqueue_q8_0_a8(v, s);
    }
    tap(label("o_b", layer, "", "y"), b_.y.get(), g_.hidden);
    enqueue_hc_expand({b_.y.get(), 0, b_.stream.get(), b_.mix_attn.get(), rows_, g_.hidden, b_.stream.get()}, s);
    tap(label("hc_post", layer, "attn", "stream"), b_.stream.get(), 4U * g_.hidden);
}

void LayerCore::ffn_input(std::uint32_t layer, cudaStream_t s) {
    const auto& w = weights(layer);
    HcMixView m;
    m.stream = b_.stream.get();
    m.rows = rows_;
    m.hidden = g_.hidden;
    m.fn = w.hc_ffn_fn;
    m.base = w.hc_ffn_base;
    m.scale = w.hc_ffn_scale;
    m.iters = g_.hc_iters;
    m.hc_eps = g_.hc_eps;
    m.mixes = b_.mix_ffn.get();
    m.partial = reinterpret_cast<float*>(b_.workspace.get());
    if (!fuse_hc_) enqueue_hc_mixes(m, s);
    NormView n;
    n.hc_stream = b_.stream.get();
    n.pre = b_.mix_attn.get();
    n.pre_stride = 24;
    n.weight = w.ffn_norm;
    n.rows = rows_;
    n.n = g_.hidden;
    n.out = b_.x.get();
    n.collapsed = tap_ ? b_.xcol.get() : nullptr;
    n.a8 = b_.xa.view();
    n.a8.rows = rows_;
    if (fuse_hc_) enqueue_hc_mixes_norm(m, n, s);
    else enqueue_norm(n, s);
    tap(label("hc_mix", layer, "ffn", "mix_ffn"), b_.mix_ffn.get(), 24U);
    tap(label("hc_pre", layer, "ffn", "x"), b_.xcol.get(), g_.hidden);
    tap(label("rms_norm", layer, "ffn", "x"), b_.x.get(), g_.hidden);
}

void LayerCore::ffn_core(std::uint32_t layer, cudaStream_t s) {
    const auto& w = weights(layer);
    RouterView r;
    r.x = b_.x.get();
    r.rows = rows_;
    r.weight = {w.router, g_.experts, g_.hidden};
    r.bias = w.bias;
    r.bias_image = w.bias_image;
    r.top_k = g_.top_k;
    r.route_scale = g_.route_scale;
    r.scores = b_.scores.get();
    r.ids = b_.route_ids.get();
    r.weights = b_.route_w.get();
    enqueue_router(r, s);
    tap(label("router", layer, "", "route_ids"), reinterpret_cast<const float*>(b_.route_ids.get()), g_.top_k);
    tap(label("router", layer, "", "route_w"), b_.route_w.get(), g_.top_k);
    SharedExpertView e;
    e.gate_up_tiled = w.gate_up.view();
    e.down_tiled = w.down.view();
    e.limit = g_.swiglu_limit;
    e.x = b_.xa.view();
    e.x.rows = rows_;
    e.out = b_.shared.get();
    e.workspace = b_.workspace.get();
    if (fuse_shared_ && rows_ == 1U) enqueue_shared_expert_fused(e, s);
    else enqueue_shared_expert(e, s);
    tap(label("shared_expert", layer, "", "shared"), b_.shared.get(), g_.hidden);
}

void LayerCore::ffn_output(std::uint32_t layer, cudaStream_t s) {
    enqueue_hc_expand({b_.y.get(), 0, b_.stream.get(), b_.mix_ffn.get(), rows_, g_.hidden, b_.stream.get(),
                       fuse_hc_ ? b_.pre_carry.get() : nullptr}, s);
    // the next attention input collapses with this site's pre (pre_carry = mix_ffn[0..4))
    if (!fuse_hc_) STRIX_CUDA_CHECK(cudaMemcpy2DAsync(b_.pre_carry.get(), 4U * sizeof(float), b_.mix_ffn.get(), 24U * sizeof(float), 4U * sizeof(float),
                                       rows_, cudaMemcpyDeviceToDevice, s));
    tap(label("hc_post", layer, "ffn", "stream"), b_.stream.get(), 4U * g_.hidden);
    tap(label("hc_post", layer, "ffn", "pre_carry"), b_.pre_carry.get(), 4U);
}

void LayerCore::head(cudaStream_t s) {
    require(has_head_, "layer core without the head");
    NormView n;
    n.hc_stream = b_.stream.get();
    n.pre = b_.pre_carry.get();
    n.pre_stride = 4;
    n.weight = output_norm_;
    n.rows = rows_;
    n.n = g_.hidden;
    n.a8 = b_.xa.view();
    n.a8.rows = rows_;
    enqueue_norm(n, s);
    Q8A8View v;
    v.tiled = output_.view();
    v.count = g_.vocab;
    v.x = b_.xa.view();
    v.x.rows = rows_;
    v.out = b_.logits.get();
    enqueue_q8_0_a8(v, s);
    tap(label("head", 0xffffffffU, "", "logits"), b_.logits.get(), g_.vocab);
    enqueue_argmax({b_.logits.get(), rows_, g_.vocab, 0, 0, b_.candidates.get()}, s);
    tap(label("argmax", 0xffffffffU, "", "token"), reinterpret_cast<const float*>(b_.candidates.get()), 3U);
}

}  // namespace strix::ds41::cuda
