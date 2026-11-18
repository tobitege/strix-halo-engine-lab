#include "strix/model_contract.hpp"

#include "strix/recurrent_state_policy.hpp"

#include <cmath>

namespace strix {

ModelContract ModelContract::qwen38_flash_next() {
    ModelContract model;
    model.recurrent_state = recurrent_state_policy().type;  // STRIX_RECURRENT_STATE: the one resolver, read at plan time
    return model;
}

std::string ModelContract::validate() const {
    if (hidden_layers == 0 || full_attention_layers == 0 || gdn_layers == 0) {
        return "layer counts must be non-zero";
    }
    if (full_attention_layers > hidden_layers ||
        gdn_layers != hidden_layers - full_attention_layers) {
        return "full-attention plus GDN layers must equal hidden layers";
    }
    if (full_attention_interval == 0 ||
        hidden_layers % full_attention_interval != 0 ||
        hidden_layers / full_attention_interval != full_attention_layers ||
        ple_layer_1based == 0 ||
        ple_layer_1based > hidden_layers ||
        ple_embed_dim == 0 ||
        ple_conv_kernel_size <= 1 ||
        ngram_size < 2 ||
        heads_per_ngram == 0 ||
        hc_count <= 1) {
        return "layer pattern, PLE layer and hyper-connection geometry must be valid";
    }
    const auto ple_heads =
        (ngram_size - 1ULL) * heads_per_ngram;
    if (ple_heads == 0 ||
        ple_embed_dim % ple_heads != 0) {
        return "PLE embedding dimension must divide the n-gram head count";
    }
    if (ngram_vocab_size_base == 0 ||
        ngram_vocab_divisor == 0 ||
        ngram_hash_seed == 0 ||
        ple_eos_token_id >= vocab_size) {
        return "PLE hash geometry and EOS token must be valid";
    }
    if (mtp_full_attention_layers == 0) {
        return "MTP layer count must be non-zero";
    }
    if (kv_heads == 0 || head_dim == 0 || bytes_bf16 == 0 || bytes_fp32 == 0) {
        return "head and dtype dimensions must be non-zero";
    }
    if (gdn_key_heads == 0 || gdn_key_head_dim == 0 ||
        gdn_value_heads == 0 || gdn_value_head_dim == 0 ||
        gdn_conv_kernel == 0) {
        return "GDN head and convolution dimensions must be non-zero";
    }
    if (gdn_value_heads % gdn_key_heads != 0) {
        return "GDN value heads must be divisible by key heads";
    }
    if (attention_kv_storage != AttentionKvStorage::bf16 &&
        attention_kv_storage != AttentionKvStorage::int8_token) {
        return "attention KV storage must be bf16 or int8_token";
    }
    if (indexer_n_heads == 0 ||
        indexer_kv_heads == 0 ||
        indexer_head_dim == 0 ||
        indexer_budget == 0 ||
        indexer_compress_ratio == 0 ||
        indexer_budget % indexer_compress_ratio != 0) {
        return "indexer geometry and budget must be valid";
    }
    if (!std::isfinite(partial_rotary_factor) ||
        partial_rotary_factor <= 0.0F ||
        partial_rotary_factor > 1.0F ||
        rotary_dim() == 0 ||
        rotary_dim() % 2ULL != 0 ||
        rotary_dim() > head_dim ||
        rotary_dim() > indexer_head_dim ||
        !std::isfinite(rope_theta) ||
        rope_theta <= 1.0F) {
        return "partial rotary geometry and theta must be valid";
    }
    if (resolved_rotary.descriptor().mode() == RotaryMode::yarn &&
        (max_position_embeddings != RotaryDescriptor::original_max ||
         rotary_dim() != RotaryDescriptor::rotary_dim || rope_theta != RotaryDescriptor::theta ||
         (qwen_rotary_table && resolved_rotary != qwen_rotary_table->identity())))
        return "resolved rotary checkpoint/identity mismatch";
    if (qwen_rotary_table && (rotary_dim() != RotaryDescriptor::rotary_dim ||
        rope_theta != RotaryDescriptor::theta || max_position_embeddings > qwen_rotary_table->extent()))
        return "resolved rotary table/model geometry mismatch";
    if (!std::isfinite(rms_norm_epsilon) ||
        rms_norm_epsilon <= 0.0F) {
        return "RMS normalization epsilon must be finite and positive";
    }
    if (!std::isfinite(gdn_qk_l2_epsilon) ||
        gdn_qk_l2_epsilon <= 0.0F) {
        return "GDN q/k L2 epsilon must be finite and positive";
    }
    if (hc_count <= 1 || hc_lowrank == 0) {
        return "hyper-connection count must exceed one and low-rank width must be non-zero";
    }
    if (vocab_size == 0 ||
        hidden_size == 0 ||
        max_position_embeddings == 0 ||
        moe_intermediate_size == 0 ||
        shared_expert_intermediate_size == 0 ||
        num_experts == 0 ||
        num_experts_per_token == 0 ||
        num_experts_per_token > num_experts) {
        return "MoE and hidden dimensions must be valid and non-zero";
    }
    return {};
}

}  // namespace strix
