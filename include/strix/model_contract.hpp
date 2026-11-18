#pragma once

#include "strix/hip_qwen_rotary_table.hpp"
#include "strix/recurrent_state_type.hpp"
#include <cstdint>
#include <string>

namespace strix {

class HipQwenRotaryTable;

// Storage of the target model's QSA attention KV cache (the MTP head keeps its
// own BF16 cache). bf16 is the canonical model dtype. int8_token is the opt-in
// kv-int8-tok-v1 identity: per token, per KV head and per K/V half, symmetric
// int8 codes of the BF16-rounded values with one FP32 scale (absmax / 127).
enum class AttentionKvStorage : std::uint32_t { bf16 = 0, int8_token = 1 };

[[nodiscard]] constexpr const char* attention_kv_storage_name(AttentionKvStorage storage) noexcept {
    return storage == AttentionKvStorage::int8_token ? "kv-int8-tok-v1" : "bf16";
}

// Captured from the official Qwen config pinned in artifacts/. These values
// describe shapes only; they do not define operator semantics.
struct ModelContract {
    std::uint64_t vocab_size = 248320;
    std::uint64_t hidden_size = 2560;
    std::uint64_t max_position_embeddings = 262144;
    std::uint64_t hidden_layers = 48;
    std::uint64_t full_attention_interval = 4;
    std::uint64_t ple_layer_1based = 2;
    std::uint64_t ple_embed_dim = 2560;
    std::uint64_t ple_conv_kernel_size = 4;
    std::uint64_t ngram_size = 3;
    std::uint64_t heads_per_ngram = 8;
    std::uint64_t ngram_vocab_size_base = 20'000'000;
    std::uint64_t ngram_vocab_divisor = 128;
    std::uint64_t ngram_hash_seed = 1234;
    std::uint64_t ple_eos_token_id = 248044;
    std::uint64_t hc_count = 4;
    std::uint64_t hc_lowrank = 320;
    std::uint64_t full_attention_layers = 12;
    std::uint64_t gdn_layers = 36;
    std::uint64_t attention_heads = 24;
    std::uint64_t kv_heads = 2;
    std::uint64_t head_dim = 256;
    std::uint64_t mtp_full_attention_layers = 1;
    bool mtp_use_dedicated_embeddings = false;
    bool mtp_hidden_state_source_layer_explicit = false;
    std::uint64_t gdn_key_heads = 16;
    std::uint64_t gdn_key_head_dim = 128;
    std::uint64_t gdn_value_heads = 48;
    std::uint64_t gdn_value_head_dim = 128;
    std::uint64_t gdn_conv_kernel = 4;
    std::uint64_t indexer_n_heads = 4;
    std::uint64_t indexer_kv_heads = 1;
    std::uint64_t indexer_head_dim = 128;
    std::uint64_t indexer_budget = 2048;
    std::uint64_t indexer_compress_ratio = 4;
    float partial_rotary_factor = 0.25F;
    // Experimental internal seam; every model copy retains the actual allocation.
    // Admission may bind the host-generated identity before the device upload.
    RotaryIdentity resolved_rotary;
    std::uint64_t effective_max_positions() const noexcept {
        return rotary_identity().descriptor().mode() == RotaryMode::yarn
            ? rotary_identity().descriptor().effective_max() : max_position_embeddings;
    }
    std::shared_ptr<const HipQwenRotaryTable> qwen_rotary_table = nullptr;
    const RotaryIdentity& rotary_identity() const noexcept {
        return qwen_rotary_table ? qwen_rotary_table->identity() : resolved_rotary;
    }
    float rope_theta = 10'000'000.0F;
    std::uint64_t moe_intermediate_size = 640;
    std::uint64_t shared_expert_intermediate_size = 640;
    std::uint64_t num_experts = 512;
    std::uint64_t num_experts_per_token = 10;
    float rms_norm_epsilon = 1e-6F;
    bool norm_topk_probability = true;
    bool gdn_output_gate_sigmoid = true;
    bool gdn_qk_l2norm = true;
    float gdn_qk_l2_epsilon = 1e-6F;
    bool gdn_conv_silu = true;
    std::uint64_t bytes_bf16 = 2;
    std::uint64_t bytes_fp32 = 4;
    RecurrentStateType recurrent_state = RecurrentStateType::f32;  // GDN matrix storage; qwen38_flash_next() takes the policy
    AttentionKvStorage attention_kv_storage = AttentionKvStorage::bf16;

    [[nodiscard]] static ModelContract qwen38_flash_next();

    [[nodiscard]] constexpr std::uint64_t rotary_dim() const noexcept {
        return static_cast<std::uint64_t>(
            static_cast<double>(head_dim) *
            partial_rotary_factor);
    }

    [[nodiscard]] constexpr std::uint64_t ple_head_count() const noexcept {
        return (ngram_size - 1ULL) * heads_per_ngram;
    }

    [[nodiscard]] constexpr std::uint64_t ple_head_dim() const noexcept {
        const auto heads = ple_head_count();
        return heads == 0 ? 0 : ple_embed_dim / heads;
    }

    [[nodiscard]] constexpr std::uint64_t ple_row_bytes_bf16() const noexcept {
        return ple_head_dim() * bytes_bf16;
    }

    [[nodiscard]] std::string validate() const;
};

// Shared numerical predicate for captures, MTP and merged QSA rows. A table
// allocation may be replaced by byte-identical coefficients for eager work;
// captured graphs additionally require the original live allocation generation.
inline bool same_rotary_model(const ModelContract& a, const ModelContract& b) {
    return a.rotary_identity() == b.rotary_identity() && a.rope_theta == b.rope_theta &&
        a.partial_rotary_factor == b.partial_rotary_factor && a.rotary_dim() == b.rotary_dim();
}
}  // namespace strix
