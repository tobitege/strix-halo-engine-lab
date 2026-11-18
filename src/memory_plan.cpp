#include "strix/memory_plan.hpp"

#include <limits>
#include <stdexcept>

namespace strix {
namespace {

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error("memory-plan addition overflow");
    }
    return left + right;
}

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error("memory-plan multiplication overflow");
    }
    return left * right;
}

std::uint64_t product(std::initializer_list<std::uint64_t> values) {
    std::uint64_t result = 1;
    for (const auto value : values) {
        result = checked_mul(result, value);
    }
    return result;
}

void add_to_total(MemoryBreakdown& plan, std::uint64_t value) {
    plan.total = checked_add(plan.total, value);
}

}  // namespace

std::uint64_t base_kv_bytes_per_token(const ModelContract& model) {
    if (model.attention_kv_storage == AttentionKvStorage::int8_token) {
        // Per layer: int8 K and V codes plus one FP32 scale per KV head and half.
        const auto codes = product({model.kv_heads, model.head_dim, 2});
        const auto scales = product({model.kv_heads, 2, model.bytes_fp32});
        return product({model.full_attention_layers, checked_add(codes, scales)});
    }
    return product({model.full_attention_layers, model.kv_heads, model.head_dim,
                    2, model.bytes_bf16});
}

std::uint64_t mtp_kv_bytes_per_token(const ModelContract& model) {
    return product({model.mtp_full_attention_layers, model.kv_heads,
                    model.head_dim, 2, model.bytes_bf16});
}

std::uint64_t mtp_draft_state_bytes_per_session(
    const ModelContract& model,
    std::uint64_t element_bytes) {
    if (element_bytes == 0) {
        throw std::invalid_argument(
            "MTP draft-state element bytes must be non-zero");
    }
    return product({
        model.hc_count,
        model.hidden_size,
        element_bytes,
    });
}

std::uint64_t mtp_indexer_bytes_per_token(
    const ModelContract& model) {
    return product({
        model.mtp_full_attention_layers,
        model.indexer_kv_heads,
        model.indexer_head_dim,
        model.bytes_bf16,
    });
}

std::uint64_t indexer_bytes_per_token(
    const ModelContract& model) {
    // QSA stores one raw BF16 index key per token for every full-attention
    // layer. The 4:1 compression ratio applies only when complete blocks are
    // pooled for scoring; it does not compress persistent raw-key storage.
    return product({
        model.full_attention_layers,
        model.indexer_kv_heads,
        model.indexer_head_dim,
        model.bytes_bf16,
    });
}

std::uint64_t provisional_indexer_bytes_per_token(
    const ModelContract& model) {
    return indexer_bytes_per_token(model);
}

std::uint64_t gdn_matrix_state_bytes_per_layer(
    const ModelContract& model) {
    return checked_mul(product({
        model.gdn_value_heads,
        model.gdn_key_head_dim,
        model.gdn_value_head_dim,
    }), recurrent_state_element_bytes(model.recurrent_state));  // f32: x4, identical to before
}

std::uint64_t gdn_matrix_state_bytes_per_session(
    const ModelContract& model) {
    return checked_mul(
        model.gdn_layers,
        gdn_matrix_state_bytes_per_layer(model));
}

std::uint64_t gdn_conv_state_bytes_per_layer(
    const ModelContract& model) {
    const auto key_width =
        product({
            model.gdn_key_heads,
            model.gdn_key_head_dim,
        });
    const auto value_width =
        product({
            model.gdn_value_heads,
            model.gdn_value_head_dim,
        });
    const auto conv_width =
        checked_add(
            checked_mul(key_width, 2),
            value_width);
    return product({
        conv_width,
        model.gdn_conv_kernel,
        model.bytes_bf16,
    });
}

std::uint64_t gdn_conv_state_bytes_per_session(
    const ModelContract& model) {
    return checked_mul(
        model.gdn_layers,
        gdn_conv_state_bytes_per_layer(model));
}

std::uint64_t gdn_state_bytes_per_session(
    const ModelContract& model) {
    return checked_add(
        gdn_matrix_state_bytes_per_session(model),
        gdn_conv_state_bytes_per_session(model));
}


std::uint64_t ple_conv_state_bytes_per_session(
    const ModelContract& model) {
    const auto short_state_len =
        checked_mul(
            model.ple_conv_kernel_size - 1ULL,
            model.ngram_size);
    const auto hc_hidden =
        checked_mul(
            model.hc_count,
            model.hidden_size);
    return product({
        hc_hidden,
        short_state_len,
        model.bytes_bf16,
    });
}

std::uint64_t ngram_history_bytes_per_session(
    const ModelContract& model) {
    // Token ids fit in uint32 for the pinned vocabulary. Only ngram_size - 1
    // preceding ids are needed to resume the hash stream.
    return checked_mul(
        model.ngram_size - 1ULL,
        sizeof(std::uint32_t));
}

std::uint64_t position_history_bytes_per_session(
    const ModelContract&) {
    // Text-only serial decode needs the next absolute position as host state.
    return sizeof(std::uint64_t);
}

MemoryBreakdown build_memory_plan(const ModelContract& model,
                                  const MemoryPlanRequest& request) {
    if (const auto error = model.validate(); !error.empty()) {
        throw std::invalid_argument(error);
    }
    if (request.sessions == 0) {
        throw std::invalid_argument("sessions must be non-zero");
    }

    MemoryBreakdown plan;
    const auto positions = checked_mul(request.context_tokens, request.sessions);
    if (model.rotary_identity().descriptor().mode() == RotaryMode::yarn &&
        (!request.context_tokens || request.context_tokens > model.effective_max_positions()))
        throw std::invalid_argument("memory-plan context is outside the model limit");
    plan.base_kv = checked_mul(base_kv_bytes_per_token(model), positions);
    plan.mtp_kv = request.include_mtp_kv
                      ? checked_mul(mtp_kv_bytes_per_token(model), positions)
                      : 0;
    plan.mtp_draft_state = request.include_mtp_draft_state
                               ? checked_mul(
                                     mtp_draft_state_bytes_per_session(
                                         model,
                                         request.mtp_draft_state_element_bytes),
                                     request.sessions)
                               : 0;
    plan.mtp_indexer = request.include_mtp_indexer
                           ? checked_mul(
                                 mtp_indexer_bytes_per_token(model),
                                 positions)
                           : 0;
    plan.indexer = request.include_indexer
                       ? checked_mul(
                             indexer_bytes_per_token(model),
                             positions)
                       : 0;
    plan.gdn_matrix_state = checked_mul(
        gdn_matrix_state_bytes_per_session(model), request.sessions);
    plan.gdn_conv_state = checked_mul(
        gdn_conv_state_bytes_per_session(model), request.sessions);
    plan.gdn_state = checked_add(
        plan.gdn_matrix_state, plan.gdn_conv_state);
    plan.ple_conv_state = checked_mul(
        ple_conv_state_bytes_per_session(model),
        request.sessions);
    plan.ngram_history = checked_mul(
        ngram_history_bytes_per_session(model),
        request.sessions);
    plan.position_history = checked_mul(
        position_history_bytes_per_session(model),
        request.sessions);
    plan.rollback = request.rollback == RollbackPolicy::full_gdn_snapshots
                        ? product({gdn_state_bytes_per_session(model),
                                   request.sessions, request.draft_tokens})
                        : 0;
    plan.resident_model = request.resident_model_bytes;
    plan.bounded_cache = request.bounded_cache_bytes;

    add_to_total(plan, plan.base_kv);
    add_to_total(plan, plan.mtp_kv);
    add_to_total(plan, plan.mtp_draft_state);
    add_to_total(plan, plan.mtp_indexer);
    add_to_total(plan, plan.indexer);
    add_to_total(plan, plan.gdn_state);
    add_to_total(plan, plan.ple_conv_state);
    add_to_total(plan, plan.ngram_history);
    add_to_total(plan, plan.position_history);
    add_to_total(plan, plan.rollback);
    add_to_total(plan, plan.resident_model);
    add_to_total(plan, plan.bounded_cache);
    return plan;
}

AdmissionDecision decide_admission(const HostBudget& host,
                                   const MemoryBreakdown& plan) {
    AdmissionDecision decision;
    decision.required = plan.total;

    const auto protected_bytes = checked_add(host.baseline_without_engine_bytes,
                                             host.recovery_reserve_bytes);
    if (protected_bytes >= host.mem_total_bytes) {
        decision.reason = "host baseline plus reserve leaves no engine budget";
        return decision;
    }
    decision.engine_limit = host.mem_total_bytes - protected_bytes;
    if (plan.total > decision.engine_limit) {
        decision.reason = "plan exceeds the engine admission limit";
        return decision;
    }
    decision.admitted = true;
    decision.headroom = decision.engine_limit - plan.total;
    decision.reason = "plan fits the explicit engine budget";
    return decision;
}

}  // namespace strix
