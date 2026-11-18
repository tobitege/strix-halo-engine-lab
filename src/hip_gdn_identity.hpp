#pragma once

#include "strix/hip_gdn_launcher.hpp"
#include "strix/q8_projection_organization.hpp"
#include "strix/qpf2k_defaults.hpp"

#include <bit>
#include <stdexcept>
#include <string_view>

// Pure host identity/admission shared by the launcher and CPU regression tests.
namespace strix::hip_gdn_identity {

// Exact R1 fusion changes launch organization only. Keep the selected kernel id
// (including its terminal projection selectors), numerical kernel plan and saved
// state/cache compatibility unchanged. The implementation hash below distinguishes
// ON/OFF in compile_launch_plan(), including graph/island organization identities.
inline void hash_u64(std::uint64_t& hash, std::uint64_t value) noexcept {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= 1099511628211ULL;
    }
}
inline void hash_float(std::uint64_t& hash, float value) noexcept {
    hash_u64(hash, std::bit_cast<std::uint32_t>(value));
}
inline constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;

inline std::string kernel_id(
    const ExecutionBindingRequest& request,
    const HipGdnLauncherConfig& config) {
    return
        "hip.gdn.bf16.h" +
        std::to_string(
            config.model.hidden_size) +
        ".kh" +
        std::to_string(
            config.model.gdn_key_heads) +
        ".vh" +
        std::to_string(
            config.model.gdn_value_heads) +
        ".d" +
        std::to_string(
            config.model.gdn_key_head_dim) +
        ".r" +
        std::to_string(
            request.logical_rows) +
        ".c" +
        std::to_string(
            request.request_concurrency) + (request.device_graph_inputs ? ".device" : ".scalar") +
        std::string(gdn_recurrent_layout_marker(config.recurrent_layout)) +
        std::string(gdn_recurrent_storage_marker(config.recurrent_storage)) +
        (config.model.recurrent_state != RecurrentStateType::f32
             ? std::string(".state-") + recurrent_state_type_name(config.model.recurrent_state) + "-rne-v1" : std::string()) +
        (config.wave8_q8_input_projections ? std::string(q8_projection_organization::gdn_wave8_marker) : "") +
        (config.wave8_q8_output_projection ? std::string(q8_projection_organization::gdn_output_wave8_marker) : "") +
        (config.dense_a8_projections ? std::string(q8_projection_organization::dense_a8_marker) : "") +
        (config.dense_trellis_projections ? ".trellis-g1-fp32-s4-v1" : "") +
        (config.prefetch_q8_input_projections ? ".prefetch8" : "") +
        (config.virtual_q8_input_projections ? ".virtual256.q8-input" : "") +
        (config.virtual_q8_output_projection ? ".virtual256.q8-output" : "");
}

inline std::uint64_t implementation_identity(
    const HipGdnLauncherConfig& config,
    bool decode_fuse = qpf2k::decode_fuse_enabled()) noexcept {
    std::uint64_t hash =
        fnv_offset;
    hash_u64(hash, 0x47444e4c41554e32ULL);
    if (decode_fuse) hash_u64(hash, 0x47444e4446555331ULL);
    hash_u64(hash, config.enable_device_graph_inputs);
    hash_u64(hash, config.model.max_position_embeddings);
    hash_u64(
        hash,
        config.model.hidden_size);
    hash_u64(
        hash,
        config.model.gdn_key_heads);
    hash_u64(
        hash,
        config.model.gdn_key_head_dim);
    hash_u64(
        hash,
        config.model.gdn_value_heads);
    hash_u64(
        hash,
        config.model.gdn_value_head_dim);
    hash_u64(
        hash,
        config.model.gdn_conv_kernel);
    hash_float(
        hash,
        config.model.rms_norm_epsilon);
    hash_float(
        hash,
        config.model.gdn_qk_l2_epsilon);
    hash_u64(
        hash,
        config.model.gdn_output_gate_sigmoid
            ? 1ULL
            : 0ULL);
    hash_u64(
        hash,
        config.model.gdn_qk_l2norm
            ? 1ULL
            : 0ULL);
    hash_u64(
        hash,
        config.model.gdn_conv_silu
            ? 1ULL
            : 0ULL);
    if (config.virtual_q8_input_projections) hash_u64(hash, 0x47444e5138563235ULL);
    if (config.virtual_q8_output_projection) hash_u64(hash, 0x47444e514f555431ULL);
    if (config.prefetch_q8_input_projections) hash_u64(hash, 0x47444e5138504631ULL);
    if (config.wave8_q8_input_projections) hash_u64(hash, 0x47444e5138574156ULL);
    if (config.wave8_q8_output_projection) hash_u64(hash, 0x47444e514f573831ULL);
    if (config.dense_a8_projections) hash_u64(hash, 0x47444e4441385631ULL); // GDNDA8V1
    if (config.dense_trellis_projections) hash_u64(hash, 0x544c473146503453ULL); // TLG1FP4S
    if (config.recurrent_layout != HipGdnRecurrentLayout::baseline) {
        hash_u64(hash, 0x47444e4c44533031ULL); // GDN LDS organization version 1.
        hash_u64(hash, static_cast<std::uint32_t>(config.recurrent_layout));
    }
    if (config.recurrent_storage != GdnRecurrentStorage::key_value) {
        hash_u64(hash, 0x47444e53544f5231ULL);
        hash_u64(hash, static_cast<std::uint32_t>(config.recurrent_storage));
    }
    return hash;
}

inline void validate_projection_id(std::string_view selected_id,
    const HipGdnLauncherConfig& config) {
    constexpr std::string_view output_suffix=".virtual256.q8-output";
    if(selected_id.ends_with(output_suffix)!=config.virtual_q8_output_projection)
        throw std::invalid_argument("HIP GDN selected kernel disagrees with virtual Q8 output configuration");
    if(config.virtual_q8_output_projection) selected_id.remove_suffix(output_suffix.size());
    if (selected_id.ends_with(".virtual256.q8-input") !=
        config.virtual_q8_input_projections)
        throw std::invalid_argument("HIP GDN selected kernel disagrees with virtual Q8 configuration");
    if (selected_id.ends_with(".prefetch8.virtual256.q8-input") != config.prefetch_q8_input_projections)
        throw std::invalid_argument("HIP GDN selected kernel disagrees with prefetch Q8 configuration");
}

} // namespace strix::hip_gdn_identity
