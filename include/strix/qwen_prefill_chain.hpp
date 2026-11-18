#pragma once
// Qwen binding of the shared heterogeneous chain planner (strix/pair/chain.hpp): the model adds only its
// dimensions and state layout, every device / link / block decision is the shared planner's.
//  - seam: 25,616 B per row at a whole- or half-layer cut (qwen_prefill_pipeline.hpp Seam).
//  - state crossing to the decoder: QSA layers (layer % 4 == 3) carry KV + indices 2,304 B/row; the MTP held
//    layer adds 2,304 B/row and 20,480 B fixed on the last stage; each GDN layer carries a recurrent matrix and a
//    convolution history of fixed size; layer 1 (PLE) a 10,240 x 9 BF16 convolution (src/cuda/models/qwen/qwen_prefill_driver.cu
//    export: recurrent_matrix / gdn_convolution / ple_convolution / mtp.held). Total 29,952 B/row + ~116.4 MB.
#include "strix/pair/chain.hpp"
#include "strix/qwen_prefill_pipeline.hpp"

namespace strix::qwen_pp {
inline constexpr std::uint32_t chain_layers = 48U;
inline constexpr double qsa_state_bytes_per_row = 2304.0;
inline constexpr double mtp_state_bytes_per_row = 2304.0;
inline constexpr double gdn_state_fixed_bytes = 3227648.0; // recurrent matrix + convolution history per GDN layer
inline pair::chain::ModelShape chain_shape(double gdn_fixed_bytes = gdn_state_fixed_bytes) {
    pair::chain::ModelShape shape;
    shape.layers = chain_layers;
    shape.seam_bytes_per_row = static_cast<double>(Seam::row_bytes);
    shape.state_bytes_per_row.assign(chain_layers, 0.0);
    shape.state_fixed_bytes.assign(chain_layers, 0.0);
    for (std::uint32_t layer = 0; layer < chain_layers; ++layer) {
        if (layer % 4U == 3U) shape.state_bytes_per_row[layer] = qsa_state_bytes_per_row;
        else shape.state_fixed_bytes[layer] = gdn_fixed_bytes;
    }
    shape.state_fixed_bytes[1] += 10240.0 * 9.0 * 2.0;                 // PLE convolution history
    shape.state_bytes_per_row[chain_layers - 1U] += mtp_state_bytes_per_row; // MTP KV of the head owner
    shape.state_fixed_bytes[chain_layers - 1U] += 10240.0 * 2.0;       // MTP held row
    return shape;
}
// A seed device from a whole-model rate (rows/s) and the shape of a measured per-layer cost vector
// (qwen_prefill_balance.hpp LayerCosts::layers of the HIP node). Search seed only, never a qualified cost.
inline pair::chain::Device chain_seed_device(std::string name, double rows_per_s, std::span<const double> shape = {},
    double block_fixed_ms = 0.0) {
    pair::chain::Device device;
    device.name = std::move(name);
    device.layer_ms_per_row = pair::chain::scaled_costs(rows_per_s, chain_layers, shape);
    device.block_fixed_ms = block_fixed_ms;
    return device;
}
} // namespace strix::qwen_pp
