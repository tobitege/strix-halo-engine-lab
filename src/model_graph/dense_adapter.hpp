#pragma once
#include "model_graph.hpp"
#include "strix/dense_model_config.hpp"

namespace strix::model_graph {
// Shared dense decoder declaration. Backend core implementations bind these operations;
// family adapters supply names/semantics, never device kernels or architecture choices.
[[nodiscard]] ModelGraph dense_decoder_graph(const model::DenseConfig& config,
                                             std::string family, std::string semantic_version);
} // namespace strix::model_graph
