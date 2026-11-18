#include "model_graph/dense_adapter.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    try {
        strix::model::DenseConfig config;
        config.hidden = 32; config.layers = 2; config.heads = 4; config.kv_heads = 2;
        config.intermediate = 48; config.head_dim = 8; config.vocab = 4; config.context = 128;
        config.rope.dimensions = 8;
        const auto graph = strix::model_graph::dense_decoder_graph(config, "dense-test", "v1");
        if (graph.model != "dense-test" || graph.semantic_version != "v1" || graph.layers.size() != 2)
            throw std::runtime_error("dense graph metadata mismatch");
        config.heads = 0;
        try {
            (void)strix::model_graph::dense_decoder_graph(config, "dense-test", "v1");
        } catch (const std::runtime_error&) {
            return 0;
        }
        throw std::runtime_error("dense graph accepted invalid geometry");
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
