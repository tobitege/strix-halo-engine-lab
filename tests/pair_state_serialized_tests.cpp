#include "strix/pair/state_decls.hpp"
#include "strix/ds41_attention.hpp"

#include <cstdio>

int main() {
    namespace ds = strix::deepseek41;
    namespace dc = strix::pair::state::decl;
    namespace attn = strix::ds41::attn;
    ds::AttentionState state;
    state.position = 32768;
    state.tokens.resize(state.position);
    for (const auto& plan : attn::released_plan()) {
        ds::AttentionLayerState layer;
        layer.ring.resize(136U * 528U);
        layer.owns_kv = plan.owns_kv;
        layer.owns_index = plan.owns_index;
        layer.ratio = plan.ratio;
        if (layer.owns_kv) {
            layer.kv.resize(state.position / layer.ratio * attn::kKvEntryBytes);
            layer.idx_keys.resize(state.position / layer.ratio * attn::kIdxEntryBytes);
            if (layer.ratio == 2) layer.carry.resize(1024);
        }
        if (layer.owns_index) layer.topk.resize(512);
        state.layers.push_back(std::move(layer));
    }
    int failures = 0;
    for (const unsigned topk : {512U, 7U, 0U}) {
        for (auto& layer : state.layers) {
            if (layer.owns_index) layer.topk.resize(topk);
            layer.topk_known = topk != 0;
        }
        const auto serialized = ds::serialize_attention_state(state);
        const auto declared = dc::ds41_serialized_bytes(state);
        // Independently pin the full fixture in addition to comparing every variable layout with write_state.
        if (declared != serialized.size() || (topk == 512 && declared != 32197432ULL)) {
            std::fprintf(stderr, "DS41 serialized size mismatch for topk=%u\n", topk);
            ++failures;
        }
        if (declared != state.bytes() + state.tokens.size() * 4U + 1848U) ++failures;
    }
    return failures == 0 ? 0 : 1;
}
