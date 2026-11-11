#include "deployment.hpp"
#include "strix/serve/json.hpp"
#include <fstream>
#include <set>

namespace strix::minimax::engine {
DeploymentModel deployment_model(const std::filesystem::path& directory) {
    using deploy::checked_add; using deploy::checked_mul;
    DeploymentModel m; std::set<std::string> names;
    bool embedding = false, head = false;
    for (const auto& file : std::filesystem::directory_iterator(directory)) {
        if (!file.is_regular_file() || file.path().extension() != ".safetensors") continue;
        std::ifstream f(file.path(), std::ios::binary);
        std::array<unsigned char, 8> raw{}; f.read(reinterpret_cast<char*>(raw.data()), 8);
        std::uint64_t n = 0;
        for (unsigned i = 0; i < 8; ++i) n |= std::uint64_t(raw[i]) << (8*i);
        if (!f || !n || n > (64ULL << 20) || n > file.file_size()-8)
            throw std::invalid_argument("MiniMax invalid safetensors header length");
        std::string text(static_cast<std::size_t>(n), '\0'); f.read(text.data(), static_cast<std::streamsize>(n));
        if (!f) throw std::invalid_argument("MiniMax truncated safetensors header");
        const auto json = serve::parse_json_unique(text);
        for (const auto& [name, t] : json.as_object()) {
            if (name == "__metadata__") continue;
            if (!names.insert(name).second) throw std::invalid_argument("MiniMax duplicate tensor header");
            const auto field = [&](const char* key) -> const serve::Json& {
                const auto* value = t.find(key);
                if (!value) throw std::invalid_argument("MiniMax missing tensor header field");
                return *value;
            };
            const auto& offsets = field("data_offsets").as_array();
            if (offsets.size() != 2 || offsets[0].as_int() < 0 || offsets[1].as_int() < offsets[0].as_int())
                throw std::invalid_argument("MiniMax invalid tensor range");
            const auto end = static_cast<std::uint64_t>(offsets[1].as_int());
            if (end > file.file_size()-8-n) throw std::invalid_argument("MiniMax tensor outside file");
            const auto bytes = end-static_cast<std::uint64_t>(offsets[0].as_int());
            if (name.ends_with("embed_tokens.weight")) {
                const auto& shape = field("shape").as_array();
                if (shape.size() != 2 || shape[1].as_int() != m.hidden || field("dtype").as_string() != "BF16")
                    throw std::invalid_argument("MiniMax unsupported model dimensions");
                embedding = true;
            }
            head |= name.find("lm_head") != std::string::npos;
            // TP2 trellis words split in half. Count both scale vectors whole;
            // non-expert linears also retain raw prefill copies. Float owners
            // (norm/router/index) are conservatively widened to FP32.
            const bool expert = name.find("experts.") != std::string::npos;
            auto charged = name.ends_with(".trellis") ? checked_add(bytes,1)/2 : bytes;
            if (name.ends_with(".trellis") || name.ends_with(".suh") || name.ends_with(".svh")) {
                if (!expert) charged = checked_mul(charged,2);
            } else if (name.find("embed_tokens") == std::string::npos) charged = checked_mul(charged,2);
            m.weights = checked_add(m.weights,charged);
        }
    }
    if (!embedding || !head || !m.weights) throw std::invalid_argument("MiniMax missing target weight headers");
    // Packed + transposed index layouts and suballocator tail padding.
    m.weights = checked_add(m.weights, checked_add(2ULL<<30, checked_mul(m.layers, checked_mul(m.hidden, 384ULL*12))));
    return m;
}
deploy::Footprint FootprintProvider::footprint(const deploy::Spec&, const deploy::Candidate& c, std::uint32_t) const {
    using deploy::checked_add; using deploy::checked_mul;
    if (c.topology != "tp2" || c.ranks != 2) throw std::invalid_argument("MiniMax requires TP2 pair");
    if (!rows_ || rows_ > 4096) throw std::invalid_argument("MiniMax prefill workspace bound requires max-rows 1..4096");
    deploy::Footprint f; f.weights = m_.weights;
    f.kv_per_token = checked_mul(m_.layers, checked_mul(2, checked_add(checked_mul(2,m_.kv_width),m_.index_width)));
    // FP32 prefill expert/activation banks, nested-bank upper bound, MSA and
    // matmul scratch reserve. Deliberately upper bounds the supported row range.
    f.workspace = checked_add(scratch_extra_,checked_add(4ULL<<30, checked_mul(rows_, checked_mul(m_.hidden,128))));
    f.staging = 768ULL<<20;
    f.extra.emplace_back("rope_positions_and_control", checked_add(checked_mul(c.context,1024),64ULL<<20));
    if (c.format.prefill != "exl3") {
        // Three TP2 expert matrices, packed nibbles plus FP16 group scales.
        // Default dims: ~1.94 decimal GB per layer/rank; all EXL3 stays live.
        const auto elements = checked_mul(checked_mul(checked_mul(3,m_.experts),m_.hidden),m_.expert_inter)/2;
        const auto layer = std::max<std::uint64_t>(1940000000ULL,
            checked_add(elements/2, checked_mul((elements+63)/64,2)));
        f.extra.emplace_back("resident_int4_packs",checked_mul(layer,c.format.resident_layers));
        if (c.format.jit_slots) f.extra.emplace_back("int4_jit_two_slots",3865470567ULL); // ceil(3.6 GiB)
    }
    return f;
}
} // namespace strix::minimax::engine
