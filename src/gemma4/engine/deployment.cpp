#include "deployment.hpp"
#include "strix/gguf_reader.hpp"

namespace strix::gemma4::engine {
DeploymentModel deployment_model(const std::filesystem::path& path) {
    const auto info = gguf::read_file(path);
    const auto number = [&](const std::string& key) {
        const auto* e = info.find_metadata(key);
        if (!e || !e->unsigned_value || *e->unsigned_value > UINT32_MAX)
            throw std::invalid_argument("Gemma deployment missing/invalid metadata: " + key);
        return static_cast<std::uint32_t>(*e->unsigned_value);
    };
    const auto* arch = info.find_metadata("general.architecture");
    if (!arch || arch->string_value != "gemma4" || number("general.file_type") != 15)
        throw std::invalid_argument("Gemma deployment requires gemma4 Q4_K_M GGUF");
    DeploymentModel m;
    m.hidden = number("gemma4.embedding_length"); m.layers = number("gemma4.block_count");
    m.inter = number("gemma4.feed_forward_length"); m.kv_heads = number("gemma4.attention.head_count_kv");
    m.window = number("gemma4.attention.sliding_window"); m.max_context = number("gemma4.context_length");
    if (m.hidden != 2560 || m.layers != 42 || m.inter != 10240 || m.kv_heads != 2 || m.window != 512 || m.max_context != 65536)
        throw std::invalid_argument("Gemma deployment unsupported native model dimensions");
    for (const auto& t : info.tensors) {
        m.weights = deploy::checked_add(m.weights,t.byte_size);
        m.largest_tensor = std::max(m.largest_tensor,t.byte_size);
        if (t.dimensions.size() >= 2) {
            std::uint64_t elements = 1;
            for (const auto dim : t.dimensions)
                elements = deploy::checked_mul(elements,deploy::checked_mul(deploy::checked_add(dim,127)/128,128));
            // Two FP16 panels bound both FP16 and hilo prepared routes.
            m.prefill_weights_upper_bound = deploy::checked_add(m.prefill_weights_upper_bound,deploy::checked_mul(elements,4));
        }
    }
    if (!m.weights) throw std::invalid_argument("Gemma deployment missing tensor headers");
    return m;
}
deploy::Footprint FootprintProvider::footprint(const deploy::Spec&, const deploy::Candidate& c, std::uint32_t) const {
    using deploy::checked_add; using deploy::checked_mul;
    if (!c.context || c.context > m_.max_context) throw std::invalid_argument("Gemma context exceeds native limit");
    deploy::Footprint f; f.weights = m_.weights;
    // Only producer layers own rings; shared-KV consumers alias them. FP32 K/V
    // plus uint64 position tags, exactly as the API's state byte accounting.
    f.kv_per_token = checked_mul(m_.global_producers, checked_add(checked_mul(8,checked_mul(m_.kv_heads,512)),8));
    const auto sliding = checked_mul(checked_mul(m_.sliding_producers,m_.window),checked_add(checked_mul(8,checked_mul(m_.kv_heads,256)),8));
    const auto state = checked_add(sliding,checked_mul(c.context,f.kv_per_token));
    // Committed state plus unpublished transaction, spare state and prefix LRU.
    f.kv_total = checked_mul(checked_mul(c.slots+1ULL,2),state);
    f.extra.emplace_back("prefix_cache_cap",prefix_bytes_);
    f.extra.emplace_back("decode_prepared_weights_upper_bound",m_.weights);
    if (prepared_) f.extra.emplace_back("prefill_prepared_weights_upper_bound",m_.prefill_weights_upper_bound);
    f.extra.emplace_back("rope_tables",checked_mul(m_.max_context,3072));
    f.workspace = checked_add(2ULL<<30,checked_mul(512,checked_mul(checked_add(m_.inter,m_.hidden),64)));
    f.staging = m_.largest_tensor;
    return f;
}
} // namespace strix::gemma4::engine
