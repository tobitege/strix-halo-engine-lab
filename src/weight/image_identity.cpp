#include <strix/weight/cache.hpp>
#include <strix/weight/source.hpp>
namespace strix::weight {
namespace {
struct Identity {
    std::vector<Byte> bytes;
    void number(std::uint64_t n) { for (int i = 0; i < 8; ++i) { bytes.push_back(static_cast<Byte>(n & 255)); n >>= 8; } }
    void string(std::string_view s) { number(s.size()); for (auto c : s) bytes.push_back(static_cast<Byte>(c)); }
    void digest(Digest256 d) { for (auto c : d.bytes) bytes.push_back(static_cast<Byte>(c)); }
};
}
Status sidecar_identity(const WeightView& w, Digest256& out) {
    try {
        Identity b; b.number(w.sidecars.size());
        for (const auto& s : w.sidecars) {
            if (!s.data.source || s.data.range.offset > s.data.source->size() || s.data.range.bytes > s.data.source->size() - s.data.range.offset)
                return {Error::malformed, "sidecar range is outside its source"};
            b.string(s.name); b.number(static_cast<unsigned>(s.scalar)); b.number(s.shape.rank);
            for (auto d : s.shape.dims) b.number(d);
            b.number(s.data.range.offset); b.number(s.data.range.bytes); b.digest(s.data.source->identity());
        }
        out = digest_bytes(b.bytes);
        return {Error::ok, {}};
    } catch (...) { return {Error::stale_source, "cannot establish sidecar content identity"}; }
}
Status make_image_key(const WeightView& w, const Plan& p, std::string arch, std::string abi, ImageKey& out) {
    if (!w.payload.source || arch.empty() || abi.empty() || !(w.storage == p.selected.storage))
        return {Error::malformed, "image identity needs source, architecture, ABI and matching storage"};
    if (w.payload.range.offset > w.payload.source->size() || w.payload.range.bytes > w.payload.source->size()-w.payload.range.offset || w.shape.rank>4)
        return {Error::malformed, "payload range or shape is invalid"};
    try {
        ImageKey key{}; key.source = w.payload.source->identity();
        auto status = sidecar_identity(w, key.sidecars); if (!status) return status;
        Identity metadata; metadata.number(w.parameters.bits); metadata.number(w.parameters.group_size);
        metadata.string(w.parameters.scale_encoding); metadata.string(w.parameters.codebook_encoding);
        metadata.string(w.parameters.transform); metadata.digest(w.parameters.semantic_metadata);
        key.metadata = digest_bytes(metadata.bytes);
        key.tensor = w.tensor; key.payload = w.payload.range; key.byte_strides = w.byte_strides;
        key.has_byte_strides = w.has_byte_strides; key.shape = w.shape; key.storage = w.storage;
        key.image = p.selected.image; key.arithmetic = p.selected.arithmetic; key.backend = p.selected.backend;
        key.arch = std::move(arch); key.kernel_abi = std::move(abi); out = std::move(key);
        return {Error::ok, {}};
    } catch (...) { return {Error::stale_source, "cannot establish source content identity"}; }
}
}
