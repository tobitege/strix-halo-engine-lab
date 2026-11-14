#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include "gemma4/contracts/g4_types.hpp"
#include <algorithm>

namespace strix::gemma4::memory {
// Payload/backing allocations only. No allocator metadata, driver reservation,
// queue/event internals or process RSS. Shared owners are charged once by their
// owning envelope, never once per alias.
inline std::uint64_t product(std::uint64_t a, std::uint64_t b) {
    if (b && a > UINT64_MAX / b) throw std::overflow_error("native memory product overflow");
    return a * b;
}
inline void add(std::uint64_t& n, std::uint64_t count, std::uint64_t size = 1) {
    const auto bytes = product(count,size);
    if (bytes > UINT64_MAX - n) throw std::overflow_error("native memory sum overflow");
    n += bytes;
}
struct Payload {
    std::uint64_t host = 0, device = 0, pinned = 0;
    [[nodiscard]] std::uint64_t total() const {
        auto n=host; add(n,device); add(n,pinned); return n;
    }
};
struct Work {
    Payload active; // queried existing buffer payload; not a complete service/RSS measure
    Payload additional; // simultaneous additional allocations, before growth
};
inline Payload stream_payload(std::uint64_t objects, std::uint64_t config_capacity,
                              std::uint64_t config_element, std::uint64_t cache_capacity,
                              std::uint64_t cache_element, std::uint64_t overlay_capacity,
                              std::uint64_t overlay_element, std::uint64_t publication,
                              bool pinned) {
    Payload n;n.host=objects;
    add(n.host,config_capacity,config_element);
    add(n.host,cache_capacity,cache_element);
    add(n.host,overlay_capacity,overlay_element);
    n.device=publication;if(pinned)n.pinned=publication;return n;
}
inline void stream_arrays(Payload& n, std::uint64_t k, std::uint64_t v, std::uint64_t tags) {
    add(n.device,k);add(n.device,v);add(n.device,tags);
}
struct StreamLayout {
    std::uint64_t objects, transaction, config_entry, cache_entry, overlay_entry, publication_entry;
};
inline std::optional<Work> exact_stream_bound(const ModelConfig& c,std::uint32_t admitted,StreamLayout s) {
#if !defined(_MSC_VER) && !defined(__GLIBCXX__)
    return {};
#else
    if(!admitted || admitted>c.max_positions || !c.window || !c.kv_heads || !c.layers || c.layer.size()!=c.layers) return {};
    for(const auto& l:c.layer) if(!l.head_dim || l.producer>=c.layers ||
        (l.attention!=AttentionKind::global && l.attention!=AttentionKind::sliding))return {};
    try {
        Work n;n.additional.host=s.objects;add(n.additional.host,s.transaction);
        auto descriptors=s.config_entry;add(descriptors,s.cache_entry);add(descriptors,s.overlay_entry,2);
        add(n.additional.host,product(c.layers,2),descriptors);
        add(n.additional.device,c.layers,s.publication_entry);
        add(n.additional.pinned,c.layers,s.publication_entry);
        add(n.additional.host,product(c.layers,2),s.publication_entry);
        std::uint64_t tag_stage=0;
        for(std::uint32_t l=0;l<c.layers;++l) if(c.layer[l].producer==l) {
            if(!c.layer[l].head_dim) return {};
            const auto width=product(c.kv_heads,c.layer[l].head_dim);
            const auto slots=c.layer[l].attention==AttentionKind::global?admitted:c.window;
            add(n.additional.device,product(slots,width),2*sizeof(float));
            add(n.additional.device,slots,sizeof(std::uint64_t));
            add(n.additional.device,width,8*2*sizeof(float));
            tag_stage=std::max(tag_stage,product(slots,2*sizeof(std::uint64_t)));
        }
        add(n.additional.host,tag_stage);
        (void)n.additional.total();return n;
    } catch(const std::overflow_error&) {return {};}
#endif
}
// Exact decode target attention scratch geometry (target.hip). The pass accepts
// at most eight rows. Rounding is storage only and does not change reductions.
struct Attention {
    std::uint64_t scores = 0, maxima = 0, partials = 0;
};
inline Attention attention(std::uint64_t context, std::uint64_t heads,
                           std::uint64_t rows, std::uint64_t chunk) {
    if (!context || !heads || !rows || rows>8 || !chunk)
        throw std::invalid_argument("unknown native attention geometry");
    auto rounded=context; add(rounded,chunk-1);
    const auto chunks=rounded/chunk;
    const auto rh=product(rows,heads);
    auto maxima_chunks=chunks;add(maxima_chunks,1);
    return {product(product(rh,product(chunks,chunk)),sizeof(float)),
            product(product(rh,maxima_chunks),sizeof(float)),
            product(product(product(rh,chunks),513),sizeof(float))};
}
inline bool fits(std::optional<std::uint64_t> work, std::uint64_t payload,
                 std::uint64_t copies, std::uint64_t limit) noexcept {
    if (!work || !*work || !payload || !copies || !limit || payload==UINT64_MAX || limit==UINT64_MAX) return false;
    try {auto n=product(payload,copies);add(n,*work);return n<=limit;}
    catch (...) {return false;}
}
} // namespace strix::gemma4::memory
