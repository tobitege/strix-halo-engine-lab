#pragma once

// The shared bridge from a family's vision::VisionAdapter to the API layer's serve::ApiMediaAdapter: decode ->
// adapter.patches -> encode_cached (EmbedCache) -> adapter.layout -> span_embed_rows -> EmbedInterner. One class
// for every family (wave4 vision spec 2a/2d; V7 open gap (a)); a family supplies its adapter, the encoder backend,
// the bound weights and the placeholder text of its chat template.
//
// A repeated image (same SHA-256 of its bytes) costs no decode, resize or patchify (V7 gap (b)): a live lease in the
// interner, or a cached encode whose span layout this bridge already holds, is used directly.
//
// Not thread-safe by itself: one ApiService calls encode() one request at a time (ApiMediaAdapter contract); the
// bridge serialises its own calls with a mutex so a second caller is safe too.

#include "strix/serve/api.hpp"
#include "strix/session/embed_rows.hpp"
#include "strix/vision/adapter.hpp"
#include "strix/vision/embed_cache.hpp"
#include "strix/vision/encoder.hpp"
#include "strix/vision/image.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>

namespace strix::vision {

struct ApiMediaOptions {
    std::string placeholder;      // text that tokenizes to exactly one id (the family's image token)
    std::uint32_t max_rows = 0;   // largest span of one image (the adapter's limit)
    std::uint32_t lm_width = 0;   // LM hidden width; rows of another width are refused at construction time of a span
    unsigned axes = 0;            // position axes the family reads (span_embed_rows)
    DecodeLimits limits;          // encoded bytes / decoded pixels per image
    // Called before the first encode (lazy weight upload, e.g. Encoder::load); retried after failure.
    std::function<void()> on_first_encode;
    // Roll back partial initialization if on_first_encode throws. Must not throw; may be empty.
    std::function<void()> on_first_encode_failure;
};

class ApiMediaBridge final : public serve::ApiMediaAdapter {
public:
    // All references must outlive the bridge. cache may be null (no cross-request reuse besides live leases).
    ApiMediaBridge(VisionAdapter& adapter, EncoderBackend& backend, const EncoderWeights& weights, EmbedCache* cache,
                   session::EmbedInterner& interner, ApiMediaOptions options);

    [[nodiscard]] std::string identity() const override { return identity_; }
    [[nodiscard]] std::string placeholder() const override { return options_.placeholder; }
    [[nodiscard]] std::uint32_t max_rows() const override { return options_.max_rows; }
    [[nodiscard]] serve::ApiMediaSpan encode(const serve::media::MediaItem& item, std::uint32_t row_budget) override;
    [[nodiscard]] std::uint64_t epoch() const override { return interner_.epoch(); }

    struct Stats {
        std::uint64_t encodes = 0, lease_hits = 0, cache_hits = 0, refused = 0;
    };
    [[nodiscard]] Stats stats() const;

private:
    struct Known {  // what a later hit needs without decoding the bytes again
        unsigned grid_h = 0, grid_w = 0;
        SpanLayout layout;
    };
    VisionAdapter& adapter_;
    EncoderBackend& backend_;
    const EncoderWeights& weights_;
    EmbedCache* cache_;
    session::EmbedInterner& interner_;
    ApiMediaOptions options_;
    std::string identity_;
    mutable std::mutex mutex_;
    bool started_ = false;
    std::map<session::EmbedDigest, Known> known_;  // bounded: cleared when it exceeds kKnownCap entries
    Stats stats_;
    static constexpr std::size_t kKnownCap = 4096;
};

}  // namespace strix::vision
