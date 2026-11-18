#include "strix/vision/api_media.hpp"

#include "strix/vision/embed_span.hpp"

#include <algorithm>
#include <chrono>
#include <span>
#include <stdexcept>
#include <utility>

namespace strix::vision {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// The client's error for an image this server cannot take (400; 413 for bytes, as serve::media).
[[noreturn]] void refuse_image(const ImageError& e) {
    switch (e.code()) {
    case ImageErrorCode::unsupported_format:
        throw serve::media::MediaError(400, "this image format is not supported by the server's decoder (send PNG or JPEG; WebP and GIF are refused)", "image");
    case ImageErrorCode::too_many_bytes:
        throw serve::media::MediaError(413, std::string("image too large: ") + e.what(), "image");
    case ImageErrorCode::too_many_pixels:
        throw serve::media::MediaError(400, std::string("image has too many pixels: ") + e.what(), "image");
    default:
        throw serve::media::MediaError(400, std::string("image could not be decoded: ") + e.what(), "image");
    }
}

}  // namespace

ApiMediaBridge::ApiMediaBridge(VisionAdapter& adapter, EncoderBackend& backend, const EncoderWeights& weights, EmbedCache* cache,
                               session::EmbedInterner& interner, ApiMediaOptions options)
    : adapter_(adapter), backend_(backend), weights_(weights), cache_(cache), interner_(interner), options_(std::move(options)) {
    if (options_.placeholder.empty() || options_.max_rows == 0U || options_.max_rows > session::kEmbedMaxRows || options_.lm_width == 0U)
        throw std::invalid_argument("vision media bridge: placeholder, max_rows and lm_width are required");
    // the cache key and the capability hash: the adapter's identity (weights digest + recipe) and the arithmetic
    identity_ = adapter_.identity() + "|" + backend_.arithmetic();  // the EmbedCache key of encode_cached
}

ApiMediaBridge::Stats ApiMediaBridge::stats() const {
    const std::lock_guard lock(mutex_);
    return stats_;
}

serve::ApiMediaSpan ApiMediaBridge::encode(const serve::media::MediaItem& item, std::uint32_t row_budget) {
    const std::lock_guard lock(mutex_);
    session::EmbedDigest digest{};
    static_assert(sizeof(item.digest) == sizeof(digest), "media digest is a SHA-256");
    std::copy(std::begin(item.digest), std::end(item.digest), digest.begin());
    const auto t0 = Clock::now();
    const auto budget = std::min(row_budget, options_.max_rows);
    const auto over_budget = [&](std::size_t rows) {
        ++stats_.refused;
        throw serve::media::MediaError(400, "image needs " + std::to_string(rows) + " rows; the request has room for " + std::to_string(budget), "image");
    };
    serve::ApiMediaSpan span;
    session::EmbedLease lease;
    const auto known = known_.find(digest);
    if (known != known_.end()) {
        if (known->second.layout.role.size() > budget) over_budget(known->second.layout.role.size());
        lease = interner_.find(digest);
        if (lease) {
            ++stats_.lease_hits;
        } else if (cache_) {
            if (const auto rows = cache_->find(identity_, digest)) {
                lease = interner_.intern(span_embed_rows(known->second.layout, *rows, digest, options_.axes));
                ++stats_.cache_hits;
            }
        }
        if (lease) {
            span.grid_h = known->second.grid_h;
            span.grid_w = known->second.grid_w;
            span.cache_hit = true;
            span.preprocess_ms = ms_since(t0);
        }
    }
    if (!lease) {
        Image image;
        try {
            image = decode(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(item.bytes.data()), item.bytes.size()), options_.limits);
        } catch (const ImageError& e) {
            ++stats_.refused;
            refuse_image(e);
        }
        const PatchGrid grid = adapter_.patches(image);
        Known k;
        k.grid_h = grid.grid_h;
        k.grid_w = grid.grid_w;
        k.layout = adapter_.layout(grid);
        if (k.layout.role.size() > budget) over_budget(k.layout.role.size());
        span.preprocess_ms = ms_since(t0);
        if (!started_) {
            try {
                if (options_.on_first_encode) options_.on_first_encode();
            } catch (...) {
                if (options_.on_first_encode_failure) options_.on_first_encode_failure();
                throw;
            }
            started_ = true;
        }
        const auto t1 = Clock::now();
        auto encoded = encode_cached(adapter_, backend_, weights_, cache_, digest, grid);
        span.encode_ms = ms_since(t1);
        if (!encoded.rows || encoded.rows->width != options_.lm_width)
            throw std::runtime_error("vision media bridge: the adapter's rows do not have the LM width");
        lease = interner_.intern(span_embed_rows(k.layout, *encoded.rows, digest, options_.axes));
        span.cache_hit = encoded.hit;
        span.grid_h = k.grid_h;
        span.grid_w = k.grid_w;
        if (encoded.hit) ++stats_.cache_hits;
        else ++stats_.encodes;
        if (known_.size() >= kKnownCap) known_.clear();
        known_[digest] = std::move(k);
    }
    span.layout = lease.ids();
    span.lease = std::move(lease);
    return span;
}

}  // namespace strix::vision
