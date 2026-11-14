#pragma once

#include "strix/serve/json.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// API media parts: one normalizer for the image shapes of Chat Completions, Messages and Responses.
// It checks the fetch policy and the limits, reads the image size from the file header (nothing is
// decoded here) and rewrites every accepted image part to {"type":"image","media":k}, where k indexes
// the returned items in prompt order. A request without image parts is returned as the same value.
namespace strix::serve::media {

enum class Api : std::uint8_t { chat_completions, messages, responses };
enum class Format : std::uint8_t { png, jpeg, webp };

// status 400 or 413; the error type is always invalid_request_error. param is the path of the part.
struct MediaError : std::runtime_error {
    MediaError(int http_status, const std::string& message, std::string where)
        : std::runtime_error(message), status(http_status), param(std::move(where)) {}
    int status;
    std::string param;
};

struct Limits {
    std::size_t max_images = 16;                       // per request
    std::size_t max_image_bytes = 20U * 1024U * 1024U; // encoded file, per image -> 413
    std::uint64_t max_pixels = 40'000'000;             // width * height from the header, per image
};

enum class FetchMode : std::uint8_t { off, allowlist };

struct FetchPolicy {
    FetchMode mode = FetchMode::off; // data: URLs are always accepted
    std::vector<std::string> hosts;  // lower case, exact match
};

// mode: "", "off" or "allowlist"; hosts: comma separated names. Anything else throws std::invalid_argument.
[[nodiscard]] FetchPolicy parse_fetch_policy(std::string_view mode, std::string_view hosts);
// STRIX_SERVE_IMAGE_FETCH and STRIX_SERVE_IMAGE_HOSTS.
[[nodiscard]] FetchPolicy fetch_policy_from_env();

struct RemoteUrl {
    std::string host; // lower case, without brackets
    std::uint16_t port = 443;
    std::string target; // path and query, starts with '/'
};

// What a fetcher has to enforce: HTTPS only, no redirects, the byte cap while reading, the timeout, and
// address_is_public() on the address it actually connects to (a name can resolve to a private address).
struct FetchRules {
    std::size_t max_bytes = 0;
    unsigned timeout_ms = 5000;
    bool follow_redirects = false;
};

// Returns the response body, or throws; any exception becomes a 400 for that part.
using Fetcher = std::function<std::string(const RemoteUrl&, const FetchRules&)>;

struct Options {
    Limits limits;
    FetchPolicy fetch;
    Fetcher fetcher; // empty: remote URLs are refused even when the policy admits the host
};

using Digest = std::array<std::uint8_t, 32>;

struct MediaItem {
    std::string bytes;      // the encoded file
    Format format = Format::png;
    std::string media_type; // from the magic bytes, never from the request
    std::uint32_t width = 0, height = 0;
    Digest digest{};        // SHA-256 of bytes: the fast pre-key, not the pixel identity
    std::string detail;     // "" when the request gave none
    std::string param;      // path of the part in the request
};

struct Normalized {
    Json body;
    std::vector<MediaItem> items;
};

// Images are taken from user turns and tool results. Parts of other roles and part types this layer
// does not own (text, video, files) are left as they are, so their existing handling stays in force.
[[nodiscard]] Normalized normalize(const Json& body, Api api, const Options& options = {});

// Pieces, exposed for callers and tests.
struct ImageInfo {
    Format format = Format::png;
    std::uint32_t width = 0, height = 0;
};
// Magic bytes and header size only; throws MediaError(400) on an unknown or truncated file.
[[nodiscard]] ImageInfo probe_image(std::string_view bytes, std::string_view param = {});
[[nodiscard]] std::string_view media_type(Format format) noexcept;
// Standard alphabet, padding optional, ASCII whitespace skipped. max_bytes bounds the output (413).
[[nodiscard]] std::string decode_base64(std::string_view text, std::size_t max_bytes, std::string_view param = {});
[[nodiscard]] Digest sha256(std::string_view bytes) noexcept;
[[nodiscard]] std::string hex(const Digest& digest);
// IPv4 or IPv6 literal. False for loopback, private, link-local, unique-local, shared (100.64/10),
// unspecified, multicast, reserved and IPv4-mapped/NAT64 forms of those, and for anything unparsable.
[[nodiscard]] bool address_is_public(std::string_view literal) noexcept;

} // namespace strix::serve::media
