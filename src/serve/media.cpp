#include "strix/serve/media.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace strix::serve::media {
namespace {

[[nodiscard]] std::uint32_t u8(std::string_view s, std::size_t i) noexcept { return static_cast<unsigned char>(s[i]); }
[[nodiscard]] std::uint32_t be16(std::string_view s, std::size_t i) noexcept { return (u8(s, i) << 8U) | u8(s, i + 1); }
[[nodiscard]] std::uint32_t le16(std::string_view s, std::size_t i) noexcept { return u8(s, i) | (u8(s, i + 1) << 8U); }
[[nodiscard]] std::uint32_t le24(std::string_view s, std::size_t i) noexcept { return le16(s, i) | (u8(s, i + 2) << 16U); }
[[nodiscard]] std::uint32_t be32(std::string_view s, std::size_t i) noexcept { return (be16(s, i) << 16U) | be16(s, i + 2); }
[[nodiscard]] std::uint32_t le32(std::string_view s, std::size_t i) noexcept { return le16(s, i) | (le16(s, i + 2) << 16U); }

[[nodiscard]] char lower(char c) noexcept { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; }
[[nodiscard]] std::string lowered(std::string_view s) {
    std::string out(s);
    for (auto& c : out) c = lower(c);
    return out;
}
[[nodiscard]] bool starts_with_nocase(std::string_view s, std::string_view prefix) noexcept {
    if (s.size() < prefix.size()) return false;
    for (std::size_t i = 0; i < prefix.size(); ++i) if (lower(s[i]) != prefix[i]) return false;
    return true;
}
[[nodiscard]] bool ends_with_nocase(std::string_view s, std::string_view suffix) noexcept {
    return s.size() >= suffix.size() && starts_with_nocase(s.substr(s.size() - suffix.size()), suffix);
}

[[noreturn]] void bad(std::string_view param, const std::string& message, int status = 400) {
    throw MediaError(status, message, std::string(param));
}

// ---- addresses -------------------------------------------------------------------------------------------

[[nodiscard]] std::optional<std::array<std::uint8_t, 4>> parse_v4(std::string_view s) noexcept {
    std::array<std::uint8_t, 4> out{};
    std::size_t at = 0;
    for (std::size_t part = 0; part < 4; ++part) {
        if (part != 0) {
            if (at >= s.size() || s[at] != '.') return std::nullopt;
            ++at;
        }
        const auto begin = at;
        unsigned value = 0;
        while (at < s.size() && s[at] >= '0' && s[at] <= '9' && at - begin < 3) value = value * 10U + static_cast<unsigned>(s[at++] - '0');
        const auto digits = at - begin;
        if (digits == 0 || value > 255U || (digits > 1 && s[begin] == '0')) return std::nullopt;
        out[part] = static_cast<std::uint8_t>(value);
    }
    if (at != s.size()) return std::nullopt;
    return out;
}

[[nodiscard]] bool v4_public(const std::uint8_t* a) noexcept {
    const unsigned b0 = a[0], b1 = a[1], b2 = a[2];
    if (b0 == 0 || b0 == 10 || b0 == 127 || b0 >= 224) return false;
    if (b0 == 100 && (b1 & 0xC0U) == 64) return false;
    if (b0 == 169 && b1 == 254) return false;
    if (b0 == 172 && (b1 & 0xF0U) == 16) return false;
    if (b0 == 192 && b1 == 168) return false;
    if (b0 == 192 && b1 == 0 && (b2 == 0 || b2 == 2)) return false;
    if (b0 == 192 && b1 == 88 && b2 == 99) return false;
    if (b0 == 198 && (b1 == 18 || b1 == 19)) return false;
    if (b0 == 198 && b1 == 51 && b2 == 100) return false;
    if (b0 == 203 && b1 == 0 && b2 == 113) return false;
    return true;
}

[[nodiscard]] std::optional<std::array<std::uint8_t, 16>> parse_v6(std::string_view s) noexcept {
    std::array<std::uint8_t, 16> out{};
    std::array<std::uint16_t, 8> groups{};
    std::size_t count = 0, at = 0;
    int gap = -1;
    if (s.size() < 2) return std::nullopt;
    if (s[0] == ':') {
        if (s[1] != ':') return std::nullopt;
        gap = 0;
        at = 2;
    }
    while (at < s.size()) {
        if (count == 8) return std::nullopt;
        const auto begin = at;
        unsigned value = 0;
        while (at < s.size() && at - begin < 4) {
            const char c = lower(s[at]);
            unsigned digit = 0;
            if (c >= '0' && c <= '9') digit = static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') digit = static_cast<unsigned>(c - 'a') + 10U;
            else break;
            value = value * 16U + digit;
            ++at;
        }
        if (at < s.size() && s[at] == '.') { // dotted tail
            const auto tail = parse_v4(s.substr(begin));
            if (!tail || count > 6) return std::nullopt;
            groups[count++] = static_cast<std::uint16_t>(((*tail)[0] << 8U) | (*tail)[1]);
            groups[count++] = static_cast<std::uint16_t>(((*tail)[2] << 8U) | (*tail)[3]);
            at = s.size();
            break;
        }
        if (at == begin) return std::nullopt;
        groups[count++] = static_cast<std::uint16_t>(value);
        if (at == s.size()) break;
        if (s[at] != ':') return std::nullopt;
        ++at;
        if (at < s.size() && s[at] == ':') {
            if (gap >= 0) return std::nullopt;
            gap = static_cast<int>(count);
            ++at;
        } else if (at == s.size()) {
            return std::nullopt; // single trailing colon
        }
    }
    if (gap < 0 && count != 8) return std::nullopt;
    if (gap >= 0 && count >= 8) return std::nullopt;
    std::array<std::uint16_t, 8> full{};
    const auto head = gap < 0 ? count : static_cast<std::size_t>(gap);
    for (std::size_t i = 0; i < head; ++i) full[i] = groups[i];
    for (std::size_t i = head; i < count; ++i) full[8 - count + i] = groups[i];
    for (std::size_t i = 0; i < 8; ++i) {
        out[2 * i] = static_cast<std::uint8_t>(full[i] >> 8U);
        out[2 * i + 1] = static_cast<std::uint8_t>(full[i] & 0xFFU);
    }
    return out;
}

[[nodiscard]] bool v6_public(const std::array<std::uint8_t, 16>& a) noexcept {
    const auto zero = [&](std::size_t from, std::size_t to) {
        for (std::size_t i = from; i < to; ++i) if (a[i] != 0) return false;
        return true;
    };
    if (zero(0, 10) && a[10] == 0xFF && a[11] == 0xFF) return v4_public(a.data() + 12); // IPv4-mapped
    if (a[0] == 0 && a[1] == 0x64 && a[2] == 0xFF && a[3] == 0x9B && zero(4, 12)) return v4_public(a.data() + 12); // NAT64
    if (a[0] == 0x20 && a[1] == 0x02) return v4_public(a.data() + 2); // 6to4
    if ((a[0] & 0xE0U) != 0x20) return false; // only global unicast 2000::/3 can be public
    if (a[0] == 0x20 && a[1] == 0x01 && a[2] == 0 && a[3] == 0) return false;       // Teredo
    if (a[0] == 0x20 && a[1] == 0x01 && a[2] == 0x0D && a[3] == 0xB8) return false; // documentation
    return true;
}

// A host whose last label is a number is an IPv4 address to URL parsers, whatever its spelling.
[[nodiscard]] bool looks_numeric(std::string_view host) noexcept {
    auto last = host;
    if (!last.empty() && last.back() == '.') last.remove_suffix(1);
    if (const auto dot = last.rfind('.'); dot != std::string_view::npos) last.remove_prefix(dot + 1);
    if (last.empty()) return true;
    if (starts_with_nocase(last, "0x")) return true;
    return std::all_of(last.begin(), last.end(), [](char c) { return c >= '0' && c <= '9'; });
}

// ---- URLs ------------------------------------------------------------------------------------------------

[[nodiscard]] RemoteUrl parse_https(std::string_view rest, std::string_view param) {
    for (const char c : rest) if (static_cast<unsigned char>(c) <= 0x20 || c == 0x7F || c == '\\') bad(param, "image URL has an invalid character");
    const auto end = rest.find_first_of("/?#");
    auto authority = rest.substr(0, end);
    RemoteUrl url;
    if (end != std::string_view::npos) {
        auto tail = rest.substr(end);
        if (const auto hash = tail.find('#'); hash != std::string_view::npos) tail = tail.substr(0, hash);
        url.target = tail.empty() || tail.front() != '/' ? "/" + std::string(tail) : std::string(tail);
    } else {
        url.target = "/";
    }
    if (authority.find('@') != std::string_view::npos) bad(param, "image URL must not carry credentials");
    std::string_view port;
    if (!authority.empty() && authority.front() == '[') {
        const auto close = authority.find(']');
        if (close == std::string_view::npos) bad(param, "image URL has an invalid host");
        url.host = lowered(authority.substr(1, close - 1));
        const auto after = authority.substr(close + 1);
        if (!after.empty()) {
            if (after.front() != ':') bad(param, "image URL has an invalid host");
            port = after.substr(1);
        }
        const auto address = parse_v6(url.host);
        if (!address || !v6_public(*address)) bad(param, "image URL host is not a public address");
    } else {
        if (const auto colon = authority.find(':'); colon != std::string_view::npos) {
            port = authority.substr(colon + 1);
            authority = authority.substr(0, colon);
        }
        url.host = lowered(authority);
        if (url.host.empty()) bad(param, "image URL has an invalid host");
        if (looks_numeric(url.host)) {
            const auto address = parse_v4(url.host);
            if (!address || !v4_public(address->data())) bad(param, "image URL host is not a public address");
        }
    }
    if (!port.empty()) {
        unsigned value = 0;
        for (const char c : port) {
            if (c < '0' || c > '9' || value > 65535U) bad(param, "image URL has an invalid port");
            value = value * 10U + static_cast<unsigned>(c - '0');
        }
        if (value == 0 || value > 65535U) bad(param, "image URL has an invalid port");
        url.port = static_cast<std::uint16_t>(value);
    }
    return url;
}

// ---- SHA-256 ---------------------------------------------------------------------------------------------

constexpr std::array<std::uint32_t, 64> sha_k = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

[[nodiscard]] constexpr std::uint32_t rotr(std::uint32_t x, unsigned n) noexcept { return (x >> n) | (x << (32U - n)); }

void sha_block(std::array<std::uint32_t, 8>& h, const unsigned char* p) noexcept {
    std::array<std::uint32_t, 64> w{};
    for (std::size_t i = 0; i < 16; ++i)
        w[i] = (static_cast<std::uint32_t>(p[4 * i]) << 24U) | (static_cast<std::uint32_t>(p[4 * i + 1]) << 16U) |
               (static_cast<std::uint32_t>(p[4 * i + 2]) << 8U) | static_cast<std::uint32_t>(p[4 * i + 3]);
    for (std::size_t i = 16; i < 64; ++i) {
        const auto s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3U);
        const auto s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10U);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    auto v = h;
    for (std::size_t i = 0; i < 64; ++i) {
        const auto s1 = rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25);
        const auto ch = (v[4] & v[5]) ^ (~v[4] & v[6]);
        const auto t1 = v[7] + s1 + ch + sha_k[i] + w[i];
        const auto s0 = rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22);
        const auto maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
        const auto t2 = s0 + maj;
        v[7] = v[6]; v[6] = v[5]; v[5] = v[4]; v[4] = v[3] + t1;
        v[3] = v[2]; v[2] = v[1]; v[1] = v[0]; v[0] = t1 + t2;
    }
    for (std::size_t i = 0; i < 8; ++i) h[i] += v[i];
}

// ---- request walk ----------------------------------------------------------------------------------------

struct Walk {
    Api api;
    const Options& options;
    std::vector<MediaItem> items;

    [[nodiscard]] std::string fetch(std::string_view url, std::string_view param) const {
        const auto cap = options.limits.max_image_bytes;
        if (starts_with_nocase(url, "data:")) {
            const auto comma = url.find(',');
            if (comma == std::string_view::npos) bad(param, "data URL has no payload");
            if (!ends_with_nocase(url.substr(5, comma - 5), ";base64")) bad(param, "data URL must be base64 encoded");
            return decode_base64(url.substr(comma + 1), cap, param);
        }
        const bool https = starts_with_nocase(url, "https://");
        if (!https && !starts_with_nocase(url, "http://")) bad(param, "unsupported image URL scheme; send a base64 data: URL");
        if (options.fetch.mode == FetchMode::off) bad(param, "remote image URLs are disabled; send a base64 data: URL");
        if (!https) bad(param, "remote image URLs must use https");
        const auto remote = parse_https(url.substr(8), param);
        const auto& hosts = options.fetch.hosts;
        if (std::find(hosts.begin(), hosts.end(), remote.host) == hosts.end()) bad(param, "image URL host is not on the allowlist");
        if (!options.fetcher) bad(param, "remote image fetch is not available on this server");
        std::string body;
        try {
            body = options.fetcher(remote, FetchRules{cap, 5000, false});
        } catch (const MediaError&) {
            throw;
        } catch (const std::exception& e) {
            bad(param, std::string("image fetch failed: ") + e.what());
        }
        if (body.size() > cap) bad(param, "image is larger than the per-image byte limit", 413);
        return body;
    }

    [[nodiscard]] Json add(std::string bytes, const Json* detail, const std::string& param) {
        MediaItem item;
        if (detail && !detail->is_null()) {
            if (!detail->is_string()) bad(param, "image detail must be a string");
            item.detail = detail->as_string();
        }
        const auto info = probe_image(bytes, param);
        if (std::uint64_t{info.width} * info.height > options.limits.max_pixels) bad(param, "image has more pixels than the per-image limit");
        item.format = info.format;
        item.media_type = std::string(media_type(info.format));
        item.width = info.width;
        item.height = info.height;
        item.digest = sha256(bytes);
        item.bytes = std::move(bytes);
        item.param = param;
        const auto index = static_cast<std::int64_t>(items.size());
        items.push_back(std::move(item));
        return Json(Json::Object{{"type", Json("image")}, {"media", Json(index)}});
    }

    void admit(const std::string& param) const {
        if (items.size() >= options.limits.max_images) bad(param, "at most " + std::to_string(options.limits.max_images) + " images per request");
    }

    [[nodiscard]] static std::string_view string_field(const Json& object, std::string_view key, const std::string& param, const char* what) {
        const auto* value = object.find(key);
        if (!value || !value->is_string()) bad(param, std::string(what) + " must be a string");
        return value->as_string();
    }

    // Returns true and fills out when the part was rewritten.
    [[nodiscard]] bool part(const Json& in, const std::string& param, Json& out) {
        if (!in.is_object()) return false;
        const auto* type_value = in.find("type");
        if (!type_value || !type_value->is_string()) return false;
        const auto& type = type_value->as_string();
        if (type == "image" && in.find("media")) bad(param, "media is a reserved member of an image part");
        if (api == Api::chat_completions && type == "image_url") {
            admit(param);
            const auto* image = in.find("image_url");
            if (image && image->is_string()) { out = add(fetch(image->as_string(), param + ".image_url"), nullptr, param); return true; }
            if (!image || !image->is_object()) bad(param + ".image_url", "image_url must be a string or an object with a url");
            const auto url = string_field(*image, "url", param + ".image_url.url", "image_url.url");
            out = add(fetch(url, param + ".image_url.url"), image->find("detail"), param);
            return true;
        }
        if (api == Api::responses && type == "input_image") {
            admit(param);
            if (const auto* file = in.find("file_id"); file && !file->is_null()) bad(param + ".file_id", "file_id is not supported; send image_url with a base64 data: URL");
            const auto url = string_field(in, "image_url", param + ".image_url", "image_url");
            out = add(fetch(url, param + ".image_url"), in.find("detail"), param);
            return true;
        }
        if (api == Api::messages && type == "image") {
            admit(param);
            const auto* source = in.find("source");
            if (!source || !source->is_object()) bad(param + ".source", "image source must be an object");
            const auto kind = string_field(*source, "type", param + ".source.type", "image source type");
            if (kind == "base64") {
                const auto data = string_field(*source, "data", param + ".source.data", "image source data");
                out = add(decode_base64(data, options.limits.max_image_bytes, param + ".source.data"), nullptr, param);
            } else if (kind == "url") {
                const auto url = string_field(*source, "url", param + ".source.url", "image source url");
                out = add(fetch(url, param + ".source.url"), nullptr, param);
            } else {
                bad(param + ".source.type", "unsupported image source type; use base64 or url");
            }
            return true;
        }
        if (api == Api::messages && type == "tool_result") return member_parts(in, "content", param, out);
        return false;
    }

    [[nodiscard]] bool parts(const Json& in, const std::string& param, Json& out) {
        if (!in.is_array()) return false;
        const auto& source = in.as_array();
        Json::Array copy;
        bool changed = false;
        for (std::size_t i = 0; i < source.size(); ++i) {
            Json replaced;
            if (part(source[i], param + "[" + std::to_string(i) + "]", replaced)) {
                if (!changed) copy = source;
                changed = true;
                copy[i] = std::move(replaced);
            }
        }
        if (changed) out = Json(std::move(copy));
        return changed;
    }

    // Rewrites the parts under object[key]; out is a copy of object with that member replaced.
    [[nodiscard]] bool member_parts(const Json& object, std::string_view key, const std::string& param, Json& out) {
        const auto* value = object.find(key);
        Json replaced;
        if (!value || !parts(*value, param + "." + std::string(key), replaced)) return false;
        auto members = object.as_object();
        for (auto& member : members) if (member.first == key) { member.second = std::move(replaced); break; }
        out = Json(std::move(members));
        return true;
    }

    [[nodiscard]] bool turn(const Json& in, const std::string& param, Json& out) {
        if (!in.is_object()) return false;
        const auto* role_value = in.find("role");
        const std::string_view role = role_value && role_value->is_string() ? std::string_view(role_value->as_string()) : std::string_view();
        if (api == Api::chat_completions) return (role == "user" || role == "tool") && member_parts(in, "content", param, out);
        if (api == Api::messages) return role == "user" && member_parts(in, "content", param, out);
        const auto* type_value = in.find("type");
        const std::string_view type = type_value && type_value->is_string() ? std::string_view(type_value->as_string()) : std::string_view();
        if (type == "function_call_output") return member_parts(in, "output", param, out);
        return (type.empty() || type == "message") && role == "user" && member_parts(in, "content", param, out);
    }
};

} // namespace

std::string_view media_type(Format format) noexcept {
    switch (format) {
        case Format::png: return "image/png";
        case Format::jpeg: return "image/jpeg";
        case Format::webp: return "image/webp";
    }
    return "application/octet-stream";
}

FetchPolicy parse_fetch_policy(std::string_view mode, std::string_view hosts) {
    FetchPolicy policy;
    if (mode.empty() || mode == "off") return policy;
    if (mode != "allowlist") throw std::invalid_argument("STRIX_SERVE_IMAGE_FETCH must be off or allowlist");
    policy.mode = FetchMode::allowlist;
    while (!hosts.empty()) {
        const auto comma = hosts.find(',');
        auto name = hosts.substr(0, comma);
        hosts = comma == std::string_view::npos ? std::string_view() : hosts.substr(comma + 1);
        while (!name.empty() && (name.front() == ' ' || name.front() == '\t')) name.remove_prefix(1);
        while (!name.empty() && (name.back() == ' ' || name.back() == '\t')) name.remove_suffix(1);
        if (name.empty()) continue;
        if (name.find_first_of("/@:*[] ") != std::string_view::npos) throw std::invalid_argument("STRIX_SERVE_IMAGE_HOSTS takes plain host names");
        policy.hosts.push_back(lowered(name));
    }
    if (policy.hosts.empty()) throw std::invalid_argument("STRIX_SERVE_IMAGE_FETCH=allowlist needs STRIX_SERVE_IMAGE_HOSTS");
    return policy;
}

FetchPolicy fetch_policy_from_env() {
    const char* mode = std::getenv("STRIX_SERVE_IMAGE_FETCH");
    const char* hosts = std::getenv("STRIX_SERVE_IMAGE_HOSTS");
    return parse_fetch_policy(mode ? mode : "", hosts ? hosts : "");
}

bool address_is_public(std::string_view literal) noexcept {
    if (literal.size() >= 2 && literal.front() == '[' && literal.back() == ']') literal = literal.substr(1, literal.size() - 2);
    if (literal.find(':') != std::string_view::npos) {
        const auto address = parse_v6(literal);
        return address && v6_public(*address);
    }
    const auto address = parse_v4(literal);
    return address && v4_public(address->data());
}

std::string decode_base64(std::string_view text, std::size_t max_bytes, std::string_view param) {
    std::string out;
    out.reserve(std::min(text.size() / 4 * 3 + 3, max_bytes));
    std::uint32_t buffer = 0;
    unsigned bits = 0;
    std::size_t symbols = 0, padding = 0;
    for (const char c : text) {
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
        if (c == '=') {
            if (++padding > 2) bad(param, "image data is not valid base64");
            continue;
        }
        std::uint32_t value = 0;
        if (c >= 'A' && c <= 'Z') value = static_cast<std::uint32_t>(c - 'A');
        else if (c >= 'a' && c <= 'z') value = static_cast<std::uint32_t>(c - 'a') + 26U;
        else if (c >= '0' && c <= '9') value = static_cast<std::uint32_t>(c - '0') + 52U;
        else if (c == '+') value = 62;
        else if (c == '/') value = 63;
        else bad(param, "image data is not valid base64");
        if (padding != 0) bad(param, "image data is not valid base64");
        ++symbols;
        buffer = (buffer << 6U) | value;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            if (out.size() >= max_bytes) bad(param, "image is larger than the per-image byte limit", 413);
            out.push_back(static_cast<char>((buffer >> bits) & 0xFFU));
        }
    }
    const auto tail = symbols % 4;
    if (tail == 1 || (padding != 0 && (tail + padding) != 4)) bad(param, "image data is not valid base64");
    if (out.empty()) bad(param, "image data is empty");
    return out;
}

Digest sha256(std::string_view bytes) noexcept {
    std::array<std::uint32_t, 8> h = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU, 0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    const auto* data = reinterpret_cast<const unsigned char*>(bytes.data());
    std::size_t at = 0;
    for (; at + 64 <= bytes.size(); at += 64) sha_block(h, data + at);
    std::array<unsigned char, 128> last{};
    const auto rest = bytes.size() - at;
    if (rest != 0) std::memcpy(last.data(), data + at, rest);
    last[rest] = 0x80;
    const std::size_t total = rest + 9 <= 64 ? 64 : 128;
    const auto bit_count = static_cast<std::uint64_t>(bytes.size()) * 8U;
    for (std::size_t i = 0; i < 8; ++i) last[total - 1 - i] = static_cast<unsigned char>((bit_count >> (8U * i)) & 0xFFU);
    sha_block(h, last.data());
    if (total == 128) sha_block(h, last.data() + 64);
    Digest out{};
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 4; ++j) out[4 * i + j] = static_cast<std::uint8_t>((h[i] >> (24U - 8U * j)) & 0xFFU);
    return out;
}

std::string hex(const Digest& digest) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (const auto byte : digest) {
        out.push_back(digits[byte >> 4U]);
        out.push_back(digits[byte & 0x0FU]);
    }
    return out;
}

ImageInfo probe_image(std::string_view b, std::string_view param) {
    ImageInfo info;
    static constexpr std::string_view png_magic("\x89PNG\r\n\x1a\n", 8);
    if (b.size() >= 8 && b.substr(0, 8) == png_magic) {
        info.format = Format::png;
        if (b.size() < 24 || b.substr(12, 4) != "IHDR") bad(param, "PNG header is truncated or invalid");
        info.width = be32(b, 16);
        info.height = be32(b, 20);
        if ((info.width | info.height) & 0x80000000U) bad(param, "PNG header is truncated or invalid");
    } else if (b.size() >= 3 && u8(b, 0) == 0xFF && u8(b, 1) == 0xD8 && u8(b, 2) == 0xFF) {
        info.format = Format::jpeg;
        std::size_t at = 2;
        for (;;) {
            if (at + 1 >= b.size() || u8(b, at) != 0xFF) bad(param, "JPEG header is truncated or invalid");
            const auto marker = u8(b, at + 1);
            if (marker == 0xFF) { ++at; continue; } // fill byte
            at += 2;
            if (marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) continue;
            if (marker == 0xD9 || marker == 0xDA || at + 2 > b.size()) bad(param, "JPEG header is truncated or invalid");
            const auto length = be16(b, at);
            if (length < 2) bad(param, "JPEG header is truncated or invalid");
            if (marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC) {
                if (length < 7 || at + 7 > b.size()) bad(param, "JPEG header is truncated or invalid");
                info.height = be16(b, at + 3);
                info.width = be16(b, at + 5);
                break;
            }
            at += length;
        }
    } else if (b.size() >= 16 && b.substr(0, 4) == "RIFF" && b.substr(8, 4) == "WEBP") {
        info.format = Format::webp;
        const auto chunk = b.substr(12, 4);
        if (chunk == "VP8 " && b.size() >= 30 && (u8(b, 20) & 1U) == 0 && u8(b, 23) == 0x9D && u8(b, 24) == 0x01 && u8(b, 25) == 0x2A) {
            info.width = le16(b, 26) & 0x3FFFU;
            info.height = le16(b, 28) & 0x3FFFU;
        } else if (chunk == "VP8L" && b.size() >= 25 && u8(b, 20) == 0x2F) {
            const auto packed = le32(b, 21);
            info.width = (packed & 0x3FFFU) + 1U;
            info.height = ((packed >> 14U) & 0x3FFFU) + 1U;
        } else if (chunk == "VP8X" && b.size() >= 30) {
            info.width = le24(b, 24) + 1U;
            info.height = le24(b, 27) + 1U;
        } else {
            bad(param, "WebP header is truncated or invalid");
        }
    } else {
        bad(param, "unsupported image format; send PNG, JPEG or WebP");
    }
    if (info.width == 0 || info.height == 0) bad(param, "image has an empty size");
    return info;
}

Normalized normalize(const Json& body, Api api, const Options& options) {
    Normalized result{body, {}};
    const std::string_view key = api == Api::responses ? "input" : "messages";
    const auto* turns = body.find(key);
    if (!turns || !turns->is_array()) return result;
    Walk walk{api, options, {}};
    const auto& source = turns->as_array();
    Json::Array copy;
    bool changed = false;
    for (std::size_t i = 0; i < source.size(); ++i) {
        Json replaced;
        if (walk.turn(source[i], std::string(key) + "[" + std::to_string(i) + "]", replaced)) {
            if (!changed) copy = source;
            changed = true;
            copy[i] = std::move(replaced);
        }
    }
    if (!changed) return result;
    auto members = body.as_object();
    for (auto& member : members) if (member.first == key) { member.second = Json(std::move(copy)); break; }
    result.body = Json(std::move(members));
    result.items = std::move(walk.items);
    return result;
}

} // namespace strix::serve::media
