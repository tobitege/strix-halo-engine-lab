#pragma once
// Host-only parsing shared by the transport opener and its tests. Every pair binary accepts the same spellings:
// usb4auto / usb4-auto (host path), usb4gpu / usb4-gpu (GPU-direct streams), usb4:dev,... and tcp-listen/tcp-connect.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::dist {
enum class TransportKind { tcp_listen, tcp_connect, usb4_host, usb4_auto, usb4_gpu };
struct TransportSpellings {
    std::string_view host = "usb4auto", gpu = "usb4gpu", host_alias = "usb4-auto";
    bool tcp = true, devices = true;
    bool usb_suffix = true; // kind:rest ignores the USB4 rest
    std::string_view gpu_alias = "usb4-gpu";
};
inline constexpr TransportSpellings pair_transport{};
struct TransportSpec {
    TransportKind kind;
    std::string host;
    std::uint16_t port = 0;
    std::vector<std::string> devices;
};
inline TransportSpec parse_transport(const std::string& spec, TransportSpellings spellings) {
    const auto colon = spec.find(':');
    const auto kind = spec.substr(0, colon);
    const auto rest = colon == std::string::npos ? "" : spec.substr(colon + 1);
    if (spellings.tcp && kind == "tcp-listen")
        return {TransportKind::tcp_listen, {}, static_cast<std::uint16_t>(std::stoul(rest)), {}};
    if (spellings.tcp && kind == "tcp-connect") {
        const auto c2 = rest.rfind(':');
        return {TransportKind::tcp_connect, rest.substr(0, c2), static_cast<std::uint16_t>(std::stoul(rest.substr(c2 + 1))), {}};
    }
    if (!spellings.usb_suffix && colon != std::string::npos)
        throw std::invalid_argument("unknown transport " + spec);
    if ((!spellings.host.empty() && kind == spellings.host) || (!spellings.host_alias.empty() && kind == spellings.host_alias))
        return {TransportKind::usb4_auto, {}, 0, {}};
    if ((!spellings.gpu.empty() && kind == spellings.gpu) || (!spellings.gpu_alias.empty() && kind == spellings.gpu_alias))
        return {TransportKind::usb4_gpu, {}, 0, {}};
    if (spellings.devices && kind == "usb4") {
        TransportSpec result{TransportKind::usb4_host, {}, 0, {}};
        for (std::size_t begin = 0; begin < rest.size();) {
            const auto end = rest.find(',', begin);
            result.devices.push_back(rest.substr(begin, end == std::string::npos ? end : end - begin));
            if (end == std::string::npos) break;
            begin = end + 1;
        }
        return result;
    }
    throw std::invalid_argument("unknown transport " + spec);
}
// Route identity after opening: a requested GPU transport is still the ordinary
// host route unless a live device exchange was negotiated. Never infer this from
// the request or the human-readable transport receipt. A USB4 route is named by the caller's canonical names
// whichever accepted spelling was requested, so a route key never depends on the spelling.
inline std::string negotiated_transport(const std::string& requested, bool direct_live,
                                        std::string_view host_name = pair_transport.host,
                                        std::string_view gpu_name = pair_transport.gpu) {
    switch (parse_transport(requested, pair_transport).kind) {
    case TransportKind::usb4_gpu: return std::string(direct_live ? gpu_name : host_name);
    case TransportKind::usb4_auto: return std::string(host_name);
    default: return requested;
    }
}
} // namespace strix::dist
