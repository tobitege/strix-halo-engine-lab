#pragma once
// Pair-route command line and link setup shared by every engine's two-node route (Qwen, Gemma 4):
// --rank 0|1, --peer HOST and --transport usb4auto|tcp are removed from argv and exported as
// <PREFIX>_RANK, <PREFIX>_PEER and <PREFIX>_TRANSPORT, so an engine's option parser never sees them and
// a re-exec'd child inherits them. open_link() opens the ordered byte transport the pair pipe runs on.
#include "strix/dist/open_transport.hpp"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::pair {
inline void set_env(const std::string& key, const char* value) {
#ifdef _WIN32
    if (_putenv_s(key.c_str(), value) != 0) throw std::runtime_error("pair prefill environment update failed");
#else
    if (setenv(key.c_str(), value, 1) != 0) throw std::runtime_error("pair prefill environment update failed");
#endif
}
// Returns the remaining arguments (argv[0] first, nullptr-terminated); argc/argv then point into it.
inline std::vector<char*> strip_flags(int& argc, char**& argv, std::string_view prefix) {
    std::vector<char*> kept{argv[0]};
    const std::string base(prefix);
    bool rank_seen = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--rank" || argument == "--peer" || argument == "--transport") {
            if (++index == argc) throw std::invalid_argument("pair prefill pair flag requires a value");
            const char* value = argv[index];
            if (argument == "--rank") {
                if (rank_seen || (std::string_view(value) != "0" && std::string_view(value) != "1"))
                    throw std::invalid_argument("pair prefill requires one --rank 0|1");
                rank_seen = true;
                set_env(base + "_RANK", value);
            } else set_env(base + (argument == "--peer" ? "_PEER" : "_TRANSPORT"), value);
        } else kept.push_back(argv[index]);
    }
    if (!rank_seen) {
        const char* inherited = std::getenv((base + "_RANK").c_str());
        if (!inherited || (std::string_view(inherited) != "0" && std::string_view(inherited) != "1"))
            throw std::invalid_argument("pair prefill requires --rank 0|1");
    }
    argc = static_cast<int>(kept.size());
    kept.push_back(nullptr);
    argv = kept.data();
    return kept;
}
inline unsigned env_rank(std::string_view prefix) {
    const char* value = std::getenv((std::string(prefix) + "_RANK").c_str());
    if (!value || (std::string_view(value) != "0" && std::string_view(value) != "1"))
        throw std::invalid_argument("pair prefill requires --rank 0|1");
    return value[0] == '0' ? 0U : 1U;
}
// transport: usb4auto (default; the striped USB4 host path), usb4gpu (the same control path plus the
// device exchange, for a link that moves seam pieces device to device: xport/pair_device_link.hpp) or
// tcp (rank 0 listens on port, rank 1 connects to peer), opened by the single shared opener
// (strix/dist/open_transport.hpp). Both ranks block here until the peer appears: the first common
// point of a pair run.
inline dist::OpenedTransport open_device_link(unsigned rank, std::string_view prefix, std::uint16_t port, bool device = true) {
    const std::string base(prefix);
    const char* transport_env = std::getenv((base + "_TRANSPORT").c_str());
    std::string spec = transport_env ? transport_env : "usb4auto";
    dist::OpenTransportOptions options;
    options.usb.rank = rank;
    if (spec == "tcp") {
        const char* peer = std::getenv((base + "_PEER").c_str());
        if (rank == 1U && !peer)
            throw std::invalid_argument("pair prefill TCP rank 1 requires --peer HOST (port " + std::to_string(port) + ")");
        spec = rank == 0U ? "tcp-listen:" + std::to_string(port) : "tcp-connect:" + std::string(peer) + ":" + std::to_string(port);
        options.usb.timeout = std::chrono::milliseconds(120000);
    } else if (device && spec == "usb4gpu") {
        options.spellings = {"usb4auto", "usb4gpu", "usb4-auto", false, false, false};
    } else if (spec != "usb4auto")
        throw std::invalid_argument(device ? "pair prefill transport must be usb4auto, usb4gpu or tcp"
                                           : "pair prefill transport must be usb4auto or tcp");
    return dist::open_transport(spec, options);
}
// The host path only (usb4auto or tcp).
inline std::unique_ptr<dist::Transport> open_link(unsigned rank, std::string_view prefix, std::uint16_t port) {
    return open_device_link(rank, prefix, port, false).host;
}
} // namespace strix::pair
