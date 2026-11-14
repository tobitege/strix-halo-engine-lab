#pragma once
// The single TCP/USB4 opener. The host transport always owns control/small
// messages; a device exchange is returned separately for the size router.
// Pair binaries call open_pair(); Communicator(group, host, direct, max_bytes) accepts a null direct (host path).
#include "strix/dist/transport_spec.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/dist/gpu_direct.hpp"

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>

namespace strix::dist {
struct OpenTransportOptions {
    TransportSpellings spellings = pair_transport;
    Usb4AutoOptions usb;
};
struct OpenedTransport {
    std::unique_ptr<Transport> host;
    std::unique_ptr<DeviceExchange> direct;
    std::string receipt;
};
inline OpenedTransport open_transport(const std::string& spec, const OpenTransportOptions& options) {
    const auto parsed = parse_transport(spec, options.spellings);
    OpenedTransport result;
    switch (parsed.kind) {
    case TransportKind::tcp_listen: result.host = tcp_listen(parsed.port, options.usb.timeout); return result;
    case TransportKind::tcp_connect: result.host = tcp_connect(parsed.host, parsed.port, options.usb.timeout); return result;
    case TransportKind::usb4_host: result.host = usb4_host(parsed.devices); return result;
    case TransportKind::usb4_auto:
    case TransportKind::usb4_gpu: break;
    }
    Usb4Topology topology;
    result.host = usb4_host_auto(options.usb, &topology);
    result.receipt = topology.receipt();
    if (parsed.kind == TransportKind::usb4_gpu) {
        std::string why, direct_receipt;
        if (usb4_gpu_direct_streams(options.usb, *result.host, &why) == 0U)
            result.receipt += "; gpu-direct unavailable (" + why + "): host path only";
        else {
            result.direct = usb4_gpu_direct(options.usb, topology, *result.host, {}, &direct_receipt);
            result.receipt += "; usb4 " + direct_receipt;
        }
    }
    return result;
}

struct PairOptions {
    std::chrono::milliseconds timeout{std::chrono::seconds(180)};  // connect / first-stream wait
    std::uint16_t transport_capabilities = all_transport_capabilities;
    bool print_receipt = false;  // "usb4 topology: <receipt>" on stderr
};
// Opens one rank of a two-rank pair. The caller may use opened.host for startup handshakes before it hands the result
// to the Communicator.
inline OpenedTransport open_pair(const std::string& spec, std::uint32_t rank, const PairOptions& pair = {}) {
    OpenTransportOptions options;
    options.usb.rank = rank;
    options.usb.timeout = pair.timeout;
    options.usb.transport_capabilities = pair.transport_capabilities;
    auto opened = open_transport(spec, options);
    if (pair.print_receipt && !opened.receipt.empty()) std::fprintf(stderr, "usb4 topology: %s\n", opened.receipt.c_str());
    return opened;
}
} // namespace strix::dist
