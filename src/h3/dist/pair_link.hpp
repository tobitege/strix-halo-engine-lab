#pragma once
// USB4 pairing of the two H3 ranks (stream discovery, cable measurement) on its own thread from program entry, so the
// text encoder and the DiT residency do not wait for the peer to arrive. Shared by the image sequence pair and the
// clip runner (world 2). The transport is the same usb4_host_auto transport as a serial pairing; only the wall overlaps.
#include "h3/common/alongside.hpp"
#include "strix/dist/communicator.hpp"
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>

namespace strix::h3::dist {

struct PairLink {
    std::unique_ptr<strix::dist::Transport> transport;
    strix::dist::Usb4Topology topology;
};
using PairLinkJob = std::future<Timed<PairLink>>;

// The ranks reach the rendezvous at different times (one may still load models): a generous timeout.
inline strix::dist::Usb4AutoOptions pair_link_options(std::uint32_t rank) {
    strix::dist::Usb4AutoOptions o;
    o.rank = rank;
    o.timeout = std::chrono::milliseconds(240000);
    return o;
}
inline PairLinkJob start_pair_link(const strix::dist::Usb4AutoOptions& options) {
    return start_alongside([options] {
        PairLink l;
        l.transport = strix::dist::usb4_host_auto(options, &l.topology);
        return l;
    });
}

}  // namespace strix::h3::dist
