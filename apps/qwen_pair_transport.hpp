#pragma once
#include "strix/qwen_prefill_pair.hpp"
#include <cstdlib>
#ifndef STRIX_DIST_PAIR_RUNTIME
#error "STRIX_DIST_PAIR_RUNTIME comes with strix_dist: link it (directly or through strix_hip_runtime)"
#endif
#if STRIX_DIST_PAIR_RUNTIME
#include "strix/dist/open_transport.hpp"
#include "../src/xport/pair_device_link.hpp"
#else
#include "strix/dist/transport.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#endif

namespace strix::qwen_pp {
#if !STRIX_DIST_PAIR_RUNTIME
// Built without the pair runtime (STRIX_DIST_PAIR_RUNTIME=0, see strix_dist in CMakeLists.txt): no link to a
// peer can be opened, so a pair prefill and a deployment pipeline refuse where they would open one. The host
// path over a given link is the same as below (pair::DeviceLink's deployment-link constructor), without the
// device route.
[[noreturn]] inline void refuse_pair_link() {
    throw std::invalid_argument("a link to a peer node needs the pair runtime, which is built on Linux only");
}
inline std::unique_ptr<dist::Transport> open_deployment_link(const std::string&, unsigned) { refuse_pair_link(); }
class PairTransport {
public:
    explicit PairTransport(unsigned) { refuse_pair_link(); }
    explicit PairTransport(std::unique_ptr<dist::Transport> link) : host_(std::move(link)) {
        if (!host_) throw std::invalid_argument("pair prefill: empty deployment link");
    }
    void send(std::span<const std::byte> data) { host_->send(data); }
    void receive(std::span<std::byte> data) { host_->receive(data); }
    void abort(const std::string& why) { host_->abort(why); }
    void finish() {}
    [[nodiscard]] bool routes(std::size_t) const noexcept { return false; }
    [[nodiscard]] bool device_checked(std::size_t) const noexcept { return false; }
    bool piece(bool, PipeStaging&, std::size_t, std::size_t) { return false; }
private:
    std::unique_ptr<dist::Transport> host_;
};
#else
// The link a deployment pipeline opens for its handshake (apps/qwen_deploy.inc); the pair prefill then keeps
// using it through PairTransport's link constructor.
inline std::unique_ptr<dist::Transport> open_deployment_link(const std::string& transport, unsigned rank) {
    if (transport == "usb4auto" || transport == "usb4-auto") {
        dist::Usb4AutoOptions usb;
        usb.rank = rank;
        return dist::usb4_host_auto(usb);
    }
    if (transport != "tcp") throw std::invalid_argument("deployment: unsupported pipeline transport");
    const char* peer = std::getenv("STRIX_QWEN_PF_PIPE_PEER");
    if (rank && !peer) throw std::invalid_argument("pipeline rank 1 requires --peer");
    const auto timeout = std::chrono::milliseconds(120000);
    return rank ? dist::tcp_connect(peer, 29872, timeout) : dist::tcp_listen(29872, timeout);
}
// Qwen's pair link: the shared pair::DeviceLink (control on the host transport, seam pieces one way on the device
// route with device CRC) opened from this engine's transport setting.
class PairTransport final : public pair::DeviceLink {
public:
    explicit PairTransport(unsigned rank) : pair::DeviceLink(rank, 0x51575050ULL /* "QWPP" */, open(rank), "qwen-pair") {}
    // A link the deployment handshake already opened (it exchanged measured
    // facts and plan IDs on it): keep using it, host path only.
    explicit PairTransport(std::unique_ptr<dist::Transport> link) : pair::DeviceLink(std::move(link), "qwen-pair") {}
private:
    static dist::OpenedTransport open(unsigned rank) {
        dist::OpenTransportOptions options;
        options.usb.rank = rank;
        options.spellings = {"usb4auto", "usb4gpu", "usb4-auto", false, false, false};
        const char* value = std::getenv("STRIX_QWEN_PF_PIPE_TRANSPORT");
        std::string spec = value ? value : "usb4gpu";
        if (spec == "tcp") {
            const char* peer = std::getenv("STRIX_QWEN_PF_PIPE_PEER");
            if (rank == 1U && !peer) throw std::invalid_argument("pair prefill TCP rank 1 requires --peer HOST (port 29872)");
            spec = rank == 0U ? "tcp-listen:29872" : "tcp-connect:" + std::string(peer) + ":29872";
            options.spellings.tcp = true;
            options.usb.timeout = std::chrono::milliseconds(120000);
        }
        return dist::open_transport(spec, options);
    }
};
#endif
} // namespace strix::qwen_pp
