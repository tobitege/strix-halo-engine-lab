#include "strix/device/hip_gate.hpp"
#include "strix/ds41_expert_server.hpp"

#if defined(STRIX_XS_WITH_HIP)
#include "strix/dist/communicator.hpp"
#include "strix/hip_ds41_expert_backend.hpp"
#endif

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

// deepseek41 expert server (strix/ds41-expert-rpc-v1, docs/models/deepseek-v4.1/expert-server.md): keeps routed experts and computes
// them for a parts-host (the device running the dense parts of every layer and its resident experts).
//   One endpoint per rank ("direct"): strix-deepseek41-expert-server MODEL.gguf --listen HOST:PORT --token-file F
//       [--experts FIRST:END] [--layers LIST] [--backend hip|host] [--busy-poll] [--keep-warm-us N] [--once] [--max-rows N]
//   One endpoint for two ranks ("relay": rank 0 listens and forwards to rank 1, which computes the high half):
//       rank 0: ... --listen HOST:PORT --token-file F --relay-rank 0 --peer usb4auto|tcp-listen:H:P|tcp-connect:H:P
//       rank 1: ... --relay-rank 1 --peer usb4auto|tcp-listen:H:P|tcp-connect:H:P
// LIST: "all" (default), "2,20", "0-39". READY {json} on stdout once listening (rank 0 / direct) or following (rank 1).
namespace {

namespace x = strix::ds41::xrpc;

struct Args {
    std::string model, listen, token_file, backend =
#if defined(STRIX_XS_WITH_HIP)
        "hip";
#else
        "host";
#endif
    std::string layers = "all", experts, peer;
    int relay_rank = -1;
    bool busy_poll = false, once = false, verbose = false;
    std::uint32_t max_rows = 64;
    unsigned peer_timeout = 120;
    std::uint32_t keep_warm_us = 0;
};

#if defined(STRIX_XS_WITH_HIP)
class TransportPeer final : public x::PeerLink {
public:
    explicit TransportPeer(std::unique_ptr<strix::dist::Transport> t) : t_(std::move(t)) {}
    void send(std::span<const std::uint8_t> b) override { t_->send(std::as_bytes(b)); }
    void receive(std::span<std::uint8_t> b) override { t_->receive(std::as_writable_bytes(b)); }
    [[nodiscard]] std::string name() const override { return t_->name(); }

private:
    std::unique_ptr<strix::dist::Transport> t_;
};
#endif

std::unique_ptr<x::PeerLink> open_peer(const std::string& spec, int rank, unsigned timeout) {
    const std::chrono::seconds limit(timeout);
    const auto colon = spec.find(':');
    const std::string kind = spec.substr(0, colon), rest = colon == std::string::npos ? "" : spec.substr(colon + 1);
    if (kind == "tcp-listen") return x::tcp_peer_listen(rest.find(':') == std::string::npos ? "0.0.0.0:" + rest : rest, limit);
    if (kind == "tcp-connect") return x::tcp_peer_connect(rest, limit);
#if defined(STRIX_XS_WITH_HIP)
    if (kind == "usb4auto") {
        strix::dist::Usb4AutoOptions o;
        o.rank = static_cast<std::uint32_t>(rank);
        o.timeout = limit;
        strix::dist::Usb4Topology topology;
        auto t = strix::dist::usb4_host_auto(o, &topology);
        std::printf("usb4 topology: %s\n", topology.receipt().c_str());
        return std::make_unique<TransportPeer>(std::move(t));
    }
#endif
    (void)rank;
    throw std::invalid_argument("unknown or unavailable peer link " + spec);
}

std::pair<std::uint32_t, std::uint32_t> parse_range(const std::string& s) {
    const auto c = s.find(':');
    if (c == std::string::npos) throw std::invalid_argument("--experts FIRST:END");
    return {static_cast<std::uint32_t>(std::stoul(s.substr(0, c))), static_cast<std::uint32_t>(std::stoul(s.substr(c + 1)))};
}

std::unique_ptr<x::ExpertBackend> make_backend(const Args& a, std::uint64_t mask, std::uint32_t first, std::uint32_t end, x::Geometry& g) {
#if defined(STRIX_XS_WITH_HIP)
    if (a.backend == "hip") return strix::ds41::xs::make_hip_backend(a.model, mask, first, end, &g, a.max_rows, a.busy_poll, a.keep_warm_us);
#endif
    if (a.backend == "host") return x::make_host_backend(a.model, mask, first, end, &g);
    throw std::invalid_argument("--backend " + a.backend + " is not available in this build");
}

int run(const Args& a) {
    // the layer count comes from the file; parse the list against the maximum first
    const std::uint64_t mask = x::parse_layer_mask(a.layers, 64U);
    x::Geometry g = x::read_geometry(a.model);
    std::uint32_t first = 0, end = 0;
    if (a.relay_rank >= 0) {  // relay: rank 0 keeps the low half, rank 1 the high half
        first = a.relay_rank == 0 ? 0U : g.split;
        end = a.relay_rank == 0 ? g.split : g.experts;
    } else if (!a.experts.empty()) {
        std::tie(first, end) = parse_range(a.experts);
    }
    std::unique_ptr<x::ExpertBackend> backend;
    const auto t0 = std::chrono::steady_clock::now();
    backend = make_backend(a, mask, first, end, g);
    const auto res = backend->residency();
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("expert server: %s, experts [%u, %u) of layers mask 0x%llx, geometry hidden %u inter %u experts %u top_k %u split %u, "
                "loaded in %.1f s\n",
                backend->engine().c_str(), res.expert_first, res.expert_end, static_cast<unsigned long long>(res.layer_mask), g.hidden,
                g.inter, g.experts, g.top_k, g.split, load_s);
    std::fflush(stdout);
    std::unique_ptr<x::PeerLink> peer;
    if (a.relay_rank >= 0) {
        peer = open_peer(a.peer, a.relay_rank, a.peer_timeout);
        // exchange residencies: each rank sends its own (16 bytes), rank 0 first
        std::array<std::uint8_t, 16> mine{}, theirs{};
        std::memcpy(mine.data(), &res.expert_first, 4);
        std::memcpy(mine.data() + 4, &res.expert_end, 4);
        std::memcpy(mine.data() + 8, &res.layer_mask, 8);
        if (a.relay_rank == 0) { peer->send(mine); peer->receive(theirs); }
        else { peer->receive(theirs); peer->send(mine); }
        x::Residency other;
        std::memcpy(&other.expert_first, theirs.data(), 4);
        std::memcpy(&other.expert_end, theirs.data() + 4, 4);
        std::memcpy(&other.layer_mask, theirs.data() + 8, 8);
        if (a.relay_rank == 1) {
            const x::Residency endpoint{0U, g.experts, other.layer_mask & res.layer_mask};
            std::printf("READY {\"role\":\"expert-server\",\"relay_rank\":1,\"peer\":\"%s\",\"experts\":[%u,%u],\"protocol\":\"%s\"}\n",
                        peer->name().c_str(), res.expert_first, res.expert_end, x::kProtocol);
            std::fflush(stdout);
            x::follow(g, endpoint, *backend, *peer);
            std::printf("{\"event\":\"STOPPED\"}\n");
            return 0;
        }
        x::ServerOptions o;
        o.listen = a.listen;
        o.token = x::read_token_file(a.token_file);
        o.geometry = g;
        o.max_rows = a.max_rows;
        o.busy_poll = a.busy_poll;
        o.once = a.once;
        o.verbose = a.verbose;
        o.peer_residency = other;
        o.on_listen = [&](std::uint16_t port) {
            std::printf("READY {\"role\":\"expert-server\",\"mode\":\"relay\",\"listen\":\"%s:%u\",\"experts\":[0,%u],\"layer_mask\":\"0x%llx\","
                        "\"world\":2,\"protocol\":\"%s\",\"peer\":\"%s\"}\n",
                        x::split_host_port(a.listen).first.c_str(), port, g.experts,
                        static_cast<unsigned long long>(res.layer_mask & other.layer_mask), x::kProtocol, peer->name().c_str());
            std::fflush(stdout);
        };
        const auto st = x::serve(o, *backend, peer.get());
        std::printf("{\"event\":\"STOPPED\",\"requests\":%llu,\"items\":%llu,\"compute_s\":%.3f,\"relay_s\":%.3f,\"server_s\":%.3f}\n",
                    static_cast<unsigned long long>(st.requests), static_cast<unsigned long long>(st.items), st.compute_s, st.relay_s, st.server_s);
        return 0;
    }
    x::ServerOptions o;
    o.listen = a.listen;
    o.token = x::read_token_file(a.token_file);
    o.geometry = g;
    o.max_rows = a.max_rows;
    o.busy_poll = a.busy_poll;
    o.once = a.once;
    o.verbose = a.verbose;
    o.on_listen = [&](std::uint16_t port) {
        std::printf("READY {\"role\":\"expert-server\",\"mode\":\"direct\",\"listen\":\"%s:%u\",\"experts\":[%u,%u],\"layer_mask\":\"0x%llx\","
                    "\"world\":1,\"protocol\":\"%s\"}\n",
                    x::split_host_port(a.listen).first.c_str(), port, res.expert_first, res.expert_end,
                    static_cast<unsigned long long>(res.layer_mask), x::kProtocol);
        std::fflush(stdout);
    };
    const auto st = x::serve(o, *backend, nullptr);
    std::printf("{\"event\":\"STOPPED\",\"requests\":%llu,\"items\":%llu,\"compute_s\":%.3f,\"server_s\":%.3f}\n",
                static_cast<unsigned long long>(st.requests), static_cast<unsigned long long>(st.items), st.compute_s, st.server_s);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    try {
        Args a;
        for (int i = 1; i < argc; ++i) {
            const std::string s = argv[i];
            const auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(s + " needs a value");
                return argv[++i];
            };
            if (s == "--listen") a.listen = next();
            else if (s == "--token-file") a.token_file = next();
            else if (s == "--backend") a.backend = next();
            else if (s == "--layers") a.layers = next();
            else if (s == "--experts") a.experts = next();
            else if (s == "--relay-rank") a.relay_rank = std::stoi(next());
            else if (s == "--peer") a.peer = next();
            else if (s == "--peer-timeout") a.peer_timeout = static_cast<unsigned>(std::stoul(next()));
            else if (s == "--max-rows") a.max_rows = static_cast<std::uint32_t>(std::stoul(next()));
            else if (s == "--busy-poll") a.busy_poll = true;
            else if (s == "--keep-warm-us") a.keep_warm_us = static_cast<std::uint32_t>(std::stoul(next()));
            else if (s == "--once") a.once = true;
            else if (s == "--verbose") a.verbose = true;
            else if (!s.empty() && s[0] == '-') throw std::invalid_argument("unknown option " + s);
            else if (a.model.empty()) a.model = s;
            else throw std::invalid_argument("unexpected argument " + s);
        }
        if (a.model.empty()) throw std::invalid_argument("usage: strix-deepseek41-expert-server MODEL.gguf --listen HOST:PORT --token-file F ...");
        if (a.relay_rank > 1) throw std::invalid_argument("--relay-rank is 0 or 1");
        if (a.relay_rank >= 0 && a.peer.empty()) throw std::invalid_argument("--relay-rank needs --peer");
        if (a.relay_rank >= 0 && !a.experts.empty()) throw std::invalid_argument("--experts is fixed by --relay-rank (the halves)");
        if (a.relay_rank != 1 && (a.listen.empty() || a.token_file.empty())) throw std::invalid_argument("--listen and --token-file are required");
        return run(a);
    } catch (const std::exception& e) {
        std::printf("{\"event\":\"POISONED\",\"error\":\"%s\"}\n", e.what());
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
