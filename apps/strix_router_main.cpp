// strix-router: places API requests across independent engine replicas (one full engine per node).
// Each replica is an ordinary strix HTTP server; the router only reads its status route and forwards bytes.

#include "strix/serve/http_server.hpp"
#include "strix/serve/replica_proxy.hpp"

#include <atomic>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

void usage() {
    std::cout <<
        "usage: strix-router --replica http://HOST:PORT [--replica ...] [options]\n"
        "  --listen HOST:PORT        address to serve on (default 127.0.0.1:8090)\n"
        "  --poll-ms N               status poll period per replica set (default 250)\n"
        "  --connect-timeout-ms N    connect timeout to a replica (default 2000)\n"
        "  --idle-timeout-s N        longest silence accepted from a replica, 0 = none (default 0)\n"
        "  --block-bytes N           prompt block size of the prefix affinity (default 1024)\n"
        "  --affinity-entries N      affinity table slots (default 4096)\n"
        "  --max-connections N       concurrent client connections (default 256)\n"
        "  --status-target PATH      replica status route (default /api/status)\n"
        "Placement: named session, then longest shared prompt prefix (unless that replica is more than one\n"
        "slot busier), then least busy. Router routes: GET /router/status, GET /health.\n";
}

long long number(std::string_view name, std::string_view text, long long low, long long high) {
    long long value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value < low || value > high)
        throw std::invalid_argument(std::string(name) + ": expected a number in [" + std::to_string(low) + ", "
            + std::to_string(high) + "]");
    return value;
}

} // namespace

int main(int argc, char** argv) {
    try {
        strix::serve::ReplicaProxyConfig proxy_config;
        strix::serve::HttpServerConfig server_config;
        server_config.port = 8090;
        server_config.max_connections = 256;
        int poll_ms = 250;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg = argv[i];
            if (arg == "--help" || arg == "-h") { usage(); return 0; }
            if (i + 1 >= argc) throw std::invalid_argument(std::string(arg) + ": missing value");
            const std::string_view value = argv[++i];
            if (arg == "--replica") {
                auto endpoint = strix::serve::parse_replica_endpoint(value);
                if (!endpoint) throw std::invalid_argument("--replica: expected http://HOST:PORT");
                proxy_config.replicas.push_back(std::move(*endpoint));
            } else if (arg == "--listen") {
                auto endpoint = strix::serve::parse_replica_endpoint(value);
                if (!endpoint) throw std::invalid_argument("--listen: expected HOST:PORT");
                server_config.bind_address = endpoint->host;
                server_config.port = endpoint->port;
            } else if (arg == "--poll-ms") poll_ms = static_cast<int>(number(arg, value, 10, 60000));
            else if (arg == "--connect-timeout-ms") proxy_config.connect_timeout_ms = static_cast<int>(number(arg, value, 1, 600000));
            else if (arg == "--idle-timeout-s") proxy_config.idle_timeout_ms = static_cast<int>(number(arg, value, 0, 86400) * 1000);
            else if (arg == "--block-bytes") proxy_config.block_bytes = static_cast<std::size_t>(number(arg, value, 16, 1 << 20));
            else if (arg == "--affinity-entries") proxy_config.affinity_entries = static_cast<std::size_t>(number(arg, value, 1, 1 << 24));
            else if (arg == "--max-connections") server_config.max_connections = static_cast<unsigned>(number(arg, value, 1, 65535));
            else if (arg == "--status-target") proxy_config.status_target = std::string(value);
            else throw std::invalid_argument("unknown option " + std::string(arg));
        }
        if (proxy_config.replicas.empty()) { usage(); return 2; }
        if (proxy_config.replicas.size() > strix::serve::ReplicaRouter::kMaxReplicas)
            throw std::invalid_argument("at most 64 replicas");

        strix::serve::ReplicaProxy proxy(proxy_config);
        proxy.refresh();
        strix::serve::HttpServer server(server_config,
            [&](const strix::serve::HttpRequest& request, strix::serve::HttpResponseWriter& response) {
                proxy.handle(request, response);
            });
        server.listen();
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        std::thread poller([&] {
            while (!g_stop.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
                if (g_stop.load()) break;
                proxy.refresh();
            }
            server.stop();
        });
        std::size_t healthy = 0;
        for (const auto& s : proxy.states()) healthy += s.healthy ? 1U : 0U;
        std::cout << "strix-router listening on " << server_config.bind_address << ":" << server.port() << ", "
                  << healthy << "/" << proxy_config.replicas.size() << " replicas healthy" << std::endl;
        server.serve();
        g_stop.store(true);
        poller.join();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "strix-router: " << e.what() << "\n";
        return 1;
    }
}
