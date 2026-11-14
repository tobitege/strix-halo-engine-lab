// Loopback test of the replica front end: two fake replicas and the router, all on HttpServer, driven by the
// blocking HTTP client. Checks SSE pass-through, cancel propagation, and that a request is never sent twice.

#include "strix/serve/http_client.hpp"
#include "strix/serve/http_server.hpp"
#include "strix/serve/replica_proxy.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#include <crtdbg.h>
#include <cstdlib>
#endif

namespace {
using namespace strix::serve;
using namespace std::chrono_literals;

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

bool wait_until(const std::atomic<bool>& flag, std::chrono::milliseconds limit = 10s) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!flag.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(5ms);
    return flag.load();
}

const std::string kEventA = "data: {\"t\":\"a\"}\n\n";
const std::string kEventB = "data: {\"t\":\"b\"}\n\n";
const std::string kDone = "data: [DONE]\n\n";

// A running HttpServer on a free loopback port.
struct Served {
    explicit Served(HttpHandler handler) {
        HttpServerConfig config;
        config.port = 0;
        server = std::make_unique<HttpServer>(config, std::move(handler));
        server->listen();
        port = server->port();
        thread = std::thread([this] { server->serve(); });
    }
    ~Served() { stop(); }
    void stop() {
        if (!server) return;
        server->stop();
        if (thread.joinable()) thread.join();
        server.reset();
    }
    std::unique_ptr<HttpServer> server;
    std::thread thread;
    std::uint16_t port = 0;
};

struct FakeReplica {
    std::string name;
    std::atomic<int> generations{0};
    std::atomic<int> busy{0};
    std::atomic<bool> release_second{false};  // set by the test once the first event reached the client
    std::atomic<bool> saw_client_gone{false};
    std::atomic<bool> draining{false};
    std::string last_auth;
    Served served;

    explicit FakeReplica(std::string id) : name(std::move(id)), served([this](const HttpRequest& q, HttpResponseWriter& r) { handle(q, r); }) {}

    void handle(const HttpRequest& q, HttpResponseWriter& r) {
        if (q.method == "GET" && q.path == "/api/status") {
            r.send(200, "application/json", std::string("{\"state\":\"") + (draining ? "draining" : "ready")
                + "\",\"capacity\":{\"slots_total\":4,\"slots_busy\":" + std::to_string(busy.load())
                + ",\"kv_bytes_free\":null,\"queue_depth\":0}}");
            return;
        }
        if (q.method == "GET" && q.path == "/v1/models") { r.send(200, "application/json", "{\"data\":[\"" + name + "\"]}"); return; }
        ++generations;
        last_auth = q.header("authorization");
        if (q.path == "/v1/chat/completions") {
            if (q.body.find("\"stream\":true") == std::string::npos) {
                r.send(200, "application/json", "{\"replica\":\"" + name + "\",\"echo\":" + std::to_string(q.body.size()) + "}",
                    {{"X-Request-Id", "req-" + name}});
                return;
            }
            if (!r.begin_stream(200, "text/event-stream")) return;
            (void)r.write(kEventA);
            // The second event is held back until the client has the first: a proxy that buffered would deadlock here.
            (void)wait_until(release_second);
            (void)r.write(kEventB);
            (void)r.write(kDone);
            r.end_stream();
            return;
        }
        if (q.path == "/endless") {
            if (!r.begin_stream(200, "text/event-stream")) return;
            const auto deadline = std::chrono::steady_clock::now() + 10s;
            while (std::chrono::steady_clock::now() < deadline) {
                if (!r.write(kEventA)) { saw_client_gone = true; return; }
                std::this_thread::sleep_for(10ms);
            }
            r.end_stream();
            return;
        }
        if (q.path == "/cut") {
            // A stream that dies after its first event: no terminator, no end of the chunked body.
            if (!r.begin_stream(200, "text/event-stream")) return;
            (void)r.write(kEventA);
            throw std::runtime_error("replica died mid-stream");
        }
        if (q.path == "/fail") { r.send(500, "application/json", "{\"error\":\"" + name + "\"}"); return; }
        r.send(404, "application/json", "{}");
    }
};

HttpClientRequest to(std::uint16_t port, std::string method, std::string target, std::string_view body = {}) {
    HttpClientRequest q;
    q.port = port;
    q.method = std::move(method);
    q.target = std::move(target);
    q.body = body;
    q.idle_timeout_ms = 20000;
    return q;
}

void client_basics(FakeReplica& a) {
    std::string body;
    auto r = http_fetch(to(a.served.port, "GET", "/api/status"), body);
    require(r.ok && r.head.status == 200 && r.request_sent && r.stage == HttpClientStage::done, "client: plain GET");
    require(body.find("\"slots_total\":4") != std::string::npos && r.body_bytes == body.size(), "client: Content-Length body");
    require(r.head.header("content-type") == "application/json", "client: header lookup");

    // Nothing listens on a port that was just released: the failure is reported before any byte is sent.
    std::uint16_t dead = 0;
    { Served gone([](const HttpRequest&, HttpResponseWriter& w) { w.send(200, "text/plain", "x"); }); dead = gone.port; }
    auto q = to(dead, "POST", "/x", "payload");
    q.connect_timeout_ms = 1500;
    r = http_fetch(q, body);
    require(!r.ok && !r.request_sent && r.stage == HttpClientStage::connect && !r.error.empty(), "client: refused connect");

    // A sink abort closes the connection and is reported as cancelled.
    HttpClientSink sink;
    sink.on_data = [](std::string_view) { return false; };
    r = http_exchange(to(a.served.port, "GET", "/v1/models"), sink);
    require(!r.ok && r.cancelled && r.request_sent && r.stage == HttpClientStage::body, "client: sink abort");
}

void run() {
    FakeReplica a("A"), b("B");
    client_basics(a);

    ReplicaProxyConfig config;
    config.replicas = {{"127.0.0.1", a.served.port}, {"127.0.0.1", b.served.port}};
    config.block_bytes = 64;
    config.connect_timeout_ms = 1500;
    ReplicaProxy proxy(config);
    require(proxy.states()[0].healthy == false, "replicas are unhealthy until the first status read");
    proxy.refresh();
    require(proxy.states()[0].healthy && proxy.states()[1].healthy, "both replicas healthy");
    Served front([&](const HttpRequest& q, HttpResponseWriter& r) { proxy.handle(q, r); });
    std::string body;

    // Router-owned routes.
    auto r = http_fetch(to(front.port, "GET", "/health"), body);
    require(r.ok && r.head.status == 200 && body.find("\"replicas_healthy\":2") != std::string::npos, "router health");

    // Non-stream request: same status, type, body and headers as the replica sends; credentials pass through.
    const std::string plain = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    auto q = to(front.port, "POST", "/v1/chat/completions", plain);
    q.headers = {{"Content-Type", "application/json"}, {"Authorization", "Bearer k"}};
    r = http_fetch(q, body);
    require(r.ok && r.head.status == 200 && body == "{\"replica\":\"A\",\"echo\":" + std::to_string(plain.size()) + "}", "plain body passes through");
    require(r.head.header("content-type") == "application/json" && r.head.header("x-request-id") == "req-A"
        && r.head.header("x-strix-replica") == "0" && r.head.header("content-length") == std::to_string(body.size()),
        "plain headers pass through");
    require(a.last_auth == "Bearer k" && a.generations == 1 && b.generations == 0, "request reached exactly one replica");

    // SSE: events arrive one by one, bytes unchanged.
    const std::string stream = "{\"model\":\"m\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
    std::string events;
    std::vector<std::size_t> pieces;
    HttpClientSink sink;
    std::string seen_type;
    sink.on_head = [&](const HttpClientHead& head) { seen_type = head.header("content-type"); return true; };
    sink.on_data = [&](std::string_view data) {
        events.append(data);
        pieces.push_back(data.size());
        if (events == kEventA) { a.release_second = true; b.release_second = true; }
        return true;
    };
    r = http_exchange(to(front.port, "POST", "/v1/chat/completions", stream), sink);
    require(r.ok && r.head.status == 200 && seen_type == "text/event-stream", "SSE head");
    require(events == kEventA + kEventB + kDone, "SSE bytes unchanged");
    require(pieces.size() >= 2 && pieces.front() == kEventA.size(), "first SSE event was delivered before the second was produced");
    require(a.generations + b.generations == 2, "one generation per request");

    // Cancel: the client leaves mid-stream, the replica sees its own client go away.
    int taken = 0;
    HttpClientSink leaving;
    leaving.on_data = [&](std::string_view) { return ++taken < 3; };
    r = http_exchange(to(front.port, "POST", "/endless", "{}"), leaving);
    require(r.cancelled && !r.ok, "client aborted the stream");
    require(wait_until(a.saw_client_gone) || wait_until(b.saw_client_gone, 1ms), "cancel reached the replica");

    // No duplicate after a failure once the request was sent: a cut stream and a 500 are both final.
    int before_a = a.generations, before_b = b.generations;
    events.clear();
    HttpClientSink collect;
    collect.on_data = [&](std::string_view data) { events.append(data); return true; };
    r = http_exchange(to(front.port, "POST", "/cut", "{}"), collect);
    require(events == kEventA, "cut stream delivers what the replica sent and nothing else");
    require((a.generations - before_a) + (b.generations - before_b) == 1, "cut stream is not replayed on the other replica");
    before_a = a.generations; before_b = b.generations;
    r = http_fetch(to(front.port, "POST", "/fail", "{}"), body);
    require(r.ok && r.head.status == 500 && (body == "{\"error\":\"A\"}" || body == "{\"error\":\"B\"}"), "replica error passes through");
    require((a.generations - before_a) + (b.generations - before_b) == 1, "a failed request is not retried");

    // Session stickiness through the front end: the same session returns to its replica whatever the load.
    b.busy = 0; a.busy = 2;
    proxy.refresh();
    q = to(front.port, "POST", "/v1/chat/completions", plain);
    q.headers = {{"X-Session-Id", "conv-7"}};
    r = http_fetch(q, body);
    require(r.ok && r.head.header("x-strix-replica") == "1", "new session placed on the less busy replica");
    a.busy = 0; b.busy = 3;
    proxy.refresh();
    r = http_fetch(q, body);
    require(r.ok && r.head.header("x-strix-replica") == "1", "session stays on its replica");
    r = http_fetch(to(front.port, "GET", "/v1/models"), body);
    require(r.ok && body == "{\"data\":[\"A\"]}", "unkeyed request goes to the less busy replica");

    // Draining replica takes nothing new.
    b.busy = 0; a.busy = 0; a.draining = true;
    proxy.refresh();
    require(!proxy.states()[0].healthy && proxy.states()[1].healthy, "draining replica is skipped");
    r = http_fetch(to(front.port, "GET", "/v1/models"), body);
    require(r.ok && body == "{\"data\":[\"B\"]}", "request avoids the draining replica");
    a.draining = false;
    proxy.refresh();

    // Replica gone between two status reads: nothing was sent to it, so the request moves to the other one.
    b.busy = 3;
    proxy.refresh();
    before_b = b.generations;
    a.served.stop();
    r = http_fetch(to(front.port, "POST", "/v1/chat/completions", plain), body);
    require(r.ok && r.head.status == 200 && r.head.header("x-strix-replica") == "1", "refused connection moves to the other replica");
    require(b.generations - before_b == 1, "moved request ran once");
    require(!proxy.states()[0].healthy, "refusing replica is marked unhealthy at once");
    r = http_fetch(to(front.port, "GET", "/router/status"), body);
    require(r.ok && body.find("\"retried_before_send\":1") != std::string::npos, "router status counts the move");

    // No replica left: 503, nothing invented.
    b.served.stop();
    r = http_fetch(to(front.port, "POST", "/v1/chat/completions", plain), body);
    require(r.ok && r.head.status == 503 && body.find("no healthy replica") != std::string::npos, "no replica: 503");
    proxy.refresh();
    r = http_fetch(to(front.port, "GET", "/health"), body);
    require(r.ok && r.head.status == 503, "router health reports no replica");

    require(parse_replica_endpoint("http://node-b:8080/") && parse_replica_endpoint("http://node-b:8080/")->port == 8080
        && parse_replica_endpoint("10.0.0.2:9000")->host == "10.0.0.2", "endpoint parsing");
    require(!parse_replica_endpoint("node-b") && !parse_replica_endpoint("node-b:0") && !parse_replica_endpoint("node-b:99999")
        && !parse_replica_endpoint(":80") && !parse_replica_endpoint("http://a/b:80"), "malformed endpoints rejected");
}

} // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    try {
        run();
    } catch (const std::exception& e) {
        std::cerr << "replica_proxy_tests: " << e.what() << "\n";
        return 1;
    }
    std::cout << "replica_proxy_tests: ok\n";
    return 0;
}
