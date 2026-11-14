#pragma once

// Minimal blocking HTTP/1.1 client: one request per connection (Connection: close), response body delivered as it
// arrives (Content-Length, chunked or read-to-close), so a Server-Sent-Events stream passes through event by event.
// No TLS, no redirects, no connection reuse.

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::serve {

struct HttpClientRequest {
    std::string host = "127.0.0.1";
    std::uint16_t port = 80;
    std::string method = "GET", target = "/";
    // Host, Content-Length and Connection are written by the client; do not pass them.
    std::vector<std::pair<std::string, std::string>> headers;
    std::string_view body;
    int connect_timeout_ms = 3000;
    // Longest silence accepted from the server while waiting for the head or for body bytes. 0 = no limit
    // (a generation may think for a long time before its first token).
    int idle_timeout_ms = 0;
    std::uint64_t max_head_bytes = 64ULL << 10U;
};

struct HttpClientHead {
    int status = 0;
    std::vector<std::pair<std::string, std::string>> headers; // names lower-cased
    [[nodiscard]] std::string header(std::string_view name) const;
};

struct HttpClientSink {
    // Called once, before any body byte. Return false to abort the exchange.
    std::function<bool(const HttpClientHead&)> on_head;
    // Decoded body bytes (chunk framing removed), in arrival order. Return false to abort.
    std::function<bool(std::string_view)> on_data;
    // Polled at least every 100 ms while waiting on the socket. Return true to abort: the connection is closed,
    // which is how a cancel reaches the server.
    std::function<bool()> cancelled;
};

enum class HttpClientStage : std::uint8_t { connect, send, head, body, done };

struct HttpClientResult {
    bool ok = false;                 // the full response was received and delivered
    bool cancelled = false;          // aborted by the sink
    HttpClientStage stage = HttpClientStage::connect; // where the exchange stopped (done when ok)
    // False only when no byte of the request left this host: the only case in which the request may be sent again
    // without risking a second execution.
    bool request_sent = false;
    HttpClientHead head;             // valid once stage > head
    std::uint64_t body_bytes = 0;
    std::string error;               // empty when ok
};

[[nodiscard]] HttpClientResult http_exchange(const HttpClientRequest& request, const HttpClientSink& sink);

// Whole-body convenience for small control requests (status routes). `body` is capped at max_body_bytes.
[[nodiscard]] HttpClientResult http_fetch(const HttpClientRequest& request, std::string& body,
    std::uint64_t max_body_bytes = 4ULL << 20U);

} // namespace strix::serve
