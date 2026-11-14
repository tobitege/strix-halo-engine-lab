#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::serve {

struct HttpRequest {
    std::string method, target, path, query, version;
    std::vector<std::pair<std::string, std::string>> headers; // names lower-cased
    std::string body;
    std::string peer;
    // First value of a header (name in lower case), empty when absent.
    [[nodiscard]] std::string header(std::string_view name) const;
};

// One response on one connection. Either send() a complete response, or begin_stream() and then
// write() chunks (HTTP/1.1 chunked transfer encoding) and end_stream(). write() returns false once
// the client has gone away; the caller should then stop producing.
class HttpResponseWriter {
public:
    virtual ~HttpResponseWriter() = default;
    virtual void send(int status, std::string_view content_type, std::string_view body,
        const std::vector<std::pair<std::string, std::string>>& headers = {}) = 0;
    virtual bool begin_stream(int status, std::string_view content_type,
        const std::vector<std::pair<std::string, std::string>>& headers = {}) = 0;
    virtual bool write(std::string_view data) = 0;
    virtual void end_stream() = 0;
    [[nodiscard]] virtual bool client_gone() const = 0;
    // A header the response head of the current request carries (plain, streamed or an error the server sends); the
    // server clears these for every request. A writer without a response head of its own ignores it.
    virtual void add_response_header(std::string name, std::string value) { (void)name; (void)value; }
};

using HttpHandler = std::function<void(const HttpRequest&, HttpResponseWriter&)>;

struct HttpServerConfig {
    std::string bind_address = "127.0.0.1";
    std::uint16_t port = 8080;          // 0 = any free port (see HttpServer::port)
    std::uint64_t max_body_bytes = 64ULL << 20U;
    // The bodies of all requests being received or handled share this budget: past it a request is refused with 503
    // before its body is read, which bounds the memory of concurrent large bodies and of what is parsed from them.
    // 0 = no shared budget.
    std::uint64_t max_inflight_body_bytes = 256ULL << 20U;
    std::uint64_t max_header_bytes = 64ULL << 10U;
    unsigned max_connections = 64;
    int idle_timeout_seconds = 300;     // waiting for the next request on a kept-alive connection
    // A started request must deliver its headers within this many seconds, and its body within that plus one second
    // per MiB (>= 1 MiB/s); a late request gets 408 and the connection closes. 0 = no request deadline.
    int request_timeout_seconds = 60;
    // One send() that makes no progress for this long marks the client gone (a non-reading client cannot hold a
    // connection or a generation); also the TCP user timeout for unacknowledged data. 0 = no write deadline.
    int write_timeout_seconds = 60;
    bool cors = false;                  // Access-Control-Allow-* on every response, OPTIONS answered
};

// Minimal HTTP/1.1 server: one thread per connection, keep-alive, Content-Length request bodies.
class HttpServer {
public:
    HttpServer(HttpServerConfig config, HttpHandler handler);
    ~HttpServer();
    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;
    // Binds and listens (throws on failure); serve() then accepts until stop().
    void listen();
    void serve();
    // Idempotent and safe concurrently with serve()/other stop() calls. Stops
    // admission and interrupts socket waits without changing request deadlines.
    // Join the caller-owned serve thread before destruction. The destructor
    // drains all connection handlers; their captured objects must outlive it.
    void stop();
    [[nodiscard]] std::uint16_t port() const noexcept;
    [[nodiscard]] unsigned active_connections() const noexcept;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_; // connection threads keep it alive until they finish
};

[[nodiscard]] std::string_view http_status_text(int status);

}  // namespace strix::serve
