// Minimal portable HTTP/1.1 (Winsock / POSIX) for the CUDA Qwen prefill service and its state hand-off.
// Server helpers and a one-shot POST as in the DeepSeek prefill service (apps/cuda_ds41_prefill_service_main.cu),
// plus a keep-alive client that reuses one connection for many requests and reconnects when the peer closes it.
#pragma once

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix::qwen_http {

#ifdef _WIN32
using Sock = SOCKET;
inline const Sock kNoSock = INVALID_SOCKET;
inline void close_sock(Sock s) { closesocket(s); }
inline void set_timeouts(Sock s, int seconds) {
    const DWORD ms = static_cast<DWORD>(seconds) * 1000U;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&ms), sizeof(ms));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&ms), sizeof(ms));
}
// Call once per process before any socket use.
inline void startup() {
    static const bool ok = [] { WSADATA w; return WSAStartup(MAKEWORD(2, 2), &w) == 0; }();
    if (!ok) throw std::runtime_error("WSAStartup failed");
}
#else
using Sock = int;
inline const Sock kNoSock = -1;
inline void close_sock(Sock s) { ::close(s); }
inline void set_timeouts(Sock s, int seconds) {
    timeval tv{};
    tv.tv_sec = seconds;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}
inline void startup() {}
#endif

inline bool send_all(Sock s, const char* p, std::size_t n) {
    while (n > 0) {
        const int piece = static_cast<int>(std::min<std::size_t>(n, 1U << 20));
#ifdef _WIN32
        const int k = ::send(s, p, piece, 0);
#else
        const auto k = static_cast<int>(::send(s, p, static_cast<std::size_t>(piece), MSG_NOSIGNAL));
#endif
        if (k <= 0) return false;
        p += k;
        n -= static_cast<std::size_t>(k);
    }
    return true;
}

inline int recv_some(Sock s, char* p, std::size_t n) {
    return static_cast<int>(::recv(s, p, static_cast<int>(std::min<std::size_t>(n, 1U << 20)), 0));
}

inline std::string lower(std::string v) {
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return v;
}

inline std::string trim(const std::string& v) {
    std::size_t a = 0, b = v.size();
    while (a < b && (v[a] == ' ' || v[a] == '\t')) ++a;
    while (b > a && (v[b - 1] == ' ' || v[b - 1] == '\t' || v[b - 1] == '\r')) --b;
    return v.substr(a, b - a);
}

// One HTTP/1.1 message (request or response): start line, lower-case headers, Content-Length body.
struct Message {
    std::string start;
    std::map<std::string, std::string> headers;
    std::string body;
    [[nodiscard]] std::string header(const std::string& k) const {
        const auto it = headers.find(k);
        return it == headers.end() ? std::string() : it->second;
    }
};

// Reads one message from `s`, starting with any bytes already in `carry` (and leaving there what follows the message,
// so a keep-alive connection can read the next one). Without Content-Length a response body runs to the connection's
// end (requests: no body). Returns an error text, empty on success.
inline std::string read_message(Sock s, Message& m, std::size_t max_body, bool is_response, std::string& carry) {
    std::string buf = std::move(carry);
    carry.clear();
    char tmp[65536];
    std::size_t head_end = std::string::npos;
    while ((head_end = buf.find("\r\n\r\n")) == std::string::npos) {
        if (buf.size() > 65536) return "headers too large";
        const int k = recv_some(s, tmp, sizeof(tmp));
        if (k <= 0) return buf.empty() ? "connection closed" : "truncated headers";
        buf.append(tmp, static_cast<std::size_t>(k));
    }
    const std::string head = buf.substr(0, head_end);
    std::size_t pos = head.find("\r\n");
    m.start = head.substr(0, pos);
    m.headers.clear();
    while (pos != std::string::npos) {
        const std::size_t next = head.find("\r\n", pos + 2);
        const std::string line = head.substr(pos + 2, next == std::string::npos ? std::string::npos : next - pos - 2);
        if (const auto colon = line.find(':'); colon != std::string::npos) m.headers[lower(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1));
        pos = next;
    }
    m.body = buf.substr(head_end + 4);
    if (!m.header("transfer-encoding").empty() && lower(m.header("transfer-encoding")) != "identity") return "chunked bodies are not supported";
    const std::string cl = m.header("content-length");
    if (cl.empty()) {
        if (!is_response) { carry = std::move(m.body); m.body.clear(); return {}; }
        for (;;) {
            const int k = recv_some(s, tmp, sizeof(tmp));
            if (k <= 0) break;
            m.body.append(tmp, static_cast<std::size_t>(k));
            if (m.body.size() > max_body) return "body too large";
        }
        return {};
    }
    std::size_t want = 0;
    try {
        want = static_cast<std::size_t>(std::stoull(cl));
    } catch (const std::exception&) {
        return "bad Content-Length";
    }
    if (want > max_body) return "body too large";
    if (m.body.size() > want) { carry = m.body.substr(want); m.body.resize(want); }
    m.body.reserve(want);
    while (m.body.size() < want) {
        const int k = recv_some(s, tmp, std::min(sizeof(tmp), want - m.body.size()));
        if (k <= 0) return "truncated body";
        m.body.append(tmp, static_cast<std::size_t>(k));
    }
    return {};
}

inline const char* reason(int status) {
    switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 409: return "Conflict";
    case 413: return "Payload Too Large";
    case 500: return "Internal Server Error";
    case 502: return "Bad Gateway";
    case 503: return "Service Unavailable";
    default: return "Status";
    }
}

inline void respond(Sock s, int status, const std::string& body, const char* type = "application/json") {
    std::string head = "HTTP/1.1 " + std::to_string(status) + " " + reason(status) + "\r\nContent-Type: " + type +
                       "\r\nContent-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n";
    if (send_all(s, head.data(), head.size())) send_all(s, body.data(), body.size());
}

struct Url {
    std::string host, port = "80", path = "/";
};

inline Url parse_url(const std::string& u) {
    if (u.rfind("http://", 0) != 0) throw std::invalid_argument("URL must start with http://");
    Url r;
    const std::string rest = u.substr(7);
    const auto slash = rest.find('/');
    std::string hostport = rest.substr(0, slash);
    if (slash != std::string::npos) r.path = rest.substr(slash);
    if (!hostport.empty() && hostport[0] == '[') {
        const auto close = hostport.find(']');
        if (close == std::string::npos) throw std::invalid_argument("bad URL host");
        r.host = hostport.substr(1, close - 1);
        if (close + 1 < hostport.size() && hostport[close + 1] == ':') r.port = hostport.substr(close + 2);
    } else if (const auto colon = hostport.rfind(':'); colon != std::string::npos) {
        r.host = hostport.substr(0, colon);
        r.port = hostport.substr(colon + 1);
    } else {
        r.host = hostport;
    }
    if (r.host.empty()) throw std::invalid_argument("URL without a host");
    return r;
}

inline Sock connect_to(const Url& u, int timeout_s) {
    startup();
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(u.host.c_str(), u.port.c_str(), &hints, &res) != 0 || res == nullptr) throw std::runtime_error("cannot resolve " + u.host);
    Sock s = kNoSock;
    for (auto* a = res; a != nullptr; a = a->ai_next) {
        s = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (s == kNoSock) continue;
        if (::connect(s, a->ai_addr, static_cast<int>(a->ai_addrlen)) == 0) break;
        close_sock(s);
        s = kNoSock;
    }
    freeaddrinfo(res);
    if (s == kNoSock) throw std::runtime_error("cannot connect to " + u.host + ":" + u.port);
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));
    // Large send buffer: the hand-off streams hundreds of MB.
    int big = 8 << 20;
    setsockopt(s, SOL_SOCKET, SO_SNDBUF, reinterpret_cast<const char*>(&big), sizeof(big));
    set_timeouts(s, timeout_s);
    return s;
}

using Headers = std::vector<std::pair<std::string, std::string>>;

// A client bound to one origin (scheme http, host, port) that keeps its connection open between requests
// (HTTP/1.1 keep-alive). If the server closes it (Connection: close or EOF), the next request reconnects. A request
// whose send fails on a reused connection is retried once on a fresh connection.
class Client {
public:
    Client(std::string origin, int timeout_s) : origin_(parse_url(origin + "/")), timeout_s_(timeout_s) {}
    ~Client() { drop(); }
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    // Returns (status, reply body).
    std::pair<int, std::string> request(const std::string& method, const std::string& path, const Headers& headers,
                                        std::span<const std::uint8_t> body, std::size_t max_reply = std::size_t{64} << 20) {
        for (int attempt = 0; attempt < 2; ++attempt) {
            const bool reused = s_ != kNoSock;
            if (!reused) { s_ = connect_to(origin_, timeout_s_); carry_.clear(); }
            std::string head = method + " " + path + " HTTP/1.1\r\nHost: " + origin_.host + ":" + origin_.port +
                               "\r\nConnection: keep-alive\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
            for (const auto& [k, v] : headers) head += k + ": " + v + "\r\n";
            head += "\r\n";
            const bool sent = send_all(s_, head.data(), head.size()) &&
                              (body.empty() || send_all(s_, reinterpret_cast<const char*>(body.data()), body.size()));
            if (!sent) {
                drop();
                if (reused) continue;
                throw std::runtime_error("send failed to " + origin_.host + ":" + origin_.port);
            }
            Message m;
            const std::string err = read_message(s_, m, max_reply, true, carry_);
            if (!err.empty()) {
                drop();
                if (reused && err == "connection closed") continue;
                throw std::runtime_error("reply from " + origin_.host + ": " + err);
            }
            if (lower(m.header("connection")) == "close" || m.header("content-length").empty()) drop();
            const auto sp = m.start.find(' ');
            if (sp == std::string::npos) throw std::runtime_error("bad status line: " + m.start);
            return {std::atoi(m.start.c_str() + sp + 1), std::move(m.body)};
        }
        throw std::runtime_error("request failed twice on " + origin_.host);
    }
    std::pair<int, std::string> get(const std::string& path, const Headers& headers = {}) {
        return request("GET", path, headers, {});
    }
    std::pair<int, std::string> post(const std::string& path, const Headers& headers, std::span<const std::uint8_t> body) {
        return request("POST", path, headers, body);
    }

private:
    void drop() {
        if (s_ != kNoSock) close_sock(s_);
        s_ = kNoSock;
        carry_.clear();
    }
    Url origin_;
    int timeout_s_;
    Sock s_ = kNoSock;
    std::string carry_;
};

}  // namespace strix::qwen_http
