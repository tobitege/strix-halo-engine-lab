#include "strix/serve/http_client.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstring>
#include <optional>

#ifdef _WIN32
// Winsock under the POSIX names used below. No SIGPIPE on Windows.
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace strix::serve {
namespace {

#ifdef _WIN32
using Socket = SOCKET;
constexpr Socket kNoSocket = INVALID_SOCKET;
using IoSize = int;
constexpr int kSendFlags = 0;
constexpr short kReadable = POLLRDNORM;
constexpr short kWritable = POLLWRNORM;
int poll_one(pollfd& p, int ms) { return WSAPoll(&p, 1, ms); }
void close_socket(Socket fd) { closesocket(fd); }
bool would_block() { const int e = WSAGetLastError(); return e == WSAEWOULDBLOCK || e == WSAEINPROGRESS; }
bool interrupted() { return WSAGetLastError() == WSAEINTR; }
bool set_nonblocking(Socket fd) { u_long on = 1; return ioctlsocket(fd, FIONBIO, &on) == 0; }
std::string socket_error() { return "Winsock error " + std::to_string(WSAGetLastError()); }
bool sockets_ready() {
    static const bool started = [] { WSADATA data; return WSAStartup(MAKEWORD(2, 2), &data) == 0; }();
    return started;
}
#else
using Socket = int;
constexpr Socket kNoSocket = -1;
using IoSize = std::size_t;
#ifdef MSG_NOSIGNAL
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;
#endif
constexpr short kReadable = POLLIN;
constexpr short kWritable = POLLOUT;
int poll_one(pollfd& p, int ms) { return ::poll(&p, 1, ms); }
void close_socket(Socket fd) { ::close(fd); }
bool would_block() { return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINPROGRESS; }
bool interrupted() { return errno == EINTR; }
bool set_nonblocking(Socket fd) {
    const int flags = fcntl(fd, F_GETFL, 0);
    return flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}
std::string socket_error() { return std::strerror(errno); }
bool sockets_ready() { return true; }
#endif

using Clock = std::chrono::steady_clock;
constexpr int kPollSliceMs = 100;

struct SocketGuard {
    Socket fd = kNoSocket;
    ~SocketGuard() { if (fd != kNoSocket) close_socket(fd); }
};

enum class Wait { ready, timeout, cancelled, error };

// Waits for `events` in slices so the sink's cancel is seen within kPollSliceMs. limit_ms <= 0 = no deadline.
Wait wait_for(Socket fd, short events, int limit_ms, const HttpClientSink& sink) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(limit_ms);
    for (;;) {
        if (sink.cancelled && sink.cancelled()) return Wait::cancelled;
        int slice = kPollSliceMs;
        if (limit_ms > 0) {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
            if (left <= 0) return Wait::timeout;
            slice = static_cast<int>(std::min<long long>(left, kPollSliceMs));
        }
        pollfd p{};
        p.fd = fd;
        p.events = events;
        const int n = poll_one(p, slice);
        if (n > 0) return Wait::ready; // readable, writable, or an error/hangup the next call reports
        if (n < 0 && !interrupted()) return Wait::error;
    }
}

std::string lower(std::string_view text) {
    std::string out(text);
    for (auto& c : out) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return out;
}

std::string_view trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) text.remove_prefix(1);
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\r')) text.remove_suffix(1);
    return text;
}

// "HTTP/1.1 200 OK\r\nName: value\r\n..." without the final blank line.
bool parse_head(std::string_view text, HttpClientHead& head) {
    const auto line_end = text.find("\r\n");
    const auto status_line = text.substr(0, line_end);
    if (!status_line.starts_with("HTTP/1.")) return false;
    const auto space = status_line.find(' ');
    if (space == std::string_view::npos || space + 4 > status_line.size()) return false;
    int status = 0;
    const auto* first = status_line.data() + space + 1;
    if (std::from_chars(first, first + 3, status).ec != std::errc{} || status < 100 || status > 599) return false;
    head.status = status;
    head.headers.clear();
    auto rest = line_end == std::string_view::npos ? std::string_view{} : text.substr(line_end + 2);
    while (!rest.empty()) {
        const auto end = rest.find("\r\n");
        const auto line = rest.substr(0, end);
        rest = end == std::string_view::npos ? std::string_view{} : rest.substr(end + 2);
        const auto colon = line.find(':');
        if (colon == std::string_view::npos || colon == 0) return false;
        head.headers.emplace_back(lower(trim(line.substr(0, colon))), std::string(trim(line.substr(colon + 1))));
    }
    return true;
}

// Incremental decoder of the chunked transfer coding. feed() returns false on a framing error or a sink abort.
class ChunkDecoder {
public:
    template<class Emit> bool feed(std::string_view data, Emit&& emit) {
        while (!data.empty() && state_ != State::done) {
            switch (state_) {
            case State::size: {
                const auto take = take_line(data);
                if (!take) { if (line_.size() > 4096) return false; break; }
                auto digits = std::string_view(line_).substr(0, line_.find(';'));
                digits = trim(digits);
                std::uint64_t size = 0;
                const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), size, 16);
                if (digits.empty() || parsed.ec != std::errc{} || parsed.ptr != digits.data() + digits.size()) return false;
                line_.clear();
                left_ = size;
                state_ = size == 0 ? State::trailer : State::data;
                break;
            }
            case State::data: {
                const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(left_, data.size()));
                if (!emit(data.substr(0, n))) { aborted_ = true; return false; }
                data.remove_prefix(n);
                left_ -= n;
                if (left_ == 0) state_ = State::data_end;
                break;
            }
            case State::data_end: {
                const auto take = take_line(data);
                if (!take) { if (line_.size() > 2) return false; break; }
                if (!line_.empty()) return false;
                state_ = State::size;
                break;
            }
            case State::trailer: {
                const auto take = take_line(data);
                if (!take) { if (line_.size() > 8192) return false; break; }
                if (line_.empty()) state_ = State::done;
                line_.clear();
                break;
            }
            case State::done: break;
            }
        }
        return true;
    }
    [[nodiscard]] bool done() const noexcept { return state_ == State::done; }
    [[nodiscard]] bool aborted() const noexcept { return aborted_; }

private:
    enum class State { size, data, data_end, trailer, done };
    // Moves bytes up to a CRLF into line_ (without the CRLF). True once the line is complete.
    bool take_line(std::string_view& data) {
        while (!data.empty()) {
            const char c = data.front();
            data.remove_prefix(1);
            if (c == '\n') {
                if (!line_.empty() && line_.back() == '\r') line_.pop_back();
                return true;
            }
            line_.push_back(c);
        }
        return false;
    }
    State state_ = State::size;
    std::uint64_t left_ = 0;
    std::string line_;
    bool aborted_ = false;
};

Socket connect_to(const HttpClientRequest& request, const HttpClientSink& sink, HttpClientResult& result) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* list = nullptr;
    const auto port = std::to_string(request.port);
    if (getaddrinfo(request.host.c_str(), port.c_str(), &hints, &list) != 0 || !list) {
        result.error = "cannot resolve " + request.host;
        return kNoSocket;
    }
    Socket connected = kNoSocket;
    for (const addrinfo* a = list; a && connected == kNoSocket && !result.cancelled; a = a->ai_next) {
        SocketGuard guard{::socket(a->ai_family, a->ai_socktype, a->ai_protocol)};
        if (guard.fd == kNoSocket) { result.error = socket_error(); continue; }
        if (!set_nonblocking(guard.fd)) { result.error = socket_error(); continue; }
        if (::connect(guard.fd, a->ai_addr, static_cast<socklen_t>(a->ai_addrlen)) != 0) {
            if (!would_block()) { result.error = "connect: " + socket_error(); continue; }
            const auto w = wait_for(guard.fd, kWritable, std::max(1, request.connect_timeout_ms), sink);
            if (w == Wait::cancelled) { result.cancelled = true; result.error = "cancelled"; continue; }
            if (w != Wait::ready) { result.error = w == Wait::timeout ? "connect timed out" : "connect: " + socket_error(); continue; }
            int failure = 0;
            socklen_t size = sizeof(failure);
            if (getsockopt(guard.fd, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&failure), &size) != 0 || failure != 0) {
                result.error = "connect refused or failed (" + std::to_string(failure) + ")";
                continue;
            }
        }
        const int on = 1;
        (void)::setsockopt(guard.fd, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&on), sizeof(on));
        connected = guard.fd;
        guard.fd = kNoSocket;
    }
    freeaddrinfo(list);
    return connected;
}

// Sends all of `data`; counts what left the host in `sent`.
bool send_all(Socket fd, std::string_view data, const HttpClientRequest& request, const HttpClientSink& sink,
    HttpClientResult& result, bool& sent) {
    while (!data.empty()) {
        const auto n = ::send(fd, data.data(), static_cast<IoSize>(std::min<std::size_t>(data.size(), 1U << 20U)), kSendFlags);
        if (n > 0) { sent = true; data.remove_prefix(static_cast<std::size_t>(n)); continue; }
        if (n < 0 && interrupted()) continue;
        if (n < 0 && would_block()) {
            const auto w = wait_for(fd, kWritable, request.idle_timeout_ms, sink);
            if (w == Wait::ready) continue;
            result.cancelled = w == Wait::cancelled;
            result.error = w == Wait::cancelled ? "cancelled" : w == Wait::timeout ? "send timed out" : "send: " + socket_error();
            return false;
        }
        result.error = "send: " + socket_error();
        return false;
    }
    return true;
}

} // namespace

std::string HttpClientHead::header(std::string_view name) const {
    for (const auto& h : headers) if (h.first == name) return h.second;
    return {};
}

HttpClientResult http_exchange(const HttpClientRequest& request, const HttpClientSink& sink) {
    HttpClientResult result;
    if (!sockets_ready()) { result.error = "socket layer unavailable"; return result; }
    SocketGuard socket{connect_to(request, sink, result)};
    if (socket.fd == kNoSocket) return result;
    result.error.clear();

    result.stage = HttpClientStage::send;
    std::string head_out = request.method + " " + (request.target.empty() ? std::string("/") : request.target) + " HTTP/1.1\r\n";
    head_out += "Host: " + request.host + ":" + std::to_string(request.port) + "\r\n";
    for (const auto& h : request.headers) head_out += h.first + ": " + h.second + "\r\n";
    if (!request.body.empty() || request.method == "POST" || request.method == "PUT" || request.method == "PATCH")
        head_out += "Content-Length: " + std::to_string(request.body.size()) + "\r\n";
    head_out += "Connection: close\r\n\r\n";
    if (!send_all(socket.fd, head_out, request, sink, result, result.request_sent)
        || !send_all(socket.fd, request.body, request, sink, result, result.request_sent))
        return result;

    result.stage = HttpClientStage::head;
    std::string buffer;       // head bytes, then the body bytes that arrived with them
    std::string scratch(64U << 10U, '\0');
    bool eof = false;
    // One read; false on error, timeout or cancel (result filled). Sets eof on an orderly close.
    const auto read_more = [&](std::string_view& got) {
        got = {};
        for (;;) {
            const auto w = wait_for(socket.fd, kReadable, request.idle_timeout_ms, sink);
            if (w != Wait::ready) {
                result.cancelled = w == Wait::cancelled;
                result.error = w == Wait::cancelled ? "cancelled" : w == Wait::timeout ? "response timed out" : "poll: " + socket_error();
                return false;
            }
            const auto n = ::recv(socket.fd, scratch.data(), static_cast<IoSize>(scratch.size()), 0);
            if (n > 0) { got = std::string_view(scratch.data(), static_cast<std::size_t>(n)); return true; }
            if (n == 0) { eof = true; return true; }
            if (interrupted() || would_block()) continue;
            result.error = "recv: " + socket_error();
            return false;
        }
    };

    std::string_view got;
    std::size_t head_end = std::string::npos;
    for (;;) {
        // 1xx interim responses are skipped; the final head follows on the same connection.
        while ((head_end = buffer.find("\r\n\r\n")) == std::string::npos) {
            if (buffer.size() > request.max_head_bytes) { result.error = "response head too large"; return result; }
            if (!read_more(got)) return result;
            if (eof) { result.error = buffer.empty() ? "connection closed before any response" : "truncated response head"; return result; }
            buffer.append(got);
        }
        if (!parse_head(std::string_view(buffer).substr(0, head_end), result.head)) { result.error = "malformed response head"; return result; }
        buffer.erase(0, head_end + 4);
        if (result.head.status >= 200) break;
    }

    result.stage = HttpClientStage::body;
    if (sink.on_head && !sink.on_head(result.head)) { result.cancelled = true; result.error = "aborted by sink"; return result; }
    const auto emit = [&](std::string_view data) {
        if (data.empty()) return true;
        result.body_bytes += data.size();
        return !sink.on_data || sink.on_data(data);
    };
    const auto aborted = [&] { result.cancelled = true; result.error = "aborted by sink"; return result; };

    const bool no_body = request.method == "HEAD" || result.head.status == 204 || result.head.status == 304;
    const bool chunked = lower(result.head.header("transfer-encoding")).find("chunked") != std::string::npos;
    std::optional<std::uint64_t> length;
    if (const auto text = result.head.header("content-length"); !chunked && !text.empty()) {
        std::uint64_t value = 0;
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) { result.error = "malformed Content-Length"; return result; }
        length = value;
    }
    if (no_body) {
        // nothing to read
    } else if (chunked) {
        ChunkDecoder decoder;
        std::string_view data = buffer;
        for (;;) {
            if (!decoder.feed(data, emit)) {
                if (decoder.aborted()) return aborted();
                result.error = "malformed chunked body";
                return result;
            }
            if (decoder.done()) break;
            if (!read_more(data)) return result;
            if (eof) { result.error = "connection closed inside a chunked body"; return result; }
        }
    } else if (length) {
        std::uint64_t left = *length;
        std::string_view data = buffer;
        for (;;) {
            const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(left, data.size()));
            if (!emit(data.substr(0, n))) return aborted();
            left -= n;
            if (left == 0) break;
            if (!read_more(data)) return result;
            if (eof) { result.error = "connection closed before the full body"; return result; }
        }
    } else {
        std::string_view data = buffer;
        for (;;) {
            if (!emit(data)) return aborted();
            if (eof) break;
            if (!read_more(data)) return result;
        }
    }
    result.stage = HttpClientStage::done;
    result.ok = true;
    return result;
}

HttpClientResult http_fetch(const HttpClientRequest& request, std::string& body, std::uint64_t max_body_bytes) {
    body.clear();
    HttpClientSink sink;
    sink.on_data = [&](std::string_view data) {
        if (body.size() + data.size() > max_body_bytes) return false;
        body.append(data);
        return true;
    };
    return http_exchange(request, sink);
}

} // namespace strix::serve
