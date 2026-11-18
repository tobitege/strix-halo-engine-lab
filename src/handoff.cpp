#include "strix/handoff.hpp"
#include "strix/env_flags.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <map>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

static_assert(std::endian::native == std::endian::little, "strix::handoff frames are little-endian; big-endian hosts are not supported");

namespace strix::handoff {
namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// ---- sockets ---------------------------------------------------------------------------------------------------
#ifdef _WIN32
using sock_t = SOCKET;
using sock_count_t = int;
constexpr sock_t kBad = INVALID_SOCKET;
void close_sock(sock_t s) { ::closesocket(s); }
// Winsock does not wake a recv()/send() blocked on another thread on shutdown(); CancelIoEx does (the call then
// fails with WSA_OPERATION_ABORTED). The handle stays open until the owner closes it, so there is no reuse race.
void shutdown_sock(sock_t s) {
    ::shutdown(s, SD_BOTH);
    ::CancelIoEx(reinterpret_cast<HANDLE>(s), nullptr);
}
constexpr int kSendFlags = 0;
struct NetInit {
    NetInit() {
        WSADATA w;
        if (WSAStartup(MAKEWORD(2, 2), &w) != 0) throw std::runtime_error("handoff: WSAStartup failed");
    }
};
void net_init() { static NetInit once; }
int last_error() { return WSAGetLastError(); }
#else
using sock_t = int;
using sock_count_t = std::size_t;
constexpr sock_t kBad = -1;
void close_sock(sock_t s) { ::close(s); }
void shutdown_sock(sock_t s) { ::shutdown(s, SHUT_RDWR); }
constexpr int kSendFlags = MSG_NOSIGNAL;
void net_init() {}
int last_error() { return errno; }
#endif

struct Socket {  // owns one connected stream
    sock_t fd = kBad;
    Socket() = default;
    explicit Socket(sock_t s) : fd(s) {}
    Socket(Socket&& o) noexcept : fd(o.fd) { o.fd = kBad; }
    Socket& operator=(Socket&& o) noexcept {
        if (this != &o) {
            reset();
            fd = o.fd;
            o.fd = kBad;
        }
        return *this;
    }
    ~Socket() { reset(); }
    void reset() {
        if (fd != kBad) close_sock(fd);
        fd = kBad;
    }
};

void set_opts(sock_t s, int buffer, bool send_side) {
    int one = 1;
    ::setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));
    if (buffer > 0) ::setsockopt(s, SOL_SOCKET, send_side ? SO_SNDBUF : SO_RCVBUF, reinterpret_cast<const char*>(&buffer), sizeof(buffer));
}

void set_timeout(sock_t s, std::chrono::milliseconds t, bool send_side) {
    const int opt = send_side ? SO_SNDTIMEO : SO_RCVTIMEO;
#ifdef _WIN32
    const DWORD ms = static_cast<DWORD>(t.count());
    ::setsockopt(s, SOL_SOCKET, opt, reinterpret_cast<const char*>(&ms), sizeof(ms));
#else
    timeval tv{};
    tv.tv_sec = static_cast<time_t>(t.count() / 1000);
    tv.tv_usec = static_cast<suseconds_t>((t.count() % 1000) * 1000);
    ::setsockopt(s, SOL_SOCKET, opt, &tv, sizeof(tv));
#endif
}

void send_all(sock_t s, const void* p, std::size_t n) {
    const char* b = static_cast<const char*>(p);
    while (n) {
        const auto k = ::send(s, b, static_cast<sock_count_t>(std::min<std::size_t>(n, std::size_t{1} << 24)), kSendFlags);
        if (k <= 0) throw std::runtime_error("handoff: send failed (error " + std::to_string(last_error()) + ")");
        b += k;
        n -= static_cast<std::size_t>(k);
    }
}

// MSG_WAITALL: one wake-up per request instead of one per arriving segment burst. Without it a receiver at 1 GbE
// burned ~1 core for 118 MB/s (8.5 s CPU per GB; 0.1-0.3 s per GB on loopback, where bursts are large). A timeout or a
// cancel still returns the partial count or an error, which the loop handles as before.
// Rollback: STRIX_HANDOFF_RECV_WAITALL=0 (recorded for --print-effective-config).
void recv_all(sock_t s, void* p, std::size_t n) {
    static const int flags = envcfg::flag("STRIX_HANDOFF_RECV_WAITALL", true) ? MSG_WAITALL : 0;
    char* b = static_cast<char*>(p);
    while (n) {
        const auto k = ::recv(s, b, static_cast<sock_count_t>(std::min<std::size_t>(n, std::size_t{1} << 24)), flags);
        if (k == 0) throw std::runtime_error("handoff: connection closed by the peer");
        if (k < 0) throw std::runtime_error("handoff: receive failed, cancelled or timed out (error " + std::to_string(last_error()) + ")");
        b += k;
        n -= static_cast<std::size_t>(k);
    }
}

sockaddr_in resolve(const std::string& host, std::uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (::getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || res == nullptr) throw std::runtime_error("handoff: cannot resolve " + host);
    sockaddr_in a{};
    std::memcpy(&a, res->ai_addr, sizeof(a));
    ::freeaddrinfo(res);
    a.sin_port = htons(port);
    return a;
}

void set_blocking(sock_t s, bool on) {
#ifdef _WIN32
    u_long nb = on ? 0 : 1;
    ::ioctlsocket(s, FIONBIO, &nb);
#else
    const int fl = ::fcntl(s, F_GETFL, 0);
    ::fcntl(s, F_SETFL, on ? (fl & ~O_NONBLOCK) : (fl | O_NONBLOCK));
#endif
}

// One non-blocking connect attempt bounded by `until`; true when connected.
bool connect_once(sock_t s, const sockaddr_in& to, Clock::time_point until) {
    set_blocking(s, false);
    if (::connect(s, reinterpret_cast<const sockaddr*>(&to), sizeof(to)) == 0) {
        set_blocking(s, true);
        return true;
    }
#ifdef _WIN32
    if (WSAGetLastError() != WSAEWOULDBLOCK) return false;
#else
    if (errno != EINPROGRESS) return false;
#endif
    const auto left = std::chrono::duration_cast<std::chrono::microseconds>(until - Clock::now()).count();
    if (left <= 0) return false;
    fd_set wr, ex;
    FD_ZERO(&wr);
    FD_ZERO(&ex);
    FD_SET(s, &wr);
    FD_SET(s, &ex);
    timeval tv{};
    tv.tv_sec = static_cast<long>(left / 1000000);
    tv.tv_usec = static_cast<long>(left % 1000000);
    if (::select(static_cast<int>(s + 1), nullptr, &wr, &ex, &tv) <= 0 || !FD_ISSET(s, &wr)) return false;
    int err = 0;
#ifdef _WIN32
    int len = sizeof(err);
#else
    socklen_t len = sizeof(err);
#endif
    ::getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&err), &len);
    if (err != 0) return false;
    set_blocking(s, true);
    return true;
}

// Retries until the deadline (the ranks may still be loading); every attempt is bounded by the deadline too.
Socket connect_to(const Endpoint& e, int buffer, std::chrono::seconds timeout) {
    net_init();
    const sockaddr_in to = resolve(e.host, e.port);
    const auto deadline = Clock::now() + timeout;
    for (;;) {
        Socket s(::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
        if (s.fd == kBad) throw std::runtime_error("handoff: socket()");
        set_opts(s.fd, buffer, true);
        if (!e.bind.empty()) {
            const sockaddr_in local = resolve(e.bind, 0);
            if (::bind(s.fd, reinterpret_cast<const sockaddr*>(&local), sizeof(local)) != 0)
                throw std::runtime_error("handoff: cannot bind the local address " + e.bind);
        }
        if (connect_once(s.fd, to, deadline)) return s;
        if (Clock::now() >= deadline) throw std::runtime_error("handoff: cannot connect to " + e.host + ":" + std::to_string(e.port));
        std::this_thread::sleep_for(std::min<Clock::duration>(std::chrono::seconds(1), deadline - Clock::now()));
    }
}

// ---- frames ----------------------------------------------------------------------------------------------------
constexpr std::uint32_t kMagic = 0x314f4853U;  // "SHO1"
constexpr std::uint16_t kVersion = 2;
enum Type : std::uint8_t { kHello = 1, kAccept = 2, kPiece = 3, kEnd = 4, kReply = 5, kAbort = 6 };
constexpr std::uint64_t kMaxControlPayload = std::uint64_t{1} << 20;

struct Frame {
    std::uint32_t magic = kMagic;
    std::uint16_t version = kVersion;
    std::uint8_t type = 0, codec = 0;
    std::uint32_t flags = 0, kind = 0, index = 0, seq = 0;
    std::uint64_t row0 = 0, rows = 0, plain_bytes = 0, wire_bytes = 0, hash = 0;
};
static_assert(sizeof(Frame) == 64, "frame header is 64 bytes");

struct Descriptor {  // the bytes the piece hash binds before the plain bytes
    std::uint32_t seq, kind, index, flags;
    std::uint64_t row0, rows, plain_bytes;
};
static_assert(sizeof(Descriptor) == 40, "descriptor is 40 bytes");

void hash_descriptor(Xxh64& h, const Frame& f) {
    const Descriptor d{f.seq, f.kind, f.index, f.flags, f.row0, f.rows, f.plain_bytes};
    h.update(&d, sizeof(d));
}

void send_frame(sock_t s, const Frame& f, const void* payload = nullptr) {
    send_all(s, &f, sizeof(f));
    if (f.wire_bytes) send_all(s, payload, f.wire_bytes);
}

Frame recv_header(sock_t s, std::uint64_t max_piece) {
    Frame f;
    recv_all(s, &f, sizeof(f));
    if (f.magic != kMagic || f.version != kVersion) throw std::runtime_error("handoff: bad frame magic / version");
    if (f.type < kHello || f.type > kAbort) throw std::runtime_error("handoff: bad frame type");
    if (f.type == kPiece ? (f.wire_bytes > max_piece || f.plain_bytes > max_piece) : f.wire_bytes > kMaxControlPayload)
        throw std::runtime_error("handoff: frame larger than allowed");
    return f;
}

std::string recv_abort_text(sock_t s, const Frame& f) {
    std::string m(f.wire_bytes, '\0');
    if (!m.empty()) recv_all(s, m.data(), m.size());
    return "handoff: aborted by the peer: " + m;
}

void send_abort(sock_t s, std::string_view why) {  // best effort, bounded
    set_timeout(s, std::chrono::milliseconds(1000), true);
    Frame f;
    f.type = kAbort;
    f.wire_bytes = f.plain_bytes = std::min<std::size_t>(why.size(), 4096);
    send_frame(s, f, why.data());
}

void put32(std::vector<std::uint8_t>& v, std::uint32_t x) { v.insert(v.end(), reinterpret_cast<std::uint8_t*>(&x), reinterpret_cast<std::uint8_t*>(&x) + 4); }
void put64(std::vector<std::uint8_t>& v, std::uint64_t x) { v.insert(v.end(), reinterpret_cast<std::uint8_t*>(&x), reinterpret_cast<std::uint8_t*>(&x) + 8); }
struct Cursor {
    const std::uint8_t* p;
    std::size_t n;
    template <class T> T get() {
        if (n < sizeof(T)) throw std::runtime_error("handoff: short HELLO");
        T x;
        std::memcpy(&x, p, sizeof(T));
        p += sizeof(T);
        n -= sizeof(T);
        return x;
    }
    std::vector<std::uint8_t> bytes(std::size_t k) {
        if (n < k) throw std::runtime_error("handoff: short HELLO");
        std::vector<std::uint8_t> v(p, p + k);
        p += k;
        n -= k;
        return v;
    }
};

constexpr std::size_t kMaxModel = 256, kMaxMeta = 65536;

void check_hello(const Hello& h) {
    if (h.model.size() > kMaxModel || h.meta.size() > kMaxMeta) throw std::invalid_argument("handoff: Hello model <= 256, meta <= 64 KiB");
    if (h.world == 0 || h.world > kMaxStreams || h.rank >= h.world) throw std::invalid_argument("handoff: Hello needs rank < world <= 64");
}

// Hello payload: u64 session | u32 rank | u32 world | u32 stream | u32 streams | u32 codec mask | u32 model bytes |
// u32 meta bytes | model | meta
std::vector<std::uint8_t> hello_payload(const Hello& h, std::uint32_t stream, std::uint32_t streams, std::uint32_t codecs) {
    check_hello(h);
    std::vector<std::uint8_t> v;
    put64(v, h.session);
    put32(v, h.rank);
    put32(v, h.world);
    put32(v, stream);
    put32(v, streams);
    put32(v, codecs);
    put32(v, static_cast<std::uint32_t>(h.model.size()));
    put32(v, static_cast<std::uint32_t>(h.meta.size()));
    v.insert(v.end(), h.model.begin(), h.model.end());
    v.insert(v.end(), h.meta.begin(), h.meta.end());
    return v;
}

// Seed of the manifest fold: the session's identity as both ends see it (a corrupted HELLO fails at END).
std::uint64_t hello_seed(const Hello& h) {
    Xxh64 x;
    const std::uint64_t head[3] = {h.session, (static_cast<std::uint64_t>(h.rank) << 32) | h.world,
                                   (static_cast<std::uint64_t>(h.model.size()) << 32) | h.meta.size()};
    x.update(head, sizeof(head));
    x.update(h.model.data(), h.model.size());
    x.update(h.meta.data(), h.meta.size());
    return x.digest();
}

struct Manifest {
    std::uint64_t pieces = 0, plain = 0, wire = 0, fold = 0;
    bool operator==(const Manifest&) const = default;
};

// Runs a caller's release callback exactly once: explicitly, or when the owner is destroyed on any path.
class Release {
public:
    Release() = default;
    explicit Release(std::function<void()> fn) : fn_(std::move(fn)) {}
    Release(Release&& o) noexcept : fn_(std::move(o.fn_)) { o.fn_ = nullptr; }
    Release& operator=(Release&& o) noexcept {
        if (this != &o) {
            run();
            fn_ = std::move(o.fn_);
            o.fn_ = nullptr;
        }
        return *this;
    }
    Release(const Release&) = delete;
    Release& operator=(const Release&) = delete;
    ~Release() { run(); }
    void run() noexcept {
        if (!fn_) return;
        auto f = std::move(fn_);
        fn_ = nullptr;
        try {
            f();
        } catch (...) {
        }
    }

private:
    std::function<void()> fn_;
};

}  // namespace

// ---- XXH64 -----------------------------------------------------------------------------------------------------
namespace {
constexpr std::uint64_t P1 = 11400714785074694791ULL, P2 = 14029467366897019727ULL, P3 = 1609587929392839161ULL,
                        P4 = 9650029242287828579ULL, P5 = 2870177450012600261ULL;
inline std::uint64_t rotl(std::uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
inline std::uint64_t round1(std::uint64_t acc, std::uint64_t in) { return rotl(acc + in * P2, 31) * P1; }
inline std::uint64_t merge(std::uint64_t acc, std::uint64_t v) { return (acc ^ round1(0, v)) * P1 + P4; }
inline std::uint64_t rd64(const std::uint8_t* p) {
    std::uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}
inline std::uint32_t rd32(const std::uint8_t* p) {
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
}  // namespace

Xxh64::Xxh64(std::uint64_t seed) noexcept : v_{seed + P1 + P2, seed + P2, seed, seed - P1}, seed_(seed), buf_{} {}

void Xxh64::update(const void* data, std::size_t bytes) noexcept {
    const auto* p = static_cast<const std::uint8_t*>(data);
    total_ += bytes;
    if (fill_ + bytes < 32) {
        if (bytes) std::memcpy(buf_ + fill_, p, bytes);
        fill_ += bytes;
        return;
    }
    if (fill_) {
        const std::size_t k = 32 - fill_;
        std::memcpy(buf_ + fill_, p, k);
        for (int i = 0; i < 4; ++i) v_[i] = round1(v_[i], rd64(buf_ + 8 * i));
        p += k;
        bytes -= k;
        fill_ = 0;
    }
    // Locals: stores to v_ through `this` would alias the byte pointer and force reloads every round.
    std::uint64_t v0 = v_[0], v1 = v_[1], v2 = v_[2], v3 = v_[3];
    while (bytes >= 32) {
        v0 = round1(v0, rd64(p));
        v1 = round1(v1, rd64(p + 8));
        v2 = round1(v2, rd64(p + 16));
        v3 = round1(v3, rd64(p + 24));
        p += 32;
        bytes -= 32;
    }
    v_[0] = v0;
    v_[1] = v1;
    v_[2] = v2;
    v_[3] = v3;
    if (bytes) std::memcpy(buf_, p, bytes);
    fill_ = bytes;
}

std::uint64_t Xxh64::digest() const noexcept {
    std::uint64_t h;
    if (total_ >= 32) {
        h = rotl(v_[0], 1) + rotl(v_[1], 7) + rotl(v_[2], 12) + rotl(v_[3], 18);
        for (int i = 0; i < 4; ++i) h = merge(h, v_[i]);
    } else {
        h = seed_ + P5;
    }
    h += total_;
    const std::uint8_t* p = buf_;
    std::size_t n = fill_;
    while (n >= 8) {
        h ^= round1(0, rd64(p));
        h = rotl(h, 27) * P1 + P4;
        p += 8;
        n -= 8;
    }
    if (n >= 4) {
        h ^= static_cast<std::uint64_t>(rd32(p)) * P1;
        h = rotl(h, 23) * P2 + P3;
        p += 4;
        n -= 4;
    }
    while (n) {
        h ^= static_cast<std::uint64_t>(*p) * P5;
        h = rotl(h, 11) * P1;
        ++p;
        --n;
    }
    h ^= h >> 33;
    h *= P2;
    h ^= h >> 29;
    h *= P3;
    h ^= h >> 32;
    return h;
}

std::uint64_t piece_hash(std::uint32_t seq, const PieceHeader& header, std::span<const bp_codec::Region> regions) {
    Frame f;
    f.seq = seq;
    f.kind = header.kind;
    f.index = header.index;
    f.flags = header.flags;
    f.row0 = header.row0;
    f.rows = header.rows;
    for (const auto& r : regions) f.plain_bytes += r.bytes;
    Xxh64 h;
    hash_descriptor(h, f);
    for (const auto& r : regions) h.update(r.data, r.bytes);
    return h.digest();
}

// ---- path map --------------------------------------------------------------------------------------------------
std::vector<Endpoint> parse_path_map(std::string_view text) {
    std::vector<Endpoint> out;
    while (!text.empty()) {
        const std::size_t comma = text.find(',');
        std::string_view item = text.substr(0, comma);
        text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1);
        Endpoint e;
        const std::size_t at = item.find('@');
        if (at != std::string_view::npos) {
            e.bind = std::string(item.substr(at + 1));
            item = item.substr(0, at);
        }
        const std::size_t colon = item.rfind(':');
        if (colon == std::string_view::npos || colon == 0 || colon + 1 == item.size()) throw std::invalid_argument("handoff: path map entry needs HOST:PORT");
        e.host = std::string(item.substr(0, colon));
        const unsigned long port = std::stoul(std::string(item.substr(colon + 1)));
        if (port == 0 || port > 65535) throw std::invalid_argument("handoff: bad port in the path map");
        e.port = static_cast<std::uint16_t>(port);
        out.push_back(std::move(e));
    }
    if (out.empty()) throw std::invalid_argument("handoff: empty path map");
    return out;
}

// ---- sender ----------------------------------------------------------------------------------------------------
struct Sender::Impl {
    // caller bytes submitted but not yet released (max_pending_plain_bytes); its own lock, never held in a callback.
    // Declared first: the queues below hold Release objects that touch it.
    std::mutex pm;
    std::condition_variable pcv;
    std::size_t pending = 0;
    std::atomic<bool> dead{false};  // failed or aborted (read without `m` by a waiting submit())

    SenderOptions opt;
    std::uint64_t session = 0;
    std::vector<Socket> socks;
    Codec codec = Codec::identity;
    Clock::time_point t_open = Clock::now();
    std::unique_ptr<bp_codec::WorkerPool> pool;

    struct Input {
        Frame f;
        std::vector<bp_codec::Region> regions;
        Release release;
        bool identity = false;  // uncoded even in a bp-v1 session
    };
    struct Output {
        Frame f;
        std::vector<std::uint8_t> wire;          // bp-v1
        std::vector<bp_codec::Region> regions;  // identity: sent from the caller's memory
        Release release;                         // identity: after sending
    };

    std::mutex m;
    std::condition_variable cv;
    std::deque<Input> inputs;
    std::deque<std::deque<Output>> outputs;  // deque: MSVC deque moves are not noexcept, a vector would copy
    bool closed = false, aborted = false, encoder_finished = false;
    std::exception_ptr error;
    std::function<void(std::exception_ptr)> failure_notify;
    std::size_t queued_wire = 0;
    std::uint32_t next_seq = 0;
    std::uint64_t wire_total = 0, plain_total = 0, pieces = 0;
    Xxh64 fold;
    double encode_s = 0, first_byte = -1, last_byte = -1;
    std::thread encoder;
    std::vector<std::thread> writers;
    bool finished = false;
    std::atomic<bool> finishing{false};  // finish() owns the joins; abort() from another thread then only signals

    Impl() : fold(0) {}

    void shutdown_all() noexcept {
        for (auto& s : socks)
            if (s.fd != kBad) shutdown_sock(s.fd);
    }

    // First failure: latch, wake every thread, and cut the sockets so a writer blocked in send() returns too.
    void fail(std::exception_ptr e) {
        std::function<void(std::exception_ptr)> notify;
        {
            std::lock_guard<std::mutex> g(m);
            if (!error) { error = e; notify = failure_notify; }
            aborted = true;
            cv.notify_all();
        }
        dead = true;
        {
            std::lock_guard<std::mutex> g(pm);
            pcv.notify_all();
        }
        shutdown_all();
        if (notify) notify(e);
        // Workers own their in-flight pieces; queued pieces can be released now, outside m.
        drop_queues();
    }

    // Hash + encode one piece (identity: hash only, the bytes stay the caller's). Runs on the encoder thread or, for a
    // batch, on a pool worker (bp_codec::encode then nests its block tasks in the same pool).
    void prepare(Input& in, Output& out) {
        out.f = in.f;
        std::uint64_t plain = 0;
        for (const auto& r : in.regions) plain += r.bytes;
        out.f.plain_bytes = plain;
        Xxh64 h;
        hash_descriptor(h, out.f);
        for (const auto& r : in.regions) h.update(r.data, r.bytes);
        out.f.hash = h.digest();
        if (codec == Codec::bp_v1 && plain > 0 && !in.identity) {
            out.wire = bp_codec::encode(in.regions, pool.get(), opt.encode);
            out.f.codec = static_cast<std::uint8_t>(Codec::bp_v1);
            out.f.wire_bytes = out.wire.size();
            in.release.run();
        } else {
            out.f.codec = static_cast<std::uint8_t>(Codec::identity);
            out.f.wire_bytes = plain;
            out.regions = std::move(in.regions);
            out.release = std::move(in.release);
        }
    }

    std::size_t encode_reservation(const Input& in) const {
        if (codec != Codec::bp_v1 || std::none_of(in.regions.begin(), in.regions.end(),
            [](const bp_codec::Region& r) { return r.bytes != 0; })) return 0;
        // bp-v1 has <=11 bits/symbol, <=4 plane headers per block, a 16-byte
        // directory entry, and <2 KiB BlockPlan objects. 8 KiB/block covers
        // metadata plus vector growth (old and new plans coexist); 2x plain
        // bounds even an explicitly permitted expanding Huffman encoding.
        // Fixed worker stacks and allocator bookkeeping are outside this budget.
        std::size_t bytes = 16;
        const auto add = [&](std::size_t n, std::size_t scale) {
            if (n > (std::numeric_limits<std::size_t>::max() - bytes) / scale)
                throw std::length_error("handoff: encoding reservation overflow");
            bytes += n * scale;
        };
        for (const auto& r : in.regions) {
            const std::size_t w = r.width == 2 || r.width == 4 ? r.width : 1;
            const auto step = std::max(w, std::min(opt.encode.block_bytes, std::size_t{1} << 30) / w * w);
            add(r.bytes, 2);
            add(r.bytes / step + (r.bytes % step != 0), 8192);
        }
        return bytes;
    }

    void encode_loop() {
        try {
#ifdef STRIX_HANDOFF_TESTING
            if (opt.before_encode) opt.before_encode();
#endif
            for (;;) {
                // bp-v1 with encode_batch > 1: every queued piece (up to encode_batch) is encoded side by side on the
                // pool, so pieces of one codec block (<= 1 MiB) no longer encode on one thread; output order and the
                // bytes of every frame are the same as one at a time.
                std::vector<Input> batch;
                std::vector<std::size_t> reservations;
                {
                    std::unique_lock<std::mutex> g(m);
                    cv.wait(g, [&] { return aborted || !inputs.empty() || closed; });
                    if (aborted) return;  // `batch` and the queues release their callbacks on destruction
                    if (inputs.empty()) return;
                    const bool wide = codec == Codec::bp_v1 && pool && pool->threads() > 0 && opt.encode_batch > 1;
                    const std::size_t take = wide ? std::min<std::size_t>(inputs.size(), opt.encode_batch) : 1;
                    for (std::size_t i = 0; i < take; ++i) {
                        const auto reserve = encode_reservation(inputs.front());
                        const auto fits = [&] { return reserve == 0 || (queued_wire == 0 && batch.empty()) ||
                            (queued_wire <= opt.max_queued_wire_bytes && reserve <= opt.max_queued_wire_bytes - queued_wire); };
                        if (!batch.empty() && !fits()) break;
                        cv.wait(g, [&] { return aborted || fits(); });
                        if (aborted) return;
                        reservations.push_back(reserve);
                        batch.push_back(std::move(inputs.front()));
                        inputs.pop_front();
                        queued_wire += reserve;
                    }
#ifdef STRIX_HANDOFF_TESTING
                    if (opt.observe_batch) opt.observe_batch(batch.size(), queued_wire);
#endif
                }
                const auto t0 = Clock::now();
                std::vector<Output> outs(batch.size());
                if (batch.size() == 1) prepare(batch[0], outs[0]);
                else pool->run(batch.size(), [&](std::size_t i) { prepare(batch[i], outs[i]); });
                const double dt = secs(t0, Clock::now());
                for (std::size_t i = 0; i < outs.size(); ++i) {
                    Output& out = outs[i];
#ifdef STRIX_HANDOFF_TESTING
                    if (opt.observe_frame) opt.observe_frame(out.f.seq,
                        {reinterpret_cast<const std::uint8_t*>(&out.f), sizeof(out.f)}, out.wire);
#endif
                    std::unique_lock<std::mutex> g(m);
                    if (i == 0) encode_s += dt;
                    if (aborted) {
                        g.unlock();
                        return;
                    }
                    if (out.wire.capacity() > reservations[i]) throw std::logic_error("handoff: encoding exceeded reservation");
                    queued_wire -= reservations[i] - out.wire.capacity();
                    plain_total += out.f.plain_bytes;
                    wire_total += out.f.wire_bytes;
                    ++pieces;
                    fold.update(&out.f.hash, 8);
                    outputs[out.f.seq % socks.size()].push_back(std::move(out));
                    cv.notify_all();
                }
            }
        } catch (...) {
            fail(std::current_exception());
        }
    }

    void write_loop(std::size_t k) {
        try {
            for (;;) {
                Output out;
                {
                    std::unique_lock<std::mutex> g(m);
                    cv.wait(g, [&] { return aborted || !outputs[k].empty() || encoder_finished; });
                    if (aborted) return;
                    if (outputs[k].empty()) return;
                    out = std::move(outputs[k].front());
                    outputs[k].pop_front();
                }
                send_all(socks[k].fd, &out.f, sizeof(out.f));
                {
                    std::lock_guard<std::mutex> g(m);
                    if (first_byte < 0) first_byte = secs(t_open, Clock::now());
                }
                if (!out.wire.empty()) {
                    send_all(socks[k].fd, out.wire.data(), out.wire.size());
                } else {
                    for (const auto& r : out.regions)
                        if (r.bytes) send_all(socks[k].fd, r.data, r.bytes);
                }
                out.release.run();
                std::lock_guard<std::mutex> g(m);
                last_byte = secs(t_open, Clock::now());
                const auto capacity = out.wire.capacity();
                std::vector<std::uint8_t>().swap(out.wire);  // release storage before returning its credit
                queued_wire -= capacity;
                cv.notify_all();
            }
        } catch (...) {
            fail(std::current_exception());
        }
    }

    void join_all() {
        if (encoder.joinable()) encoder.join();
        for (auto& w : writers)
            if (w.joinable()) w.join();
    }

    // After the threads stopped: the caller's buffers of pieces that never left are released (exactly once).
    void drop_queues() {
        std::deque<Input> in;
        std::deque<std::deque<Output>> out;
        {
            std::lock_guard<std::mutex> g(m);
            in.swap(inputs);
            out.swap(outputs);
        }
        // destroyed here, outside the lock: their Release objects run the callbacks
    }
};

Sender::Sender(const Endpoint& to, const Hello& hello, SenderOptions options) : impl_(std::make_unique<Impl>()) {
    auto& I = *impl_;
    I.opt = options;
    check_hello(hello);
    if (options.streams == 0 || options.streams > kMaxStreams) throw std::invalid_argument("handoff: 1..64 streams");
    const unsigned n = options.streams;
    I.session = hello.session;
    I.fold = Xxh64(hello_seed(hello));
    const std::uint32_t offer = options.codec == Codec::bp_v1 ? 3U : 1U;
    for (unsigned k = 0; k < n; ++k) {
        I.socks.push_back(connect_to(to, options.socket_buffer, options.connect_timeout));
        set_timeout(I.socks.back().fd, std::chrono::duration_cast<std::chrono::milliseconds>(options.reply_timeout), false);
        set_timeout(I.socks.back().fd, std::chrono::duration_cast<std::chrono::milliseconds>(options.send_timeout), true);
        const auto payload = hello_payload(hello, k, n, offer);
        Frame f;
        f.type = kHello;
        f.wire_bytes = f.plain_bytes = payload.size();
        send_frame(I.socks.back().fd, f, payload.data());
    }
    std::uint8_t accepted = 0;
    for (unsigned k = 0; k < n; ++k) {
        const Frame a = recv_header(I.socks[k].fd, 0);
        if (a.type == kAbort) throw std::runtime_error(recv_abort_text(I.socks[k].fd, a));
        if (a.type != kAccept || a.wire_bytes != 0) throw std::runtime_error("handoff: expected ACCEPT");
        if (a.codec > 1 || ((offer >> a.codec) & 1U) == 0) throw std::runtime_error("handoff: the receiver accepted a codec that was not offered");
        if (k == 0) accepted = a.codec;
        else if (a.codec != accepted) throw std::runtime_error("handoff: streams accepted different codecs");
    }
    I.codec = accepted == static_cast<std::uint8_t>(Codec::bp_v1) ? Codec::bp_v1 : Codec::identity;
    if (I.codec == Codec::bp_v1) I.pool = std::make_unique<bp_codec::WorkerPool>(options.encode_threads);
    I.t_open = Clock::now();
    I.outputs.resize(n);
    try {
        I.encoder = std::thread([&I] { I.encode_loop(); });
        for (unsigned k = 0; k < n; ++k) I.writers.emplace_back([&I, k] { I.write_loop(k); });
    } catch (...) {
        I.fail(std::current_exception());
        I.join_all();
        throw;
    }
}

Sender::~Sender() {
    if (impl_ && !impl_->finished && !impl_->finishing) abort("sender destroyed before finish()");
}

void Sender::submit(const PieceHeader& header, std::span<const bp_codec::Region> regions, std::function<void()> release,
                    bool identity) {
    auto& I = *impl_;
    std::size_t bytes = 0;
    for (const auto& r : regions) bytes += r.bytes;
    // Release ownership first, so the callback runs exactly once even if this call throws.
    Release rel([&I, bytes, fn = std::move(release)]() mutable {
        if (fn) fn();
        std::lock_guard<std::mutex> g(I.pm);
        I.pending -= bytes;
        I.pcv.notify_all();
    });
    {
        std::unique_lock<std::mutex> g(I.pm);
        I.pending += bytes;
        if (I.opt.max_pending_plain_bytes)
            I.pcv.wait(g, [&] {
                return I.pending == bytes || I.pending <= I.opt.max_pending_plain_bytes || I.dead.load();
            });
    }
    std::lock_guard<std::mutex> g(I.m);
    if (I.error) std::rethrow_exception(I.error);
    if (I.closed || I.finished) throw std::logic_error("handoff: submit after finish()");
    if (I.next_seq == 0xffffffffU) throw std::length_error("handoff: too many pieces in one session");
    Impl::Input in;
    in.f.type = kPiece;
    in.f.kind = header.kind;
    in.f.index = header.index;
    in.f.flags = header.flags;
    in.f.row0 = header.row0;
    in.f.rows = header.rows;
    in.f.seq = I.next_seq++;
    in.regions.assign(regions.begin(), regions.end());
    in.release = std::move(rel);
    in.identity = identity;
    I.inputs.push_back(std::move(in));
    I.cv.notify_all();
}

void Sender::submit(const PieceHeader& header, std::vector<std::uint8_t> bytes, std::uint32_t width) {
    auto owned = std::make_shared<std::vector<std::uint8_t>>(std::move(bytes));
    const bp_codec::Region r{owned->data(), owned->size(), width};
    submit(header, std::span<const bp_codec::Region>(&r, 1), [owned]() mutable { owned.reset(); });
}

Reply Sender::finish() {
    auto& I = *impl_;
    {
        std::lock_guard<std::mutex> g(I.m);
        if (I.finishing && I.error) std::rethrow_exception(I.error);  // aborted before finish()
        if (I.finished || I.finishing) throw std::logic_error("handoff: finish() twice");
        I.finishing = true;
        I.closed = true;
        I.cv.notify_all();
    }
    if (I.encoder.joinable()) I.encoder.join();
    {
        std::lock_guard<std::mutex> g(I.m);
        I.encoder_finished = true;
        I.cv.notify_all();
    }
    for (auto& w : I.writers)
        if (w.joinable()) w.join();
    I.finished = true;
    std::exception_ptr failed;
    {
        std::lock_guard<std::mutex> g(I.m);
        failed = I.error;
    }
    if (failed) {
        I.shutdown_all();
        I.drop_queues();
        std::rethrow_exception(failed);
    }
    try {
        const Manifest man{I.pieces, I.plain_total, I.wire_total, I.fold.digest()};
        for (auto& s : I.socks) {
            Frame f;
            f.type = kEnd;
            f.wire_bytes = f.plain_bytes = sizeof(man);
            send_frame(s.fd, f, &man);
        }
        const Frame r = recv_header(I.socks[0].fd, 0);
        if (r.type == kAbort) throw std::runtime_error(recv_abort_text(I.socks[0].fd, r));
        if (r.type != kReply) throw std::runtime_error("handoff: expected REPLY");
        Reply out;
        out.status = r.kind;
        out.payload.resize(r.wire_bytes);
        if (r.wire_bytes) recv_all(I.socks[0].fd, out.payload.data(), out.payload.size());
        return out;
    } catch (...) {
        std::lock_guard<std::mutex> g(I.m);
        if (I.error) std::rethrow_exception(I.error);  // an abort() from another thread cut the sockets: say why
        throw;
    }
}

void Sender::abort(std::string_view why) noexcept {
    auto& I = *impl_;
    // A writer may be inside a frame: cutting the sockets makes the receiver fail on a short frame instead of ever
    // accepting a partial piece. It also wakes a finish() waiting for the reply on another thread.
    I.fail(std::make_exception_ptr(std::runtime_error("handoff: sender aborted: " + std::string(why))));
    if (I.finishing.exchange(true)) return;  // finish() is running elsewhere: it joins, releases and rethrows
    I.join_all();
    I.drop_queues();
    I.finished = true;
}

void Sender::on_failure(std::function<void(std::exception_ptr)> notify) {
    auto& I = *impl_;
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> g(I.m);
        I.failure_notify = notify;
        error = I.error;
    }
    if (error && notify) notify(error);
}

SenderStats Sender::stats() const {
    auto& I = *impl_;
    std::lock_guard<std::mutex> g(I.m);
    return {I.pieces, I.plain_total, I.wire_total, I.encode_s, I.first_byte, I.last_byte, I.codec};
}

Codec Sender::codec() const noexcept { return impl_->codec; }

// ---- receiver --------------------------------------------------------------------------------------------------
struct Session::Impl {
    ReceiverOptions opt;
    Hello hello;
    Codec codec = Codec::identity;
    std::vector<Socket> socks;
    std::unique_ptr<bp_codec::WorkerPool> pool;
    Clock::time_point t_accept = Clock::now();

    struct Got {
        Frame f;
        std::vector<std::uint8_t> wire;
    };
    std::mutex m;
    std::condition_variable cv;
    std::map<std::uint32_t, Got> ready;  // by seq
    std::size_t buffered = 0;
    std::uint64_t next_seq = 0;
    std::vector<std::uint64_t> expect;  // per stream: the next seq it must carry (k, k + N, ...)
    std::vector<bool> stream_ended;
    unsigned ended_streams = 0;
    bool have_manifest = false;
    Manifest manifest;
    std::exception_ptr error;
    std::exception_ptr reported;  // the first error next() threw: every later call throws it again
    bool stopping = false;
    std::vector<std::thread> readers;
    std::uint64_t plain_total = 0, wire_total = 0, pieces = 0;
    Xxh64 fold;
    double decode_s = 0, wait_s = 0, first_piece = -1, end = -1;
    bool complete = false, replied = false;

    Impl() : fold(0) {}

    void cut() noexcept {
        for (auto& s : socks)
            if (s.fd != kBad) shutdown_sock(s.fd);
    }

    void fail(std::exception_ptr e) {
        {
            std::lock_guard<std::mutex> g(m);
            if (!error) error = e;
            cv.notify_all();
        }
        cut();
    }

    void read_loop(std::size_t k) {
        try {
            const sock_t s = socks[k].fd;
            const std::uint64_t n = socks.size();
            for (;;) {
                const Frame f = recv_header(s, opt.max_piece_bytes);
                if (f.type == kAbort) throw std::runtime_error(recv_abort_text(s, f));
                if (f.type == kEnd) {
                    if (f.wire_bytes != sizeof(Manifest)) throw std::runtime_error("handoff: bad END");
                    Manifest man;
                    recv_all(s, &man, sizeof(man));
                    std::lock_guard<std::mutex> g(m);
                    if (have_manifest && !(man == manifest)) throw std::runtime_error("handoff: streams disagree on the manifest");
                    // stream k carries seqs k, k + N, ...: it must have sent exactly those below the piece count
                    const std::uint64_t want = man.pieces > k ? k + (man.pieces - k + n - 1) / n * n : k;
                    if (expect[k] != want) throw std::runtime_error("handoff: a stream ended with missing or extra pieces");
                    manifest = man;
                    have_manifest = true;
                    stream_ended[k] = true;
                    ++ended_streams;
                    cv.notify_all();
                    return;
                }
                if (f.type != kPiece) throw std::runtime_error("handoff: unexpected frame type in a session");
                const bool identity = f.codec == static_cast<std::uint8_t>(Codec::identity);
                if (identity ? f.wire_bytes != f.plain_bytes : f.codec != static_cast<std::uint8_t>(codec))
                    throw std::runtime_error("handoff: piece codec / size inconsistent with the session");
                {
                    std::unique_lock<std::mutex> g(m);
                    if (f.seq != expect[k]) throw std::runtime_error("handoff: piece seq out of order or duplicated on its stream");
                    expect[k] += n;
                    // Admit the piece the consumer needs next unconditionally (no deadlock across streams).
                    cv.wait(g, [&] {
                        return stopping || error || f.seq == next_seq || buffered == 0 || buffered + f.wire_bytes <= opt.max_buffered_wire_bytes;
                    });
                    if (stopping || error) return;
                    buffered += f.wire_bytes;
                }
                Got got{f, std::vector<std::uint8_t>(f.wire_bytes)};
                if (f.wire_bytes) recv_all(s, got.wire.data(), got.wire.size());
                std::lock_guard<std::mutex> g(m);
                ready.emplace(f.seq, std::move(got));
                cv.notify_all();
            }
        } catch (...) {
            fail(std::current_exception());
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> g(m);
            stopping = true;
            cv.notify_all();
        }
        cut();
        for (auto& t : readers)
            if (t.joinable()) t.join();
    }
};

Session::Session(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Session::~Session() {
    if (impl_) impl_->stop();
}

const Hello& Session::hello() const noexcept { return impl_->hello; }
Codec Session::codec() const noexcept { return impl_->codec; }

bool Session::next(Piece& piece) {
    auto& I = *impl_;
    {
        std::lock_guard<std::mutex> g(I.m);
        if (I.reported) std::rethrow_exception(I.reported);
    }
    try {
        return next_checked(piece);
    } catch (...) {
        std::lock_guard<std::mutex> g(I.m);
        if (!I.reported) I.reported = std::current_exception();
        if (!I.error) I.error = I.reported;
        std::rethrow_exception(I.reported);
    }
}

bool Session::next_checked(Piece& piece) {
    auto& I = *impl_;
    Impl::Got got;
    std::exception_ptr later;  // a failure that arrived after this piece: reported after checking the piece itself
    {
        const auto w0 = Clock::now();
        std::unique_lock<std::mutex> g(I.m);
        if (I.error && !I.ready.count(static_cast<std::uint32_t>(I.next_seq))) std::rethrow_exception(I.error);  // also after completion
        if (I.complete) return false;
        I.cv.wait(g, [&] { return I.error || I.ready.count(static_cast<std::uint32_t>(I.next_seq)) || I.ended_streams == I.socks.size(); });
        I.wait_s += secs(w0, Clock::now());
        const auto it = I.next_seq <= 0xffffffffULL ? I.ready.find(static_cast<std::uint32_t>(I.next_seq)) : I.ready.end();
        if (it == I.ready.end()) {
            if (I.error) std::rethrow_exception(I.error);
            // every stream ended: the session is complete only if nothing is missing and the manifest agrees
            const Manifest& man = I.manifest;
            if (I.next_seq != man.pieces || !I.ready.empty() || man.plain != I.plain_total || man.wire != I.wire_total || man.fold != I.fold.digest()) {
                I.error = std::make_exception_ptr(std::runtime_error("handoff: manifest mismatch (pieces, bytes or checksums differ from what arrived)"));
                g.unlock();
                I.cut();
                std::rethrow_exception(I.error);
            }
            I.complete = true;
            I.end = secs(I.t_accept, Clock::now());
            return false;
        }
        got = std::move(it->second);
        I.ready.erase(it);
        I.buffered -= got.f.wire_bytes;
        ++I.next_seq;
        later = I.error;
        I.cv.notify_all();
    }
    const Frame& f = got.f;
    const auto d0 = Clock::now();
    try {
        if (f.codec == static_cast<std::uint8_t>(Codec::identity)) {
            piece.plain.swap(got.wire);
        } else {
            if (bp_codec::decoded_size(got.wire) != f.plain_bytes) throw std::runtime_error("handoff: bp-v1 size differs from the frame");
            piece.plain.resize(f.plain_bytes);
            bp_codec::decode(got.wire, piece.plain, I.pool.get());
        }
        Xxh64 h;
        hash_descriptor(h, f);
        h.update(piece.plain.data(), piece.plain.size());
        if (h.digest() != f.hash) throw std::runtime_error("handoff: piece checksum mismatch (seq " + std::to_string(f.seq) + ")");
    } catch (...) {
        // This piece is the first fault in seq order: report it, latched, even if a later failure latched first.
        const std::exception_ptr own = std::current_exception();
        {
            std::lock_guard<std::mutex> g(I.m);
            I.error = own;
            I.cv.notify_all();
        }
        I.cut();
        throw;
    }
    if (later) std::rethrow_exception(later);
    piece.header = {f.kind, f.index, f.flags, f.row0, f.rows};
    piece.seq = f.seq;
    piece.hash = f.hash;
    std::lock_guard<std::mutex> g(I.m);
    I.decode_s += secs(d0, Clock::now());
    if (I.first_piece < 0) I.first_piece = secs(I.t_accept, Clock::now());
    I.fold.update(&f.hash, 8);
    I.plain_total += f.plain_bytes;
    I.wire_total += f.wire_bytes;
    ++I.pieces;
    return true;
}

void Session::reply(std::uint32_t status, std::span<const std::uint8_t> payload) {
    auto& I = *impl_;
    {
        std::lock_guard<std::mutex> g(I.m);
        if (I.error) std::rethrow_exception(I.error);
        if (!I.complete) throw std::logic_error("handoff: reply() before the session completed");
        if (I.replied) throw std::logic_error("handoff: reply() twice");
        I.replied = true;
    }
    Frame f;
    f.type = kReply;
    f.kind = status;
    f.wire_bytes = f.plain_bytes = payload.size();
    send_frame(I.socks[0].fd, f, payload.data());
}

void Session::abort(std::string_view why) noexcept {
    auto& I = *impl_;
    {
        std::lock_guard<std::mutex> g(I.m);
        if (!I.error) I.error = std::make_exception_ptr(std::runtime_error("handoff: receiver aborted: " + std::string(why)));
        I.cv.notify_all();
    }
    try {
        send_abort(I.socks[0].fd, why);
    } catch (...) {
    }
    I.stop();
}

void Session::cancel(std::string_view why) noexcept {
    auto& I = *impl_;
    {
        std::lock_guard<std::mutex> g(I.m);
        if (!I.error) I.error = std::make_exception_ptr(std::runtime_error("handoff: receiver cancelled: " + std::string(why)));
        I.cv.notify_all();
    }
    I.cut();
}

ReceiverStats Session::stats() const {
    auto& I = *impl_;
    std::lock_guard<std::mutex> g(I.m);
    return {I.pieces, I.plain_total, I.wire_total, I.decode_s, I.wait_s, I.first_piece, I.end};
}

Listener::Listener(std::uint16_t port, const std::string& bind) : fd_(static_cast<std::intptr_t>(kBad)), port_(port) {
    net_init();
    Socket holder(::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
    const sock_t s = holder.fd;
    if (s == kBad) throw std::runtime_error("handoff: socket()");
    int one = 1;
    ::setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one), sizeof(one));
    sockaddr_in a{};
    if (bind.empty()) {
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_ANY);
        a.sin_port = htons(port);
    } else {
        a = resolve(bind, port);
    }
    if (::bind(s, reinterpret_cast<const sockaddr*>(&a), sizeof(a)) != 0 || ::listen(s, 16) != 0)
        throw std::runtime_error("handoff: cannot listen on port " + std::to_string(port));
    sockaddr_in got{};
#ifdef _WIN32
    int len = sizeof(got);
#else
    socklen_t len = sizeof(got);
#endif
    ::getsockname(s, reinterpret_cast<sockaddr*>(&got), &len);
    port_ = ntohs(got.sin_port);
    fd_ = static_cast<std::intptr_t>(s);
    holder.fd = kBad;  // owned by the listener now
}

Listener::~Listener() {
    const auto fd = static_cast<sock_t>(fd_.exchange(static_cast<std::intptr_t>(kBad)));
    if (fd != kBad) close_sock(fd);
}

// A stream accepted by Listener::accept() and still in its HELLO handshake: registered so shutdown() can wake it.
// Declared after the Socket that owns the descriptor, so it unregisters before that socket is closed.
struct PendingTrack {
    Listener* L = nullptr;
    std::intptr_t fd = static_cast<std::intptr_t>(kBad);
    PendingTrack(Listener* l, sock_t s) : L(l), fd(static_cast<std::intptr_t>(s)) {
        std::lock_guard<std::mutex> g(L->pending_m_);
        if (L->down_) throw std::runtime_error("handoff: listener shut down");
        L->pending_.push_back(fd);
    }
    PendingTrack(PendingTrack&& o) noexcept : L(o.L), fd(o.fd) { o.L = nullptr; }
    PendingTrack(const PendingTrack&) = delete;
    PendingTrack& operator=(const PendingTrack&) = delete;
    PendingTrack& operator=(PendingTrack&& o) noexcept {  // std::sort moves the pending streams
        if (this != &o) {
            release();
            L = o.L;
            fd = o.fd;
            o.L = nullptr;
        }
        return *this;
    }
    ~PendingTrack() { release(); }
    void release() noexcept {
        if (!L) return;
        std::lock_guard<std::mutex> g(L->pending_m_);
        for (std::size_t k = 0; k < L->pending_.size(); ++k)
            if (L->pending_[k] == fd) {
                L->pending_.erase(L->pending_.begin() + static_cast<std::ptrdiff_t>(k));
                break;
            }
        L = nullptr;
    }
};

void Listener::shutdown() noexcept {
    {
        std::lock_guard<std::mutex> g(pending_m_);
        down_ = true;
        for (const auto fd : pending_) shutdown_sock(static_cast<sock_t>(fd));
    }
#ifdef _WIN32
    // Winsock: closing the socket is what wakes a blocked accept().
    const auto fd = static_cast<sock_t>(fd_.exchange(static_cast<std::intptr_t>(kBad)));
    if (fd != kBad) close_sock(fd);
#else
    // Linux: shutdown() wakes accept() with EINVAL; the descriptor stays valid until the destructor (no reuse race).
    const auto fd = static_cast<sock_t>(fd_.load());
    if (fd != kBad) ::shutdown(fd, SHUT_RDWR);
#endif
}

std::uint16_t Listener::port() const noexcept { return port_; }

std::unique_ptr<Session> Listener::accept(ReceiverOptions options) {
    auto I = std::make_unique<Session::Impl>();
    I->opt = options;
    const sock_t lfd = static_cast<sock_t>(fd_.load());
    if (lfd == kBad) throw std::runtime_error("handoff: listener shut down");
    struct Pending {
        Socket s;
        PendingTrack t;  // after s: unregisters before s closes
        std::uint32_t stream, codecs;
    };
    std::vector<Pending> got;
    std::uint32_t streams = 1;
    while (got.size() < streams) {
        if (!got.empty()) {  // the other streams of a session must follow within the idle timeout
            fd_set rd;
            FD_ZERO(&rd);
            FD_SET(lfd, &rd);
            timeval tv{};
            tv.tv_sec = static_cast<long>(options.idle_timeout.count() / 1000);
            tv.tv_usec = static_cast<long>((options.idle_timeout.count() % 1000) * 1000);
            if (::select(static_cast<int>(lfd + 1), &rd, nullptr, nullptr, &tv) <= 0) throw std::runtime_error("handoff: missing streams of a session");
        }
        Socket s(::accept(lfd, nullptr, nullptr));
        if (s.fd == kBad) throw std::runtime_error("handoff: accept failed");
        PendingTrack t(this, s.fd);  // throws if shut down meanwhile
        set_opts(s.fd, options.socket_buffer, false);
        set_timeout(s.fd, options.idle_timeout, false);
        set_timeout(s.fd, options.idle_timeout, true);
        try {
            const Frame f = recv_header(s.fd, 0);
            if (f.type != kHello) throw std::runtime_error("handoff: expected HELLO");
            std::vector<std::uint8_t> p(f.wire_bytes);
            if (!p.empty()) recv_all(s.fd, p.data(), p.size());
            Cursor c{p.data(), p.size()};
            Hello h;
            h.session = c.get<std::uint64_t>();
            h.rank = c.get<std::uint32_t>();
            h.world = c.get<std::uint32_t>();
            const auto stream = c.get<std::uint32_t>(), n = c.get<std::uint32_t>(), codecs = c.get<std::uint32_t>();
            const auto model_bytes = c.get<std::uint32_t>(), meta_bytes = c.get<std::uint32_t>();
            if (model_bytes > kMaxModel || meta_bytes > kMaxMeta) throw std::runtime_error("handoff: HELLO model / meta too large");
            const auto model = c.bytes(model_bytes);
            h.model.assign(model.begin(), model.end());
            h.meta = c.bytes(meta_bytes);
            if (c.n != 0) throw std::runtime_error("handoff: trailing bytes in HELLO");
            if (n == 0 || n > kMaxStreams || stream >= n) throw std::runtime_error("handoff: bad stream index in HELLO");
            if (h.world == 0 || h.world > kMaxStreams || h.rank >= h.world) throw std::runtime_error("handoff: bad rank / world in HELLO");
            if (got.empty()) {
                I->hello = std::move(h);
                streams = n;
            } else if (h.session != I->hello.session || n != streams || h.model != I->hello.model || h.rank != I->hello.rank ||
                       h.world != I->hello.world || h.meta != I->hello.meta) {
                try {
                    send_abort(s.fd, "busy: another session is being received");
                } catch (...) {
                }
                continue;
            }
            for (const auto& g : got)
                if (g.stream == stream) throw std::runtime_error("handoff: duplicate stream index");
            got.push_back({std::move(s), std::move(t), stream, codecs});
        } catch (const std::runtime_error&) {
            if (got.empty()) continue;  // a stray or broken first connection: wait for the next session
            throw;
        }
    }
    std::sort(got.begin(), got.end(), [](const Pending& a, const Pending& b) { return a.stream < b.stream; });
    std::uint32_t mask = 3U;  // identity is always accepted
    for (const auto& g : got) mask &= g.codecs | 1U;
    I->codec = (options.accept_bp_v1 && (mask & 2U)) ? Codec::bp_v1 : Codec::identity;
    if (I->codec == Codec::bp_v1) I->pool = std::make_unique<bp_codec::WorkerPool>(options.decode_threads);
    I->fold = Xxh64(hello_seed(I->hello));
    for (auto& g : got) {
        Frame a;
        a.type = kAccept;
        a.codec = static_cast<std::uint8_t>(I->codec);
        send_frame(g.s.fd, a);
        I->socks.push_back(std::move(g.s));
    }
    I->expect.resize(I->socks.size());
    for (std::size_t k = 0; k < I->expect.size(); ++k) I->expect[k] = k;
    I->stream_ended.assign(I->socks.size(), false);
    I->t_accept = Clock::now();
    Session::Impl* raw = I.get();
    std::unique_ptr<Session> out(new Session(std::move(I)));  // owner first: its destructor stops started readers
    for (std::size_t k = 0; k < raw->socks.size(); ++k) raw->readers.emplace_back([raw, k] { raw->read_loop(k); });
    return out;
}

}  // namespace strix::handoff
