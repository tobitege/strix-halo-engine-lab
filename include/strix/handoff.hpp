#pragma once
// strix::handoff: persistent framed TCP streams for the prompt-state hand-off (prefill host -> serving ranks).
//
// One session = one request's state for one destination rank, carried by 1..64 TCP connections ("streams") to the
// same endpoint. Piece seq s travels on stream s % N; the receiver merges the streams back into one seq-ordered
// sequence. A piece is a model-defined unit (a layer block, a row sub-block, a prefill chunk, a fixed component); the
// library never looks inside it.
//
// Frame (little-endian, 64-byte header + payload):
//   u32 magic "SHO1" | u16 version (2) | u8 type | u8 codec | u32 flags | u32 kind | u32 index | u32 seq
//   | u64 row0 | u64 rows | u64 plain_bytes | u64 wire_bytes | u64 hash
// Types: HELLO (sender -> receiver, once per stream; payload = Hello), ACCEPT (receiver -> sender, codec = the
// accepted codec, one of those offered), PIECE (payload = the piece, identity or one bp-v1 frame), END (on every
// stream after its last piece; payload = manifest {u64 pieces, u64 plain bytes, u64 wire bytes, u64 fold}), REPLY
// (receiver -> sender on stream 0 after a complete session: kind = status, payload = model-defined bytes), ABORT
// (either way, payload = a message).
//
// Integrity (threat model: transport and codec defects, not adversaries; detection is probabilistic, ~2^-64 per piece):
//   piece hash = XXH64 (seed 0) of the 40-byte descriptor {u32 seq, kind, index, flags; u64 row0, rows, plain_bytes}
//   followed by the plain bytes, so a corrupted destination (layer, rows) fails like corrupted data. It is checked
//   on every piece after decoding (bp-v1 also checks each 1 MiB block inside the codec). fold = XXH64 (seed = XXH64 of
//   the session's HELLO {session, rank, world, model, meta}) of the piece hashes in seq order, so the request identity
//   is checked end to end too. A session completes only when every stream sent END with the same
//   manifest, every seq < pieces arrived exactly once on its own stream, and the totals and fold match. The first
//   failure latches: the session never returns another piece and cannot reply.
//
// Relay: a piece flagged kRelay is replicated state; the receiving rank (normally rank 0) forwards its plain bytes
// to its peer rank(s) on the engine's own interconnect. The library only carries the flag; the model adapter relays.
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <exception>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "strix/bp_codec.hpp"

namespace strix::handoff {

enum class Codec : std::uint8_t { identity = 0, bp_v1 = 1 };
inline constexpr std::uint32_t kRelay = 1U;  // PieceHeader::flags: replicated piece, the receiver forwards it
inline constexpr unsigned kMaxStreams = 64;

struct PieceHeader {
    std::uint32_t kind = 0;   // model-defined piece type
    std::uint32_t index = 0;  // model-defined id (layer, chunk, component)
    std::uint32_t flags = 0;  // kRelay | model-defined bits 8..31
    std::uint64_t row0 = 0, rows = 0;
};

struct Hello {
    std::uint64_t session = 0;  // request id (lineage), identical on every stream of the session
    std::string model;          // state format tag, e.g. "minimax-m3.1/kv-v1" (<= 256 bytes)
    std::uint32_t rank = 0, world = 1;  // rank < world <= 64
    std::vector<std::uint8_t> meta;     // model-defined (prompt length, token digest, ...), <= 64 KiB
};

// rank -> where its listener is, and which local address to send from (a second NIC is configuration only).
struct Endpoint {
    std::string host;
    std::uint16_t port = 0;
    std::string bind;  // local source address, "" = any
};
// "HOST:PORT[@BIND][,HOST:PORT[@BIND]...]": entry i is rank i.
[[nodiscard]] std::vector<Endpoint> parse_path_map(std::string_view text);

struct SenderOptions {
    Codec codec = Codec::bp_v1;   // falls back to identity when the receiver does not accept it
    unsigned streams = 1;          // TCP connections to the same endpoint (1..64); piece seq s goes on stream s % N
    unsigned encode_threads = 8;   // bp-v1 worker pool
    // bp-v1: queued pieces encoded side by side on the pool (1 = one at a time), within the byte budget.
    // The encoded frame bytes and submission order are unchanged.
    unsigned encode_batch = 16;
    // Reservations include encoding workspace and unsent buffers. One oversized piece may run alone for progress.
    std::size_t max_queued_wire_bytes = std::size_t{512} << 20;
    std::size_t max_pending_plain_bytes = 0;  // submitted-but-unreleased caller bytes before submit() waits; 0 = no bound
    int socket_buffer = 32 << 20;
    std::chrono::seconds connect_timeout{900};  // the ranks may still be loading
    std::chrono::seconds reply_timeout{600};    // ACCEPT (a receiver may finish another session first) and REPLY
    std::chrono::seconds send_timeout{60};      // a send() that makes no progress this long fails the session
    bp_codec::EncodeOptions encode{};
#ifdef STRIX_HANDOFF_TESTING
    // Host regression hooks, absent from production builds. Called on the encoder thread.
    std::function<void()> before_encode;
    std::function<void(std::size_t, std::size_t)> observe_batch;  // pieces, total reserved bytes
    std::function<void(std::uint32_t, std::span<const std::uint8_t>, std::span<const std::uint8_t>)> observe_frame;
#endif
};

struct SenderStats {
    std::uint64_t pieces = 0, plain_bytes = 0, wire_bytes = 0;
    double encode_seconds = 0;   // encoder thread busy time
    double first_byte = -1, last_byte = -1;  // seconds after open, first / last payload byte handed to the socket
    Codec codec = Codec::identity;
};

struct Reply {
    std::uint32_t status = 0;
    std::vector<std::uint8_t> payload;
};

// Sends one session. submit() returns at once (unless max_pending_plain_bytes is reached): a background encoder
// (bp-v1 on a worker pool) prepares piece i + 1 while piece i is on the wire, and pieces leave in submit order. The
// plain bytes of a piece must stay valid until its release callback runs. Every submitted callback runs exactly once,
// on a library thread or in finish()/abort()/the destructor: after encoding for bp-v1, after sending for identity, or
// when the session fails. Callbacks must not throw (an exception is swallowed) and must not call into this Sender.
class Sender {
public:
    Sender(const Endpoint& to, const Hello& hello, SenderOptions options = {});
    ~Sender();  // aborts an unfinished session
    Sender(const Sender&) = delete;
    Sender& operator=(const Sender&) = delete;

    // `width` hints the element size of the bytes (2 for BF16, 4 for FP32) to the codec. `identity` sends this piece
    // uncoded even in a bp-v1 session (every receiver accepts an identity frame; a latency-critical piece skips the codec).
    void submit(const PieceHeader& header, std::span<const bp_codec::Region> regions, std::function<void()> release = {},
                bool identity = false);
    void submit(const PieceHeader& header, std::vector<std::uint8_t> bytes, std::uint32_t width = 1);
    // Every piece on the wire, END + manifest sent on every stream; waits for and returns the receiver's reply.
    // Rethrows the first transport or encoder error.
    Reply finish();
    // Safe from any thread, also while finish() waits on another one (which then throws).
    void abort(std::string_view why) noexcept;
    // One failure observer; also called immediately for an already failed Sender. Runs outside Sender locks,
    // must not throw or call this Sender. Captures must outlive the Sender (or use weak ownership).
    void on_failure(std::function<void(std::exception_ptr)> notify);
    [[nodiscard]] SenderStats stats() const;
    [[nodiscard]] Codec codec() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct ReceiverOptions {
    unsigned decode_threads = 4;
    std::size_t max_buffered_wire_bytes = std::size_t{1} << 30;  // received-but-unconsumed bytes before reading waits
    std::uint64_t max_piece_bytes = std::uint64_t{4} << 30;      // larger plain or wire pieces are rejected
    int socket_buffer = 32 << 20;
    std::chrono::milliseconds idle_timeout{120000};  // no byte received (or no send progress) this long: the session fails
    bool accept_bp_v1 = true;
};

struct ReceiverStats {
    std::uint64_t pieces = 0, plain_bytes = 0, wire_bytes = 0;
    double decode_seconds = 0, wait_seconds = 0;  // in next(): decoding + checking / waiting for the network
    double first_piece = -1, end = -1;            // seconds after accept
};

struct Piece {
    PieceHeader header;
    std::uint32_t seq = 0;            // position in the session (0, 1, ...)
    std::uint64_t hash = 0;           // the verified piece hash (descriptor + plain bytes)
    std::vector<std::uint8_t> plain;  // reused across next() calls
};

class Session;

class Listener {
public:
    explicit Listener(std::uint16_t port, const std::string& bind = "");
    ~Listener();
    Listener(const Listener&) = delete;
    Listener& operator=(const Listener&) = delete;
    // Accepts the streams of the next session (all of them carry the same Hello) and answers their HELLO.
    [[nodiscard]] std::unique_ptr<Session> accept(ReceiverOptions options = {});
    [[nodiscard]] std::uint16_t port() const noexcept;
    // Stops listening: an accept() blocked on another thread (or any later one) throws, including one waiting in the
    // HELLO handshake of an already connected stream. Sessions already returned by accept() are not affected (cancel
    // them with Session::cancel).
    void shutdown() noexcept;

private:
    friend struct PendingTrack;
    std::atomic<std::intptr_t> fd_;
    std::uint16_t port_;
    std::mutex pending_m_;
    std::vector<std::intptr_t> pending_;  // streams accept() holds in their HELLO handshake (woken by shutdown())
    bool down_ = false;
};

class Session {
public:
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    [[nodiscard]] const Hello& hello() const noexcept;
    [[nodiscard]] Codec codec() const noexcept;
    // The next piece in seq order, verified and decoded; false once the whole session is complete and checked. Throws
    // on any transport, checksum, codec, ordering or manifest error, or an ABORT from the sender; after a throw every
    // later call throws that same first error (also after abort(), even once complete).
    bool next(Piece& piece);
    // Only after next() returned false: the status and payload for the sender's finish().
    void reply(std::uint32_t status, std::span<const std::uint8_t> payload = {});
    void abort(std::string_view why) noexcept;
    // From any thread (e.g. a listener shutting down): fails the session with that reason (a blocked or later next()
    // throws it) and cuts its streams, without joining anything; the owner still aborts or destroys the session.
    void cancel(std::string_view why) noexcept;
    [[nodiscard]] ReceiverStats stats() const;

private:
    friend class Listener;
    struct Impl;
    explicit Session(std::unique_ptr<Impl> impl);
    bool next_checked(Piece& piece);
    std::unique_ptr<Impl> impl_;
};

// Streaming XXH64 (seed 0 unless given); equal to bp_codec::xxh64 over the concatenated bytes.
class Xxh64 {
public:
    explicit Xxh64(std::uint64_t seed = 0) noexcept;
    void update(const void* data, std::size_t bytes) noexcept;
    [[nodiscard]] std::uint64_t digest() const noexcept;

private:
    std::uint64_t v_[4];
    std::uint64_t total_ = 0, seed_;
    std::uint8_t buf_[32];
    std::size_t fill_ = 0;
};

// The piece hash of the frame format: XXH64 of the descriptor followed by the plain bytes (regions concatenated).
[[nodiscard]] std::uint64_t piece_hash(std::uint32_t seq, const PieceHeader& header, std::span<const bp_codec::Region> regions);

}  // namespace strix::handoff
