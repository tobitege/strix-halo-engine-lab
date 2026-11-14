#pragma once
#include "strix/dist/exchange_direction.hpp"
#include "strix/dist/staged_ready.hpp"
#include "strix/dist/transport_caps.hpp"
#include "strix/dist/host_wait.hpp"
#include "strix/dist/transport.hpp"

// Two-rank communicator for the distributed runtime: whole-message transports between two ranks and a
// device-buffer exchange on top of them.
//
// Wire format: a message is its payload followed by a 128-byte trailer, so the trailer is the last
// thing to arrive; its sequence number tells a receiver (or, later, a device poll on the receive window)
// that the payload before it is complete, and a mismatch in epoch, plan, rank, tag, sequence, size or
// CRC-32C fails the exchange. Message sizes always come from data both ranks already share (routes, rows,
// plan); no shape metadata travels.
//
// Synchronization contract (also for every future backend): the bytes of a device buffer leave only
// after the producing work is complete as seen by the CPU, and a receive buffer is reused only after the
// work reading it is complete as seen by the CPU. Implicit device-side fences are not relied on: memory
// written by HIP kernels publishes none to the DMA engines.

#include <hip/hip_runtime.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace strix::dist {

class DeviceExchange; // gpu_direct.hpp

inline constexpr std::uint64_t trailer_magic = 0x31544c5254534944ULL; // "DISTRLT1"

struct Trailer {
    std::uint64_t magic = trailer_magic;
    std::uint64_t epoch = 0;
    std::uint64_t plan = 0;
    std::uint64_t sequence = 0;
    std::uint64_t request = 0;
    std::uint64_t bytes = 0;
    std::uint32_t rank = 0;
    std::uint32_t tag = 0;
    std::uint32_t layer = 0;
    std::uint32_t crc32c = 0;
    std::uint64_t staged_ns = 0; // sender's wall clock when the payload was staged (for clock-aligned traces)
    // STRIX_DIST_CRC_DEFER=1: the CRC of the sender's previous message when that one went with crc32c = 0
    // (flags & trailer_crc_deferred); carried_sequence is its sequence + 1, 0 when nothing is carried.
    std::uint64_t carried_sequence = 0;
    std::uint32_t carried_crc32c = 0;
    std::uint32_t flags = 0;
    std::uint64_t reserved[5] = {};
};
static_assert(sizeof(Trailer) == 128);
inline constexpr std::uint32_t trailer_crc_deferred = 1U; // this message's CRC travels in the next trailer

// CRC-32C (Castagnoli) of the payload; SSE4.2 when available. STRIX_DIST_CRC_PARALLEL=1
// selects vector folding when supported, otherwise three independent chains on
// large inputs; both preserve the identical 32 bits.
[[nodiscard]] std::uint32_t crc32c(std::span<const std::byte> bytes) noexcept;

[[nodiscard]] std::pair<std::unique_ptr<Transport>, std::unique_ptr<Transport>> memory_pair();
// TCP with TCP_NODELAY; the listener accepts one peer. Timeouts bound the connection setup and every
// later blocking send or receive.
[[nodiscard]] std::unique_ptr<Transport> tcp_listen(std::uint16_t port, std::chrono::milliseconds timeout);
[[nodiscard]] std::unique_ptr<Transport> tcp_connect(const std::string& host, std::uint16_t port, std::chrono::milliseconds timeout);
// Host path of USB4 streams: `devices` are stream device files, opened read-write (both ranks list the
// paired streams in the same order). A message is split over the streams at 64-byte boundaries, in
// proportion to integer weights (equal when none are given; the trailer ends up in the last part); each
// stream has a persistent thread that writes its part and reads the peer's part of the same message.
[[nodiscard]] std::unique_ptr<Transport> usb4_host(const std::vector<std::string>& devices,
                                                   const std::vector<std::uint32_t>& weights = {});
// The same transport over descriptors that are already open (the streams of a finished pairing, or socket
// pairs in tests); it owns and closes them. Messages shorter than `min_part` bytes per stream use fewer
// streams; a transfer without progress for `timeout` fails the transport.
[[nodiscard]] std::unique_ptr<Transport> usb4_host_descriptors(std::vector<int> fds, std::vector<std::string> names,
                                                               std::vector<std::uint32_t> weights, std::size_t min_part,
                                                               std::chrono::milliseconds timeout);

// Automatic pairing of every live USB4 stream between the two ranks (any cabling: straight or crossed,
// any number of cables, 20 or 40 Gb/s links).
//   1. Candidates: the stream entries of the layout file (keys "<service>/<stream>", e.g. "0-2.0/data0",
//      with "device"); a stream is opened only if the layout does not mark it unattached or unopenable
//      and its service is present in /sys/bus/thunderbolt/devices (a peer is attached on that port).
//      Link speed, lanes and the peer host come from sysfs; the layout's v2 fields are used when present.
//   2. Handshake on every opened stream (repeated until answered): protocol version, rank, process
//      nonce, local stream index and device number, host name; then an acknowledgement echoing the
//      peer's nonce and stream index, which proves both directions. Streams that stay silent or answer
//      one way only are closed and reported; streams answered by a different peer process or the same
//      rank fail the pairing.
//   3. The accepted pairs are ordered by rank 0's stream index; streams of one cable (one XDomain) are
//      grouped, and each cable's throughput is measured with a short exchange on all its streams.
//   4. Rank 0 sends the plan (stream order and integer striping weights, proportional to the measured
//      cable throughput and shared equally among the cable's streams); rank 1 checks it against its own
//      pairing and echoes it. Rank 0's environment can replace the weights (STRIX_DIST_USB4_WEIGHTS=W0,W1,...),
//      re-measure them with every cable at once (STRIX_DIST_USB4_CALIBRATE=joint, two rounds) and set the
//      streams per message size (STRIX_DIST_USB4_STREAMS=UP_TO:N,...); the plan carries all of it to rank 1.
// The detected topology is returned in `topology` (and its receipt() is the log line).
struct Usb4AutoOptions {
    std::uint32_t rank = 0;
    std::string layout;                          // empty: $STRIX_DIST_USB4_LAYOUT or /run/usb4stream-layout.json
    std::string sysfs = "/sys/bus/thunderbolt/devices";
    // Streams never to open (device files or layout keys); $STRIX_DIST_USB4_EXCLUDE (comma-separated) adds to it.
    std::vector<std::string> exclude;
    std::chrono::milliseconds timeout{60000};    // for the peer to appear on its first stream
    std::chrono::milliseconds grace{5000};       // for the remaining streams once one stream is paired
    std::size_t probe_bytes = std::size_t{16} << 20U; // per cable and direction; 0: weights from the link speed
    // Messages shorter than this per stream use fewer streams (rank 0's value, carried in the plan; 0 = always all;
    // $STRIX_DIST_USB4_MIN_PART overrides).
    std::size_t min_part_bytes = 0;
    // A transfer that makes no progress for this long fails the transport ($STRIX_DIST_USB4_TIMEOUT_MS overrides).
    std::chrono::milliseconds io_timeout{120000};
    // Exact set agreed by both ranks in the existing startup handshake.
    std::uint16_t transport_capabilities = all_transport_capabilities;
};
struct Usb4StreamPair {
    std::string key;           // layout key of the local stream
    std::string device;        // local device file
    std::string xdomain;       // local XDomain (one per cable)
    std::string cable;         // the layout's cable label, if any
    std::int32_t domain = -1;
    std::int32_t usb4_port = -1;
    double link_gbps = 0.0;    // raw link rate, min of both directions
    std::uint32_t rank0_index = 0;
    std::int32_t remote_device = -1; // the peer's tbstream number
    std::uint32_t remote_index = 0;
    double measured_gbps = 0.0; // the cable's exchange throughput, one direction
    std::uint32_t weight = 0;
};
struct Usb4Topology {
    std::string local_host, remote_host, remote_unique_id;
    std::vector<Usb4StreamPair> streams;  // in striping order
    std::vector<std::string> rejected;    // candidates not used, with the reason
    std::size_t min_part_bytes = 0;
    [[nodiscard]] std::string receipt() const;
};
[[nodiscard]] std::unique_ptr<Transport> usb4_host_auto(const Usb4AutoOptions& options, Usb4Topology* topology = nullptr);
// The layout's streams as seen now: attached means its service is present in sysfs; gpu_direct means its
// configfs `gpu_direct` reads 1 (such streams are left to the GPU-direct path: see gpu_direct.hpp).
struct Usb4LayoutStream {
    std::string key, service, stream, device, configfs, xdomain;
    bool attached = false, gpu_direct = false;
};
[[nodiscard]] std::vector<Usb4LayoutStream> usb4_layout_streams(const Usb4AutoOptions& options);

// Device-memory transport (the zero-copy path of a USB4 stream with device buffers). Planned interface:
// windows are registered once; submit(window, bytes) may be called only after the CPU has seen the
// producer complete, and repost(window) only after the CPU has seen every reader of that window
// complete; completion of a receive is the trailer at the end of the window carrying the expected
// sequence. Not implemented yet.
class DirectTransport {
public:
    virtual ~DirectTransport() = default;
    virtual std::uint32_t register_window(void* device, std::size_t bytes) = 0;
    virtual void submit(std::uint32_t window, std::size_t bytes) = 0;
    virtual void repost(std::uint32_t window) = 0;
    virtual bool received(std::uint32_t window, std::uint64_t sequence) = 0;
};

struct Group {
    std::uint32_t rank = 0;
    std::uint32_t world = 2;
    std::uint64_t epoch = 0;
    std::uint64_t plan = 0;
    std::uint16_t transport_capabilities = all_transport_capabilities;
};
struct Tag {
    std::uint32_t kind = 0;
    std::uint32_t layer = 0;
    std::uint64_t request = 0;
};
enum class TagKind : std::uint32_t { half_sum = 1, raw_slots = 2, argmax = 3, control = 4 };

struct CommunicatorStats {
    std::uint64_t exchanges = 0;
    std::uint64_t bytes_sent = 0;
    double seconds = 0.0;  // wall time inside exchange()
    double stage = 0.0;    // device to pinned host copy, with the CPU wait for it and the producer
    double wire = 0.0;     // transport exchange, including the wait for the peer
    double check = 0.0;    // trailer and CRC checks, and the host to device copy being enqueued
    double overlap = 0.0;  // between start() returning and finish() being called (work enqueued meanwhile)
    double host_reuse_wait = 0.0; // ordinary pinned receive staging, caller wait
    double pipeline_reuse_wait = 0.0; // chunked receive staging, caller or worker wait
    double finish_wait = 0.0; // inside finish(), waiting for the progress thread
    std::chrono::steady_clock::time_point last_entry{};
    std::chrono::steady_clock::time_point last_staged{};   // payload staged, sending starts
    std::chrono::steady_clock::time_point last_received{}; // peer message complete
};

// Where a message of up to `up_to` bytes goes: the device exchange (GPU-direct) on `rings` rings (0: the engine's
// rule), or the host path. Routes are ordered by `up_to`; the last one covers every size. A mixed route
// (`host_percent` > 0, direct) sends the first host_percent % of the bytes (rounded down to 64 B, at most max_bytes)
// over the host path and the rest over the device exchange, both at once; each part lands at its own offset of the
// receive buffer, so the bytes delivered are the same as on either path alone.
struct Route {
    std::size_t up_to = 0;
    bool direct = true;
    unsigned rings = 0;
    unsigned host_percent = 0;
};

// One per rank. Collective calls are made in the same order on both ranks.
class Communicator {
public:
    Communicator(Group group, std::unique_ptr<Transport> transport, std::size_t max_bytes);
    // With a device exchange (GPU-direct): exchanges up to its capacity go device buffer to device buffer with no
    // host copy and no CPU wait on the caller's thread; larger ones (and the host-side collectives of the setup)
    // use `transport`. The same calls, in the same order on both ranks.
    Communicator(Group group, std::unique_ptr<Transport> transport, std::unique_ptr<DeviceExchange> direct, std::size_t max_bytes);
    ~Communicator();
    Communicator(const Communicator&) = delete;
    Communicator& operator=(const Communicator&) = delete;

    // Resize host capacity between collectives; preserves sequence numbers and the route table. grow_pipeline = false:
    // only the (fused) staging buffers grow; the chunked host pipeline keeps its capacity and messages above it take the
    // plain host path (for large staged messages only, without a second pair of pipeline buffers).
    void resize_bytes(std::size_t bytes, bool grow_pipeline = true);
    [[nodiscard]] std::size_t max_bytes() const noexcept { return max_bytes_; }

    // Sends `bytes` of device buffer `local` to the peer and writes the peer's `bytes` into device buffer
    // `remote`. The copy of `local` is ordered after its producer on `stream` and the CPU waits for it
    // before sending; the copy into `remote` is enqueued on `stream` (later work on the stream sees it),
    // and the CPU waits for that copy before the receive buffer is used again by the next exchange. Both
    // ranks pass the same size and tag.
    // STRIX_DIST_HOST_CHUNK_BYTES=N (default 0) pipelines full-host messages above N
    // on separate copy streams, with an independently checked trailer per chunk.
    // Both ranks must use the same N; direct/mixed routes and staged exchanges keep
    // their existing protocol. No consumer work is released after a failed chunk.
    // STRIX_DIST_HOST_CRC_OVERLAP=1 additionally prepares TX CRCs and validates RX
    // chunks on persistent workers while the progress thread moves other chunks.
    void exchange(hipStream_t stream, const void* local, void* remote, std::size_t bytes, const Tag& tag);
    // One-way device transfer of `bytes` (GPU-direct routes only): the sending rank passes sending = true and its
    // device buffer, the receiving rank sending = false and the buffer to fill; both at the same point with the same
    // size and tag. Nothing travels in the unused direction and no early receive is posted. The receiver's GPU checks
    // the in-band identity and checksum as for exchange(); check_exchanges() reports a refusal. Returns false, with
    // nothing sent, when this size has no pure device route (host or mixed route, or an engine without a one-way
    // form): the caller then uses its host path. Both ranks get the same answer (same route table).
    [[nodiscard]] bool transfer(hipStream_t stream, bool sending, void* device, std::size_t bytes, const Tag& tag);
    // The same exchange in two halves, so that work enqueued in between overlaps the transfer. start()
    // enqueues the copy of `local` on `stream` and returns; the progress thread waits on the CPU for that
    // copy, then sends and receives. finish() waits for the peer's message, checks it, and enqueues its
    // copy into `remote` on `stream`. One exchange at a time: start, (enqueue other work), finish.
    void start(hipStream_t stream, const void* local, std::size_t bytes, const Tag& tag);
    void finish(hipStream_t stream, void* remote);
    // Host-resident payloads (control frames, forwarded messages): `local` goes straight into the pinned send buffer and
    // the peer's message is copied out of the pinned receive buffer into `remote` (empty: not wanted), with no device
    // buffer and no device copy on either side of the wire. The message on the wire (payload, trailer, sequence) is the
    // same as exchange()'s, so a peer may use either call for the same collective. The calling thread does the transfer;
    // not between start() and finish().
    void exchange_host(std::span<const std::byte> local, std::span<std::byte> remote, const Tag& tag);
    // Per-communicator opt-in, without changing other engines/process environment.
    void enable_async_start() { async_start_ = true; }
    // Enable before the first collective. Per-communicator scheduling policy for messages above GPU ring capacity:
    // start the first chunk before caller work; finish drains it and the tail.
    // Same bytes, chunk identities, route table and buffer lifetime as before.
    void enable_direct_chunk_overlap() { direct_chunk_overlap_ = true; }
    // Fatal teardown: stop and join host submission before caller events/buffers
    // can be retired. The communicator remains poisoned; caller must drain GPU work.
    void cancel_pending(const std::string& reason) noexcept;
    // STRIX_DIST_ASYNC_START=1 or enable_async_start(): full-host chunked path only. Publish remote at start;
    // destination_last_use must already be recorded after its last reader. Records
    // source_free after all off-stream D2H: wait on it before overwriting local.
    // finish only joins validation/copy submission and waits on the final H2D event.
    // One exchange at a time. Returns false (ordinary start contract) on other paths;
    // neither caller event is used then. Keep events/buffers alive until their
    // queued device work completes; finish joins submission, not device execution.
    bool start_async(hipStream_t stream, const void* local, void* remote, std::size_t bytes, const Tag& tag,
                     hipEvent_t destination_last_use, hipEvent_t source_free);
    // GPU-direct: consumes a receive posted early for a guessed next exchange (DeviceExchange::quiesce), so that the
    // session can stay idle. Both ranks at the same point, between exchanges; nothing to do on the host path.
    void quiesce(hipStream_t stream);
    // Relinquish a borrowed transport to another progress owner between complete
    // exchanges: retire deferred validation, direct receives and device readers.
    void release_transport(hipStream_t stream);
    // Opt-in small host path, STRIX_DIST_STAGED_READY_BYTES=1..1048576 (0: unavailable).
    // One stream must outlive the communicator. Kernels write/read the coherent
    // views in place and call publish_staged_block. No HIP calls in start/finish/
    // release on this path. prepare returns {} for routes/sizes not admitted.
    [[nodiscard]] ProducerStage prepare_staged_ready(hipStream_t stream, std::size_t bytes);
    void start_staged_ready(hipStream_t stream, std::size_t bytes, const Tag& tag);
    [[nodiscard]] ConsumerStage finish_staged_ready();
    void release_staged_ready(hipStream_t stream);
    // Fused staging (host path only): send_staging(bytes) is the pinned send buffer as a device pointer when an exchange
    // of `bytes` would take the host path (nullptr otherwise: use start / finish). The caller's kernel writes the message
    // there on `stream`, then start_staged(stream, bytes, tag) sends it (no device-to-host copy); finish_staged(stream)
    // returns the peer's message in the pinned receive buffer (no host-to-device copy, checked as in finish), valid for
    // kernels enqueued on `stream` before release_staged(stream), which must follow before the next start.
    [[nodiscard]] void* send_staging(std::size_t bytes) const;
    void start_staged(hipStream_t stream, std::size_t bytes, const Tag& tag);
    [[nodiscard]] const void* finish_staged(hipStream_t stream);
    void release_staged(hipStream_t stream);
    [[nodiscard]] const CommunicatorStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const Group& group() const noexcept { return group_; }
    [[nodiscard]] std::string transport_name() const;
    // The transport, for a module that borrows it between this communicator's exchanges (both ranks at the same point,
    // nothing in flight here: e.g. an in-process A/B of XferExchange against CommExchange).
    [[nodiscard]] Transport& transport() noexcept { return *transport_; }
    [[nodiscard]] const DeviceExchange* device_exchange() const noexcept { return direct_.get(); }
    // The size router (with a device exchange): STRIX_DIST_ROUTE=table (default: the measured table; with host
    // streams, small messages on the host path, two rings up to 1 MB, mixed 35 % host above; without, one ring up to
    // 32 KB, two up to 256 KB, every ring above),
    // auto (measured at construction at idle, rank 0 decides), gpu (every size direct, the engine's ring rule), host
    // (the host path up to max_bytes), or an explicit table "UP_TO:host|gpu[:RINGS]|mix:RINGS:HOST_PERCENT,..." (UP_TO
    // in bytes or "inf"; mix: see Route).
    // The host path is never chosen when the transport is only a control channel (TCP rendezvous of the
    // all-GPU-direct mode).
    [[nodiscard]] const std::vector<Route>& routes() const noexcept { return routes_; }
    [[nodiscard]] std::string route_text() const;
    // Replaces the route table (same syntax as an explicit STRIX_DIST_ROUTE). Both ranks, same table, between exchanges.
    void set_routes(const std::string& table);
    // Throws if a device exchange has failed or was refused by the receiving GPU. A failed exchange already stopped
    // its stream (later work does not run and the stream's synchronisation reports an error); this names the cause.
    void check_exchanges() const;

private:
    Group group_;
    xport::HostWaitMetric slot_wait_{"communicator", "slot"};
    xport::HostWaitMetric peer_wait_{"communicator", "peer"};
    xport::HostWaitMetric worker_slot_wait_{"communicator-worker", "slot"};
    std::unique_ptr<Transport> transport_;
    std::size_t max_bytes_;
    std::byte* send_host_ = nullptr;
    std::byte* receive_host_ = nullptr;
    void* send_device_alias_ = nullptr;
    std::size_t d2h_kernel_bytes_ = 0; // STRIX_DIST_HOST_D2H_KERNEL_BYTES, default off; at most 1 MiB
    std::size_t staged_ready_bytes_ = 0, staged_prepared_bytes_ = 0;
    std::byte *staged_send_host_ = nullptr, *staged_receive_host_ = nullptr;
    void *staged_send_device_ = nullptr, *staged_receive_device_ = nullptr;
    StagedReady *producer_flag_ = nullptr, *consumer_flag_ = nullptr;
    StagedReady *producer_flag_device_ = nullptr, *consumer_flag_device_ = nullptr;
    unsigned* staged_counters_ = nullptr;
    std::uint64_t staged_token_ = 0, consumer_pending_token_ = 0;
    hipStream_t staged_stream_ = nullptr;
    bool staged_exposed_ = false, staged_prepared_ = false, staged_started_ = false, staged_receive_held_ = false;
    void wait_staged_flag(StagedReady* flag, std::uint64_t token, std::chrono::steady_clock::time_point deadline);
    // STRIX_DIST_CRC_DEFER (default on, =0 off; both ranks the same), ready-view exchanges only: the TX CRC is computed while
    // the message is on the wire and travels in the NEXT trailer; finish_staged_ready() returns once the trailer's identity
    // is checked, and the progress thread then computes the RX CRC, checked against the peer's next trailer. Every
    // exchange (deferred or not) carries and checks the previous deferred CRC, so a step that ends with an ordinary
    // exchange (the argmax / logits) has every payload checked before its result is returned; a mismatch fails it.
    bool crc_defer_ = false, deferring_ = false, tail_pending_ = false;
    std::uint64_t tx_carry_sequence_ = 0, rx_pending_sequence_ = 0; // sequence + 1; progress thread only
    std::uint32_t tx_carry_crc_ = 0, rx_pending_crc_ = 0;
    void drain_tail(); // waits for the previous job's deferred CRC work (it reads the staged buffers)
    std::uint64_t sequence_ = 0;
    hipEvent_t receive_done_ = nullptr;
    hipEvent_t send_ready_ = nullptr;
    bool receive_pending_ = false;
    bool started_ = false;
    std::size_t bytes_ = 0;
    Tag tag_;
    std::chrono::steady_clock::time_point started_at_{}, returned_at_{};
    // STRIX_DIST_TRACE=FILE: one line per exchange (FILE.r<rank>), wall-clock ns, written when the next exchange
    // starts. Host path ("H"): entry, producer done (seen by the CPU), staged (D2H + CRC done), the peer's staged
    // time, received, validated, consumer (H2D end on the GPU timeline). GPU-direct ("G", from the device exchange):
    // entry, scatter seen, TX submitted, RX complete, TX complete, consumer (gather end).
    std::FILE* trace_ = nullptr;
    hipEvent_t pre_copy_ = nullptr;
    std::uint64_t t_entry_ = 0, t_produced_ = 0, t_staged_ = 0, t_received_ = 0, t_validated_ = 0, peer_staged_ns_ = 0;
    hipEvent_t h2d_done_ = nullptr;
    struct PendingTrace {
        std::uint64_t sequence = 0;
        std::size_t bytes = 0;
        std::uint32_t tag = 0, layer = 0;
        std::uint64_t entry = 0, produced = 0, staged = 0, peer_staged = 0, received = 0, validated = 0;
    } pending_;
    bool trace_pending_ = false;
    void flush_trace();
    void validate_peer(bool deferred = false); // the peer's trailer and CRC of the received message (deferred: identity only)
    // Bounded waits (STRIX_DIST_WAIT_TIMEOUT_MS, default 120000): the producer's copy, the reuse of the receive buffer,
    // and the progress job. On expiry (or any failed host exchange) the communicator stops for good, prints its state
    // (the collective in flight and its phase, the last collectives, the streams' progress) to stderr and closes the
    // transport so the peer's pending reads end at once.
    enum class Phase : int { idle = 0, reuse_wait = 1, producer = 2, transfer = 3, check = 4 };
    struct History {
        std::uint64_t sequence = 0, bytes = 0;
        std::uint32_t tag = 0, layer = 0;
        std::uint64_t request = 0, cmd_out = 0, cmd_in = 0; // the first 8 bytes sent / received (a command word)
        char path = 'H';
    };
    std::chrono::milliseconds wait_timeout_{120000};
    std::string terminal_;
    std::atomic<bool> cancel_{false};
    std::atomic<int> phase_{0};
    std::atomic<std::uint64_t> phase_since_{0}, cmd_out_{0}, cmd_in_{0};
    History current_;
    std::array<History, 32> history_{};
    std::size_t history_next_ = 0;
    void set_phase(Phase p);
    [[noreturn]] void stop(const std::string& why);
public:
    // The communicator's state for a failure report: phase, the collective in flight, the last collectives, the
    // transport's progress.
    [[nodiscard]] std::string diagnose() const;
private:
    static bool check_on_caller();
    class Progress;
    std::unique_ptr<Progress> progress_;
    class Pipeline;
    std::unique_ptr<Pipeline> pipeline_;
    bool pipeline_started_ = false;
    bool async_start_ = false;
    void start_pipeline(hipStream_t stream, const void* local, std::size_t bytes, void* remote = nullptr,
                        hipEvent_t destination_last_use = nullptr, hipEvent_t source_free = nullptr);
    void finish_pipeline(hipStream_t stream, void* remote);
    std::unique_ptr<DeviceExchange> direct_;
    bool direct_started_ = false;
    bool direct_chunk_overlap_ = false;
    std::vector<Route> routes_;
    void configure_routes();
    [[nodiscard]] const Route& route(std::size_t bytes) const;
    void start_host(hipStream_t stream, const void* local, std::size_t bytes);
    void finish_host(hipStream_t stream, void* remote);
    bool staged_ = false;  // the exchange in flight is a fused-staging one (start_staged)
    bool fused_ = true;    // fused staging (STRIX_DIST_STAGING=copy: off): staging buffers coherent, send_staging() available
    const void* direct_local_ = nullptr;
    // The device exchange's part: bytes, offset in the message (non-zero when a mixed route sent the head over the
    // host path), and the mixed route's rings.
    std::size_t direct_bytes_ = 0, direct_offset_ = 0;
    unsigned mix_rings_ = 0;
    // The exchange in flight is a one-way transfer(): its direction, else both.
    ExchangeDirection direction_ = ExchangeDirection::both;
    void direct_hint(std::size_t bytes, unsigned rings);
    bool one_way_probed_ = false, one_way_ = false;
    CommunicatorStats stats_;
};

} // namespace strix::dist
