#pragma once

// GPU-direct exchanges for the two-rank communicator: device buffer to device buffer over USB4 streams in
// gpu_direct mode (DMA-BUF windows of GPU memory, the driver's multi-slot GPU queue with kernel-validated
// terminal frames). No host copies, and the stream that enqueues the exchange never waits on the CPU.
//
// Per stream: a TX and an RX arena of ring_frames x 4 KB in device memory, exported as DMA-BUF and imported
// once; `depth` slots per direction. One exchange k (sequence k + 1) on every stream it uses:
//   start  (caller's thread): device-to-device copy of the caller's buffer into TX slot k % depth, an event
//          after it; the progress thread is posted the exchange.
//   progress thread: retire RX k - depth once the CPU has seen its consumer copy complete, PREPARE + SUBMIT
//          RX k (same sequence and size as the peer's TX k), wait on the CPU for the producer copy's event,
//          PREPARE + SUBMIT TX k, poll EVENT until RX k and TX k complete, then write the sequence into a
//          host-mapped flag.
//   finish (caller's thread): the stream waits for the flag (hipStreamWaitValue32), then device-to-device
//          copies the RX slot into the caller's buffer and records the consumer event. Nothing blocks.
// Ordering: both ranks post and submit in sequence order; a TX that reaches the peer before its RX is posted
// waits on end-to-end flow control. The kernel checks each received terminal (epoch, sequence, payload size,
// frame count, slot) against the local RX operation, so ranks that disagree on the order or the size of an
// exchange fail instead of mixing data. CPU witnesses (C-1: DMA-BUF reservations do not see HIP work): the
// producer copy is complete before SUBMIT, the consumer copy before an RX slot is reused.
// Failure: an error or a deadline (no completion within `timeout`) poisons the exchange, releases the stream
// waiting on the flag (with a value that marks the failure) and makes every later call throw.

#include "strix/dist/exchange_direction.hpp"
#include "strix/linux_gpu_queue.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strix::dist {

class Transport;

struct DeviceExchangeStats {
    std::uint64_t exchanges = 0;
    std::uint64_t bytes = 0;
    // Payload bytes this rank put on / took off the rings (a symmetric exchange counts in both, a one-way
    // transfer in one), and the one-way transfers among `exchanges`.
    std::uint64_t bytes_tx = 0, bytes_rx = 0, one_way = 0;
    double producer_wait = 0.0;  // progress thread: waiting for the producer copy before SUBMIT (seconds)
    double completion_wait = 0.0; // from the last SUBMIT until RX and TX are complete
    double eagain_retries = 0.0;
    std::uint64_t preposted = 0, prepost_misses = 0; // exchanges whose RX was posted early; wrong guesses
    // Breakdown (seconds, summed over exchanges): CPU side of the progress thread, then, with
    // STRIX_DIST_GPU_TIMING=1, the stream's own timeline (`timed` exchanges).
    double submit = 0.0;   // witness seen -> every TX submitted
    double rx_wait = 0.0;  // TX submitted -> every RX complete, flag written
    double tx_tail = 0.0;  // flag written -> every TX complete
    std::uint64_t timed = 0;
    double scatter_gpu = 0.0, wait_gpu = 0.0, gather_gpu = 0.0; // scatter kernel; scatter end -> gather start; gather kernel
};

// Identity of one collective, carried in-band and checked by the receiver (as the host path's trailer).
struct ExchangeIdentity {
    std::uint64_t plan = 0, request = 0;
    std::uint32_t tag = 0, layer = 0;
};

// A device-memory exchange engine behind the communicator (one exchange at a time, both ranks in the same
// order with the same sizes).
class DeviceExchange {
public:
    virtual ~DeviceExchange() = default;
    [[nodiscard]] virtual std::size_t capacity() const noexcept = 0;
    virtual void start(hipStream_t stream, const void* local, std::size_t bytes, const ExchangeIdentity& identity) = 0;
    virtual void finish(hipStream_t stream, void* remote) = 0;
    [[nodiscard]] virtual std::string name() const = 0;
    [[nodiscard]] virtual DeviceExchangeStats stats() const = 0;
    // Empty while every exchange so far was accepted; otherwise the reason (a host-side error, or the receiving
    // GPU's refusal of a header: identity, size, checksum). Call it after synchronising a stream whose results
    // are about to be used.
    [[nodiscard]] virtual std::string failure() const = 0;
    // Rings available, and how many the next start() uses (0: the engine's own rule from the size). Both ranks set
    // the same value before the same exchange.
    [[nodiscard]] virtual unsigned rings() const noexcept { return 1U; }
    // Diagnostics: GPU read-path timings of the RX arena vs ordinary device memory (no exchange; call it between
    // exchanges). Empty if the engine has none.
    [[nodiscard]] virtual std::string diagnose(std::size_t) { return {}; }
    virtual void use_rings(unsigned) {}
    // One-way transfer: the next start() moves bytes in one direction only. The sending rank calls
    // use_direction(send) and the receiving rank use_direction(receive) before the same exchange; the sender posts no
    // receive and the receiver no transmit, so nothing travels in the unused direction. The receiver's GPU checks the
    // same in-band identity and checksum, and the sender's operation completes only against the receive the peer
    // posted (the rings' flow control). A one-way transfer posts no early receive. start() takes `local` on the
    // sender (ignored on the receiver), finish() takes `remote` on the receiver (ignored on the sender). False: this
    // exchange has no one-way form; the caller keeps the symmetric exchange.
    [[nodiscard]] virtual bool use_direction(ExchangeDirection) { return false; }
    // The expected size and ring count (as use_rings) of the exchange after the next start(): its receive is posted
    // early, before this exchange's TX, so the peer never transmits into an empty receive ring. A wrong guess costs one
    // internal exchange of the guessed geometry, made the same way by both ranks (the hints are the same calls).
    virtual void expect_next(std::size_t, unsigned) {}
    // Leaves no receive posted early (it would count against the driver's DMA deadline while idle): an early receive
    // is consumed by an internal exchange of its geometry. Both ranks at the same point, between exchanges (e.g. at
    // the end of a request in a server before going idle).
    virtual void quiesce(hipStream_t) {}
    // STRIX_DIST_TRACE (set by the communicator): one "G" line per exchange into `trace`, written on the caller's
    // thread when the next exchange starts (or at destruction): see Communicator's trace.
    virtual void set_trace(std::FILE*) {}
};

// One gpu_direct stream ready to import: its queue descriptor, the endpoint identity it must report, and the
// arenas' backing (exported device memory, or fakes in tests).
struct GpuDirectArena {
    void* tx = nullptr;          // device pointer of the TX window
    void* rx = nullptr;          // device pointer of the RX window
    transport::GpuQueuePool pool; // the DMA-BUF descriptors and their owners
};
struct GpuDirectLink {
    std::string name;
    transport::GpuQueueIo io;
    transport::GpuQueueExpected expected; // identity, ring frames, depth, terminal mode
    GpuDirectArena arena;
};

struct GpuDirectOptions {
    std::uint64_t epoch = 0;        // peer-agreed, nonzero, fresh
    std::uint64_t pool_identity = 0; // peer-agreed, nonzero
    std::chrono::milliseconds timeout{30000};
    std::size_t min_part = 4096;     // a message uses floor(bytes / min_part) streams (at least one)
};

// The exchange engine over its links, with the link-level steps of a bootstrap that has no host-mode stream.
// They run before the first exchange, both ranks calling the same steps in the same order:
//   hello(): one fixed-size message each way on every link independently (a ring is point to point, so what
//            lands on local link j was sent by the peer's end of the same cable): the cable map;
//   reorder(): this rank's links in the agreed order (the peer's link j is then the other end of local link j);
//   round_trip_us(): p50 of `rounds` synchronous round trips of `bytes` on every link, in link order.
class GpuDirectLinks : public DeviceExchange {
public:
    static constexpr std::size_t hello_bytes = 256;
    [[nodiscard]] virtual std::size_t link_count() const noexcept = 0;
    [[nodiscard]] virtual std::string link_name(std::size_t j) const = 0;
    [[nodiscard]] virtual std::vector<std::string> hello(const std::vector<std::string>& mine, std::chrono::milliseconds timeout) = 0;
    virtual void reorder(const std::vector<std::size_t>& order) = 0; // new link j = old link order[j]
    virtual void keep(std::size_t count) = 0; // only the first `count` links from now on (the others are released)
    [[nodiscard]] virtual std::vector<double> round_trip_us(std::size_t rounds, std::size_t bytes) = 0;
    // Everything both ranks must have identical for the parts of a message to be cut the same way and land in the
    // same slots: wire format version, link count, depth, slot bytes, min_part, split rule.
    [[nodiscard]] virtual std::string parameters() const = 0;
};

// Imports and arms every link, then returns the exchange engine. Both ranks pass their links in the same
// (paired) order with the same depth, or reorder them before the first exchange.
[[nodiscard]] std::unique_ptr<GpuDirectLinks> gpu_direct_exchange(std::vector<GpuDirectLink> links, GpuDirectOptions options);

// Device arena of `window_bytes` per direction exported as DMA-BUF (hipMemGetHandleForAddressRange).
[[nodiscard]] GpuDirectArena export_gpu_direct_arena(std::size_t window_bytes, std::uint64_t generation);

// Setup over real streams: the layout's streams whose configfs `gpu_direct` is 1 (never touched here), paired
// cable by cable through the host-path streams of `control` (which must be a usb4_host_auto transport of the
// same pairing; `topology` is its receipt), epoch and pool identity agreed over `control`. Throws with a
// clear reason when no stream is in gpu_direct mode on both ranks.
struct Usb4Topology;
struct Usb4AutoOptions;
struct GpuDirectSetup {
    std::uint32_t depth = 4;
    std::uint32_t max_streams = 8; // use at most this many paired gpu_direct streams
    std::chrono::milliseconds timeout{30000};
};
[[nodiscard]] std::unique_ptr<DeviceExchange> usb4_gpu_direct(const Usb4AutoOptions& options, const Usb4Topology& topology,
                                                              Transport& control, GpuDirectSetup setup, std::string* receipt = nullptr);

// The link-level bootstrap over an engine's links (both ranks, before the first exchange): the session parameters
// must be identical; HELLO on every ring, checked against each ring's endpoint identities both ways (a foreign
// endpoint fails; crossed cables and stream numbering that differs between the hosts are fine: the pairing is
// physical, the names are local); rank 1's links put in rank 0's order, then both put in rank 0's latency ranking. `control` carries only the small
// agreement messages. Returns a receipt (one entry per ring, in the final order).
struct GpuDirectRing {
    std::string name;                // for messages: "0-2.0/data1 /dev/tbstream2"
    std::string stream;              // dataN, the same at both ends of a cable
    std::string local_id, peer_id;   // endpoint identities from QUERY (hex)
    [[nodiscard]] std::string key_token() const { auto k = name.substr(0, name.find(' ')); return k.empty() ? "-" : k; }
};
[[nodiscard]] std::string gpu_direct_bootstrap(GpuDirectLinks& exchange, std::uint32_t rank, Transport& control,
                                               const std::vector<GpuDirectRing>& rings, std::uint64_t nonce);

// Fallback check before any GPU-direct setup, both ranks (one control message): the number of gpu_direct streams both
// ranks have attached (0 if either has none; `why` then says which). A caller falls back to the host path when it is 0,
// so both ranks make the same choice; failures later in the setup still throw (fail closed, with the cause).
[[nodiscard]] std::size_t usb4_gpu_direct_streams(const Usb4AutoOptions& options, Transport& control, std::string* why = nullptr);

// Setup with every stream in gpu_direct mode (no host-mode stream): `control` is an out-of-band transport (TCP over
// Ethernet), used for the rendezvous only (inventory, epoch, pool identity, the "armed" barrier and the agreed link
// order). The cable map comes from an in-band HELLO on every ring; the rings are then ranked by measured latency
// (link 0, the fastest, carries the messages that use one ring). Exchanges never touch `control`.
[[nodiscard]] std::unique_ptr<DeviceExchange> usb4_gpu_direct_all(const Usb4AutoOptions& options, Transport& control, GpuDirectSetup setup,
                                                                  std::string* receipt = nullptr);

} // namespace strix::dist
