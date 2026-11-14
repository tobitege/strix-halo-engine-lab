#pragma once
#include "staging.hpp"
#include "row_fold.hpp"
// Shared TP2 staging/progress engine; engine-specific wire ABI lives in Traits.
#include "strix/dist/communicator.hpp"

#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>

namespace strix::xport {

// Where the TX / RX slots live. host_coherent: hipHostMalloc coherent+mapped (CPU-cached, GPU uncached). device_fine:
// fine-grained device memory (hipExtMallocWithFlags hipDeviceMallocFinegrained), which libraries that only accept
// device pointers (strix_gemm) can write directly; the CPU reads / writes it in place.
// device_uncached: hipDeviceMallocUncached. Validation found device_fine UNSAFE for the CPU side (stale TX bytes read
// after the flag; see design notes): only host_coherent is a default; the device kinds are for the coherence probe.
enum class Memory : std::uint32_t { host_coherent = 0, device_fine = 1, device_uncached = 2 };

// How the TX flag of a message is written once its producers are enqueued.
enum class Publish : std::uint32_t {
    kernel = 0,       // a 1-block kernel after the producers (system-scope release store)
    write_value = 1,  // hipStreamWriteValue64 after the producers
    producer = 2,     // the producer's own last block publishes (glm_xfer_device.hpp), nothing is enqueued
};

// A received message for consumers that wait for it themselves (glmx_wait in glm_xfer_device.hpp) instead of a
// stream wait packet: every block of the first consumer kernel calls glmx_wait(view) before reading the payload.
struct RxView {
    const void* payload = nullptr;
    const std::uint64_t* flag = nullptr;  // device pointer of the RX flag
    std::uint64_t sequence = 0;
    // Device-memory gate (measurement: with every block polling the host flag, a 196 KB consumer's 192 pollers slowed the whole
    // exchange 2.4x): block 0 polls the host flag and opens the gate, the other blocks poll the gate in L2. nullptr:
    // every block polls the host flag.
    std::uint64_t* gate = nullptr;
};

// What a producer kernel needs to publish its own message (Publish::producer): see glm_xfer_device.hpp.
struct ProducerTarget {
    std::uint64_t* flag = nullptr;    // device pointer of the TX flag
    unsigned* counter = nullptr;      // device scratch, reset by the last block
    std::uint64_t sequence = 0;       // value to publish
};

template<class Traits>
struct ChannelOptions {
    std::uint32_t rank = 0;
    std::uint64_t epoch = 1;
    std::uint64_t plan = Traits::kPlan;  // "GLMX"
    std::size_t capacity = 0;            // largest payload in bytes
    unsigned depth = 4;                  // TX / RX slots
    hipStream_t stream = nullptr;
    Publish publish = Publish::kernel;   // default for send()
    bool crc = Traits::kCrc;
    bool crc_deferred = Traits::kCrcDeferred;             // CRC off the critical path (see kInlineCrcMax); measured exact, never slower
                                          // (N=1 -24 %, N=8 -36 % exchange cost); STRIX_GLMX_CRC=1 restores inline
    Memory tx_memory = Memory::host_coherent, rx_memory = Memory::host_coherent;
    int progress_cpu = -1;
    int crc_cpu = -1;
    std::uint64_t test_corrupt_sequence = 0;  // tests only (STRIX_GLMX_TEST_CORRUPT): flip one received byte of this sequence
    unsigned idle_spin_us = 3000;        // progress thread: spin this long without a job before sleeping
    bool small_spin = false;            // shared opt-in ordered small-message progress
    std::chrono::milliseconds timeout{120000};
    // Exchange row fold (row_fold.hpp): largest message a fold may be requested for; 0 = no fold slots are allocated
    // and fold_next() is refused. The fold slots are host-coherent like the default RX slots.
    std::size_t fold_capacity = 0;
    // Reads the STRIX_GLMX_* variables over these values.
    void from_env();
};

template<class Traits>
class Channel {
public:
    template<class Ops> using Staging = StagingChannel<typename Traits::Tag, Ops>;
    using Options = ChannelOptions<Traits>;
    using Tag = typename Traits::Tag;
    using Kind = typename Traits::Kind;
    using Trailer = typename Traits::Trailer;
    using Schedule = typename Traits::Schedule;
    using Payload = typename Traits::Payload;
    struct Ticket { Tag tag; std::uint64_t sequence; };
    struct StagingView { void* data; ProducerTarget producer; };
    static_assert(Schedule::row_bytes(1) == Traits::kWidth * sizeof(Payload));
    static constexpr std::size_t kInlineCrcMax = Traits::kInlineCrc;
    static constexpr std::uint64_t kTrailerMagic = Traits::kMagic;
    static constexpr std::uint16_t kTrailerDeferred = Traits::kDeferred, kTrailerPrev = Traits::kPrevious;
    Channel(std::unique_ptr<dist::Transport> owned, Options options);
    Channel(dist::Transport& borrowed, Options options);
    ~Channel();
    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;

    // Payload buffer (device-accessible, host-coherent) of the next exchange; producers write the message here.
    [[nodiscard]] void* tx();
    // For a producer that publishes itself (Publish::producer): the flag, its counter and the value, for the next send.
    [[nodiscard]] ProducerTarget producer_target();
    // The channel owns this staging storage; producers publish through its existing flags.
    [[nodiscard]] StagingView staging() { return {tx(), producer_target()}; }
    [[nodiscard]] Ticket post(const Tag& tag, StagingView source, std::size_t bytes, Publish how) {
        if (source.data != tx() || source.producer.sequence != next_ + 1)
            throw std::invalid_argument("xport: stale or foreign staging view");
        send(bytes, tag, how);
        const Ticket ticket{tag, next_};
        tickets_.push_back(ticket);
        return ticket;
    }
    // This ordered-stream backend admits waits in posting order. Wire matching still checks
    // the complete tag in process(), independently of the physical staging slot.
    [[nodiscard]] const void* wait(const Ticket& ticket, hipStream_t stream) {
        if (stream != options_.stream || ticket.sequence != received_ + 1)
            throw std::invalid_argument("xport: wait must use the posting stream and order");
        for (auto i = tickets_.begin(); i != tickets_.end(); ++i) {
            if (i->sequence != ticket.sequence) continue;
            if (i->tag.kind != ticket.tag.kind || i->tag.layer != ticket.tag.layer || i->tag.step != ticket.tag.step)
                throw std::invalid_argument("xport: ticket tag differs");
            const void* remote = recv();
            tickets_.erase(i);
            return remote;
        }
        throw std::invalid_argument("xport: unknown ticket");
    }
    // The message of `bytes` is produced by the work enqueued so far on the stream: publish it (per `how`) and hand
    // the transfer to the progress thread. Never blocks.
    void send(std::size_t bytes, const Tag& tag, Publish how);
    void send(std::size_t bytes, const Tag& tag) { send(bytes, tag, options_.publish); }
    // Copy `bytes` of device buffer `local` into tx() with one kernel whose last block publishes, then send.
    void send_copy(const void* local, std::size_t bytes, const Tag& tag);
    // SP-only direct coherent host aliases. Both buffers must remain live and
    // unchanged until RX completion; consumers must retire before reuse. Unlike
    // send_copy this publishes readiness only, and receives into the consumer.
    // Uses a separate trailer exchange; the peer must use this same API.
    void send_host(const void* local, void* remote, std::size_t bytes, const Tag& tag);
    // The stream waits for the peer's message of the oldest send() not yet received; returns its payload (RX slot).
    [[nodiscard]] const void* recv();
    // As recv() without the stream wait packet: the consumer kernels wait in-kernel (glmx_wait), saving the command
    // processor's wake-up and the launch behind it. Same lifetime rules as recv().
    [[nodiscard]] RxView recv_view();
    // The local payload of the exchange recv() just returned (its TX slot), for consumers of (part0, part1).
    [[nodiscard]] const void* last_local() const noexcept { return last_local_; }
    // Exchange row fold. fold_next(op): the next send()'s exchange is folded by the progress thread before its RX flag
    // is published: fold slot = op(rank 0's partial, rank 1's partial) over the message bytes. last_folded(): the fold
    // slot (device pointer) of the exchange recv() / recv_view() just returned, or nullptr if it was not folded. The
    // slot has the lifetime of the RX slot; a consumer reads it only after the RX flag (stream wait or in-kernel wait).
    [[nodiscard]] bool can_fold(std::size_t bytes) const noexcept {
        return fx_host_ != nullptr && bytes != 0 && bytes <= options_.fold_capacity && bytes % sizeof(float) == 0;
    }
    void fold_next(RowFold op) {
        if (op != RowFold::none && fx_host_ == nullptr) throw std::logic_error("xport: fold_next() without fold slots");
        fold_next_ = op;
    }
    [[nodiscard]] const void* last_folded() const noexcept { return last_folded_; }
    // CPU: waits until every posted exchange is complete, then check(). Between steps / requests only.
    void drain();
    // Throws with the cause if an exchange failed (the stream was released with garbage).
    void check() const;
    [[nodiscard]] Stats stats() const;
    [[nodiscard]] std::uint64_t sent() const noexcept { return next_; }
    [[nodiscard]] const Options& options() const noexcept { return options_; }
    [[nodiscard]] std::string name() const;

private:
    struct Job {
        std::uint64_t sequence = 0;
        std::size_t bytes = 0;
        Tag tag;
        bool seal = false;  // host-only 0-byte message carrying the last deferred CRC (drain())
        const std::byte* direct_tx = nullptr;
        std::byte* direct_rx = nullptr;
        RowFold fold = RowFold::none;
    };
    struct RxTrace { std::uint64_t sequence, kind, bytes, tx_seen_ns, wire_done_ns, rx_published_ns; };
    class CrcWorker;
    static constexpr std::size_t kQueue = 256;
    void init();
    void run();
    void process(const Job& job);
    void post(const Job& job);
    [[noreturn]] void fail(const std::string& why);
    std::byte* tx_slot(std::uint64_t sequence) const;
    std::byte* rx_slot(std::uint64_t sequence) const;

    std::unique_ptr<dist::Transport> owned_;
    std::vector<Ticket> tickets_; // tagged API only; legacy GLM send/recv has no extra work
    dist::Transport* transport_ = nullptr;
    Options options_;
    std::size_t stride_ = 0;
    std::byte* tx_host_ = nullptr;
    std::byte* rx_host_ = nullptr;
    std::byte* tx_dev_ = nullptr;
    std::byte* rx_dev_ = nullptr;
    void* tx_alloc_ = nullptr;
    void* rx_alloc_ = nullptr;
    std::uint64_t* flags_host_ = nullptr;  // [0] TX flag (GPU writes), [8] RX flag (CPU writes)
    std::uint64_t* flags_dev_ = nullptr;
    unsigned* counter_dev_ = nullptr;
    std::uint64_t* gate_dev_ = nullptr;
    std::uint64_t next_ = 0;       // sequences 1.. ; next_ = last sequence handed out by send()
    std::uint64_t received_ = 0;   // last sequence whose recv() was enqueued
    std::uint64_t tx_taken_ = 0;   // last sequence whose tx() was handed out
    std::uint64_t jobs_posted_ = 0;  // caller side job counter (sends + seals)
    bool tail_deferred_ = false;     // caller side: the last send's CRC is deferred and not yet sealed
    // progress thread, deferred CRCs
    bool prev_deferred_ = false;
    std::uint32_t prev_tx_crc_ = 0;
    std::uint64_t prev_rx_ticket_ = 0, prev_sequence_ = 0;
    std::byte seal_tx_[64 + 64] = {}, seal_rx_[64 + 64] = {};
    std::unique_ptr<CrcWorker> crc_worker_;
    const void* last_local_ = nullptr;
    // row fold: slots (host-coherent), caller-side request state
    std::byte* fx_host_ = nullptr;
    std::byte* fx_dev_ = nullptr;
    std::size_t fx_stride_ = 0;
    RowFold fold_next_ = RowFold::none;
    std::vector<std::uint8_t> folded_;  // per slot: the exchange in it was folded (caller only)
    const void* last_folded_ = nullptr;
    // <ENV>RX_TRACE=<path>: the progress thread records per exchange when it saw the producer, when the wire call
    // returned and when it published the RX flag (CLOCK_MONOTONIC ns); written to <path>.<rank> by the destructor.
    std::string rx_trace_path_;
    std::vector<RxTrace> rx_trace_;
    std::uint64_t folds_ = 0;  // progress thread only
    Job jobs_[kQueue];
    std::atomic<std::uint64_t> posted_{0}, done_{0};
    std::atomic<bool> stop_{false}, failed_{false}, sleeping_{false};
    std::mutex mutex_;
    std::condition_variable wake_;
    std::string failure_;
    mutable std::mutex failure_mutex_;
    Stats stats_;
    HostWaitMetric slot_wait_{"channel", "slot"}; // caller only
    HostWaitMetric peer_wait_{"channel", "peer"}; // progress thread only
    std::thread thread_;
};

// Greedy head split by vocabulary (rank 0 owns ids [0, vocab0), rank 1 the rest): one exchange of `rows` x 8 bytes.
// Each rank's candidate per row is (max, id) over its logits with the first maximum winning (lowest id: the host scan
// `if (l[i] > l[best]) best = i`), written into the TX slot by one kernel that publishes itself; after the exchange a
// pick kernel writes per row the peer-agreed token, rank 0 winning ties (lower ids), into `out_ids` (device memory),
// so the next step's embedding can be read on the GPU without the CPU. A row with any non-finite logit on either rank
// gives kNonFinite (the caller treats it as an error after the step). `logits` is rows x ld, the first vn of each row.
inline constexpr std::uint32_t kNonFinite = 0xFFFFFFFFU;
template<class Traits>
void argmax_exchange(Channel<Traits>& x, const float* logits, unsigned rows, unsigned vn, unsigned ld, std::uint32_t id0, std::uint64_t step,
                     std::uint32_t* out_ids);

}  // namespace strix::xport
