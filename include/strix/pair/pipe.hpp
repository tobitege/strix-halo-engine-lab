#pragma once
// Two-stage pipeline-parallel prefill seam shared by every engine (Qwen, Gemma 4, ...): a two-slot
// producer/consumer pipe over one ordered byte transport (strix::dist::Transport send/receive), with
// pinned staging, per-piece and whole-frame CRC32C and a consumption ACK per chunk. Engines supply only
// the seam header validation (their own geometry and identity words) and the device staging.
// Frame header words owned here: [1] sequence (producer), [40] whole CRC, [41] chunk count (producer),
// [42] wire version 3. All other words belong to the engine's validator.
#include <array>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "strix/dist/pipeline_timeline.hpp"

namespace strix::pair {
inline constexpr std::size_t seam_piece_bytes = 16U * 1024U * 1024U;
// Where a two-stage layer pipeline is cut. Stage A runs layers [0, layer) and, for a half-layer cut, the first
// sub-block of `layer` (attention before its feed-forward half); stage B runs the rest. A half-layer cut lets an
// engine whose layers have two residual sub-blocks balance the stages to half a layer. Text form: "23", "22.5".
struct StageCut {
    std::uint32_t layer = 0;
    bool half = false;
    constexpr StageCut() = default;
    constexpr StageCut(std::uint32_t whole, bool with_half = false) : layer(whole), half(with_half) {}
    // First layer whose first sub-block (and so its sequence state) belongs to stage B.
    constexpr std::uint32_t state_begin() const noexcept { return layer + (half ? 1U : 0U); }
    // Position in half-layer units; orders cuts and range ends.
    constexpr std::uint64_t units() const noexcept { return 2ULL * layer + (half ? 1U : 0U); }
    // Seam header word: a whole-layer cut keeps the plain layer number.
    constexpr std::uint64_t wire() const noexcept { return layer | (half ? 1ULL << 32U : 0ULL); }
    friend constexpr bool operator==(const StageCut&, const StageCut&) = default;
    std::string text() const { return std::to_string(layer) + (half ? ".5" : ""); }
    // "N" or "N.5"; anything else is rejected.
    static StageCut parse(std::string_view text) {
        StageCut cut;
        const auto* const end = text.data() + text.size();
        const auto parsed = std::from_chars(text.data(), end, cut.layer);
        if (parsed.ec != std::errc{}) throw std::invalid_argument("stage cut requires a layer number");
        if (parsed.ptr != end) {
            if (std::string_view(parsed.ptr, static_cast<std::size_t>(end - parsed.ptr)) != ".5")
                throw std::invalid_argument("stage cut requires a layer number or a half layer (N.5)");
            cut.half = true;
        }
        return cut;
    }
};
class SeamCrc {
public:
    void update(std::span<const std::byte> bytes) noexcept {
        for (const auto value : bytes) crc_ = table_[(crc_ ^ std::to_integer<unsigned>(value)) & 255U] ^ (crc_ >> 8U);
    }
    std::uint32_t value() const noexcept { return ~crc_; }
    // Concatenate finalized CRCs over GF(2); do not scan each piece twice.
    void append(std::uint32_t piece_crc, std::size_t bytes) noexcept {
        if (bytes == 0U) return;
        std::array<std::uint32_t, 32> odd{}, even{};
        odd[0] = 0x82f63b78U;
        std::uint32_t row = 1;
        for (std::size_t index = 1; index < odd.size(); ++index) { odd[index] = row; row <<= 1U; }
        square(even, odd); square(odd, even);
        auto combined = value();
        do {
            square(even, odd);
            if ((bytes & 1U) != 0U) combined = times(even, combined);
            bytes >>= 1U;
            if (bytes == 0U) break;
            square(odd, even);
            if ((bytes & 1U) != 0U) combined = times(odd, combined);
            bytes >>= 1U;
        } while (bytes != 0U);
        crc_ = ~(combined ^ piece_crc);
    }
private:
    std::uint32_t crc_ = 0xffffffffU;
    static std::uint32_t times(const std::array<std::uint32_t, 32>& matrix, std::uint32_t vector) noexcept {
        std::uint32_t result = 0;
        std::size_t index = 0;
        while (vector != 0U) { if ((vector & 1U) != 0U) result ^= matrix[index]; vector >>= 1U; ++index; }
        return result;
    }
    static void square(std::array<std::uint32_t, 32>& result, const std::array<std::uint32_t, 32>& matrix) noexcept {
        for (std::size_t index = 0; index < result.size(); ++index) result[index] = times(matrix, matrix[index]);
    }
    static constexpr auto table_ = [] {
        std::array<std::uint32_t, 256> values{};
        for (std::uint32_t index = 0; index < 256U; ++index) {
            auto value = index;
            for (unsigned bit = 0; bit < 8U; ++bit) value = (value >> 1U) ^ ((value & 1U) ? 0x82f63b78U : 0U);
            values[index] = value;
        }
        return values;
    }();
};
// A slot owns staging until consumption ACK. Native implementations use pinned host
// memory and a separate device snapshot, never the next chunk's mutable arena.
struct PipeStaging {
    virtual ~PipeStaging() = default;
    virtual std::span<std::byte> resize(std::size_t bytes) = 0;
    virtual void copy(bool save, std::size_t offset, void* device, std::size_t bytes) = 0;
    // `rows` equal-length pieces at a constant pitch on both sides (row r: offset + r * pitch <-> device + r * device_pitch).
    // Same bytes as `rows` copy() calls; native staging issues one strided copy instead of one call per row.
    virtual void copy_rows(bool save, std::size_t offset, std::size_t pitch, void* device, std::size_t device_pitch,
        std::size_t bytes, std::size_t rows) {
        for (std::size_t row = 0; row < rows; ++row)
            copy(save, offset + row * pitch, static_cast<std::byte*>(device) + row * device_pitch, bytes);
    }
    virtual void seal(std::size_t bytes) = 0;
    virtual void await_piece(std::size_t offset) = 0;
    virtual void received_piece(std::size_t offset, std::size_t bytes) = 0;
    virtual void finish_receive() = 0;
    // Optional immutable device snapshot for a routed byte transfer. Host-only
    // staging keeps the original send/receive path.
    virtual void* device_piece(std::size_t) { return nullptr; }
    virtual void read_device_piece(std::size_t, std::size_t) {}
    // Device-checked pieces. route_device() installs the engine's size rule (the pieces its transport moves device
    // to device); a staging that can compute the seam CRC-32C on the device then answers device_routed() for them,
    // never copies such a piece to the host, and returns its CRC from device_crc() (received: the bytes just landed
    // in the device snapshot; otherwise the sealed snapshot). The value is the host CRC-32C of the same bytes, so
    // the wire is unchanged. Only for a pipe whose checksum is CRC-32C (the default one and dist::crc32c).
    virtual void route_device(std::function<bool(std::size_t bytes)>) {}
    virtual bool device_routed(std::size_t, std::size_t) { return false; }
    virtual std::uint32_t device_crc(std::size_t, std::size_t, bool) { throw std::logic_error("pair staging has no device CRC"); }
};
std::shared_ptr<PipeStaging> make_hip_pipe_staging(std::size_t capacity);
using PipeHeader = std::array<std::uint64_t, 48>;
struct PipeFrame {
    // Native little-endian, identical-build wire v3: header, CRC+payload pieces, whole CRC.
    PipeHeader header{};
    std::vector<std::byte> bytes;
    std::shared_ptr<PipeStaging> staging;
    std::span<std::byte> payload;
    void resize(std::size_t size) {
        if (staging) payload = staging->resize(size);
        else { bytes.resize(size); payload = bytes; }
    }
};
struct PipeOptions {
    // Rank 1: validates a received header (sequence, version, count and the engine's words) and returns the
    // payload bytes to receive. Throws on mismatch; nothing is installed before it passes.
    std::function<std::size_t(const PipeHeader&, std::uint64_t sequence)> validate;
    std::string profile_label = "pair-profile"; // stderr prefix of the per-chunk transfer profile line
    bool profile = false;
    // Optional STRIX_PAIR_TIMING spans: the engine owns the timeline and its column names and maps the
    // pipe's own waits onto its columns (step = chunk sequence).
    dist::PipelineTimeline* timeline = nullptr;
    struct TimingColumns {
        std::size_t d2h_wait = 0, crc = 0, wire = 0, ack_wait = 0, header_wait = 0, h2d_wait = 0;
    } timing;
};
class Pipe {
public:
    using Send = std::function<void(std::span<const std::byte>)>;
    using Receive = std::function<void(std::span<std::byte>)>;
    using Checksum = std::function<std::uint32_t(std::span<const std::byte>)>;
    // Optional size router: moves one staged piece itself (true) or leaves it to send/receive (false).
    using TransferPiece = std::function<bool(bool, PipeStaging&, std::size_t, std::size_t)>;
    Pipe(unsigned rank, std::uint64_t count, Send send, Receive receive, std::function<void()> abort,
        PipeOptions options, std::function<std::shared_ptr<PipeStaging>()> staging = {}, Checksum checksum = {},
        TransferPiece transfer_piece = {})
        : rank_(rank), count_(count), send_(std::move(send)), receive_(std::move(receive)), abort_(std::move(abort)),
          options_(std::move(options)), checksum_(std::move(checksum)), transfer_piece_(std::move(transfer_piece)),
          timeline_(options_.timeline && options_.timeline->enabled() ? options_.timeline : nullptr) {
        if (rank_ > 1U || count_ == 0U) throw std::invalid_argument("pair prefill rank must be 0 or 1");
        if (!options_.validate) throw std::invalid_argument("pair pipe requires a seam header validator");
        if (staging) for (auto& frame : frames_) frame.staging = staging();
        worker_ = std::thread([this] { progress(); });
    }
    ~Pipe() { stop(); }
    Pipe(const Pipe&) = delete;
    Pipe& operator=(const Pipe&) = delete;
    PipeFrame& acquire(std::uint64_t sequence) {
        std::unique_lock lock(mutex_);
        const auto slot = static_cast<std::size_t>(sequence % 2U);
        cv_.wait(lock, [&] { return error_ || states_[slot] == (rank_ == 0U ? free : ready); });
        rethrow();
        states_[slot] = busy;
        return frames_[slot];
    }
    void release(std::uint64_t sequence) {
        std::lock_guard lock(mutex_);
        states_[static_cast<std::size_t>(sequence % 2U)] = rank_ == 0U ? ready : consumed;
        cv_.notify_all();
    }
    std::uint64_t count() const noexcept { return count_; }
    void finish(std::uint64_t count) {
        if (count != count_) throw std::invalid_argument("pair prefill chunk count mismatch");
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] { return error_ || completed_ == count; });
        rethrow();
        // Every chunk is acknowledged: let the worker exit without aborting the transport, so the same
        // transport can still carry the pair handoff state handoff after the prefill.
        stopping_ = true; cv_.notify_all();
        lock.unlock();
        if (worker_.joinable()) worker_.join();
    }
private:
    enum State { free, ready, busy, consumed };
    unsigned rank_;
    std::uint64_t count_;
    Send send_;
    Receive receive_;
    std::function<void()> abort_;
    std::array<PipeFrame, 2> frames_;
    std::array<State, 2> states_{free, free};
    std::mutex mutex_;
    std::condition_variable cv_;
    std::exception_ptr error_;
    bool stopping_ = false;
    std::uint64_t completed_ = 0;
    std::thread worker_;
    PipeOptions options_;
    Checksum checksum_;
    TransferPiece transfer_piece_;
    dist::PipelineTimeline* timeline_ = nullptr;
    template<class F> decltype(auto) timed(std::uint64_t step, std::size_t column, F&& operation) {
        if (timeline_) return timeline_->measure(step, column, std::forward<F>(operation));
        return operation();
    }
    std::uint32_t piece_crc(std::span<const std::byte> bytes) const {
        if (checksum_) return checksum_(bytes);
        SeamCrc checksum; checksum.update(bytes); return checksum.value();
    }
    void rethrow() { if (error_) std::rethrow_exception(error_); }
    void stop() {
        { std::lock_guard lock(mutex_); if (stopping_) return; stopping_ = true; cv_.notify_all(); }
        abort_();
        if (worker_.joinable()) worker_.join();
    }
    void acknowledge(std::uint64_t sequence) {
        const auto slot = static_cast<std::size_t>(sequence % 2U);
        const std::array<std::uint64_t, 2> ack{0x50503241434bULL, sequence};
        if (rank_ == 0U) {
            std::array<std::uint64_t, 2> reply{};
            timed(sequence, options_.timing.ack_wait, [&] { receive_(std::as_writable_bytes(std::span(reply))); });
            if (reply != ack) throw std::runtime_error("pair prefill consumption acknowledgement mismatch");
        } else {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [&] { return stopping_ || states_[slot] == consumed; });
            if (stopping_) throw std::runtime_error("pair prefill stopped");
            lock.unlock();
            timed(sequence, options_.timing.wire, [&] { send_(std::as_bytes(std::span(ack))); });
        }
        std::lock_guard lock(mutex_);
        states_[slot] = free; ++completed_; cv_.notify_all();
    }
    void transfer(PipeFrame& frame, std::uint64_t sequence) {
        const auto& column = options_.timing;
        SeamCrc whole;
        for (std::size_t offset = 0; offset < frame.payload.size(); offset += seam_piece_bytes) {
            auto piece = frame.payload.subspan(offset, std::min(seam_piece_bytes, frame.payload.size() - offset));
            std::uint32_t crc = 0;
            // A device-checked piece never visits the host: its CRC comes from the device snapshot.
            const bool device = frame.staging && transfer_piece_ && frame.staging->device_routed(offset, piece.size());
            if (rank_ == 0U) {
                if (device) crc = timed(sequence, column.crc, [&] { return frame.staging->device_crc(offset, piece.size(), false); });
                else {
                    if (frame.staging) timed(sequence, column.d2h_wait, [&] { frame.staging->await_piece(offset); });
                    crc = timed(sequence, column.crc, [&] { return piece_crc(piece); });
                }
                whole.append(crc, piece.size());
                timed(sequence, column.wire, [&] {
                    send_(std::as_bytes(std::span(&crc, 1)));
                    const bool moved = frame.staging && transfer_piece_ && transfer_piece_(true, *frame.staging, offset, piece.size());
                    if (device && !moved) throw std::logic_error("pair prefill device-checked piece was not moved by the transport");
                    if (!moved) send_(piece);
                });
            } else {
                const bool routed = timed(sequence, column.wire, [&] {
                    receive_(std::as_writable_bytes(std::span(&crc, 1)));
                    const bool routed_piece = frame.staging && transfer_piece_ && transfer_piece_(false, *frame.staging, offset, piece.size());
                    if (!routed_piece) receive_(piece);
                    return routed_piece;
                });
                if (device && !routed) throw std::logic_error("pair prefill device-checked piece was not moved by the transport");
                if (crc != timed(sequence, column.crc, [&] {
                        return device ? frame.staging->device_crc(offset, piece.size(), true) : piece_crc(piece); }))
                    throw std::runtime_error("pair prefill seam piece CRC mismatch");
                whole.append(crc, piece.size());
                if (frame.staging && !routed)
                    timed(sequence, column.h2d_wait, [&] { frame.staging->received_piece(offset, piece.size()); });
            }
        }
        auto crc = whole.value();
        if (rank_ == 0U) timed(sequence, column.wire, [&] { send_(std::as_bytes(std::span(&crc, 1))); });
        else {
            std::uint32_t expected = 0;
            timed(sequence, column.wire, [&] { receive_(std::as_writable_bytes(std::span(&expected, 1))); });
            if (crc != expected) throw std::runtime_error("pair prefill seam whole CRC mismatch");
            if (frame.staging) timed(sequence, column.h2d_wait, [&] { frame.staging->finish_receive(); });
        }
        frame.header[40] = crc;
    }
    void progress() {
        try {
            for (std::uint64_t sequence = 0; sequence < count_; ++sequence) {
                const auto slot = static_cast<std::size_t>(sequence % 2U);
                auto& frame = frames_[slot];
                {
                    std::unique_lock lock(mutex_);
                    cv_.wait(lock, [&] { return stopping_ || states_[slot] == (rank_ == 0U ? ready : free); });
                    if (stopping_) return;
                }
                auto transfer_start = options_.profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                if (rank_ == 0U) {
                    if (!frame.staging) frame.payload = frame.bytes;
                    frame.header[42] = 3U;
                    timed(sequence, options_.timing.wire, [&] { send_(std::as_bytes(std::span(frame.header))); });
                } else {
                    timed(sequence, options_.timing.header_wait, [&] { receive_(std::as_writable_bytes(std::span(frame.header))); });
                    frame.resize(options_.validate(frame.header, sequence));
                }
                // The receiver excludes waiting for the next header (producer compute),
                // but includes payload, CRC and authenticated H2D completion.
                if (rank_ == 1U && options_.profile) transfer_start = std::chrono::steady_clock::now();
                transfer(frame, sequence);
                if (options_.profile)
                    std::fprintf(stderr, "%s {\"kind\":\"transfer\",\"rank\":%u,\"chunk\":%llu,\"rows\":%llu,\"ms\":%.9f}\n",
                        options_.profile_label.c_str(), rank_, static_cast<unsigned long long>(sequence),
                        static_cast<unsigned long long>(frame.header[3]), // [3]: rows, by convention of every engine
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - transfer_start).count());
                if (rank_ == 1U) { std::lock_guard lock(mutex_); states_[slot] = ready; cv_.notify_all(); }
                // Receive/stage the lookahead before waiting for B(sequence-1).
                // ACK still means consumption; neither slot can be reused early.
                if (sequence != 0U) acknowledge(sequence - 1U);
            }
            acknowledge(count_ - 1U);
        } catch (...) {
            { std::lock_guard lock(mutex_); if (!stopping_) error_ = std::current_exception(); cv_.notify_all(); }
            abort_();
        }
    }

};
} // namespace strix::pair
