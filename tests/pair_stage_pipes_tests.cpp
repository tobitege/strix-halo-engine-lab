// Interleaved virtual-stage seam (include/strix/pair/stage_map.hpp): StageMap, balance_map and StagePipes over
// loopback transports with random delays (no deadlock with two slots per direction, order and CRC kept, ACK =
// consumption), and wire v3 byte identity with a plain Pipe at v = 1.
#include "strix/pair/stage_map.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace {
static_assert(!std::is_move_constructible_v<strix::pair::StagePipes>, "receive validator requires a stable owner");
static_assert(!std::is_move_assignable_v<strix::pair::StagePipes>, "receive validator requires a stable owner");

void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
// One ordered byte stream (one direction of a loopback transport).
class Stream {
public:
    void send(std::span<const std::byte> data) {
        std::lock_guard lock(mutex_);
        bytes_.insert(bytes_.end(), data.begin(), data.end());
        if (record_) log_.insert(log_.end(), data.begin(), data.end());
        cv_.notify_all();
    }
    void receive(std::span<std::byte> data) {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] { return aborted_ || bytes_.size() >= data.size(); });
        if (aborted_) throw std::runtime_error("loopback aborted");
        std::copy_n(bytes_.begin(), data.size(), data.begin());
        bytes_.erase(bytes_.begin(), bytes_.begin() + static_cast<std::ptrdiff_t>(data.size()));
    }
    void abort() { std::lock_guard lock(mutex_); aborted_ = true; cv_.notify_all(); }
    void record() { record_ = true; }
    std::vector<std::byte> log() { std::lock_guard lock(mutex_); return log_; }
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::byte> bytes_;
    std::vector<std::byte> log_;
    bool aborted_ = false, record_ = false;
};
// A full-duplex channel: a -> b and b -> a.
struct Channel {
    Stream ab, ba;
    void abort() { ab.abort(); ba.abort(); }
    strix::pair::PipeLink end(unsigned side) {
        auto& out = side == 0U ? ab : ba;
        auto& in = side == 0U ? ba : ab;
        return {[&out](std::span<const std::byte> d) { out.send(d); }, [&in](std::span<std::byte> d) { in.receive(d); },
            [this] { abort(); }};
    }
};
std::byte pattern(std::uint64_t chunk, std::uint32_t boundary, std::size_t i) {
    return static_cast<std::byte>((chunk * 131U + boundary * 17U + i * 7U) & 0xffU);
}
std::size_t payload_bytes(std::uint64_t chunk, std::uint32_t boundary) { return 64U + chunk * 3U + boundary; }
strix::pair::PipeOptions options() {
    strix::pair::PipeOptions out;
    out.validate = [](const strix::pair::PipeHeader& header, std::uint64_t sequence) {
        if (header[0] != 0x5345414dULL || header[1] != sequence) throw std::runtime_error("engine header mismatch");
        return static_cast<std::size_t>(header[3]);
    };
    return out;
}
// Runs both ranks through the shared order; returns the last stage's chunk order on rank 1.
std::vector<std::uint64_t> run(const strix::pair::StageMap& map, std::uint64_t chunks, unsigned seed,
    Channel* forward_log = nullptr, std::size_t extra_bytes = 0) {
    Channel forward, reverse;
    Channel& fwd = forward_log ? *forward_log : forward;
    std::vector<std::uint64_t> finished;
    std::atomic<bool> failed{false};
    std::string error;
    std::mutex error_mutex;
    const auto size = [&](std::uint64_t chunk, std::uint32_t boundary) {
        return payload_bytes(chunk, boundary) + (boundary == 0U ? extra_bytes : 0U);
    };
    const auto rank_body = [&](unsigned rank) {
        try {
            std::mt19937 rng(seed * 2U + rank);
            strix::pair::StagePipes pipes(rank, map, chunks, fwd.end(rank), reverse.end(rank), options());
            for (const auto step : pipes.order()) {
                if (rng() % 4U == 0U) {  // short random delay (sleep_for rounds up to a scheduler tick on some hosts)
                    const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(rng() % 60U);
                    while (std::chrono::steady_clock::now() < until) std::this_thread::yield();
                }
                if (step.stage > 0U) {
                    const auto b = step.stage - 1U;
                    auto& in = pipes.acquire_in(step.chunk, b);
                    require(in.payload.size() == size(step.chunk, b), "payload size kept");
                    for (std::size_t i = 0; i < in.payload.size(); ++i)
                        require(in.payload[i] == pattern(step.chunk, b, i), "payload bytes kept");
                    pipes.release_in(step.chunk, b);
                }
                if (step.stage + 1U < map.stages()) {
                    auto& out = pipes.acquire_out(step.chunk, step.stage);
                    out.header[0] = 0x5345414dULL;
                    out.header[3] = size(step.chunk, step.stage);
                    out.resize(size(step.chunk, step.stage));
                    for (std::size_t i = 0; i < out.payload.size(); ++i) out.payload[i] = pattern(step.chunk, step.stage, i);
                    pipes.release_out(step.chunk, step.stage);
                } else {
                    finished.push_back(step.chunk);
                }
            }
            pipes.finish();
        } catch (const std::exception& e) {
            std::lock_guard lock(error_mutex);
            if (!failed.exchange(true)) error = e.what();
            fwd.abort(); reverse.abort();
        }
    };
    std::thread r0(rank_body, 0U), r1(rank_body, 1U);
    r0.join(); r1.join();
    if (failed) throw std::runtime_error("stage pipes run failed: " + error);
    return finished;
}
} // namespace

int main() try {
    using strix::pair::StageMap;
    using strix::pair::StageCut;
    {  // StageMap text / parse / ownership
        const auto map = StageMap::parse("0,8,15,23,30,38,45,53,60", 60U);
        require(map.stages() == 8U && map.virtual_stages() == 4U && map.text() == "0,8,15,23,30,38,45,53,60", "parse/text");
        require(map.owns(0U, 0U) && map.owns(1U, 8U) && map.owns(0U, 15U) && map.owns(1U, 59U), "ownership alternates");
        const auto single = StageMap::single(StageCut(22U, true), 48U);
        require(single.text() == "0,22.5,48" && single.stage_of(22U) == 0U && single.stage_of(23U) == 1U, "half cut keeps the cut layer's state on A");
        bool threw = false;
        try { (void)StageMap::parse("0,10,20,30", 30U); } catch (const std::invalid_argument&) { threw = true; }
        require(threw, "odd stage count rejected");
        threw = false;
        try { (void)StageMap::parse("0,10,20", 30U); } catch (const std::invalid_argument&) { threw = true; }
        require(threw, "wrong endpoint rejected");
        require(StageMap::parse("0,24,48", 48U).fnv() != StageMap::parse("0,23,48", 48U).fnv(), "map FNV differs");
    }
    {  // balance_map: uniform MiniMax-like 60 layers (v = 5) and a Qwen-like 48 layers with a head tail
        const std::vector<double> mm(60U, 81.0);
        require(strix::pair::balance_map(mm, 0.0, 5U, [](std::uint32_t) { return true; }).text() == "0,6,12,18,24,30,36,42,48,54,60",
            "uniform 60 layers in 10 stages");
        std::vector<double> qwen(48U, 50.0);
        for (std::uint32_t l = 3; l < 48U; l += 4U) qwen[l] = 80.0;  // every 4th layer full attention, rest GDN
        const auto q = strix::pair::balance_map(qwen, 120.0, 2U, [](std::uint32_t) { return true; });
        require(q.stages() == 4U && q.cuts.back().layer == 48U && q.cuts[3].layer > 36U, "head tail shortens the last stage");
        const auto odd = strix::pair::balance_map(qwen, 0.0, 2U, [](std::uint32_t l) { return l % 2U == 0U; });
        for (const auto& cut : odd.cuts) require(cut.layer % 2U == 0U, "legal cuts only");
    }
    {  // v = 1: frames stay wire v3, byte-identical to a plain Pipe
        const auto map = StageMap::single(StageCut(24U), 48U);
        Channel staged;
        staged.ab.record();
        (void)run(map, 3U, 7U, &staged);
        Channel plain;
        plain.ab.record();
        const auto rank_body = [&](unsigned rank) {
            strix::pair::Pipe pipe(rank, 3U, plain.end(rank).send, plain.end(rank).receive, [&] { plain.abort(); }, options());
            for (std::uint64_t c = 0; c < 3U; ++c) {
                auto& frame = pipe.acquire(c);
                if (rank == 0U) {
                    frame.header[0] = 0x5345414dULL; frame.header[1] = c; frame.header[41] = 3U;
                    frame.header[3] = payload_bytes(c, 0U);
                    frame.resize(payload_bytes(c, 0U));
                    for (std::size_t i = 0; i < frame.payload.size(); ++i) frame.payload[i] = pattern(c, 0U, i);
                }
                pipe.release(c);
            }
            pipe.finish(3U);
        };
        std::thread r0(rank_body, 0U), r1(rank_body, 1U);
        r0.join(); r1.join();
        require(staged.ab.log() == plain.ab.log(), "v=1 forward bytes identical to a plain pipe (wire v3)");
    }
    {  // At least 1,000 random schedules, including partial groups after a full group.
        const std::array<const char*, 4> maps{"0,24,48", "0,12,24,36,48", "0,8,16,24,32,40,48", "0,5,10,15,20,25,30,35,40,45,48"};
        const std::array<std::uint64_t, 6> chunk_counts{2U, 3U, 4U, 5U, 7U, 8U};
        unsigned runs = 0;
        for (unsigned seed = 0; runs < 1000U; ++seed)
            for (const auto* text : maps)
                for (const auto chunks : chunk_counts) {
                    const auto map = StageMap::parse(text, 48U);
                    const auto done = run(map, chunks, seed);
                    require(done.size() == chunks, "every chunk finished");
                    for (std::uint64_t c = 0; c < chunks; ++c) require(done[c] == c, "last stage in chunk order");
                    ++runs;
                }
        std::printf("stage pipes: %u schedules\n", runs);
    }
    {  // A full piece followed by a partial piece, with payload and CRC checks through both ranks.
        const auto done = run(StageMap::parse("0,12,24,36,48", 48U), 1U, 0U, nullptr, strix::pair::seam_piece_bytes);
        require(done == std::vector<std::uint64_t>{0U}, "multi-piece frame consumed");
    }
    for (const std::size_t word : {42U, 43U, 44U}) {  // Reject malformed v4 headers before engine validation.
        const auto map = StageMap::parse("0,12,24,36,48", 48U);
        Channel f, r;
        std::atomic<unsigned> engine_calls{0U};
        auto opts = options();
        opts.validate = [&](const strix::pair::PipeHeader&, std::uint64_t) -> std::size_t {
            ++engine_calls;
            throw std::runtime_error("engine validator reached");
        };
        strix::pair::StagePipes pipes(1U, map, 1U, f.end(1U), r.end(1U), opts);
        strix::pair::PipeHeader header{};
        header[42] = 4U;
        header[44] = map.fnv();
        header[word] ^= 1U;
        f.ab.send(std::as_bytes(std::span(header)));
        bool threw = false;
        try { (void)pipes.acquire_in(0U, 0U); } catch (const std::runtime_error&) { threw = true; }
        require(threw && engine_calls == 0U, "malformed v4 header rejected before engine validation");
    }
    {  // out-of-order use is refused
        const auto map = StageMap::parse("0,12,24,36,48", 48U);
        Channel f, r;
        strix::pair::StagePipes pipes(0U, map, 4U, f.end(0U), r.end(0U), options());
        bool threw = false;
        try { (void)pipes.acquire_out(1U, 0U); } catch (const std::logic_error&) { threw = true; }
        require(threw && pipes.sequence(1U, 2U) == 5U && pipes.count(0U) == 8U && pipes.count(1U) == 4U, "order enforced");
        f.abort(); r.abort();
    }
    std::puts("PASS: pair interleaved stage pipes");
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
