// Production ttft_runs compiled against deterministic host device/transport fakes.
#include "strix/repeat_digests.hpp"
#include <algorithm>
#include <chrono>
#include <span>
#include <stdexcept>

struct Args {
    std::uint32_t repeat = 2, ttft_warmup = 1;
    bool timing = true;
};
namespace strix::dist {
struct Communicator {
    struct Group { unsigned rank = 0; } value;
    std::vector<std::uint32_t> rounds;
    const Group& group() const { return value; }
};
}
namespace strix::deepseek41 {
struct PrefillResult {
    double seconds = 0, engram_seconds = 0, head_seconds = 0;
    std::uint32_t next_token = 7;
    std::vector<std::uint64_t> logits{11}, stream_hashes{22};
};
struct DevicePrefill {
    std::string drift;
    unsigned measured = 0, warmups = 0;
    PrefillResult run(std::span<const std::uint32_t> rows) {
        PrefillResult r;
        if (rows.size() == 1U) { // deliberately different warm-up, excluded from reference
            ++warmups;
            r.logits[0] = 99;
        } else if (++measured == 2U) {
            if (drift == "logits") ++r.logits[0];
            if (drift == "streams") ++r.stream_hashes[0];
            if (drift == "token") ++r.next_token;
        }
        return r;
    }
};
}
struct TtftClock {
    void line(const char*) const {}
    static double ms(std::chrono::steady_clock::duration d) {
        return std::chrono::duration<double, std::milli>(d).count();
    }
} ttft_clock;
std::uint64_t fnv(const std::vector<std::uint64_t>& values) { return values.at(0); }
void barrier(strix::dist::Communicator& comm, std::uint32_t round) { comm.rounds.push_back(round); }
void print_result(const char*, const strix::deepseek41::PrefillResult&, std::size_t, bool, const std::string&) {}
bool receipt_written = false;
void write_receipts(const Args&, const strix::deepseek41::DevicePrefill& pre,
                    const strix::deepseek41::PrefillResult& r, unsigned rank) {
    if (pre.measured != 2U || pre.warmups != 1U ||
        (pre.drift == "token" && r.next_token != 8U) ||
        (pre.drift == "logits" && r.logits[0] != 12U) ||
        (pre.drift == "streams" && r.stream_hashes[0] != 23U))
        throw std::runtime_error("receipt is not the last measured result");
    receipt_written = true;
    std::printf("receipt rank %u\n", rank);
}
#include "repeat_adapter.inc"
#include "ttft_runs.inc"

int main(int argc, char** argv) try {
    if (argc != 3) throw std::runtime_error("expected mode and drift");
    const std::string mode = argv[1];
    strix::dist::Communicator comm;
    comm.value.rank = mode == "pair1" ? 1U : 0U;
    strix::deepseek41::DevicePrefill pre{argv[2]};
    const Args args;
    const std::vector<std::uint32_t> prompt{1, 2, 3, 4};
    try {
        const auto r = ttft_runs(args, pre, prompt, mode == "single" ? nullptr : &comm, mode != "pair1", {});
        if (r.next_token != 7U) throw std::runtime_error("unexpected continuation token");
    } catch (const std::exception&) {
        if (!receipt_written) throw std::runtime_error("failed before preserving receipt");
        if (mode != "single" && comm.rounds != std::vector<std::uint32_t>{900, 0, 1})
            throw std::runtime_error("barrier sequence changed");
        throw;
    }
    if (!receipt_written) throw std::runtime_error("timing path omitted receipts");
    if (pre.warmups != 1U || pre.measured != 2U) throw std::runtime_error("wrong repetition count");
    if (mode != "single" && comm.rounds != std::vector<std::uint32_t>{900, 0, 1})
        throw std::runtime_error("barrier sequence changed");
    std::puts("continuation allowed");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
}
