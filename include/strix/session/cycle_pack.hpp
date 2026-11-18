#pragma once
// One completion per serve cycle (STRIX_SERVE_CYCLE_PACK).
//
// A batched decode cycle ends with small device records per member (status words, first-error records, union
// summaries). Read one by one, every record costs a copy, a host wait and a wake-up, member after member. A
// CyclePack collects the records of one cycle: add() reserves a slot per record, gather() copies all of them into
// one mapped host block with ONE kernel launch, wait() fences the stream ONCE, and the members then check their
// records from host memory. Scheduling only: the same records reach the same host checks, so tokens are unchanged.
//
// Shared by every batch runner (Qwen cohort window first). The header is host-only (streams are void*), so engines,
// servers and host tests include it without HIP.

#include "strix/lever.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strix::session {

inline constexpr lever::Decl serve_cycle_pack_lever{
    "serve-cycle-pack", "STRIX_SERVE_CYCLE_PACK", lever::Exactness::bitwise, false, {}, {},
    "one packed completion per batched decode cycle: same records, same checks, one wait"};

// Standalone experimental declaration: not a valid production lever table row yet.
// No regime inputs: resolve through envcfg once; changing the switch requires restart.
[[nodiscard]] inline lever::Choice serve_cycle_pack_choice(std::uint64_t) {
    static_assert(serve_cycle_pack_lever.regime.axis == lever::Axis::none);
    static const auto choice = lever::select(serve_cycle_pack_lever, lever::Inputs{});
    return choice;
}

// A fenced status is valid only until the owner's next submission.
class CyclePackProof {
public:
    void invalidate() noexcept { ready_ = false; }
    [[nodiscard]] bool ready() const noexcept { return ready_; }
    void publish(std::array<std::uint32_t, 2> status) noexcept { status_ = status; ready_ = true; }
    bool consume(std::array<std::uint32_t, 2>& status) noexcept {
        if (!ready_) return false;
        status = status_; invalidate(); return true;
    }
private:
    bool ready_ = false;
    std::array<std::uint32_t, 2> status_{};
};

// Retain cleanup ownership until each settlement succeeds or fails its owner.
// Fail must be noexcept. A pack/finish exception also fails all still-taken members.
template <class Fail> class CyclePackSettlement {
public:
    CyclePackSettlement(std::size_t members, Fail fail) : take_(members, false), fail_(std::move(fail)) {}
    CyclePackSettlement(const CyclePackSettlement&) = delete;
    CyclePackSettlement& operator=(const CyclePackSettlement&) = delete;
    ~CyclePackSettlement() { for (std::size_t k = 0; k < take_.size(); ++k) if (take_[k]) fail_(k); }
    void take(std::size_t k) { take_[k] = true; }
    const std::vector<bool>& selected() const noexcept { return take_; }
    template <class Settle> void settle(Settle&& settle_one) {
        std::exception_ptr error;
        for (std::size_t k = 0; k < take_.size(); ++k) if (take_[k]) {
            try { settle_one(k); }
            catch (...) { if (!error) error = std::current_exception(); fail_(k); }
            take_[k] = false;
        }
        if (error) std::rethrow_exception(error);
    }
private:
    std::vector<bool> take_;
    Fail fail_;
};

// Word layout of one cycle's records: each record is a 4-byte multiple placed at the next free word.
struct CyclePackLayout {
    static constexpr std::uint32_t max_records = 64;
    static constexpr std::uint32_t max_words = 2048;  // 8 KiB of records per cycle
    std::uint32_t records = 0;
    std::uint32_t words = 0;
    std::array<std::uint32_t, max_records> offset{};
    std::array<std::uint32_t, max_records> size{};

    void reset() noexcept { records = 0; words = 0; }
    // Capacity-only planning, before any owner completion state is changed.
    bool try_add(std::size_t bytes, std::size_t count = 1) {
        if (bytes == 0 || bytes % 4U != 0 || count > max_records - records) return false;
        if (count == 0) return true;
        if (bytes / 4U > (max_words - words) / count) return false;
        for (std::size_t k = 0; k < count; ++k) add(bytes);
        return true;
    }
    // Returns the record's first word.
    std::uint32_t add(std::size_t bytes) {
        if (bytes == 0 || bytes % 4U != 0) throw std::invalid_argument("cycle pack record must be a non-empty 4-byte multiple");
        if (records == max_records) throw std::length_error("cycle pack holds at most 64 records per cycle");
        if (bytes / 4U > max_words - words) throw std::length_error("cycle pack holds at most 8 KiB of records per cycle");
        const auto n = static_cast<std::uint32_t>(bytes / 4U);
        offset[records] = words;
        size[records] = n;
        ++records;
        words += n;
        return words - n;
    }
};

// Device side (src/session/cycle_pack.hip). One pack per runner: records of one cycle at a time.
class CyclePack {
public:
    CyclePack();
    ~CyclePack();
    CyclePack(const CyclePack&) = delete;
    CyclePack& operator=(const CyclePack&) = delete;

    // Starts a cycle: drops the previous cycle's records.
    void begin() noexcept;
    // Reserves a slot for `bytes` (4-byte multiple) read from `device_source`; returns the host address the record
    // lands at. The slot is poisoned (0xA5A5A5A5 words) until gather() overwrites it and is valid after wait().
    [[nodiscard]] const void* add(const void* device_source, std::size_t bytes);
    // Copies every record of the cycle into the mapped host block: one kernel on `stream`.
    void gather(void* stream);
    // The single host wait of the cycle: synchronises `stream` (which must carry gather()).
    void wait(void* stream);
    [[nodiscard]] std::uint32_t records() const noexcept;
    // Counters for receipts: cycles packed and records gathered since creation.
    [[nodiscard]] std::uint64_t cycles() const noexcept;
    [[nodiscard]] std::uint64_t gathered() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::session
