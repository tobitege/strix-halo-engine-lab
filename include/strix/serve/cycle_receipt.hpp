// Per-cycle serving receipt and its ledger: host only, model agnostic, no device types.
//
// A scheduler fills one CycleReceipt per decode cycle (one cohort forward plus its draft and host work) and
// hands it to a CycleLedger at cycle end. The ledger keeps totals, a per-lane-count table of the cycles in
// which no prompt unit ran (the steady-state window of a load test), the sums of a least-squares cost model
//     total_ms = fixed_ms + row_ms * physical_rows + lane_ms * lanes
// and a fixed ring of the most recent receipts. Nothing is allocated per cycle and no kernel is fenced: the
// receipt is built from numbers the cycle already has.
//
// Switch: STRIX_SERVE_RECEIPT=1 (cycle_receipt_enabled()). Unset, engines do not build receipts and their
// outputs and timing are unchanged. Engines only adapt: they fill the descriptor; the arithmetic is here.
#pragma once

#include "strix/env_flags.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace strix::serve {

inline constexpr std::size_t kCycleDraftDepths = 8;
// by_lanes rows of the ledger: lane counts above this share the last row.
inline constexpr std::size_t kCycleLaneBuckets = 64;

struct CycleReceipt {
    // lanes: requests that decoded in this cycle. physical_rows: verify rows the forward ran;
    // useful_rows: rows that carried a real candidate position; padding_rows: the rest.
    // emitted: tokens committed to the lanes by this cycle.
    std::uint32_t lanes = 0, physical_rows = 0, useful_rows = 0, padding_rows = 0, emitted = 0;
    // draft_rows_by_depth[d]: lanes whose draft reached depth d + 1 (depths past the array share the last
    // entry), so acceptance by depth comes out of the same receipt.
    std::array<std::uint32_t, kCycleDraftDepths> draft_rows_by_depth{};
    // total_ms is the wall time of the decode cycle only; a prompt unit of the same scheduler step is not in it.
    double draft_ms = 0.0, verify_ms = 0.0, host_ms = 0.0, total_ms = 0.0;
    // A prompt unit ran in the same scheduler step: the cycle is outside the steady-state window.
    bool prompt_unit_ran = false;
    // Optional monotonic clock stamps of the cycle (0 when the engine does not record them).
    std::int64_t begin_ns = 0, end_ns = 0;

    [[nodiscard]] bool consistent() const noexcept {
        return static_cast<std::uint64_t>(useful_rows) + padding_rows == physical_rows && std::isfinite(total_ms) &&
            std::isfinite(draft_ms) && std::isfinite(verify_ms) && std::isfinite(host_ms) && total_ms >= 0.0;
    }
    // Sets the three row fields from the physical and useful counts (useful is clamped to physical).
    void set_rows(std::uint32_t physical, std::uint32_t useful) noexcept {
        physical_rows = physical;
        useful_rows = std::min(physical, useful);
        padding_rows = physical - useful_rows;
    }
    void add_draft_depth(std::size_t depth, std::uint32_t rows = 1) noexcept {
        if (depth == 0) return;
        draft_rows_by_depth[std::min(depth, kCycleDraftDepths) - 1] += rows;
    }
};

[[nodiscard]] inline bool cycle_receipt_enabled() noexcept {
    static const bool on = env_flag_default_off("STRIX_SERVE_RECEIPT");
    return on;
}

// One log line per cycle; tools/serve_bench.py reads it back ("serve cycle key=value ...").
[[nodiscard]] inline std::string format_cycle_receipt(const CycleReceipt& r, std::uint64_t sequence = 0) {
    std::ostringstream o;
    o << "serve cycle seq=" << sequence << " lanes=" << r.lanes << " rows=" << r.physical_rows << " useful="
      << r.useful_rows << " pad=" << r.padding_rows << " emitted=" << r.emitted << " draft_rows=";
    for (std::size_t i = 0; i < kCycleDraftDepths; ++i) o << (i ? "/" : "") << r.draft_rows_by_depth[i];
    o << std::fixed << std::setprecision(6) << " draft_ms=" << r.draft_ms << " verify_ms=" << r.verify_ms
      << " host_ms=" << r.host_ms << " ms=" << r.total_ms << " prompt=" << (r.prompt_unit_ran ? 1 : 0)
      << " begin_ns=" << r.begin_ns << " end_ns=" << r.end_ns;
    return o.str();
}

class CycleLedger {
public:
    struct Fit {
        // total_ms = fixed_ms + row_ms * physical_rows + lane_ms * lanes over the observed cycles.
        double fixed_ms = 0.0, row_ms = 0.0, lane_ms = 0.0, residual_sd = 0.0;
        std::uint64_t cycles = 0;
        // Coefficients the data identify: 3 = all; 2 = rows and lanes moved together (or one of them never
        // moved), so the slope is reported on the one that moved, rows first, and the other is 0; 1 = mean only.
        std::uint32_t terms = 0;
    };
    struct Totals {
        std::uint64_t cycles = 0, prompt_cycles = 0, lanes = 0, physical_rows = 0, useful_rows = 0, padding_rows = 0,
            emitted = 0;
        std::array<std::uint64_t, kCycleDraftDepths> draft_rows_by_depth{};
        double draft_ms = 0.0, verify_ms = 0.0, host_ms = 0.0, total_ms = 0.0;
    };
    // Cycles with this many lanes in which no prompt unit ran.
    struct LaneBucket {
        std::uint64_t cycles = 0, physical_rows = 0, useful_rows = 0, emitted = 0;
        double total_ms = 0.0;
    };
    // Raw sums of the fit, so two snapshots can be subtracted and refitted (a load test's own window).
    struct FitSums {
        double n = 0.0, r = 0.0, l = 0.0, rr = 0.0, rl = 0.0, ll = 0.0, y = 0.0, ry = 0.0, ly = 0.0, yy = 0.0;
    };

    explicit CycleLedger(std::size_t ring_capacity = 4096) : ring_(std::max<std::size_t>(ring_capacity, 1)) {}

    // False (and counted as rejected) when useful + padding != physical or a time is not finite.
    bool add(const CycleReceipt& r) {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (!r.consistent()) {
            ++rejected_;
            return false;
        }
        ring_[static_cast<std::size_t>(sequence_ % ring_.size())] = r;
        ++sequence_;
        ++totals_.cycles;
        totals_.prompt_cycles += r.prompt_unit_ran ? 1U : 0U;
        totals_.lanes += r.lanes;
        totals_.physical_rows += r.physical_rows;
        totals_.useful_rows += r.useful_rows;
        totals_.padding_rows += r.padding_rows;
        totals_.emitted += r.emitted;
        for (std::size_t i = 0; i < kCycleDraftDepths; ++i) totals_.draft_rows_by_depth[i] += r.draft_rows_by_depth[i];
        totals_.draft_ms += r.draft_ms;
        totals_.verify_ms += r.verify_ms;
        totals_.host_ms += r.host_ms;
        totals_.total_ms += r.total_ms;
        if (!r.prompt_unit_ran) {
            auto& b = by_lanes_[std::min<std::size_t>(r.lanes, kCycleLaneBuckets)];
            ++b.cycles;
            b.physical_rows += r.physical_rows;
            b.useful_rows += r.useful_rows;
            b.emitted += r.emitted;
            b.total_ms += r.total_ms;
        }
        if (r.lanes != 0) {
            const double rows = static_cast<double>(r.physical_rows), lanes = static_cast<double>(r.lanes);
            sums_.n += 1.0;
            sums_.r += rows;
            sums_.l += lanes;
            sums_.rr += rows * rows;
            sums_.rl += rows * lanes;
            sums_.ll += lanes * lanes;
            sums_.y += r.total_ms;
            sums_.ry += rows * r.total_ms;
            sums_.ly += lanes * r.total_ms;
            sums_.yy += r.total_ms * r.total_ms;
        }
        return true;
    }

    [[nodiscard]] static Fit fit(const FitSums& s) noexcept {
        Fit f;
        if (!(s.n >= 1.0)) return f;
        f.cycles = static_cast<std::uint64_t>(s.n);
        const double mr = s.r / s.n, ml = s.l / s.n, my = s.y / s.n;
        const double srr = std::max(0.0, s.rr - s.n * mr * mr), sll = std::max(0.0, s.ll - s.n * ml * ml);
        const double srl = s.rl - s.n * mr * ml, sry = s.ry - s.n * mr * my, sly = s.ly - s.n * ml * my;
        const double syy = std::max(0.0, s.yy - s.n * my * my);
        // A regressor "moved" when its variance is not rounding noise of its own square sum.
        const bool rows_moved = srr > 1e-9 * std::max(1.0, s.rr), lanes_moved = sll > 1e-9 * std::max(1.0, s.ll);
        const double det = srr * sll - srl * srl;
        if (rows_moved && lanes_moved && det > 1e-9 * srr * sll) {
            f.row_ms = (sry * sll - sly * srl) / det;
            f.lane_ms = (sly * srr - sry * srl) / det;
            f.terms = 3;
        } else if (rows_moved) {
            f.row_ms = sry / srr;
            f.terms = 2;
        } else if (lanes_moved) {
            f.lane_ms = sly / sll;
            f.terms = 2;
        } else {
            f.terms = 1;
        }
        f.fixed_ms = my - f.row_ms * mr - f.lane_ms * ml;
        const double sse = std::max(0.0, syy - f.row_ms * sry - f.lane_ms * sly);
        f.residual_sd = s.n > static_cast<double>(f.terms) ? std::sqrt(sse / (s.n - static_cast<double>(f.terms))) : 0.0;
        return f;
    }
    // Least squares over the observed (rows, lanes) of every cycle with at least one lane.
    [[nodiscard]] Fit fit() const {
        const std::lock_guard<std::mutex> lock(mutex_);
        return fit(sums_);
    }

    [[nodiscard]] Totals totals() const {
        const std::lock_guard<std::mutex> lock(mutex_);
        return totals_;
    }
    [[nodiscard]] FitSums fit_sums() const {
        const std::lock_guard<std::mutex> lock(mutex_);
        return sums_;
    }
    [[nodiscard]] LaneBucket lanes(std::size_t count) const {
        const std::lock_guard<std::mutex> lock(mutex_);
        return by_lanes_[std::min(count, kCycleLaneBuckets)];
    }
    [[nodiscard]] std::uint64_t rejected() const {
        const std::lock_guard<std::mutex> lock(mutex_);
        return rejected_;
    }
    [[nodiscard]] std::size_t ring_capacity() const noexcept { return ring_.size(); }
    // The most recent receipts, oldest first, at most `limit`.
    [[nodiscard]] std::vector<CycleReceipt> recent(std::size_t limit = std::numeric_limits<std::size_t>::max()) const {
        const std::lock_guard<std::mutex> lock(mutex_);
        return recent_locked(limit);
    }
    void clear() {
        const std::lock_guard<std::mutex> lock(mutex_);
        sequence_ = 0;
        rejected_ = 0;
        totals_ = {};
        sums_ = {};
        by_lanes_ = {};
    }

    // Compact JSON object. `recent_limit` receipts of the ring are appended under "recent" (0: none).
    // Counters are cumulative; a client subtracts two snapshots ("totals", "by_lanes", "fit_sums").
    [[nodiscard]] std::string json(std::size_t recent_limit = 0) const {
        const std::lock_guard<std::mutex> lock(mutex_);
        const Fit f = fit(sums_);
        std::ostringstream o;
        o << std::setprecision(std::numeric_limits<double>::max_digits10);
        o << "{\"schema\":\"strix.serve.cycle_ledger.v1\",\"cycles\":" << totals_.cycles << ",\"rejected\":" << rejected_
          << ",\"ring_capacity\":" << ring_.size() << ",\"totals\":{\"prompt_cycles\":" << totals_.prompt_cycles
          << ",\"lanes\":" << totals_.lanes << ",\"physical_rows\":" << totals_.physical_rows << ",\"useful_rows\":"
          << totals_.useful_rows << ",\"padding_rows\":" << totals_.padding_rows << ",\"emitted\":" << totals_.emitted
          << ",\"draft_rows_by_depth\":";
        write_array(o, totals_.draft_rows_by_depth);
        o << ",\"draft_ms\":" << totals_.draft_ms << ",\"verify_ms\":" << totals_.verify_ms << ",\"host_ms\":"
          << totals_.host_ms << ",\"total_ms\":" << totals_.total_ms << "},\"fit\":{\"fixed_ms\":" << f.fixed_ms
          << ",\"row_ms\":" << f.row_ms << ",\"lane_ms\":" << f.lane_ms << ",\"residual_sd\":" << f.residual_sd
          << ",\"cycles\":" << f.cycles << ",\"terms\":" << f.terms << "},\"fit_sums\":{\"n\":" << sums_.n << ",\"r\":"
          << sums_.r << ",\"l\":" << sums_.l << ",\"rr\":" << sums_.rr << ",\"rl\":" << sums_.rl << ",\"ll\":" << sums_.ll
          << ",\"y\":" << sums_.y << ",\"ry\":" << sums_.ry << ",\"ly\":" << sums_.ly << ",\"yy\":" << sums_.yy
          << "},\"by_lanes\":[";
        bool first = true;
        for (std::size_t lanes = 0; lanes <= kCycleLaneBuckets; ++lanes) {
            const LaneBucket& b = by_lanes_[lanes];
            if (b.cycles == 0) continue;
            o << (first ? "" : ",") << "{\"lanes\":" << lanes << ",\"cycles\":" << b.cycles << ",\"physical_rows\":"
              << b.physical_rows << ",\"useful_rows\":" << b.useful_rows << ",\"emitted\":" << b.emitted
              << ",\"total_ms\":" << b.total_ms << '}';
            first = false;
        }
        o << "],\"recent\":[";
        first = true;
        for (const CycleReceipt& r : recent_locked(recent_limit)) {
            o << (first ? "" : ",") << "{\"lanes\":" << r.lanes << ",\"physical_rows\":" << r.physical_rows
              << ",\"useful_rows\":" << r.useful_rows << ",\"padding_rows\":" << r.padding_rows << ",\"emitted\":"
              << r.emitted << ",\"draft_rows_by_depth\":";
            write_array(o, r.draft_rows_by_depth);
            o << ",\"draft_ms\":" << r.draft_ms << ",\"verify_ms\":" << r.verify_ms << ",\"host_ms\":" << r.host_ms
              << ",\"total_ms\":" << r.total_ms << ",\"prompt_unit_ran\":" << (r.prompt_unit_ran ? "true" : "false")
              << ",\"begin_ns\":" << r.begin_ns << ",\"end_ns\":" << r.end_ns << '}';
            first = false;
        }
        o << "]}";
        return o.str();
    }

private:
    template <typename T>
    static void write_array(std::ostringstream& o, const std::array<T, kCycleDraftDepths>& values) {
        o << '[';
        for (std::size_t i = 0; i < values.size(); ++i) o << (i ? "," : "") << values[i];
        o << ']';
    }
    [[nodiscard]] std::vector<CycleReceipt> recent_locked(std::size_t limit) const {
        const std::size_t held = static_cast<std::size_t>(std::min<std::uint64_t>(sequence_, ring_.size()));
        const std::size_t count = std::min(held, limit);
        std::vector<CycleReceipt> out;
        out.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
            out.push_back(ring_[static_cast<std::size_t>((sequence_ - count + i) % ring_.size())]);
        return out;
    }

    mutable std::mutex mutex_;
    std::vector<CycleReceipt> ring_;
    std::uint64_t sequence_ = 0, rejected_ = 0;
    Totals totals_{};
    FitSums sums_{};
    std::array<LaneBucket, kCycleLaneBuckets + 1> by_lanes_{};
};

// The process ledger an engine's scheduler adds to and its status route serves.
[[nodiscard]] inline CycleLedger& cycle_ledger() {
    static CycleLedger ledger;
    return ledger;
}

// Exit dump: one "serve ledger <json>" line, only when the switch is on and a cycle was recorded.
inline void dump_cycle_ledger(std::FILE* out) {
    if (!out || !cycle_receipt_enabled() || cycle_ledger().totals().cycles == 0) return;
    const std::string text = cycle_ledger().json();
    std::fprintf(out, "serve ledger %s\n", text.c_str());
}

} // namespace strix::serve
