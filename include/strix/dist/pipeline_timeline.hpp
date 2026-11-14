#pragma once
// Per-step exposure timeline for pipelined two-node work (shared; Qwen pair prefill is the first user).
// STRIX_PAIR_TIMING=1 enables it. OFF: no clock reads, no allocation, no output, no added synchronization.
// Spans are host steady-clock intervals of blocking calls on the compute and progress threads. A step's
// compute span is exact for GPU work only when the step ends in a host fence (the Qwen seam does).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace strix::dist {
inline bool pipeline_timing_enabled() {
    const char* value = std::getenv("STRIX_PAIR_TIMING");
    return value && std::string_view(value) == "1";
}
class PipelineTimeline {
public:
    using Clock = std::chrono::steady_clock;
    // columns: names of the accumulated spans, e.g. {"chunk","seam-wait","wire"}; must outlive the timeline.
    PipelineTimeline(std::string owner, unsigned rank, std::vector<const char*> columns,
        bool enabled = pipeline_timing_enabled())
        : owner_(std::move(owner)), rank_(rank), columns_(std::move(columns)), enabled_(enabled), epoch_(Clock::now()) {}
    bool enabled() const noexcept { return enabled_; }
    Clock::time_point now() const { return enabled_ ? Clock::now() : Clock::time_point{}; }
    void add(std::uint64_t step, std::size_t column, Clock::time_point begin, Clock::time_point end) {
        if (!enabled_) return;
        add_ms(step, column, std::chrono::duration<double, std::milli>(end - begin).count());
    }
    void add_since(std::uint64_t step, std::size_t column, Clock::time_point begin) {
        if (enabled_) add(step, column, begin, Clock::now());
    }
    void add_ms(std::uint64_t step, std::size_t column, double ms) {
        if (!enabled_ || column >= columns_.size()) return;
        std::lock_guard lock(mutex_);
        auto& row = row_of(step);
        row.ms[column] += ms;
    }
    // Step start/end on the owner's critical path (ms since the timeline was created).
    void mark(std::uint64_t step, Clock::time_point begin, Clock::time_point end) {
        if (!enabled_) return;
        std::lock_guard lock(mutex_);
        auto& row = row_of(step);
        row.start = std::chrono::duration<double, std::milli>(begin - epoch_).count();
        row.end = std::chrono::duration<double, std::milli>(end - epoch_).count();
    }
    template<class F> decltype(auto) measure(std::uint64_t step, std::size_t column, F&& operation) {
        if (!enabled_) return operation();
        struct Timer {
            PipelineTimeline& owner; std::uint64_t step; std::size_t column; Clock::time_point begin = Clock::now();
            ~Timer() { owner.add_since(step, column, begin); }
        } timer{*this, step, column};
        return operation();
    }
    // One line per step and a per-rank sum line. wall_ms < 0 omits the wall field.
    void report(double wall_ms = -1.0, std::FILE* out = stderr) const {
        if (!enabled_) return;
        std::lock_guard lock(mutex_);
        std::vector<double> sums(columns_.size(), 0.0);
        for (std::size_t step = 0; step < rows_.size(); ++step) {
            const auto& row = rows_[step];
            if (!row.used) continue;
            std::string line = "pair-timing owner=" + owner_ + " rank=" + std::to_string(rank_) + " step=" + std::to_string(step);
            char buffer[64];
            std::snprintf(buffer, sizeof buffer, " start-ms=%.3f end-ms=%.3f", row.start, row.end); line += buffer;
            for (std::size_t column = 0; column < columns_.size(); ++column) {
                sums[column] += row.ms[column];
                std::snprintf(buffer, sizeof buffer, " %s=%.3f", columns_[column], row.ms[column]); line += buffer;
            }
            std::fprintf(out, "%s\n", line.c_str());
        }
        std::string line = "pair-timing owner=" + owner_ + " rank=" + std::to_string(rank_) + " sum steps=" +
            std::to_string(std::count_if(rows_.begin(), rows_.end(), [](const Row& row) { return row.used; }));
        char buffer[64];
        if (wall_ms >= 0.0) { std::snprintf(buffer, sizeof buffer, " wall-ms=%.3f", wall_ms); line += buffer; }
        for (std::size_t column = 0; column < columns_.size(); ++column) {
            std::snprintf(buffer, sizeof buffer, " %s=%.3f", columns_[column], sums[column]); line += buffer;
        }
        std::fprintf(out, "%s\n", line.c_str());
    }
private:
    struct Row { bool used = false; double start = 0, end = 0; std::vector<double> ms; };
    Row& row_of(std::uint64_t step) {
        const auto index = static_cast<std::size_t>(step);
        if (index >= rows_.size()) rows_.resize(index + 1U);
        auto& row = rows_[index];
        if (!row.used) { row.used = true; row.ms.assign(columns_.size(), 0.0); }
        return row;
    }
    std::string owner_;
    unsigned rank_;
    std::vector<const char*> columns_;
    bool enabled_;
    Clock::time_point epoch_;
    mutable std::mutex mutex_;
    std::vector<Row> rows_;
};
} // namespace strix::dist
