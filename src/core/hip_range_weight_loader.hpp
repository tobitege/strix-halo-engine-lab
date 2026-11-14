#pragma once
// File byte ranges -> device memory through core::WeightLoadQueue: the adapter for engines whose loader is
// "copy [offset, offset + bytes) of the model file to this device pointer" (GGUF tensors, tensor slices, row runs).
// load() only queues: ranges are cut into pieces, small consecutive ranges share one ticket, and nothing waits.
// ready() is the one wait (end-of-load publication, or before the first consumer of the bytes).
#include "hip_weight_uploader.hpp"
#include "weight_file_source.hpp"

#include <string>

namespace strix::core {

class HipRangeWeightLoader {
public:
    static constexpr std::uint64_t default_staging_bytes = std::uint64_t{512} << 20;
    explicit HipRangeWeightLoader(const std::string& path, std::uint64_t staging_bytes = default_staging_bytes)
        : source_(path), queue_(uploader_, weight_load_options_from_environment(staging_bytes)), tickets_(queue_) {}
    HipRangeWeightLoader(const HipRangeWeightLoader&) = delete;
    HipRangeWeightLoader& operator=(const HipRangeWeightLoader&) = delete;
    [[nodiscard]] int fd() const { return source_.fd(); }
    [[nodiscard]] bool direct() const { return source_.direct(); }

    // Queues the copy; `device` must stay allocated until ready().
    void load(std::uint64_t offset, std::uint64_t bytes, std::uint8_t* device) { tickets_.load(source_, offset, bytes, device); }
    // Everything queued so far is in device memory; rethrows the first read or upload failure.
    void ready() {
        tickets_.flush();
        queue_.publish();
    }
    [[nodiscard]] WeightLoadStats stats() const { return queue_.stats(); }
    [[nodiscard]] const WeightLoadOptions& options() const { return queue_.options(); }
    [[nodiscard]] std::uint64_t peak_pinned_bytes() const { return uploader_.peak_pinned_bytes(); }

private:
    FileWeightSource source_;
    HipWeightUploader uploader_;
    WeightLoadQueue queue_;  // destroyed first: joins its workers and drains the copies
    WeightRangeTickets tickets_;  // piece and ticket boundaries from the aligned staging size
};

// One line per load for receipts: wall time, read rate, summed worker times, peak pinned staging.
inline std::string weight_load_line(const WeightLoadStats& s, std::uint64_t peak_pinned, const WeightLoadOptions& o) {
    char line[512];
    std::snprintf(line, sizeof(line),
                  "weight load queue: %llu tickets, %.3f GiB read, wall %.2f s (%.2f GiB/s), read %.2f s + convert %.2f s over %u workers, "
                  "staging wait %.2f s, peak pinned %.1f MiB of %.1f MiB budget, %llu staging allocations",
                  static_cast<unsigned long long>(s.tickets), static_cast<double>(s.read_bytes) / 1073741824.0, s.wall_seconds,
                  s.wall_seconds > 0 ? static_cast<double>(s.read_bytes) / 1073741824.0 / s.wall_seconds : 0.0, s.read_seconds,
                  s.convert_seconds, o.workers, s.admit_wait_seconds, static_cast<double>(peak_pinned) / 1048576.0,
                  static_cast<double>(o.staging_budget_bytes) / 1048576.0, static_cast<unsigned long long>(s.staging_allocations));
    return line;
}

}  // namespace strix::core
