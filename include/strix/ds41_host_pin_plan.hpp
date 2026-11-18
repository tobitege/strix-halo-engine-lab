#pragma once
// CPU-only host placement for an ascending, single-pass expert schedule.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix::ds41 {

// automatic: interleave or layer by the measured decision table (resolve_host_pin_order), once at load.
enum class HostPinOrder { layer, deadline, early, interleave, automatic };

inline HostPinOrder parse_host_pin_order(std::string_view value) {
    if (value == "layer") return HostPinOrder::layer;
    if (value == "deadline") return HostPinOrder::deadline;
    if (value == "early") return HostPinOrder::early;
    if (value == "interleave") return HostPinOrder::interleave;
    if (value == "auto") return HostPinOrder::automatic;
    throw std::invalid_argument("--host-pin-order must be auto, layer, deadline, early or interleave");
}
inline const char* host_pin_order_name(HostPinOrder order) {
    return order == HostPinOrder::deadline ? "deadline" : order == HostPinOrder::early ? "early"
         : order == HostPinOrder::interleave ? "interleave" : order == HostPinOrder::automatic ? "auto" : "layer";
}

// Zero bytes means fully device resident. The first S nonzero layers already occupy
// the idle-preload slots, regardless of tier. Deadline placement keeps those on FILE
// and pins the longest affordable suffix of the remaining streamed schedule. In
// particular, small decoder tails are considered before full encoder layers.
// Stop at the first non-fitting layer: filling the remainder with earlier HOST
// layers would put a host-copy burst before a later FILE deadline. The loader has
// no independent disk lookahead past that burst (its four staging pages are bounded).
// This is a suffix policy, not a general heterogeneous knapsack optimizer. It avoids
// internal HOST/FILE transitions and keeps the disk prefix continuous at short rows.
inline std::vector<std::uint8_t> deadline_host_layers(std::span<const std::uint64_t> bytes,
                                                    std::uint32_t slots, double budget_bytes) {
    std::vector<std::uint8_t> host(bytes.size(), 0);
    std::vector<std::size_t> streamed;
    for (std::size_t l = 0; l < bytes.size(); ++l)
        if (bytes[l]) streamed.push_back(l);
    const auto preloaded = std::min<std::size_t>(std::max(1U, slots), streamed.size());
    for (std::size_t j = streamed.size(); j > preloaded;) {
        const auto l = streamed[--j];
        if (budget_bytes < static_cast<double>(bytes[l])) break;
        budget_bytes -= static_cast<double>(bytes[l]);
        host[l] = 1;
    }
    return host;
}

// Early placement: the legacy ascending fill (every streamed layer that still fits, in
// layer order), minus the first S streamed layers that the idle preload already holds.
// Pinned layers then sit at the start of the schedule, where little compute hides a
// FILE read, while later FILE reads overlap the compute of the pinned ones (measured: the
// suffix order was no faster than the legacy one at 30K for that reason).
inline std::vector<std::uint8_t> early_host_layers(std::span<const std::uint64_t> bytes,
                                                 std::uint32_t slots, double budget_bytes) {
    std::vector<std::uint8_t> host(bytes.size(), 0);
    std::size_t seen = 0;
    const std::size_t preloaded = std::max(1U, slots);
    for (std::size_t l = 0; l < bytes.size(); ++l) {
        if (!bytes[l]) continue;
        if (seen++ < preloaded) continue;
        if (budget_bytes < static_cast<double>(bytes[l])) continue;
        budget_bytes -= static_cast<double>(bytes[l]);
        host[l] = 1;
    }
    return host;
}

// Interleaved placement: a two-resource schedule of the streamed layers in execution order. The copy stream moves every
// streamed byte (host or file) at pcie_gbps; a file-tier layer is first read from the disk at disk_gbps into the file
// ring (ring_bytes of pinned pages), which can run ahead of the copies while it has room. With the layer and early
// orders the file layers come as one block after the host ones, so the disk idles through the host copies and the copy
// stream then waits for the disk (measured on one RTX: 61.5 GB host at 25 GB/s then 12.2 GB file at 6.6 GB/s, serial).
// Here a layer goes to FILE when bytes must come from the file anyway and its read, started when the disk and the ring
// are free, ends before the copy stream would finish that layer as a host one (the read hides behind earlier copies);
// otherwise HOST while the budget lasts, FILE when it is spent. Unlike deadline / early, the first S streamed layers
// (the slots' preload, loaded after each run) are pinned first: with back-to-back requests there is no idle time and
// a file preload (7.6 GB: 1.2 s from the disk vs 0.19 s from pinned memory) ran into the next run (measured: 1.25 s
// wait at layer 0). Only the source of the same bytes changes: no resident set, permutation or arithmetic changes.
// The rates only steer the placement (a wrong rate costs time, never bits).
inline std::vector<std::uint8_t> interleave_host_layers(std::span<const std::uint64_t> bytes, std::uint32_t slots,
                                                      double budget_bytes, double ring_bytes, double pcie_gbps = 25.0,
                                                      double disk_gbps = 6.6) {
    std::vector<std::uint8_t> host(bytes.size(), 0);
    const std::size_t preloaded = std::max(1U, slots);
    double rest = 0.0;
    std::size_t seen = 0;
    for (std::size_t l = 0; l < bytes.size(); ++l) {
        if (!bytes[l]) continue;
        if (seen++ >= preloaded) rest += static_cast<double>(bytes[l]);
        else if (budget_bytes >= static_cast<double>(bytes[l])) {  // the preload, from pinned memory
            host[l] = 1;
            budget_bytes -= static_cast<double>(bytes[l]);
        }
    }
    double file_need = std::max(0.0, rest - budget_bytes);  // bytes that come from the file whatever the order
    double t_copy = 0.0, t_disk = 0.0, ring_free_at = 0.0, ring_held = 0.0;
    seen = 0;
    for (std::size_t l = 0; l < bytes.size(); ++l) {
        if (!bytes[l] || seen++ < preloaded) continue;
        const double b = static_cast<double>(bytes[l]);
        const double copy = b / (pcie_gbps * 1e9), read = b / (disk_gbps * 1e9);
        // the read starts when the disk is free and the ring has room (else after the previous file layer's copies);
        // a layer larger than the ring is read in step with its own copies
        double start = t_disk;
        if (b > ring_bytes) start = std::max(start, t_copy);
        else if (b + ring_held > ring_bytes) start = std::max(start, ring_free_at);
        const double read_end = start + read;
        const bool fits = budget_bytes >= b;
        const bool to_file = !fits || (file_need >= b && read_end <= t_copy + copy);
        if (to_file) {
            t_disk = read_end;
            t_copy = std::max(t_copy + copy, read_end);
            ring_held = std::min(b, ring_bytes);
            ring_free_at = t_copy;
            file_need = std::max(0.0, file_need - b);
        } else {
            host[l] = 1;
            budget_bytes -= b;
            t_copy += copy;
        }
    }
    return host;
}

// The host set of a resolved order (bytes[l]: streamed bytes of layer l, 0 = resident; budget and ring in bytes).
inline std::vector<std::uint8_t> ordered_host_layers(HostPinOrder order, std::span<const std::uint64_t> bytes, std::uint32_t slots,
                                                   double budget_bytes, double ring_bytes) {
    if (order == HostPinOrder::automatic) throw std::invalid_argument("resolve automatic host order before placement");
    if (order == HostPinOrder::layer) {
        std::vector<std::uint8_t> host(bytes.size(), 0);
        for (std::size_t l = 0; l < bytes.size(); ++l)
            if (bytes[l] && budget_bytes >= static_cast<double>(bytes[l])) {
                host[l] = 1;
                budget_bytes -= static_cast<double>(bytes[l]);
            }
        return host;
    }
    if (order == HostPinOrder::early) return early_host_layers(bytes, slots, budget_bytes);
    if (order == HostPinOrder::interleave) return interleave_host_layers(bytes, slots, budget_bytes, ring_bytes);
    if (order == HostPinOrder::deadline) return deadline_host_layers(bytes, slots, budget_bytes);
    throw std::invalid_argument("unsupported host placement order");
}

// File-read ring of the streamed experts. ring_mib 0 = auto (the default): the interleave order takes a 4 GiB ring out
// of the pinned host budget, so the order changes placement, not pinned RAM (the measured equal-RAM arms: 68 GiB layer
// vs 64 GiB + 4 GiB ring interleave); other orders, no streamed slots or a budget under two rings (not measured) keep the
// four 64 MiB staging pages. An explicit ring is added to the budget, as before. Idempotent: a resolved ring is nonzero.
inline constexpr std::uint32_t kLegacyFileRingMib = 256;
inline constexpr std::uint32_t kInterleaveFileRingMib = 4096;
struct FileRing {
    double host_gib;
    std::uint32_t ring_mib;
};
inline FileRing resolve_file_ring(HostPinOrder order, double host_gib, std::uint32_t ring_mib, std::uint32_t slots) {
    if (ring_mib != 0) return {host_gib, ring_mib};
    constexpr double ring_gib = double(kInterleaveFileRingMib) / 1024.0;
    if (order == HostPinOrder::interleave && slots > 0 && host_gib >= 2.0 * ring_gib) return {host_gib - ring_gib, kInterleaveFileRingMib};
    return {host_gib, kLegacyFileRingMib};
}

// AUTO WHERE IT WINS: the order the automatic default resolves to, decided once at load (the pinned set is fixed for
// the process). Decision variable: the bytes the layer order would stream from the file, F = streamed - pinnable,
// pinnable = min(budget, min(free physical RAM, free commit) - floor); unknown room declines growth. Measured on one RTX PRO
// (sm_120, PCIe 5 x16, NVMe ~6.6 GB/s), equal pinned RAM, ABAB, 3 reps, all outputs bit-identical:
//   F  8.4 GB: interleave 2K warm -5 % / cold -10 %, 8K warm +3 % / cold +6 %, 32K -1 %   -> layer (8K loses)
//   F 12.2 GB: 2K -16 %, 8K -15 %, 32K -1 %; F 19.9-23.7 GB: 2K -13..-15 %, 8K -5..-6 %  -> interleave
// Why it loses below the crossover: the ring (4 GiB) comes out of the pinned budget, so interleave streams ~4-6 GB
// more from the disk; with a small file tier the 8K compute already hides the reads and the extra disk bytes show.
// Crossover 10 GB (between the 8.4 and 12.2 GB receipts). Explicit orders (--host-pin-order layer / interleave) override.
inline constexpr double kInterleaveMinFileBytes = 10e9;
struct HostPinChoice {
    HostPinOrder order;
    double host_gib;
    std::uint32_t ring_mib;
    double layer_file_bytes;  // F (automatic only; -1 otherwise)
    const char* reason;
};
// Unknown telemetry cannot admit optional ring growth. Both resources retain the floor.
inline double host_pin_headroom(double physical, double commit, double floor_bytes) {
    return std::max(0.0, std::min(physical, commit) - floor_bytes);
}

// Allocate a ring transactionally: release partial optional growth before retrying legacy staging.
// Callbacks keep device allocation out of the shared planner and permit host fault injection.
template<class Room, class Allocate, class Release>
HostPinChoice allocate_file_ring(HostPinChoice choice, bool automatic_growth, std::size_t page_padding,
                                Room room, Allocate allocate, Release release) {
    const auto pages = std::max(2U, choice.ring_mib / 64U);
    const double page_bytes = double((64ULL << 20U) + page_padding);
    if ((!automatic_growth || room() >= double(pages) * page_bytes) && allocate(pages, automatic_growth)) return choice;
    release();
    if (!automatic_growth) throw std::runtime_error("explicit file ring allocation failed");
    const double restored = choice.host_gib + double(choice.ring_mib) / 1024.0;
    choice.host_gib = std::min(restored, std::max(0.0, room() - 4.0 * page_bytes) / double(1ULL << 30));
    choice.order = HostPinOrder::layer;
    choice.ring_mib = kLegacyFileRingMib;
    choice.reason = "automatic file ring declined: legacy staging and layer placement";
    if (!allocate(4U, false)) throw std::runtime_error("legacy file ring allocation failed");
    return choice;
}

inline HostPinChoice resolve_host_pin_order(HostPinOrder requested, std::span<const std::uint64_t> streamed_bytes, double host_gib,
                                            double pinnable_bytes, std::uint32_t ring_mib, std::uint32_t slots) {
    if (requested != HostPinOrder::automatic) {
        const auto ring = resolve_file_ring(requested, host_gib, ring_mib, slots);
        return {requested, ring.host_gib, ring.ring_mib, -1.0, "explicit order"};
    }
    // pinnable_bytes < 0: host memory telemetry unknown, so the automatic order admits no pinned budget. A budget the
    // caller asked for is refused with this reason instead of silently dropped; an explicit order pins without telemetry.
    if (pinnable_bytes < 0.0 && host_gib > 0.0)
        throw std::runtime_error("host pin order auto: host memory telemetry unavailable, the requested host budget is refused "
                                 "(pass an explicit host pin order to pin it without telemetry)");
    double streamed = 0.0;
    for (const auto b : streamed_bytes) streamed += static_cast<double>(b);
    const double available_gib = std::max(0.0, pinnable_bytes) / double(1ULL << 30);
    host_gib = std::min(host_gib, available_gib);
    const double pinned = host_gib * double(1ULL << 30);
    const double file = std::max(0.0, streamed - pinned);
    const bool on = slots > 0 && host_gib >= 2.0 * double(kInterleaveFileRingMib) / 1024.0 && file >= kInterleaveMinFileBytes;
    const auto order = on ? HostPinOrder::interleave : HostPinOrder::layer;
    auto ring = resolve_file_ring(order, host_gib, ring_mib, slots);
    ring.host_gib = std::min(ring.host_gib, std::max(0.0, available_gib - double(ring.ring_mib) / 1024.0));
    return {order, ring.host_gib, ring.ring_mib, file,
            on ? "layer-order file tier >= 10 GB: interleave (measured 12-24 GB: 2K -13..-16 %, 8K -5..-15 %)"
               : slots == 0 || host_gib < 2.0 * double(kInterleaveFileRingMib) / 1024.0
                    ? "no streamed slots or admitted host budget under 8 GiB: layer"
                   : "layer-order file tier < 10 GB: layer (measured 8.4 GB: interleave 8K +3 % warm, +6 % cold)"};
}

}  // namespace strix::ds41
