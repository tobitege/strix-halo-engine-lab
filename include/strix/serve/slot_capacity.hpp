#pragma once
// Slot capacity accounting shared by every serving family.
//
// A slot has two kinds of bytes:
//   retained     persistent per-slot state (KV, recurrent state, arenas, drafter stacks). Paid by every built slot.
//   active peak  reusable scratch a slot needs only while it runs.
// Engines fill these terms and the memory observations; the arithmetic, the limiting term and the report live here.
// Pure host code: no device types, no engine dependency.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::serve {
// Pure admission policy; callers observe memory only when demand requires growth.
// Grow when every built slot is busy and work waits, or (spare) as soon as the last free slot is taken, so a request
// arriving during a long prompt finds a built slot (nested admission needs one inside the block).
inline bool slot_should_grow(bool queued, std::size_t active, std::size_t built,
    std::size_t capacity, bool memory_ok, bool spare = false) {
    return (queued || spare) && active == built && built < capacity && memory_ok;
}
inline std::uint32_t eager_slots(std::uint32_t capacity, std::uint32_t eager = 4) {
    return eager ? std::min(capacity, eager) : capacity;
}
inline bool slot_memory_fits(std::uint64_t retained, std::uint64_t active_peak,
    std::optional<std::uint64_t> available,
    std::optional<std::uint64_t> gtt_available, std::uint64_t headroom) {
    if (!available || !gtt_available) return false;
    const auto bytes = std::min(*available, *gtt_available);
    if (bytes < headroom || bytes - headroom < retained) return false;
    const auto remaining = bytes - headroom - retained;
    return remaining >= active_peak;
}
inline std::uint64_t slot_retained_bytes(std::uint64_t state_device, std::uint64_t state_host,
    std::uint64_t arena, std::uint64_t metadata, std::uint64_t external) {
    std::uint64_t total = 0;
    for (const auto n : {state_device, state_host, arena, metadata, external}) {
        if (n > UINT64_MAX-total) throw std::overflow_error("HTTP slot retained overflow");
        total += n;
    }
    return total;
}
struct ServingMemory {
    std::uint64_t shared_host = 0, shared_device = 0, slot_host = 0, slot_device = 0;
};
// At the serving boundary bridge shared setup is resident (or released).
// Reserve only unallocated cache/capture payload plus fixed headroom. Fresh
// observations already include the resident fraction of both shared payloads.
inline std::uint64_t shared_growth_headroom(std::uint64_t fixed,
    std::uint64_t cache_cap, std::uint64_t cache_resident,
    std::uint64_t staging_cap, std::uint64_t staging_resident) {
    for (const auto remaining : {cache_cap-std::min(cache_cap,cache_resident),
            staging_cap-std::min(staging_cap,staging_resident)}) {
        if (remaining > UINT64_MAX-fixed) throw std::overflow_error("HTTP shared growth overflow");
        fixed += remaining;
    }
    return fixed;
}
inline bool bridge_growth_fits(std::uint64_t retained, const ServingMemory& cost,
    std::optional<std::uint64_t> host, std::optional<std::uint64_t> device,
    std::uint64_t remaining_headroom) {
    return slot_memory_fits(retained,cost.slot_host,host,host,remaining_headroom) &&
        slot_memory_fits(retained,cost.slot_device,device,device,remaining_headroom);
}
struct BridgeGrowthObservation {
    std::optional<std::uint64_t> host, device;
    std::uint64_t cache_resident=0, staging_resident=0;
};
// Capture the pre-bridge admission terms, but read residency and availability
// anew at each serving-boundary call. Shared setup peaks have no future charge.
template<class Observe>
auto make_bridge_growth_check(std::uint64_t retained, ServingMemory cost,
    std::uint64_t fixed_headroom, std::uint64_t cache_cap, std::uint64_t staging_cap, Observe observe) {
    return [=] {
        const auto pressure=observe();
        const auto remaining=shared_growth_headroom(fixed_headroom,cache_cap,pressure.cache_resident,
            staging_cap,pressure.staging_resident);
        return bridge_growth_fits(retained,cost,pressure.host,pressure.device,remaining);
    };
}
struct SlotCapacity {
    std::uint32_t slots = 0;
    std::string limiting = "missing memory observations";
};
// Slots that fit one pool: the first slot is already resident and pays only its active peak; every further slot
// pays retained + active peak. Saturates at UINT32_MAX.
inline std::uint64_t slots_in_pool(std::uint64_t retained, std::uint64_t active_peak,
    std::uint64_t bytes, std::uint64_t headroom) {
    if (bytes < headroom || bytes - headroom < active_peak) return UINT64_C(0);
    const auto extra = (bytes - headroom - active_peak) / (retained + active_peak);
    return UINT64_C(1) + std::min(extra, static_cast<std::uint64_t>(UINT32_MAX) - 1U);
}
// Observations are remaining capacity after the primary stack, before the probe.
// Retained bytes charge only extra stacks; active peaks charge every runnable slot.
inline SlotCapacity resolve_slot_capacity(std::uint64_t retained, std::uint64_t active_peak,
    std::optional<std::uint64_t> available, std::optional<std::uint64_t> gtt_available,
    std::uint64_t headroom, std::uint32_t cap = 0) {
    if (!available || !gtt_available || !retained) return {};
    if (active_peak > std::numeric_limits<std::uint64_t>::max() - retained)
        return {0, "cost overflow"};
    const auto uma = slots_in_pool(retained, active_peak, *available, headroom);
    const auto gtt = slots_in_pool(retained, active_peak, *gtt_available, headroom);
    auto slots = std::min({uma, gtt, static_cast<std::uint64_t>(UINT32_MAX)});
    std::string limiting = uma <= gtt ? "MemAvailable budget" : "GTT budget";
    if (cap && cap <= slots) { slots = cap; limiting = "requested cap"; }
    return {static_cast<std::uint32_t>(slots), std::move(limiting)};
}
// The primary stack/shared prefill is already resident at runtime. Future
// bridge owners have distinct host/GTT costs; do not charge calibration host
// payload to GTT or copied sidecar device backing only to host.
inline SlotCapacity resolve_bridge_capacity(std::uint64_t retained,
    const ServingMemory& cost, std::optional<std::uint64_t> host,
    std::optional<std::uint64_t> device, std::uint64_t host_headroom,
    std::uint64_t device_headroom, std::uint32_t cap = 0) {
    if (cost.shared_host > UINT64_MAX-host_headroom || cost.shared_device > UINT64_MAX-device_headroom)
        return {0,"shared cost overflow"};
    auto h=resolve_slot_capacity(retained,cost.slot_host,host,host,host_headroom+cost.shared_host,cap);
    auto d=resolve_slot_capacity(retained,cost.slot_device,device,device,device_headroom+cost.shared_device,cap);
    if (h.slots<=d.slots) { h.limiting="MemAvailable: "+h.limiting; return h; }
    d.limiting="GTT: "+d.limiting; return d;
}

// ---- Report: the same answer with every term visible ----------------------------------------------------------
struct SlotCost {
    std::uint64_t retained_bytes = 0;     // persistent per-slot bytes
    std::uint64_t active_peak_bytes = 0;  // reusable scratch while the slot runs
};
// One memory pool a slot draws from (host RAM, device-visible memory, a KV page pool, ...).
struct CapacityPool {
    std::string name;
    std::optional<std::uint64_t> available_bytes;  // observed; unknown stays unknown
    std::uint64_t headroom_bytes = 0;              // fixed reserve plus shared one-time costs
    SlotCost cost;
    std::uint64_t keep_bytes = 0;                  // kept-prefix budget charged before slots are counted (see keep_budget_grant)
};
struct CapacityTerm {
    std::string name;
    std::optional<std::uint64_t> available_bytes;
    std::uint64_t headroom_bytes = 0;
    SlotCost cost;
    std::uint32_t slots = 0;  // what this pool alone allows, before the cap
    std::uint64_t keep_bytes = 0;  // kept-prefix budget charged to this pool (0: none; the report omits it)
};
struct CapacityReport {
    std::uint32_t slots = 0;
    std::string limiting = "missing memory observations";
    std::vector<CapacityTerm> terms;
};
// Kept-prefix (conversation cache) budget for a pool. Live slots win: the grant is what is left after headroom and
// `min_slots` slots (the first pays its active peak, each further one retained + active peak), capped by the
// request. An unobserved pool or a missing slot cost grants nothing. Store the grant in CapacityPool::keep_bytes
// (with_keep_budget does) so report_slot_capacity charges it before slots are counted: a smaller grant shrinks the
// cache, never the slots.
inline std::uint64_t keep_budget_grant(const CapacityPool& pool, std::uint64_t requested, std::uint32_t min_slots = 1) {
    if (!requested || !pool.available_bytes || !pool.cost.retained_bytes) return 0;
    const auto& c = pool.cost;
    if (c.active_peak_bytes > UINT64_MAX - c.retained_bytes) return 0;
    const auto per_extra = c.retained_bytes + c.active_peak_bytes;
    const std::uint64_t extra = min_slots > 1 ? static_cast<std::uint64_t>(min_slots - 1U) : 0U;
    if (extra && per_extra > UINT64_MAX / extra) return 0;
    const auto live = extra * per_extra;
    if (c.active_peak_bytes > UINT64_MAX - live) return 0;
    const auto reserved = c.active_peak_bytes + live;
    if (pool.headroom_bytes > UINT64_MAX - reserved) return 0;
    const auto floor = pool.headroom_bytes + reserved;
    return *pool.available_bytes > floor ? std::min(requested, *pool.available_bytes - floor) : 0;
}
inline CapacityPool with_keep_budget(CapacityPool pool, std::uint64_t requested, std::uint32_t min_slots = 1) {
    pool.keep_bytes = keep_budget_grant(pool, requested, min_slots);
    return pool;
}
// Minimum over the pools; ties name the first pool. A pool without an observation, or with no retained cost, gives
// zero slots: capacity is never raised on an estimate.
inline CapacityReport report_slot_capacity(const std::vector<CapacityPool>& pools, std::uint32_t cap = 0) {
    CapacityReport report;
    if (pools.empty()) return report;
    std::uint64_t best = UINT32_MAX;
    std::string limiting;
    for (const auto& pool : pools) {
        CapacityTerm term{pool.name, pool.available_bytes, pool.headroom_bytes, pool.cost, 0U, pool.keep_bytes};
        const auto headroom = pool.keep_bytes > UINT64_MAX - pool.headroom_bytes ? UINT64_MAX
                                                                                  : pool.headroom_bytes + pool.keep_bytes;
        std::string why = pool.name + " budget";
        if (!pool.available_bytes) why = "missing memory observations: " + pool.name;
        else if (!pool.cost.retained_bytes) why = "missing slot cost: " + pool.name;
        else if (pool.cost.active_peak_bytes > UINT64_MAX - pool.cost.retained_bytes) why = "cost overflow: " + pool.name;
        else term.slots = static_cast<std::uint32_t>(slots_in_pool(pool.cost.retained_bytes,
            pool.cost.active_peak_bytes, *pool.available_bytes, headroom));
        if (limiting.empty() || term.slots < best) { best = term.slots; limiting = std::move(why); }
        report.terms.push_back(std::move(term));
    }
    if (cap && cap <= best) { best = cap; limiting = "requested cap"; }
    report.slots = static_cast<std::uint32_t>(best);
    report.limiting = std::move(limiting);
    return report;
}
// Report for the host + device pair. `slots` and `limiting` are those of resolve_bridge_capacity; the terms add the
// per-pool numbers behind them.
inline CapacityReport report_bridge_capacity(std::uint64_t retained,
    const ServingMemory& cost, std::optional<std::uint64_t> host,
    std::optional<std::uint64_t> device, std::uint64_t host_headroom,
    std::uint64_t device_headroom, std::uint32_t cap = 0) {
    const auto resolved = resolve_bridge_capacity(retained, cost, host, device, host_headroom, device_headroom, cap);
    CapacityReport report{resolved.slots, resolved.limiting, {}};
    const auto term = [&](const char* name, std::optional<std::uint64_t> bytes, std::uint64_t headroom,
            std::uint64_t shared, std::uint64_t peak) {
        CapacityTerm t{name, bytes, headroom, {retained, peak}, 0U};
        const bool sum_ok = shared <= UINT64_MAX - headroom;
        if (sum_ok) t.headroom_bytes = headroom + shared;
        if (sum_ok && bytes && retained && peak <= UINT64_MAX - retained)
            t.slots = static_cast<std::uint32_t>(slots_in_pool(retained, peak, *bytes, t.headroom_bytes));
        report.terms.push_back(std::move(t));
    };
    term("MemAvailable", host, host_headroom, cost.shared_host, cost.slot_host);
    term("GTT", device, device_headroom, cost.shared_device, cost.slot_device);
    return report;
}
namespace detail {
inline std::string bytes_or_na(const std::optional<std::uint64_t>& value) {
    return value ? std::to_string(*value) : std::string("na");
}
} // namespace detail
// One log line: `serve-capacity slots=8 limiting="MemAvailable: MemAvailable budget" MemAvailable{...} GTT{...}`.
inline std::string format_capacity_report(const CapacityReport& report) {
    std::string out = "serve-capacity slots=" + std::to_string(report.slots) + " limiting=\"" + report.limiting + "\"";
    for (const auto& t : report.terms)
        out += " " + t.name + "{available=" + detail::bytes_or_na(t.available_bytes) +
            " headroom=" + std::to_string(t.headroom_bytes) + " retained=" + std::to_string(t.cost.retained_bytes) +
            " active_peak=" + std::to_string(t.cost.active_peak_bytes) + " slots=" + std::to_string(t.slots) +
            (t.keep_bytes ? " keep=" + std::to_string(t.keep_bytes) : std::string()) + "}";
    return out;
}
// JSON object for a status route. Names are engine-chosen identifiers; quotes and backslashes are escaped.
inline std::string capacity_report_json(const CapacityReport& report) {
    const auto quote = [](std::string_view text) {
        std::string out = "\"";
        for (const char c : text) {
            if (c == '"' || c == '\\') out += '\\';
            out += static_cast<unsigned char>(c) < 0x20U ? ' ' : c;
        }
        return out + "\"";
    };
    std::string out = "{\"slots\":" + std::to_string(report.slots) + ",\"limiting\":" + quote(report.limiting) + ",\"terms\":[";
    for (std::size_t i = 0; i < report.terms.size(); ++i) {
        const auto& t = report.terms[i];
        out += std::string(i ? "," : "") + "{\"name\":" + quote(t.name) + ",\"available_bytes\":" +
            (t.available_bytes ? std::to_string(*t.available_bytes) : std::string("null")) +
            ",\"headroom_bytes\":" + std::to_string(t.headroom_bytes) +
            ",\"retained_bytes\":" + std::to_string(t.cost.retained_bytes) +
            ",\"active_peak_bytes\":" + std::to_string(t.cost.active_peak_bytes) +
            ",\"slots\":" + std::to_string(t.slots) +
            (t.keep_bytes ? ",\"keep_bytes\":" + std::to_string(t.keep_bytes) : std::string()) + "}";
    }
    return out + "]}";
}

// ---- Switches ---------------------------------------------------------------------------------------------------
// STRIX_SERVE_SLOT_EXACT=1: engines may charge what a slot really allocates instead of their conservative bound.
// Unset (default): every charge is unchanged.
inline bool slot_exact_from(const char* value) { return value && std::string_view(value) == "1"; }
inline bool slot_exact() {
    static const bool on = slot_exact_from(std::getenv("STRIX_SERVE_SLOT_EXACT"));
    return on;
}
// STRIX_SERVE_CENSUS=1: print the memory census lines (before load, after load, at the slot probe).
inline bool census_enabled_from(const char* value) { return value && std::string_view(value) == "1"; }
inline bool census_enabled() {
    static const bool on = census_enabled_from(std::getenv("STRIX_SERVE_CENSUS"));
    return on;
}

// ---- Memory census ----------------------------------------------------------------------------------------------
// One observation of host and device-visible memory at a named point. `fields` carries whatever else the engine can
// read (kernel counters, registered bytes, what it charged), in bytes, in print order.
struct MemoryCensus {
    std::string point;
    std::optional<std::uint64_t> host_total, host_available, device_limit, device_used;
    std::vector<std::pair<std::string, std::uint64_t>> fields;
};
// Appends `<prefix><Name>` = bytes for every wanted `Name:   N kB` line of a /proc/meminfo or /proc/self/status
// style text. Lines that are missing or malformed are skipped; the text is never trusted beyond that.
inline void append_kb_fields(MemoryCensus& census, std::string_view text,
    std::initializer_list<std::string_view> names, std::string_view prefix = {}) {
    for (const auto name : names) {
        std::size_t at = 0;
        while (at < text.size()) {
            auto end = text.find('\n', at);
            if (end == std::string_view::npos) end = text.size();
            const auto line = text.substr(at, end - at);
            at = end + 1U;
            if (line.size() <= name.size() || line.substr(0, name.size()) != name || line[name.size()] != ':') continue;
            auto rest = line.substr(name.size() + 1U);
            while (!rest.empty() && (rest.front() == ' ' || rest.front() == '\t')) rest.remove_prefix(1U);
            std::uint64_t value = 0;
            std::size_t digits = 0;
            bool overflow = false;
            while (digits < rest.size() && rest[digits] >= '0' && rest[digits] <= '9') {
                const auto digit = static_cast<std::uint64_t>(rest[digits] - '0');
                if (value > (UINT64_MAX - digit) / 10U) { overflow = true; break; }
                value = value * 10U + digit;
                ++digits;
            }
            rest.remove_prefix(digits);
            while (!rest.empty() && (rest.front() == ' ' || rest.front() == '\t')) rest.remove_prefix(1U);
            const bool kb = rest.substr(0, 2U) == "kB";
            if (overflow || !digits || !kb || value > UINT64_MAX / 1024U) break;
            census.fields.emplace_back(std::string(prefix) + std::string(name), value * 1024U);
            break;
        }
    }
}
// `serve-census point=probe host_total=.. host_available=.. device_limit=.. device_used=.. device_free=.. k=v ...`
inline std::string format_memory_census(const MemoryCensus& census) {
    std::string out = "serve-census point=" + census.point +
        " host_total=" + detail::bytes_or_na(census.host_total) +
        " host_available=" + detail::bytes_or_na(census.host_available) +
        " device_limit=" + detail::bytes_or_na(census.device_limit) +
        " device_used=" + detail::bytes_or_na(census.device_used);
    std::optional<std::uint64_t> device_free;
    if (census.device_limit && census.device_used && *census.device_used <= *census.device_limit)
        device_free = *census.device_limit - *census.device_used;
    out += " device_free=" + detail::bytes_or_na(device_free);
    for (const auto& [name, bytes] : census.fields) out += " " + name + "=" + std::to_string(bytes);
    return out;
}
// What moved between two points, signed, for the two headline numbers and every field both carry:
// `serve-census-delta from=before_load to=probe host_available=-96636764160 device_used=+30923764531 ...`
inline std::string format_census_delta(const MemoryCensus& from, const MemoryCensus& to) {
    const auto signed_delta = [](std::uint64_t a, std::uint64_t b) {
        return b >= a ? "+" + std::to_string(b - a) : "-" + std::to_string(a - b);
    };
    const auto opt = [&](const std::optional<std::uint64_t>& a, const std::optional<std::uint64_t>& b) {
        return a && b ? signed_delta(*a, *b) : std::string("na");
    };
    std::string out = "serve-census-delta from=" + from.point + " to=" + to.point +
        " host_available=" + opt(from.host_available, to.host_available) +
        " device_used=" + opt(from.device_used, to.device_used);
    for (const auto& [name, bytes] : to.fields)
        for (const auto& [other, before] : from.fields)
            if (other == name) { out += " " + name + "=" + signed_delta(before, bytes); break; }
    return out;
}
} // namespace strix::serve
