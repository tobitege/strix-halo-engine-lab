// Slot capacity in the shared serve layer: move equivalence against a frozen copy of the app-local arithmetic,
// the capacity report, the census formatting and the switches.
#include "strix/serve/slot_capacity.hpp"
#include "../apps/http_slot_capacity.hpp"

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>
#ifdef _MSC_VER
#include <crtdbg.h>
#include <cstdlib>
#endif

namespace legacy {
// Frozen copy of apps/http_slot_capacity.hpp as it was before the move. Do not edit: it is the reference.
inline bool http_slot_should_grow(bool queued, std::size_t active, std::size_t built,
    std::size_t capacity, bool memory_ok, bool spare = false) {
    return (queued || spare) && active == built && built < capacity && memory_ok;
}
inline std::uint32_t http_eager_slots(std::uint32_t capacity, std::uint32_t eager = 4) {
    return eager ? std::min(capacity, eager) : capacity;
}
inline bool http_slot_memory_fits(std::uint64_t retained, std::uint64_t active_peak,
    std::optional<std::uint64_t> available,
    std::optional<std::uint64_t> gtt_available, std::uint64_t headroom) {
    if (!available || !gtt_available) return false;
    const auto bytes = std::min(*available, *gtt_available);
    if (bytes < headroom || bytes - headroom < retained) return false;
    const auto remaining = bytes - headroom - retained;
    return remaining >= active_peak;
}
inline std::uint64_t http_slot_retained_bytes(std::uint64_t state_device, std::uint64_t state_host,
    std::uint64_t arena, std::uint64_t metadata, std::uint64_t external) {
    std::uint64_t total = 0;
    for (const auto n : {state_device, state_host, arena, metadata, external}) {
        if (n > UINT64_MAX-total) throw std::overflow_error("HTTP slot retained overflow");
        total += n;
    }
    return total;
}
struct HttpServingMemory {
    std::uint64_t shared_host = 0, shared_device = 0, slot_host = 0, slot_device = 0;
};
inline std::uint64_t http_shared_growth_headroom(std::uint64_t fixed,
    std::uint64_t cache_cap, std::uint64_t cache_resident,
    std::uint64_t staging_cap, std::uint64_t staging_resident) {
    for (const auto remaining : {cache_cap-std::min(cache_cap,cache_resident),
            staging_cap-std::min(staging_cap,staging_resident)}) {
        if (remaining > UINT64_MAX-fixed) throw std::overflow_error("HTTP shared growth overflow");
        fixed += remaining;
    }
    return fixed;
}
inline bool http_bridge_growth_fits(std::uint64_t retained, const HttpServingMemory& cost,
    std::optional<std::uint64_t> host, std::optional<std::uint64_t> device,
    std::uint64_t remaining_headroom) {
    return http_slot_memory_fits(retained,cost.slot_host,host,host,remaining_headroom) &&
        http_slot_memory_fits(retained,cost.slot_device,device,device,remaining_headroom);
}
struct HttpSlotCapacity {
    std::uint32_t slots = 0;
    std::string limiting = "missing memory observations";
};
inline HttpSlotCapacity resolve_http_slot_capacity(std::uint64_t retained, std::uint64_t active_peak,
    std::optional<std::uint64_t> available, std::optional<std::uint64_t> gtt_available,
    std::uint64_t headroom, std::uint32_t cap = 0) {
    if (!available || !gtt_available || !retained) return {};
    if (active_peak > std::numeric_limits<std::uint64_t>::max() - retained)
        return {0, "cost overflow"};
    const auto per_slot = retained + active_peak;
    const auto count = [&](std::uint64_t bytes) {
        if (bytes < headroom || bytes - headroom < active_peak) return UINT64_C(0);
        const auto extra = (bytes - headroom - active_peak) / per_slot;
        return UINT64_C(1) + std::min(extra, static_cast<std::uint64_t>(UINT32_MAX) - 1U);
    };
    const auto uma = count(*available), gtt = count(*gtt_available);
    auto slots = std::min({uma, gtt, static_cast<std::uint64_t>(UINT32_MAX)});
    std::string limiting = uma <= gtt ? "MemAvailable budget" : "GTT budget";
    if (cap && cap <= slots) { slots = cap; limiting = "requested cap"; }
    return {static_cast<std::uint32_t>(slots), std::move(limiting)};
}
inline HttpSlotCapacity resolve_http_bridge_capacity(std::uint64_t retained,
    const HttpServingMemory& cost, std::optional<std::uint64_t> host,
    std::optional<std::uint64_t> device, std::uint64_t host_headroom,
    std::uint64_t device_headroom, std::uint32_t cap = 0) {
    if (cost.shared_host > UINT64_MAX-host_headroom || cost.shared_device > UINT64_MAX-device_headroom)
        return {0,"shared cost overflow"};
    auto h=resolve_http_slot_capacity(retained,cost.slot_host,host,host,host_headroom+cost.shared_host,cap);
    auto d=resolve_http_slot_capacity(retained,cost.slot_device,device,device,device_headroom+cost.shared_device,cap);
    if (h.slots<=d.slots) { h.limiting="MemAvailable: "+h.limiting; return h; }
    d.limiting="GTT: "+d.limiting; return d;
}
} // namespace legacy

namespace {
int failures = 0;
void expect(bool ok, const char* what, int line) {
    if (!ok && ++failures <= 20) std::fprintf(stderr, "FAIL line %d: %s\n", line, what);
}
#define EXPECT(x) expect((x), #x, __LINE__)

using Opt = std::optional<std::uint64_t>;
const std::uint64_t kGiB = UINT64_C(1) << 30U;
const std::uint64_t kMiB = UINT64_C(1) << 20U;

template<class F> std::string outcome(F f) {
    try { return std::to_string(f()); } catch (const std::overflow_error& e) { return std::string("throw:") + e.what(); }
}

void move_equivalence() {
    namespace s = strix::serve;
    namespace app = strix::decode_cli;
    const std::vector<std::uint64_t> bytes = {0, 1, 2, 9, 10, 11, 12, 100, 101, 111, 112, 123, 1000,
        3 * kGiB, 26 * kGiB + kGiB / 2, 33 * kGiB + kGiB / 2, 121 * kGiB, UINT64_MAX - 1U, UINT64_MAX};
    const std::vector<Opt> pools = {std::nullopt, Opt{0}, Opt{1}, Opt{100}, Opt{112}, Opt{1000}, Opt{27 * kGiB},
        Opt{33 * kGiB}, Opt{121 * kGiB}, Opt{UINT64_MAX}};
    const std::vector<std::uint64_t> costs = {0, 1, 2, 10, 727 * kMiB, 2386 * kMiB, UINT64_MAX};
    const std::vector<std::uint64_t> rooms = {0, 100, 4393 * kMiB, UINT64_MAX};
    const std::vector<std::uint32_t> caps = {0, 1, 4, 8, 75, 1000, UINT32_MAX};
    std::uint64_t compared = 0;
    for (const auto retained : costs) for (const auto peak : costs) for (const auto& host : pools)
    for (const auto& device : pools) for (const auto room : rooms) {
        EXPECT(legacy::http_slot_memory_fits(retained, peak, host, device, room) ==
            s::slot_memory_fits(retained, peak, host, device, room));
        EXPECT(s::slot_memory_fits(retained, peak, host, device, room) ==
            app::http_slot_memory_fits(retained, peak, host, device, room));
        for (const auto cap : caps) {
            const auto old = legacy::resolve_http_slot_capacity(retained, peak, host, device, room, cap);
            const auto now = s::resolve_slot_capacity(retained, peak, host, device, room, cap);
            const auto fwd = app::resolve_http_slot_capacity(retained, peak, host, device, room, cap);
            EXPECT(old.slots == now.slots && old.limiting == now.limiting);
            EXPECT(fwd.slots == now.slots && fwd.limiting == now.limiting);
            ++compared;
        }
    }
    // Host + device pair with distinct shared and per-slot costs.
    const std::vector<std::uint64_t> small = {0, 2, 727 * kMiB, UINT64_MAX};
    for (const auto retained : costs) for (const auto shared_host : small) for (const auto shared_device : small)
    for (const auto slot_host : small) for (const auto slot_device : small) for (const auto& host : pools)
    for (const auto& device : pools) for (const auto room : rooms) for (const auto cap : {0U, 8U}) {
        const legacy::HttpServingMemory lcost{shared_host, shared_device, slot_host, slot_device};
        const s::ServingMemory cost{shared_host, shared_device, slot_host, slot_device};
        const auto old = legacy::resolve_http_bridge_capacity(retained, lcost, host, device, room, room / 2U, cap);
        const auto now = s::resolve_bridge_capacity(retained, cost, host, device, room, room / 2U, cap);
        const auto fwd = app::resolve_http_bridge_capacity(retained, cost, host, device, room, room / 2U, cap);
        const auto report = s::report_bridge_capacity(retained, cost, host, device, room, room / 2U, cap);
        EXPECT(old.slots == now.slots && old.limiting == now.limiting);
        EXPECT(fwd.slots == now.slots && fwd.limiting == now.limiting);
        EXPECT(report.slots == now.slots && report.limiting == now.limiting && report.terms.size() == 2U);
        // Without a cap the answer is the smaller of the two published terms.
        if (!cap && report.terms.size() == 2U && now.limiting != "shared cost overflow")
            EXPECT(report.slots == std::min(report.terms[0].slots, report.terms[1].slots));
        EXPECT(legacy::http_bridge_growth_fits(retained, lcost, host, device, room) ==
            s::bridge_growth_fits(retained, cost, host, device, room));
        EXPECT(s::bridge_growth_fits(retained, cost, host, device, room) ==
            app::http_bridge_growth_fits(retained, cost, host, device, room));
        ++compared;
    }
    for (const auto a : bytes) for (const auto b : bytes) for (const auto c : {UINT64_C(0), UINT64_C(7), UINT64_MAX}) {
        EXPECT(outcome([&] { return legacy::http_slot_retained_bytes(a, b, c, 1, 0); }) ==
            outcome([&] { return s::slot_retained_bytes(a, b, c, 1, 0); }));
        EXPECT(outcome([&] { return s::slot_retained_bytes(a, b, c, 1, 0); }) ==
            outcome([&] { return app::http_slot_retained_bytes(a, b, c, 1, 0); }));
        for (const auto d : {UINT64_C(0), UINT64_C(50), UINT64_MAX}) {
            EXPECT(outcome([&] { return legacy::http_shared_growth_headroom(a, b, c, d, 3); }) ==
                outcome([&] { return s::shared_growth_headroom(a, b, c, d, 3); }));
            EXPECT(outcome([&] { return s::shared_growth_headroom(a, b, c, d, 3); }) ==
                outcome([&] { return app::http_shared_growth_headroom(a, b, c, d, 3); }));
        }
        ++compared;
    }
    for (int mask = 0; mask < 8; ++mask) for (std::size_t active = 0; active < 4; ++active)
    for (std::size_t built = 0; built < 4; ++built) for (std::size_t capacity = 0; capacity < 5; ++capacity) {
        const bool queued = (mask & 1) != 0, memory_ok = (mask & 2) != 0, spare = (mask & 4) != 0;
        EXPECT(legacy::http_slot_should_grow(queued, active, built, capacity, memory_ok, spare) ==
            s::slot_should_grow(queued, active, built, capacity, memory_ok, spare));
        EXPECT(s::slot_should_grow(queued, active, built, capacity, memory_ok, spare) ==
            app::http_slot_should_grow(queued, active, built, capacity, memory_ok, spare));
    }
    for (const auto capacity : {0U, 1U, 3U, 4U, 5U, 16U, UINT32_MAX}) {
        EXPECT(legacy::http_eager_slots(capacity) == s::eager_slots(capacity));
        for (const auto eager : {0U, 1U, 4U, 8U, UINT32_MAX})
            EXPECT(legacy::http_eager_slots(capacity, eager) == s::eager_slots(capacity, eager) &&
                s::eager_slots(capacity, eager) == app::http_eager_slots(capacity, eager));
    }
    EXPECT(compared > 10000U);
    // The growth check reads fresh observations at each call.
    Opt host = 1000, device = 1000;
    auto check = s::make_bridge_growth_check(10, s::ServingMemory{0, 0, 2, 2}, 100, 50, 20,
        [&] { return s::BridgeGrowthObservation{host, device, 50, 20}; });
    auto old_name = app::make_http_bridge_growth_check(10, app::HttpServingMemory{0, 0, 2, 2}, 100, 50, 20,
        [&] { return app::HttpBridgeGrowthObservation{host, device, 50, 20}; });
    EXPECT(check() && old_name());
    host = 111;
    EXPECT(!check() && !old_name());
    host = 112;
    EXPECT(check() && old_name());
}

void report() {
    namespace s = strix::serve;
    // The measured case: 2.33 GB retained + 0.71 GB active peak per slot, 4.29 GB headroom.
    const s::SlotCost slot{2386 * kMiB, 727 * kMiB};
    const auto headroom = 4393 * kMiB;
    const auto at = [&](std::uint64_t host_gib, std::uint64_t device_gib) {
        return s::report_slot_capacity({{"host", host_gib * kGiB, headroom, slot},
            {"device", device_gib * kGiB, headroom, slot}});
    };
    const auto probe = at(28, 60);
    EXPECT(probe.slots == 8U && probe.limiting == "host budget");
    EXPECT(probe.terms.size() == 2U && probe.terms[0].slots == 8U && probe.terms[1].slots == 19U);
    EXPECT(probe.slots == s::resolve_slot_capacity(slot.retained_bytes, slot.active_peak_bytes,
        28 * kGiB, 60 * kGiB, headroom).slots);
    const auto free = at(92, 92);
    EXPECT(free.slots == 29U && free.limiting == "host budget");  // a tie names the first pool
    EXPECT(at(60, 28).limiting == "device budget");
    const auto capped = s::report_slot_capacity({{"host", 92 * kGiB, headroom, slot}}, 16U);
    EXPECT(capped.slots == 16U && capped.limiting == "requested cap" && capped.terms[0].slots == 29U);
    EXPECT(s::report_slot_capacity({{"host", 92 * kGiB, headroom, slot}}, 64U).slots == 29U);
    // Never a slot count from a missing observation or a missing cost.
    const auto blind = s::report_slot_capacity({{"host", 92 * kGiB, headroom, slot},
        {"device", std::nullopt, headroom, slot}});
    EXPECT(blind.slots == 0U && blind.limiting == "missing memory observations: device");
    EXPECT(s::report_slot_capacity({{"host", 92 * kGiB, headroom, {}}}).limiting == "missing slot cost: host");
    EXPECT(s::report_slot_capacity({{"host", 92 * kGiB, headroom, {UINT64_MAX, 1}}}).limiting == "cost overflow: host");
    EXPECT(s::report_slot_capacity({}).slots == 0U);
    EXPECT(s::report_slot_capacity({{"host", headroom - 1U, headroom, slot}}).slots == 0U);
    // Pools with different costs: persistent bytes in one pool, scratch in another.
    const auto split = s::report_slot_capacity({{"kv", 100, 0, {10, 0}}, {"scratch", 100, 0, {1, 30}}});
    EXPECT(split.slots == 3U && split.limiting == "scratch budget" && split.terms[0].slots == 11U);

    const auto line = s::format_capacity_report(s::report_slot_capacity(
        {{"host", 112, 100, {10, 2}}, {"device", std::nullopt, 7, {10, 2}}}));
    EXPECT(line == "serve-capacity slots=0 limiting=\"missing memory observations: device\""
        " host{available=112 headroom=100 retained=10 active_peak=2 slots=1}"
        " device{available=na headroom=7 retained=10 active_peak=2 slots=0}");
    const auto bridge = s::report_bridge_capacity(10, s::ServingMemory{5, 0, 2, 3}, 141, 1000, 100, 50, 0);
    EXPECT(s::format_capacity_report(bridge) == "serve-capacity slots=3 limiting=\"MemAvailable: MemAvailable budget\""
        " MemAvailable{available=141 headroom=105 retained=10 active_peak=2 slots=3}"
        " GTT{available=1000 headroom=50 retained=10 active_peak=3 slots=73}");
    EXPECT(s::capacity_report_json(bridge) ==
        "{\"slots\":3,\"limiting\":\"MemAvailable: MemAvailable budget\",\"terms\":["
        "{\"name\":\"MemAvailable\",\"available_bytes\":141,\"headroom_bytes\":105,\"retained_bytes\":10,"
        "\"active_peak_bytes\":2,\"slots\":3},"
        "{\"name\":\"GTT\",\"available_bytes\":1000,\"headroom_bytes\":50,\"retained_bytes\":10,"
        "\"active_peak_bytes\":3,\"slots\":73}]}");
    s::CapacityReport odd{1, "a \"quoted\" \\ term\n", {{"p", std::nullopt, 0, {}, 0}}};
    EXPECT(s::capacity_report_json(odd) == "{\"slots\":1,\"limiting\":\"a \\\"quoted\\\" \\\\ term \",\"terms\":["
        "{\"name\":\"p\",\"available_bytes\":null,\"headroom_bytes\":0,\"retained_bytes\":0,"
        "\"active_peak_bytes\":0,\"slots\":0}]}");
}

void census() {
    namespace s = strix::serve;
    const std::string meminfo =
        "MemTotal:       129930240 kB\nMemFree:          1048576 kB\nMemAvailable:   27787264 kB\n"
        "Cached:         31457280 kB\nSwapCached:            0 kB\nUnevictable:    30198988 kB\nMlocked:\t  16 kB\n"
        "HugePages_Total:       0\nBroken:  12 MB\nEmpty:\nHuge: 99999999999999999999 kB\nEdge: 18014398509481984 kB\n";
    s::MemoryCensus c;
    c.point = "probe";
    c.host_total = UINT64_C(129930240) * 1024U; c.host_available = UINT64_C(27787264) * 1024U;
    c.device_limit = 120 * kGiB; c.device_used = 31 * kGiB;
    s::append_kb_fields(c, meminfo, {"MemFree", "Cached", "Unevictable", "Mlocked", "Missing", "HugePages_Total",
        "Broken", "Empty", "Huge", "Edge", "Mem"});
    EXPECT(c.fields.size() == 4U);
    if (c.fields.size() == 4U) {
        EXPECT(c.fields[0].first == "MemFree" && c.fields[0].second == kGiB);
        EXPECT(c.fields[1].first == "Cached" && c.fields[1].second == 30 * kGiB);
        EXPECT(c.fields[3].first == "Mlocked" && c.fields[3].second == 16384U);
    }
    s::append_kb_fields(c, "Name:\tx\nVmRSS:\t 2097152 kB\nVmPin:\t 1024 kB", {"VmRSS", "VmPin"}, "self_");
    EXPECT(c.fields.size() == 6U);
    if (c.fields.size() == 6U) {
        EXPECT(c.fields[4].first == "self_VmRSS" && c.fields[4].second == 2 * kGiB);
        EXPECT(c.fields[5].first == "self_VmPin" && c.fields[5].second == 1048576U);
    }
    c.fields.emplace_back("registered", 7U);
    EXPECT(s::format_memory_census(c) == "serve-census point=probe host_total=133048565760 host_available=28454158336"
        " device_limit=128849018880 device_used=33285996544 device_free=95563022336"
        " MemFree=1073741824 Cached=32212254720 Unevictable=30923763712 Mlocked=16384"
        " self_VmRSS=2147483648 self_VmPin=1048576 registered=7");
    s::MemoryCensus before;
    before.point = "before_load";
    before.host_available = 121 * kGiB; before.device_used = kGiB;
    before.fields = {{"Cached", 2 * kGiB}, {"MemFree", 100 * kGiB}, {"only_before", 1}};
    EXPECT(s::format_census_delta(before, c) == "serve-census-delta from=before_load to=probe"
        " host_available=-101468602368 device_used=+32212254720 MemFree=-106300440576 Cached=+30064771072");
    s::MemoryCensus blind;
    blind.point = "x";
    EXPECT(s::format_memory_census(blind) ==
        "serve-census point=x host_total=na host_available=na device_limit=na device_used=na device_free=na");
    EXPECT(s::format_census_delta(blind, c).find("host_available=na device_used=na") != std::string::npos);
    blind.device_limit = 1; blind.device_used = 2;  // inconsistent counters: no free value
    EXPECT(s::format_memory_census(blind).find("device_free=na") != std::string::npos);
}

void switches() {
    namespace s = strix::serve;
    EXPECT(!s::slot_exact_from(nullptr) && !s::slot_exact_from("") && !s::slot_exact_from("0") &&
        !s::slot_exact_from("true") && !s::slot_exact_from("11") && s::slot_exact_from("1"));
    EXPECT(!s::census_enabled_from(nullptr) && !s::census_enabled_from("0") && s::census_enabled_from("1"));
}
} // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    move_equivalence();
    report();
    census();
    switches();
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::puts("slot capacity tests: ok");
    return 0;
}
