#include "strix/deploy/memory_plan.hpp"

#include "strix/device/profiles.hpp"
#include "strix/model_description.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace strix::deploy {
namespace {

constexpr std::uint64_t kNoCap = std::numeric_limits<std::uint64_t>::max();

std::uint64_t sat_add(std::uint64_t a, std::uint64_t b) { return a > kNoCap - b ? kNoCap : a + b; }
std::uint64_t sat_mul(std::uint64_t a, std::uint64_t b) { return (a && b > kNoCap / a) ? kNoCap : a * b; }
std::uint64_t ceil_div(std::uint64_t a, std::uint64_t b) { return b ? (a + b - 1) / b : a; }

std::string gib(std::uint64_t bytes) {
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(2);
    os << static_cast<double>(bytes) / 1073741824.0;
    return os.str();
}

std::string q(const std::string& s) {
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out + "\"";
}

struct Shape {
    std::uint32_t positions, slots;
};

std::uint64_t sessions(const RouteNeeds& n, Shape s) {
    return sat_mul(s.slots, sat_add(n.session_fixed, sat_mul(s.positions, n.session_per_position)));
}

} // namespace

RouteNeeds parse_route_needs(std::string_view toml) {
    const auto parsed = model_description::parse(toml);
    RouteNeeds n;
    const auto fail = [](const std::string& key, const char* what) { throw std::invalid_argument("route needs: " + key + ": " + what); };
    const auto u64 = [&](const std::string& key, const model_description::Value& v) {
        const auto* x = std::get_if<std::int64_t>(&v.data);
        if (!x || *x < 0) fail(key, "expected a nonnegative integer");
        return static_cast<std::uint64_t>(*x);
    };
    const auto u32 = [&](const std::string& key, const model_description::Value& v) {
        const auto x = u64(key, v);
        if (x > 0xffffffffULL) fail(key, "out of range");
        return static_cast<std::uint32_t>(x);
    };
    const auto flag = [&](const std::string& key, const model_description::Value& v) {
        const auto* x = std::get_if<bool>(&v.data);
        if (!x) fail(key, "expected true or false");
        return *x;
    };
    bool schema = false;
    for (const auto& [key, v] : parsed.fields) {
        if (key == "schema") {
            const auto* s = std::get_if<std::string>(&v.data);
            if (!s || *s != "strix-route-needs-v1") fail(key, "expected strix-route-needs-v1");
            schema = true;
        } else if (key == "route") {
            const auto* s = std::get_if<std::string>(&v.data);
            if (!s) fail(key, "expected a string");
            n.route = *s;
        } else if (key == "device_fixed") n.device_fixed = u64(key, v);
        else if (key == "weights_resident") n.weights_resident = u64(key, v);
        else if (key == "tier_units") n.tier_units = u64(key, v);
        else if (key == "tier_unit_bytes") n.tier_unit_bytes = u64(key, v);
        else if (key == "min_device_units") n.min_device_units = u64(key, v);
        else if (key == "host_tier") n.host_tier = flag(key, v);
        else if (key == "file_tier") n.file_tier = flag(key, v);
        else if (key == "host_fixed") n.host_fixed = u64(key, v);
        else if (key == "session_fixed") n.session_fixed = u64(key, v);
        else if (key == "session_per_position") n.session_per_position = u64(key, v);
        else if (key == "positions") n.positions = u32(key, v);
        else if (key == "min_positions") n.min_positions = u32(key, v);
        else if (key == "slots") n.slots = u32(key, v);
        else if (key == "min_slots") n.min_slots = u32(key, v);
        else if (key == "margin_bytes") n.margin_bytes = u64(key, v);
        else if (key == "bytes_per_token") n.bytes_per_token = u64(key, v);
        else if (key == "tiered_bytes_per_token") n.tiered_bytes_per_token = u64(key, v);
        else if (key == "hop_bytes_per_token") n.hop_bytes_per_token = u64(key, v);
        else if (key == "decode_efficiency") {
            if (const auto* d = std::get_if<double>(&v.data)) n.decode_efficiency = *d;
            else fail(key, "expected a number");
            if (!(n.decode_efficiency >= 0.0 && n.decode_efficiency <= 1.0)) fail(key, "expected 0 to 1");
        } else fail(key, "unknown key");
    }
    if (!schema) fail("schema", "required (strix-route-needs-v1)");
    if (n.route.empty()) fail("route", "required");
    return n;
}

MemoryPlan plan_memory(const device::MemoryBudget& budget, const RouteNeeds& n) {
    MemoryPlan p;
    p.budget = budget;
    p.route = n.route;
    p.device_fixed = n.device_fixed;
    p.weights_resident = n.weights_resident;
    p.margin_bytes = n.margin_bytes;
    p.unit_bytes = n.tier_unit_bytes;
    const bool tiered = n.host_tier || n.file_tier;
    // Without a host or file tier every tiered unit must sit on the device.
    const std::uint64_t min_units = tiered ? std::min(n.min_device_units, n.tier_units) : n.tier_units;
    const std::uint64_t device = budget.device_capped() ? budget.device_bytes
                                 : budget.physical_device_bytes ? budget.physical_device_bytes : kNoCap;
    const auto mandatory = [&](Shape s) {
        return sat_add(sat_add(sat_add(n.device_fixed, n.weights_resident), sat_add(sessions(n, s), n.margin_bytes)),
                       sat_mul(min_units, n.tier_unit_bytes));
    };
    const std::uint32_t min_positions = std::min(n.min_positions ? n.min_positions : n.positions, n.positions);
    const std::uint32_t min_slots = std::max<std::uint32_t>(1, std::min(n.min_slots, n.slots));
    const Shape smallest{min_positions, min_slots};
    p.min_device_bytes = mandatory(smallest);

    // Positions are kept and slots drop first; then positions halve (not below min_positions).
    bool fit = false;
    Shape chosen{n.positions, std::max<std::uint32_t>(1, n.slots)};
    for (std::uint32_t pos = n.positions;; pos = std::max(min_positions, pos / 2)) {
        for (std::uint32_t slots = std::max<std::uint32_t>(1, n.slots); slots >= min_slots && !fit; --slots)
            if (mandatory({pos, slots}) <= device) {
                chosen = {pos, slots};
                fit = true;
            }
        if (fit || pos == min_positions) break;
    }
    if (!fit) {
        p.refused = true;
        p.refusal = n.route + ": needs at least " + gib(p.min_device_bytes) + " GiB of device memory at " +
                    std::to_string(min_positions) + " positions x " + std::to_string(min_slots) + " slot(s) (" +
                    (budget.device_capped() ? budget.device_source + " gives " + gib(budget.device_bytes) + " GiB" : "the card") +
                    ")";
        return p;
    }
    p.positions = chosen.positions;
    p.slots = chosen.slots;
    p.sessions_bytes = sessions(n, chosen);
    if (chosen.slots < n.slots)
        p.notes.push_back("session slots " + std::to_string(n.slots) + " -> " + std::to_string(chosen.slots) + " (device memory)");
    if (chosen.positions < n.positions)
        p.notes.push_back("positions " + std::to_string(n.positions) + " -> " + std::to_string(chosen.positions) +
                          " (device memory at 1 slot)");

    const std::uint64_t used = mandatory(chosen);
    const std::uint64_t spare_units = n.tier_unit_bytes ? (device == kNoCap ? kNoCap : (device - used) / n.tier_unit_bytes) : 0;
    p.units_device = std::min(n.tier_units, sat_add(min_units, spare_units));
    std::uint64_t rest = n.tier_units - p.units_device;
    p.device_used = sat_add(used, sat_mul(p.units_device - min_units, n.tier_unit_bytes));

    const std::uint64_t host = budget.host_capped() ? budget.host_bytes : kNoCap;
    if (host != kNoCap && n.host_fixed > host) {
        p.refused = true;
        p.min_host_bytes = n.host_fixed;
        p.refusal = n.route + ": the process needs " + gib(n.host_fixed) + " GiB of host memory before any weight tier (" +
                    budget.host_source + " gives " + gib(host) + " GiB)";
        return p;
    }
    if (n.host_tier && rest) {
        const std::uint64_t room = host == kNoCap ? kNoCap : host - n.host_fixed;
        p.units_host = std::min(rest, n.tier_unit_bytes ? room / n.tier_unit_bytes : 0);
        rest -= p.units_host;
    }
    p.host_used = sat_add(n.host_fixed, sat_mul(p.units_host, n.tier_unit_bytes));
    if (rest && !n.file_tier) {
        p.refused = true;
        p.min_host_bytes = sat_add(n.host_fixed, sat_mul(n.tier_units - p.units_device, n.tier_unit_bytes));
        p.refusal = n.route + ": " + std::to_string(rest) + " weight units fit neither the device nor the host; the minimum is " +
                    gib(p.min_host_bytes) + " GiB of host memory at this device budget";
        return p;
    }
    p.units_file = rest;
    if (n.tier_units) {
        p.notes.push_back(std::to_string(p.units_device) + " of " + std::to_string(n.tier_units) + " weight units in VRAM (" +
                          gib(sat_mul(p.units_device, n.tier_unit_bytes)) + " GiB)");
        if (p.units_host)
            p.notes.push_back(std::to_string(p.units_host) + " units in pinned host memory (" +
                              gib(sat_mul(p.units_host, n.tier_unit_bytes)) + " GiB)");
        if (p.units_file)
            p.notes.push_back(std::to_string(p.units_file) + " units streamed from the model file (" +
                              gib(sat_mul(p.units_file, n.tier_unit_bytes)) + " GiB)");
    }
    return p;
}

std::string MemoryPlan::json() const {
    std::ostringstream os;
    os << "{\"schema\":\"strix-memory-plan-v1\",\"route\":" << q(route) << ",\"budget\":" << budget.json()
       << ",\"refused\":" << (refused ? "true" : "false") << ",\"refusal\":" << q(refusal)
       << ",\"min_device_bytes\":" << min_device_bytes << ",\"min_host_bytes\":" << min_host_bytes
       << ",\"positions\":" << positions << ",\"slots\":" << slots << ",\"device\":{\"fixed\":" << device_fixed
       << ",\"weights_resident\":" << weights_resident << ",\"sessions\":" << sessions_bytes << ",\"margin\":" << margin_bytes
       << ",\"used\":" << device_used << "},\"units\":{\"bytes\":" << unit_bytes << ",\"device\":" << units_device
       << ",\"host_pinned\":" << units_host << ",\"file\":" << units_file << "},\"host_used\":" << host_used << ",\"notes\":[";
    for (std::size_t i = 0; i < notes.size(); ++i) os << (i ? "," : "") << q(notes[i]);
    os << "]}";
    return os.str();
}

ShapePlan plan_shape(const device::Descriptor& shape, const device::Descriptor* host, const RouteNeeds& needs) {
    ShapePlan s;
    s.count = std::max<std::uint32_t>(1, shape.count);
    s.shape = std::to_string(s.count) + "x " + (shape.profile.empty() ? shape.id : shape.profile) +
              (shape.interconnect.empty() ? std::string() : " over " + shape.interconnect);
    const std::uint64_t n = s.count;
    RouteNeeds per = needs;
    per.weights_resident = ceil_div(needs.weights_resident, n);
    per.tier_units = ceil_div(needs.tier_units, n);
    per.min_device_units = ceil_div(needs.min_device_units, n);
    per.session_fixed = ceil_div(needs.session_fixed, n);
    per.session_per_position = ceil_div(needs.session_per_position, n);
    auto budget = device::resolve_budget(shape, host, 0);
    if (budget.host_capped()) budget.host_bytes /= n; // one host feeds every card of the shape
    s.per_device = plan_memory(budget, per);
    s.per_device.route = needs.route + " (per device of " + s.shape + ")";
    const double bw = budget.bandwidth_gbytes_ps * 1e9 * needs.decode_efficiency;
    if (s.per_device.refused || bw <= 0 || needs.bytes_per_token == 0) return s;
    const double t_mem = static_cast<double>(needs.bytes_per_token) / bw;
    double t_spill = 0;
    const auto* profile = device::find_profile(shape.profile);
    if (per.tier_units && profile && !profile->unified) {
        const auto* link = device::find_interconnect(profile->host_link);
        const double frac = static_cast<double>(s.per_device.units_host + s.per_device.units_file) / static_cast<double>(per.tier_units);
        if (link) t_spill = static_cast<double>(needs.tiered_bytes_per_token) * frac / (link->gbytes_ps * 1e9);
    }
    double t_hop = 0;
    if (s.count > 1)
        if (const auto* ic = device::find_interconnect(shape.interconnect))
            t_hop = static_cast<double>(s.count - 1) * (static_cast<double>(needs.hop_bytes_per_token) / (ic->gbytes_ps * 1e9) + ic->rtt_us * 1e-6);
    s.spill_ms = t_spill * 1e3;
    s.hop_ms = t_hop * 1e3;
    s.predicted_ms_per_token = (t_mem + t_spill + t_hop) * 1e3;
    s.predicted_decode_tok_s = 1e3 / s.predicted_ms_per_token;
    return s;
}

std::string ShapePlan::json() const {
    std::ostringstream os;
    os.precision(4);
    os << "{\"schema\":\"strix-shape-plan-v1\",\"shape\":" << q(shape) << ",\"count\":" << count
       << ",\"predicted\":{\"model\":\"bandwidth\",\"decode_c1_tok_s\":" << predicted_decode_tok_s
       << ",\"ms_per_token\":" << predicted_ms_per_token << ",\"spill_ms\":" << spill_ms << ",\"hop_ms\":" << hop_ms
       << "},\"per_device\":" << per_device.json() << "}";
    return os.str();
}

} // namespace strix::deploy
