#pragma once
// Shared route receipt: which kernel family actually ran each product, which levers were requested versus effective,
// and a reason code for every decline or fallback. One process-wide table of relaxed atomic counters; a launch site
// pays a few integer increments on the host (no device work, no mutex, no output change).
//
//   route::launched(Product::moe, Family::iu4, rows);                      // after the enqueue
//   route::bypass(Product::dense, Family::f16_wmma, Reason::short_call, rows);  // wanted family not taken, and why
//   route::lever("glm.pf_wmma", "1", "1", Reason::selected);               // start-up: requested vs effective
//   const route::LayerScope scope(L, Reason::layer_mask);                  // attributes launches below to layer L
//
// json() is the one-line block "strix-route-receipt-v1" every CLI prints at exit (arm_exit_print(), armed by the
// device bind) and the server returns per request (delta of the request window) and in its status view.
// fingerprint() hashes the numerical levers; the two ranks of a pair compare it at start-up (compare_ranks /
// require_same_route) and a difference is an error. Switch: STRIX_ROUTE_RECEIPT=0 turns the counters and the printed
// block off; the lever list and the rank comparison stay. See docs/engine/route-receipt.md.

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "strix/env_flags.hpp"

namespace strix::route {

enum class Product : std::uint8_t { dense, attention, index, router, moe, moe_gate_up, moe_down, head, core, count_ };
enum class Family : std::uint8_t { exact, f16_wmma, iu4, fma_rows, fallback, count_ };
enum class Reason : std::uint8_t {
    selected, disabled, wrong_route, layer_mask, short_call, format, shape, dependency, assets, memory, build,
    qualification, nested, no_match, peer_disagreement, resource_limit, launch_error, count_
};

inline constexpr std::size_t kProducts = static_cast<std::size_t>(Product::count_);
inline constexpr std::size_t kFamilies = static_cast<std::size_t>(Family::count_);
inline constexpr std::size_t kReasons = static_cast<std::size_t>(Reason::count_);
inline constexpr std::size_t kLayers = 128;
inline constexpr const char* kSchema = "strix-route-receipt-v1";

inline const char* name(Product p) {
    static constexpr const char* n[kProducts] = {"dense", "attention", "index", "router", "moe", "moe_gate_up", "moe_down", "head", "core"};
    return n[static_cast<std::size_t>(p)];
}
inline const char* name(Family f) {
    static constexpr const char* n[kFamilies] = {"exact", "f16_wmma", "iu4", "fma_rows", "fallback"};
    return n[static_cast<std::size_t>(f)];
}
inline const char* name(Reason r) {
    static constexpr const char* n[kReasons] = {"selected", "disabled", "wrong_route", "layer_mask", "short_call", "format", "shape",
        "dependency", "assets", "memory", "build", "qualification", "nested", "no_match", "peer_disagreement", "resource_limit", "launch_error"};
    return n[static_cast<std::size_t>(r)];
}

struct Count { std::uint64_t launches = 0, rows = 0; };

// Plain copy of the counters (a snapshot, or the difference of two).
struct Totals {
    std::array<std::array<Count, kFamilies>, kProducts> product{};
    std::array<std::array<Count, kReasons>, kProducts> bypass{};
    std::array<std::array<std::uint8_t, kReasons>, kProducts> wanted{};  // family the bypassed call asked for
    std::array<std::array<Count, kFamilies>, kLayers> layer{};
    std::array<std::array<std::uint64_t, kReasons>, kLayers> layer_reason{};  // layer visits by reason
    [[nodiscard]] bool empty() const {
        for (const auto& p : product) for (const auto& c : p) if (c.launches) return false;
        for (const auto& p : bypass) for (const auto& c : p) if (c.launches) return false;
        return true;
    }
};

struct Lever {
    std::string name, requested, effective, detail;
    Reason reason = Reason::selected;
    bool numeric = true;  // part of the numerical-route fingerprint
};

namespace detail {
struct Cell { std::atomic<std::uint64_t> launches{0}, rows{0}; };
struct Table {
    Cell product[kProducts][kFamilies];
    Cell bypass[kProducts][kReasons];
    std::atomic<std::uint8_t> wanted[kProducts][kReasons];
    Cell layer[kLayers][kFamilies];
    std::atomic<std::uint64_t> layer_reason[kLayers][kReasons];
};
inline Table g_table;
inline std::atomic<int> g_on{-1};
inline std::atomic<int> g_rank{-1};
inline thread_local int t_layer = -1;
// Never destroyed: the exit print reads them after other statics are gone.
inline std::mutex& lever_mutex() { static auto* m = new std::mutex; return *m; }
inline std::vector<Lever>& levers() { static auto* v = new std::vector<Lever>; return *v; }
inline char g_engine[32] = {};  // written once at start-up under lever_mutex
inline int resolve_on() {
    const int on = envcfg::flag("STRIX_ROUTE_RECEIPT", true) ? 1 : 0;
    g_on.store(on, std::memory_order_relaxed);
    return on;
}
inline void add(Cell& c, std::uint64_t rows) {
    c.launches.fetch_add(1, std::memory_order_relaxed);
    c.rows.fetch_add(rows, std::memory_order_relaxed);
}
inline void put(std::string& out, std::string_view text) {
    out += '"';
    for (const char ch : text) {
        if (ch == '"' || ch == '\\') { out += '\\'; out += ch; }
        else if (static_cast<unsigned char>(ch) < 0x20U) out += ' ';
        else out += ch;
    }
    out += '"';
}
inline std::uint64_t fnv(std::string_view text, std::uint64_t h) {
    for (const char ch : text) { h ^= static_cast<unsigned char>(ch); h *= 1099511628211ULL; }
    return h;
}
inline std::string hex(std::uint64_t v) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
    return b;
}
}  // namespace detail

// Counters on? Resolved once from STRIX_ROUTE_RECEIPT (default on); set_enabled is for tests and A/B arms.
inline bool enabled() {
    const int on = detail::g_on.load(std::memory_order_relaxed);
    return (on < 0 ? detail::resolve_on() : on) != 0;
}
inline void set_enabled(bool on) { detail::g_on.store(on ? 1 : 0, std::memory_order_relaxed); }
inline void set_rank(int rank) { detail::g_rank.store(rank, std::memory_order_relaxed); }
inline int rank() { return detail::g_rank.load(std::memory_order_relaxed); }
inline void set_engine(std::string_view engine) {
    std::lock_guard<std::mutex> lock(detail::lever_mutex());
    const std::size_t n = engine.size() < sizeof detail::g_engine - 1 ? engine.size() : sizeof detail::g_engine - 1;
    std::memcpy(detail::g_engine, engine.data(), n);
    detail::g_engine[n] = 0;
}

// One executed launch (or one logical call that enqueued a family) of `rows` logical rows.
inline void launched(Product p, Family f, std::uint64_t rows) {
    if (!enabled()) return;
    detail::add(detail::g_table.product[static_cast<std::size_t>(p)][static_cast<std::size_t>(f)], rows);
    if (const int L = detail::t_layer; L >= 0) detail::add(detail::g_table.layer[static_cast<std::size_t>(L)][static_cast<std::size_t>(f)], rows);
}
// A call that wanted family `wanted` and did not get it, with the reason of the deciding branch.
inline void bypass(Product p, Family wanted, Reason why, std::uint64_t rows) {
    if (!enabled()) return;
    const auto pi = static_cast<std::size_t>(p), ri = static_cast<std::size_t>(why);
    detail::add(detail::g_table.bypass[pi][ri], rows);
    detail::g_table.wanted[pi][ri].store(static_cast<std::uint8_t>(wanted), std::memory_order_relaxed);
}

// Attributes the launches of this thread to one layer until the scope ends, and counts why the layer runs the
// family it runs (selected = the requested family).
class LayerScope {
public:
    LayerScope(unsigned layer, Reason why) : previous_(detail::t_layer) {
        if (layer >= kLayers || !enabled()) return;
        detail::t_layer = static_cast<int>(layer);
        detail::g_table.layer_reason[layer][static_cast<std::size_t>(why)].fetch_add(1, std::memory_order_relaxed);
    }
    ~LayerScope() { detail::t_layer = previous_; }
    LayerScope(const LayerScope&) = delete;
    LayerScope& operator=(const LayerScope&) = delete;
private:
    int previous_;
};

// Start-up: one lever, requested versus effective. The same name replaces its earlier record.
inline void lever(std::string_view lever_name, std::string_view requested, std::string_view effective, Reason why = Reason::selected,
                  bool numeric = true, std::string_view detail_text = {}) {
    std::lock_guard<std::mutex> lock(detail::lever_mutex());
    Lever next{std::string(lever_name), std::string(requested), std::string(effective), std::string(detail_text), why, numeric};
    for (auto& l : detail::levers()) if (l.name == next.name) { l = std::move(next); return; }
    detail::levers().push_back(std::move(next));
}
inline std::vector<Lever> levers() {
    std::lock_guard<std::mutex> lock(detail::lever_mutex());
    return detail::levers();
}

// Hash of the effective value of every numerical lever, independent of the order they were recorded in.
inline std::uint64_t fingerprint(const std::vector<Lever>& list) {
    std::uint64_t sum = 0, n = 0;
    for (const auto& l : list) {
        if (!l.numeric) continue;
        sum += detail::fnv(l.effective, detail::fnv("=", detail::fnv(l.name, 1469598103934665603ULL)));
        ++n;
    }
    return detail::fnv(std::to_string(n), sum ^ 1469598103934665603ULL);
}
inline std::uint64_t fingerprint() { return fingerprint(levers()); }

// The numerical levers as "name=effective" lines (what a rank shows when the fingerprints differ).
inline std::string numeric_levers(const std::vector<Lever>& list) {
    std::string out;
    for (const auto& l : list) if (l.numeric) out += (out.empty() ? "" : ", ") + l.name + "=" + l.effective;
    return out;
}

// Throws when the peer rank resolved a different numerical route.
inline void require_same_route(std::uint64_t local, std::uint64_t peer, const std::string& local_levers = {}) {
    if (local == peer) return;
    throw std::runtime_error("route receipt: the numerical-route fingerprint differs between the ranks (local " + detail::hex(local) +
                             ", peer " + detail::hex(peer) + "); local levers: " + (local_levers.empty() ? "none" : local_levers));
}
// Start-up comparison over a symmetric host exchange: exchange(out, in, bytes) sends 16 bytes and receives the peer's.
template <class Exchange>
inline std::uint64_t compare_ranks(Exchange&& exchange) {
    const auto list = levers();
    const std::uint64_t out[2] = {0x3154504352545352ULL, fingerprint(list)};  // "RSTRCPT1"
    std::uint64_t in[2] = {0, 0};
    exchange(static_cast<const void*>(out), static_cast<void*>(in), sizeof out);
    if (in[0] != out[0]) throw std::runtime_error("route receipt: the peer rank did not send a route fingerprint");
    require_same_route(out[1], in[1], numeric_levers(list));
    return out[1];
}

// Approximate process-window telemetry: independent relaxed loads are not a coherent transaction.
// Concurrent launches can advance rows, calls and visits at different instants; request windows overlap.
inline Totals snapshot() {
    Totals t;
    const auto& g = detail::g_table;
    const auto get = [](const detail::Cell& c) { return Count{c.launches.load(std::memory_order_relaxed), c.rows.load(std::memory_order_relaxed)}; };
    for (std::size_t p = 0; p < kProducts; ++p) {
        for (std::size_t f = 0; f < kFamilies; ++f) t.product[p][f] = get(g.product[p][f]);
        for (std::size_t r = 0; r < kReasons; ++r) {
            t.bypass[p][r] = get(g.bypass[p][r]);
            t.wanted[p][r] = g.wanted[p][r].load(std::memory_order_relaxed);
        }
    }
    for (std::size_t l = 0; l < kLayers; ++l) {
        for (std::size_t f = 0; f < kFamilies; ++f) t.layer[l][f] = get(g.layer[l][f]);
        for (std::size_t r = 0; r < kReasons; ++r) t.layer_reason[l][r] = g.layer_reason[l][r].load(std::memory_order_relaxed);
    }
    return t;
}
// now - before (the counters of a window). Wanted families are those of `now`.
inline Totals delta(const Totals& now, const Totals& before) {
    Totals d = now;
    const auto sub = [](Count& a, const Count& b) { a.launches -= b.launches; a.rows -= b.rows; };
    for (std::size_t p = 0; p < kProducts; ++p) {
        for (std::size_t f = 0; f < kFamilies; ++f) sub(d.product[p][f], before.product[p][f]);
        for (std::size_t r = 0; r < kReasons; ++r) sub(d.bypass[p][r], before.bypass[p][r]);
    }
    for (std::size_t l = 0; l < kLayers; ++l) {
        for (std::size_t f = 0; f < kFamilies; ++f) sub(d.layer[l][f], before.layer[l][f]);
        for (std::size_t r = 0; r < kReasons; ++r) d.layer_reason[l][r] -= before.layer_reason[l][r];
    }
    return d;
}
// Test helper: zero the counters and forget the levers.
inline void reset() {
    auto& g = detail::g_table;
    const auto zero = [](detail::Cell& c) { c.launches.store(0); c.rows.store(0); };
    for (std::size_t p = 0; p < kProducts; ++p) {
        for (std::size_t f = 0; f < kFamilies; ++f) zero(g.product[p][f]);
        for (std::size_t r = 0; r < kReasons; ++r) { zero(g.bypass[p][r]); g.wanted[p][r].store(0); }
    }
    for (std::size_t l = 0; l < kLayers; ++l) {
        for (std::size_t f = 0; f < kFamilies; ++f) zero(g.layer[l][f]);
        for (std::size_t r = 0; r < kReasons; ++r) g.layer_reason[l][r].store(0);
    }
    std::lock_guard<std::mutex> lock(detail::lever_mutex());
    detail::levers().clear();
}

// {"schema":"strix-route-receipt-v1",...}; one line, no trailing newline. `scope` says what the counters cover
// ("process" or "request"); a request window on a server with concurrent requests also holds their launches.
// full = false leaves the lever and switch lists out (the per-request block; the fingerprint names the route).
inline std::string json(const Totals& t, std::string_view scope = "process", bool full = true) {
    const auto list = levers();
    std::string out = "{\"schema\":\"";
    out += kSchema;
    out += "\",\"scope\":";
    detail::put(out, scope);
    {
        std::lock_guard<std::mutex> lock(detail::lever_mutex());
        if (detail::g_engine[0] != 0) { out += ",\"engine\":"; detail::put(out, detail::g_engine); }
    }
    if (rank() >= 0) out += ",\"rank\":" + std::to_string(rank());
    out += ",\"counters\":";
    out += enabled() ? "true" : "false";
    out += ",\"fingerprint\":\"" + detail::hex(fingerprint(list)) + "\"";
    bool first = true;
    if (full) out += ",\"levers\":[";
    if (full) for (const auto& l : list) {
        if (!first) out += ',';
        first = false;
        out += "{\"name\":"; detail::put(out, l.name);
        out += ",\"requested\":"; detail::put(out, l.requested);
        out += ",\"effective\":"; detail::put(out, l.effective);
        out += ",\"reason\":\""; out += name(l.reason); out += '"';
        if (!l.numeric) out += ",\"numeric\":false";
        if (!l.detail.empty()) { out += ",\"detail\":"; detail::put(out, l.detail); }
        out += '}';
    }
    // The governed switch registry: what the process was asked for explicitly; the rest ran on defaults.
    if (full) out += "],\"switches\":{\"set\":{";
    std::size_t defaults = 0;
    first = true;
    if (full) for (const auto& e : envcfg::registered()) {
        if (!e.set) { ++defaults; continue; }
        if (!first) out += ',';
        first = false;
        detail::put(out, e.name); out += ':'; detail::put(out, e.value);
    }
    if (full) out += "},\"defaults\":" + std::to_string(defaults) + "}";
    out += ",\"products\":{";
    first = true;
    for (std::size_t p = 0; p < kProducts; ++p) {
        bool any = false;
        for (std::size_t f = 0; f < kFamilies; ++f) {
            const Count& c = t.product[p][f];
            if (!c.launches) continue;
            out += any ? "," : (first ? "" : ",");
            if (!any) { out += '"'; out += name(static_cast<Product>(p)); out += "\":{"; }
            any = true; first = false;
            out += '"'; out += name(static_cast<Family>(f));
            out += "\":{\"launches\":" + std::to_string(c.launches) + ",\"rows\":" + std::to_string(c.rows) + "}";
        }
        if (any) out += '}';
    }
    out += "},\"bypass\":[";
    first = true;
    for (std::size_t p = 0; p < kProducts; ++p)
        for (std::size_t r = 0; r < kReasons; ++r) {
            const Count& c = t.bypass[p][r];
            if (!c.launches) continue;
            if (!first) out += ',';
            first = false;
            out += "{\"product\":\""; out += name(static_cast<Product>(p));
            out += "\",\"wanted\":\""; out += name(static_cast<Family>(t.wanted[p][r] < kFamilies ? t.wanted[p][r] : 0));
            out += "\",\"reason\":\""; out += name(static_cast<Reason>(r));
            out += "\",\"calls\":" + std::to_string(c.launches) + ",\"rows\":" + std::to_string(c.rows) + "}";
        }
    // Layers: runs of consecutive layers that ran the same families for the same reasons.
    out += "],\"layers\":[";
    first = true;
    const auto mask = [&](std::size_t l) { unsigned m = 0; for (std::size_t f = 0; f < kFamilies; ++f) if (t.layer[l][f].launches) m |= 1U << f; return m; };
    const auto why = [&](std::size_t l) { unsigned m = 0; for (std::size_t r = 0; r < kReasons; ++r) if (t.layer_reason[l][r]) m |= 1U << r; return m; };
    for (std::size_t l = 0; l < kLayers;) {
        const unsigned m = mask(l);
        if (!m) { ++l; continue; }
        std::size_t end = l + 1;
        const unsigned w = why(l);
        while (end < kLayers && mask(end) == m && why(end) == w) ++end;
        if (!first) out += ',';
        first = false;
        out += "{\"layers\":\"" + std::to_string(l) + (end - l > 1 ? "-" + std::to_string(end - 1) : "") + "\",\"reasons\":{";
        bool first_reason = true;
        for (std::size_t r = 0; r < kReasons; ++r) {
            if (!(w & (1U << r))) continue;
            std::uint64_t visits = 0;
            for (std::size_t k = l; k < end; ++k) visits += t.layer_reason[k][r];
            out += first_reason ? "\"" : ",\"";
            first_reason = false;
            out += name(static_cast<Reason>(r));
            out += "\":" + std::to_string(visits);
        }
        out += '}';
        for (std::size_t f = 0; f < kFamilies; ++f) {
            if (!(m & (1U << f))) continue;
            Count sum;
            for (std::size_t k = l; k < end; ++k) { sum.launches += t.layer[k][f].launches; sum.rows += t.layer[k][f].rows; }
            out += ",\""; out += name(static_cast<Family>(f));
            out += "\":{\"launches\":" + std::to_string(sum.launches) + ",\"rows\":" + std::to_string(sum.rows) + "}";
        }
        out += '}';
        l = end;
    }
    out += "]}";
    return out;
}
inline std::string json() { return json(snapshot()); }

// Prints "strix-route-receipt {json}" on stderr when the process exits (once per process; nothing with the
// receipt off). The device bind arms it, so every binary that binds a device prints it; an engine without family
// hooks still reports its levers and the explicitly set switches.
inline void arm_exit_print() {
    static std::once_flag once;
    std::call_once(once, [] {
        if (!enabled()) return;
        std::atexit([] {
            if (!enabled()) return;
            std::fprintf(stderr, "strix-route-receipt %s\n", json(snapshot()).c_str());
            std::fflush(stderr);
        });
    });
}

}  // namespace strix::route
