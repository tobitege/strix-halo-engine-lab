#pragma once
// Per-request session reservation shared by every serving backend (HIP, CUDA) and by each family's CLI.
//
// A session's device state is fixed bytes (recurrent state, carries) plus bytes per position (KV, index, pooled keys,
// draft caches). Sizing every session at the context cap makes the slot count a function of the cap, not of the work:
// at 34,816 positions one CUDA Qwen session is 1.27 GiB and three fit, while a 1K-prompt, 128-token request needs
// 1,280 positions (0.27 GiB). Here a request reserves what it can write (prompt + max_tokens + the engine's slack),
// rounded to a granule and capped at the context, and the slots are whatever the free memory holds.
//
// Pure host code: no device types. Engines supply the byte function (the same one that sizes their allocations) and the
// free-memory observation; the decision, its reason string and the report live here. Scheduling only: a request's
// tokens never depend on the capacity it was given (positions beyond what it writes are never read).
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::serve {

struct ReservationRule {
    std::uint64_t max_positions = 0;  // context cap: no session is larger
    std::uint64_t granule = 256;      // capacities are multiples of this (requests of similar size reuse a session)
    std::uint64_t slack = 0;          // positions the engine may write past prompt + max_tokens (verify window, drafts)
    std::uint64_t floor = 0;          // smallest capacity (0: one granule)
};

// Positions one request reserves: prompt + max_tokens + slack, rounded up to the granule, within [floor, max_positions].
inline std::uint64_t reservation_positions(const ReservationRule& r, std::uint64_t prompt, std::uint64_t max_tokens) {
    if (r.max_positions == 0) throw std::invalid_argument("reservation rule without max_positions");
    const std::uint64_t g = r.granule ? r.granule : 1U;
    const std::uint64_t lo = r.floor ? r.floor : g;
    std::uint64_t need = prompt;
    for (const auto add : {max_tokens, r.slack}) need = add > r.max_positions - std::min(need, r.max_positions) ? r.max_positions : need + add;
    if (need < lo) need = lo;
    if (need >= r.max_positions) return r.max_positions;
    const std::uint64_t up = (need + g - 1U) / g * g;
    return up < r.max_positions ? up : r.max_positions;
}

// One slot as the admission sees it.
struct SlotSession {
    bool busy = false;          // runs a request
    bool built = false;         // owns a session
    bool pinned = false;        // its session can be resized but never released (e.g. an engine's session 0)
    std::uint64_t positions = 0;
    std::uint64_t bytes = 0;    // device bytes of its session (0 when not built)
};

enum class AdmitAction {
    reuse,   // slot's session already holds the request
    resize,  // slot's idle session is re-sized to `positions` (its state is discarded)
    build,   // a new session of `positions` in a slot without one
    wait,    // does not fit now; retry when a running request ends
    reject,  // never fits (nothing runs and every idle session is already counted as released)
};

struct AdmitPlan {
    AdmitAction action = AdmitAction::wait;
    std::size_t slot = 0;
    std::uint64_t positions = 0, bytes = 0;  // the session the request gets
    std::vector<std::size_t> release;        // idle, unpinned sessions to free first (largest first)
    std::string why;                         // observable reason (every branch)
};

using SessionBytes = std::function<std::uint64_t(std::uint64_t positions)>;

inline const char* admit_action_name(AdmitAction a) {
    switch (a) {
    case AdmitAction::reuse: return "reuse";
    case AdmitAction::resize: return "resize";
    case AdmitAction::build: return "build";
    case AdmitAction::wait: return "wait";
    case AdmitAction::reject: return "reject";
    }
    return "?";
}

// Where a request of `need` positions goes. free_bytes: the device's free memory now; headroom: what must stay free
// after the allocation (e.g. the WDDM spill margin). Order of preference:
//   1. an idle session that already holds it (the smallest such: large sessions stay for long requests);
//   2. a new session in an empty slot (keeps the idle sessions), if free memory holds it;
//   3. re-size an idle session that is too small (largest first: it returns the most bytes);
//   4. 2 or 3 after releasing idle unpinned sessions, largest first, only as many as needed;
//   5. wait while something runs, else reject.
inline AdmitPlan plan_admission(const std::vector<SlotSession>& slots, std::uint64_t need, const SessionBytes& bytes_at,
                                std::uint64_t free_bytes, std::uint64_t headroom) {
    AdmitPlan p;
    p.positions = need;
    bool any_busy = false, any_free = false;
    std::size_t reuse = slots.size(), empty = slots.size(), grow = slots.size();
    for (std::size_t i = 0; i < slots.size(); ++i) {
        const auto& s = slots[i];
        any_busy = any_busy || s.busy;
        if (s.busy) continue;
        any_free = true;
        if (!s.built) { if (empty == slots.size()) empty = i; continue; }
        if (s.positions >= need) { if (reuse == slots.size() || s.positions < slots[reuse].positions) reuse = i; }
        else if (grow == slots.size() || s.bytes > slots[grow].bytes) grow = i;
    }
    if (!any_free) { p.why = "every slot runs a request"; return p; }
    if (reuse != slots.size()) {
        p.action = AdmitAction::reuse;
        p.slot = reuse;
        p.positions = slots[reuse].positions;
        p.bytes = slots[reuse].bytes;
        p.why = "idle session of " + std::to_string(p.positions) + " positions holds " + std::to_string(need);
        return p;
    }
    p.bytes = bytes_at(need);
    const auto fits = [&](std::uint64_t avail) { return avail >= headroom && avail - headroom >= p.bytes; };
    const auto mib = [](std::uint64_t b) { return std::to_string(b >> 20U); };
    // idle unpinned sessions other than `keep`, largest first
    const auto releasable = [&](std::size_t keep) {
        std::vector<std::size_t> r;
        for (std::size_t i = 0; i < slots.size(); ++i)
            if (i != keep && !slots[i].busy && slots[i].built && !slots[i].pinned) r.push_back(i);
        std::sort(r.begin(), r.end(), [&](std::size_t a, std::size_t b) { return slots[a].bytes != slots[b].bytes ? slots[a].bytes > slots[b].bytes : a < b; });
        return r;
    };
    const auto try_target = [&](std::size_t target, AdmitAction act, std::uint64_t returned) {
        std::uint64_t avail = free_bytes + returned;
        std::vector<std::size_t> rel;
        if (!fits(avail))
            for (const auto i : releasable(target)) {
                rel.push_back(i);
                avail += slots[i].bytes;
                if (fits(avail)) break;
            }
        if (!fits(avail)) return false;
        p.action = act;
        p.slot = target;
        p.release = std::move(rel);
        p.why = std::string(act == AdmitAction::build ? "new session" : "re-size idle session") + " of " + std::to_string(need) +
                " positions (" + mib(p.bytes) + " MiB): free " + mib(free_bytes) + " MiB + returned " + mib(avail - free_bytes) +
                " MiB, headroom " + mib(headroom) + " MiB" + (p.release.empty() ? "" : ", releasing " + std::to_string(p.release.size()) + " idle");
        return true;
    };
    if (empty != slots.size() && fits(free_bytes) && try_target(empty, AdmitAction::build, 0)) return p;  // nothing released
    if (grow != slots.size() && try_target(grow, AdmitAction::resize, slots[grow].bytes)) return p;
    if (empty != slots.size() && try_target(empty, AdmitAction::build, 0)) return p;
    // A smaller pinned target can return bytes that the largest target cannot release.
    // Retry each eligible target with its own release list before declaring the pool infeasible.
    for (std::size_t i = 0; i < slots.size(); ++i) {
        const auto& s = slots[i];
        if (i != grow && !s.busy && s.built && s.positions < need &&
            try_target(i, AdmitAction::resize, s.bytes)) return p;
    }
    p.release.clear();
    p.action = any_busy ? AdmitAction::wait : AdmitAction::reject;
    p.why = "session of " + std::to_string(need) + " positions (" + mib(p.bytes) + " MiB) + headroom " + mib(headroom) +
            " MiB exceeds free " + mib(free_bytes) + " MiB" + (any_busy ? " until a running request ends" : " with every idle session released");
    return p;
}

}  // namespace strix::serve
