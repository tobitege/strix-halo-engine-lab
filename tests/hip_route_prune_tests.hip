// RoutePrune fixture (no model): the shared prune kernel against an independent host reference for every mode, and the
// packed route planner with prune=true against a host CSR of the kept assignments (expert ascending, then row, then
// slot). prune=false on the same rows must still plan every assignment.
#include "../src/moe_core/device_plan.hpp"
#include "../src/moe_core/route_prune.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using strix::moe_core::RoutePrune;

struct Tables { unsigned *summary, *offsets, *touched, *rows, *slots, *experts, *begins; };
struct View { const unsigned* routes; const float* weights; const unsigned* status; Tables csr; unsigned top_k; };

struct Policy {
    static constexpr unsigned kExperts = 512U, kTopK = 10U, kThreads = 1024U, kAssignmentsPerWave = 64U;
    static constexpr unsigned kCountLimit = 65535U, kHistogramBins = 8U, kInvalid = 1U;
    using View = ::View;
    __device__ static const Tables& tables(const View& v) { return v.csr; }
    __device__ static const unsigned* routes(const View& v, unsigned r) { return v.routes + r * v.top_k; }
    __device__ static unsigned expert(const View& v, unsigned r, unsigned s) { return v.routes[r * v.top_k + s]; }
    __device__ static unsigned status(const View& v, unsigned r) { return v.status[r]; }
    __device__ static bool kept(const View& v, unsigned r, unsigned s) { return v.weights[r * v.top_k + s] != 0.0F; }
    template <bool Certificate>
    __device__ static bool invalid(const View& v, unsigned r, unsigned s, unsigned e, unsigned experts, bool& unsafe) {
        (void)unsafe;
        return e >= experts || !isfinite(v.weights[r * v.top_k + s]);
    }
    __device__ static void certificate(const View&) {}
    __device__ static void failure(const View& v, unsigned t, unsigned bits, unsigned count, unsigned rows) {
        if (t < 32U) v.csr.summary[t] = t == 0U ? bits : t == 3U ? count : t == 6U ? rows : 0U;
    }
    __device__ static void summary(const View& v, unsigned t, unsigned status, unsigned touched, unsigned batches, unsigned count,
                                   unsigned duplicates, unsigned maximum, unsigned rows, const unsigned* hist) {
        (void)hist;
        if (t < 32U) v.csr.summary[t] = t == 0U ? status : t == 1U ? touched : t == 2U ? batches : t == 3U ? count :
            t == 4U ? duplicates : t == 5U ? maximum : t == 6U ? rows : 0U;
    }
};
using Plan = strix::moe_core::device_plan::PackedPlan<Policy>;
// A legacy unpruned policy need not implement the optional kept hook.
struct UnprunedPolicy : Policy {
    __device__ static bool kept(View, unsigned, unsigned) = delete;
};
using UnprunedPlan = strix::moe_core::device_plan::PackedPlan<UnprunedPolicy>;

int failures = 0;
void expect(bool ok, const char* what) { if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); } }
void check(hipError_t e, const char* why) { if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", why, hipGetErrorString(e)); std::exit(2); } }
template <class T> T* dev(std::size_t n) { void* p = nullptr; check(hipMalloc(&p, n * sizeof(T)), "alloc"); return static_cast<T*>(p); }
template <class T> void up(T* d, const std::vector<T>& h) { check(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload"); }
template <class T> std::vector<T> down(const T* d, std::size_t n) {
    std::vector<T> h(n); check(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "download"); return h;
}

// Independent host reference: sort the slots by (weight descending, slot ascending) and apply the mode in rank order.
std::vector<float> host_prune(const std::vector<float>& w, unsigned rows, unsigned k, RoutePrune p) {
    std::vector<float> out(w);
    for (unsigned r = 0; r < rows; ++r) {
        const float* x = w.data() + std::size_t{r} * k;
        float sum = 0.0F;
        for (unsigned s = 0; s < k; ++s) sum = sum + x[s];
        std::vector<unsigned> order(k);
        for (unsigned s = 0; s < k; ++s) order[s] = s;
        for (unsigned a = 0; a < k; ++a) for (unsigned b = a + 1U; b < k; ++b)
            if (x[order[b]] > x[order[a]] || (x[order[b]] == x[order[a]] && order[b] < order[a])) std::swap(order[a], order[b]);
        std::vector<bool> keep(k, false);
        const float bound = p.value * sum;
        for (unsigned rank = 0; rank < k; ++rank) {
            const unsigned s = order[rank];
            float before = 0.0F;  // same slot-order accumulation as the kernel
            for (unsigned q = 0; q < k; ++q) if (x[q] > x[s] || (x[q] == x[s] && q < s)) before = before + x[q];
            keep[s] = rank == 0U || (p.mode == RoutePrune::relative ? x[s] >= bound : p.mode == RoutePrune::top_p ? before < bound
                                     : static_cast<float>(rank) < p.value);
        }
        float kept = 0.0F;
        for (unsigned s = 0; s < k; ++s) if (keep[s]) kept = kept + x[s];
        const float scale = p.renorm && kept > 0.0F ? sum / kept : 1.0F;
        for (unsigned s = 0; s < k; ++s) out[std::size_t{r} * k + s] = !keep[s] ? 0.0F : p.renorm ? x[s] * scale : x[s];
    }
    return out;
}

}  // namespace

int main() {
    constexpr unsigned rows = 3000U, k = Policy::kTopK, experts = Policy::kExperts, cap = 8U, count = rows * k;
    std::mt19937 rng(20261113U);
    std::vector<unsigned> routes(count), status(rows, 0U);
    std::vector<float> weights(count);
    for (unsigned r = 0; r < rows; ++r) {
        // distinct experts, skewed towards a few (uneven batches), slots in expert order (weights unordered by slot)
        std::vector<unsigned> pick;
        while (pick.size() < k) {
            const unsigned e = (rng() % 4U) ? rng() % 48U : rng() % experts;
            bool seen = false;
            for (unsigned q : pick) seen |= q == e;
            if (!seen) pick.push_back(e);
        }
        float sum = 0.0F;
        for (unsigned s = 0; s < k; ++s) {
            routes[r * k + s] = pick[s];
            weights[r * k + s] = std::exp(-static_cast<float>(rng() % 1000U) / 140.0F);
            sum += weights[r * k + s];
        }
        for (unsigned s = 0; s < k; ++s) weights[r * k + s] /= sum;
        if (r % 97U == 0U) weights[r * k + 3U] = weights[r * k + 7U];  // exact ties
    }
    auto* d_routes = dev<unsigned>(count); auto* d_weights = dev<float>(count); auto* d_status = dev<unsigned>(rows);
    up(d_routes, routes); up(d_status, status);
    const unsigned max_batches = experts + (count + cap - 1U) / cap;
    Tables t{dev<unsigned>(32), dev<unsigned>(experts + 1U), dev<unsigned>(experts), dev<unsigned>(count), dev<unsigned>(count),
             dev<unsigned>(max_batches), dev<unsigned>(max_batches + 1U)};
    auto* scratch = static_cast<unsigned*>(nullptr);
    { void* p = nullptr; check(hipMalloc(&p, Plan::scratch_bytes(rows)), "scratch"); scratch = static_cast<unsigned*>(p); }
    const View view{d_routes, d_weights, d_status, t, k};

    const RoutePrune specs[] = {{RoutePrune::off, 0.0F, 0U}, {RoutePrune::relative, 0.02F, 0U}, {RoutePrune::relative, 0.05F, 1U},
                                {RoutePrune::top_p, 0.9F, 0U}, {RoutePrune::top_p, 0.98F, 1U}, {RoutePrune::cap, 7.0F, 0U},
                                {RoutePrune::cap, 9.0F, 1U}, {RoutePrune::relative, 0.5F, 0U}};
    for (const auto p : specs) {
        char name[96];
        std::snprintf(name, sizeof name, "mode %u value %g renorm %u", p.mode, static_cast<double>(p.value), p.renorm);
        up(d_weights, weights);
        std::vector<float> want = weights;
        if (p.active()) {
            check(strix::moe_core::enqueue_route_prune(d_weights, k * sizeof(float), rows, k, p, nullptr), "prune");
            want = host_prune(weights, rows, k, p);
        }
        if (p.active()) check(Plan::enqueue(view, rows, experts, k, cap, scratch, nullptr, false, true), "plan");
        else {
            expect(UnprunedPlan::enqueue(view, rows, experts, k, cap, scratch, nullptr, false, true) == hipErrorInvalidValue,
                   "pruning without kept must be refused");
            check(UnprunedPlan::enqueue(view, rows, experts, k, cap, scratch, nullptr, false, false), "legacy plan");
        }
        check(hipDeviceSynchronize(), "sync");
        const auto got = down(d_weights, count);
        unsigned differ = 0U, kept = 0U;
        for (unsigned a = 0; a < count; ++a) { differ += std::memcmp(&got[a], &want[a], 4U) != 0; kept += want[a] != 0.0F; }
        expect(differ == 0U, name);
        expect(!p.active() || p.mode != RoutePrune::cap || kept == rows * static_cast<unsigned>(p.value), "cap keeps exactly k' slots per row");
        // host CSR of the kept assignments
        std::vector<unsigned> offsets(experts + 1U, 0U), h_rows, h_slots;
        for (unsigned e = 0; e < experts; ++e) {
            offsets[e] = static_cast<unsigned>(h_rows.size());
            for (unsigned r = 0; r < rows; ++r) for (unsigned s = 0; s < k; ++s)
                if (routes[r * k + s] == e && want[r * k + s] != 0.0F) { h_rows.push_back(r); h_slots.push_back(s); }
        }
        offsets[experts] = static_cast<unsigned>(h_rows.size());
        const auto summary = down(t.summary, 32), d_off = down(t.offsets, experts + 1U);
        const auto d_rows = down(t.rows, count), d_slots = down(t.slots, count);
        expect(summary[0] == 0U && summary[3] == kept && summary[6] == rows, "summary: no failure, planned count, rows");
        expect(d_off == offsets, "expert offsets");
        bool same = true;
        for (unsigned a = 0; a < kept; ++a) same &= d_rows[a] == h_rows[a] && d_slots[a] == h_slots[a];
        expect(same, "planned rows and slots");
        // batches: per touched expert ceil(n / cap) runs, begins ascending, final begin = planned count
        unsigned batches = 0U, touched = 0U;
        for (unsigned e = 0; e < experts; ++e) { const unsigned n = offsets[e + 1U] - offsets[e]; batches += (n + cap - 1U) / cap; touched += n != 0U; }
        const auto d_begins = down(t.begins, max_batches + 1U), d_experts = down(t.experts, max_batches);
        expect(summary[1] == touched && summary[2] == batches && d_begins[batches] == kept, "touched experts, batches, final begin");
        bool runs = true;
        for (unsigned b = 0; b < batches; ++b) {
            const unsigned e = d_experts[b], begin = d_begins[b], end = d_begins[b + 1U];
            runs &= e < experts && begin < end && end - begin <= cap && begin >= offsets[e] && end <= offsets[e + 1U];
        }
        expect(runs, "batch runs stay inside their expert");
        std::printf("%-34s kept %u of %u (%.1f %% cut) batches %u touched %u\n", name, kept, count, 100.0 * (count - kept) / count, batches, touched);
    }
    std::printf(failures ? "route prune tests: %d FAILED\n" : "route prune tests: PASS\n", failures);
    return failures ? 1 : 0;
}
