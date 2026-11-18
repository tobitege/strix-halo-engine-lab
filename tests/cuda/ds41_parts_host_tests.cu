// deepseek41 CUDA parts-host kernels, bit for bit, without a network: resident routes (against enqueue_moe_routes
// when every expert is resident, and against a host construction for arbitrary residencies), and the seeded combine
// against xrpc::combine_host with answers planned by xrpc::plan_response for one or two servers (FOLD and RAW items,
// row subsets), and against enqueue_moe_combine on the one-device raw vectors.

#include "strix/cuda/ds41_parts_host.hpp"
#include "strix/ds41_expert_rpc.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace dc = strix::ds41::cuda;
namespace x = strix::ds41::xrpc;

namespace {

int g_failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++g_failures;
        if (g_failures < 40) std::printf("FAIL %s\n", what.c_str());
    }
}

template <class T>
std::vector<T> get(const dc::DeviceBuffer<T>& b, std::size_t n) { return b.download(n); }
template <class T>
void put(dc::DeviceBuffer<T>& b, const std::vector<T>& v) {
    if (b.size() < v.size()) b.resize(v.size());
    if (!v.empty()) b.upload(v.data(), v.size());
}
template <class T>
bool same(const std::vector<T>& a, const std::vector<T>& b, std::size_t n) { return std::memcmp(a.data(), b.data(), n * sizeof(T)) == 0; }

std::vector<std::uint32_t> random_ids(std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts, std::mt19937& rng) {
    std::vector<std::uint32_t> ids;
    std::vector<std::uint32_t> all(experts);
    std::iota(all.begin(), all.end(), 0U);
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::shuffle(all.begin(), all.end(), rng);
        ids.insert(ids.end(), all.begin(), all.begin() + top_k);
    }
    return ids;
}

struct Routes {
    dc::DeviceBuffer<std::uint32_t> experts, begins, rows, slots, order, low, counts;
    explicit Routes(std::size_t n, std::uint32_t r) : experts(n), begins(n + 1), rows(n), slots(n), order(n), low(r), counts(2) {}
    dc::MoeRoutes view() { return {experts.get(), begins.get(), rows.get(), slots.get(), order.get(), low.get(), counts.get()}; }
};

void test_resident_routes(std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts, double fraction, std::mt19937& rng) {
    const auto ids = random_ids(rows, top_k, experts, rng);
    const std::size_t n = std::size_t{rows} * top_k;
    const std::uint32_t split = experts / 2U;
    std::vector<std::int32_t> slot_of(experts, -1);
    std::vector<std::uint32_t> perm(experts);
    std::iota(perm.begin(), perm.end(), 0U);
    std::shuffle(perm.begin(), perm.end(), rng);
    const auto resident = static_cast<std::uint32_t>(fraction * experts);
    // physical slots in a scrambled order (any compact layout must work)
    for (std::uint32_t i = 0; i < resident; ++i) slot_of[perm[i]] = static_cast<std::int32_t>((i * 7U + 3U) % resident);
    if (fraction >= 1.0) for (std::uint32_t e = 0; e < experts; ++e) slot_of[e] = static_cast<std::int32_t>(e);
    dc::DeviceBuffer<std::uint32_t> d_ids;
    put(d_ids, ids);
    dc::DeviceBuffer<std::int32_t> d_slot;
    put(d_slot, slot_of);
    Routes a(n, rows);
    STRIX_CUDA_CHECK(cudaMemset(a.experts.get(), 0xff, n * 4U));
    dc::ResidentRoutesView v;
    v.ids = d_ids.get();
    v.slot_of = d_slot.get();
    v.rows = rows;
    v.top_k = top_k;
    v.experts = experts;
    v.split = split;
    v.out = a.view();
    dc::enqueue_resident_routes(v, nullptr);
    STRIX_CUDA_CHECK(cudaDeviceSynchronize());
    const std::string tag = "resident routes rows " + std::to_string(rows) + " fraction " + std::to_string(fraction);
    // host expectation
    std::vector<std::uint32_t> uni;
    for (const auto e : ids) if (slot_of[e] >= 0) uni.push_back(e);
    std::sort(uni.begin(), uni.end());
    uni.erase(std::unique(uni.begin(), uni.end()), uni.end());
    std::vector<std::uint32_t> ex, begins, er, es;
    std::uint32_t most = 0;
    for (const auto e : uni) {
        ex.push_back(static_cast<std::uint32_t>(slot_of[e]));
        begins.push_back(static_cast<std::uint32_t>(er.size()));
        std::uint32_t c = 0;
        for (std::size_t s = 0; s < n; ++s)
            if (ids[s] == e) { er.push_back(static_cast<std::uint32_t>(s / top_k)); es.push_back(static_cast<std::uint32_t>(s)); ++c; }
        most = std::max(most, c);
    }
    begins.push_back(static_cast<std::uint32_t>(er.size()));
    std::vector<std::uint32_t> order(n), low(rows);
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::vector<std::uint32_t> sl(top_k);
        std::iota(sl.begin(), sl.end(), r * top_k);
        std::sort(sl.begin(), sl.end(), [&](std::uint32_t p, std::uint32_t q) { return ids[p] < ids[q]; });
        std::copy(sl.begin(), sl.end(), order.begin() + std::size_t{r} * top_k);
        low[r] = static_cast<std::uint32_t>(std::count_if(sl.begin(), sl.end(), [&](std::uint32_t s) { return ids[s] < split; }));
    }
    const auto counts = get(a.counts, 2);
    expect(counts[0] == uni.size() && counts[1] == most, tag + " counts");
    expect(same(get(a.experts, uni.size()), ex, uni.size()), tag + " experts");
    expect(same(get(a.begins, uni.size() + 1), begins, uni.size() + 1), tag + " begins");
    expect(same(get(a.rows, er.size()), er, er.size()) && same(get(a.slots, es.size()), es, es.size()), tag + " entries");
    expect(same(get(a.order, n), order, n) && same(get(a.low, rows), low, rows), tag + " order / low");
    if (fraction >= 1.0) {  // identical to the all-resident routes
        Routes b(n, rows);
        dc::DeviceBuffer<std::uint32_t> scratch(experts);
        dc::MoeRoutesView m;
        m.ids = d_ids.get();
        m.rows = rows;
        m.top_k = top_k;
        m.experts = experts;
        m.split = split;
        m.scratch = scratch.get();
        m.out = b.view();
        dc::enqueue_moe_routes(m, nullptr);
        STRIX_CUDA_CHECK(cudaDeviceSynchronize());
        const auto u = counts[0];
        expect(same(get(a.experts, u), get(b.experts, u), u) && same(get(a.begins, u + 1), get(b.begins, u + 1), u + 1) &&
                   same(get(a.rows, n), get(b.rows, n), n) && same(get(a.slots, n), get(b.slots, n), n) &&
                   same(get(a.order, n), get(b.order, n), n) && same(get(a.low, rows), get(b.low, rows), rows) &&
                   same(get(a.counts, 2), get(b.counts, 2), 2),
               tag + " equal to enqueue_moe_routes");
    }
}

float random_value(std::mt19937& rng) {
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    std::uniform_int_distribution<int> e(-12, 6);
    return std::ldexp(u(rng), e(rng));
}

// One window: residency (local fraction, one or two servers split at `cut`), answers from the one-device raw vectors.
void test_seeded_combine(std::uint32_t rows, double local_fraction, int servers, bool no_fold, std::mt19937& rng) {
    const std::uint32_t K = 6, E = 384, H = 256, split = 192;
    x::Geometry g;
    g.hidden = H;
    g.inter = 256;
    g.experts = E;
    g.top_k = K;
    g.split = split;
    const auto ids = random_ids(rows, K, E, rng);
    const std::size_t n = std::size_t{rows} * K;
    std::vector<float> raw(n * H), weights(n), shared(std::size_t{rows} * H);
    for (auto& v : raw) v = random_value(rng);
    for (auto& v : weights) v = std::ldexp(std::uniform_real_distribution<float>(0.01F, 1.0F)(rng), -1);
    for (auto& v : shared) v = random_value(rng);
    std::vector<char> local(E, 0);
    for (auto& l : local) l = std::uniform_real_distribution<double>(0, 1)(rng) < local_fraction ? 1 : 0;
    const std::uint32_t cut = std::uniform_int_distribution<std::uint32_t>(1, E - 1)(rng);
    std::vector<x::Residency> res(static_cast<std::size_t>(servers));
    res[0] = {0, servers == 1 ? E : cut, ~std::uint64_t{0}};
    if (servers == 2) res[1] = {cut, E, ~std::uint64_t{0}};
    std::vector<std::uint32_t> remote(rows, 0U);
    std::vector<std::vector<std::uint32_t>> mask(res.size(), std::vector<std::uint32_t>(rows, 0U));
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t k = 0; k < K; ++k) {
            const auto e = ids[r * K + k];
            if (local[e]) continue;
            const std::size_t srv = servers == 2 && e >= cut ? 1U : 0U;
            mask[srv][r] |= 1U << k;
            remote[r] |= 1U << k;
        }
    std::vector<x::ExpertsRequest> reqs(res.size());
    std::vector<std::vector<std::uint32_t>> row_maps(res.size());
    std::vector<x::ResponsePlan> plans(res.size());
    std::vector<std::vector<float>> vecs(res.size());
    std::vector<x::Answer> answers;
    std::vector<std::int32_t> base;
    std::int32_t total = 0;
    for (std::size_t srv = 0; srv < res.size(); ++srv) {
        auto& q = reqs[srv];
        for (std::uint32_t r = 0; r < rows; ++r) {
            if (!mask[srv][r]) continue;
            row_maps[srv].push_back(r);
            q.ids.insert(q.ids.end(), ids.begin() + r * K, ids.begin() + (r + 1) * K);
            q.weights.insert(q.weights.end(), weights.begin() + r * K, weights.begin() + (r + 1) * K);
            q.remote.push_back(mask[srv][r]);
        }
        if (row_maps[srv].empty()) continue;
        q.layer = 3;
        q.rows = static_cast<std::uint32_t>(row_maps[srv].size());
        q.top_k = K;
        q.flags = no_fold ? x::kNoFold : 0U;
        plans[srv] = x::plan_response(q, g, res[srv]);
        const auto& rm = row_maps[srv];
        const auto raw_of = [&](std::uint32_t slot) { return raw.data() + (std::size_t{rm[slot / K]} * K + slot % K) * H; };
        vecs[srv].resize(plans[srv].items.size() * H);
        for (std::size_t i = 0; i < plans[srv].items.size(); ++i) {
            const auto& it = plans[srv].items[i];
            if (it.kind == x::PlanItem::Kind::fold) x::fold_item(plans[srv], i, q.weights, H, raw_of, vecs[srv].data() + i * H);
            else std::memcpy(vecs[srv].data() + i * H, raw_of(it.slot), H * 4U);
        }
        answers.push_back({&plans[srv], vecs[srv].data(), rm});
        base.push_back(total);
        total += static_cast<std::int32_t>(plans[srv].items.size());
    }
    const auto cp = x::plan_combine(rows, K, ids, remote, split, answers);
    std::vector<float> y_host(std::size_t{rows} * H);
    x::combine_host(cp, answers, weights, shared, H, [&](std::uint32_t slot) { return raw.data() + std::size_t{slot} * H; }, y_host.data());
    // device: remote slots of the local raw buffer hold NaN (the combine must not read them)
    std::vector<float> local_raw(raw);
    for (std::uint32_t s = 0; s < n; ++s)
        if ((remote[s / K] >> (s % K)) & 1U) std::fill(local_raw.begin() + std::size_t{s} * H, local_raw.begin() + std::size_t{s + 1} * H, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> remote_vecs;
    for (const auto& a : answers) remote_vecs.insert(remote_vecs.end(), a.vectors, a.vectors + a.plan->items.size() * H);
    std::vector<std::int32_t> tables(dc::seeded_combine_table_ints(rows, K));
    const auto index = [&](std::int64_t src) { return static_cast<std::int32_t>(src); };
    for (std::uint32_t i = 0; i < 2U * rows; ++i) { tables[i] = index(cp.seed[i]); tables[2U * rows + i] = static_cast<std::int32_t>(cp.start[i]); }
    for (std::uint32_t i = 0; i < n; ++i) tables[4U * rows + i] = index(cp.source[i]);
    dc::DeviceBuffer<float> d_raw, d_full, d_remote, d_w, d_shared, d_y(std::size_t{rows} * H), d_ref(std::size_t{rows} * H);
    dc::DeviceBuffer<std::uint32_t> d_order, d_low;
    dc::DeviceBuffer<std::int32_t> d_tab;
    put(d_raw, local_raw);
    put(d_full, raw);
    remote_vecs.push_back(0.0F);
    put(d_remote, remote_vecs);
    put(d_w, weights);
    put(d_shared, shared);
    put(d_order, cp.order);
    put(d_low, cp.low);
    put(d_tab, tables);
    dc::SeededCombineView c;
    c.raw = d_raw.get();
    for (std::size_t i = 0; i < answers.size(); ++i) c.remote[i] = d_remote.get() + std::size_t(base[i]) * H;
    c.shared = d_shared.get();
    c.weights = d_w.get();
    c.order = d_order.get();
    c.low_counts = d_low.get();
    c.tables = d_tab.get();
    c.rows = rows;
    c.top_k = K;
    c.hidden = H;
    c.out = d_y.get();
    dc::enqueue_seeded_combine(c, nullptr);
    dc::MoeCombineView m;
    m.raw = d_full.get();
    m.shared = d_shared.get();
    m.weights = d_w.get();
    m.order = d_order.get();
    m.low_counts = d_low.get();
    m.rows = rows;
    m.top_k = K;
    m.hidden = H;
    m.out = d_ref.get();
    dc::enqueue_moe_combine(m, nullptr);
    STRIX_CUDA_CHECK(cudaDeviceSynchronize());
    const auto y = get(d_y, y_host.size()), ref = get(d_ref, y_host.size());
    const std::string tag = "seeded combine rows " + std::to_string(rows) + " local " + std::to_string(local_fraction) + " servers " +
                            std::to_string(servers) + (no_fold ? " no-fold" : "");
    expect(same(y, y_host, y.size()), tag + " = combine_host");
    expect(same(y, ref, y.size()), tag + " = one-device combine");
}

}  // namespace

int main() {
    try {
        const auto device = dc::select_device(0);
        std::printf("device %s (cc %d.%d)\n", device.name.c_str(), device.major, device.minor);
        std::mt19937 rng(20261021);
        for (const std::uint32_t rows : {1U, 2U, 4U, 8U, 64U, 700U})
            for (const double f : {0.0, 0.1, 0.5, 0.9, 1.0}) test_resident_routes(rows, 6, 384, f, rng);
        test_resident_routes(5, 8, 64, 0.5, rng);
        int windows = 0;
        for (int rep = 0; rep < 60; ++rep)
            for (const std::uint32_t rows : {1U, 2U, 3U, 4U, 8U, 13U})
                for (const double f : {0.0, 0.3, 0.7, 1.0})
                    for (const int servers : {1, 2}) {
                        test_seeded_combine(rows, f, servers, rep % 5 == 4, rng);
                        ++windows;
                    }
        std::printf("seeded combine: %d windows\n", windows);
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf(g_failures ? "%d FAILURES\n" : "all passed\n", g_failures);
    return g_failures ? 1 : 0;
}
