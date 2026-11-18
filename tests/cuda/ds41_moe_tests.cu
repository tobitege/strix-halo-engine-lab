// deepseek41 CUDA routed experts against the host reference, bit for bit: A8, device routes (both launch forms,
// restricted ranges), the decode GEMVs, the grouped GEMM, half sums and the combine, on synthetic blocks at a small
// and at the model's shape; with STRIX_DS41_EXPERTS_GGUF also on the real expert tensors of a layer
// (STRIX_DS41_LAYER, default 20), routed by STRIX_DS41_ROUTES (p8-routes format) when given.

#include "strix/cuda/ds41_core.hpp"
#include "strix/cuda/ds41_moe.hpp"
#include "strix/ds41_formats.hpp"
#include "strix/ds41_moe.hpp"
#include "strix/gguf_reader.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ds = strix::ds41;
namespace dc = strix::ds41::cuda;

namespace {

int g_failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++g_failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}
void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class T>
struct Dev {
    T* p = nullptr;
    std::size_t n = 0;
    Dev() = default;
    explicit Dev(std::size_t count) : n(count) { cuda_check(cudaMalloc(&p, std::max<std::size_t>(1, count) * sizeof(T)), "malloc"); }
    explicit Dev(const std::vector<T>& v) : Dev(v.size()) { put(v); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    ~Dev() { cudaFree(p); }
    void put(const std::vector<T>& v) { cuda_check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload"); }
    [[nodiscard]] std::vector<T> get(std::size_t count) const {
        std::vector<T> v(count);
        cuda_check(cudaMemcpy(v.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return v;
    }
    void fill(int byte) { cuda_check(cudaMemset(p, byte, std::max<std::size_t>(1, n) * sizeof(T)), "memset"); }
};

std::uint64_t fnv(const void* data, std::size_t bytes) {
    std::uint64_t h = 14695981039346656037ULL;
    const auto* b = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}
template <class T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b, std::size_t count, const std::string& what) {
    if (a.size() < count || b.size() < count) { expect(false, what + " (size)"); return false; }
    for (std::size_t i = 0; i < count; ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(T)) != 0) {
            std::ostringstream s;
            s << what << " differs at " << i << " of " << count;
            if constexpr (std::is_floating_point_v<T>) s << ": " << a[i] << " vs " << b[i];
            else s << ": " << static_cast<long long>(a[i]) << " vs " << static_cast<long long>(b[i]);
            expect(false, s.str());
            return false;
        }
    return true;
}

std::uint16_t to_half_bits(float f) {
    // finite normal halves only (test data)
    const auto u = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = (u >> 16U) & 0x8000U;
    int e = static_cast<int>((u >> 23U) & 0xffU) - 127 + 15;
    e = std::clamp(e, 1, 30);
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(e) << 10U) | ((u >> 13U) & 0x3ffU));
}

// Synthetic expert tensors: random IQ2_XXS / Q2_K blocks with finite FP16 scales (some subnormal).
std::vector<std::uint8_t> random_iq2(std::size_t blocks, std::mt19937& rng) {
    std::vector<std::uint8_t> out(blocks * ds::iq2_xxs_block_bytes);
    std::uniform_int_distribution<int> byte(0, 255), pick(0, 15);
    std::lognormal_distribution<float> mag(-7.0F, 1.0F);
    for (std::size_t b = 0; b < blocks; ++b) {
        auto* p = out.data() + b * ds::iq2_xxs_block_bytes;
        for (std::size_t i = 2; i < ds::iq2_xxs_block_bytes; ++i) p[i] = static_cast<std::uint8_t>(byte(rng));
        std::uint16_t d = to_half_bits(mag(rng));
        if (pick(rng) == 0) d = static_cast<std::uint16_t>(byte(rng) | ((byte(rng) & 3) << 8));  // subnormal
        p[0] = static_cast<std::uint8_t>(d & 0xff); p[1] = static_cast<std::uint8_t>(d >> 8);
    }
    return out;
}
std::vector<std::uint8_t> random_q2k(std::size_t blocks, std::mt19937& rng) {
    std::vector<std::uint8_t> out(blocks * ds::q2_k_block_bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    std::lognormal_distribution<float> mag(-8.0F, 1.0F);
    for (std::size_t b = 0; b < blocks; ++b) {
        auto* p = out.data() + b * ds::q2_k_block_bytes;
        for (std::size_t i = 0; i < 80; ++i) p[i] = static_cast<std::uint8_t>(byte(rng));
        const std::uint16_t d = to_half_bits(mag(rng)), m = to_half_bits(mag(rng));
        p[80] = static_cast<std::uint8_t>(d & 0xff); p[81] = static_cast<std::uint8_t>(d >> 8);
        p[82] = static_cast<std::uint8_t>(m & 0xff); p[83] = static_cast<std::uint8_t>(m >> 8);
    }
    return out;
}
std::vector<float> random_x(std::size_t n, std::mt19937& rng, float scale) {
    std::normal_distribution<float> normal(0.0F, scale);
    std::vector<float> x(n);
    for (auto& v : x) v = normal(rng);
    // a zero group and a tiny group (A8 special cases)
    if (n >= 128) {
        std::fill(x.begin() + 32, x.begin() + 64, 0.0F);
        for (int i = 64; i < 96; ++i) x[static_cast<std::size_t>(i)] = 1e-37F * static_cast<float>(i - 80);
    }
    return x;
}
std::vector<std::uint32_t> random_ids(std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts, std::mt19937& rng, bool same) {
    std::vector<std::uint32_t> ids;
    std::vector<std::uint32_t> all(experts);
    for (std::uint32_t e = 0; e < experts; ++e) all[e] = e;
    std::vector<std::uint32_t> first;
    for (std::uint32_t r = 0; r < rows; ++r) {
        if (!same || r == 0) {
            std::shuffle(all.begin(), all.end(), rng);
            first.assign(all.begin(), all.begin() + top_k);
        }
        ids.insert(ids.end(), first.begin(), first.end());
    }
    return ids;
}

struct DevRoutes {
    Dev<std::uint32_t> experts, begins, rows, slots, order, low_counts, counts, scratch;
    DevRoutes(std::uint32_t r, std::uint32_t top_k, std::uint32_t n_experts)
        : experts(std::size_t{r} * top_k), begins(std::size_t{r} * top_k + 1), rows(std::size_t{r} * top_k), slots(std::size_t{r} * top_k),
          order(std::size_t{r} * top_k), low_counts(r), counts(2), scratch(n_experts) {}
    [[nodiscard]] dc::MoeRoutes view() const {
        return {experts.p, begins.p, rows.p, slots.p, order.p, low_counts.p, counts.p};
    }
};

// --------------------------------------------------------------------------------------------------- routes
void test_routes(std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts, std::uint32_t first, std::uint32_t end, std::uint32_t base,
                 std::mt19937& rng) {
    const auto ids = random_ids(rows, top_k, experts, rng, false);
    const std::uint32_t split = experts / 2;
    auto ref = ds::build_window_routes(ids, rows, top_k, split);
    if (end) ref = ds::restrict_routes(ref, first, end);
    Dev<std::uint32_t> dids(ids);
    DevRoutes r(rows, top_k, experts);
    dc::MoeRoutesView b{dids.p, rows, top_k, experts, split, first, end, base, r.scratch.p, r.view()};
    dc::enqueue_moe_routes(b, nullptr);
    cuda_check(cudaDeviceSynchronize(), "routes");
    const auto counts = r.counts.get(2);
    const std::uint32_t u = counts[0];
    std::uint32_t max_rows = 0;
    for (std::size_t j = 0; j + 1 < ref.begins.size(); ++j) max_rows = std::max(max_rows, ref.begins[j + 1] - ref.begins[j]);
    bool ok = u == ref.experts.size() && counts[1] == max_rows;
    auto experts_ref = ref.experts;
    for (auto& e : experts_ref) e -= base;
    ok = ok && same_bits(experts_ref, r.experts.get(u), u, "route experts");
    ok = ok && same_bits(ref.begins, r.begins.get(u + 1), u + 1, "route begins");
    ok = ok && same_bits(ref.rows, r.rows.get(ref.rows.size()), ref.rows.size(), "route rows");
    ok = ok && same_bits(ref.slots, r.slots.get(ref.slots.size()), ref.slots.size(), "route slots");
    ok = ok && same_bits(ref.order, r.order.get(ref.order.size()), ref.order.size(), "route order");
    ok = ok && same_bits(ref.low_counts, r.low_counts.get(rows), rows, "route low counts");
    expect(ok, "routes");
    std::printf("routes rows %u top_k %u experts %u range [%u, %u) base %u: %s (union %u)\n", rows, top_k, experts, first, end, base,
                ok ? "equal" : "DIFFER", u);
}

// --------------------------------------------------------------------------------------------- whole window
struct WindowResult {
    std::uint64_t raw_hash = 0, y_hash = 0;
};
WindowResult run_window(const dc::MoeShape& shape, const std::vector<std::uint8_t>& gate, const std::vector<std::uint8_t>& up,
                        const std::vector<std::uint8_t>& down, const std::vector<std::uint32_t>& ids, std::uint32_t rows, std::mt19937& rng,
                        bool grouped, const std::string& label, std::uint32_t expert_index_base = 0,
                        const std::vector<std::uint32_t>* compact = nullptr) {
    const std::uint32_t H = shape.hidden, I = shape.inter, K = shape.top_k, slots = rows * K, split = shape.experts / 2;
    const auto x = random_x(std::size_t{rows} * H, rng, 1.0F);
    std::vector<float> weights(slots), shared(std::size_t{rows} * H);
    std::uniform_real_distribution<float> uw(0.05F, 0.6F);
    for (auto& w : weights) w = uw(rng);
    std::normal_distribution<float> ns(0.0F, 0.05F);
    for (auto& v : shared) v = ns(rng);

    // host reference (gate/up/down given as whole tensors of the experts used; compact maps id -> tensor index)
    std::vector<ds::A8Activation> xa;
    for (std::uint32_t r = 0; r < rows; ++r) xa.push_back(ds::quantize_a8(std::span<const float>(x.data() + std::size_t{r} * H, H)));
    auto routes = ds::build_window_routes(ids, rows, K, split);
    ds::ExpertTensors t;
    t.gate = gate; t.up = up; t.down = down; t.hidden = H; t.inter = I; t.limit = shape.limit;
    auto host_routes = routes;
    if (compact)
        for (auto& e : host_routes.experts) e = (*compact)[e];
    const auto ref = ds::reference_window(t, xa, host_routes, weights, shared, K);
    const auto low = ds::reference_half_sum(routes, weights, ref.raw, rows, K, H, 0);
    const auto high = ds::reference_half_sum(routes, weights, ref.raw, rows, K, H, 1);

    // device
    Dev<std::uint8_t> dg(gate), du(up), dd(down);
    Dev<float> dx(x), dw(weights), dsh(shared);
    Dev<std::int8_t> xc(x.size());
    Dev<float> xd(x.size() / 32);
    Dev<std::int32_t> xsum(x.size() / 32);  // the A8 quantizer writes the sums
    const dc::A8Rows xa8{xc.p, xd.p, xsum.p, rows, H, H};
    dc::QuantizeA8View qv;
    qv.x = dx.p; qv.rows = rows; qv.width = H; qv.out = xa8;
    dc::enqueue_quantize_a8(qv, nullptr);
    Dev<std::uint32_t> dids(ids);
    DevRoutes r(rows, K, shape.experts);
    dc::enqueue_moe_routes({dids.p, rows, K, shape.experts, split, 0, 0, 0, r.scratch.p, r.view()}, nullptr);
    cuda_check(cudaDeviceSynchronize(), "routes");
    if (compact) {  // the test keeps only the used experts resident: remap the union ids to tensor indices
        const auto u = r.counts.get(2)[0];
        auto e = r.experts.get(u);
        for (auto& v : e) v = (*compact)[v] - expert_index_base;
        cuda_check(cudaMemcpy(r.experts.p, e.data(), 4 * u, cudaMemcpyHostToDevice), "remap");
    }
    Dev<std::int8_t> ac(std::size_t{slots} * I);
    Dev<float> ad(std::size_t{slots} * I / 32), go(std::size_t{slots} * I), uo(std::size_t{slots} * I), raw(std::size_t{slots} * H),
        y(std::size_t{rows} * H), lo(std::size_t{rows} * H), hi(std::size_t{rows} * H), canon(std::size_t{rows} * H);
    Dev<std::int32_t> as(std::size_t{slots} * I / 32);
    ac.fill(0x55); raw.fill(0x7f);
    dc::RoutedGateUpView g;
    g.shape = shape; g.weights = {dg.p, du.p, dd.p}; g.routes = r.view(); g.max_union = std::min(slots, shape.experts);
    g.max_rows = std::min(rows, 8U); g.x = xa8; g.act = {ac.p, ad.p, as.p, slots, I, I}; g.gate_out = go.p; g.up_out = uo.p;
    dc::RoutedDownView d;
    d.shape = shape; d.weights = g.weights; d.routes = r.view(); d.max_union = g.max_union; d.max_rows = g.max_rows;
    d.act = g.act; d.raw = raw.p;
    if (grouped) {
        Dev<unsigned char> scratch(dc::routed_grouped_workspace_bytes(rows, shape));
        dc::enqueue_routed_gate_up_grouped(g, rows, scratch.p, nullptr);
        dc::enqueue_routed_down_grouped(d, rows, scratch.p, nullptr);
        cuda_check(cudaDeviceSynchronize(), "grouped");
    } else {
        dc::enqueue_routed_gate_up(g, nullptr);
        dc::enqueue_routed_down(d, nullptr);
    }
    dc::MoeCombineView c{raw.p, dsh.p, dw.p, r.order.p, r.low_counts.p, rows, K, H, -1, y.p};
    dc::enqueue_moe_combine(c, nullptr);
    dc::MoeCombineView h = c;
    h.out = lo.p; h.half = 0;
    dc::enqueue_moe_combine(h, nullptr);
    h.out = hi.p; h.half = 1;
    dc::enqueue_moe_combine(h, nullptr);
    dc::enqueue_moe_canonical_sum(dsh.p, lo.p, hi.p, std::size_t{rows} * H, canon.p, nullptr);
    cuda_check(cudaDeviceSynchronize(), "window");

    std::vector<std::int8_t> ref_codes(std::size_t{slots} * I);
    std::vector<float> ref_deltas(std::size_t{slots} * I / 32);
    std::vector<std::int32_t> ref_sums(std::size_t{slots} * I / 32);
    for (std::uint32_t s = 0; s < slots; ++s) {
        std::copy(ref.activation[s].codes.begin(), ref.activation[s].codes.end(), ref_codes.begin() + std::size_t{s} * I);
        std::copy(ref.activation[s].deltas.begin(), ref.activation[s].deltas.end(), ref_deltas.begin() + std::size_t{s} * I / 32);
        std::copy(ref.activation[s].sums.begin(), ref.activation[s].sums.end(), ref_sums.begin() + std::size_t{s} * I / 32);
    }
    bool ok = true;
    ok &= same_bits(ref.gate, go.get(ref.gate.size()), ref.gate.size(), label + " gate");
    ok &= same_bits(ref.up, uo.get(ref.up.size()), ref.up.size(), label + " up");
    ok &= same_bits(ref_codes, ac.get(ref_codes.size()), ref_codes.size(), label + " act codes");
    ok &= same_bits(ref_deltas, ad.get(ref_deltas.size()), ref_deltas.size(), label + " act deltas");
    ok &= same_bits(ref_sums, as.get(ref_sums.size()), ref_sums.size(), label + " act sums");
    const auto draw = raw.get(ref.raw.size());
    ok &= same_bits(ref.raw, draw, ref.raw.size(), label + " raw");
    const auto dy = y.get(ref.y.size());
    ok &= same_bits(ref.y, dy, ref.y.size(), label + " y");
    ok &= same_bits(low, lo.get(low.size()), low.size(), label + " low half");
    ok &= same_bits(high, hi.get(high.size()), high.size(), label + " high half");
    ok &= same_bits(ref.y, canon.get(ref.y.size()), ref.y.size(), label + " canonical");
    WindowResult res{fnv(draw.data(), draw.size() * 4), fnv(dy.data(), dy.size() * 4)};
    std::printf("%s rows %u union %zu: %s (raw fnv %016llx, y fnv %016llx; host %016llx %016llx)\n", label.c_str(), rows,
                routes.experts.size(), ok ? "bitwise equal" : "DIFFERS", static_cast<unsigned long long>(res.raw_hash),
                static_cast<unsigned long long>(res.y_hash), static_cast<unsigned long long>(fnv(ref.raw.data(), ref.raw.size() * 4)),
                static_cast<unsigned long long>(fnv(ref.y.data(), ref.y.size() * 4)));
    return res;
}

void test_synthetic(const dc::MoeShape& shape, std::uint32_t resident, const std::vector<std::uint32_t>& row_counts, bool grouped,
                    std::mt19937& rng, const std::string& label) {
    // `resident` random experts are materialized; ids are drawn from all `shape.experts` and mapped onto them
    const std::size_t gb = shape.gate_row_bytes() * shape.inter / ds::iq2_xxs_block_bytes * resident;
    const std::size_t db = shape.down_row_bytes() * shape.hidden / ds::q2_k_block_bytes * resident;
    const auto gate = random_iq2(gb, rng), up = random_iq2(gb, rng);
    const auto down = random_q2k(db, rng);
    std::vector<std::uint32_t> compact(shape.experts);
    for (std::uint32_t e = 0; e < shape.experts; ++e) compact[e] = e % resident;
    for (const auto rows : row_counts) {
        for (const bool same : {false, true}) {
            if (same && rows == 1) continue;
            auto ids = random_ids(rows, shape.top_k, shape.experts, rng, same);
            // distinct resident tensors per row are not needed: rows only need distinct ids
            run_window(shape, gate, up, down, ids, rows, rng, grouped, label + (same ? " same-experts" : ""), 0,
                       resident < shape.experts ? &compact : nullptr);
        }
    }
}

// ---------------------------------------------------------------------------------------------- real tensors
std::vector<std::uint32_t> read_routes(const char* path, std::uint32_t layer, std::uint32_t top_k) {
    std::vector<std::uint32_t> ids;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ls(line);
        std::string item;
        while (ls >> item) {
            const auto colon = item.find(':');
            if (colon == std::string::npos || std::stoul(item.substr(0, colon)) != layer) continue;
            std::istringstream is(item.substr(colon + 1));
            std::string v;
            std::uint32_t k = 0;
            while (std::getline(is, v, ',') && k < top_k) { ids.push_back(static_cast<std::uint32_t>(std::stoul(v))); ++k; }
        }
    }
    return ids;
}

void test_real(const char* path, std::uint32_t layer, const char* routes_path, std::mt19937& rng) {
    const auto file = strix::gguf::read_file(path);
    const auto find = [&](const std::string& name) -> const strix::gguf::TensorInfo& {
        for (const auto& t : file.tensors)
            if (t.name == name) return t;
        throw std::runtime_error("missing tensor " + name);
    };
    const std::string p = "blk." + std::to_string(layer) + ".";
    const auto& tg = find(p + "ffn_gate_exps.weight");
    const auto& tu = find(p + "ffn_up_exps.weight");
    const auto& td = find(p + "ffn_down_exps.weight");
    expect(tg.ggml_type == 16 && tu.ggml_type == 16 && td.ggml_type == 10, "real tensor types IQ2_XXS / Q2_K");
    dc::MoeShape shape;
    shape.hidden = static_cast<std::uint32_t>(tg.dimensions[0]);
    shape.inter = static_cast<std::uint32_t>(tg.dimensions[1]);
    shape.experts = static_cast<std::uint32_t>(tg.dimensions[2]);
    std::vector<std::uint32_t> all_ids;
    if (routes_path) all_ids = read_routes(routes_path, layer, shape.top_k);
    std::ifstream in(path, std::ios::binary);
    for (const std::uint32_t rows : {1U, 2U, 5U}) {
        std::vector<std::uint32_t> ids;
        if (all_ids.size() >= std::size_t{rows} * shape.top_k * 4)
            ids.assign(all_ids.begin() + 3 * rows * shape.top_k, all_ids.begin() + 4 * rows * shape.top_k);
        else
            ids = random_ids(rows, shape.top_k, shape.experts, rng, false);
        // load the union experts only, in ascending id
        std::vector<std::uint32_t> uni(ids);
        std::sort(uni.begin(), uni.end());
        uni.erase(std::unique(uni.begin(), uni.end()), uni.end());
        std::vector<std::uint32_t> compact(shape.experts, 0);
        const std::size_t gbytes = shape.gate_expert_bytes(), dbytes = shape.down_expert_bytes();
        std::vector<std::uint8_t> gate(gbytes * uni.size()), up(gbytes * uni.size()), down(dbytes * uni.size());
        for (std::size_t i = 0; i < uni.size(); ++i) {
            compact[uni[i]] = static_cast<std::uint32_t>(i);
            in.seekg(static_cast<std::streamoff>(tg.absolute_offset + uni[i] * gbytes));
            in.read(reinterpret_cast<char*>(gate.data() + i * gbytes), static_cast<std::streamsize>(gbytes));
            in.seekg(static_cast<std::streamoff>(tu.absolute_offset + uni[i] * gbytes));
            in.read(reinterpret_cast<char*>(up.data() + i * gbytes), static_cast<std::streamsize>(gbytes));
            in.seekg(static_cast<std::streamoff>(td.absolute_offset + uni[i] * dbytes));
            in.read(reinterpret_cast<char*>(down.data() + i * dbytes), static_cast<std::streamsize>(dbytes));
        }
        for (const bool grouped : {false, true})
            run_window(shape, gate, up, down, ids, rows, rng, grouped,
                       "real layer " + std::to_string(layer) + (grouped ? " grouped" : " gemv"), 0, &compact);
    }
}

}  // namespace

int main() {
    try {
        const auto device = dc::select_device(0);
        std::printf("device %s (cc %d.%d, %d SMs)\n", device.name.c_str(), device.major, device.minor, device.sms);
        std::mt19937 rng(20261020);
        test_routes(1, 6, 384, 0, 0, 0, rng);
        test_routes(8, 6, 384, 0, 0, 0, rng);
        test_routes(8, 6, 384, 192, 384, 192, rng);
        test_routes(8, 6, 384, 0, 192, 0, rng);
        test_routes(700, 6, 384, 0, 0, 0, rng);       // > 4096 slots: the three-launch form
        test_routes(2048, 6, 384, 192, 384, 192, rng);
        test_routes(3, 8, 64, 0, 0, 0, rng);

        dc::MoeShape small;
        small.hidden = 1024; small.inter = 512; small.experts = 24; small.top_k = 6;
        test_synthetic(small, 24, {1, 2, 3, 4, 5, 8, 13}, false, rng, "synthetic small gemv");
        test_synthetic(small, 24, {1, 8, 40, 130}, true, rng, "synthetic small grouped");
        dc::MoeShape model;  // the model's shape, 384 ids over 12 resident random experts
        test_synthetic(model, 12, {1, 2, 4, 8}, false, rng, "synthetic model-shape gemv");
        test_synthetic(model, 12, {3, 70}, true, rng, "synthetic model-shape grouped");

        if (const char* path = std::getenv("STRIX_DS41_EXPERTS_GGUF")) {
            const char* layer = std::getenv("STRIX_DS41_LAYER");
            test_real(path, layer ? static_cast<std::uint32_t>(std::atoi(layer)) : 20U, std::getenv("STRIX_DS41_ROUTES"), rng);
        }
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf(g_failures ? "%d FAILURES\n" : "all passed\n", g_failures);
    return g_failures ? 1 : 0;
}
