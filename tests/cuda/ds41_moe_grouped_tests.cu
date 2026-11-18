// deepseek41 CUDA routed experts: the prefill grouped GEMM against the decode GEMVs (themselves the host reference's
// bits, strix-cuda-ds41-moe-tests) at prefill row counts, bit for bit: gate and up products, the A8 activations
// (codes, deltas, sums), the raw slot vectors and y, through the entry points and through enqueue_moe_layer.
// Routes: uniform random (every expert used), skewed with designed per-expert counts (0, 1, 2, 15..17, 31..33,
// 63..65, 127..129 entries, an expert in every row and one in all rows but one), and recorded routes tiled.
// Synthetic: all 384 experts resident at a reduced shape (hidden 1024, inter 512), also against the host reference
// at the smaller row counts. With STRIX_DS41_EXPERTS_GGUF: the real layers (STRIX_DS41_LAYERS, default "2,20"),
// all 384 experts resident (~4 GB), routed by STRIX_DS41_ROUTES (p8-routes format) where given.

#include "strix/cuda/ds41_common.hpp"
#include "strix/cuda/ds41_core.hpp"
#include "strix/cuda/ds41_moe.hpp"
#include "strix/ds41_formats.hpp"
#include "strix/ds41_moe.hpp"
#include "strix/gguf_reader.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
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

template <class T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b, const std::string& what) {
    if (a.size() != b.size()) { expect(false, what + " (size)"); return false; }
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(T)) != 0) {
            std::ostringstream s;
            s << what << " differs at " << i << " of " << a.size();
            if constexpr (std::is_floating_point_v<T>) s << ": " << a[i] << " vs " << b[i];
            else s << ": " << static_cast<long long>(a[i]) << " vs " << static_cast<long long>(b[i]);
            expect(false, s.str());
            return false;
        }
    return true;
}

// ------------------------------------------------------------------------------------------------- inputs
std::uint16_t to_half_bits(float f) {
    const auto u = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = (u >> 16U) & 0x8000U;
    const int e = std::clamp(static_cast<int>((u >> 23U) & 0xffU) - 127 + 15, 1, 30);
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(e) << 10U) | ((u >> 13U) & 0x3ffU));
}
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
std::vector<float> random_x(std::size_t n, std::size_t width, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> x(n);
    for (auto& v : x) v = normal(rng);
    for (std::size_t r = 0; r * width < n; r += 7) {  // a zero group and a tiny group in every 7th row (A8 special cases)
        float* row = x.data() + r * width;
        std::fill(row + 32, row + 64, 0.0F);
        for (int i = 64; i < 96; ++i) row[i] = 1e-37F * static_cast<float>(i - 80);
    }
    return x;
}

// uniform: every row K distinct experts drawn uniformly
std::vector<std::uint32_t> uniform_ids(std::uint32_t rows, std::uint32_t K, std::uint32_t experts, std::mt19937& rng) {
    std::vector<std::uint32_t> all(experts), ids;
    for (std::uint32_t e = 0; e < experts; ++e) all[e] = e;
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t k = 0; k < K; ++k) std::swap(all[k], all[k + rng() % (experts - k)]);
        ids.insert(ids.end(), all.begin(), all.begin() + K);
    }
    return ids;
}
// designed: experts 0.. get the listed entry counts (expert 0 none), the others fill the rows skewed towards low ids
std::vector<std::uint32_t> designed_ids(std::uint32_t rows, std::uint32_t K, std::uint32_t experts, std::mt19937& rng) {
    const std::vector<std::uint32_t> counts{0, 1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, rows, rows - 1U};
    const auto designated = static_cast<std::uint32_t>(counts.size());
    std::vector<std::vector<std::uint32_t>> row_ids(rows);
    std::vector<std::uint32_t> order(rows);
    for (std::uint32_t r = 0; r < rows; ++r) order[r] = r;
    for (std::uint32_t e = designated; e-- > 0;) {  // the big counts first
        std::shuffle(order.begin(), order.end(), rng);
        std::uint32_t placed = 0;
        for (std::uint32_t i = 0; i < rows && placed < std::min(counts[e], rows); ++i)
            if (row_ids[order[i]].size() < K) { row_ids[order[i]].push_back(e); ++placed; }
        if (placed != std::min(counts[e], rows)) throw std::runtime_error("designed routes: no room");
    }
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<std::uint32_t> ids;
    for (auto& r : row_ids) {
        while (r.size() < K) {
            const double v = u(rng);
            const auto e = designated + static_cast<std::uint32_t>(v * v * v * (experts - designated));
            if (std::find(r.begin(), r.end(), e) == r.end()) r.push_back(e);
        }
        std::shuffle(r.begin(), r.end(), rng);
        ids.insert(ids.end(), r.begin(), r.end());
    }
    return ids;
}
std::vector<std::vector<std::uint32_t>> read_routes(const char* path, std::uint32_t layer, std::uint32_t K) {
    std::vector<std::vector<std::uint32_t>> tokens;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ls(line);
        std::string item;
        while (ls >> item) {
            const auto colon = item.find(':');
            if (colon == std::string::npos || std::stoul(item.substr(0, colon)) != layer) continue;
            std::vector<std::uint32_t> ids;
            std::istringstream is(item.substr(colon + 1));
            std::string v;
            while (std::getline(is, v, ',') && ids.size() < K) ids.push_back(static_cast<std::uint32_t>(std::stoul(v)));
            tokens.push_back(ids);
        }
    }
    return tokens;
}
std::vector<std::uint32_t> tiled_ids(const std::vector<std::vector<std::uint32_t>>& tokens, std::uint32_t rows, std::uint32_t first) {
    std::vector<std::uint32_t> ids;
    for (std::uint32_t r = 0; r < rows; ++r) {
        const auto& t = tokens[(first + r) % tokens.size()];
        ids.insert(ids.end(), t.begin(), t.end());
    }
    return ids;
}

// ------------------------------------------------------------------------------------------------- one case
struct Experts {
    const std::uint8_t* gate;
    const std::uint8_t* up;
    const std::uint8_t* down;
};
struct HostExperts {  // for the host reference (synthetic only)
    const std::vector<std::uint8_t>* gate = nullptr;
    const std::vector<std::uint8_t>* up = nullptr;
    const std::vector<std::uint8_t>* down = nullptr;
};

struct PathOut {
    std::vector<float> gate, up, raw, y;
    std::vector<std::int8_t> codes;
    std::vector<float> deltas;
    std::vector<std::int32_t> sums;
};

void run_case(const dc::MoeShape& shape, const Experts& w, const std::vector<std::uint32_t>& ids, std::uint32_t rows, std::mt19937& rng,
              const std::string& label, const HostExperts* host) {
    const std::uint32_t H = shape.hidden, I = shape.inter, K = shape.top_k, slots = rows * K, split = shape.experts / 2U;
    const auto x = random_x(std::size_t{rows} * H, H, rng);
    std::vector<float> weights(slots), shared(std::size_t{rows} * H);
    std::uniform_real_distribution<float> uw(0.05F, 0.6F);
    for (auto& v : weights) v = uw(rng);
    std::normal_distribution<float> ns(0.0F, 0.05F);
    for (auto& v : shared) v = ns(rng);
    std::vector<std::uint32_t> per_expert(shape.experts, 0);
    for (const auto e : ids) ++per_expert[e];
    const auto used = static_cast<std::uint32_t>(std::count_if(per_expert.begin(), per_expert.end(), [](std::uint32_t c) { return c > 0; }));
    const auto most = *std::max_element(per_expert.begin(), per_expert.end());

    dc::DeviceBuffer<float> dx(x.size()), dw(weights.size()), dsh(shared.size());
    dx.upload(x); dw.upload(weights); dsh.upload(shared);
    dc::DeviceBuffer<std::int8_t> xc(x.size());
    dc::DeviceBuffer<float> xd(x.size() / 32U);
    dc::DeviceBuffer<std::int32_t> xs(x.size() / 32U);
    const dc::A8Rows xa8{xc.get(), xd.get(), xs.get(), rows, H, H};
    dc::QuantizeA8View qv;
    qv.x = dx.get(); qv.rows = rows; qv.width = H; qv.out = xa8;
    dc::enqueue_quantize_a8(qv, nullptr);
    dc::DeviceBuffer<std::uint32_t> did(ids.size());
    did.upload(ids);

    const std::size_t ws_bytes = dc::moe_layer_workspace_bytes(rows, shape);
    dc::DeviceBuffer<std::uint8_t> ws(ws_bytes);
    const auto parts = dc::moe_layer_workspace_views(ws.get(), rows, shape);
    dc::MoeRoutesView rv;
    rv.ids = did.get(); rv.rows = rows; rv.top_k = K; rv.experts = shape.experts; rv.split = split;
    rv.scratch = parts.routes_scratch; rv.out = parts.routes;
    dc::enqueue_moe_routes(rv, nullptr);

    dc::DeviceBuffer<std::int8_t> ac(std::size_t{slots} * I);
    dc::DeviceBuffer<float> ad(std::size_t{slots} * I / 32U), go(std::size_t{slots} * I), uo(std::size_t{slots} * I),
        raw(std::size_t{slots} * H), y(std::size_t{rows} * H);
    dc::DeviceBuffer<std::int32_t> as(std::size_t{slots} * I / 32U);
    dc::DeviceBuffer<std::uint8_t> gscratch(dc::routed_grouped_workspace_bytes(rows, shape));

    auto run = [&](bool grouped) {
        // poison the outputs: every element must be written
        STRIX_CUDA_CHECK(cudaMemset(ac.get(), grouped ? 0x55 : 0x33, ac.size()));
        STRIX_CUDA_CHECK(cudaMemset(ad.get(), grouped ? 0x7f : 0x3f, ad.size() * 4U));
        STRIX_CUDA_CHECK(cudaMemset(as.get(), grouped ? 0x7f : 0x3f, as.size() * 4U));
        STRIX_CUDA_CHECK(cudaMemset(go.get(), grouped ? 0x7f : 0x3f, go.size() * 4U));
        STRIX_CUDA_CHECK(cudaMemset(uo.get(), grouped ? 0x7f : 0x3f, uo.size() * 4U));
        STRIX_CUDA_CHECK(cudaMemset(raw.get(), grouped ? 0x7f : 0x3f, raw.size() * 4U));
        dc::RoutedGateUpView g;
        g.shape = shape; g.weights = {w.gate, w.up, w.down}; g.routes = parts.routes; g.max_union = std::min(slots, shape.experts);
        g.max_rows = std::min(rows, 8U); g.x = xa8; g.act = {ac.get(), ad.get(), as.get(), slots, I, I};
        g.gate_out = go.get(); g.up_out = uo.get();
        dc::RoutedDownView d;
        d.shape = shape; d.weights = g.weights; d.routes = parts.routes; d.max_union = g.max_union; d.max_rows = g.max_rows;
        d.act = g.act; d.raw = raw.get();
        if (grouped) {
            dc::enqueue_routed_gate_up_grouped(g, rows, gscratch.get(), nullptr);
            dc::enqueue_routed_down_grouped(d, rows, gscratch.get(), nullptr);
        } else {
            dc::enqueue_routed_gate_up(g, nullptr);
            dc::enqueue_routed_down(d, nullptr);
        }
        dc::MoeCombineView c{raw.get(), dsh.get(), dw.get(), parts.routes.order, parts.routes.low_counts, rows, K, H, -1, y.get()};
        dc::enqueue_moe_combine(c, nullptr);
        STRIX_CUDA_CHECK(cudaDeviceSynchronize());
        PathOut o;
        o.gate = go.download(); o.up = uo.download(); o.raw = raw.download(); o.y = y.download();
        o.codes = ac.download(); o.deltas = ad.download(); o.sums = as.download();
        return o;
    };
    const PathOut gemv = run(false);
    const PathOut grouped = run(true);
    bool ok = true;
    ok &= same_bits(gemv.gate, grouped.gate, label + " gate");
    ok &= same_bits(gemv.up, grouped.up, label + " up");
    ok &= same_bits(gemv.codes, grouped.codes, label + " act codes");
    ok &= same_bits(gemv.deltas, grouped.deltas, label + " act deltas");
    ok &= same_bits(gemv.sums, grouped.sums, label + " act sums");
    ok &= same_bits(gemv.raw, grouped.raw, label + " raw");
    ok &= same_bits(gemv.y, grouped.y, label + " y");

    // the layer driver (its own workspace carve): grouped from 64 rows vs GEMVs only
    std::vector<float> layer_y[2], layer_raw[2];
    for (int gi = 0; gi < 2; ++gi) {
        STRIX_CUDA_CHECK(cudaMemset(raw.get(), 0x7f, raw.size() * 4U));
        STRIX_CUDA_CHECK(cudaMemset(y.get(), 0x7f, y.size() * 4U));
        dc::MoeLayerView l;
        l.shape = shape; l.weights = {w.gate, w.up, w.down}; l.rows = rows; l.split = split; l.x = xa8; l.ids = did.get();
        l.route_weights = dw.get(); l.shared = dsh.get(); l.raw = raw.get(); l.y = y.get(); l.workspace = ws.get(); l.capacity = rows;
        l.grouped_min_rows = gi ? 64U : 0U;
        dc::enqueue_moe_layer(l, nullptr);
        STRIX_CUDA_CHECK(cudaDeviceSynchronize());
        layer_y[gi] = y.download();
        layer_raw[gi] = raw.download();
    }
    ok &= same_bits(gemv.raw, layer_raw[0], label + " layer gemv raw");
    ok &= same_bits(gemv.y, layer_y[0], label + " layer gemv y");
    ok &= same_bits(gemv.raw, layer_raw[1], label + " layer grouped raw");
    ok &= same_bits(gemv.y, layer_y[1], label + " layer grouped y");

    // the layer by ranges (enqueue_moe_layer_ranges): a random half of the experts "resident" in one buffer, the rest
    // in groups of 37 in a "slot" buffer, both in virtual order (resident ascending id, then the rest ascending id)
    {
        const std::uint32_t E = shape.experts;
        std::vector<std::uint32_t> ids_all(E);
        for (std::uint32_t e = 0; e < E; ++e) ids_all[e] = e;
        std::shuffle(ids_all.begin(), ids_all.end(), rng);
        const std::uint32_t res = E / 2U;
        std::vector<std::uint32_t> order(ids_all.begin(), ids_all.begin() + res), rest(ids_all.begin() + res, ids_all.end());
        std::sort(order.begin(), order.end());
        std::sort(rest.begin(), rest.end());
        order.insert(order.end(), rest.begin(), rest.end());
        std::vector<std::uint32_t> perm(E);
        for (std::uint32_t v = 0; v < E; ++v) perm[order[v]] = v;
        const std::uint64_t gb = shape.gate_expert_bytes(), db = shape.down_expert_bytes();
        dc::DeviceBuffer<std::uint8_t> rbuf(res * (2U * gb + db)), sbuf((E - res) * (2U * gb + db));
        for (std::uint32_t v = 0; v < E; ++v) {
            const bool r = v < res;
            const std::uint64_t n = r ? res : E - res, k = r ? v : v - res;
            std::uint8_t* base = r ? rbuf.get() : sbuf.get();
            const std::uint64_t e = order[v];
            STRIX_CUDA_CHECK(cudaMemcpy(base + k * gb, w.gate + e * gb, gb, cudaMemcpyDeviceToDevice));
            STRIX_CUDA_CHECK(cudaMemcpy(base + n * gb + k * gb, w.up + e * gb, gb, cudaMemcpyDeviceToDevice));
            STRIX_CUDA_CHECK(cudaMemcpy(base + 2U * n * gb + k * db, w.down + e * db, db, cudaMemcpyDeviceToDevice));
        }
        dc::DeviceBuffer<std::uint32_t> dperm(E), mapped(std::size_t{rows} * K);
        dperm.upload(perm);
        std::vector<dc::MoeRange> ranges;
        ranges.push_back({0U, res, 0U, {rbuf.get(), rbuf.get() + res * gb, rbuf.get() + 2U * res * gb}, nullptr});
        const std::uint64_t sn = E - res;
        cudaEvent_t ev;
        STRIX_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        STRIX_CUDA_CHECK(cudaEventRecord(ev, nullptr));
        for (std::uint32_t f = res; f < E; f += 37U)
            ranges.push_back({f, std::min(E, f + 37U), res, {sbuf.get(), sbuf.get() + sn * gb, sbuf.get() + 2U * sn * gb}, ev});
        for (int gi = 0; gi < 2; ++gi) {
            STRIX_CUDA_CHECK(cudaMemset(raw.get(), 0x7f, raw.size() * 4U));
            STRIX_CUDA_CHECK(cudaMemset(y.get(), 0x7f, y.size() * 4U));
            dc::MoeRangesView rv2;
            auto& l = rv2.layer;
            l.shape = shape; l.rows = rows; l.split = split; l.x = xa8; l.ids = did.get();
            l.route_weights = dw.get(); l.shared = dsh.get(); l.raw = raw.get(); l.y = y.get(); l.workspace = ws.get(); l.capacity = rows;
            l.grouped_min_rows = gi ? 64U : 0U;
            rv2.perm = dperm.get();
            rv2.mapped_ids = mapped.get();
            rv2.ranges = ranges.data();
            rv2.range_count = static_cast<std::uint32_t>(ranges.size());
            dc::enqueue_moe_layer_ranges(rv2, nullptr);
            STRIX_CUDA_CHECK(cudaDeviceSynchronize());
            ok &= same_bits(gemv.raw, raw.download(), label + (gi ? " ranges grouped raw" : " ranges gemv raw"));
            ok &= same_bits(gemv.y, y.download(), label + (gi ? " ranges grouped y" : " ranges gemv y"));
        }
        cudaEventDestroy(ev);
    }

    const char* host_note = "";
    if (host) {  // the host reference (whole tensors of all experts)
        std::vector<ds::A8Activation> xa;
        for (std::uint32_t r = 0; r < rows; ++r) xa.push_back(ds::quantize_a8(std::span<const float>(x.data() + std::size_t{r} * H, H)));
        const auto routes = ds::build_window_routes(ids, rows, K, split);
        ds::ExpertTensors t;
        t.gate = *host->gate; t.up = *host->up; t.down = *host->down; t.hidden = H; t.inter = I; t.limit = shape.limit;
        const auto ref = ds::reference_window(t, xa, routes, weights, shared, K);
        std::vector<std::int8_t> codes(std::size_t{slots} * I);
        for (std::uint32_t s = 0; s < slots; ++s)
            std::copy(ref.activation[s].codes.begin(), ref.activation[s].codes.end(), codes.begin() + std::size_t{s} * I);
        ok &= same_bits(ref.gate, grouped.gate, label + " host gate");
        ok &= same_bits(codes, grouped.codes, label + " host act codes");
        ok &= same_bits(ref.raw, grouped.raw, label + " host raw");
        ok &= same_bits(ref.y, grouped.y, label + " host y");
        host_note = ", host reference";
    }
    std::printf("%-48s rows %5u union %3u most %5u: %s (grouped vs gemv: gate, up, act, raw, y; layer driver, ranges%s)\n", label.c_str(), rows,
                used, most, ok ? "bitwise equal" : "DIFFERS", host_note);
    std::fflush(stdout);
}

void load(std::ifstream& in, std::uint64_t offset, std::uint64_t bytes, std::uint8_t* device) {
    constexpr std::uint64_t chunk = 64ULL << 20U;
    std::vector<char> host(chunk);
    in.seekg(static_cast<std::streamoff>(offset));
    for (std::uint64_t at = 0; at < bytes; at += chunk) {
        const auto n = std::min(chunk, bytes - at);
        in.read(host.data(), static_cast<std::streamsize>(n));
        STRIX_CUDA_CHECK(cudaMemcpy(device + at, host.data(), n, cudaMemcpyHostToDevice));
    }
}

}  // namespace

int main() {
    try {
        const auto device = dc::select_device(0);
        std::printf("device %s (cc %d.%d, %d SMs)\n", device.name.c_str(), device.major, device.minor, device.sms);
        std::mt19937 rng(20261021);

        {  // synthetic: all 384 experts resident at a reduced shape
            dc::MoeShape s;
            s.hidden = 1024; s.inter = 512;
            const auto gate = random_iq2(s.gate_expert_bytes() / ds::iq2_xxs_block_bytes * s.experts, rng);
            const auto up = random_iq2(s.gate_expert_bytes() / ds::iq2_xxs_block_bytes * s.experts, rng);
            const auto down = random_q2k(s.down_expert_bytes() / ds::q2_k_block_bytes * s.experts, rng);
            dc::DeviceBuffer<std::uint8_t> dg(gate.size()), du(up.size()), dd(down.size());
            dg.upload(gate); du.upload(up); dd.upload(down);
            const Experts w{dg.get(), du.get(), dd.get()};
            const HostExperts h{&gate, &up, &down};
            for (const std::uint32_t rows : {64U, 65U, 200U})
                run_case(s, w, uniform_ids(rows, s.top_k, s.experts, rng), rows, rng, "synthetic uniform", &h);
            run_case(s, w, designed_ids(256, s.top_k, s.experts, rng), 256, rng, "synthetic designed counts", &h);
            for (const std::uint32_t rows : {1024U, 4096U}) {
                run_case(s, w, uniform_ids(rows, s.top_k, s.experts, rng), rows, rng, "synthetic uniform", nullptr);
                run_case(s, w, designed_ids(rows, s.top_k, s.experts, rng), rows, rng, "synthetic designed counts", nullptr);
            }
        }

        if (const char* path = std::getenv("STRIX_DS41_EXPERTS_GGUF")) {
            const auto file = strix::gguf::read_file(path);
            const auto find = [&](const std::string& name) -> const strix::gguf::TensorInfo& {
                for (const auto& t : file.tensors)
                    if (t.name == name) return t;
                throw std::runtime_error("missing tensor " + name);
            };
            const char* layers_env = std::getenv("STRIX_DS41_LAYERS");
            const char* routes_path = std::getenv("STRIX_DS41_ROUTES");
            std::istringstream ls(layers_env ? layers_env : "2,20");
            std::string item;
            while (std::getline(ls, item, ',')) {
                const auto layer = static_cast<std::uint32_t>(std::stoul(item));
                const std::string p = "blk." + std::to_string(layer) + ".";
                const auto& tg = find(p + "ffn_gate_exps.weight");
                const auto& tu = find(p + "ffn_up_exps.weight");
                const auto& td = find(p + "ffn_down_exps.weight");
                expect(tg.ggml_type == 16 && tu.ggml_type == 16 && td.ggml_type == 10, "real tensor types IQ2_XXS / Q2_K");
                dc::MoeShape s;
                s.hidden = static_cast<std::uint32_t>(tg.dimensions[0]);
                s.inter = static_cast<std::uint32_t>(tg.dimensions[1]);
                s.experts = static_cast<std::uint32_t>(tg.dimensions[2]);
                dc::DeviceBuffer<std::uint8_t> dg(tg.byte_size), du(tu.byte_size), dd(td.byte_size);
                {
                    std::ifstream in(path, std::ios::binary);
                    load(in, tg.absolute_offset, tg.byte_size, dg.get());
                    load(in, tu.absolute_offset, tu.byte_size, du.get());
                    load(in, td.absolute_offset, td.byte_size, dd.get());
                }
                const Experts w{dg.get(), du.get(), dd.get()};
                const std::string name = "real layer " + std::to_string(layer);
                if (routes_path) {
                    const auto tokens = read_routes(routes_path, layer, s.top_k);
                    expect(!tokens.empty(), name + " recorded routes");
                    if (!tokens.empty())
                        for (const std::uint32_t rows : {64U, 333U, 2048U})
                            run_case(s, w, tiled_ids(tokens, rows, 17U * rows % static_cast<std::uint32_t>(tokens.size())), rows, rng,
                                     name + " recorded routes", nullptr);
                }
                run_case(s, w, uniform_ids(1024, s.top_k, s.experts, rng), 1024, rng, name + " uniform", nullptr);
                run_case(s, w, designed_ids(4096, s.top_k, s.experts, rng), 4096, rng, name + " designed counts", nullptr);
            }
        }
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf(g_failures ? "%d FAILURES\n" : "all passed\n", g_failures);
    return g_failures ? 1 : 0;
}
