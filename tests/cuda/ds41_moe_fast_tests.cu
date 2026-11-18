// deepseek41 CUDA routed experts, the prefill FAST arm (cuda_ds41_moe_fast.hpp) against the exact path
// (enqueue_moe_layer, the host reference's bits) on the real expert tensors of one layer, all 384 experts resident.
//   strix-cuda-ds41-moe-fast-tests <experts.gguf> [routes.txt] [layer = 20] [rows = 333,2048,8192] [--bench]
// Rows of tokens cycled from the recorded routes (p8-routes format; uniform random routes without a file), x normal
// random (the normed MoE input's scale), router weights in [0.05, 0.6], shared output small normal. Checks per row
// count: relative L2 error of the routed sum (y - shared) and of y, cosine of the routed sum per row (min, mean) of
// fast vs exact, every value finite; on 4 rows also both arms against an FP64 product of the decoded weights on the
// unquantized x (the exact arm's own A8 error for scale). --bench: device ms per layer, exact vs fast (median of 3).
// STRIX_FMOE_SIGN_LUT=1 and/or STRIX_FMOE_WIDE_TILES=1 checks specialization vs legacy bytes, ranges/permutations/events and graph replay.
// STRIX_FMOE_EXPERT_COUNT=N with uniform routes checks a prefix of the real experts (small portability runs).
// STRIX_FMOE_DUMP_PREFIX=PATH writes raw outputs outside the source tree for comparisons across frozen builds.

#include "strix/cuda/ds41_common.hpp"
#include "strix/cuda/ds41_core.hpp"
#include "strix/cuda/ds41_moe.hpp"
#include "strix/cuda/ds41_moe_fast.hpp"
#include "strix/ds41_formats.hpp"
#include "strix/ds41_moe.hpp"
#include "strix/gguf_reader.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

void check_packed_signs() {
    // The packed negation must match scalar signed codes for every codebook/sign pair. This also checks the
    // no-cross-byte-carry condition of the format's nonzero magnitudes (a zero magnitude would need other logic).
    bool equal = true;
    for (std::uint32_t s = 0; s < 128U; ++s) {
        const auto sign = ds::iq2_xxs_sign_byte(s);
        for (const auto grid : ds::iq2_xxs_grid) {
            for (std::uint32_t half = 0; half < 2U; ++half) {
                std::uint32_t mask = 0;
                for (std::uint32_t b = 0; b < 4U; ++b)
                    if ((sign >> (4U * half + b)) & 1U) mask |= 255U << (8U * b);
                const auto word = static_cast<std::uint32_t>(grid >> (32U * half));
                const auto packed = (word ^ mask) + (mask & 0x01010101U);
                for (std::uint32_t b = 0; b < 4U; ++b) {
                    const auto magnitude = static_cast<int>((word >> (8U * b)) & 255U);
                    const auto scalar = static_cast<std::uint8_t>((sign >> (4U * half + b)) & 1U ? -magnitude : magnitude);
                    equal &= magnitude > 0 && ((packed >> (8U * b)) & 255U) == scalar;
                }
            }
        }
    }
    expect(equal, "all 32768 packed IQ2 sign/codebook words match scalar codes");
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
            if (ids.size() != K) throw std::runtime_error("recorded route has fewer than top_k entries");
            tokens.push_back(ids);
        }
    }
    return tokens;
}
std::vector<std::uint32_t> uniform_ids(std::uint32_t rows, std::uint32_t K, std::uint32_t experts, std::mt19937& rng) {
    std::vector<std::uint32_t> all(experts), ids;
    for (std::uint32_t e = 0; e < experts; ++e) all[e] = e;
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t k = 0; k < K; ++k) std::swap(all[k], all[k + rng() % (experts - k)]);
        ids.insert(ids.end(), all.begin(), all.begin() + K);
    }
    return ids;
}

void load(std::ifstream& in, std::uint64_t offset, std::uint64_t bytes, std::uint8_t* device) {
    constexpr std::uint64_t chunk = 64ULL << 20U;
    std::uint8_t* host = nullptr;
    STRIX_CUDA_CHECK(cudaMallocHost(&host, chunk));
    in.seekg(static_cast<std::streamoff>(offset));
    for (std::uint64_t at = 0; at < bytes; at += chunk) {
        const auto n = std::min(chunk, bytes - at);
        in.read(reinterpret_cast<char*>(host), static_cast<std::streamsize>(n));
        if (!in) throw std::runtime_error("short read of expert weights");
        STRIX_CUDA_CHECK(cudaMemcpy(device + at, host, n, cudaMemcpyHostToDevice));
    }
    STRIX_CUDA_CHECK(cudaFreeHost(host));
}

// FP64 product of one row's routed sum (sum_k w_k * down(swiglu(gate . x, up . x))) on the decoded weights
std::vector<double> fp64_routed(std::ifstream& in, const strix::gguf::TensorInfo& tg, const strix::gguf::TensorInfo& tu,
                                const strix::gguf::TensorInfo& td, const dc::MoeShape& s, const float* x, const std::uint32_t* ids,
                                const float* w) {
    const std::uint32_t H = s.hidden, I = s.inter;
    std::vector<double> out(H, 0.0);
    auto read_rows = [&](const strix::gguf::TensorInfo& t, std::uint64_t expert_bytes, std::uint32_t expert, std::vector<std::uint8_t>& buf) {
        buf.resize(expert_bytes);
        in.seekg(static_cast<std::streamoff>(t.absolute_offset + expert * expert_bytes));
        in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(expert_bytes));
    };
    std::vector<std::uint8_t> gb, ub, db;
    std::vector<float> dec(256);
    auto row_dot_iq2 = [&](const std::uint8_t* row, std::uint32_t width, const auto& v) {
        double acc = 0.0;
        for (std::uint32_t b = 0; b < width / 256U; ++b) {
            ds::decode_iq2_xxs(std::span<const std::uint8_t, ds::iq2_xxs_block_bytes>(row + b * ds::iq2_xxs_block_bytes, ds::iq2_xxs_block_bytes),
                               std::span<float, 256>(dec.data(), 256));
            for (std::uint32_t j = 0; j < 256U; ++j) acc += static_cast<double>(dec[j]) * static_cast<double>(v[256U * b + j]);
        }
        return acc;
    };
    for (std::uint32_t k = 0; k < s.top_k; ++k) {
        read_rows(tg, s.gate_expert_bytes(), ids[k], gb);
        read_rows(tu, s.gate_expert_bytes(), ids[k], ub);
        read_rows(td, s.down_expert_bytes(), ids[k], db);
        std::vector<double> act(I);
        for (std::uint32_t o = 0; o < I; ++o) {
            const double g = row_dot_iq2(gb.data() + o * s.gate_row_bytes(), H, x), u = row_dot_iq2(ub.data() + o * s.gate_row_bytes(), H, x);
            const double gc = std::min(g, static_cast<double>(s.limit)), uc = std::clamp(u, -static_cast<double>(s.limit), static_cast<double>(s.limit));
            act[o] = gc / (1.0 + std::exp(-gc)) * uc;
        }
        for (std::uint32_t o = 0; o < H; ++o) {
            const std::uint8_t* row = db.data() + o * s.down_row_bytes();
            double acc = 0.0;
            for (std::uint32_t b = 0; b < I / 256U; ++b) {
                ds::decode_q2_k(std::span<const std::uint8_t, ds::q2_k_block_bytes>(row + b * ds::q2_k_block_bytes, ds::q2_k_block_bytes),
                                std::span<float, 256>(dec.data(), 256));
                for (std::uint32_t j = 0; j < 256U; ++j) acc += static_cast<double>(dec[j]) * act[256U * b + j];
            }
            out[o] += static_cast<double>(w[k]) * acc;
        }
    }
    return out;
}

// STRIX_FMOE_ARMS: the fast arms to run (digits, e.g. "1"; default all)
bool arm_on(int arm) {
    const char* e = std::getenv("STRIX_FMOE_ARMS");
    return !e || std::strchr(e, '0' + arm) != nullptr;
}

struct Errors {
    double rel = 0, cos_min = 1, cos_mean = 0;
};
// routed parts a, b ([rows][H]); relative L2 error of b against a, cosine per row
Errors compare(const std::vector<float>& a, const std::vector<float>& b, std::uint32_t rows, std::uint32_t H) {
    Errors e;
    double num = 0, den = 0;
    for (std::uint32_t r = 0; r < rows; ++r) {
        double ab = 0, aa = 0, bb = 0;
        for (std::uint32_t i = 0; i < H; ++i) {
            const double x = a[std::size_t{r} * H + i], y = b[std::size_t{r} * H + i];
            ab += x * y; aa += x * x; bb += y * y;
            num += (x - y) * (x - y);
        }
        den += aa;
        const double c = aa > 0 && bb > 0 ? ab / std::sqrt(aa * bb) : (aa == bb ? 1.0 : 0.0);
        e.cos_min = std::min(e.cos_min, c);
        e.cos_mean += c;
    }
    e.rel = std::sqrt(num / den);
    e.cos_mean /= rows;
    return e;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <experts.gguf> [routes.txt|-] [layer] [rows,...] [--bench]\n", argv[0]);
        return 2;
    }
    bool bench = false;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench") bench = true;
        else args.emplace_back(argv[i]);
    }
    const std::string path = args[0];
    const std::string routes_path = args.size() > 1 ? args[1] : "-";
    const std::uint32_t layer = args.size() > 2 ? static_cast<std::uint32_t>(std::stoul(args[2])) : 20U;
    std::vector<std::uint32_t> row_counts{333, 2048, 8192};
    if (args.size() > 3) {
        row_counts.clear();
        std::istringstream is(args[3]);
        std::string v;
        while (std::getline(is, v, ',')) row_counts.push_back(static_cast<std::uint32_t>(std::stoul(v)));
    }
    try {
        check_packed_signs();
        const auto device = dc::select_device(0);
        std::printf("device %s, %d SMs\n", device.name.c_str(), device.sms);
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
        if (tg.ggml_type != 16U || tu.ggml_type != 16U || td.ggml_type != 10U || tg.dimensions.size() != 3U ||
            tu.dimensions != tg.dimensions || td.dimensions.size() != 3U || td.dimensions[0] != tg.dimensions[1] ||
            td.dimensions[1] != tg.dimensions[0] || td.dimensions[2] != tg.dimensions[2])
            throw std::runtime_error("benchmark requires matching IQ2_XXS gate/up and Q2_K down expert tensors");
        if (row_counts.empty() || std::find(row_counts.begin(), row_counts.end(), 0U) != row_counts.end())
            throw std::runtime_error("benchmark row counts must be nonempty and positive");
        dc::MoeShape s;
        s.hidden = static_cast<std::uint32_t>(tg.dimensions[0]);
        s.inter = static_cast<std::uint32_t>(tg.dimensions[1]);
        s.experts = static_cast<std::uint32_t>(tg.dimensions[2]);
        // A prefix of the real experts permits small portability checks on a shared device without changing the
        // production dimensions of either matrix. Use uniform routes for this diagnostic (no trace truncation).
        if (const char* limit = std::getenv("STRIX_FMOE_EXPERT_COUNT")) {
            const auto n = static_cast<std::uint32_t>(std::stoul(limit));
            if (n < s.top_k || n > s.experts || routes_path != "-") throw std::runtime_error("invalid expert prefix diagnostic");
            s.experts = n;
        }
        const auto gate_bytes = s.experts * s.gate_expert_bytes(), down_bytes = s.experts * s.down_expert_bytes();
        dc::DeviceBuffer<std::uint8_t> dg(gate_bytes), du(gate_bytes), dd(down_bytes);
        std::ifstream in(path, std::ios::binary);
        load(in, tg.absolute_offset, gate_bytes, dg.get());
        load(in, tu.absolute_offset, gate_bytes, du.get());
        load(in, td.absolute_offset, down_bytes, dd.get());
        std::printf("layer %u: %u experts resident (%.2f GB)\n", layer, s.experts, static_cast<double>(2U * gate_bytes + down_bytes) / 1e9);
        std::vector<std::vector<std::uint32_t>> tokens;
        if (routes_path != "-") {
            tokens = read_routes(routes_path.c_str(), layer, s.top_k);
            if (tokens.empty()) throw std::runtime_error("no recorded routes for the layer");
        }

        const std::uint32_t cap = *std::max_element(row_counts.begin(), row_counts.end());
        const std::uint32_t H = s.hidden, K = s.top_k;
        dc::DeviceBuffer<float> dx(std::size_t{cap} * H), dw(std::size_t{cap} * K), dsh(std::size_t{cap} * H), raw(std::size_t{cap} * K * H),
            y(std::size_t{cap} * H);
        dc::DeviceBuffer<std::int8_t> xc(std::size_t{cap} * H);
        dc::DeviceBuffer<float> xd(std::size_t{cap} * H / 32U);
        dc::DeviceBuffer<std::int32_t> xs(std::size_t{cap} * H / 32U);
        dc::DeviceBuffer<std::uint32_t> did(std::size_t{cap} * K);
        dc::DeviceBuffer<std::uint8_t> ws_exact(dc::moe_layer_workspace_bytes(cap, s)), ws_fast(dc::moe_fast_workspace_bytes(cap, s));
        dc::Stream stream;
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        std::mt19937 rng(20261020);

        for (const std::uint32_t rows : row_counts) {
            std::vector<std::uint32_t> ids;
            if (!tokens.empty())
                for (std::uint32_t r = 0; r < rows; ++r) {
                    const auto& t = tokens[(r * 7U + 3U) % tokens.size()];
                    ids.insert(ids.end(), t.begin(), t.end());
                }
            else ids = uniform_ids(rows, K, s.experts, rng);
            std::vector<std::uint32_t> histogram(s.experts, 0U);
            for (auto id : ids) {
                if (id >= s.experts) throw std::runtime_error("route id outside expert count");
                ++histogram[id];
            }
            auto sorted = histogram;
            std::sort(sorted.begin(), sorted.end());
            const auto used = std::count_if(sorted.begin(), sorted.end(), [](auto n) { return n != 0U; });
            std::printf("hist rows %u source %s recorded_tokens %zu used %zu/%u counts min/median/p95/max %u/%u/%u/%u avoidable_weight_MB %.3f\n",
                        rows, tokens.empty() ? "uniform" : "cycled-recorded", tokens.size(), static_cast<std::size_t>(used), s.experts,
                        sorted.front(), sorted[sorted.size() / 2U], sorted[(sorted.size() - 1U) * 95U / 100U], sorted.back(),
                        static_cast<double>((s.experts - used) * s.expert_bytes()) / 1e6);
            std::vector<float> x(std::size_t{rows} * H), w(std::size_t{rows} * K), sh(std::size_t{rows} * H);
            std::normal_distribution<float> nx(0.0F, 1.0F), ns(0.0F, 0.05F);
            std::uniform_real_distribution<float> uw(0.05F, 0.6F);
            for (auto& v : x) v = nx(rng);
            for (auto& v : w) v = uw(rng);
            for (auto& v : sh) v = ns(rng);
            STRIX_CUDA_CHECK(cudaMemcpy(dx.get(), x.data(), x.size() * 4U, cudaMemcpyHostToDevice));
            STRIX_CUDA_CHECK(cudaMemcpy(dw.get(), w.data(), w.size() * 4U, cudaMemcpyHostToDevice));
            STRIX_CUDA_CHECK(cudaMemcpy(dsh.get(), sh.data(), sh.size() * 4U, cudaMemcpyHostToDevice));
            STRIX_CUDA_CHECK(cudaMemcpy(did.get(), ids.data(), ids.size() * 4U, cudaMemcpyHostToDevice));
            const dc::A8Rows xa8{xc.get(), xd.get(), xs.get(), rows, H, H};
            dc::QuantizeA8View qv;
            qv.x = dx.get(); qv.rows = rows; qv.width = H; qv.out = xa8;
            dc::enqueue_quantize_a8(qv, stream.get());

            dc::MoeLayerView ev;
            ev.shape = s; ev.weights = {dg.get(), du.get(), dd.get()}; ev.rows = rows; ev.split = s.experts / 2U; ev.x = xa8;
            ev.ids = did.get(); ev.route_weights = dw.get(); ev.shared = dsh.get(); ev.raw = raw.get(); ev.y = y.get();
            ev.workspace = ws_exact.get(); ev.capacity = cap;
            dc::MoeFastLayerView fv;
            fv.shape = s; fv.weights = ev.weights; fv.rows = rows; fv.split = ev.split; fv.x = dx.get(); fv.ids = did.get();
            fv.route_weights = dw.get(); fv.shared = dsh.get(); fv.raw = raw.get(); fv.y = y.get(); fv.workspace = ws_fast.get(); fv.capacity = cap;

            // arms: 0 exact; fast 1 INT8 gate/up (the exact path's A8 input) + INT8 down (A8 activation), 2 FP16 gate/up +
            // FP16 down (unquantized input and activation), 3 INT8 gate/up + FP16 down
            auto enqueue = [&](int arm, bool legacy = false) {
                if (arm == 0) {
                    dc::enqueue_moe_layer(ev, stream.get());
                    return;
                }
                dc::MoeFastLayerView v = fv;
                if (arm == 1 && !legacy)
                    if (const char* entries = std::getenv("STRIX_FMOE_SIGN_LUT"))
                        v.signed_lut = std::stoul(entries) != 0U;
                if (arm != 2) v.xa = xa8;
                if (arm == 1 && !legacy)
                    if (const char* wide = std::getenv("STRIX_FMOE_WIDE_TILES")) v.wide_tiles = std::stoul(wide) != 0U;
                v.down_i8 = arm == 1;
                dc::enqueue_moe_fast_layer(v, stream.get());
            };
            auto run = [&](int arm, bool legacy = false) {
                STRIX_CUDA_CHECK(cudaMemsetAsync(y.get(), 0x7f, std::size_t{rows} * H * 4U, stream.get()));
                enqueue(arm, legacy);
                stream.sync();
                std::vector<float> out(std::size_t{rows} * H);
                STRIX_CUDA_CHECK(cudaMemcpy(out.data(), y.get(), out.size() * 4U, cudaMemcpyDeviceToHost));
                return out;
            };
            const auto y_exact = run(0);
            std::vector<float> r_exact(y_exact.size());
            for (std::size_t i = 0; i < y_exact.size(); ++i) r_exact[i] = y_exact[i] - sh[i];
            const char* names[4] = {"exact", "fast-i8 ", "fast-f16", "fast-i8f"};
            std::vector<float> r_arm[4];
            for (int arm = 1; arm <= 3; ++arm) {
                if (!arm_on(arm)) continue;
                const auto y_fast = run(arm);
                const auto y_fast2 = run(arm);
                bool finite = true;
                for (const float v : y_fast) finite &= std::isfinite(v);
                r_arm[arm].resize(y_fast.size());
                for (std::size_t i = 0; i < y_fast.size(); ++i) r_arm[arm][i] = y_fast[i] - sh[i];
                const Errors er = compare(r_exact, r_arm[arm], rows, H), ey = compare(y_exact, y_fast, rows, H);
                const bool deterministic = std::memcmp(y_fast.data(), y_fast2.data(), y_fast.size() * 4U) == 0;
                std::printf("rows %5u %s: vs exact routed sum rel L2 %.3e, cos min %.6f mean %.6f; y rel L2 %.3e; finite %s, repeatable %s\n",
                            rows, names[arm], er.rel, er.cos_min, er.cos_mean, ey.rel, finite ? "yes" : "NO",
                            deterministic ? "yes" : "NO");
                expect(finite, "fast output finite");
                expect(deterministic, "fast output repeatable");
                expect(er.rel < 0.05 && er.cos_min > 0.99, "fast routed sum close to the exact path");
                if (arm == 1 && (std::getenv("STRIX_FMOE_SIGN_LUT") || std::getenv("STRIX_FMOE_WIDE_TILES"))) {
                    const auto baseline = run(arm, true);
                    const bool identical = std::memcmp(y_fast.data(), baseline.data(), y_fast.size() * sizeof(float)) == 0;
                    std::printf("           specialized vs legacy fast bytes: %s\n", identical ? "IDENTICAL" : "DIFFERENT");
                    expect(identical, "signed LUT preserves legacy fast output bytes");
                    dc::MoeFastLayerView v = fv;
                    v.xa = xa8;
                    if (const char* lut = std::getenv("STRIX_FMOE_SIGN_LUT")) v.signed_lut = std::stoul(lut) != 0U;
                    if (const char* wide = std::getenv("STRIX_FMOE_WIDE_TILES")) v.wide_tiles = std::stoul(wide) != 0U;
                    // Ranges have local tensor bases and an explicit identity map. The trace order and the caller's
                    // route weights stay untouched; the final combine must match the whole-layer launch exactly.
                    std::vector<std::uint32_t> identity(s.experts);
                    for (std::uint32_t i = 0; i < s.experts; ++i) identity[i] = i;
                    dc::DeviceBuffer<std::uint32_t> perm(s.experts), mapped(std::size_t{rows} * K);
                    perm.upload(identity);
                    const std::uint32_t cut = s.experts / 3U + 1U;
                    dc::MoeRange ranges[2];
                    ranges[0] = {0U, cut, 0U, ev.weights, nullptr};
                    ranges[1] = {cut, s.experts, cut,
                                 {dg.get() + cut * s.gate_expert_bytes(), du.get() + cut * s.gate_expert_bytes(),
                                  dd.get() + cut * s.down_expert_bytes()}, nullptr};
                    dc::enqueue_moe_fast_layer_ranges(v, perm.get(), mapped.get(), ranges, 2U, stream.get());
                    stream.sync();
                    const auto ranged = y.download(std::size_t{rows} * H);
                    expect(std::memcmp(y_fast.data(), ranged.data(), y_fast.size() * sizeof(float)) == 0,
                           "signed LUT range/local-base/mapping bytes");
                    identity[0] = 1U;
                    identity[1] = 0U;
                    perm.upload(identity);
                    cudaEvent_t ready = nullptr;
                    STRIX_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
                    STRIX_CUDA_CHECK(cudaEventRecord(ready, stream.get()));
                    dc::MoeRange swapped[3];
                    swapped[0] = {0U, 1U, 0U,
                                  {dg.get() + s.gate_expert_bytes(), du.get() + s.gate_expert_bytes(),
                                   dd.get() + s.down_expert_bytes()}, ready};
                    swapped[1] = {1U, 2U, 1U, ev.weights, nullptr};
                    swapped[2] = {2U, s.experts, 2U,
                                  {dg.get() + 2U * s.gate_expert_bytes(), du.get() + 2U * s.gate_expert_bytes(),
                                   dd.get() + 2U * s.down_expert_bytes()}, nullptr};
                    dc::enqueue_moe_fast_layer_ranges(v, perm.get(), mapped.get(), swapped, 3U, stream.get());
                    stream.sync();
                    const auto reordered = y.download(std::size_t{rows} * H);
                    expect(std::memcmp(y_fast.data(), reordered.data(), y_fast.size() * sizeof(float)) == 0,
                           "signed LUT nonidentity permutation/event/range bytes");
                    STRIX_CUDA_CHECK(cudaEventDestroy(ready));
                    // Capture only after the ordinary launch initialized device attributes and the codebook.
                    cudaGraph_t graph = nullptr;
                    cudaGraphExec_t exec = nullptr;
                    STRIX_CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
                    dc::enqueue_moe_fast_layer(v, stream.get());
                    STRIX_CUDA_CHECK(cudaStreamEndCapture(stream.get(), &graph));
                    STRIX_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
                    STRIX_CUDA_CHECK(cudaMemsetAsync(y.get(), 0x7f, y_fast.size() * sizeof(float), stream.get()));
                    STRIX_CUDA_CHECK(cudaGraphLaunch(exec, stream.get()));
                    stream.sync();
                    const auto graphed = y.download(std::size_t{rows} * H);
                    expect(std::memcmp(y_fast.data(), graphed.data(), y_fast.size() * sizeof(float)) == 0,
                           "signed LUT graph replay bytes");
                    STRIX_CUDA_CHECK(cudaGraphExecDestroy(exec));
                    STRIX_CUDA_CHECK(cudaGraphDestroy(graph));
                    if (rows == row_counts.front()) {
                        auto bad = v;
                        bad.signed_lut = true;
                        bad.xa = {};
                        bool rejected = false;
                        try { dc::enqueue_moe_fast_layer(bad, stream.get()); }
                        catch (const std::invalid_argument&) { rejected = true; }
                        expect(rejected, "signed LUT with FP16 gate rejected before launch");
                        bad.rows = 0U;
                        dc::enqueue_moe_fast_layer(bad, stream.get());  // empty work is a no-op
                    }
                }
                // A frozen baseline can be compared byte for byte across kernel builds. Opt-in diagnostic artifacts
                // belong outside the source tree; no dump or extra synchronization in the production driver.
                if (const char* prefix = std::getenv("STRIX_FMOE_DUMP_PREFIX")) {
                    const auto name = std::string(prefix) + "-" + std::to_string(layer) + "-" + std::to_string(rows) + "-" +
                                      std::to_string(arm) + ".f32";
                    std::ofstream out(name, std::ios::binary);
                    out.write(reinterpret_cast<const char*>(y_fast.data()), static_cast<std::streamsize>(y_fast.size() * sizeof(float)));
                    if (!out) throw std::runtime_error("cannot write output dump " + name);
                }
            }

            if (rows == row_counts.front() && arm_on(1) && arm_on(2) && arm_on(3)) {  // FP64 truth on a few rows
                std::vector<float> tt, ta[4];
                for (const std::uint32_t r : {0U, 1U, 2U, rows - 1U}) {
                    const auto truth = fp64_routed(in, tg, tu, td, s, x.data() + std::size_t{r} * H, ids.data() + std::size_t{r} * K,
                                                   w.data() + std::size_t{r} * K);
                    for (std::uint32_t i = 0; i < H; ++i) {
                        tt.push_back(static_cast<float>(truth[i]));
                        ta[0].push_back(r_exact[std::size_t{r} * H + i]);
                        ta[1].push_back(r_arm[1][std::size_t{r} * H + i]);
                        ta[2].push_back(r_arm[2][std::size_t{r} * H + i]);
                        ta[3].push_back(r_arm[3][std::size_t{r} * H + i]);
                    }
                }
                const Errors q0 = compare(tt, ta[0], 4, H), q1 = compare(tt, ta[1], 4, H), q2 = compare(tt, ta[2], 4, H),
                             q3 = compare(tt, ta[3], 4, H);
                std::printf("           vs FP64 of the decoded weights on unquantized x (4 rows): rel exact %.3e, fast-i8 %.3e, fast-f16 %.3e, "
                            "fast-i8f %.3e (cos min %.6f %.6f %.6f %.6f)\n",
                            q0.rel, q1.rel, q2.rel, q3.rel, q0.cos_min, q1.cos_min, q2.cos_min, q3.cos_min);
                expect(q1.rel < 0.03 && q2.rel < 0.02, "fast arms close to the FP64 product");
            }
            if (bench) {
                auto time = [&](int arm) {
                    std::vector<float> samples;
                    for (int rep = 0; rep < 3; ++rep) {
                        cudaEventRecord(e0, stream.get());
                        enqueue(arm);
                        cudaEventRecord(e1, stream.get());
                        stream.sync();
                        float ms = 0;
                        cudaEventElapsedTime(&ms, e0, e1);
                        samples.push_back(ms);
                    }
                    std::sort(samples.begin(), samples.end());
                    return samples[samples.size() / 2U];
                };
                const double macs = static_cast<double>(rows) * K * 3.0 * H * s.inter;
                const float t0 = time(0);
                std::printf("           layer ms (median of 3): exact %.2f (%.0f TOPS)", t0, 2.0 * macs / (t0 * 1e9));
                for (int arm = 1; arm <= 3; ++arm) {
                    if (!arm_on(arm)) continue;
                    const float ta_ms = time(arm);
                    std::printf(", %s %.2f (x%.2f)", names[arm], ta_ms, t0 / ta_ms);
                }
                std::printf("\n");
            }
            std::fflush(stdout);
        }
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf(g_failures ? "%d FAILURES\n" : "all passed\n", g_failures);
    return g_failures ? 1 : 0;
}
