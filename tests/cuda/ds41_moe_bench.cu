// deepseek41 CUDA routed experts: timings on the real expert tensors of one layer, all 384 experts resident.
//   strix-cuda-ds41-moe-bench <experts.gguf> <routes.txt> [layer = 20] [windows = 40] [grouped rows = 512,2048,8192]
// Decode: windows of R consecutive recorded tokens (R = 1, 2, 4, 8), cold L2 (a 256 MB read before every window);
// per window routes + gate/up + down + combine, and gate/up and down alone. GB/s = union experts' weight bytes / time.
// Prefill: R rows of tokens cycled from the recorded routes through the grouped entry points.

#include "strix/cuda/ds41_common.hpp"
#include "strix/cuda/ds41_moe.hpp"
#include "strix/gguf_reader.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace dc = strix::ds41::cuda;

namespace {

__global__ void flush_kernel(const int4* p, std::size_t n, int* sink) {
    int acc = 0;
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += std::size_t{gridDim.x} * blockDim.x) {
        const int4 v = __ldcg(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678) *sink = acc;
}

std::vector<std::vector<std::uint32_t>> read_routes(const char* path, std::uint32_t layer) {
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
            while (std::getline(is, v, ',')) ids.push_back(static_cast<std::uint32_t>(std::stoul(v)));
            tokens.push_back(ids);
        }
    }
    return tokens;
}

void load(std::ifstream& in, std::uint64_t offset, std::uint64_t bytes, std::uint8_t* device) {
    constexpr std::uint64_t chunk = 64ULL << 20U;
    std::uint8_t* host = nullptr;
    STRIX_CUDA_CHECK(cudaMallocHost(&host, chunk));
    in.seekg(static_cast<std::streamoff>(offset));
    for (std::uint64_t at = 0; at < bytes; at += chunk) {
        const auto n = std::min(chunk, bytes - at);
        in.read(reinterpret_cast<char*>(host), static_cast<std::streamsize>(n));
        STRIX_CUDA_CHECK(cudaMemcpy(device + at, host, n, cudaMemcpyHostToDevice));
    }
    STRIX_CUDA_CHECK(cudaFreeHost(host));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <experts.gguf> <routes.txt> [layer] [windows] [grouped rows,...]\n", argv[0]);
        return 2;
    }
    const std::uint32_t layer = argc > 3 ? static_cast<std::uint32_t>(std::atoi(argv[3])) : 20U;
    const std::uint32_t windows = argc > 4 ? static_cast<std::uint32_t>(std::atoi(argv[4])) : 40U;
    std::vector<std::uint32_t> grouped_rows{512, 2048, 8192};
    if (argc > 5) {
        grouped_rows.clear();
        std::istringstream is(argv[5]);
        std::string v;
        while (std::getline(is, v, ',')) grouped_rows.push_back(static_cast<std::uint32_t>(std::stoul(v)));
    }
    try {
        const auto device = dc::select_device(0);
        std::printf("device %s, %d SMs\n", device.name.c_str(), device.sms);
        const auto file = strix::gguf::read_file(argv[1]);
        const auto find = [&](const std::string& name) -> const strix::gguf::TensorInfo& {
            for (const auto& t : file.tensors)
                if (t.name == name) return t;
            throw std::runtime_error("missing tensor " + name);
        };
        const std::string p = "blk." + std::to_string(layer) + ".";
        const auto& tg = find(p + "ffn_gate_exps.weight");
        const auto& tu = find(p + "ffn_up_exps.weight");
        const auto& td = find(p + "ffn_down_exps.weight");
        dc::MoeShape shape;
        dc::DeviceBuffer<std::uint8_t> gate(tg.byte_size), up(tu.byte_size), down(td.byte_size);
        {
            std::ifstream in(argv[1], std::ios::binary);
            load(in, tg.absolute_offset, tg.byte_size, gate.get());
            load(in, tu.absolute_offset, tu.byte_size, up.get());
            load(in, td.absolute_offset, td.byte_size, down.get());
        }
        std::printf("layer %u experts resident: %.2f GB\n", layer, static_cast<double>(tg.byte_size + tu.byte_size + td.byte_size) / 1e9);
        const auto tokens = read_routes(argv[2], layer);
        if (tokens.empty()) throw std::runtime_error("no routes for the layer");

        const std::uint32_t cap = std::max<std::uint32_t>(8U, *std::max_element(grouped_rows.begin(), grouped_rows.end()));
        const std::uint32_t H = shape.hidden, K = shape.top_k;
        std::mt19937 rng(7);
        std::vector<std::int8_t> codes(std::size_t{cap} * H);
        std::uniform_int_distribution<int> c(-127, 127);
        for (auto& v : codes) v = static_cast<std::int8_t>(c(rng));
        std::vector<float> deltas(std::size_t{cap} * H / 32, 0.01F), w(std::size_t{cap} * K, 0.2F);
        dc::DeviceBuffer<std::int8_t> xc(codes.size());
        xc.upload(codes);
        dc::DeviceBuffer<float> xd(deltas.size()), rw(w.size()), shared(std::size_t{cap} * H), raw(std::size_t{cap} * K * H),
            y(std::size_t{cap} * H);
        xd.upload(deltas);
        rw.upload(w);
        shared.zero();
        dc::DeviceBuffer<std::uint32_t> ids(std::size_t{cap} * K);
        dc::DeviceBuffer<std::uint8_t> ws(dc::moe_layer_workspace_bytes(cap, shape));
        dc::DeviceBuffer<int4> flush((256ULL << 20U) / sizeof(int4));
        dc::DeviceBuffer<int> sink(1);
        STRIX_CUDA_CHECK(cudaMemset(flush.get(), 1, flush.size() * sizeof(int4)));
        dc::Stream stream;
        cudaEvent_t e0, e1, e2, e3;
        cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);

        auto layer_view = [&](std::uint32_t rows, std::uint32_t grouped_min) {
            dc::MoeLayerView v;
            v.shape = shape;
            v.weights = {gate.get(), up.get(), down.get()};
            v.rows = rows;
            v.x = {xc.get(), xd.get(), nullptr, rows, H, H};
            v.ids = ids.get();
            v.route_weights = rw.get();
            v.shared = shared.get();
            v.raw = raw.get();
            v.y = y.get();
            v.workspace = ws.get();
            v.capacity = cap;
            v.grouped_min_rows = grouped_min;
            return v;
        };
        auto set_ids = [&](std::uint32_t first, std::uint32_t rows) {
            std::vector<std::uint32_t> h;
            for (std::uint32_t r = 0; r < rows; ++r) {
                const auto& t = tokens[(first + r) % tokens.size()];
                h.insert(h.end(), t.begin(), t.begin() + K);
            }
            ids.upload(h);
            std::set<std::uint32_t> u(h.begin(), h.end());
            return static_cast<std::uint32_t>(u.size());
        };
        const double expert_bytes = static_cast<double>(shape.expert_bytes());
        const double gate_bytes = 2.0 * static_cast<double>(shape.gate_expert_bytes()), down_bytes = static_cast<double>(shape.down_expert_bytes());

        std::printf("\ndecode windows (cold L2, %u windows of consecutive recorded tokens; sums over windows)\n", windows);
        std::printf("R  union/win  layer us/win  GB/s   | gate/up us  GB/s  | down us  GB/s  | routes+combine us\n");
        for (const std::uint32_t R : {1U, 2U, 4U, 8U}) {
            double t_layer = 0, t_gu = 0, t_down = 0, unions = 0;
            for (std::uint32_t win = 0; win < windows + 1U; ++win) {
                const auto u = set_ids(win * R, R);
                const auto v = layer_view(R, 0);
                // layer: the whole driver
                flush_kernel<<<1024, 256, 0, stream.get()>>>(flush.get(), flush.size(), sink.get());
                cudaEventRecord(e0, stream.get());
                dc::enqueue_moe_layer(v, stream.get());
                cudaEventRecord(e1, stream.get());
                // the two expert launches alone (routes already built by the driver in the workspace)
                flush_kernel<<<1024, 256, 0, stream.get()>>>(flush.get(), flush.size(), sink.get());
                dc::RoutedGateUpView g;
                dc::RoutedDownView d;
                {
                    const auto parts = dc::moe_layer_workspace_views(ws.get(), cap, shape);
                    g.shape = d.shape = shape;
                    g.weights = d.weights = v.weights;
                    g.routes = d.routes = parts.routes;
                    g.max_union = d.max_union = std::min(R * K, shape.experts);
                    g.max_rows = d.max_rows = R;
                    g.x = v.x;
                    g.act = d.act = parts.act;
                    d.raw = raw.get();
                }
                cudaEventRecord(e2, stream.get());
                dc::enqueue_routed_gate_up(g, stream.get());
                cudaEventRecord(e3, stream.get());
                stream.sync();
                float ms_layer = 0, ms_gu = 0;
                cudaEventElapsedTime(&ms_layer, e0, e1);
                cudaEventElapsedTime(&ms_gu, e2, e3);
                flush_kernel<<<1024, 256, 0, stream.get()>>>(flush.get(), flush.size(), sink.get());
                cudaEventRecord(e2, stream.get());
                dc::enqueue_routed_down(d, stream.get());
                cudaEventRecord(e3, stream.get());
                stream.sync();
                float ms_down = 0;
                cudaEventElapsedTime(&ms_down, e2, e3);
                if (win == 0U) continue;  // warm-up
                t_layer += ms_layer; t_gu += ms_gu; t_down += ms_down; unions += u;
            }
            const double n = windows, us = 1e3 / n;
            std::printf("%u  %6.1f     %8.1f     %6.0f  | %8.1f  %6.0f | %7.1f  %5.0f | %6.1f\n", R, unions / n, t_layer * us,
                        unions * expert_bytes / (t_layer * 1e6), t_gu * us, unions * gate_bytes / (t_gu * 1e6), t_down * us,
                        unions * down_bytes / (t_down * 1e6), (t_layer - t_gu - t_down) * us);
        }

        std::printf("\nprefill (grouped entry points, rows of tokens cycled from the recorded routes)\n");
        std::printf("rows   union  layer ms   us/row   gate/up ms  down ms   TMAC/s  TOPS   | gemv layer ms  TOPS\n");
        std::printf("                                                         (gate/up+down; layer, 2 ops per MAC)\n");
        for (const auto rows : grouped_rows) {
            const auto u = set_ids(0, rows);
            const auto v = layer_view(rows, 1);
            dc::enqueue_moe_layer(v, stream.get());  // warm-up, also leaves the routes and activations
            stream.sync();
            const auto parts = dc::moe_layer_workspace_views(ws.get(), cap, shape);
            dc::RoutedGateUpView g;
            dc::RoutedDownView d;
            g.shape = d.shape = shape;
            g.weights = d.weights = v.weights;
            g.routes = d.routes = parts.routes;
            g.max_union = d.max_union = std::min(rows * K, shape.experts);
            g.x = v.x;
            g.act = d.act = parts.act;
            d.raw = raw.get();
            float best = 1e30F, best_gu = 1e30F, best_down = 1e30F;
            for (int rep = 0; rep < 3; ++rep) {
                cudaEventRecord(e0, stream.get());
                dc::enqueue_moe_layer(v, stream.get());
                cudaEventRecord(e1, stream.get());
                stream.sync();
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                best = std::min(best, ms);
                cudaEventRecord(e0, stream.get());
                dc::enqueue_routed_gate_up_grouped(g, rows, parts.grouped, stream.get());
                cudaEventRecord(e1, stream.get());
                dc::enqueue_routed_down_grouped(d, rows, parts.grouped, stream.get());
                cudaEventRecord(e2, stream.get());
                stream.sync();
                float a = 0, b = 0;
                cudaEventElapsedTime(&a, e0, e1);
                cudaEventElapsedTime(&b, e1, e2);
                best_gu = std::min(best_gu, a);
                best_down = std::min(best_down, b);
            }
            float gemv = 1e30F;  // the same layer on the decode GEMVs
            for (int rep = 0; rep < (rows > 4096U ? 1 : 2); ++rep) {
                cudaEventRecord(e0, stream.get());
                dc::enqueue_moe_layer(layer_view(rows, 0), stream.get());
                cudaEventRecord(e1, stream.get());
                stream.sync();
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                gemv = std::min(gemv, ms);
            }
            const double macs = static_cast<double>(rows) * K * 3.0 * shape.hidden * shape.inter;
            std::printf("%5u  %5u  %8.2f  %7.2f   %8.2f  %8.2f   %5.1f  %6.1f | %8.2f  %5.1f\n", rows, u, best, best * 1e3 / rows, best_gu,
                        best_down, macs / ((best_gu + best_down) * 1e-3) / 1e12, 2.0 * macs / (best * 1e-3) / 1e12, gemv,
                        2.0 * macs / (gemv * 1e-3) / 1e12);
        }
    } catch (const std::exception& e) {
        std::printf("FAIL %s\n", e.what());
        return 1;
    }
    return 0;
}
