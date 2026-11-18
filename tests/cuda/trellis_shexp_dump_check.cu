// Kernel-level identity check of the decode step's trellis shared expert: the CUDA trellis GEMV with the pair library's
// split policy against the pair's own per-row outputs (a dump written on the AMD node by the engine's trellis adapter:
// "TRLDUMP1", per record layer / which (0 gate, 1 up, 2 down) / K / I / N, x[8][I], then y[R][N] for R = 1..8).
// Usage: strix-cuda-trellis-shexp-dump-check PACK_DIR DUMP.bin   (exit 0 when every value of every R is bitwise equal)
#include "strix/cuda/trellis.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tr = strix::ds41::cuda::trellis;

namespace {
struct Blob {
    std::vector<std::uint8_t> bytes;
    std::vector<std::uint64_t> shape;
};
Blob read_tensor(const std::string& path, const std::string& name) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 8);
    std::string h(static_cast<std::size_t>(n), '\0');
    f.read(h.data(), static_cast<std::streamsize>(n));
    const auto at = h.find("\"" + name + "\"");
    if (at == std::string::npos) throw std::runtime_error("no tensor " + name);
    const std::string d = h.substr(at, h.find('}', at) - at);
    const auto nums = [&](const char* k) {
        std::vector<std::uint64_t> out;
        const auto p = d.find(std::string("\"") + k + "\"");
        const auto open = d.find('[', p), close = d.find(']', p);
        std::string body = d.substr(open + 1, close - open - 1);
        std::size_t i = 0;
        while (i < body.size()) {
            const auto j = body.find(',', i);
            const auto t = body.substr(i, j == std::string::npos ? std::string::npos : j - i);
            if (t.find_first_of("0123456789") != std::string::npos) out.push_back(std::stoull(t));
            if (j == std::string::npos) break;
            i = j + 1;
        }
        return out;
    };
    Blob b;
    b.shape = nums("shape");
    const auto off = nums("data_offsets");
    b.bytes.resize(static_cast<std::size_t>(off.at(1) - off.at(0)));
    f.seekg(static_cast<std::streamoff>(8 + n + off[0]));
    f.read(reinterpret_cast<char*>(b.bytes.data()), static_cast<std::streamsize>(b.bytes.size()));
    return b;
}
template <class T>
T* dev(const void* host, std::size_t bytes) {
    void* p = nullptr;
    STRIX_CUDA_CHECK(cudaMalloc(&p, bytes));
    if (host) STRIX_CUDA_CHECK(cudaMemcpy(p, host, bytes, cudaMemcpyHostToDevice));
    return static_cast<T*>(p);
}
std::uint32_t vector_splits(std::uint32_t input, std::uint32_t output) {
    const std::uint32_t s = output <= 1024U ? 32U : (output <= 4096U && input >= 4096U ? 16U : 4U);
    return s < input / 16U ? s : input / 16U;
}
}  // namespace

int main(int argc, char** argv) try {
    if (argc < 3) {
        std::fprintf(stderr, "usage: strix-cuda-trellis-shexp-dump-check PACK_DIR DUMP.bin\n");
        return 2;
    }
    const std::string pack = argv[1];
    std::ifstream f(argv[2], std::ios::binary);
    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "TRLDUMP1", 8) != 0) throw std::runtime_error("not a TRLDUMP1 file");
    std::uint32_t count = 0;
    f.read(reinterpret_cast<char*>(&count), 4);
    const char* parts[3] = {"w1", "w3", "w2"};
    const char* names[3] = {"gate", "up", "down"};
    std::uint64_t total = 0, differ = 0;
    cudaStream_t st;
    STRIX_CUDA_CHECK(cudaStreamCreate(&st));
    for (std::uint32_t rec = 0; rec < count; ++rec) {
        std::uint32_t h[5];
        f.read(reinterpret_cast<char*>(h), 20);
        const std::uint32_t layer = h[0], which = h[1], bits = h[2], I = h[3], N = h[4];
        std::vector<float> x(std::size_t{8} * I);
        f.read(reinterpret_cast<char*>(x.data()), static_cast<std::streamsize>(x.size() * 4));
        std::vector<std::vector<float>> ys(9);
        for (std::uint32_t R = 1; R <= 8; ++R) {
            ys[R].resize(std::size_t{R} * N);
            f.read(reinterpret_cast<char*>(ys[R].data()), static_cast<std::streamsize>(ys[R].size() * 4));
        }
        const std::string path = pack + "/dense.L" + std::to_string(layer) + ".safetensors";
        const std::string p = "layers." + std::to_string(layer) + ".ffn.shared_experts." + parts[which] + ".";
        const auto packed = read_tensor(path, p + "trellis"), suh = read_tensor(path, p + "suh"), svh = read_tensor(path, p + "svh");
        tr::Linear v;
        v.weights.packed = dev<std::uint16_t>(packed.bytes.data(), packed.bytes.size());
        v.weights.suh = dev<std::uint16_t>(suh.bytes.data(), suh.bytes.size());
        v.weights.svh = dev<std::uint16_t>(svh.bytes.data(), svh.bytes.size());
        v.weights.packed_bytes = packed.bytes.size();
        v.weights.suh_bytes = suh.bytes.size();
        v.weights.svh_bytes = svh.bytes.size();
        v.weights.input = I;
        v.weights.output = N;
        v.weights.bits = bits;
        v.splits = vector_splits(I, N);
        v.mode = tr::Mode::gemv_quality;
        v.x_stride = I;
        v.y_stride = N;
        v.rows = 8;
        const auto wsb = tr::workspace_bytes(v);
        void* ws = dev<std::uint8_t>(nullptr, wsb);
        float* dx = dev<float>(x.data(), x.size() * 4);
        float* dy = dev<float>(nullptr, std::size_t{8} * N * 4);
        std::uint64_t rec_differ = 0;
        for (std::uint32_t R = 1; R <= 8; ++R) {
            v.rows = R;
            v.x = dx;
            v.y = dy;
            v.x_bytes = std::size_t{R} * I * 4;
            v.y_bytes = std::size_t{R} * N * 4;
            v.workspace = ws;
            v.workspace_bytes = wsb;
            tr::enqueue_linear(v, st);
            STRIX_CUDA_CHECK(cudaStreamSynchronize(st));
            std::vector<float> y(std::size_t{R} * N);
            STRIX_CUDA_CHECK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
            for (std::size_t i = 0; i < y.size(); ++i) {
                std::uint32_t a, b;
                std::memcpy(&a, &y[i], 4);
                std::memcpy(&b, &ys[R][i], 4);
                rec_differ += a != b;
            }
            total += y.size();
        }
        differ += rec_differ;
        std::printf("L%u %s K%u %ux%u S%u: %llu values differ over R=1..8\n", layer, names[which], bits, I, N, v.splits,
                    static_cast<unsigned long long>(rec_differ));
        cudaFree(const_cast<std::uint16_t*>(v.weights.packed));
        cudaFree(const_cast<std::uint16_t*>(v.weights.suh));
        cudaFree(const_cast<std::uint16_t*>(v.weights.svh));
        cudaFree(ws);
        cudaFree(dx);
        cudaFree(dy);
    }
    std::printf("dump check: %u records, %llu values, %llu differ -> %s\n", count, static_cast<unsigned long long>(total),
                static_cast<unsigned long long>(differ), differ == 0 ? "PASS" : "FAIL");
    return differ == 0 ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
