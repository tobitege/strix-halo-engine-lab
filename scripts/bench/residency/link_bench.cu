// Card <-> host link costs for tiered expert residency (docs/engine/memory/expert-residency.md).
//   H2D / D2H pinned copy bandwidth at expert-sized and large transfers, pageable H2D,
//   zero-copy: a kernel reading host-mapped memory (what the card pays to read CPU expert outputs in place),
//   round trip: kernel publishes a flag in host-mapped memory -> host thread answers -> kernel sees the answer
//   (the per-layer exchange of the tiered route without any stream sync), vs event-sync + memcpy (the naive form).
// Build: nvcc -O3 -arch=sm_120 -o link_bench link_bench.cu      Run: link_bench [device=0] (<= 1 GiB of card memory)
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)
using Clock = std::chrono::steady_clock;

__global__ void read_kernel(const uint4* p, std::size_t n, unsigned long long* out) {
    unsigned long long acc = 0;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += std::size_t{gridDim.x} * blockDim.x) {
        const uint4 v = p[i];
        acc += v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345) *out = acc;
}

// iteration i: write ping = i, spin until pong == i (host answers), `iters` times; records nothing else.
__global__ void ping_kernel(volatile unsigned* ping, volatile unsigned* pong, unsigned iters) {
    if (threadIdx.x != 0) return;
    for (unsigned i = 1; i <= iters; ++i) {
        *ping = i;
        __threadfence_system();
        while (*pong != i) {}
    }
}

static double time_copy(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind, int reps, cudaStream_t s) {
    CK(cudaMemcpyAsync(dst, src, bytes, kind, s));
    CK(cudaStreamSynchronize(s));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a, s));
    for (int r = 0; r < reps; ++r) CK(cudaMemcpyAsync(dst, src, bytes, kind, s));
    CK(cudaEventRecord(b, s));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return ms / reps;
}

int main(int argc, char** argv) {
    const int dev = argc > 1 ? std::atoi(argv[1]) : 0;
    CK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, dev));
    std::printf("device %d %s, %d SMs\n", dev, prop.name, prop.multiProcessorCount);
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const std::size_t big = std::size_t{512} << 20;
    void *d = nullptr, *h = nullptr;
    CK(cudaMalloc(&d, big));
    CK(cudaHostAlloc(&h, big, cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(h, 1, big);
    std::vector<unsigned char> pageable(big, 1);
    for (const std::size_t bytes : {std::size_t{10} << 10, std::size_t{3145728}, std::size_t{4} * 3145728, std::size_t{64} << 20, big}) {
        const int reps = bytes < (std::size_t{1} << 20) ? 200 : bytes <= (std::size_t{16} << 20) ? 50 : 5;
        const double h2d = time_copy(d, h, bytes, cudaMemcpyHostToDevice, reps, s);
        const double d2h = time_copy(h, d, bytes, cudaMemcpyDeviceToHost, reps, s);
        const double pg = time_copy(d, pageable.data(), bytes, cudaMemcpyHostToDevice, reps, s);
        std::printf("copy %9zu B: H2D pinned %7.3f ms %6.2f GB/s | D2H pinned %7.3f ms %6.2f GB/s | H2D pageable %7.3f ms %6.2f GB/s\n",
                    bytes, h2d, bytes / h2d / 1e6, d2h, bytes / d2h / 1e6, pg, bytes / pg / 1e6);
    }
    // zero-copy read of host-mapped memory by a kernel
    void* hd = nullptr;
    CK(cudaHostGetDevicePointer(&hd, h, 0));
    unsigned long long* sink = nullptr;
    CK(cudaMalloc(&sink, 8));
    for (const std::size_t bytes : {std::size_t{40} << 10, std::size_t{4} << 20, std::size_t{256} << 20}) {
        for (const int blocks : {8, prop.multiProcessorCount * 4}) {
            read_kernel<<<blocks, 256, 0, s>>>(static_cast<const uint4*>(hd), bytes / 16, sink);
            CK(cudaStreamSynchronize(s));
            cudaEvent_t a, b;
            CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
            const int reps = 20;
            CK(cudaEventRecord(a, s));
            for (int r = 0; r < reps; ++r) read_kernel<<<blocks, 256, 0, s>>>(static_cast<const uint4*>(hd), bytes / 16, sink);
            CK(cudaEventRecord(b, s));
            CK(cudaEventSynchronize(b));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, a, b));
            std::printf("zero-copy kernel read %9zu B, %3d blocks: %8.4f ms %6.2f GB/s\n", bytes, blocks, ms / reps, bytes / (ms / reps) / 1e6);
            CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
        }
    }
    // flag round trip through host-mapped memory (no stream sync), host thread spinning
    volatile unsigned* flags = static_cast<volatile unsigned*>(h);
    flags[0] = flags[64] = 0;
    unsigned* dflags = static_cast<unsigned*>(hd);
    const unsigned iters = 2000;
    std::atomic<bool> go{true};
    std::thread responder([&] {
        unsigned seen = 0;
        while (go.load(std::memory_order_relaxed) && seen < iters) {
            const unsigned v = flags[0];
            if (v != seen) { seen = v; std::atomic_thread_fence(std::memory_order_seq_cst); flags[64] = v; }
        }
    });
    const auto t0 = Clock::now();
    ping_kernel<<<1, 32, 0, s>>>(dflags, dflags + 64, iters);
    CK(cudaStreamSynchronize(s));
    const double us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    go = false;
    responder.join();
    std::printf("mapped-flag round trip (card -> host thread -> card): %.2f us per exchange (%u exchanges)\n", us / iters, iters);
    // naive: kernel, event sync, D2H 3 KB, host work none, H2D 10 KB, next kernel
    const auto t1 = Clock::now();
    const int naive = 500;
    for (int i = 0; i < naive; ++i) {
        read_kernel<<<1, 32, 0, s>>>(static_cast<const uint4*>(d), 64, sink);
        CK(cudaMemcpyAsync(h, d, 3072, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpyAsync(d, h, 10240, cudaMemcpyHostToDevice, s));
    }
    CK(cudaStreamSynchronize(s));
    std::printf("naive exchange (launch + D2H 3 KB + stream sync + H2D 10 KB): %.2f us per exchange\n",
                std::chrono::duration<double, std::micro>(Clock::now() - t1).count() / naive);
    CK(cudaFree(sink)); CK(cudaFree(d)); CK(cudaFreeHost(h));
    return 0;
}
