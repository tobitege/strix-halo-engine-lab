// Host <-> device staging bandwidth for the state hand-off: pageable vs pinned, H2D and D2H, alone and while a
// compute kernel keeps the GPU busy on another stream (does the copy overlap compute, and what does it cost it).
// One source for both backends:
//   CUDA: nvcc -O2 -std=c++17 -o copy_probe copy_probe.cpp -x cu
//   HIP:  hipcc -O2 -std=c++17 --offload-arch=gfx1151 -o copy_probe copy_probe.cpp
// Run:   copy_probe [DEVICE] [SIZES_MIB...]   (default device 0, sizes 1 16 256)
// Prints one JSON line per (size, direction, host memory, with/without busy kernel): GB/s, and the busy kernel's
// slowdown when the copy runs beside it.
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define gpu(x) hip##x
#define GPU_H2D hipMemcpyHostToDevice
#define GPU_D2H hipMemcpyDeviceToHost
#define GPU_PINNED(p, n) hipHostMalloc(p, n, hipHostMallocDefault)
#define GPU_FREE_PINNED hipHostFree
#define BACKEND "hip"
typedef hipDeviceProp_t gpu_prop_t;
#else
#include <cuda_runtime.h>
#define gpu(x) cuda##x
#define GPU_H2D cudaMemcpyHostToDevice
#define GPU_D2H cudaMemcpyDeviceToHost
#define GPU_PINNED(p, n) cudaMallocHost(p, n)
#define GPU_FREE_PINNED cudaFreeHost
#define BACKEND "cuda"
typedef cudaDeviceProp gpu_prop_t;
#endif
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { auto e_ = (x); if (e_ != gpu(Success)) { std::fprintf(stderr, "%s failed: %s\n", #x, gpu(GetErrorString)(e_)); std::exit(1); } } while (0)

__global__ void busy(float* p, int iters) {
    float v = p[threadIdx.x + blockIdx.x * blockDim.x];
    for (int i = 0; i < iters; ++i) v = v * 0.999f + 0.001f;
    p[threadIdx.x + blockIdx.x * blockDim.x] = v;
}

using Clock = std::chrono::steady_clock;

static double now_s() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    const int dev = argc > 1 ? std::atoi(argv[1]) : 0;
    std::vector<size_t> sizes;
    for (int i = 2; i < argc; ++i) sizes.push_back(static_cast<size_t>(std::atoll(argv[i])) << 20);
    if (sizes.empty()) sizes = {size_t{1} << 20, size_t{16} << 20, size_t{256} << 20};
    CK(gpu(SetDevice)(dev));
    gpu_prop_t prop;
    CK(gpu(GetDeviceProperties)(&prop, dev));
    std::printf("{\"backend\":\"%s\",\"device\":%d,\"name\":\"%s\"}\n", BACKEND, dev, prop.name);
    gpu(Stream_t) cs, ks;
    CK(gpu(StreamCreateWithFlags)(&cs, gpu(StreamNonBlocking)));
    CK(gpu(StreamCreateWithFlags)(&ks, gpu(StreamNonBlocking)));
    const int blocks = prop.multiProcessorCount * 8, threads = 256;
    float* kbuf = nullptr;
    CK(gpu(Malloc)(&kbuf, sizeof(float) * blocks * threads));
    CK(gpu(Memset)(kbuf, 0, sizeof(float) * blocks * threads));
    // calibrate the busy kernel to ~200 ms
    int iters = 1 << 16;
    for (;;) {
        const double t0 = now_s();
        busy<<<blocks, threads, 0, ks>>>(kbuf, iters);
        CK(gpu(StreamSynchronize)(ks));
        const double dt = now_s() - t0;
        if (dt > 0.15 || iters > (1 << 28)) break;
        iters *= 2;
    }
    gpu(Event_t) k_start, k_end, c_start, c_end;
    CK(gpu(EventCreate)(&k_start));
    CK(gpu(EventCreate)(&k_end));
    CK(gpu(EventCreate)(&c_start));
    CK(gpu(EventCreate)(&c_end));
    auto elapsed = [&](gpu(Event_t) begin, gpu(Event_t) end) {
        float ms = 0;
        CK(gpu(EventElapsedTime)(&ms, begin, end));
        return static_cast<double>(ms) / 1000.0;
    };
    double kernel_alone = 0;
    {
        CK(gpu(EventRecord)(k_start, ks));
        busy<<<blocks, threads, 0, ks>>>(kbuf, iters);
        CK(gpu(EventRecord)(k_end, ks));
        CK(gpu(EventSynchronize)(k_end));
        kernel_alone = elapsed(k_start, k_end);
    }
    for (const size_t n : sizes) {
        void* d = nullptr;
        CK(gpu(Malloc)(&d, n));
        void* pinned = nullptr;
        CK(GPU_PINNED(&pinned, n));
        void* pageable = std::malloc(n);
        std::memset(pageable, 1, n);
        std::memset(pinned, 1, n);
        for (int dir = 0; dir < 2; ++dir)
            for (int pin = 0; pin < 2; ++pin)
                for (int with_kernel = 0; with_kernel < 2; ++with_kernel) {
                    void* h = pin ? pinned : pageable;
                    const auto kind = dir == 0 ? GPU_H2D : GPU_D2H;
                    auto copy = [&] {
                        if (dir == 0) CK(gpu(MemcpyAsync)(d, h, n, kind, cs));
                        else CK(gpu(MemcpyAsync)(h, d, n, kind, cs));
                    };
                    copy();  // warm
                    CK(gpu(StreamSynchronize)(cs));
                    const int reps = n >= (size_t{64} << 20) ? 4 : 32;
                    double k_dt = 0, overlap = 0;
                    const double t0 = now_s();
                    if (with_kernel) {
                        CK(gpu(EventRecord)(k_start, ks));
                        busy<<<blocks, threads, 0, ks>>>(kbuf, iters);
                        CK(gpu(EventRecord)(k_end, ks));
                    }
                    CK(gpu(EventRecord)(c_start, cs));
                    for (int r = 0; r < reps; ++r) copy();
                    CK(gpu(EventRecord)(c_end, cs));
                    CK(gpu(EventSynchronize)(c_end));
                    const double copy_wall = now_s() - t0;
                    const double c_dt = elapsed(c_start, c_end);
                    if (with_kernel) {
                        CK(gpu(EventSynchronize)(k_end));
                        k_dt = elapsed(k_start, k_end);
                        const double copy_start = elapsed(k_start, c_start);
                        overlap = std::max(0.0, std::min(k_dt, copy_start + c_dt) - std::max(0.0, copy_start));
                    }
                    std::printf("{\"mib\":%zu,\"dir\":\"%s\",\"host\":\"%s\",\"busy_kernel\":%d,\"GBps\":%.2f,\"kernel_alone_s\":%.4f,\"kernel_with_copy_s\":%.4f,"
                                "\"copy_device_s\":%.4f,\"copy_wall_s\":%.4f,\"kernel_copy_interval_overlap_s\":%.4f}\n",
                        n >> 20, dir == 0 ? "h2d" : "d2h", pin ? "pinned" : "pageable", with_kernel,
                        static_cast<double>(n) * reps / c_dt / 1e9, with_kernel ? kernel_alone : 0.0, k_dt, c_dt, copy_wall, overlap);
                    std::fflush(stdout);
                }
        std::free(pageable);
        CK(GPU_FREE_PINNED(pinned));
        CK(gpu(Free)(d));
    }
    CK(gpu(EventDestroy)(k_start));
    CK(gpu(EventDestroy)(k_end));
    CK(gpu(EventDestroy)(c_start));
    CK(gpu(EventDestroy)(c_end));
    CK(gpu(Free)(kbuf));
    CK(gpu(StreamDestroy)(cs));
    CK(gpu(StreamDestroy)(ks));
    return 0;
}
