// xslot_bench: cost of the TP2 exchange slot memory type for the MiniMax prefill post (and its producers).
//
// The prefill post (src/hip_minimax_moe.hip post_exchange_kernel, mode 2 = STRIX_MM_XB_POSTILP default) reads both
// partials in place from the exchange slots (src/minimax/pfx/pf_exchange.hip: TX and RX are hipHostMallocCoherent,
// i.e. GPU-uncached), adds them to the residual h, writes h and the normed y. This bench runs the same per-row work
// (one 256-thread block per row, float4 partial staging through LDS, the same expressions) with the partials in
//   coh = hipHostMalloc Mapped|Coherent (today), nc = hipHostMalloc Mapped|NonCoherent (GPU L2-cached),
//   dev = hipMalloc (device floor), unc = hipExtMallocWithFlags uncached device memory (the TX uncached option),
// times a producer-style write of one partial into each type, and checks that a kernel ordered behind a host flag
// (hipStreamWaitValue64, the exchange's own RX ordering) sees CPU-written data in every type after it has read
// (and possibly cached) the previous contents of the same slot.
//
// usage: xslot_bench [rows=4096] [reps=20] [check_iters=64]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr unsigned kHidden = 6144, kThreads = 256, kPer = kHidden / kThreads;
constexpr unsigned kPadEngine = 51200U - (2U * kHidden + kThreads) * 4U;  // total static LDS = 51,200 B

void ck(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}

// Pad: extra static LDS (bytes) to reproduce the engine kernel's footprint (post_exchange_kernel<2>: 51,200 B).
template <unsigned Pad>
__global__ __launch_bounds__(kThreads) void post_like(unsigned rows, float* hglob, const float* p0g, const float* p1g,
                                                      const float* norm_w, float* yglob) {
    __shared__ __attribute__((aligned(16))) float hs[kHidden];
    __shared__ __attribute__((aligned(16))) float ys[kHidden];
    __shared__ float part[kThreads];
    __shared__ float pad[Pad / 4U + 1U];
    const unsigned r = blockIdx.x, t = threadIdx.x;
    if constexpr (Pad != 0) reinterpret_cast<volatile float*>(pad)[(t * 61U) % (Pad / 4U)] = 0.0F;
    float hv[kPer], nw[kPer];
#pragma unroll
    for (unsigned q = 0; q < kPer; ++q) {
        hv[q] = hglob[static_cast<std::size_t>(r) * kHidden + t + q * kThreads];
        nw[q] = norm_w[t + q * kThreads];
    }
    float* h = hglob + static_cast<std::size_t>(r) * kHidden;
    const auto* p0v = reinterpret_cast<const float4*>(p0g + static_cast<std::size_t>(r) * kHidden);
    const auto* p1v = reinterpret_cast<const float4*>(p1g + static_cast<std::size_t>(r) * kHidden);
    for (unsigned k = t; k < kHidden / 4U; k += kThreads) {
        reinterpret_cast<float4*>(hs)[k] = p0v[k];
        reinterpret_cast<float4*>(ys)[k] = p1v[k];
    }
    __syncthreads();
    float acc = 0.0F;
#pragma unroll
    for (unsigned q = 0; q < kPer; ++q) {
        const unsigned j = t + q * kThreads;
        const float x = hv[q] + (hs[j] + ys[j]);
        h[j] = x;
        hs[j] = x;
        acc = __builtin_fmaf(x, x, acc);
    }
    part[t] = acc;
    __syncthreads();
    for (unsigned s = kThreads / 2; s > 0; s >>= 1) {
        if (t < s) part[t] += part[t + s];
        __syncthreads();
    }
    const float inv = 1.0F / __builtin_sqrtf(part[0] / static_cast<float>(kHidden) + 1e-6F);
    float* y = yglob + static_cast<std::size_t>(r) * kHidden;
#pragma unroll
    for (unsigned q = 0; q < kPer; ++q) {
        const unsigned j = t + q * kThreads;
        y[j] = hs[j] * inv * nw[q];
    }
}

// Producer epilogue stand-in: read one device row, write it (scaled) into the slot.
__global__ __launch_bounds__(kThreads) void write_like(unsigned rows, const float* src, float* dst) {
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * kHidden;
    const auto* s = reinterpret_cast<const float4*>(src + base);
    auto* d = reinterpret_cast<float4*>(dst + base);
    for (unsigned k = threadIdx.x; k < kHidden / 4U; k += kThreads) {
        float4 v = s[k];
        v.x *= 0.5F; v.y *= 0.5F; v.z *= 0.5F; v.w *= 0.5F;
        d[k] = v;
    }
}

// Every element must equal `expect`; mismatches counted.
__global__ __launch_bounds__(kThreads) void check_kernel(std::size_t n, const float* p, float expect, unsigned* bad) {
    unsigned local = 0;
    for (std::size_t i = blockIdx.x * std::size_t{kThreads} + threadIdx.x; i < n; i += std::size_t{gridDim.x} * kThreads)
        local += p[i] != expect;
    if (local) atomicAdd(bad, local);
}

struct Buf {
    std::string kind;
    float* dev = nullptr;   // device pointer
    float* host = nullptr;  // host pointer (nullptr for device-only kinds)
};

Buf alloc(const std::string& kind, std::size_t bytes) {
    Buf b{kind};
    if (kind == "coh" || kind == "nc") {
        const unsigned f = hipHostMallocMapped | (kind == "coh" ? hipHostMallocCoherent : hipHostMallocNonCoherent);
        ck(hipHostMalloc(reinterpret_cast<void**>(&b.host), bytes, f), "hipHostMalloc");
        ck(hipHostGetDevicePointer(reinterpret_cast<void**>(&b.dev), b.host, 0), "hipHostGetDevicePointer");
        std::memset(b.host, 0, bytes);
    } else if (kind == "dev") {
        ck(hipMalloc(reinterpret_cast<void**>(&b.dev), bytes), "hipMalloc");
        ck(hipMemset(b.dev, 0, bytes), "memset");
    } else {
        ck(hipExtMallocWithFlags(reinterpret_cast<void**>(&b.dev), bytes, hipDeviceMallocUncached), "uncached");
        ck(hipMemset(b.dev, 0, bytes), "memset");
    }
    return b;
}

void release(Buf& b) {
    if (b.host) (void)hipHostFree(b.host);
    else (void)hipFree(b.dev);
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
    const auto positive = [](const char* text, int& value) {
        const char* end = text + std::strlen(text);
        const auto parsed = std::from_chars(text, end, value);
        return parsed.ec == std::errc{} && parsed.ptr == end && value > 0;
    };
    int row_count = 4096, reps = 20, iters = 64;
    if (argc > 4 || (argc > 1 && !positive(argv[1], row_count)) || (argc > 2 && !positive(argv[2], reps)) ||
        (argc > 3 && !positive(argv[3], iters)) || reps > std::numeric_limits<int>::max() - 2 ||
        iters == std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "usage: xslot_bench [rows=4096] [reps=20] [check_iters=64] (positive integers within loop limits)\n");
        return 2;
    }
    const unsigned rows = static_cast<unsigned>(row_count);
    const std::size_t n = std::size_t{rows} * kHidden, bytes = n * sizeof(float);
    hipStream_t st;
    ck(hipStreamCreateWithFlags(&st, hipStreamNonBlocking), "stream");
    hipEvent_t e0, e1;
    ck(hipEventCreate(&e0), "ev");
    ck(hipEventCreate(&e1), "ev");
    float *h, *y, *w, *src;
    ck(hipMalloc(reinterpret_cast<void**>(&h), bytes), "h");
    ck(hipMalloc(reinterpret_cast<void**>(&y), bytes), "y");
    ck(hipMalloc(reinterpret_cast<void**>(&src), bytes), "src");
    ck(hipMalloc(reinterpret_cast<void**>(&w), kHidden * sizeof(float)), "w");
    ck(hipMemset(h, 0, bytes), "m");
    ck(hipMemset(src, 0, bytes), "m");
    ck(hipMemset(w, 0, kHidden * sizeof(float)), "m");
    unsigned* bad;
    ck(hipMalloc(reinterpret_cast<void**>(&bad), sizeof(unsigned)), "bad");
    std::uint64_t* flag_h;
    std::uint64_t* flag_d;
    ck(hipHostMalloc(reinterpret_cast<void**>(&flag_h), 4096, hipHostMallocMapped | hipHostMallocCoherent), "flag");
    __atomic_store_n(flag_h, 0, __ATOMIC_RELEASE);
    ck(hipHostGetDevicePointer(reinterpret_cast<void**>(&flag_d), flag_h, 0), "flag");
    // Floor: the post moves 6 rows of kHidden fp32 per row (h, p0, p1 read; h, y write; norm_w cached).
    const double post_mb = 5.0 * static_cast<double>(bytes) / 1e6;
    std::printf("xslot rows %u reps %d: post bytes %.1f MB (p0+p1 %.1f MB), producer write %.1f MB\n", rows, reps, post_mb,
                2.0 * bytes / 1e6, bytes / 1e6);
    bool stale = false;
    for (const char* kind : {"coh", "nc", "dev", "unc"}) {
        __atomic_store_n(flag_h, 0, __ATOMIC_RELEASE);  // previous trial synchronized every consumer
        Buf p0 = alloc(kind, bytes), p1 = alloc(kind, bytes);
        std::vector<double> tp, tq, tw;
        for (int i = 0; i < reps + 2; ++i) {
            float ms = 0.0F;
            ck(hipEventRecord(e0, st), "rec");
            hipLaunchKernelGGL(post_like<0>, dim3(rows), dim3(kThreads), 0, st, rows, h, p0.dev, p1.dev, w, y);
            ck(hipEventRecord(e1, st), "rec");
            ck(hipEventSynchronize(e1), "sync");
            ck(hipEventElapsedTime(&ms, e0, e1), "el");
            if (i >= 2) tp.push_back(ms);
            ck(hipEventRecord(e0, st), "rec");
            hipLaunchKernelGGL(post_like<kPadEngine>, dim3(rows), dim3(kThreads), 0, st, rows, h, p0.dev, p1.dev, w, y);
            ck(hipEventRecord(e1, st), "rec");
            ck(hipEventSynchronize(e1), "sync");
            ck(hipEventElapsedTime(&ms, e0, e1), "el");
            if (i >= 2) tq.push_back(ms);
            ck(hipEventRecord(e0, st), "rec");
            hipLaunchKernelGGL(write_like, dim3(rows), dim3(kThreads), 0, st, rows, src, p0.dev);
            ck(hipEventRecord(e1, st), "rec");
            ck(hipEventSynchronize(e1), "sync");
            ck(hipEventElapsedTime(&ms, e0, e1), "el");
            if (i >= 2) tw.push_back(ms);
        }
        const double mp = median(tp), mw = median(tw);
        std::printf("kind %-3s post %.3f ms (%.0f GB/s, min %.3f max %.3f)  post+pad(engine LDS 51,200 B) %.3f ms  write %.3f ms (%.0f GB/s)\n",
                    kind, mp, post_mb / mp, *std::min_element(tp.begin(), tp.end()), *std::max_element(tp.begin(), tp.end()),
                    median(tq), mw, bytes / 1e6 / mw);
        // Coherence (host kinds): kernel ordered behind the host flag reads data the CPU wrote after the GPU had read
        // the previous contents of the slot (stale-line hazard of a cached type).
        if (p1.host != nullptr) {
            unsigned total_bad = 0, bad_iters = 0;
            for (int it = 1; it <= iters; ++it) {
                ck(hipMemsetAsync(bad, 0, sizeof(unsigned), st), "mz");
                // touch the previous contents first (as the previous message's post did)
                hipLaunchKernelGGL(check_kernel, dim3(256), dim3(kThreads), 0, st, n, p1.dev, static_cast<float>(it - 1), bad);
                ck(hipStreamSynchronize(st), "s");
                ck(hipMemsetAsync(bad, 0, sizeof(unsigned), st), "mz");
                ck(hipStreamWaitValue64(st, flag_d, static_cast<std::uint64_t>(it), hipStreamWaitValueGte, ~std::uint64_t{0}), "wv");
                hipLaunchKernelGGL(check_kernel, dim3(256), dim3(kThreads), 0, st, n, p1.dev, static_cast<float>(it), bad);
                std::this_thread::sleep_for(std::chrono::milliseconds(2));  // kernel queued behind the flag
                std::fill(p1.host, p1.host + n, static_cast<float>(it));       // the progress thread's recv into RX
                std::atomic_thread_fence(std::memory_order_release);
                __atomic_store_n(flag_h, static_cast<std::uint64_t>(it), __ATOMIC_RELEASE);
                ck(hipStreamSynchronize(st), "s");
                unsigned b = 0;
                ck(hipMemcpy(&b, bad, sizeof(unsigned), hipMemcpyDeviceToHost), "cp");
                total_bad += b;
                bad_iters += b != 0;
            }
            std::printf("kind %-3s coherence: %d iters, %u bad iters, %u stale elements -> %s\n", kind, iters, bad_iters,
                        total_bad, bad_iters == 0 ? "OK" : "STALE");
            stale |= bad_iters != 0;
        }
        release(p0);
        release(p1);
    }
    std::printf("done\n");
    return stale ? 1 : 0;
}
