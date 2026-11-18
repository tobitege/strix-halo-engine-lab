// kvmicro: does memory pressure on the node evict this process's GPU queues through its userptr host memory,
// and does a mitigation stop it? Holds GTT_GB of device memory (like a loaded model), HOST_MB of hipHostMalloc
// buffers (like engine staging), runs a ~1 ms GPU kernel that reads host memory in a loop for SECS seconds and
// reports iteration-time stalls + own evicted_ms. Mitigation from argv[4]: none | mlock | pin (mlock + io_uring
// long-term pin of every ROCr/locked host range found in anonymous mappings) ; env HSA_USERPTR_FOR_PAGED_MEM=0 = GTT host.
// Usage: kvmicro SECS GTT_GB HOST_MB MODE [ODIRECT_FILE]
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

__global__ void spin(const float* __restrict__ host, float* out, int n, int iters) {
    float acc = 0.f; const int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = 0; k < iters; ++k) acc = acc * 0.999f + host[(i + k * 4099) % n];
    out[i] = acc;
}

static long evicted_ms() {
    std::string p = "/sys/class/kfd/kfd/proc/" + std::to_string(getpid());
    long t = 0; char cmd[512]; std::snprintf(cmd, sizeof cmd, "cat %s/stats_*/evicted_ms 2>/dev/null", p.c_str());
    FILE* f = popen(cmd, "r"); if (!f) return -1; long v; while (std::fscanf(f, "%ld", &v) == 1) t += v; pclose(f); return t;
}

struct Range { char* base; size_t bytes; };
// ROCr-owned (HSA) or hipHostRegister'd (LOCKED) host ranges inside anonymous private mappings.
static std::vector<Range> rocr_host_ranges() {
    std::vector<Range> out; std::ifstream m("/proc/self/maps"); std::string line;
    while (std::getline(m, line)) {
        unsigned long lo, hi; char perm[8] = {}; unsigned long off; char dev[16] = {}; unsigned long ino; char path[256] = {};
        if (std::sscanf(line.c_str(), "%lx-%lx %7s %lx %15s %lu %255s", &lo, &hi, perm, &off, dev, &ino, path) < 6) continue;
        if (perm[0] != 'r' || perm[1] != 'w') continue;
        if (perm[3] == 's' && line.find("/dev/zero") != std::string::npos) { out.push_back({reinterpret_cast<char*>(lo), hi - lo}); continue; }
        if (perm[3] != 'p' || ino != 0 || path[0] == '[') continue;
        for (unsigned long a = lo; a < hi;) {
            hsa_amd_pointer_info_t info{}; info.size = sizeof(info);
            if (hsa_amd_pointer_info(reinterpret_cast<void*>(a), &info, nullptr, nullptr, nullptr) == HSA_STATUS_SUCCESS &&
                (info.type == HSA_EXT_POINTER_TYPE_HSA || info.type == HSA_EXT_POINTER_TYPE_LOCKED) && info.hostBaseAddress && info.sizeInBytes) {
                auto b = reinterpret_cast<unsigned long>(info.hostBaseAddress); auto e = b + info.sizeInBytes;
                b = std::max(b, lo); e = std::min(e, hi);
                if (e > b) out.push_back({reinterpret_cast<char*>(b), e - b});
                a = std::max(e, a + 4096);
            } else a += 4096;
        }
    }
    return out;
}

static int pin_ranges(const std::vector<Range>& rs, size_t& pinned) {
    io_uring_params p{}; int fd = static_cast<int>(syscall(__NR_io_uring_setup, 1, &p));
    if (fd < 0) { std::perror("io_uring_setup"); return -1; }
    std::vector<iovec> iov;
    for (auto r : rs) for (size_t o = 0; o < r.bytes; o += (1UL << 30)) iov.push_back({r.base + o, std::min(r.bytes - o, 1UL << 30)});
    if (syscall(__NR_io_uring_register, fd, IORING_REGISTER_BUFFERS, iov.data(), static_cast<unsigned>(iov.size())) != 0) { std::perror("IORING_REGISTER_BUFFERS"); return -1; }
    pinned = 0; for (auto& v : iov) pinned += v.iov_len; return fd;
}

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage\n"); return 64; }
    const double secs = std::atof(argv[1]); const size_t gtt_gb = std::strtoul(argv[2], nullptr, 10);
    const size_t host_mb = std::strtoul(argv[3], nullptr, 10); const std::string mode = argv[4];
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<void*> dev; for (size_t g = 0; g < gtt_gb; ++g) { void* p; CK(hipMalloc(&p, 1UL << 30)); CK(hipMemset(p, 0, 1UL << 30)); dev.push_back(p); }
    // host buffers: one big staging + many 4-40 MB ones (engine-like shape)
    std::vector<float*> host; size_t left = host_mb << 20; std::vector<Range> own;
    auto halloc = [&](size_t sz) {
        float* p = nullptr;
        if (mode == "anonpin") { // engine-owned private anonymous pages, locked + long-term pinned, then registered
            void* m = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); if (m == MAP_FAILED) std::exit(3);
            madvise(m, sz, MADV_DONTFORK); std::memset(m, 1, sz); if (mlock(m, sz) != 0) std::perror("mlock");
            own.push_back({static_cast<char*>(m), sz}); CK(hipHostRegister(m, sz, hipHostRegisterMapped)); void* d = nullptr; CK(hipHostGetDevicePointer(&d, m, 0)); p = static_cast<float*>(d);
        } else { CK(hipHostMalloc(reinterpret_cast<void**>(&p), sz, hipHostMallocDefault)); std::memset(p, 1, sz); }
        host.push_back(p); left -= sz; };
    for (size_t sz : {512UL << 20, 40UL << 20, 40UL << 20, 20UL << 20}) { if (sz > left) break; halloc(sz); }
    while (left >= (4UL << 20)) halloc(4UL << 20);
    CK(hipDeviceSynchronize());
    if (argc > 5) { // O_DIRECT read into pinned host memory (the weight-load pattern)
        int fd = open(argv[5], O_RDONLY | O_DIRECT); ssize_t r = fd >= 0 ? pread(fd, own.empty() ? static_cast<void*>(host[0]) : static_cast<void*>(own[0].base), 1 << 20, 0) : -1;
        std::printf("odirect read rc=%zd errno=%d\n", r, r < 0 ? errno : 0); if (fd >= 0) close(fd);
    }
    size_t pinned = 0; int ring = -1; size_t nr = 0;
    const auto tp0 = std::chrono::steady_clock::now();
    if (mode == "mlock" || mode == "shmpin") { auto rs = rocr_host_ranges(); nr = rs.size(); for (auto r : rs) if (mlock(r.base, r.bytes) != 0) std::perror("mlock");
        if (mode == "shmpin") ring = pin_ranges(rs, pinned); }
    if (mode == "anonpin") { nr = own.size(); ring = pin_ranges(own, pinned); }
    const double pin_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0).count();
    const double setup = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    { std::ifstream mi("/proc/meminfo"); std::string l; while (std::getline(mi, l)) if (!l.rfind("MemAvailable", 0) || !l.rfind("GPUActive", 0) || !l.rfind("Mlocked", 0)) std::printf("  %s\n", l.c_str()); }
    float* out; CK(hipMalloc(&out, 256 * 256 * sizeof(float)));
    const int n = static_cast<int>((4UL << 20) / sizeof(float));
    std::vector<double> it; const long ev0 = evicted_ms(); hipStream_t s; CK(hipStreamCreate(&s));
    const auto tl = std::chrono::steady_clock::now();
    for (size_t k = 0; std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count() < secs; ++k) {
        const auto a = std::chrono::steady_clock::now();
        spin<<<256, 256, 0, s>>>(host[1 + k % (host.size() - 1)], out, n, 2000); CK(hipStreamSynchronize(s));
        it.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
    }
    const long ev1 = evicted_ms(); auto v = it; std::sort(v.begin(), v.end());
    const double p50 = v[v.size() / 2]; double stall = 0; int nst = 0;
    for (double x : it) if (x > 5 * p50 && x > 3.0) { stall += x - p50; ++nst; }
    std::printf("mode=%s userptr_env=%s setup=%.1fs pin_ranges=%zu pinned=%zuMB pin_ms=%.1f iters=%zu p50=%.3fms p99=%.3fms max=%.1fms stalls=%d stall_ms=%.0f evicted_ms=%ld\n",
        mode.c_str(), std::getenv("HSA_USERPTR_FOR_PAGED_MEM") ? std::getenv("HSA_USERPTR_FOR_PAGED_MEM") : "-", setup, nr, pinned >> 20, pin_ms,
        it.size(), p50, v[v.size() * 99 / 100], v.back(), nst, stall, ev1 - ev0);
    if (ring >= 0) close(ring);
    for (auto r : own) { (void)hipHostUnregister(r.base); munmap(r.base, r.bytes); }
    return 0;
}
