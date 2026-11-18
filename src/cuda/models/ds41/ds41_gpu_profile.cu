// deepseek41 CUDA prefill: device profile (cuda_ds41_gpu_profile.hpp).
#include "strix/cuda/ds41_gpu_profile.hpp"

#include "strix/cuda/ds41_common.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace strix::ds41::cuda {
namespace {

// The PCIe link from the NVIDIA management library, loaded at run time (absent: unknown).
void query_pcie(GpuProfile& p) {
    using Init = int (*)();
    using ByBus = int (*)(const char*, void**);
    using Get = int (*)(void*, unsigned*);
#ifdef _WIN32
    HMODULE lib = LoadLibraryA("nvml.dll");
    if (!lib) lib = LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
    if (!lib) return;
    auto sym = [&](const char* n) { return reinterpret_cast<void*>(GetProcAddress(lib, n)); };
#else
    void* lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!lib) return;
    auto sym = [&](const char* n) { return dlsym(lib, n); };
#endif
    const auto init = reinterpret_cast<Init>(sym("nvmlInit_v2"));
    const auto shutdown = reinterpret_cast<Init>(sym("nvmlShutdown"));
    const auto by_bus = reinterpret_cast<ByBus>(sym("nvmlDeviceGetHandleByPciBusId_v2"));
    const auto cur_gen = reinterpret_cast<Get>(sym("nvmlDeviceGetCurrPcieLinkGeneration"));
    const auto cur_width = reinterpret_cast<Get>(sym("nvmlDeviceGetCurrPcieLinkWidth"));
    const auto max_gen = reinterpret_cast<Get>(sym("nvmlDeviceGetMaxPcieLinkGeneration"));
    const auto max_width = reinterpret_cast<Get>(sym("nvmlDeviceGetMaxPcieLinkWidth"));
    if (init && by_bus && init() == 0) {
        void* dev = nullptr;
        if (by_bus(p.pci_bus_id.c_str(), &dev) == 0) {
            unsigned v = 0;
            if (cur_gen && cur_gen(dev, &v) == 0) p.pcie_gen = static_cast<int>(v);
            if (cur_width && cur_width(dev, &v) == 0) p.pcie_width = static_cast<int>(v);
            if (max_gen && max_gen(dev, &v) == 0) p.pcie_max_gen = static_cast<int>(v);
            if (max_width && max_width(dev, &v) == 0) p.pcie_max_width = static_cast<int>(v);
        }
        if (shutdown) shutdown();
    }
#ifdef _WIN32
    FreeLibrary(lib);
#else
    dlclose(lib);
#endif
}

std::filesystem::path default_cache_dir() {
    if (const char* v = std::getenv("STRIX_GPU_PROFILE_DIR")) return v;
#ifdef _WIN32
    if (const char* v = std::getenv("LOCALAPPDATA")) return std::filesystem::path(v) / "strix";
#endif
    if (const char* v = std::getenv("XDG_CACHE_HOME")) return std::filesystem::path(v) / "strix";
    if (const char* v = std::getenv("HOME")) return std::filesystem::path(v) / ".cache" / "strix";
    return std::filesystem::current_path();
}

double read_cached(const std::filesystem::path& f) {
    std::ifstream in(f);
    if (!in) return 0.0;
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();
    const auto k = s.find("\"h2d_gbps\":");
    if (k == std::string::npos) return 0.0;
    return std::atof(s.c_str() + k + 11);
}

// Pinned 256 MiB source, 64 MiB pieces on one stream, after one warm round: the best of four windows of seconds / 4
// (the host's other load drifts; the best window is what the link and memory can sustain).
double probe_h2d(double seconds) {
    constexpr std::size_t kBytes = std::size_t{256} << 20, kPiece = std::size_t{64} << 20;
    void* host = nullptr;
    void* dev = nullptr;
    cudaStream_t st = nullptr;
    STRIX_CUDA_CHECK(cudaHostAlloc(&host, kBytes, cudaHostAllocDefault));
    STRIX_CUDA_CHECK(cudaMalloc(&dev, kBytes));
    STRIX_CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    auto round = [&] {
        for (std::size_t at = 0; at < kBytes; at += kPiece)
            STRIX_CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(dev) + at, static_cast<char*>(host) + at, kPiece, cudaMemcpyHostToDevice, st));
    };
    round();
    STRIX_CUDA_CHECK(cudaStreamSynchronize(st));
    double best = 0.0;
    for (int w = 0; w < 4; ++w) {
        const auto t0 = std::chrono::steady_clock::now();
        std::size_t bytes = 0;
        double el = 0.0;
        while (el < seconds / 4.0) {
            round();
            STRIX_CUDA_CHECK(cudaStreamSynchronize(st));
            bytes += kBytes;
            el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        best = std::max(best, double(bytes) / el / 1e9);
    }
    cudaStreamDestroy(st);
    cudaFree(dev);
    cudaFreeHost(host);
    return best;
}

std::string escape(const std::string& s) {
    std::string o;
    for (const char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o;
}

}  // namespace

GpuProfile query_gpu_profile(int device, bool probe, bool force_probe, double seconds, const std::string& cache_dir) {
    GpuProfile p;
    p.device = device;
    STRIX_CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    STRIX_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    p.name = prop.name;
    p.cc_major = prop.major;
    p.cc_minor = prop.minor;
    p.sms = prop.multiProcessorCount;
    char hex[40] = {0};
    for (int i = 0; i < 16; ++i) std::snprintf(hex + 2 * i, 3, "%02x", static_cast<unsigned>(static_cast<unsigned char>(prop.uuid.bytes[i])));
    p.uuid = hex;
    char bus[32] = {0};
    if (cudaDeviceGetPCIBusId(bus, sizeof bus, device) == cudaSuccess) p.pci_bus_id = bus;
    STRIX_CUDA_CHECK(cudaMemGetInfo(&p.free_bytes, &p.total_bytes));
    const std::filesystem::path dir = cache_dir.empty() ? default_cache_dir() : std::filesystem::path(cache_dir);
    const std::filesystem::path file = dir / ("strix-gpu-" + p.uuid + ".json");
    p.cache_file = file.string();
    if (const char* v = std::getenv("STRIX_GPU_PROBE"); v && std::string(v) == "force") force_probe = true;
    {
        // a value older than a day is measured again (drivers, BIOS and slots change)
        std::error_code ec;
        const auto age = std::filesystem::file_time_type::clock::now() - std::filesystem::last_write_time(file, ec);
        if (!ec && age > std::chrono::hours(24)) force_probe = true;
    }
    if (!force_probe) {
        p.h2d_gbps = read_cached(file);
        p.h2d_cached = p.h2d_gbps > 0.0;
    }
    if (probe && !p.h2d_cached) {
        p.h2d_gbps = probe_h2d(seconds);
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        std::ofstream out(file);
        if (out) {
            const std::time_t now = std::time(nullptr);
            char when[32] = {0};
            std::strftime(when, sizeof when, "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
            out << "{\"name\":\"" << escape(p.name) << "\",\"uuid\":\"" << p.uuid << "\",\"h2d_gbps\":" << p.h2d_gbps << ",\"measured\":\"" << when
                << "\"}\n";
        }
    }
    query_pcie(p);  // after the probe: an idle link may have trained down
    return p;
}

std::string to_json(const GpuProfile& p) {
    std::ostringstream s;
    s << "{\"device\":" << p.device << ",\"name\":\"" << escape(p.name) << "\",\"uuid\":\"" << p.uuid << "\",\"pci\":\"" << p.pci_bus_id
      << "\",\"cc\":\"" << p.cc_major << "." << p.cc_minor << "\",\"sms\":" << p.sms << ",\"total_gib\":" << double(p.total_bytes) / double(1ULL << 30)
      << ",\"free_gib\":" << double(p.free_bytes) / double(1ULL << 30) << ",\"pcie\":{\"gen\":" << p.pcie_gen << ",\"width\":" << p.pcie_width
      << ",\"max_gen\":" << p.pcie_max_gen << ",\"max_width\":" << p.pcie_max_width << "},\"h2d_gbps\":" << p.h2d_gbps
      << ",\"h2d_cached\":" << (p.h2d_cached ? "true" : "false") << ",\"cache\":\"" << escape(p.cache_file) << "\"}";
    return s.str();
}

}  // namespace strix::ds41::cuda
