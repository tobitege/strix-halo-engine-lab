#include "strix/cuda_ds41_common.hpp"
#include "strix/device/cuda_enumerator.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace strix::ds41::cuda {

namespace {
__global__ void probe_kernel() {}
}  // namespace

void check(cudaError_t status, const char* what, const char* file, int line) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorName(status) + " (" + cudaGetErrorString(status) + ") in " + what +
                             " at " + file + ":" + std::to_string(line));
}

DeviceInfo select_device(int ordinal) {
    // The start gate picks the card (declared, not excluded, with a role) and restricts the process to it. An index is
    // never an identity, so `ordinal` cannot move work to another card: it has to agree with the bound device.
    const auto& bound = strix::device::bind_cuda();
    if (ordinal > 0 && ordinal != bound.visible.ordinal)
        throw std::runtime_error("ds41 cuda: device index " + std::to_string(ordinal) + " is not available: the process is bound to declared device " +
                                 bound.device.id + " (select a card with --device-id or STRIX_DEVICE)");
    ordinal = bound.visible.ordinal;
    cudaDeviceProp p{};
    STRIX_CUDA_CHECK(cudaGetDeviceProperties(&p, ordinal));
    DeviceInfo d;
    d.ordinal = ordinal;
    d.name = p.name;
    d.major = p.major;
    d.minor = p.minor;
    d.sms = p.multiProcessorCount;
    d.total_bytes = p.totalGlobalMem;
    // The kernels use the INT8 mma.sync / ldmatrix / cp.async of sm_80+ only (FP8 conversions fall back to the exact
    // software path below sm_89), so any 8.6+ device runs them if the build has its code (CMAKE_CUDA_ARCHITECTURES,
    // e.g. "120;89;86"); a device without an image fails the probe kernel's attribute query here.
    const auto where = "ds41 cuda: device " + d.name + " (compute capability " + std::to_string(d.major) + "." + std::to_string(d.minor) + ")";
    if (d.major < 8 || (d.major == 8 && d.minor < 6)) throw std::runtime_error(where + ": needs 8.6 or newer");
    cudaFuncAttributes attr{};
    const cudaError_t probe = cudaFuncGetAttributes(&attr, probe_kernel);
    if (probe != cudaSuccess) {
        (void)cudaGetLastError();
        throw std::runtime_error(where + ": this build has no code for it (" + cudaGetErrorName(probe) +
                                 "); add its architecture to CMAKE_CUDA_ARCHITECTURES");
    }
    d.binary_version = attr.binaryVersion;
    return d;
}

void Workspace::reserve(std::size_t bytes) {
    if (bytes > storage_.size()) storage_.resize(bytes);
    used_ = 0;
}

void* Workspace::take_bytes(std::size_t bytes, std::size_t align) {
    const std::size_t at = (used_ + align - 1U) / align * align;
    if (at + bytes > storage_.size())
        throw std::runtime_error("ds41 cuda workspace: " + std::to_string(at + bytes) + " bytes needed, " +
                                 std::to_string(storage_.size()) + " reserved");
    used_ = at + bytes;
    return storage_.get() + at;
}

Stream::Stream() { STRIX_CUDA_CHECK(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking)); }
Stream::~Stream() {
    if (s_) cudaStreamDestroy(s_);
}
void Stream::sync() const { STRIX_CUDA_CHECK(cudaStreamSynchronize(s_)); }

EventTimer::EventTimer() {
    STRIX_CUDA_CHECK(cudaEventCreate(&a_));
    STRIX_CUDA_CHECK(cudaEventCreate(&b_));
}
EventTimer::~EventTimer() {
    if (a_) cudaEventDestroy(a_);
    if (b_) cudaEventDestroy(b_);
}
void EventTimer::start(cudaStream_t s) { STRIX_CUDA_CHECK(cudaEventRecord(a_, s)); }
float EventTimer::stop(cudaStream_t s) {
    STRIX_CUDA_CHECK(cudaEventRecord(b_, s));
    STRIX_CUDA_CHECK(cudaEventSynchronize(b_));
    float ms = 0.0F;
    STRIX_CUDA_CHECK(cudaEventElapsedTime(&ms, a_, b_));
    return ms;
}

std::uint64_t fnv1a(const float* values, std::size_t n) noexcept {
    std::uint64_t h = 14695981039346656037ULL;
    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t b = __builtin_bit_cast(std::uint32_t, values[i]);
        for (int k = 0; k < 4; ++k, b >>= 8U) {
            h ^= b & 0xffU;
            h *= 1099511628211ULL;
        }
    }
    return h;
}

}  // namespace strix::ds41::cuda
