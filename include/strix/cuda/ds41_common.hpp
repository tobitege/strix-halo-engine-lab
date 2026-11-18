#pragma once

// deepseek41 CUDA backend (sm_120/89/86): common pieces shared by every kernel family (layer core, attention, routed
// experts) and by the layer assembly.
//
// Conventions (all families):
//   * namespace strix::ds41::cuda; every kernel entry point is `enqueue_<site>(const <Site>View&, cudaStream_t)`:
//     it only enqueues on the stream (no synchronisation, no allocation) and checks the launch with
//     STRIX_CUDA_CHECK(cudaGetLastError()). Views hold device pointers, sizes and strides; scratch comes from the
//     caller (a Workspace) and is named in the view, with a `<site>_workspace_bytes(...)` function giving its size.
//   * Rows are independent tokens (a decode window of 1..8 rows or a prefill block of T rows); every form of a site
//     gives a row the same bits whatever the number of rows, as the host reference (ds41-exact-v1).
//   * Activation layouts are those of the host forward buffers, row-major with an explicit stride in elements:
//     the residual stream is [row][hc][hidden] FP32; an hc mix is 24 floats per row (pre 4, post 4, comb 16, the
//     HcMix layout); A8 activations are codes [row][stride] int8 and deltas / sums [row][stride / 32].
//   * Weights keep their GGUF bytes: a weight [in, out] is `out` contiguous rows of `in` values (Q8_0: 34-byte
//     blocks {f16 d, 32 int8 codes}).
//   * Translation units are built with --fmad=false -prec-div=true -prec-sqrt=true -ftz=false (no contraction, IEEE
//     division and root, denormals kept), so FP32 code written in the declared order gives the reference bits.

#include "strix/cuda/core/types.hpp"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strix::ds41::cuda {

// Throws std::runtime_error naming the call and the CUDA error.
void check(cudaError_t status, const char* what, const char* file, int line);
#define STRIX_CUDA_CHECK(call) ::strix::ds41::cuda::check((call), #call, __FILE__, __LINE__)

struct DeviceInfo {
    int ordinal = 0;
    std::string name;
    int major = 0, minor = 0, sms = 0;
    std::size_t total_bytes = 0;
    int binary_version = 0;  // the architecture of the code the device runs (120, 89, 86: the build's image for it)
};
// Selects the device and checks that the build has code for it (compute capability 8.6+, an image of its
// architecture in CMAKE_CUDA_ARCHITECTURES: sm_120 for Blackwell, sm_89 for Ada, sm_86 for Ampere GA10x).
DeviceInfo select_device(int ordinal = 0);

// Bytes currently held by every DeviceBuffer of the process (accounting: deltas around a load give its device bytes
// without cudaMemGetInfo, which counts the other processes of a shared GPU).
inline std::atomic<std::size_t>& device_buffer_bytes() {
    static std::atomic<std::size_t> bytes{0};
    return bytes;
}

// Owning device allocation of n values of T (cudaMalloc; 256-byte aligned).
template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) { resize(n); }
    ~DeviceBuffer() { release(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { release(); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; }
        return *this;
    }
    void resize(std::size_t n) {
        release();
        if (n) STRIX_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p_), n * sizeof(T)));
        n_ = n;
        device_buffer_bytes() += n * sizeof(T);
    }
    // Blocking copy, complete on return (also for non-blocking streams, which do not order against cudaMemcpy).
    void upload(const T* host, std::size_t n, std::size_t offset = 0) {
        STRIX_CUDA_CHECK(cudaMemcpy(p_ + offset, host, n * sizeof(T), cudaMemcpyHostToDevice));
        STRIX_CUDA_CHECK(cudaDeviceSynchronize());
    }
    // Copy ordered on `stream` (host memory must stay valid until it completes; pinned for a true async copy).
    void upload_async(const T* host, std::size_t n, cudaStream_t stream, std::size_t offset = 0) {
        STRIX_CUDA_CHECK(cudaMemcpyAsync(p_ + offset, host, n * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    void upload(const std::vector<T>& host) { if (host.size() > n_) resize(host.size()); upload(host.data(), host.size()); }
    [[nodiscard]] std::vector<T> download(std::size_t n = 0) const {
        std::vector<T> out(n ? n : n_);
        STRIX_CUDA_CHECK(cudaDeviceSynchronize());
        if (!out.empty()) STRIX_CUDA_CHECK(cudaMemcpy(out.data(), p_, out.size() * sizeof(T), cudaMemcpyDeviceToHost));
        return out;
    }
    void zero() {
        if (n_) STRIX_CUDA_CHECK(cudaMemset(p_, 0, n_ * sizeof(T)));
        STRIX_CUDA_CHECK(cudaDeviceSynchronize());
    }
    [[nodiscard]] T* get() const noexcept { return p_; }
    [[nodiscard]] std::size_t size() const noexcept { return n_; }

private:
    void release() noexcept {
        if (p_) cudaFree(p_);
        device_buffer_bytes() -= n_ * sizeof(T);
        p_ = nullptr;
        n_ = 0;
    }
    T* p_ = nullptr;
    std::size_t n_ = 0;
};

// Bump allocator over one device allocation: per-step scratch of the kernels (take() is host-side bookkeeping only;
// reset() makes every byte reusable, so the caller orders reuse on its stream).
class Workspace {
public:
    explicit Workspace(std::size_t bytes = 0) { reserve(bytes); }
    void reserve(std::size_t bytes);
    void reset() noexcept { used_ = 0; }
    [[nodiscard]] void* take_bytes(std::size_t bytes, std::size_t align = 256);
    template <class T>
    [[nodiscard]] T* take(std::size_t n, std::size_t align = 256) { return static_cast<T*>(take_bytes(n * sizeof(T), align)); }
    [[nodiscard]] std::size_t capacity() const noexcept { return storage_.size(); }
    [[nodiscard]] std::size_t used() const noexcept { return used_; }

private:
    DeviceBuffer<std::uint8_t> storage_;
    std::size_t used_ = 0;
};

// Non-blocking stream owned by the caller of the kernels.
class Stream {
public:
    Stream();
    ~Stream();
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    [[nodiscard]] cudaStream_t get() const noexcept { return s_; }
    void sync() const;

private:
    cudaStream_t s_ = nullptr;
};

// Elapsed device time between two events recorded on a stream.
class EventTimer {
public:
    EventTimer();
    ~EventTimer();
    EventTimer(const EventTimer&) = delete;
    EventTimer& operator=(const EventTimer&) = delete;
    void start(cudaStream_t s);
    // Records the stop event, waits for it and returns the milliseconds since start().
    float stop(cudaStream_t s);

private:
    cudaEvent_t a_ = nullptr, b_ = nullptr;
};

// ---- views ------------------------------------------------------------------------------------------------------
inline constexpr std::uint32_t q8_block_values = 32, q8_block_bytes = 34, a8_group_values = 32;
[[nodiscard]] constexpr std::size_t q8_0_row_bytes(std::size_t width) noexcept { return width / q8_block_values * q8_block_bytes; }

// GGUF Q8_0 weight: `rows` rows of `width` values (width a multiple of 32); rows are 16-byte aligned when the base
// is and width % 256 is 0 or width / 32 * 34 % 16 is 0 (every width of this model).
struct Q8Weights {
    const std::uint8_t* data = nullptr;
    std::uint32_t rows = 0, width = 0;
    [[nodiscard]] std::size_t row_bytes() const noexcept { return q8_0_row_bytes(width); }
};
// F16 / F32 weights: `rows` rows of `width` values.
struct F16Weights {
    const std::uint16_t* data = nullptr;
    std::uint32_t rows = 0, width = 0;
};
struct F32Weights {
    const float* data = nullptr;
    std::uint32_t rows = 0, width = 0;
};
// A8 activations of `rows` rows (quantize_a8 per row): codes [row][stride], deltas and sums [row][stride / 32].
// width <= stride is the part a site reads; stride and width multiples of 32.
using A8Rows = ::strix::cuda::core::A8Rows;
// Device storage for an A8Rows view.
struct A8Buffer {
    DeviceBuffer<std::int8_t> codes;
    DeviceBuffer<float> deltas;
    DeviceBuffer<std::int32_t> sums;
    std::uint32_t rows = 0, width = 0;
    void resize(std::uint32_t rows_, std::uint32_t width_) {
        rows = rows_; width = width_;
        codes.resize(std::size_t{rows} * width);
        deltas.resize(std::size_t{rows} * width / a8_group_values);
        sums.resize(std::size_t{rows} * width / a8_group_values);
    }
    [[nodiscard]] A8Rows view() const noexcept { return {codes.get(), deltas.get(), sums.get(), rows, width, width}; }
};

// The op hash of the host forward: FNV-1a 64 over the little-endian bytes of the FP32 values.
[[nodiscard]] std::uint64_t fnv1a(const float* values, std::size_t n) noexcept;

}  // namespace strix::ds41::cuda
