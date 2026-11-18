#pragma once
// CUDA core v0: model-agnostic device building blocks. No model names here; callers pass views, dims and a stream.
// Numeric identity rule (Home: shared, backend core CUDA): every reduction has ONE documented fixed order that does not
// depend on the row count or the launch shape, all arithmetic is IEEE FP32 (no FMA contraction: the library is built with
// --fmad=false, and the kernels spell mul/add as __fmul_rn/__fadd_rn), division/sqrt are IEEE. Transcendentals go through
// one point (cc_expf/cc_log1pf in src/cuda_core/common.cuh) so a HIP twin can adopt the same bits.
#include <cstddef>
#include <cstdint>
#include <utility>

#include <cuda_runtime.h>

namespace strix::cuda_core {

// Move-only CUDA handle, also used for allocations during partial construction.
template <class T, auto Release>
class UniqueHandle {
    T value_{};
public:
    UniqueHandle() = default;
    explicit UniqueHandle(T value) noexcept : value_(value) {}
    ~UniqueHandle() { reset(); }
    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;
    UniqueHandle(UniqueHandle&& other) noexcept : value_(std::exchange(other.value_, T{})) {}
    UniqueHandle& operator=(UniqueHandle&& other) noexcept {
        if (this != &other) { reset(); value_ = std::exchange(other.value_, T{}); }
        return *this;
    }
    void reset(T value = {}) noexcept { if (value_) (void)Release(value_); value_ = value; }
    T get() const noexcept { return value_; }
    T* put() noexcept { reset(); return &value_; }
};

enum class WeightFormat : std::uint8_t {
    bf16,  // 2 bytes per element, row-major [out][in]
    q8_0,  // GGUF Q8_0: per 32 elements {f16 d, int8 q[32]} = 34 bytes, row-major [out][in/32 blocks]
    f32,   // 4 bytes per element (small tensors)
    nvfp4, // TURBO: e2m1 + e4m3 scale per 16 + FP32 global scale (gscale); tile layout in nvfp4.hpp (rows 1..32)
};

// A 2-D weight [out][in] in device memory (no ownership).
struct DenseWeight {
    const void* data = nullptr;
    WeightFormat format = WeightFormat::bf16;
    std::uint32_t out = 0;
    std::uint32_t in = 0;
    float gscale = 1.0f;  // nvfp4 only: per-tensor FP32 global scale
};

// Rows x groups x n view of an FP32 tensor: element (r, g, i) at base[r * row_stride + g * group_stride + i].
// A zero stride selects the contiguous default (row_stride = groups * n, group_stride = n).
struct GroupedView {
    std::uint32_t rows = 1, groups = 1, n = 0;
    std::uint32_t row_stride = 0, group_stride = 0;
    [[nodiscard]] __host__ __device__ std::uint32_t rs() const noexcept { return row_stride ? row_stride : groups * n; }
    [[nodiscard]] __host__ __device__ std::uint32_t gs() const noexcept { return group_stride ? group_stride : n; }
};

// Programmatic dependent launch between consecutive kernels (default on); takes effect for graphs captured afterwards.
void set_programmatic_launch(bool on);

constexpr std::uint32_t kMaxRows = 32;  // verify window

// Per-row indirection for multi-session decode (null members = single-session behaviour). Device arrays [rows]:
// pos[r] = absolute position of row r (null: *pos0_dev + r); slot[r] = cache slot of row r (null: slot 0); a cache
// buffer holds [slot][...] blocks of slot_stride elements. Rows of one slot are consecutive and ascending in position.
struct RowMap {
    const std::uint32_t* pos = nullptr;
    const std::uint32_t* slot = nullptr;
    std::size_t slot_stride = 0;
};

}  // namespace strix::cuda_core
