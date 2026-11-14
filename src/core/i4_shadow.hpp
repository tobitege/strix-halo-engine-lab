#pragma once
// Measurement-only helpers: compare rank-local outputs without changing the live result.
#include <array>
#include <cstddef>
#include <cstdio>
#include <limits>

#include <hip/hip_runtime.h>

namespace strix::core::i4_shadow {

// One block owns the two output scalars, so no atomics or device reset are needed.
// Accumulate in double before narrowing for the small blocking host readback.
static __global__ __launch_bounds__(256) void squared_error_kernel(const float* test, const float* reference,
                                                                  std::size_t count, float* sums) {
    double error = 0.0, energy = 0.0;
    for (std::size_t i = threadIdx.x; i < count; i += blockDim.x) {
        const double value = static_cast<double>(reference[i]);
        const double delta = static_cast<double>(test[i]) - value;
        error += delta * delta;
        energy += value * value;
    }
    __shared__ double errors[256];
    __shared__ double energies[256];
    const unsigned lane = threadIdx.x;
    errors[lane] = error;
    energies[lane] = energy;
    __syncthreads();
    for (unsigned stride = 128U; stride != 0U; stride >>= 1U) {
        if (lane < stride) {
            errors[lane] += errors[lane + stride];
            energies[lane] += energies[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0U) {
        sums[0] = static_cast<float>(errors[0]);
        sums[1] = static_cast<float>(energies[0]);
    }
}

// Caller owns two device floats and orders reuse on stream; safe even for count == 0.
inline hipError_t reduce(const float* test, const float* reference, std::size_t count, float* sums, hipStream_t stream) {
    hipLaunchKernelGGL(squared_error_kernel, dim3(1), dim3(256), 0, stream, test, reference, count, sums);
    return hipGetLastError();
}

template <std::size_t Layers> class Accumulator {
    struct Sums {
        std::size_t rows = 0;
        double w4a4 = 0.0, a4 = 0.0, reference = 0.0;
    };
    std::array<Sums, Layers> layers_{};

    static double relative(double error, double reference) {
        if (reference == 0.0) return error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
        return error / reference;
    }

public:
    // sums = {W4A4 error, exact energy, A4 error, exact energy} from two reductions.
    void add(unsigned layer, std::size_t rows, const std::array<float, 4>& sums) {
        Sums& value = layers_.at(layer);
        value.rows += rows;
        value.w4a4 += static_cast<double>(sums[0]);
        value.a4 += static_cast<double>(sums[2]);
        value.reference += static_cast<double>(sums[1]);
    }

    // Totals divide summed error by summed exact energy (not the mean of layer ratios).
    void print(const char* engine, unsigned rank) const {
        Sums total;
        for (std::size_t layer = 0; layer < Layers; ++layer) {
            const Sums& value = layers_[layer];
            if (value.rows == 0) continue;
            std::printf("%s rank %u: fast_i4 shadow L=%u rows=%zu e_w4a4=%.6f e_a4=%.6f\n", engine, rank,
                        static_cast<unsigned>(layer), value.rows, relative(value.w4a4, value.reference), relative(value.a4, value.reference));
            total.rows += value.rows;
            total.w4a4 += value.w4a4;
            total.a4 += value.a4;
            total.reference += value.reference;
        }
        std::printf("%s rank %u: fast_i4 shadow total rows=%zu e_w4a4=%.6f e_a4=%.6f\n", engine, rank, total.rows,
                    relative(total.w4a4, total.reference), relative(total.a4, total.reference));
        std::fflush(stdout);
    }
};

}  // namespace strix::core::i4_shadow
