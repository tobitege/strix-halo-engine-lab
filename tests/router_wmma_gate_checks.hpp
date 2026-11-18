#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace router_wmma_test {
inline constexpr float probe_input = 1.0F + 0x1p-12F;  // rounds to 1 in FP16

// One nonzero finite product, independent of accumulation order. Column 1 is
// zero: an injected FP16 infinity there makes WMMA NaN, but exact stays finite.
template <class Weight>
void prepare_probe(std::vector<float>& x, unsigned width, std::vector<Weight>& w,
                   unsigned experts, Weight one) {
    std::fill_n(x.begin(), width, 0.0F);
    x[0] = probe_input;
    for (unsigned e = 0; e < experts; ++e) {
        w[std::size_t{e} * width] = one;
        w[std::size_t{e} * width + 1U] = Weight{};
    }
}
inline void inject_overflow(std::vector<float>& x, unsigned width, unsigned row) {
    const auto offset = std::size_t{row} * width;
    std::fill_n(x.begin() + static_cast<std::ptrdiff_t>(offset), width, 0.0F);
    x[offset] = probe_input;
    x[offset + 1U] = 1.0e5F;
}
inline bool complete(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); });
}
inline bool wmma_probe(float value, float exact, float expected, float tolerance) {
    return std::isfinite(value) && std::isfinite(exact) &&
           std::fabs(value - expected) <= tolerance && std::fabs(value - exact) > 1.0e-5F;
}
}  // namespace router_wmma_test
