#pragma once
#include <cmath>

namespace strix::cuda::core::nvfp4::test {
// Shared by host regression tests and the native general-scale fixtures.
inline bool within_accumulation_tolerance(float actual, float reference, double abs_sum) {
    const double err = std::fabs(double(actual) - reference);
    const double tolerance = 2.e-6 * abs_sum + 1.e-5;
    return std::isfinite(actual) && std::isfinite(reference) &&
        std::isfinite(abs_sum) && abs_sum >= 0 &&
        std::isfinite(err) && std::isfinite(tolerance) && err <= tolerance;
}
}
