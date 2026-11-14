#pragma once
// Element-wise comparison of a computed tensor with a reference, for the gates of every image tower.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>

namespace strix::vision {

struct AbsDiff {
    double worst = 0.0;          // largest finite |got - want|
    std::size_t over = 0;        // elements above the tolerance
    std::size_t non_finite = 0;  // elements where either operand is NaN or infinite
    bool pass = false;
};

inline AbsDiff abs_diff(std::span<const float> got, std::span<const float> want, double tolerance) {
    AbsDiff r;
    if (got.size() != want.size() || !(tolerance >= 0.0)) return r;  // pass stays false
    for (std::size_t i = 0; i < got.size(); ++i) {
        // A NaN compares false with everything, so it is counted, never reduced: max() and > would both drop it.
        if (!std::isfinite(got[i]) || !std::isfinite(want[i])) {
            ++r.non_finite;
            continue;
        }
        const double d = std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
        r.worst = std::max(r.worst, d);
        r.over += d > tolerance;
    }
    r.pass = r.non_finite == 0 && r.over == 0;
    return r;
}

}  // namespace strix::vision
