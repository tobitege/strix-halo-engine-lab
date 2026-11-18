#include "tolerance.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

using strix::cuda::core::nvfp4::test::within_accumulation_tolerance;
static void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
int main() try {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    check(within_accumulation_tolerance(1.f, 1.f, 1.), "equal finite values");
    check(within_accumulation_tolerance(0.00001f, 0.f, 0.), "within tolerance");
    check(!within_accumulation_tolerance(0.00002f, 0.f, 0.), "above tolerance");
    check(!within_accumulation_tolerance(nan, 1.f, 1.), "injected NaN actual accepted");
    check(!within_accumulation_tolerance(1.f, nan, 1.), "NaN reference accepted");
    check(!within_accumulation_tolerance(nan, nan, 1.), "matching NaNs accepted");
    for (const float invalid : {inf, -inf}) {
        check(!within_accumulation_tolerance(invalid, 1.f, 1.), "infinite actual accepted");
        check(!within_accumulation_tolerance(1.f, invalid, 1.), "infinite reference accepted");
        check(!within_accumulation_tolerance(invalid, invalid, 1.), "matching infinities accepted");
    }
    check(!within_accumulation_tolerance(1.f, 1.f, nan), "NaN bound accepted");
    check(!within_accumulation_tolerance(1.f, 1.f, inf), "infinite bound accepted");
    check(!within_accumulation_tolerance(1.f, 1.f, -1.), "negative absolute sum accepted");
    std::cout << "NVFP4 tolerance: finite bounds and nonfinite rejection PASS\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
