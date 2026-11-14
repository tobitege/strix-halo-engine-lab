// Exchange row fold (xport/row_fold.hpp): the vectorised host sum equals a one-addition-per-element scalar oracle
// bit for bit, on raw bit patterns (every exponent, subnormals, signed zeros, infinities) and at every tail length.
#include "xport/row_fold.hpp"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
std::uint32_t bits(float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, 4);
    return u;
}
float from(std::uint32_t u) {
    float v;
    std::memcpy(&v, &u, 4);
    return v;
}
}  // namespace

int main() {
    using strix::xport::RowFold;
    const std::size_t n = std::size_t{1} << 22;
    std::vector<float> a(n), b(n), out(n + 1), ref(n);
    std::uint64_t z = 0x9e3779b97f4a7c15ULL;
    for (std::size_t i = 0; i < n; ++i) {
        z ^= z << 13U;
        z ^= z >> 7U;
        z ^= z << 17U;
        std::uint32_t ua = static_cast<std::uint32_t>(z), ub = static_cast<std::uint32_t>(z >> 32U);
        switch (i % 8) {
        case 1: ub = (ub & 0x807fffffU) | (ua & 0x7f800000U); break;  // same exponent
        case 2: ub = ua ^ 0x80000000U; break;                         // exact cancellation
        case 3: ua &= 0x807fffffU; ub &= 0x807fffffU; break;          // subnormals
        case 4: ua &= 0x80000000U; ub &= 0x80000000U; break;          // signed zeros
        default: break;
        }
        a[i] = from(ua);
        b[i] = from(ub);
        volatile float x = a[i], y = b[i];  // scalar oracle: one binary32 addition
        volatile float s = x + y;
        ref[i] = s;
    }
    int failures = 0;
    auto check = [&](std::size_t offset, std::size_t count) {
        const float guard = from(0x7fc01234U);
        for (std::size_t i = 0; i <= count; ++i) out[i] = guard;
        strix::xport::fold_rows(RowFold::sum_f32, out.data(), a.data() + offset, b.data() + offset, count * 4);
        std::size_t bad = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const float r = ref[offset + i];
            if (r != r ? out[i] == out[i] : bits(out[i]) != bits(r)) ++bad;
        }
        if (bits(out[count]) != 0x7fc01234U) ++bad;  // nothing written past the message
        if (bad != 0) {
            std::cerr << "row_fold: " << bad << " differences at offset " << offset << " count " << count << "\n";
            ++failures;
        }
    };
    check(0, n);
    for (std::size_t count = 0; count <= 70; ++count) check(count * 3 + 1, count);
    // RowFold::none leaves the output untouched.
    out[0] = 7.0F;
    strix::xport::fold_rows(RowFold::none, out.data(), a.data(), b.data(), 64);
    if (out[0] != 7.0F) ++failures;
    if (failures != 0) return 1;
    std::cout << "xport row fold tests passed\n";
    return 0;
}
