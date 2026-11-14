#include "../src/moe_core/dense_a8_small_layout.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "dense A8 reference requires strict floating point"
#endif

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
float half(unsigned h) {
    const unsigned sign = (h & 0x8000U) << 16U, exp = (h >> 10U) & 31U, mant = h & 1023U;
    if (!exp) {
        const float v = std::ldexp(static_cast<float>(mant), -24);
        return std::bit_cast<float>(std::bit_cast<unsigned>(v) | sign);
    }
    return std::bit_cast<float>(sign | (exp == 31U ? 0x7f800000U : (exp + 112U) << 23U) | (mant << 13U));
}
unsigned canonical(float v) { return std::isnan(v) ? 0x7fc00000U : std::bit_cast<unsigned>(v); }
int signed_byte(unsigned v) { return static_cast<int>(v & 127U) - static_cast<int>(v & 128U); }
unsigned load32(const unsigned char* p) {
    return unsigned(p[0]) | (unsigned(p[1]) << 8U) | (unsigned(p[2]) << 16U) | (unsigned(p[3]) << 24U);
}

// Independent byte-addressed scalar Q8_0 decoder vs the component's paired
// dword extraction, transposed wave layout and two exact int16-K partial dots.
void run(unsigned rows, unsigned width, unsigned flavor) {
    using namespace strix::dense_a8_small;
    const unsigned blocks = width / 32U;
    std::mt19937 rng(417U + rows * 11U + width + flavor);
    std::vector<unsigned char> weights(16U * blocks * 34U);
    std::vector<int> codes(rows * width);
    std::vector<float> deltas(rows * blocks);
    constexpr unsigned scales[]{0x0000U, 0x8000U, 0x0001U, 0x03ffU, 0x0400U, 0x3555U,
        0xb555U, 0x3c00U, 0xbc00U, 0x7bffU, 0x7c00U, 0xfc00U, 0x7e01U};
    for (unsigned n = 0; n < 16U; ++n) {
        for (unsigned b = 0; b < blocks; ++b) {
            auto* p = weights.data() + (n * blocks + b) * 34U;
            const unsigned h = scales[(n + b) % (flavor == 0U ? 10U : 13U)];
            p[0] = static_cast<unsigned char>(h); p[1] = static_cast<unsigned char>(h >> 8U);
            for (unsigned k = 0; k < 32U; ++k)
                p[2U + k] = static_cast<unsigned char>(flavor == 2U ? (k & 1U ? 127U : 128U) : rng() & 255U);
        }
    }
    for (auto& c : codes) c = static_cast<int>(rng() % 255U) - 127;
    for (unsigned b = 0; b < blocks; ++b)
        for (unsigned r = 0; r < rows; ++r)
            deltas[b * rows + r] = flavor == 1U && r == 1U && b == 0U
                ? std::bit_cast<float>(0x7fc00000U) : std::ldexp(1.0F + float((b + r) % 7U) / 8.0F, -9);
    std::vector<float> actual(rows * 16U, 0.0F);
    std::vector<unsigned> visits(rows * 16U, 0U);
    for (unsigned lane = 0; lane < 32U; ++lane) {
        const unsigned r = lane & 15U;
        for (unsigned e = 0; e < 8U; ++e) {
            const unsigned n = output_row(lane, e);
            require(weight_row(scale_lane(lane, e)) == n, "scale shuffle/output mapping");
            if (r >= rows) continue;
            float acc = 0.0F;
            for (unsigned pair = 0; pair < blocks / 2U; ++pair) {
                const auto* p = weights.data() + n * blocks * 34U + pair * 68U;
                unsigned q[17];
                for (unsigned j = 0; j < 17U; ++j) q[j] = load32(p + 4U * j);
                for (unsigned s = 0; s < 2U; ++s) {
                    int biased = 0x4b400000;
                    for (unsigned h = 0; h < 2U; ++h) {
                        int partial = 0;
                        for (unsigned k = 0; k < 4U; ++k) {
                            const unsigned word = code_word(q, s, h, k);
                            for (unsigned j = 0; j < 4U; ++j)
                                partial += signed_byte(word >> (8U * j)) * codes[r * width + pair * 64U + s * 32U + h * 16U + k * 4U + j];
                        }
                        biased += partial;
                    }
                    const float dot = std::bit_cast<float>(static_cast<unsigned>(biased)) - 12582912.0F;
                    const float scale = half(scale_bits(q, s)) * deltas[(2U * pair + s) * rows + r];
                    acc = std::fma(scale, dot, acc);
                }
            }
            actual[r * 16U + n] = acc;
            ++visits[r * 16U + n];
        }
    }
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned n = 0; n < 16U; ++n) {
            float expected = 0.0F;
            for (unsigned b = 0; b < blocks; ++b) {
                const auto* p = weights.data() + (n * blocks + b) * 34U;
                int dot = 0;
                for (unsigned k = 0; k < 32U; ++k) dot += signed_byte(p[2U + k]) * codes[r * width + b * 32U + k];
                const float scale = half(unsigned(p[0]) | (unsigned(p[1]) << 8U)) * deltas[b * rows + r];
                expected = std::fma(scale, static_cast<float>(dot), expected);
            }
            require(visits[r * 16U + n] == 1U, "masked stores must cover each real output once");
            require(canonical(actual[r * 16U + n]) == canonical(expected), "dequant/dot chain bits differ");
        }
    }
}
}  // namespace

int main() try {
    unsigned cases = 0;
    for (unsigned rows = 2; rows <= 8U; ++rows)
        for (unsigned width : {64U, 128U, 1024U, 8192U})
            for (unsigned flavor = 0; flavor < 3U; ++flavor) { run(rows, width, flavor); ++cases; }
    std::cout << "dense A8 small CPU layout/dequant/dot: " << cases << " bitwise cases passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
