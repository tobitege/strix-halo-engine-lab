// Host rejection contract, also exercised against CUDA when enabled.
#include "strix/cuda/core/nvfp4_a8.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#ifdef STRIX_TEST_A8_CUDA
#include "a8_nonfinite_device.hpp"
#endif

int main() {
    namespace a8 = strix::cuda::core::nvfp4_a8;
    const std::array<std::uint16_t, 5> scales{0x7e00, 0xfe00, 0x7c01, 0x7c00, 0xfc00};
    for (const auto bits : scales) {
        for (const std::uint8_t code : {std::uint8_t{0}, std::uint8_t{1}}) {
            std::array<std::uint8_t, 34> q8{};
            q8.fill(code);
            std::memcpy(q8.data(), &bits, sizeof(bits));
            std::array<std::uint8_t, 16> codes{};
            std::array<std::uint8_t, 2> groups{};
            float tensor_scale = 0;
#ifdef STRIX_TEST_A8_CUDA
            if (!check_device(q8, true)) return 1;
#endif
            try {
                a8::reference_convert_q8_0(q8.data(), 1, 32, codes.data(), groups.data(), &tensor_scale);
            } catch (const std::invalid_argument& e) {
                if (std::strcmp(e.what(), "nonfinite q8_0 weight") == 0) continue;
                std::fputs("unexpected rejection message\n", stderr);
                return 1;
            }
            std::fputs("nonfinite Q8_0 scale was accepted\n", stderr);
            return 1;
        }
    }
#ifdef STRIX_TEST_A8_CUDA
    std::array<std::uint8_t, 34> finite{};
    const std::uint16_t one = 0x3c00;
    std::memcpy(finite.data(), &one, sizeof(one));
    for (std::size_t j = 2; j < finite.size(); ++j) finite[j] = static_cast<std::uint8_t>(j * 7);
    if (!check_device(finite, false)) return 1;
#endif
    return 0;
}
