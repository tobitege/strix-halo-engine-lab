#pragma once

// Host-only verdicts shared by the dense-core and vision device fixtures.
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <istream>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace dense_fixture {

inline void require(bool pass, const char* message) {
    if (!pass) throw std::runtime_error(message);
}

inline bool same_bytes(std::span<const float> a, std::span<const float> b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size_bytes()) == 0);
}

template <class Output> void require_repeatable(const Output& got, const Output& first) {
    require(same_bytes(got.hidden, first.hidden) && got.taps.size() == first.taps.size(), "large forward is not repeatable");
    for (std::size_t tap = 0; tap < got.taps.size(); ++tap)
        require(same_bytes(got.taps[tap], first.taps[tap]), "large forward tap is not repeatable");
}

// Inputs in the core fixture are small dyadic values: every product and partial
// sum is exactly representable in FP32, independently of the WMMA reduction order.
inline std::vector<float> exact_reference(std::span<const float> x, std::span<const float> w,
                                          unsigned rows, unsigned in, unsigned n, unsigned pad, float guard) {
    std::vector<float> y(std::size_t{rows + 1U} * (n + pad), guard);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned c = 0; c < n; ++c) {
            float sum = 0.0F;
            for (unsigned k = 0; k < in; ++k) sum += x[std::size_t{r} * in + k] * w[std::size_t{c} * in + k];
            y[std::size_t{r} * (n + pad) + c] = sum;
        }
    return y;
}

inline void check_dense(std::span<const float> got, std::span<const float> want,
                        unsigned rows, unsigned n, unsigned pad, float guard) {
    require(got.size() == std::size_t{rows + 1U} * (n + pad), "dense output size");
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned c = 0; c < n; ++c) {
            const float v = got[std::size_t{r} * (n + pad) + c];
            require(v != guard && std::isfinite(v), "dense live output was not written or is nonfinite");
        }
    require(same_bytes(got, want), "dense reference or padding mismatch");
}

template <class Call> void require_numeric_failure(Call call) {
    try {
        call();
    } catch (const std::runtime_error& e) {
        const std::string_view message(e.what()), prefix = "vision encoder: numeric failure bits ";
        unsigned bits = 0;
        if (message.starts_with(prefix)) {
            const auto number = message.substr(prefix.size());
            const auto parsed = std::from_chars(number.data(), number.data() + number.size(), bits);
            // The encoder's nonfinite (1) or operand-range (8) report is expected.
            if (parsed.ec == std::errc{} && parsed.ptr == number.data() + number.size() && (bits & 9U) != 0U) return;
        }
        throw;  // Allocation, device and unrelated numeric errors are failures.
    }
    throw std::runtime_error("vision encoder: missing numeric failure");
}

// Same-machine OFF-arm reference, one dimensioned record per linear, in call order.
// Streams keep the comparison testable without a GPU or temporary files.
inline void linear_reference(std::istream* input, std::ostream* output, std::span<const float> values,
                             unsigned rows, unsigned in, unsigned n) {
    require(values.size() == std::size_t{rows} * n, "linear reference size");
    for (float v : values) require(std::isfinite(v), "linear nonfinite output");
    const std::uint32_t shape[] = {rows, in, n};
    const auto bytes = static_cast<std::streamsize>(values.size_bytes());
    if (output) {
        output->write(reinterpret_cast<const char*>(shape), sizeof(shape));
        output->write(reinterpret_cast<const char*>(values.data()), bytes);
        require(static_cast<bool>(*output), "linear reference write failed");
    } else {
        require(input != nullptr, "linear reference required");
        std::uint32_t saved_shape[3]{};
        input->read(reinterpret_cast<char*>(saved_shape), sizeof(saved_shape));
        require(static_cast<bool>(*input) && std::memcmp(shape, saved_shape, sizeof(shape)) == 0, "linear reference shape mismatch");
        std::vector<float> saved(values.size());
        input->read(reinterpret_cast<char*>(saved.data()), bytes);
        require(static_cast<bool>(*input) && same_bytes(values, saved), "linear OFF reference mismatch or truncation");
    }
}

inline void finish_reference(std::istream& input) {
    require(input.peek() == std::char_traits<char>::eof() && input.eof() && !input.bad(), "linear reference trailing bytes or read error");
}

}  // namespace dense_fixture
