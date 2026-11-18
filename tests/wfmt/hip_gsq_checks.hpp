#pragma once
// Host-only checks shared by the HIP fixture and its fault-injection tests.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace strix::wfmt_gsq {
struct Snapshot {
    std::vector<float> output, dd, raw;
    std::vector<signed char> dcodes;
    std::vector<int> ds;
};
inline bool same(const Snapshot& a, const Snapshot& b) {
    const auto eq = [](const auto& x, const auto& y) {
        return x.size() == y.size() && (x.empty() || !std::memcmp(x.data(), y.data(), x.size() * sizeof x[0]));
    };
    return eq(a.output, b.output) && eq(a.dd, b.dd) && eq(a.raw, b.raw) && eq(a.dcodes, b.dcodes) && eq(a.ds, b.ds);
}
inline void check_snapshot(const Snapshot& s, const std::vector<unsigned>& status, unsigned csr_status) {
    if (csr_status || std::any_of(status.begin(), status.end(), [](unsigned v) { return v != 0U; }))
        throw std::runtime_error("GSQ fixture: device/CSR status error");
    for (const auto* values : {&s.output, &s.dd, &s.raw})
        if (!std::all_of(values->begin(), values->end(), [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("GSQ fixture: nonfinite output");
    if (std::none_of(s.raw.begin(), s.raw.end(), [](float v) { return v != 0.0F; }))
        throw std::runtime_error("GSQ fixture: all-zero down output");
}
// Two independent one-assignment batches retain both slots, even for duplicate
// routes. The caller rebases the selected token's pointers to local row zero.
inline std::array<unsigned, 41> row_csr(const std::vector<unsigned>& words, unsigned batches,
                                      unsigned rows, unsigned slots, unsigned experts, unsigned begins, unsigned row) {
    std::array<unsigned, 41> result{};
    unsigned found = 0;
    for (unsigned b = 0; b < batches; ++b)
        for (unsigned i = words[begins + b]; i < words[begins + b + 1U]; ++i) {
            if (words[rows + i] != row) continue;
            const auto slot = words[slots + i];
            if (slot >= 2U || (found & (1U << slot))) throw std::runtime_error("GSQ fixture: invalid row slots");
            result[36U + slot] = words[experts + b];
            found |= 1U << slot;
        }
    if (found != 3U) throw std::runtime_error("GSQ fixture: missing row slot");
    result[1] = result[36] == result[37] ? 1U : 2U;
    result[2] = 2U; result[3] = 2U; result[4] = result[36] == result[37] ? 1U : 0U;
    result[5] = 1U; result[6] = 1U; result[7] = 1U;
    result[35] = 1U; result[39] = 1U; result[40] = 2U;
    return result;
}
} // namespace strix::wfmt_gsq
