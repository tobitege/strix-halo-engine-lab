#pragma once
// Per-layer policy set shared by the FAST_I4 engines: which layers of a lossy format keep the exact (EXL3) operator.
// Host-only; engines read one environment value through parse_layer_set and test layers on the scheduling path.
// Grammar: "" or "none" = empty set; a comma list of layer indices and inclusive ranges ("3,7-9"); a leading '!'
// complements the list inside [first, end) ("!12" = every layer but 12). Indices are absolute layer numbers.
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::core {

struct LayerSet {
    static constexpr unsigned capacity = 256;
    std::array<std::uint64_t, capacity / 64U> bits{};
    constexpr bool test(unsigned layer) const { return layer < capacity && ((bits[layer / 64U] >> (layer % 64U)) & 1U) != 0; }
    constexpr void set(unsigned layer) { bits[layer / 64U] |= std::uint64_t{1} << (layer % 64U); }
    constexpr unsigned count() const {
        unsigned n = 0;
        for (unsigned l = 0; l < capacity; ++l) n += test(l) ? 1U : 0U;
        return n;
    }
};

inline LayerSet parse_layer_set(const char* value, unsigned first, unsigned end, const char* name) {
    auto bad = [&] {
        return std::invalid_argument(std::string(name) + " must be none or a comma list of layers/ranges in [" +
                                     std::to_string(first) + ", " + std::to_string(end) + "), optionally prefixed by !");
    };
    if (end > LayerSet::capacity || first >= end) throw std::invalid_argument("parse_layer_set: layer range");
    LayerSet listed;
    std::string_view s = value == nullptr ? "" : value;
    if (s.empty() || s == "none") return listed;
    const bool complement = s.front() == '!';
    if (complement) s.remove_prefix(1);
    auto number = [&](std::string_view t) {
        if (t.empty() || t.size() > 3) throw bad();
        unsigned v = 0;
        for (const char c : t) {
            if (c < '0' || c > '9') throw bad();
            v = v * 10U + static_cast<unsigned>(c - '0');
        }
        if (v < first || v >= end) throw bad();
        return v;
    };
    while (!s.empty()) {
        const auto comma = s.find(',');
        const std::string_view item = s.substr(0, comma);
        const auto dash = item.find('-');
        const unsigned a = number(item.substr(0, dash));
        const unsigned b = dash == std::string_view::npos ? a : number(item.substr(dash + 1));
        if (b < a) throw bad();
        for (unsigned l = a; l <= b; ++l) listed.set(l);
        if (comma == std::string_view::npos) break;
        s.remove_prefix(comma + 1);
        if (s.empty()) throw bad();
    }
    if (!complement) return listed;
    LayerSet out;
    for (unsigned l = first; l < end; ++l)
        if (!listed.test(l)) out.set(l);
    return out;
}

}  // namespace strix::core
