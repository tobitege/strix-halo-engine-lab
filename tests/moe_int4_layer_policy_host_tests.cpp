// Host checks: the stored INT4 layer / scope policy and the fast full-range IU4 scale rule.
#include "core/iu4.hpp"
#include "moe_core/int4_layer_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int checks = 0, failures = 0;
#define CHECK(c) do { ++checks; if (!(c)) { ++failures; \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

bool refused(const std::string& spec) {
    try {
        (void)strix::moe_core::parse_int4_source_layers(spec, "LAYERS");
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

// Relative squared error of one 64-value group under a rule.
double group_error(const std::vector<float>& w, bool full) {
    float amax = 0, pos = 0, neg = 0;
    for (float v : w) {
        amax = std::max(amax, std::fabs(v));
        pos = std::max(pos, v);
        neg = std::max(neg, -v);
    }
    const float s = full ? strix::core::iu4_fast_full_scale(pos, neg) : (amax == 0.0f ? 1.0f : std::max(amax / 7.0f, 0x1p-24f));
    const float inv = 1.0f / s;
    double err = 0, energy = 0;
    for (float v : w) {
        const int q = full ? strix::core::iu4_requant_full(v, inv) : strix::core::iu4_requant(v, inv);
        if (full) CHECK(q >= -8 && q <= 7);
        else CHECK(q >= -7 && q <= 7);
        const double d = double(v) - double(q) * double(s);
        err += d * d;
        energy += double(v) * double(v);
    }
    return energy == 0 ? 0 : err / energy;
}
}  // namespace

int main() {
    namespace mc = strix::moe_core;
    using strix::core::iu4_fast_full_scale;
    using strix::core::iu4_requant_full;

    // Layer spec.
    CHECK(mc::parse_int4_source_layers("none", "L") == 0);
    CHECK(mc::parse_int4_source_layers("0-14", "L") == 0x7fffULL);
    CHECK(mc::parse_int4_source_layers("3", "L") == 8ULL);
    CHECK(mc::parse_int4_source_layers("0,2,5-6", "L") == 0x65ULL);
    CHECK(mc::parse_int4_source_layers("63", "L") == (std::uint64_t{1} << 63U));
    CHECK(mc::parse_int4_source_layers("0-63", "L") == ~std::uint64_t{0});
    for (const char* bad : {"", "all", "64", "5-2", "1,", ",1", "1--2", "a", "1-", "-1", "001", "0 - 3"}) CHECK(refused(bad));
    try {
        (void)mc::parse_int4_source_layers("x", "MY_OPTION");
        CHECK(false);
    } catch (const std::invalid_argument& e) {
        CHECK(std::string(e.what()).rfind("MY_OPTION", 0) == 0);
    }

    // Scope.
    CHECK(mc::parse_int4_scope("gate_up", "S") == 2U);
    CHECK(mc::parse_int4_scope("all", "S") == 3U);
    bool threw = false;
    try {
        (void)mc::parse_int4_scope("down", "S");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    // Policy.
    mc::Int4LayerPolicy p;
    CHECK(p.matrices == 2U && !p.down_int4() && p.int4(0) && p.count(42) == 42U);
    p.source_mask = mc::parse_int4_source_layers("0-14", "L");
    CHECK(!p.int4(0) && !p.int4(14) && p.int4(15) && p.int4(41) && p.count(42) == 27U && p.count(15) == 0U);
    mc::Int4LayerPolicy q = p;
    q.matrices = 3;
    CHECK(q.down_int4() && q.tag(0) != p.tag(0));
    CHECK(p.tag(0) != p.tag(2001));
    mc::Int4LayerPolicy r = p;
    r.source_mask = mc::parse_int4_source_layers("0-13", "L");
    CHECK(r.tag(2001) != p.tag(2001) && p.tag(2001) == mc::Int4LayerPolicy{p}.tag(2001));

    // Fast full-range rule: the scale, the clamp and round-to-nearest-even.
    CHECK(iu4_fast_full_scale(0.0f, 0.0f) == 1.0f);
    CHECK(iu4_fast_full_scale(7.0f, 0.0f) == 0.9f * 1.0f);
    CHECK(iu4_fast_full_scale(0.0f, 8.0f) == 0.9f * 1.0f);
    CHECK(iu4_fast_full_scale(7.0f, 16.0f) == 0.9f * 2.0f);
    CHECK(iu4_fast_full_scale(0x1p-30f, 0.0f) == 0x1p-24f);
    CHECK(iu4_requant_full(100.0f, 1.0f) == 7 && iu4_requant_full(-100.0f, 1.0f) == -8);
    CHECK(iu4_requant_full(-7.6f, 1.0f) == -8 && iu4_requant_full(7.6f, 1.0f) == 7);
    CHECK(iu4_requant_full(0.5f, 1.0f) == 0 && iu4_requant_full(1.5f, 1.0f) == 2 && iu4_requant_full(-2.5f, 1.0f) == -2);
    CHECK(strix::core::iu4_scale_rule_valid(0) && strix::core::iu4_scale_rule_valid(2001) && !strix::core::iu4_scale_rule_valid(1));

    // Gaussian g64 groups: the fast full-range rule has the lower error (documented 0.0096 against 0.0116).
    std::mt19937 gen(20261112U);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    double legacy = 0, full = 0;
    const int groups = 4096;
    for (int g = 0; g < groups; ++g) {
        std::vector<float> w(64);
        for (float& v : w) v = normal(gen);
        legacy += group_error(w, false);
        full += group_error(w, true);
    }
    legacy /= groups;
    full /= groups;
    std::printf("gaussian g64 relative error: legacy %.5f, fast full range %.5f\n", legacy, full);
    CHECK(full < legacy);
    CHECK(legacy > 0.010 && legacy < 0.013 && full > 0.008 && full < 0.011);

    std::printf("moe int4 layer policy host tests: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
