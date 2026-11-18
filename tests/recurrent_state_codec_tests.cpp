// Host tests of the shared recurrent state codec (src/attn_core/recurrent_state.hpp) and storage policy.
// The codec's nearest-even rule is checked against its definition (distance to the neighbouring f16 / bf16
// values in double), not against another implementation.
#include "attn_core/recurrent_state.hpp"
#include "strix/recurrent_state_policy.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#if defined(_MSC_VER)
#include <crtdbg.h>
#endif

namespace {
using namespace strix;
using namespace strix::attn_core::rstate;

int failures = 0;
void expect(bool ok, const char* what, std::uint32_t bits) {
    if (!ok && failures++ < 12) std::printf("FAIL %s at 0x%08x\n", what, bits);
}

double f16_value(std::uint16_t h) { return static_cast<double>(f32_from_f16_host(h)); }
double bf16_value(std::uint16_t h) { return static_cast<double>(f32_from_bf16(h)); }

// x rounds to h: no representable neighbour is nearer, ties go to the even encoding.
template <class Value>
void check_nearest(float x, std::uint16_t h, Value value, std::uint16_t inf_bits, double overflow, const char* what) {
    const auto bits = f32_bits(x);
    const double v = static_cast<double>(x);
    const auto mag = static_cast<std::uint16_t>(h & 0x7fffU);
    if (std::fabs(v) >= overflow) { // at or past the midpoint above max finite (ties go to the even inf)
        expect(mag == inf_bits, what, bits);
        return;
    }
    expect(mag < inf_bits, what, bits);
    const double y = value(h);
    expect((y < 0) == (v < 0) || y == 0.0, what, bits);
    const double err = std::fabs(v - y);
    for (const int step : {-1, 1}) {
        const int n = static_cast<int>(mag) + step;
        if (n < 0 || n >= inf_bits) continue;
        const auto other = static_cast<std::uint16_t>((h & 0x8000U) | static_cast<unsigned>(n));
        const double e2 = std::fabs(v - value(other));
        expect(err < e2 || (err == e2 && (mag & 1U) == 0U), what, bits);
    }
}

void codec_tests() {
    // f16 decode/encode round trip over every encoding.
    for (std::uint32_t h = 0; h < 0x10000U; ++h) {
        const auto half = static_cast<std::uint16_t>(h);
        const float x = f32_from_f16_host(half);
        const bool nan = (h & 0x7c00U) == 0x7c00U && (h & 0x3ffU) != 0U;
        expect(nan ? f16_from_f32_host(x) == static_cast<std::uint16_t>((h & 0x8000U) | 0x7e00U)
                   : f16_from_f32_host(x) == half, "f16 round trip", h);
        const float b = f32_from_bf16(half);
        const bool bnan = (h & 0x7f80U) == 0x7f80U && (h & 0x7fU) != 0U;
        expect(bnan ? bf16_from_f32(b) == static_cast<std::uint16_t>((h & 0x8000U) | 0x7fc0U) : bf16_from_f32(b) == half,
            "bf16 round trip", h);
    }
    // Nearest-even sweep over float bit patterns (stride prime, both signs) plus the f16 boundaries.
    std::uint32_t bits = 1U;
    for (std::uint32_t i = 0; i < 2000000U; ++i, bits += 2147U) {
        const float x = f32_from_bits(bits);
        if (std::isnan(x)) {
            expect(f16_from_f32_host(x) == static_cast<std::uint16_t>((bits >> 16 & 0x8000U) | 0x7e00U), "f16 nan", bits);
            expect(bf16_from_f32(x) == static_cast<std::uint16_t>((bits >> 16 & 0x8000U) | 0x7fc0U), "bf16 nan", bits);
            continue;
        }
        if (std::isinf(x)) continue;
        check_nearest(x, f16_from_f32_host(x), f16_value, 0x7c00U, 65520.0, "f16 nearest");
        check_nearest(x, bf16_from_f32(x), bf16_value, 0x7f80U, 3.3961775292304173e38, "bf16 nearest");
    }
    for (const float x : {65504.0F, 65519.99F, 65520.0F, 6.103515625e-5F, 6.0975552e-5F, 5.9604645e-8F, 2.9802322e-8F,
                          2.9802326e-8F, 8.940697e-8F, -0.0F, 1.0F, -1.0F, 0.33333334F}) {
        const auto h = f16_from_f32_host(x);
        if (std::fabs(x) >= 65520.0F) expect((h & 0x7fffU) == 0x7c00U, "f16 overflow", f32_bits(x));
        else check_nearest(x, h, f16_value, 0x7c00U, 65520.0, "f16 boundary");
    }
    expect(f16_from_f32_host(2.9802322e-8F) == 0U, "2^-25 tie to zero", 0U);
    expect(f16_from_f32_host(8.940697e-8F) == 2U, "1.5 x 2^-24 tie to even", 0U);
#if defined(__FLT16_MAX__) && !defined(_MSC_VER)
    // Where the host compiler has a native f16, it must agree bit for bit (finite values).
    bits = 7U;
    for (std::uint32_t i = 0; i < 1000000U; ++i, bits += 4099U) {
        const float x = f32_from_bits(bits);
        if (std::isnan(x)) continue;
        const _Float16 native = static_cast<_Float16>(x);
        std::uint16_t nb;
        std::memcpy(&nb, &native, 2);
        expect(nb == f16_from_f32_host(x), "native _Float16 agreement", bits);
    }
#endif
}

void policy_tests() {
    expect(recurrent_state_policy_from(nullptr).type == RecurrentStateType::f32, "default f32", 0U);
    expect(!recurrent_state_policy_from("").fast, "empty f32", 0U);
    const auto f16 = recurrent_state_policy_from("f16");
    expect(f16.type == RecurrentStateType::f16 && f16.fast, "f16 fast", 0U);
    expect(recurrent_state_policy_from("bf16").type == RecurrentStateType::bf16, "bf16", 0U);
    bool threw = false;
    try { (void)recurrent_state_policy_from("fp16"); } catch (const std::exception&) { threw = true; }
    expect(threw, "unknown value rejected", 0U);
    expect(recurrent_state_bytes(48U * 128U * 128U, RecurrentStateType::f16) == 1572864U, "f16 layer bytes", 0U);
    expect(recurrent_state_bytes(48U * 128U * 128U, RecurrentStateType::f32) == 3145728U, "f32 layer bytes", 0U);
    expect(recurrent_state_route_refusal(RecurrentStateType::f32, false, "x").empty(), "f32 admitted anywhere", 0U);
    expect(!recurrent_state_route_refusal(RecurrentStateType::bf16, false, "x").empty(), "untyped route refused", 0U);
    expect(std::string(recurrent_state_contract(RecurrentStateType::f16)) != recurrent_state_contract(RecurrentStateType::bf16),
        "distinct contract tags", 0U);
}
}  // namespace

int main() {
#if defined(_MSC_VER)
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    codec_tests();
    policy_tests();
    if (failures) { std::printf("%d failures\n", failures); return 1; }
    std::printf("recurrent state codec: all passed\n");
    return 0;
}
