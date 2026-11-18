// CPU backend scaffold: the reference and the selected table against the existing host references.
//  - quantize_a8 and gemv_q8_0_a8 against strix::ds41::quantize_a8 / product_q8_0 (bitwise)
//  - rmsnorm through the shared strix::ops::norm::Norm<> interface against ops::norm::reference (bitwise)
#include "cpu/core/backend.hpp"
#include "ops/norm/reference.hpp"
#include "strix/ds41_formats.hpp"

#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace {
unsigned failures = 0, checks = 0;
void expect(bool ok, const char* what, std::size_t i) {
    ++checks;
    if (!ok && failures++ < 10) std::printf("FAIL %s at %zu\n", what, i);
}
bool same(float a, float b) { return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b); }

void test_retained_selection() {
    const bool owns_table = !std::is_reference_v<decltype(strix::cpu::select(strix::cpu::Caps{}))>;
    expect(owns_table, "selection returns an owned table", 0);
    if (!owns_table) return;  // Fail the old API without deliberately invoking its data race.
    const auto capabilities = strix::cpu::caps();
    const auto& retained = strix::cpu::select(capabilities);
    std::atomic<bool> started{false};
    bool worker_ok = true;
    std::thread worker([&] {
        std::array<std::uint8_t, 34> weight{};
        weight[1] = 0x3c;  // half 1
        for (std::size_t j = 2; j < weight.size(); ++j) weight[j] = 1;
        std::array<std::int8_t, 32> codes{};
        codes.fill(1);
        const float delta = 1.0F;
        float y = 0.0F;
        const strix::cpu::Q8a8View v{weight.data(), codes.data(), &delta, &y, 34, 32, 1, 1, 32, 1, 0};
        started.store(true, std::memory_order_release);
        for (unsigned i = 0; i < 4096; ++i) {
            retained.gemv_q8_0_a8(v, {});
            worker_ok = worker_ok && same(y, 32.0F);
        }
    });
    while (!started.load(std::memory_order_acquire)) std::this_thread::yield();
    const strix::cpu::Stream stream{};
    const float x = 1.0F;
    float y = 0.0F;
    for (unsigned i = 0; i < 4096; ++i) {
        const auto next = strix::cpu::select(i % 2U ? strix::cpu::caps() : strix::cpu::Caps{});
        expect(next.gemv_q8_0_a8 != nullptr, "reselection", i);
        strix::cpu::Norm::run(&x, nullptr, 0, 1, 1e-6F, &y, 1, &stream);
    }
    worker.join();
    expect(worker_ok, "retained table concurrent selection and norm", 0);
}

void test_invalid_widths(const strix::cpu::Kernels& k) {
    for (const std::uint32_t width : {31U, 33U}) {
        std::array<float, 64> x{};
        x.fill(1.0F);
        std::array<std::int8_t, 64> codes{};
        codes.fill(17);
        std::array<float, 2> deltas{19.0F, 19.0F};
        std::array<std::int32_t, 2> sums{23, 23};
        bool rejected = false;
        try { k.quantize_a8(x.data(), width, codes.data(), deltas.data(), sums.data()); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "a8 rejects partial group", width);
        for (const auto code : codes) expect(code == 17, "rejected a8 codes untouched", width);
        for (const auto delta : deltas) expect(same(delta, 19.0F), "rejected a8 deltas untouched", width);
        for (const auto sum : sums) expect(sum == 23, "rejected a8 sums untouched", width);
        std::array<std::uint8_t, 68> weight{};
        for (const std::uint32_t accumulate : {0U, 1U}) {
            float y = 29.0F;
            const strix::cpu::Q8a8View v{weight.data(), codes.data(), deltas.data(), &y,
                                        68, width, 1, 1, 64, 1, accumulate};
            rejected = false;
            try { k.gemv_q8_0_a8(v, {}); }
            catch (const std::invalid_argument&) { rejected = true; }
            expect(rejected, "gemv rejects partial group", width);
            expect(same(y, 29.0F), "rejected gemv output untouched", width);
        }
    }
}

void test_product_order(const strix::cpu::Kernels& k) {
    // Independent powers-of-two fixture: lane 0 rounds (2^24 + 1) before cancelling
    // lane 16; lane 1 then leaves 1. Seeding a lane with the existing output is wrong.
    std::array<std::uint8_t, 33 * 34> weight{};
    std::array<std::int8_t, 33 * 32> codes{};
    codes.fill(1);
    std::array<float, 33> deltas{};
    for (std::size_t g = 0; g < 33; ++g) {
        weight[g * 34 + 1] = 0x3c;
        weight[g * 34 + 2] = 1;
    }
    deltas[0] = 0x1p24F;
    deltas[16] = -0x1p24F;
    deltas[1] = deltas[32] = 1.0F;
    float y = -1.0F;
    strix::cpu::Q8a8View v{weight.data(), codes.data(), deltas.data(), &y,
                          33 * 34, 33 * 32, 1, 1, 33 * 32, 1, 0};
    k.gemv_q8_0_a8(v, {});
    expect(same(y, 1.0F), "group and butterfly rounding order", 0);
    y = -1.0F;
    v.accumulate = 1;
    k.gemv_q8_0_a8(v, {});
    expect(same(y, 0.0F), "accumulate after cancellation", 0);
}

void test_padded_tiny_subnormal(const strix::cpu::Kernels& k) {
    // Two rows, one group each; odd weight stride also exercises unaligned scales.
    std::array<std::uint8_t, 70> weight{};
    weight[0] = 1;  // smallest positive half subnormal: 2^-24
    weight[35] = 0xff; weight[36] = 0x83;  // largest negative subnormal: -1023 * 2^-24
    for (std::size_t j = 0; j < 32; ++j) weight[2 + j] = weight[37 + j] = 1;
    std::array<float, 32> x{};
    std::array<std::int8_t, 128> codes{};
    codes.fill(17);
    std::array<float, 4> deltas{19.0F, 19.0F, 19.0F, 19.0F};
    std::array<std::int32_t, 4> sums{23, 23, 23, 23};
    for (std::size_t r = 0; r < 2; ++r) {
        x.fill(r == 0 ? 0x1p-120F : 127.0F);
        k.quantize_a8(x.data(), 32, codes.data() + r * 64, deltas.data() + r * 2, sums.data() + r * 2);
        expect(same(deltas[r * 2], 1.0F), "tiny/normal delta", r);
        expect(sums[r * 2] == (r == 0 ? 0 : 4064), "tiny/normal sum", r);
        for (std::size_t j = 0; j < 32; ++j) {
            expect(codes[r * 64 + j] == (r == 0 ? 0 : 127), "tiny/normal code", j);
            expect(codes[r * 64 + 32 + j] == 17, "code padding", j);
        }
        expect(same(deltas[r * 2 + 1], 19.0F) && sums[r * 2 + 1] == 23, "scale/sum padding", r);
    }
    std::array<float, 6> y{};
    y.fill(29.0F);
    const strix::cpu::Q8a8View v{weight.data(), codes.data(), deltas.data(), y.data(), 35, 32, 2, 2, 64, 3, 0};
    k.gemv_q8_0_a8(v, {});
    expect(same(y[0], 0.0F) && same(y[1], 0.0F), "tiny group product", 0);
    expect(same(y[3], 4064.0F * 0x1p-24F), "positive half subnormal product", 0);
    expect(same(y[4], -4157472.0F * 0x1p-24F), "negative half subnormal product", 0);
    expect(same(y[2], 29.0F) && same(y[5], 29.0F), "output padding", 0);
}

struct HostNormTraits {
    using Input = float;
    using Output = float;
    using Scale = float;
    using ScaleFormat = int;
    using Stream = void*;
    static constexpr float epsilon = 1e-6F;
    static constexpr auto reduction_order = strix::ops::norm::ReductionOrder::lane_strided_fma_lds_halving;
    static constexpr unsigned lanes(unsigned n) { return strix::cpu::NormTraits::lanes(n); }
    static float inv_sqrt(float x) { return 1.0F / std::sqrt(x); }
    static float load(const float* w, int, std::size_t i) { return w[i]; }
};

std::uint16_t float_to_half_bits(float f) {  // fixture helper: normal finite scales only
    const std::uint32_t b = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = (b >> 16U) & 0x8000U;
    const int e = static_cast<int>((b >> 23U) & 0xffU) - 127 + 15;
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(e) << 10U) | ((b >> 13U) & 0x3ffU));
}

void test_q8_0(const strix::cpu::Kernels& k, std::mt19937& rng, std::uint32_t width, std::uint32_t count,
               std::uint32_t rows) {
    namespace d = strix::ds41;
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::uniform_int_distribution<int> code(-128, 127);
    std::uniform_real_distribution<float> sc(1e-3F, 2e-2F);
    const std::uint32_t row_bytes = width / 32U * 34U;
    std::vector<std::uint8_t> w(static_cast<std::size_t>(count) * row_bytes);
    for (std::size_t b = 0; b < w.size() / 34U; ++b) {
        const std::uint16_t h = float_to_half_bits(sc(rng) * (b % 7U == 0 ? -1.0F : 1.0F));
        w[b * 34U] = static_cast<std::uint8_t>(h & 0xffU);
        w[b * 34U + 1] = static_cast<std::uint8_t>(h >> 8U);
        for (std::size_t j = 0; j < 32; ++j) w[b * 34U + 2 + j] = static_cast<std::uint8_t>(code(rng));
    }
    std::vector<float> x(static_cast<std::size_t>(rows) * width);
    for (auto& v : x) v = nd(rng);
    for (std::size_t j = 0; j < 32 && rows > 1; ++j) x[width + j] = 0.0F;  // an all-zero group
    std::vector<std::int8_t> codes(x.size());
    std::vector<float> deltas(x.size() / 32U);
    std::vector<std::int32_t> sums(x.size() / 32U);
    std::vector<d::A8Activation> ref_x;
    for (std::uint32_t r = 0; r < rows; ++r) {
        k.quantize_a8(x.data() + static_cast<std::size_t>(r) * width, width, codes.data() + static_cast<std::size_t>(r) * width,
                      deltas.data() + static_cast<std::size_t>(r) * width / 32U, sums.data() + static_cast<std::size_t>(r) * width / 32U);
        ref_x.push_back(d::quantize_a8(std::span<const float>(x.data() + static_cast<std::size_t>(r) * width, width)));
        for (std::size_t i = 0; i < width; ++i) expect(codes[r * width + i] == ref_x[r].codes[i], "a8 code", i);
        for (std::size_t g = 0; g < width / 32U; ++g) {
            expect(same(deltas[r * width / 32U + g], ref_x[r].deltas[g]), "a8 delta", g);
            expect(sums[r * width / 32U + g] == ref_x[r].sums[g], "a8 sum", g);
        }
    }
    std::vector<float> y(static_cast<std::size_t>(rows) * count, 0.0F);
    const strix::cpu::Q8a8View v{w.data(), codes.data(), deltas.data(), y.data(), row_bytes, width, count, rows, width, count, 0};
    k.gemv_q8_0_a8(v, strix::cpu::Stream{});
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t o = 0; o < count; ++o) {
            const float ref = d::product_q8_0(std::span<const std::uint8_t>(w.data() + static_cast<std::size_t>(o) * row_bytes, row_bytes),
                                              ref_x[r], width);
            expect(same(y[static_cast<std::size_t>(r) * count + o], ref), "q8_0 gemv", o);
        }
    // accumulate: out += product, one rounding after the product
    std::vector<float> y2(y.size());
    for (std::size_t i = 0; i < y.size(); ++i)
        y2[i] = i % 3U == 0 ? -y[i] : (i % 3U == 1 ? std::nextafter(-y[i], 1.0F) : nd(rng));
    const auto initial = y2;
    const strix::cpu::Q8a8View va{w.data(), codes.data(), deltas.data(), y2.data(), row_bytes, width, count, rows, width, count, 1};
    k.gemv_q8_0_a8(va, strix::cpu::Stream{});
    for (std::size_t i = 0; i < y.size(); ++i) expect(same(y2[i], initial[i] + y[i]), "q8_0 accumulate", i);
}

void test_norm(std::mt19937& rng, unsigned n, unsigned rows) {
    std::normal_distribution<float> nd(0.0F, 3.0F);
    std::vector<float> x(static_cast<std::size_t>(n) * rows), w(n), y(x.size()), ref(x.size());
    for (auto& v : x) v = nd(rng);
    for (auto& v : w) v = 1.0F + 0.1F * nd(rng);
    const strix::cpu::Stream s{};
    strix::cpu::Norm::run(x.data(), w.data(), 0, n, 1e-6F, y.data(), rows, &s);
    strix::ops::norm::reference<HostNormTraits>(x.data(), w.data(), 0, n, 1e-6F, ref.data(), rows);
    for (std::size_t i = 0; i < y.size(); ++i) expect(same(y[i], ref[i]), "rmsnorm", i);
    auto in_place = x;
    strix::cpu::Norm::run(in_place.data(), w.data(), 0, n, 1e-6F, in_place.data(), rows, &s);
    for (std::size_t i = 0; i < y.size(); ++i) expect(same(in_place[i], ref[i]), "rmsnorm in place", i);
    strix::cpu::Norm::run(x.data(), nullptr, 0, n, 1e-6F, y.data(), rows, &s);
    strix::ops::norm::reference<HostNormTraits>(x.data(), nullptr, 0, n, 1e-6F, ref.data(), rows);
    for (std::size_t i = 0; i < y.size(); ++i) expect(same(y[i], ref[i]), "rmsnorm unscaled", i);
    in_place = x;
    strix::cpu::Norm::run(in_place.data(), nullptr, 0, n, 1e-6F, in_place.data(), rows, &s);
    for (std::size_t i = 0; i < y.size(); ++i) expect(same(in_place[i], ref[i]), "rmsnorm unscaled in place", i);
}
}  // namespace

int main() {
    test_retained_selection();
    std::mt19937 rng(1112U);
    const auto host = strix::cpu::caps();
    const auto& selected = strix::cpu::select(host);
    std::printf("cpu backend: selected table %s (avx2=%d avx512f=%d vnni=%d amx=%d neon=%d sve=%d)\n",
                strix::cpu::isa_name(selected.isa), host.avx2, host.avx512f, host.avx512vnni, host.amx_int8,
                host.neon, host.sve);
    for (const auto* k : {&strix::cpu::reference_kernels(), &selected}) {
        test_invalid_widths(*k);
        test_product_order(*k);
        test_padded_tiny_subnormal(*k);
        test_q8_0(*k, rng, 32, 5, 1);
        test_q8_0(*k, rng, 2048, 64, 3);
        test_q8_0(*k, rng, 7168 + 32, 17, 2);  // groups not a multiple of the lane count
    }
    test_norm(rng, 96, 3);
    for (const unsigned n : {1U, 127U, 128U, 129U, 511U, 512U, 513U}) test_norm(rng, n, 2);
    test_norm(rng, 4096, 2);
    test_norm(rng, 7168, 1);
    std::printf("%u checks, %u failures\n", checks, failures);
    return failures ? 1 : 0;
}
