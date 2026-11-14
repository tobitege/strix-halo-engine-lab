#include "strix/hip_qwen_dpp_reduce.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned stride = 128, blocks = 256;

void check(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
struct DeviceMemory {
    void* data = nullptr;
    explicit DeviceMemory(std::size_t bytes) { check(hipMalloc(&data, bytes)); }
    ~DeviceMemory() { if (data) static_cast<void>(hipFree(data)); }
};

// Pin operand order for NaN payloads as well as signed zero. The test compares
// raw exchange bits separately, then each step of the actual reduction trees.
__device__ __forceinline__ float add(float a, float b) {
    float out;
    asm volatile("v_add_f32 %0, %1, %2" : "=v"(out) : "v"(a), "v"(b));
    return out;
}
__device__ __forceinline__ unsigned different(float a, float b) {
    return __float_as_uint(a) != __float_as_uint(b);
}
template <unsigned D>
__device__ unsigned exchanges(unsigned bits, unsigned lane) {
    namespace dpp = strix::qwen_dpp;
    const float value = __uint_as_float(bits);
    unsigned bad = different(dpp::xor32<D>(value), __shfl_xor(value, D, 32));
    bad |= static_cast<unsigned>(dpp::xor32<D>(bits) != __shfl_xor(bits, D, 32)) << 1U;
    const int signed_bits = __float_as_int(value);
    bad |= static_cast<unsigned>(dpp::xor32<D>(signed_bits) != __shfl_xor(signed_bits, D, 32)) << 2U;
    const double wide = __longlong_as_double(static_cast<long long>(
        (std::uint64_t{bits} << 32U) | (bits ^ 0x12345678U)));
    bad |= static_cast<unsigned>(__double_as_longlong(dpp::xor32<D>(wide)) !=
        __double_as_longlong(__shfl_xor(wide, D, 32))) << 8U;
    // Execute with the original EXEC before masking consumers. Otherwise the
    // fixture itself would mask the source lanes and test a different exchange.
    const float down = dpp::down_tree32<D>(value);
    const float old_down = __shfl_down(value, D, 32);
    const unsigned down_u = dpp::down_tree32<D>(bits);
    const unsigned old_down_u = __shfl_down(bits, D, 32);
    if (lane < D) {
        bad |= different(down, old_down) << 3U;
        bad |= static_cast<unsigned>(down_u != old_down_u) << 4U;
    }
    return bad;
}

__global__ void compare(const unsigned* input, const std::uint64_t* masks, unsigned* failures) {
    const unsigned at = blockIdx.x * stride + threadIdx.x;
    const unsigned physical_lane = threadIdx.x % static_cast<unsigned>(warpSize);
    const unsigned lane = threadIdx.x & 31U;
    if ((masks[blockIdx.x] & (std::uint64_t{1} << physical_lane)) == 0) return;
    const unsigned bits = input[at];
    unsigned bad = exchanges<1>(bits, lane) | exchanges<2>(bits, lane) |
        exchanges<4>(bits, lane) | exchanges<8>(bits, lane) | exchanges<16>(bits, lane);
    const float value = __uint_as_float(bits);

    // Descending full-wave max/sum and bounded integer sum (quantizers/router).
    float sum = value, ref_sum = value, maximum = value, ref_maximum = value;
    float plain = value, ref_plain = value;
    int total = static_cast<int>(bits & 255U) - 128, ref_total = total;
    for (unsigned d = 16; d; d /= 2) {
        sum = add(sum, strix::qwen_dpp::xor32(sum, d));
        ref_sum = add(ref_sum, __shfl_xor(ref_sum, static_cast<int>(d), 32));
        plain += strix::qwen_dpp::xor32(plain, d);
        ref_plain += __shfl_xor(ref_plain, static_cast<int>(d), 32);
        maximum = fmaxf(maximum, strix::qwen_dpp::xor32(maximum, d));
        ref_maximum = fmaxf(ref_maximum, __shfl_xor(ref_maximum, static_cast<int>(d), 32));
        total += strix::qwen_dpp::xor32(total, d);
        ref_total += __shfl_xor(ref_total, static_cast<int>(d), 32);
        bad |= (different(sum, ref_sum) | different(plain, ref_plain) | different(maximum, ref_maximum) |
            static_cast<unsigned>(total != ref_total)) << 5U;
    }
    // Ascending MoE 1/2/4, including the 2/4-only pair epilogue.
    sum = value; ref_sum = value;
    plain = value; ref_plain = value;
    float pair = value, ref_pair = value;
    for (unsigned d = 1; d <= 4; d *= 2) {
        sum = add(sum, strix::qwen_dpp::xor32(sum, d));
        ref_sum = add(ref_sum, __shfl_xor(ref_sum, static_cast<int>(d), 32));
        plain += strix::qwen_dpp::xor32(plain, d);
        ref_plain += __shfl_xor(ref_plain, static_cast<int>(d), 32);
        if (d != 1) {
            pair = add(pair, strix::qwen_dpp::xor32(pair, d));
            ref_pair = add(ref_pair, __shfl_xor(ref_pair, static_cast<int>(d), 32));
        }
        bad |= (different(sum, ref_sum) | different(plain, ref_plain) | different(pair, ref_pair)) << 6U;
    }
    // Half-row maxima (MoE FP16 epilogues), integer min/max (QSA selection)
    // and the ascending integer pair tree (MoE dot products).
    maximum = value; ref_maximum = value;
    unsigned minimum_u = bits, ref_minimum_u = bits, maximum_u = bits, ref_maximum_u = bits;
    for (unsigned d = 8; d; d /= 2) {
        maximum = fmaxf(maximum, strix::qwen_dpp::xor32(maximum, d));
        ref_maximum = fmaxf(ref_maximum, __shfl_xor(ref_maximum, static_cast<int>(d), 32));
        bad |= different(maximum, ref_maximum) << 9U;
    }
    for (unsigned d = 16; d; d /= 2) {
        minimum_u = min(minimum_u, strix::qwen_dpp::xor32(minimum_u, d));
        ref_minimum_u = min(ref_minimum_u, __shfl_xor(ref_minimum_u, static_cast<int>(d), 32));
        maximum_u = max(maximum_u, strix::qwen_dpp::xor32(maximum_u, d));
        ref_maximum_u = max(ref_maximum_u, __shfl_xor(ref_maximum_u, static_cast<int>(d), 32));
        bad |= static_cast<unsigned>(minimum_u != ref_minimum_u || maximum_u != ref_maximum_u) << 10U;
    }
    total = static_cast<int>(bits & 255U) - 128; ref_total = total;
    for (unsigned d = 1; d <= 4; d *= 2) {
        total += strix::qwen_dpp::xor32(total, d);
        ref_total += __shfl_xor(ref_total, static_cast<int>(d), 32);
        bad |= static_cast<unsigned>(total != ref_total) << 11U;
    }
    // Guarded virtual-leaf projection trees: compare EVERY live lane at every
    // stage, not just lane zero. Reverse operand order covers QSA/HC too.
    sum = value; ref_sum = value;
    plain = value; ref_plain = value;
    float reverse = value, ref_reverse = value;
    for (unsigned d = 16; d; d /= 2) {
        const float peer = strix::qwen_dpp::down_tree32(sum, d);
        const float old_peer = __shfl_down(ref_sum, d, 32);
        const float peer_reverse = strix::qwen_dpp::down_tree32(reverse, d);
        const float old_peer_reverse = __shfl_down(ref_reverse, d, 32);
        const float plain_peer = strix::qwen_dpp::down_tree32(plain, d);
        const float old_plain_peer = __shfl_down(ref_plain, d, 32);
        if (lane < d) {
            sum = add(sum, peer); ref_sum = add(ref_sum, old_peer);
            reverse = add(peer_reverse, reverse); ref_reverse = add(old_peer_reverse, ref_reverse);
            plain += plain_peer; ref_plain += old_plain_peer;
            bad |= (different(sum, ref_sum) | different(plain, ref_plain) | different(reverse, ref_reverse)) << 7U;
        }
    }
    failures[at] = bad;
}

unsigned random_bits(unsigned& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
} // namespace

int main() try {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) return 77;
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0));
    // The helper deliberately retains HIP shuffles on unqualified ISAs.
    if (std::string(properties.gcnArchName).find("gfx1151") != 0) return 77;
    constexpr std::array<unsigned, 16> specials{0U, 0x80000000U, 0x7f800000U, 0xff800000U,
        0x7fc00001U, 0x7fc12345U, 0xffc54321U, 0x7f800001U,
        0xff800001U, 1U, 0x80000001U, 0x007fffffU, 0x00800000U,
        0x7f7fffffU, 0xff7fffffU, 0x3f800000U};
    std::vector<unsigned> input(blocks * stride), failures(input.size());
    std::vector<std::uint64_t> masks(blocks);
    unsigned rng = 0x58d001U;
    for (unsigned b = 0; b < blocks; ++b) {
        // Full/prefix masks, either logical half of wave64, single holes,
        // checkerboards and arbitrary divergence. Actual block tails are
        // independent of this mask and exercised by every launch shape below.
        if (b == 0) masks[b] = ~std::uint64_t{0};
        else if (b < 64) masks[b] = (std::uint64_t{1} << b) - 1;
        else if (b < 128) masks[b] = ~(std::uint64_t{1} << (b - 64));
        else if (b == 128) masks[b] = 0xffffffff00000000ULL;
        else if (b == 129) masks[b] = 0xaaaaaaaaaaaaaaaaULL;
        else if (b == 130) masks[b] = 0x5555555555555555ULL;
        else masks[b] = (std::uint64_t{random_bits(rng)} << 32) | random_bits(rng);
        for (unsigned l = 0; l < stride; ++l)
            input[b * stride + l] = (b % 4 == 0) ? specials[(b / 4 + l) % specials.size()] : random_bits(rng);
    }
    DeviceMemory data(input.size() * sizeof(unsigned)), mask_data(masks.size() * sizeof(std::uint64_t)),
        result(failures.size() * sizeof(unsigned));
    check(hipMemcpy(data.data, input.data(), input.size() * sizeof(unsigned), hipMemcpyHostToDevice));
    check(hipMemcpy(mask_data.data, masks.data(), masks.size() * sizeof(std::uint64_t), hipMemcpyHostToDevice));
    for (unsigned threads : {1U, 7U, 15U, 16U, 17U, 31U, 32U, 33U, 47U, 63U, 64U, 65U, 96U, 127U, 128U}) {
        check(hipMemset(result.data, 0, failures.size() * sizeof(unsigned)));
        hipLaunchKernelGGL(compare, dim3(blocks), dim3(threads), 0, nullptr,
            static_cast<const unsigned*>(data.data), static_cast<const std::uint64_t*>(mask_data.data),
            static_cast<unsigned*>(result.data));
        check(hipGetLastError());
        check(hipMemcpy(failures.data(), result.data, failures.size() * sizeof(unsigned), hipMemcpyDeviceToHost));
        for (unsigned i = 0; i < failures.size(); ++i) if (failures[i]) {
            std::fprintf(stderr, "DPP mismatch threads=%u block=%u lane=%u bits=%08x checks=%02x\n",
                threads, i / stride, i % stride, input[i], failures[i]);
            return 1;
        }
    }
    std::puts("Qwen DPP exchanges and ordered reductions: byte-equal");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
