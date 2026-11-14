#pragma once
// Shared greedy argmax core (R2-07). Every engine's argmax/greedy kernel uses
// these primitives; engines only adapt payload layout and output format.
//
// Ordering contract (first-max): a candidate wins when its numeric key is
// larger, or when keys are equal and its token id is lower. The integer key
// implements the CPU finite numeric comparison exactly (subnormals included,
// independent of the device FP denormal mode); both zeros map to one key.
// Non-finite logits never win: they set `bad` and are not `valid`.
#include <cstdint>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>

namespace strix::argmax {

struct Partial {
    float value;
    std::uint32_t token, bad, valid;
};
static_assert(sizeof(Partial) == 16);

__device__ __forceinline__ std::uint32_t key(float value) {
    auto bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) == 0) bits = 0;
    return (bits & 0x80000000U) ? ~bits : (bits ^ 0x80000000U);
}

__device__ __forceinline__ bool nonfinite(float value) {
    return (__float_as_uint(value) & 0x7f800000U) == 0x7f800000U;
}

// True when (value_b, token_b) beats (value_a, token_a) under first-max.
// Callers must only pass finite values (or values already known valid).
__device__ __forceinline__ bool better(float value_b, std::uint32_t token_b,
                                       float value_a, std::uint32_t token_a) {
    const auto kb = key(value_b);
    const auto ka = key(value_a);
    return kb > ka || (kb == ka && token_b < token_a);
}

__device__ __forceinline__ Partial empty() { return Partial{0.0F, 0, 0, 0}; }

__device__ __forceinline__ Partial load(float value, std::uint32_t token) {
    const auto bad = nonfinite(value) ? 1U : 0U;
    return Partial{value, token, bad, bad ? 0U : 1U};
}

// Works for any payload exposing value/token/bad/valid (extra fields ride
// along with the winner).
template <class T>
__device__ __forceinline__ T choose(T a, const T& b) {
    const auto bad = a.bad | b.bad;
    if (b.valid && (!a.valid || better(b.value, b.token, a.value, a.token))) a = b;
    a.bad = bad;
    return a;
}

// 0 = ok, bit0 = a non-finite logit was seen, bit1 = no valid candidate.
template <class T>
__device__ __forceinline__ std::uint32_t status(const T& best) {
    return best.bad | (best.valid ? 0U : 2U);
}

// Shared-memory tree reduction over `Threads` lanes (power of two). Every lane
// must call it; on return values[0] holds the reduction of all lanes.
template <unsigned Threads, class T, class Combine>
__device__ __forceinline__ T block_reduce(T* values, unsigned lane, T value, Combine combine) {
    static_assert(Threads && (Threads & (Threads - 1U)) == 0, "power of two");
    values[lane] = value;
    __syncthreads();
    for (unsigned stride = Threads / 2; stride; stride >>= 1U) {
        if (lane < stride) values[lane] = combine(values[lane], values[lane + stride]);
        __syncthreads();
    }
    return values[0];
}

template <unsigned Threads, class T>
__device__ __forceinline__ T block_argmax(T* values, unsigned lane, T value) {
    return block_reduce<Threads>(values, lane, value,
                                 [](const T& a, const T& b) { return choose(a, b); });
}

// Strided fold of `count` partials by one block, then block reduction.
template <unsigned Threads, class T>
__device__ __forceinline__ T block_argmax_partials(T* values, unsigned lane, const T* partials,
                                                   std::uint64_t count, T init) {
    T value = init;
    for (std::uint64_t i = lane; i < count; i += Threads) value = choose(value, partials[i]);
    return block_argmax<Threads>(values, lane, value);
}

}  // namespace strix::argmax
#endif
