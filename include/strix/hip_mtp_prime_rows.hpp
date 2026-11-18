#pragma once
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif
#include <cstdint>
#include <span>
#include "strix/hip_mtp_input_fusion.hpp"

namespace strix {
struct MtpNativeContext;
// Priming-only views: decode/draft eight-row structs and entry points retain their ABI.
inline constexpr std::uint32_t hip_mtp_prime_rows_max=128U;
struct HipMtpPrimeStride {
    std::uintptr_t base=0;
    std::uint64_t stride=0;
#if defined(__HIPCC__)
    __host__ __device__ __forceinline__
#else
    inline
#endif
    std::uintptr_t operator[](std::uint64_t row) const {return base+row*stride;}
};
struct HipMtpEmbeddingPrimeRows {
    std::uint32_t value[hip_mtp_prime_rows_max]{};
    HipMtpPrimeStride output;
};
struct HipMtpInputFusionPrimeRows {
    std::uintptr_t wide_hidden[hip_mtp_prime_rows_max]{};
    HipMtpPrimeStride embedding,normalized,output;
};
struct HipMtpHyperPrimeRows {HipMtpPrimeStride input,output,workspace,injection;};
struct HipMtpCachePrimeRows {HipMtpPrimeStride input,index_key,key,value;};
static_assert(sizeof(HipMtpPrimeStride)==16U);
static_assert(sizeof(HipMtpEmbeddingPrimeRows)==528U);
static_assert(sizeof(HipMtpInputFusionPrimeRows)==1072U);
static_assert(sizeof(HipMtpHyperPrimeRows)==64U && sizeof(HipMtpCachePrimeRows)==64U);
// Shared kernel instantiations use these strided scratch views; hidden inputs remain arbitrary pointers.
void launch_hip_mtp_embedding_prime_rows(const MtpNativeContext&,std::span<const std::uint32_t>,HipMtpPrimeStride,void*);
void launch_hip_mtp_input_fusion_prime_rows(std::span<const HipMtpInputFusionRequest>,void*);
void enqueue_hip_mtp_hyper_mix_q8_prime_rows(const HipMtpHyperPrimeRows&,std::uint32_t,const float*,const void*,
    const float*,const void*,float,void*);
void launch_hip_mtp_cache_prime_rows(const MtpNativeContext&,const HipMtpCachePrimeRows&,std::uint32_t,void*);
} // namespace strix
