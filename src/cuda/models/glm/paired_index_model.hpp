#pragma once
// Independent IEEE approximation of a two-product FP16 / FP32-accumulator dot.
// Inputs must already be exactly widened binary16. No AMD truncation model.
#if defined(__CUDACC__)
#define STRIX_PAIR_HD __host__ __device__
#else
#define STRIX_PAIR_HD
#endif
namespace strix::glm::rtxpf {
STRIX_PAIR_HD inline float paired_half_model(float a,float b,float c,float d,float acc) {
    // Half products are exact in double. Recover cancellation lost while
    // adding the products before adding the accumulator (TwoSum).
    double p=double(a)*double(b),q=double(c)*double(d),s=p+q;
    double z=s-p,err=(p-(s-z))+(q-z);
    return float((s+double(acc))+err);
}
}
#undef STRIX_PAIR_HD
