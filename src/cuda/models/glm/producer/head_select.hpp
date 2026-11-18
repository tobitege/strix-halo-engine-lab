#pragma once
// Deterministic greedy selection shared by the device argmax and host checks.
// Order: logit descending, then vocabulary ID ascending. Numeric equality
// (so +0 == -0) is a tie; nonfinite logits never win and reject the token.
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#ifdef __CUDACC__
#define GLM_HOST_DEVICE __host__ __device__
#else
#define GLM_HOST_DEVICE
#endif
namespace strix::glm::rtxpf::producer {
inline constexpr std::uint32_t no_token=0xffffffffu;
GLM_HOST_DEVICE inline bool head_better(float a,std::uint32_t ia,float b,std::uint32_t ib){
    if(ib==no_token)return ia!=no_token;if(ia==no_token)return false;
    return a>b||(a==b&&ia<ib);
}
struct HeadChoice {std::uint32_t token=no_token;std::uint32_t nonfinite=0;float logit=0;};
inline HeadChoice head_argmax(std::span<const float> logits){
    HeadChoice c;
    for(std::size_t j=0;j<logits.size();++j){float v=logits[j];if(!std::isfinite(v)){++c.nonfinite;continue;}if(head_better(v,std::uint32_t(j),c.logit,c.token)){c.logit=v;c.token=std::uint32_t(j);}}
    return c;
}
// The only accepted outcome: a finite winner and no nonfinite logits.
inline std::uint32_t accept_choice(const HeadChoice& c){
    if(c.nonfinite)throw std::runtime_error("nonfinite logits; token rejected");
    if(c.token==no_token)throw std::runtime_error("empty logits; token rejected");
    return c.token;
}
}
