#pragma once
// Portable arithmetic contract of the FAST_I4 policies. Ops supplies the lane layout,
// RN32/RN16 boundaries and H128 implementation; the Binding supplies model math/scales.
#if defined(__HIPCC__)
#define STRIX_I4_INLINE __device__ __forceinline__
#else
#define STRIX_I4_INLINE inline
#endif
#include <type_traits>
#include <utility>
namespace strix::core::moe_i4 {
// Optional binding hook: suh_row(expert,m) resolves the expert's input-scale row once (a callable c -> float with the
// values of suh(expert,m,c)), so a lane does not repeat the group lookup and pointer load per channel. Same values.
template<class B,class=void> struct has_suh_row : std::false_type {};
template<class B> struct has_suh_row<B,std::void_t<decltype(std::declval<const B&>().suh_row(0U,0U))>> : std::true_type {};
template<class Binding> struct SuhOf {
    const Binding& b; unsigned e,m;
    STRIX_I4_INLINE float operator()(unsigned c) const { return b.suh(e,m,c); }
};
template<class Binding> STRIX_I4_INLINE auto suh_of(const Binding& b,unsigned e,unsigned m) {
    if constexpr (has_suh_row<Binding>::value) return b.suh_row(e,m);
    else return SuhOf<Binding>{b,e,m};
}
template<class Ops,class Binding,unsigned N>
STRIX_I4_INLINE void input_values(const Binding& b,unsigned expert,unsigned pair,unsigned m,unsigned col,float (&v)[N],const float* cached=nullptr) {
    const auto suh=suh_of(b,expert,m);
    float s[N];   // all scale loads first: none waits behind another value's arithmetic
    for(unsigned j=0;j<N;++j) s[j]=suh(col+Ops::channel(j));
    for(unsigned j=0;j<N;++j) {
        const unsigned c=col+Ops::channel(j);
        const float x=cached ? cached[j] : b.x[static_cast<unsigned long long>(b.token(pair))*Binding::H+c];
        v[j]=Ops::round16(Ops::materialize(x*s[j]));
    }
    Ops::transform(v);
    for(unsigned j=0;j<N;++j) v[j]=Ops::round16(v[j]);
}
template<class Ops,class Binding,unsigned N>
STRIX_I4_INLINE void activation_values(const Binding& b,unsigned expert,unsigned col,const float (&g)[N],const float (&u)[N],float (&v)[N]) {
    const auto suh=suh_of(b,expert,2);
    float s[N];
    for(unsigned j=0;j<N;++j) s[j]=suh(col+Ops::channel(j));
    for(unsigned j=0;j<N;++j) v[j]=Ops::round16(Ops::materialize(b.activation(g[j],u[j])*s[j]));
    Ops::transform(v);
    for(unsigned j=0;j<N;++j) v[j]=Ops::round16(v[j]);
}
} // namespace strix::core::moe_i4
#undef STRIX_I4_INLINE
