#include "strix/cuda/core/nvfp4.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace strix::cuda::core::nvfp4 {
float e2m1(std::uint8_t c) {
    constexpr float v[] = {0,.5f,1,1.5f,2,3,4,6};
    return (c & 8) ? -v[c & 7] : v[c & 7];
}
float e4m3(std::uint8_t c) {
    if (c > 126) return std::numeric_limits<float>::quiet_NaN();
    const int e = c >> 3, m = c & 7;
    return e ? std::ldexp(1.f + float(m) / 8, e - 7) : std::ldexp(float(m), -9);
}
std::uint8_t encode_e2m1(float x) {
    if (!std::isfinite(x)) throw std::invalid_argument("nonfinite FP4 input");
    const float a = std::min(std::fabs(x), 6.f);
    unsigned best = 0;
    float d = a;
    for (unsigned i=1; i<8; ++i) {
        const float t=std::fabs(a-e2m1(std::uint8_t(i)));
        if (t<d || (t==d && !(i&1))) { best=i; d=t; }
    }
    return std::uint8_t(best | (std::signbit(x) ? 8 : 0));
}
std::uint8_t encode_e4m3(float x) {
    if (!std::isfinite(x) || x<0) throw std::invalid_argument("invalid block scale");
    unsigned best=0; float d=x;
    for (unsigned i=1;i<127;++i) {
        const float t=std::fabs(x-e4m3(std::uint8_t(i)));
        if(t<d || (t==d && !(i&1))) {best=i;d=t;}
    }
    return std::uint8_t(best);
}
bool valid(const View& v) {
    const auto matrix_ok=[&](const Matrix& a, std::uint32_t rows) {
        return a.codes && a.scales && a.code_stride>=v.k/2 && a.scale_stride>=v.k/16 &&
            a.code_stride%4==0 && a.scale_stride%4==0 &&
            reinterpret_cast<std::uintptr_t>(a.codes)%4==0 &&
            reinterpret_cast<std::uintptr_t>(a.scales)%4==0 &&
            std::isfinite(a.tensor_scale) && a.tensor_scale>0 &&
            a.code_stride <= std::numeric_limits<std::size_t>::max()/rows &&
            a.scale_stride <= std::numeric_limits<std::size_t>::max()/rows;
    };
    return v.m && v.n && v.k && v.k%64==0 && v.output &&
        reinterpret_cast<std::uintptr_t>(v.output)%alignof(float)==0 &&
        (!v.residual || reinterpret_cast<std::uintptr_t>(v.residual)%alignof(float)==0) &&
        v.output_stride>=v.n && (!v.residual || v.residual_stride>=v.n) &&
        v.output_stride<=std::numeric_limits<std::size_t>::max()/v.m/sizeof(float) &&
        (!v.residual || v.residual_stride<=std::numeric_limits<std::size_t>::max()/v.m/sizeof(float)) &&
        matrix_ok(v.a,v.m) && matrix_ok(v.b,v.n);
}
Plan resolve(const View& v) {
    if (!valid(v)) throw std::invalid_argument("invalid NVFP4 dense view");
    Plan p{v.m,v.n,v.k,1,v.m<=8?Route::decode:Route::gemm,0};
    if(p.route==Route::decode) p.splits=std::min(16u,v.k/64);
    if(p.splits>1) {
        const auto max=std::numeric_limits<std::size_t>::max();
        if(std::size_t(v.n)>max/v.m/p.splits/sizeof(float))
            throw std::invalid_argument("NVFP4 workspace overflow");
        p.workspace_bytes=std::size_t(v.m)*v.n*p.splits*sizeof(float);
    }
    return p;
}
void quantize(const float* x,std::size_t xs,std::uint32_t rows,std::uint32_t k,
              std::uint8_t* q,std::size_t qs,std::uint8_t* s,std::size_t ss,float ts) {
    const auto max=std::numeric_limits<std::size_t>::max();
    if(!x || !q || !s || !rows || !k || k%16 || xs<k || qs<k/2 || ss<k/16 ||
       !std::isfinite(ts) || ts<=0 || xs>max/rows/sizeof(float) || qs>max/rows || ss>max/rows)
        throw std::invalid_argument("invalid quantize view");
    for(unsigned r=0;r<rows;++r) for(unsigned b=0;b<k/16;++b) {
        float amax=0;
        for(unsigned j=0;j<16;++j) {
            const float z=x[r*xs+b*16+j];
            if(!std::isfinite(z)) throw std::invalid_argument("nonfinite quantize input");
            amax=std::max(amax,std::fabs(z));
        }
        // Clamp before conversion, including an overflowing scale quotient.
        const float desired=std::min(448.f,(amax/6)/ts);
        const auto sc=encode_e4m3(desired); s[r*ss+b]=sc;
        const float scale=e4m3(sc)*ts;
        for(unsigned j=0;j<16;j+=2) {
            const float u=scale?std::clamp(x[r*xs+b*16+j]/scale,-6.f,6.f):0;
            const float w=scale?std::clamp(x[r*xs+b*16+j+1]/scale,-6.f,6.f):0;
            q[r*qs+b*8+j/2]=std::uint8_t(encode_e2m1(u)|(encode_e2m1(w)<<4));
        }
    }
}
static float value(const Matrix& a,unsigned r,unsigned k) {
    const auto c=a.codes[r*a.code_stride+k/2];
    return e2m1(std::uint8_t(c>>((k&1)*4)))*e4m3(a.scales[r*a.scale_stride+k/16]);
}
void reference(const View& v,const Plan& p) {
    const Plan expected=resolve(v);
    if(p.m!=expected.m || p.n!=expected.n || p.k!=expected.k || p.splits!=expected.splits ||
       p.route!=expected.route || p.workspace_bytes!=expected.workspace_bytes)
        throw std::invalid_argument("NVFP4 plan mismatch");
    const unsigned blocks=v.k/64;
    for(unsigned m=0;m<v.m;++m) for(unsigned n=0;n<v.n;++n) {
        float total=0;
        for(unsigned z=0;z<p.splits;++z) {
            float sum=0;
            const unsigned begin=unsigned(std::uint64_t(blocks)*z/p.splits)*64;
            const unsigned end=unsigned(std::uint64_t(blocks)*(z+1)/p.splits)*64;
            for(unsigned k=begin;k<end;++k) sum=std::fma(value(v.a,m,k),value(v.b,n,k),sum);
            total=total+sum;
        }
        float out=total*(v.a.tensor_scale*v.b.tensor_scale);
        if(v.residual) out=out+v.residual[m*v.residual_stride+n];
        v.output[m*v.output_stride+n]=out;
    }
}
}
