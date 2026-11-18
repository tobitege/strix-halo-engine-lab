// Compact old/new migration fixture. All arithmetic is exercised through the incumbent cores.
#include "../../src/hip_q8_rows_gemv.hpp"
#include "../../src/core/f16_dense_q8a8.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <initializer_list>
#include <vector>
#include "byte_equal.hpp"
#if __has_include("../../src/hip/weight/registry.hpp")
#include "../../src/hip/weight/registry.hpp"
#include "strix/weight/source.hpp"
#include "strix/weight/registry.hpp"
#define WFMT_HAS_NATIVE_SOURCES 1
#endif
#ifdef STRIX_WFMT_CAPTURE
#include "../../scripts/bench/wfmt/hip_transfer_capture.hpp"
#endif

namespace {
using strix::weight::test::byte_equal;
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
void context(const std::string& name,const char* arithmetic,std::initializer_list<std::uint64_t> shape,const char* format) {
#ifdef STRIX_WFMT_CAPTURE
    strix::weight::capture::set_context(name.c_str(),arithmetic,shape,format);
#else
    (void)name;(void)arithmetic;(void)shape;(void)format;
#endif
}

struct Buffer {
    void* p = nullptr;
    explicit Buffer(std::size_t n) { check(hipMalloc(&p,n)); check(hipMemset(p,0,n)); }
    ~Buffer() { (void)hipFree(p); }
    template<class T> T* as() const { return static_cast<T*>(p); }
    template<class T> void upload(const std::vector<T>& v) const { check(hipMemcpy(p,v.data(),v.size()*sizeof(T),hipMemcpyHostToDevice)); }
    template<class T> std::vector<T> read(std::size_t n) const {
        std::vector<T> v(n); check(hipMemcpy(v.data(),p,n*sizeof(T),hipMemcpyDeviceToHost)); return v;
    }
};
__device__ float* partial_trace;
__device__ unsigned* partial_count;
inline constexpr unsigned kPartialSlots=4096U;
__device__ void trace_value(float v) {
    const unsigned t=((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    const unsigned i=partial_count[t]++;
    if(i<kPartialSlots) partial_trace[t*kPartialSlots+i]=v;
}

template<unsigned R,bool Diagnostic=false> struct Site {
    using Input = float;
    static constexpr unsigned rows=R;
    static constexpr bool fused_fma=true;
    const void* input[R];
    strix::q8_rows::Segments segments;
    std::uint32_t width,outputs;
    float* output;
    __device__ static float coefficient(unsigned scale,std::int8_t code) {
        float result;
        asm("v_fma_mix_f32 %0, %1, %2, neg(0) op_sel_hi:[1,0,0]" : "=v"(result) : "v"(scale), "v"(static_cast<float>(code)));
        return result;
    }
    __device__ static void capture_partial(float value) requires Diagnostic { trace_value(value); }
    __device__ static void fma(float& acc,float c,float x) {
        acc=__builtin_fmaf(c,x,acc);
        if constexpr(Diagnostic) capture_partial(acc);
    }
    __device__ static float add(float lower,float upper) {
        float result; asm("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(lower), "v"(upper)); return result;
    }
    __device__ void store(std::uint32_t o,unsigned r,float v) const { output[r*outputs+o]=v; }
};
struct RawCols {
    const _Float16* weight; float* out; unsigned in,count;
    __device__ strix::core::f16_dense_tall::ColumnTile tile(unsigned i) const {
        return {weight+16U*i*in,out+16U*i,count,16U*i<count};
    }
};
template<unsigned Form> struct DiagnosticCols: strix::core::f16_dense_tall::q8a8::Cols<8U,Form> {
    __device__ static void capture_partial(const strix::core::f16_dense_tall::Float8& value) {
        for(unsigned i=0;i<8U;++i) trace_value(value[i]);
    }
};
template<unsigned Form> __global__ void f16_partial_kernel(strix::core::f16_dense_tall::q8a8::View v) {
    namespace d=strix::core::f16_dense_tall;
    __shared__ d::Storage<1> lds;
    d::core<1>(nullptr,v.in,v.rows,DiagnosticCols<Form>{{v}},lds,d::q8a8::Store{v.count,v.tile0,v.accumulate});
}
__global__ void raw_kernel(const float* x,unsigned k,unsigned rows,RawCols c) {
    __shared__ strix::core::f16_dense_tall::Storage<1> lds;
    strix::core::f16_dense_tall::core<1>(x,k,rows,c,lds,strix::core::f16_dense_tall::q8a8::Store{c.count,0U,0U},true);
}
__global__ void operands_kernel(std::uint16_t* out) {
    using namespace strix::core::f16_dense_tall::q8a8;
    const unsigned i=blockIdx.x*256U+threadIdx.x;
    if(i>=256U*16U) return;
    constexpr std::uint16_t scales[16]={0,1,0x3ff,0x400,0x7bff,0x8000,0x8001,0x83ff,0x8400,0x3c00,0xbc00,0x3555,0xb555,0x7c00,0xfc00,0x7e00};
    const unsigned code=i%256U, bits=scales[i/256U];
    const std::uint32_t word=code*0x01010101U;
    const auto a=detail::w16_scalar(word,0U,static_cast<float>(__builtin_bit_cast(_Float16,static_cast<std::uint16_t>(bits))));
    const auto pair=detail::w16_pair(word^0x80808080U,0U,__builtin_bit_cast(detail::Half2,bits|(bits<<16U)));
    out[3U*i]=__builtin_bit_cast(std::uint16_t,a);
    out[3U*i+1U]=__builtin_bit_cast(std::uint16_t,pair[0]);
    out[3U*i+2U]=__builtin_bit_cast(std::uint16_t,pair[1]);
}
template<unsigned Form> __global__ void prepared_kernel(strix::core::f16_dense_tall::q8a8::View v,std::uint32_t* out) {
    using namespace strix::core::f16_dense_tall;
    q8a8::Cols<8U,Form> c{v};
    U4 data[8];c.stage(0,threadIdx.x%16U,0,data);
    for(unsigned i=0;i<8U;++i) reinterpret_cast<U4*>(out)[threadIdx.x*8U+i]=data[i];
}
struct StridedCols {
    const _Float16* weight;float* out;unsigned in,count,stride;
    __device__ strix::core::f16_dense_tall::ColumnTile tile(unsigned i) const {return {nullptr,out+16U*i,count,16U*i<count};}
    __device__ void stage(unsigned tile,unsigned col,unsigned unit,strix::core::f16_dense_tall::U4* nb) const {
        namespace d=strix::core::f16_dense_tall;
        const auto* p=reinterpret_cast<const unsigned char*>(weight)+(static_cast<std::size_t>(2U*unit)*stride+tile*16U+col)*16U;
#ifdef WFMT_HAS_NATIVE_SOURCES
        ::strix::weight::hip::RawF16Source::strided_stage(p,stride*16U,nb);
#else
#pragma unroll
        for(unsigned i=0;i<8U;++i) nb[i]=*(d::detail::GlobalU4*)(p+static_cast<std::size_t>(i)*stride*16U);
#endif
    }
};
__global__ void strided_raw_kernel(const float* x,unsigned k,unsigned rows,StridedCols cols) {
    namespace d=strix::core::f16_dense_tall;
    __shared__ d::Storage<1> lds;
    d::core<1>(x,k,rows,cols,lds,d::q8a8::Store{cols.count,0U,0U});
}
void strided_raw_case() {
    context("hip.raw-f16.strided-columns","hip-raw-f16-wmma@1",{7U,17U,128U},"plain.f16@1");
    constexpr unsigned rows=7U,n=17U,padded=32U,k=128U;
    std::vector<_Float16> w(padded*k),packed(padded*k);std::vector<float> x(rows*k);
    for(unsigned o=0;o<padded;++o) for(unsigned j=0;j<k;++j) {
        w[o*k+j]=static_cast<_Float16>(static_cast<float>(static_cast<int>((o*7U+j)%31U)-15)*0.03125F);
        packed[(j/8U*padded+o)*8U+j%8U]=w[o*k+j];
    }
    for(unsigned i=0;i<x.size();++i) x[i]=static_cast<float>(static_cast<int>(i%15U)-7)*0.125F;
    Buffer dw(w.size()*2U),dp(packed.size()*2U),dx(x.size()*4U),dy(rows*n*4U);dw.upload(w);dp.upload(packed);dx.upload(x);
    hipLaunchKernelGGL(raw_kernel,dim3(1,2),dim3(256),0,0,dx.as<float>(),k,rows,RawCols{dw.as<_Float16>(),dy.as<float>(),k,n});check(hipGetLastError());const auto raw=dy.read<float>(rows*n);
    hipLaunchKernelGGL(strided_raw_kernel,dim3(1,2),dim3(256),0,0,dx.as<float>(),k,rows,StridedCols{dp.as<_Float16>(),dy.as<float>(),k,n,padded});check(hipGetLastError());const auto strided=dy.read<float>(rows*n);
    if(!byte_equal(raw,strided)) throw std::runtime_error("strided raw F16 source layout differs");
}
std::vector<unsigned char> weights(unsigned k,unsigned n) {
    std::vector<unsigned char> w(n*(k/32U)*34U);
    for(unsigned b=0;b<n*(k/32U);++b) {
        const std::uint16_t scale=static_cast<std::uint16_t>(0x2800U+(b%8U)*0x100U);
        std::memcpy(w.data()+34U*b,&scale,2);
        for(unsigned j=0;j<32;++j) w[34U*b+2U+j]=static_cast<unsigned char>((b*17U+j*13U)%63U-32U);
    }
    return w;
}
std::vector<unsigned char> q6_weights(unsigned k,unsigned n) {
    std::vector<unsigned char> w(n*(k/32U)*26U,0);
    constexpr std::uint16_t scales[8]={1U,0x3ffU,0x400U,0x2800U,0x3c00U,0xbc00U,0x8000U,0x3555U};
    for(unsigned o=0;o<n;++o) for(unsigned chunk=0;chunk<k/256U;++chunk) {
        const auto c=strix::q6p::chunk_offsets(k,chunk);auto* row=w.data()+o*(k/32U)*26U;
        for(unsigned j=0;j<128U;++j) row[c.qs+j]=static_cast<unsigned char>(j*7U+o*17U+chunk*13U);
        for(unsigned j=0;j<64U;++j) row[c.qh+j]=static_cast<unsigned char>((j*13U+o*23U+chunk*7U)^0xa5U);
        for(unsigned b=0;b<8U;++b) {const auto h=scales[(o+chunk+b)%8U];std::memcpy(row+c.d+2U*b,&h,2U);}
    }
    return w;
}
template<unsigned R> void rows_case(unsigned k,unsigned n) {
    using namespace strix::q8_rows;
    const auto name="hip.rows.r"+std::to_string(R)+".k"+std::to_string(k)+".n"+std::to_string(n);
    context(name,"hip-q8-rows-fma-mix-fixture@1",{R,n,k},"gguf.q8_0@1");
    auto w=weights(k,n); std::vector<float> x(R*k);
    for(unsigned i=0;i<x.size();++i) x[i]=static_cast<float>(static_cast<int>(i%127U)-63)*0.015625F;
    Buffer dw(w.size()), dx(x.size()*4U),dy(R*n*4U);dw.upload(w);dx.upload(x);
    Site<R> s{};s.segments=segments({{dw.p,n}});s.width=k;s.outputs=n;s.output=dy.as<float>();
    for(unsigned r=0;r<R;++r) s.input[r]=dx.as<float>()+r*k;
    const auto read=[&] { check(hipGetLastError()); auto y=dy.read<float>(R*n); for(float v:y) if(!std::isfinite(v)) throw std::runtime_error("nonfinite row output");return y; };
    if constexpr(R<=8U) {
        hipLaunchKernelGGL((project_rows_kernel<Site<R>,2U,false,false>),dim3((n+15U)/16U),dim3(256),0,0,s);const auto a=read();
        hipLaunchKernelGGL((project_rows_kernel<Site<R>,2U,true,false>),dim3((n+15U)/16U),dim3(256),R*k*4U,0,s);const auto b=read();
        if(!byte_equal(a,b)) throw std::runtime_error("staged rows differ");
        hipLaunchKernelGGL((project_rows_stepped_kernel<Site<R>,2U,1U>),dim3((n+15U)/16U),dim3(256),2U*R*256U*4U,0,s);const auto c=read();
        if(!byte_equal(a,c)) throw std::runtime_error("stepped rows differ");
        // Seeded planar low/high codes cover signs, extrema and per-group FP16 scales.
        context(name+".q6p32","hip-q8-rows-fma-mix-fixture@1",{R,n,k},"image.q6p32@1");
        auto q6=q6_weights(k,n);
        Buffer dq(q6.size());dq.upload(q6);s.segments=segments({{dq.p,n}});
        hipLaunchKernelGGL((project_rows_kernel<Site<R>,2U,false,true>),dim3((n+15U)/16U),dim3(256),0,0,s);read();
        hipLaunchKernelGGL((project_rows_kernel<Site<R>,2U,true,true>),dim3((n+15U)/16U),dim3(256),R*k*4U,0,s);read();
    } else {
        constexpr unsigned sub=4U;
        hipLaunchKernelGGL((project_wide_rows_kernel<Site<R>,2U,sub>),dim3(((n+15U)/16U)*(R/sub)),dim3(256),0,0,s);const auto a=read();
        hipLaunchKernelGGL((project_wide_rows_staged_kernel<Site<R>,2U,sub>),dim3(((n+15U)/16U)*(R/sub)),dim3(256),0,0,s);const auto b=read();
        if(!byte_equal(a,b)) throw std::runtime_error("wide staged differs");
        constexpr unsigned waves=R/sub<8U?R/sub:8U;
        hipLaunchKernelGGL((project_wide_rows_coef_kernel<Site<R>,2U,sub,8U,false>),dim3(((n+1U)/2U)*(R/sub/waves)),dim3(32U*waves),0,0,s);const auto c=read();
        if(!byte_equal(a,c)) throw std::runtime_error("coefficient source differs");
        // Production admits the lane kernel only for width % 512 == 0 (an even chunk count; see
        // lane_admits in src/hip_q8_rows_gemv.hpp); other widths are outside its contract.
        if constexpr(R%32U==0U) if(k%512U==0U) {
            constexpr unsigned groups=(65U+63U)/64U;
            const unsigned g=(n+63U)/64U;
            Buffer codes(groups*16U*(k/256U)*1024U),scales(groups*(k/256U)*64U*16U),xs(R*k*4U);
            hipLaunchKernelGGL((lane_pack_weights_kernel<Site<R>>),dim3((g*(k/256U)*64U*16U+255U)/256U),dim3(256),0,0,s,codes.as<unsigned char>(),scales.as<std::uint16_t>(),g);
            codes.read<unsigned char>(g*16U*(k/256U)*1024U);scales.read<std::uint16_t>(g*(k/256U)*64U*8U);
            hipLaunchKernelGGL((lane_pack_rows_kernel<Site<R>>),dim3((R*k+255U)/256U),dim3(256),0,0,s,xs.as<float>());
            xs.read<float>(R*k);
            hipLaunchKernelGGL((project_wide_rows_lane_kernel<Site<R>,8U,8U,2U>),dim3(g*(R/32U)),dim3(256),0,0,s,codes.as<unsigned char>(),scales.as<std::uint16_t>(),xs.as<float>());const auto d=read();
            if(!byte_equal(a,d)) {unsigned diff=0;for(unsigned i=0;i<a.size();++i) diff+=std::bit_cast<std::uint32_t>(a[i])!=std::bit_cast<std::uint32_t>(d[i]);
                std::fprintf(stderr,"lane-vs-wide R=%u n=%u k=%u differ=%u/%zu\n",R,n,k,diff,a.size());
                throw std::runtime_error("lane source differs");}
        }
    }
}
template<unsigned R> void signed_case(unsigned k,unsigned n) {
    using namespace strix::q8_rows;
    const auto name="hip.signed-a8.r"+std::to_string(R)+".k"+std::to_string(k)+".n"+std::to_string(n);
    context(name,"hip-signed-a8-block-chain@1",{R,n,k},"gguf.q8_0@1");
    auto w=weights(k,n);std::vector<float> x(R*k);
    for(unsigned i=0;i<x.size();++i) x[i]=static_cast<float>(static_cast<int>(i%31U)-15)*0.03125F;
    Buffer dw(w.size()),dx(x.size()*4U),dy(R*n*4U),codes(dense_a8_pitch<R>()*k),deltas(dense_a8_pitch<R>()*(k/32U)*4U);
    dw.upload(w);dx.upload(x);
    Site<R> site{};site.segments=segments({{dw.p,n}});site.width=k;site.outputs=n;site.output=dy.as<float>();
    for(unsigned r=0;r<R;++r) site.input[r]=dx.as<float>()+r*k;
    hipLaunchKernelGGL((dense_a8_quantize_kernel<Site<R>>),dim3((R*(k/8U)+255U)/256U),dim3(256),0,0,site,codes.as<std::int8_t>(),deltas.as<float>());check(hipGetLastError());
    const auto ac=codes.read<std::int8_t>(R*k);const auto ad=deltas.read<float>(dense_a8_pitch<R>()*(k/32U));
#ifdef STRIX_WFMT_CAPTURE
    strix::weight::capture::emit("activation_codes",ac.data(),ac.size());
    strix::weight::capture::emit("activation_scales",ad.data(),ad.size()*sizeof(float));
#endif
    if constexpr(R<=8U) {
        hipLaunchKernelGGL((project_rows_dense_a8_kernel<Site<R>,1U,false,false>),dim3((n+7U)/8U),dim3(256),0,0,site,codes.as<std::int8_t>(),deltas.as<float>());check(hipGetLastError());const auto direct=dy.read<float>(R*n);
        hipLaunchKernelGGL((project_rows_dense_a8_kernel<Site<R>,1U,true,false>),dim3((n+7U)/8U),dim3(256),dense_a8_stage_bytes<R>(k),0,site,codes.as<std::int8_t>(),deltas.as<float>());check(hipGetLastError());const auto staged=dy.read<float>(R*n);
        if(!byte_equal(direct,staged)) throw std::runtime_error("signed-A8 source staging differs");
        context(name+".q6p32","hip-signed-a8-block-chain@1",{R,n,k},"image.q6p32@1");
        auto q6=q6_weights(k,n);
        Buffer dq(q6.size());dq.upload(q6);site.segments=segments({{dq.p,n}});
        hipLaunchKernelGGL((project_rows_dense_a8_kernel<Site<R>,1U,false,true>),dim3((n+7U)/8U),dim3(256),0,0,site,codes.as<std::int8_t>(),deltas.as<float>());check(hipGetLastError());dy.read<float>(R*n);
    } else {
        hipLaunchKernelGGL((project_rows_dense_a8_wmma_kernel<Site<R>,1U,1U>),dim3(((n+63U)/64U)*(R/16U)),dim3(128),0,0,site,codes.as<std::int8_t>(),deltas.as<float>());check(hipGetLastError());dy.read<float>(R*n);
        if constexpr(R%64U==0U) {
            hipLaunchKernelGGL((project_rows_dense_a8_wmma_lds2_kernel<Site<R>,4U,true>),dim3(((n+127U)/128U)*(R/64U)),dim3(128),0,0,site,codes.as<std::int8_t>(),deltas.as<float>());check(hipGetLastError());dy.read<float>(R*n);
        }
    }
}
void partial_case(bool q6) {
    using namespace strix::q8_rows;
    context(q6?"hip.rows.partial.q6p32":"hip.rows.partial.q8","hip-q8-rows-fma-mix-fixture@1",{1U,1U,256U},q6?"image.q6p32@1":"gguf.q8_0@1");
    auto w=weights(256U,1U);
    if(q6) w=q6_weights(256U,1U);
    std::vector<float> x(256U,0.125F);
    Buffer dw(w.size()),dx(x.size()*4U),dy(4U),trace(256U*kPartialSlots*4U),counts(256U*4U);
    dw.upload(w);dx.upload(x);
    auto* tp=trace.as<float>();auto* cp=counts.as<unsigned>();
    check(hipMemcpyToSymbol(HIP_SYMBOL(partial_trace),&tp,sizeof(tp)));
    check(hipMemcpyToSymbol(HIP_SYMBOL(partial_count),&cp,sizeof(cp)));
    Site<1U,true> site{};site.input[0]=dx.p;site.segments=segments({{dw.p,1U}});site.width=256;site.outputs=1;site.output=dy.as<float>();
    if(q6) hipLaunchKernelGGL((project_rows_kernel<Site<1U,true>,2U,false,true>),dim3(1),dim3(256),0,0,site);
    else hipLaunchKernelGGL((project_rows_kernel<Site<1U,true>,2U,false,false>),dim3(1),dim3(256),0,0,site);
    check(hipGetLastError());dy.read<float>(1);
    auto values=trace.read<float>(256U*kPartialSlots);
    auto used=counts.read<unsigned>(256U);
    for(unsigned count:used) if(count>kPartialSlots) throw std::runtime_error("diagnostic partial capacity");
#ifdef STRIX_WFMT_CAPTURE
    strix::weight::capture::emit("partials",values.data(),values.size()*sizeof(float));
#endif
}
template<unsigned R> void row_partials() {
    using namespace strix::q8_rows;
    constexpr unsigned k=512U,chunks=k/256U;
    static_assert(k%512U==0U); // lane pipeline requires an even chunk count
    context("hip.rows.partial.r"+std::to_string(R),"hip-q8-rows-fma-mix-fixture@1",{R,1U,k},"gguf.q8_0@1");
    auto w=weights(k,1U);std::vector<float> x(R*k,0.125F);
    constexpr unsigned blocks=R>8U?R/4U:1U,threads=blocks*256U;
    Buffer dw(w.size()),dx(x.size()*4U),dy(R*4U),trace(threads*kPartialSlots*4U),counts(threads*4U);
    dw.upload(w);dx.upload(x);
    auto* tp=trace.as<float>();auto* cp=counts.as<unsigned>();
    check(hipMemcpyToSymbol(HIP_SYMBOL(partial_trace),&tp,sizeof(tp)));check(hipMemcpyToSymbol(HIP_SYMBOL(partial_count),&cp,sizeof(cp)));
    Site<R,true> site{};site.segments=segments({{dw.p,1U}});site.width=k;site.outputs=1;site.output=dy.as<float>();
    for(unsigned r=0;r<R;++r) site.input[r]=dx.as<float>()+r*k;
    const auto read=[&] {
        check(hipGetLastError());auto output=dy.read<float>(R);auto values=trace.read<float>(threads*kPartialSlots);const auto used=counts.read<unsigned>(threads);
        unsigned total=0;for(unsigned n:used) {if(n>kPartialSlots) throw std::runtime_error("row diagnostic partial capacity");total+=n;}
#ifdef STRIX_WFMT_CAPTURE
        if(!total) throw std::runtime_error("missing row diagnostic partials");
        strix::weight::capture::emit("partials",values.data(),values.size()*sizeof(float));
#else
        (void)total;
#endif
        check(hipMemset(trace.p,0,threads*kPartialSlots*4U));check(hipMemset(counts.p,0,threads*4U));
        return output;
    };
    if constexpr(R<=8U) {
        hipLaunchKernelGGL((project_rows_kernel<Site<R,true>,2U,true,false>),dim3(1),dim3(256),R*k*4U,0,site);read();
        hipLaunchKernelGGL((project_rows_stepped_kernel<Site<R,true>,2U,1U>),dim3(1),dim3(256),2U*R*256U*4U,0,site);read();
    } else {
        hipLaunchKernelGGL((project_wide_rows_kernel<Site<R,true>,2U,4U>),dim3(blocks),dim3(256),0,0,site);const auto wide=read();
        hipLaunchKernelGGL((project_wide_rows_staged_kernel<Site<R,true>,2U,4U>),dim3(blocks),dim3(256),0,0,site);read();
        hipLaunchKernelGGL((project_wide_rows_coef_kernel<Site<R,true>,2U,4U,8U,false>),dim3(1),dim3(256),0,0,site);read();
        Buffer codes(16U*chunks*1024U),scales(chunks*64U*16U),xs(R*k*4U);
        hipLaunchKernelGGL((lane_pack_weights_kernel<Site<R,true>>),dim3(chunks*4U),dim3(256),0,0,site,codes.as<unsigned char>(),scales.as<std::uint16_t>(),1U);
        codes.read<unsigned char>(16U*chunks*1024U);scales.read<std::uint16_t>(chunks*64U*8U);
        hipLaunchKernelGGL((lane_pack_rows_kernel<Site<R,true>>),dim3(R*chunks),dim3(256),0,0,site,xs.as<float>());xs.read<float>(R*k);
        hipLaunchKernelGGL((project_wide_rows_lane_kernel<Site<R,true>,8U,8U,2U>),dim3(1),dim3(256),0,0,site,codes.as<unsigned char>(),scales.as<std::uint16_t>(),xs.as<float>());const auto lane=read();
        if(!byte_equal(wide,lane)) throw std::runtime_error("diagnostic lane source differs");
    }
}
template<unsigned R> void signed_partials() {
    using namespace strix::q8_rows;
    context("hip.signed-a8.partial.r"+std::to_string(R),"hip-signed-a8-block-chain@1",{R,1U,256U},"gguf.q8_0@1");
    constexpr unsigned threads=R<=8U?256U:R*8U;
    auto w=weights(256U,1U);std::vector<float> x(R*256U,0.125F);
    Buffer dw(w.size()),dx(x.size()*4U),dy(R*4U),codes(dense_a8_pitch<R>()*256U),deltas(dense_a8_pitch<R>()*8U*4U),trace(threads*kPartialSlots*4U),counts(threads*4U);
    dw.upload(w);dx.upload(x);auto* tp=trace.as<float>();auto* cp=counts.as<unsigned>();
    check(hipMemcpyToSymbol(HIP_SYMBOL(partial_trace),&tp,sizeof(tp)));check(hipMemcpyToSymbol(HIP_SYMBOL(partial_count),&cp,sizeof(cp)));
    Site<R,true> site{};site.segments=segments({{dw.p,1U}});site.width=256;site.outputs=1;site.output=dy.as<float>();
    for(unsigned r=0;r<R;++r) site.input[r]=dx.as<float>()+r*256U;
    hipLaunchKernelGGL((dense_a8_quantize_kernel<Site<R,true>>),dim3((R*32U+255U)/256U),dim3(256),0,0,site,codes.as<std::int8_t>(),deltas.as<float>());
    codes.read<std::int8_t>(R*256U);deltas.read<float>(dense_a8_pitch<R>()*8U);
    const auto read=[&] {
        check(hipGetLastError());dy.read<float>(R);auto values=trace.read<float>(threads*kPartialSlots);const auto used=counts.read<unsigned>(threads);
        unsigned total=0;for(unsigned n:used) {if(n>kPartialSlots) throw std::runtime_error("signed diagnostic partial capacity");total+=n;}
#ifdef STRIX_WFMT_CAPTURE
        if(!total) throw std::runtime_error("missing signed diagnostic partials");
        strix::weight::capture::emit("partials",values.data(),values.size()*sizeof(float));
#endif
        check(hipMemset(trace.p,0,threads*kPartialSlots*4U));check(hipMemset(counts.p,0,threads*4U));
    };
    if constexpr(R<=8U) {
        hipLaunchKernelGGL((project_rows_dense_a8_kernel<Site<R,true>,1U,false,false>),dim3(1),dim3(256),0,0,site,codes.as<std::int8_t>(),deltas.as<float>());read();
        hipLaunchKernelGGL((project_rows_dense_a8_kernel<Site<R,true>,1U,true,false>),dim3(1),dim3(256),dense_a8_stage_bytes<R>(256U),0,site,codes.as<std::int8_t>(),deltas.as<float>());read();
        auto q6=q6_weights(256U,1U);Buffer dq(q6.size());dq.upload(q6);site.segments=segments({{dq.p,1U}});
        hipLaunchKernelGGL((project_rows_dense_a8_kernel<Site<R,true>,1U,false,true>),dim3(1),dim3(256),0,0,site,codes.as<std::int8_t>(),deltas.as<float>());read();
    } else {
        hipLaunchKernelGGL((project_rows_dense_a8_wmma_kernel<Site<R,true>,1U,1U>),dim3(R/16U),dim3(128),0,0,site,codes.as<std::int8_t>(),deltas.as<float>());read();
        if constexpr(R%64U==0U) {
            hipLaunchKernelGGL((project_rows_dense_a8_wmma_lds2_kernel<Site<R,true>,4U,true>),dim3(R/64U),dim3(128),0,0,site,codes.as<std::int8_t>(),deltas.as<float>());read();
        }
    }
}
void f16_partials() {
    namespace d=strix::core::f16_dense_tall;
    context("hip.dense-f16.partial","dense-q8a8-f16-v1@1",{1U,1U,128U},"gguf.q8_0@1");
    auto w=weights(128U,1U);std::vector<std::int8_t> a(128U,7);std::vector<float> delta(4U,0.03125F);
    Buffer dw(w.size()),da(a.size()),ds(16U),dy(4U),trace(256U*kPartialSlots*4U),counts(256U*4U);
    dw.upload(w);da.upload(a);ds.upload(delta);
    auto* tp=trace.as<float>();auto* cp=counts.as<unsigned>();
    check(hipMemcpyToSymbol(HIP_SYMBOL(partial_trace),&tp,sizeof(tp)));check(hipMemcpyToSymbol(HIP_SYMBOL(partial_count),&cp,sizeof(cp)));
    d::q8a8::View v{dw.as<std::uint8_t>(),da.as<std::int8_t>(),ds.as<float>(),dy.as<float>(),136U,0U,128U,1U,1U,128U,1U,0U,0U};
    const auto read=[&] {
        check(hipGetLastError());dy.read<float>(1U);auto values=trace.read<float>(256U*kPartialSlots);const auto used=counts.read<unsigned>(256U);
        for(unsigned n:used) if(n>kPartialSlots) throw std::runtime_error("F16 diagnostic partial capacity");
#ifdef STRIX_WFMT_CAPTURE
        strix::weight::capture::emit("partials",values.data(),values.size()*sizeof(float));
#endif
        check(hipMemset(trace.p,0,256U*kPartialSlots*4U));check(hipMemset(counts.p,0,256U*4U));
    };
    hipLaunchKernelGGL((f16_partial_kernel<0U>),dim3(1),dim3(256),0,0,v);read();
    hipLaunchKernelGGL((f16_partial_kernel<1U>),dim3(1),dim3(256),0,0,v);read();
}
void dense_case(unsigned k,unsigned n,unsigned rows) {
    namespace d=strix::core::f16_dense_tall;
    const auto name="hip.dense-q8a8.r"+std::to_string(rows)+".k"+std::to_string(k)+".n"+std::to_string(n);
    context(name,"dense-q8a8-f16-v1@1",{rows,n,k},"gguf.q8_0@1");
    auto w=weights(k,n);std::vector<std::int8_t> a(rows*k);std::vector<float> scales(rows*k/32U,0.03125F),x(rows*k);
    for(unsigned i=0;i<a.size();++i) {a[i]=static_cast<std::int8_t>(static_cast<int>(i%63U)-31);x[i]=static_cast<float>(a[i])*0.03125F;}
    Buffer dw(w.size()),da(a.size()),ds(scales.size()*4U),dx(x.size()*4U),dy(rows*n*4U);
    dw.upload(w);da.upload(a);ds.upload(scales);dx.upload(x);
    d::q8a8::View v{dw.as<std::uint8_t>(),da.as<std::int8_t>(),ds.as<float>(),dy.as<float>(),k/32U*34U,0,k,n,rows,k,n,0,0};
    if(rows==1U) {
        Buffer prepared(16U*8U*16U);
        hipLaunchKernelGGL((prepared_kernel<0U>),dim3(1),dim3(16),0,0,v,prepared.as<std::uint32_t>());check(hipGetLastError());const auto a0=prepared.read<std::uint32_t>(16U*8U*4U);
        hipLaunchKernelGGL((prepared_kernel<1U>),dim3(1),dim3(16),0,0,v,prepared.as<std::uint32_t>());check(hipGetLastError());const auto a1=prepared.read<std::uint32_t>(16U*8U*4U);
        if(!byte_equal(a0,a1)) throw std::runtime_error("prepared scalar packed fragments differ");
#ifdef STRIX_WFMT_CAPTURE
        strix::weight::capture::emit("prepared",a0.data(),a0.size()*sizeof(std::uint32_t));
#endif
    }
    hipLaunchKernelGGL((d::q8a8::kernel<1U,8U,0U>),dim3((rows+127U)/128U,(n+15U)/16U),dim3(256),0,0,v);check(hipGetLastError());const auto scalar=dy.read<float>(rows*n);
    hipLaunchKernelGGL((d::q8a8::kernel<1U,8U,1U>),dim3((rows+127U)/128U,(n+15U)/16U),dim3(256),0,0,v);check(hipGetLastError());const auto packed=dy.read<float>(rows*n);
    if(!byte_equal(scalar,packed)) throw std::runtime_error("scalar packed source differs");
    std::vector<_Float16> f16(((n+15U)/16U)*16U*k);
    for(unsigned o=0;o<((n+15U)/16U)*16U;++o) for(unsigned j=0;j<k;++j) {
        const unsigned ro=std::min(o,n-1U);std::uint16_t scale;std::memcpy(&scale,w.data()+(ro*k/32U+j/32U)*34U,2U);
        f16[o*k+j]=static_cast<_Float16>(static_cast<float>(std::bit_cast<_Float16>(scale))*static_cast<float>(static_cast<std::int8_t>(w[(ro*k/32U+j/32U)*34U+2U+j%32U])));
    }
    context(name+".raw-f16","hip-raw-f16-wmma@1",{rows,n,k},"plain.f16@1");
    Buffer df(f16.size()*2U);df.upload(f16);
    hipLaunchKernelGGL(raw_kernel,dim3((rows+127U)/128U,(n+15U)/16U),dim3(256),0,0,dx.as<float>(),k,rows,RawCols{df.as<_Float16>(),dy.as<float>(),k,n});check(hipGetLastError());const auto raw=dy.read<float>(rows*n);
    if(!byte_equal(scalar,raw)) throw std::runtime_error("raw F16 source differs");
}
}
int main() try {
#ifdef WFMT_HAS_NATIVE_SOURCES
    namespace wh=strix::weight::hip;
    static_assert(std::is_trivially_copyable_v<wh::MatrixView>);
    auto fixture=weights(128U,17U);
    strix::weight::WeightView view{};
    view.storage={"gguf.q8_0",1};view.shape={{17U,128U,0U,0U},2};
    view.payload={strix::weight::memory_source(std::as_bytes(std::span(fixture))),{0U,fixture.size()}};
    if(!strix::weight::format_geometry(view.storage,view.parameters,view.geometry)) throw std::runtime_error("Q8 source geometry");
    wh::LaunchShape shape{1U,17U,128U,136U,128U,17U,0U};
    const auto* registered=wh::find_source(view.storage,"f16-tile");
    if(wh::sources().size()!=23U || !registered || !registered->validate(view,shape)) throw std::runtime_error("HIP source registration validation");
    shape.activation_stride=127U;
    if(registered->validate(view,shape)) throw std::runtime_error("HIP source admitted invalid activation stride");
    shape.activation_stride=128U;shape.k=64U;
    if(registered->validate(view,shape)) throw std::runtime_error("HIP F16 source admitted partial stage");
    shape.k=96U;registered=wh::find_source(view.storage,"signed-a8");
    if(!registered || registered->paired_k!=64U || registered->validate(view,shape)) throw std::runtime_error("HIP signed source admitted odd block pair");
    if(wh::find_source({"gguf.iq2_xxs",1},"f16-tile")) throw std::runtime_error("unregistered HIP source unexpectedly admitted");
    const auto* gsq=wh::find_source({"gguf.iq3_s",1},"f16-tile");
    const auto* gsq_a8=wh::find_source({"gguf.iq3_s",1},"signed-a8");
    const auto* gsq_pair=wh::find_source({"gguf.iq4_nl",1},"f16-tile");
    if(!gsq || !gsq_a8 || !gsq_pair || gsq->paired_k!=0U || gsq_pair->paired_k!=64U || gsq->validate(view,shape))
        throw std::runtime_error("HIP GSQ source registration");
#endif
    strided_raw_case();
    context("hip.q8-f16-operands","dense-q8a8-f16-v1@1",{16U,256U},"gguf.q8_0@1");
    Buffer op(4096U*3U*2U);hipLaunchKernelGGL(operands_kernel,dim3(16),dim3(256),0,0,op.as<std::uint16_t>());check(hipGetLastError());op.read<std::uint16_t>(4096U*3U);
    partial_case(false);partial_case(true);row_partials<1>();row_partials<32>();signed_partials<1>();signed_partials<16>();signed_partials<64>();f16_partials();
    signed_case<1>(256U,17U);signed_case<2>(256U,17U);signed_case<7>(256U,17U);signed_case<8>(256U,17U);
    signed_case<16>(256U,17U);signed_case<32>(256U,17U);signed_case<64>(256U,17U);signed_case<128>(256U,17U);
    for(unsigned n:{1U,17U,65U}) {rows_case<1>(256U,n);rows_case<2>(512U,n);rows_case<7>(256U,n);rows_case<8>(512U,n);rows_case<16>(256U,n);rows_case<32>(512U,n);rows_case<64>(256U,n);rows_case<64>(512U,n);}
    for(unsigned rows:{1U,2U,7U,8U,9U,31U,32U,33U,63U,64U,65U,127U,128U,129U}) dense_case(128U,17U,rows);
    std::cout<<"HIP dense source migration: operands/raw/Q8 scalar-packed/Q6/direct/staged/stepped/wide/coef/lane PASS\n";return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
