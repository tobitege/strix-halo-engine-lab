#include "../../src/cuda/weight/gsq_sources.cuh"
#include "strix/weight/source.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

namespace strix::weight::test {
namespace {
namespace cw=::strix::cuda::weight;
struct GroupArch { using SignedGroup=cw::CudaSignedGroup32; };
template<unsigned T> using Source=cw::OperandSource<cw::GgmlTag<T>,cw::SignedA8Tag,GroupArch,cw::GsqIq3SA8Dot4Cuda>;
template<unsigned T> __global__ void groups(typename Source<T>::View view,cw::CudaSignedGroup32* output,float* product,unsigned count) {
    const unsigned g=blockIdx.x*blockDim.x+threadIdx.x;
    if(g>=count) return;
    const auto decoded=Source<T>::signed_group(view,g/(view.k/32),g%(view.k/32));
    output[g]=decoded;
    int lo=0,hi=0;
    for(unsigned j=0;j<32;++j) {
        const int a=int((g*37+j*13)%255)-127;
        if constexpr(T==22) { if(j<16) lo+=a*decoded.codes[j]; else hi+=a*decoded.codes[j]; }
        else lo+=a*decoded.codes[j];
    }
    float v=__fmul_rn(decoded.scales[0],float(lo));
    if constexpr(T==22) v=__fadd_rn(v,__fmul_rn(decoded.scales[1],float(hi)));
    product[g]=__fmul_rn(0x1.23456p-8F,v);
}
float multiply(float a,float b) { volatile float c=a*b; return c; }
float add(float a,float b) { volatile float c=a+b; return c; }
template<unsigned T> bool exercise(const char* name,unsigned rows,unsigned blocks) {
    constexpr unsigned be=cw::GsqDecode<T>::be,bb=cw::GsqDecode<T>::bb;
    const unsigned width=be*blocks,count=rows*width/32;
    std::vector<Byte> bytes(std::size_t(rows)*blocks*bb);
    std::uint32_t seed=0x71521+T;
    for(auto& b:bytes) { seed=1664525U*seed+1013904223U; b=Byte(seed>>24); }
    constexpr std::array<std::uint16_t,6> scales={0,0x8000,1,0x03ff,0x3c00,0x7bff};
    for(unsigned b=0;b<rows*blocks;++b) {
        auto* x=reinterpret_cast<std::uint8_t*>(bytes.data())+b*bb;
        const auto half=scales[b%scales.size()]; x[0]=std::uint8_t(half); x[1]=std::uint8_t(half>>8);
        if constexpr(T==21 || T==18) for(unsigned q=0;q<64;++q) x[2+q]=std::uint8_t(b);
        if constexpr(T==21) for(unsigned q=0;q<8;++q) x[66+q]=b&256?255:0;
        if constexpr(T==22) {
            for(unsigned q=0;q<32;++q) x[2+q]=std::uint8_t(b);
            for(unsigned q=0;q<8;++q) x[66+q]=std::uint8_t(0x55*(b>>8));
        }
    }
    const auto& registry=builtin_registry();
    const auto* descriptor=find_format(registry,{name,1});
    if(!descriptor) return false;
    WeightView view{}; view.storage=descriptor->id; view.shape={{{rows,width,0,0}},2};
    if(!descriptor->geometry(view.parameters,view.geometry)) return false;
    view.payload={memory_source(bytes),{0,bytes.size()}};
    cw::LaunchShape launch{1,rows,width,0,{},1};
    if(!Source<T>::validate(view,launch)) return false;
    auto invalid=launch; invalid.k=width-1;
    if(Source<T>::validate(view,invalid)) return false;
    invalid=launch; invalid.expert_stride=1;
    if(Source<T>::validate(view,invalid)) return false;
    std::uint8_t* raw=nullptr; cw::CudaSignedGroup32* decoded=nullptr; float* products=nullptr;
    if(cudaMalloc(&raw,bytes.size())!=cudaSuccess) return false;
    if(cudaMalloc(&decoded,count*sizeof(*decoded))!=cudaSuccess) { cudaFree(raw); return false; }
    if(cudaMalloc(&products,count*sizeof(*products))!=cudaSuccess) { cudaFree(decoded); cudaFree(raw); return false; }
    bool ok=cudaMemcpy(raw,bytes.data(),bytes.size(),cudaMemcpyHostToDevice)==cudaSuccess;
    typename Source<T>::View device{};
    const cw::UploadedSlices upload{{raw,bytes.size()},{}};
    ok=ok && bool(Source<T>::make_view(view,upload,launch,device));
    std::vector<cw::CudaSignedGroup32> got(count); std::vector<float> actual(count);
    if(ok) {
        groups<T><<<(count+127)/128,128>>>(device,decoded,products,count);
        ok=cudaGetLastError()==cudaSuccess && cudaMemcpy(got.data(),decoded,count*sizeof(*decoded),cudaMemcpyDeviceToHost)==cudaSuccess &&
           cudaMemcpy(actual.data(),products,count*sizeof(*products),cudaMemcpyDeviceToHost)==cudaSuccess;
    }
    for(unsigned g=0;ok && g<count;++g) {
        SignedGroup32 oracle{};
        ok=bool(descriptor->signed_group_reference(view,g,oracle));
        if(!ok || std::memcmp(got[g].codes,oracle.codes.data(),32)) { ok=false; break; }
        const unsigned scale_count=T==22?2:1;
        if(oracle.scale_count!=scale_count || std::memcmp(got[g].scales,oracle.scales.data(),4*scale_count)) { ok=false; break; }
        int lo=0,hi=0;
        for(unsigned j=0;j<32;++j) {
            const int a=int((g*37+j*13)%255)-127;
            if constexpr(T==22) { if(j<16) lo+=a*oracle.codes[j]; else hi+=a*oracle.codes[j]; }
            else lo+=a*oracle.codes[j];
        }
        float v=multiply(oracle.scales[0],float(lo));
        if constexpr(T==22) v=add(v,multiply(oracle.scales[1],float(hi)));
        const float expected=multiply(0x1.23456p-8F,v);
        if(std::memcmp(&expected,&actual[g],4)) ok=false;
    }
    cudaFree(products); cudaFree(decoded); cudaFree(raw);
    std::printf("cuda-gsq T%u rows%u K%u groups%u OperandExact codes/scales/ordered-products %s\n",T,rows,width,count,ok?"PASS":"FAIL");
    return ok;
}
template<unsigned T> bool formats(const char* name) { return exercise<T>(name,3,2) && exercise<T>(name,1,T==22?1024:512); }
}
bool cuda_gsq_case() {
    return formats<18>("gguf.iq3_xxs") && formats<20>("gguf.iq4_nl") && formats<21>("gguf.iq3_s") &&
           formats<22>("gguf.iq2_s") && formats<23>("gguf.iq4_xs") && formats<42>("gguf.q2_0.strix");
}
}
