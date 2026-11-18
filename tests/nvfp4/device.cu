#include "strix/cuda/core/nvfp4_cuda.hpp"
#include "strix/env_flags.hpp"
#include "tolerance.hpp"
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#ifdef STRIX_NVFP4_NVML
#include <nvml.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using namespace strix::cuda::core::nvfp4;
static void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void require(bool b,const char* s) { if(!b) throw std::runtime_error(s); }
template<class T> struct Buffer {
    T* p=nullptr;
    explicit Buffer(std::size_t n) { if(n) ck(cudaMalloc(&p,n*sizeof(T))); }
    ~Buffer() { if(p) cudaFree(p); }
    Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
    void put(const std::vector<T>& x) { ck(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice)); }
    void get(std::vector<T>& x) { ck(cudaMemcpy(x.data(),p,x.size()*sizeof(T),cudaMemcpyDeviceToHost)); }
};
struct Packed {
    unsigned rows,k; std::size_t qs,ss;
    std::vector<std::uint8_t> q,s;
    Buffer<std::uint8_t> dq,ds;
    Packed(unsigned r,unsigned width,bool exact,bool padded): rows(r),k(width),qs(width/2+(padded?12:0)),
       ss(width/16+(padded?4:0)),q(r*qs),s(r*ss),dq(q.size()),ds(s.size()) {
        std::mt19937 gen(r*17+width);
        for(auto& c:q) c=std::uint8_t(gen());
        for(auto& c:s) c=exact?56:std::uint8_t(gen()%127);
        dq.put(q); ds.put(s);
    }
    Matrix host() const { return {q.data(),s.data(),qs,ss,.5f}; }
    Matrix device() const { return {dq.p,ds.p,qs,ss,.5f}; }
};
__global__ void codecs(const float* x,std::uint8_t* q,std::uint8_t* s,unsigned n) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {
        q[i]=__nv_cvt_float_to_fp4(x[i],__NV_E2M1,cudaRoundNearest);
        s[i]=__nv_cvt_float_to_fp8(fabsf(x[i]),__NV_SATFINITE,__NV_E4M3);
    }
}
static void codec_test() {
    std::vector<float> x;
    for(unsigned i=0;i<126;++i) {
        const float a=e4m3(std::uint8_t(i)),b=e4m3(std::uint8_t(i+1)),m=(a+b)*.5f;
        for(float z:{a,m,std::nextafter(m,0.f),std::nextafter(m,std::numeric_limits<float>::infinity())}) {x.push_back(z);x.push_back(-z);}
    }
    for(unsigned i=0;i<7;++i) {
        const float m=(e2m1(std::uint8_t(i))+e2m1(std::uint8_t(i+1)))*.5f;
        for(float z:{m,std::nextafter(m,0.f),std::nextafter(m,std::numeric_limits<float>::infinity())}) {x.push_back(z);x.push_back(-z);}
    }
    Buffer<float> dx(x.size()); dx.put(x);
    Buffer<std::uint8_t> dq(x.size()),ds(x.size());
    codecs<<<unsigned((x.size()+127)/128),128>>>(dx.p,dq.p,ds.p,unsigned(x.size())); ck(cudaGetLastError());
    std::vector<std::uint8_t> q(x.size()),s(x.size()); dq.get(q);ds.get(s);
    for(unsigned i=0;i<x.size();++i) {
        require(q[i]==encode_e2m1(x[i]),"CUDA FP4 codec mismatch");
        require(s[i]==encode_e4m3(std::fabs(x[i])),"CUDA FP8 codec mismatch");
    }
}
static void quantizer_test() {
    constexpr unsigned rows=3,k=192,xs=k+8,qs=k/2+12,ss=k/16+4;
    std::vector<float> x(rows*xs);
    std::mt19937 gen(872);
    for(auto& z:x) z=std::ldexp(float(int(gen()%8193)-4096),int(gen()%30)-20);
    std::fill(x.begin(),x.begin()+16,0.f);
    Buffer<float> dx(x.size());dx.put(x);
    std::vector<std::uint8_t> q(rows*qs,0xab),s(rows*ss,0xcd),gotq(q),gots(s);
    Buffer<std::uint8_t> dq(q.size()),ds(s.size());
    for(float ts:{.00317f,.5f,1000.f}) {
        dq.put(gotq);ds.put(gots);
        quantize(x.data(),xs,rows,k,q.data(),qs,s.data(),ss,ts);
        ck(quantize_enqueue(dx.p,xs,rows,k,dq.p,qs,ds.p,ss,ts,nullptr));
        std::vector<std::uint8_t> actualq(q.size()),actuals(s.size());dq.get(actualq);ds.get(actuals);
        require(q==actualq && s==actuals,"quantizer byte mismatch or padding store");
    }
    std::cout<<"host/device activation quantizer byte equality PASS\n";
}
static void fixture(unsigned m,unsigned n,unsigned k,bool exact,bool residual,bool alias=false) {
    Packed a(m,k,exact,true),b(n,k,exact,true);
    const unsigned ys=n+5;
    std::vector<float> ref(std::size_t(m)*ys,-77),got(ref),again(ref),res(ref.size(),.25f);
    Buffer<float> dy(got.size()),dr(res.size()); dy.put(alias?res:got);dr.put(res);
    View host{a.host(),b.host(),ref.data(),residual?res.data():nullptr,ys,ys,m,n,k};
    const auto p=resolve(host); reference(host,p);
    View dev{a.device(),b.device(),dy.p,residual?(alias?dy.p:dr.p):nullptr,ys,ys,m,n,k};
    Buffer<std::uint8_t> work(p.workspace_bytes);
    ck(enqueue(dev,p,work.p,p.workspace_bytes,nullptr));dy.get(got);
    dy.put(alias?res:again);
    ck(enqueue(dev,p,work.p,p.workspace_bytes,nullptr));dy.get(again);
    require(std::memcmp(got.data(),again.data(),got.size()*sizeof(float))==0,"repeat determinism");
    if(m==3 || m==65) {
        cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        cudaGraph_t graph;cudaGraphExec_t exec;
        dy.put(alias?res:std::vector<float>(got.size(),-77.f));
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        ck(enqueue(dev,p,work.p,p.workspace_bytes,stream));
        ck(cudaStreamEndCapture(stream,&graph));
        ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));dy.get(again);
        require(std::memcmp(got.data(),again.data(),got.size()*sizeof(float))==0,"stream/graph equality");
        ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));
    }
    double maxerr=0;
    for(unsigned r=0;r<m;++r) for(unsigned c=0;c<ys;++c) {
        const auto i=std::size_t(r)*ys+c;
        if(c>=n) {require(got[i]==(alias?.25f:-77.f),"output padding overwritten");continue;}
        if(exact) require(std::memcmp(&got[i],&ref[i],sizeof(float))==0,"exact dyadic mismatch");
        else {
            double abs_sum=0;
            for(unsigned j=0;j<k;++j) {
                const double u=e2m1(std::uint8_t(a.q[r*a.qs+j/2]>>((j&1)*4)))*e4m3(a.s[r*a.ss+j/16]);
                const double w=e2m1(std::uint8_t(b.q[c*b.qs+j/2]>>((j&1)*4)))*e4m3(b.s[c*b.ss+j/16]);
                abs_sum+=std::fabs(u*w)*.25;
            }
            const double err=std::fabs(double(got[i])-ref[i]);
            if(!test::within_accumulation_tolerance(got[i],ref[i],abs_sum)) {
                std::cerr<<"at "<<r<<','<<c<<" got "<<got[i]<<" expected "<<ref[i]<<" abs_sum "<<abs_sum<<'\n';
                throw std::runtime_error("FP32 accumulation tolerance");
            }
            maxerr=std::max(maxerr,err/std::max(1.,abs_sum));
        }
    }
    auto bad=p; ++bad.splits;
    require(enqueue(dev,bad,work.p,p.workspace_bytes,nullptr)==cudaErrorInvalidValue,"plan rejection");
    if(p.workspace_bytes) require(enqueue(dev,p,nullptr,0,nullptr)==cudaErrorInvalidValue,"workspace rejection");
    std::cout<<"fixture "<<m<<'x'<<n<<'x'<<k<<" exact="<<exact<<" max_error/abs_sum="<<maxerr<<" PASS\n";
}
static float timing(const View& v,const Plan& p,void* work,cudaEvent_t begin,cudaEvent_t end,unsigned iters) {
    ck(cudaEventRecord(begin));
    for(unsigned i=0;i<iters;++i) ck(enqueue(v,p,work,p.workspace_bytes,nullptr));
    ck(cudaEventRecord(end));ck(cudaEventSynchronize(end)); float ms=0;
    ck(cudaEventElapsedTime(&ms,begin,end)); return ms/iters;
}
struct Telemetry {
#ifdef STRIX_NVFP4_NVML
    nvmlDevice_t device=nullptr;
    bool initialized=false;
    Telemetry() {
        initialized=nvmlInit_v2()==NVML_SUCCESS;
        char pci[32]{};
        if(initialized && cudaDeviceGetPCIBusId(pci,sizeof(pci),0)==cudaSuccess)
            if(nvmlDeviceGetHandleByPciBusId_v2(pci,&device)!=NVML_SUCCESS) device=nullptr;
    }
    ~Telemetry() { if(initialized) nvmlShutdown(); }
    void sample(unsigned batch,double ms,unsigned rows) {
        unsigned sm=0,mem=0;nvmlUtilization_t util{};
        if(device && nvmlDeviceGetClockInfo(device,NVML_CLOCK_SM,&sm)==NVML_SUCCESS &&
           nvmlDeviceGetClockInfo(device,NVML_CLOCK_MEM,&mem)==NVML_SUCCESS &&
           nvmlDeviceGetUtilizationRates(device,&util)==NVML_SUCCESS) {
            std::cout<<"TELEMETRY warm_batch="<<batch<<" sm_MHz="<<sm<<" memory_MHz="<<mem
                <<" busy_pct="<<util.gpu<<" busy_clock_cycles_per_row_est="
                <<sm*1000.*ms*(util.gpu/100.)/rows<<'\n';
        } else std::cout<<"TELEMETRY unavailable\n";
    }
#else
    void sample(unsigned,double,unsigned) { std::cout<<"TELEMETRY unavailable (NVML not linked)\n"; }
#endif
};
static void bench(unsigned m,unsigned n,unsigned k) {
    Packed a(m,k,true,false),b(n,k,true,false);
    Buffer<float> out(std::size_t(m)*n);
    View v{a.device(),b.device(),out.p,nullptr,n,0,m,n,k}; const auto p=resolve(v);
    Buffer<std::uint8_t> work(p.workspace_bytes);
    cudaEvent_t start,end;ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
    const float cold=timing(v,p,work.p,start,end,1);
    const float pilot=timing(v,p,work.p,start,end,1);
    const unsigned iterations=std::min(1000u,std::max(2u,unsigned(std::ceil(25.f/std::max(.001f,pilot)))));
    // Capture one complete operator to avoid host submission gaps in tiny decode
    // timings. Capture/instantiation are outside measured GPU intervals.
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    cudaGraph_t graph;cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    ck(enqueue(v,p,work.p,p.workspace_bytes,stream));
    ck(cudaStreamEndCapture(stream,&graph));
    ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    Telemetry telemetry;
    std::vector<float> warm;
    for(unsigned i=0;i<3;++i) {
        ck(cudaEventRecord(start,stream));
        for(unsigned j=0;j<iterations;++j) ck(cudaGraphLaunch(exec,stream));
        ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));float ms=0;
        ck(cudaEventElapsedTime(&ms,start,end));warm.push_back(ms/iterations);
        telemetry.sample(i,warm.back(),m);
    }
    std::sort(warm.begin(),warm.end()); const double ms=warm[1];
    const double bytes=(double(m)+n)*k*.5625+double(m)*n*4;
    std::cout<<std::fixed<<std::setprecision(4)<<"BENCH M="<<m<<" N="<<n<<" K="<<k<<" splits="<<p.splits
       <<" cold_ms="<<cold<<" warm_median_ms="<<ms<<" TFLOPS="<<2.*m*n*k/(ms*1.e9)
       <<" useful_GBps="<<bytes/(ms*1.e6)<<" workspace_bytes="<<p.workspace_bytes
       <<" warm_iterations="<<iterations<<" warm_mode=graph identity="<<kArithmeticId<<'\n';
    ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));
    ck(cudaStreamDestroy(stream));
    ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));
}
int main(int argc,char** argv) try {
    const char* visible=strix::envcfg::text("CUDA_VISIBLE_DEVICES");
    require(visible && std::strcmp(visible,"0")==0,"must set CUDA_VISIBLE_DEVICES=0");
    ck(cudaSetDevice(0));cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));
    require(prop.major==12 && prop.minor==0,"SM120 required");
    std::cout<<"device0="<<prop.name<<" SMs="<<prop.multiProcessorCount<<'\n';
    if(argc==5 && std::strcmp(argv[1],"--bench")==0) {
        bench(unsigned(std::stoul(argv[2])),unsigned(std::stoul(argv[3])),unsigned(std::stoul(argv[4]))); return 0;
    }
    require(argc==1,"usage: nvfp4_device [--bench M N K]");
    codec_test();quantizer_test();
    for(unsigned m=1;m<=8;++m) {
        fixture(m,19,64,true,true);
        fixture(m,67,1088,false,true,m==3);
    }
    for(unsigned m:{9u,17u,65u}) {
        fixture(m,35,128,true,false);
        fixture(m,33,192,false,true,m==17);
    }
    fixture(1,19,1088,true,true);fixture(8,19,1088,true,true);
    fixture(8,17,17408,false,true);fixture(9,3,17408,false,false);
    ck(cudaDeviceSynchronize());std::cout<<"NVFP4 native PASS\n";return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
