#include "cuda/weight/nvfp4_source.cuh"
#include "../nvfp4/tolerance.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace native=strix::cuda::core::nvfp4;
namespace source=strix::cuda::weight;
namespace {
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void check(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
template<class T> struct Buffer {
    T* pointer{};std::size_t count{};
    explicit Buffer(std::size_t size):count(size){if(size)check(cudaMalloc(reinterpret_cast<void**>(&pointer),sizeof(T)*size));}
    ~Buffer(){cudaFree(pointer);}
    void put(const std::vector<T>& values){check(cudaMemcpy(pointer,values.data(),sizeof(T)*count,cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T> result(count);check(cudaMemcpy(result.data(),pointer,sizeof(T)*count,cudaMemcpyDeviceToHost));return result;}
};
void fixture(unsigned m,unsigned n,unsigned k) {
    std::vector<std::uint8_t> aq(m*k/2),as(m*k/16),bq(n*k/2),bs(n*k/16);
    for(std::size_t i=0;i<aq.size();++i)aq[i]=std::uint8_t(i*17+43);
    for(std::size_t i=0;i<bq.size();++i)bq[i]=std::uint8_t(i*31+7);
    for(std::size_t i=0;i<as.size();++i)as[i]=std::uint8_t(48+i%15);
    for(std::size_t i=0;i<bs.size();++i)bs[i]=std::uint8_t(40+i%23);
    std::vector<float> reference(m*n);
    native::View host{{aq.data(),as.data(),k/2,k/16,.5f},{bq.data(),bs.data(),k/2,k/16,1.5f},
        reference.data(),nullptr,n,0,m,n,k};
    const auto plan=native::resolve(host);native::reference(host,plan);
    Buffer<std::uint8_t> daq(aq.size()),das(as.size()),dbq(bq.size()),dbs(bs.size());
    daq.put(aq);das.put(as);dbq.put(bq);dbs.put(bs);
    Buffer<float> old_output(m*n),new_output(m*n);Buffer<std::uint8_t> scratch(plan.workspace_bytes);
    native::View device{{daq.pointer,das.pointer,k/2,k/16,.5f},{dbq.pointer,dbs.pointer,k/2,k/16,1.5f},
        old_output.pointer,nullptr,n,0,m,n,k};
    check(native::enqueue(device,plan,scratch.pointer,plan.workspace_bytes,nullptr));
    device.output=new_output.pointer;
    strix::weight::ExecutionContext context{};context.backend=strix::weight::Backend::cuda;context.device_index=0;
    const source::Nvfp4Arguments args{device,plan};
    require(bool(source::enqueue_nvfp4(args,scratch.pointer,plan.workspace_bytes,context)),"shared source launch");
    const auto old=old_output.get(),got=new_output.get();
    require(std::memcmp(old.data(),got.data(),old.size()*sizeof(float))==0,"legacy native outputs differ through shared source");
    for(unsigned row=0;row<m;++row)for(unsigned col=0;col<n;++col) {
        double abs_sum=0;
        for(unsigned x=0;x<k;++x) {
            const auto a=native::e2m1((aq[row*k/2+x/2]>>(4*(x%2)))&15)*native::e4m3(as[row*k/16+x/16])*.5f;
            const auto b=native::e2m1((bq[col*k/2+x/2]>>(4*(x%2)))&15)*native::e4m3(bs[col*k/16+x/16])*1.5f;
            abs_sum+=std::fabs(double(a)*b);
        }
        require(native::test::within_accumulation_tolerance(got[row*n+col],reference[row*n+col],abs_sum),"native FP4 scalar tolerance");
    }
    context.backend=strix::weight::Backend::hip;
    require(source::enqueue_nvfp4(args,scratch.pointer,plan.workspace_bytes,context).code==strix::weight::Error::unsupported_backend,
        "unqualified backend must reject native FP4");
}
}
namespace strix::weight::test {
bool cuda_nvfp4_case() try {
    int current=-1;check(cudaGetDevice(&current));require(current==0,"native FP4 fixture requires device index zero");
    int major=0,minor=0;check(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,0));
    check(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,0));
    if(major!=12 || minor!=0) {
        source::Nvfp4Arguments args{};strix::weight::ExecutionContext context{};context.backend=strix::weight::Backend::cuda;
        require(source::enqueue_nvfp4(args,nullptr,0,context).code==strix::weight::Error::unavailable_isa,
            "non-SM120 launch must return unavailable_isa");
        std::cerr<<"non-SM120 rejection PASS; positive SM120 qualification is mandatory separately\n";return false;
    }
    fixture(1,7,64);fixture(3,65,128);fixture(9,33,64);fixture(65,65,128);
    std::cout<<"native FP4 shared source: legacy_bitwise forwarding and FAST scalar tolerance PASS\n";return true;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return false;}
}
#ifdef STRIX_WFMT_NVFP4_STANDALONE
int main(){return strix::weight::test::cuda_nvfp4_case()?0:1;}
#endif
