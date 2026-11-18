#include "../native.cuh"
#include "reference.hpp"
#include <fstream>
#include <random>
#include <string>
#include <numeric>
#include <chrono>
#include <memory>
#include <cstring>

namespace n = strix::glm::rtxpf;
using namespace n;
using namespace n::bench;
void ck(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
void need(bool b,const char* s) { if(!b)throw std::runtime_error(s); }
struct Buffer {
    void* p=nullptr;
    explicit Buffer(std::size_t bytes) { ck(cudaMalloc(&p,bytes)); }
    ~Buffer() { if(p)cudaFree(p); }
    Buffer(const Buffer&)=delete;
    template<class T>T* as()const {return static_cast<T*>(p);}
};
template<class T>void read(std::istream& f,T* p,std::size_t count) {need(bool(f.read(reinterpret_cast<char*>(p),count*sizeof(T))),"short input");}
template<class T>void upload(T* p,const std::vector<T>& v) {ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
struct Weights {
    unsigned h=4096,i=2048,e=288;
    std::vector<HostExpert> host;
    std::vector<Expert> device;
    std::vector<std::unique_ptr<Buffer>> storage;
    std::unique_ptr<Buffer> descriptors;
    Matrix put(const HostMatrix& m) {
        auto alloc=[&](const std::vector<U16>& v) {auto b=std::make_unique<Buffer>(v.size()*2);auto p=b->as<U16>();upload(p,v);storage.push_back(std::move(b));return p;};
        return {alloc(m.packed),alloc(m.suh),alloc(m.svh),m.input,m.output,m.bits};
    }
    Weights(const std::string& path,const std::string& kind,bool smoke) {
        std::ifstream file;
        if(!path.empty()) {
            file.open(path,std::ios::binary);need(bool(file),"weights archive open");char magic[8];read(file,magic,8);need(std::string(magic,8)=="GLMNAT01","archive magic");
            read(file,&e,1);read(file,&h,1);read(file,&i,1);
        } else { e=kind=="routed"?288:1;i=kind=="dense"?12288:2048;if(smoke){e=kind=="routed"?33:1;h=i=128;} }
        need((e==1||(e>=8&&e<=288))&&h>=128&&h<=16384&&i>=128&&i<=16384&&h%128==0&&i%128==0,"archive dimensions");
        std::mt19937 rng(20261027);
        auto make=[&](unsigned in,unsigned out) {
            unsigned bits=kind=="dense"?5:kind=="shared"?6:4;
            if(file.is_open())read(file,&bits,1);
            need(bits>=3&&bits<=6,"archive rate");
            HostMatrix m{in,out,bits};m.packed.resize(std::size_t(in)*out*bits/16);m.suh.resize(in);m.svh.resize(out);
            if(file.is_open()) {read(file,m.packed.data(),m.packed.size());read(file,m.suh.data(),in);read(file,m.svh.data(),out);}
            else {for(auto& x:m.packed)x=U16(rng());for(auto& x:m.suh)x=fh((rng()&1)?0.125f:-0.125f);for(auto& x:m.svh)x=fh((rng()&1)?0.125f:-0.125f);}
            for(U16 x:m.suh)need(std::isfinite(hf(x)),"nonfinite suh");for(U16 x:m.svh)need(std::isfinite(hf(x)),"nonfinite svh");
            return m;
        };
        // Check exact archive length before allocations/uploads.
        host.reserve(e);device.reserve(e);
        for(unsigned j=0;j<e;++j)host.push_back({make(h,i),make(h,i),make(i,h)});
        if(file.is_open())need(file.peek()==std::char_traits<char>::eof(),"archive trailing bytes");
        for(const auto& x:host)device.push_back({put(x.gate),put(x.up),put(x.down)});
        descriptors=std::make_unique<Buffer>(device.size()*sizeof(Expert));upload(descriptors->as<Expert>(),device);
    }
};
struct Events {
    Profile p{};cudaEvent_t begin,end;unsigned groups;
    explicit Events(unsigned e):groups((e+15)/16) {
        ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
        for(unsigned j=0;j<groups;++j)for(auto ptr:{&p.begin[j],&p.decoded[j],&p.gate_up[j],&p.down[j],
            &p.decode_start[j],&p.ready[j],&p.input[j],&p.gu_start[j],&p.gu_gemm[j],&p.swiglu[j],&p.down_input[j],&p.down_start[j],&p.down_gemm[j]})ck(cudaEventCreate(ptr));
        ck(cudaEventCreate(&p.merge_begin));ck(cudaEventCreate(&p.merge_end));
    }
    ~Events(){cudaEventDestroy(begin);cudaEventDestroy(end);for(unsigned j=0;j<groups;++j)for(auto v:{p.begin[j],p.decoded[j],p.gate_up[j],p.down[j],p.decode_start[j],p.ready[j],p.input[j],p.gu_start[j],p.gu_gemm[j],p.swiglu[j],p.down_input[j],p.down_start[j],p.down_gemm[j]})cudaEventDestroy(v);cudaEventDestroy(p.merge_begin);cudaEventDestroy(p.merge_end);}
    float elapsed(cudaEvent_t a,cudaEvent_t b){float t;ck(cudaEventElapsedTime(&t,a,b));return t;}
};
struct Execution {
    CutlassExecution e{};
    explicit Execution(bool enabled) {
        if(!enabled)return;
        ck(cudaStreamCreateWithFlags(&e.decode_stream,cudaStreamNonBlocking));
        ck(cudaEventCreate(&e.dispatch));
        for(unsigned j=0;j<18;++j){ck(cudaEventCreate(&e.ready[j]));ck(cudaEventCreate(&e.retired[j]));}
        ck(prepare_cutlass(e));
    }
    ~Execution(){if(e.decode_stream){cudaStreamDestroy(e.decode_stream);cudaEventDestroy(e.dispatch);
        for(unsigned j=0;j<18;++j){cudaEventDestroy(e.ready[j]);cudaEventDestroy(e.retired[j]);}}}
};
struct Opt {
    std::string weights,histogram,routes,input,kind="routed",path="legacy";
    unsigned iters=3,rows=0,batches=0;bool smoke=false,run=false;
};
void run_bench(const Opt& o) {
    Weights w(o.weights,o.kind,o.smoke);
    bool c1=o.path=="cutlass"; Execution execution(c1);
    need(o.kind=="routed"?w.e>=8:w.e==1,"archive expert count differs from requested kind");
    std::printf("source=%s kind=%s experts=%u H=%u I=%u identity=%s reference=scalar-FP32-fma exact-decoded-FP16\n",o.weights.empty()?"synthetic":"model-archive",o.kind.c_str(),w.e,w.h,w.i,c1?cutlass_identity:identity);
    unsigned routes=w.e==1?1:8;
    std::mt19937 rng(20261027);
    for(unsigned C:o.smoke?std::vector<unsigned>{1,17,257}:o.rows?std::vector<unsigned>{o.rows}:std::vector<unsigned>{1024,8192,32768}) {
        std::vector<float> x(std::size_t(C)*w.h);
        std::normal_distribution<float> normal(0,0.05f);for(auto& v:x)v=normal(rng);
        if(!o.input.empty()){std::ifstream f(o.input,std::ios::binary);read(f,x.data(),x.size());need(f.peek()==EOF,"input trailing bytes (use one shape per file)");}
        Buffer xd(x.size()*4),yd(x.size()*4);upload(xd.as<float>(),x);
        std::size_t bytes=c1?workspace_bytes_cutlass(C,w.e,w.h,w.i):routes==1?workspace_bytes_dense(C,w.h,w.i):workspace_bytes(C,w.e,w.h,w.i);Buffer ws(bytes);
        Buffer idd(std::size_t(C)*routes*4),wd(std::size_t(C)*routes*4);
        std::vector<float> weights(std::size_t(C)*routes,routes==1?1.f:2.5f/routes);
        std::vector<U32> ids(std::size_t(C)*routes);
        for(unsigned distribution=0;distribution<(routes==1?1:2);++distribution) {
            std::vector<double> popularity(w.e,1.0);
            if(distribution) {std::lognormal_distribution<double> ln(0,0.6);for(auto& v:popularity)v=ln(rng);}
            if(!o.histogram.empty()) {std::ifstream f(o.histogram,std::ios::binary);read(f,popularity.data(),popularity.size());need(f.peek()==EOF,"histogram trailing bytes");}
            unsigned positive=0;for(double v:popularity){need(std::isfinite(v)&&v>=0,"histogram finite/nonnegative");positive+=v>0;}need(positive>=routes,"not enough distinct experts");
            std::discrete_distribution<unsigned> pick(popularity.begin(),popularity.end());
            for(unsigned r=0;r<C;++r)for(unsigned s=0;s<routes;++s) {
                unsigned e;bool duplicate;
                do {e=pick(rng);duplicate=false;for(unsigned t=0;t<s;++t)duplicate|=ids[r*routes+t]==e;}while(duplicate);
                ids[r*routes+s]=e;
            }
            if(!o.routes.empty()){std::ifstream f(o.routes,std::ios::binary);read(f,ids.data(),ids.size());read(f,weights.data(),weights.size());need(f.peek()==EOF,"routes trailing bytes");}
            for(unsigned r=0;r<C;++r)for(unsigned s=0;s<routes;++s){need(ids[r*routes+s]<w.e && std::isfinite(weights[r*routes+s]),"invalid route");for(unsigned t=0;t<s;++t)need(ids[r*routes+s]!=ids[r*routes+t],"duplicate route");}
            upload(idd.as<U32>(),ids);upload(wd.as<float>(),weights);
            Call call{C,w.e,w.h,w.i,xd.as<float>(),idd.as<U32>(),wd.as<float>(),w.descriptors->as<Expert>(),yd.as<float>()};
            Events ev(w.e);
            auto enqueue_call=[&](bool profile){if(c1)ck(enqueue_cutlass(call,ws.p,bytes,0,execution.e,profile?&ev.p:nullptr));else if(routes==1)ck(enqueue_dense(C,w.h,w.i,call.x,call.matrices,call.y,ws.p,bytes,0,profile?&ev.p:nullptr));else ck(enqueue(call,ws.p,bytes,0,profile?&ev.p:nullptr));};
            enqueue_call(false);ck(cudaDeviceSynchronize());
            if(o.batches){
                // Expert-batch exactness (prefill through a small cache arena): the 288 experts in
                // o.batches whole-expert batches, each with only its experts' descriptors (the
                // others zeroed, never read) and its routes (others masked to 0xFFFFFFFF), over
                // the same rows and workspace, merging only after the last batch, must give the
                // full call's output BITWISE; the full call is also repeated (determinism).
                std::vector<float> full(x.size()),again(x.size()),batched(x.size());
                ck(cudaMemcpy(full.data(),yd.p,full.size()*4,cudaMemcpyDeviceToHost));
                enqueue_call(false);ck(cudaDeviceSynchronize());ck(cudaMemcpy(again.data(),yd.p,again.size()*4,cudaMemcpyDeviceToHost));
                need(std::memcmp(full.data(),again.data(),full.size()*4)==0,"full routed call is not bitwise deterministic");
                Buffer masked(std::size_t(C)*routes*4),desc(std::size_t(w.e)*sizeof(Expert));ck(cudaMemset(yd.p,0xff,x.size()*4));
                for(unsigned b=0;b<o.batches;++b){
                    std::vector<Expert> part(w.e);std::vector<U32> mid(ids.size());
                    for(unsigned e=0;e<w.e;++e)if(e%o.batches==b)part[e]=w.device[e];else std::memset(&part[e],0,sizeof(Expert));
                    for(std::size_t k=0;k<ids.size();++k)mid[k]=ids[k]%o.batches==b?ids[k]:0xFFFFFFFFu;
                    upload(desc.as<Expert>(),part);upload(masked.as<U32>(),mid);
                    Call cb=call;cb.ids=masked.as<U32>();cb.matrices=desc.as<Expert>();cb.combine=b+1==o.batches;
                    ck(enqueue_cutlass(cb,ws.p,bytes,0,execution.e,nullptr));ck(cudaDeviceSynchronize());
                }
                ck(cudaMemcpy(batched.data(),yd.p,batched.size()*4,cudaMemcpyDeviceToHost));
                need(std::memcmp(full.data(),batched.data(),full.size()*4)==0,"expert-batched routed output differs from the full call");
                std::printf("BATCH-EXACT rows=%u distribution=%u batches=%u: batched routed output == full call bitwise (full call deterministic)\n",C,distribution,o.batches);
                continue;
            }
            std::vector<double> times;double dec=0,gu=0,dn=0,exposed=0,decode_exposed=0,input_ms=0,gu_ms=0,epi_ms=0,dt_ms=0,dg_ms=0,oe_ms=0,merge_ms=0,problem_ms=0;
            for(unsigned iter=0;iter<o.iters;++iter) {
                ck(cudaEventRecord(ev.begin));enqueue_call(true);ck(cudaEventRecord(ev.end));ck(cudaEventSynchronize(ev.end));
                times.push_back(ev.elapsed(ev.begin,ev.end));
                for(unsigned g=0;g<ev.groups;++g) {
                    if(c1) {
                        double decode_time=ev.elapsed(ev.p.decode_start[g],ev.p.decoded[g]);
                        dec+=decode_time;
                        decode_exposed+=std::clamp(double(ev.elapsed(ev.p.input[g],ev.p.decoded[g])),0.0,decode_time);
                        exposed+=ev.elapsed(ev.p.input[g],ev.p.ready[g]);
                        input_ms+=ev.elapsed(ev.p.begin[g],ev.p.input[g]);
                        gu_ms+=ev.elapsed(ev.p.gu_start[g],ev.p.gu_gemm[g]);
                        problem_ms+=ev.elapsed(ev.p.ready[g],ev.p.gu_start[g]);
                        epi_ms+=ev.elapsed(ev.p.gu_gemm[g],ev.p.swiglu[g]);
                        dt_ms+=ev.elapsed(ev.p.swiglu[g],ev.p.down_input[g]);
                        dg_ms+=ev.elapsed(ev.p.down_start[g],ev.p.down_gemm[g]);
                        problem_ms+=ev.elapsed(ev.p.down_input[g],ev.p.down_start[g]);
                        oe_ms+=ev.elapsed(ev.p.down_gemm[g],ev.p.down[g]);
                        gu+=ev.elapsed(ev.p.begin[g],ev.p.input[g])+ev.elapsed(ev.p.ready[g],ev.p.swiglu[g]);
                        dn+=ev.elapsed(ev.p.swiglu[g],ev.p.down[g]);
                    } else {dec+=ev.elapsed(ev.p.begin[g],ev.p.decoded[g]);gu+=ev.elapsed(ev.p.decoded[g],ev.p.gate_up[g]);dn+=ev.elapsed(ev.p.gate_up[g],ev.p.down[g]);}
                }
                if(c1) {
                    merge_ms+=ev.elapsed(ev.p.merge_begin,ev.p.merge_end);
                    // All events have completed. Intersect timestamps on the common
                    // device clock, rather than calling all hidden time GEMM overlap.
                    auto stamp=[&](cudaEvent_t t){return double(ev.elapsed(ev.begin,t));};
                    std::vector<unsigned> counts(w.e);
                    for(auto id:ids)++counts[id];
                    for(unsigned g=0;g<ev.groups;++g) {
                        unsigned first=g*16,last=std::min(first+16,w.e),rates[7]{},empty=0;
                        std::size_t in=0,out=0,skipped_in=0,skipped_out=0;
                        for(unsigned j=first;j<last;++j) {
                            bool skip=counts[j]==0;empty+=skip;
                            for(const auto* m:{&w.host[j].gate,&w.host[j].up,&w.host[j].down}) {
                                std::size_t ni=m->packed.size()*2,no=std::size_t(m->input)*m->output*2;
                                if(skip){skipped_in+=ni;skipped_out+=no;}
                                else {in+=ni;out+=no;++rates[m->bits];}
                            }
                        }
                        double a=stamp(ev.p.decode_start[g]),b=stamp(ev.p.decoded[g]);
                        double overlap=0;
                        for(unsigned j=0;j<ev.groups;++j) {
                            for(auto pair:{std::pair{ev.p.gu_start[j],ev.p.gu_gemm[j]},
                                           std::pair{ev.p.down_start[j],ev.p.down_gemm[j]}})
                                overlap+=std::max(0.0,std::min(b,stamp(pair.second))-std::max(a,stamp(pair.first)));
                        }
                        double wait=ev.elapsed(ev.p.input[g],ev.p.ready[g]);
                        std::printf("decode_window iter=%u window=%u experts=%u empty=%u K3=%u K4=%u K5=%u K6=%u decode_ms=%.6f bytes_in=%zu bytes_out=%zu logical_GB_s=%.6f GEMM_overlap_ms=%.6f exposed_wait_ms=%.6f skipped_bytes_in=%zu skipped_bytes_out=%zu\n",
                            iter,g,last-first,empty,rates[3],rates[4],rates[5],rates[6],b-a,in,out,
                            b>a?(in+out)/((b-a)*1e6):0.0,overlap,wait,skipped_in,skipped_out);
                    }
                }
            }
            std::sort(times.begin(),times.end());double ms=times[times.size()/2];
            double mac=double(C)*routes*3*w.h*w.i;
            std::printf("C=%u routing=%s workspace_bytes=%zu full_moe_median_ms=%.6f useful_TMAC_s=%.6f decode_ms=%.6f gate_up_transform_mma_epilogue_ms=%.6f down_transform_mma_epilogue_ms=%.6f\n",C,!o.routes.empty()?"captured":!o.histogram.empty()?"captured-histogram-resampled":distribution?"lognormal-0.6":"uniform",bytes,ms,mac/(ms*1e9),dec/o.iters,gu/o.iters,dn/o.iters);
            std::printf("grouped_useful_TMAC_s=%.6f (includes transforms/epilogues, excludes decode/dispatch/merge)\n",mac/((gu+dn)/o.iters*1e9));
            if(c1) {
                std::printf("input_transform_ms=%.6f gate_up_GEMM_ms=%.6f epilogue_SwiGLU_round_ms=%.6f down_transform_ms=%.6f down_GEMM_ms=%.6f output_epilogue_ms=%.6f merge_ms=%.6f problem_list_ms=%.6f decode_exposed_wait_ms=%.6f decode_exposed_estimate_ms=%.6f decode_hidden_estimate_ms=%.6f GEMM_useful_TMAC_s=%.6f\n",
                    input_ms/o.iters,gu_ms/o.iters,epi_ms/o.iters,dt_ms/o.iters,dg_ms/o.iters,oe_ms/o.iters,merge_ms/o.iters,problem_ms/o.iters,exposed/o.iters,decode_exposed/o.iters,std::max(0.0,dec-decode_exposed)/o.iters,mac/((gu_ms+dg_ms)/o.iters*1e9));
            }
            std::vector<float> y(x.size()),input(x.begin(),x.begin()+w.h);
            ck(cudaMemcpy(y.data(),yd.p,y.size()*4,cudaMemcpyDeviceToHost));
            auto ref=moe(input,w.host,routes==8?ids.data():nullptr,routes==8?weights.data():nullptr,routes);
            metrics(y.data(),ref.data(),w.h,"CPU-scalar-full-row-0");
            // Same inputs twice: a deterministic-merge diagnostic, independent
            // of the statistical comparison with the scalar arithmetic.
            enqueue_call(false);ck(cudaDeviceSynchronize());std::vector<float> again(x.size());ck(cudaMemcpy(again.data(),yd.p,again.size()*4,cudaMemcpyDeviceToHost));
            need(std::memcmp(y.data(),again.data(),y.size()*4)==0,"repeat output differs");
            // All outputs against SIMT chronological FP32 FMA accumulation.
            // These diagnostic kernels contain no WMMA/HMMA instructions.
            ck(enqueue_reference(call,ws.p,bytes,0));ck(cudaDeviceSynchronize());
            ck(cudaMemcpy(again.data(),yd.p,again.size()*4,cudaMemcpyDeviceToHost));
            metrics(y.data(),again.data(),y.size(),"SIMT-scalar-all-rows");
            metrics(again.data(),ref.data(),w.h,"SIMT-vs-CPU-full-row-0");
        }
    }
}
int main(int argc,char** argv) {
    try {
        Opt o;
        for(int j=1;j<argc;++j) {
            std::string a=argv[j];
            auto next=[&](){need(j+1<argc,"missing option value");return std::string(argv[++j]);};
            if(a=="--run")o.run=true;else if(a=="--smoke")o.smoke=true;
            else if(a=="--weights")o.weights=next();else if(a=="--histogram")o.histogram=next();
            else if(a=="--routes")o.routes=next();else if(a=="--input")o.input=next();
            else if(a=="--path")o.path=next();
            else if(a=="--kind")o.kind=next();else if(a=="--iters")o.iters=unsigned(std::stoul(next()));
            else if(a=="--rows")o.rows=unsigned(std::stoul(next()));
            else if(a=="--batch-check")o.batches=unsigned(std::stoul(next()));
            else throw std::runtime_error("usage: glm_native_bench --run [--path legacy|cutlass] [--weights ARCHIVE] [--kind routed|shared|dense] [--iters 1..10] [--rows 1024|8192|32768] [--smoke] [--histogram F64] [--routes IDS_THEN_WEIGHTS] [--input F32] [--batch-check 2..32]");
        }
        need(o.path=="legacy"||o.path=="cutlass","invalid path");
        need(o.run,"GPU execution requires --run; use host_tests.exe for CPU-only verification");
        need(o.iters>0&&o.iters<=10,"iterations must be 1..10");
        need(o.kind=="routed"||o.kind=="shared"||o.kind=="dense","invalid kind");
        need(!o.rows||o.rows==1024||o.rows==8192||o.rows==32768,"unsupported row count");
        need(!o.batches||(o.batches>=2&&o.batches<=32&&o.path=="cutlass"&&o.kind=="routed"),"--batch-check B: 2..32 batches, --path cutlass --kind routed");
        need((o.routes.empty()&&o.input.empty())||o.rows,"captured inputs/routes require --rows");
        int count=0;ck(cudaGetDeviceCount(&count));need(count==1,"expose exactly the intended PRO via CUDA_VISIBLE_DEVICES");
        cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,0));need(p.major==12&&p.minor==0&&std::string(p.name).find("RTX PRO 6000")!=std::string::npos,"requires sm_120 RTX PRO 6000");
        ck(cudaSetDevice(0));std::printf("device=%s sm=%d%d\n",p.name,p.major,p.minor);
        run_bench(o);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
