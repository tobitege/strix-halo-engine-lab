#include "../kda.cuh"
#include "reference.hpp"
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <chrono>

using namespace strix::glm::rtxpf;
using namespace strix::glm::rtxpf::bench;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
struct Buffer {
    void* p=nullptr;explicit Buffer(std::size_t n){ck(cudaMalloc(&p,n));}
    ~Buffer(){if(p)cudaFree(p);}Buffer(const Buffer&)=delete;
    template<class T>T* as()const{return static_cast<T*>(p);}
};
template<class T>void read(std::istream& f,T* p,std::size_t n){need(bool(f.read(reinterpret_cast<char*>(p),n*sizeof(T))),"short archive");}
struct Tensor {std::string dtype;std::vector<unsigned> shape;std::vector<char> data;};
struct Weights {
    unsigned layer,h,heads;KdaWeights w{};
    std::map<std::string,Tensor> tensors;
    std::vector<std::unique_ptr<Buffer>> storage;
    Weights(const std::string& path){
        std::ifstream f(path,std::ios::binary);need(bool(f),"open attention archive");char magic[8];read(f,magic,8);need(std::string(magic,8)=="GLMATT01","attention archive magic");
        unsigned count;read(f,&layer,1);read(f,&h,1);read(f,&heads,1);read(f,&count,1);
        need(h==4096&&heads==64&&count==17,"KDA archive geometry/count");
        for(unsigned j=0;j<count;++j){unsigned len,nd;read(f,&len,1);need(len>0&&len<=128,"tensor name length");std::string name(len,' ');read(f,name.data(),len);
            char dt[4];read(f,dt,4);read(f,&nd,1);need(nd<=3,"tensor rank");Tensor t;t.dtype=std::string(dt,std::find(dt,dt+4,'\0'));t.shape.resize(nd);read(f,t.shape.data(),nd);
            std::uint64_t bytes,n=1;read(f,&bytes,1);for(unsigned extent:t.shape){need(extent>0&&extent<=32768,"tensor extent");n*=extent;}
            need(t.dtype=="I16"||t.dtype=="I32"||t.dtype=="F16"||t.dtype=="BF16"||t.dtype=="F32","tensor dtype");
            need(bytes==n*((t.dtype=="I32"||t.dtype=="F32")?4:2)&&bytes<=256*1024*1024,"tensor byte count");
            t.data.resize(std::size_t(bytes));read(f,t.data.data(),t.data.size());need(tensors.emplace(name,std::move(t)).second,"duplicate tensor");}
        need(f.peek()==std::char_traits<char>::eof(),"archive trailing bytes");
        auto get=[&](const std::string& name,const std::string& dtype,std::vector<unsigned> shape)->const Tensor&{
            auto it=tensors.find(name);need(it!=tensors.end(),"missing KDA tensor");const auto& t=it->second;need(t.dtype==dtype&&t.shape==shape,"KDA tensor shape/type");
            if(dtype=="F16"||dtype=="BF16"){auto p=reinterpret_cast<const U16*>(t.data.data());unsigned mask=dtype=="F16"?0x7c00:0x7f80;for(std::size_t j=0;j<t.data.size()/2;++j)need((p[j]&mask)!=mask,"nonfinite weights");}
            if(dtype=="F32"){auto p=reinterpret_cast<const float*>(t.data.data());for(std::size_t j=0;j<t.data.size()/4;++j)need(std::isfinite(p[j]),"nonfinite weights");}return t;};
        // Complete host validation before ANY CUDA API, including allocations.
        unsigned c=heads*128;
        for(auto role:{std::string("qkv_proj"),std::string("o_proj")}){unsigned in=role=="qkv_proj"?h:c,out=role=="qkv_proj"?3*c:h;
            const auto& marker=get(role+".mul1","I32",{});U32 value;std::memcpy(&value,marker.data.data(),4);need(value==0x83dcd12dU,"mul1 marker");
            auto it=tensors.find(role+".trellis");need(it!=tensors.end()&&it->second.shape.size()==3,"trellis rank");unsigned bits=it->second.shape[2]/16;need(bits>=3&&bits<=6,"trellis rate");
            get(role+".trellis","I16",{in/16,out/16,16*bits});get(role+".suh","F16",{in});get(role+".svh","F16",{out});}
        get("b_proj.weight","F16",{heads,h});get("f_a_proj.weight","F16",{128,h});get("g_a_proj.weight","F16",{128,h});
        get("f_b_proj.weight","F16",{c,128});get("g_b_proj.weight","F16",{c,128});get("conv1d.weight","BF16",{3*c,1,4});get("o_norm.weight","BF16",{128});get("A_log","F32",{heads});get("dt_bias","F32",{c});
        auto put=[&](const std::string& name){auto& t=tensors.at(name);auto b=std::make_unique<Buffer>(t.data.size());ck(cudaMemcpy(b->p,t.data.data(),t.data.size(),cudaMemcpyHostToDevice));void* p=b->p;storage.push_back(std::move(b));return p;};
        auto matrix=[&](const std::string& name,unsigned in,unsigned out){return Matrix{static_cast<U16*>(put(name+".trellis")),static_cast<U16*>(put(name+".suh")),static_cast<U16*>(put(name+".svh")),in,out,tensors.at(name+".trellis").shape[2]/16};};
        w={h,heads,matrix("qkv_proj",h,3*c),matrix("o_proj",c,h),static_cast<U16*>(put("b_proj.weight")),static_cast<U16*>(put("f_a_proj.weight")),static_cast<U16*>(put("f_b_proj.weight")),static_cast<U16*>(put("g_a_proj.weight")),static_cast<U16*>(put("g_b_proj.weight")),static_cast<U16*>(put("conv1d.weight")),static_cast<U16*>(put("o_norm.weight")),static_cast<float*>(put("A_log")),static_cast<float*>(put("dt_bias"))};
    }
};
std::vector<float> download(Buffer& b,std::size_t n){std::vector<float> out(n);ck(cudaMemcpy(out.data(),b.p,n*4,cudaMemcpyDeviceToHost));return out;}
int main(int argc,char** argv){try{
    std::string path;unsigned rows=0,iters=1;bool allow=false;
    for(int j=1;j<argc;++j){std::string a=argv[j];if(a=="--allow-gpu")allow=true;else if(a=="--weights"&&j+1<argc)path=argv[++j];else if(a=="--rows"&&j+1<argc)rows=unsigned(std::stoul(argv[++j]));else if(a=="--iters"&&j+1<argc)iters=unsigned(std::stoul(argv[++j]));else throw std::runtime_error("usage: --weights FILE --rows C --iters N --allow-gpu");}
    need(!path.empty()&&rows>0&&rows<=32768&&iters>0&&iters<=10,"weights/rows/iters required");
    if(!allow){std::puts("GPU execution disabled; pass --allow-gpu in an authorized window");return 2;}
    Weights weights(path);ck(prepare_kda());auto w=weights.w;std::size_t ny=std::size_t(rows)*w.hidden,ns=std::size_t(w.heads)*128*128,ncv=std::size_t(w.heads)*128*9;
    std::mt19937 rng(20261027);auto random=[&](std::size_t n,float scale){std::vector<float> a(n);for(auto& x:a)x=float(int(rng()%2001)-1000)/1000*scale;return a;};
    auto x=random(ny,1.7320508f),s0=random(ns,0.001f),cv0=random(ncv,0.01f);
    Buffer input(ny*4),y(ny*4),state(ns*4),cv(ncv*4),workspace(kda_workspace_bytes(rows,w.hidden,w.heads));
    ck(cudaMemcpy(input.p,x.data(),ny*4,cudaMemcpyHostToDevice));
    auto reset=[&](){ck(cudaMemcpy(state.p,s0.data(),ns*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(cv.p,cv0.data(),ncv*4,cudaMemcpyHostToDevice));};
    KdaCall call{rows,w,input.as<float>(),y.as<float>(),state.as<float>(),cv.as<float>()};
    cudaEvent_t begin,end;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
    auto run=[&](bool ref){reset();auto start=std::chrono::steady_clock::now();ck(cudaEventRecord(begin));ck(enqueue_kda(call,workspace.p,kda_workspace_bytes(rows,w.hidden,w.heads),nullptr,ref));ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));
        double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();float ms;ck(cudaEventElapsedTime(&ms,begin,end));return std::pair<double,double>{ms,wall};};
    // Warmup is excluded from timings. Every invocation starts from identical
    // nonzero state/carry. Decode is once per layer and INCLUDED in all times.
    run(false);auto expected=download(y,ny),es=download(state,ns),ec=download(cv,ncv);
    std::vector<double> times,walls;
    double c=w.heads*128.,linear=rows*(4.*w.hidden*c+2.*w.hidden*128+2.*128*c+double(w.hidden)*w.heads),recurrence=double(rows)*3*w.heads*128*128;
    for(unsigned j=0;j<iters;++j){auto [ms,wall]=run(false);times.push_back(ms);walls.push_back(wall);auto actual=download(y,ny),as=download(state,ns),ac=download(cv,ncv);
        need(std::memcmp(actual.data(),expected.data(),ny*4)==0&&std::memcmp(as.data(),es.data(),ns*4)==0&&std::memcmp(ac.data(),ec.data(),ncv*4)==0,"native repeated output/state nondeterministic");}
    std::sort(times.begin(),times.end());std::sort(walls.begin(),walls.end());double ms=times[times.size()/2];
    std::printf("identity=%s layer=%u C=%u iterations=%u input=synthetic_normalized seed=20261027 initial_state=nonzero decode=included workspace_bytes=%zu device_ms=%.6f wall_ms=%.6f useful_projection_TMAC_s=%.6f useful_layer_TMAC_s=%.6f useful_projection_MAC=%.0f useful_recurrence_MAC=%.0f\n",kda_identity,weights.layer,rows,iters,kda_workspace_bytes(rows,w.hidden,w.heads),ms,walls[walls.size()/2],linear/(ms*1.e9),(linear+recurrence)/(ms*1.e9),linear,recurrence);
    auto [refms,refwall]=run(true);auto reference=download(y,ny),rs=download(state,ns),rc=download(cv,ncv);
    metrics(expected.data(),reference.data(),ny,"KDA_layer_output");metrics(es.data(),rs.data(),ns,"KDA_exported_S");metrics(ec.data(),rc.data(),ncv,"KDA_exported_conv");
    std::printf("reference=FP32_sequential_full_layer device_ms=%.6f wall_ms=%.6f determinism=PASS qualification=DRAFT\n",refms,refwall);
    ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
