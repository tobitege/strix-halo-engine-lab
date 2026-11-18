#include "../dsa.cuh"
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
    unsigned layer,h,heads;DsaWeights w{};
    std::map<std::string,Tensor> tensors;
    std::vector<std::unique_ptr<Buffer>> storage;
    Weights(const std::string& path){
        std::ifstream f(path,std::ios::binary);need(bool(f),"open attention archive");char magic[8];read(f,magic,8);need(std::string(magic,8)=="GLMATT01","attention archive magic");
        unsigned count;read(f,&layer,1);read(f,&h,1);read(f,&heads,1);read(f,&count,1);
        need(layer==7&&h==4096&&heads==64&&count==29,"DSA archive geometry/count");
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
        struct Shape {const char* role;unsigned in,out;};
        const Shape shapes[]={{"q_a_proj",4096,1536},{"q_b_proj",1536,16384},{"kv_a_proj_with_mqa",4096,512},{"o_proj",16384,4096},{"indexer.wq_b",1536,4096}};
        for(auto item:shapes){std::string role=item.role;auto in=item.in,out=item.out;
            const auto& marker=get(role+".mul1","I32",{});U32 value;std::memcpy(&value,marker.data.data(),4);need(value==0x83dcd12dU,"mul1 marker");
            auto it=tensors.find(role+".trellis");need(it!=tensors.end()&&it->second.shape.size()==3,"trellis rank");unsigned bits=it->second.shape[2]/16;need(bits>=3&&bits<=6,"trellis rate");
            get(role+".trellis","I16",{in/16,out/16,16*bits});get(role+".suh","F16",{in});get(role+".svh","F16",{out});}
        get("q_a_layernorm.weight","BF16",{1536});get("kv_a_layernorm.weight","BF16",{512});get("kv_b_proj.weight","F16",{32768,512});
        get("indexer.wk.weight","F16",{128,4096});get("indexer.index_kpool_compress_gate","F16",{128,4096});get("indexer.weights_proj.weight","F16",{32,4096});
        get("indexer.k_norm.weight","F16",{128});get("indexer.k_norm.bias","F16",{128});get("indexer.index_kpool_compress_ape","F32",{4,128});
        auto put=[&](const std::string& name){auto& t=tensors.at(name);auto b=std::make_unique<Buffer>(t.data.size());ck(cudaMemcpy(b->p,t.data.data(),t.data.size(),cudaMemcpyHostToDevice));void* p=b->p;storage.push_back(std::move(b));return p;};
        auto matrix=[&](const std::string& name,unsigned in,unsigned out){return Matrix{static_cast<U16*>(put(name+".trellis")),static_cast<U16*>(put(name+".suh")),static_cast<U16*>(put(name+".svh")),in,out,tensors.at(name+".trellis").shape[2]/16};};
        w={matrix("q_a_proj",4096,1536),matrix("q_b_proj",1536,16384),matrix("kv_a_proj_with_mqa",4096,512),matrix("o_proj",16384,4096),matrix("indexer.wq_b",1536,4096),
            static_cast<U16*>(put("q_a_layernorm.weight")),static_cast<U16*>(put("kv_a_layernorm.weight")),static_cast<U16*>(put("kv_b_proj.weight")),static_cast<U16*>(put("indexer.wk.weight")),
            static_cast<U16*>(put("indexer.index_kpool_compress_gate")),static_cast<U16*>(put("indexer.weights_proj.weight")),static_cast<U16*>(put("indexer.k_norm.weight")),static_cast<U16*>(put("indexer.k_norm.bias")),static_cast<float*>(put("indexer.index_kpool_compress_ape"))};
    }
};
template<class T>std::vector<T> download(Buffer& b,std::size_t n){std::vector<T> out(n);ck(cudaMemcpy(out.data(),b.p,n*sizeof(T),cudaMemcpyDeviceToHost));return out;}
struct Events {
    std::vector<cudaEvent_t> begin,end;DsaProfile p{};unsigned windows;
    Events(unsigned rows):begin(dsa_phases*dsa_max_windows),end(begin.size()),windows((rows+63)/64){
        for(unsigned phase=0;phase<dsa_phases;++phase)for(unsigned w=0;w<((phase==0||phase==2)?1:windows);++w){auto j=phase*dsa_max_windows+w;ck(cudaEventCreate(&begin[j]));ck(cudaEventCreate(&end[j]));}p={begin.data(),end.data()};}
    ~Events(){for(auto e:begin)if(e)cudaEventDestroy(e);for(auto e:end)if(e)cudaEventDestroy(e);}
    std::vector<double> times(){std::vector<double> result(dsa_phases);for(unsigned phase=0;phase<dsa_phases;++phase)for(unsigned w=0;w<((phase==0||phase==2)?1:windows);++w){float ms;auto j=phase*dsa_max_windows+w;ck(cudaEventElapsedTime(&ms,begin[j],end[j]));result[phase]+=ms;}return result;}
};
std::vector<float> bf_values(const std::vector<U16>& b){std::vector<float> a(b.size());for(std::size_t j=0;j<b.size();++j){U32 bits=U32(b[j])<<16;std::memcpy(&a[j],&bits,4);}return a;}
void state_metrics(const std::vector<U16>& a,const std::vector<U16>& b,const char* name){auto x=bf_values(a),y=bf_values(b);metrics(x.data(),y.data(),a.size(),name);std::size_t diff=0;for(std::size_t j=0;j<a.size();++j)diff+=a[j]!=b[j];std::printf("state_bits %s differing=%zu total=%zu\n",name,diff,a.size());}
int main(int argc,char** argv){try{
    std::string path,input_path;unsigned rows=0,iters=1;bool allow=false,mma_index=false,self_check=false;DsaIndex index_mode=DsaIndex::fp32;
    for(int j=1;j<argc;++j){std::string a=argv[j];if(a=="--self-check-exact")self_check=true;else if(a=="--input"&&j+1<argc)input_path=argv[++j];else if(a=="--allow-gpu")allow=true;else if(a=="--mma-index")mma_index=true;else if(a=="--index-variant"&&j+1<argc){std::string v=argv[++j];
        if(v=="fp32")index_mode=DsaIndex::fp32;else if(v=="mma")index_mode=DsaIndex::mma;
        else if(v=="mma-proj-fp32-score")index_mode=DsaIndex::mma_proj_fp32_score;
        else if(v=="fp32-proj-mma-score")index_mode=DsaIndex::fp32_proj_mma_score;
        else if(v=="tf32x3")index_mode=DsaIndex::tf32x3;else throw std::runtime_error("unknown index variant");}
        else if(a=="--weights"&&j+1<argc)path=argv[++j];else if(a=="--rows"&&j+1<argc)rows=unsigned(std::stoul(argv[++j]));else if(a=="--iters"&&j+1<argc)iters=unsigned(std::stoul(argv[++j]));else throw std::runtime_error("usage: --weights FILE --rows C --iters N [--index-variant fp32|mma|mma-proj-fp32-score|fp32-proj-mma-score|tf32x3] --allow-gpu");}
    need(!self_check||!input_path.empty(),"exact check requires captured real-layer --input");
    need(!path.empty()&&rows>0&&rows<=32768&&iters>0&&iters<=3,"weights/rows/iters required");
    if(!allow){std::puts("GPU execution disabled; pass --allow-gpu in an authorized window");return 2;}
    Weights weights(path);need(weights.layer==7,"real layer-7 benchmark requires layer 7 archive");ck(prepare_dsa());unsigned cap=(rows+3)/4*4;
    std::size_t ny=std::size_t(rows)*4096,nl=std::size_t(rows)*512,ni=std::size_t(rows)*128,np=std::size_t(rows/4)*128,nt=std::size_t(rows)*dsa_max_tokens;
    std::mt19937 rng(20261027);std::vector<float> x(ny);for(auto& v:x)v=float(int(rng()%2001)-1000)/1000*1.7320508f;
    if(!input_path.empty()){std::ifstream in(input_path,std::ios::binary);need(bool(in.read(reinterpret_cast<char*>(x.data()),ny*4))&&in.peek()==std::char_traits<char>::eof(),"captured input extent");}
    Buffer input(ny*4),y(ny*4),lat(std::size_t(cap)*512*2),key(std::size_t(cap)*128*2),gate(std::size_t(cap)*128*2),pooled(std::size_t(cap/4)*128*4);
    Buffer packed(dsa_export_bytes(rows));
    Buffer workspace(dsa_workspace_bytes(rows,cap)),flag(4),tokens(nt*4),counts(rows*4),diag(sizeof(DsaDiagnostics));
    ck(cudaMemcpy(input.p,x.data(),ny*4,cudaMemcpyHostToDevice));
    DsaCall call{};call.mma_index=mma_index;call.index=index_mode;call.rows=rows;call.w=weights.w;call.state={lat.as<U16>(),key.as<U16>(),gate.as<U16>(),pooled.as<float>(),cap};call.x=input.as<float>();call.y=y.as<float>();call.nonfinite=flag.as<unsigned>();call.tokens=tokens.as<U32>();call.counts=counts.as<U32>();
    Events events(rows);cudaEvent_t begin,end;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
    auto run=[&](bool ref,bool profile){ck(cudaMemset(flag.p,0,4));ck(cudaMemset(diag.p,0,sizeof(DsaDiagnostics)));auto start=std::chrono::steady_clock::now();ck(cudaEventRecord(begin));
        ck(enqueue_dsa(call,workspace.p,dsa_workspace_bytes(rows,cap),nullptr,ref,profile?&events.p:nullptr));ck(cudaEventRecord(end));ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,begin,end));
        unsigned bad;ck(cudaMemcpy(&bad,flag.p,4,cudaMemcpyDeviceToHost));need(!bad,"nonfinite projection/state/attention");return std::pair<double,double>{ms,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()};};
    auto verify_export=[&](){ck(enqueue_dsa_export(call.state,rows,packed.p,dsa_export_bytes(rows),nullptr));auto blob=download<unsigned char>(packed,dsa_export_bytes(rows));std::size_t offset=0;
        for(auto item:{std::pair<Buffer*,std::size_t>{&lat,nl*2},{&key,ni*2},{&gate,ni*2},{&pooled,np*4}}){auto raw=download<unsigned char>(*item.first,item.second);need(std::memcmp(blob.data()+offset,raw.data(),raw.size())==0,"node snapshot order/extent");offset+=raw.size();}need(offset==blob.size(),"node snapshot trailing bytes");};
    auto candidate_timing=run(false,false);verify_export();auto expected=download<float>(y,ny),ep=download<float>(pooled,np);auto el=download<U16>(lat,nl),ek=download<U16>(key,ni),eg=download<U16>(gate,ni);
    auto et=download<U32>(tokens,nt),en=download<U32>(counts,rows);
    if(self_check){
        need(!mma_index&&index_mode==DsaIndex::fp32,"exact check requires FP32 identity");
        call.legacy_fp32_index=true;auto old_timing=run(false,false);verify_export();
        std::printf("old_new layer_device_ms_old=%.6f layer_device_ms_new=%.6f input=captured weights=model_real samples=one_pair\n",old_timing.first,candidate_timing.first);
        std::uint64_t differing=0;
        auto exact=[&](const auto& a,const auto& b,const char* label){std::size_t count=0;for(std::size_t i=0;i<a.size();++i)count+=std::memcmp(&a[i],&b[i],sizeof(a[i]))!=0;
            bool equal=a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*sizeof(a[0]))==0;differing+=count;
            std::printf("old_new %s memcmp=%s differing=%zu elements=%zu\n",label,equal?"EXACT":"MISMATCH",count,a.size());};
        exact(expected,download<float>(y,ny),"layer_output");exact(ep,download<float>(pooled,np),"pooled_FP32");
        exact(el,download<U16>(lat,nl),"latent_BF16");exact(ek,download<U16>(key,ni),"key_BF16");exact(eg,download<U16>(gate,ni),"gate_BF16");
        exact(en,download<U32>(counts,rows),"selection_counts");auto old_tokens=download<U32>(tokens,nt);std::vector<U32> valid_new,valid_old;
        for(unsigned r=0;r<rows;++r)for(unsigned j=0;j<en[r];++j){valid_new.push_back(et[std::size_t(r)*dsa_max_tokens+j]);valid_old.push_back(old_tokens[std::size_t(r)*dsa_max_tokens+j]);}
        exact(valid_new,valid_old,"selected_ids");
        DsaExactReport report;ck(self_check_dsa_index(call,workspace.p,dsa_workspace_bytes(rows,cap),nullptr,report));
        std::printf("old_new raw_projections=%llu projection_elements=%llu scores=%llu score_elements=%llu ordered_keys=%llu memcmp_failures=%llu qualification=DRAFT\n",
            report.projection_differing,report.projection_elements,report.score_differing,report.score_elements,report.key_differing,report.memcmp_failures);
        ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));
        return differing||report.projection_differing||report.score_differing||report.key_differing||report.memcmp_failures?3:0;
    }
    auto check_tokens=[&](){auto actual=download<U32>(tokens,nt),an=download<U32>(counts,rows);need(an==en,"native selected cardinality nondeterministic");for(unsigned r=0;r<rows;++r){need(en[r]<=2051,"too many tokens");for(unsigned j=0;j<en[r];++j){auto at=std::size_t(r)*2051+j;need(actual[at]==et[at],"native selected sets nondeterministic");need(et[at]<=r,"noncausal selected token");if(j)need(et[at-1]<et[at],"noncanonical token order");}}};
    auto same=[](const auto& a,const auto& b){return a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*sizeof(a[0]))==0;};
    std::vector<double> times,walls;std::vector<std::vector<double>> phase_times(dsa_phases);
    for(unsigned iter=0;iter<iters;++iter){auto [ms,wall]=run(false,true);times.push_back(ms);walls.push_back(wall);auto phases=events.times();for(unsigned p=0;p<dsa_phases;++p)phase_times[p].push_back(phases[p]);
        need(same(download<float>(y,ny),expected)&&same(download<float>(pooled,np),ep)&&same(download<U16>(lat,nl),el)&&same(download<U16>(key,ni),ek)&&same(download<U16>(gate,ni),eg),"native output/state nondeterministic");check_tokens();}
    auto median=[](std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];};double ms=median(times);
    double linear=double(rows)*(4096.*1536+1536.*16384+4096.*512+16384.*4096+1536.*4096+4096.*(128+128+32));
    double absorption=double(rows)*64*2*256*512,index=0,attention=0;
    for(unsigned r=0;r<rows;++r){if((r+1)/4>512)index+=double((r+1)/4)*32*128;attention+=double(en[r])*64*512*2;}
    std::printf("identity=%s layer=%u C=%u iterations=%u input=%s seed=%s weights=model_real decode=included workspace_bytes=%zu device_ms=%.6f wall_ms=%.6f useful_projection_MAC=%.0f useful_absorption_MAC=%.0f useful_index_MAC=%.0f useful_attention_MAC=%.0f useful_TMAC_s=%.6f\n",dsa_index_identity(mma_index?DsaIndex::mma:index_mode),weights.layer,rows,iters,input_path.empty()?"synthetic_normalized":"captured",input_path.empty()?"20261027":"not_applied",dsa_workspace_bytes(rows,cap),ms,median(walls),linear,absorption,index,attention,(linear+absorption+index+attention)/(ms*1.e9));
    std::vector<double> phase_mac={0,double(rows)*(4096.*512+4096.*256),0,double(rows)*(4096.*1536+1536.*16384+1536.*4096+4096.*32),absorption/2,index,0,attention,absorption/2,double(rows)*16384*4096};
    for(unsigned p=0;p<dsa_phases;++p)std::printf("phase %s median_ms=%.6f useful_MAC=%.0f useful_TMAC_s=%.6f\n",dsa_phase_names[p],median(phase_times[p]),phase_mac[p],phase_mac[p]/(median(phase_times[p])*1.e9));
    std::printf("export_layout=PASS bytes=%zu order=latent,index_key,index_gate,pooled_keys\n",dsa_export_bytes(rows));
    // Retain native and scalar lists for two independent comparisons against
    // the paired-half model, using its bounded score window and cutoff.
    Buffer scalar_tokens(nt*4),scalar_counts(rows*4),diag2(sizeof(DsaDiagnostics));
    call.tokens=scalar_tokens.as<U32>();call.counts=scalar_counts.as<U32>();call.compare_tokens=tokens.as<U32>();call.compare_counts=counts.as<U32>();call.diagnostics=diag.as<DsaDiagnostics>();
    auto [refms,refwall]=run(true,true);verify_export();auto reference=download<float>(y,ny),rp=download<float>(pooled,np);auto rl=download<U16>(lat,nl),rk=download<U16>(key,ni),rg=download<U16>(gate,ni);
    metrics(expected.data(),reference.data(),ny,"DSA_layer_output");state_metrics(el,rl,"latent_BF16");state_metrics(ek,rk,"index_key_BF16");state_metrics(eg,rg,"index_gate_BF16");metrics(ep.data(),rp.data(),np,"pooled_keys_FP32");
    DsaDiagnostics d{};ck(cudaMemcpy(&d,diag.p,sizeof(d),cudaMemcpyDeviceToHost));
    std::printf("selection differing_rows=%llu differing_pool_memberships=%llu near_tie_pool_memberships=%llu outside_tolerance_pool_memberships=%llu near_tie_abs=%.9g near_tie_rel=%.9g tolerance=abs+rel*max(abs(score),abs(reference_cutoff))\n",d.differing_rows,d.differing_pools,d.near_tie_pools,d.outside_tolerance_pools,call.near_tie_abs,call.near_tie_rel);
    auto ref_phases=events.times();for(unsigned p=0;p<dsa_phases;++p)std::printf("reference_phase %s ms=%.6f\n",dsa_phase_names[p],ref_phases[p]);
    std::printf("reference=FP32_full_layer device_ms=%.6f wall_ms=%.6f determinism=PASS selection=%s qualification=DRAFT\n",refms,refwall,d.outside_tolerance_pools?"MISMATCH":"PASS_WITH_STATED_TIE_TOLERANCE");
    call.node_mimic=true;call.index_only=true;call.tokens=nullptr;call.counts=nullptr;
    call.compare2_tokens=scalar_tokens.as<U32>();call.compare2_counts=scalar_counts.as<U32>();call.diagnostics2=diag2.as<DsaDiagnostics>();
    ck(cudaMemset(diag2.p,0,sizeof(DsaDiagnostics)));
    auto [nodems,nodewall]=run(true,false);
    DsaDiagnostics dn{},ds{};ck(cudaMemcpy(&dn,diag.p,sizeof(dn),cudaMemcpyDeviceToHost));ck(cudaMemcpy(&ds,diag2.p,sizeof(ds),cudaMemcpyDeviceToHost));
    auto selection=[](const char* pair,const DsaDiagnostics& v){std::printf("selection_pair=%s differing_rows=%llu differing_pool_memberships=%llu near_tie_pool_memberships=%llu outside_tolerance_pool_memberships=%llu\n",pair,v.differing_rows,v.differing_pools,v.near_tie_pools,v.outside_tolerance_pools);};
    selection("variant_vs_node_mimic",dn);selection("node_mimic_vs_scalar_fp32",ds);
    std::printf("node_mimic=paired_half_ieee_rn32 split_lane_order=matched score=ascending_FP32_FMA cutoff_reference=node_mimic index_only_ms=%.6f wall_ms=%.6f bitwise_node=UNQUALIFIED qualification=DRAFT speed_gate_ms=300 speed_gate=%s agreement_gate=%s\n",nodems,nodewall,ms<=300?"PASS":"FAIL",(dn.differing_pools<=ds.differing_pools&&dn.outside_tolerance_pools<=ds.outside_tolerance_pools)?"PASS_WITH_STATED_TIE_TOLERANCE":"MISMATCH");
    ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));return (dn.differing_pools>ds.differing_pools||dn.outside_tolerance_pools>ds.outside_tolerance_pools)?3:0; // dedicated selection mismatch, all diagnostics already emitted
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
