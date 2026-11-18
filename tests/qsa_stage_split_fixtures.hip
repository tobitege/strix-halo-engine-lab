// Standalone production stage-score/selection partition fixture (no GPU required to build):
// amdclang++ -O3 -std=gnu++20 -Wall -Wextra -Werror -Wno-unused-function --offload-arch=gfx1151 -Iinclude -x hip tests/qsa_stage_split_fixtures.hip -o qsa_stage_split_fixtures
// Run in fresh processes with STRIX_QPF2K_SCORE_ROWS=16/32/64/auto and
// STRIX_QSA_SCORE_ROWS64=0/1; repeat with --packed and --selector-edges.
// --ring-rows 64/128/256/512/1024 selects groups per slot; default uses the
// engine byte policy (or STRIX_QSA_SCORE_RING_ROWS) at the guarded capacity.
// Defaults cross T=1024/2048/4096/8192 and max complete-block counts
// 4096/4097/65535/65536/65537/131072/262144. --rows N --blocks N narrows a run.
// The embedded reference materializes all T score/maxima rows and uses one
// whole-call select_batch_kernel; split uses two leased multi-group slots and one
// select_batch_kernel per slice. Both invoke the actual shared production
// stage-score launcher with band disabled (qsa_index_band.hip gates that route).
// This tests the legacy score/selection lifetime boundary;
// norm/store/pool publication and complete-engine outputs require engine gates.
// Reference above 65536 deliberately bypasses embedded admission only in this
// standalone fixture. At T8192/262144 allow ~21 GiB combined host/device memory
// for reference storage and its pre-selection snapshot. Diagnostics are untimed.
#include "qsa_flash_prod.inc"
#include "strix/qsa_score_scratch.hpp"
#include <array>
#include <memory>
namespace strix { namespace {
#include "../src/hip_qsa_stage_score.inc"
} }
using namespace strix;
static void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what)+": "+hipGetErrorString(e));
}
struct Backend {
    using Event=hipEvent_t;
    int device() const { int d=-1;check(hipGetDevice(&d),"device");return d; }
    void activate(int d) { check(hipSetDevice(d),"activate"); }
    void* allocate(std::uint64_t n) { void* p=nullptr;check(hipMalloc(&p,n),"allocate");return p; }
    Event event() { Event e=nullptr;check(hipEventCreateWithFlags(&e,hipEventDisableTiming),"event");return e; }
    void record(Event e,std::uintptr_t s) { check(hipEventRecord(e,reinterpret_cast<hipStream_t>(s)),"record"); }
    void wait(Event e) { check(hipEventSynchronize(e),"wait"); }
    void drain(std::uintptr_t s) { check(hipStreamSynchronize(reinterpret_cast<hipStream_t>(s)),"drain"); }
    void destroy(Event e) { check(hipEventDestroy(e),"destroy"); }
    void free(void* p) { check(hipFree(p),"free"); }
};
template<class T> struct Device {
    T* p=nullptr;
    explicit Device(std::size_t n) { check(hipMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)),"device allocation"); }
    ~Device() { if(p) (void)hipFree(p); }
    Device(const Device&)=delete;
};
template<class T> struct Pinned {
    T* p=nullptr;
    explicit Pinned(std::size_t n) { check(hipHostMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)),"pinned allocation"); }
    ~Pinned() { if(p) (void)hipHostFree(p); }
    Pinned(const Pinned&)=delete;
};
static constexpr std::uint32_t canary=0xa5a5a5a5U;
static void equal(const void* a,const void* b,std::size_t bytes,const char* kind,unsigned group) {
    if(std::memcmp(a,b,bytes)) throw std::runtime_error(std::string(kind)+" mismatch at group "+std::to_string(group));
}
static void guard(const std::vector<std::uint32_t>& words,std::size_t stride,unsigned rows,
    const std::vector<std::uint32_t>& live,unsigned lead,const char* what) {
    for(unsigned r=0;r<rows;++r) for(std::size_t i=0;i<stride;++i)
        if(i<lead || i>=lead+live[r]) {
            if(words[r*stride+i]!=canary)
                throw std::runtime_error(std::string(what)+" canary overwritten at row "+std::to_string(r));
        } else if(words[r*stride+i]==canary) {
            throw std::runtime_error(std::string(what)+" live word unwritten at row "+std::to_string(r));
        }
}
static std::vector<std::uint32_t> copy(const void* p,std::size_t words) {
    std::vector<std::uint32_t> v(words);if(!words) return v;check(hipMemcpy(v.data(),p,words*4U,hipMemcpyDeviceToHost),"copy");return v;
}
// Optional adversarial selector-only rows, after scoring. Exercise ties, signed
// zero, NaN/inf, too-few-maxima and candidate overflow through the same selector.
__global__ void selector_edges(qsa_flash::ScoreGroup<64U>* desc,unsigned first,unsigned count) {
    const unsigned r=blockIdx.y,g=first+blockIdx.z;
    if(blockIdx.z>=count || r%16U>=4U) return;
    const auto sc=desc[g];const auto blocks=sc.views.blocks[r];
    const unsigned b=(blockIdx.x*blockDim.x+threadIdx.x)*4U;
    if(b>=blocks) return;
    float x[4];
    for(unsigned j=0;j<4U;++j) {
        const unsigned i=b+j;
        float v=r%16U==0U?0.0F:r%16U==1U?(i<19U?1.0F:-INFINITY):
            r%16U==2U?(i%7U==0U?INFINITY:i%3U==0U?__uint_as_float(0x7fc00001U):-INFINITY):
            (i%2U?-0.0F:0.0F);
        x[j]=i<blocks?v:-INFINITY;
        if(i<blocks) sc.views.block_scores[r][i]=v;
    }
    sc.maxima[static_cast<std::uint64_t>(r)*sc.maxima_stride+b/4U]=fmaxf(fmaxf(x[0],x[1]),fmaxf(x[2],x[3]));
}
static void run(unsigned T,unsigned blocks,bool packed,bool edges,unsigned ring_rows,hipStream_t stream) {
    constexpr unsigned dim=128U,heads=4U,topk=512U,lead=16U,mlead=4U;
    if(T<64U || T>16384U || T%64U || blocks<T/4U) throw std::invalid_argument("fixture rows/blocks");
    const unsigned groups=T/64U, stride=blocks+64U, ms=(blocks+3U)/4U+16U, ss=topk*4U+3U+16U;
    const auto nscore=static_cast<std::size_t>(T)*stride,nmax=static_cast<std::size_t>(T)*ms;
    std::vector<float> keys(static_cast<std::size_t>(blocks)*dim),queries(static_cast<std::size_t>(T)*heads*dim);
    std::uint64_t seed=0x9e3779b97f4a7c15ULL;
    const auto next=[&] { seed^=seed<<13U;seed^=seed>>7U;seed^=seed<<17U;
        return static_cast<float>(static_cast<int>(seed%257U)-128)*0.015625F; };
    for(auto& x:keys) x=next();for(auto& x:queries) x=next();
    const float specials[]{0.0F,-0.0F,INFINITY,std::numeric_limits<float>::quiet_NaN(),0x1p-120F,0x1p120F};
    for(unsigned g=0;g<groups;++g) {
        std::fill_n(queries.data()+static_cast<std::size_t>(g*64U)*heads*dim,heads*dim,0.0F);
        for(unsigned i=0;i<6U;++i) queries[static_cast<std::size_t>(g*64U+i+1U)*heads*dim+i]=specials[i];
    }
    for(unsigned b=1;b<blocks;b+=997U) keys[static_cast<std::size_t>(b)*dim]=specials[b%6U];
    Device<float> dk(keys.size()),dq(queries.size()),ref(nscore),maxima(nmax);
    Device<std::uint16_t> pk(keys.size());Device<uint2> ranges(blocks);
    Device<std::uint32_t> selected(static_cast<std::size_t>(T)*ss),split_selected(static_cast<std::size_t>(T)*ss);
    Device<unsigned long long> refhash(4U),splithash(4U);
    check(hipMemcpy(dk.p,keys.data(),keys.size()*4U,hipMemcpyHostToDevice),"keys");
    check(hipMemcpy(dq.p,queries.data(),queries.size()*4U,hipMemcpyHostToDevice),"queries");
    std::vector<std::uint16_t> bits(keys.size());
    for(std::size_t i=0;i<bits.size();++i) { std::uint32_t word;std::memcpy(&word,&keys[i],4U);bits[i]=static_cast<std::uint16_t>(word>>16U); }
    check(hipMemcpy(pk.p,bits.data(),bits.size()*2U,hipMemcpyHostToDevice),"packed");
    keys.clear();keys.shrink_to_fit();queries.clear();queries.shrink_to_fit();bits.clear();bits.shrink_to_fit();
    check(hipMemsetAsync(ref.p,0xa5,nscore*4U,stream),"ref canary");
    check(hipMemsetAsync(maxima.p,0xa5,nmax*4U,stream),"max canary");
    check(hipMemsetAsync(selected.p,0xa5,static_cast<std::size_t>(T)*ss*4U,stream),"select canary");
    check(hipMemsetAsync(split_selected.p,0xa5,static_cast<std::size_t>(T)*ss*4U,stream),"split select canary");
    check(hipMemsetAsync(refhash.p,0,32U,stream),"trace reset");check(hipMemsetAsync(splithash.p,0,32U,stream),"trace reset");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::key_range_kernel<>),dim3((blocks+255U)/256U),dim3(256),0,stream,dk.p,0U,blocks,ranges.p);
    Pinned<qsa_flash::ScoreGroup<64U>> embedded(groups),split(groups);
    Pinned<qsa_flash::SelectRow> refsel(T),splitsel(T);
    std::vector<std::uint32_t> live(T),mlive(T),slive(T);
    for(unsigned g=0;g<groups;++g) {
        auto& sc=embedded.p[g];sc={};sc.maxima=maxima.p+static_cast<std::size_t>(g)*64U*ms+mlead;sc.maxima_stride=ms;
        for(unsigned r=0;r<64U;++r) {
            const unsigned row=g*64U+r,visible=blocks*4U-T+row+1U,complete=visible/4U;
            live[row]=complete>topk?complete:0U;mlive[row]=(live[row]+3U)/4U;slive[row]=std::min(complete,topk)*4U+visible%4U;
            sc.views.index_query[r]=dq.p+static_cast<std::size_t>(row)*heads*dim;
            sc.views.block_scores[r]=ref.p+static_cast<std::size_t>(row)*stride+lead;sc.views.blocks[r]=live[row];
            sc.blocks=std::max(sc.blocks,live[row]);
        }
        sc.maxima_stride=(sc.blocks+3U)/4U+8U; // incumbent group-local stride, plus row guards
        for(unsigned r=0;r<64U;++r) {
            const unsigned row=g*64U+r;
            refsel.p[row]={sc.views.block_scores[r],selected.p+static_cast<std::size_t>(row)*ss+4U,
                sc.maxima+static_cast<std::size_t>(r)*sc.maxima_stride,blocks*4U-T+row+1U,0U};
        }
    }
    const void* pool=packed?static_cast<void*>(pk.p):static_cast<void*>(dk.p);
    qpf2k_stage_score_launch(embedded.p,pool,packed,ranges.p,dim,blocks,0U,groups,stream,false);
    if(edges) hipLaunchKernelGGL(selector_edges,dim3((blocks+1023U)/1024U,64U,groups),dim3(256),0,stream,embedded.p,0U,groups);
    check(hipGetLastError(),"embedded score");check(hipStreamSynchronize(stream),"embedded score completion");
    // Snapshot exactly the bytes live at selection, before the whole-call selector.
    auto pre=copy(ref.p,nscore);
    guard(pre,stride,T,live,lead,"embedded pre score");
    const auto trace=[&](const qsa_flash::ScoreGroup<64U>& sc,const qsa_flash::SelectRow* sel,unsigned g,unsigned kind,unsigned long long* acc) {
        qsa_flash::TraceViews<64U> v{};
        for(unsigned r=0;r<64U;++r) { const unsigned row=g*64U+r;
            v.data[r]=kind==1U?sel[row].selected:reinterpret_cast<const std::uint32_t*>(sc.views.block_scores[r]);
            v.count[r]=kind==1U?slive[row]:live[row];v.position[r]=blocks*4U-T+row; }
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::trace_kernel<64U>),dim3(64),dim3(256),0,stream,v,kind,0U,acc);
    };
    for(unsigned g=0;g<groups;++g) trace(embedded.p[g],refsel.p,g,0U,refhash.p);
    hipLaunchKernelGGL(qsa_flash::select_batch_kernel,dim3(T),dim3(kSelectThreads),0,stream,refsel.p,4U,topk);
    for(unsigned g=0;g<groups;++g) { trace(embedded.p[g],refsel.p,g,1U,refhash.p);trace(embedded.p[g],refsel.p,g,2U,refhash.p); }
    check(hipGetLastError(),"embedded selection");check(hipStreamSynchronize(stream),"embedded selection completion");
    const QsaScratchBinding binding{0,reinterpret_cast<std::uintptr_t>(stream),1U,static_cast<std::uint64_t>(blocks+64U)*4U};
    const QsaScoreRingGeometry geometry(binding.maximum,4U,ring_rows);
    QsaScoreScratch<Backend> ring(Backend{},binding,geometry.total_bytes(),geometry.rows);
    const auto& geo=ring.geometry();const QsaPrefillStagePlan plan(T,qsa_flash::score_rows(blocks),geo.rows);
    for(unsigned slice=0;slice<plan.slices();++slice) {
        QsaScoreScratch<Backend>::Lease lease(ring);
        check(hipMemsetAsync(ring.data(lease.slot),0xa5,geo.total_bytes()/2U,stream),"slot canary reset");
        auto* scores=static_cast<float*>(ring.data(lease.slot));auto* mx=scores+geo.score_slot_bytes()/4U;
        const unsigned first=plan.first(slice)/64U,count=plan.count(slice)/64U;
        for(unsigned k=0;k<count;++k) {
            const unsigned g=first+k;
            auto& sc=split.p[g];sc=embedded.p[g];
            sc.maxima=mx+static_cast<std::size_t>(k)*64U*ms+mlead;
            for(unsigned r=0;r<64U;++r) {
                const unsigned row=g*64U+r;
                sc.views.block_scores[r]=scores+static_cast<std::size_t>(k*64U+r)*geo.blocks+lead;
                splitsel.p[row]={sc.views.block_scores[r],split_selected.p+static_cast<std::size_t>(row)*ss+4U,
                    sc.maxima+static_cast<std::size_t>(r)*sc.maxima_stride,refsel.p[row].visible,0U};
            }
        }
        qpf2k_stage_score_launch(split.p,pool,packed,ranges.p,dim,blocks,first,count,stream,false);
        if(edges) hipLaunchKernelGGL(selector_edges,dim3((blocks+1023U)/1024U,64U,count),dim3(256),0,stream,split.p,first,count);
        for(unsigned g=first;g<first+count;++g) trace(split.p[g],splitsel.p,g,0U,splithash.p);
        check(hipGetLastError(),"split score");check(hipStreamSynchronize(stream),"split pre snapshot");
        for(unsigned k=0;k<count;++k) {
            const unsigned g=first+k;
            const auto actual=copy(scores+static_cast<std::size_t>(k)*64U*stride,64U*stride),
                actualmax=copy(mx+static_cast<std::size_t>(k)*64U*ms,64U*ms),
                expectedmax=copy(maxima.p+static_cast<std::size_t>(g)*64U*ms,64U*ms);
            equal(actual.data(),pre.data()+static_cast<std::size_t>(g)*64U*stride,actual.size()*4U,"pre score",g);
            equal(actualmax.data(),expectedmax.data(),actualmax.size()*4U,"maxima",g);
            std::vector<std::uint32_t> group_mlive(mlive.begin()+g*64U,mlive.begin()+(g+1U)*64U);
            guard(actualmax,split.p[g].maxima_stride,64U,group_mlive,mlead,"split maxima");
            for(std::size_t i=64U*split.p[g].maxima_stride;i<actualmax.size();++i)
                if(actualmax[i]!=canary) throw std::runtime_error("maxima group tail canary overwritten");
        }
        hipLaunchKernelGGL(qsa_flash::select_batch_kernel,dim3(count*64U),dim3(kSelectThreads),0,stream,splitsel.p+first*64U,4U,topk);
        for(unsigned g=first;g<first+count;++g) { trace(split.p[g],splitsel.p,g,1U,splithash.p);trace(split.p[g],splitsel.p,g,2U,splithash.p); }
        check(hipGetLastError(),"split selection");check(hipStreamSynchronize(stream),"split post snapshot");
        for(unsigned k=0;k<count;++k) {
            const unsigned g=first+k;
            const auto actual=copy(scores+static_cast<std::size_t>(k)*64U*stride,64U*stride),
                post=copy(ref.p+static_cast<std::size_t>(g)*64U*stride,64U*stride);
            equal(actual.data(),post.data(),actual.size()*4U,"post score",g);
            std::vector<std::uint32_t> group_live(live.begin()+g*64U,live.begin()+(g+1U)*64U);
            guard(actual,stride,64U,group_live,lead,"post score");
            const auto aselect=copy(split_selected.p+static_cast<std::size_t>(g)*64U*ss,64U*ss),
                eselect=copy(selected.p+static_cast<std::size_t>(g)*64U*ss,64U*ss);
            equal(aselect.data(),eselect.data(),aselect.size()*4U,"selected",g);
            std::vector<std::uint32_t> group_slive(slive.begin()+g*64U,slive.begin()+(g+1U)*64U);
            guard(aselect,ss,64U,group_slive,4U,"selected");
            const auto actualmax=copy(mx+static_cast<std::size_t>(k)*64U*ms,64U*ms),
                expectedmax=copy(maxima.p+static_cast<std::size_t>(g)*64U*ms,64U*ms);
            equal(actualmax.data(),expectedmax.data(),actualmax.size()*4U,"post maxima",g);
        }
        // A short final slice must not write any unused group's reservation.
        const auto unused_scores=copy(scores+static_cast<std::size_t>(count)*64U*stride,(geo.rows-count*64U)*stride),
            unused_maxima=copy(mx+static_cast<std::size_t>(count)*64U*ms,(geo.rows-count*64U)*ms);
        const auto unused_guard=[](const auto& words) {
            for(auto word:words) if(word!=canary) throw std::runtime_error("unused slot rows overwritten");
        };
        unused_guard(unused_scores);unused_guard(unused_maxima);
    }
    check(hipStreamSynchronize(stream),"trace completion");
    const auto rh=copy(refhash.p,8U),sh=copy(splithash.p,8U);equal(rh.data(),sh.data(),32U,"trace hashes",0U);
    std::printf("EXACT T %u blocks %u slices %u score_rows %u ring_rows %u ring_bytes %llu packed %u selector_edges %u (pre/maxima/selected/post/canaries/trace)\n",
        T,blocks,plan.slices(),plan.score_rows,geo.rows,static_cast<unsigned long long>(geo.total_bytes()),packed?1U:0U,edges?1U:0U);
}
int main(int argc,char** argv) {
    try {
        std::vector<unsigned> rows{1024U,2048U,4096U,8192U},blocks{4096U,4097U,65535U,65536U,65537U,131072U,262144U};
        bool packed=false,edges=false;
        unsigned ring_rows=qsa_score_ring_rows_override(std::getenv("STRIX_QSA_SCORE_RING_ROWS"));
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--rows" && i+1<argc) rows={static_cast<unsigned>(std::stoul(argv[++i]))};
            else if(arg=="--blocks" && i+1<argc) blocks={static_cast<unsigned>(std::stoul(argv[++i]))};
            else if(arg=="--ring-rows" && i+1<argc) ring_rows=qsa_score_ring_rows_override(argv[++i]);
            else if(arg=="--packed") packed=true;
            else if(arg=="--selector-edges") edges=true;
            else throw std::invalid_argument("unknown or incomplete fixture argument");
        }
        int devices=0;const auto available=hipGetDeviceCount(&devices);
        if(available!=hipSuccess || !devices) { std::fprintf(stderr,"SKIP: no ROCm-capable device\n");return 77; }
        check(hipSetDevice(0),"set device");hipStream_t stream=nullptr;check(hipStreamCreate(&stream),"stream");
        for(unsigned t:rows) for(unsigned b:blocks) run(t,b,packed,edges,ring_rows,stream);
        check(hipStreamDestroy(stream),"stream destroy");return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what());return 1; }
}
