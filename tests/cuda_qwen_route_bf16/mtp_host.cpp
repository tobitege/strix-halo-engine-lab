#include "strix/cuda/qwen_mtp_diagnostic.hpp"
#include <cstdio>
#include <limits>
namespace md=strix::cuda::qwen_mtp_diagnostic;
static unsigned checks=0;
static void ck(bool b) { ++checks;if (!b) throw std::runtime_error("host check"); }
template<class F> static void no(F f) { bool caught=false;try {f();}catch(const std::invalid_argument&){caught=true;}ck(caught); }
int main() {
    try {
        for (unsigned mask=0;mask<64;++mask) {
            const bool arm=(mask&1)!=0,c=(mask&2)!=0,d=(mask&4)!=0,x=(mask&8)!=0,s=(mask&16)!=0,g=(mask&32)!=0;
            if (!arm || (c&&d&&!x&&s&&!g)) { md::admission(arm,c,d,x,s,g);ck(true); }
            else no([&]{md::admission(arm,c,d,x,s,g);});
        }
        const md::Binding b{0x46bf863a1bc53cd4ULL,2,3,4096,5,1};md::binding(b,b);ck(true);
        for (unsigned i=0;i<6;++i) {
            auto a=b;
            switch(i) {case 0:++a.target;break;case 1:++a.engine;break;case 2:++a.session;break;
                case 3:++a.capacity;break;case 4:++a.sidecar;break;default:++a.phase;break;}
            no([&]{md::binding(a,b);});
        }
        for (unsigned width=1;width<=8;++width) for (unsigned mismatch=1;mismatch<=width;++mismatch)
            for (unsigned budget=1;budget<=8;++budget) {
                std::vector<std::uint32_t> rows(width),ids(width);
                for (unsigned i=0;i<width;++i) {rows[i]=i;ids[i]=i+1;}
                if (mismatch<width) rows[mismatch]=999;
                const auto a=md::select(rows,ids,budget);
                ck(a.keep==std::min(mismatch,budget));ck(a.pending==ids[a.keep-1]);ck(!a.stopped);
                for (unsigned stop=0;stop<width;++stop) {
                    const auto r=md::select(rows,ids,budget,ids[stop]);
                    ck(r.keep==std::min(a.keep,stop+1));ck(r.stopped==(stop<a.keep));
                }
            }
        no([]{md::select({}, {},1);});no([]{md::select({1},{},1);});no([]{md::select({1},{2},0);});
        ck(md::payload_bytes(8,248320,8ULL*248320*4)==8ULL*248320*4);
        no([]{md::payload_bytes(9,1,100);});no([]{md::payload_bytes(8,UINT64_MAX,UINT64_MAX);});
        no([]{md::payload_bytes(8,248320,1);});
        const std::vector<float> raw{1,2,2, 0,-1,3};const std::vector<std::uint32_t> ids{1,2};
        md::observe(raw,ids,2,3);ck(true);
        no([&]{md::observe(raw,{2,2},2,3);}); // tie must select first maximum
        no([&]{md::observe(raw,ids,8,SIZE_MAX);});
        no([&]{md::observe(raw,{1},2,3);});no([&]{md::observe(raw,ids,1,3);});
        for (std::size_t i=0;i<raw.size();++i) for (const auto bad:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
            auto a=raw;a[i]=bad;no([&]{md::observe(a,ids,2,3);});
        }
        #ifdef NDEBUG
        const auto dir=std::filesystem::temp_directory_path()/"strix-qwen-mtp-publication-host-ndebug";
#else
        const auto dir=std::filesystem::temp_directory_path()/"strix-qwen-mtp-publication-host-normal";
#endif
        std::filesystem::create_directory(dir);
        ck(!std::filesystem::exists(dir/"COMPLETE.txt"));
        no([&]{md::publish_complete(dir,"");});
        bool failed=false;try {md::publish_complete(dir/"missing-parent","pass");}catch(const std::runtime_error&){failed=true;}
        ck(failed && !std::filesystem::exists(dir/"COMPLETE.txt"));
        md::publish_complete(dir,"complete\n");ck(std::filesystem::exists(dir/"COMPLETE.txt"));
        ck(!std::filesystem::exists(dir/"COMPLETE.pending"));no([&]{md::publish_complete(dir,"duplicate");});
        std::filesystem::remove(dir/"COMPLETE.txt");std::filesystem::remove(dir);
        std::printf("MTP policy/controller/observer HOST %u checks PASS\n",checks);return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what());return 1; }
}
