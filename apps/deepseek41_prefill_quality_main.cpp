#include "strix/device/hip_gate.hpp"
#include "strix/deepseek41_prefill.hpp"
#include "strix/dist/communicator.hpp"
#include "deepseek41_quality_arms.hpp"
#include <algorithm>
#include <chrono>
#include <iterator>
#include <cmath>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

namespace {
using strix::deepseek41::PrefillOptions;
using strix::deepseek41::PrefillResult;
using strix::deepseek41::quality::arm_precision;

std::vector<double> log_softmax(const float* z, std::size_t n) {
    double mx = -INFINITY;
    for (std::size_t i = 0; i < n; ++i) mx = std::max(mx, double(z[i]));
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) sum += std::exp(double(z[i]) - mx);
    const double l = std::log(sum);
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = double(z[i]) - mx - l;
    return out;
}

// Per-layer divergence of a candidate trace against the exact trace (this rank's rows).
std::string trace_json(const PrefillResult& ref, const PrefillResult& got, std::size_t stream_row) {
    std::ostringstream o;
    o << "[";
    for (std::size_t l = 0; l < std::min(ref.trace.size(), got.trace.size()); ++l) {
        const auto& a = ref.trace[l];
        const auto& b = got.trace[l];
        unsigned router = 0, router_ids = 0, index = 0, rows = 0, router_tail = 0, index_tail = 0;
        const std::size_t nr = a.row_hashes.size(), K = nr ? a.router_ids.size() / nr : 0;
        for (std::size_t r = 0; K && r < nr && a.router_ids.size() == b.router_ids.size(); ++r) {
            // sized first, then copied: gcc 14 -O3 reports a false -Wfree-nonheap-object on the iterator-range form
            std::vector<std::uint32_t> x(K), y(K);
            std::copy_n(a.router_ids.data() + r * K, K, x.begin());
            std::copy_n(b.router_ids.data() + r * K, K, y.begin());
            std::sort(x.begin(), x.end());
            std::sort(y.begin(), y.end());
            if (x != y) {
                ++router;
                if (r + 64 >= nr) ++router_tail;
                std::vector<std::uint32_t> d;
                std::set_difference(x.begin(), x.end(), y.begin(), y.end(), std::back_inserter(d));
                router_ids += static_cast<unsigned>(d.size());
            }
        }
        for (std::size_t r = 0; r < a.index_sets.size() && a.index_sets.size() == b.index_sets.size(); ++r)
            if (a.index_sets[r] != b.index_sets[r]) {
                ++index;
                if (r + 64 >= a.index_sets.size()) ++index_tail;
            }
        std::size_t first_row = a.row_hashes.size();
        for (std::size_t r = 0; r < a.row_hashes.size() && a.row_hashes.size() == b.row_hashes.size(); ++r)
            if (a.row_hashes[r] != b.row_hashes[r]) {
                ++rows;
                first_row = std::min(first_row, r);
            }
        double worst = 0, mean = 0;
        const std::size_t tr = stream_row ? a.tail.size() / stream_row : 0;
        for (std::size_t r = 0; r < tr && a.tail.size() == b.tail.size(); ++r) {
            double num = 0, den = 0;
            for (std::size_t i = 0; i < stream_row; ++i) {
                const double x = a.tail[r * stream_row + i], y = b.tail[r * stream_row + i];
                num += (y - x) * (y - x);
                den += x * x;
            }
            const double rel = std::sqrt(num / std::max(den, 1e-300));
            worst = std::max(worst, rel);
            mean += rel / double(tr);
        }
        o << (l ? "," : "") << "{\"l\":" << l << ",\"router_rows\":" << router << ",\"router_tail\":" << router_tail
          << ",\"router_ids\":" << router_ids << ",\"index_rows\":" << index << ",\"index_tail\":" << index_tail
          << ",\"index_owner\":" << (a.index_sets.empty() ? 0 : 1) << ",\"stream_rows\":" << rows
          << ",\"first_row\":" << (first_row == a.row_hashes.size() ? -1 : long(first_row)) << ",\"tail_rel_max\":" << worst
          << ",\"tail_rel_mean\":" << mean << "}";
    }
    o << "]";
    return o.str();
}
}  // namespace

int main(int argc,char** argv) try {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    const auto start=std::chrono::steady_clock::now();
    if(argc!=5 && argc!=6)throw std::runtime_error("MODEL manifest.tsv outdir rank [budget_seconds]");
    const double budget=argc==6?std::stod(argv[5]):0;
    if(budget && budget<60)throw std::runtime_error("budget must be >=60 seconds");
    unsigned rank=static_cast<unsigned>(std::stoul(argv[4]));
    if(rank>1)throw std::runtime_error("rank");
    const char* te=std::getenv("STRIX_QUALITY_TRACE");const bool trace=te && std::string(te)=="1";
    // STRIX_QUALITY_DUMP: unset/1 every arm, 0 exact only, 2 exact and the non-null arms (no n_* noise samples)
    const char* de=std::getenv("STRIX_QUALITY_DUMP");const bool dump_all=!(de && (std::string(de)=="0"||std::string(de)=="2"));
    const bool dump_candidates=de && std::string(de)=="2";
    strix::dist::Usb4AutoOptions u;u.rank=rank;u.timeout=std::chrono::seconds(20);
    auto transport=strix::dist::usb4_host_auto(u,nullptr);
    strix::dist::Communicator comm(strix::dist::Group{rank,2,1,0x616d647072656669ULL},std::move(transport),std::size_t(32)<<20);
    strix::deepseek41::PrefillOptions o;o.rank=rank;o.world=2;o.communicator=&comm;
    o.max_positions=8192;o.row_split=false;o.cone=false;o.quality_head_rows=64;
    strix::deepseek41::DevicePrefill prefill(argv[1],o);
    std::printf("loaded rank=%u seconds=%.3f weights=%llu overlay=%s trace=%d\n",rank,prefill.load_seconds(),static_cast<unsigned long long>(prefill.weight_bytes()),prefill.overlay_description().c_str(),trace);std::fflush(stdout);
    std::ifstream manifest(argv[2]);if(!manifest)throw std::runtime_error("manifest");
    struct Flags {
        std::uint32_t* p=nullptr;
        Flags(){if(hipMalloc(&p,8)!=hipSuccess)throw std::runtime_error("budget flags allocation");}
        ~Flags(){if(p)(void)hipFree(p);}
    } flags;
    unsigned budget_round=0;
    bool control_done=false;
    std::string name,path;unsigned n;
    for(std::string line;std::getline(manifest,line);){
        if(line.empty()||line[0]=='#')continue;
        std::istringstream fields(line);std::string arms_field;
        if(!(fields>>name>>n>>path))throw std::runtime_error("manifest row");
        std::vector<std::string> arms;
        if(fields>>arms_field){
            if(arms_field.find_first_not_of("0123456789")==std::string::npos){  // legacy bit mask: exact 1, dense 2, attention 4
                const auto mask=std::stoul(arms_field);
                if(mask>7)throw std::runtime_error("manifest arm mask");
                if(mask&1U)arms.push_back("exact");
                if(mask&2U)arms.push_back("dense");
                if(mask&4U)arms.push_back("attention");
            }else{std::stringstream as(arms_field);for(std::string a;std::getline(as,a,',');)arms.push_back(a);}
        }else arms={"exact","dense","attention"};
        if(name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-")!=std::string::npos)throw std::runtime_error("case name");
        std::ifstream f(path);std::vector<std::uint32_t> ids;std::uint32_t id;while(f>>id)ids.push_back(id);
        if(ids.size()!=n)throw std::runtime_error("prompt length");
        PrefillResult ref;bool have_ref=false;
        for(const auto& label:arms){
            if(label.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_")!=std::string::npos)throw std::runtime_error("arm label "+label);
            auto precision=arm_precision(label);precision.trace=trace;
            const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            std::uint32_t stop=budget && elapsed+(n>4096?45:15)>budget,peer_stop=0;
            if(hipMemcpy(flags.p,&stop,4,hipMemcpyHostToDevice)!=hipSuccess)throw std::runtime_error("budget flag upload");
            comm.exchange(nullptr,flags.p,flags.p+1,4,strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control),15U,++budget_round});
            if(hipMemcpy(&peer_stop,flags.p+1,4,hipMemcpyDeviceToHost)!=hipSuccess)throw std::runtime_error("budget flag download");
            if(stop || peer_stop){std::printf("budget_stop next=%s arm=%s elapsed=%.3f\n",name.c_str(),label.c_str(),elapsed);std::fflush(stdout);return 0;}
            prefill.set_precision_for_new_sequence(precision);
            auto r=prefill.run(ids);
            if(label=="exact" && !control_done){
                auto repeat=prefill.run(ids);
                if(repeat.teacher_logits.size()!=r.teacher_logits.size() || repeat.logits.size()!=r.logits.size() ||
                   std::memcmp(repeat.teacher_logits.data(),r.teacher_logits.data(),r.teacher_logits.size()*4) ||
                   std::memcmp(repeat.logits.data(),r.logits.data(),r.logits.size()*4))throw std::runtime_error("repeated incumbent run drift");
                std::printf("exact_repeat_control=bitwise_equal case=%s\n",name.c_str());std::fflush(stdout);control_done=true;
            }
            if(r.teacher_logits.size()!=64*r.logits.size())throw std::runtime_error("teacher logit shape");
            if(std::memcmp(r.logits.data(),r.teacher_logits.data()+63*r.logits.size(),r.logits.size()*4))throw std::runtime_error("batched head differs from last-position incumbent head");
            std::uint64_t hash=14695981039346656037ULL;
            for(float value:r.teacher_logits){std::uint32_t bits;std::memcpy(&bits,&value,4);for(unsigned b=0;b<4;++b){hash=(hash^(bits&255U))*1099511628211ULL;bits>>=8;}}
            unsigned nonfinite=0;for(float v:r.teacher_logits)nonfinite+=!std::isfinite(v);
            if(nonfinite)throw std::runtime_error("nonfinite teacher logits");
            if(rank==0 && (dump_all || label=="exact" || (dump_candidates && label.rfind("n_",0)!=0))){std::ofstream out(std::string(argv[3])+"/"+name+"-"+label+".f32",std::ios::binary);out.write(reinterpret_cast<const char*>(r.teacher_logits.data()),static_cast<std::streamsize>(r.teacher_logits.size()*4));if(!out)throw std::runtime_error("write logits");}
            std::printf("{\"case\":\"%s\",\"arm\":\"%s\",\"rank\":%u,\"tokens\":%u,\"rows\":64,\"seconds\":%.6f,\"next\":%u,\"nonfinite\":%u,\"hash\":\"%016llx\"}\n",name.c_str(),label.c_str(),rank,n,r.seconds,r.next_token,nonfinite,static_cast<unsigned long long>(hash));std::fflush(stdout);
            if(label=="exact"){ref=std::move(r);have_ref=true;continue;}
            if(!have_ref || rank!=0)continue;
            // In-process KL(exact || arm) and top-1 per position (the strict gate re-computes it from the dumps).
            const std::size_t V=ref.logits.size();
            std::ostringstream kl;kl.precision(9);double sum=0;unsigned top=0;
            for(unsigned p=0;p<64;++p){
                const float* x=ref.teacher_logits.data()+std::size_t(p)*V;const float* y=r.teacher_logits.data()+std::size_t(p)*V;
                auto lp=log_softmax(x,V),lq=log_softmax(y,V);double k=0;
                for(std::size_t i=0;i<V;++i)k+=std::exp(lp[i])*(lp[i]-lq[i]);
                sum+=k;top+=std::max_element(x,x+V)-x==std::max_element(y,y+V)-y;kl<<(p?",":"")<<k;
            }
            std::size_t stream_row=0;
            if(!ref.trace.empty() && !ref.trace[0].tail.empty())stream_row=ref.trace[0].tail.size()/64;
            std::printf("{\"diag\":1,\"case\":\"%s\",\"arm\":\"%s\",\"mean_kl\":%.9g,\"top1\":%u,\"kl\":[%s],\"layers\":%s}\n",name.c_str(),label.c_str(),sum/64,top,kl.str().c_str(),
                        trace?trace_json(ref,r,stream_row).c_str():"[]");
            std::fflush(stdout);
        }
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"prefill-quality: %s\n",e.what());return 2;}
