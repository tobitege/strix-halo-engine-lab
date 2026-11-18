#include "strix/device/hip_gate.hpp"
#include "strix/deepseek41_prefill.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/deepseek41_model.hpp"
#include "strix/ds41_fast_profile.hpp"
#include "strix/model_content_hash.hpp"
#include "deepseek41_quality_arms.hpp"
#include <filesystem>
#include <span>
#include <unistd.h>
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
std::string sha_hex(const void* p, std::size_t n) {
    return strix::model_content_identity_hex(strix::sha256_bytes(std::span<const std::byte>(static_cast<const std::byte*>(p), n)));
}
std::string sha_hex(const std::string& s) { return sha_hex(s.data(), s.size()); }

// Gate mode: one execution witness (tools/strix-gate, schema strix-fast-execution/v1) per arm, from the executed
// configuration and the dumps this process wrote. Canonical numerical configuration: every precision field of the arm
// except the null control (null_ulp_layer, null_seed), the loaded run form, the FAST dense threshold and the process's
// STRIX_* environment, one "key=value" line each in a fixed order (environment sorted).
std::string numerical_config(const PrefillOptions& a, const PrefillOptions& load, bool replay_oracle) {
    std::ostringstream o;
    o << "ds41-numcfg-v1\nfp16_dense=" << a.fp16_dense << "\nbf16_attention=" << a.bf16_attention << "\nfast_attention=" << a.fast_attention
      << "\nfast_attention_flash=" << a.fast_attention_flash << "\nfast_moe=" << a.fast_moe << "\nfast_dense=" << a.fast_dense
      << "\nfp16_projections=" << a.fp16_projections << "\napprox_layers=" << a.approx_layer_first << "-" << a.approx_layer_end
      << "\nfp16_min_rows=" << a.fp16_min_rows << "\nfp16_keep_a8=" << a.fp16_keep_a8 << "\nfp16_fast=" << a.fp16_fast
      << "\nfp16_wsplit=" << a.fp16_wsplit << "\ndecoder_replay_rows=" << a.decoder_replay_rows
      << "\ndecoder_replay_oracle=" << replay_oracle << "\nprefill_cone=1\ncontinuation_cone=0\ncontinuation_decoder_replay_rows=0"
      << "\nrow_split=" << load.row_split << "\nquality_head_rows=" << load.quality_head_rows << "\nmax_positions=" << load.max_positions
      << "\nwindow_rows=" << load.window_rows << "\nworld=" << load.world
      << "\nfast_dense_min_rows=" << strix::ds41::fast_profile::dense_min_rows() << "\n";
    std::vector<std::string> env;
    for (char** e = environ; e && *e; ++e)
        if (std::strncmp(*e, "STRIX_", 6) == 0) env.emplace_back(*e);
    std::sort(env.begin(), env.end());
    for (const auto& e : env) o << "env " << e << "\n";
    return o.str();
}
struct GateArm {
    std::string run_id, profile, control, numcfg;
    std::map<std::string, std::string> prompts;  // name -> the sealed prompts map entry (JSON)
};
void write_witnesses(const std::string& dir, const std::map<std::string, GateArm>& arms, unsigned rank, const std::string& binary,
                     const std::string& model) {
    for (const auto& [label, a] : arms) {
        std::ostringstream o;
        o << "{\"schema\":\"strix-fast-execution/v1\",\"run_id\":\"" << a.run_id << "\",\"rank\":" << rank << ",\"binary_sha256\":\""
          << binary << "\",\"engine\":\"ds41\",\"profile\":\"" << a.profile << "\",\"model_sha256\":\"" << model
          << "\",\"numerical_config_sha256\":\"" << sha_hex(a.numcfg) << "\",\"control\":" << a.control << ",\"prompts\":{";
        bool first = true;
        for (const auto& [name, entry] : a.prompts) {
            o << (first ? "" : ",") << "\"" << name << "\":" << entry;
            first = false;
        }
        o << "}}\n";
        const auto d = std::filesystem::path(dir) / label;
        std::ofstream(d / ("execution-r" + std::to_string(rank) + ".json")) << o.str();
        std::ofstream(d / "numerical-config.txt") << a.numcfg;
        std::printf("gate_witness arm=%s run_id=%s profile=%s dir=%s prompts=%zu\n", label.c_str(), a.run_id.c_str(), a.profile.c_str(),
                    d.string().c_str(), a.prompts.size());
    }
    std::fflush(stdout);
}
}  // namespace

int main(int argc,char** argv) try {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    const auto start=std::chrono::steady_clock::now();
    // GATE_DIR: gate mode (below); budget_seconds 0 = no budget
    if(argc<5 || argc>7)throw std::runtime_error("MODEL manifest.tsv outdir rank [budget_seconds [GATE_DIR]]");
    const double budget=argc>=6?std::stod(argv[5]):0;
    const std::string gate_dir=argc==7?argv[6]:"";
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
    const strix::LoaderCapture gate_capture(!gate_dir.empty());
    strix::deepseek41::DevicePrefill prefill(argv[1],o);
    const auto gate_files=strix::LoaderCapture::snapshot();
    std::printf("loaded rank=%u seconds=%.3f weights=%llu overlay=%s trace=%d\n",rank,prefill.load_seconds(),static_cast<unsigned long long>(prefill.weight_bytes()),prefill.overlay_description().c_str(),trace);std::fflush(stdout);
    std::ifstream manifest(argv[2]);if(!manifest)throw std::runtime_error("manifest");
    struct Flags {
        std::uint32_t* p=nullptr;
        Flags(){if(hipMalloc(&p,8)!=hipSuccess)throw std::runtime_error("budget flags allocation");}
        ~Flags(){if(p)(void)hipFree(p);}
    } flags;
    unsigned budget_round=0;
    bool control_done=false;
    // Gate mode (GATE_DIR): per prompt of n tokens every arm prefills P = (n - 64) / 128 * 128 rows from position 0 (run:
    // the arm's policy, fast3b<R> with the bound), then the 64 rows [P, P + 64) teacher-forced from that state (extend,
    // the exact plan), and rank 0 dumps 65 rows for tools/strix-gate/fast_logit_gate.py: the run's last row (position
    // P - 1), then the 64 (GATE_DIR/<arm>/q-<case>-tf-r0.f32, full vocabulary), with the prompt's first P + 64 tokens as
    // GATE_DIR/tok/<case>.u32 and, at the end, GATE_DIR/<arm>/execution-r0.json from the executed configuration.
    std::map<std::string,GateArm> gate_arms;
    std::string gate_binary,gate_model;
    if(!gate_dir.empty()){
        strix::verify_loader_snapshot(gate_files);
        gate_model=strix::deepseek41::quality::model_witness(argv[1],prefill.overlay_path());
        strix::verify_loader_snapshot(gate_files);
        std::string peer_model(gate_model.size(),'\0');
        comm.exchange_host(std::as_bytes(std::span(gate_model)),std::as_writable_bytes(std::span(peer_model)),
                           strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control),0x6d6f646cU,0U});
        if(peer_model!=gate_model)throw std::runtime_error("gate mode: ranks loaded different model content");
    }
    if(!gate_dir.empty() && rank==0){
        gate_binary=strix::model_content_identity_hex(strix::sha256_file_content("/proc/self/exe",nullptr));
    }
    const auto unix_now=static_cast<long long>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
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
        std::vector<float> gate_ref;  // gate mode: the exact arm's 65 rows of this prompt
        for(const auto& label:arms){
            if(label.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_")!=std::string::npos)throw std::runtime_error("arm label "+label);
            auto precision=arm_precision(label);precision.trace=trace;
            const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            std::uint32_t stop=budget && elapsed+(n>4096?45:15)>budget,peer_stop=0;
            if(hipMemcpy(flags.p,&stop,4,hipMemcpyHostToDevice)!=hipSuccess)throw std::runtime_error("budget flag upload");
            comm.exchange(nullptr,flags.p,flags.p+1,4,strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control),15U,++budget_round});
            if(hipMemcpy(&peer_stop,flags.p+1,4,hipMemcpyDeviceToHost)!=hipSuccess)throw std::runtime_error("budget flag download");
            if(stop || peer_stop){
                std::printf("budget_stop next=%s arm=%s elapsed=%.3f\n",name.c_str(),label.c_str(),elapsed);std::fflush(stdout);
                if(!gate_dir.empty())strix::verify_loader_snapshot(gate_files);
                if(!gate_dir.empty() && rank==0)write_witnesses(gate_dir,gate_arms,rank,gate_binary,gate_model);
                return 0;
            }
            prefill.set_precision_for_new_sequence(precision);
            if(!gate_dir.empty()){
                if(n<192)throw std::runtime_error("gate mode: prompts of at least 192 tokens");
                const std::uint32_t P=(n-64)/128*128,total=P+64;
                const std::span<const std::uint32_t> all(ids.data(),total);
                const auto t0=std::chrono::steady_clock::now();
                std::uint32_t next=0;
                auto executed=precision;
                bool replay_oracle=false;
                auto pass=[&]{
                    prefill.set_cone(true);  // the prefill as served (exact suffix replay, or the arm's bounded plan)
                    auto a=prefill.run(all.first(P));
                    executed=strix::deepseek41::quality::executed_precision(precision,a);
                    replay_oracle=a.decoder_replay_oracle;
                    prefill.set_cone(false);  // the 64 teacher rows: every row of every layer
                    auto b=prefill.extend(all.subspan(P,64));
                    const std::size_t V=a.logits.size();
                    if(b.teacher_logits.size()!=64*V || b.logits.size()!=V)throw std::runtime_error("gate mode: teacher logit shape");
                    std::vector<float> z(65*V);
                    std::copy(a.logits.begin(),a.logits.end(),z.begin());
                    std::copy(b.teacher_logits.begin(),b.teacher_logits.end(),z.begin()+std::ptrdiff_t(V));
                    next=a.next_token;
                    return z;
                };
                auto z=pass();
                const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
                if(label=="exact" && !control_done){
                    if(const auto again=pass();again.size()!=z.size() || std::memcmp(again.data(),z.data(),z.size()*4))throw std::runtime_error("repeated incumbent run drift (gate mode)");
                    std::printf("exact_repeat_control=bitwise_equal case=%s\n",name.c_str());std::fflush(stdout);control_done=true;
                }
                unsigned nonfinite=0;for(float v:z)nonfinite+=!std::isfinite(v);
                if(nonfinite)throw std::runtime_error("nonfinite gate logits");
                const std::size_t V=z.size()/65;
                std::uint64_t hash=14695981039346656037ULL;
                for(float value:z){std::uint32_t bits;std::memcpy(&bits,&value,4);for(unsigned b=0;b<4;++b){hash=(hash^(bits&255U))*1099511628211ULL;bits>>=8;}}
                if(rank==0){
                    namespace fs=std::filesystem;
                    const fs::path ad=fs::path(gate_dir)/label,td=fs::path(gate_dir)/"tok";
                    fs::create_directories(ad);fs::create_directories(td);
                    const std::string tok_name=name+".u32",dump_name="q-"+name+"-tf-r0.f32";
                    std::ofstream(td/tok_name,std::ios::binary).write(reinterpret_cast<const char*>(all.data()),std::streamsize(total)*4);
                    std::ofstream(ad/dump_name,std::ios::binary).write(reinterpret_cast<const char*>(z.data()),std::streamsize(z.size())*4);
                    auto& g=gate_arms[label];
                    if(g.run_id.empty()){
                        g.run_id=label+"-"+std::to_string(unix_now)+"-"+std::to_string(::getpid());
                        g.profile=strix::deepseek41::quality::arm_profile(executed);
                        g.control=precision.null_ulp_layer<0?std::string("null"):"{\"kind\":\""+std::string(precision.null_ulp_layer==0?"layer0-null":"layer-null")+
                                  "\",\"layer\":"+std::to_string(precision.null_ulp_layer)+",\"magnitude\":1,\"unit\":\"ulp\"}";
                        g.numcfg=numerical_config(executed,o,replay_oracle);
                    }
                    std::ostringstream e;
                    e<<"{\"tokens\":"<<total<<",\"tokens_sha256\":\""<<sha_hex(all.data(),std::size_t(total)*4)<<"\",\"rows\":65,\"files\":{\"tf\":{\"name\":\""
                     <<dump_name<<"\",\"bytes\":"<<z.size()*4<<",\"sha256\":\""<<sha_hex(z.data(),z.size()*4)<<"\",\"width\":"<<V<<"}}}";
                    g.prompts[name]=e.str();
                }
                std::printf("{\"case\":\"%s\",\"arm\":\"%s\",\"rank\":%u,\"tokens\":%u,\"prefill\":%u,\"rows\":65,\"seconds\":%.6f,\"next\":%u,\"nonfinite\":%u,\"hash\":\"%016llx\"}\n",
                            name.c_str(),label.c_str(),rank,total,P,seconds,next,nonfinite,static_cast<unsigned long long>(hash));
                if(label=="exact"){gate_ref=std::move(z);std::fflush(stdout);continue;}
                if(gate_ref.size()!=z.size() || rank!=0){std::fflush(stdout);continue;}
                double sum=0;unsigned top=0;std::ostringstream kl;kl.precision(9);
                for(unsigned q=0;q<65;++q){
                    const float* x=gate_ref.data()+std::size_t(q)*V;const float* y=z.data()+std::size_t(q)*V;
                    auto lp=log_softmax(x,V),lq=log_softmax(y,V);double k=0;
                    for(std::size_t i=0;i<V;++i)k+=std::exp(lp[i])*(lp[i]-lq[i]);
                    sum+=k;top+=std::max_element(x,x+V)-x==std::max_element(y,y+V)-y;kl<<(q?",":"")<<k;
                }
                std::printf("{\"diag\":1,\"case\":\"%s\",\"arm\":\"%s\",\"rows\":65,\"mean_kl\":%.9g,\"top1\":%u,\"kl\":[%s]}\n",name.c_str(),label.c_str(),sum/65,top,kl.str().c_str());
                std::fflush(stdout);
                continue;
            }
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
    if(!gate_dir.empty())strix::verify_loader_snapshot(gate_files);
    if(!gate_dir.empty() && rank==0)write_witnesses(gate_dir,gate_arms,rank,gate_binary,gate_model);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"prefill-quality: %s\n",e.what());return 2;}
