// Same-target execution diagnostic, not a model-quality campaign.
// qwen_route_bf16_e2e MODEL TABLES ROPE PROMPT_IDS N G MAX_POS OUTPUT candidate|off [all|serial|window:W|chunk:R]
// Requires explicit STRIX_QRTX_GRAPHS=0 for candidate. No MTP/cache composition.
#include "strix/cuda/qwen_prefill.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qp = strix::cuda::qwen_prefill;
namespace qr = strix::cuda::qwen_route;
namespace fs = std::filesystem;
static constexpr std::size_t vocab = 248320U;

static void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}
template<class T> static void dump(const fs::path& path, const std::vector<T>& data) {
    require(data.size() <= static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())/sizeof(T),
            "diagnostic file extent");
    std::ofstream f(path,std::ios::binary|std::ios::trunc);
    require(bool(f),"open diagnostic file");
    if (!data.empty()) f.write(reinterpret_cast<const char*>(data.data()),static_cast<std::streamsize>(data.size()*sizeof(T)));
    f.flush(); require(bool(f),"flush diagnostic file"); f.close(); require(bool(f),"close diagnostic file");
}
static std::vector<std::uint32_t> tokens(const char* path) {
    std::ifstream f(path);
    require(bool(f),"open prompt IDs");
    std::string text((std::istreambuf_iterator<char>(f)),{});
    for (auto& c:text) if (c==',') c=' ';
    std::istringstream in(text);
    std::vector<std::uint32_t> out;
    std::uint64_t id;
    while (in>>id) { require(id<vocab,"prompt token outside vocabulary"); out.push_back(static_cast<std::uint32_t>(id)); }
    require(in.eof(),"invalid prompt IDs");
    return out;
}
static void same_state(const qp::CanonicalState& a,const qp::CanonicalState& b,const std::string& arm) {
    require(a.target==b.target && a.length==b.length,arm+" target/length");
    require(a.rows==b.rows,arm+" complete canonical KV/index");
    require(a.fixed==b.fixed,arm+" complete GDN/history/ngram");
    require(a.held==b.held,arm+" complete held row");
}
static void save_state(const fs::path& dir,const qp::CanonicalState& s) {
    fs::create_directories(dir);
    dump(dir/"kv-index.raw",s.rows);
    dump(dir/"fixed.raw",s.fixed);
    dump(dir/"held.raw",s.held);
    std::vector<std::uint64_t> metadata{s.target,s.length};
    dump(dir/"target-length.u64",metadata);
}
template<class F> static void refuses(F f,const std::string& name) {
    bool refused=false;
    try { f(); } catch (const std::invalid_argument& e) {
        refused=std::string(e.what()).find(name)!=std::string::npos;
    }
    require(refused,"expected named refusal: "+name);
}
int main(int argc,char** argv) {
    try {
        if (argc!=10 && argc!=11) {
            std::fprintf(stderr,"usage: qwen_route_bf16_e2e MODEL TABLES ROPE PROMPT_IDS N G MAX_POS OUTPUT candidate|off [all|serial|window:W|chunk:R]\n");
            return 2;
        }
        const auto n=std::stoull(argv[5]), g64=std::stoull(argv[6]), max64=std::stoull(argv[7]);
        require(g64>0 && g64<=UINT32_MAX && max64<=UINT32_MAX,"G/MAX_POS range");
        const auto g=static_cast<std::uint32_t>(g64);
        const auto prompt=tokens(argv[4]);
        require(n>=2 && n<=prompt.size() && n<max64 && g64<=max64-n,"N/G/context range");
        const std::string mode=argv[9];
        require(mode=="candidate" || mode=="off","candidate|off");
        const bool candidate=mode=="candidate";
        const std::string arm=argc==11 ? argv[10] : "all";
        bool valid_arm=arm=="all" || arm=="serial";
        for (unsigned w=1;w<=8;++w) valid_arm=valid_arm || arm=="window:"+std::to_string(w);
        for (const unsigned r : {1U,7U,16U,17U,257U,1024U}) valid_arm=valid_arm || arm=="chunk:"+std::to_string(r);
        require(valid_arm,"unknown diagnostic arm");
        const fs::path output=argv[8];
        require(!fs::exists(output),"diagnostic output must be new");
        fs::create_directories(output);
        qp::Config c;
        c.model_gguf=argv[1];c.tables_dir=argv[2];c.rope_table=argv[3];
        c.max_positions=static_cast<std::uint32_t>(max64);
        c.decode=true;c.export_state=false;c.release_by_layer=false;
        c.block_schedule={1024};c.moe_route_bf16=candidate;
        require(qr::environment(c.moe_route_bf16).enabled==candidate,"candidate suppressed by OFF master");
        qp::Prefill p(c);
        require(p.route_storage_policy().enabled==candidate,"candidate suppressed by OFF master");
        const auto id=p.numerical_identity();
        std::printf("target %016llx prefill %s decode %s storage %s\n",
            static_cast<unsigned long long>(id.hash),id.prefill.data(),id.decode.data(),id.storage.data());
        const qp::Sink drop=[](const qp::Piece&){};
        p.run(prompt,n-1U,drop);
        const auto initial=p.capture_canonical();
        save_state(output/"initial",initial);
        std::vector<std::uint32_t> consumed, ids;
        std::vector<float> logits;
        require(g64<=std::numeric_limits<std::size_t>::max()/vocab,"logit extent");
        logits.reserve(static_cast<std::size_t>(g)*vocab);
        consumed.push_back(prompt[static_cast<std::size_t>(n-1U)]);
        for (std::uint32_t i=0;i<g;++i) {
            const auto result=p.step(&consumed[i],1,n-1U+i,true);
            require(result.ok && result.argmax.size()==1 && result.logits.size()==vocab,"serial result extent/status");
            ids.push_back(result.argmax[0]);
            logits.insert(logits.end(),result.logits.begin(),result.logits.end());
            if (i+1U<g) consumed.push_back(result.argmax[0]);
        }
        dump(output/"serial-consumed.u32",consumed);
        dump(output/"serial-ids.u32",ids);
        dump(output/"serial-logits.f32",logits);
        const auto final=p.capture_canonical();
        save_state(output/"serial-final",final);
        // Restore bindings are checked before mutation/copy.
        auto wrong=initial;wrong.target^=1U;
        refuses([&]{p.restore_canonical(wrong);},"target identity");
        wrong=initial;++wrong.engine_binding;
        refuses([&]{p.restore_canonical(wrong);},"engine binding");
        same_state(final,p.capture_canonical(),"refused restore preserves state");
        if (candidate) {
            refuses([&]{p.set_graphs(true);},"graphs");
            refuses([&]{p.set_graph_table(true);},"graphs");
            refuses([&]{p.set_multiseg(true);},"multisegment");
            refuses([&]{p.create_session();},"cohort");
            refuses([&]{p.enable_prefill_overlap(16);},"prefill-overlap");
            refuses([&]{p.draft(0U,1U,0.7F);},"MTP-sidecar");
            refuses([&]{p.import_state(0,1,{},nullptr);},"untagged-state-import");
            std::uint8_t byte=0;
            refuses([&]{p.export_fixed(0,&byte);},"prefix-cache");
        }
        // All comparisons are against the serial run of THIS target.
        // Complete raw arrays and final canonical bytes are persisted for each arm.
        for (std::uint32_t width=1;width<=8;++width) {
            if (arm!="all" && arm!="window:"+std::to_string(width)) continue;
            p.restore_canonical(initial);
            same_state(initial,p.capture_canonical(),"restored initial");
            std::vector<float> arm_logits;
            std::vector<std::uint32_t> arm_ids;
            std::uint32_t at=0;
            while (at<g) {
                const auto rows=std::min(width,g-at);
                if (at==0) {
                    const auto before=p.capture_canonical();
                    const auto speculative=p.verify(consumed.data(),rows,n-1U,true);
                    require(speculative.logits.size()==static_cast<std::size_t>(rows)*vocab,"zero-keep logits extent");
                    require(std::memcmp(speculative.logits.data(),logits.data(),speculative.logits.size()*4U)==0,
                            "zero-keep complete raw logits");
                    p.commit_cohort({0});
                    same_state(before,p.capture_canonical(),"zero-keep rollback");
                }
                const auto result=p.verify(consumed.data()+at,rows,n-1U+at,true);
                require(result.ok && result.argmax.size()==rows &&
                        result.logits.size()==static_cast<std::size_t>(rows)*vocab,"window result extent/status");
                const auto offset=static_cast<std::size_t>(at)*vocab;
                if (std::memcmp(result.logits.data(),logits.data()+offset,result.logits.size()*4U)!=0) {
                    dump(output/"failure-window-logits.f32",result.logits);
                    dump(output/"failure-window-ids.u32",result.argmax);
                    throw std::runtime_error("window complete raw logits width "+std::to_string(width)+
                                             " at "+std::to_string(at));
                }
                require(std::equal(result.argmax.begin(),result.argmax.end(),ids.begin()+at),"window complete IDs");
                // Exercise rollback within a window, then recompute its unkept tail.
                const auto keep=rows>1U ? rows-1U : 1U;
                arm_logits.insert(arm_logits.end(),result.logits.begin(),result.logits.begin()+static_cast<std::ptrdiff_t>(keep)*static_cast<std::ptrdiff_t>(vocab));
                arm_ids.insert(arm_ids.end(),result.argmax.begin(),result.argmax.begin()+keep);
                p.commit(keep);at+=keep;
            }
            require(arm_logits.size()==logits.size() && std::memcmp(arm_logits.data(),logits.data(),logits.size()*4U)==0,
                    "complete committed logit stream");
            require(arm_ids==ids,"complete committed ID stream");
            const auto dir=output/("window-"+std::to_string(width));
            fs::create_directories(dir);
            dump(dir/"logits.f32",arm_logits);dump(dir/"ids.u32",arm_ids);
            const auto state=p.capture_canonical();save_state(dir/"state",state);
            same_state(final,state,"window final canonical");
        }
        // Chunk boundaries include sub-window, pad-to-16 and tiled transitions.
        for (const std::uint32_t chunk : {1U,7U,16U,17U,257U,1024U}) {
            if (arm!="all" && arm!="chunk:"+std::to_string(chunk)) continue;
            std::uint64_t from=0;
            while (from<n-1U) from=p.prefill_chunk(0,prompt,n-1U,from,chunk);
            const auto state=p.capture_canonical();
            save_state(output/("chunk-"+std::to_string(chunk)),state);
            same_state(initial,state,"chunked prefill "+std::to_string(chunk));
        }
        std::ofstream receipt(output/"COMPLETE.txt",std::ios::trunc);
        receipt<<"SAME_TARGET_EXECUTION_PASS arm="<<arm<<"\n"<<p.describe()<<"\n";
        receipt.flush();require(bool(receipt),"completion flush");receipt.close();require(bool(receipt),"completion close");
        std::printf("SAME TARGET EXECUTION PASS arm=%s; model quality NOT EVALUATED\n",arm.c_str());
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"FAIL %s\n",e.what());
        return 1;
    }
}
