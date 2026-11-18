// Actual production proposer + FAST target diagnostic. No qualification claim.
// MODEL SIDECAR TABLES ROPE PROMPT_IDS N G CAP NEW_OUTPUT ARM
// ARM real | correct | early | late | reject | zero | partial | stop | budget
#include "strix/cuda/qwen_prefill.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace qp=strix::cuda::qwen_prefill;
namespace md=strix::cuda::qwen_mtp_diagnostic;
namespace fs=std::filesystem;
constexpr std::size_t V=248320;
static void check(bool b,const char* s) { if (!b) throw std::runtime_error(s); }
template<class T> static void dump(const fs::path& p,const std::vector<T>& a) {
    std::ofstream f(p,std::ios::binary|std::ios::trunc);check(bool(f),"open export");
    if (!a.empty()) f.write(reinterpret_cast<const char*>(a.data()),static_cast<std::streamsize>(a.size()*sizeof(T)));
    f.flush();check(bool(f),"flush export");f.close();check(bool(f),"close export");
}
static void save(const fs::path& p,const qp::CanonicalState& a) {
    fs::create_directories(p);dump(p/"target-rows.raw",a.rows);dump(p/"fixed.raw",a.fixed);
    dump(p/"held.raw",a.held);dump(p/"mtp-live-pairs.raw",a.mtp_rows);
    dump(p/"pool-valid.u32",a.pool_valid);
    for (std::size_t i=0;i<a.pools.size();++i) dump(p/("pool-"+std::to_string(i)+".f32"),a.pools[i]);
    dump(p/"identity-position.u64",std::vector<std::uint64_t>{a.target,a.length,a.engine_binding,
        a.mtp_binding.session,a.mtp_binding.capacity,a.mtp_binding.sidecar,a.mtp_binding.phase});
}
static void same(const qp::CanonicalState& actual,const qp::CanonicalState& oracle) {
    check(actual.target==oracle.target && actual.length==oracle.length,"canonical target/position");
    check(actual.rows==oracle.rows,"full target KV/index");
    check(actual.mtp_rows==oracle.mtp_rows,"full committed MTP KV/index");
    check(actual.fixed==oracle.fixed && actual.held==oracle.held,"GDN/PLE/ngram/held state");
    check(actual.pools.size()==13 && oracle.pools.size()==13,"pool coverage");
    for (std::size_t i=0;i<13;++i) {
        check(actual.pool_valid[i]<=oracle.pool_valid[i],"retained pool lacks independent reference");
        check(actual.pools[i].empty() || std::memcmp(actual.pools[i].data(),oracle.pools[i].data(),actual.pools[i].size()*4)==0,"actual retained pool bytes");
        for (const auto v:actual.pools[i]) check(std::isfinite(v),"nonfinite retained pool");
    }
}
static void rows_equal(const std::vector<float>& a,const std::vector<float>& b) {
    check(a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*4)==0,"full finite raw vocabulary equality");
}
int main(int argc,char** argv) {
    try {
        check(argc==11,"MODEL SIDECAR TABLES ROPE PROMPT_IDS N G CAP NEW_OUTPUT ARM");
        const auto n=std::stoull(argv[6]),g=std::stoull(argv[7]),cap=std::stoull(argv[8]);
        check(n>=2 && g>=8 && g<=64 && cap<=UINT32_MAX && n<cap && n+g+9<cap,"bounded N/G/context");
        // At most eight simultaneous full snapshots, three logit journals, and
        // bounded per-cycle raw rows. Reject excess observability; never shrink capacity.
        const auto observed=(n+g+8)*32000ULL+128ULL*1024*1024;
        check(observed<=UINT64_MAX/8 && observed*8+g*V*12+8*V*8<=2ULL*1024*1024*1024,"2 GiB observability ceiling");
        (void)md::payload_bytes(8,V,8*V*4); // bounded before model allocation
        const std::string arm=argv[10];
        check(arm=="real" || arm=="correct" || arm=="early" || arm=="late" || arm=="reject" ||
              arm=="zero" || arm=="partial" || arm=="stop" || arm=="budget" || arm=="cancel","unknown arm");
        const fs::path out=argv[9];check(!fs::exists(out),"new output required");fs::create_directories(out);
        std::ifstream input(argv[5]);check(bool(input),"prompt open");
        std::vector<std::uint32_t> prompt;std::uint64_t token;
        while (input>>token) { check(token<V,"prompt vocabulary");if (prompt.size()<n) prompt.push_back(static_cast<std::uint32_t>(token)); }
        check(input.eof() && prompt.size()>=n,"prompt parse/extent");prompt.resize(static_cast<std::size_t>(n));
        qp::Config c;c.model_gguf=argv[1];c.mtp_sidecar_gguf=argv[2];c.tables_dir=argv[3];c.rope_table=argv[4];
        c.max_positions=static_cast<std::uint32_t>(cap);c.decode=true;c.export_state=false;c.release_by_layer=false;
        c.block_schedule={1024};c.moe_route_bf16=true;c.diagnostic_bf16_mtp=true;
        qp::Prefill e(c);check(e.numerical_identity().hash==0x46bf863a1bc53cd4ULL,"FAST target required");
        const qp::Sink drop=[](const qp::Piece&){};
        std::vector<std::uint32_t> history(prompt.begin(),prompt.end()-1);
        e.run(history,history.size(),drop);const auto initial=e.capture_canonical();save(out/"initial",initial);
        for (unsigned field=0;field<9;++field) {
            auto bad=initial;
            switch(field) {case 0:++bad.mtp_binding.target;break;case 1:++bad.mtp_binding.engine;break;
                case 2:++bad.mtp_binding.session;break;case 3:++bad.mtp_binding.capacity;break;
                case 4:++bad.mtp_binding.sidecar;break;case 5:++bad.mtp_binding.phase;break;
                case 6:bad.rows.pop_back();break;case 7:bad.mtp_rows.push_back(0);break;default:bad.pool_valid[0]=UINT32_MAX;break;}
            bool refused=false;try {e.restore_canonical(bad);}catch(const std::invalid_argument&) {refused=true;}
            check(refused,"mismatched/stale/alias/extent restore refusal");same(initial,e.capture_canonical());
        }
        dump(out/"memory-api.u64",std::vector<std::uint64_t>{e.session_bytes(),e.row_bytes(),e.fixed_bytes(),observed*8+g*V*12+8*V*8,cap,1024,1});
        // Independent greedy reference of THIS target, using the same single weight owner.
        std::uint32_t pending=prompt.back();std::vector<std::uint32_t> plain_ids,plain_inputs;
        std::vector<float> plain_raw;plain_raw.reserve(static_cast<std::size_t>(g)*V);
        for (std::size_t i=0;i<g;++i) {
            plain_inputs.push_back(pending);auto r=e.step(&pending,1,e.length(0),true);
            md::observe(r.logits,r.argmax,1,V);plain_ids.push_back(r.argmax[0]);pending=r.argmax[0];
            plain_raw.insert(plain_raw.end(),r.logits.begin(),r.logits.end());
        }
        dump(out/"plain-ids.u32",plain_ids);dump(out/"plain-inputs.u32",plain_inputs);dump(out/"plain-raw.f32",plain_raw);
        e.restore_canonical(initial);pending=prompt.back();std::vector<std::uint32_t> emitted;
        std::uint32_t cycle=0;bool stop=false;
        while (emitted.size()<g && !stop) {
            const auto dir=out/("cycle-"+std::to_string(cycle++));fs::create_directories(dir);
            const auto before=e.capture_canonical();const auto pos=e.length(0);
            std::vector<std::uint32_t> rows{pending};
            // Only this arm is credited with actual MTP proposal composition.
            if (arm=="real") {
                const auto d=e.draft(pending,7,0.7F);
                check(d.tokens.size()==d.probability.size() && d.tokens.size()<=7,"actual draft coverage");
                for (std::size_t i=0;i<d.tokens.size();++i) {
                    check(d.tokens[i]<V && std::isfinite(d.probability[i]) && d.probability[i]>=0 && d.probability[i]<=1,"actual draft output");
                }
                rows.insert(rows.end(),d.tokens.begin(),d.tokens.end());dump(dir/"actual-proposals.u32",d.tokens);dump(dir/"actual-probability.f32",d.probability);
            } else {
                const auto width=std::min<std::size_t>(8,g-emitted.size());
                rows.assign(plain_inputs.begin()+static_cast<std::ptrdiff_t>(emitted.size()),plain_inputs.begin()+static_cast<std::ptrdiff_t>(emitted.size()+width));
                if (rows.size()>1 && (arm=="early" || arm=="reject")) rows[1]=(rows[1]+1)%V;
                if (rows.size()>1 && arm=="late") rows.back()=(rows.back()+1)%V;
                if (arm=="reject") for (std::size_t i=2;i<rows.size();++i) rows[i]=(rows[i]+1)%V;
            }
            auto r=e.verify(rows.data(),static_cast<std::uint32_t>(rows.size()),pos,true);
            md::observe(r.logits,r.argmax,rows.size(),V);dump(dir/"fed.u32",rows);dump(dir/"verify-ids.u32",r.argmax);dump(dir/"verify-raw.f32",r.logits);
            auto budget=static_cast<std::uint32_t>(g-emitted.size());if (arm=="budget") budget=1;
            const auto selected=md::select(rows,r.argmax,budget,arm=="stop"?r.argmax[0]:UINT32_MAX);
            auto keep=selected.keep;if (arm=="partial") keep=1;
            if ((arm=="zero" && cycle==1) || arm=="cancel") keep=0;
            e.commit_cohort({keep});
            // Capture actual kept state BEFORE any restore, rebuild, or reference computation.
            const auto actual=e.capture_canonical();save(dir/"actual-before-reference",actual);
            dump(dir/"controller.u64",std::vector<std::uint64_t>{pending,keep,keep?r.argmax[keep-1]:pending,budget,selected.stopped?1ULL:0ULL});
            // Scalar feeding of the complete proposed window checks even rejected raw rows.
            e.run(history,history.size(),drop);qp::CanonicalState kept=e.capture_canonical();std::vector<float> scalar_raw;
            for (std::size_t i=0;i<rows.size();++i) {
                const auto sr=e.step(&rows[i],1,e.length(0),true);md::observe(sr.logits,sr.argmax,1,V);
                scalar_raw.insert(scalar_raw.end(),sr.logits.begin(),sr.logits.end());
                if (i+1==keep) kept=e.capture_canonical();
            }
            dump(dir/"scalar-full-window.f32",scalar_raw);rows_equal(r.logits,scalar_raw);save(dir/"scalar-kept",kept);same(actual,kept);
            if (!keep) { e.restore_canonical(before);if (arm=="cancel") return 3;continue; }
            for (std::size_t i=0;i<keep;++i) {
                check(r.argmax[i]==plain_ids[emitted.size()+i],"complete emitted IDs vs FAST plain");
                check(std::memcmp(r.logits.data()+i*V,plain_raw.data()+(emitted.size()+i)*V,V*4)==0,"committed raw rows vs FAST plain");
            }
            pending=r.argmax[keep-1];history.insert(history.end(),rows.begin(),rows.begin()+keep);
            emitted.insert(emitted.end(),r.argmax.begin(),r.argmax.begin()+keep);
            dump(dir/"full-history.u32",history);dump(dir/"emitted-total.u32",emitted);
            // Independent continuation, then replay the actual retained pools without repairing them.
            e.restore_canonical(kept);const auto oracle_next=e.step(&pending,1,e.length(0),true);const auto oracle_state=e.capture_canonical();
            e.restore_canonical(actual);const auto replay=e.step(&pending,1,e.length(0),true);md::observe(replay.logits,replay.argmax,1,V);
            rows_equal(replay.logits,oracle_next.logits);check(replay.argmax==oracle_next.argmax,"replay IDs");same(e.capture_canonical(),oracle_state);
            dump(dir/"continuation-raw.f32",replay.logits);save(dir/"continuation-state",e.capture_canonical());
            e.restore_canonical(actual);stop=selected.stopped;
        }
        check(!emitted.empty() && (stop || emitted.size()==g),"complete generation budget");
        dump(out/"emitted.u32",emitted);save(out/"final",e.capture_canonical());
        md::publish_complete(out,"SAME_TARGET_DIAGNOSTIC arm="+arm+" actual_proposer="+(arm=="real"?"true":"false")+"\n"+e.describe()+"\n");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what());return 1; }
}
