// Native ABI doubles in this translation unit are host control evidence only.
#include "strix/deepseek41_batch_dspark.hpp"
#include "strix/deepseek41_prefill.hpp"
#include "strix/deepseek41_session_cache.hpp"
#include "strix/deepseek41_serve_sched.hpp"
#include "strix/deepseek41_serve_observe.hpp"
#include "strix/deepseek41_serve_cache_policy.hpp"
#include "strix/deploy/binding.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/pair/lockstep.hpp"
#include "strix/ds41_attention.hpp"
#include "strix/deepseek41_attention_host.hpp"
#include "strix/serve/api.hpp"
#include "strix/serve/deepseek_chat.hpp"
#include "strix/serve/session_api.hpp"
#include "strix/serve/session_runtime.hpp"
#if __has_include("deepseek41/http_session_owner.hpp")
#include "deepseek41/http_session_owner.hpp"
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>

namespace witness {
std::vector<std::string> trace;
void record(std::string s) { trace.push_back(std::move(s)); }
void check(bool b, const char* text) { if (!b) throw std::runtime_error(text); }
std::string bytes(std::span<const std::byte> b) {
    static constexpr char hex[]="0123456789abcdef";
    std::string s; for(auto c:b) {auto v=std::to_integer<unsigned>(c);s+=hex[v>>4];s+=hex[v&15];}return s;
}
std::map<unsigned,unsigned> positions, capacities;
unsigned pool=0, context=64, steps=0;
std::uint64_t import_digest=0;
std::string scenario;
std::function<void()> after_step;
std::function<void(std::span<const std::byte>,std::span<std::byte>)> peer;
strix::serve::GenerationQueue* queue=nullptr;
}
// Actual native public declarations, with deterministic device implementations.
namespace strix::deepseek41 {
struct BatchForward::Impl {};
BatchForward::BatchForward(const std::filesystem::path&, const BatchForwardOptions&) : impl_(std::make_unique<Impl>()) {}
BatchForward::~BatchForward() = default;
std::uint32_t BatchForward::row_limit() const noexcept {return max_rows;}
void BatchForward::reset(std::uint32_t s) { witness::record("reset physical="+std::to_string(101+s*17)); witness::positions[s]=0; }
bool BatchForward::reserve(std::uint32_t s,std::uint32_t n) {
    witness::record("reserve physical="+std::to_string(101+s*17)+" rows="+std::to_string(n));
    witness::capacities.erase(s);
    unsigned used=0;for(auto [id,v]:witness::capacities){(void)id;used+=v;}
    if(witness::pool && used+n>witness::pool)return false;
    witness::capacities[s]=n;reset(s);return true;
}
void BatchForward::release(std::uint32_t s) {witness::record("release physical="+std::to_string(101+s*17));witness::capacities.erase(s);}
std::uint32_t BatchForward::capacity(std::uint32_t s) const {return witness::pool?witness::capacities[s]:witness::context;}
std::uint64_t BatchForward::pool_free_positions() const {unsigned used=0;for(auto [s,n]:witness::capacities){(void)s;used+=n;}return witness::pool-used;}
std::uint64_t BatchForward::pool_free_bytes() const {return pool_free_positions()*288;}
std::uint64_t BatchForward::weight_bytes() const noexcept {return 0;}
std::uint64_t BatchForward::state_bytes_per_slot() const noexcept {return 0;}
std::uint64_t BatchForward::workspace_bytes() const noexcept {return 0;}
std::uint64_t BatchForward::pool_bytes() const noexcept {return witness::pool*288;}
std::string BatchForward::trellis_identity() const {return {};}
std::uint64_t BatchForward::trellis_bytes() const noexcept {return 0;}
std::uint64_t BatchForward::trellis_library_bytes() const noexcept {return 0;}
std::pair<std::uint64_t,std::uint64_t> BatchForward::device_memory() const {return {0,0};}
void* BatchForward::stream() const noexcept {return nullptr;}
void BatchForward::quiesce() {witness::record("quiesce");}
void BatchForward::import_state(std::uint32_t s,const AttentionState& a) {witness::record("import physical="+std::to_string(101+s*17)+" position="+std::to_string(a.position));witness::positions[s]=a.position;}
AttentionState BatchForward::export_state(std::uint32_t s,std::uint32_t) const {
    AttentionState a;a.position=witness::positions[s];
    if(!witness::scenario.empty()){
        a.layers.resize(1);auto& l=a.layers[0];auto p=ds41::attn::released_plan()[0];
        l.owns_kv=p.owns_kv;l.owns_index=p.owns_index;l.ratio=p.ratio;
        l.ring.resize(AttentionHost::ring_capacity*ds41::attn::kWindowEntryBytes,23);
    }return a;
}
const std::vector<std::uint32_t>& BatchForward::step(std::span<const BatchRow> rows) {
    witness::check(++witness::steps<1000,"native step bound");
    std::string t="step";tokens_.clear();
    for(auto r:rows){t+=" physical="+std::to_string(101+r.slot*17)+" token="+std::to_string(r.token);++witness::positions[r.slot];tokens_.push_back(r.token+1);}
    witness::record(t);if(witness::after_step)witness::after_step();return tokens_;
}
void BatchForward::exchange_host(std::span<const std::byte> out,std::span<std::byte> in) {
    witness::record("exchange "+witness::bytes(out));std::copy(out.begin(),out.end(),in.begin());
    if(witness::peer)witness::peer(out,in);
}
// Native speculation double: distinct accepted and rejected suffixes. The
// actual lead still owns grouped planning, commits, flush and publication order.
struct BatchSpeculator::Slots {};
struct BatchSpeculator::ParkedPlan {PlannedStep plan;};
BatchSpeculator::BatchSpeculator(BatchForward& f,ds41::dspark::DeviceDrafter& d,std::uint32_t n,BatchSpecOptions o)
    :target_(f),drafter_(d),options_(o),controller_(n),slots_(std::make_unique<Slots>()),owners_(n),in_step_(n),coop_pending_(n){}
BatchSpeculator::~BatchSpeculator() = default;
void BatchSpeculator::start(std::uint32_t s,std::span<const std::uint32_t>) {witness::record("spec start "+std::to_string(s));}
void BatchSpeculator::imported(std::uint32_t s,std::span<const std::uint32_t>,std::uint32_t,std::uint32_t,std::array<const float*,3>,std::uint32_t,void*) {witness::record("spec imported "+std::to_string(s));}
const PlannedStep& BatchSpeculator::plan(std::span<BatchSequence* const> seqs,std::span<const std::uint32_t> remaining) {
    plan_=plan_step(seqs,options_.prompt_rows);witness::record("spec plan");
    if(allow_drafts_)for(unsigned i=0;i<seqs.size();++i){const auto& s=*seqs[i];
        if(!s.reading_prompt() && remaining[i]>1 && plan_.rows.size()<BatchForward::max_rows){
            auto at=std::find(plan_.owner.begin(),plan_.owner.end(),i)-plan_.owner.begin()+1;
            const bool accepted=(witness::steps%2)==0;
            plan_.rows.insert(plan_.rows.begin()+at,{s.slot,s.generated.back()+(accepted?1:9)});
            plan_.owner.insert(plan_.owner.begin()+at,i);plan_.emits.insert(plan_.emits.begin()+at,true);
        }
    }return plan_;
}
std::vector<std::pair<std::uint32_t,std::uint32_t>> BatchSpeculator::finish(std::span<BatchSequence* const> seqs,std::span<const std::uint32_t> tokens) {
    witness::record("spec finish");PlannedStep kept;std::vector<unsigned> ids;
    std::vector<std::pair<unsigned,unsigned>> commits;
    for(unsigned i=0;i<seqs.size();++i){unsigned count=0;bool accept=true;
        for(unsigned k=0;k<plan_.rows.size();++k)if(plan_.owner[k]==i){
            if(count && plan_.rows[k].token!=tokens[k-1])accept=false;
            if(accept){kept.rows.push_back(plan_.rows[k]);kept.owner.push_back(i);kept.emits.push_back(plan_.emits[k]);ids.push_back(tokens[k]);++count;}
        }
        unsigned all=static_cast<unsigned>(std::count(plan_.owner.begin(),plan_.owner.end(),i));
        if(count<all){commits.push_back({seqs[i]->slot,count});witness::record("commit "+std::to_string(seqs[i]->slot)+" kept="+std::to_string(count));witness::positions[seqs[i]->slot]-=all-count;}
    }apply_step(seqs,kept,ids);return commits;
}
std::vector<std::uint32_t> BatchSpeculator::take_commits() {return {};}
std::shared_ptr<BatchSpeculator::ParkedPlan> BatchSpeculator::park() {witness::record("spec park");return std::make_shared<ParkedPlan>(plan_);}
void BatchSpeculator::resume(const std::shared_ptr<ParkedPlan>& p) {witness::record("spec resume");plan_=p->plan;}
}
namespace strix::ds41::dspark {
struct DeviceDrafter::Impl {};
DeviceDrafter::DeviceDrafter(const Weights&) :impl_(std::make_unique<Impl>()){}
DeviceDrafter::~DeviceDrafter()=default;
BatchWindowController::BatchWindowController(std::uint32_t n):slot_seen_(n),slot_hits_(n){}
}
namespace strix::deepseek41 {
// Clock-derived session envelope ID, fixed independently on both replay paths.
std::string fixture_session_generation(){return "host-clock-generation";}
}
namespace strix::dist {
struct CommunicatorStats {
    std::uint64_t exchanges=0,bytes_sent=0;
    double seconds=0,stage=0,wire=0,check=0,overlap=0,host_reuse_wait=0,pipeline_reuse_wait=0,finish_wait=0;
};
class Communicator {
public:
    CommunicatorStats stats() const {return {};}
    void* device_exchange() const {return nullptr;}
};
}
namespace strix::ds41::hetero {
class Stage {
public:
    explicit Stage(deepseek41::BatchForward& f):fwd(f){}
    void reset(unsigned s){witness::record("front reset "+std::to_string(s));}
    void import_state(unsigned s,const deepseek41::AttentionState&){witness::record("front import "+std::to_string(s));}
    void submit(std::span<const deepseek41::BatchRow>){witness::record("front submit");}
    const std::vector<std::uint32_t>& step(std::span<const deepseek41::BatchRow> r){witness::record("front step");return fwd.step(r);}
    void commit(unsigned,unsigned){witness::record("front commit");}
    deepseek41::BatchForward& fwd;
};
}
namespace {
using strix::deepseek41::BatchForward;
using strix::deepseek41::BatchRow;
using strix::deepseek41::BatchSequence;
namespace serve=strix::serve;
using Clock=std::chrono::steady_clock;
#include "args.inc"
[[noreturn]] void poisoned(unsigned,const std::string& e){throw std::runtime_error(e);}
#include "control.inc"
#include "outcome.inc"
constexpr std::uint32_t kMinRows=9;
// Explicit double of the native prefill ABI. No physical resource is adopted.
class SlotPrefill {
public:
    class Task {
    public:
        Task(unsigned s,std::vector<unsigned> p):slot(s),prompt(std::move(p)),ends{static_cast<unsigned>(prompt.size())}{}
        bool mid_piece()const{return false;} unsigned piece_end()const{return 0;}
        unsigned rows()const{return at;}const std::vector<unsigned>& default_ends()const{return ends;}
        unsigned units_within(double)const{return 1;}unsigned unit_position()const{return units;}
        double unit_max_ms()const{return 0;}double step_max_ms()const{return 0;}
        bool done()const{return at==prompt.size();}
        bool step(unsigned end,unsigned){at=std::min(end,at+16);++units;witness::record("prefill unit "+std::to_string(slot)+" rows="+std::to_string(at));if(witness::after_step)witness::after_step();return done();}
        void cancel(){witness::record("prefill abort "+std::to_string(slot));}
        PrefillOutcome finish(){witness::record("prefill finish "+std::to_string(slot));witness::positions[slot]=static_cast<unsigned>(prompt.size());PrefillOutcome o;o.next_token=prompt.back()+1;o.pieces=1;return o;}
        unsigned slot,at=0,units=0;std::vector<unsigned> prompt,ends;
    };
    bool has_cache()const{return false;}
    strix::deepseek41::PrefixLookup lookup(const std::vector<unsigned>&){return {};}
    bool prepare(const std::vector<unsigned>&,const strix::deepseek41::PrefixLookup&){return true;}
    void forget(const std::vector<unsigned>&,const strix::deepseek41::PrefixLookup&){}
    bool admit(unsigned,bool=true){return false;}
    std::unique_ptr<Task> begin(unsigned s,const std::vector<unsigned>& p,const strix::deepseek41::PrefixLookup&,bool,unsigned){witness::record("prefill begin "+std::to_string(s));return std::make_unique<Task>(s,p);}
    PrefillOutcome run(unsigned s,const std::vector<unsigned>& p,const strix::deepseek41::PrefixLookup&,bool,unsigned){Task t(s,p);return t.finish();}
    void on_prefill(unsigned,const std::vector<unsigned>&){}
    void on_reset(unsigned){}
    unsigned on_step(const std::vector<BatchRow>&,const std::vector<unsigned>&){return 0;}
    void snapshot(unsigned){}
    std::uint64_t disk_bytes()const{return 0;}double disk_seconds()const{return 0;}
    strix::deepseek41::PrefixCacheStats cache_stats()const{return {};}
};
class Watchdog {public:void pause(bool){} void progress(){}};
#include "transfer.inc"
using Sessions=strix::deepseek41::PrefixCache::Sessions;
using SessionCheckpoint=strix::deepseek41::SessionCheckpoint;
using SessionBytes=strix::deepseek41::SessionBytes;
strix::deepseek41::SessionExchange session_exchange(BatchForward& f){return [&f](std::span<const std::uint8_t> b){return strix::deepseek41::session_peer_exchange(b,[&](auto o,auto i){f.exchange_host(o,i);});};}
#include "session.inc"
#include "records.inc"
std::string plans_since_ready(){return "host-native-double";}
struct Server {void stop(){witness::record("server stop");if(!witness::scenario.empty()){auto q=witness::queue->snapshot();witness::record("queue "+std::to_string(q.queued)+" "+std::to_string(q.running)+" "+std::to_string(q.completed)+" "+std::to_string(q.failed)+" "+std::to_string(q.cancelled)+" "+std::to_string(q.prompt_tokens)+" "+std::to_string(q.generated_tokens));}}};
using Prepare=std::function<void(serve::GenerationQueue&,serve::ApiService&,serve::ApiConfig&,ImportBox&,SessionBox&)>;
void actual_lead(Args a,const Prepare& prepare) {
    BatchForward fwd({},{});strix::dist::Communicator comm;Watchdog dog;
    std::unique_ptr<strix::deepseek41::BatchSpeculator> spec;
    strix::ds41::dspark::DeviceDrafter drafter({});
    if(witness::scenario=="spec" || witness::scenario=="hetero-spec")spec=std::make_unique<strix::deepseek41::BatchSpeculator>(fwd,drafter,a.slots,strix::deepseek41::BatchSpecOptions{});
    SlotPrefill native_prefill;SlotPrefill* prefill=a.device_prefill?&native_prefill:nullptr;
    strix::ds41::hetero::Stage native_stage(fwd);
    strix::ds41::hetero::Stage* stage=witness::scenario.starts_with("hetero")?&native_stage:nullptr;
    double hetero_step_seconds=0;
    // Rank checksum reporting is outside this host control fixture.
    struct StepTokenDigest { void add(std::span<const std::uint32_t>) {} } step_tokens;
    const auto sched=strix::deepseek41::sched::options_from_env(a.device_prefill);
    serve::GenerationQueue queue(a.max_queue);
    witness::queue=&queue;
    witness::peer=[](auto out,auto in){if(out.size()!=512)return;Control c;std::memcpy(&c,out.data(),offsetof(Control,extra_rows));
        if(c.command==static_cast<unsigned>(Command::import_state))witness::import_digest=c.pkey;
        if(c.command==static_cast<unsigned>(Command::heartbeat)){c.pkey=witness::import_digest?witness::import_digest:1;std::memcpy(in.data(),&c,offsetof(Control,extra_rows));}};
    serve::ApiConfig config;config.model_id="host-ds41";config.default_max_tokens=3;
    config.sampling_policy="greedy";config.default_thinking=false;
    std::vector<std::string> words{"a","b","c","d","e","f","g","h","i","j","<eos>"};
    auto tok=serve::BpeTokenizer::from_parts(words,std::vector<std::int32_t>(words.size(),1),{},"qwen35");
    serve::ApiService api(config,tok,queue,a.context,false);api.set_state("ready");
    ServeState state;ImportBox imports;SessionBox session_jobs;
    std::unique_ptr<Sessions> sessions;
    if(witness::scenario=="named-conflict"){
        strix::deepseek41::SessionCompat compat;compat.fields["fixture"]="host-control";
        sessions=std::make_unique<Sessions>(std::move(compat),0,1U<<20U,std::filesystem::path{},8,0);
    }
    Server server;
    config.on_drain=[&]{state.draining.store(true);queue.shutdown();};
    prepare(queue,api,config,imports,session_jobs);
    std::thread http_thread([]{});
#include "ownership.inc"
#ifdef DS41_OWNER_ADOPTED
    bool http_stopped=false;
    native_stop_http=[&]{if(!http_stopped){server.stop();http_stopped=true;}};
#endif
#define session_generation fixture_session_generation
#include "owner.inc"
#undef session_generation

struct Writer final:serve::HttpResponseWriter {
    int status=0;std::string data;
    void send(int s,std::string_view,std::string_view b,const std::vector<std::pair<std::string,std::string>>&)override{status=s;data=b;}
    bool begin_stream(int s,std::string_view,const std::vector<std::pair<std::string,std::string>>&)override{status=s;return true;}
    bool write(std::string_view b)override{data+=b;return true;}void end_stream()override{}
    bool client_gone()const override{return false;}
};
std::string run_case() {
    witness::scenario.clear();witness::after_step={};witness::peer={};witness::pool=0;
    witness::trace.clear();witness::positions.clear();witness::capacities.clear();witness::steps=0;stop_requested().store(false);
    Args a;a.context=64;a.slots=2;a.device_prefill=false;a.pool=0;
    std::vector<std::shared_ptr<serve::GenerationTicket>> tickets;
    std::thread observer;
    actual_lead(a,[&](auto& q,auto&,auto&,auto&,auto&){
        for(unsigned n=0;n<4;++n){serve::GenerationJob j;j.prompt={n+1,n+2};j.max_tokens=n+1;
            if(n==1)j.stop_tokens={5};tickets.push_back(q.submit(std::move(j)));}
        observer=std::thread([&]{for(auto& t:tickets){bool done=false;while(!done)(void)t->wait(std::chrono::milliseconds(10),done,false);}stop_requested().store(true);});
    });
    observer.join();
    std::ostringstream out;for(const auto& t:witness::trace)out<<t<<'\n';
    for(const auto& t:tickets){const auto s=t->stats();out<<"receipt "<<s.prompt_tokens<<' '<<s.generated_tokens<<' '<<s.cached_tokens<<' '<<s.cycles<<' '<<s.finish<<' '<<s.sampling<<' '<<t->error()<<'\n';}
    return out.str();
}
void set_env(const char* k,const char* v) {
#ifdef _WIN32
    witness::check(_putenv_s(k,v)==0,"fixture environment");
#else
    witness::check(setenv(k,v,1)==0,"fixture environment");
#endif
}
void normalize(serve::Json& j) {
    if(j.is_object())for(auto& [key,value]:j.as_object()){
        if(key=="queue_ms" || key=="prefill_ms" || key=="decode_ms" || key=="first_token_ms" || key=="elapsed_ms" || key=="decode_tokens_per_second")value=serve::Json(0);
        else normalize(value);
    }else if(j.is_array())for(auto& v:j.as_array())normalize(v);
}
std::string receipt(const serve::GenerationStats& s){
    auto j=serve::Json::object();
#define FIELD(k) j.set(#k,serve::Json(s.k))
    FIELD(error_status);FIELD(error_code);FIELD(prompt_tokens);FIELD(cached_tokens);FIELD(generated_tokens);
    FIELD(cycles);FIELD(proposed);FIELD(accepted);FIELD(ngram_rows);FIELD(ngram_accepted);FIELD(tool_rows);FIELD(tool_accepted);
    FIELD(source_shadow);FIELD(source_scored);FIELD(source_right);FIELD(frame_cycles);FIELD(frame_shadow);
    FIELD(imported_tokens);FIELD(finish);FIELD(sampling);
#undef FIELD
    auto extra=serve::Json::array();for(auto [k,v]:s.engine_extra){auto e=serve::Json::object();e.set(k,serve::parse_json(v));extra.push(std::move(e));}j.set("engine_extra",std::move(extra));
    j.set("constraint",serve::Json(s.constraint.has_value()));
    j.set("first_token_reported",serve::Json(s.first_token_ms.has_value()));return j.dump();
}
std::string extended_case(const std::string& name){
    witness::scenario=name;witness::trace.clear();witness::positions.clear();witness::capacities.clear();
    witness::steps=0;witness::after_step={};witness::import_digest=0;stop_requested().store(false);
    const bool device=name.starts_with("defer") || name.starts_with("interleave");
    set_env("STRIX_DS41_SERVE_SCHED",name.starts_with("defer")?"defer":name.starts_with("interleave")?"interleave":"off");
    set_env("STRIX_DS41_SERVE_SCHED_LOG","0");set_env("STRIX_DS41_SERVE_SCHED_TDEV","100,0");
    set_env("STRIX_DS41_SERVE_SCHED_STARVE_MS","100000");set_env("STRIX_DS41_SERVE_SCHED_MIN_ROWS","9");
    set_env("STRIX_DS41_SERVE_SHORT_FIRST",name=="defer-short"?"1":"0");
    Args a;a.context=64;a.slots=3;a.device_prefill=device;a.pool=name.starts_with("pool")?16:0;
    witness::pool=a.pool;witness::context=a.context;
    std::vector<std::shared_ptr<serve::GenerationTicket>> tickets;
    std::vector<std::vector<unsigned>> outputs;
    std::thread observer;std::future<void> http;
    Writer writer;std::exception_ptr thread_error;
    actual_lead(a,[&](auto& q,auto& api,auto&,auto& imports,auto&){
        unsigned count=device?7:4;
        for(unsigned n=0;n<count;++n){
            serve::GenerationJob j;j.prompt={1+n,2+n};j.max_tokens=n==0?7:3;
            if(device && n>0)j.prompt.assign(12+n*3,3+n);
            if(name=="invalid" && n==2)j.prompt.clear();
            if(name=="invalid" && n==3)j.prompt.assign(63,1);
            if(name=="pool-refuse" && n==2)j.prompt.assign(20,3);
            if(name=="stop" && n==1)j.stop_tokens={5};
            if(name=="named-conflict" && n<2)j.session_id="conversation";
            tickets.push_back(q.submit(std::move(j)));
        }
        if(name=="import"){
            auto j=std::make_shared<ImportJob>();j->state.position=2;j->state.tokens={1,2};j->state.next_token=3;
            j->bytes=strix::deepseek41::serialize_attention_state(j->state);j->digest=j->state.digest();j->queued=Clock::now();
            imports.jobs.push_back(j);
        }
        if(name=="defer-cancel" || name=="interleave-cancel")witness::after_step=[&]{if(witness::steps==1)tickets.back()->cancel();};
        if(name=="drain")witness::after_step=[&]{if(witness::steps==1)stop_requested().store(true);};
        if(name=="http"){
            http=std::async(std::launch::async,[&api,&writer]{serve::HttpRequest req;req.method="POST";req.path="/v1/completions";
                req.body=R"({"model":"host-ds41","prompt":[1,2],"max_tokens":3,"temperature":0,"seed":19})";api.handle(req,writer);});
            const auto until=Clock::now()+std::chrono::seconds(3);
            while(q.snapshot().queued<count+1 && http.wait_for(std::chrono::milliseconds(0))!=std::future_status::ready && Clock::now()<until)std::this_thread::yield();
            witness::check(q.snapshot().queued==count+1,"HTTP request did not queue");
        }
        outputs.resize(tickets.size());
        observer=std::thread([&]{try{
            for(unsigned i=0;i<tickets.size();++i){bool done=false;const auto until=Clock::now()+std::chrono::seconds(5);
                while(!done && Clock::now()<until){auto ids=tickets[i]->wait(std::chrono::milliseconds(10),done,false);outputs[i].insert(outputs[i].end(),ids.begin(),ids.end());}
                witness::check(done,"request did not finish");
            }
            if(http.valid()){witness::check(http.wait_for(std::chrono::seconds(3))==std::future_status::ready,"HTTP did not finish");http.get();}
        }catch(...){thread_error=std::current_exception();}stop_requested().store(true);});
    });
    observer.join();if(thread_error)std::rethrow_exception(thread_error);
    std::ostringstream out;out<<"CASE "<<name<<'\n';for(auto& t:witness::trace)out<<t<<'\n';
    for(unsigned i=0;i<tickets.size();++i){
        const auto stats=tickets[i]->stats();
        witness::check(stats.generated_tokens==outputs[i].size(),"native receipt must replace logical publications");
        if(stats.finish!="cancelled" && tickets[i]->error().empty()){
            std::vector<unsigned> expected;
            const unsigned first=device && i>0?4+i:3+i;
            for(unsigned k=0;k<(i==0?7U:3U);++k){expected.push_back(first+k);if(name=="stop" && i==1 && first+k==5)break;}
            witness::check(outputs[i]==expected,"independent complete token oracle");
        }
        out<<"tokens";for(auto t:outputs[i])out<<' '<<t;out<<'\n'<<receipt(stats)<<' '<<tickets[i]->error()<<'\n';
    }
    if(name=="http"){witness::check(writer.status==200,"HTTP completion status");auto j=serve::parse_json(writer.data);j.set("id","<completion-id>");j.set("created",0);normalize(j);out<<"HTTP "<<writer.status<<' '<<j.dump()<<'\n';}
    witness::after_step={};witness::queue=nullptr;return out.str();
}
std::string extended(){
    std::string out;
    for(const auto* n:{"ordinary","invalid","stop","http","pool-wait","pool-refuse","defer","defer-short","defer-cancel","interleave","interleave-cancel","import","named-conflict","hetero","spec","hetero-spec","drain"})out+=extended_case(n);
    set_env("STRIX_DS41_SERVE_SCHED","");set_env("STRIX_DS41_SERVE_SCHED_LOG","");set_env("STRIX_DS41_SERVE_SCHED_TDEV","");
    set_env("STRIX_DS41_SERVE_SCHED_STARVE_MS","");set_env("STRIX_DS41_SERVE_SCHED_MIN_ROWS","");set_env("STRIX_DS41_SERVE_SHORT_FIRST","");return out;
}
void failure_case(bool unknown){
    witness::scenario="failure";witness::trace.clear();witness::positions.clear();witness::capacities.clear();
    witness::steps=0;witness::pool=0;stop_requested().store(false);
    set_env("STRIX_DS41_SERVE_SCHED","defer");set_env("STRIX_DS41_SERVE_SCHED_LOG","0");
    set_env("STRIX_DS41_SERVE_SCHED_TDEV","100,0");set_env("STRIX_DS41_SERVE_SCHED_STARVE_MS","100000");
    Args a;a.slots=3;a.context=64;a.device_prefill=true;
    std::vector<std::shared_ptr<serve::GenerationTicket>> tickets;
    bool failed=false;
    try{actual_lead(a,[&](auto& q,auto&,auto&,auto&,auto&){
        for(unsigned n=0;n<7;++n){serve::GenerationJob j;j.prompt.assign(n?12+n:2,1);j.max_tokens=8;tickets.push_back(q.submit(std::move(j)));}
        witness::after_step=[&]{if(unknown)throw 37;throw std::runtime_error("native primary failure");};
    });}catch(const std::runtime_error& e){failed=std::string(e.what())==(unknown?"engine failure":"native primary failure");}
    witness::check(failed,"primary native failure after HTTP teardown");
    for(auto& t:tickets){bool done=false;(void)t->wait(std::chrono::milliseconds(0),done,false);witness::check(done && !t->error().empty(),"failure must settle active and pending requests");}
    witness::check(std::count(witness::trace.begin(),witness::trace.end(),"server stop")==1,"HTTP stop occurs exactly once");
    witness::check(std::any_of(witness::trace.begin(),witness::trace.end(),[](const auto& s){return s.starts_with("queue 0 0 0 7 0 ");}),"no leaked or double-finished native pending work");
    std::cout<<(unknown?"UNKNOWN FAILURE\n":"STANDARD FAILURE\n");
    for(const auto& event:witness::trace)std::cout<<event<<'\n';
    for(const auto& t:tickets)std::cout<<receipt(t->stats())<<' '<<t->error()<<'\n';
    witness::after_step={};set_env("STRIX_DS41_SERVE_SCHED","");set_env("STRIX_DS41_SERVE_SCHED_LOG","");
    set_env("STRIX_DS41_SERVE_SCHED_TDEV","");set_env("STRIX_DS41_SERVE_SCHED_STARVE_MS","");
}
}
int main(int argc,char** argv) try {
    if(argc==2 && std::string(argv[1])=="--fail-sentinel")throw std::runtime_error("deliberate active-check sentinel");
    if(argc==2 && std::string(argv[1])=="--failure"){failure_case(false);failure_case(true);std::cout<<"DS41 actual lead failure cleanup PASS\n";return 0;}
    const bool extra=argc>1 && (std::string(argv[1])=="--extended" || std::string(argv[1])=="--capture-extended" || std::string(argv[1])=="--compare-extended");
    const auto result=extra?extended():run_case();
    if(argc==3 && std::string(argv[1])=="--capture"){std::ofstream(argv[2],std::ios::binary)<<result;return 0;}
    if(argc==3 && std::string(argv[1])=="--compare"){std::ifstream f(argv[2],std::ios::binary);std::string old((std::istreambuf_iterator<char>(f)),{});witness::check(old==result,"actual lead reference mismatch");}
    if(argc==3 && std::string(argv[1])=="--capture-extended"){std::ofstream(argv[2],std::ios::binary)<<result;return 0;}
    if(argc==3 && std::string(argv[1])=="--compare-extended"){std::ifstream f(argv[2],std::ios::binary);std::string old((std::istreambuf_iterator<char>(f)),{});witness::check(old==result,"supplemental actual lead reference mismatch");}
    std::cout<<"DS41 actual lead owner PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<"DS41 owner FAIL: "<<e.what()<<'\n';return 1;}
