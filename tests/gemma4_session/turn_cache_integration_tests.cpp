// Reuse the loaded tiny model and independent serial causal reference. The
// production Api owns every HTTP job, queue, slot, sampler and cache operation.
#define STRIX_G4_CONTINUATION_TEST_SUPPORT
#include "continuation_tests.cpp"
#include "gemma4/engine/api.hpp"
#ifdef STRIX_G4_NATIVE_TURN_PROBE
#include "gemma4/engine/runtime.hpp"
#endif
#include "strix/model_content_hash.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {
using Json=strix::serve::Json;
namespace srv=strix::serve;
void environment(const char* key,const char* value) {
#ifdef _WIN32
    _putenv_s(key,value);
#else
    setenv(key,value,1);
#endif
}
std::shared_ptr<const g::serve::Tokenizer> tokenizer(const std::filesystem::path& directory,bool thought=false) {
    std::filesystem::create_directories(directory);
    const auto asset=directory/(thought?"thinking.json":"tokenizer.json");
    const auto script=directory/(thought?"thinking.py":"tokenizer.py");
    // Extend the existing tokenizer-fixture generator; only the ID layout is
    // specialized for the existing tiny model, never its math.
    const auto helper=std::filesystem::path(__FILE__).parent_path().parent_path()/"gemma4_serve/make_deployment_fixture.py";
    std::ofstream f(script);
    f<<"import sys,runpy,json,pathlib\n";
    f<<"sys.argv=['fixture',"<<Json(asset.string()).dump()<<"]\n";
    f<<"data=runpy.run_path("<<Json(helper.string()).dump()<<")\n";
    f<<R"(a=data['asset']; v=a['model']['vocab']
# Keep the production adapter's fixed BOS/EOS/turn boundaries. Fill the
# enlarged tiny vocabulary with round-trippable ordinary text, then map roles.
def token(t,i):
 old=next(k for k,n in v.items() if n==i)
 if t in v: v[old],v[t]=v[t],i
 else: del v[old]; v[t]=i
for i in range(128): token('Z'+chr(97+i//26)+chr(97+i%26),i)
for i,t in enumerate('abcdefghijklmnopqrstuvwxyz\n',10): token(t,i)
specials={1:'<eos>',2:'<bos>',106:'<turn|>',107:'<|channel>',108:'<|tool_call>',109:'<tool_call|>',110:'<|tool_response>',111:'<|"|>',112:'<|turn>',113:'<channel|>',114:'<|think|>'}
for i,t in specials.items(): token(t,i)
token('Z',9)
merges=[]
for j,c in enumerate('abcde'):
 token('Z'+c,5000+j); merges.append(['Z',c])
for t,n in list(v.items()):
 if n<128 and t.startswith('Z') and len(t)==3: merges.append([t[:2],t[2]])
a['model']['merges']=merges
special=[t for t,n in v.items() if n<128 and (n<10 or n>36)]
a['added_tokens']=[{'id':v[t],'content':t,'single_word':False,'lstrip':False,'rstrip':False,'normalized':False,'special':True} for t in special]
)";
    if(thought) f<<R"(old=next(k for k,n in v.items() if n==127); n=v.pop(old); v['<|channel>thought\n']=n
a['model']['merges']=[pair for pair in a['model']['merges'] if ''.join(pair)!=old]
for x in a['added_tokens']:
 if x['content']==old: x['content']='<|channel>thought\n'
)";
    f<<"pathlib.Path(sys.argv[1]).write_text(json.dumps(a,ensure_ascii=False,separators=(',',':')),encoding='utf-8')\n";
    f.close();CHECK(f.good());
    // The interpreter is named python or python3 depending on the platform.
    CHECK(std::system(("python \""+script.string()+"\"").c_str())==0 || std::system(("python3 \""+script.string()+"\"").c_str())==0);
    return g::serve::Tokenizer::load(asset);
}
g::ModelManifest http_model() {
    auto m=tiny();m.config.vocab=128;
    // Reuse the existing fixture coefficients, extending only vocabulary rows
    // so the unchanged production adapter's fixed ID 106 is representable.
    const auto shapes=g::required_tensors(m.config);
    for(auto& record:m.tensors) {
        const auto& shape=shapes.at(record.name);if(shape==record.shape)continue;
        CHECK(shape.size()==2 && record.shape.size()==2 && shape[1]==record.shape[1]);
        auto values=std::make_shared<std::vector<float>>(std::size_t(shape[0]*shape[1]));
        for(std::uint32_t r=0;r<shape[0];++r)for(std::uint32_t c=0;c<shape[1];++c)
            (*values)[r*shape[1]+c]=g::weight_at(record.view,r%record.view.outputs,c);
        record.shape=shape;record.view={reinterpret_cast<const std::uint8_t*>(values->data()),std::uint32_t(shape[0]),std::uint32_t(shape[1]),std::size_t(shape[1]*4),g::WeightFormat::f32};
        m.storage.push_back(values);
    }return m;
}
void identify(g::ModelManifest& m) {
    strix::Sha256Fast h;
    for(const auto& t:m.tensors) {
        h.update(std::as_bytes(std::span(t.name.data(),t.name.size())));
        h.update(std::as_bytes(std::span(t.shape)));
        h.update({reinterpret_cast<const std::byte*>(t.view.data),std::size_t(t.view.outputs*t.view.row_bytes)});
    }
    m.model_hash=strix::model_content_identity_hex(h.finalize());
}
struct Log {
    struct Row {std::uint64_t position;std::uint32_t token;std::vector<float> logits;};
    std::vector<Row> rows;
    std::vector<std::string> digests;
    std::size_t forks=0,fail_fork=0,queries=0,fail_query=0;
    std::size_t calls=0,fail_call=0;
    bool rollback_failure=false,failed=false,unknown_bytes=false;
    std::vector<g::memory::Payload> native_payloads;
};
class TraceStream;
class TraceTxn final:public g::KvTxn {
public:
    TraceTxn(std::unique_ptr<g::KvTxn> tx,TraceStream& s):physical(std::move(tx)),stream(s){}
    std::uint64_t frontier()const override{return physical->frontier();}
    void reserve_rows(std::size_t n)override{physical->reserve_rows(n);}
    void commit(std::size_t n)override;
    std::unique_ptr<g::KvTxn> physical;
    TraceStream& stream;
};
class TraceStream final:public e::StreamState {
public:
    TraceStream(std::unique_ptr<e::StreamState> p,Log& l):physical(std::move(p)),log(l){}
    std::unique_ptr<g::KvTxn> begin()override{return std::make_unique<TraceTxn>(physical->begin(),*this);}
    std::uint64_t frontier()const override{return physical->frontier();}
    std::string digest()const override{return physical->digest();}
    std::optional<std::uint64_t> retained_bytes()const override {
        if(++log.queries==log.fail_query)throw std::runtime_error("injected retained query failure");
        if(log.unknown_bytes)return {};
        const auto n=physical->retained_bytes();if(!n)return {};
        if(const auto payload=physical->native_payload()) {
            CHECK(payload->total()==*n);log.native_payloads.push_back(*payload);
        }
        CHECK(*n<=UINT64_MAX-sizeof(TraceStream));return *n+sizeof(TraceStream);
    }
    std::unique_ptr<e::StreamState> fork(std::uint32_t n)const override {
        if(++log.forks==log.fail_fork || (log.failed && log.rollback_failure))throw std::runtime_error("injected eager copy failure");
        return std::make_unique<TraceStream>(physical->fork(n),log);
    }
    std::unique_ptr<e::StreamState> physical;Log& log;
};
void TraceTxn::commit(std::size_t n){physical->commit(n);stream.log.digests.push_back(stream.digest());}
class RealBackend final:public e::Backend {
public:
    RealBackend(const g::ModelManifest& m,g::RopeTables t,e::Backend* device=nullptr):host(m,t,1),replacement(device){}
    e::Backend& target(){if(replacement)return *replacement;return host;}
    const e::Backend& target()const{if(replacement)return *replacement;return host;}
    bool input_continuation_supported()const override{return supported && target().input_continuation_supported();}
    std::unique_ptr<e::StreamState> create_stream()override{return std::make_unique<TraceStream>(target().create_stream(),log);}
    std::unique_ptr<e::StreamState> create_stream(std::uint32_t n)override{return std::make_unique<TraceStream>(target().create_stream(n),log);}
    const g::ModelConfig& config()const override{return target().config();}
    Json identity()const override{auto j=target().identity();if(unknown_identity)j.set("model_hash","");
        if(memory_wrapper)j.set("backend","host-memory-wrapper-evidence");return j;}
    std::optional<std::uint64_t> native_turn_work_bound()const override {
        return memory_wrapper?reported_work:target().native_turn_work_bound();
    }
    g::ForwardResult forward_rows(std::span<const g::RowPlan> rows)override {
        ++log.calls;if(on_call)on_call();auto physical=std::vector<g::RowPlan>(rows.begin(),rows.end());
        for(auto& r:physical) {
            r.transaction=dynamic_cast<TraceTxn&>(*r.transaction).physical.get();
            // Operator tracing explicitly asks the real native target to expose
            // logits even for greedy rows; it does not change their arithmetic.
            if(replacement)r.export_logits=true;
        }
        auto out=target().forward_rows(physical);
        CHECK(out.logits.size()==out.rows*out.vocab);
        for(const auto& r:rows)log.rows.push_back({r.position,r.token,{out.logits.begin()+r.output_row*out.vocab,out.logits.begin()+(r.output_row+1)*out.vocab}});
        if(log.calls==log.fail_call){log.failed=true;throw std::runtime_error("injected final target failure");}
        return out;
    }
    e::HostBackend host;Log log;bool supported=true,unknown_identity=false;e::Backend* replacement=nullptr;
    std::function<void()> on_call;
    // Exercises the family native-bound seam with real HostTarget execution.
    // This wrapper is host evidence, never native execution/measurement.
    bool memory_wrapper=false;
    std::optional<std::uint64_t> reported_work=65536;
};
struct Writer final:srv::HttpResponseWriter {
    int status=0;std::string bytes;std::shared_ptr<std::atomic<bool>> gone=std::make_shared<std::atomic<bool>>(false);
    void send(int s,std::string_view,std::string_view b,const std::vector<std::pair<std::string,std::string>>&)override{status=s;bytes=b;}
    bool begin_stream(int s,std::string_view,const std::vector<std::pair<std::string,std::string>>&)override{status=s;return !gone->load();}
    bool write(std::string_view s)override{bytes+=s;return !gone->load();}
    void end_stream()override{}
    bool client_gone()const override{return gone->load();}
};
struct Settings {
    std::string conversation="conversation",templ="tokenizer-assets/render-v1/ordinary",topology="host-rank-0",keep;
    bool no_store=false,unknown=false,throws=false;std::size_t requests=0;
};
e::TurnCacheConfig configuration(Settings& settings) {
    e::TurnCacheConfig c;c.policy.enabled=true;c.policy.completed_turns=true;
    c.policy.max_entries=2;c.policy.max_bytes=2U<<20;
    c.max_state_bytes=256U<<10;c.max_transient_bytes=8U<<20;
    c.request=[&](const srv::HttpRequest&,const e::ChatRequest&)->std::optional<e::TurnCacheRequest>{
        ++settings.requests;
        if(settings.throws)throw std::runtime_error("injected metadata failure");
        if(settings.unknown)return {};
        return e::TurnCacheRequest{{settings.conversation,settings.templ,settings.topology,settings.keep},
            settings.no_store?srv::CacheRetention::no_store:srv::CacheRetention::retain};
    };return c;
}
Json body(bool stream=false,bool thinking=false) {
    return Json(Json::Object{{"messages",Json(Json::Array{Json(Json::Object{{"role","user"},{"content","hi"}})})},
        {"enable_thinking",thinking},{"max_tokens",3},{"temperature",1.0},{"seed",19},{"dflash","off"},{"stream",stream}});
}
Writer http(e::Api& api,const Json& b) {
    srv::HttpRequest r;r.method="POST";r.path="/v1/chat/completions";r.body=b.dump();Writer w;api.handle(r,w);return w;
}
Json response(const Writer& w) {
    if(w.status!=200)std::cerr<<"HTTP "<<w.status<<" "<<w.bytes<<"\n";
    CHECK(w.status==200);
    if(w.bytes.starts_with("data: ")) {
        for(std::size_t i=0;i<w.bytes.size();) {
            auto e=w.bytes.find("\n\n",i);CHECK(e!=std::string::npos);
            auto line=w.bytes.substr(i+6,e-i-6);i=e+2;
            if(line!="[DONE]"){auto j=srv::parse_json_unique(line);if(j.find("strix"))return j;}
        }
        throw std::runtime_error("missing SSE final receipt");
    }
    return srv::parse_json_unique(w.bytes);
}
std::vector<std::uint32_t> generated(const Json& j) {
    std::vector<std::uint32_t> ids;for(const auto& n:j.find("strix")->find("generated_ids")->as_array())ids.push_back(std::uint32_t(n.as_int()));return ids;
}
const Json& phase(const Json& j){return j.find("strix")->find("phases")->as_array().back();}
std::vector<std::uint32_t> prompt(const Json& b,const g::serve::Adapter& a){return a.prepare(e::parse_chat_request(b).input).ids;}
Json next_turn(const Json& first,const std::vector<std::uint32_t>& ids,const g::serve::Tokenizer& t,bool stream=false) {
    auto b=body(stream);auto messages=Json(first.find("messages")->as_array());
    messages.push(Json(Json::Object{{"role","assistant"},{"content",t.decode(ids)}}));
    messages.push(Json(Json::Object{{"role","user"},{"content","why"}}));b.set("messages",messages);
    b.set("max_tokens",4);b.set("seed",733);return b;
}
// Normalize only existing nondeterminism and documented physical cache-work
// statistics. Every other receipt/HTTP/SSE byte is compared as serialized.
Json stable(Json j) {
    if(j.is_object()) {
        Json::Object out;
        for(const auto& [k,v]:j.as_object()) {
            if(k=="id" || k=="created" || k=="timings" || k=="serve_prefix" || k=="cached_tokens" ||
                k=="target_calls" || k=="target_rows" || k=="steps" || k=="admitted_at_step")continue;
            out.emplace_back(k,stable(v));
        }return Json(std::move(out));
    }
    if(j.is_array()){auto out=Json::array();for(const auto& v:j.as_array())out.push(stable(v));return out;}
    return j;
}
std::string stable_wire(const Writer& w) {
    if(!w.bytes.starts_with("data: "))return stable(srv::parse_json_unique(w.bytes)).dump();
    std::string out;
    for(std::size_t i=0;i<w.bytes.size();) {
        const auto end=w.bytes.find("\n\n",i);CHECK(end!=std::string::npos);
        const auto data=w.bytes.substr(i+6,end-i-6);i=end+2;
        out+="data: "+(data=="[DONE]"?data:stable(srv::parse_json_unique(data)).dump())+"\n\n";
    }return out;
}
void causal(const g::ModelManifest& m,g::RopeTables t,const g::serve::Adapter& a,const Json& b,const Json& j,
    const Log& log,std::size_t row_begin,std::size_t cached,bool sealed) {
    const auto r=e::parse_chat_request(b);auto p=prompt(b,a);Reference ref(m,t,r.generation.sampling);ref.append(p);
    for(const auto id:generated(j))CHECK(ref.decode()==id);
    CHECK(phase(j).find("rng_draws")->as_int()==std::int64_t(ref.draws));
    CHECK(phase(j).find("state_digest")->as_string()==ref.state.digest());
    if(sealed && ref.pending){ref.consume(*ref.pending);ref.pending.reset();}
    CHECK(!log.digests.empty() && log.digests.back()==ref.state.digest());
    auto all=p;const auto ids=generated(j);all.insert(all.end(),ids.begin(),ids.end());if(!sealed)all.pop_back();
    CHECK(log.rows.size()-row_begin==all.size()-cached);
    Reference rows(m,t,{});
    for(std::size_t i=0;i<all.size();++i) {
        rows.consume(all[i]);
        if(i>=cached) {
            const auto& row=log.rows.at(row_begin+i-cached);CHECK(row.position==i && row.token==all[i]);
            CHECK(bits(row.logits,rows.logits));
        }
    }
    CHECK(std::uint64_t(phase(j).find("target_rows")->as_int())==p.size()+ids.size()-1-cached);
    std::cout<<"causal prompt="<<p.size()<<" cached="<<cached<<" output="<<ids.size()<<" physical="<<all.size()-cached<<" sealed="<<sealed<<'\n';
}
void two_turns(const g::ModelManifest& m,g::RopeTables t,std::shared_ptr<const g::serve::Tokenizer> tok) {
    g::serve::Adapter adapter(tok);
    for(const auto mode:{0,1,2,3}) {
        const bool stream=mode==1;
        environment("STRIX_G4_SERVE_BATCH",mode==2?"0":"1");
        environment("STRIX_G4_SERVE_PREFIX",mode?"0":"1");
        Settings settings;RealBackend backend(m,t),cold(m,t);
        backend.memory_wrapper=cold.memory_wrapper=mode==3;
        e::Api api(backend,tok,{},e::DraftMode::off,"model",{}, {},e::Api::Owner::standalone,configuration(settings));
        e::Api independent(cold,tok,{},e::DraftMode::off,"model");
        auto first=body(stream);const auto w1=http(api,first);const auto j1=response(w1);
        CHECK(api.runtime().cache_stats().turn_admitted==1);causal(m,t,adapter,first,j1,backend.log,0,0,true);
        const auto cache_tokens=prompt(first,adapter).size()+generated(j1).size();
        const auto count=backend.log.rows.size();auto second=next_turn(first,generated(j1),*tok,stream);
        const auto p=prompt(second,adapter);auto actual=prompt(first,adapter);auto ids=generated(j1);actual.insert(actual.end(),ids.begin(),ids.end());
        if(!(p.size()>actual.size() && std::equal(actual.begin(),actual.end(),p.begin()))) {
            std::cerr<<"actual="<<tok->decode(actual)<<"\nrendered="<<tok->decode(p)<<"\n";
            for(auto x:actual)std::cerr<<x<<',';std::cerr<<"\n";for(auto x:p)std::cerr<<x<<',';std::cerr<<"\n";
        }
        CHECK(p.size()>actual.size() && std::equal(actual.begin(),actual.end(),p.begin()));
        RealBackend enabled_backend(m,t);Settings cold_settings;
        enabled_backend.memory_wrapper=mode==3;
        e::Api enabled_cold(enabled_backend,tok,{},e::DraftMode::off,"model",{}, {},e::Api::Owner::standalone,configuration(cold_settings));
        const auto warm=http(api,second),off=http(independent,second),enabled=http(enabled_cold,second);
        const auto j2=response(warm),gold=response(off),enabled_gold=response(enabled);
        CHECK(stable_wire(warm)==stable_wire(enabled));CHECK(enabled_cold.runtime().cache_stats().hits==0);
        causal(m,t,adapter,second,enabled_gold,enabled_backend.log,0,0,true);
        CHECK(settings.requests==2);
        CHECK(generated(j2)==generated(gold));CHECK(stable_wire(warm)==stable_wire(off));
        CHECK(api.runtime().cache_stats().hits==1 && api.runtime().cache_stats().turn_admitted==2);
        causal(m,t,adapter,second,j2,backend.log,count,cache_tokens,true);causal(m,t,adapter,second,gold,cold.log,0,0,false);
        CHECK(generated(j2).size()==4); // fresh output allowance, not previous job's remaining budget
        // A third actual HTTP request tests per-conversation replacement.
        CHECK(api.runtime().cache_stats().entries==1);
        std::cout<<"retained_charge="<<api.runtime().cache_stats().bytes<<" retained_budget="<<(2U<<20)
            <<" payload_cap="<<(256U<<10)<<" transient_budget="<<(8U<<20)<<" slots=1 mode="<<mode<<'\n';
    }
    environment("STRIX_G4_SERVE_BATCH","1");environment("STRIX_G4_SERVE_PREFIX","1");
}
void refusals(const g::ModelManifest& m,g::RopeTables t,std::shared_ptr<const g::serve::Tokenizer> tok) {
    g::serve::Adapter adapter(tok);
    for(int mode=0;mode<20;++mode) {
        Settings settings;RealBackend backend(m,t);auto config=configuration(settings);
        if(mode==0)settings.no_store=true;
        if(mode==1)settings.unknown=true;
        if(mode==2)settings.templ.clear();
        if(mode==3)settings.throws=true;
        if(mode==4)config.policy.max_bytes=1;
        if(mode==5)config.max_state_bytes=1;
        if(mode==6)config.max_transient_bytes=1;
        if(mode==7)config.max_state_bytes=UINT64_MAX;
        if(mode==8)backend.supported=false;
        if(mode==9)config.policy.max_entries=0;
        if(mode==10)backend.log.unknown_bytes=true;
        if(mode==11)config.policy.max_entries=UINT64_MAX;
        if(mode==12)backend.unknown_identity=true;
        if(mode>=14) {backend.memory_wrapper=true;environment("STRIX_G4_SERVE_PREFIX","0");}
        if(mode==14)backend.reported_work.reset();
        if(mode==15)backend.reported_work=0;
        if(mode==16)backend.reported_work=UINT64_MAX;
        if(mode==17)config.max_transient_bytes=65536+5*config.max_state_bytes-1;
        if(mode==18)config.max_state_bytes=0;
        if(mode==19)backend.log.unknown_bytes=true;
        e::DraftFactory drafts;if(mode==13)drafts=[] {return std::unique_ptr<e::DraftSession>{};};
        e::Api api(backend,tok,drafts,e::DraftMode::off,"model",{}, {},e::Api::Owner::standalone,config);
        auto b=body();const auto j=response(http(api,b));CHECK(api.runtime().cache_entries()==0);
        causal(m,t,adapter,b,j,backend.log,0,0,false);
        if(mode>=14)CHECK(backend.log.forks==0); // no rollback/candidate/install copy before admission
    }
    environment("STRIX_G4_SERVE_PREFIX","1");
}
void ordinary_seed(Json& b,const g::ModelManifest& m,g::RopeTables t,const g::serve::Adapter& adapter) {
    const auto ids=prompt(b,adapter);auto r=e::parse_chat_request(b);
    for(std::uint64_t seed=0;seed<1000;++seed) {
        r.generation.sampling.seed=seed;Reference ref(m,t,r.generation.sampling);ref.append(ids);bool ordinary=true;
        for(std::size_t i=0;i<r.generation.max_tokens;++i) {
            const auto bytes=adapter.tokenizer().token_bytes(ref.decode());
            if(bytes.find('<')!=std::string::npos || bytes=="\n") {ordinary=false;break;}
        }
        if(ordinary){b.set("seed",seed);return;}
    }throw std::runtime_error("no ordinary tiny-model seed");
}
void mismatches(const g::ModelManifest& m,g::RopeTables t,std::shared_ptr<const g::serve::Tokenizer> tok) {
    for(int mode=0;mode<5;++mode) {
        std::cout<<"mismatch "<<mode<<"\n";
        Settings settings;RealBackend backend(m,t);
        auto other=m;other.tensors=m.tensors;auto changed=std::make_shared<std::vector<float>>(std::size_t(other.tensors[0].view.outputs)*other.tensors[0].view.inputs);
        for(std::uint32_t r=0;r<other.tensors[0].view.outputs;++r)for(std::uint32_t c=0;c<other.tensors[0].view.inputs;++c)
            (*changed)[r*other.tensors[0].view.inputs+c]=g::weight_at(other.tensors[0].view,r,c);
        changed->at(0)+=0.0001F;other.tensors[0].view.data=reinterpret_cast<const std::uint8_t*>(changed->data());other.storage.push_back(changed);identify(other);
        e::HostBackend changed_model(other,t,1);
        e::Api api(backend,tok,{},e::DraftMode::off,"model",{}, {},e::Api::Owner::standalone,configuration(settings));
        auto first=body();auto j=response(http(api,first));CHECK(api.runtime().cache_entries()==1);
        auto second=next_turn(first,generated(j),*tok);const auto n=backend.log.rows.size();
        if(mode==0){auto messages=Json(second.find("messages")->as_array());
            messages.as_array()[0]=Json(messages.as_array()[0].as_object());messages.as_array()[0].set("content","why");second.set("messages",messages);}
        if(mode==1)settings.templ="tokenizer-assets/render-v2/ordinary";
        if(mode==2)settings.topology="host-other-order";
        if(mode==3)settings.conversation="other";
        if(mode==4)backend.replacement=&changed_model;
        ordinary_seed(second,mode==4?other:m,t,g::serve::Adapter(tok));
        CHECK(response(http(api,second)).find("choices"));CHECK(api.runtime().cache_stats().hits==0);
        CHECK(backend.log.rows.at(n).position==0);
    }
}
void retention(const g::ModelManifest& m,g::RopeTables t,std::shared_ptr<const g::serve::Tokenizer> tok) {
    for(bool keep:{false,true}) {
        Settings settings;settings.keep=keep?"protected":"";RealBackend backend(m,t);auto c=configuration(settings);c.policy.max_entries=1;
        e::Api api(backend,tok,{},e::DraftMode::off,"model",{}, {},e::Api::Owner::standalone,c);
        auto b=body();auto j=response(http(api,b));CHECK(api.runtime().cache_entries()==1);
        settings.conversation="second";settings.keep.clear();response(http(api,b));CHECK(api.runtime().cache_entries()==1);
        settings.conversation="conversation";auto next=next_turn(b,generated(j),*tok);response(http(api,next));
        CHECK(api.runtime().cache_stats().hits==std::uint64_t(keep));
        CHECK(api.runtime().cache_stats().bytes<=c.policy.max_bytes);
    }
}
void failures(const g::ModelManifest& m,g::RopeTables t,std::shared_ptr<const g::serve::Tokenizer> tok) {
    // Identify exact finalization fork/query positions with a successful run;
    // this observes real eager copies, not synthetic cache insertion.
    std::size_t forks=0,queries=0,calls=0;
    {Settings settings;RealBackend backend(m,t);e::Api api(backend,tok,{},e::DraftMode::off,"model",{}, {},e::Api::Owner::standalone,configuration(settings));
     response(http(api,body()));forks=backend.log.forks;queries=backend.log.queries;calls=backend.log.calls;CHECK(forks>=2 && queries>=3);}
    for(int mode=0;mode<5;++mode) {
        std::cout<<"failure "<<mode<<"\n";
        Settings settings;RealBackend backend(m,t);
        if(mode==0 || mode==3){backend.log.fail_call=calls;backend.log.rollback_failure=mode==3;}
        if(mode==1)backend.log.fail_fork=forks; // post-commit candidate copy
        if(mode==2)backend.log.fail_query=queries; // prompt capture accounting
        e::Api api(backend,tok,{},e::DraftMode::off,"model",{}, {},e::Api::Owner::standalone,configuration(settings));
        const auto delivered=http(api,body());const auto first=response(delivered);
        RealBackend gold_backend(m,t);e::Api gold_api(gold_backend,tok,{},e::DraftMode::off,"model");
        const auto gold=http(gold_api,body());CHECK(stable_wire(delivered)==stable_wire(gold));
        CHECK(phase(first).find("target_rows")->as_int()==phase(response(gold)).find("target_rows")->as_int());
        if(mode!=4){CHECK(api.runtime().cache_stats().turn_failed==1);CHECK(api.runtime().cache_entries()==0);}
        if(mode==4) {
            const auto next=next_turn(body(),generated(first),*tok);
            backend.log.fail_fork=backend.log.forks+1;auto failed=http(api,next);CHECK(failed.status==500);
            CHECK(api.runtime().cache_stats().restore_failed==1 && api.runtime().cache_stats().hits==0);
        }
        backend.log.fail_call=backend.log.fail_fork=backend.log.fail_query=0;backend.log.failed=false;backend.log.rollback_failure=false;
        CHECK(response(http(api,body())).find("choices")); // fresh generation, no stale pending/cache leakage
    }
}
void stops_and_capacity(const g::ModelManifest& m,g::RopeTables t,std::shared_ptr<const g::serve::Tokenizer> tok) {
    const g::serve::Adapter adapter(tok);
    for(const auto stop:{1U,106U}) {
        auto b=body();Reference ref(m,t,{});ref.append(prompt(b,adapter));e::Sampling sampling;sampling.temperature=1;
        std::uint64_t seed=0;for(;seed<10000;++seed){sampling.seed=seed;std::uint64_t draws=0;if(e::select_token(ref.logits,sampling,draws)==stop)break;}
        CHECK(seed<10000);b.set("seed",seed);Settings settings;RealBackend backend(m,t);
        e::Api api(backend,tok,{},e::DraftMode::off,"model",{}, {},e::Api::Owner::standalone,configuration(settings));
        auto j=response(http(api,b));CHECK(generated(j)==std::vector<std::uint32_t>{stop});
        CHECK(j.find("choices")->as_array()[0].find("finish_reason")->as_string()=="stop");
        CHECK(api.runtime().cache_stats().turn_admitted==1);causal(m,t,adapter,b,j,backend.log,0,0,true);
    }
    Settings settings;RealBackend backend(m,t);e::Api api(backend,tok,{},e::DraftMode::off,"model",{}, {},e::Api::Owner::standalone,configuration(settings));
    auto b=body();b.set("temperature",0.0);b.set("max_tokens",std::uint64_t(m.config.max_positions-prompt(b,adapter).size()));
    const auto j=response(http(api,b));CHECK(generated(j).size()==std::size_t(b.find("max_tokens")->as_int()));
    CHECK(backend.log.rows.back().position==m.config.max_positions-1);CHECK(api.runtime().cache_stats().turn_admitted==1);
    causal(m,t,adapter,b,j,backend.log,0,0,true);
    const auto calls=backend.log.calls;b.set("max_tokens",std::uint64_t(b.find("max_tokens")->as_int()+1));
    CHECK(http(api,b).status==400 && backend.log.calls==calls); // HTTP context admission remains unchanged
}
void no_store_hit(const g::ModelManifest& m,g::RopeTables t,std::shared_ptr<const g::serve::Tokenizer> tok) {
    Settings settings;RealBackend backend(m,t);e::Api api(backend,tok,{},e::DraftMode::off,"model",{}, {},e::Api::Owner::standalone,configuration(settings));
    auto first=body();auto j=response(http(api,first));const auto before=api.runtime().cache_stats();const auto count=backend.log.rows.size();
    settings.no_store=true;auto b=next_turn(first,generated(j),*tok);auto next=response(http(api,b));
    CHECK(api.runtime().cache_stats().hits==1 && api.runtime().cache_stats().turn_admitted==before.turn_admitted);
    CHECK(api.runtime().cache_stats().bytes==before.bytes);g::serve::Adapter adapter(tok);
    causal(m,t,adapter,b,next,backend.log,count,prompt(first,adapter).size()+generated(j).size(),false);
}
void disconnects(const g::ModelManifest& m,g::RopeTables t,std::shared_ptr<const g::serve::Tokenizer> tok) {
    for(bool during:{false,true}) {
        Settings settings;RealBackend backend(m,t);e::Api api(backend,tok,{},e::DraftMode::off,"model",{}, {},e::Api::Owner::standalone,configuration(settings));
        Writer writer;if(during)backend.on_call=[&]{writer.gone->store(true);};else writer.gone->store(true);
        srv::HttpRequest req;req.method="POST";req.path="/v1/chat/completions";req.body=body().dump();api.handle(req,writer);
        CHECK(api.runtime().cache_entries()==0 && api.runtime().cache_stats().turn_admitted==0);
        backend.on_call={};CHECK(response(http(api,body())).find("choices"));
    }
}
void thinking(const g::ModelManifest& m,g::RopeTables t,std::shared_ptr<const g::serve::Tokenizer> tok) {
    g::serve::Adapter adapter(tok);auto b=body(false,true);b.set("max_tokens",5);b.set("thinking_budget",2);
    // Find an actual sampler seed whose first selected tiny-model ID is the
    // tokenizer's whole thought header. The logits/selection are never replaced.
    auto p=prompt(b,adapter);Reference ref(m,t,{});ref.append(p);
    e::Sampling policy;policy.temperature=1;std::uint64_t seed=0;
    const auto header=tok->find("<|channel>thought\n");
    for(;seed<10000;++seed) {
        policy.seed=seed;std::uint64_t draws=0;if(e::select_token(ref.logits,policy,draws)!=header)continue;
        Reference within(m,t,policy);within.append(p);CHECK(within.decode()==header);
        const auto second=within.decode();const auto bytes=tok->token_bytes(second);
        if(bytes.find('<')!=std::string::npos || bytes=="\n")continue;
        g::serve::OutputParser parser;parser.push(tok->token_bytes(header));parser.push(bytes);parser.finish();
        const auto continuation=g::serve::continue_after_budget(e::parse_chat_request(b).input,parser.output(),2,2,5);
        auto final_policy=policy;final_policy.draws=2;Reference final(m,t,final_policy);final.append(tok->encode(continuation.prompt));
        bool ordinary=true;for(int i=0;i<3;++i)if(tok->token_bytes(final.decode()).find('<')!=std::string::npos)ordinary=false;
        if(ordinary)break;
    }
    CHECK(seed<10000);b.set("seed",seed);
    Settings settings;RealBackend backend(m,t);e::Api api(backend,tok,{},e::DraftMode::off,"model",{}, {},e::Api::Owner::standalone,configuration(settings));
    const auto j=response(http(api,b));CHECK(j.find("strix")->find("budget_continued")->as_bool());
    CHECK(j.find("strix")->find("phases")->size()==2);CHECK(settings.requests==1);
    if(api.runtime().cache_stats().turn_admitted!=1)std::cerr<<"thinking reply "<<j.dump()<<"\n";
    CHECK(api.runtime().cache_stats().turn_admitted==1);
    const auto all_ids=generated(j);CHECK(all_ids.size()==5);
    const auto first_rows=p.size()+1; // phase-0 last published thought ID remains pending
    CHECK(backend.log.rows.size()>first_rows && backend.log.rows[first_rows].position==0);
    auto sampling=policy;sampling.seed=seed;sampling.draws=2;
    Reference final_ref(m,t,sampling);
    // Obtain the actually evaluated final causal input from the observed real
    // rows. No cache state is created/committed by the test between requests.
    std::vector<std::uint32_t> sealed;
    for(std::size_t i=first_rows;i<backend.log.rows.size();++i){CHECK(backend.log.rows[i].position==sealed.size());sealed.push_back(backend.log.rows[i].token);}
    CHECK(sealed.size()>=3);const auto final_prompt=sealed.size()-3;
    final_ref.append(std::span(sealed).first(final_prompt));
    for(std::size_t i=2;i<all_ids.size();++i)CHECK(final_ref.decode()==all_ids[i]);
    CHECK(phase(j).find("state_digest")->as_string()==final_ref.state.digest());
    CHECK(phase(j).find("rng_draws")->as_int()==std::int64_t(final_ref.draws));
    Reference row_ref(m,t,{});
    for(std::size_t i=0;i<sealed.size();++i){row_ref.consume(sealed[i]);CHECK(bits(row_ref.logits,backend.log.rows[first_rows+i].logits));}
    CHECK(backend.log.digests.back()==row_ref.state.digest());
    std::cout<<"thinking phase0_rows="<<first_rows<<" final_prompt="<<final_prompt<<" sealed="<<sealed.size()<<"\n";
    // Standard visible history drops old reasoning under the actual renderer:
    // it must miss. Raw assistant content is normalized by the renderer too.
    auto visible=body(false,true);auto messages=Json(b.find("messages")->as_array());
    messages.push(j.find("choices")->as_array()[0].find("message")->as_object());
    messages.push(Json(Json::Object{{"role","user"},{"content","why"}}));visible.set("messages",messages);
    ordinary_seed(visible,m,t,adapter);settings.no_store=true;
    response(http(api,visible));CHECK(api.runtime().cache_stats().hits==0);
    auto exact=body(false,true);messages=Json(b.find("messages")->as_array());
    CHECK(std::equal(p.begin(),p.end(),sealed.begin()));
    messages.push(Json(Json::Object{{"role","assistant"},{"content",tok->decode(std::span(sealed).subspan(p.size()))}}));
    messages.push(Json(Json::Object{{"role","user"},{"content","why"}}));exact.set("messages",messages);
    ordinary_seed(exact,m,t,adapter);const auto next_prompt=prompt(exact,adapter);
    CHECK(next_prompt.size()>sealed.size());
    CHECK(!std::equal(sealed.begin(),sealed.end(),next_prompt.begin()));
    const auto row_begin=backend.log.rows.size();const auto cold_again=response(http(api,exact));
    CHECK(api.runtime().cache_stats().hits==0 && backend.log.rows[row_begin].position==0);
    causal(m,t,adapter,exact,cold_again,backend.log,row_begin,0,false);
    // A completed FINAL continuation may retain its actually evaluated closed
    // thinking transcript. An ordinary incomplete thought never seals.
    auto incomplete=b;incomplete=Json(b.as_object());incomplete.as_object().erase(std::remove_if(incomplete.as_object().begin(),incomplete.as_object().end(),[](const auto& x){return x.first=="thinking_budget";}),incomplete.as_object().end());
    settings.no_store=false;
    for(const auto maximum:{1,2}) {
        incomplete.set("max_tokens",maximum);const auto before=api.runtime().cache_stats();const auto rows=backend.log.rows.size();
        const auto partial=response(http(api,incomplete));CHECK(api.runtime().cache_stats().turn_admitted==before.turn_admitted);
        CHECK(api.runtime().cache_stats().turn_refused==before.turn_refused+1);
        CHECK(backend.log.rows.size()-rows==prompt(incomplete,adapter).size()+generated(partial).size()-1);
    }
}
}
#if !defined(STRIX_G4_NATIVE_TURN_PROBE) && !defined(STRIX_G4_SHARED_API_TEST_SUPPORT)
int main(int argc,char** argv) {
    try {
        CHECK(argc==2);environment("STRIX_G4_SERVE_SLOTS","1");environment("STRIX_G4_SERVE_BATCH","1");environment("STRIX_G4_SERVE_PREFIX","1");
        const auto directory=std::filesystem::absolute(argv[1]);auto tok=tokenizer(directory);auto model=http_model();identify(model);Tables tables;auto rope=tables.view();
        strix::Sha256Fast rope_hash;rope_hash.update(std::as_bytes(rope.sliding));rope_hash.update(std::as_bytes(rope.global));
        rope.hash=strix::model_content_identity_hex(rope_hash.finalize());
        two_turns(model,rope,tok);refusals(model,rope,tok);mismatches(model,rope,tok);
        retention(model,rope,tok);failures(model,rope,tok);stops_and_capacity(model,rope,tok);no_store_hit(model,rope,tok);disconnects(model,rope,tok);thinking(model,rope,tokenizer(directory,true));
        std::cout<<"real Api completed-turn integration PASS\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
#elif defined(STRIX_G4_NATIVE_TURN_PROBE)
// Linked into the loaded native operator executable. All HTTP/queue/runtime
// helpers above are identical to the always-active host integration; only the
// explicitly supplied physical backend changes. No generated tiny fixture is
// used by this entry point.
int gemma4_native_api_turns(e::Backend& native,const g::ModelManifest& model,g::RopeTables rope,
    const std::filesystem::path& identity_path,const Json& identity,const std::string& ticket) {
    const auto required=[&](const char* key)->const Json& {const auto* p=identity.find(key);CHECK(p);return *p;};
    auto tokenizer_path=std::filesystem::path(required("tokenizer").as_string());
    if(tokenizer_path.is_relative())tokenizer_path=identity_path.parent_path()/tokenizer_path;
    CHECK(std::filesystem::exists(tokenizer_path));auto tok=g::serve::Tokenizer::load(tokenizer_path);
    auto first=required("api_request");first.set("stream",ticket=="api-sse");
    const auto request=e::parse_chat_request(first);
    CHECK(!request.input.enable_thinking && request.generation.max_tokens>0 && request.generation.max_tokens<=4);
    const auto finite=[&](const char* key) {const auto n=required(key).as_int();CHECK(n>0);return std::uint64_t(n);};
    const auto row_budget=finite("api_row_budget");CHECK(row_budget<=128);
    Settings settings;auto config=configuration(settings);
    config.max_state_bytes=finite("max_state_bytes");config.max_transient_bytes=finite("max_transient_bytes");
    config.policy.max_bytes=finite("retained_budget");
    environment("STRIX_G4_SERVE_SLOTS","1");environment("STRIX_G4_SERVE_BATCH","1");environment("STRIX_G4_SERVE_PREFIX","0");
    CHECK(native.native_turn_work_bound());
    const auto buffers=native.native_buffer_payload();CHECK(buffers);
    std::cout<<"existing native service buffers device="<<buffers->device<<" pinned="<<buffers->pinned
        <<" (excludes host service objects and driver internals)\n";
    RealBackend backend(model,rope,&native);g::serve::Adapter adapter(tok);
    CHECK(prompt(first,adapter).size()+request.generation.max_tokens<=row_budget);
    if(ticket=="api-pressure")config.max_transient_bytes=1;
    if(ticket=="api-retained-pressure")config.policy.max_bytes=1;
    if(ticket=="api-no-retention")config.policy.enabled=false;
    e::Api api(backend,tok,{},e::DraftMode::off,"model",{}, {},e::Api::Owner::standalone,config);
    const auto j1=response(http(api,first));
    const bool retains=ticket!="api-pressure" && ticket!="api-retained-pressure" && ticket!="api-no-retention";
    CHECK(api.runtime().cache_stats().turn_admitted==std::uint64_t(retains));
    causal(model,rope,adapter,first,j1,backend.log,0,0,retains);
    const auto cache_tokens=prompt(first,adapter).size()+generated(j1).size();
    auto second=next_turn(first,generated(j1),*tok,ticket=="api-sse");
    const auto second_prompt=prompt(second,adapter);auto sealed=prompt(first,adapter);
    const auto answer=generated(j1);sealed.insert(sealed.end(),answer.begin(),answer.end());
    CHECK(second_prompt.size()>sealed.size() && std::equal(sealed.begin(),sealed.end(),second_prompt.begin()));
    CHECK(second_prompt.size()+4<=row_budget && second_prompt.size()+4<=model.config.max_positions);
    if(ticket=="api-mismatch")settings.templ+="-changed";
    if(ticket=="api-no-store")settings.no_store=true;
    std::uint64_t cancelled_hits=0;
    if(ticket=="api-cancel") {
        auto cancelled=second;cancelled.set("stream",true);Writer disconnected;
        backend.on_call=[gone=disconnected.gone]{gone->store(true);};
        srv::HttpRequest wire;wire.method="POST";wire.path="/v1/chat/completions";wire.body=cancelled.dump();
        const auto charge=api.runtime().cache_stats().bytes;
        api.handle(wire,disconnected);backend.on_call={};
        CHECK(api.runtime().cache_stats().turn_admitted==1 && api.runtime().cache_stats().bytes==charge);
        cancelled_hits=api.runtime().cache_stats().hits;
        settings.no_store=true; // retry also proves release and fresh cancellation
    }
    const auto begin=backend.log.rows.size(),forks=backend.log.forks;
    const auto old_charge=api.runtime().cache_stats().bytes;
    const bool hit=retains && ticket!="api-mismatch";
    // First copy restores the warm entry; the next copy is terminal rollback.
    // Fail optional finalization before growth, preserving the completed reply.
    if(ticket=="api-fault")backend.log.fail_fork=backend.log.forks+2;
    const auto warm=http(api,second);const auto j2=response(warm);
    auto cold_native=e::make_device_backend(model,rope,false,e::NativeInputContinuation::exact_g4_v1);
    RealBackend cold(model,rope,cold_native.get());Settings cold_settings;auto cold_config=config;
    cold_config.request=configuration(cold_settings).request;
    if(ticket=="api-no-store" || ticket=="api-cancel")cold_settings.no_store=true;
    e::Api independent(cold,tok,{},e::DraftMode::off,"model",{}, {},e::Api::Owner::standalone,cold_config);
    const auto reference=http(independent,second);CHECK(stable_wire(warm)==stable_wire(reference));
    CHECK(api.runtime().cache_stats().hits==std::uint64_t(hit)+cancelled_hits);
    CHECK(generated(j2)==generated(response(reference)));
    const bool stores=retains && ticket!="api-no-store" && ticket!="api-fault" && ticket!="api-cancel";
    if(ticket!="api-fault")causal(model,rope,adapter,second,j2,backend.log,begin,hit?cache_tokens:0,stores);
    else CHECK(api.runtime().cache_stats().turn_failed>0 && api.runtime().cache_stats().bytes==old_charge);
    if(ticket=="api-no-store")CHECK(api.runtime().cache_stats().bytes==old_charge);
    if(!retains)CHECK(backend.log.forks==forks && forks==0);
    std::cout<<"native API ticket="<<ticket<<" cache_retained_charge="<<api.runtime().cache_stats().bytes
        <<" native_additional_bound="<<*native.native_turn_work_bound()<<" payload_ceiling="<<config.max_state_bytes
        <<" transient_budget="<<config.max_transient_bytes<<" retained_budget="<<config.policy.max_bytes
        <<" hits="<<api.runtime().cache_stats().hits<<" copies="<<backend.log.forks<<'\n';
    for(const auto& p:backend.log.native_payloads) std::cout<<"native retained payload host="<<p.host
        <<" device="<<p.device<<" pinned="<<p.pinned<<" total="<<p.total()<<'\n';
    return 0;
}
#endif
