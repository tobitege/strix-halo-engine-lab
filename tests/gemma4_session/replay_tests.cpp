// One harness is compiled against the original source before migration and the
// adopted source afterwards. Expected data is never written by the latter.
#include "gemma4/engine/api.hpp"
#ifndef G4_CAPTURE_BASELINE
#include "gemma4/engine/session_engine.hpp"
#include "gemma4/engine/session_service.hpp"
#include "gemma4/engine/runtime.hpp"
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

namespace {
namespace e = strix::gemma4::engine;
namespace g = strix::gemma4;
using Json = strix::serve::Json;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void env(const char* key, const char* value) {
#ifdef _WIN32
    _putenv_s(key, value ? value : "");
#else
    if (value) setenv(key, value, 1); else unsetenv(key);
#endif
}
std::uint64_t mix(std::uint64_t h, std::uint32_t t) {
    h ^= t + UINT64_C(0x9e3779b97f4a7c15) + (h << 6U) + (h >> 2U);
    h *= UINT64_C(0xff51afd7ed558ccd); return h ^ (h >> 33U);
}
struct Stream;
struct Tx final : g::KvTxn {
    Stream& s; std::uint64_t start; std::vector<std::uint32_t> rows;
    Tx(Stream&, std::uint64_t);
    std::uint64_t frontier() const override { return start; }
    void reserve_rows(std::size_t n) override { rows.reserve(n); }
    void commit(std::size_t) override;
};
struct Stream final : e::StreamState {
    std::vector<std::uint64_t> hashes{0x1234567}; bool poisoned = false;
    std::unique_ptr<g::KvTxn> begin() override { return std::make_unique<Tx>(*this, frontier()); }
    std::uint64_t frontier() const override { return hashes.size()-1; }
    std::string digest() const override {
        if (poisoned) throw std::runtime_error("failed state");
        return std::to_string(hashes.back());
    }
    std::unique_ptr<e::StreamState> fork(std::uint32_t) const override { return std::make_unique<Stream>(*this); }
};
Tx::Tx(Stream& state, std::uint64_t f) : s(state), start(f) {}
void Tx::commit(std::size_t n) {
    check(n <= rows.size(), "commit beyond rows");
    for (std::size_t i=0; i<n; ++i) s.hashes.push_back(mix(s.hashes.back(), rows[i]));
}
struct Backend final : e::Backend {
    g::ModelConfig c; Json trace = Json::array(); bool same = true, fail = false;
    std::mutex gate_mutex; std::condition_variable gate_changed;
    bool gate = false, entered = false, released = false;
    void rendezvous() {
        std::unique_lock lock(gate_mutex);
        if (!gate || entered) return;
        entered=true; gate_changed.notify_all();
        gate_changed.wait(lock,[&]{return released;});
    }
    Backend() {
        c.hidden=8; c.vocab=4096; c.max_positions=8192; c.kv_heads=1; c.window=8;
        c.layer={{g::AttentionKind::sliding,8,0},{g::AttentionKind::global,8,1}};
    }
    std::unique_ptr<e::StreamState> create_stream() override { return std::make_unique<Stream>(); }
    std::uint32_t pick(std::uint64_t h) const { return h%29==0 ? 106U : 2097U+std::uint32_t(h%24); }
    std::uint64_t row(Tx& tx, std::uint64_t p, std::uint32_t token) {
        check(p==tx.start+tx.rows.size(), "noncontiguous row"); tx.rows.push_back(token);
        auto h=tx.s.hashes.back(); for (auto t:tx.rows) h=mix(h,t); return h;
    }
    g::ForwardResult forward_rows(std::span<const g::RowPlan> plans) override {
        rendezvous();
        Json event=Json::array();
        for (const auto& p:plans) event.push(Json(Json::Object{{"slot",std::uint64_t(p.stream)},
            {"generation",p.generation},{"position",p.position},{"token",std::uint64_t(p.token)}}));
        trace.push(Json(Json::Object{{"forward",event}}));
        if (fail) { fail=false; for (const auto& p:plans) static_cast<Tx*>(p.transaction)->s.poisoned=true;
            throw std::runtime_error("injected execution failure"); }
        g::ForwardResult out; out.rows=plans.size(); out.vocab=c.vocab; out.tap_width=5*c.hidden;
        out.logits.assign(out.rows*c.vocab, -8); out.taps.assign(out.rows*out.tap_width,0.5F);
        bool hidden=false; for (const auto& p:plans) hidden |= p.export_hidden;
        if (hidden) out.hidden.assign(out.rows*c.hidden,0.25F);
        for (const auto& p:plans) {
            auto h=row(*static_cast<Tx*>(p.transaction),p.position,p.token);
            out.logits[p.output_row*c.vocab+pick(h)]=8;
            out.logits[p.output_row*c.vocab+pick(mix(h,7))]+=7.5F;
        }
        return out;
    }
    std::size_t prefill_chunk_rows() const override { return 16; }
    bool prefill_matches_decode() const override { return same; }
    bool prefill_batch_supported() const override { return true; }
    e::PrefillRows prefill_rows(g::KvTxn& transaction, std::span<const std::uint32_t> ids, bool hidden, bool taps) override {
        rendezvous();
        auto& tx=static_cast<Tx&>(transaction); Json tokens=Json::array(); for(auto id:ids) tokens.push(std::uint64_t(id));
        trace.push(Json(Json::Object{{"prefill",tokens},{"position",tx.start},{"same",same}}));
        for(auto id:ids) row(tx,tx.start+tx.rows.size(),id^(same?0U:0x40000000U));
        e::PrefillRows out; out.tap_width=5*c.hidden;
        if(taps) out.taps.assign(ids.size()*out.tap_width,0.5F);
        if(hidden) out.hidden.assign(ids.size()*c.hidden,0.25F);
        return out;
    }
    std::vector<e::PrefillRows> prefill_rows_batch(std::span<const e::PrefillItem> items, bool hidden, bool taps) override {
        trace.push(Json(Json::Object{{"prefill_batch",std::uint64_t(items.size())}}));
        std::vector<e::PrefillRows> out;
        for(const auto& item:items) out.push_back(prefill_rows(*item.transaction,item.tokens,hidden,taps)); return out;
    }
    const g::ModelConfig& config() const override { return c; }
    Json identity() const override { return Json(Json::Object{{"identity","synthetic-row-v1"},{"prefill_identity",same?"synthetic-row-v1":"synthetic-prefill-v1"}}); }
};
struct Draft final : e::DraftSession {
    Backend& b; Stream* s=nullptr;
    explicit Draft(Backend& backend) : b(backend) {}
    void attach(e::StreamState& state) override { s=dynamic_cast<Stream*>(&state); }
    void append(std::span<const float>,std::span<const std::uint64_t>) override {}
    bool seed(std::span<const std::uint32_t>,std::span<const float>) override { return true; }
    std::array<std::uint32_t,7> propose(std::uint32_t anchor,std::uint64_t) override {
        std::array<std::uint32_t,7> out{}; auto h=mix(s->hashes.back(),anchor);
        for(std::size_t i=0;i<out.size();++i) { out[i]=b.pick(h); h=mix(h,out[i]); }
        out[3]=2097; return out;
    }
};
Json normalize(const Json& j, const std::string& key={}) {
    // Keep shapes and every scheduler/cache/draft/state value. Only clocks and
    // measured durations are masked, including timing arrays in cost receipts.
    if(key=="created" || key.ends_with("_ms") || key.find("_ms_")!=std::string::npos) return Json(0);
    if(key=="id" && j.is_string() && j.as_string().starts_with("chatcmpl-g4-")) return Json("chatcmpl-g4-ID");
    // Host facts (kernel release, driver parameter) exist only on some platforms.
    const auto host_fact=[](const std::string& k){ return k=="kernel" || k=="amdgpu.iommu_perfopt"; };
    if(j.is_object()) { auto out=Json::object(); for(const auto& [k,v]:j.as_object()) { if(!host_fact(k)) out.set(k,normalize(v,k)); } return out; }
    if(j.is_array()) { auto out=Json::array(); for(const auto& v:j.as_array()) out.push(normalize(v)); return out; }
    return j;
}
struct Writer final : strix::serve::HttpResponseWriter {
    int status=0; std::string body; bool gone=false;
    void send(int s,std::string_view,std::string_view text,const std::vector<std::pair<std::string,std::string>>&) override { status=s;body=text; }
    bool begin_stream(int s,std::string_view,const std::vector<std::pair<std::string,std::string>>&) override {status=s;return true;}
    bool write(std::string_view text) override {if(gone)return false;body+=text;return true;}
    void end_stream() override {}
    bool client_gone() const override {return gone;}
};
std::string normalized_wire(const std::string& body) {
    if(!body.starts_with("data: ")) return normalize(strix::serve::parse_json_unique(body)).dump();
    std::string out; std::size_t at=0;
    while(at<body.size()) {auto end=body.find("\n\n",at);check(end!=std::string::npos,"bad SSE");
        auto line=body.substr(at+6,end-at-6); out+="data: "+(line=="[DONE]"?line:normalize(strix::serve::parse_json_unique(line)).dump())+"\n\n";at=end+2;}
    return out;
}
Json read(const std::filesystem::path& p) {std::ifstream f(p);check(bool(f),"missing fixture");return strix::serve::parse_json_unique(std::string(std::istreambuf_iterator<char>(f),{}));}
void write(const std::filesystem::path& p,const Json& j) {std::ofstream f(p,std::ios::binary);f<<j.dump()<<'\n';check(bool(f),"fixture write failed");}
Json api_replay(const std::shared_ptr<const g::serve::Tokenizer>& tokenizer,const Json& requests,Json& trace) {
    Backend b; e::Api api(b,tokenizer); Json out=Json::array();
    for(const auto& request:requests.as_array()) {
        if(const auto* same=request.find("same")) b.same=same->as_bool();
        if(const auto* fail=request.find("fail")) b.fail=fail->as_bool();
        strix::serve::HttpRequest r;r.method="POST";r.path="/v1/chat/completions";r.body=request.find("body")->dump();
        Writer w;api.handle(r,w); out.push(Json(Json::Object{{"case",*request.find("case")},{"status",w.status},{"wire",normalized_wire(w.body)}}));
    }
    trace=b.trace;return out;
}
Json concurrent_api(const std::shared_ptr<const g::serve::Tokenizer>& tokenizer,const Json& requests,Json& trace) {
    env("STRIX_G4_SERVE_SLOTS","4"); Backend b; b.gate=true;
    e::Api api(b,tokenizer); std::array<Writer,4> writers; std::vector<std::thread> threads;
    const std::array<std::size_t,4> order{5,2,3,4};
    for(std::size_t i=0;i<4;++i) {
        threads.emplace_back([&,i]{strix::serve::HttpRequest r;r.method="POST";r.path="/v1/chat/completions";
            r.body=requests.as_array()[order[i]].find("body")->dump();api.handle(r,writers[i]);});
        if(i==0) {std::unique_lock lock(b.gate_mutex);check(b.gate_changed.wait_for(lock,std::chrono::seconds(10),[&]{return b.entered;}),"backend gate timeout");}
        else {
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
            for(;;) {strix::serve::HttpRequest r;r.method="GET";r.path="/api/status";Writer w;api.handle(r,w);
                const auto j=strix::serve::parse_json_unique(w.body);
                if(std::uint64_t(j.find("capacity")->find("queue_depth")->as_int())==i)break;
                check(std::chrono::steady_clock::now()<deadline,"queue admission timeout");std::this_thread::yield();}
        }
    }
    {std::lock_guard lock(b.gate_mutex);b.released=true;}b.gate_changed.notify_all();
    for(auto& thread:threads)thread.join();Json out=Json::array();
    for(const auto& w:writers)out.push(Json(Json::Object{{"status",w.status},{"wire",normalized_wire(w.body)}}));
    trace=b.trace;env("STRIX_G4_SERVE_SLOTS","1");return out;
}
Json coordinator_replay(Json& trace) {
    Backend b; auto costs=std::make_shared<e::CostModel>();
    e::Coordinator c(b,[&]{return std::make_unique<Draft>(b);},costs);c.set_prefill_limit(1);c.set_prefill_budget(4);
    Json out=Json::array(); std::vector<std::size_t> slots;
    for(unsigned i=0;i<4;++i) { e::Request r;r.prompt.assign(i?7+i*13:2,2100+i);r.max_tokens=12+i;r.stop_ids={106};
        r.dflash=i==0?e::DraftMode::on:e::DraftMode::off;r.draft_k=7;
        if(i==2){r.sampling.temperature=0.7F;r.sampling.seed=11;r.sampling.draws=3;}
        slots.push_back(c.submit(r)); }
    std::size_t tick=0;
    for(bool more=true;more;) { if(tick==3)c.cancel(slots[3]);more=c.step();++tick;check(tick<300,"coordinator stuck"); }
    for(auto slot:slots) {auto row=normalize(c.receipt(slot));Json ids=Json::array();for(auto id:c.result(slot).ids) ids.push(std::uint64_t(id));row.set("ids",ids);out.push(row);}
    e::Request reused;reused.prompt={2100,2101};reused.max_tokens=3;reused.stop_ids.clear();reused.dflash=e::DraftMode::off;
    auto slot=c.submit(reused);c.run();out.push(normalize(c.receipt(slot)));trace=b.trace;return out;
}
#ifndef G4_CAPTURE_BASELINE
Json adopted_coordinator_replay(Json& trace) {
    using namespace strix::serve;
    struct Context final : GenerationContext {e::Request request;};
    Backend b;std::mutex mutex;
    e::SessionEngine engine(b,[&]{return std::make_unique<Draft>(b);},std::make_shared<e::CostModel>(),mutex,4,1,4);
    GenerationQueue queue;SessionHooks hooks;Json results=Json::array();
    hooks.validate=[](const GenerationJob& j){check(bool(std::dynamic_pointer_cast<Context>(j.family_context)),"wrong context");};
    hooks.opened=[&](const SessionView& v){engine.prepare(v.state,std::dynamic_pointer_cast<Context>(v.job.family_context)->request);};
    hooks.ready=[&](const SessionView& v){engine.start(v.state);};
    hooks.initial_cursor=[&](const SessionView& v){return engine.initial_cursor(v.state);};
    hooks.schedule=[](std::span<const SessionView> views){SessionWork w{SessionPhase::mixed,{},{}};
        for(std::size_t i=0;i<views.size();++i)w.lanes.push_back(i);return std::vector<SessionWork>{w};};
    std::map<std::uint32_t,Json> ended;std::map<std::uint32_t,std::vector<std::uint32_t>> histories;
    hooks.identity={"synthetic-rows","synthetic-row-v1","gemma4-prompt-v1",1};hooks.prompt_restart_cache=true;
    hooks.prompt_checkpoint=[&](const SessionView& v,SessionCheckpoint boundary){const auto r=engine.result(v.state);
        return boundary==SessionCheckpoint::terminal && (r.status==e::Status::length || r.status==e::Status::stopped) && engine.cache_candidate(v.state);};
    hooks.receipt=[&](const SessionView& v,GenerationStats&){auto row=normalize(engine.lane_receipt(v.state));
        histories[v.state.slot]=engine.store().tokens(v.state);
        Json ids=Json::array();for(auto id:engine.result(v.state).ids)ids.push(std::uint64_t(id));row.set("ids",ids);ended[v.state.slot]=row;};
    strix::session::Policy p;p.max_entries=16;p.leave_one_token=true;p.replace_duplicates=true;
    p.retain_one_oversize=true;p.charge_token_bytes=false;
    SessionRuntime runtime(queue,engine,engine.store(),p,hooks);
    const auto submit=[&](e::Request r,CacheRetention retention=CacheRetention::server_policy){auto context=std::make_shared<Context>();context->request=r;GenerationJob j;
        j.prompt=r.prompt;j.max_tokens=std::uint32_t(r.max_tokens);j.stop_tokens=r.stop_ids;j.family_context=context;
        j.cache_retention=retention;j.sampling={r.sampling.temperature,r.sampling.top_p,r.sampling.top_k,0,r.sampling.seed,true};return queue.submit(std::move(j));};
    std::vector<std::shared_ptr<GenerationTicket>> tickets;
    for(unsigned i=0;i<4;++i){e::Request r;r.prompt.assign(i?7+i*13:2,2100+i);r.max_tokens=12+i;r.stop_ids={106};
        r.dflash=i==0?e::DraftMode::on:e::DraftMode::off;r.draft_k=7;
        if(i==2){r.sampling.temperature=0.7F;r.sampling.seed=11;r.sampling.draws=3;}tickets.push_back(submit(r));}
    for(std::size_t tick=0;tick<300;++tick){if(tick==3)tickets[3]->cancel();runtime.step();if(!runtime.live())break;}
    check(ended.size()==4,"missing adopted lane receipt");for(unsigned i=0;i<4;++i)results.push(ended.at(i));
    for(const auto& t:tickets)check(t->error().empty(),"adopted grouped execution failed");
    e::Request r;r.prompt={2100,2101};r.max_tokens=3;r.stop_ids.clear();r.dflash=e::DraftMode::off;
    auto reused=submit(r);for(std::size_t tick=0;tick<100;++tick){runtime.step();if(!runtime.live())break;}
    check(reused->error().empty(),"reused adopted slot failed");auto row=ended.at(0);
    // The frozen reused receipt did not include an ids member.
    auto reused_receipt=Json::object();for(const auto& [k,v]:row.as_object())if(k!="ids")reused_receipt.set(k,v);
    results.push(reused_receipt);trace=strix::serve::parse_json_unique(b.trace.dump());
    const auto entries=runtime.cache_entries();const auto prefix=histories.at(0);
    r.prompt=prefix;r.prompt.push_back(2110);r.sampling.temperature=0.7F;r.sampling.seed=13;r.sampling.draws=5;
    auto warm=submit(r,CacheRetention::no_store);
    for(std::size_t tick=0;tick<100;++tick){runtime.step();if(!runtime.live())break;}
    check(warm->error().empty() && warm->stats().cached_tokens==prefix.size() &&
        ended.at(0).find("rng_draws")->as_int()==8 && runtime.cache_entries()==entries,
        "no-store permits physical warm reuse and restarts sampler without retaining output");
    r.prompt={2010,2011};auto cold=submit(r,CacheRetention::no_store);
    for(std::size_t tick=0;tick<100;++tick){runtime.step();if(!runtime.live())break;}
    check(cold->error().empty() && cold->stats().cached_tokens==0 && runtime.cache_entries()==entries,
        "cold no-store request does not add a checkpoint");
    return results;
}
void adapter_state_tests() {
    namespace ss=strix::session;
    Backend b;std::mutex mutex;e::SessionEngine engine(b,{}, {},mutex,2,0,0);
    check(!engine.caps().fork_lanes && !engine.caps().shared_prefix_cow,"deep copies must not advertise COW");
    auto h=engine.store().acquire();e::Request r;r.prompt={2100,2101};r.max_tokens=4;r.stop_ids.clear();
    r.dflash=e::DraftMode::off;r.sampling.temperature=0.7F;r.sampling.seed=17;r.sampling.draws=5;
    engine.prepare(h,r);engine.start(h);
    std::size_t offset=0;std::uint64_t draws=5;
    const auto advance=[&]{ss::SamplerState sampler;sampler.parameters={0.7F,1,0,0,17,true};sampler.rng_draws=draws;
        sampler.max_tokens=std::uint32_t(r.max_tokens-engine.result(h).ids.size());
        ss::AdvanceItem item{h,std::span(r.prompt).subspan(offset),sampler,{}};
        auto out=engine.advance({&item,1}).at(0);offset+=out.input_tokens;draws=out.output.rng_draws;return out;};
    while(engine.result(h).ids.empty())advance();
    const auto full=engine.store().capture(h);const auto saved=engine.result(h);const auto saved_offset=offset;
    const auto first=advance();const auto expected=engine.result(h);
    const auto digest=engine.lane_receipt(h).find("state_digest")->as_string();
    engine.store().restore(h,full);offset=saved_offset;draws=saved.rng_draws;
    const auto repeated=advance();
    check(first.output.tokens==repeated.output.tokens && engine.result(h).rng_draws==expected.rng_draws &&
        engine.lane_receipt(h).find("state_digest")->as_string()==digest,"full checkpoint restores pending, RNG and physical state");
    check(full.pending.has_value() && full.sealed(),"full lane snapshot retains pending anchor");
    check(engine.cache_candidate(h),"cache candidate available");
    const auto prompt=engine.store().capture_prompt(h);
    check(prompt.sealed() && !prompt.pending && prompt.purpose==ss::SnapshotPurpose::prompt_reuse,"separate prompt checkpoint");
    engine.store().release(h);const auto stale=h;h=engine.store().acquire();
    r.prompt=prompt.tokens;r.prompt.push_back(2110);r.sampling.draws=1;
    engine.prepare(h,r);
    check(engine.store().restore_prompt(h,prompt),"physical prompt reuse");engine.start(h);offset=prompt.tokens.size();draws=1;
    const auto warm=advance();check(warm.output.rng_draws==2 && engine.result(h).cached_tokens==prompt.consumed,
        "prompt restore starts the new sampler, not the former cursor");
    bool rejected=false;try{(void)engine.result(stale);}catch(const std::exception&){rejected=true;}
    check(rejected,"old slot generation is rejected");engine.store().release(h);

    // Two long prefills would run as two collectives. The second cancellation
    // becomes visible DURING the synchronous advance after the first completes.
    Backend boundary;std::mutex boundary_mutex;e::SessionEngine bounded(boundary,{}, {},boundary_mutex,2,0,0);
    std::array<ss::KvHandle,2> handles{bounded.store().acquire(),bounded.store().acquire()};
    e::Request long_request;long_request.prompt.assign(50,2100);long_request.max_tokens=4;long_request.dflash=e::DraftMode::off;
    long_request.stop_ids.clear();for(auto handle:handles){bounded.prepare(handle,long_request);bounded.start(handle);}
    ss::SamplerState sampler;sampler.max_tokens=4;
    std::array<ss::AdvanceItem,2> items{{{handles[0],long_request.prompt,sampler,{}},
        {handles[1],long_request.prompt,sampler,[&]{return boundary.trace.size()!=0;}}}};
    const auto progress=bounded.advance(items);
    check(progress[0].input_tokens==16 && progress[1].input_tokens==0 &&
        progress[1].output.finish==ss::FinishReason::cancelled && boundary.trace.size()==1,
        "cancellation observed between collectives, without mid-kernel cancellation");
    for(auto handle:handles)bounded.store().release(handle);
}
void service_factory_tests(const std::shared_ptr<const g::serve::Tokenizer>& tokenizer,const std::string& tokenizer_path) {
    using namespace strix::serve;
    ServiceFactories factories;std::weak_ptr<Backend> loaded;unsigned loads=0,inspection_loads=0;
    e::install_session_service(factories,[&](const e::Options& options){auto b=std::make_shared<Backend>();
        ++loads;inspection_loads+=options.inspect?1:0;loaded=b;return e::SessionAssets{b,b.get(),tokenizer,{}};});
    const auto invoke=[&](std::vector<std::string> args,const ServeShell& shell=ServeShell{}) {
        std::vector<char*> argv;for(auto& a:args)argv.push_back(a.data());std::ostringstream out,error;
        const auto code=run(int(argv.size()),argv.data(),factories,out,error,shell);
        if(code!=0)throw std::runtime_error(error.str());return out.str();};
    const std::vector<std::string> base{"serve","--registry-root",G4_SESSION_ROOT,"--model","gemma4"};
    auto args=base;args.push_back("--inspect");check(invoke(args).find("built")!=std::string::npos && loads==0,"metadata inspection does no loading");
    args=base;args.insert(args.end(),{"--","--help"});check(invoke(args).find("--draft-mtp")!=std::string::npos && loads==0,"family help does no loading");
    const std::vector<std::string> family{"--","synthetic.gguf","--rope-tables","tables","--tokenizer",tokenizer_path,"--draft","off"};
    args=base;args.push_back("--family-inspect");args.insert(args.end(),family.begin(),family.end());
    check(!invoke(args).empty() && loads==1 && inspection_loads==1 && loaded.expired(),"family inspection uses deferred loader and releases assets");
    ServeShell shell;shell.serve=[&](const HttpServerConfig& config,HttpHandler handler,SessionRuntime& runtime){
        check(config.port==0 && runtime.capabilities().mixed_advance,"actual service has shared runtime");
        Writer writer;std::atomic<bool> done=false;
        std::thread client([&]{HttpRequest r;r.method="POST";r.path="/v1/chat/completions";
            r.body=R"({"messages":[{"role":"user","content":"Hello."}],"max_tokens":4})";handler(r,writer);done=true;});
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        while(!done && std::chrono::steady_clock::now()<deadline){runtime.step();std::this_thread::yield();}
        if(!done)runtime.shutdown();client.join();check(done && writer.status==200,"factory service executes handler through external owner");
    };
    args=base;args.insert(args.end(),{"--port","0"});args.insert(args.end(),family.begin(),family.end());
    (void)invoke(args,shell);check(loaded.expired() && loads==2,"service owns and releases model assets");
    // Forwarded family listener arguments retain their original interpretation.
    args=base;args.insert(args.end(),{"--port","12"});args.insert(args.end(),family.begin(),family.end());
    args.insert(args.end(),{"--port","0"});(void)invoke(args,shell);
    const auto before_check=loads;
    const auto model=std::filesystem::path(tokenizer_path).parent_path()/"deployment.gguf";
    {std::ofstream file(model,std::ios::binary);
        const auto u32=[&](std::uint32_t n){for(unsigned i=0;i<4;++i)file.put(char((n>>(8*i))&255));};
        const auto u64=[&](std::uint64_t n){for(unsigned i=0;i<8;++i)file.put(char((n>>(8*i))&255));};
        const auto text=[&](const std::string& value){u64(value.size());file<<value;};
        file<<"GGUF";u32(3);u64(1);u64(8);text("general.architecture");u32(8);text("gemma4");
        for(const auto& [key,n]:std::vector<std::pair<const char*,std::uint32_t>>{{"general.file_type",15},
            {"gemma4.embedding_length",2560},{"gemma4.block_count",42},{"gemma4.feed_forward_length",10240},
            {"gemma4.attention.head_count_kv",2},{"gemma4.attention.sliding_window",512},{"gemma4.context_length",65536}}){text(key);u32(4);u32(n);}
        text("token_embd.weight");u32(2);u64(256);u64(1);u32(12);u64(0);
        while(std::uint64_t(file.tellp())%32)file.put('\0');for(unsigned i=0;i<144;++i)file.put('\0');}
    env("STRIX_DEPLOY_FAKE_MEM_GIB","192");
    args=base;args.insert(args.end(),{"--",model.string(),"--draft","off","--deploy-check"});
    const auto plan=strix::serve::parse_json_unique(invoke(args));env("STRIX_DEPLOY_FAKE_MEM_GIB",nullptr);
    check(loads==before_check && plan.find("schema")->as_string()=="strix-deployment-v1",
        "deploy-check sizes headers and exits before loading service assets");
}
#endif
}
int gemma4_session_replay(int argc,char** argv) {
    check(argc>=2,"tokenizer path required");
    env("STRIX_G4_SERVE_SLOTS","1");env("STRIX_G4_DRAFT_MULTI","0");env("STRIX_G4_MTP_CONF","0");
    env("STRIX_G4_SERVE_PREFIX_MB","0");
    const auto root=std::filesystem::path(G4_SESSION_FIXTURES);
    const auto tokenizer=g::serve::Tokenizer::load(argv[1]);
    Json api_trace,coordinator_trace,concurrent_trace;auto requests=read(root/"requests.json");
    auto actual=api_replay(tokenizer,requests,api_trace);
    auto concurrent=concurrent_api(tokenizer,requests,concurrent_trace);
    auto coordinator=coordinator_replay(coordinator_trace);
    Json traces(Json::Object{{"api",api_trace},{"concurrent",concurrent_trace},{"concurrent_results",concurrent},
        {"coordinator",coordinator_trace},{"coordinator_results",coordinator}});
    if(argc==3 && std::string(argv[2])=="--capture") {
#ifdef G4_CAPTURE_BASELINE
        write(root/"legacy_expected.json",actual);write(root/"legacy_trace.json",traces);
        std::cout<<"Captured original Api/Coordinator oracle\n";return 0;
#else
        throw std::runtime_error("capture is disabled after migration");
#endif
    }
    if(actual.dump()!=read(root/"legacy_expected.json").dump()) {
        write(std::filesystem::path(argv[1]).parent_path()/"actual_http.json",actual);
        write(std::filesystem::path(argv[1]).parent_path()/"actual_trace.json",traces);
        throw std::runtime_error("HTTP/SSE/receipt oracle mismatch (actual_http.json in build directory)");
    }
    check(traces.dump()==read(root/"legacy_trace.json").dump(),"grouped row/state/receipt oracle mismatch");
#ifndef G4_CAPTURE_BASELINE
    Json adopted_trace;const auto adopted=adopted_coordinator_replay(adopted_trace);
    check(adopted.dump()==coordinator.dump(),"adopted mixed runtime receipts differ from frozen old Coordinator");
    check(adopted_trace.dump()==coordinator_trace.dump(),"adopted mixed runtime row trace differs from frozen old Coordinator");
    adapter_state_tests();service_factory_tests(tokenizer,argv[1]);
#endif
    std::cout<<"PASS frozen Gemma4 Api/Coordinator replay\n";return 0;
}
