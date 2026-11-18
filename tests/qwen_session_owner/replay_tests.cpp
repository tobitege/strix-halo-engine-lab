// The native ABI below is a bounded host double, not a device scheduler model.
// QWEN_ENTRY compiles the verbatim archived owner block. The current selection
// compiles the production binding and links the actual shared owner.
#include "strix/serve/api.hpp"
#include "strix/serve/health.hpp"
#include "strix/token_sampling.hpp"
#include "../../apps/http_session_completion.hpp"
#ifndef QWEN_ENTRY
#include "qwen/http_session_owner.hpp"
#endif
#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <future>
#include <thread>

#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string("check: ") + #x); } while (false)
namespace strix::decode_cli {
namespace serve = strix::serve;
static std::vector<std::string> events;
static std::mutex* named_mutex = nullptr;
static bool available(std::mutex& mutex) {
    return std::async(std::launch::async,[&] {
        const bool acquired=mutex.try_lock();if(acquired)mutex.unlock();return acquired;
    }).get();
}
struct Receipt { std::string json() const { return "{\"witness\":7}"; } };
struct Options { std::uint64_t context_tokens=64; Receipt product_profile; };
struct MtpConstraintRow {};
struct HttpConstraintRequests {
    std::shared_ptr<MtpConstraintRow> make(const std::shared_ptr<const serve::ConstraintSpec>& s) {
        events.push_back(s ? "constraint:on" : "constraint:off"); return s ? std::make_shared<MtpConstraintRow>() : nullptr;
    }
};
struct MtpServeResult : serve::GenerationStats {
    std::uint64_t restored_tokens=0, final_state_position=3;
    double prompt_ms=1;
    std::string error, cache_refusal="native-only", final_state_digest="final", final_target_state_digest, prompt_state_digest="prompt", final_state_components="{\"a\":1}";
    Receipt shared_execution;
    unsigned service=3;
};
static std::string batch_service_receipt(unsigned s) { return "\"points\":"+std::to_string(s); }
static void add_prefill_stats(serve::GenerationStats& s) { s.engine_extra.emplace_back("prefill", "{\"identity\":17}"); }
static void append_mtp_pool_receipt(serve::GenerationStats& s,const MtpServeResult&) { s.engine_extra.emplace_back("pool","11"); }
struct MtpBatchRequest {
    std::shared_ptr<MtpConstraintRow> constraint;
    serve::CacheRetention cache_retention=serve::CacheRetention::server_policy;
    std::vector<std::uint32_t> prompt, stop_tokens;
    std::uint64_t max_tokens=0;
    std::shared_ptr<TokenSampler> sampler;
    std::optional<bool> ngram_draft,tool_draft,ngram_peers;
    std::uint32_t draft_depth_max=0;
    std::function<bool(std::span<const std::uint32_t>)> on_tokens;
    std::function<void(serve::GenerationProgress)> on_progress;
    std::function<void(const MtpServeResult&)> on_finish;
    std::function<bool()> cancelled;
};
static std::atomic<bool> stop{false};
static std::atomic<bool>& http_serve_stop_requested() { return stop; }
struct Probe {
    unsigned pending_generation() {events.push_back("probe:pending");return 9;}
    void complete(unsigned n,serve::BackendProof p) {CHECK(n==9 && p==serve::BackendProof::work_completion);events.push_back("probe:complete");}
    void fail() {events.push_back("probe:fail");}
};
struct MtpBatchRunner {
    using AdmissionSink=std::function<void(MtpBatchRequest)>;
    using AdmissionPump=std::function<void(const AdmissionSink&,const std::function<bool()>&)>;
    struct Slot { bool active=false; MtpBatchRequest request; MtpServeResult result; };
    std::vector<Slot> physical=std::vector<Slot>(1);
    std::size_t slots_capacity=3;
    std::mutex mutex;
    std::mutex* session_mutex=&mutex;
    // No prefix store in the replay: the queue protection binding is empty (owner protection inert).
    strix::serve::QueuedPrefixProtection::Binding queue_protection_binding() { return {}; }
    bool session_run=true, memory_refusal=false, check_mutex=false, completed=true, empty_error=false, unknown_error=false;
    bool stop_when_empty=true;
    int fault=0;
    serve::GenerationQueue& queue;
    std::function<void()> arrivals;
    std::function<std::size_t()> queued;
    AdmissionPump pump;
    explicit MtpBatchRunner(serve::GenerationQueue& q):queue(q) {}
    std::size_t slots() const {return physical.size();}
    std::size_t active() const {return std::count_if(physical.begin(),physical.end(),[](const auto& s){return s.active;});}
    bool has_free_slot() const {return active()<slots();}
    void add_slot(bool q) {
        events.push_back(std::string("grow:")+(q?"queued":"spare"));
        if(fault==1)throw std::runtime_error("growth failure");
        if(slots()<slots_capacity && active()==slots()) {
            if(memory_refusal)events.push_back("memory:refused");
            else {physical.emplace_back();events.push_back("built:"+std::to_string(slots()-1));}
        }
    }
    void set_service_admission(std::function<std::size_t()> q,AdmissionPump p) {queued=std::move(q);pump=std::move(p);}
    void admit(MtpBatchRequest r) {
        if(check_mutex)CHECK(!available(mutex));
        if(fault==2) {if(unknown_error)throw 9;throw std::runtime_error(empty_error ? "" : "before admit failure");}
        const auto it=std::find_if(physical.begin(),physical.end(),[](const auto& s){return !s.active;});
        CHECK(it!=physical.end());auto& s=*it;
        events.push_back("admit:"+std::to_string(r.prompt.front())+":"+std::to_string(it-physical.begin()));
        if(r.sampler)events.push_back("sampler:"+std::to_string(r.sampler->policy().seed)+":"+std::to_string(r.sampler->options().request_identity));
        s.request=std::move(r);s.result={};s.active=true;
        if(fault==3)throw std::runtime_error("during admit failure");
    }
    template<bool Proof=false> bool step() {
        if(check_mutex)CHECK(!available(mutex));
        events.push_back(Proof?"step:proof":"step");
        if(arrivals) {auto fn=std::move(arrivals);fn();}
        if(queued && queued()) {events.push_back("nested:enter");pump([&](MtpBatchRequest r){events.push_back("nested:prepare");admit(std::move(r));},[&]{events.push_back("can_prepare");return has_free_slot();});}
        if(fault==4)throw std::runtime_error("step failure");
        // Reverse finish order makes logical generations diverge from native
        // first-free physical selection on the next admission.
        for(std::size_t i=physical.size();i-->0;) {
            auto& s=physical[i];if(!s.active)continue;
            auto& r=s.request;auto& result=s.result;
            if(r.on_progress)r.on_progress({serve::GenerationPhase::decode,r.prompt.size(),0,r.prompt.size()});
            if(r.cancelled && r.cancelled())result.finish="cancelled";
            else {
                auto id=std::uint32_t(r.prompt.front()+result.generated_tokens);
                ++result.generated_tokens;result.cycles+=2;result.proposed+=3;result.accepted+=1;
                result.ngram_rows=5;result.ngram_accepted=4;result.tool_rows=7;result.tool_accepted=6;
                result.source_shadow=8;result.source_scored=9;result.source_right=10;result.frame_cycles=11;result.frame_shadow=12;
                events.push_back("output:"+std::to_string(id));
                if(!r.on_tokens(std::span(&id,1)))result.finish="cancelled";
                else if(std::find(r.stop_tokens.begin(),r.stop_tokens.end(),id)!=r.stop_tokens.end())result.finish="stop";
                else if(result.generated_tokens<r.max_tokens)continue;
            }
            s.active=false;events.push_back("finish:"+std::to_string(i));
            if(fault==5)throw std::runtime_error("finish failure");
            r.on_finish(result);r.on_finish={};
        }
        if(stop_when_empty && !active() && !queue.has_queued_work())queue.shutdown();
        return completed;
    }
    void fail_requests(const std::string& e) {
        events.push_back("fail_requests:"+e);
        for(auto& s:physical)if(s.request.on_finish) {s.result.finish="execution_failure";s.result.error=e;s.request.on_finish(s.result);s.request.on_finish={};}
    }
    void abandon() {events.push_back("abandon");for(auto& s:physical){s.active=false;s.request={};}}
};
struct MtpBridgeEngine {
    bool bad=false;
    template<class... T> explicit MtpBridgeEngine(T&&...) {}
    serve::GenerationStats generate(serve::GenerationJob& j) {
        if(named_mutex)CHECK(available(*named_mutex));
        events.push_back("named:"+j.session_id);
        if(j.constraint)throw serve::GenerationRequestError({},"named constraint refused");
        if(j.on_admitted)j.on_admitted();
        const std::uint32_t id=j.prompt.front();j.on_tokens(std::span(&id,1));
        serve::GenerationStats s;s.generated_tokens=1;s.prompt_tokens=j.prompt.size();s.engine_extra.emplace_back("session","{\"fixed\":true}");return s;
    }
    bool failed()const{return bad;}
};
static serve::GenerationJob job(unsigned id,unsigned count=1) {
    serve::GenerationJob j;j.prompt={id};j.max_tokens=count;
    j.on_admitted=[id]{events.push_back("admitted:"+std::to_string(id));};
    return j;
}
static serve::Json stats_json(const serve::GenerationStats& s) {
    auto j=serve::Json::object();
    // Only real queue wall time is outside this deterministic comparison.
    j.set("prompt",s.prompt_tokens);j.set("cached",s.cached_tokens);j.set("generated",s.generated_tokens);
    j.set("cycles",s.cycles);j.set("proposed",s.proposed);j.set("accepted",s.accepted);
    j.set("ngram_rows",s.ngram_rows);j.set("ngram_accepted",s.ngram_accepted);j.set("tool_rows",s.tool_rows);j.set("tool_accepted",s.tool_accepted);
    j.set("source_shadow",s.source_shadow);j.set("source_scored",s.source_scored);j.set("source_right",s.source_right);j.set("frame_cycles",s.frame_cycles);j.set("frame_shadow",s.frame_shadow);
    j.set("prefill_ms",s.prefill_ms);j.set("decode_ms",s.decode_ms);j.set("finish",s.finish);j.set("sampling",s.sampling);
    j.set("error_status",s.error_status);j.set("error_code",s.error_code);
    auto extra=serve::Json::array();for(const auto& [k,v]:s.engine_extra)extra.as_array().emplace_back(serve::Json::Array{k,v});j.set("extra",std::move(extra));return j;
}
static serve::Json scenario(unsigned mode) {
    events.clear();stop=false;Options options;HttpConstraintRequests constraints;Probe probe;
    bool probing=true,samples=true;std::atomic<bool> failed{false};serve::GenerationQueue queue;
    MtpBatchRunner runner(queue);
    std::vector<std::shared_ptr<serve::GenerationTicket>> tickets;
    auto submit=[&](serve::GenerationJob j){auto t=queue.submit(std::move(j));CHECK(t);tickets.push_back(t);};
    submit(job(10,mode==0?2:1));
    if(mode==0) {submit(job(20));runner.arrivals=[&]{submit(job(30));};}
    if(mode==1) {auto j=job(20);j.session_id="fixed";submit(j);submit(job(30));}
    if(mode==2) {auto j=job(20);j.sampling.temperature=101;submit(j);submit(job(30));}
    if(mode==3) {auto j=job(20);j.sampling.temperature=0.7;j.sampling.seed=81;submit(j);}
    if(mode==4) {tickets.front()->cancel();submit(job(20));}
    if(mode==5) {auto j=job(20,3);j.stop_tokens={20};submit(j);}
    if(mode==6) {runner.memory_refusal=true;submit(job(20));}
    if(mode>=7 && mode<=11) {runner.fault=int(mode-6);submit(job(20));}
    if(mode==12) {runner.arrivals=[&]{auto j=job(20);j.session_id="nested";submit(j);submit(job(30));};}
    if(mode==13) {auto j=job(20);j.session_id="fixed";j.constraint=std::make_shared<serve::ConstraintSpec>();submit(j);}
    if(mode==14) {auto j=job(20);j.output=std::make_shared<serve::GenerationOutput>();j.output->consume_tokens=[](auto)->bool{throw std::runtime_error("publisher failure");};submit(j);}
    if(mode==15) {auto j=job(20,3);j.output=std::make_shared<serve::GenerationOutput>();j.output->consume_tokens=[](auto){return false;};submit(j);}
    if(mode==16)runner.arrivals=[&]{queue.shutdown();};

#ifdef QWEN_ENTRY
#include "entry_owner.inc"
#else
#include "current_binding.inc"
    std::string failure;
    try { owner.run(); } catch(const std::exception& e) {failure=e.what();}
#endif
    // The shell shuts the queue down after normal completion/failure.
    queue.shutdown();
    CHECK(queue.snapshot().running==0 && queue.snapshot().queued==0);
    auto out=serve::Json::object();out.set("failure",failure);out.set("failed",failed.load());
    auto trace=serve::Json::array();for(const auto& e:events)trace.as_array().emplace_back(e);out.set("events",std::move(trace));
    auto replies=serve::Json::array();
    for(auto& t:tickets) {bool done=false;auto ids=t->wait(std::chrono::milliseconds(1),done,false);CHECK(done);
        CHECK(!t->stats().first_token_ms && !t->stats().constraint && t->stats().imported_tokens==0);
        auto reply=serve::Json::object(),tokens=serve::Json::array();for(auto id:ids)tokens.as_array().emplace_back(std::uint64_t(id));
        reply.set("ids",std::move(tokens));reply.set("stats",stats_json(t->stats()));reply.set("error",t->error());replies.as_array().push_back(std::move(reply));}
    out.set("replies",std::move(replies));
    // Independently prescribed output streams, including committed-but-not-
    // published output at a throwing presentation callback (mode 14).
    static const std::vector<std::vector<std::vector<std::uint32_t>>> expected_ids{
        {{10,11},{20},{30}},{{10},{20},{30}},{{10},{},{30}},{{10},{20}},{{},{20}},
        {{10},{20}},{{10},{20}},{{},{}},{{},{}},{{},{}},{{},{}},{{},{20}},
        {{10},{20},{30}},{{10},{}},{{},{}},{{10},{20}},{{10}}
    };
    CHECK(tickets.size()==expected_ids.at(mode).size());
    for(std::size_t i=0;i<tickets.size();++i) {
        auto expected=serve::Json::array();for(auto id:expected_ids[mode][i])expected.push(std::uint64_t(id));
        CHECK(out.find("replies")->as_array()[i].find("ids")->dump()==expected.dump());
        const auto s=tickets[i]->stats();
        if(std::find(s.engine_extra.begin(),s.engine_extra.end(),std::pair<std::string,std::string>{"pool","11"})!=s.engine_extra.end()) {
            const std::vector<std::pair<std::string,std::string>> extras{
                {"profile","{\"witness\":7}"},{"prefill","{\"identity\":17}"},{"pool","11"},
                {"state_cache_refusal","\"native-only\""},{"batch_service","{\"points\":3}"},
                {"shared_execution","{\"witness\":7}"},{"final_state_digest","\"final\""},
                {"final_state_components","{\"a\":1}"},{"final_state_position","3"},{"prompt_state_digest","\"prompt\""}
            };
            CHECK(s.engine_extra==extras && s.prompt_tokens==1 && s.cached_tokens==0 && s.prefill_ms==1 && s.decode_ms==0);
            CHECK(s.cycles==2*s.generated_tokens && s.proposed==3*s.generated_tokens && s.accepted==s.generated_tokens);
            if(s.generated_tokens) {
                CHECK(s.ngram_rows==5 && s.ngram_accepted==4 && s.tool_rows==7 && s.tool_accepted==6 &&
                    s.source_shadow==8 && s.source_scored==9 && s.source_right==10 && s.frame_cycles==11 && s.frame_shadow==12);
                const serve::GenerationProgress progress{serve::GenerationPhase::decode,1,0,1};
                CHECK(tickets[i]->progress()==progress);
            }
        }
    }
    if(mode==0) {
        const std::vector<std::string> trace{
            "grow:queued","constraint:off","admit:10:0","grow:queued","built:1","probe:pending","step:proof",
            "nested:enter","can_prepare","constraint:off","nested:prepare","admit:20:1","can_prepare","output:20","finish:1","output:10","probe:complete",
            "grow:queued","constraint:off","admit:30:1","grow:spare","built:2","probe:pending","step:proof",
            "output:30","finish:1","output:11","finish:0","probe:complete","grow:spare","grow:spare"
        };CHECK(events==trace);
    }
    if(mode==2)CHECK(tickets[1]->error().find("sampling policy refused:")==0 && tickets.back()->error().empty());
    if(mode==5)CHECK(tickets[1]->stats().finish=="stop" && tickets[1]->stats().generated_tokens==1);
    return out;
}

struct ByteTokenizer final : serve::ApiTokenizer {
    std::vector<std::uint32_t> encode(std::string_view s,bool) const override {return {s.begin(),s.end()};}
    std::string decode(std::span<const std::uint32_t> ids,bool) const override {std::string s;for(auto id:ids)s+=token_bytes(id);return s;}
    std::string token_bytes(std::uint32_t id) const override {CHECK(id<256);return std::string(1,char(id));}
    std::size_t vocab_size()const override{return 256;}
    std::uint32_t find(std::string_view)const override{return UINT32_MAX;}
};
struct Writer final : serve::HttpResponseWriter {
    int status=0;bool gone=false,ended=false,disconnect=false;
    std::promise<void>* disconnected=nullptr;
    std::string body,content_type;
    std::vector<std::pair<std::string,std::string>> headers;
    void send(int s,std::string_view t,std::string_view b,const std::vector<std::pair<std::string,std::string>>& h)override{status=s;content_type=t;body=b;headers.insert(headers.end(),h.begin(),h.end());}
    bool begin_stream(int s,std::string_view t,const std::vector<std::pair<std::string,std::string>>& h)override{status=s;content_type=t;headers.insert(headers.end(),h.begin(),h.end());return true;}
    bool write(std::string_view b)override{body+=b;if(disconnect && !gone){gone=true;if(disconnected)disconnected->set_value();}return !gone;}
    void end_stream()override{ended=true;}
    bool client_gone()const override{return gone;}
    void add_response_header(std::string n,std::string v)override{headers.emplace_back(std::move(n),std::move(v));}
};
static std::string stable_http_json(std::string raw) {
    auto j=serve::parse_json(raw);CHECK(j.dump()==raw);
    // Exactly three volatile observations, at these explicit envelope paths.
    // No receipt field is removed, no recursive name-based stripping occurs.
    if(auto* id=j.find("id")) {CHECK(id->as_string().starts_with("cmpl-"));j.set("id","cmpl-reference");}
    if(auto* created=j.find("created")) {CHECK(created->is_number());j.set("created",0);}
    if(auto* receipt=j.find("strix"))if(auto* wait=receipt->find("queue_ms")) {
        CHECK(wait->as_double()>=0);auto copy=*receipt;copy.set("queue_ms",0);j.set("strix",std::move(copy));
    }
    return j.dump();
}
static serve::Json http_scenario(unsigned mode) {
    events.clear();stop=false;Options options;HttpConstraintRequests constraints;Probe probe;
    bool probing=true,samples=true;std::atomic<bool> failed{false};serve::GenerationQueue queue;
    MtpBatchRunner runner(queue);
    auto config=serve::ApiConfig{};config.model_id="qwen-owner-host";config.default_max_tokens=2;config.default_thinking=false;
    config.stream_receipts=true;config.sse_keepalive_seconds=0;config.sse_progress_seconds=0;
    config.engine_progress=true;config.http_slots=3;
    serve::ApiService api(config,serve::ApiTokenizerBinding(std::make_shared<ByteTokenizer>()),queue,64,true);
    Writer writer;writer.disconnect=mode==2;
    serve::HttpRequest request;request.method="POST";request.path="/v1/completions";
    request.body=mode==0?"{\"prompt\":\"A\",\"max_tokens\":2,\"return_token_ids\":true}":
        "{\"prompt\":\"A\",\"max_tokens\":1,\"stream\":true,\"return_token_ids\":true}";
    // The double's step-only auto-stop cannot retire a request cancelled by
    // the real queue before native admission. Give the harness its own explicit
    // phases; production remains alive at idle until its caller requests stop.
    runner.stop_when_empty=false;
    std::promise<void> idle,disconnected;
    auto idle_phase=idle.get_future();auto disconnect_phase=disconnected.get_future();
    writer.disconnected=&disconnected;
    bool first_idle=true;std::atomic<bool> api_returned{false};
    queue.set_idle_control([&] {
        if(first_idle) {
            first_idle=false;idle.set_value();
            if(mode==2) {
                CHECK(disconnect_phase.wait_for(std::chrono::seconds(3))==std::future_status::ready);
                const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(3);
                const auto waiting=[&]{return queue.has_waiting([](const auto&,auto){return true;});};
                while(waiting() && std::chrono::steady_clock::now()<until)std::this_thread::yield();
                CHECK(queue.snapshot().queued==1 && !queue.stopping() && !waiting());
            }
        } else {
            CHECK(api_returned && queue.snapshot().queued==0 && queue.snapshot().running==0);
            queue.shutdown();
        }
    });
    std::exception_ptr owner_error;
#ifndef QWEN_ENTRY
#include "current_binding.inc"
#endif
    auto run=std::async(std::launch::async,[&] {
        try {
#ifdef QWEN_ENTRY
#include "entry_owner.inc"
            CHECK(failure.empty());
#else
            owner.run();
#endif
        } catch(...) {owner_error=std::current_exception();queue.shutdown();}
    });
    queue.wake_control();
    const bool at_idle=idle_phase.wait_for(std::chrono::seconds(3))==std::future_status::ready;
    if(!at_idle){queue.shutdown();run.get();CHECK(at_idle);}
    try {api.handle(request,writer);}catch(...){queue.shutdown();run.wait();throw;}
    api_returned=true;queue.wake_control();
    const bool ready=run.wait_for(std::chrono::seconds(3))==std::future_status::ready;
    if(!ready)queue.shutdown();run.get();CHECK(ready);
    if(owner_error)std::rethrow_exception(owner_error);
    CHECK(writer.status==200 && queue.snapshot().running==0 && queue.snapshot().queued==0);
#ifndef QWEN_ENTRY
    CHECK(owner.runtime().live()==0 && !owner.runtime().reservation());
#endif
    if(mode==2) {
        CHECK(queue.snapshot().cancelled==1 && queue.snapshot().completed==0 && queue.snapshot().failed==0);
        CHECK(std::none_of(events.begin(),events.end(),[](const auto& e){return e.starts_with("admit:") || e.starts_with("step");}));
    } else CHECK(queue.snapshot().completed==1 && queue.snapshot().cancelled==0 && queue.snapshot().failed==0);
    std::string body;
    if(mode==0) {
        body=stable_http_json(writer.body);
        auto j=serve::parse_json(body);CHECK(j.find("choices")->as_array().front().find("text")->as_string()=="AB");
        CHECK(j.find("strix")->find("token_ids")->dump()=="[65,66]");
    } else {
        if(mode==1)CHECK(writer.ended);
        std::size_t at=0;
        while(at<writer.body.size()) {
            auto end=writer.body.find('\n',at);if(end==std::string::npos)end=writer.body.size();
            auto line=writer.body.substr(at,end-at);
            if(line.starts_with("data: {"))line="data: "+stable_http_json(line.substr(6));
            body+=line;if(end<writer.body.size())body+='\n';at=end+1;
        }
        if(mode==1)CHECK(body.find("data: [DONE]\n\n")!=std::string::npos);
    }
    auto out=serve::Json::object();out.set("body",body);out.set("status",writer.status);out.set("content_type",writer.content_type);
    auto headers=serve::Json::array();for(const auto& [k,v]:writer.headers)headers.push(serve::Json::Array{k,v});out.set("headers",std::move(headers));
    return out;
}

#ifndef QWEN_ENTRY
static void logical_generation_and_lifecycle() {
    events.clear();stop=false;serve::GenerationQueue queue;
    Options options;HttpConstraintRequests constraints;Probe probe;bool probing=true,samples=true;std::atomic<bool> failed{false};
    MtpBatchRunner runner(queue);
#include "current_binding.inc"
    auto& runtime=owner.runtime();
    CHECK(runtime.capabilities().owner_control && !runtime.capabilities().batch_decode && !runtime.capabilities().fork_lanes);
    CHECK(!runtime.cache_owner() && runtime.cache_entries()==0 && runtime.cache_stats().bytes==0);
    auto ticket=queue.submit(job(20));CHECK(runtime.reserve(serve::QueueSelection::fifo));
    const auto reserved=runtime.reservation();CHECK(reserved && reserved->ticket==ticket);
    CHECK(runtime.reserve(serve::QueueSelection::fifo) && reserved==runtime.reservation());
    auto first=runtime.activate(0,true);const std::uint32_t token=20;
    CHECK(runtime.publish(first.state,std::span(&token,1)));
    serve::GenerationStats formed;formed.generated_tokens=1;formed.engine_extra={{"session","{\"bytes\":true}"}};
    runtime.complete_native(first.state,formed);
    auto next_ticket=queue.submit(job(30));CHECK(runtime.reserve(serve::QueueSelection::fifo));auto next=runtime.activate();
    CHECK(next.state.slot==first.state.slot && next.state.generation!=first.state.generation);
    CHECK(runtime.ticket_cancelled(first.state) && !runtime.ticket_cancelled(next.state));
    unsigned rejected=0;
    try{runtime.publish(first.state,std::span(&token,1));}catch(const std::invalid_argument&){++rejected;}
    try{runtime.complete_native(first.state,{});}catch(const std::invalid_argument&){++rejected;}
    try{runtime.publish_progress(first.state,{});}catch(const std::invalid_argument&){++rejected;}
    try{runtime.notify_admitted(first.state);}catch(const std::invalid_argument&){++rejected;}
    CHECK(rejected==4 && runtime.live()==1 && next_ticket->stats().generated_tokens==0);
    runtime.notify_admitted(next.state);
    const serve::GenerationProgress native_progress{serve::GenerationPhase::prefill,1,0,1};
    runtime.publish_progress(next.state,native_progress);CHECK(next_ticket->progress()==native_progress);
    runtime.complete_native(next.state,{});CHECK(runtime.live()==0);
    CHECK(ticket->stats().generated_tokens==1 && ticket->stats().engine_extra==formed.engine_extra);
    // Occupy logical slot 0 without touching the runner. Production admission
    // must publish through logical slot 1 while native first-free is slot 0.
    auto held_ticket=queue.submit(job(40));CHECK(runtime.reserve(serve::QueueSelection::fifo));auto held=runtime.activate();
    CHECK(held.state.slot==0);
    auto native_ticket=queue.submit(job(50));
    for(unsigned turn=0;turn<4 && native_ticket->stats().generated_tokens==0;++turn)runtime.step();
    CHECK(runtime.live()==1 && native_ticket->stats().generated_tokens==1 && held_ticket->stats().generated_tokens==0);
    CHECK(std::find(events.begin(),events.end(),"admit:50:0")!=events.end());
    runtime.complete_native(held.state,{});
    queue.shutdown();owner.run();
}
static void idle_request_wake() {
    events.clear();stop=false;serve::GenerationQueue queue;
    Options options;HttpConstraintRequests constraints;Probe probe;bool probing=true,samples=true;std::atomic<bool> failed{false};
    MtpBatchRunner runner(queue);
#include "current_binding.inc"
    std::promise<void> idle;auto idle_ready=idle.get_future();
    queue.set_idle_control([&]{idle.set_value();});
    auto run=std::async(std::launch::async,[&]{owner.run();});
    queue.wake_control();
    const bool waiting=idle_ready.wait_for(std::chrono::seconds(2))==std::future_status::ready;
    if(!waiting){queue.shutdown();run.get();CHECK(waiting);}
    const auto start=std::chrono::steady_clock::now();
    auto ticket=queue.submit(job(60));
    bool done=false;std::vector<std::uint32_t> ids;const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
    while(!done && std::chrono::steady_clock::now()<deadline) {
        auto part=ticket->wait(std::chrono::duration_cast<std::chrono::milliseconds>(deadline-std::chrono::steady_clock::now()),done,false);
        ids.insert(ids.end(),part.begin(),part.end());
    }
    if(!done)queue.shutdown();
    const bool ready=run.wait_for(std::chrono::seconds(2))==std::future_status::ready;
    if(!ready)queue.shutdown();run.get();
    CHECK(ready && done && ids==std::vector<std::uint32_t>{60} && std::chrono::steady_clock::now()-start<std::chrono::milliseconds(750));
}
static void native_boundary_contracts(bool counter=false) {
    events.clear();stop=false;serve::GenerationQueue queue;
    Options options;HttpConstraintRequests constraints;Probe probe;bool probing=true,samples=true;std::atomic<bool> failed{false};
    MtpBatchRunner runner(queue);runner.check_mutex=true;runner.completed=counter;
#include "current_binding.inc"
    std::vector<std::shared_ptr<serve::GenerationTicket>> tickets;
    auto submit=[&](auto j){auto t=queue.submit(j);CHECK(t);tickets.push_back(t);};
    if(counter) {
        auto j=job(65);j.sampling.temperature=0.7;j.sampling.seed=41;submit(j);j.prompt={75};submit(j);
    } else {
        named_mutex=&runner.mutex;
        submit(job(10,3));
        runner.arrivals=[&] {submit(job(20,2));auto j=job(30);j.session_id="barrier";submit(j);submit(job(40));};
    }
    try {owner.run();}catch(...){named_mutex=nullptr;throw;}
    named_mutex=nullptr;
    CHECK(!failed && queue.snapshot().running==0 && queue.snapshot().queued==0);
    if(counter) {
        CHECK(std::count(events.begin(),events.end(),"sampler:41:1")==1 && std::count(events.begin(),events.end(),"sampler:41:2")==1);
    } else {
        const auto named=std::find(events.begin(),events.end(),"named:barrier");CHECK(named!=events.end());
        CHECK(std::find(events.begin(),named,"output:12")!=named && std::find(events.begin(),named,"output:21")!=named);
        CHECK(std::find(named,events.end(),"admit:40:0")!=events.end());
        CHECK(std::find(events.begin(),events.end(),"probe:complete")==events.end()); // native step reported no proof
        CHECK(tickets.size()==4 && tickets[0]->stats().generated_tokens==3 && tickets[1]->stats().generated_tokens==2);
        const std::vector<std::pair<std::string,std::string>> session_receipt{{"session","{\"fixed\":true}"}};
        CHECK(tickets[2]->stats().engine_extra==session_receipt);
    }
}
static void reserved_failure_cleanup() {
    events.clear();stop=false;serve::GenerationQueue queue;
    Options options;HttpConstraintRequests constraints;Probe probe;bool probing=true,samples=true;std::atomic<bool> failed{false};
    MtpBatchRunner runner(queue);runner.fault=4;runner.check_mutex=true;
#include "current_binding.inc"
    auto ordinary=queue.submit(job(10,2));std::shared_ptr<serve::GenerationTicket> named,queued;
    runner.arrivals=[&] {auto j=job(20);j.session_id="waiting";named=queue.submit(j);queued=queue.submit(job(30));};
    bool threw=false;try{owner.run();}catch(const std::runtime_error& e){threw=std::string(e.what())=="step failure";}
    CHECK(threw && failed && named && queued);
    CHECK(ordinary->stats().finish=="execution_failure" && ordinary->error()=="step failure");
    CHECK(named->stats().finish=="length" && named->error()=="step failure");
    CHECK(queued->stats().finish=="cancelled" && queued->error()=="server shutting down");
    CHECK(owner.runtime().live()==0 && !owner.runtime().reservation() && queue.snapshot().running==0 && queue.snapshot().queued==0);
    CHECK(available(runner.mutex) && std::find(events.begin(),events.end(),"named:waiting")==events.end());
    const auto failed_at=std::find(events.begin(),events.end(),"fail_requests:step failure");
    CHECK(failed_at!=events.end() && std::find(failed_at,events.end(),"abandon")!=events.end());
}
static void owner_failure_texts() {
    for(bool unknown:{false,true}) {
        events.clear();stop=false;serve::GenerationQueue queue;
        Options options;HttpConstraintRequests constraints;Probe probe;bool probing=true,samples=true;std::atomic<bool> failed{false};
        MtpBatchRunner runner(queue);runner.fault=2;runner.empty_error=true;runner.unknown_error=unknown;
#include "current_binding.inc"
        auto live=queue.submit(job(10));auto queued=queue.submit(job(20));
        bool threw=false;try{owner.run();}catch(...){threw=true;}
        CHECK(threw && failed && live->error()=="engine failure" && live->stats().finish=="length");
        CHECK(queued->error()=="server shutting down" && !owner.runtime().reservation() && owner.runtime().live()==0);
        CHECK(queue.snapshot().running==0 && queue.snapshot().queued==0 && available(runner.mutex));
    }
}
#endif
}
#ifndef QWEN_ENTRY
// The actual owner, eight occupied slots, and no native nested-pump calls.
static void queue_protection_without_nesting() {
    namespace serve = strix::serve;
    using Owner = strix::qwen::HttpSessionOwner;
    serve::GenerationQueue queue;
    std::vector<std::shared_ptr<serve::GenerationTicket>> tickets;
    for (std::uint32_t i = 0; i < 10; ++i) {
        serve::GenerationJob job; job.prompt = {i + 1}; job.max_tokens = 1;
        tickets.push_back(queue.submit(std::move(job)));
        CHECK(tickets.back());
    }
    std::size_t held = 0, completed = 0, steps = 0;
    std::vector<Owner::Callbacks> active;
    Owner::Native native;
    native.queue_protection.prepare = [](auto prompt) -> serve::QueuedPrefixProtection::Prepared {
        return std::make_shared<const std::uint32_t>(prompt.front());
    };
    native.queue_protection.protect = [&](const auto&) -> serve::QueuedPrefixProtection::Handle {
        ++held;
        return std::shared_ptr<void>(new int(0), [&](void* p) { delete static_cast<int*>(p); --held; });
    };
    native.add_slot = [](bool) {};
    native.has_free_slot = [&] { return active.size() < 8; };
    native.active = [&] { return active.size(); };
    native.submit = [&](const auto&, Owner::Callbacks cb) { active.push_back(std::move(cb)); };
    // STRIX_BATCH_NEST_ROWS=0: the runner never invokes this pump.
    native.service_admission = [](auto, auto) {};
    native.step = [&] {
        ++steps;
        if (steps == 1) {
            CHECK(active.size() == 8 && held == 10);
            tickets.back()->cancel();
            const std::uint32_t token = 1;
            for (auto& cb : active) CHECK(cb.tokens(std::span(&token, 1)));
            CHECK(held == 2); // only the two waiting jobs still hold references
            return;
        }
        CHECK(held == 1); // cancelled waiting job was released by a normal turn
        for (auto& cb : active) { cb.finish({}, {}); ++completed; }
        active.clear();
        if (completed == 9) queue.shutdown();
    };
    native.fail_requests = [](const auto&) {};
    native.abandon = [&] { active.clear(); };
    Owner owner(queue, 64, 8, std::move(native));
    owner.run();
    CHECK(steps == 3 && held == 0 && completed == 9);
    CHECK(owner.queue_protection_stats().released_left == 1);
    CHECK(owner.queue_protection_stats().handed_over == 9);
}
#endif
int main(int argc,char** argv) try {
    if(argc==2 && std::string(argv[1])=="--failure-probe") {CHECK(argc==0);}
#ifndef QWEN_ENTRY
    if(argc==2 && std::string(argv[1])=="--queue-protection") {queue_protection_without_nesting();return 0;}
    if(argc==2 && std::string(argv[1])=="--readiness-phases") {
        for(unsigned mode=0;mode<3;++mode)(void)strix::decode_cli::http_scenario(mode);
        std::cout<<"Qwen idle, completion, pre-admission disconnect and teardown phases passed\n";return 0;
    }
    if(argc==2 && std::string(argv[1])=="--sampling-counter") {strix::decode_cli::native_boundary_contracts(true);return 0;}
#endif
    if(argc==3 && (std::string(argv[1])=="--http-capture" || std::string(argv[1])=="--http-compare")) {
        auto all=strix::serve::Json::array();for(unsigned i=0;i<3;++i)all.push(strix::decode_cli::http_scenario(i));
        const auto bytes=all.dump()+"\n";
        if(std::string(argv[1])=="--http-capture"){std::ofstream f(argv[2],std::ios::binary);CHECK(f);f<<bytes;}
        else {std::ifstream f(argv[2],std::ios::binary);CHECK(f);std::string expected((std::istreambuf_iterator<char>(f)),{});CHECK(bytes==expected);}
        return 0;
    }
    auto all=strix::serve::Json::array();
    for(unsigned i=0;i<=16;++i)all.as_array().push_back(strix::decode_cli::scenario(i));
    const auto bytes=all.dump()+"\n";
    if(argc==3 && std::string(argv[1])=="--capture") {std::ofstream f(argv[2],std::ios::binary);CHECK(f);f<<bytes;}
    else if(argc==3 && std::string(argv[1])=="--compare") {std::ifstream f(argv[2],std::ios::binary);CHECK(f);std::string expected((std::istreambuf_iterator<char>(f)),{});CHECK(bytes==expected);}
    else {
#ifndef QWEN_ENTRY
        strix::decode_cli::logical_generation_and_lifecycle();
        strix::decode_cli::idle_request_wake();
        strix::decode_cli::native_boundary_contracts();
        strix::decode_cli::reserved_failure_cleanup();
        strix::decode_cli::owner_failure_texts();
        for(unsigned i=0;i<3;++i)(void)strix::decode_cli::http_scenario(i);
#endif
        std::cout<<"Qwen native-control host checks passed\n";
    }
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
