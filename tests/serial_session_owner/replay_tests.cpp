#include "strix/serve/api.hpp"
#include "strix/serve/health.hpp"
#ifndef SERIAL_ENTRY
#include "strix/serve/serial_session_owner.hpp"
#include "strix/serve/session_api.hpp"
#endif
#include <atomic>
#include <algorithm>
#include <fstream>
#include <future>
#include <cmath>
#include <iostream>
#include <thread>
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string("serial check: ")+ #x); } while(false)
namespace strix::qwen::profile { struct Resolution { std::string json() const {return "{\"profile\":17}";} }; }
namespace strix::decode_cli {
namespace serve = strix::serve;
using serve::SessionApiError;
struct MtpConstraintRow { serve::ConstraintVerdict verdict(const std::string&) {return {}; } };
struct HttpConstraintRequests { std::shared_ptr<MtpConstraintRow> make(const std::shared_ptr<const serve::ConstraintSpec>& s) {return s ? std::make_shared<MtpConstraintRow>() : nullptr;} };
struct MtpServeRequest {
 serve::BackendProbeSlot* backend_probe=nullptr;
 std::shared_ptr<MtpConstraintRow> constraint;
 std::function<void()> on_admitted;
 std::string session_id,session_operation,session_durability;
 serve::CacheRetention cache_retention=serve::CacheRetention::server_policy;
 std::vector<std::uint32_t> prompt,stop_tokens;
 std::uint32_t max_tokens=0;
 std::optional<bool> ngram_draft;
 std::function<bool(std::span<const std::uint32_t>)> on_tokens;
 std::function<void(serve::GenerationProgress)> on_progress;
};
struct MtpServeResult {
 std::uint64_t restored_tokens=1,generated_tokens=2,cycles=3,proposed=4,accepted=2,ngram_rows=6,ngram_accepted=1,final_state_position=9;
 double prompt_ms=2,decode_ms=4;
 std::optional<serve::ConstraintVerdict> constraint;
 std::string finish="length",final_state_digest="final-state",final_target_state_digest,final_state_components="{\"all\":[1,2,3]}",prompt_state_digest="prompt-state",cache_refusal="native",session_json="{\"owner\":\"native\"}",rtx_import_json="{\"import\":true}",error;
};
using MtpServeRunner=std::function<MtpServeResult(const MtpServeRequest&)>;
static void append_mtp_pool_receipt(serve::GenerationStats& s,const MtpServeResult&) {s.engine_extra.emplace_back("pool","7");}
static void add_prefill_stats(serve::GenerationStats& s) {s.engine_extra.emplace_back("identity","\"0123456789abcdef\"");}
#include "serial_bridge.inc"
#ifndef SERIAL_ENTRY
#include "serial_routes.inc"
#endif
}
using namespace strix;
static std::chrono::steady_clock::time_point submitted;
static auto clock_submit() {return submitted;}
static void execute(serve::GenerationQueue& q,serve::GenerationEngine& engine,const std::function<void(const serve::GenerationStats&)>& observer={}) {
#ifdef SERIAL_ENTRY
 q.run(engine,observer);
#else
 if(!observer) {
  auto& queue=q;
#include "serial_binding.inc"
  owner.runtime().run();return;
 }
 serve::SerialSessionOwner owner(q,engine,observer);
 CHECK(!owner.runtime().cache_owner() && !owner.runtime().capabilities().prefix_cache);
 owner.runtime().run();CHECK(owner.runtime().live()==0 && !owner.runtime().reservation());
#endif
}
static serve::Json stats_json(const serve::GenerationStats& s) {
 auto j=serve::Json::object();
#define FIELD(n) j.set(#n,static_cast<double>(s.n))
 FIELD(error_status);FIELD(prompt_tokens);FIELD(cached_tokens);FIELD(generated_tokens);FIELD(cycles);FIELD(proposed);FIELD(accepted);FIELD(ngram_rows);FIELD(ngram_accepted);FIELD(tool_rows);FIELD(tool_accepted);FIELD(source_shadow);FIELD(source_scored);FIELD(source_right);FIELD(frame_cycles);FIELD(frame_shadow);FIELD(imported_tokens);FIELD(prefill_ms);FIELD(decode_ms);
#undef FIELD
 j.set("error_code",s.error_code);j.set("finish",s.finish);j.set("sampling",s.sampling);
 CHECK(s.queue_ms>=0);j.set("queue_ms",0);
 // Only service-envelope time is volatile. Check its attribution separately.
 j.set("first_token_ms",s.first_token_ms ? serve::Json(0) : serve::Json());
 if(s.constraint){auto c=serve::Json::object();c.set("status",s.constraint->status);c.set("reason",s.constraint->reason);j.set("constraint",c);}
 auto receipt=serve::Json::array();for(const auto& [k,v]:s.engine_extra)receipt.push(serve::Json::Array{k,v});j.set("receipt",receipt);return j;
}
struct Fake final : serve::GenerationEngine {
 serve::GenerationQueue& queue;unsigned mode;std::function<void()> cancel;
 std::vector<std::string>& events;
 Fake(serve::GenerationQueue& q,unsigned m,std::vector<std::string>& e):queue(q),mode(m),events(e){}
 std::uint64_t context_tokens()const override{return 64;}
 serve::GenerationStats generate(serve::GenerationJob& j)override {
  auto id=j.prompt.front();events.push_back("generate:"+std::to_string(id));
  if(j.on_admitted)j.on_admitted();if(j.on_progress)j.on_progress({serve::GenerationPhase::prefill,1,0,1});
  serve::GenerationStats s;s.prompt_tokens=9;s.cached_tokens=2;s.imported_tokens=1;s.generated_tokens=2;s.decode_ms=4;s.first_token_ms=7;s.engine_extra={{"complete","{\"rows\":[4,8],\"rng\":9}"}};
  if(id==1) {
   if(mode==1)throw serve::GenerationRequestError(s,"local");
   if(mode==2)throw std::runtime_error("standard");
   if(mode==3)throw std::runtime_error("");
   if(mode==4)throw 7;
   if(mode==5)serve::observe_disarm();
   if(mode==6 && cancel)cancel();
   if(mode==7)queue.shutdown();
  }
  std::uint32_t ids[]={id,id+1};j.on_tokens(ids);return s;
 }
};
static serve::Json scenario(unsigned mode,bool observed) {
 serve::observe_disarm();if(observed)serve::observe_arm(0);
 serve::GenerationQueue queue;submitted=std::chrono::steady_clock::now()-std::chrono::milliseconds(120);
 std::vector<std::string> events;Fake engine(queue,mode,events);
 serve::GenerationJob j;j.prompt={1};auto a=queue.submit(j);j.prompt={3};auto b=queue.submit(j);CHECK(a&&b);
 if(mode==8)a->cancel();engine.cancel=[a]{a->cancel();};
 if(mode==9){j.output=std::make_shared<serve::GenerationOutput>();j.output->consume_tokens=[](auto)->bool{throw std::runtime_error("callback");}; j.prompt={5};auto t=queue.submit(j);CHECK(t);}
 unsigned calls=0;
 execute(queue,engine,[&](const auto& s){events.push_back("observe:"+s.finish);++calls;if(queue.snapshot().queued==0)queue.shutdown();});
 CHECK(calls==(mode==7?1U:mode==9?3U:2U));
 auto result=serve::Json::object();auto ts=serve::Json::array();
 for(const auto& t:{a,b}){bool done=false;auto ids=t->wait(std::chrono::milliseconds(0),done,false);CHECK(done);auto x=stats_json(t->stats());x.set("error",t->error());x.set("observed",t->observed());auto arr=serve::Json::array();for(auto id:ids)arr.push(static_cast<int>(id));x.set("ids",arr);ts.push(x);}
 const auto s=a->stats();CHECK(s.queue_ms>=119);if(s.first_token_ms)CHECK(std::abs(*s.first_token_ms-(7+(a->observed()?s.queue_ms:0)))<0.0001);
 auto q=queue.snapshot();CHECK(q.running==0&&q.queued==0&&q.busy_seconds>=0);
 auto total=serve::Json::array();for(auto n:{q.completed,q.failed,q.cancelled,q.prompt_tokens,q.cached_tokens,q.generated_tokens})total.push(static_cast<double>(n));
 auto o=queue.observed();double ttft_sum=0;std::uint64_t observed_requests=0;
 for(auto ticket:{a,b})if(ticket->observed()&&ticket->error().empty()) {++observed_requests;if(ticket->stats().first_token_ms)ttft_sum+=*ticket->stats().first_token_ms/1000;}
 CHECK(o.requests==observed_requests&&std::abs(o.ttft_seconds.sum-ttft_sum)<1e-9);
 auto obs=serve::Json::array();for(auto n:{o.requests,o.prompt_tokens,o.cached_tokens,o.imported_tokens,o.computed_tokens,o.ttft_seconds.count,o.decode_tokens_per_second.count})obs.push(static_cast<double>(n));
 if(observed && mode==0){CHECK(o.requests==2&&o.computed_tokens==12&&o.ttft_seconds.count==2&&o.decode_tokens_per_second.count==2);CHECK(o.ttft_seconds.sum>=0.254);}
 auto buckets=serve::Json::array();for(auto n:o.ttft_seconds.counts)buckets.push(static_cast<double>(n));obs.push(buckets);buckets=serve::Json::array();for(auto n:o.decode_tokens_per_second.counts)buckets.push(static_cast<double>(n));obs.push(buckets);obs.push(o.decode_tokens_per_second.sum);
 auto ev=serve::Json::array();for(auto& e:events)ev.push(e);result.set("tickets",ts);result.set("totals",total);result.set("observed",obs);result.set("events",ev);serve::observe_disarm();return result;
}
static serve::Json bridge(unsigned mode) {
 using namespace decode_cli;serve::GenerationQueue queue;HttpConstraintRequests constraints;std::vector<std::string> events;
 MtpServeRunner native=[&](const MtpServeRequest& r) {
  events.push_back("native:"+r.session_id);CHECK(r.cache_retention==serve::CacheRetention::no_store && r.ngram_draft==false);
  if(mode==1)throw SessionApiError(409,"session_incompatible","incompatible");if(mode==2)throw std::runtime_error("native failure");if(mode==3)throw 17;
  if(r.on_admitted)r.on_admitted();if(r.on_progress)r.on_progress({serve::GenerationPhase::prefill,1,0,1});
  std::uint32_t ids[]={65,66};r.on_tokens(ids);return MtpServeResult{};
 };
 MtpBridgeEngine engine(native,64,constraints,true,true,true);serve::GenerationJob j;j.prompt={10};j.session_id="named";j.cache_retention=serve::CacheRetention::no_store;j.ngram_draft=false;
 if(mode==4)j.constraint=std::make_shared<serve::ConstraintSpec>();
 auto a=queue.submit(j);auto b=queue.submit(j);unsigned count=0;execute(queue,engine,[&](const auto&){if(++count==2)queue.shutdown();});
 auto out=serve::Json::array();for(auto t:{a,b}){auto v=stats_json(t->stats());v.set("error",t->error());out.push(v);}out.push(engine.failed());return out;
}
struct ByteTokenizer final : strix::serve::ApiTokenizer {
    std::vector<std::uint32_t> encode(std::string_view s,bool) const override {return {s.begin(),s.end()};}
    std::string decode(std::span<const std::uint32_t> ids,bool) const override {std::string s;for(auto id:ids)s+=token_bytes(id);return s;}
    std::string token_bytes(std::uint32_t id) const override {CHECK(id<256);return std::string(1,char(id));}
    std::size_t vocab_size()const override{return 256;}
    std::uint32_t find(std::string_view)const override{return UINT32_MAX;}
};
struct Writer final : strix::serve::HttpResponseWriter {
    int status=0;bool gone=false,ended=false,disconnect=false;
    std::string body,content_type;
    std::vector<std::pair<std::string,std::string>> headers;
    void send(int s,std::string_view t,std::string_view b,const std::vector<std::pair<std::string,std::string>>& h)override{status=s;content_type=t;body=b;headers.insert(headers.end(),h.begin(),h.end());}
    bool begin_stream(int s,std::string_view t,const std::vector<std::pair<std::string,std::string>>& h)override{status=s;content_type=t;headers.insert(headers.end(),h.begin(),h.end());return true;}
    bool write(std::string_view b)override{body+=b;if(disconnect)gone=true;return !gone;}
    void end_stream()override{ended=true;}
    bool client_gone()const override{return gone;}
    void add_response_header(std::string n,std::string v)override{headers.emplace_back(std::move(n),std::move(v));}
};
static std::string stable_http_json(std::string raw) {
    auto j=strix::serve::parse_json(raw);CHECK(j.dump()==raw);
    // Exactly three volatile observations, at these explicit envelope paths.
    // No receipt field is removed, no recursive name-based stripping occurs.
    if(auto* id=j.find("id")) {CHECK(id->as_string().starts_with("cmpl-"));j.set("id","cmpl-reference");}
    if(auto* created=j.find("created")) {CHECK(created->is_number());j.set("created",0);}
    if(auto* receipt=j.find("strix"))if(auto* wait=receipt->find("queue_ms")) {
        CHECK(wait->as_double()>=0);auto copy=*receipt;copy.set("queue_ms",0);j.set("strix",std::move(copy));
    }
    return j.dump();
}
static serve::Json http(unsigned mode) {
 using namespace decode_cli;serve::GenerationQueue queue;HttpConstraintRequests constraints;
 MtpServeRunner native=[&](const MtpServeRequest& r){MtpServeResult result;result.generated_tokens=r.max_tokens;std::vector<std::uint32_t> ids;for(unsigned n=0;n<r.max_tokens;++n)ids.push_back(65+n);if(r.on_admitted)r.on_admitted();r.on_tokens(ids);return result;};
 MtpBridgeEngine engine(native,64,constraints,true,true,true);
 auto config=serve::ApiConfig{};config.model_id="serial-host";config.default_max_tokens=2;config.default_thinking=false;config.stream_receipts=true;config.sse_keepalive_seconds=0;config.sse_progress_seconds=0;
 serve::ApiService api(config,serve::ApiTokenizerBinding(std::make_shared<ByteTokenizer>()),queue,64,false);
 Writer writer;writer.disconnect=mode==2;
 serve::HttpRequest request;request.method="POST";request.path="/v1/completions";
 request.body=mode==0?"{\"prompt\":\"A\",\"max_tokens\":2,\"return_token_ids\":true}":"{\"prompt\":\"A\",\"max_tokens\":1,\"stream\":true,\"return_token_ids\":true}";
 auto worker=std::async(std::launch::async,[&]{execute(queue,engine);});
 try{api.handle(request,writer);}catch(...){queue.shutdown();worker.get();throw;}
 queue.shutdown();worker.get();CHECK(writer.status==200);
 std::string body;
 if(mode==0){body=stable_http_json(writer.body);auto j=serve::parse_json(body);CHECK(j.find("choices")->as_array().front().find("text")->as_string()=="AB");}
 else {std::size_t at=0;while(at<writer.body.size()){auto end=writer.body.find('\n',at);if(end==std::string::npos)end=writer.body.size();auto line=writer.body.substr(at,end-at);if(line.starts_with("data: {"))line="data: "+stable_http_json(line.substr(6));body+=line;if(end<writer.body.size())body+='\n';at=end+1;}if(mode==1)CHECK(writer.ended && body.find("[DONE]")!=std::string::npos);}
 auto out=serve::Json::object();out.set("body",body);out.set("status",writer.status);out.set("content_type",writer.content_type);auto headers=serve::Json::array();for(auto& [k,v]:writer.headers)headers.push(serve::Json::Array{k,v});out.set("headers",headers);return out;
}
static void wake_and_throw() {
 serve::GenerationQueue q;std::vector<std::string> events;Fake engine(q,0,events);std::promise<void> idle;auto ready=idle.get_future();q.set_idle_control([&]{idle.set_value();});
 auto run=std::async(std::launch::async,[&]{execute(q,engine);});q.wake_control();CHECK(ready.wait_for(std::chrono::seconds(2))==std::future_status::ready);
 serve::GenerationJob j;j.prompt={1};auto t=q.submit(j);bool done=false;auto ids=t->wait(std::chrono::seconds(2),done,false);if(!done)ids=t->wait(std::chrono::seconds(2),done,false);q.shutdown();run.get();CHECK(done&&t->stats().generated_tokens==2);
 serve::GenerationQueue other;Fake e(other,0,events);auto a=other.submit(j);auto b=other.submit(j);bool threw=false;
 try{execute(other,e,[](const auto&){throw std::runtime_error("observer escaped");});}catch(const std::runtime_error& x){threw=std::string(x.what())=="observer escaped";}
 CHECK(threw&&!other.stopping()&&other.snapshot().completed==1&&other.snapshot().queued==1);bool finished=false;a->wait(std::chrono::milliseconds(0),finished,false);CHECK(!finished);other.shutdown();CHECK(b->error()=="server shutting down");
}

#ifndef SERIAL_ENTRY
static void leases_and_expiry() {
 serve::GenerationQueue q;std::vector<std::string> events;Fake e(q,0,events);serve::SerialSessionOwner owner(q,e);auto& r=owner.runtime();
 CHECK(r.step(false)); // idle deterministic entry must never block
 auto caps=r.capabilities();CHECK(caps.owner_control&&caps.max_slots==1&&!caps.batch_decode&&!caps.fork_lanes&&!caps.external_append&&!caps.logprobs&&!caps.constraints&&!caps.turn_commit);
 serve::GenerationJob j;j.prompt={1};auto a=q.submit(j);a->cancel();CHECK(r.reserve_serial());auto first=r.activate(0,true);CHECK(r.ticket_cancelled(first.state));r.complete_native(first.state,{});CHECK(a->stats().finish=="cancelled");
 auto b=q.submit(j);CHECK(r.reserve_serial());auto next=r.activate(0,true);CHECK(next.state.slot==first.state.slot&&next.state.generation!=first.state.generation);
 unsigned refusals=0;try{r.publish(first.state,{});}catch(const std::invalid_argument&){++refusals;}try{r.complete_native(first.state,{});}catch(const std::invalid_argument&){++refusals;}try{r.notify_admitted(first.state);}catch(const std::invalid_argument&){++refusals;}try{r.publish_progress(first.state,{});}catch(const std::invalid_argument&){++refusals;}CHECK(refusals==4&&r.live()==1&&r.ticket_cancelled(first.state));
 r.complete_native(next.state,{});CHECK(b->error().empty()&&r.live()==0);q.shutdown();r.run();
 struct Expiring final:serve::GenerationEngine {
  std::uint64_t context_tokens()const override{return 64;}
  serve::GenerationStats generate(serve::GenerationJob& job)override {
   CHECK(job.observed);std::this_thread::sleep_for(std::chrono::milliseconds(50));CHECK(!job.observation_complete());serve::GenerationStats s;s.generated_tokens=2;s.first_token_ms=7;return s;
  }
 } expiring;
 serve::GenerationQueue eq;serve::observe_disarm();serve::observe_arm(0.01);auto t=eq.submit(j);execute(eq,expiring,[&](const auto&){eq.shutdown();});CHECK(!t->observed()&&eq.observed().requests==0&&t->stats().first_token_ms==7&&eq.snapshot().busy_seconds>=0.05);serve::observe_disarm();
 // Empty and unknown idle callback exceptions retain caller-owned admission.
 for(bool unknown:{false,true}) {serve::GenerationQueue idle;idle.set_idle_control([&]{if(unknown)throw 1;throw std::runtime_error("");});idle.wake_control();bool threw=false;try{execute(idle,e);}catch(...){threw=true;}CHECK(threw&&!idle.stopping());idle.shutdown();}
}
static void native_control_routes() {
 using namespace decode_cli;std::vector<std::string> commands;
 MtpServeRunner native=[&](const MtpServeRequest& r){commands.push_back(r.session_operation+":"+r.session_id+":"+r.session_durability);return MtpServeResult{};};
 for(const auto& op:{"list","stats","delete","checkpoint","park","resume"}) {
  serve::HttpRequest request;request.method="POST";request.path="/v1/sessions/named/"+std::string(op);request.body=R"({"durability":"disk"})";
  if(std::string(op)=="list"){request.method="GET";request.path="/v1/sessions";}
  if(std::string(op)=="stats"){request.method="GET";request.path="/v1/cache/stats";}
  if(std::string(op)=="delete"){request.method="DELETE";request.path="/v1/sessions/named";}
  Writer writer;CHECK(handle_session_route(true,native,request,writer)&&writer.status==200&&writer.body==MtpServeResult{}.session_json);
 }
 CHECK((commands==std::vector<std::string>{"list::","stats::","delete:named:","checkpoint:named:disk","park:named:","resume:named:"}));
 serve::HttpRequest req;req.method="POST";req.path="/v1/sessions/bad.id/park";Writer writer;CHECK(handle_session_route(true,native,req,writer)&&writer.status==400&&commands.size()==6);
 CHECK(!handle_session_route(false,native,req,writer));
 serve::GenerationQueue q;std::promise<void> idle;auto ready=idle.get_future();serve::BackendProbeSlot probe([&]{q.wake_control();});
 q.set_idle_control([&]{probe.boundary(serve::BackendProof::control_only);idle.set_value();});
 std::vector<std::string> events;Fake e(q,0,events);auto owner=std::async(std::launch::async,[&]{execute(q,e);});(void)probe.snapshot();CHECK(ready.wait_for(std::chrono::seconds(2))==std::future_status::ready);CHECK(probe.snapshot().proof==serve::BackendProof::control_only);
 q.shutdown();owner.get();probe.fail();CHECK(probe.snapshot().state==serve::BackendState::failed);
}

static void native_policy_receipts() {
 using namespace decode_cli;
 for(bool cancel:{false,true})for(bool local_error:{false,true}) {
  serve::GenerationQueue q;HttpConstraintRequests constraints;std::shared_ptr<serve::GenerationTicket> t;
  serve::ConstraintVerdict verdict;verdict.status="accepted";verdict.accepting=true;verdict.phase="answer";verdict.reason="native";verdict.execution="serial-host";verdict.closure="budget";verdict.reasoning_tokens=3;verdict.forced_ids={11,12};verdict.execution_fallback="policy";verdict.profile_rows=17;verdict.profile_window_rows=2;verdict.profile_known_rows=9;verdict.profile_mask_ms=1;verdict.profile_select_ms=2;verdict.profile_window_row_ms=3;
  MtpServeRunner native=[&](const MtpServeRequest& r){CHECK(r.constraint&&r.session_id=="named"&&r.stop_tokens==std::vector<std::uint32_t>{63}&&r.max_tokens==2);if(cancel)t->cancel();std::uint32_t ids[]={65,66};CHECK(r.on_tokens(ids)==!cancel);MtpServeResult result;result.constraint=verdict;if(local_error)result.error="refused with receipt";return result;};
  MtpBridgeEngine engine(native,64,constraints,true,true,true);serve::GenerationJob j;j.prompt={10};j.max_tokens=2;j.stop_tokens={63};j.session_id="named";auto spec=std::make_shared<serve::ConstraintSpec>();spec->transform.frequency=1;spec->logprobs=true;spec->thinking=serve::ThinkingPolicy{};j.constraint=spec;t=q.submit(j);
  execute(q,engine,[&](const auto&){q.shutdown();});auto stats=t->stats();CHECK(stats.sampling=="greedy-logit-transform-v1"&&stats.generated_tokens==2&&stats.engine_extra.size()==10&&stats.constraint);
  const auto& v=*stats.constraint;CHECK(v.accepting==verdict.accepting&&v.phase==verdict.phase&&v.execution==verdict.execution&&v.closure==verdict.closure&&v.reasoning_tokens==3&&v.forced_ids==verdict.forced_ids&&v.execution_fallback==verdict.execution_fallback&&v.profile_rows==17&&v.profile_window_rows==2&&v.profile_known_rows==9&&v.profile_mask_ms==1&&v.profile_select_ms==2&&v.profile_window_row_ms==3);
  if(cancel&&!local_error)CHECK(v.status=="incomplete"&&v.reason=="cancelled"&&stats.finish=="cancelled");else CHECK(v.status==verdict.status&&v.reason==verdict.reason);
  CHECK(t->error()==(local_error?"refused with receipt":""));bool done=false;CHECK((t->wait(std::chrono::milliseconds(0),done,false)==std::vector<std::uint32_t>{65,66})&&done);
 }
}

static void http_refusals() {
 for(const auto& body:{
  "{\"prompt\":\"A\",\"response_format\":{\"type\":\"json_object\"}}",
  "{\"prompt\":\"A\",\"logprobs\":2}",
  "{\"prompt\":\"A\",\"strix\":{\"thinking_policy\":\"thinking-budget-v1\",\"max_thinking_tokens\":3}}",
  "{\"prompt\":\"A\",\"frequency_penalty\":1}"}) {
  serve::GenerationQueue q;auto config=serve::ApiConfig{};config.default_thinking=false;
  serve::ApiService api(config,serve::ApiTokenizerBinding(std::make_shared<ByteTokenizer>()),q,64,false);Writer w;serve::HttpRequest request;request.method="POST";request.path="/v1/completions";request.body=body;
  if(request.body.find("thinking_policy")!=std::string::npos){request.path="/v1/chat/completions";request.body=R"({"messages":[{"role":"user","content":"A"}],"strix":{"thinking_policy":"thinking-budget-v1","max_thinking_tokens":3}})";}
  std::vector<std::string> events;Fake e(q,0,events);auto owner=std::async(std::launch::async,[&]{execute(q,e);});api.handle(request,w);q.shutdown();owner.get();
  if(w.status!=400)throw std::runtime_error("expected refusal: "+request.body+" returned "+w.body);
  CHECK(q.snapshot().completed==0);auto error=serve::parse_json(w.body);CHECK(error.find("error"));q.shutdown();
 }
}
#endif

int main(int argc,char** argv)try {
 if(argc==2&&std::string(argv[1])=="--failure-probe")CHECK(argc==0);
 serve::GenerationQueue::test_now=clock_submit;
 submitted=std::chrono::steady_clock::now()-std::chrono::milliseconds(120);
 auto all=serve::Json::array();for(unsigned m=0;m<10;++m)for(bool observed:{false,true})all.push(scenario(m,observed));
 for(unsigned m=0;m<5;++m)all.push(bridge(m));for(unsigned m=0;m<3;++m)all.push(http(m));wake_and_throw();
 auto bytes=all.dump()+"\n";
#ifndef SERIAL_ENTRY
 leases_and_expiry();http_refusals();native_control_routes();native_policy_receipts();
#endif
 if(argc==3&&std::string(argv[1])=="--capture"){std::ofstream f(argv[2],std::ios::binary);CHECK(f);f<<bytes;}
 else if(argc==3&&std::string(argv[1])=="--compare"){std::ifstream f(argv[2],std::ios::binary);CHECK(f);std::string expected((std::istreambuf_iterator<char>(f)),{});CHECK(expected==bytes);}
 else std::cout<<"serial owner host checks passed\n";
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
