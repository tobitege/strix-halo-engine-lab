// The fixture supplies the actual templated lead/follower, native double and wire.
#define main archived_cpu_main
#include "../glm_serve_cpu_tests.cpp"
#undef main
#include <fstream>
#include <sstream>
#include <iomanip>
#include "strix/serve/api.hpp"

struct WitnessTokenizer final : sv::ApiTokenizer {
 std::vector<std::uint32_t> encode(std::string_view text,bool =true)const override {return {text.begin(),text.end()};}
 std::string decode(std::span<const std::uint32_t> ids,bool =false)const override {std::string s;for(auto id:ids)s+=token_bytes(id);return s;}
 std::string token_bytes(std::uint32_t id)const override {return std::string(1,static_cast<char>('a'+id%26));}
 std::size_t vocab_size()const override{return 200000;}
 std::uint32_t find(std::string_view)const override{return 0;}
};
struct WitnessWriter final : sv::HttpResponseWriter {
 int status=0;std::string bytes;bool disconnect=false,gone=false;
 void send(int s,std::string_view,std::string_view b,const std::vector<std::pair<std::string,std::string>>&)override{status=s;bytes+=b;}
 bool begin_stream(int s,std::string_view,const std::vector<std::pair<std::string,std::string>>&)override{status=s;return true;}
 bool write(std::string_view b)override {bytes+=b;if(disconnect){gone=true;return false;}return true;}
 void end_stream()override{}
 bool client_gone()const override{return gone;}
};
void normalize_http(sv::Json& j) {
 if(j.is_object()) {
  std::erase_if(j.as_object(),[](const auto& item){
   const auto& k=item.first;
   return k=="id"||k=="created"||k=="created_at"||k=="queue_ms"||k=="prefill_ms"||k=="decode_ms"||k=="first_token_ms"||k=="decode_tokens_per_second"||k=="decode_tok_s"||k=="prefill_tok_s"||k=="elapsed_ms";
  });
  for(auto& item:j.as_object())normalize_http(item.second);
 } else if(j.is_array())for(auto& item:j.as_array())normalize_http(item);
}
std::string http_case(const std::string& raw_path) {
 WitnessWriter writers[3];writers[2].disconnect=true;
 FakeEng engine(4);engine.stops=false;engine.delay_us=10000;
 gs::Channel channel({},0);channel.hello(hello_for(4,512,1));
 sv::GenerationQueue queue(8);std::atomic<bool> stop{false};
 gs::LeadConfig cfg;cfg.slots=4;cfg.capacity=512;
 gs::Lead<FakeEng> lead(engine,channel,queue,cfg,stop);
 sv::ApiConfig config;config.model_id="glm-owner-fixture";config.stop_tokens={};
 config.sse_keepalive_seconds=0;config.sse_progress_seconds=0;
 sv::ApiService api(config,sv::ApiTokenizerBinding(std::make_shared<WitnessTokenizer>()),queue,512,false);
 std::thread owner([&]{lead.loop();});
 for(unsigned i=0;i<3;++i) {
  sv::HttpRequest request;request.method="POST";request.path="/v1/completions";
  request.body=std::string("{\"prompt\":\"abc\",\"max_tokens\":4,\"stream\":")+(i?"true}":"false}");
  api.handle(request,writers[i]);CHECK(writers[i].status==200);
 }
 queue.shutdown();owner.join();
 std::ofstream raw(raw_path,std::ios::binary);std::ostringstream out;
 for(unsigned i=0;i<3;++i) {
  raw<<"HTTP "<<i<<' '<<writers[i].status<<'\n'<<writers[i].bytes<<'\n';
  out<<"HTTP "<<i<<' '<<writers[i].status<<'\n';
  if(i==0) {auto json=sv::parse_json(writers[i].bytes);normalize_http(json);out<<json.dump()<<'\n';}
  else {std::istringstream lines(writers[i].bytes);std::string line;
   while(std::getline(lines,line)) {if(line.ends_with('\r'))line.pop_back();
    if(line.starts_with("data: {") ) {auto json=sv::parse_json(line.substr(6));normalize_http(json);out<<"data: "<<json.dump()<<'\n';}
    else if(!line.empty())out<<line<<'\n';
   }
  }
 }
 CHECK(writers[1].bytes.find("[DONE]")!=std::string::npos);CHECK(writers[2].gone);
 auto q=queue.snapshot();CHECK(!q.running&&!q.queued);out<<"HTTP TERMINAL "<<q.completed<<' '<<q.cancelled<<' '<<q.failed<<'\n';
 return out.str();
}

struct WitnessEngine : FakeEng {
 using FakeEng::FakeEng;
 std::function<void(std::uint32_t,std::uint64_t)> prefilled;
 std::function<void()> expected;
 void import_expect(std::uint32_t id,std::span<const std::uint32_t> tokens) {
  FakeEng::import_expect(id,tokens);if(expected)expected();
 }
 bool abandoned=false;
 void owner_abandon() {dec.reset();seed.reset();staged.clear();abandoned=true;}
 std::uint32_t prefill(std::uint32_t s,std::span<const std::uint32_t> t,std::uint64_t p) {
  auto y=FakeEng::prefill(s,t,p); if(prefilled) prefilled(s,p+t.size()); return y;
 }
};
std::string replay_case(unsigned mode, unsigned world) {
 std::ostringstream out;
 Loopback lb; Pipe pipe; WitnessEngine e0(4),e1(4);
 e0.stops=e1.stops=false; e0.pipe=e1.pipe=&pipe; e1.rank=1;
 e1.fail_expect=mode==5; e1.fail_restore=mode==6;
 std::vector<std::string> wire[2];
 auto side=[&](int rank) { auto exchange=lb.side(rank);
  return gs::Channel::Wire([&,rank,exchange](auto send,auto recv,auto tag,auto round) {
   std::ostringstream s; for(auto v:send) s<<std::hex<<std::setw(2)<<std::setfill('0')<<std::to_integer<unsigned>(v);
   wire[rank].push_back(s.str()); exchange(send,recv,tag,round);
  });
 };
 gs::Channel c0(world==2?side(0):gs::Channel::Wire{},0),c1(world==2?side(1):gs::Channel::Wire{},1);
 auto hello=hello_for(4,96,world); if(mode==7||mode==14) hello.prefix_cache_bytes=1<<20;
 std::string follower_error,lead_error;
 std::thread follower;
 if(world==2) follower=std::thread([&]{try {gs::follow(e1,c1,hello,4,96);}catch(const std::exception& e){follower_error=e.what();}});
 c0.hello(hello);
 sv::GenerationQueue queue(16); std::atomic<bool> stop{false};
 gs::LeadConfig cfg;cfg.slots=4;cfg.capacity=96;cfg.batch=mode!=1;cfg.prefill_chunk=3;
 cfg.heartbeat_seconds=3600;cfg.mtp=mode==2||mode==3;cfg.g2=mode==3;cfg.mtp_cycles=2;
 cfg.prefix_cache_bytes=hello.prefix_cache_bytes;
 if((mode>=4&&mode<=6)||mode==15) {cfg.import_min_tokens=2;cfg.rtx_prefill=[](auto,auto){return true;};}
 gs::Lead<WitnessEngine> lead(e0,c0,queue,cfg,stop);
 std::vector<std::shared_ptr<sv::GenerationTicket>> tickets;
 for(unsigned j=0;j<4;++j) {
  sv::GenerationJob job;job.prompt.assign(j==1?17:5+j,1000+j);job.max_tokens=12+j;
  if(mode==14)job.prompt.assign(5+4*j,1000);
  if(mode==8) {if(j==1)job.prompt.clear();if(j==2)job.prompt.resize(96);if(j==3)job.max_tokens=0;}
  if(mode==9) {job.prompt.resize(90);job.max_tokens=20;}
  if(mode==10) {std::uint64_t h=0;for(auto t:job.prompt)h=FakeEng::fold(h,t);job.stop_tokens={e0.tok(h)};}
  if(mode==11&&j==0) {
   job.output=std::make_shared<sv::GenerationOutput>();
   job.output->consume_tokens=[&](auto){tickets.at(1)->cancel();return true;};
  }
  tickets.push_back(queue.submit(std::move(job)));
 }
 if(mode==12) e0.prefilled=[&](auto s,auto){if(s==1)tickets.at(1)->cancel();};
 if(mode==13) tickets.at(2)->cancel();
 if(mode==15)e0.expected=[&]{tickets.at(0)->cancel();};
 if(mode==16)e0.prefilled=[&](auto s,auto){if(s==1)stop.store(true);};
 std::thread owner([&]{try{lead.loop();}catch(const std::exception& e){lead_error=e.what();}catch(...){lead_error="nonstandard";}if(mode==16)queue.shutdown();});
 for(unsigned j=0;j<tickets.size();++j) {
  bool done=false; std::vector<std::uint32_t> ids;
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
  while(!done&&std::chrono::steady_clock::now()<deadline) {
   auto v=tickets[j]->wait(std::chrono::milliseconds(10),done,false);ids.insert(ids.end(),v.begin(),v.end());
  }
  CHECK(done);
  out<<"REQUEST "<<j<<" IDS";for(auto id:ids)out<<' '<<id;out<<'\n';
  auto st=tickets[j]->stats();
  out<<"RECEIPT "<<st.error_status<<' '<<st.error_code<<' '<<st.prompt_tokens<<' '<<st.cached_tokens<<' '<<st.generated_tokens
   <<' '<<st.cycles<<' '<<st.proposed<<' '<<st.accepted<<' '<<st.ngram_rows<<' '<<st.ngram_accepted<<' '<<st.tool_rows<<' '<<st.tool_accepted
   <<' '<<st.source_shadow<<' '<<st.source_scored<<' '<<st.source_right<<' '<<st.frame_cycles<<' '<<st.frame_shadow<<' '<<st.imported_tokens<<' '<<st.finish<<' '<<st.sampling<<' '<<tickets[j]->error()<<'\n';
  for(auto [key,value]:st.engine_extra) {
   auto json=sv::parse_json(value);
   // Only clocks are excluded; route, import count and cache hits remain.
   for(auto clock:{"wait_ms","listen_ms","staged_ms","commit_ms","relay_ms","restore_ms"}) std::erase_if(json.as_object(),[&](auto& v){return v.first==clock;});
   out<<"EXTRA "<<key<<' '<<json.dump()<<'\n';
  }
 }
 queue.shutdown();owner.join();if(follower.joinable())follower.join();
 CHECK(lead_error.empty());CHECK(follower_error.empty());
 const auto q=queue.snapshot();
 out<<"QUEUE "<<q.queued<<' '<<q.running<<' '<<q.completed<<' '<<q.failed<<' '<<q.cancelled<<' '<<q.prompt_tokens<<' '<<q.cached_tokens<<' '<<q.generated_tokens<<'\n';
 for(unsigned rank=0;rank<world;++rank) {
  for(auto& s:rank?e1.log:e0.log)out<<"NATIVE "<<rank<<' '<<s<<'\n';
  for(auto& s:wire[rank])out<<"WIRE "<<rank<<' '<<s<<'\n';
 }
 return out.str();
}

#ifndef GLM_OWNER_ORIGINAL
void mtp_lifetime_case(unsigned mode) {
 WitnessEngine engine(4);engine.stops=false;
 gs::Channel channel({},0);channel.hello(hello_for(4,96,1));
 sv::GenerationQueue queue(8);std::atomic<bool> stop{false};
 gs::LeadConfig cfg;cfg.slots=4;cfg.capacity=96;cfg.mtp=true;cfg.batch=false;cfg.mtp_cycles=1;
 gs::Lead<WitnessEngine> lead(engine,channel,queue,cfg,stop);
 sv::GenerationJob job;job.prompt={1000,1001};job.max_tokens=40;
 job.output=std::make_shared<sv::GenerationOutput>();unsigned published=0;
 job.output->consume_tokens=[&](auto ids) {
  published+=static_cast<unsigned>(ids.size());if(published<3)return true;
  if(mode==0){stop.store(true);return true;}
  if(mode==1)throw std::runtime_error("injected open MTP publication failure");
  throw 19;
 };
 auto active=queue.submit(job);job.output.reset();auto waiting=queue.submit(job);
 bool escaped=false;try{lead.loop();}catch(...){escaped=true;}
 queue.shutdown();CHECK(escaped==(mode!=0));CHECK(engine.abandoned==(mode!=0));
 CHECK(!engine.dec);CHECK(lead.runtime().live()==0);CHECK(queue.snapshot().running==0);
 bool done=false;auto ids=active->wait(std::chrono::milliseconds(0),done,false);CHECK(done);
 CHECK(!active->error().empty());CHECK(active->stats().generated_tokens==published);
 (void)waiting->wait(std::chrono::milliseconds(0),done,false);CHECK(done);
 CHECK(std::any_of(engine.log.begin(),engine.log.end(),[](const auto& s){return s.starts_with("run ");}));
 const bool left=std::any_of(engine.log.begin(),engine.log.end(),[](const auto& s){return s.starts_with("leave ");});
 CHECK(left==(mode==0));
 std::printf("MTP LIFETIME mode=%u published=%u ids",mode,published);for(auto id:ids)std::printf(" %u",id);std::printf("\n");
 for(auto& event:engine.log)std::printf("MTP NATIVE %s\n",event.c_str());
}
void failure_case(bool unknown, bool callback) {
 WitnessEngine engine(4);engine.stops=false;
 gs::Channel channel({},0);channel.hello(hello_for(4,96,1));
 sv::GenerationQueue queue(16);std::atomic<bool> stop{false};
 gs::LeadConfig cfg;cfg.slots=4;cfg.capacity=96;cfg.prefill_chunk=3;
 gs::Lead<WitnessEngine> lead(engine,channel,queue,cfg,stop);
 std::vector<std::shared_ptr<sv::GenerationTicket>> tickets;
 for(unsigned i=0;i<6;++i) {
  sv::GenerationJob j;j.prompt.assign(i?20:5,1000+i);j.max_tokens=40;
  if(callback&&i==1) {j.output=std::make_shared<sv::GenerationOutput>();
   j.output->consume_tokens=[unknown](auto)->bool {if(unknown)throw 17;throw std::runtime_error("injected publication failure");};}
  tickets.push_back(queue.submit(std::move(j)));
 }
 if(!callback)engine.prefilled=[unknown](auto s,auto){if(s==1){if(unknown)throw 17;throw std::runtime_error("injected prefill failure");}};
 bool escaped=false;
 try {lead.loop();}catch(...){escaped=true;}
 CHECK(escaped);CHECK(engine.abandoned);CHECK(!engine.dec);
 for(auto& t:tickets) {bool done=false;(void)t->wait(std::chrono::milliseconds(0),done,false);CHECK(done);CHECK(!t->error().empty());}
 // shutdown() finishes entries that were never taken without adding them to
 // taken-work totals. Preserve that existing queue contract exactly.
 for(unsigned i=2;i<tickets.size();++i) {CHECK(tickets[i]->stats().finish=="cancelled");CHECK(tickets[i]->error()=="server shutting down");}
 const auto q=queue.snapshot();CHECK(q.running==0&&q.queued==0);CHECK(q.failed==2&&q.cancelled==0&&q.completed==0);CHECK(lead.runtime().live()==0);
 std::printf("FAILURE unknown=%u callback=%u running=%llu failed=%llu leases=%zu abandoned=%u\n",unknown,callback,
  (unsigned long long)q.running,(unsigned long long)q.failed,lead.runtime().live(),engine.abandoned);
}

void handle_and_slot_case() {
 WitnessEngine engine(4);engine.stops=false;
 gs::Channel channel({},0);channel.hello(hello_for(4,96,1));
 sv::GenerationQueue queue(8),other_queue(8);std::atomic<bool> stop{false};
 gs::LeadConfig cfg;cfg.slots=4;cfg.capacity=96;
 gs::Lead<WitnessEngine> lead(engine,channel,queue,cfg,stop),other(engine,channel,other_queue,cfg,stop);
 auto& r=lead.runtime();auto& foreign=other.runtime();
 sv::GenerationJob j;j.prompt={1000,1001};j.max_tokens=4;
 auto held=queue.submit(j);CHECK(r.reserve_fifo());const auto submitted=r.reservation()->submitted,started=r.reservation()->started;
 auto h=r.activate(0,true).state;CHECK(h.slot==0);CHECK(r.inspect(h).submitted==submitted&&r.inspect(h).started==started);
 CHECK(throws_with([&]{foreign.notify_admitted(h);},"stale"));
 CHECK(throws_with([&]{foreign.publish(h,{});},"stale"));
 CHECK(throws_with([&]{foreign.publish_progress(h,{});},"stale"));
 CHECK(throws_with([&]{foreign.complete_native(h,{});},"stale"));
 CHECK(foreign.ticket_cancelled(h));
 auto actual=queue.submit(j);
 // Logical 0 remains occupied, while physical slot 0 is still free. Native
 // resource 101 must be used by the real planner for logical lease 1.
 std::thread owner([&]{lead.loop();});bool done=false;std::vector<std::uint32_t> ids;
 while(!done){auto v=actual->wait(std::chrono::milliseconds(10),done,false);ids.insert(ids.end(),v.begin(),v.end());}
 queue.shutdown();owner.join();
 CHECK(ids.size()==4);CHECK(engine.ready.at(0));CHECK(!engine.ready.at(1));CHECK(r.live()==0);
 CHECK(throws_with([&]{r.publish(h,{});},"stale"));
 CHECK(throws_with([&]{r.notify_admitted(h);},"stale"));
 CHECK(throws_with([&]{r.publish_progress(h,{});},"stale"));
 CHECK(throws_with([&]{r.complete_native(h,{});},"stale"));
 CHECK(r.capabilities().owner_control&&r.capabilities().max_slots==4&&r.capabilities().max_context==96);
 CHECK(!r.capabilities().batch_decode&&!r.capabilities().prefix_cache&&!r.capabilities().fork_lanes);
 CHECK(!r.cache_owner());
 std::printf("ISOLATION logical=1 physical_resource=101 native_slot=0 ids=%zu leases=%zu\n",ids.size(),r.live());
 // New owner tests lease reuse separately, before either native planner runs.
 sv::GenerationQueue reuse_queue(8);gs::Lead<WitnessEngine> reuse(engine,channel,reuse_queue,cfg,stop);
 auto& rr=reuse.runtime();(void)reuse_queue.submit(j);CHECK(rr.reserve_fifo());auto old=rr.activate().state;
 rr.complete_native(old,{});(void)reuse_queue.submit(j);CHECK(rr.reserve_fifo());auto fresh=rr.activate().state;
 CHECK(fresh.slot==old.slot&&!(fresh==old));CHECK(throws_with([&]{rr.publish(old,{});},"stale"));
 CHECK(throws_with([&]{rr.notify_admitted(old);},"stale"));CHECK(throws_with([&]{rr.publish_progress(old,{});},"stale"));
 CHECK(throws_with([&]{rr.complete_native(old,{});},"stale"));rr.complete_native(fresh,{});
}
#endif
int main(int argc,char** argv) {
 if(argc==2&&std::string(argv[1])=="--negative") {std::fprintf(stderr,"GLM owner negative sentinel\n");return 43;}
 if(argc>=3&&std::string(argv[1])=="--http") {
  auto result=http_case(argv[2]);std::printf("%s",result.c_str());
  if(argc==4) {std::ifstream f(argv[3],std::ios::binary);std::string old((std::istreambuf_iterator<char>(f)),{});std::erase(old,'\r');CHECK(old==result);}
  return failures?1:0;
 }
 if(argc>=2&&std::string(argv[1])=="--extended") {
  std::ostringstream result;
  for(unsigned world:{1U,2U})for(unsigned mode:{14U,15U,16U}) {
   if(world==1&&mode==15)continue;
   result<<"CASE "<<world<<' '<<mode<<'\n'<<replay_case(mode,world);
  }
  std::printf("%s",result.str().c_str());
  if(argc==3){std::ifstream f(argv[2],std::ios::binary);std::string old((std::istreambuf_iterator<char>(f)),{});std::erase(old,'\r');CHECK(old==result.str());}
  return failures?1:0;
 }
#ifndef GLM_OWNER_ORIGINAL
 if(argc==2&&std::string(argv[1])=="--lifetime") {for(bool unknown:{false,true})for(bool callback:{false,true})failure_case(unknown,callback);for(unsigned mode=0;mode<3;++mode)mtp_lifetime_case(mode);handle_and_slot_case();return failures?1:0;}
#endif
 std::ostringstream all;
 for(unsigned world:{1U,2U})for(unsigned mode=0;mode<14;++mode) {
  // Fake import pairing requires both ranks.
  if(world==1&&mode>=4&&mode<=6)continue;
  all<<"CASE "<<world<<' '<<mode<<'\n'<<replay_case(mode,world);
 }
 std::printf("%s",all.str().c_str());
 if(argc==3&&std::string(argv[1])=="--compare") {
  std::ifstream input(argv[2],std::ios::binary);std::string old((std::istreambuf_iterator<char>(input)),{});
  std::erase(old,'\r');
  // Captures written by the ticket runner contain native LF bytes.
  CHECK(old==all.str());
 }
 return failures?1:0;
}
