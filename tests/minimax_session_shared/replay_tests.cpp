// Independent original-source capture and read-only replay of the production
// scheduler and HTTP handler. Capture is absent from the normal executable.
#include "minimax/serve/minimax_api.hpp"
#include <condition_variable>
#include <fstream>
#include <regex>
#include <sstream>
#include <thread>
#include <unordered_map>
struct Scenario;
void minimax_replay_scenario(Scenario&);
#define main minimax_source_policy_tests
#include "lead_fixture.inc"
#undef main
namespace {
using Json = sv::Json;
std::mutex submitted_mutex;
std::condition_variable submitted_cv;
bool submitted = false;
std::size_t submission_hooks = 0;
bool probing_publication = false;
std::vector<int> publication_order;
std::unordered_map<sv::GenerationTicket*,std::vector<std::uint32_t>> published;
Json traces = Json::array();
std::string capture_directory;
void need(bool value, const char* text) { if (!value) throw std::runtime_error(text); }
Json number(std::uint64_t n) { return Json(n); }
std::string read(const std::string& path) { std::ifstream f(path,std::ios::binary); need(bool(f),"cannot read fixture"); return {std::istreambuf_iterator<char>(f),{}}; }
std::string normalize(std::string text) {
    // The sole HTTP normalization: seconds since the epoch are not a policy.
    return std::regex_replace(text,std::regex("\\\"created\\\":[0-9]+"),"\"created\":0");
}
Json ids_json(std::span<const std::uint32_t> ids) { auto a=Json::array(); for(auto id:ids)a.push(number(id)); return a; }
struct Writer final : sv::HttpResponseWriter {
    Json output = Json::array();
    std::vector<std::pair<std::string,std::string>> added;
    void add_response_header(std::string k,std::string v) override { added.emplace_back(std::move(k),std::move(v)); }
    void head(int s,std::string_view type,const std::vector<std::pair<std::string,std::string>>& h) {
        auto o=Json::object();o.set("status",Json(std::int64_t(s)));o.set("type",Json(std::string(type)));
        auto headers=Json::array();for(const auto& [k,v]:h)headers.push(Json(Json::Array{Json(k),Json(v)}));
        for(const auto& [k,v]:added)headers.push(Json(Json::Array{Json(k),Json(v)}));
        o.set("headers",std::move(headers)); output.push(std::move(o));
    }
    void send(int s,std::string_view t,std::string_view b,const std::vector<std::pair<std::string,std::string>>& h) override {
        head(s,t,h);output.push(Json(normalize(std::string(b))));
    }
    bool begin_stream(int s,std::string_view t,const std::vector<std::pair<std::string,std::string>>& h) override {head(s,t,h);return true;}
    bool write(std::string_view data) override {output.push(Json(normalize(std::string(data))));return true;}
    void end_stream() override {output.push(Json("end_stream"));}
    bool client_gone() const override {return false;}
};
Json http_replay() {
    auto result=Json::array();
    const auto requests=sv::parse_json(read(std::string(MM_FIXTURES)+"/requests.json"));
    for(const auto& req:requests.as_array()) {
        Scenario s(ms::Sched::interleave);
        ms::ApiTokenizer tok;tok.eos_id=static_cast<unsigned>(req.find("eos")->as_int());
        tok.encode=[](std::string_view text){return std::vector<std::uint32_t>(text.size(),7);};
        tok.decode=[](std::span<const std::uint32_t> ids){std::string out;for(auto id:ids)out+="<"+std::to_string(id)+">";return out;};
        tok.stream_bytes=[](std::uint32_t id){return "<"+std::to_string(id)+">";};
        ms::MinimaxApiOptions options; options.context_tokens=s.engine.capacity(); options.http_slots=s.engine.slots();
        options.take_extra=[&](const sv::GenerationTicket& t)->std::optional<ml::Extra>{
            std::lock_guard lock(s.shared.m);auto it=s.shared.extras.find(&t);if(it==s.shared.extras.end())return {};
            auto x=it->second;s.shared.extras.erase(it);return x;
        };
        ms::Observer observer({});ms::MinimaxApi api(tok,s.queue,options,observer);
        Writer w;sv::HttpRequest request;request.method="POST";request.path="/v1/completions";request.target=request.path;
        request.body=req.find("body")->dump();request.headers.emplace_back("x-request-id","replay-request");
        {std::lock_guard lock(submitted_mutex);submitted=false;}
        std::exception_ptr failure;
        std::jthread handler([&]{try{api.handle(request,w);}catch(...){failure=std::current_exception();}});
        bool arrived=false;
        {std::unique_lock lock(submitted_mutex);arrived=submitted_cv.wait_for(lock,std::chrono::seconds(5),[]{return submitted;});}
        if(!arrived){s.queue.shutdown();handler.join();if(failure)std::rethrow_exception(failure);
            throw std::runtime_error("HTTP submission timeout");}
        // No arrival race: submission is synchronized before the owner starts.
        // The actual Lead executes its production control/row/stop path.
        (void)s.run(12000);
        handler.join();if(failure)std::rethrow_exception(failure);
        result.push(std::move(w.output));
    }
    sv::GenerationQueue::test_now=std::chrono::steady_clock::now;
    return result;
}
void compare_or_capture(const char* file,const Json& actual) {
    const auto path=std::string(MM_FIXTURES)+"/"+file;
#ifdef MM_CAPTURE_BASELINE
    if(!capture_directory.empty()) {
        std::ofstream f(capture_directory+"/"+file,std::ios::binary);
        f<<actual.dump()<<'\n';need(bool(f),"capture write failed");return;
    }
#endif
    const auto bytes=actual.dump()+"\n";
#ifdef MM_DIAGNOSTICS
    if(bytes!=read(path)) { std::ofstream f(std::string(MM_DIAGNOSTICS)+"/"+file,std::ios::binary);f<<bytes; }
#endif
    need(bytes==read(path),file);
}
}
void minimax_replay_submitted() {std::lock_guard lock(submitted_mutex);++submission_hooks;submitted=true;submitted_cv.notify_all();}
void minimax_replay_tokens(sv::GenerationTicket* t,std::span<const std::uint32_t> ids) {
    if(probing_publication) {
        publication_order.push_back(2);
        bool done=false;
        need(t->wait(std::chrono::milliseconds(0),done).empty(),"observer must precede publication");
    }
    auto& out=published[t];out.insert(out.end(),ids.begin(),ids.end());
}
void minimax_replay_scenario(Scenario& s) {
    auto o=Json::object();o.set("control",Json(sequence({s.engine.controls,s.queue.work})));
    auto tickets=Json::array();
    for(const auto& [id,t]:s.tickets) {
        auto r=Json::object();r.set("id",number(id));r.set("tokens",ids_json(published[t.get()]));
        const auto st=t->stats();auto x=s.shared.extras.find(t.get());
        r.set("timings",ms::reply_timings(st,x==s.shared.extras.end()?ml::Extra{}:x->second));
        r.set("finish",Json(st.finish));r.set("error",Json(t->error()));r.set("cached_tokens",number(st.cached_tokens));
        r.set("prompt_tokens",number(st.prompt_tokens));r.set("generated_tokens",number(st.generated_tokens));
        tickets.push(std::move(r));
    }
    o.set("tickets",std::move(tickets));
    const auto q=s.queue.snapshot();auto totals=Json::object();
    totals.set("queued",number(q.queued));totals.set("running",number(q.running));totals.set("completed",number(q.completed));
    totals.set("cancelled",number(q.cancelled));totals.set("failed",number(q.failed));totals.set("prompt_tokens",number(q.prompt_tokens));
    totals.set("cached_tokens",number(q.cached_tokens));totals.set("generated_tokens",number(q.generated_tokens));
    totals.set("busy_seconds",Json(q.busy_seconds));o.set("queue",std::move(totals));
    auto work=Json::array();for(auto v:{s.queue.work.take_locks,s.queue.work.cancel_visits,s.queue.work.selection_visits,s.queue.work.lookahead_locks,s.queue.work.lookahead_visits})work.push(number(v));
    o.set("queue_work",std::move(work));o.set("syncs",number(s.engine.syncs));o.set("quiesces",number(s.engine.quiesces));
    traces.push(std::move(o));published.clear();
}
int main(int argc,char** argv) {
    try {
#ifdef MM_CAPTURE_BASELINE
        if(argc==3 && std::string(argv[1])=="--capture") capture_directory=argv[2];
        else need(argc==1,"usage: replay [--capture BUILD_OUTPUT_DIR]");
#else
        (void)argv;need(argc==1,"capture is compiled out of this executable");
        // Exercise the actual transformed production queue, including both
        // publication entry paths and every parser/cancellation boolean pair.
        // Always active in Release; no frozen policy/wire assertions change.
        for(bool run_queue:{false,true}) for(bool keep:{false,true}) for(bool cancel:{false,true}) {
            sv::GenerationQueue queue;
            sv::GenerationJob job;
            job.output=std::make_shared<sv::GenerationOutput>();
            std::shared_ptr<sv::GenerationTicket> ticket;
            publication_order.clear();probing_publication=true;
            job.output->consume_tokens=[&](std::span<const std::uint32_t>){
                publication_order.push_back(1);if(cancel)ticket->cancel();return keep;
            };
            const auto before=submission_hooks;
            ticket=queue.submit(std::move(job));
            need(ticket && submission_hooks==before+1,"exactly one submission observer");
            const std::vector<std::uint32_t> ids{17,23};
            bool result=true;
            if(run_queue) {
                struct Probe final:sv::GenerationEngine {
                    const std::vector<std::uint32_t>& ids;bool& result;sv::GenerationQueue& queue;
                    Probe(const std::vector<std::uint32_t>& i,bool& r,sv::GenerationQueue& q):ids(i),result(r),queue(q){}
                    sv::GenerationStats generate(sv::GenerationJob& j) override {
                        result=j.on_tokens(ids);queue.shutdown();return {};
                    }
                    std::uint64_t context_tokens() const override{return 64;}
                } engine(ids,result,queue);
                queue.run(engine);
            } else {
                auto taken=queue.take(false);need(bool(taken),"probe admission");
                result=taken->job.on_tokens(ids);queue.finish(*taken,{});
            }
            need(result==(!cancel && keep),"publication and parser cancellation result");
            need(publication_order==std::vector<int>({1,2}),"parser then exactly one token observer");
            bool done=false;
            need(ticket->wait(std::chrono::milliseconds(0),done)==ids,"publish IDs even on parser stop/cancel");
            need(done && published[ticket.get()]==ids,"exactly one observed publication");
            probing_publication=false;published.clear();
        }
#endif
        const int policy_result = minimax_source_policy_tests();
        if(policy_result!=0) {
            FakeEngine probe;(void)ms::session_hello(probe,false);
            std::array<unsigned char,512> expected{};
            expected[0]=0x4d;expected[1]=0x4d;expected[2]=0x53;expected[3]=0x43;expected[4]=3;
            std::fill_n(expected.begin()+216,8,0xff);
            std::cerr<<"Source hello differing byte offsets:";
            for(std::size_t i=0;i<expected.size();++i)
                if(std::to_integer<unsigned char>(probe.last_wire[i])!=expected[i])std::cerr<<' '<<i;
            std::cerr<<'\n';
        }
        const auto http=http_replay();
        compare_or_capture("legacy_expected.json",http);compare_or_capture("legacy_trace.json",traces);
        std::cout<<"MiniMax frozen traces and HTTP/SSE bytes matched: "<<traces.as_array().size()<<" control scenarios and "<<http.as_array().size()<<" HTTP/SSE replies\n";
        return policy_result;
    }catch(const std::exception& e){std::cerr<<"MiniMax replay failure: "<<e.what()<<'\n';return 1;}
}
