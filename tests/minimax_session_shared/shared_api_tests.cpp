// Always-active host tests: actual tokenizer/rendering, ApiService, queue,
// SessionRuntime and native planner adapter, with mutable host target rows.
#define main minimax_policy_fixture_entry
#include "../minimax_lead_tests.cpp"
#undef main
#include "minimax/serve/minimax_api.hpp"
#include "minimax/serve/minimax_tokenizer.hpp"
#include "minimax/serve/minimax_chat.hpp"
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>

namespace {
using Json=sv::Json;
using Headers=std::vector<std::pair<std::string,std::string>>;
void need(bool value,const char* text) { if (!value) throw std::runtime_error(text); }
template<class F> void until(F predicate,const char* why) {
    const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while (!predicate() && std::chrono::steady_clock::now()<end) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    need(predicate(),why);
}
Json& member(Json& object,std::string_view key) {
    for(auto& [name,value]:object.as_object())if(name==key)return value;
    throw std::runtime_error("missing synthetic tokenizer member");
}
std::shared_ptr<const ms::MinimaxTokenizer> tokenizer() {
    auto root=sv::parse_json(R"({"normalizer":{"type":"NFC"},"decoder":{"type":"ByteLevel"},"model":{"type":"BPE","vocab":{},"merges":[]},"pre_tokenizer":{"pretokenizers":[{"type":"Split","pattern":{},"behavior":"Removed","invert":true},{"type":"ByteLevel","use_regex":false,"add_prefix_space":false}]},"added_tokens":[]})");
    const std::string pattern=
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
        "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
        "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    member(member(member(root,"pre_tokenizer"),"pretokenizers").as_array()[0],"pattern").set("Regex",Json(pattern));
    auto& vocab=member(member(root,"model"),"vocab");
    unsigned extra=0;
    for (unsigned b=0;b<256;++b) {
        const unsigned cp=(b>=33 && b<=126) || (b>=161 && b<=172) || b>=174 ? b : 256+extra++;
        std::string text;
        if (cp<128) text.push_back(static_cast<char>(cp));
        else { text.push_back(static_cast<char>(0xc0U|(cp>>6U))); text.push_back(static_cast<char>(0x80U|(cp&63U))); }
        vocab.set(text,Json(static_cast<std::uint64_t>(b)));
    }
    unsigned id=256;
    for (const char* text:{"]~b]","[e~[","]!p~[", "[im_end]", "[im_start]", "<mm:think>","</mm:think>"}) {
        auto token=Json::object();token.set("id",Json(static_cast<std::uint64_t>(id++)));token.set("content",Json(text));
        token.set("special",Json(text[0]!='<'));member(root,"added_tokens").push(std::move(token));
    }
    const auto path=std::filesystem::current_path()/"minimax-shared-api-tokenizer.json";
    { std::ofstream file(path,std::ios::binary); file << root.dump(); need(bool(file),"write tokenizer fixture"); }
    return ms::MinimaxTokenizer::load(path.string());
}
struct Writer final : sv::HttpResponseWriter {
    mutable std::mutex mutex;
    int status=0;std::string body;Headers headers;
    std::vector<std::string> events;
    std::atomic<bool> gone{false};bool fail_begin=false;unsigned fail_after=~0U,ends=0;
    void add_response_header(std::string k,std::string v) override { headers.emplace_back(std::move(k),std::move(v)); }
    void send(int code,std::string_view,std::string_view text,const Headers&) override {status=code;body=text;}
    bool begin_stream(int code,std::string_view,const Headers&) override {status=code;return !fail_begin;}
    bool write(std::string_view text) override {
        std::lock_guard lock(mutex);
        if (gone || events.size()>=fail_after) { gone=true;return false; }
        events.emplace_back(text);return true;
    }
    void end_stream() override {++ends;}
    bool client_gone() const override {return gone;}
    bool contains(std::string_view text) const {std::lock_guard lock(mutex);for(const auto& e:events)if(e.find(text)!=std::string::npos)return true;return false;}
};
sv::HttpRequest request(std::string path,std::string body,std::string id={}) {
    sv::HttpRequest r;r.method="POST";r.path=std::move(path);r.target=r.path;r.body=std::move(body);
    if(!id.empty())r.headers.emplace_back("x-request-id",std::move(id));return r;
}
struct Target : FakeEngine {
    mutable std::mutex mutex;std::condition_variable cv;
    std::vector<unsigned> script;std::vector<std::vector<unsigned>> prompts;
    std::array<std::vector<unsigned>,8> rows;std::array<unsigned,8> cursor{};
    std::atomic<unsigned> calls{0};std::atomic<bool> paused{false};bool hold=false;
    unsigned slots() const {return 1;}
    void prefill_begin(unsigned slot,std::span<const unsigned> prompt,bool context,std::uint64_t id,std::uint64_t parent,std::uint64_t generation) {
        ++calls;{std::lock_guard lock(mutex);prompts.emplace_back(prompt.begin(),prompt.end());rows[slot]=prompts.back();cursor[slot]=0;}
        FakeEngine::prefill_begin(slot,prompt,context,id,parent,generation);
    }
    bool prefill_unit(unsigned layers,unsigned* next) {
        const bool finished=FakeEngine::prefill_unit(layers,next);
        if(finished){*next=script.front();active.next=*next;}return finished;
    }
    std::vector<unsigned> step(std::span<const mm::Row> input,unsigned) {
        std::unique_lock lock(mutex);paused=true;cv.notify_all();cv.wait(lock,[&]{return !hold;});paused=false;
        std::vector<unsigned> output;
        for(auto row:input){rows[row.slot].resize(static_cast<std::size_t>(row.pos+1));rows[row.slot][row.pos]=row.token;
            output.push_back(script[std::min<std::size_t>(++cursor[row.slot],script.size()-1)]);}
        return output;
    }
    void release(){std::lock_guard lock(mutex);hold=false;cv.notify_all();}
    void prepare(std::vector<unsigned> ids,bool pause=false){std::lock_guard lock(mutex);script=std::move(ids);hold=pause;}
    auto last_prompt(){std::lock_guard lock(mutex);return prompts.back();}
};
struct Fixture {
    std::shared_ptr<const ms::MinimaxTokenizer> tok=tokenizer();
    Target target;sv::GenerationQueue queue{1};ml::Shared shared;std::atomic<bool> stop{false};
    ms::Observer observer{ms::ObserverInfo{}};std::atomic<unsigned> takes{0},encodes{0},decodes{0};
    std::unique_ptr<ml::Lead<Target,Target>> lead;std::unique_ptr<ms::MinimaxApi> api;std::jthread owner;std::exception_ptr error;std::mutex error_mutex;
    Fixture() {
        FakeClock::advance=[](double){};sv::GenerationQueue::test_now=std::chrono::steady_clock::now;
        shared.policy.sched=ms::Sched::off;shared.policy.lookup=false;shared.policy.nest=false;
        lead=std::make_unique<ml::Lead<Target,Target>>(target,target,queue,shared,4096,false,stop);
        ms::MinimaxApiOptions options;options.context_tokens=16384;options.default_max_tokens=32;options.http_slots=1;
        options.take_extra=[this](const sv::GenerationTicket& ticket)->std::optional<ms::RequestExtra>{
            ++takes;std::lock_guard lock(shared.m);auto it=shared.extras.find(&ticket);if(it==shared.extras.end())return {};
            auto extra=it->second;shared.extras.erase(it);return extra;
        };
        options.admin_sched=[this](const sv::HttpRequest&){std::lock_guard lock(shared.m);auto j=Json::object();j.set("sched",Json(shared.policy.sched==ms::Sched::off ? "off":"interleave"));return j;};
        auto binding=ms::api_tokenizer(tok);auto encode=binding.encode;
        binding.encode=[this,encode](std::string_view text){++encodes;return encode(text);};
        auto decode=binding.decode;binding.decode=[this,decode](std::span<const std::uint32_t> ids){++decodes;return decode(ids);};
        api=std::make_unique<ms::MinimaxApi>(std::move(binding),queue,std::move(options),observer);
        target.prepare(tok->encode("ok[e~[",true));
        owner=std::jthread([this]{try{lead->loop();}catch(...){std::lock_guard lock(error_mutex);error=std::current_exception();queue.shutdown();}});
    }
    ~Fixture(){target.release();stop=true;queue.shutdown();if(owner.joinable())owner.join();}
    void run(std::string path,std::string body,Writer& writer,std::string id={}){api->handle(request(std::move(path),std::move(body),std::move(id)),writer);{std::lock_guard lock(error_mutex);if(error)std::rethrow_exception(error);}}
};
void prompt_routes() {
    sv::observe_disarm();Fixture f;
    const auto body=R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":32,"return_token_ids":true})";
    Writer chat;f.run("/v1/chat/completions",body,chat);need(chat.status==200,"chat accepted");
    const auto prompt=f.target.last_prompt();
    need(prompt==f.tok->encode(ms::render_minimax_chat(*sv::parse_json(body).find("messages")),true),"renderer/tokenizer independent expected prompt");
    Writer messages;f.run("/v1/messages",body,messages);need(messages.status==200 && f.target.last_prompt()==prompt,"messages prompt equality");
    Writer responses;f.run("/v1/responses",R"({"input":"hi","max_output_tokens":32,"return_token_ids":true})",responses);
    need(responses.status==200 && f.target.last_prompt()==prompt,"responses prompt equality");
    for (const auto* writer:{&chat,&messages,&responses}) {
        std::string id;for(const auto& [key,value]:writer->headers)if(key=="X-Request-Id")id=value;
        const auto body=sv::parse_json(writer->body);
        need(id==body.find("id")->as_string(),"all generation routes echo their completion identity");
        need(body.find("timings") && body.find("strix")->find("token_ids")->as_array().back().as_int()==f.tok->eos_id(),"all generation routes retain family timings and EOS IDs");
    }
    auto reply=sv::parse_json(chat.body);need(reply.find("usage")->find("completion_tokens")->as_int()==3,"EOS included in usage");
    need(reply.find("strix")->find("token_ids")->as_array().back().as_int()==f.tok->eos_id(),"EOS included in IDs");
    need(f.takes==3 && f.encodes==3 && f.decodes==3,"one render/encode, callback decode and receipt per route");
    const auto delegated=ms::api_tokenizer(f.tok).delegated;
    need(delegated->find("[e~[")==f.tok->eos_id() && delegated->vocab_size()==f.tok->vocab_size(),"exact delegated IDs and vocabulary");
    need(delegated->type(f.tok->eos_id())==sv::ApiTokenizer::TokenType::control,"exact special type");
    need(delegated->encode("[e~[",false)==f.tok->encode("[e~[",false),"special parsing delegated");
    Writer tokens;f.run("/tokenize",R"({"text":"hi[e~["})",tokens);
    need(tokens.status==200 && sv::parse_json(tokens.body).find("tokens")->size()==3,"legacy text tokenize");
    Writer decode;f.run("/detokenize",R"({"tokens":[104,105,257]})",decode);
    need(decode.status==200 && sv::parse_json(decode.body).find("text")->as_string()=="hi[e~[","exact detokenizer skip policy");
    Writer tokenize_chat;f.run("/tokenize",body,tokenize_chat);need(sv::parse_json(tokenize_chat.body).find("tokens")->size()==prompt.size(),"chat tokenize once");
    const auto calls=f.target.calls.load();
    for(const char* field:{R"("stop":"END")",R"("logprobs":true)",R"("thinking_budget":2)",R"("ngram_draft":true)",R"("frequency_penalty":1)",R"("response_format":{"type":"json_object"})"}) {
        Writer refused;f.run("/v1/chat/completions",std::string(R"({"messages":[{"role":"user","content":"hi"}],)")+field+"}",refused);
        need(refused.status==400,"unsupported field refused by name");need(sv::parse_json(refused.body).find("error")->find("param"),"refusal names field");
    }
    Writer budget;f.run("/v1/messages",R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":20,"thinking":{"type":"enabled","budget_tokens":1}})",budget);
    need(budget.status==400 && f.target.calls==calls && f.takes==3,"validation performs no owner work and takes no receipts");
    Writer response_budget;f.run("/v1/responses",R"({"input":"hi","reasoning":{"budget_tokens":1}})",response_budget);
    need(response_budget.status==400 && response_budget.body.find("reasoning.budget_tokens")!=std::string::npos,"unknown Responses reasoning policy refused by name");
    Writer invalid_thinking;f.run("/v1/messages",R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":20,"thinking":{"type":1}})",invalid_thinking);
    need(invalid_thinking.status==400 && invalid_thinking.body.find("thinking.type")!=std::string::npos && f.target.calls==calls && f.takes==3,"invalid thinking mode refuses by name before owner work");
    f.target.prepare(f.tok->encode("<mm:think>why</mm:think> answer[e~[",true));Writer reasoning;
    f.run("/v1/chat/completions",body,reasoning);const auto reasoned=sv::parse_json(reasoning.body);const auto message=reasoned.find("choices")->as_array()[0].find("message");
    need(message->find("reasoning_content")->as_string()=="why" && message->find("content")->as_string()=="answer","family final reasoning split");
    Writer reasoned_messages;f.run("/v1/messages",body,reasoned_messages);
    const auto blocks=sv::parse_json(reasoned_messages.body);
    need(blocks.find("content")->as_array()[0].find("thinking")->as_string()=="why","normalized messages retain parsed family reasoning");
    std::vector<unsigned> tool_prompt;
    for(unsigned route=0;route<3;++route) {
        const auto text=route==0 ? R"({"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","description":"test","parameters":{"type":"object"}}}],"chat_template_kwargs":{"thinking_mode":"adaptive","reasoning_effort":"high"},"reasoning_effort":"medium","max_tokens":32})" :
            route==1 ? R"({"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"f","description":"test","input_schema":{"type":"object"}}],"chat_template_kwargs":{"thinking_mode":"adaptive","reasoning_effort":"high"},"reasoning_effort":"medium","max_tokens":32})" :
            R"({"input":"hi","tools":[{"type":"function","name":"f","description":"test","parameters":{"type":"object"}}],"chat_template_kwargs":{"thinking_mode":"adaptive","reasoning_effort":"high"},"reasoning_effort":"medium","max_output_tokens":32})";
        const auto before=f.encodes.load();Writer writer;
        f.run(route==0 ? "/v1/chat/completions" : route==1 ? "/v1/messages" : "/v1/responses",text,writer);
        need(writer.status==200 && f.encodes==before+1,"normalized tools and template controls encode once");
        if(route==0)tool_prompt=f.target.last_prompt();else need(tool_prompt==f.target.last_prompt(),"tools and extension fields preserved through normalization");
    }
    f.target.prepare(f.tok->encode("ok[e~[",true));
    for(const auto& [path,text]:std::vector<std::pair<std::string,std::string>>{
        {"/v1/messages",R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":32,"stream":true})"},
        {"/v1/responses",R"({"input":"hi","max_output_tokens":32,"stream":true})"}}) {
        Writer writer;f.run(path,text,writer);
        need(writer.status==200 && writer.ends==1 && writer.contains(path=="/v1/messages" ? "message_stop" : "response.completed"),"new routes share SSE terminal lifecycle");
    }
    const auto before_raw=f.encodes.load();Writer raw;
    f.run("/v1/completions",R"({"prompt":"hi"})",raw);
    need(raw.status==200 && f.encodes==before_raw+1,"legacy encode callback remains active with delegated metadata");

}
void live_and_terminal() {
    sv::observe_disarm();sv::observe_arm();Fixture f;
    f.target.prepare(f.tok->encode("ok[e~[",true),true);Writer live;
    auto request=std::async(std::launch::async,[&]{f.run("/v1/chat/completions",R"({"messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true},"return_token_ids":true})",live,"live-request");});
    try { until([&]{return f.target.paused.load() && live.contains("\"content\":\"o\"");},"live content before terminal while owner paused"); }
    catch(...) { f.target.release();request.wait();throw; }
    need(f.takes==0,"no terminal receipt while owner paused");f.target.release();request.get();
    need(f.takes==1 && live.contains("[DONE]") && live.contains("\"completion_tokens\":3") && live.ends==1,"live terminal IDs usage receipt");
    for(unsigned mode=0;mode<3;++mode) {
        Writer failure;if(mode==0)failure.fail_begin=true;else if(mode==1)failure.fail_after=0;else failure.gone=true;
        f.run("/v1/completions",R"({"prompt":"hi","stream":true})",failure);
        need(f.takes==mode+2,"once-only cleanup after begin/write/disconnect");
        need(!failure.contains("[DONE]") && failure.ends==0,"no writes or terminator after delivery failure");
    }
    {std::lock_guard lock(f.shared.m);need(f.shared.extras.empty(),"owner receipt table fully drained");}
    need(f.observer.status(f.queue.snapshot(),false,true).find("requests")->size()==4,"one observation per accepted request");
    sv::observe_disarm();
    f.target.prepare(f.tok->encode("ok[e~[",true),true);Writer held;
    auto first=std::async(std::launch::async,[&]{f.run("/v1/completions",R"({"prompt":"hi","stream":true})",held,"unobserved-request");});
    until([&]{return f.target.paused.load();},"owner paused for queue exhaustion");
    sv::observe_arm();
    Writer queued;queued.gone=true;
    auto second=std::async(std::launch::async,[&]{f.run("/v1/completions",R"({"prompt":"hi","stream":true})",queued,"queued-request");});
    until([&]{return f.queue.snapshot().queued==1;},"second request queued");
    Writer refused;f.run("/v1/completions",R"({"prompt":"hi"})",refused);need(refused.status==503,"full queue refusal");
    need(f.takes==4,"queue rejection creates no terminal receipt");f.target.release();first.get();second.get();
    need(f.takes==6,"queued cancellation drained once");
    const auto status=f.observer.status(f.queue.snapshot(),false,true);
    bool queued_record=false;
    for(const auto& record:status.find("requests")->as_array()) {
        need(record.find("id")->as_string()!="unobserved-request","unobserved concurrent request has no ring record");
        if(record.find("id")->as_string()=="queued-request") {queued_record=true;need(record.find("output_tokens")->as_int()==0,"queued cancellation executes zero rows");}
    }
    need(queued_record,"observed concurrent request recorded once");
    Writer reused;f.run("/v1/completions",R"({"prompt":"hi"})",reused);need(reused.status==200 && f.takes==7,"queue and physical slot reusable");
}
void surface_and_validation() {
    sv::observe_disarm();Fixture f;
    const auto get=[&](const std::string& path,const std::string& query={}) {
        Writer writer;auto req=request(path,"");req.method="GET";req.query=query;f.api->handle(req,writer);
        need(writer.status==200,"service route accepted");return writer.body;
    };
    need(get("/health")==R"({"status":"ok"})","frozen health bytes");
    need(sv::parse_json(get("/v1/models")).find("data")->as_array()[0].find("context_length")->as_int()==16384,"model context");
    const auto caps=sv::parse_json(get("/v1/capabilities"));
    need(!caps.find("logprobs")->as_bool() && !caps.find("json_schema")->as_bool() && !caps.find("thinking_budget")->as_bool(),"no advanced policy advertised");
    need(!sv::observe_armed(),"metadata routes do not arm");
    need(!sv::parse_json(get("/api/status","notdetail=true")).find("armed")->as_bool(),"only exact detail query arms");
    need(sv::parse_json(get("/api/status")).find("capacity")->find("slots_total")->as_int()==1,"legacy slot capacity without new generation headers");
    need(get("/metrics").find("mm_")!=std::string::npos && sv::observe_armed(),"metrics arm and legacy counters");
    need(sv::parse_json(get("/admin/sched")).find("sched")->as_string()=="off","same scheduler domain owner");
    ms::MinimaxApiOptions options;options.session_enabled=true;options.context_tokens=16384;
    ms::MinimaxApi session_api(ms::api_tokenizer(f.tok),f.queue,options,f.observer);
    for(const char* value:{"null","2","\"\"","\"bad/id\""}) {
        Writer writer;session_api.handle(request("/v1/completions",std::string(R"({"prompt":"hi","session":)")+value+"}"),writer);
        need(writer.status==400 && sv::parse_json(writer.body).find("error")->find("code")->as_string()=="invalid_request","native session validation before work");
    }
    need(f.target.calls==0 && f.takes==0,"route/validation operations do not invent scheduler work");
    // Callback-only callers remain source-compatible and fail only metadata work.
    auto callbacks=ms::api_tokenizer(f.tok);callbacks.delegated.reset();
    ms::MinimaxApi fake_api(callbacks,f.queue,options,f.observer);
    Writer missing;fake_api.handle(request("/detokenize",R"({"tokens":[1]})"),missing);
    need(missing.status==400 && f.target.calls==0,"missing optional metadata refused before queue");
    Writer drain;f.run("/admin/drain","",drain);need(drain.status==200,"shared drain installed");
    Writer health;auto req=request("/health","");req.method="GET";f.api->handle(req,health);need(health.status==503,"draining health");
    Writer refused;f.run("/v1/completions",R"({"prompt":"hi"})",refused);need(refused.status==503 && f.takes==0,"drain refuses with no scheduler receipt");
}

void callback_requirements() {
    sv::observe_disarm();Fixture f;
    auto callbacks=ms::api_tokenizer(f.tok);
    callbacks.delegated.reset();callbacks.stream_bytes={};
    ms::MinimaxApiOptions options;options.context_tokens=16384;
    options.take_extra=[&](const sv::GenerationTicket& ticket)->std::optional<ms::RequestExtra> {
        ++f.takes;std::lock_guard lock(f.shared.m);
        auto it=f.shared.extras.find(&ticket);if(it==f.shared.extras.end())return {};
        auto extra=it->second;f.shared.extras.erase(it);return extra;
    };
    ms::MinimaxApi api(std::move(callbacks),f.queue,std::move(options),f.observer);
    for(const char* path:{"/v1/completions","/v1/chat/completions"}) {
        Writer reply;
        const auto body=std::string(path)=="/v1/completions" ? R"({"prompt":"hi"})" : R"({"messages":[{"role":"user","content":"hi"}]})";
        api.handle(request(path,body),reply);
        const auto parsed=sv::parse_json(reply.body);
        need(reply.status==200 && parsed.find("usage")->find("completion_tokens")->as_int()==3,"nonstream callbacks need only encode and decode");
    }
    need(f.target.calls==2 && f.takes==2,"nonstream callback jobs drain exactly once");
    Writer stream;api.handle(request("/v1/completions",R"({"prompt":"hi","stream":true})"),stream);
    need(stream.status==400 && f.target.calls==2 && f.takes==2,"missing stream callback refused before queue work");
    {std::lock_guard lock(f.shared.m);need(f.shared.extras.empty(),"callback-only jobs leave no terminal receipt");}
}

}
int main() try {
    prompt_routes();live_and_terminal();surface_and_validation();callback_requirements();
    std::cout << "PASS shared frontend production owner/tokenizer routes and terminal lifecycle\n";return 0;
} catch(const std::exception& error){std::cerr << error.what() << '\n';return 1;}
