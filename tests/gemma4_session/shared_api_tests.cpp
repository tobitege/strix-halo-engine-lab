// Reuse tiny loaded weights and the independent serial oracle, never a copied
// frontend or a mock queue. Every request below enters production ApiService.
#define STRIX_G4_SHARED_API_TEST_SUPPORT
#include "turn_cache_integration_tests.cpp"
#include "strix/serve/api.hpp"
#include "gemma4/features/features.hpp"
#include <thread>

namespace {
Writer route(e::Api& api,const char* path,const Json& b=Json::object(),const char* method="POST") {
    srv::HttpRequest r;r.method=method;r.path=path;r.body=b.dump();Writer w;api.handle(r,w);return w;
}
Json terminal(const Writer& w) {
    CHECK(w.status==200);
    if(w.bytes.starts_with("data: ")) {
        Json final;
        for(std::size_t at=0;at<w.bytes.size();) {
            const auto end=w.bytes.find("\n\n",at);CHECK(end!=std::string::npos);
            const auto line=w.bytes.substr(at+6,end-at-6);at=end+2;
            if(line=="[DONE]")continue;
            auto j=srv::parse_json_unique(line);
            if(const auto* receipt=j.find("strix");receipt && receipt->find("generated_ids"))final=j;
        }
        CHECK(final.find("strix"));return final;
    }
    if(!w.bytes.starts_with("event:"))return response(w);
    Json last;
    for(std::size_t at=0;at<w.bytes.size();) {
        const auto end=w.bytes.find("\n\n",at);CHECK(end!=std::string::npos);
        const auto event=w.bytes.substr(at,end-at);at=end+2;
        const auto data=event.find("\ndata: ");CHECK(data!=std::string::npos);
        auto j=srv::parse_json_unique(event.substr(data+7));
        if(const auto* r=j.find("response");r && r->find("strix"))last=*r;
        if(j.find("strix"))last=j;
    }
    CHECK(last.find("strix"));return last;
}
Json converted(const Json& b,const std::string& path) {
    if(path=="/v1/chat/completions")return b;
    auto out=Json::object();
    for(const auto* key:{"max_tokens","temperature","seed","dflash","stream"})out.set(key,*b.find(key));
    if(path=="/v1/messages")out.set("messages",*b.find("messages"));
    else {
        Json normalized=Json::object();
        for(const auto& [key,value]:out.as_object())if(key!="max_tokens")normalized.set(key,value);
        normalized.set("max_output_tokens",*b.find("max_tokens"));normalized.set("input",*b.find("messages"));out=std::move(normalized);
    }
    return out;
}
void routes(const g::ModelManifest& m,g::RopeTables rope,std::shared_ptr<const g::serve::Tokenizer> tok) {
    g::serve::Adapter adapter(tok);auto b=body();const auto expected=prompt(b,adapter);
    std::vector<std::uint32_t> ids;
    for(const auto* path:{"/v1/chat/completions","/v1/messages","/v1/responses"})for(bool stream:{false,true}) {
        std::cout<<"route "<<path<<" stream="<<stream<<std::endl;
        RealBackend backend(m,rope);e::Api api(backend,tok,{},e::DraftMode::off,"model");
        auto input=body(stream);auto request=converted(input,path);
        const auto w=route(api,path,request);const auto j=terminal(w);
        CHECK(backend.log.rows.size()>=expected.size());
        for(std::size_t i=0;i<expected.size();++i)CHECK(backend.log.rows[i].position==i && backend.log.rows[i].token==expected[i]);
        if(ids.empty())ids=generated(j);else CHECK(ids==generated(j));
        CHECK(j.find("strix")->find("phases")->size()==1);
        if(stream) {
            if(std::string(path)=="/v1/chat/completions")CHECK(w.bytes.ends_with("data: [DONE]\n\n"));
            else CHECK(w.bytes.find(std::string(path)=="/v1/messages"?"event: message_stop":"event: response.")!=std::string::npos);
        }
        const auto status=srv::parse_json_unique(route(api,"/api/status",{},"GET").bytes);
        CHECK(status.find("queue")->find("completed")->as_int()==1 && status.find("queue")->find("running")->as_int()==0);
        for(const auto* get:{"/health","/v1/models","/v1/capabilities","/metrics"})CHECK(route(api,get,{},"GET").status==200);
        auto tokenized=route(api,"/tokenize",b);CHECK(tokenized.status==200);
        const auto tokens=srv::parse_json_unique(tokenized.bytes);
        CHECK(tokens.find("count")->as_int()==std::int64_t(expected.size()));
        for(std::size_t i=0;i<expected.size();++i)CHECK(tokens.find("tokens")->as_array()[i].as_int()==expected[i]);
        auto detokenized=route(api,"/detokenize",Json(Json::Object{{"tokens",*tokens.find("tokens")}}));
        CHECK(detokenized.status==200 && srv::parse_json_unique(detokenized.bytes).find("text")->as_string()==tok->decode(expected));
    }
    RealBackend backend(m,rope);e::Api api(backend,tok,{},e::DraftMode::off,"model");
    for(bool stream:{false,true}) {
        Json raw=Json::array();for(auto id:expected)raw.push(std::uint64_t(id));
        auto request=Json(Json::Object{{"prompt",raw},{"max_tokens",3},{"temperature",1.0},{"seed",19},{"stream",stream}});
        const auto j=terminal(route(api,"/v1/completions",request));CHECK(generated(j)==ids);
        CHECK(j.find("usage")->find("completion_tokens")->as_int()==std::int64_t(ids.size()));
    }
    for(const double temperature:{3.5,1e-50}) {
        auto request=body();request.set("temperature",temperature);
        const auto actual=generated(response(http(api,request)));
        e::Sampling sampling;sampling.temperature=float(temperature);sampling.seed=19;
        Reference reference(m,rope,sampling);reference.append(expected);
        std::vector<std::uint32_t> wanted;
        for(std::size_t i=0;i<3;++i){const auto id=reference.decode();wanted.push_back(id);if(id==1 || id==106 || id==110)break;}
        CHECK(actual==wanted);
    }
    const auto shared=g::serve::api_tokenizer(tok);
    CHECK(shared->encode("<bos> hi")==tok->encode("<bos> hi"));
    CHECK(shared->encode("<bos> hi",false)==tok->encode("<bos> hi",false));
    CHECK(shared->eos_token()==tok->find("<eos>"));
    CHECK(shared->find("missing-token")==UINT32_MAX);
    bool refused=false;try{(void)shared->type(1);}catch(const std::invalid_argument&){refused=true;}CHECK(refused);
    const auto j=response(http(api,b));const auto content=j.find("choices")->as_array()[0].find("message")->find("content")->as_string();
    CHECK(!content.empty());auto stopped=body();stopped.set("stop",content.substr(0,1));
    const auto stop=response(http(api,stopped));CHECK(stop.find("choices")->as_array()[0].find("finish_reason")->as_string()=="stop");
    CHECK(stop.find("choices")->as_array()[0].find("message")->find("content")->as_string().empty());
    CHECK(stop.find("usage")->find("completion_tokens")->as_int()==std::int64_t(generated(stop).size()));
}
void invalid(const g::ModelManifest& m,g::RopeTables rope,std::shared_ptr<const g::serve::Tokenizer> tok) {
    RealBackend backend(m,rope);e::Api api(backend,tok,{},e::DraftMode::off,"model");
    for(const auto* path:{"/v1/chat/completions","/v1/messages","/v1/responses"}) {
        for(const auto& field:std::vector<std::pair<std::string,Json>>{{"logprobs",true},{"min_p",0.1},{"presence_penalty",1},
            {"response_format",Json(Json::Object{{"type","json_object"}})},{"ngram_draft",true},{"session","public"},
            {"dflash","invalid"},{"temperature",-1},{"stream",1},{"unknown_policy",true},{"max_tokens",1.5}}) {
            auto request=converted(body(),path);request.set(field.first,field.second);
            const auto before=backend.log.calls;const auto w=route(api,path,request);
            CHECK(w.status==400 && w.bytes.find(field.first)!=std::string::npos && backend.log.calls==before);
        }
    }
    srv::HttpRequest duplicate;duplicate.method="POST";duplicate.path="/v1/chat/completions";
    duplicate.body=R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":2,"max_tokens":3})";
    Writer duplicate_reply;api.handle(duplicate,duplicate_reply);CHECK(duplicate_reply.status==400 && backend.log.calls==0);
    const auto status=srv::parse_json_unique(route(api,"/api/status",{},"GET").bytes);
    CHECK(status.find("queue")->find("completed")->as_int()==0);
    auto request=converted(body(),"/v1/messages");request.set("thinking",Json(Json::Object{{"type","enabled"},{"budget_tokens",2}}));
    CHECK(route(api,"/v1/messages",request).status==400 && backend.log.calls==0);
    auto text=converted(body(),"/v1/responses");text.set("text",Json(Json::Object{{"format",Json(Json::Object{{"type","json_object"}})}}));
    CHECK(route(api,"/v1/responses",text).status==400 && backend.log.calls==0);
    CHECK(route(api,"/v1/gemma4/features",Json::object()).status==400 && backend.log.calls==0);
}
// One backend lock must serialize feature export with a live generation.
class FeatureBackend final : public e::Backend {
public:
    FeatureBackend(const g::ModelManifest& m,g::RopeTables rope):host(m,rope,1),identity_(g::features::freeze(m)){identity_.rope=host.identity().find("rope_hash")->as_string();}
    const g::ModelConfig& config()const override{return host.config();}
    Json identity()const override{auto out=host.identity();out.set("feature_identity",identity_.json());return out;}
    std::unique_ptr<e::StreamState> create_stream()override{return host.create_stream();}
    std::unique_ptr<e::StreamState> create_stream(std::uint32_t n)override{return static_cast<e::Backend&>(host).create_stream(n);}
    g::ForwardResult forward_rows(std::span<const g::RowPlan> rows)override {
        CHECK(inside.fetch_add(1)==0);
        struct Exit {std::atomic<int>& n;~Exit(){--n;}} exit{inside};
        {std::unique_lock lock(mutex);
            if(!entered){entered=true;changed.notify_all();changed.wait(lock,[&]{return released;});}}
        return host.forward_rows(rows);
    }
    e::HostBackend host;g::features::Identity identity_;
    std::mutex mutex;std::condition_variable changed;bool entered=false,released=false;std::atomic<int> inside=0;
};
void features_with_live_work(const g::ModelManifest& m,g::RopeTables rope,std::shared_ptr<const g::serve::Tokenizer> tok) {
    FeatureBackend backend(m,rope);e::Api api(backend,tok,{},e::DraftMode::off,"model");
    Writer chat,feature;std::exception_ptr chat_error,feature_error;std::atomic<bool> feature_started=false;
    const auto request=Json(Json::Object{{"identity",backend.identity_.json()},{"prefix_ids",Json(Json::Array{10,11})},
        {"branches",Json(Json::Array{Json(Json::Object{{"id","branch"},{"token_ids",Json(Json::Array{12,13})},
            {"read_position",1},{"option_close_positions",Json::array()}})})},{"code_ids",Json(Json::Array{10})},{"chunk_rows",1}});
    std::thread generation([&]{try{chat=http(api,body());}catch(...){chat_error=std::current_exception();}});
    {
        std::unique_lock lock(backend.mutex);
        if(!backend.changed.wait_for(lock,std::chrono::seconds(15),[&]{return backend.entered;})) {
            backend.released=true;lock.unlock();backend.changed.notify_all();generation.join();throw std::runtime_error("generation rendezvous failed");
        }
    }
    std::thread exporting([&]{feature_started=true;try{feature=route(api,"/v1/gemma4/features",request);}catch(...){feature_error=std::current_exception();}});
    while(!feature_started)std::this_thread::yield();
    {std::lock_guard lock(backend.mutex);backend.released=true;}backend.changed.notify_all();
    generation.join();exporting.join();
    if(chat_error)std::rethrow_exception(chat_error);if(feature_error)std::rethrow_exception(feature_error);
    if(chat.status!=200 || feature.status!=200)std::cerr<<"chat "<<chat.status<<" "<<chat.bytes<<" feature "<<feature.status<<" "<<feature.bytes<<std::endl;
    CHECK(chat.status==200 && feature.status==200 && backend.inside==0);
    CHECK(srv::parse_json_unique(feature.bytes).find("branch_isolation")->as_bool());
    CHECK(route(api,"/admin/drain").status==200);
    CHECK(route(api,"/v1/gemma4/features",request).status==503);
}
void wire_no_store(const g::ModelManifest& m,g::RopeTables rope,std::shared_ptr<const g::serve::Tokenizer> tok) {
    Settings settings;RealBackend backend(m,rope);e::Api api(backend,tok,{},e::DraftMode::off,"model",{}, {},e::Api::Owner::standalone,configuration(settings));
    auto first=body();const auto reply=response(http(api,first));CHECK(api.runtime().cache_stats().turn_admitted==1);
    const auto charge=api.runtime().cache_stats().bytes;
    auto second=next_turn(first,generated(reply),*tok);second.set("state_cache",Json(Json::Object{{"retain",false}}));
    response(http(api,second));CHECK(api.runtime().cache_stats().hits==1);
    CHECK(api.runtime().cache_stats().turn_admitted==1 && api.runtime().cache_stats().bytes==charge);
}
// A scripted protocol producer isolates stream ordering/error framing from
// numerical fidelity. The real model/oracle tests above remain independent.
class PresentationEngine final : public srv::GenerationEngine {
public:
    bool fail=false;
    srv::GenerationStats generate(srv::GenerationJob& job)override {
        if(fail)throw std::runtime_error("injected protocol failure");
        g::serve::OutputParser parser;
        std::uint64_t count=0;
        for(const auto* piece:{"<|channel>thought\n","why","<channel|>","before","<|tool_call>","call:f{a:1}","<tool_call|>","after","<turn|>"}) {
            const auto delta=parser.push(piece);
            job.output->publish({delta.content,delta.reasoning,delta.tool_calls.size()?g::serve::shared_calls(delta.tool_calls):Json()});
            const std::uint32_t token=10;
            if(job.on_tokens)job.on_tokens(std::span(&token,1));++count;
        }
        parser.finish();job.output->parsed=g::serve::shared_output(parser.output());
        job.output->receipt=Json(Json::Object{{"protocol_fixture",true}});
        srv::GenerationStats stats;stats.prompt_tokens=job.prompt.size();stats.generated_tokens=count;stats.finish="tool_calls";return stats;
    }
    std::uint64_t context_tokens()const override{return 8192;}
};
void tool_stream_protocol(std::shared_ptr<const g::serve::Tokenizer> tok) {
    for(bool fail:{false,true}) {
        srv::GenerationQueue queue;PresentationEngine engine;engine.fail=fail;
        srv::ApiConfig config;config.model_id="model";config.default_thinking=false;config.default_max_tokens=16;
        config.compact_chat_envelope=true;config.chat_id_prefix="chatcmpl-g4-";
        config.sse_keepalive_seconds=0;config.sse_progress_seconds=0;
        g::serve::Adapter adapter(tok);
        config.prepare_chat=[&](const Json& body) {
            const auto chat=e::parse_chat_request(body);const auto prepared=adapter.prepare(chat.input);
            srv::ApiChatPrompt prompt;prompt.tokens=prepared.ids;prompt.thinking=chat.input.enable_thinking;return prompt;
        };
        config.prepare_job=[](const srv::HttpRequest&,const Json&,srv::GenerationJob& job){job.output=std::make_shared<srv::GenerationOutput>();};
        srv::ApiService api(config,srv::ApiTokenizerBinding(g::serve::api_tokenizer(tok)),queue,8192,true);
        std::thread owner([&]{queue.run(engine);});Writer writer;
        srv::HttpRequest request;request.method="POST";request.path="/v1/chat/completions";request.body=body(true).dump();
        Writer nonstream;
        try{
            api.handle(request,writer);
            if(!fail){request.body=body(false).dump();api.handle(request,nonstream);}
        }catch(...){queue.shutdown();owner.join();throw;}
        queue.shutdown();owner.join();CHECK(writer.status==200 && writer.bytes.ends_with("data: [DONE]\n\n"));
        if(fail) {
            CHECK(writer.bytes.find("injected protocol failure")!=std::string::npos);
            CHECK(writer.bytes.find("chat.completion.chunk")==std::string::npos);
        } else {
            const auto parsed=srv::parse_json_unique(nonstream.bytes);
            CHECK(parsed.find("choices")->as_array()[0].find("message")->find("reasoning_content")->as_string()=="why");
            const auto before=writer.bytes.find("\"content\":\"before\"");
            const auto calls=writer.bytes.find("\"tool_calls\":");
            const auto after=writer.bytes.find("\"content\":\"after\"");
            CHECK(before!=std::string::npos && calls!=std::string::npos && after!=std::string::npos && before<calls && calls<after);
            CHECK(writer.bytes.find("\"tool_calls\":",calls+1)==std::string::npos);
            CHECK(writer.bytes.find("call_0")!=std::string::npos);
        }
    }
}
void legacy_hook_contract() {
    srv::GenerationQueue queue;srv::ApiConfig config;
    config.engine_routes=[](const srv::HttpRequest& request,srv::HttpResponseWriter&)->bool {
        if(request.path=="/legacy-error")throw std::invalid_argument("legacy hook failure");
        if(request.path=="/typed-error")throw srv::ApiRouteError(409,"typed route refusal");
        return false;
    };
    srv::ApiService api(config,{},queue,64,false);
    for(const auto& item:std::vector<std::pair<std::string,int>>{{"/health",200},{"/legacy-error",500},{"/typed-error",409}}) {
        srv::HttpRequest request;request.method="GET";request.path=item.first;Writer writer;api.handle(request,writer);CHECK(writer.status==item.second);
    }
    srv::HttpRequest request;request.method="POST";request.path="/tokenize";request.body=R"({"prompt":"hi"})";
    Writer writer;api.handle(request,writer);CHECK(writer.status==503 && writer.bytes.find("tokenizer")!=std::string::npos);
    CHECK(queue.snapshot().queued==0 && queue.snapshot().running==0);
}
void sse_disconnects(const g::ModelManifest& m,g::RopeTables rope,std::shared_ptr<const g::serve::Tokenizer> tok) {
    for(bool during:{false,true}) {
        Settings settings;RealBackend backend(m,rope);e::Api api(backend,tok,{},e::DraftMode::off,"model",{}, {},e::Api::Owner::standalone,configuration(settings));
        Writer writer;
        if(during)backend.on_call=[gone=writer.gone]{gone->store(true);};else writer.gone->store(true);
        srv::HttpRequest request;request.method="POST";request.path="/v1/chat/completions";request.body=body(true).dump();
        api.handle(request,writer);backend.on_call={};
        const auto status=srv::parse_json_unique(route(api,"/api/status",{},"GET").bytes);
        CHECK(status.find("queue")->find("running")->as_int()==0 && api.runtime().cache_stats().turn_admitted==0);
        CHECK(response(http(api,body())).find("choices"));
    }
}
// Hold the external owner after its first selected token. A terminal-only
// parser passes replay bytes but cannot satisfy this live stream boundary.
void live_stream(const g::ModelManifest& m,g::RopeTables rope,std::shared_ptr<const g::serve::Tokenizer> tok) {
    struct LiveWriter final : srv::HttpResponseWriter {
        std::mutex mutex;std::condition_variable changed;bool visible=false;Writer final;
        void send(int s,std::string_view t,std::string_view b,const std::vector<std::pair<std::string,std::string>>& h)override {
            std::lock_guard lock(mutex);final.send(s,t,b,h);
        }
        bool begin_stream(int s,std::string_view t,const std::vector<std::pair<std::string,std::string>>& h)override {
            std::lock_guard lock(mutex);return final.begin_stream(s,t,h);
        }
        bool write(std::string_view bytes)override {
            std::lock_guard lock(mutex);const bool ok=final.write(bytes);
            visible|=bytes.find("\"content\":\"")!=std::string_view::npos && bytes.find("\"content\":\"\"")==std::string_view::npos;
            changed.notify_all();return ok;
        }
        void end_stream()override{}
        bool client_gone()const override{return false;}
    } writer;
    RealBackend backend(m,rope);e::Api api(backend,tok,{},e::DraftMode::off,"model",{}, {},e::Api::Owner::external);
    auto b=body(true);g::serve::Adapter adapter(tok);const auto ids=prompt(b,adapter);
    std::atomic<bool> completed=false;std::exception_ptr failure;
    std::thread client([&]{try{srv::HttpRequest r;r.method="POST";r.path="/v1/chat/completions";r.body=b.dump();api.handle(r,writer);}
        catch(...){failure=std::current_exception();}completed=true;});
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(15);
    try {
        for(;;) {
            const auto status=srv::parse_json_unique(route(api,"/api/status",{},"GET").bytes);
            if(status.find("queue")->find("queued")->as_int()==1)break;
            CHECK(std::chrono::steady_clock::now()<deadline);std::this_thread::yield();
        }
        while(backend.log.rows.size()<ids.size()) {
            api.runtime().step(false);CHECK(std::chrono::steady_clock::now()<deadline);
        }
        CHECK(api.runtime().live()==1 && !completed);
        {std::unique_lock lock(writer.mutex);
            CHECK(writer.changed.wait_for(lock,std::chrono::seconds(3),[&]{return writer.visible;}));}
        CHECK(api.runtime().live()==1 && !completed); // no terminal result or receipt yet
        while(!completed){api.runtime().step(false);CHECK(std::chrono::steady_clock::now()<deadline);std::this_thread::yield();}
    }catch(...){api.runtime().shutdown();while(api.runtime().live())api.runtime().step(false);client.join();throw;}
    client.join();if(failure)std::rethrow_exception(failure);
    CHECK(generated(terminal(writer.final)).size()==3 && api.runtime().live()==0);
}
void grouped(const g::ModelManifest& m,g::RopeTables rope,std::shared_ptr<const g::serve::Tokenizer> tok) {
    environment("STRIX_G4_SERVE_SLOTS","2");RealBackend backend(m,rope);
    e::Api api(backend,tok,{},e::DraftMode::off,"model",{}, {},e::Api::Owner::external);
    std::array<Writer,3> writers;std::array<std::exception_ptr,3> failures;
    std::vector<std::thread> clients;std::atomic<int> completed=0;
    for(std::size_t i=0;i<writers.size();++i)clients.emplace_back([&,i]{
        try{writers[i]=http(api,body(i==1));}catch(...){failures[i]=std::current_exception();}++completed;
    });
    try {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(15);
        for(;;) {
            const auto j=srv::parse_json_unique(route(api,"/api/status",{},"GET").bytes);
            if(j.find("queue")->find("queued")->as_int()==3)break;
            CHECK(std::chrono::steady_clock::now()<deadline);std::this_thread::yield();
        }
        CHECK(backend.log.calls==0);
        while(completed!=3){api.runtime().step(false);CHECK(std::chrono::steady_clock::now()<deadline);std::this_thread::yield();}
    }catch(...){api.runtime().shutdown();while(api.runtime().live())api.runtime().step(false);for(auto& c:clients)c.join();throw;}
    for(auto& c:clients)c.join();for(auto f:failures)if(f)std::rethrow_exception(f);
    bool grouped=false;
    for(const auto& w:writers) {
        const auto j=response(w);CHECK(generated(j).size()==3);
        grouped|=phase(j).find("serve_batch")->find("max_batch")->as_int()==2;
    }
    CHECK(grouped && api.runtime().live()==0);
    environment("STRIX_G4_SERVE_SLOTS","1");
}
}
int main(int argc,char** argv) {
    try {
        CHECK(argc==2);environment("STRIX_G4_SERVE_SLOTS","1");environment("STRIX_G4_SERVE_BATCH","1");environment("STRIX_G4_SERVE_PREFIX","0");
        auto tok=tokenizer(std::filesystem::absolute(argv[1]));auto m=http_model();identify(m);Tables tables;auto rope=tables.view();
        legacy_hook_contract();live_stream(m,rope,tok);routes(m,rope,tok);invalid(m,rope,tok);grouped(m,rope,tok);wire_no_store(m,rope,tok);features_with_live_work(m,rope,tok);tool_stream_protocol(tok);sse_disconnects(m,rope,tok);
        // Existing independent model/cursor/cache proofs now execute through the
        // same production frontend, including failed restoration and accounting.
        two_turns(m,rope,tok);refusals(m,rope,tok);mismatches(m,rope,tok);retention(m,rope,tok);
        failures(m,rope,tok);stops_and_capacity(m,rope,tok);disconnects(m,rope,tok);
        thinking(m,rope,tokenizer(std::filesystem::absolute(argv[1]),true));
        std::cout<<"shared ApiService real Gemma4 adopter PASS\n";return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
