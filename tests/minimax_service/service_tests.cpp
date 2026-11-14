#include "minimax/serve/session_service.hpp"
#include "../../tests/session_replay/legacy_reference.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
using namespace session_fixture;
namespace ms = strix::minimax::serve;
namespace {
struct Counts {unsigned loaded=0,destroyed=0,runtime=0,followed=0,closed=0,failed=0;};
struct Assets final : ms::ServiceAssets {
    Counts& counts;
    bool follower, fail_follow;
    std::optional<int> exit;
    FakeEngine engine{1};
    sv::SessionRuntime owner;
    Assets(Counts& c,const ms::ServePreparation& p,bool follow_failure,bool owner_failure)
        :counts(c),follower(p.opt.rank!=0),fail_follow(follow_failure),
         owner(queue,engine,engine.store,{},hooks(owner_failure)) {
        ++counts.loaded;
        if(p.check_only || !p.session_arguments.empty())exit=4;
        info.model=p.model_id; info.context_tokens=64;info.slots=1;info.transport="host-fixture";
        tokenizer.eos_id=0;
        tokenizer.encode=[](std::string_view){return std::vector<std::uint32_t>{10,11};};
        tokenizer.decode=[](std::span<const std::uint32_t> ids){std::string s;for(auto id:ids)if(id)s+=static_cast<char>(id);return s;};
        tokenizer.stream_bytes=[](std::uint32_t id){return id ? std::string(1,static_cast<char>(id)) : std::string{};};
        session_enabled=p.admin;
        if(session_enabled) sessions.tasks.publish_metadata(sv::Json::object(),sv::Json::object());
    }
    sv::SessionHooks hooks(bool fail) {
        auto h=runtime_hooks(engine);
        if(fail)h.idle=[] {throw std::runtime_error("owner failed");};
        h.stopped=[this]{sessions.tasks.close();};
        return h;
    }
    ~Assets() override {++counts.destroyed;}
    std::optional<int> startup() override {return exit;}
    sv::SessionRuntime& runtime() override {++counts.runtime;CHECK(!follower);return owner;}
    int follow() override {++counts.followed;CHECK(follower);if(fail_follow)throw std::runtime_error("follower failed");return 0;}
    void close() noexcept override {++counts.closed;ServiceAssets::close();owner.shutdown();}
    void fail() noexcept override {++counts.failed;close();}
};
struct Result {int code;std::string out,error;};
Result invoke(std::vector<std::string> args,const sv::ServiceFactories& f,const sv::ServeShell& shell={}) {
    std::vector<char*> argv;for(auto& a:args)argv.push_back(a.data());
    std::ostringstream out,error;
    const int code=sv::run(static_cast<int>(argv.size()),argv.data(),f,out,error,shell);
    return {code,out.str(),error.str()};
}
std::vector<std::string> base(const std::string& root,const std::string& model="minimax") {
    return {"serve","--registry-root",root,"--model",model,"--","MODEL_DIR","--rank","0"};
}
std::vector<std::string> with(std::vector<std::string> a,std::initializer_list<std::string> b){a.insert(a.end(),b);return a;}
}
int main(int argc,char** argv) {
    try {
        CHECK(argc==2);const std::string root=argv[1];
        Counts c;bool setup_fail=false,follow_fail=false,owner_fail=false,invalid_assets=false;
        sv::ServiceFactories factories;
        ms::install_session_service(factories,[&](const ms::ServePreparation& p)->std::unique_ptr<ms::ServiceAssets>{
            if(setup_fail)throw std::runtime_error("setup failed");
            auto a=std::make_unique<Assets>(c,p,follow_fail,owner_fail);
            if(invalid_assets)a->tokenizer={};
            return a;
        });
        sv::ServiceFactories production;sv::install_service_factories(production);
#ifdef STRIX_EXPECT_MINIMAX_SERVICE
        CHECK(production.find("minimax"));
#else
        CHECK(!production.find("minimax"));
#endif
        for(const auto& id:{"minimax","minimax_m3_1_flash","minimax-m3.1-flash"}) {
            const auto r=invoke({"serve","--registry-root",root,"--model",id,"--inspect"},factories);
            CHECK(r.code==0 && r.out.find("models/minimax-m3.1-flash.toml")!=std::string::npos);
            CHECK(r.out.find("built")!=std::string::npos);
        }
        CHECK(invoke({"serve","--registry-root",root,"--description","models/minimax-m3.1-flash.toml","--inspect"},factories).code==0);
        CHECK(invoke({"serve","--registry-root",root,"--model","minimax","--","--help"},factories).code==0);
        CHECK(invoke(with(base(root),{"--print-effective-config"}),factories).code==0);
        CHECK(c.loaded==0);
        auto entries=strix::model_registry::load((std::filesystem::path(root)/"models/registry.toml").string());
        const auto row=std::find_if(entries.begin(),entries.end(),[](const auto& e){return e.family=="minimax";});
        CHECK(row!=entries.end());
        sv::ServiceSelection invalid{*row,{},root,{"MODEL_DIR"}};
        bool refused=false;
        try {factories.find("minimax")->validate(invalid);}catch(const std::exception&){refused=true;}
        CHECK(refused && c.loaded==0);
        for(const auto& bad:std::vector<std::vector<std::string>>{
            {"--rank","2"},{"--rank","-1"},{"--slots","0"},{"--slots","9"},
            {"--port","65536"},{"--capacity","1x"},{"--max-rows","0"},{"--unknown"},
            {"--dspark-mode","bad"},{"--resume-check-child-layers","0"},
            {"--session-save","A","--session-load","B"},{"--session-generate","-1"},{"--transport"}}) {
            auto a=base(root);a.insert(a.end(),bad.begin(),bad.end());CHECK(invoke(a,factories).code==2);
        }
        CHECK(c.loaded==0);
        const auto fixture=std::filesystem::current_path()/"service-model";
        std::filesystem::create_directories(fixture);
        std::filesystem::copy_file(std::filesystem::path(root)/"models/minimax-m3.1-flash.toml",fixture/"minimax-m3.1-flash.toml",std::filesystem::copy_options::overwrite_existing);
        for(const auto& option:{"--print-model-graph","--print-plan"}) {
            auto a=base(root);a[6]=fixture.string();a.push_back(option);CHECK(invoke(a,factories).code==0);
        }
        CHECK(invoke(with(base(root),{"--deploy","missing.toml","--deploy-check"}),factories).code==2);
        CHECK(c.loaded==0);
        sv::ServiceFactories native;
        ms::install_session_service(native);
        const auto unavailable=invoke(base(root),native);
        CHECK(unavailable.code==2 && unavailable.error.find("native execution unavailable")!=std::string::npos);
        sv::ServeShell shell;
        shell.serve=[&](const sv::HttpServerConfig& config,sv::HttpHandler handle,sv::SessionRuntime& owner) {
            CHECK(config.port==0);std::exception_ptr failure;
            std::thread t([&]{try{owner.run();}catch(...){failure=std::current_exception();}});
            try {
                Writer health;handle({"GET","/health","/health"},health);CHECK(health.status==200);
                Writer models;handle({"GET","/v1/models","/v1/models"},models);CHECK(models.body.find("family-id")!=std::string::npos);
                sv::HttpRequest req;req.method="POST";req.path="/v1/completions";
                req.body=R"({"prompt":"hi","max_tokens":3,"temperature":0.7,"return_token_ids":true})";
                Writer reply;handle(req,reply);CHECK(reply.status==200);
                CHECK(reply.body.find("temperature ignored: this engine decodes greedy (exact)")!=std::string::npos);
                CHECK(reply.body.find("[65,66,0]")!=std::string::npos);
                req.path="/admin/sched";req.method="GET";req.body="";
                Writer admin;handle(req,admin);CHECK(admin.status==200);
                req.path="/v1/sessions";
                Writer sessions;handle(req,sessions);CHECK(sessions.status==200);
                req.path="/v1/cache/stats";
                Writer stats;handle(req,stats);CHECK(stats.status==200);
            } catch(...) {owner.shutdown();t.join();throw;}
            owner.shutdown();t.join();if(failure)std::rethrow_exception(failure);
        };
        CHECK(invoke(with(base(root),{"--port","0","--model-id","family-id","--admin"}),factories,shell).code==0);
        CHECK(c.loaded==1 && c.destroyed==1 && c.runtime==1);
        auto follower=base(root);follower.back()="1";
        sv::ServeShell forbidden;
        forbidden.serve=[](const sv::HttpServerConfig&,sv::HttpHandler,sv::SessionRuntime&){throw std::runtime_error("follower bound listener");};
        CHECK(invoke(follower,factories,forbidden).code==0);
        CHECK(c.followed==1 && c.runtime==1 && c.destroyed==2);
        follow_fail=true;CHECK(invoke(follower,factories,forbidden).error.find("follower failed")!=std::string::npos);follow_fail=false;
        CHECK(invoke(with(base(root),{"--resume-check","fixture.u32","--check-only"}),factories,forbidden).code==4);
        CHECK(invoke(with(follower,{"--session-prompt","fixture.u32"}),factories,forbidden).code==4);
        CHECK(c.runtime==1 && c.followed==2);
        setup_fail=true;CHECK(invoke(base(root),factories).error.find("setup failed")!=std::string::npos);setup_fail=false;
        invalid_assets=true;CHECK(invoke(base(root),factories).error.find("tokenizer")!=std::string::npos);invalid_assets=false;
        // Actual listener failure closes the actual service, without an owner thread.
        CHECK(invoke(with(base(root),{"--bind","invalid-bind-address"}),factories).code==2);
        CHECK(c.failed>0);
        owner_fail=true;
        const auto owner_result=invoke(with(base(root),{"--port","0"}),factories);
        CHECK(owner_result.code==2 && owner_result.error.find("owner failed")!=std::string::npos);owner_fail=false;
        sv::ServeShell handler_failure;
        handler_failure.serve=[](const sv::HttpServerConfig&,sv::HttpHandler handle,sv::SessionRuntime&) {
            struct ThrowWriter final : sv::HttpResponseWriter {
                void send(int,std::string_view,std::string_view,const std::vector<std::pair<std::string,std::string>>&) override {throw std::runtime_error("handler failed");}
                bool begin_stream(int,std::string_view,const std::vector<std::pair<std::string,std::string>>&) override {return false;}
                bool write(std::string_view) override {return false;}void end_stream() override{}bool client_gone() const override{return false;}
            } w;
            handle({"GET","/health","/health"},w);
        };
        CHECK(invoke(base(root),factories,handler_failure).code==2);
        CHECK(c.loaded==c.destroyed);
        std::cout<<"MiniMax real service host checks passed\n";return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
