#include "strix/serve/run.hpp"
#include "session_replay/legacy_reference.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

using namespace session_fixture;
namespace {
struct Counts { unsigned validated = 0, inspected = 0, created = 0, destroyed = 0, handled = 0; };
struct Service final : sv::SessionService {
    Counts& counts;
    FakeEngine engine{1};
    sv::GenerationQueue queue;
    sv::SessionRuntime owner;
    explicit Service(Counts& c) : counts(c), owner(queue, engine, engine.store, {}, runtime_hooks(engine)) { ++counts.created; }
    ~Service() override { ++counts.destroyed; }
    sv::SessionRuntime& runtime() override { return owner; }
    void handle(const sv::HttpRequest&, sv::HttpResponseWriter& w) override {
        ++counts.handled; w.send(200, "text/plain", "constructed", {});
    }
};
struct Follower final : sv::SessionService {
    Counts& counts;
    bool fail;
    Follower(Counts& c,bool throws):counts(c),fail(throws){++counts.created;}
    ~Follower() override {++counts.destroyed;}
    sv::ServiceRole role() const noexcept override {return sv::ServiceRole::follower;}
    int follow() override {++counts.handled;if(fail)throw std::runtime_error("peer failed");return 7;}
    sv::SessionRuntime& runtime() override {throw std::runtime_error("follower queried a generation runtime");}
    void configure_listener(sv::HttpServerConfig&) const override {throw std::runtime_error("follower configured a listener");}
    void handle(const sv::HttpRequest&,sv::HttpResponseWriter&) override {throw std::runtime_error("follower received HTTP");}
};
struct Invocation { int code; std::string out, error; };
Invocation invoke(std::vector<std::string> args, const sv::ServiceFactories& f, const sv::ServeShell& shell = {}) {
    std::vector<char*> pointers;
    for (auto& s : args) pointers.push_back(s.data());
    std::ostringstream out, error;
    const auto code = sv::run(static_cast<int>(pointers.size()), pointers.data(), f, out, error, shell);
    return {code, out.str(), error.str()};
}
void write(const std::filesystem::path& p, const std::string& text) { std::ofstream f(p); f << text; CHECK(bool(f)); }
}
int main(int argc, char** argv) {
    try {
        CHECK(argc == 2);
        const std::string root = argv[1];
        sv::ServiceFactories production;sv::install_service_factories(production);
#ifdef STRIX_EXPECT_GEMMA4_SERVICE
        CHECK(production.find("gemma4"));
        CHECK(invoke({"serve","--registry-root",root,"--model","gemma4","--inspect"},production).out.find("built")!=std::string::npos);
        CHECK(invoke({"serve","--registry-root",root,"--model","gemma4","--","--help"},production).code==0);
#else
        CHECK(!production.find("gemma4"));
        CHECK(invoke({"serve","--registry-root",root,"--model","gemma4"},production).code==3);
#endif
#ifdef STRIX_EXPECT_MINIMAX_SERVICE
        CHECK(production.find("minimax"));
        CHECK(invoke({"serve","--registry-root",root,"--model","minimax","--inspect"},production).out.find("built")!=std::string::npos);
        CHECK(invoke({"serve","--registry-root",root,"--model","minimax","--","--help"},production).code==0);
#else
        CHECK(!production.find("minimax"));
        CHECK(invoke({"serve","--registry-root",root,"--model","minimax"},production).code==3);
#endif
        {
            Counts follower_counts;bool fail=false;
            sv::ServiceFactories peers;
            peers.install("minimax",{[](const sv::ServiceSelection&){},[](const sv::ServiceSelection&,std::ostream&){},
                [&](const sv::ServiceSelection&){return std::make_unique<Follower>(follower_counts,fail);}, {}});
            sv::ServeShell forbidden;
            forbidden.serve=[](const sv::HttpServerConfig&,sv::HttpHandler,sv::SessionRuntime&){throw std::runtime_error("follower entered HTTP shell");};
            const std::vector<std::string> args{"serve","--registry-root",root,"--model","minimax"};
            CHECK(invoke(args,peers,forbidden).code==7);
            CHECK(follower_counts.created==1 && follower_counts.destroyed==1 && follower_counts.handled==1);
            fail=true;const auto result=invoke(args,peers,forbidden);
            CHECK(result.code==2 && result.error.find("peer failed")!=std::string::npos);
            CHECK(follower_counts.created==2 && follower_counts.destroyed==2 && follower_counts.handled==2);
        }
        Counts c;
        sv::ServiceFactories factories;
        std::vector<std::string> forwarded;
        factories.install("gemma4", {
            [&](const sv::ServiceSelection& s) { ++c.validated; CHECK(s.description.find("dims.hidden")); forwarded = s.arguments; },
            [&](const sv::ServiceSelection&, std::ostream& out) { ++c.inspected; out << "family read-only\n"; },
            [&](const sv::ServiceSelection&) { return std::make_unique<Service>(c); }, {}});
        CHECK(invoke({"serve", "--help", "--registry", "absent"}, factories).code == 0);
        CHECK(c.created == 0 && c.validated == 0);
        auto listing = invoke({"serve", "--registry-root", root, "--list-models"}, factories);
        CHECK(listing.code == 0 && listing.out.find("gemma4\tbuilt") != std::string::npos && c.created == 0);
        CHECK(invoke({"serve", "--registry-root", root, "--model", "gemma4", "--inspect"}, factories).code == 0);
        CHECK(c.validated == 0 && c.inspected == 0);
        CHECK(invoke({"serve", "--registry-root", root, "--model", "gemma4", "--family-inspect"}, factories).code == 0);
        CHECK(c.created == 0 && c.inspected == 1);
        sv::ServeShell shell;
        shell.serve = [&](const sv::HttpServerConfig& config, sv::HttpHandler handler, sv::SessionRuntime& runtime) {
            CHECK(config.port == 0 && config.bind_address == "127.0.0.1" && c.destroyed == 0);
            Writer writer; handler({}, writer); CHECK(writer.status == 200 && writer.body == "constructed");
            CHECK(runtime.capabilities().max_slots == 1);
        };
        CHECK(invoke({"serve", "--registry-root", root, "--model", "gemma4", "--port", "0",
            "--family-option", "a b", "", "--", "--model", "kept"}, factories, shell).code == 0);
        CHECK(forwarded == std::vector<std::string>({"--family-option", "a b", "", "--model", "kept"}));
        CHECK(c.created == 1 && c.destroyed == 1 && c.handled == 1);
        shell.serve = [](const sv::HttpServerConfig&, sv::HttpHandler, sv::SessionRuntime&) { throw std::runtime_error("shell failure"); };
        CHECK(invoke({"serve", "--registry-root", root, "--model", "gemma4"}, factories, shell).code == 2);
        CHECK(c.created == 2 && c.destroyed == 2);
        auto unknown = invoke({"serve", "--registry-root", root, "--model", "absent"}, factories);
        CHECK(unknown.code == 2 && unknown.error.find("unknown") != std::string::npos);
        auto unbuilt = invoke({"serve", "--registry-root", root, "--model", "minimax"}, factories);
        CHECK(unbuilt.code == 3 && unbuilt.error.find("known family") != std::string::npos);
        CHECK(invoke({"serve", "--registry-root", root, "--model", "nemotron_3_5_bolt_e4b", "--inspect"}, factories).code == 0);
        CHECK(invoke({"serve", "--registry-root", root, "--model", "nemotron-3.5-bolt-e4b", "--inspect"}, factories).code == 0);
        CHECK(invoke({"serve", "--registry-root", root, "--model", "gemma4", "--port", "2x"}, factories).code == 2);
        // All generated registry fixtures stay in the authorized build tree.
        namespace fs = std::filesystem;
        const fs::path temp = fs::current_path() / "registry-fixture";
        fs::create_directories(temp);
        const auto row = [](const std::string& id, const std::string& name) {
            return "\n[models." + id + "]\nname='" + name + "'\nfamily='fixture'\ndescription='" + name + ".toml'\n";
        };
        write(temp / "a.toml", "schema='strix-model/v1'\nname='a'\n");
        write(temp / "b.toml", "schema='strix-model/v1'\nname='b'\n");
        write(temp / "registry.toml", "schema='strix-registry/v1'\n" + row("a", "a") + row("b", "b"));
        const std::vector<std::string> base{"serve", "--registry-root", temp.string(), "--registry", "registry.toml", "--model", "fixture"};
        CHECK(invoke(base, factories).error.find("ambiguous") != std::string::npos);
        auto args = base; args.insert(args.end(), {"--description", "a.toml", "--inspect"});
        CHECK(invoke(args, factories).code == 0);
        write(temp / "a.toml", "schema='strix-model/v1'\nname='wrong'\n");
        CHECK(invoke(args, factories).error.find("schema/name") != std::string::npos);
        write(temp / "a.toml", "schema='invalid'\nname='a'\n"); CHECK(invoke(args, factories).code == 2);
        write(temp / "registry.toml", "schema='invalid'\n"); CHECK(invoke(args, factories).code == 2);
        write(temp / "registry.toml", "schema='strix-registry/v1'\n[models.a]\nname='a'\nfamily=7\ndescription='a.toml'\n");
        CHECK(invoke(args, factories).code == 2);
        CHECK(c.created == 2);
        bool duplicate = false;
        try { factories.install("gemma4", {[](const sv::ServiceSelection&) {}, [](const sv::ServiceSelection&, std::ostream&) {},
            [](const sv::ServiceSelection&) -> std::unique_ptr<sv::SessionService> { return {}; }, {}}); }
        catch (const std::invalid_argument&) { duplicate = true; }
        CHECK(duplicate);
        sv::ServiceFactories empty_service;
        empty_service.install("gemma4", {[](const sv::ServiceSelection&) {}, [](const sv::ServiceSelection&, std::ostream&) {},
            [](const sv::ServiceSelection&) -> std::unique_ptr<sv::SessionService> { return {}; }, {}});
        CHECK(invoke({"serve", "--registry-root", root, "--model", "gemma4"}, empty_service).error.find("no service") != std::string::npos);
        sv::ServiceFactories invalid_deployment;
        invalid_deployment.install("gemma4", {[](const sv::ServiceSelection&) { throw std::invalid_argument("deployment refused"); },
            [](const sv::ServiceSelection&, std::ostream&) {},
            [&](const sv::ServiceSelection&) { return std::make_unique<Service>(c); }, {}});
        CHECK(invoke({"serve", "--registry-root", root, "--model", "gemma4"}, invalid_deployment).code == 2);
        CHECK(c.created == 2);
        std::cout << "session registry checks passed\n"; return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
