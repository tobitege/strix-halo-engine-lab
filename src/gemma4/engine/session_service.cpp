#include "session_service.hpp"
#include "runtime.hpp"
#include "gemma4/model/description.hpp"
#include <ostream>

namespace strix::gemma4::engine {
namespace {
struct Prepared final : strix::serve::ServicePreparation { ServePreparation value; };
const ServePreparation& prepared(const strix::serve::ServiceSelection& s) {
    const auto p=std::dynamic_pointer_cast<const Prepared>(s.preparation);
    if(!p)throw std::invalid_argument("Gemma4 service has not passed validation");
    return p->value;
}
class Service final : public strix::serve::SessionService {
    SessionAssets assets_;
    ServePreparation settings_;
    Api api_;
public:
    Service(const ServePreparation& p,SessionAssets assets)
        :assets_(std::move(assets)),settings_(p),api_(*assets_.backend,assets_.tokenizer,assets_.drafts,p.options.request.dflash,
            "nemotron-bolt-e4b",p.options.request,p.deployment,Api::Owner::external,{},p.deployment_slots) {}
    strix::serve::SessionRuntime& runtime() override {return api_.runtime();}
    void configure_listener(strix::serve::HttpServerConfig& config) const override {
        if(settings_.bind_given)config.bind_address=settings_.options.bind;
        if(settings_.port_given)config.port=settings_.options.port;
    }
    void handle(const strix::serve::HttpRequest& r,strix::serve::HttpResponseWriter& w) override {api_.handle(r,w);}
};
}
void install_session_service(strix::serve::ServiceFactories& factories) { install_session_service(factories,{}); }
void install_session_service(strix::serve::ServiceFactories& factories,SessionAssetLoader loader) {
    if(!loader)loader=[](const Options& options){auto assets=std::make_shared<Runtime>(options);
        return SessionAssets{assets,assets->backend.get(),assets->tokenizer,assets->drafts};};
    strix::serve::ServiceFactory f;
    f.validate=[](const strix::serve::ServiceSelection& s) {
        (void)validate_model_description(s.description);
        auto p=std::make_shared<Prepared>();std::vector<std::string> args{"strix-serve"};
        args.insert(args.end(),s.arguments.begin(),s.arguments.end());std::vector<char*> argv;
        for(auto& arg:args)argv.push_back(arg.data());
        p->value=prepare_serve(int(argv.size()),argv.data());
        if(!p->value.options.help && !p->value.exit_code && !p->value.options.inspect && p->value.options.tokenizer.empty())
            throw std::invalid_argument("serve requires --tokenizer");
        s.preparation=std::move(p);
    };
    f.early_exit=[loader](const strix::serve::ServiceSelection& s,std::ostream& out)->std::optional<int> {
        const auto& p=prepared(s);
        if(p.exit_code){out<<p.output;return p.exit_code;}
        if(p.options.inspect){auto assets=loader(p.options);out<<effective_config(*assets.backend,p.options.request,bool(assets.drafts)).dump()<<'\n';return 0;}
        return std::nullopt;
    };
    f.inspect=[loader](const strix::serve::ServiceSelection& s,std::ostream& out) {
        auto options=prepared(s).options;options.inspect=true;
        auto assets=loader(options);out<<effective_config(*assets.backend,options.request,bool(assets.drafts)).dump()<<'\n';
    };
    f.create=[loader](const strix::serve::ServiceSelection& s){auto assets=loader(prepared(s).options);
        if(!assets.owner || !assets.backend || !assets.tokenizer)throw std::invalid_argument("incomplete Gemma4 service assets");
        return std::make_unique<Service>(prepared(s),std::move(assets));};
    factories.install("gemma4",std::move(f));
}
} // namespace strix::gemma4::engine
