#include "gemma4/engine/api.hpp"
#include "gemma4/engine/runtime.hpp"
#include "gemma4/engine/deployment.hpp"
#include <cstdlib>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    namespace e = strix::gemma4::engine;
    try {
        auto options = e::parse_options(argc,argv);
        // Up to serve_slots() requests decode together: session-fixed choices (the EXL3 core identity) see that
        // stream count unless --streams sets it.
        bool streams_given = false;
        for (int i = 1; i < argc; ++i) streams_given |= std::string_view(argv[i]) == "--streams";
        if (!streams_given) options.streams = std::uint32_t(e::serve_slots());
        if (options.help) {
            std::cout << "strix-gemma4-serve MODEL --rope-tables TABLE_DIR --tokenizer tokenizer.json [--bind 127.0.0.1 --port 8080] [--dflash on|off|auto] [--draft-mtp DIR | --draft DRAFT.gguf --draft-rope TABLE.g4dr | --draft-eagle FILE] [--threads 1..16]\n"
                "drafters: --draft-mtp DIR (Gemma-4-E4B-it MTP assistant: a drafter-mtp/ package or a model dir containing one; recommended),\n"
                "  --draft DRAFT.gguf --draft-rope TABLE.g4dr (DFlash), --draft-eagle EAGLE.gguf (e4b-eagle fallback)\n"
                "env: STRIX_G4_DRAFT_KIND=dflash|mtp|eagle (how --draft is read), STRIX_G4_MTP_TOPK=1..64 (clusters, default 64),\n"
                "  STRIX_G4_MTP_Q8=1 (Q8_0 assistant matrices), STRIX_G4_MTP_GRAPHS=0 (no HIP graph replay), STRIX_G4_DFLASH=0 (no drafting)\n"
                "profiles: --profile default|fast|quality (default: default; explicit options and env win); the MTP drafter is found at\n"
                "  <model dir>/drafter-mtp unless --draft-mtp DIR is given; --draft off or STRIX_G4_DRAFT=0 disables it\n"
                "deployment: --deploy FILE|auto [--deploy-check] [--deploy-node INDEX for replica inventory]\n";
            return 0;
        }
        std::shared_ptr<const strix::deploy::Plan> deployment_plan;
        strix::serve::DeploymentCallbacks deployment;
        if (options.deploy_check && options.deploy.empty()) options.deploy = "auto";
        if (!options.deploy.empty()) {
            auto spec = strix::deploy::load_spec(options.deploy);
            if (!options.exl3_arm.empty() || !options.draft.empty() || !options.draft_mtp.empty() || !options.draft_eagle.empty())
                throw std::invalid_argument("deployment Q4_K_M target only: disable drafts with --draft off");
            if (const auto* p = std::getenv("STRIX_G4_PAIR"); p && std::string_view(p) != "0")
                throw std::invalid_argument("Gemma pair route is not present in this build");
            if (spec.profile != "auto" && spec.profile != strix::profile::name(options.profile.name))
                throw std::invalid_argument("deployment profile differs from active --profile");
            for (const auto& entry : options.profile.entries)
                if (entry.status == strix::gemma4::profile::Status::unqualified)
                    throw std::invalid_argument("deployment pending profile execution is not admitted");
            if (!strix::deploy::pinned(spec,"slots") && streams_given) spec.slots = options.streams;
            const auto model = e::deployment_model(options.model);
            if (!strix::deploy::pinned(spec,"context")) spec.context = model.max_context;
            // Replicas are independent launches. The spec supplies measured peer
            // facts; each process (and its inspection) replaces its own indexed observation.
            const auto facts = strix::deploy::inspection_devices(spec,options.deploy_node,strix::deploy::probe_device());
            std::uint64_t prefix_bytes = 4ULL<<30;
            if (const auto* mb = std::getenv("STRIX_G4_SERVE_PREFIX_MB")) {
                std::size_t end = 0; const auto n = std::stoull(mb,&end);
                if (end != std::string_view(mb).size() || n > (1ULL<<20)) throw std::invalid_argument("deployment invalid prefix cache cap");
                prefix_bytes = n<<20;
            }
            if (const auto* on = std::getenv("STRIX_G4_SERVE_PREFIX"); on && std::string_view(on) == "0") prefix_bytes = 0;
            if (const auto* batch = std::getenv("STRIX_G4_SERVE_BATCH"); batch && std::string_view(batch) == "0") {
                prefix_bytes = 0;
                if (spec.slots > 1) throw std::invalid_argument("deployment slots conflict with STRIX_G4_SERVE_BATCH=0");
                spec.slots = 1;
            }
            const auto* prefill = std::getenv("STRIX_G4_PREFILL_IDENTITY");
            const auto* identity = std::getenv("STRIX_G4_IDENTITY");
            const bool prepared = (prefill && std::string_view(prefill) != "exact") ||
                (identity && std::string_view(identity) == "g4-a8-v1");
            auto plan = strix::deploy::resolve(spec,facts,e::FootprintProvider(model,facts.size()>1,prefix_bytes,prepared));
            if (options.deploy_check) { std::cout << plan.to_json() << '\n'; return plan.refused ? 2 : 0; }
            strix::deploy::require_plan(plan);
            options.deployment_context = plan.chosen.context; options.streams = plan.chosen.slots;
#ifdef _WIN32
            _putenv_s("STRIX_G4_SERVE_SLOTS",std::to_string(options.streams).c_str());
#else
            ::setenv("STRIX_G4_SERVE_SLOTS",std::to_string(options.streams).c_str(),1);
#endif
            deployment_plan = std::make_shared<const strix::deploy::Plan>(std::move(plan));
            deployment.deployment_json = [deployment_plan] { return deployment_plan->to_json(); };
            deployment.deployment_id = [deployment_plan] { return deployment_plan->hash(); };
        }
        e::Runtime runtime(options);
        if (options.inspect) { std::cout << e::effective_config(*runtime.backend,options.request,bool(runtime.drafts)).dump() << '\n'; return 0; }
        if (!runtime.tokenizer) throw std::invalid_argument("serve requires --tokenizer");
        e::Api api(*runtime.backend,runtime.tokenizer,runtime.drafts,options.request.dflash,"nemotron-bolt-e4b",options.request,deployment);
        strix::serve::HttpServerConfig config; config.bind_address = options.bind; config.port = options.port;
        strix::serve::HttpServer server(config,[&](const auto& request, auto& writer) { api.handle(request,writer); });
        server.listen(); std::cerr << "Gemma 4 service ready\n"; server.serve();
        return 0;
    } catch (const std::exception& error) { std::cerr << "gemma4 serve: " << error.what() << '\n'; return 1; }
}
