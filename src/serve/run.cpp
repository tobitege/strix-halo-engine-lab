#include "strix/serve/run.hpp"

#include <charconv>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <set>
#include <thread>

#ifdef STRIX_SERVE_GEMMA4
namespace strix::gemma4::engine { void install_session_service(strix::serve::ServiceFactories&); }
#endif

#ifdef STRIX_SERVE_MINIMAX
namespace strix::minimax::serve { void install_session_service(strix::serve::ServiceFactories&); }
#endif

namespace strix::serve {
void ServiceFactories::install(std::string family, ServiceFactory f) {
    if (family.empty() || !f.create || !f.validate || !f.inspect)
        throw std::invalid_argument("service factory requires family, validation, inspection and construction");
    if (!factories_.emplace(std::move(family), std::move(f)).second)
        throw std::invalid_argument("duplicate service factory");
}
const ServiceFactory* ServiceFactories::find(const std::string& family) const {
    const auto it = factories_.find(family);
    return it == factories_.end() ? nullptr : &it->second;
}
void install_service_factories(ServiceFactories& factories) {
#ifdef STRIX_SERVE_GEMMA4
    gemma4::engine::install_session_service(factories);
#else
    (void)factories;
#endif
#ifdef STRIX_SERVE_MINIMAX
    minimax::serve::install_session_service(factories);
#endif
}
namespace {
std::string text(const model_description::ModelDescription& d, const char* key) {
    const auto* value = d.find(key);
    const auto* s = value ? std::get_if<std::string>(&value->data) : nullptr;
    if (!s) throw std::invalid_argument(std::string("description missing string: ") + key);
    return *s;
}
void serve(const HttpServerConfig& config, HttpHandler handler, SessionRuntime& runtime, SessionService& service) {
    HttpServer server(config, std::move(handler));
    std::once_flag stopped;
    const auto stop = [&] { std::call_once(stopped, [&] { service.shutdown(); server.stop(); }); };
    std::exception_ptr owner_error;
    std::thread owner;
    try {
        server.listen();
        service.listening(config, server.port());
        owner = std::thread([&] {
            try { runtime.run(); } catch (...) { owner_error = std::current_exception(); }
            stop();
        });
        server.serve();
    } catch (...) {
        service.failed(); runtime.shutdown(); stop();
        if(owner.joinable())owner.join();
        throw;
    }
    runtime.shutdown(); stop();
    if(owner.joinable())owner.join();
    if(owner_error)std::rethrow_exception(owner_error);
}

}
int run(int argc, char** argv, const ServiceFactories& factories, std::ostream& out,
        std::ostream& error, const ServeShell& shell) {
    try {
        std::string root = ".", registry = "models/registry.toml", model, description;
        std::vector<std::string> family_args;
        HttpServerConfig config;
        bool list = false, inspect = false, family_inspect = false;
        const auto value = [&](int& i) {
            if (++i >= argc) throw std::invalid_argument("missing option value");
            return std::string(argv[i]);
        };
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--") { while (++i < argc) family_args.emplace_back(argv[i]); break; }
            if (arg == "--help" || arg == "-h") {
                out << "strix-serve --model FAMILY|ID|NAME [family arguments]\n"
                    "  --registry-root DIR --registry PATH --description PATH\n"
                    "  --list-models --inspect --family-inspect --bind ADDRESS --port N\n"
                    "Use -- before family arguments that overlap shell options.\n"
                    "Registry metadata does not imply a built execution adapter.\n";
                return 0;
            }
            if (arg == "--registry-root") root = value(i);
            else if (arg == "--registry") registry = value(i);
            else if (arg == "--model") model = value(i);
            else if (arg == "--description") description = value(i);
            else if (arg == "--list-models") list = true;
            else if (arg == "--inspect") inspect = true;
            else if (arg == "--family-inspect") family_inspect = true;
            else if (arg == "--bind") config.bind_address = value(i);
            else if (arg == "--port") {
                const auto number = value(i);
                unsigned port = 0;
                const auto parsed = std::from_chars(number.data(), number.data() + number.size(), port);
                if (parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size() || port > 65535)
                    throw std::invalid_argument("invalid port");
                config.port = static_cast<std::uint16_t>(port);
            } else family_args.push_back(arg);
        }
        namespace fs = std::filesystem;
        const auto entries = model_registry::load((fs::path(root) / registry).string());
        std::set<std::string> names;
        for (const auto& e : entries) {
            if (e.id.empty() || e.name.empty() || e.family.empty() || e.description.empty() || !names.insert(e.name).second)
                throw std::invalid_argument("invalid or duplicate registry model");
            const fs::path path(e.description);
            if (path.is_absolute()) throw std::invalid_argument("registry description must be relative");
            for (const auto& part : path) if (part == "..") throw std::invalid_argument("registry description escapes root");
        }
        const auto load = [&](const model_registry::Entry& e) {
            auto d = model_description::load((fs::path(root) / e.description).string());
            if (text(d, "schema") != "strix-model/v1" || text(d, "name") != e.name)
                throw std::invalid_argument("description schema/name differs from registry");
            return d;
        };
        if (list) {
            for (const auto& e : entries) {
                (void)load(e);
                out << e.id << '\t' << e.name << '\t' << e.family << '\t'
                    << (factories.find(e.family) ? "built" : "unbuilt") << '\n';
            }
            return 0;
        }
        if (model.empty() && description.empty()) throw std::invalid_argument("--model is required");
        std::vector<const model_registry::Entry*> selected;
        // An explicit registry ID/name takes precedence over a family alias.
        bool exact = false;
        for (const auto& e : entries) if (e.id == model || e.name == model) exact = true;
        for (const auto& e : entries) {
            if (!model.empty() && !(exact ? e.id == model || e.name == model : e.family == model)) continue;
            if (!description.empty() && fs::path(e.description).lexically_normal() != fs::path(description).lexically_normal()) continue;
            selected.push_back(&e);
        }
        if (selected.empty()) throw std::invalid_argument("unknown model family or description: " + model);
        if (selected.size() != 1) throw std::invalid_argument("ambiguous model family; choose --description or a registry ID");
        ServiceSelection selection{*selected[0], load(*selected[0]), root, std::move(family_args), {}};
        const auto* factory = factories.find(selection.model.family);
        if (inspect) {
            out << selection.model.id << '\t' << selection.model.name << '\t' << selection.model.family
                << '\t' << selection.model.description << '\t' << (factory ? "built" : "unbuilt") << '\n';
            return 0;
        }
        if (!factory) {
            error << "known family '" << selection.model.family << "' has no built shared serve adapter; legacy entry points remain separate\n";
            return 3;
        }
        factory->validate(selection);
        if(factory->early_exit) if(auto result=factory->early_exit(selection,out)) return *result;
        if (family_inspect) { factory->inspect(selection, out); return 0; }
        auto service = factory->create(selection);
        if (!service) throw std::runtime_error("factory returned no service");
        if (auto result=service->startup()) return *result;
        if (service->role() == ServiceRole::follower) {
            try { const int result=service->follow(); service->shutdown(); return result; }
            catch (...) { service->failed(); service->shutdown(); throw; }
        }
        service->configure_listener(config);
        auto& runtime = service->runtime();
        try {
            const HttpHandler handler = [&](const HttpRequest& request, HttpResponseWriter& writer) { service->handle(request, writer); };
            if (shell.serve) shell.serve(config, handler, runtime);
            else serve(config, handler, runtime, *service);
        } catch (...) { service->failed(); runtime.shutdown(); service->shutdown(); throw; }
        runtime.shutdown(); service->shutdown(); service->stopped();
        return 0;
    } catch (const std::exception& e) { error << "strix-serve: " << e.what() << '\n'; return 2; }
    catch (...) { error << "strix-serve: service failure\n"; return 2; }
}
int run(int argc, char** argv) {
    ServiceFactories factories;
    install_service_factories(factories);
    return run(argc, argv, factories, std::cout, std::cerr);
}
} // namespace strix::serve
