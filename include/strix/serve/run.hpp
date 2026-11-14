#pragma once

#include "strix/model_registry.hpp"
#include "strix/serve/session_runtime.hpp"

#include <iosfwd>
#include <map>
#include <stdexcept>

namespace strix::serve {
struct ServicePreparation { virtual ~ServicePreparation() = default; };
struct ServiceSelection {
    model_registry::Entry model;
    model_description::ModelDescription description;
    std::string root;
    // Family arguments only, in original order, with empty arguments intact.
    std::vector<std::string> arguments;
    mutable std::shared_ptr<const ServicePreparation> preparation = {};
};
// An adapter owns its assets, engine, store, queue, runtime and production
// handler. Members must be ordered so assets outlive their users. No main()
// or alternate queue-owner-loop callback belongs in this interface.
enum class ServiceRole { leader, follower };
class SessionService {
public:
    virtual ~SessionService() = default;
    virtual ServiceRole role() const noexcept { return ServiceRole::leader; }
    // Only the explicit follower role uses this synchronous protocol lifecycle.
    // It never binds a listener or starts a generation queue owner.
    virtual int follow() { throw std::logic_error("service has no follower protocol"); }
    // Paired startup/check/session CLI may finish before either rank serves.
    // Native setup has already completed through create(); this returns its
    // cached CLI/check-only result, never an alternate generation owner.
    virtual std::optional<int> startup() { return {}; }
    virtual SessionRuntime& runtime() = 0;
    // Called after bind, before owner startup; port is the actual bound port.
    virtual void listening(const HttpServerConfig&, std::uint16_t) {}
    // Wake family control tasks before HTTP drains and service destruction.
    virtual void shutdown() noexcept {}
    virtual void failed() noexcept { shutdown(); }
    virtual void stopped() {}
    virtual void handle(const HttpRequest&, HttpResponseWriter&) = 0;
    virtual void configure_listener(HttpServerConfig&) const {}
};
struct ServiceFactory {
    // Family schema/deployment/CLI validation, before construction/device work.
    std::function<void(const ServiceSelection&)> validate;
    std::function<void(const ServiceSelection&, std::ostream&)> inspect;
    std::function<std::unique_ptr<SessionService>(const ServiceSelection&)> create;
    // Optional family help/config/deployment inspection exit. Runs after
    // validation and before service construction, with no owner thread.
    std::function<std::optional<int>(const ServiceSelection&, std::ostream&)> early_exit = {};
};
class ServiceFactories {
public:
    void install(std::string family, ServiceFactory);
    [[nodiscard]] const ServiceFactory* find(const std::string& family) const;
private:
    std::map<std::string, ServiceFactory> factories_;
};
// Called explicitly by run(), so future static adapters must be referenced
// here through explicit install functions, including conditionally built Gemma4.
void install_service_factories(ServiceFactories&);

struct ServeShell {
    // Test seam around listen/serve only; construction/ownership/cleanup still
    // execute in run(). Empty selects the production HttpServer shell.
    std::function<void(const HttpServerConfig&, HttpHandler, SessionRuntime&)> serve;
};
int run(int argc, char** argv);
int run(int argc, char** argv, const ServiceFactories&, std::ostream& out,
        std::ostream& error, const ServeShell& shell = {});
} // namespace strix::serve
