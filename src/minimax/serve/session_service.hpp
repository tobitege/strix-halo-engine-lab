#pragma once
#include "strix/serve/run.hpp"
#include "minimax/serve/minimax_api.hpp"
#include "minimax/serve/minimax_lead.hpp"
#include "minimax/engine/deployment.hpp"
#include "minimax/engine/mm_defaults.hpp"
#include "minimax/serve/minimax_dspark_policy.hpp"
namespace strix::minimax::serve {
struct ServeOptions {
    std::string model_dir;
    std::uint32_t rank = 0, slots = defaults::Cli{}.slots;
    std::uint64_t capacity = defaults::Cli{}.capacity;
    std::uint32_t max_rows = defaults::Cli{}.max_rows;
};
struct ServePreparation final : strix::serve::ServicePreparation {
    ServeOptions opt;
    std::string transport_spec = "usb4gpu", bind = "127.0.0.1", model_id = "minimax-m3.1-flash";
    std::string dspark_dir, dspark_exl3, check, check_nested;
    std::uint16_t port = 8080;
    std::uint32_t max_tokens = 4096, check_tokens = 256, check_child_layers = ~0U;
    bool admin = false, check_only = false, bind_given = false, port_given = false;
    std::optional<int> exit_code;
    std::string output;
    dspark::Mode dspark_mode = dspark::Mode::automatic;
    dspark::Prior dspark_prior = dspark::Prior::structural_v1;
    std::uint32_t dspark_inflight = 1;
    std::vector<std::string> session_arguments;
    std::optional<strix::deploy::Spec> deploy_spec;
    std::shared_ptr<engine::FootprintProvider> deploy_provider;
};
ServePreparation prepare_serve(const std::vector<std::string>& arguments);
// This seam replaces only native assets/Runner execution. The real service,
// factory, validation, role, HTTP handlers and shell lifetime remain in use.
// Derived native members (including Lead) die before these borrowed base fields.
struct ServiceAssets {
    strix::serve::GenerationQueue queue{64};
    lead::Shared shared;
    SessionService sessions;
    bool session_enabled = false;
    // Construct once after the existing resolver admits the service RAM budget.
    // Named payloads and resident metadata consume the SAME credits; no second B.
    std::shared_ptr<PrefixStateCache> cache_owner;
    void prepare_cache_owner(std::uint64_t budget) {
        if (!cache_owner) cache_owner = strix::session::make_cache_owner(budget);
    }
    template<class Factory>
    auto with_cache_owner(std::uint64_t budget, Factory&& factory) {
        prepare_cache_owner(budget);
        return factory(cache_owner);
    }
    template<class Engine, class Runner, class Time = std::chrono::steady_clock>
    auto make_lead(Engine& engine, Runner& runner, std::uint64_t max_rows, bool taps,
        std::atomic<bool>& stop,
        std::function<std::string(const std::vector<std::uint32_t>&)> prompt = {},
        SessionService* service = nullptr) {
        prepare_cache_owner(UINT64_MAX);
        return std::make_unique<lead::Lead<Engine, Runner, Time>>(engine, runner, queue,
            shared, max_rows, taps, stop, std::move(prompt), service, cache_owner);
    }
    ObserverInfo info;
    ApiTokenizer tokenizer;
    strix::serve::DeploymentCallbacks deployment;
    virtual ~ServiceAssets() = default;
    virtual std::optional<int> startup() { return {}; }
    virtual strix::serve::SessionRuntime& runtime() = 0;
    virtual int follow() = 0;
    virtual void fail() noexcept { close(); }
    virtual void close() noexcept { sessions.tasks.close(); queue.shutdown(); }
};
using SessionAssetLoader = std::function<std::unique_ptr<ServiceAssets>(const ServePreparation&)>;
void install_session_service(strix::serve::ServiceFactories&, SessionAssetLoader);
void install_session_service(strix::serve::ServiceFactories&);
int legacy_serve(int argc, char** argv);
} // namespace strix::minimax::serve
