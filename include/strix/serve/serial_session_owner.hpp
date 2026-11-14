#pragma once

#include "strix/serve/session_runtime.hpp"

namespace strix::serve {

// Adapts one synchronous native request to the common owner. Logical leases
// carry generations only: no slot number ever addresses native model state.
class SerialSessionOwner {
public:
    SerialSessionOwner(GenerationQueue&, GenerationEngine&,
        std::function<void(const GenerationStats&)> observe = {});
    SessionRuntime& runtime() noexcept { return runtime_; }

private:
    class ControlEngine final : public session::Engine {
    public:
        explicit ControlEngine(std::uint64_t context) : context_(context) {}
        void prefill(std::span<const std::uint32_t>, session::KvHandle&) override;
        session::StepResult decode(session::KvHandle&, const session::SamplerState&) override;
        session::Capabilities caps() const override;
        profile::ExecutionReceipt receipt() const override { return {}; }
    private:
        std::uint64_t context_;
    };
    static session::Hooks logical_hooks();
    SessionHooks owner_hooks(std::function<void(const GenerationStats&)>);
    void turn(SessionRuntime&);
    GenerationEngine& native_;
    ControlEngine engine_;
    session::KvStore leases_;
    SessionRuntime runtime_;
};

} // namespace strix::serve
