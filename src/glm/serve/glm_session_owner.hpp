#pragma once

#include "strix/serve/session_runtime.hpp"

namespace strix::glm::serve {

// Request ownership only. Physical slots, grouped native work and rank control
// belong to Lead; logical handles never address native state.
class GlmSessionOwner {
public:
    GlmSessionOwner(strix::serve::GenerationQueue&, std::uint64_t context, std::uint32_t slots,
                    std::function<void(strix::serve::SessionRuntime&)> turn,
                    std::function<bool()> drained, std::function<bool()> stop,
                    std::function<void()> closing, std::function<void()> stopped,
                    std::function<void()> failed);
    strix::serve::SessionRuntime& runtime() noexcept { return runtime_; }
private:
    class ControlEngine final : public session::Engine {
    public:
        ControlEngine(std::uint64_t, std::uint32_t);
        void prefill(std::span<const std::uint32_t>, session::KvHandle&) override;
        session::StepResult decode(session::KvHandle&, const session::SamplerState&) override;
        session::Capabilities caps() const override { return caps_; }
        profile::ExecutionReceipt receipt() const override { return {}; }
    private:
        session::Capabilities caps_;
    } engine_;
    session::KvStore leases_;
    strix::serve::SessionRuntime runtime_;
};

} // namespace strix::glm::serve
