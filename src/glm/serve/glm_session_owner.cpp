#include "glm/serve/glm_session_owner.hpp"

namespace strix::glm::serve {
namespace {
[[noreturn]] void unsupported() {
    throw std::invalid_argument("GLM request owner has no generic native state execution");
}
session::Hooks logical_hooks() {
    session::Hooks h;
    h.initialize = [](session::KvHandle) {};
    h.release = [](session::KvHandle) {};
    h.reset = [](session::KvHandle) { unsupported(); };
    h.view = [](session::KvHandle, std::uint32_t) -> session::RankState { unsupported(); };
    h.capture = h.view;
    h.restore = [](session::KvHandle, std::uint32_t, const session::RankState&) { unsupported(); };
    h.truncate = [](session::KvHandle, std::uint32_t, std::uint64_t) { unsupported(); };
    return h;
}
session::Policy no_retention() {
    session::Policy p;
    p.enabled = false; p.store = false; p.completed_turns = false;
    return p;
}
strix::serve::SessionHooks control_hooks(
    std::function<void(strix::serve::SessionRuntime&)> turn, std::function<bool()> drained,
    std::function<bool()> stop, std::function<void()> closing,
    std::function<void()> stopped, std::function<void()> failed) {
    strix::serve::SessionHooks h;
    h.control_turn = std::move(turn); h.native_drained = std::move(drained);
    h.stop_requested = std::move(stop); h.closing = std::move(closing);
    h.stopped = std::move(stopped); h.failed = std::move(failed);
    // Preserve GLM's existing API and native admission checks.
    h.native_validate = [](const strix::serve::GenerationJob&) {};
    h.validate = [](const strix::serve::GenerationJob&) {};
    h.disable_native_cache = true; h.drain_native_on_shutdown = true;
    return h;
}
}

GlmSessionOwner::ControlEngine::ControlEngine(std::uint64_t context, std::uint32_t slots) {
    caps_.max_context = context; caps_.max_slots = slots; caps_.max_batch = slots;
    caps_.owner_control = true;
}
void GlmSessionOwner::ControlEngine::prefill(std::span<const std::uint32_t>, session::KvHandle&) { unsupported(); }
session::StepResult GlmSessionOwner::ControlEngine::decode(session::KvHandle&, const session::SamplerState&) { unsupported(); }

GlmSessionOwner::GlmSessionOwner(strix::serve::GenerationQueue& queue, std::uint64_t context,
    std::uint32_t slots, std::function<void(strix::serve::SessionRuntime&)> turn,
    std::function<bool()> drained, std::function<bool()> stop, std::function<void()> closing,
    std::function<void()> stopped, std::function<void()> failed)
    : engine_(context, slots), leases_(slots, 2, logical_hooks()),
      runtime_(queue, engine_, leases_, no_retention(), control_hooks(std::move(turn), std::move(drained),
          std::move(stop), std::move(closing), std::move(stopped), std::move(failed))) {}

} // namespace strix::glm::serve
