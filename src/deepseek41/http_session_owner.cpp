#include "deepseek41/http_session_owner.hpp"

namespace strix::deepseek41 {
namespace {
[[noreturn]] void unsupported() {
    throw std::invalid_argument("DS41 request owner has no generic native state execution");
}
session::Policy no_retention() {
    session::Policy p; p.enabled = false; p.store = false; p.completed_turns = false; return p;
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
serve::SessionHooks control_hooks(std::function<void(serve::SessionRuntime&)> turn,
                                std::function<bool()> drained, std::function<void()> failed,
                                std::function<void()> stopped) {
    serve::SessionHooks h;
    h.control_turn = std::move(turn); h.native_drained = std::move(drained); h.failed = std::move(failed);
    h.stopped = std::move(stopped);
    // Existing family API and admission branches validate the native ABI.
    h.native_validate = [](const serve::GenerationJob&) {};
    h.validate = [](const serve::GenerationJob&) {};
    h.disable_native_cache = true; h.drain_native_on_shutdown = true;
    return h;
}
}
HttpSessionOwner::ControlEngine::ControlEngine(std::uint64_t context, std::uint32_t slots) {
    caps_.max_context = context; caps_.max_slots = slots; caps_.max_batch = slots; caps_.owner_control = true;
}
void HttpSessionOwner::ControlEngine::prefill(std::span<const std::uint32_t>, session::KvHandle&) { unsupported(); }
session::StepResult HttpSessionOwner::ControlEngine::decode(session::KvHandle&, const session::SamplerState&) { unsupported(); }
HttpSessionOwner::HttpSessionOwner(serve::GenerationQueue& q, std::uint64_t context, std::uint32_t slots,
                                 std::function<void(serve::SessionRuntime&)> turn,
                                 std::function<bool()> drained, std::function<void()> failed,
                                 std::function<void()> stopped)
    : engine_(context, slots), leases_(slots, 2, logical_hooks()),
      runtime_(q, engine_, leases_, no_retention(), control_hooks(std::move(turn), std::move(drained), std::move(failed), std::move(stopped))) {}
} // namespace strix::deepseek41
