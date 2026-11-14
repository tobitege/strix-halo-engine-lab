#include "strix/serve/serial_session_owner.hpp"

namespace strix::serve {
namespace {
[[noreturn]] void unsupported() {
    throw std::invalid_argument("serial native-control owner has no generic state execution");
}
session::Policy no_retention() {
    session::Policy p;
    p.enabled = false; p.store = false; p.completed_turns = false;
    return p;
}
}

void SerialSessionOwner::ControlEngine::prefill(std::span<const std::uint32_t>, session::KvHandle&) { unsupported(); }
session::StepResult SerialSessionOwner::ControlEngine::decode(session::KvHandle&, const session::SamplerState&) { unsupported(); }
session::Capabilities SerialSessionOwner::ControlEngine::caps() const {
    session::Capabilities c;
    c.max_context = context_; c.max_slots = 1; c.max_batch = 1; c.owner_control = true;
    return c;
}
session::Hooks SerialSessionOwner::logical_hooks() {
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
SessionHooks SerialSessionOwner::owner_hooks(std::function<void(const GenerationStats&)> observe) {
    SessionHooks h;
    h.control_turn = [this](SessionRuntime& owner) { turn(owner); };
    // The existing GenerationEngine owns its complete request admission ABI.
    h.native_validate = [](const GenerationJob&) {};
    h.validate = [](const GenerationJob&) {};
    h.disable_native_cache = true;
    h.serial_queue_protocol = true;
    h.serial_observe = std::move(observe);
    return h;
}
SerialSessionOwner::SerialSessionOwner(GenerationQueue& queue, GenerationEngine& native,
    std::function<void(const GenerationStats&)> observe)
    : native_(native), engine_(native.context_tokens()), leases_(1, 1, logical_hooks()),
      runtime_(queue, engine_, leases_, no_retention(), owner_hooks(std::move(observe))) {}

void SerialSessionOwner::turn(SessionRuntime& owner) {
    if (!owner.reserve_serial(true)) return;
    auto lease = owner.activate(0, true);
    GenerationStats stats;
    std::string error;
    if (owner.ticket_cancelled(lease.state)) stats.finish = "cancelled";
    else {
        // One request copy at admission. Terminal receipts move once; token
        // publication never copies receipts or native state.
        auto job = lease.taken.job;
        job.on_tokens = [&owner, h=lease.state](auto ids) { return owner.publish(h, ids, true, true); };
        job.on_admitted = [&owner, h=lease.state] { owner.notify_admitted(h); };
        job.on_progress = [&owner, h=lease.state](auto p) { owner.publish_progress(h, p); };
        try { stats = native_.generate(job); }
        catch (const GenerationRequestError& e) { stats = e.stats; error = e.what(); }
        catch (const std::exception& e) { error = e.what(); }
        catch (...) { error = "engine failure"; }
    }
    owner.complete_native(lease.state, std::move(stats), std::move(error));
}

} // namespace strix::serve
