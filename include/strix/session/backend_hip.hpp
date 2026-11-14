#pragma once
#include "strix/session/backend.hpp"

namespace strix::session {

// Opt-in adapter. Construction binds the current device without replacing any
// production resource owner. Native objects live exclusively in the .cpp file.
class HipBackend : public Backend {
public:
    explicit HipBackend(std::size_t budget_bytes);
    ~HipBackend() override;
    [[nodiscard]] std::string_view backend_name() const noexcept override { return "hip"; }
    [[nodiscard]] MemoryBudget memory_budget() const override;
protected:
    void execute_plan(ExecutionPlan&) override;
private:
    struct State;
    std::shared_ptr<State> state_;
    std::shared_ptr<detail::BufferState> allocate_buffer(std::size_t, std::size_t) override;
    std::shared_ptr<detail::QueueState> make_queue() override;
    std::shared_ptr<detail::EventState> record_event(detail::QueueState&) override;
    void wait_event(detail::QueueState&, detail::EventState&) override;
    void finish(detail::QueueState&) override;
    void drain(detail::QueueState&) noexcept override;
    void transfer(detail::QueueState&, const detail::BoundBuffer*, const detail::BoundBuffer*,
                  void*, const void*, std::size_t) override;
};
}
