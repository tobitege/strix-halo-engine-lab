#pragma once
#include "strix/session/backend.hpp"

namespace strix::session {

// Correctness reference for explicitly registered operations, not a model loader.
class CpuBackend : public Backend {
public:
    explicit CpuBackend(std::size_t budget_bytes = 64U * 1024U * 1024U);
    ~CpuBackend() override;
    [[nodiscard]] std::string_view backend_name() const noexcept override { return "cpu"; }
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
    void transfer(detail::QueueState&, const detail::BoundBuffer*, const detail::BoundBuffer*,
                  void*, const void*, std::size_t) override;
};
}
