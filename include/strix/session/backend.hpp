#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::session {

class Backend;

class UnsupportedOperation : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

namespace detail {
// Implementation extension points; none exposes a native device object.
struct Resource {
    virtual ~Resource() = default;
    std::shared_ptr<const void> owner;
    bool released = false;
};
struct BufferState : Resource { std::size_t bytes = 0, alignment = 1; };
struct QueueState : Resource { bool failed = false; };
struct EventState : Resource {};
}

template<class State> class ResourceHandle {
public:
    ResourceHandle() = default;
    ResourceHandle(ResourceHandle&&) noexcept = default;
    ResourceHandle& operator=(ResourceHandle&& other) noexcept {
        if (this != &other) { reset(); state_ = std::move(other.state_); }
        return *this;
    }
    ~ResourceHandle() { reset(); }
    ResourceHandle(const ResourceHandle&) = delete;
    ResourceHandle& operator=(const ResourceHandle&) = delete;
    void reset() noexcept {
        if (state_) state_->released = true;
        state_.reset();
    }
    [[nodiscard]] explicit operator bool() const noexcept { return bool(state_); }
private:
    friend class Backend;
    friend struct BufferSlice;
    std::shared_ptr<State> state_;
};
using Buffer = ResourceHandle<detail::BufferState>;
using Queue = ResourceHandle<detail::QueueState>;
using Event = ResourceHandle<detail::EventState>;

// A non-owning checked view. Releasing the owning Buffer expires all slices.
struct BufferSlice {
    BufferSlice() = default;
    BufferSlice(const Buffer& buffer, std::size_t byte_offset, std::size_t byte_count)
        : offset(byte_offset), bytes(byte_count), state_(buffer.state_) {}
    std::size_t offset = 0, bytes = 0;
private:
    friend class Backend;
    std::weak_ptr<detail::BufferState> state_;
};

struct MemoryBudget {
    // Available is an admission ceiling, not a physical-fit promise. Unknown
    // observations remain zero with probe_known=false. Reserved includes padding.
    std::size_t limit = 0, reserved = 0, available = 0;
    std::size_t observed_free = 0, observed_total = 0;
    bool probe_known = false;
};

struct OperationKey {
    std::string operation, dtype, quantization, backend;
    bool operator==(const OperationKey&) const = default;
};
struct Operation {
    OperationKey key;
    std::vector<BufferSlice> inputs;
    BufferSlice output;
    std::size_t elements = 0;
};
struct OperationSchema {
    std::size_t inputs = 0, element_bytes = 1, alignment = 1;
    std::size_t max_elements = static_cast<std::size_t>(-1);
};
namespace detail {
struct BoundBuffer {
    std::shared_ptr<BufferState> state;
    std::size_t offset = 0, bytes = 0;
};
struct BoundOperation {
    std::vector<BoundBuffer> inputs;
    BoundBuffer output;
    std::size_t elements = 0;
};
}

// A family defines a host-side plan type containing its complete forward batch:
// causal/independent rows, physical state bindings, optional taps/head requests,
// and output storage. The plan's schema is shared by that family's execution
// implementations; it contains no scheduling or token-selection policy.
//
// Engine builds the plan, Backend executes it, Engine verifies/selects and
// commits or rolls back. For example, a many-stream verify plan stays one call,
// rather than becoming a generic loop of single-token forwards. A host executor
// and an accelerator executor can implement the same plan without changing the
// Engine's model semantics, sampling, drafting or batch planner.
class ExecutionPlan {
public:
    explicit ExecutionPlan(Backend& owner) noexcept : owner_(owner) {}
    virtual ~ExecutionPlan() = default;
    ExecutionPlan(const ExecutionPlan&) = delete;
    ExecutionPlan& operator=(const ExecutionPlan&) = delete;

    [[nodiscard]] Backend& owner() const noexcept { return owner_; }

private:
    Backend& owner_;
};

// Host dispatch only: no virtual calls in kernels and no universal device-kind
// switch. Concrete backends implement the family's plan schema, reject unknown
// plan types before side effects, and retain their own allocation/shard hooks.
// This interface neither owns logical KvHandles nor prescribes physical KV.
class Backend {
public:
    Backend() = default;
    virtual ~Backend() = default;
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    // Resource calls are synchronous, externally serialized by the caller.
    // A successful call completes all its work; host spans are borrowed only
    // for that call. No implicit default queue and no cross-owner resources.
    [[nodiscard]] Buffer allocate(std::size_t bytes, std::size_t alignment = 64);
    [[nodiscard]] Queue create_queue();
    [[nodiscard]] Event record(Queue&);
    void wait(Queue&, const Event&);
    void synchronize(Queue&);
    void upload(Queue&, BufferSlice destination, std::span<const std::byte>);
    void download(Queue&, BufferSlice source, std::span<std::byte>);
    void copy(Queue&, BufferSlice destination, BufferSlice source);
    [[nodiscard]] virtual MemoryBudget memory_budget() const { return {}; }
    [[nodiscard]] bool supports(const OperationKey&) const noexcept;
    [[nodiscard]] std::optional<OperationSchema> query(const OperationKey&) const noexcept;
    [[nodiscard]] virtual std::string_view backend_name() const noexcept { return {}; }
    void dispatch(Queue&, const Operation&);

    // Synchronous completion boundary, even if execution internally overlaps
    // streams/exchanges. All writes and host outputs are complete on return;
    // no borrowed plan, input, output or callback is retained. The backend and
    // its resources outlive the plan and every Engine/state binding using it.
    // A plan may write tentative rows. On failure Backend drains outstanding
    // work before throwing; Engine then restores or invalidates affected state.
    void execute(ExecutionPlan& plan) {
        if (&plan.owner() != this)
            throw std::invalid_argument("execution plan belongs to another backend");
        registry_sealed_ = true;
        execute_plan(plan);
    }

protected:
    virtual void execute_plan(ExecutionPlan& plan) = 0;

    using Executor = std::function<void(detail::QueueState&, const detail::BoundOperation&)>;
    // Register at backend construction, never while executing. Exact keys only;
    // duplicate/foreign/empty keys and invalid schemas are rejected.
    void register_operation(OperationKey, OperationSchema, Executor);
    virtual std::shared_ptr<detail::BufferState> allocate_buffer(std::size_t, std::size_t);
    virtual std::shared_ptr<detail::QueueState> make_queue();
    virtual std::shared_ptr<detail::EventState> record_event(detail::QueueState&);
    virtual void wait_event(detail::QueueState&, detail::EventState&);
    virtual void finish(detail::QueueState&);
    virtual void transfer(detail::QueueState&, const detail::BoundBuffer*,
                          const detail::BoundBuffer*, void*, const void*, std::size_t);
    // Invoked after execution errors, before borrowed resources are released.
    // Failed queues remain poisoned. Native implementations quarantine any
    // allocations whose completion cannot be established.
    virtual void drain(detail::QueueState&) noexcept {}

private:
    struct Registration { OperationKey key; OperationSchema schema; Executor execute; };
    std::vector<Registration> registry_;
    mutable bool registry_sealed_ = false;
    const std::shared_ptr<const void> owner_ = std::make_shared<const int>(0);
    detail::QueueState& checked(const Queue&) const;
    detail::BoundBuffer checked(const BufferSlice&) const;
    void run(detail::QueueState&, const std::function<void()>&);
};

} // namespace strix::session
