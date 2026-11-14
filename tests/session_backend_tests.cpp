#include "strix/session/backend_cpu.hpp"
#include "strix/session/backend_hip.hpp" // Header must compile without a native SDK.

#include <array>
#include <iostream>
#include <limits>
#include <type_traits>

namespace {
namespace ss = strix::session;
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class Error, class F> void rejects(F&& call) {
    try { std::forward<F>(call)(); }
    catch (const Error&) { return; }
    throw std::runtime_error("expected refusal");
}
template<class T, std::size_t N> auto bytes(const std::array<T, N>& values) {
    return std::as_bytes(std::span(values));
}
template<class T, std::size_t N> auto writable_bytes(std::array<T, N>& values) {
    return std::as_writable_bytes(std::span(values));
}
const ss::OperationKey add{"add", "f32", "none", "cpu"};

void transfers_and_ordering() {
    ss::CpuBackend backend;
    auto first = backend.create_queue(), second = backend.create_queue();
    auto a = backend.allocate(64), b = backend.allocate(64, 256);
    const std::array<std::uint32_t, 4> input{7, 19, 31, 41};
    std::array<std::uint32_t, 4> out{};
    backend.upload(first, {a, 12, sizeof(input)}, bytes(input));
    auto uploaded = backend.record(first);
    backend.wait(second, uploaded);
    backend.copy(second, {b, 28, sizeof(input)}, {a, 12, sizeof(input)});
    auto copied = backend.record(second);
    backend.wait(first, copied);
    backend.download(first, {b, 28, sizeof(input)}, writable_bytes(out));
    require(out == input, "offset upload/copy/download and cross-queue dependency");
    backend.synchronize(first);
    backend.synchronize(second);
    // Events represent immutable completed prefixes, not an unrecorded mutable
    // native object. Destroying their source queue does not invalidate them.
    first.reset();
    backend.wait(second, uploaded);
    backend.copy(second, {b, 0, 0}, {a, 64, 0});
    backend.download(second, {a, 64, 0}, {});
    backend.upload(second, {a, 64, 0}, {});
    auto moved = std::move(a);
    require(!a && bool(moved), "move transfers buffer ownership");
    backend.download(second, {moved, 12, sizeof(input)}, writable_bytes(out));
    require(out == input, "moved buffer retains contents");
}

void validation_is_atomic() {
    ss::CpuBackend backend, foreign;
    auto q = backend.create_queue(), other = foreign.create_queue();
    auto a = backend.allocate(32), b = foreign.allocate(32);
    const std::array<std::uint32_t, 4> sentinel{42, 42, 42, 42}, replacement{1, 2, 3, 4};
    std::array<std::uint32_t, 4> out{};
    backend.upload(q, {a, 8, sizeof(sentinel)}, bytes(sentinel));
    auto unchanged = [&] {
        backend.download(q, {a, 8, sizeof(out)}, writable_bytes(out));
        require(out == sentinel, "refusal leaves device destination unchanged");
    };
    rejects<std::out_of_range>([&] { backend.upload(q, {a, 31, sizeof(replacement)}, bytes(replacement)); });
    unchanged();
    const auto huge = std::numeric_limits<std::size_t>::max();
    rejects<std::out_of_range>([&] { backend.copy(q, {a, huge, 2}, {a, 0, 2}); });
    rejects<std::out_of_range>([&] { backend.copy(q, {a, 8, huge}, {a, 0, huge}); });
    rejects<std::invalid_argument>([&] { backend.copy(q, {a, 8, 16}, {a, 0, 16}); });
    rejects<std::invalid_argument>([&] { backend.copy(q, {a, 8, 16}, {a, 0, 8}); });
    rejects<std::invalid_argument>([&] { backend.upload(q, {a, 8, 8}, bytes(replacement)); });
    rejects<std::invalid_argument>([&] { backend.upload(other, {a, 8, 16}, bytes(replacement)); });
    rejects<std::invalid_argument>([&] { backend.copy(q, {a, 8, 16}, {b, 0, 16}); });
    rejects<std::invalid_argument>([&] { backend.upload(q, {b, 0, 16}, bytes(replacement)); });
    unchanged();
    out = sentinel;
    rejects<std::out_of_range>([&] { backend.download(q, {a, 32, 16}, writable_bytes(out)); });
    rejects<std::invalid_argument>([&] { backend.download(q, {b, 0, 16}, writable_bytes(out)); });
    require(out == sentinel, "refusal leaves host destination unchanged");
    ss::Buffer empty;
    rejects<std::invalid_argument>([&] { backend.download(q, {empty, 0, 16}, writable_bytes(out)); });
    auto released = backend.allocate(16);
    const ss::BufferSlice stale{released, 0, 16};
    released.reset();
    rejects<std::invalid_argument>([&] { backend.download(q, stale, writable_bytes(out)); });
    auto e = foreign.record(other);
    rejects<std::invalid_argument>([&] { backend.wait(q, e); });
    ss::Event invalid;
    rejects<std::invalid_argument>([&] { backend.wait(q, invalid); });
    auto own_event = backend.record(q);
    own_event.reset();
    rejects<std::invalid_argument>([&] { backend.wait(q, own_event); });
    ss::Queue invalid_queue;
    rejects<std::invalid_argument>([&] { backend.synchronize(invalid_queue); });
    q.reset();
    rejects<std::invalid_argument>([&] { (void)backend.record(q); });
}

void budgets_and_lifetimes() {
    ss::CpuBackend backend(256);
    const auto initial = backend.memory_budget();
    require(initial.limit == 256 && initial.available == 256 && initial.reserved == 0 &&
        !initial.probe_known && initial.observed_free == 0 && initial.observed_total == 0, "CPU conservative budget");
    auto buffer = backend.allocate(128, 64);
    const auto used = backend.memory_budget();
    require(used.reserved == 191 && used.available == 65, "alignment padding charged to budget");
    rejects<std::length_error>([&] { (void)backend.allocate(66, 1); });
    rejects<std::invalid_argument>([&] { (void)backend.allocate(0); });
    rejects<std::invalid_argument>([&] { (void)backend.allocate(1, 0); });
    rejects<std::invalid_argument>([&] { (void)backend.allocate(1, 3); });
    rejects<std::overflow_error>([&] { (void)backend.allocate(std::numeric_limits<std::size_t>::max(), 64); });
    require(backend.memory_budget().reserved == used.reserved, "failed allocation preserves accounting");
    buffer.reset();
    require(backend.memory_budget().available == 256, "RAII restores budget");
    ss::Buffer orphan;
    ss::Queue orphan_queue;
    ss::Event orphan_event;
    ss::BufferSlice stale;
    {
        ss::CpuBackend temporary;
        orphan = temporary.allocate(16);
        stale = {orphan, 0, 16};
        orphan_queue = temporary.create_queue();
        orphan_event = temporary.record(orphan_queue);
    }
    auto q = backend.create_queue();
    std::array<std::byte, 16> out{};
    rejects<std::invalid_argument>([&] { backend.download(q, stale, out); });
    rejects<std::invalid_argument>([&] { backend.synchronize(orphan_queue); });
    rejects<std::invalid_argument>([&] { backend.wait(q, orphan_event); });
    // Resources safely destroy after their Backend; they retain only the
    // resource implementation/accounting needed for cleanup, never its address.
    orphan.reset();
    orphan_queue.reset();
    orphan_event.reset();
}

void registry_and_arithmetic() {
    ss::CpuBackend backend;
    require(backend.backend_name() == "cpu" && backend.supports(add), "CPU add registered");
    const auto schema = backend.query(add);
    require(schema && schema->inputs == 2 && schema->element_bytes == sizeof(float) &&
        schema->alignment == alignof(float), "query describes typed binding contract");
    auto q = backend.create_queue();
    auto a = backend.allocate(64), b = backend.allocate(64), result = backend.allocate(64);
    const std::array<float, 4> x{1.5F, -2.0F, 0.25F, 100.0F}, y{2.0F, 7.0F, -0.5F, -10.0F};
    const std::array<float, 4> sentinel{99, 99, 99, 99};
    std::array<float, 4> out{};
    backend.upload(q, {a, 4, sizeof(x)}, bytes(x));
    backend.upload(q, {b, 12, sizeof(y)}, bytes(y));
    backend.upload(q, {result, 8, sizeof(sentinel)}, bytes(sentinel));
    ss::Operation operation{add, {{a, 4, sizeof(x)}, {b, 12, sizeof(y)}}, {result, 8, sizeof(out)}, 4};
    for (const auto& key : std::array<ss::OperationKey, 5>{{
        {"missing", "f32", "none", "cpu"}, {"add", "f16", "none", "cpu"},
        {"add", "f32", "unknown", "cpu"}, {"add", "f32", "none", "hip"}, {"", "", "", ""}}}) {
        require(!backend.supports(key) && !backend.query(key), "unknown key fails closed");
        auto refused = operation;
        refused.key = key;
        rejects<ss::UnsupportedOperation>([&] { backend.dispatch(q, refused); });
    }
    auto bad = operation;
    bad.elements = std::numeric_limits<std::size_t>::max();
    rejects<std::overflow_error>([&] { backend.dispatch(q, bad); });
    bad = operation;
    bad.inputs.pop_back();
    rejects<std::invalid_argument>([&] { backend.dispatch(q, bad); });
    bad = operation;
    bad.inputs[1] = {b, 13, sizeof(y)};
    rejects<std::invalid_argument>([&] { backend.dispatch(q, bad); });
    bad = operation;
    bad.inputs[1] = {b, 12, 12};
    rejects<std::invalid_argument>([&] { backend.dispatch(q, bad); });
    bad = operation;
    bad.inputs[1] = {result, 8, sizeof(out)};
    rejects<std::invalid_argument>([&] { backend.dispatch(q, bad); });
    ss::CpuBackend foreign;
    auto foreign_buffer = foreign.allocate(64);
    bad = operation;
    bad.inputs[1] = {foreign_buffer, 0, sizeof(out)};
    rejects<std::invalid_argument>([&] { backend.dispatch(q, bad); });
    backend.download(q, {result, 8, sizeof(out)}, writable_bytes(out));
    require(out == sentinel, "all preflight refusals preserve arithmetic destination");
    backend.dispatch(q, operation);
    backend.download(q, {result, 8, sizeof(out)}, writable_bytes(out));
    require(out == std::array<float, 4>{3.5F, 5.0F, -0.25F, 90.0F}, "registry executes real typed F32 add");
    auto underaligned = backend.allocate(16, 1);
    bad = operation;
    bad.inputs[1] = {underaligned, 0, 16};
    rejects<std::invalid_argument>([&] { backend.dispatch(q, bad); });
    struct Plan final : ss::ExecutionPlan { using ExecutionPlan::ExecutionPlan; } plan(backend);
    rejects<ss::UnsupportedOperation>([&] { backend.execute(plan); });
}

class PlanOnly final : public ss::Backend {
    void execute_plan(ss::ExecutionPlan&) override {}
};
class Failing final : public ss::Backend {
public:
    int executions = 0, drains = 0, completions = 0;
    std::shared_ptr<ss::detail::BufferState> retained;
    Failing() {
        register_operation({"fail", "bytes", "none", "fixture"}, {0, 1, 1},
            [this](ss::detail::QueueState&, const ss::detail::BoundOperation& operation) {
                retained = operation.output.state;
                ++executions;
                throw std::runtime_error("execution failure");
            });
    }
    void register_key(ss::OperationKey key, ss::OperationSchema schema = {}) {
        register_operation(std::move(key), schema, [](ss::detail::QueueState&, const ss::detail::BoundOperation&) {});
    }
    std::string_view backend_name() const noexcept override { return "fixture"; }
private:
    void execute_plan(ss::ExecutionPlan&) override {}
    std::shared_ptr<ss::detail::BufferState> allocate_buffer(std::size_t, std::size_t) override {
        return std::make_shared<ss::detail::BufferState>();
    }
    std::shared_ptr<ss::detail::QueueState> make_queue() override { return std::make_shared<ss::detail::QueueState>(); }
    void finish(ss::detail::QueueState&) override { ++completions; }
    void drain(ss::detail::QueueState&) noexcept override { ++drains; }
};
void compatibility_and_errors() {
    PlanOnly old;
    require(!old.supports(add) && !old.query(add) && old.memory_budget().available == 0, "plan-only default fails closed");
    rejects<ss::UnsupportedOperation>([&] { (void)old.allocate(8); });
    rejects<ss::UnsupportedOperation>([&] { (void)old.create_queue(); });
    Failing backend;
    rejects<std::invalid_argument>([&] { backend.register_key({"fail", "bytes", "none", "fixture"}); });
    rejects<std::invalid_argument>([&] { backend.register_key({"", "bytes", "none", "fixture"}); });
    rejects<std::invalid_argument>([&] { backend.register_key({"new", "bytes", "none", "foreign"}); });
    rejects<std::invalid_argument>([&] { backend.register_key({"new", "bytes", "none", "fixture"}, {0, 0, 1}); });
    rejects<std::invalid_argument>([&] { backend.register_key({"new", "bytes", "none", "fixture"}, {0, 1, 3}); });
    require(backend.supports({"fail", "bytes", "none", "fixture"}), "fixture registration survives invalid entries");
    rejects<std::logic_error>([&] { backend.register_key({"late", "bytes", "none", "fixture"}); });
    auto q = backend.create_queue();
    auto b = backend.allocate(8);
    const ss::Operation failure{{"fail", "bytes", "none", "fixture"}, {}, {b, 0, 8}, 8};
    backend.synchronize(q);
    rejects<std::runtime_error>([&] { backend.dispatch(q, failure); });
    require(backend.executions == 1 && backend.drains == 1 && backend.completions == 1, "error drained before return");
    rejects<std::runtime_error>([&] { backend.dispatch(q, failure); });
    rejects<std::runtime_error>([&] { backend.synchronize(q); });
    require(backend.executions == 1 && backend.drains == 1, "poisoned queue refuses further work");
    b.reset();
    // Native failure paths can retain a state for quarantine after the public
    // owner releases it. Such retention must not revive non-owning slices.
    auto fresh = backend.create_queue();
    rejects<std::invalid_argument>([&] { backend.dispatch(fresh, failure); });
    require(backend.executions == 1 && bool(backend.retained), "released view refused even while implementation retains state");
}
static_assert(!std::is_copy_constructible_v<ss::Buffer> && std::is_nothrow_move_constructible_v<ss::Buffer>);
static_assert(!std::is_copy_constructible_v<ss::Queue> && !std::is_copy_constructible_v<ss::Event>);
}

int main() {
    try {
        transfers_and_ordering();
        validation_is_atomic();
        budgets_and_lifetimes();
        registry_and_arithmetic();
        compatibility_and_errors();
        std::cout << "session backend tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "session backend test failure: " << error.what() << '\n';
        return 1;
    }
}
