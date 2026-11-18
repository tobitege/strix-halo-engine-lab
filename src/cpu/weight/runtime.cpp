#include <strix/weight/cache.hpp>
#include <strix/weight/source.hpp>
#include <cstdlib>
#include <limits>
namespace strix::weight {
ExecutionContext cpu_execution_context() {
    static const auto callbacks = [] {
        auto cb = std::make_shared<RuntimeCallbacks>();
        cb->allocate = [](std::uint64_t bytes, void*& p) -> Status {
            if (bytes > std::numeric_limits<std::size_t>::max()) return {Error::memory_budget, "host allocation size overflow"};
            p = std::malloc(static_cast<std::size_t>(bytes ? bytes : 1));
            return p ? Status{Error::ok, {}} : Status{Error::memory_budget, "host image allocation failed"};
        };
        cb->free = [](void* p) { std::free(p); };
        cb->record_event = [](ExecutionContext&, void*& event) -> Status { event = nullptr; return {Error::ok, {}}; };
        cb->wait_event = [](ExecutionContext&, void*) -> Status { return {Error::ok, {}}; };
        cb->event_complete = [](void*) { return true; };
        cb->destroy_event = [](void*) {};
        cb->retire = [](void* p, Retirement, ExecutionContext) { std::free(p); };
        cb->convert = [](const WeightView& w, const Plan& plan, void* p, ExecutionContext&) -> Status {
            if (plan.selected.strategy != Strategy::native || plan.selected.resources.image_bytes != w.payload.range.bytes)
                return {Error::unsupported_arithmetic, "CPU conversion source is not registered"};
            return w.payload.source->read(w.payload.range, {static_cast<Byte*>(p), static_cast<std::size_t>(w.payload.range.bytes)});
        };
        return cb;
    }();
    return {Backend::cpu, nullptr, 0, callbacks};
}
}
