#pragma once

#include "strix/execution_buffers.hpp"
#include "strix/execution_io.hpp"
#include "strix/model_contract.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace strix {
namespace core { struct RowLayoutPolicy; }  // src/core/row_layout.hpp

struct ExecutionArenaAllocation {
    ExecutionBufferUse use;
    std::uint32_t first_step = 0;
    std::uint32_t last_step = 0;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
};

struct ExecutionArenaPlan {
    std::uint64_t alignment = 1;
    std::uint64_t total_bytes = 0;
    std::uint64_t naive_nonaliased_bytes = 0;
    std::vector<ExecutionArenaAllocation> allocations;
};

// Two, four or eight known inputs share one physical arena. Each row view retains the exact
// serial payload sizes and uses; only offsets change. Strides are indexed in
// the same order as physical.allocations, including lifetime-reused regions.
struct KnownWindowArena {
    ExecutionArenaPlan physical;
    std::vector<std::uint64_t> row_stride;
    std::vector<ExecutionArenaPlan> row_view;
};

[[nodiscard]] KnownWindowArena build_known_window_arena(
    const ExecutionArenaPlan& serial, std::uint64_t alignment = 256, std::uint32_t logical_rows = 2);
// Separate bounded prefill geometry. Does not widen the closed window API.
// Terminal mode repacks headless rows and appends one logits stripe. Logits have
// stride zero and zero payload in omitted row views; only the last row exposes
// the stripe. materialize_views=false builds physical geometry for validation.
[[nodiscard]] KnownWindowArena build_known_prefill_arena(
    const ExecutionArenaPlan& serial, std::uint64_t alignment, std::uint32_t logical_rows,
    std::uint64_t external_qsa_score_bytes = 0, bool terminal_logits = false, bool materialize_views = true,
    std::uint64_t compact_qsa_output_savings = 0, const core::RowLayoutPolicy* row_layout = nullptr);
// Row strides come from the shared row layout policy (core/row_layout.hpp): the payload rounded to `alignment`, moved
// off page multiples. row_layout == nullptr uses the process policy (core::row_layout_policy()).

[[nodiscard]] ExecutionArenaPlan build_serial_execution_arena_plan(
    const ModelContract& model,
    const ExecutionBufferRequest& buffer_request,
    const SerialExecutionPlan& execution_plan,
    const SerialExecutionIoPlan& io_plan,
    std::uint64_t alignment = 256);

[[nodiscard]] const ExecutionArenaAllocation* find_arena_allocation(
    const ExecutionArenaPlan& plan,
    ExecutionBufferUse use) noexcept;

// Index of plans an owner keeps alive and unmodified for the index's lifetime (a compiled known
// window's row views). Built once; used by the ArenaLookupScopes opened with it.
class ArenaLookupIndex {
public:
    ArenaLookupIndex() = default;
    explicit ArenaLookupIndex(std::span<const ExecutionArenaPlan* const> plans);
    // Row views of one compiled arena (a tiled prefill's T rows): when every view holds the same uses in the same
    // order, one use -> position map answers them all (the first allocation of the use, as the linear search
    // returns); otherwise an empty index. Kept views must stay alive and unmodified for the index's lifetime.
    [[nodiscard]] static ArenaLookupIndex uniform_views(std::span<const ExecutionArenaPlan> views);
    [[nodiscard]] bool empty() const noexcept { return !plans_ && !uniform_; }

private:
    friend class ArenaLookupScope;
    std::shared_ptr<const void> plans_;
    std::shared_ptr<const void> uniform_;
};

// While one is alive, find_arena_allocation on this thread indexes each plan it is asked about once
// (first allocation of every use, as the linear search returns) and answers later lookups of that
// plan from the index. The owner must keep the plans it looks up alive and unmodified for the
// scope's lifetime (one admitted prefill block's launch: about 2000 lookups per row).
// Opened with a persistent index, plans of that index are answered from it without re-indexing
// (a plan whose allocation storage or size no longer matches is searched linearly); other plans
// are indexed per scope as above.
class ArenaLookupScope {
public:
    ArenaLookupScope();
    explicit ArenaLookupScope(const ArenaLookupIndex& persistent);
    ~ArenaLookupScope();
    ArenaLookupScope(const ArenaLookupScope&) = delete;
    ArenaLookupScope& operator=(const ArenaLookupScope&) = delete;

private:
    void* previous_;
};

}  // namespace strix
