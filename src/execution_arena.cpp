#include "strix/execution_arena.hpp"
#include "strix/known_prefill_limits.hpp"

#include "core/row_layout.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace strix {
namespace {

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right) {
    if (right >
        std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(
            "execution arena addition overflow");
    }
    return left + right;
}


std::uint64_t align_up(
    std::uint64_t value,
    std::uint64_t alignment) {
    if (alignment == 0) {
        throw std::invalid_argument(
            "execution arena alignment must be non-zero");
    }
    const auto remainder = value % alignment;
    return remainder == 0
        ? value
        : checked_add(
              value,
              alignment - remainder);
}

bool same_use(
    ExecutionBufferUse left,
    ExecutionBufferUse right) noexcept {
    return left == right;
}

std::uint64_t bytes_for(
    const ExecutionBufferPlan& sizes,
    ExecutionBufferClass buffer_class) {
    switch (buffer_class) {
    case ExecutionBufferClass::expanded_hidden:
        return sizes.expanded_hidden_bytes;
    case ExecutionBufferClass::hidden:
        return sizes.hidden_bytes;
    case ExecutionBufferClass::injection_weights:
        return sizes.injection_weight_bytes;
    case ExecutionBufferClass::route_indices:
        return sizes.route_index_bytes;
    case ExecutionBufferClass::route_weights:
        return sizes.route_weight_bytes;
    case ExecutionBufferClass::logits:
        return sizes.logits_bytes;
    case ExecutionBufferClass::mlp_intermediate:
        return sizes.mlp_intermediate_bytes;
    case ExecutionBufferClass::gdn_workspace:
        return sizes.gdn_workspace_bytes;
    case ExecutionBufferClass::ple_workspace:
        return sizes.ple_workspace_bytes;
    case ExecutionBufferClass::indexed_attention_workspace:
        return sizes.indexed_attention_workspace_bytes;
    case ExecutionBufferClass::hyper_workspace:
        return sizes.hyper_workspace_bytes;
    case ExecutionBufferClass::router_workspace:
        return sizes.router_workspace_bytes;
    }
    throw std::logic_error(
        "unknown execution buffer class");
}


}  // namespace

ExecutionArenaPlan build_serial_execution_arena_plan(
    const ModelContract& model,
    const ExecutionBufferRequest& buffer_request,
    const SerialExecutionPlan& execution_plan,
    const SerialExecutionIoPlan& io_plan,
    std::uint64_t alignment) {
    if (alignment == 0) {
        throw std::invalid_argument(
            "execution arena alignment must be non-zero");
    }

    validate_serial_execution_io_plan(
        execution_plan, io_plan);

    if ((io_plan.router_workspace_slots != 0U) != buffer_request.parallel_moe_router)
        throw std::invalid_argument("router workspace admission differs from buffer request");
    if (buffer_request.parallel_moe_router && alignment % 16U != 0)
        throw std::invalid_argument("parallel router arena requires at least 16-byte alignment");

    if (buffer_request.q4k_a8g32_gate_up && alignment % 256U != 0)
        throw std::invalid_argument("Q4K A8 runtime arena requires 256-byte alignment");

    const auto sizes =
        build_serial_execution_buffer_plan(
            model, buffer_request);

    ExecutionArenaPlan plan;
    plan.alignment = alignment;

    for (std::size_t step_index = 0;
         step_index < io_plan.steps.size();
         ++step_index) {
        const auto& io = io_plan.steps[step_index];

        for (const auto use : io.writes) {
            const auto found =
                std::find_if(
                    plan.allocations.begin(),
                    plan.allocations.end(),
                    [&](const ExecutionArenaAllocation& allocation) {
                        return same_use(
                            allocation.use, use);
                    });
            if (found != plan.allocations.end()) {
                throw std::invalid_argument(
                    "execution buffer generation is written more than once");
            }

            const auto bytes =
                bytes_for(
                    sizes,
                    use.buffer.buffer_class);
            if (bytes == 0) {
                throw std::invalid_argument(
                    "execution arena buffer size must be non-zero");
            }

            plan.allocations.push_back(
                ExecutionArenaAllocation{
                    use,
                    static_cast<std::uint32_t>(step_index),
                    static_cast<std::uint32_t>(step_index),
                    0,
                    bytes,
                });
            plan.naive_nonaliased_bytes =
                checked_add(
                    plan.naive_nonaliased_bytes,
                    align_up(bytes, alignment));
        }

        for (const auto use : io.reads) {
            const auto found =
                std::find_if(
                    plan.allocations.begin(),
                    plan.allocations.end(),
                    [&](const ExecutionArenaAllocation& allocation) {
                        return same_use(
                            allocation.use, use);
                    });
            if (found == plan.allocations.end()) {
                throw std::invalid_argument(
                    "execution arena read has no producing generation");
            }
            found->last_step =
                std::max(
                    found->last_step,
                    static_cast<std::uint32_t>(
                        step_index));
        }

        for (const auto use : io.temporaries) {
            const auto found =
                std::find_if(
                    plan.allocations.begin(),
                    plan.allocations.end(),
                    [&](const ExecutionArenaAllocation& allocation) {
                        return same_use(
                            allocation.use, use);
                    });
            if (found != plan.allocations.end()) {
                throw std::invalid_argument(
                    "execution temporary generation is allocated more than once");
            }

            const auto bytes =
                bytes_for(
                    sizes,
                    use.buffer.buffer_class);
            if (bytes == 0) {
                throw std::invalid_argument(
                    "execution temporary size must be non-zero");
            }

            plan.allocations.push_back(
                ExecutionArenaAllocation{
                    use,
                    static_cast<std::uint32_t>(step_index),
                    static_cast<std::uint32_t>(step_index),
                    0,
                    bytes,
                });
            plan.naive_nonaliased_bytes =
                checked_add(
                    plan.naive_nonaliased_bytes,
                    align_up(bytes, alignment));
        }
    }

    std::stable_sort(
        plan.allocations.begin(),
        plan.allocations.end(),
        [](const auto& left, const auto& right) {
            if (left.first_step != right.first_step) {
                return left.first_step <
                       right.first_step;
            }
            if (left.bytes != right.bytes) {
                return left.bytes > right.bytes;
            }
            if (left.use.buffer.buffer_class !=
                right.use.buffer.buffer_class) {
                return left.use.buffer.buffer_class <
                       right.use.buffer.buffer_class;
            }
            if (left.use.buffer.slot !=
                right.use.buffer.slot) {
                return left.use.buffer.slot <
                       right.use.buffer.slot;
            }
            return left.use.generation <
                   right.use.generation;
        });

    std::vector<const ExecutionArenaAllocation*> active;

    for (auto& allocation : plan.allocations) {
        active.erase(
            std::remove_if(
                active.begin(),
                active.end(),
                [&](const ExecutionArenaAllocation* other) {
                    return other->last_step <
                           allocation.first_step;
                }),
            active.end());

        std::sort(
            active.begin(),
            active.end(),
            [](const auto* left, const auto* right) {
                return left->offset < right->offset;
            });

        std::uint64_t candidate = 0;
        for (const auto* other : active) {
            candidate =
                align_up(candidate, alignment);
            const auto candidate_end =
                checked_add(
                    candidate,
                    allocation.bytes);
            const auto other_end =
                checked_add(
                    other->offset,
                    other->bytes);

            if (candidate_end <= other->offset) {
                break;
            }
            if (candidate >= other_end) {
                continue;
            }

            candidate = other_end;
        }

        allocation.offset =
            align_up(candidate, alignment);

        const auto end =
            checked_add(
                allocation.offset,
                allocation.bytes);
        plan.total_bytes =
            std::max(
                plan.total_bytes,
                end);
        active.push_back(&allocation);
    }

    plan.total_bytes =
        align_up(
            plan.total_bytes,
            alignment);
    return plan;
}

namespace {
KnownWindowArena build_known_rows_arena(
    const ExecutionArenaPlan& serial, std::uint64_t alignment, std::uint32_t logical_rows,
    std::uint64_t external_qsa_score_bytes = 0, bool terminal_logits = false, bool materialize_views = true, std::uint64_t compact_qsa_output_savings = 0,
    const core::RowLayoutPolicy& layout = core::row_layout_policy()) {
    if (!serial.alignment || !alignment || alignment % serial.alignment != 0 ||
        !serial.total_bytes || serial.allocations.empty()) {
        throw std::invalid_argument("known window requires a complete aligned serial arena");
    }
    for (std::size_t i = 0; i < serial.allocations.size(); ++i) {
        const auto& current = serial.allocations[i];
        if (!current.bytes || current.first_step > current.last_step ||
            current.offset % serial.alignment != 0 ||
            checked_add(current.offset, current.bytes) > serial.total_bytes) {
            throw std::invalid_argument("known window serial allocation is invalid");
        }
        for (std::size_t j = 0; j < i; ++j) {
            const auto& previous = serial.allocations[j];
            if (current.use == previous.use) {
                throw std::invalid_argument("known window serial use is duplicated");
            }
            const bool live = current.first_step <= previous.last_step &&
                              previous.first_step <= current.last_step;
            if (live && current.offset < checked_add(previous.offset, previous.bytes) &&
                previous.offset < checked_add(current.offset, current.bytes)) {
                throw std::invalid_argument("known window source aliases live serial allocations");
            }
        }
    }
    KnownWindowArena result;
    result.physical.alignment = alignment;
    result.physical.allocations = serial.allocations;
    ExecutionArenaAllocation terminal;
    if(terminal_logits) {
        const auto it=std::find_if(result.physical.allocations.begin(),result.physical.allocations.end(),
            [](const auto& a){return execution_use_is_logits(a.use);});
        if(it==result.physical.allocations.end()) throw std::invalid_argument("terminal arena requires a logits generation");
        terminal=*it;result.physical.allocations.erase(it);
        if(std::any_of(result.physical.allocations.begin(),result.physical.allocations.end(),
            [](const auto& a){return execution_use_is_logits(a.use);}))
            throw std::invalid_argument("terminal arena requires exactly one logits generation");
    }
    for (auto& allocation : result.physical.allocations) {
        if ((external_qsa_score_bytes || compact_qsa_output_savings) && allocation.use.buffer.buffer_class == ExecutionBufferClass::indexed_attention_workspace) {
            const auto removed=checked_add(external_qsa_score_bytes ? align_up(external_qsa_score_bytes,alignment) : 0U, compact_qsa_output_savings);
            if (allocation.bytes<=removed) throw std::invalid_argument("score-free QSA workspace is undersized");
            allocation.bytes-=removed;
        }
        // The shared row layout policy decides the physical stride: the payload rounded to the arena alignment,
        // moved off page multiples for row counts that kernels walk in lockstep (core/row_layout.hpp).
        const auto stride = core::physical_row_stride(layout, align_up(allocation.bytes, alignment), logical_rows,
            alignment < core::kRowAlignment ? core::kRowAlignment : alignment);
        allocation.bytes = 0;
        for(std::uint32_t row=0;row<logical_rows;++row) allocation.bytes=checked_add(allocation.bytes,stride);
        allocation.offset = 0;
        result.physical.naive_nonaliased_bytes = checked_add(
            result.physical.naive_nonaliased_bytes, allocation.bytes);
    }
    std::stable_sort(result.physical.allocations.begin(), result.physical.allocations.end(),
        [](const auto& left, const auto& right) {
            if (left.first_step != right.first_step) return left.first_step < right.first_step;
            if (left.bytes != right.bytes) return left.bytes > right.bytes;
            if (left.use.buffer.buffer_class != right.use.buffer.buffer_class)
                return left.use.buffer.buffer_class < right.use.buffer.buffer_class;
            if (left.use.buffer.slot != right.use.buffer.slot)
                return left.use.buffer.slot < right.use.buffer.slot;
            return left.use.generation < right.use.generation;
        });
    std::vector<const ExecutionArenaAllocation*> active;
    for (auto& allocation : result.physical.allocations) {
        active.erase(std::remove_if(active.begin(), active.end(), [&](const auto* previous) {
            return previous->last_step < allocation.first_step;
        }), active.end());
        std::sort(active.begin(), active.end(), [](const auto* left, const auto* right) {
            return left->offset < right->offset;
        });
        std::uint64_t candidate = 0;
        for (const auto* previous : active) {
            candidate = align_up(candidate, alignment);
            if (checked_add(candidate, allocation.bytes) <= previous->offset) break;
            const auto previous_end = checked_add(previous->offset, previous->bytes);
            if (candidate < previous_end) candidate = previous_end;
        }
        allocation.offset = align_up(candidate, alignment);
        result.physical.total_bytes = std::max(result.physical.total_bytes,
            checked_add(allocation.offset, allocation.bytes));
        active.push_back(&allocation);
    }
    result.physical.total_bytes = align_up(result.physical.total_bytes, alignment);
    if(terminal_logits) {
        terminal.offset=result.physical.total_bytes;
        terminal.bytes=align_up(terminal.bytes,alignment);
        result.physical.total_bytes=checked_add(result.physical.total_bytes,terminal.bytes);
        result.physical.naive_nonaliased_bytes=checked_add(result.physical.naive_nonaliased_bytes,terminal.bytes);
        result.physical.allocations.push_back(terminal);
    }
    if(materialize_views) result.row_view.assign(logical_rows,result.physical);
    // STRIX_ROW_LAYOUT_VERBOSE=1: one line per row buffer (payload, physical stride, rows until a page offset repeats).
    static const bool layout_verbose = envcfg::flag("STRIX_ROW_LAYOUT_VERBOSE", false);
    for (std::size_t i = 0; i < result.physical.allocations.size(); ++i) {
        const auto& physical = result.physical.allocations[i];
        const auto* original = find_arena_allocation(serial, physical.use);
        if (!original) throw std::logic_error("known window lost a serial allocation");
        const bool terminal_use=terminal_logits && execution_use_is_logits(physical.use);
        const auto stride = terminal_use?0U:physical.bytes / logical_rows;
        result.row_stride.push_back(stride);
        if (layout_verbose)
            std::fprintf(stderr, "row layout: rows %u class %u slot %u payload %llu stride %llu page-period %llu\n",
                logical_rows, static_cast<unsigned>(physical.use.buffer.buffer_class), physical.use.buffer.slot,
                static_cast<unsigned long long>(original->bytes), static_cast<unsigned long long>(stride),
                static_cast<unsigned long long>(core::row_alias_period(stride)));
        for (std::size_t row = 0; row < result.row_view.size(); ++row) {
            auto& allocation = result.row_view[row].allocations[i];
            allocation.offset = checked_add(physical.offset, static_cast<std::uint64_t>(row)*stride);
            allocation.bytes = terminal_use && row+1U!=logical_rows?0U:original->bytes;
            if ((external_qsa_score_bytes || compact_qsa_output_savings) && allocation.use.buffer.buffer_class == ExecutionBufferClass::indexed_attention_workspace)
                allocation.bytes-=checked_add(external_qsa_score_bytes ? align_up(external_qsa_score_bytes,alignment) : 0U, compact_qsa_output_savings);
        }
    }
    return result;
}
} // namespace

KnownWindowArena build_known_window_arena(
    const ExecutionArenaPlan& serial,std::uint64_t alignment,std::uint32_t logical_rows) {
    if (logical_rows!=2U && logical_rows!=4U && logical_rows!=8U)
        throw std::invalid_argument("known window rows must be two, four or eight");
    return build_known_rows_arena(serial,alignment,logical_rows);
}
KnownWindowArena build_known_prefill_arena(
    const ExecutionArenaPlan& serial,std::uint64_t alignment,std::uint32_t logical_rows,
    std::uint64_t external_qsa_score_bytes, bool terminal_logits, bool materialize_views, std::uint64_t compact_qsa_output_savings,
    const core::RowLayoutPolicy* row_layout) {
    if(logical_rows<16U||logical_rows>known_prefill_max_rows)
        throw std::invalid_argument("known prefill rows must be within "+known_prefill_rows_range());
    return build_known_rows_arena(serial,alignment,logical_rows,external_qsa_score_bytes,terminal_logits,materialize_views,compact_qsa_output_savings,
        row_layout ? *row_layout : core::row_layout_policy());
}

namespace {
struct UseKey {
    ExecutionBufferUse use;
    friend bool operator==(const UseKey&, const UseKey&) = default;
};
struct UseHash {
    std::size_t operator()(const UseKey& key) const noexcept {
        auto h = key.use.generation * 0x9E3779B97F4A7C15ULL;
        h ^= (static_cast<std::uint64_t>(key.use.buffer.slot) << 8U) ^ static_cast<std::uint64_t>(key.use.buffer.buffer_class);
        return static_cast<std::size_t>(h ^ (h >> 29U));
    }
};
struct PlanIndex {
    const ExecutionArenaAllocation* data = nullptr;
    std::size_t size = 0;
    std::unordered_map<UseKey, std::size_t, UseHash> first;
};
using LookupCache = std::unordered_map<const ExecutionArenaPlan*, PlanIndex>;
// Views with identical use sequences: one first-position map, and each view's allocation storage and size.
struct UniformViews {
    std::unordered_map<UseKey, std::size_t, UseHash> first;
    std::unordered_map<const ExecutionArenaPlan*, std::pair<const ExecutionArenaAllocation*, std::size_t>> plans;
};
struct ActiveLookup {
    const LookupCache* persistent = nullptr;
    const UniformViews* uniform = nullptr;
    LookupCache transient;
};
thread_local ActiveLookup* active_lookup = nullptr;

void index_plan(PlanIndex& index, const ExecutionArenaPlan& plan) {
    index.first.clear();
    for (std::size_t i = 0; i < plan.allocations.size(); ++i)
        index.first.try_emplace(UseKey{plan.allocations[i].use}, i);
    index.data = plan.allocations.data();
    index.size = plan.allocations.size();
}

bool index_matches(const PlanIndex& index, const ExecutionArenaPlan& plan) noexcept {
    return index.data == plan.allocations.data() && index.size == plan.allocations.size();
}

// Answer of a matching index; false when its hit no longer holds that use (plan changed in place).
bool indexed_answer(const PlanIndex& index, const ExecutionArenaPlan& plan, ExecutionBufferUse use,
    const ExecutionArenaAllocation*& out) {
    const auto found = index.first.find(UseKey{use});
    if (found == index.first.end()) { out = nullptr; return true; }
    out = &plan.allocations[found->second];
    return out->use == use;
}

const ExecutionArenaAllocation* indexed_allocation(
    LookupCache& cache, const ExecutionArenaPlan& plan, ExecutionBufferUse use) {
    auto& index = cache[&plan];
    // First lookup of this plan (or another plan now at its address): index it.
    if (!index_matches(index, plan)) index_plan(index, plan);
    const ExecutionArenaAllocation* allocation = nullptr;
    if (!indexed_answer(index, plan, use, allocation)) throw std::logic_error("arena lookup index is stale");
    return allocation;
}
}  // namespace

ArenaLookupIndex::ArenaLookupIndex(std::span<const ExecutionArenaPlan* const> plans) {
    auto cache = std::make_shared<LookupCache>();
    for (const auto* plan : plans)
        if (plan) index_plan((*cache)[plan], *plan);
    plans_ = std::move(cache);
}

ArenaLookupScope::ArenaLookupScope() : previous_(active_lookup) { active_lookup = new ActiveLookup; }

ArenaLookupIndex ArenaLookupIndex::uniform_views(std::span<const ExecutionArenaPlan> views) {
    ArenaLookupIndex index;
    if (views.empty()) return index;
    const auto& lead = views.front().allocations;
    for (const auto& view : views) {
        if (view.allocations.size() != lead.size()) return index;
        for (std::size_t i = 0; i < lead.size(); ++i)
            if (!(view.allocations[i].use == lead[i].use)) return index;
    }
    auto uniform = std::make_shared<UniformViews>();
    for (std::size_t i = 0; i < lead.size(); ++i) uniform->first.try_emplace(UseKey{lead[i].use}, i);
    uniform->plans.reserve(views.size());
    for (const auto& view : views) uniform->plans.emplace(&view, std::pair{view.allocations.data(), view.allocations.size()});
    index.uniform_ = std::move(uniform);
    return index;
}

ArenaLookupScope::ArenaLookupScope(const ArenaLookupIndex& persistent) : previous_(active_lookup) {
    active_lookup = new ActiveLookup;
    active_lookup->persistent = static_cast<const LookupCache*>(persistent.plans_.get());
    active_lookup->uniform = static_cast<const UniformViews*>(persistent.uniform_.get());
}

ArenaLookupScope::~ArenaLookupScope() {
    delete active_lookup;
    active_lookup = static_cast<ActiveLookup*>(previous_);
}

const ExecutionArenaAllocation* find_arena_allocation(
    const ExecutionArenaPlan& plan,
    ExecutionBufferUse use) noexcept {
    if (active_lookup) {
        // A uniform view still at its indexed storage and size: its use sits where it sits in every view.
        if (const auto* uniform = active_lookup->uniform) {
            const auto view = uniform->plans.find(&plan);
            if (view != uniform->plans.end() && view->second.first == plan.allocations.data() &&
                view->second.second == plan.allocations.size()) {
                const auto found = uniform->first.find(UseKey{use});
                if (found == uniform->first.end()) return nullptr;
                const auto* allocation = &plan.allocations[found->second];
                if (allocation->use == use) return allocation;
            }
        }
        const auto* persistent = active_lookup->persistent;
        const auto fixed = persistent ? persistent->find(&plan) : LookupCache::const_iterator{};
        if (persistent && fixed != persistent->end()) {
            // An owner's plan: answered from its index while that still matches, else searched.
            const ExecutionArenaAllocation* allocation = nullptr;
            if (index_matches(fixed->second, plan) && indexed_answer(fixed->second, plan, use, allocation))
                return allocation;
        } else {
            try {
                return indexed_allocation(active_lookup->transient, plan, use);
            } catch (...) {
                // Out of memory while indexing, or a plan changed in place: drop its index and search.
                active_lookup->transient.erase(&plan);
            }
        }
    }
    const auto found =
        std::find_if(
            plan.allocations.begin(),
            plan.allocations.end(),
            [&](const ExecutionArenaAllocation& allocation) {
                return allocation.use == use;
            });
    return found == plan.allocations.end()
        ? nullptr
        : &*found;
}

}  // namespace strix
