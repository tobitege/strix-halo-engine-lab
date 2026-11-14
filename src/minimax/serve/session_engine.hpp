#pragma once

#include "strix/session/backend.hpp"
#include "strix/session/engine.hpp"
#include "strix/minimax/prefill_types.hpp"
#include "minimax/dspark/minimax_dspark.hpp"
#include "paired_state.hpp"
#include "minimax_control.hpp"
#include <array>

namespace strix::minimax::serve {
struct SessionDraft {
    std::array<std::uint32_t, dspark::block> tokens{};
    std::array<float, dspark::block> confidence{};
};

// One synchronous native operation, retaining the native grouped/layer shape.
// Control exchange and scheduling policy remain outside backend resources.
struct SessionPlan final : strix::session::ExecutionPlan {
    enum class Kind { begin, unit, step, draft, capture, restore, input_admit, input } kind;
    SessionPlan(strix::session::Backend& b, Kind k) : ExecutionPlan(b), kind(k) {}
    std::uint32_t slot = 0, layers = 0, append = 0, anchor = 0;
    std::uint64_t task = 0, parent = 0, generation = 0, pos = 0, resident = 0;
    bool context = false, done = false;
    std::span<const std::uint32_t> prompt;
    std::span<const engine::Row> rows;
    std::span<const strix::session::KvHandle> bindings;
    std::vector<std::uint32_t> next;
    SessionDraft drafts;
    paired::Frontier frontier;
    std::array<strix::session::RankState,2> state;
    std::uint32_t rank = 0;
    paired::InputProfile input_profile = paired::InputProfile::off;
    paired::Identity native_identity;
};

template<class Runner> class SessionBackend final : public strix::session::Backend {
public:
    explicit SessionBackend(Runner& runner) : runner_(runner) {}
    std::string_view backend_name() const noexcept override { return "minimax-native"; }
protected:
    void execute_plan(strix::session::ExecutionPlan& base) override {
        auto* p = dynamic_cast<SessionPlan*>(&base);
        if (!p) throw strix::session::UnsupportedOperation("unsupported MiniMax plan");
        switch (p->kind) {
        case SessionPlan::Kind::begin:
            resident_begin(runner_, p->slot, p->prompt, p->context, p->task, p->parent, p->generation, p->resident); break;
        case SessionPlan::Kind::unit: {
            std::uint32_t next = 0;
            p->done = runner_.prefill_unit(p->layers, &next);
            if (p->done) p->next.push_back(next);
            break;
        }
        case SessionPlan::Kind::step:
            p->next = runner_.step(p->rows, p->append);
            if (p->append) runner_.synchronize();
            break;
        case SessionPlan::Kind::draft: {
            const auto d = runner_.draft(p->slot, p->anchor, p->pos);
            p->drafts.tokens = d.tokens; p->drafts.confidence = d.confidence; break;
        }
        case SessionPlan::Kind::input_admit:
            if constexpr (requires { runner_.input_supported(p->input_profile); runner_.state(*p);
                                     runner_.input(p->rows,p->bindings); p->native_identity=runner_.input_identity(); }) {
                if (!runner_.input_supported(p->input_profile))
                    throw strix::session::UnsupportedOperation("MiniMax native input profile unavailable");
                p->native_identity=runner_.input_identity();
            } else throw strix::session::UnsupportedOperation("MiniMax runner has no input profile");
            break;
        case SessionPlan::Kind::input:
            if constexpr (requires { runner_.input(p->rows,p->bindings); })
                p->next=runner_.input(p->rows,p->bindings);
            else throw strix::session::UnsupportedOperation("MiniMax runner has no native input support");
            break;
        case SessionPlan::Kind::capture: case SessionPlan::Kind::restore:
            if constexpr (requires { runner_.state(*p); }) runner_.state(*p);
            else throw strix::session::UnsupportedOperation("MiniMax runner has no paired state support");
            break;
        }
    }
private:
    Runner& runner_;
};

// Binding and canonical history for the native planner. Partial layer work
// advances a bounded checkpoint only; no input token is committed until the
// complete native prefill has finished. Generic eager paired snapshots require
// explicit typed activation and a native state-capable runner; defaults refuse.
class SessionEngine final : public strix::session::Engine {
public:
    SessionEngine(strix::session::Backend&, std::uint32_t slots, std::uint64_t capacity,
                  std::function<void(strix::session::KvHandle)> release,
                  std::function<engine::PfProgress()> progress, bool paired_state = false,
                  paired::InputProfile input_profile = paired::InputProfile::off);
    strix::session::Hooks storage_hooks();
    void attach(strix::session::KvStore& store) noexcept { store_ = &store; }
    void begin(strix::session::KvHandle, std::span<const std::uint32_t>, bool,
               std::uint64_t task, std::uint64_t parent, std::uint64_t resident = 0);
    bool unit(strix::session::KvHandle, unsigned layers, std::uint32_t* next);
    std::vector<std::uint32_t> step(std::span<const engine::Row>, std::uint32_t append,
                                    std::span<const strix::session::KvHandle>);
    SessionDraft draft(strix::session::KvHandle, std::uint32_t anchor, std::uint64_t pos);
    // Commit only the prefix actually published, including a pending stop ID.
    void commit(strix::session::KvHandle, std::uint32_t);
    void observe_named(strix::session::KvHandle, std::span<const std::uint32_t>, std::optional<std::uint32_t>);
    std::uint64_t checkpoint(strix::session::KvHandle) const;
    void prefill(std::span<const std::uint32_t>, strix::session::KvHandle&) override;
    strix::session::StepResult decode(strix::session::KvHandle&, const strix::session::SamplerState&) override;
    void prefill_batch(std::span<const strix::session::PrefillItem>) override;
    std::vector<strix::session::StepResult> decode_batch(std::span<const strix::session::DecodeItem>) override;
    std::vector<strix::session::AdvanceResult> advance(std::span<const strix::session::AdvanceItem>) override;
    void append_external(std::span<const std::uint32_t>, strix::session::KvHandle&) override;
    void commit_pending(strix::session::KvHandle&) override;
    void cancel_lane(strix::session::KvHandle&) override;
    void resume_lane(strix::session::KvHandle&) override;
    strix::profile::ExecutionReceipt receipt(strix::session::KvHandle) const;
    strix::session::Capabilities caps() const override;
    strix::profile::ExecutionReceipt receipt() const override;
private:
    struct Binding {
        strix::session::KvHandle handle;
        std::vector<std::uint32_t> prompt;
        std::uint64_t progress = 0, progress_limit = 0;
        bool open = false;
        std::optional<std::uint32_t> retained;
        strix::session::FinishReason finish = strix::session::FinishReason::none;
    };
    Binding& checked(strix::session::KvHandle);
    const Binding& checked(strix::session::KvHandle) const;
    strix::session::Backend& backend_;
    std::vector<Binding> bindings_;
    std::uint64_t capacity_;
    std::function<void(strix::session::KvHandle)> release_;
    std::function<engine::PfProgress()> progress_;
    strix::session::KvStore* store_ = nullptr;
    bool paired_state_ = false;
    paired::InputProfile input_profile_ = paired::InputProfile::off;
    void require_input() const;
    void validate_input(strix::session::KvHandle, std::span<const std::uint32_t>, bool terminal = false) const;
    static void validate_sampler(const strix::session::SamplerState&);
    struct InputWork {
        strix::session::KvHandle handle;
        std::vector<std::uint32_t> tokens;
        std::optional<std::uint32_t> retained;
    };
    // Store/caller publication is deferred until ALL grouped cycles succeed.
    void consume(std::span<InputWork>, const std::function<void()>& before_rows = {}, bool single_group = false);
    paired::Frontier captured_frontier_;
    std::array<strix::session::RankState,2> captured_;
    strix::session::RankState capture_rank(strix::session::KvHandle, std::uint32_t, bool);
};
} // namespace strix::minimax::serve
