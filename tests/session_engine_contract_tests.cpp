// Include each public header before test support; neither needs family/HIP code.
#include "strix/session/backend.hpp"
#include "strix/session/engine.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

namespace {
namespace ss = strix::session;

static_assert(std::is_same_v<ss::SamplingParameters, strix::serve::SamplingParameters>);
static_assert(std::is_same_v<decltype(ss::StepResult{}.tokens)::value_type, std::uint32_t>);
static_assert(std::is_same_v<decltype(&ss::Engine::prefill),
    void (ss::Engine::*)(std::span<const std::uint32_t>, ss::KvHandle&)>);
static_assert(std::is_same_v<decltype(&ss::Engine::decode),
    ss::StepResult (ss::Engine::*)(ss::KvHandle&, const ss::SamplerState&)>);
static_assert(std::is_same_v<decltype(&ss::Engine::receipt),
    strix::profile::ExecutionReceipt (ss::Engine::*)() const>);
static_assert(std::has_virtual_destructor_v<ss::Engine>);
static_assert(std::has_virtual_destructor_v<ss::Backend>);
static_assert(std::has_virtual_destructor_v<ss::ExecutionPlan>);
static_assert(!std::is_copy_constructible_v<ss::Engine>);
static_assert(!std::is_copy_constructible_v<ss::Backend>);

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Exception, class F>
void rejects(F&& call, const char* message) {
    try { std::forward<F>(call)(); }
    catch (const Exception&) { return; }
    throw std::runtime_error(message);
}

// Fixture-local model state, deliberately NOT a definition or surrogate object
// of KvHandle. The required handle overloads are checked above and delegate to
// these same bound-state paths. Runtime tests use bound paths so no fabricated
// reference, incomplete-type dereference or dependency on the store is needed.
struct State {
    std::vector<std::uint32_t> consumed;
    std::optional<std::uint32_t> pending, next;
    std::uint64_t recurrent = 0, rng_draws = 0;
    [[nodiscard]] ss::Frontier frontier() const {
        return {static_cast<std::uint64_t>(consumed.size()), pending};
    }
    bool operator==(const State&) const = default;
};

class ForwardPlan final : public ss::ExecutionPlan {
public:
    ForwardPlan(ss::Backend& backend, State& bound, std::vector<std::uint32_t> ids)
        : ExecutionPlan(backend), state(bound), inputs(std::move(ids)) {}
    State& state;
    std::vector<std::uint32_t> inputs, predictions;
    std::vector<State> checkpoints;
};

class FakeBackend final : public ss::Backend {
public:
    std::size_t calls = 0;
    std::vector<std::uint32_t> evaluated;
    bool fail_after_row = false;
    std::function<void()> on_completed;
private:
    void execute_plan(ss::ExecutionPlan& plan) override {
        auto* forward = dynamic_cast<ForwardPlan*>(&plan);
        if (!forward) throw std::invalid_argument("unsupported execution plan");
        ++calls;
        for (const auto token : forward->inputs) {
            evaluated.push_back(token);
            auto& state = forward->state;
            state.consumed.push_back(token);
            state.pending.reset();
            state.recurrent = state.recurrent * 131U + token;
            // A tiny deterministic target transition, not a sampling algorithm.
            state.next = token + std::uint32_t{1};
            forward->predictions.push_back(*state.next);
            forward->checkpoints.push_back(state);
            if (fail_after_row) throw std::runtime_error("injected execution failure");
        }
        if (on_completed) on_completed();
    }
};

class FakeEngine final : public ss::Engine {
public:
    explicit FakeEngine(FakeBackend& backend, bool eager_prefill = false, bool* destroyed = nullptr)
        : backend_(backend), eager_(eager_prefill), destroyed_(destroyed) {}
    ~FakeEngine() override { if (destroyed_) *destroyed_ = true; }
    std::vector<std::uint32_t> drafts;
    ss::SamplingParameters observed_sampling;
    State state;

    void prefill(std::span<const std::uint32_t> tokens, ss::KvHandle&) override { prefill_bound(tokens); }
    ss::StepResult decode(ss::KvHandle&, const ss::SamplerState& sampler) override { return decode_bound(sampler); }

    void prefill_bound(std::span<const std::uint32_t> tokens) {
        if (tokens.empty()) return;
        const auto logical = state.consumed.size() + (state.pending ? 1U : 0U);
        if (tokens.size() > caps().max_context - logical)
            throw std::invalid_argument("context limit");
        const auto saved = state;
        std::vector<std::uint32_t> input;
        input.reserve(tokens.size() + 1);
        if (state.pending) input.push_back(*state.pending);
        input.insert(input.end(), tokens.begin(), eager_ ? tokens.end() : tokens.end() - 1);
        ForwardPlan plan(backend_, state, std::move(input));
        try {
            if (!plan.inputs.empty()) backend_.execute(plan);
            if (!eager_) state.pending = tokens.back();
        } catch (...) { state = saved; throw; }
    }

    ss::StepResult decode_bound(const ss::SamplerState& sampler) {
        observed_sampling = sampler.parameters;
        const auto& p = sampler.parameters;
        // This fake advertises only greedy. Exercise explicit refusal rather
        // than adding a second sampler to the contract fixture.
        if (p.temperature != 0 || p.top_p != 1 || p.top_k != 0 || p.min_p != 0 || p.seeded || sampler.constraint)
            throw std::invalid_argument("unsupported selection policy");
        if (sampler.rng_draws != state.rng_draws) throw std::invalid_argument("sampler cursor mismatch");
        const auto saved = state;
        ss::StepResult out;
        out.before = out.after = state.frontier();
        out.rng_draws = state.rng_draws;
        if (sampler.cancellation.stop_requested()) { out.finish = ss::FinishReason::cancelled; return out; }
        if (sampler.max_tokens == 0) { out.finish = ss::FinishReason::length; return out; }
        const auto logical = state.consumed.size() + (state.pending ? 1U : 0U);
        if (logical == 0) throw std::invalid_argument("no input");
        if (logical == caps().max_context) { out.finish = ss::FinishReason::context; return out; }
        const auto limit = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            sampler.max_tokens, caps().max_context - logical));
        const auto proposed = std::min(drafts.size(), static_cast<std::size_t>(limit - 1U));
        std::vector<std::uint32_t> input;
        if (state.pending) input.push_back(*state.pending);
        input.insert(input.end(), drafts.begin(), drafts.begin() + static_cast<std::ptrdiff_t>(proposed));
        ForwardPlan plan(backend_, state, std::move(input));
        try {
            if (!plan.inputs.empty()) backend_.execute(plan);
            out.proposed_drafts = static_cast<std::uint32_t>(proposed);
            // Cancellation at the safe boundary rolls back tentative attention
            // AND recurrent state, preserving the original unconsumed anchor.
            if (sampler.cancellation.stop_requested()) {
                state = saved;
                out.finish = ss::FinishReason::cancelled;
                return out;
            }
            for (std::size_t i = 0; i <= proposed; ++i) {
                const auto next = saved.pending ? plan.predictions.at(i) :
                    (i == 0 ? saved.next.value() : plan.predictions.at(i - 1));
                out.tokens.push_back(next);
                const bool accepted = i < proposed && drafts[i] == next;
                if (accepted) ++out.accepted_drafts;
                if (std::find(sampler.stop_tokens.begin(), sampler.stop_tokens.end(), next) != sampler.stop_tokens.end()) {
                    out.finish = ss::FinishReason::stop;
                    break;
                }
                if (out.tokens.size() == limit) {
                    out.finish = limit == sampler.max_tokens ? ss::FinishReason::length : ss::FinishReason::context;
                    break;
                }
                if (!accepted) break;
            }
            const auto retained = out.tokens.size() - 1U + (saved.pending ? 1U : 0U);
            state = retained == 0 ? saved : plan.checkpoints.at(retained - 1U);
            state.pending = out.tokens.back();
            out.after = state.frontier();
            return out;
        } catch (...) { state = saved; throw; }
    }

    ss::Capabilities caps() const override {
        ss::Capabilities out;
        out.max_context = 32;
        out.max_slots = 1;
        out.speculative = true;
        return out;
    }
    strix::profile::ExecutionReceipt receipt() const override {
        strix::profile::ExecutionReceipt out;
        out.engine = "session-contract";
        out.identity.arithmetic_contract = "host-transition-v1";
        out.identity.state_format = "host-recurrent-v1";
        out.implementation = "host";
        out.digests.emplace_back("recurrent", strix::profile::hex_identity(state.recurrent));
        return out;
    }
private:
    FakeBackend& backend_;
    bool eager_;
    bool* destroyed_;
};

void check_growth(const ss::StepResult& result) {
    const auto before = result.before.consumed_tokens + (result.before.pending ? 1U : 0U);
    const auto after = result.after.consumed_tokens + (result.after.pending ? 1U : 0U);
    require(after == before + result.tokens.size(), "logical frontier growth");
    if (!result.tokens.empty()) require(result.after.pending == result.tokens.back(), "last output remains pending");
    else require(result.before == result.after, "empty result preserves anchor");
}

void defaults_and_refusals() {
    const ss::Capabilities c;
    require(c.max_context == 0 && c.max_slots == 0 && c.max_batch == 0, "zero capability limits");
    require(!c.pair && !c.prefix_cache && !c.batch_prefill && !c.batch_decode && !c.speculative &&
        !c.temperature && !c.top_p && !c.top_k && !c.min_p && !c.seed && !c.constraints &&
        !c.thinking_budget && !c.logit_transforms && !c.logprobs && !c.fork_lanes &&
        !c.shared_prefix_cow && !c.lane_cancellation && !c.external_append && !c.lane_resume &&
        !c.request_lane_batch && !c.mixed_advance, "optional capabilities fail closed");
    FakeBackend backend;
    require(!backend.supports({"add", "f32", "none", "cpu"}) &&
        !backend.memory_budget().probe_known, "plan-only backend does not advertise portable resources");
    rejects<ss::UnsupportedOperation>([&] { (void)backend.allocate(16); }, "plan-only resources explicitly unsupported");
    FakeEngine engine(backend);
    ss::Engine& api = engine;
    rejects<std::invalid_argument>([&] { api.prefill_batch({}); }, "no scalar prefill fallback");
    rejects<std::invalid_argument>([&] { (void)api.decode_batch({}); }, "no scalar decode fallback");
    rejects<std::invalid_argument>([&] { (void)api.advance({}); }, "mixed planning is unsupported by default");
    ss::KvHandle unused;
    rejects<std::invalid_argument>([&] { (void)api.fork_lane({}); }, "fork fails closed");
    rejects<std::invalid_argument>([&] { api.cancel_lane(unused); }, "cancel fails closed");
    rejects<std::invalid_argument>([&] { api.resume_lane(unused); }, "resume fails closed");
    rejects<std::invalid_argument>([&] { api.append_external({}, unused); }, "append fails closed");
    rejects<std::invalid_argument>([&] { (void)engine.decode_bound({}); }, "empty lane rejected");
    engine.prefill_bound(std::array<std::uint32_t, 2>{10, 11});
    const auto saved = engine.state;
    const auto calls = backend.calls;
    ss::SamplerState sampler;
    sampler.parameters = {0.75, 0.9, 17, 0.05, 123456789012345ULL, true};
    rejects<std::invalid_argument>([&] { (void)engine.decode_bound(sampler); }, "unsupported sampling rejected");
    const auto& observed = engine.observed_sampling;
    require(observed.temperature == 0.75 && observed.top_p == 0.9 && observed.top_k == 17 &&
        observed.min_p == 0.05 && observed.seed == 123456789012345ULL && observed.seeded,
        "all existing sampling parameters reach adapter unchanged");
    sampler = {};
    sampler.constraint = std::make_shared<strix::serve::ConstraintSpec>();
    rejects<std::invalid_argument>([&] { (void)engine.decode_bound(sampler); }, "unsupported policy rejected");
    sampler = {};
    sampler.rng_draws = 1;
    rejects<std::invalid_argument>([&] { (void)engine.decode_bound(sampler); }, "RNG cursor mismatch rejected");
    require(engine.state == saved && backend.calls == calls, "refusals precede execution");
}

void backend_ownership_and_lifetimes() {
    FakeBackend first, second;
    State state;
    ForwardPlan plan(first, state, {7, 8});
    ss::Backend& execution = first;
    rejects<std::invalid_argument>([&] { second.execute(plan); }, "foreign backend rejected");
    require(second.calls == 0 && state.consumed.empty(), "foreign plan has no effects");
    execution.execute(plan);
    require(state.consumed == std::vector<std::uint32_t>({7, 8}) && state.recurrent == 925 &&
        plan.predictions == std::vector<std::uint32_t>({8, 9}), "polymorphic backend executes complete plan");
    struct UnknownPlan final : ss::ExecutionPlan { using ExecutionPlan::ExecutionPlan; } unknown(first);
    rejects<std::invalid_argument>([&] { execution.execute(unknown); }, "unknown plan rejected");
    require(first.calls == 1, "unknown plan refused before work");
    bool destroyed = false;
    {
        std::unique_ptr<ss::Engine> engine = std::make_unique<FakeEngine>(first, false, &destroyed);
        require(engine->caps().max_slots == 1, "polymorphic Engine inspection");
    }
    require(destroyed, "Engine deleted through virtual destructor");
    ForwardPlan later(first, state, {9});
    execution.execute(later);
    require(state.consumed.back() == 9, "borrowed backend outlives Engine");
}

void serial_append_and_owned_results() {
    FakeBackend backend;
    FakeEngine engine(backend);
    std::array<std::uint32_t, 2> prompt{0x80000000U, 11};
    engine.prefill_bound(prompt);
    prompt.fill(99);
    require(engine.state.consumed == std::vector<std::uint32_t>({0x80000000U}) && engine.state.pending == 11U,
        "prefill owns copied unsigned IDs and retains anchor");
    const auto saved = engine.state;
    engine.prefill_bound({});
    require(engine.state == saved, "empty append does nothing");
    ss::SamplerState sampler;
    sampler.max_tokens = 8;
    const auto first = engine.decode_bound(sampler);
    check_growth(first);
    require(first.tokens == std::vector<std::uint32_t>({12}) && first.after.consumed_tokens == 2 &&
        first.finish == ss::FinishReason::none && first.rng_draws == 0 && first.proposed_drafts == 0 &&
        first.accepted_drafts == 0, "ordinary step consumes input, emits unevaluated output, no RNG draw");
    engine.prefill_bound(std::array<std::uint32_t, 2>{20, 21});
    const auto next = engine.decode_bound(sampler);
    check_growth(next);
    require(next.tokens == std::vector<std::uint32_t>({22}) &&
        backend.evaluated == std::vector<std::uint32_t>({0x80000000U, 11, 12, 20, 21}),
        "append consumes old anchor once and appends only suffix");
    require(first.tokens == std::vector<std::uint32_t>({12}) && first.after.pending == 12U, "result survives later calls");

    FakeBackend eager_backend;
    FakeEngine eager(eager_backend, true);
    eager.prefill_bound(std::array<std::uint32_t, 2>{10, 11});
    const auto calls = eager_backend.calls;
    const auto selected = eager.decode_bound(sampler);
    check_growth(selected);
    require(!selected.before.pending && selected.before.consumed_tokens == 2 &&
        selected.tokens == std::vector<std::uint32_t>({12}) && eager_backend.calls == calls,
        "fully consumed prefill selects cached logits without evaluating output");
}

void speculative_commit_and_rollback() {
    const std::array<std::uint32_t, 2> prompt{10, 11};
    FakeBackend backend, serial_backend;
    FakeEngine engine(backend), serial(serial_backend);
    engine.prefill_bound(prompt);
    serial.prefill_bound(prompt);
    engine.drafts = {12, 999, 14};
    ss::SamplerState sampler;
    sampler.max_tokens = 8;
    const auto out = engine.decode_bound(sampler);
    check_growth(out);
    require(out.tokens == std::vector<std::uint32_t>({12, 13}) && out.proposed_drafts == 3 &&
        out.accepted_drafts == 1 && out.after.pending == 13U, "speculation emits accepted prefix and correction only");
    for (std::size_t i = 0; i < out.tokens.size(); ++i) (void)serial.decode_bound(sampler);
    require(engine.state == serial.state, "rejected rows roll back attention and recurrent state to serial route");
    require(backend.calls == 2 && backend.evaluated.back() == 14, "one complete speculative plan executed");
    engine.drafts = {14, 15};
    const auto accepted = engine.decode_bound(sampler);
    check_growth(accepted);
    require(accepted.tokens == std::vector<std::uint32_t>({14, 15, 16}) && accepted.accepted_drafts == 2 &&
        accepted.after.pending == 16U, "full acceptance retains unconsumed bonus anchor");

    FakeBackend stop_backend;
    FakeEngine stopped(stop_backend);
    stopped.prefill_bound(prompt);
    stopped.drafts = {12, 13, 14};
    const std::array<std::uint32_t, 1> stops{12};
    sampler.stop_tokens = stops;
    const auto stop = stopped.decode_bound(sampler);
    check_growth(stop);
    require(stop.tokens == std::vector<std::uint32_t>({12}) && stop.finish == ss::FinishReason::stop &&
        stop.after.consumed_tokens == 2 && stop.accepted_drafts == 1 && stopped.state.consumed.back() == 11,
        "stop included once; stop and later speculative rows are not retained as consumed");
    require(stopped.state.recurrent == (10U * 131U + 11U), "post-stop recurrent writes removed");

    FakeBackend length_backend;
    FakeEngine limited(length_backend);
    limited.prefill_bound(prompt);
    limited.drafts = {12, 13, 14};
    sampler = {};
    sampler.max_tokens = 2;
    const auto length = limited.decode_bound(sampler);
    check_growth(length);
    require(length.tokens == std::vector<std::uint32_t>({12, 13}) && length.finish == ss::FinishReason::length &&
        length.proposed_drafts == 1 && length_backend.evaluated.back() == 12, "budget bounds proposals and bonus output");
}

void cancellation_limits_and_failures() {
    FakeBackend backend;
    FakeEngine engine(backend);
    engine.prefill_bound(std::array<std::uint32_t, 2>{10, 11});
    const auto saved = engine.state;
    ss::SamplerState sampler;
    std::stop_source cancel;
    sampler.cancellation = cancel.get_token();
    backend.on_completed = [&] { (void)cancel.request_stop(); };
    const auto cancelled = engine.decode_bound(sampler);
    check_growth(cancelled);
    require(cancelled.finish == ss::FinishReason::cancelled && engine.state == saved,
        "safe-boundary cancellation restores complete state and pending anchor");
    const auto calls = backend.calls;
    check_growth(engine.decode_bound(sampler));
    require(backend.calls == calls, "entry cancellation does no backend work");
    backend.on_completed = {};
    sampler = {};
    sampler.max_tokens = 0;
    const auto zero = engine.decode_bound(sampler);
    check_growth(zero);
    require(zero.finish == ss::FinishReason::length && engine.state == saved && backend.calls == calls,
        "zero budget preserves anchor without execution");
    backend.fail_after_row = true;
    rejects<std::runtime_error>([&] { (void)engine.decode_bound({}); }, "execution failure propagates");
    require(engine.state == saved, "decode failure restores recurrent and attention state");
    rejects<std::runtime_error>([&] { engine.prefill_bound(std::array<std::uint32_t, 2>{20, 21}); },
        "prefill execution failure propagates");
    require(engine.state == saved, "failed append preserves old anchor");
    backend.fail_after_row = false;
    std::vector<std::uint32_t> too_long(33, 7);
    rejects<std::invalid_argument>([&] { engine.prefill_bound(too_long); }, "oversize append refused");
    require(engine.state == saved, "capacity rejection preserves old state");
    FakeEngine full(backend);
    const std::vector<std::uint32_t> full_prompt(32, 7);
    full.prefill_bound(full_prompt);
    const auto full_state = full.state;
    const auto context = full.decode_bound({});
    check_growth(context);
    require(context.finish == ss::FinishReason::context && full.state == full_state, "full context preserves unconsumed input");
}

void existing_receipt_json() {
    FakeBackend backend;
    FakeEngine engine(backend);
    engine.prefill_bound(std::array<std::uint32_t, 2>{10, 11});
    (void)engine.decode_bound({});
    ss::Engine& api = engine;
    auto receipt = api.receipt();
    // Exact output of the existing serializer, including its actual identity ID.
    const auto expected = std::string("{\"schema\":\"strix-engine-receipt/v1\",\"engine\":\"session-contract\",\"identity\":") +
        receipt.identity.json() + ",\"implementation\":\"host\",\"build\":\"\",\"kernel\":\"\",\"device\":\"\",\"topology\":\"\"," +
        "\"command\":\"\",\"digests\":{\"recurrent\":\"0000000000000529\"}}";
    require(receipt.json() == expected, "existing receipt schema and serialized execution digest");
    require(receipt.witness() == "strix-engine-witness " + expected + "\n", "existing witness format");
    require(receipt.identity.state_format == "host-recurrent-v1", "receipt identifies complete hybrid state format");
    const auto identity = receipt.identity.id();
    receipt.identity.state_format = "other-state-v1";
    require(receipt.identity.id() != identity, "state format participates in numerical identity");
    receipt.digests.emplace_back("recurrent", "duplicate");
    rejects<std::invalid_argument>([&] { (void)receipt.json(); }, "existing duplicate-digest guard retained");
    require(api.receipt().json() == expected, "receipt result owns its strings and digest storage");
}

void row_contract() {
    ss::KvHandle handle;
    ss::SamplerState sampler;
    ss::DecodeItem old_decode{handle, sampler};
    ss::AdvanceItem old_advance{handle, {}, sampler, {}};
    require(!old_decode.row_hints && !old_advance.row_hints, "legacy aggregates omit row hints");
    require(!ss::StepResult{}.rows.physical_rows && !ss::AdvanceResult{}.rows.useful_rows,
            "unknown work is not measured zero");
    ss::validate_row_hints({}, false);
    ss::validate_row_hints(ss::RowHints{}, false);
    const std::array<std::uint32_t, 3> shapes{2, 4, 8};
    auto h = ss::resolve_row_hints(ss::RowHints{7, 1.5}, true, shapes);
    require(h.physical_row_cap == 4U && h.expected_useful_rows == 1.5, "qualified downward clamp");
    h = ss::resolve_row_hints(ss::RowHints{UINT32_MAX, 0.0}, true, shapes);
    require(h.physical_row_cap == 8U && h.expected_useful_rows == 0, "cap range and zero prediction");
    for (auto cap : {0U, 1U}) rejects<std::invalid_argument>([&] {
        (void)ss::resolve_row_hints(ss::RowHints{cap, {}}, true, shapes);
    }, "zero or unavailable shape refuses");
    for (double prediction : {-1.0, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN(), double(UINT32_MAX) + 1.0})
        rejects<std::invalid_argument>([&] { ss::validate_row_hints(ss::RowHints{{}, prediction}, true); },
                                       "invalid predicted rows");
    rejects<std::invalid_argument>([&] { ss::validate_row_hints(ss::RowHints{2, {}}, false); }, "unsupported cap");
    rejects<std::invalid_argument>([&] { ss::validate_row_hints(ss::RowHints{{}, 0.5}, false); }, "unsupported prediction");
    for (const auto invalid : {std::array<std::uint32_t, 3>{0, 2, 4}, {2, 2, 4}, {4, 2, 8}})
        rejects<std::invalid_argument>([&] { (void)ss::resolve_row_hints(ss::RowHints{8, {}}, true, invalid); },
                                       "invalid qualification shapes");
    rejects<std::invalid_argument>([] { (void)ss::resolve_row_hints(ss::RowHints{8, {}}, true, {}); },
                                   "missing native qualification");
}

} // namespace

int main() {
    try {
        defaults_and_refusals();
        backend_ownership_and_lifetimes();
        serial_append_and_owned_results();
        speculative_commit_and_rollback();
        cancellation_limits_and_failures();
        existing_receipt_json();
        row_contract();
        std::cout << "session engine contract tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "session engine contract test failure: " << error.what() << '\n';
        return 1;
    }
}
