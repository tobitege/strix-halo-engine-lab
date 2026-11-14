#pragma once

#include "strix/engine_profile.hpp"
#include "strix/serve/engine.hpp"
#include "strix/session/embed_rows.hpp"
#include "strix/session/kv.hpp"

#include <cstdint>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <vector>

namespace strix::session {

struct KvHandle;
struct StateSnapshot;

// Counts describe committed model input, not generated-output accounting.
// [0, consumed_tokens) has canonical model state. pending, when present, is
// the admitted token at consumed_tokens, not yet part of that committed state.
// Logical history length is consumed_tokens + pending.has_value(). Tentative
// backend rows may exist beyond it but are never reusable committed state.
struct Frontier {
    std::uint64_t consumed_tokens = 0;
    std::optional<std::uint32_t> pending;
    bool operator==(const Frontier&) const = default;
};

// Preserve every existing serve sampling parameter, including min_p and the
// distinction between an explicit seed and an engine-resolved seed. This is
// configuration for the family's existing sampler, not another sampler.
using SamplingParameters = serve::SamplingParameters;

struct SamplerState {
    SamplingParameters parameters;
    // Expected cumulative cursor in this lane's target sampler stream. Engine
    // owns the actual family RNG and resolved seed (initialized once, never
    // reseeded per decode); the caller carries StepResult::rng_draws forward.
    // A cursor mismatch is rejected before work. Counts are sampler draws, not
    // token counts: greedy draws none, and no one-draw-per-token rule is imposed.
    std::uint64_t rng_draws = 0;
    // Remaining output allowance for THIS call, including any speculative bonus
    // token and stop token. One by default; zero returns length without work.
    std::uint32_t max_tokens = 1;
    std::span<const std::uint32_t> stop_tokens;
    std::stop_token cancellation;
    // The existing selection policies, with their original shared ownership.
    // A non-null unsupported policy or unsupported sampling field is rejected
    // before work; never silently downgraded to greedy or ignored.
    std::shared_ptr<const serve::ConstraintSpec> constraint;
};

enum class FinishReason { none, stop, length, context, cancelled };

// Per-lane proposal for ONE native grouped call. Units are evaluated forward
// rows, not sessions, sampler draws or an output-token allowance. A cap of zero
// is invalid; unset preserves the family's existing planner. Expected work is a
// finite prediction in [0, UINT32_MAX], potentially fractional, never measured
// acceptance. Neither field changes selection policy or SamplerState budgets.
struct RowHints {
    std::optional<std::uint32_t> physical_row_cap;
    std::optional<double> expected_useful_rows;
};
inline void validate_row_hints(const std::optional<RowHints>& hints, bool supported) {
    if (!hints) return;
    const auto& h = *hints;
    if ((h.physical_row_cap && !*h.physical_row_cap) ||
        (h.expected_useful_rows && (!std::isfinite(*h.expected_useful_rows) ||
            *h.expected_useful_rows < 0 || *h.expected_useful_rows > UINT32_MAX)))
        throw std::invalid_argument("invalid row hints");
    if (!supported && (h.physical_row_cap || h.expected_useful_rows))
        throw std::invalid_argument("engine does not support row hints");
}
// Adapters call after validating ALL items, before any mutation. Clamp a cap
// DOWN to the largest qualified shape <= it; never manufacture an R1 path or
// exceed a cap to reach a minimum. No admissible shape is a prevalidation error.
// qualified_shapes are positive, strictly increasing native per-lane shapes.
inline RowHints resolve_row_hints(const std::optional<RowHints>& hints, bool supported,
                                  std::span<const std::uint32_t> qualified_shapes) {
    validate_row_hints(hints, supported);
    if (!hints || (!hints->physical_row_cap && !hints->expected_useful_rows)) return {};
    std::uint32_t previous = 0, admitted = 0;
    for (auto shape : qualified_shapes) {
        if (!shape || shape <= previous) throw std::invalid_argument("invalid qualified row shapes");
        previous = shape;
        if (!hints->physical_row_cap || shape <= *hints->physical_row_cap) admitted = shape;
    }
    if (!admitted) throw std::invalid_argument("no qualified row shape within cap");
    auto out = *hints;
    if (out.physical_row_cap) out.physical_row_cap = admitted;
    return out;
}
// Actual per-call lane-attributed forward work. Unset means unavailable, never
// measured zero. physical = useful + padding when all three are known. Useful
// rows include evaluated prompt/target/draft work even if later rejected; they
// are NOT accepted output IDs. Group-only padding is charged exactly once to
// the first result in input order, never repeated in every lane. Sum each field
// only if every lane reports it; partial knowledge cannot yield a group total.
struct EffectiveRows {
    std::optional<std::uint64_t> physical_rows;
    std::optional<std::uint64_t> useful_rows;
    std::optional<std::uint64_t> padding_rows;
};

struct StepResult {
    // Newly committed OUTPUT IDs in order, including the first stop ID. Empty
    // only at a terminal boundary. Draft proposals and rejected suffixes never
    // appear here. The caller owns this storage; later calls cannot mutate it.
    std::vector<std::uint32_t> tokens;
    Frontier before, after;
    FinishReason finish = FinishReason::none;
    // Cumulative target-stream cursor after commit, not a per-call delta. Real
    // speculative acceptance draws on that stream must be included. Independent
    // proposal RNGs remain family-owned; they must not steal target draws.
    // Rollback restores all RNG/policy cursors to the committed route, according
    // to the existing sampler's algorithm, not an invented serial draw schedule.
    std::uint64_t rng_draws = 0;
    std::uint32_t proposed_drafts = 0;
    std::uint32_t accepted_drafts = 0; // only those retained in tokens
    EffectiveRows rows{};
};

// Effective admission limits for this Engine's loaded model/configuration,
// never hardware maxima or aspirational support. Zero limits and false flags
// mean unavailable/unsupported; callers must not infer support from a receipt.
// max_context includes the pending token; max_batch counts sessions per call,
// not speculative rows. Slots are independent live lanes. Pair means this
// configuration executes a coordinated pair, not merely that a transport exists.
struct Capabilities {
    std::uint64_t max_context = 0;
    std::uint32_t max_slots = 0;
    std::uint32_t max_batch = 0;
    bool pair = false;
    bool prefix_cache = false;
    bool batch_prefill = false;
    bool batch_decode = false;
    bool speculative = false;
    bool temperature = false;
    bool top_p = false;
    bool top_k = false;
    bool min_p = false;
    bool seed = false;
    bool constraints = false;
    bool thinking_budget = false;
    bool logit_transforms = false;
    bool logprobs = false;
    // Independent continuation from a coherent snapshot requires actual shared
    // immutable prefix owners AND write isolation, not an eager prefix copy.
    bool fork_lanes = false;
    bool shared_prefix_cow = false;
    bool lane_cancellation = false;
    bool external_append = false;
    bool lane_resume = false;
    // batch_decode alone may cover unrelated requests. This flag explicitly
    // admits several forked lanes of ONE request via decode_batch below.
    bool request_lane_batch = false;
    bool mixed_advance = false;
    // Native layer/control plans require the runtime's owned reservation API
    // and a bounded family control turn. Scalar prefill/decode are unsupported
    // in this mode; callers must not substitute a serial engine route.
    bool owner_control = false;
    bool row_hints = false; // only qualified native adapters may opt in
    bool turn_commit = false;
    // The engine resolves virtual ids (session/embed_rows.hpp) in its embed stage.
    // Without it any id >= 2^31 is a prevalidation error (reject_embed_ids).
    bool embed_rows = false;
};

struct PrefillItem {
    std::span<const std::uint32_t> tokens;
    KvHandle& state;
};

struct DecodeItem {
    KvHandle& state;
    const SamplerState& sampler;
    std::optional<RowHints> row_hints = std::nullopt;
};

// Additive mixed-phase boundary for native family planners. Every live lane is
// explicit; prompt is the still-unadmitted suffix. One call may commit a bounded
// prompt chunk OR output for each lane, and may leave a paced lane untouched.
// The planner retains one grouped forward, rather than scalarizing its rows.
struct AdvanceItem {
    KvHandle& state;
    std::span<const std::uint32_t> prompt;
    const SamplerState& sampler;
    // Concurrent, read-only observation of ticket/family cancellation. Call at
    // quiescent chunk/collective boundaries only; never interrupt a kernel.
    std::function<bool()> cancelled;
    std::optional<RowHints> row_hints = std::nullopt;
};
struct AdvanceResult {
    std::size_t input_tokens = 0;
    StepResult output; // before/after cover BOTH admitted input and output
    // Execution errors are lane-local results, after quiescence. Any output in
    // this result was committed and remains publishable before the error ends
    // the lane. Unreported tentative state is invalid and must be released.
    std::string error;
    // Authoritative work for this entire advance (input AND output). output.rows
    // remains unset in advance results: never count the same work twice.
    EffectiveRows rows{};
};

// Lower-level model/session contract, independently adaptable to the existing
// serve::GenerationEngine and GenerationQueue. Engine owns model semantics and
// selects its family's Backend plans; HTTP, tokenization and queue ownership
// need not be moved to implement it. No scheduling policy is supplied here.
//
// Ownership/concurrency: the store owns generation-checked handles; the backend
// owns physical state/shards. Engine validates provenance, generation, admission
// and exclusive lane access before work, without transferring handle ownership.
// All calls (including inspection) are externally serialized per Engine unless
// an implementation documents stronger guarantees. No same handle appears twice
// in a batch or in overlapping calls. Borrowed spans/references live through the
// synchronous call only; results/receipts are owning values. A cancellation's
// stop source may be signalled concurrently. Backends/assets outlive their users.
//
// Commit/rollback: ordinary decode emits one selected token; speculation may
// emit an accepted prefix plus a correction/bonus. For any nonempty result, after
// retains tokens.back() as pending, and logical history grows by tokens.size().
// Thus no output token has to be evaluated just to publish it. Empty terminal
// results preserve both frontier and sampler cursor, including an existing
// unconsumed anchor. Before/after are absolute frontiers, not output counts.
//
// Engine applies stop IDs, output/context budgets and selection policies before
// committing/publishing. It must not select or publish a post-stop token. A
// backend may evaluate tentative speculative rows beyond a stop; Engine removes
// them before return, including restoring recurrent/conv, attention/index state,
// drafter state and RNG/policy cursors. Cancellation is observed at entry and at
// family-safe commit boundaries, never mid-kernel/collective. A completed commit
// can win a cancellation race; return every committed output in that case. The
// caller stops at a terminal result and explicitly appends/resumes a new turn
// under the adapter's documented prefill/new-turn contract. The optional lane
// coordination hooks below require resume_lane before external append.
//
// Invalid/unsupported requests throw std::invalid_argument before mutation.
// Execution failures throw after quiescence and rollback; if restoration fails,
// invalidate the affected handles through the store's hooks before throwing.
// An exception must never leave apparently reusable partial state or lose a
// committed-but-unreported output. Batch failures obey this for every item.
// Prefix/snapshot integration must capture this entire frontier and all
// recurrent/rollback/selection state on every rank under matching numerical
// identities; truncating attention rows alone is insufficient for hybrid models.
class Engine {
public:
    Engine() = default;
    virtual ~Engine() = default;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Append exactly these new input IDs to admitted logical history; this is
    // not an implicit reset or a repeated full prompt. Empty input is a no-op.
    // Consume the previous pending anchor before the suffix, without replaying
    // it twice. On success either consume the whole suffix and retain its final
    // logits, or leave its last ID pending. Both routes must yield the family's
    // existing prefill arithmetic and are reported by the next decode's before.
    // No output selection/RNG draws occur here. Rejection/failure follows the
    // rollback rules above. Allocation, release and reset belong to the store.
    virtual void prefill(std::span<const std::uint32_t> tokens, KvHandle& state) = 0;
    [[nodiscard]] virtual StepResult decode(KvHandle& state, const SamplerState& sampler) = 0;
    [[nodiscard]] virtual Capabilities caps() const = 0;
    [[nodiscard]] virtual profile::ExecutionReceipt receipt() const = 0;
    // Explicit quiescent successful-turn sealing at stop/length/context, without
    // resume permission. Consume the published pending anchor exactly once via
    // family forward, selecting/publishing nothing and drawing no target RNG.
    // Preserve generated IDs, finish and selection cursors. A supported sealed
    // lane is a no-op. Unsupported (even empty), invalid, cancelled, failed,
    // stale/foreign/poisoned or in-flight lanes refuse before writes. Failure
    // restores all attempted ranks (including the throwing rank), or poisons
    // only unrecoverable lanes. This is separate from empty input no-ops.
    // Complete family capture_prompt candidates may be prepared on success;
    // an incomplete resume snapshot must never be relabelled prompt-reusable.
    virtual void commit_pending(KvHandle&) {
        throw std::invalid_argument("engine does not support turn commit");
    }
    // Unsupported by default. Validate every item, duplicate/foreign handles,
    // suffix, sampler cursor and allowances before mutation. Results preserve
    // item order. Throw only for prevalidation or fully rolled-back group work;
    // partial failures return per-lane errors so unrelated commits survive.
    [[nodiscard]] virtual std::vector<AdvanceResult> advance(std::span<const AdvanceItem>) {
        throw std::invalid_argument("engine does not support mixed advance");
    }

    // Optional independent-lane admission. Validate caps, snapshot provenance,
    // complete rank state and context/slot capacity BEFORE allocating/mutating.
    // The returned generation is new; the snapshot remains immutable/reusable.
    // Family hooks restore all recurrent/draft/RNG/policy/cursors independently.
    [[nodiscard]] virtual KvHandle fork_lane(const StateSnapshot&) {
        throw std::invalid_argument("engine does not support lane fork");
    }

    // At a quiescent point cancel ONLY this lane, preserving committed work,
    // frontiers and RNG. A completed commit wins a concurrent stop race. These
    // synchronous hooks cannot be invoked during in-flight work; asynchronous
    // requests use SamplerState::cancellation and join before append/resume.
    // A rejected cancellation grants no append/resume permission: quiesce and
    // explicitly resume a new turn first, or release the lane. With these lane
    // coordination hooks, terminal lanes refuse further decode/prefill/external
    // append until resume_lane succeeds; prefill alone cannot bypass that guard.
    virtual void cancel_lane(KvHandle&) {
        throw std::invalid_argument("engine does not support lane cancellation");
    }
    // Resume preserves history, pending anchor, seed and all sampler draws.
    // Caller supplies a fresh (not already stopped) cancellation source on the
    // next decode. New per-call output budget does not reset cumulative cursors.
    virtual void resume_lane(KvHandle&) {
        throw std::invalid_argument("engine does not support lane resume");
    }
    // Explicit external input admission at a quiescent point of an active turn.
    // Reuse prefill's append/evaluation path, consume any old pending anchor once
    // and count suffix IDs as INPUT, never as generated outputs or RNG draws.
    // Prevalidate IDs, stale/foreign/poisoned handles, terminal/in-flight state,
    // capabilities and logical context (consumed+pending+suffix) before mutation.
    // Reserve metadata before physical writes; failure restores all ranks/cursors
    // or poisons only affected lanes. Empty suffix follows prefill's no-op rule.
    virtual void append_external(std::span<const std::uint32_t>, KvHandle&) {
        throw std::invalid_argument("engine does not support external append");
    }

    // Capabilities::embed_rows only. Names the table that resolves the virtual ids of
    // the rows this lane consumes next (prefill, append_external, advance suffixes);
    // it replaces the lane's earlier table, and the engine keeps it until the lane is
    // released or bound again. Rows already consumed need no table. Null unbinds.
    // Token rows are resolved as before; prompts without a virtual id are unaffected
    // whether or not a table is bound. Before any write the engine validates every
    // virtual id against the bound table (EmbedTable::validate); an id with no table
    // bound is a prevalidation error.
    virtual void bind_embeds(KvHandle&, std::shared_ptr<const EmbedTable> table) {
        if (table) throw std::invalid_argument("engine does not support embedding rows");
    }

    // Optional hooks for existing multi-stream coordinators and grouped row
    // planners. Items/results keep caller order; family code owns admission,
    // grouping, drafting and forward-call shape. There is deliberately no
    // default scalar loop: unsupported batching fails before touching any lane,
    // even for an empty batch. An implementing adapter validates ALL items first,
    // including duplicate generation-checked handles, cursors and capacity. For
    // request_lane_batch this is also the one-request lane batching API; there
    // is no per-element virtual dispatch. Results retain item order. Cancellation
    // of one lane cannot erase another's commit. On execution failure either
    // roll back ALL unreported commits (every attempted rank, including throwing
    // ranks), or invalidate every lane whose complete restoration failed. Never
    // keep a valid-looking committed output that an exception hid from callers.
    virtual void prefill_batch(std::span<const PrefillItem>) {
        throw std::invalid_argument("engine does not support batch prefill");
    }
    [[nodiscard]] virtual std::vector<StepResult> decode_batch(std::span<const DecodeItem>) {
        throw std::invalid_argument("engine does not support batch decode");
    }
};

} // namespace strix::session
