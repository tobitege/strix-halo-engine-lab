#pragma once

#include "strix/deploy/footprint.hpp"
#include "strix/deploy/link.hpp"
#include "strix/deploy/plan.hpp"
#include "strix/device/declaration.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Unbound shared host planner for one prefill (no production engine/CLI/server adapter yet): from the declaration, its link
// graph and measured stage costs, enumerate "prefill where it decodes" (local), "prefill on one unit and hand the
// state off" (handoff) and "split the layers across units in blocks" (layers); predict each one's time with its
// uncertainty and bytes per link; refuse what does not fit, is not executable or not qualified; choose by objective.
//
// What is shared and what the model supplies: the planner knows no model and no device type. The model's
// RouteProvider states its layer count, which cuts are legal and how many bytes cross them, the prompt state per layer
// range, the memory of a stage (through its FootprintProvider) and which executable, qualified routes exist
// (Protocol). Rates come only from measured cost files (strix-stage-costs-v1), keyed by unit and arithmetic identity;
// there is no "if CUDA then route X" anywhere.
namespace strix::deploy {

// ---- Measured stage costs ---------------------------------------------------------------------------------------
//   schema = "strix-stage-costs-v1"
//   model = "deepseek4.1"
//   [costs.rtx-32k]                       # the key names the entry in plans and reasons
//   unit = ["rtx"]                        # device ids that run this stage together (one device, or a group)
//   topology = "single"                   # how the unit runs internally: single | tp2 | pipeline (FootprintProvider)
//   identity = "ds41-cuda-prefill-v1"     # arithmetic identity: weights/quant, arithmetic, reduction order, policy
//   context = 32768                       # prompt rows the receipt measured
//   chunk = 0                             # rows per block of that run (0 = the engine's own schedule)
//   total_ms = 11650.0                    # measured prefill time of `context` rows, fixed costs included
//   fixed_ms = 0.0                        # optional: the part of total_ms that does not scale with rows
//   layers = [1.0, 1.0, ...]              # optional: relative work of each model layer on this unit (any scale)
//   tail = 0.0                            # optional: head / final-norm work, same scale as layers
//   spread = 0.02                         # optional: relative half-width from the receipt's ABAB spread (0.15)
//   warm = true                           # warm (last prefill of the process) or cold (first)
//   receipt = "rtx 32K warm, ABAB x3"      # where the numbers come from
struct StageCost {
    std::string name;
    std::vector<std::string> unit;
    std::string topology = "single";
    std::string identity;
    std::uint64_t context = 0;
    std::uint32_t chunk = 0;
    double total_ms = 0, fixed_ms = 0;
    std::vector<double> layers;
    double tail = 0;
    double spread = 0.15;
    bool warm = true;
    std::string receipt;
};
struct StageCostTable {
    std::string model;
    std::vector<StageCost> entries;
};
// Throws std::invalid_argument naming the offending key.
[[nodiscard]] StageCostTable parse_stage_costs(std::string_view toml);
[[nodiscard]] StageCostTable load_stage_costs(const std::string& path);

// ---- What the model supplies --------------------------------------------------------------------------------------
// An executable route the engine implements. `gate` is the id of its quality gate receipt; a protocol without a gate
// is still predicted and reported, never selected.
struct Protocol {
    std::string kind;                   // local | handoff | layers
    std::string name;                   // transport: e.g. "handoff-sho1-v2/bp-v1", "pipe-crc-ack-v1", "none"
    std::string identity;               // identity of the whole route, e.g. "ds41-cuda-prefill-hip-trellis-v1"
    std::vector<std::string> stages;    // stage identities in order: local [I]; handoff [producer, consumer];
                                        // layers [stage 0, stage 1, ...] (the consumer decodes with the last identity)
    std::vector<std::uint32_t> chunks;  // layers: block sizes the route implements (empty = one block)
    std::string gate;
    double import_ms = -1;              // consumer import fixed cost, measured; < 0 = not measured (not selectable)
    double import_ms_per_mb = 0;        // consumer import cost per MB of state
    std::uint64_t piece_bytes = 0;      // size of one acknowledged message; 0 = one round trip per transfer (SHO1)
};

class RouteProvider {
public:
    virtual ~RouteProvider() = default;
    [[nodiscard]] virtual const FootprintProvider& footprints() const = 0;
    [[nodiscard]] virtual std::uint32_t layers() const = 0;
    // Bytes per row that cross layer boundary `cut` (0 < cut < layers) with `context` rows of history, including every
    // dependency the later stage needs; nullopt = not a legal cut.
    [[nodiscard]] virtual std::optional<std::uint64_t> seam_bytes(std::uint32_t cut, std::uint64_t context) const = 0;
    // Prompt state of layers [begin, end) after `rows` rows, with the fixed parts that range owns (closure is the
    // model's job: a model whose layers share sources returns them with every range that needs them).
    [[nodiscard]] virtual std::uint64_t state_bytes(std::uint64_t rows, std::uint32_t begin, std::uint32_t end) const = 0;
    [[nodiscard]] virtual std::vector<Protocol> protocols() const = 0;
    // Memory of a stage that holds only layers [begin, end); nullopt = unknown, and the option is refused.
    // Adapter prerequisite: bind each arithmetic identity to an explicit footprint Format. The current
    // first-matching format selection is only a synthetic planning convention, not qualified admission.
    [[nodiscard]] virtual std::optional<Footprint> range_footprint(const Spec&, const Candidate&, std::uint32_t /*rank*/,
                                                                   std::uint32_t /*begin*/, std::uint32_t /*end*/) const {
        return std::nullopt;
    }
};

// ---- Request and inventory ----------------------------------------------------------------------------------------
// Live load the server passes; the CLI passes none. A device with queued work or a running prefill takes no prefill
// stage; active decodes do not block a unit from receiving a hand-off (it batches the new stream).
struct UnitLoad {
    std::uint32_t active_decodes = 0, queued = 0;
    bool prefill_busy = false;
};
struct Inventory {
    device::Declaration declaration;
    std::map<std::string, DeviceFacts, std::less<>> facts; // measured memory by device id; absent = unknown (refused)
    std::map<std::string, UnitLoad, std::less<>> load;
};
enum class Objective { ttft, throughput };
[[nodiscard]] std::string_view objective_name(Objective objective) noexcept;
[[nodiscard]] Objective parse_objective(std::string_view text); // throws std::invalid_argument

struct RouteRequest {
    Spec spec;                        // model, serving context, slots, memory margin, format
    std::uint64_t prompt = 0;         // rows to prefill; 0 = spec.context
    Objective objective = Objective::ttft;
    std::vector<std::string> decode;  // unit that decodes afterwards; empty = where the route ends
    std::string pin;                  // option label to force; refused (with the reason) when it is not selectable
};

// ---- Result -------------------------------------------------------------------------------------------------------
struct RouteStage {
    std::vector<std::string> unit;
    std::string identity, topology, cost; // cost = stage-cost entry used
    std::uint32_t begin = 0, end = 0;     // prefill range; [0,0) with ranks = decoder-only full-model admission
    double ms = 0;                        // whole-prompt compute on this stage (layers: one block)
    std::vector<RankBudget> ranks;        // memory admission per device of the unit (fit_rank)
};
struct RouteTransfer {
    std::string what;                     // seam, state, handoff
    std::string from, to;
    std::vector<std::string> edges;       // link graph edge names
    std::uint64_t bytes = 0;              // per request (seams: all blocks)
    double ms = 0;                        // time if alone on its path (seams: one block)
    std::string rate;                     // measured | nominal | integrated
};
struct RouteOption {
    std::string label, kind, protocol, identity, gate;
    std::vector<RouteStage> stages;
    std::vector<RouteTransfer> transfers;
    std::vector<std::string> decode;
    std::uint32_t chunk = 0;
    std::uint64_t bytes = 0;              // all bytes moved over links for the request
    double predicted_ms = 0, uncertainty_ms = 0, rows_per_s = 0;
    bool predicted = false;               // a time could be computed
    bool selectable = false;
    std::string reason;                   // first refusal, or the notes of a selectable option
};
struct RoutePlan {
    RouteRequest request;
    std::vector<RouteOption> options;     // enumeration order (local, handoff, layers)
    int chosen = -1;
    std::string reason;                   // why the chosen option won, or why nothing was chosen
    [[nodiscard]] const RouteOption* selected() const noexcept {
        return chosen >= 0 ? &options[static_cast<std::size_t>(chosen)] : nullptr;
    }
    [[nodiscard]] std::string to_json() const; // strix-route-v1, includes route_id = hash()
    [[nodiscard]] std::string hash() const;    // canonical JSON, FNV-1a-64, 16 lowercase hex digits
};

// Shared planner entry point, not yet bound to production callers. Idle load is deterministic;
// live prefill load changes name the load in their reason. Limits refuse the entire plan rather than
// choose from a truncated search: 100,000 search steps, 4,096 layers/costs, 256 devices, 128 protocols,
// 64 stages/chunks per protocol. Endpoint paths are cached within one call.
[[nodiscard]] RoutePlan plan_route(const RouteRequest& request, const Inventory& inventory, const RouteProvider& provider,
                                   const StageCostTable& costs);

// Contrast with receipts: prediction vs measurement per option label. A miss beyond the stated uncertainty and above
// 30 % must be explained before the costs are reused.
struct Receipt { std::string label; double measured_ms = 0; };
struct Contrast {
    std::string label;
    double predicted_ms = 0, uncertainty_ms = 0, measured_ms = 0, error = 0; // error = measured / predicted - 1
    bool within = false;   // |measured - predicted| <= uncertainty
    bool explain = false;  // outside the uncertainty and |error| > 0.30
    bool found = false;
};
[[nodiscard]] std::vector<Contrast> contrast(const RoutePlan& plan, const std::vector<Receipt>& receipts);

} // namespace strix::deploy
