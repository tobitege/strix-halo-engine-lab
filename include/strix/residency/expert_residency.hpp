// Shared host residency planner for engines that can hold only part of their routed experts on a card.
// Host only: no CUDA, no model code. Adapters pass byte counts and observed expert ids.
//
// Contract: when resolve() returns mode == full, the result is IDENTICAL to today's route (everything resident,
// no planner, no pool, no host expert copy). The caller then does nothing new; host_expert_bytes is 0 and
// slots_per_layer == experts. Only mode == tiered activates the pool and the Planner.
//
// Slots are numbered 0..slots_per_layer-1 per layer; the zero (always-miss) slot is NOT counted, the adapter
// maps it. Planner::table(layer)[expert] is the slot or -1 (the adapter routes -1 to the CPU / zero slot).
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace strix::residency {

enum class Mode { automatic, full, tiered };

struct Options {
    Mode mode = Mode::automatic;
    std::uint64_t vram_budget = 0;  // bytes; 0 = whole card
    // bytes; the bound device profile's cap (strix::device::MemoryBudget::device_bytes), 0 = none. Set by the route
    // from the start gate, never by a flag: the plan then honours the same cap as the server's admission view.
    std::uint64_t device_cap = 0;
    // false (STRIX_TIER_HOST_CHECK=0): tiered starts even when the host cannot hold every non-resident expert;
    // cold experts then page from the model file (the reason says by how much). Budget emulation only.
    bool host_check = true;
    bool mode_explicit = false;
};

// Consumes `--expert-residency auto|full|tiered` and `--vram-budget-gib N` (decimal ok, > 0). Returns true when
// `arg` is one of the two flags (value taken from next()); false when it is not ours. A bad value throws
// std::invalid_argument. Serve and CLI both call this, so they accept the same flags.
bool parse_flag(std::string_view arg, const std::function<std::string()>& next, Options& out);
// STRIX_EXPERT_RESIDENCY / STRIX_VRAM_BUDGET_GIB via envcfg; only fills what the flags left unset (flags win).
// Callers: parse flags first, then call with_environment (explicit auto also wins over the environment).
Options with_environment(Options in);

struct ModelBytes {
    std::uint64_t core = 0;      // always-resident weights incl. the MTP head when on
    std::uint64_t mtp_head = 0;  // informational (already inside core)
    std::uint32_t layers = 0, experts = 0, top_k = 0;
    std::vector<std::uint64_t> expert_bytes;  // per layer, one expert (gate+up+down)
    std::uint64_t host_tables = 0;            // host-only tables in every mode (e.g. PLE)
};
struct CardBytes {
    std::uint64_t total = 0, free = 0;
};
struct SessionBytes {
    std::uint64_t per_position = 0, fixed = 0, scratch = 0, margin = 0;
    std::uint32_t sessions = 1, positions = 0;
    std::uint64_t per_group = 0;
    std::uint32_t group_size = 1;
    std::function<std::uint64_t(std::uint32_t)> position_scratch;
};
// Shared checked storage contracts, also used by adapters before allocating or exposing tensors.
std::uint64_t grouped_bytes(std::uint64_t positions, std::uint32_t group, std::uint64_t bytes);
std::uint64_t session_bytes(const SessionBytes&, std::uint32_t positions);
void validate_expert_tensor(const std::vector<std::uint64_t>& dimensions, std::uint64_t bytes,
                            std::uint64_t input, std::uint64_t output, std::uint64_t experts,
                            std::uint64_t block_elements, std::uint64_t block_bytes);
bool allocation_fits(std::uint64_t budget, std::uint64_t initial_free, std::uint64_t free_now,
                     std::uint64_t bytes);
struct HostBytes {
    std::uint64_t available = 0;
};

struct Decision {
    Mode mode = Mode::full;  // full | tiered
    bool ok = false;
    std::string reason;  // one line
    std::string error;
    std::uint64_t budget = 0, expert_pool = 0, session_total = 0, host_expert_bytes = 0;
    std::uint32_t slots_per_layer = 0;  // uniform, excludes the zero slot
    std::uint32_t positions = 0;
    std::uint32_t positions_requested = 0;
    // budget is the engine allowance after live usage; cap_source: card | residency (--vram-budget-gib / STRIX_VRAM_BUDGET_GIB) | device_profile
    std::uint64_t cap = 0;
    std::string cap_source = "card";
};

// Resolve the engine allowance: the explicit residency budget limits this engine; the device profile caps
// total card usage, so existing physical usage is subtracted first. An explicit allowance wins a tie.
struct BudgetCap {
    std::uint64_t bytes = 0, available = 0;
    std::string source = "card";
};
BudgetCap budget_cap(const Options&, const CardBytes&);

// Apply a total-device cap to physical live memory before narrowing to this engine's remaining allowance.
CardBytes live_budget_view(const CardBytes& physical, std::uint64_t device_cap,
                           std::uint64_t engine_allowance = 0, std::uint64_t engine_used = 0);

Decision resolve(const Options&, const ModelBytes&, const CardBytes&, const SessionBytes&, const HostBytes&);
// {"expert_residency":...,"reason":...,"vram_budget_bytes":...,"expert_pool_bytes":...,"slots_per_layer":...,
//  "positions":...,"positions_requested":...,"host_expert_bytes":...} for --print-effective-config.
std::string decision_json(const Decision&);

struct PlannerConfig {
    double decay = 0.97;
    double margin_reuses = 1.5;
    std::uint64_t upload_bytes_per_step = 64ull << 20;
};

struct Move {
    static constexpr std::uint32_t kNone = 0xffffffffu;
    std::uint32_t layer = 0, slot = 0, evict = kNone, expert = 0;  // evict: expert leaving the slot, or kNone
};

struct Receipt {
    std::uint64_t active_bytes = 0, hit_bytes = 0, promoted_bytes = 0, promotions = 0, evictions = 0, steps = 0;
    double cpu_ms = 0, upload_ms = 0, exchange_ms = 0;  // filled by the adapter
    std::uint64_t resident_bytes = 0;
    std::uint32_t context_filled = 0;  // filled by the adapter
    double byte_hit_rate() const { return active_bytes ? double(hit_bytes) / double(active_bytes) : 1.0; }
};
std::string receipt_json(const Receipt&);

// Score = decayed count of steps that used the expert. Promote candidate c over victim v iff
// score_c - score_v * bytes_v / bytes_c > margin_reuses (saved time per resident byte = reuse rate / CPU bandwidth,
// so scores compare directly in reuses). Free slots fill first; promoted bytes per plan() <= upload_bytes_per_step
// (a single expert larger than the budget is still allowed alone, otherwise it could never move); an expert
// observed in the current step is never evicted; ties by (layer, expert) ascending. plan() is a pure function of
// the state: calling it twice without commit() returns the same moves.
class Planner {
public:
    Planner(const ModelBytes&, std::uint32_t slots_per_layer, PlannerConfig = {});
    void seed(std::uint32_t layer, const std::uint32_t* counts);  // prior, counts[experts]; normalised to a rate
    void observe(std::uint32_t layer, const std::uint32_t* ids, std::uint32_t n);  // one step, all rows, dups ok
    void end_step();                                                                // decay + hit/active bytes
    std::vector<Move> plan() const;
    void commit(const Move&);  // the copy is in: table updated, receipt counters advanced
    const std::vector<std::int32_t>& table(std::uint32_t layer) const { return table_[layer]; }
    void fill_from_seed();  // initial residency: best seeded experts per layer (no receipt cost)
    Receipt receipt() const;

private:
    std::uint32_t layers_, experts_, slots_, top_k_;
    PlannerConfig cfg_;
    std::vector<std::uint64_t> bytes_;
    std::vector<std::vector<double>> score_;
    std::vector<std::vector<std::int32_t>> table_;       // expert -> slot or -1
    std::vector<std::vector<std::int32_t>> slot_expert_;  // slot -> expert or -1
    std::vector<std::vector<std::uint8_t>> used_;         // observed in the current (or just ended) step
    bool ended_ = true;
    Receipt rc_;
};

}  // namespace strix::residency
