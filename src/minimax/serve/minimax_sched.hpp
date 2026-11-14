#pragma once

// Rank-0 scheduling arithmetic. All times are milliseconds; no device or wall-clock reads.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <cstdint>
#include <span>
#include <istream>
#include <vector>
#include <string_view>

namespace strix::minimax::serve {
// Prefill layer-time model of rank 0 (interleave): per rows bucket (log2), EMA of ms per layer from the measured units,
// plus a measured fixed cost per unit (restore, cache table, drain). Prior 10 ms + 0.1 ms x rows per layer (measurement: chats ~25 ms,
// 1K 108 ms, 3K 314 ms per layer).
struct LayerModel {
    std::array<double, 16> per_layer{}, rows_seen{};  // rows_seen: EMA of the measured rows, to scale within a bucket
    double fixed_ms = 5.0;
    // Last eight host-return samples, separate from the smoothed throughput model.
    std::array<std::array<double, 8>, 16> recent{}, recent_rows{};
    std::array<unsigned, 16> recent_next{};
    [[nodiscard]] double upper_layer_ms(std::uint64_t rows) const {
        const unsigned b = bucket(rows);
        double upper = layer_ms(rows);
        for (unsigned i = 0; i < recent[b].size(); ++i)
            if (recent_rows[b][i] > 0)
                upper = std::max(upper, recent[b][i] * std::max(1.0, static_cast<double>(rows) / recent_rows[b][i]));
        return upper;
    }
    // Small consecutive units keep teaching the model even when a high estimate limits their size.
    unsigned pending_bucket = 0;
    std::uint64_t pending_layers = 0;
    double pending_ms = 0;
    void reset_pending() {
        pending_layers = 0;
        pending_ms = 0;
    }
    static unsigned bucket(std::uint64_t rows) {
        unsigned b = 0;
        while ((std::uint64_t{1} << b) < rows && b < 15) ++b;
        return b;
    }
    [[nodiscard]] double layer_ms(std::uint64_t rows) const {
        const unsigned b = bucket(rows);
        const double v = per_layer[b];
        // never below the measured per-row rate: a bucket spans 2x rows (cost grows at most linearly)
        if (v <= 0) return 10.0 + 0.1 * static_cast<double>(rows);
        return rows_seen[b] > 0 ? v * std::max(1.0, static_cast<double>(rows) / rows_seen[b]) : v;
    }
    void observe(std::uint64_t rows, unsigned layers, double ms) {
        if (layers == 0) return;
        const unsigned b = bucket(rows);
        const auto sample = recent_next[b]++ % recent[b].size();
        recent[b][sample] = std::max(0.0, ms) / layers;
        recent_rows[b][sample] = static_cast<double>(rows);
        if (pending_layers > 0 && pending_bucket != b) reset_pending();
        pending_bucket = b;
        pending_layers += layers;
        pending_ms += std::max(0.0, ms - fixed_ms);
        // Use the estimate before this sample; the residual includes the per-unit host work.
        const double fixed = std::clamp(ms - layers * layer_ms(rows), 0.0, 50.0);
        fixed_ms = 0.7 * fixed_ms + 0.3 * fixed;
        if (pending_layers < 4) return;
        double x = pending_ms / static_cast<double>(pending_layers);
        double& c = per_layer[b];
        if (c > 0) x = std::clamp(x, 0.5 * c, 2.0 * c);
        rows_seen[b] = c <= 0 ? static_cast<double>(rows) : 0.7 * rows_seen[b] + 0.3 * static_cast<double>(rows);
        c = c <= 0 ? x : 0.7 * c + 0.3 * x;
        reset_pending();
    }
};

enum class Sched : std::uint32_t { off = 0, interleave = 1 };
inline Sched sched_mode(const char* value) {
    const std::string_view s = value == nullptr ? "interleave" : value;
    if (s == "interleave" || s == "1") return Sched::interleave;
    if (s == "off" || s == "0") return Sched::off;
    throw std::invalid_argument("STRIX_MM_SERVE_SCHED: off | interleave");
}

// Strict comparisons: a same-time first emission does not protect a candidate.
template<class Time> bool protects(Time submitted, Time first_emit, bool emitted) {
    return emitted && first_emit < submitted;
}
// A waiting child predates both the parent submission and the start of its prefill.
// This excludes burst-mates; later children need a pre-existing decoder instead.
template<class Time> bool waiting_short(Time submitted, Time parent_submitted, Time parent_started) {
    return submitted < parent_submitted && submitted < parent_started;
}
inline bool nest_timing(bool whole_admission, bool protected_decoder, bool waiting) {
    return !whole_admission && (protected_decoder || waiting);
}
inline bool off_admission(Sched sched, bool protected_decoder, bool nesting) {
    return sched == Sched::off || (!protected_decoder && !nesting);
}
inline bool reorder_prompts(Sched sched, bool enabled, bool head_protected) {
    return sched == Sched::interleave && enabled && head_protected;
}

enum class Command : std::uint32_t { heartbeat = 1, prefill = 2, step = 3, shutdown = 4, unit = 5, cancel = 6, draft = 7, taps = 8, park = 9, nest = 10, resume_parked = 11, check = 12, retire = 13, session = 14, state = 15, input = 16 };

// A follower only replays the lead's payload. No policy, cancellation ticket or clock is consulted.
template<class Control, class Engine, class Runner, class Receive>
bool replay_command(const Control& c, Engine& eng, Runner& run, Receive receive) try {
    const auto command = static_cast<Command>(c.command);
    switch (command) {
    case Command::prefill: case Command::nest: case Command::unit: case Command::cancel:
    case Command::park: case Command::resume_parked: case Command::check:
        if (c.task_id == 0) throw std::runtime_error("control channel: missing task ID");
        break;
    default: break;
    }
    switch (command) {
    case Command::heartbeat: break;
    case Command::shutdown:
        run.cancel_all();
        return false;
    case Command::prefill:
    case Command::nest: {
        if ((command == Command::nest) != (c.parent_task_id != 0) ||
            c.bank != (c.parent_task_id != 0 ? 1U : 0U))
            throw std::runtime_error("control channel: begin bank / parent mismatch");
        const auto prompt = receive(c.plen);
        // A runner without resident-prefix support still replays every whole prefill.
        if constexpr (requires { run.prefill_begin(c.pslot, prompt, true, c.task_id, c.parent_task_id, c.slot_generation, c.pstart); })
            run.prefill_begin(c.pslot, prompt, (c.pflags & 1U) != 0, c.task_id, c.parent_task_id, c.slot_generation, c.pstart);
        else {
            if (c.pstart != 0) throw std::runtime_error("control channel: resident prefill unsupported by this runner");
            run.prefill_begin(c.pslot, prompt, (c.pflags & 1U) != 0, c.task_id, c.parent_task_id, c.slot_generation);
        }
        break;
    }
    case Command::unit: {
        if (c.layers == 0) throw std::runtime_error("control channel: zero unit layers");
        std::uint32_t next = 0;
        run.require_active(c.task_id, c.bank);
        (void)run.prefill_unit(c.layers, &next);
        break;
    }
    case Command::cancel: run.prefill_cancel(c.task_id); break;
    case Command::park: eng.prefill_park(c.task_id); break;
    case Command::resume_parked: eng.prefill_resume_parked(c.task_id); break;
    case Command::check: run.prefill_check(c.task_id, c.plen); break;
    case Command::taps:
        if (eng.prefill_open() || c.pflags > 1) throw std::runtime_error("control channel: unsafe tap change");
        eng.set_taps(c.pflags != 0);
        break;
    case Command::retire:
        if ((c.append & ~255U) != 0) throw std::runtime_error("control channel: context mask");
        run.retire_contexts(c.append);
        break;
    case Command::draft:
        if (c.dslots > std::size(c.dslot)) throw std::runtime_error("control channel: draft count");
        if (run.dd == nullptr) throw std::runtime_error("control channel: draft without a drafter");
        for (std::uint32_t i = 0; i < c.dslots; ++i) (void)run.draft(c.dslot[i], c.danchor[i], c.dpos[i]);
        break;
    case Command::input:
        if(c.rows==0 || c.rows>std::size(c.row) || c.append || c.dslots!=c.rows)
            throw std::runtime_error("control channel: input shape");
        if constexpr(requires {run.input_control(c);})run.input_control(c);
        else throw std::runtime_error("control channel: native input OFF");
        break;
    case Command::step:
        if (c.rows > std::size(c.row)) throw std::runtime_error("control channel: step count");
        (void)run.step(std::span(c.row, c.rows), c.append);
        break;
    default: throw std::runtime_error("control channel: unknown command");
    }
    return true;
} catch (...) {
    eng.abort("control channel: replay failure");
    throw;
}

inline std::vector<std::uint32_t> nested_fixture(std::istream& input, unsigned bank_rows,
                                                std::uint64_t capacity, unsigned tokens) {
    std::vector<std::uint32_t> prompt;
    for (std::uint32_t token; input.read(reinterpret_cast<char*>(&token), sizeof(token));)
        prompt.push_back(token);
    if (!input.eof() || input.gcount() != 0 || prompt.empty() || prompt.size() > bank_rows ||
        prompt.size() >= capacity || tokens >= capacity - prompt.size())
        throw std::invalid_argument("resume-check nested: invalid complete fixture / bank / capacity (NOT a pass)");
    return prompt;
}

// Elapsed nesting lifetime includes control, setup and publication; engine service is counted once.
inline double nest_switch_ms(double elapsed, double service) {
    return std::max(0.0, elapsed - service);
}
inline unsigned resume_layers(unsigned requested) { return std::max(1U, requested); }

enum class DecodeAction { cancel, context, step };
inline DecodeAction decode_action(bool cancelled, bool full) {
    return cancelled ? DecodeAction::cancel : full ? DecodeAction::context : DecodeAction::step;
}

inline double service_ms(const char* value) {
    if (value == nullptr || *value == 0) return 250;
    char* end = nullptr;
    const double n = std::strtod(value, &end);
    if (*end != 0 || !std::isfinite(n) || n < 0 || n > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("STRIX_MM_SERVE_SERVICE_MS: finite nonnegative milliseconds required");
    return n;
}
inline double gap_ms(const char* value) {
    if (value == nullptr || *value == 0) return 1500;
    char* end = nullptr;
    const double n = std::strtod(value, &end);
    if (*end != 0 || !std::isfinite(n) || n < 1 || n > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("STRIX_MM_SERVE_GAP_MS: finite milliseconds >= 1 required");
    return n;
}
// No service target may override a decoder deadline. Margin covers control/setup and host tail;
// these are predictions, not a hard bound on an indivisible device operation.
inline constexpr double control_margin_ms = 20;
inline double remaining_ms(double gap, double cycle, double waited, double margin = 0) {
    if (!std::isfinite(gap) || !std::isfinite(cycle) || !std::isfinite(waited) || !std::isfinite(margin)) return 0;
    return std::max(0.0, gap - std::max(0.0, cycle) - std::max(0.0, waited) - std::max(0.0, margin));
}
inline bool runs_whole(bool fresh, double predicted, double service, bool decoding, double gap, double cycle,
                       double waited, double margin = 0) {
    if (!std::isfinite(predicted) || predicted < 0) return false;
    if (!decoding && service <= 0) return true;
    const double budget = decoding ? remaining_ms(gap, cycle, waited, margin) : service;
    return fresh && predicted <= budget;
}
inline unsigned unit_layers(double service, bool decoding, double gap, double cycle,
                            double fixed, double layer, double waited = 0, double margin = 0) {
    const double budget = decoding ? remaining_ms(gap, cycle, waited, margin) :
                          service > 0 ? service : std::numeric_limits<double>::infinity();
    if (!std::isfinite(budget)) return ~0U;
    // Zero means decode first. Only the caller may force one indivisible layer AFTER a decode
    // when even a fresh deadline cannot fit it (a documented unavoidable target overrun).
    const double n = std::floor(std::max(0.0, budget - fixed) / std::max(0.001, layer));
    return static_cast<unsigned>(std::clamp(n, 0.0, static_cast<double>(std::numeric_limits<unsigned>::max())));
}
// One level, two children, and a returned parent unit between children. No clock reads.
struct NestQuota {
    unsigned children = 0;
    bool progressed = true;
    double wall_ms = 0;
    [[nodiscard]] bool eligible() const { return children < 2 && progressed && wall_ms < 2600; }
    void admitted() { ++children; progressed = false; }
};
inline bool nest_fits(std::uint64_t rows, unsigned capacity, bool enabled, bool parent_interleave,
                      bool child_interleave, bool ready, bool parked, bool free_slot, const NestQuota& quota,
                      double predicted, double gap, double waited, double cycle, double switch_ms, bool decoding) {
    if (!enabled || !parent_interleave || !child_interleave || !ready || parked || !free_slot ||
        !quota.eligible() || rows == 0 || rows > std::min(512U, capacity) || !std::isfinite(gap) || gap < 1 || !std::isfinite(predicted) || predicted < 0 ||
        !std::isfinite(switch_ms) || switch_ms < 0) return false;
    const double safe = std::min(1300.0, gap);
    return quota.wall_ms < 2 * safe && predicted <= remaining_ms(safe, decoding ? cycle : 0, decoding ? waited : 0,
                                      switch_ms + control_margin_ms);
}

// Scheduling targets are copied at admission and retained across every prefill resume.
struct RequestPolicy {
    Sched sched = Sched::interleave;
    double service_ms = 250;
    double gap_ms = 1500;
    [[nodiscard]] bool runs_whole(bool fresh, double predicted, bool decoding, double cycle, double waited) const {
        return sched == Sched::off || serve::runs_whole(fresh, predicted, service_ms, decoding, gap_ms, cycle, waited);
    }
    [[nodiscard]] unsigned unit_layers(bool decoding, double cycle, double fixed, double layer) const {
        return sched == Sched::off ? ~0U : serve::unit_layers(service_ms, decoding, gap_ms, cycle, fixed, layer);
    }
};
}  // namespace strix::minimax::serve
