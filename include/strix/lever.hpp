#pragma once

// One typed declaration per lever: its switch, exactness class, default and per-regime rule.
//
// A lever is an implementation choice with an off switch (a kernel variant, a schedule, a cache). Today a lever is
// usually a getenv at its use site, so its default, its exactness and the range where it wins live in a comment.
// A `Decl` row states them as data, and `select` is the only decision rule:
//
//   explicit switch ("0"/"1")  >  gate of a bit-changing lever  >  per-regime rule  >  default
//
//   bitwise        output bits identical to the reference: ON by default (possibly only in the regimes it wins).
//   bit_changing   changes output bits: OFF unless its gate is open (the gate is the engine's profile/receipt
//                  decision, see engine_profile.hpp); the explicit switch still selects it for measurements.
//
// A lever that wins only in some regimes keeps both arms and declares the crossover on one axis (rows, context
// tokens or concurrent streams). `valid` rejects malformed tables at compile time (static_assert on the table);
// `conflict` reports two levers declared incompatible that are both selected, so the caller can refuse instead of
// running an unmeasured combination.
//
// Header-only and host-only: no HIP/CUDA types, usable from kernels' host launchers, apps, servers and tests.

#include "strix/env_flags.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::lever {

enum class Exactness { bitwise, bit_changing };
enum class Axis { none, rows, context, concurrency };

// With an axis: the lever is ON below the crossover when `on_below`, ON at or above it when `on_from`.
struct Regime {
    Axis axis = Axis::none;
    std::uint64_t crossover = 0;
    bool on_below = false;
    bool on_from = false;
};

struct Decl {
    std::string_view name{};                     // receipt key, unique in the table
    std::string_view env{};                      // STRIX_* switch: "0" off, "1" on, unset/empty = automatic
    Exactness exactness = Exactness::bitwise;
    bool default_on = false;                     // used when `regime.axis == Axis::none`
    Regime regime{};
    std::span<const std::string_view> conflicts{};  // names of levers that must not be ON together with this one
    std::string_view source{};                   // measurement receipt or qualification plan the row rests on
};

// What the request looks like when the decision is taken. Zero means "not known": a regime rule on an unknown
// axis value falls back to OFF for that lever (the reference arm).
struct Inputs {
    std::uint64_t rows = 0;
    std::uint64_t context = 0;
    std::uint64_t concurrency = 0;
};

enum class Origin { builtin, regime, explicit_switch, gate_closed };

struct Choice {
    bool on = false;
    Origin origin = Origin::builtin;
};

[[nodiscard]] constexpr const char* origin_name(Origin origin) {
    switch (origin) {
    case Origin::builtin: return "default";
    case Origin::regime: return "regime";
    case Origin::explicit_switch: return "explicit";
    case Origin::gate_closed: return "gate-closed";
    }
    return "default";
}

[[nodiscard]] constexpr std::uint64_t axis_value(Axis axis, const Inputs& in) {
    switch (axis) {
    case Axis::rows: return in.rows;
    case Axis::context: return in.context;
    case Axis::concurrency: return in.concurrency;
    case Axis::none: break;
    }
    return 0;
}

[[nodiscard]] constexpr const Decl* find(std::span<const Decl> table, std::string_view name) {
    for (const Decl& row : table)
        if (row.name == name) return &row;
    return nullptr;
}

// Table well-formedness, usable in static_assert:
//   names and switches are non-empty and unique; switches are STRIX_* names;
//   a regime rule has a crossover and exactly one ON side (both sides equal is a plain default);
//   a row without a regime rule carries no crossover or side;
//   a bitwise lever is ON somewhere (a bitwise lever that is OFF everywhere is not a lever);
//   every conflict names another row of the table.
[[nodiscard]] constexpr bool valid(std::span<const Decl> table) {
    for (std::size_t i = 0; i < table.size(); ++i) {
        const Decl& row = table[i];
        if (row.name.empty() || row.env.substr(0, 6) != "STRIX_" || row.env.size() == 6) return false;
        for (std::size_t j = 0; j < i; ++j)
            if (table[j].name == row.name || table[j].env == row.env) return false;
        const Regime& r = row.regime;
        if (r.axis == Axis::none) {
            if (r.crossover != 0 || r.on_below || r.on_from) return false;
        } else if (r.crossover == 0 || r.on_below == r.on_from) {
            return false;
        }
        if (row.exactness == Exactness::bitwise && r.axis == Axis::none && !row.default_on) return false;
        for (const std::string_view other : row.conflicts)
            if (other == row.name || find(table, other) == nullptr) return false;
    }
    return true;
}

// The automatic decision: no switch, gate as given.
[[nodiscard]] constexpr Choice automatic(const Decl& row, const Inputs& in, bool gate_open) {
    if (row.exactness == Exactness::bit_changing && !gate_open) return {false, Origin::gate_closed};
    if (row.regime.axis == Axis::none) return {row.default_on, Origin::builtin};
    const std::uint64_t value = axis_value(row.regime.axis, in);
    if (value == 0) return {false, Origin::regime};
    return {value < row.regime.crossover ? row.regime.on_below : row.regime.on_from, Origin::regime};
}

// `value` is the raw text of the lever's switch (nullptr or empty: automatic). Anything but "0"/"1" throws
// std::invalid_argument naming the switch, like every reader in env_flags.hpp.
[[nodiscard]] inline Choice select(const Decl& row, const Inputs& in, const char* value, bool gate_open = false) {
    if (value == nullptr || *value == '\0') return automatic(row, in, gate_open);
    const std::string_view text(value);
    if (text == "0") return {false, Origin::explicit_switch};
    if (text == "1") return {true, Origin::explicit_switch};
    throw std::invalid_argument(std::string(row.env) + "=" + value + ": expected 0 or 1");
}

// Reads the process environment and records the decision in the envcfg registry, so
// --print-effective-config lists the lever with the value it resolved to.
[[nodiscard]] inline Choice select(const Decl& row, const Inputs& in, bool gate_open = false) {
    const std::string env(row.env);
    const char* raw = std::getenv(env.c_str());
    const Choice choice = select(row, in, raw, gate_open);
    envcfg::record(env.c_str(), raw != nullptr && *raw != '\0' ? raw : nullptr, choice.on ? "1" : "0");
    return choice;
}

// First pair of levers that are both ON although one declares the other as a conflict; {nullptr, nullptr} when
// the selection is consistent. `on(row)` returns the selected state of a row.
struct Conflict {
    const Decl* first = nullptr;
    const Decl* second = nullptr;
    [[nodiscard]] constexpr explicit operator bool() const { return first != nullptr; }
};

template <class On>
[[nodiscard]] constexpr Conflict conflict(std::span<const Decl> table, On&& on) {
    for (const Decl& row : table) {
        if (row.conflicts.empty() || !on(row)) continue;
        for (const std::string_view other : row.conflicts) {
            const Decl* peer = find(table, other);
            if (peer != nullptr && on(*peer)) return {&row, peer};
        }
    }
    return {};
}

// Throws std::invalid_argument naming both switches when the selection contains a declared conflict.
template <class On>
inline void require_consistent(std::span<const Decl> table, On&& on) {
    if (const Conflict c = conflict(table, on))
        throw std::invalid_argument(std::string(c.first->env) + " and " + std::string(c.second->env) +
                                    " are declared incompatible and are both on");
}

}  // namespace strix::lever
