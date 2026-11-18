#pragma once
// Host-only pieces shared by the TP2 resident adapters (apps/deepseek41_resident.hpp, apps/minimax_resident.hpp) on top
// of the engine-neutral runner (strix/serve/resident_session.hpp):
//   - LaunchEnv: the process environment of every declared lever as the session was launched. A switchable lever whose
//     environment equals its launch value is put back to the launch state (the engine's own resolution of it); only a
//     value that differs is applied explicitly. So a lever no arm touches never changes, and a cached static that is
//     first read during a plan is still read with its launch value.
//   - agree_bytes / barrier_bytes: the runner's 8-field TP2 agreement and per-slot barrier over any same-size host byte
//     exchange (DeviceForward::exchange_host, Engine::exchange_bytes).
#include "strix/serve/resident_session.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::serve::resident::tp2 {
namespace rp = strix::serve::resident;

struct LaunchEnv {
    std::vector<std::pair<std::string, std::optional<std::string>>> env;  // table order

    [[nodiscard]] const std::optional<std::string>* find(std::string_view name) const {
        for (const auto& e : env)
            if (e.first == name) return &e.second;
        return nullptr;
    }
};

// Snapshot of every lever of `table` (all classes) from `getenv_fn`; call it once, before the first plan.
template<class GetEnv>
[[nodiscard]] LaunchEnv capture_launch(const rp::LeverTable& table, GetEnv&& getenv_fn) {
    LaunchEnv l;
    for (const rp::LeverEntry& e : table) {
        const std::string name(e.name);
        const char* v = getenv_fn(name.c_str());
        l.env.emplace_back(name, v ? std::optional<std::string>(v) : std::nullopt);
    }
    return l;
}

// The lever's current value when it differs from its launch value; nullopt = at the launch value (restore the launch
// state). A lever absent from the snapshot counts as launched unset.
template<class GetEnv>
[[nodiscard]] std::optional<std::string> changed(const LaunchEnv& launch, std::string_view name, GetEnv&& getenv_fn) {
    const char* v = getenv_fn(std::string(name).c_str());
    const auto* l = launch.find(name);
    const bool launched = l != nullptr && l->has_value();
    if (v == nullptr) {
        if (!launched) return std::nullopt;
        // Arms only set values and the runner restores the baseline, so an unset lever that was set at launch means
        // someone else changed the environment: refuse rather than guess.
        throw std::runtime_error("resident: " + std::string(name) + " was set at launch and is now unset");
    }
    if (launched && **l == v) return std::nullopt;
    return std::string(v);
}

// Checks that every override of `arm` for the levers that `allowed(name, value)` knows is accepted; throws PlanError.
template<class Allowed>
void validate_values(const rp::Arm& arm, const rp::LeverTable& table, Allowed&& allowed) {
    for (const auto& kv : arm.env) {
        if (rp::lever_class(table, kv.first) == rp::LeverClass::reload) continue;
        if (!allowed(std::string_view(kv.first), std::string_view(kv.second)))
            throw rp::PlanError("resident plan: arm \"" + arm.name + "\": " + kv.first + "=" + kv.second +
                                " is not a value this session can switch to");
    }
}

// The runner's agreement over a byte exchange `ex(const void* out, void* in, std::size_t n)` (both ranks, same size):
// field 0 is ANDed, fields 1..7 must match on both ranks (a mismatch throws on both, since both see it).
template<class Exchange>
[[nodiscard]] bool agree_bytes(const std::array<std::uint32_t, 8>& mine, Exchange&& ex) {
    std::array<std::uint32_t, 8> peer{};
    ex(mine.data(), peer.data(), sizeof(mine));
    for (std::size_t i = 1; i < mine.size(); ++i)
        if (mine[i] != peer[i])
            throw std::runtime_error("resident agreement: field " + std::to_string(i) + " differs (" +
                                     std::to_string(mine[i]) + " here, " + std::to_string(peer[i]) + " on the peer)");
    return mine[0] != 0 && peer[0] != 0;
}

// Both ranks at the same slot before the timed run.
template<class Exchange>
void barrier_bytes(std::uint64_t seq, Exchange&& ex) {
    const std::uint64_t mine = seq ^ 0x52424152ULL;  // "RBAR"
    std::uint64_t peer = 0;
    ex(&mine, &peer, sizeof(mine));
    if (peer != mine)
        throw std::runtime_error("resident barrier: peer at slot " + std::to_string(peer ^ 0x52424152ULL) + ", here " +
                                 std::to_string(seq));
}

// "0" / "1" only.
[[nodiscard]] inline bool is_bit(std::string_view v) noexcept { return v == "0" || v == "1"; }
// Decimal digits only (no sign, no spaces), at most 9 of them.
[[nodiscard]] inline bool is_uint(std::string_view v) noexcept {
    if (v.empty() || v.size() > 9) return false;
    for (const char c : v)
        if (c < '0' || c > '9') return false;
    return true;
}
[[nodiscard]] inline unsigned to_uint(std::string_view v) {
    if (!is_uint(v)) throw std::invalid_argument("resident: not an unsigned integer: " + std::string(v));
    unsigned x = 0;
    for (const char c : v) x = x * 10U + static_cast<unsigned>(c - '0');
    return x;
}

}  // namespace strix::serve::resident::tp2
