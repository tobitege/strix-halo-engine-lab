#pragma once
// Storage element type of a recurrent (linear-attention) state matrix, model- and backend-agnostic.
// Arithmetic is FP32 for every type: a kernel loads an element, widens it to float, updates in FP32 and narrows
// once on store (round to nearest even). f32 is the incumbent (bit-exact) contract; f16 / bf16 are FAST
// storage contracts: they change the bits of every state that is stored, so outputs differ from the f32 run
// even though every speculative / cohort / serial route of the SAME type stays bitwise-equal to each other.
#include <cstddef>
#include <cstdint>

namespace strix {

enum class RecurrentStateType : std::uint8_t { f32 = 0, f16 = 1, bf16 = 2 };

[[nodiscard]] constexpr std::size_t recurrent_state_element_bytes(RecurrentStateType type) noexcept {
    return type == RecurrentStateType::f32 ? 4U : 2U;
}

[[nodiscard]] constexpr const char* recurrent_state_type_name(RecurrentStateType type) noexcept {
    return type == RecurrentStateType::f16 ? "f16" : type == RecurrentStateType::bf16 ? "bf16" : "f32";
}

// Contract tag stored with any persisted / snapshotted state (session cache, checkpoints, exported files):
// a state written under one tag is never restored under another (the reader refuses, no silent conversion).
[[nodiscard]] constexpr const char* recurrent_state_contract(RecurrentStateType type) noexcept {
    return type == RecurrentStateType::f16 ? "recurrent-state-f16-rne-v1" :
        type == RecurrentStateType::bf16 ? "recurrent-state-bf16-rne-v1" : "recurrent-state-f32";
}

// Bytes of `elements` state elements of `type` (sizing of buffers, snapshots and cache records).
[[nodiscard]] constexpr std::uint64_t recurrent_state_bytes(std::uint64_t elements, RecurrentStateType type) noexcept {
    return elements * recurrent_state_element_bytes(type);
}

}  // namespace strix
