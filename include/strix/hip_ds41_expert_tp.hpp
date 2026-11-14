#pragma once

// Device pieces of deepseek41 expert tensor parallelism (ds41_expert_tp.hpp, ds41-expert-tp2-i1152-v1).

#include <cstdint>
#include <functional>

namespace strix::ds41::expert_tp {

// rows Q2_K down rows of 756 bytes (src, device) to one rank's 420-byte rows (dst, device), as pack_down_row, or (packed)
// its 380-byte rows as pack_down_row_packed.
void enqueue_pack_down_rows(const std::uint8_t* src, std::uint8_t* dst, std::uint64_t rows, unsigned rank, bool packed, void* stream);

// Loading one rank's part of a routed tensor [experts][rows][row bytes] of `bytes` bytes at `offset` in the file:
// part_bytes is its size; load_part reads it with `load(file offset, bytes, device destination)` (synchronous) into `out`:
// gate/up rows straight in the rank's local order, down rows (down = true, `hidden` rows per expert) whole through a
// device staging buffer of 16 experts and packed (enqueue_pack_down_rows on `stream`, synchronised) in the layout of
// packed_from_env() (device_down_row_bytes() per row).
using LoadFn = std::function<void(std::uint64_t, std::uint64_t, std::uint8_t*)>;
[[nodiscard]] std::uint64_t part_bytes(std::uint64_t bytes, std::uint32_t experts, bool down, std::uint32_t hidden);
void load_part(const LoadFn& load, std::uint64_t offset, std::uint64_t bytes, std::uint32_t experts, bool down, std::uint32_t hidden,
               unsigned rank, std::uint8_t* out, void* stream, const std::function<void()>& loaded = {});
// `loaded` (optional) makes a queued `load` synchronous where load_part itself consumes the bytes: it returns when
// everything passed to `load` so far is in device memory. Without it `load` must be synchronous, as before.

}  // namespace strix::ds41::expert_tp
