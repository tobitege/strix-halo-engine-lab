#pragma once
// Optional two-node Gemma 4 route (STRIX_G4_PAIR=1 with --rank 0|1 [--transport usb4auto|usb4gpu|tcp --peer HOST]).
// Prompt prefill is pipeline-parallel over the two nodes (prefill/pair.hpp): rank 0 runs layers [0, cut) of
// every prompt chunk, rank 1 layers [cut, 42) of the previous chunk at the same time. Rank 1 receives rank
// 0's K/V rows with the seam and commits them, so after the prompt it holds the complete stream and
// decodes alone ("prefill on two nodes, decode on one"; no separate KV hand-off). Rank 0 exits after its
// last chunk. STRIX_G4_PAIR_CUT selects the cut (default pair_default_cut, chosen by measurement).
#include "coordinator.hpp"
#include "runtime.hpp"
#include <span>
#include <vector>

namespace strix::gemma4::engine {
inline constexpr unsigned pair_default_cut = 21;
[[nodiscard]] bool pair_enabled(); // STRIX_G4_PAIR=1
// Chunks the coordinator gives prefill_rows for this prompt alone (both ranks size the pipe with it).
[[nodiscard]] std::size_t pair_chunk_count(std::size_t prompt, std::size_t chunk);
// Runs one request; rank 1 returns the coordinator's receipt (with "pair" fields), rank 0 its stage receipt.
// emit receives ids (empty on rank 0) and the receipt.
int run_pair(const Options& options, const Runtime& runtime, std::span<const std::uint32_t> prompt,
             const std::function<void(std::span<const std::uint32_t>, strix::serve::Json)>& emit);
} // namespace strix::gemma4::engine
