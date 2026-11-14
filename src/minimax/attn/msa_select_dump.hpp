#pragma once
// Diagnostic dump of the prefill block selection (quality gates), usable from plain C++ without HIP headers.
namespace strix::minimax::msa::fused {

// path != nullptr: every later sparse prefill batch synchronizes its stream and appends one record to the file
// (truncated here): 8 x u32 header {"SELD", version 1, STRIX_IDX_WMMA mode (index_score_mode.hpp), first position
// low / high, rows, KV heads, top-k}, then rows x heads selection counts and rows x heads x top-k block ids (u32;
// slots at or beyond the count are unspecified). Layers follow call order. nullptr closes the file; throws when the
// file cannot be opened or a write failed. Off by default; never enabled on timed runs.
void set_select_dump(const char* path);

}  // namespace strix::minimax::msa::fused
