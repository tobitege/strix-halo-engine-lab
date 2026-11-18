#pragma once

// Class of every environment switch the DS41 prefill app reads (strix-deepseek41-prefill: strix_ds41_device_prefill and the
// libraries it links). The resident runner (several arms in one process) uses it to know which switches an arm may change:
//   runtime   read on every call already
//   cached    a strix::lever_cache (include/strix/lever_cache.hpp): re-resolved after lever_environment_changed()
//   construct re-resolved by DevicePrefill::apply_levers() (read once per DevicePrefill object before)
//   reload    frozen at load/construction (weights, buffers, plans, admissions): changing it needs a new object; the default,
//             a name that is not listed also means reload
// Sorted by name, no duplicates. The sources are the files under the repository root that contain the STRIX_DS41_ literals
// (the files of the app's libraries, then the headers they include with such literals); a host test scans exactly these.
// Host-only: no HIP types.

#include <array>
#include <string_view>

namespace strix::deepseek41 {

enum class PrefillLeverClass : unsigned char { runtime, construct, cached, reload };

struct PrefillLever {
    std::string_view name;
    PrefillLeverClass cls;
};

inline constexpr std::string_view kPrefillLeverSources[] = {
    "apps/deepseek41_prefill_main.cpp",
    "include/strix/deepseek41_batch_forward.hpp",
    "include/strix/deepseek41_pair_profile_table.hpp",
    "include/strix/ds41_engram.hpp",
    "include/strix/ds41_engram_tp.hpp",
    "include/strix/ds41_exact_flags.hpp",
    "include/strix/ds41_fast_profile.hpp",
    "include/strix/ds41_touch_batch.hpp",
    "src/deepseek41_pair_defaults.cpp",
    "src/deepseek41_prefix_cache.cpp",
    "src/ds41_engram.cpp",
    "src/ds41_expert_tp.cpp",
    "src/ds41_prepared_load.cpp",
    "src/ds41_trellis.cpp",
    "src/hip_deepseek41_batch_forward.hip",
    "src/hip_deepseek41_device_forward.hip",
    "src/hip_deepseek41_prefill.hip",
    "src/hip_ds41_attention.hip",
    "src/hip_ds41_dense_fp16.hip",
    "src/hip_ds41_f8_rows_fast.hip",
    "src/hip_ds41_formats.hip",
    "src/hip_ds41_layer.hip",
    "src/hip_ds41_prefill.hip",
    "src/hip_ds41_rn_table.hip",
    "src/hip_ds41_rn_wide.hip",
    "src/hip_ds41_routes_fast.hip",
    "src/hip_ds41_trellis.hip",
    "src/hip_ds41_trellis_group.hip",
};

inline constexpr PrefillLever kPrefillLevers[] = {
    {"STRIX_DS41_ATTN_A8", PrefillLeverClass::reload},
    {"STRIX_DS41_ATTN_LDS_SWIZZLE", PrefillLeverClass::cached},
    {"STRIX_DS41_ATTN_LEAN", PrefillLeverClass::cached},
    {"STRIX_DS41_ATTN_LIVE_CHUNKS", PrefillLeverClass::reload},  // startup pair resolver (deepseek41_pair_defaults) reads/exports it once
    {"STRIX_DS41_ATTN_MIN_BLOCKS", PrefillLeverClass::cached},
    {"STRIX_DS41_ATTN_SEG_HPW", PrefillLeverClass::cached},
    {"STRIX_DS41_ATTN_TILE_F16", PrefillLeverClass::reload},  // startup pair resolver (deepseek41_pair_defaults) reads/exports it once
    {"STRIX_DS41_ATTN_WAVE_CAP", PrefillLeverClass::cached},
    {"STRIX_DS41_ATTN_WINDOW_FOLD", PrefillLeverClass::reload},  // startup pair resolver (deepseek41_pair_defaults) reads/exports it once
    {"STRIX_DS41_BASELINE", PrefillLeverClass::reload},
    {"STRIX_DS41_BATCH_ENGRAM_ASYNC", PrefillLeverClass::reload},
    {"STRIX_DS41_BATCH_PACKED", PrefillLeverClass::reload},
    {"STRIX_DS41_BATCH_PREFETCH", PrefillLeverClass::reload},
    {"STRIX_DS41_BATCH_QA_KV", PrefillLeverClass::reload},
    {"STRIX_DS41_BATCH_ROWS16", PrefillLeverClass::reload},
    {"STRIX_DS41_BATCH_SEGMENTS", PrefillLeverClass::reload},
    {"STRIX_DS41_BATCH_SHARED", PrefillLeverClass::reload},
    {"STRIX_DS41_BF16_ATTN_VARIANT", PrefillLeverClass::cached},
    {"STRIX_DS41_CB4", PrefillLeverClass::reload},
    {"STRIX_DS41_COMPRESS_AHEAD", PrefillLeverClass::cached},
    {"STRIX_DS41_COMPRESS_FAST", PrefillLeverClass::cached},
    {"STRIX_DS41_COMPRESS_RING", PrefillLeverClass::reload},
    {"STRIX_DS41_COPY_ROWS", PrefillLeverClass::reload},  // startup pair resolver (deepseek41_pair_defaults) reads/exports it once
    {"STRIX_DS41_DECODER_REPLAY", PrefillLeverClass::reload},  // pair profile table row (bounded decoder replay policy)
    {"STRIX_DS41_DECODER_REPLAY_ORACLE", PrefillLeverClass::reload},  // measurement-only exact cone rows of the replay
    {"STRIX_DS41_DECODE_PREFETCH", PrefillLeverClass::reload},
    {"STRIX_DS41_DECODE_XPORT", PrefillLeverClass::reload},  // startup resolver applies this policy before resident arms
    {"STRIX_DS41_DEFER_DECODER", PrefillLeverClass::cached},
    {"STRIX_DS41_DENSE_FORM", PrefillLeverClass::cached},
    {"STRIX_DS41_DENSE_SELECT", PrefillLeverClass::reload},  // startup pair resolver (deepseek41_pair_defaults) reads/exports it once
    {"STRIX_DS41_DENSE_STATS", PrefillLeverClass::cached},
    {"STRIX_DS41_DENSE_TALL", PrefillLeverClass::cached},
    {"STRIX_DS41_DENSE_TALL_ROTATE", PrefillLeverClass::cached},
    {"STRIX_DS41_DSPARK_CALIB", PrefillLeverClass::reload},
    {"STRIX_DS41_ENGRAM_ASYNC", PrefillLeverClass::reload},
    {"STRIX_DS41_ENGRAM_DEVICE_READY", PrefillLeverClass::runtime},
    {"STRIX_DS41_ENGRAM_F16_PF", PrefillLeverClass::reload},
    {"STRIX_DS41_ENGRAM_FAST", PrefillLeverClass::cached},
    {"STRIX_DS41_ENGRAM_FROM_LEAD", PrefillLeverClass::reload},
    {"STRIX_DS41_ENGRAM_LOOKAHEAD", PrefillLeverClass::cached},
    {"STRIX_DS41_ENGRAM_LOOKAHEAD_LOG", PrefillLeverClass::cached},
    {"STRIX_DS41_ENGRAM_PAYLOAD_CACHE", PrefillLeverClass::reload},
    {"STRIX_DS41_ENGRAM_PAYLOAD_CACHE_MIB", PrefillLeverClass::reload},
    {"STRIX_DS41_ENGRAM_PF_FMA", PrefillLeverClass::reload},
    {"STRIX_DS41_ENGRAM_PF_TILE", PrefillLeverClass::reload},
    {"STRIX_DS41_ENGRAM_PIECE_READ", PrefillLeverClass::cached},
    {"STRIX_DS41_ENGRAM_READY", PrefillLeverClass::reload},
    {"STRIX_DS41_ENGRAM_RN", PrefillLeverClass::cached},
    {"STRIX_DS41_ENGRAM_SPLIT_READ", PrefillLeverClass::runtime},
    {"STRIX_DS41_ENGRAM_THREADS", PrefillLeverClass::reload},
    {"STRIX_DS41_ENGRAM_TP", PrefillLeverClass::cached},
    {"STRIX_DS41_ENGRAM_URING", PrefillLeverClass::reload},  // reader backend built at load
    {"STRIX_DS41_ENGRAM_URING_DEPTH", PrefillLeverClass::reload},
    {"STRIX_DS41_EP_MAP", PrefillLeverClass::reload},
    {"STRIX_DS41_EP_OVERLAP", PrefillLeverClass::reload},
    {"STRIX_DS41_EXPERT_ROWS", PrefillLeverClass::reload},
    {"STRIX_DS41_EXPERT_ROWS_FORM", PrefillLeverClass::reload},
    {"STRIX_DS41_EXPERT_SLOTS", PrefillLeverClass::reload},
    {"STRIX_DS41_EXPERT_SLOTS_Y", PrefillLeverClass::reload},
    {"STRIX_DS41_EXPERT_TP", PrefillLeverClass::reload},  // decode-side expert TP: plan and weights at load
    {"STRIX_DS41_EXPERT_TP_LAYOUT", PrefillLeverClass::reload},  // packed or dup layout of the loaded expert weights (set_packed is load-time too)
    {"STRIX_DS41_F16_AS_F8", PrefillLeverClass::reload},
    {"STRIX_DS41_F16_LANES", PrefillLeverClass::cached},
    {"STRIX_DS41_F16_PAIR", PrefillLeverClass::cached},
    {"STRIX_DS41_F16_TILE", PrefillLeverClass::reload},
    {"STRIX_DS41_F16_TILED", PrefillLeverClass::reload},
    {"STRIX_DS41_F16_WIDE", PrefillLeverClass::reload},
    {"STRIX_DS41_FAST_ATTN", PrefillLeverClass::reload},  // startup pair resolver (deepseek41_pair_defaults) reads/exports it once
    {"STRIX_DS41_FAST_ATTN_Q16", PrefillLeverClass::reload},  // static once per process (hip_deepseek41_prefill.hip)
    {"STRIX_DS41_FAST_DENSE", PrefillLeverClass::reload},  // startup pair resolver (deepseek41_pair_defaults) reads/exports it once
    {"STRIX_DS41_FAST_DENSE_BATCH", PrefillLeverClass::cached},  // scheduling only (batched o_a groups), same bits
    {"STRIX_DS41_FAST_DENSE_MIN_ROWS", PrefillLeverClass::reload},  // startup pair resolver (deepseek41_pair_defaults) reads/exports it once
    {"STRIX_DS41_FAST_ENGRAM", PrefillLeverClass::reload},  // startup pair resolver exports the Engram policy
    {"STRIX_DS41_FAST_I4", PrefillLeverClass::runtime},
    {"STRIX_DS41_FAST_I4_SCOPE", PrefillLeverClass::reload},
    {"STRIX_DS41_FAST_MOE", PrefillLeverClass::reload},  // startup pair resolver (deepseek41_pair_defaults) reads/exports it once
    {"STRIX_DS41_FAST_MOE_DOWN_ROWS", PrefillLeverClass::runtime},  // getenv on every call
    {"STRIX_DS41_FAST_MOE_F16", PrefillLeverClass::reload},  // startup pair resolver / profile table row (MoE FP16 operands)
    {"STRIX_DS41_FAST_MOE_ORDER", PrefillLeverClass::runtime},  // getenv on every call
    {"STRIX_DS41_FAST_PREFILL", PrefillLeverClass::reload},  // startup pair resolver (deepseek41_pair_defaults) reads/exports it once
    {"STRIX_DS41_FAST_SHARED", PrefillLeverClass::reload},  // startup pair resolver / profile table row (shared expert FAST)
    {"STRIX_DS41_FP16_CACHE", PrefillLeverClass::reload},
    {"STRIX_DS41_FP16_CACHE_READONLY", PrefillLeverClass::reload},
    {"STRIX_DS41_FP16_TUNE", PrefillLeverClass::reload},
    {"STRIX_DS41_FP8_W", PrefillLeverClass::cached},  // launch shape only (outputs per wave), not a layout
    {"STRIX_DS41_HC_COLLAPSE_ONCE", PrefillLeverClass::cached},  // scheduling only (hc collapse site), same bits
    {"STRIX_DS41_HC_SITE", PrefillLeverClass::reload},
    {"STRIX_DS41_HC_SITE_FAST", PrefillLeverClass::cached},
    {"STRIX_DS41_HOST_EXCHANGE", PrefillLeverClass::reload},
    {"STRIX_DS41_IDX_CAND_FILTER", PrefillLeverClass::cached},
    {"STRIX_DS41_IDX_DIRECT", PrefillLeverClass::cached},
    {"STRIX_DS41_LATE_HC_LAYERS", PrefillLeverClass::reload},
    {"STRIX_DS41_MOE_HEAD", PrefillLeverClass::reload},
    {"STRIX_DS41_MOE_PREFILL", PrefillLeverClass::reload},
    {"STRIX_DS41_MOE_ROUTED", PrefillLeverClass::reload},
    {"STRIX_DS41_MOE_SHARED", PrefillLeverClass::reload},
    {"STRIX_DS41_NATIVE_FP8", PrefillLeverClass::reload},
    {"STRIX_DS41_NATIVE_FP8_SITES", PrefillLeverClass::reload},
    {"STRIX_DS41_NORM_ILP", PrefillLeverClass::cached},
    {"STRIX_DS41_OB_SPLIT", PrefillLeverClass::reload},
    {"STRIX_DS41_OP_TIMING", PrefillLeverClass::reload},
    {"STRIX_DS41_OVERLAY", PrefillLeverClass::reload},  // selects the weights loaded
    {"STRIX_DS41_PACKED_W", PrefillLeverClass::cached},  // launch shape only; three call sites with different defaults
    {"STRIX_DS41_PACKS", PrefillLeverClass::reload},
    {"STRIX_DS41_PF_CORE", PrefillLeverClass::cached},  // set_pf_core still overrides
    {"STRIX_DS41_PREFETCH_PAST_EXPERTS", PrefillLeverClass::reload},
    {"STRIX_DS41_PREFETCH_ROW_MB", PrefillLeverClass::reload},
    {"STRIX_DS41_PREFETCH_SIDE", PrefillLeverClass::reload},
    {"STRIX_DS41_PREFILL_APPROX_LAYERS", PrefillLeverClass::construct},
    {"STRIX_DS41_PREFILL_ASYNC_GATHER", PrefillLeverClass::construct},
    {"STRIX_DS41_PREFILL_BF16_ATTN", PrefillLeverClass::construct},
    {"STRIX_DS41_PREFILL_DENSE_MIN_ROWS", PrefillLeverClass::construct},
    {"STRIX_DS41_PREFILL_DENSE_SITES", PrefillLeverClass::construct},
    {"STRIX_DS41_PREFILL_EARLY_QB", PrefillLeverClass::construct},  // row_split only
    {"STRIX_DS41_PREFILL_EXPERT_TP", PrefillLeverClass::reload},  // expert ownership and the weights loaded
    {"STRIX_DS41_PREFILL_FINAL_KV", PrefillLeverClass::cached},
    {"STRIX_DS41_PREFILL_FP16_A8", PrefillLeverClass::construct},
    {"STRIX_DS41_PREFILL_FP16_DENSE", PrefillLeverClass::construct},
    {"STRIX_DS41_PREFILL_FP16_FAST", PrefillLeverClass::construct},
    {"STRIX_DS41_PREFILL_FP16_MIN_ROWS", PrefillLeverClass::construct},
    {"STRIX_DS41_PREFILL_FP16_PROJ", PrefillLeverClass::construct},
    {"STRIX_DS41_PREFILL_FP16_WSPLIT", PrefillLeverClass::construct},
    {"STRIX_DS41_PREFILL_HC_COLLAPSE", PrefillLeverClass::runtime},
    {"STRIX_DS41_PREFILL_HC_FUSE", PrefillLeverClass::runtime},
    {"STRIX_DS41_PREFILL_JOINT", PrefillLeverClass::construct},
    {"STRIX_DS41_PREFILL_JOINT_LAYER", PrefillLeverClass::reload},  // construction or capacity policy requires a fresh resident object
    {"STRIX_DS41_PREFILL_JOINT_LAYER_ROWS", PrefillLeverClass::reload},  // construction or capacity policy requires a fresh resident object
    {"STRIX_DS41_PREFILL_JOINT_MAX_ROWS", PrefillLeverClass::construct},
    {"STRIX_DS41_PREFILL_LEASE", PrefillLeverClass::runtime},
    {"STRIX_DS41_PREFILL_MEM_GUARD_GIB", PrefillLeverClass::reload},  // checked once before loading
    {"STRIX_DS41_PREFILL_MOE_BAND", PrefillLeverClass::reload},
    {"STRIX_DS41_PREFILL_MOE_PASS_WINDOW", PrefillLeverClass::reload},  // startup resolver applies this policy before resident arms
    {"STRIX_DS41_PREFILL_MOE_PERSIST", PrefillLeverClass::construct},
    {"STRIX_DS41_PREFILL_MOE_R2", PrefillLeverClass::reload},
    {"STRIX_DS41_PREFILL_MOE_R2_DOWN", PrefillLeverClass::reload},
    {"STRIX_DS41_PREFILL_MOE_R2_GU", PrefillLeverClass::reload},
    {"STRIX_DS41_PREFILL_MOE_V2", PrefillLeverClass::reload},
    {"STRIX_DS41_PREFILL_MOE_V2_MASK", PrefillLeverClass::reload},
    {"STRIX_DS41_PREFILL_MOE_V2_PIN", PrefillLeverClass::reload},
    {"STRIX_DS41_PREFILL_MOE_V2_PIPE", PrefillLeverClass::reload},
    {"STRIX_DS41_PREFILL_MOE_V2_Q8", PrefillLeverClass::reload},
    {"STRIX_DS41_PREFILL_MOE_V2_SIGNS", PrefillLeverClass::reload},
    {"STRIX_DS41_PREFILL_MOE_WINDOW_CAP", PrefillLeverClass::reload},  // construction or capacity policy requires a fresh resident object
    {"STRIX_DS41_PREFILL_OVERLAP", PrefillLeverClass::construct},
    {"STRIX_DS41_PREFILL_O_A8_FUSED", PrefillLeverClass::runtime},
    {"STRIX_DS41_PREFILL_O_B_ONE", PrefillLeverClass::runtime},
    {"STRIX_DS41_PREFILL_O_LOW_A8_FUSED", PrefillLeverClass::runtime},
    {"STRIX_DS41_PREFILL_PACK", PrefillLeverClass::construct},  // pack buffers are allocated whatever the value
    {"STRIX_DS41_PREFILL_ROUTER_V2", PrefillLeverClass::runtime},
    {"STRIX_DS41_PREFILL_ROUTER_WIDE", PrefillLeverClass::cached},
    {"STRIX_DS41_PREFILL_SELECT", PrefillLeverClass::reload},
    {"STRIX_DS41_PREFILL_SLABS", PrefillLeverClass::reload},  // picks the window at load
    {"STRIX_DS41_PREFILL_SPLIT_MIN_ROWS", PrefillLeverClass::reload},  // sizes residual streams and half sums
    {"STRIX_DS41_PREFILL_STAGE", PrefillLeverClass::cached},
    {"STRIX_DS41_PREFILL_STAGED_GATHER", PrefillLeverClass::construct},  // needs staged_half (reload) as well
    {"STRIX_DS41_PREFILL_STAGED_HALF", PrefillLeverClass::reload},  // sizes the communicator staging buffers at load
    {"STRIX_DS41_PREFILL_WINDOW", PrefillLeverClass::reload},  // sizes the scratch and row-owner maps at load
    {"STRIX_DS41_PREFILL_WINDOW_DRAIN", PrefillLeverClass::reload},  // startup resolver applies this policy before resident arms
    {"STRIX_DS41_PREFILL_WINDOW_MEAN", PrefillLeverClass::reload},
    {"STRIX_DS41_PREPARED_ROOT", PrefillLeverClass::reload},  // where the weights are loaded from
    {"STRIX_DS41_PROFILE_ALLOW_PENDING", PrefillLeverClass::reload},
    {"STRIX_DS41_QA_KV", PrefillLeverClass::reload},
    {"STRIX_DS41_R1FUSE_QNORM_A8", PrefillLeverClass::reload},
    {"STRIX_DS41_R1FUSE_ROUTES", PrefillLeverClass::cached},
    {"STRIX_DS41_RN_SHAPE", PrefillLeverClass::cached},
    {"STRIX_DS41_RN_WIDE", PrefillLeverClass::cached},
    {"STRIX_DS41_ROUTER_BF16", PrefillLeverClass::reload},
    {"STRIX_DS41_ROUTER_FAST", PrefillLeverClass::cached},
    {"STRIX_DS41_ROUTER_ROWS", PrefillLeverClass::cached},
    {"STRIX_DS41_ROUTES_FAST", PrefillLeverClass::cached},
    {"STRIX_DS41_SEED_EARLY", PrefillLeverClass::reload},
    {"STRIX_DS41_SELECT_V1", PrefillLeverClass::cached},
    {"STRIX_DS41_SERVE_WARM", PrefillLeverClass::reload},
    {"STRIX_DS41_SHARED_DOWN_FUSED", PrefillLeverClass::reload},  // admission at load; route frozen once used
    {"STRIX_DS41_SHARED_GROUPED", PrefillLeverClass::reload},  // admission at load
    {"STRIX_DS41_SHEXP_TP", PrefillLeverClass::reload},
    {"STRIX_DS41_SPEC_ARBITER", PrefillLeverClass::reload},  // startup pair resolver (deepseek41_pair_defaults) reads/exports it once
    {"STRIX_DS41_SPLIT_SHARED", PrefillLeverClass::reload},
    {"STRIX_DS41_TOUCH_BATCH", PrefillLeverClass::cached},
    {"STRIX_DS41_TP_DOWN", PrefillLeverClass::reload},
    {"STRIX_DS41_TP_DOWN_FORM", PrefillLeverClass::reload},
    {"STRIX_DS41_TRELLIS", PrefillLeverClass::reload},
    {"STRIX_DS41_TRELLIS_DENSE_FULL", PrefillLeverClass::reload},  // selects the dense plan at load
    {"STRIX_DS41_TRELLIS_HEADOUT", PrefillLeverClass::reload},  // baked into the plan at load (m.headout)
    {"STRIX_DS41_TRELLIS_MIX", PrefillLeverClass::reload},
    {"STRIX_DS41_TRELLIS_OA_GROUP", PrefillLeverClass::reload},
    {"STRIX_DS41_TRELLIS_OA_GROUP_LIB", PrefillLeverClass::reload},
    {"STRIX_DS41_TRELLIS_QAKV_PAIR", PrefillLeverClass::cached},
    {"STRIX_DS41_TRELLIS_ROW_VIEWS", PrefillLeverClass::cached},
    {"STRIX_DS41_TRELLIS_SITES", PrefillLeverClass::reload},
    {"STRIX_LOAD_HASH_OUT", PrefillLeverClass::reload},
    {"STRIX_SESSION_CACHE", PrefillLeverClass::reload},
    {"STRIX_SESSION_DIR", PrefillLeverClass::reload},
    {"STRIX_SESSION_DISK", PrefillLeverClass::reload},
    {"STRIX_SESSION_DISK_BYTES", PrefillLeverClass::reload},
    {"STRIX_SESSION_MAX_PARKED", PrefillLeverClass::reload},
    {"STRIX_SESSION_RAM_BYTES", PrefillLeverClass::reload},
};

}  // namespace strix::deepseek41
