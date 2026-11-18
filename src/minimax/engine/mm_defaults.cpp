#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif
// MiniMax M3.1 defaults-on resolver: the table of default-ON items, the baseline master switch and the canonical
// effective configuration (see mm_defaults.hpp).
#include "minimax/engine/mm_defaults.hpp"
#include "strix/env_flags.hpp"
#include "minimax/engine/mm_profile_table.hpp"
#include "minimax/engine/route_identity.hpp"
#include "core/layer_set.hpp"
#include "moe_core/grouped_policy.hpp"
#include "minimax/serve/minimax_sched.hpp"
#include "minimax/dspark/minimax_auto.hpp"
#include "strix/session/speculation_default.hpp"

#include <array>
#include <cstdio>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <mutex>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <set>
#include <sstream>
#include <stdexcept>

#ifndef STRIX_MM_GU4_SCALAR
#define STRIX_MM_GU4_SCALAR 1
#endif

namespace strix::minimax::defaults {
namespace {

// Measurement evidence for the adopted production defaults.
constexpr Item kItems[] = {
    {"serve_dspark", "STRIX_MM_SERVE_DSPARK", "auto", "0", "Frozen structural-v1 admission; per-content node qualification pending"},
    {"serve_lookup", "STRIX_MM_LOOKUP", "1", "0",
     "Serve measurement: token ids identical in every arm; copy-heavy rewrite C1 without a DSpark draft 23.65/23.44 -> 26.01/25.69 tok/s, agent / prose / C4 neutral"},
    {"serve_sched", "STRIX_MM_SERVE_SCHED", "interleave", "off", "Bounded serving policy; GPU qualification pending"},
    {"serve_service", "STRIX_MM_SERVE_SERVICE_MS", "250", "0", "Soft service opportunity target; GPU qualification pending"},
    {"serve_nest", "STRIX_MM_SERVE_NEST", "1", "0", "Nested short admission; extended resume-check GPU qualification pending"},
    {"serve_short_first", "STRIX_MM_SERVE_SHORT_FIRST", "1", "0", "Stable shortest-first admission, at most eight overtakes; qualification pending"},
    // decode
    {"moe_wide", "STRIX_MMX_MOE_WIDE", "2", "0",
     "Measurement: rows == alone, wide expert kernels at N2-4, N2 -3.0..-3.5 %, N4 -4..-6 %"},
    {"moe_gu_interleave", "STRIX_MMX_MOE_GU_INTERLEAVE", "1", "0",
     "Decode MoE measurement: bitwise (610 N/U/S cases, every row == alone), B8 gate/up interleave diverse U8-32 -1.5..-3 %, B1/B4 unchanged"},
    {"moe_single", "STRIX_MMX_MOE_SINGLE", "1", "0",
     "Decode MoE measurement: bitwise (610 N/U/S cases, every row == alone), B8 U>=25 singleton reader -3 %, B1/B4 unchanged"},
    {"moe_down_split", "STRIX_MMX_MOE_DOWN_SPLIT", "1", "0",
     "Decode MoE measurement: bitwise (610 N/U/S cases, every row == alone), B4 down partials + ordered finish, fresh-process "
     "OFF/ON/OFF/ON/OFF B/A U4 0.986/0.986, U8 0.993/0.989, U12 0.991/0.997, U16 0.986/1.000; B4 wide W4 Depth4 only"},
    {"index_conc", "STRIX_MM_INDEX_CONC", "1", "0",
     "Measurement: ids identical, 17/18 in-process block pairs faster, ~-0.2 ms/token"},
    {"attend_diet", "STRIX_MM_ATTEND_DIET", "1", "0",
     "Attention measurement t9: bitwise, v3 32 KB LDS vs 42 KB at sparse N2/W1 8K 83.9 -> 76.2, 32K 139.0 -> 131.6, "
     "128K 350.2 -> 340.8 us; only on those grids (1K N8, W8, 8K N4 slower stay at 42 KB)"},
    {"attend4_decode", "STRIX_MM_ATTEND4_DECODE", "1", "0",
     "Attention measurement t12: bitwise, v4 one head per wave vs v3-full at sparse N2/W1 8K 78.6 -> 75.2, 32K 132.2 -> "
     "128.3, 128K 346.4 -> 340.6 us; only on those grids (1K N1/N8, 8K N4 slower keep v3)"},
    {"select_once", "STRIX_MM_SELECT_ONCE", "1", "0",
     "Attention measurement s1b/s2: bitwise (decode suite, v3/diet/v4), selection launch + attend select=false on sparse "
     "calls of >= 4 rows at >= 16K: index+attend 32K N8 455.3 -> 436.0, N4 241.5 -> 237.1, W8 220.4 -> 206.4 us"},
    {"xfer_crc_deferred", "STRIX_MMX_CRC", "deferred", "1",
     "Measurement: tokens identical in 7 processes, each deferred arm below its inline neighbours by 0.04-0.12 ms/token"},
    {"xb_wait", "STRIX_MM_XB_WAIT", "1", "0", "Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token"},
    {"xb_inplace", "STRIX_MM_XB_INPLACE", "1", "0", "Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token"},
    {"xb_publish", "STRIX_MM_XB_PUBLISH", "1", "0",
     "Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token (falls back without the v4/overlap cores)"},
    {"xb_postilp", "STRIX_MM_XB_POSTILP", "2", "0", "Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token"},
    {"xb_warm", "STRIX_MM_XB_WARM", "8", "0", "Exchange boundary measurement: exact, 1K 48.56 -> 44.40 ms/token"},
    {"xb_warm_o", "STRIX_MM_XB_WARM_O", "8", "0",
     "Decode measurement: exact (decode logits bitwise both ranks, N1/N4/N8 + verify window), C1 8K in-process ABA x3 44.64-44.66 -> 44.01-44.07 ms/token (8 MB; 14 MB slower)"},
    {"moe_tight_grid", "STRIX_MMX_MOE_TIGHT_GRID", "1", "0",
     "Decode measurement: exact (decode logits bitwise both ranks, N1/N4/N8 + verify window), C1 8K in-process ABA x3 45.12-45.18 -> 44.60-44.68 ms/token"},
    {"moe_pipe", "STRIX_MMX_MOE_PIPE", "2", "0",
     "Decode measurement: exact (decode logits bitwise both ranks, N1/N2/N4 + verify window), calls of 2-4 rows, in-process ABA x3 N2 58.9-59.3 -> 57.3-57.7 ms/step (-2.7 %), N4 81.8-82.6 -> 80.6-80.8 ms/step (-1.6 %)"},
    {"dspark_attn", "STRIX_MM_DSPARK_ATTN", "2", "0",
     "Drafter measurement: bitwise, drafter attention B/A 0.964-0.975 / 0.918-0.924 / 0.815-0.819 at ctx 300/1000/4095"},
    {"draft_lin", "STRIX_MM_DRAFT_LIN", "opt", "lib", "Optimized draft linears (measurement evidence)"},
    {"linear_v11", "STRIX_MMX_LINEAR_V11", "1", "0",
     "Draft linear measurement: same bits as v10 (bitwise, ids == greedy 36/36), DSpark draft block -11.9 %"},
    // DSpark Markov chain (acceptance only, verify unchanged; C / |H| / top-C act on the cand form)
    {"markov_cand", "STRIX_MM_MARKOV", "cand", "full",
     "Markov measurement: cand C4096/H8192 + device exchange + top-C v2, verify exact, acceptance unchanged"},
    {"markov_cand_c", "STRIX_MM_MARKOV_C", "4096", "1024",
     "Markov measurement: C 4096 (offline approx1-K3: C4096/H8192 0 acceptance changes on 1857 anchors)"},
    {"markov_cand_h", "STRIX_MM_MARKOV_H", "8192", "2048",
     "Markov measurement: |H| 8192 (offline approx1-K3: C4096/H8192 0 acceptance changes on 1857 anchors)"},
    {"markov_topc_v2", "STRIX_MM_MARKOV_TOPC", "v2", "v1",
     "Markov measurement: top-C v2 = v1 canonical set / scores / picks (payload gates v1/v2 vs CPU)"},
    // prefill
    {"pf_msa_batch", "STRIX_MM_PF_MSA_BATCH", "1", "0", "Prefill measurement: exact, pfall TTFT 1K -17 %, 8K -20 %"},
    {"pf_dense_mlp", "STRIX_MM_PF_DENSE_MLP", "1", "0",
     "Prefill: exact (== strix_gemm bitwise, the pf_dense_qkv contract); dense MLP of layers 0-2 on the v4 core for 4096-row pieces"},
    {"pf_flash_keys", "STRIX_MM_PF_FLASH_KEYS", "2", "0",
     "Prefill: schedule only, same bits (FAST flash output fnv identical); key fragments in two halves around the value loads (attn_core kKsKeysHalves)"},
    {"pf_msa_rg_rows", "STRIX_MM_PF_MSA_RG_ROWS", "auto", "4",
     "Prefill: schedule only, same bits (row-group flash mode 3 output independent of rows); auto = 8 rows per work-group on dense windows, 4 on sparse rows (attn_core RowGroup Rows; bench scripts/bench/mmattn)"},
    {"pf_headless", "STRIX_MM_PF_HEADLESS", "1", "0",
     "Prefill: exact (state + final rows fnv identical); last layer cache-only for every chunk but the last (strix/prefill_headless.hpp)"},
    {"pf_attend4", "STRIX_MM_PF_ATTEND4", "1", "0", "Prefill measurement: exact, pfall TTFT 1K -17 %, 8K -20 %"},
    {"pf_router_batch", "STRIX_MM_PF_ROUTER_BATCH", "1", "0", "Prefill measurement: exact, pfall TTFT 1K -17 %, 8K -20 %"},
    {"pf_mtfit", "STRIX_MMX_PREFILL_MTFIT", "1", "0", "Prefill measurement: exact, pfall TTFT 1K -17 %, 8K -20 %"},
    {"pf_msa_q16", "STRIX_MM_PF_MSA_Q16", "1", "0",
     "Prefill: existing exact query-image preparation for the explicit FAST attention route; stores the same BF16 norm/RoPE query image once instead of repeating the flash prologue; no arithmetic identity promotion"},
    {"pf_msa_rework", "STRIX_MM_PF_MSA_REWORK", "10", "0", "Prefill attention measurement: exact; mode 10 v2 (in-block chunk loop with LDS-staged block list/partials, attend4 112 VGPR / 0 scratch) vs 8 attend -3.2 %, TTFT 8K 13.418 -> 13.230 s (-1.4 %, 2 pairs), 32K 61.24 -> 60.94 s (-0.5 %, 5 pairs, within spread), state/final-row fnv identical in every arm, =8 restores the previous default; mode 8 (mode 7 with the 16-key tile max as an exact ds_swizzle lane tree) vs 7 8K in-process -2.6 %, 16K -0.5 %, 32K fresh process 8/7/8 + 7/8/7 -2.6 % (-1.67 s), result hash identical on both ranks; mode 7 (combine fused into the last chunk block) vs 6 attend -8.3 %/call at 28672:4096, TTFT bracketed in-process A/B vs mode 6 8K -5.0 % (-0.78 s), 16K -3.0 % (-0.93 s); 32K fresh process 6/7/6 + 7/6/7 -2.4 % (-1.59 s), result hash identical on both ranks; mode 6 (tile corrections in the packed exp lanes) vs 5 8K in-process -1.6/0/0 %, mode 3 8K B/A 0.963 / 0.957 / 0.972"},
    {"pf_idx_tile", "STRIX_MM_PF_IDX_TILE", "1", "0",
     "Prefill index measurement: exact (state/final rows identical), index scoring 8K -6.7 %, 32K -6.4 %; TTFT 8K -1.9 %, 32K -0.6 %, 1K neutral"},
    {"pf_idx_rows", "STRIX_MM_PF_IDX_ROWS", "64", "32",
     "Prefill index measurement: exact (state/final rows identical), Q64 vs Q32 index scoring 8K -35 %, 32K -28 %; TTFT 8K -1.5 %, 32K -1.8 %, 1K neutral"},
    {"pf_moe_pipe", "STRIX_MMX_PREFILL_PIPE", "1", "0", "Prefill measurement: exact, 8K B/A 0.973 / 0.935 / 0.924"},
    {"pf_moe_pipe_pair", "STRIX_MMX_PREFILL_PIPE_PAIR", "auto", "0",
     "Prefill MoE measurement: exact (partials/xt/at/output bitwise), paired MT8 bands for calls >= 2048 rows: MoE call -0.6..-1.0 % 2048 / -0.9..-1.5 % 4096 rows, 1024 mixed"},
    {"pf_moe_tail_stream", "STRIX_MMX_PREFILL_TAIL_STREAM", "1", "0",
     "Prefill MoE measurement: exact (partials/xt/at/output bitwise), tail classes beside MT8: MoE call -1.9 to -3.0 % real routes"},
    {"pf_moe_pipe_tail", "STRIX_MMX_PREFILL_PIPE_TAIL", "1", "0",
     "Prefill MoE measurement: bitwise (pipe-tail fixture 12,480 cells, logits and state fnv of both ranks identical), tail rows (MT 1..7) on the pipelined kernel: PAIR TTFT 8K 12.68 -> 12.47 s, 32K 51.81 -> 50.91 s"},
    {"pf_gather_tok", "STRIX_MMX_PREFILL_GATHER_TOK", "1", "0",
     "Prefill MoE measurement: exact (tiled xt/at, partials, output bitwise), gather 6.35 -> 4.14 ms per 4096-row call"},
    {"pf_router_rows", "STRIX_MM_PF_ROUTER_ROWS", "32", "8",
     "Prefill router measurement: exact (routes/weights bitwise vs fused and 4/8-row forms), post+router 4096 rows 5.40 -> 4.82 ms; TTFT 1K -0.9 %, 8K -0.8 %"},
    {"pf_msa_scratch_mb", "STRIX_MM_PF_MSA_SCRATCH_MB", "1152", "512",
     "Prefill attention: exact (independent windows), one MSA batch per 4096-row piece instead of three; kept at 512 below 4 GiB headroom"},
    {"pf_dense_o", "STRIX_MM_PF_DENSE_O", "1", "0",
     "Prefill measurement: exact (== strix_gemm bitwise, T 9..4096), o_proj 7.89 -> 7.59 ms per 4096-row piece; full pieces only"},
    {"pf_dense_qkv", "STRIX_MM_PF_DENSE_QKV", "1", "0",
     "Prefill measurement: exact (== strix_gemm bitwise, T 9..4096), q+k+v 12.76 -> 8.87 ms at 4096 rows; with pf_gather_tok TTFT 8K -4.0..-4.3 %, 32K -4.3 %"},
    {"pf_kv_tile", "STRIX_MM_PF_KV_TILE", "auto", "off",
     "Prefill measurement: exact, 8K B/A 0.960 / 0.949 / 1.017; needs the Trellis tile hook, else a logged no-op"},
    {"pf_tx_uncached", "STRIX_MM_PF_TX_UNCACHED", "1", "0",
     "Prefill exchange measurement: exact, exchange-send 0.23 -> 0.002 s GPU, ~1.3 % (prefill module route)"},
    {"pf_async_start", "STRIX_MM_PF_ASYNC_START", "1", "0",
     "Grouped all-member >=2048-row gate (configurable); TP2 4096+3844 IDs identical, -2.3/-2.5 % with late fence; full state/workload qualification pending"},
    {"pf_late_src_fence", "STRIX_MM_PF_LATE_SRC_FENCE", "1", "0",
     "Admitted grouped async source lifetime; TP2 IDs identical, combined -2.3/-2.5 %; full state/workload qualification pending"},
    {"pf_order", "STRIX_MMX_PF_ORDER", "invariant", "adaptive",
     "Chunk-invariant linear order mmx-pf-linear-s1-v1 (bits independent of chunking; cache identity prefill_order=): gate32 6 x 32K "
     "units vs 3 L0 nulls each, pooled dNLL -0.053 CI95 [-0.148,+0.041] inside null spread, task 40/42 == 40/42; "
     "a defaulted value falls back to adaptive when the invariant route is unmet"},
    // transport (host path, bits unchanged)
    {"dist_crc_parallel", "STRIX_DIST_CRC_PARALLEL", "1", "0", "Host transport measurement: ABA x3 TTFT 1K -12.8 %, decode -1 %, bits unchanged"},
    {"dist_host_crc_overlap", "STRIX_DIST_HOST_CRC_OVERLAP", "1", "0", "Host transport measurement: ABA x3 TTFT 1K -12.8 %, decode -1 %, bits unchanged"},
    {"dist_host_chunk", "STRIX_DIST_HOST_CHUNK_BYTES", "1048576", "0", "Host transport measurement: ABA x3 TTFT 1K -12.8 %, decode -1 %, bits unchanged"},
    {"pfx_joined", "STRIX_MM_PFX_JOINED", "1", "0",
     "Prefill exchange wire form: identity and CRC list ride with the last chunk (two rendezvous fewer per message); payload bytes unchanged"},
};

// Route switches whose code default is the production route; printed as set or "default".
constexpr const char* kRoute[] = {"STRIX_MM_XFER", "STRIX_MM_PFX", "STRIX_MM_PF_INTERLEAVE"};

std::mutex g_mutex;
bool g_applied = false;
std::set<std::string> g_exported;

int export_env(const char* name, const char* value, int overwrite) {
#ifdef _WIN32
    if (!overwrite && std::getenv(name)) return 0;
    return ::_putenv_s(name, value);
#else
    return ::setenv(name, value, overwrite);
#endif
}
const char* process_env(const char* name) { return std::getenv(name); }

}  // namespace

std::span<const Item> items() { return kItems; }

std::size_t pf_async_min_rows(const char* value) {
    if (value == nullptr) return 2048;
    std::size_t rows = 0;
    const char* end = value + std::strlen(value);
    const auto parsed = std::from_chars(value, end, rows);
    if (parsed.ec != std::errc{} || parsed.ptr != end)
        throw std::invalid_argument("STRIX_MM_PF_ASYNC_MIN_ROWS: nonnegative row count required");
    return rows;
}

bool baseline(const Lookup& env) {
    const char* v = env(kBaselineEnv);
    return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
}

std::vector<Resolved> resolve(const Lookup& env) {
    const bool base = baseline(env);
    std::vector<Resolved> out;
    out.reserve(std::size(kItems));
    for (const Item& it : kItems) {
        // Empty item values are explicit in the legacy resolver too. Baseline's
        // separate nonempty/not-"0" truthiness above remains MiniMax policy.
        auto value = profile::resolve_default(it, base, env);
        out.push_back({&it, std::move(value.value), value.source});
    }
    return out;
}

namespace {
bool fast_i4_format(const Lookup& env) {
    const char* f = env("STRIX_MMX_MODEL_FORMAT");
    return f != nullptr && std::strcmp(f, "fast_i4") == 0;
}
}  // namespace

const char* msa_flash_decline_reason(bool rework, bool fused, unsigned layers) {
    if (!rework) return "STRIX_MM_PF_MSA_REWORK=0";
    if (!fused) return "STRIX_MM_ATTN_PREFILL=off";
    if (layers == 0) return "STRIX_MM_PF_MSA_FLASH_LAYERS selects no layers";
    return nullptr;
}

PrefillPolicy prefill_policy(const Lookup& env) {
    namespace fp = strix::fast_profile;
    const fp::Member& flash = kFastMembers[0];
    const char* raw = env(flash.env);
    if (fp::set(raw) && std::strcmp(raw, "0") != 0 && !fp::accepted(flash, raw))
        throw std::invalid_argument("STRIX_MM_PF_MSA_FLASH must be 0, 1, 2 or 3");
    PrefillPolicy p;
    // The MSA flash member runs on the exact EXL3 MoE base only (its gate fails on FAST_I4); an explicit value still wins.
    const bool i4 = fast_i4_format(env);
    p.canonical = fp::expand(kFastBundle, env, [&](const fp::Member&) { return !i4; });
    const auto it = p.canonical.find(flash.env);
    const std::string mode = it != p.canonical.end() ? it->second : "0";
    p.msa_flash = mode == "0" ? 0U : static_cast<unsigned>(mode[0] - '0');
    p.msa_flash_requested = p.msa_flash;
    p.msa_flash_asked = fp::set(raw) ? raw : it != p.canonical.end() ? "fast_prefill" : "default";
    const char* layers = env("STRIX_MM_PF_MSA_FLASH_LAYERS");
    p.msa_flash_layers_requested = fp::set(layers) ? layers : kMsaFlashLayers;
    const auto mask = strix::core::parse_layer_set(p.msa_flash_layers_requested.c_str(), 0, kMsaLayers, "STRIX_MM_PF_MSA_FLASH_LAYERS");
    p.msa_flash_layers = fp::layer_ranges(kMsaLayers, [&](unsigned l) { return mask.test(l); });
    if (p.msa_flash != 0) {
        bool rework = true;
        // Only the rework item and the baseline switch decide it: forwarding every item through a recording lookup
        // (the engine passes envcfg::text) put the whole defaults set into the FAST arm's receipt, which the floor
        // gate then refused against an exact reference ("differs beyond its FAST layer set").
        const Lookup rework_env = [&](const char* k) -> const char* {
            return std::strcmp(k, "STRIX_MM_PF_MSA_REWORK") == 0 || std::strcmp(k, kBaselineEnv) == 0 ? env(k) : nullptr;
        };
        for (const auto& r : resolve(rework_env))
            if (std::strcmp(r.item->env, "STRIX_MM_PF_MSA_REWORK") == 0) rework = r.value != "0";
        const char* fused = env("STRIX_MM_ATTN_PREFILL");
        if (const char* reason = msa_flash_decline_reason(rework, fused == nullptr || std::strcmp(fused, "off") != 0, mask.count())) {
            p.msa_flash = 0;
            p.note = reason;
        }
    }
    if (p.msa_flash != 0)
        p.policies = p.msa_flash == 1 && msa_flash_policy(p.msa_flash_layers) != nullptr
                         ? std::string(msa_flash_policy(p.msa_flash_layers))
                         : std::string("msa-flash-bf16") + (p.msa_flash == 2 ? "-2term" : p.msa_flash == 3 ? "-rg4" : "") + "-l" + p.msa_flash_layers;
    if (i4 && fp::switch_on(env(kFastBundle.env)) && !fp::set(raw))
        p.note = std::string(flash.policy) + " not selected by " + kFastBundle.env + " on STRIX_MMX_MODEL_FORMAT=fast_i4";
    // Index projections on the shared WMMA core (opt-in member, mm_profile_table.hpp kIdxWmmaPolicy).
    const char* iw = env("STRIX_MM_PF_IDX_WMMA");
    if (fp::set(iw) && std::strcmp(iw, "0") != 0 && std::strcmp(iw, "1") != 0)
        throw std::invalid_argument("STRIX_MM_PF_IDX_WMMA must be 0 or 1");
    p.idx_wmma_asked = fp::set(iw) ? iw : "default";
    p.idx_wmma = fp::set(iw) && std::strcmp(iw, "1") == 0;
    const char* il = env("STRIX_MM_PF_IDX_WMMA_LAYERS");
    const auto imask = strix::core::parse_layer_set(fp::set(il) ? il : kIdxWmmaLayers, 0, kMsaLayers, "STRIX_MM_PF_IDX_WMMA_LAYERS");
    p.idx_wmma_layers = fp::layer_ranges(kMsaLayers, [&](unsigned l) { return imask.test(l); });
    if (p.idx_wmma && imask.count() == 0) {
        p.idx_wmma = false;
        p.note += std::string(p.note.empty() ? "" : "; ") + "STRIX_MM_PF_IDX_WMMA_LAYERS selects no layers";
    }
    // Resolve the same prefill layout as dispatch; decode's <=8-row policy is irrelevant here.
    const auto index = route::resolve_index_projection(env("STRIX_MM_INDEX_PROJ"), true);
    const bool packed = route::index_projection_route(index, 9) == route::IndexLayout::packed;
    const bool fp32 = route::fp32_profile(env("STRIX_MM_PREFILL_IDENTITY"));
    if (p.idx_wmma && (!packed || fp32)) {
        p.idx_wmma = false;
        p.note += std::string(p.note.empty() ? "" : "; ") +
                  (!packed ? "index WMMA requires STRIX_MM_INDEX_PROJ=attn" : "index WMMA requires the default prefill identity");
    }
    // Shared TP2 exchange format: an explicit f16 is its own named FAST policy (tp2-xchg-f16-v0), like an explicit
    // FAST member; the exact baseline (STRIX_MM_BASELINE=1) refuses it (fail closed) and no profile selects it.
    const auto xf = strix::xport::resolve_exchange_format(env(strix::xport::kExchangeFormatEnv), !baseline(env),
                                                          "tp2-xchg-f16-v0, not under STRIX_MM_BASELINE=1");
    p.exchange_fmt = xf.format;
    p.exchange_fmt_reason = xf.reason;
    if (xf.format == strix::xport::ExchangeFormat::f16)
        p.policies += std::string(p.policies.empty() ? "" : "+") + "tp2-xchg-f16-v0";
    if (p.idx_wmma) p.policies += (p.policies.empty() ? "" : "+") + idx_wmma_policy(p.idx_wmma_layers);
    // FAST router product (bundle member or its own switch): the layers it runs on; layers without a router (dense)
    // are ignored. Admitted like the flash member (EXL3 MoE base only); an explicit value wins.
    const char* router = env("STRIX_MM_PF_ROUTER_WMMA");
    if (fp::set(router) && std::strcmp(router, "0") != 0 && std::strcmp(router, "1") != 0)
        throw std::invalid_argument("STRIX_MM_PF_ROUTER_WMMA must be 0 or 1");
    const auto rit = p.canonical.find("STRIX_MM_PF_ROUTER_WMMA");
    const char* rlayers = env("STRIX_MM_PF_ROUTER_WMMA_LAYERS");
    const auto rmask = strix::core::parse_layer_set(fp::set(rlayers) ? rlayers : kRouterWmmaLayers, 0, kMsaLayers,
                                                    "STRIX_MM_PF_ROUTER_WMMA_LAYERS");
    p.router_wmma_layers = fp::layer_ranges(kMsaLayers, [&](unsigned l) { return rmask.test(l); });
    p.router_wmma = rit != p.canonical.end() && rit->second == "1" && rmask.count() != 0;
    return p;
}

std::string idx_wmma_policy(const std::string& layers) { return std::string(kIdxWmmaPolicy) + "-l" + layers; }

strix::pair::PipelineChoice pair_pipeline_choice(std::uint64_t rows) {
    // Hypothetical reference only: memory is assumed, not measured. This choice cannot enable execution.
    // MMPIPE-PLAN section 1 (8K FAST stack): P(u) = 580 + 1.9946 u ms per chunk over both ranks (W 8.75 s at 4,096
    // rows, 9.33 s at 2,048), FP32 seam 22 ms per 4,096 rows at 4.58 GB/s, TP2 resident 9,810 ms at 8K.
    strix::pair::PipelineCost cost{580.0, 1.9946, 0.0, 0.0, 0.0, 0.0};
    cost.hop_ms_per_row = 22.0 / 4096.0;
    cost.tp_ms_per_row = 9810.0 / 8192.0;
    static constexpr std::array<std::uint32_t, 4> chunks{1024U, 2048U, 4096U, 8192U};
    static constexpr std::array<std::uint32_t, 6> virtual_stages{1U, 2U, 3U, 5U, 6U, 10U};
    const strix::pair::LayoutDecl reference_layout{strix::pair::Residency::tensor_split, 31.0e9, 41.6e9,
        "src/minimax/engine/engine.hip:89-91,1120-1131"};
    return strix::pair::choose_pipeline(rows, cost, chunks, [](std::uint32_t) { return true; }, virtual_stages,
        [](std::uint32_t v) { return 2U * v <= 60U; }, reference_layout);
}

const profile::Engine& profile_engine() {
    static const profile::Engine engine{"minimax-pair", "", profile_table, false, "minimax-turbo-v1"};
    return engine;
}

profile::Resolution profile_resolution(const Lookup& env) {
    const auto p = prefill_policy(env);
    const auto base = strix::fast_profile::lookup(p.canonical, env);
    const std::string effective_mode = std::to_string(p.msa_flash);
    profile::Inputs in;
    in.env = [&](const char* key) -> const char* {
        // The layer set is read only while the flash attention runs: an unused setting is not part of the identity.
        if (std::strcmp(key, "STRIX_MM_PF_MSA_FLASH") == 0)
            return p.msa_flash == 0 && p.msa_flash_asked == "default" ? nullptr : effective_mode.c_str();
        if (std::strcmp(key, "STRIX_MM_PF_MSA_FLASH_LAYERS") == 0)
            return p.msa_flash == 0 ? nullptr : p.msa_flash_layers.c_str();
        if (std::strcmp(key, "STRIX_MM_PF_IDX_WMMA") == 0)
            return !p.idx_wmma && p.idx_wmma_asked == "default" ? nullptr : p.idx_wmma ? "1" : "0";
        if (std::strcmp(key, "STRIX_MM_PF_IDX_WMMA_LAYERS") == 0)
            return p.idx_wmma ? p.idx_wmma_layers.c_str() : nullptr;
        // The same for the FAST router product: unset or 0 is the builtin, its layer set counts only while it runs.
        if (std::strcmp(key, "STRIX_MM_PF_ROUTER_WMMA") == 0) return p.router_wmma ? "1" : nullptr;
        if (std::strcmp(key, "STRIX_MM_PF_ROUTER_WMMA_LAYERS") == 0) return p.router_wmma ? p.router_wmma_layers.c_str() : nullptr;
        return base(key);
    };
    // Product profile: '--profile NAME' of rank and serve, exported by print_or_apply() (one value for both apps).
    if (const char* name = env(kProductProfileEnv); name != nullptr && *name != 0) in.profile = profile::parse(name);
    return profile::resolve(profile_engine(), in);
}

bool product_option(const std::string& a, const std::function<std::string()>& next, std::string* timing) {
    if (a == profile::turbo::kWeightsOption || a == profile::turbo::kBitsOption || a == profile::turbo::kPackOption) {
        (void)next();  // exported by print_or_apply()
        return true;
    }
    if (a == "--timing-profile") {
        if (timing == nullptr) throw std::invalid_argument("--timing-profile is a strix-minimax-rank option");
        *timing = next();
        return true;
    }
    if (a != "--profile") return false;
    const std::string value = next();
    if (profile::is_name(value)) return true;  // the product profile, exported by print_or_apply()
    // Explicit compatibility: the old timing spelling NAME:PROMPT.u32:STEPS keeps working on rank, with one notice.
    if (timing == nullptr || value.find(':') == std::string::npos)
        throw std::invalid_argument("--profile must be default, fast, quality or turbo (got '" + value +
                                    "'); the per-phase timing session is --timing-profile NAME:PROMPT.u32:STEPS");
    std::fprintf(stderr, "strix-minimax-rank: '--profile %s' is the old spelling of '--timing-profile %s'; "
                         "--profile now names the product profile\n", value.c_str(), value.c_str());
    *timing = value;
    return true;
}

void check_peer(const profile::Resolution& mine, const std::function<void(const void*, void*, std::size_t)>& exchange) {
    profile::check_peer(mine, [&](std::span<const std::byte> out, std::span<std::byte> in) {
        exchange(out.data(), in.data(), out.size());
    });
}
void check_peer(const std::function<void(const void*, void*, std::size_t)>& exchange) {
    check_peer(profile_resolution(process_env), exchange);
}

std::string linear_order(const Lookup& env) {
    for (const auto& r : resolve(env))  // a table default counts exactly like an explicit value
        if (std::strcmp(r.item->env, "STRIX_MMX_PF_ORDER") == 0) return r.value;
    const char* v = env("STRIX_MMX_PF_ORDER");
    return v != nullptr && *v != '\0' ? std::string(v) : std::string("adaptive");
}

std::string prefill_identity(const Lookup& env) {
    const auto p = prefill_policy(env);
    // MSA attention policies under prefill_msa (unchanged form), the index projection member under prefill_idxproj.
    std::string msa, idx;
    for (std::size_t a = 0; a < p.policies.size();) {
        const std::size_t b = std::min(p.policies.find('+', a), p.policies.size());
        const std::string one = p.policies.substr(a, b - a);
        if (one == "tp2-xchg-f16-v0") { a = b + 1; continue; }
        std::string& dst = one.rfind(kIdxWmmaPolicy, 0) == 0 ? idx : msa;
        dst += (dst.empty() ? "" : "+") + one;
        a = b + 1;
    }
    std::string out = msa.empty() ? std::string() : "prefill_msa=" + msa;
    if (!idx.empty()) out += std::string(out.empty() ? "" : ";") + "prefill_idxproj=" + idx;
    if (p.exchange_fmt != strix::xport::ExchangeFormat::f32)
        out += std::string(out.empty() ? "" : ";") + "tp2_exchange_fmt=" + strix::xport::exchange_format_name(p.exchange_fmt);
    if (p.router_wmma) out += std::string(out.empty() ? "" : ";") + "prefill_router=mm-pf-router-wmma-v1-l" + p.router_wmma_layers;
    if (fast_i4_format(env)) out += std::string(out.empty() ? "" : ";") + "moe_prefill_format=fast_i4";
    // Chunk-invariant linear order (MoE gate/up/down + dense q/k/v/o, fixed S = 1): different bits from the adaptive
    // order, so a parked state written under one never restores into the other.
    if (linear_order(env) == "invariant") out += std::string(out.empty() ? "" : ";") + "prefill_order=mmx-pf-linear-s1-v1";
    // TURBO: versioned name + effective levers + weight identity, even with every lever off (empty outside turbo).
    if (const auto pr = profile_resolution(env); pr.is_turbo())
        out += std::string(out.empty() ? "" : ";") + pr.cache_namespace();
    return out;
}

profile::ExecutionReceipt host_receipt(const Lookup& env) {
    profile::ExecutionReceipt receipt;
    receipt.engine = "minimax-m3.1";
    receipt.implementation = "minimax-pair";
    receipt.build = STRIX_MM_GU4_SCALAR ? "gu4_scalar=on" : "gu4_scalar=off";
    // Declared in include/strix/minimax/engine.hpp: FP32 partials, then a
    // fixed rank-order sum. This is distinct from any other engine's contract
    // and does not identify the assets or a complete reduction/kernel plan.
    receipt.identity.arithmetic_contract = "MiniMax M3.1 FP32 partials with fixed rank-order sum";
    const char* prefill = env("STRIX_MM_PREFILL_IDENTITY");
    receipt.identity.prefill_lineage = route::fp32_profile(prefill) ? "fp32" : "production";
    // Kernel, device, topology, asset hashes, state format and the complete
    // reduction plan are unknown at this host-only policy boundary.
    return receipt;
}

const lever::Decl& split_frame_row() {
    static constexpr lever::Decl row = xport::split_frame_decl("mm_split_frame", 4, "LNKS2 prediction, ABAB pending");
    return row;
}

xport::SplitFramePolicy split_frame_policy() {
    static const xport::SplitFramePolicy policy = xport::split_frame_policy(split_frame_row());  // once: CLI, serve and engine agree
    return policy;
}

void apply() {
    std::lock_guard<std::mutex> g(g_mutex);
    if (g_applied) return;
    g_applied = true;
    const Lookup env = process_env;
    for (const Resolved& r : resolve(env)) {
        if (r.source == Source::env) continue;
        if (export_env(r.item->env, r.value.c_str(), 0) == 0) strix::envcfg::note_default_export(r.item->env, r.value.c_str());
        g_exported.insert(r.item->env);
    }
    if (baseline(env))
        std::printf("minimax defaults: %s=1, the %zu optimization items are off (an explicit item value wins)\n", kBaselineEnv,
                    std::size(kItems));
}

bool from_defaults(const char* env) {
    std::lock_guard<std::mutex> g(g_mutex);
    return g_exported.count(env) != 0;
}

void decline_default(const char* env, const char* value, const std::string& reason) {
    std::lock_guard<std::mutex> g(g_mutex);
    if (g_exported.count(env) == 0) throw std::logic_error(std::string("decline_default: ") + env + " was not exported by the defaults");
    (void)export_env(env, value, 1);
    std::printf("minimax defaults: %s default declined -> %s (%s)\n", env, value, reason.c_str());
}

Cli parse_cli(const std::string& app, const std::vector<std::string>& args) {
    if (app != "rank" && app != "serve") throw std::invalid_argument("parse_cli: app must be rank or serve");
    Cli c;
    c.app = app;
    if (app == "serve") c.transport = "usb4gpu";
    std::size_t i = 0;
    if (app == "rank") {  // MODEL_DIR RANK TRANSPORT OUT_DIR [options]
        if (args.size() > 2 && args[2].rfind("--", 0) != 0) c.transport = args[2];
        i = 4;
    } else {
        i = 1;  // MODEL_DIR [options]
    }
    for (; i < args.size(); ++i) {
        const std::string& a = args[i];
        const bool has = i + 1 < args.size();
        if (a == "--dspark-prior") {
            if (!has) throw std::invalid_argument("--dspark-prior requires a version");
            c.dspark_prior = dspark::prior_name(dspark::parse_prior(args[++i].c_str()));
        } else if (strix::session::speculation_opt_out_argument(a) && app == "serve") {
            if (c.dspark_mode.empty()) c.dspark_mode = "off";  // an explicit --dspark-mode wins
        } else if (a == "--dspark-mode" && app == "serve") {
            if (!has) throw std::invalid_argument("--dspark-mode requires off | auto | on");
            c.dspark_mode = dspark::mode_name(dspark::parse_mode(args[++i]));
        } else if (a == "--slots" && has) c.slots = static_cast<std::uint32_t>(std::stoul(args[++i]));
        else if (a == "--capacity" && has) c.capacity = std::stoull(args[++i]);
        else if (a == "--max-rows" && has) c.max_rows = static_cast<std::uint32_t>(std::stoul(args[++i]));
        else if (a == "--transport" && has && app == "serve") c.transport = args[++i];
    }
    return c;
}

namespace {
struct ServePolicy {
    serve::Sched sched;
    double service;
    double gap;
    bool short_first;
    bool nest;
};

// Pure host preflight, including defaults/baseline before apply() exports them.
ServePolicy serve_policy(const Lookup& env) {
    const auto resolved = resolve(env);
    const auto value = [&](const char* name) -> const char* {
        for (const auto& r : resolved)
            if (std::strcmp(r.item->env, name) == 0) return r.value.c_str();
        return env(name);
    };
    const auto flag = [&](const char* name) {
        const std::string v = value(name);
        if (v != "0" && v != "1") throw std::invalid_argument(std::string(name) + ": 0 | 1");
        return v == "1";
    };
    return {serve::sched_mode(value("STRIX_MM_SERVE_SCHED")),
            serve::service_ms(value("STRIX_MM_SERVE_SERVICE_MS")),
            serve::gap_ms(value("STRIX_MM_SERVE_GAP_MS")),
            flag("STRIX_MM_SERVE_SHORT_FIRST"), flag("STRIX_MM_SERVE_NEST")};
}
} // namespace
// Pure policy only: inspection must not query memory, storage or home paths.
SessionConfig session_policy(const Cli& cli, const Lookup& env, bool explicit_cli) {
    SessionConfig c;
    const auto value = [&](const char* key) { const char* v = env(key); return std::string(v ? v : ""); };
    const auto mode = value("STRIX_SESSION_CACHE");
    c.route = cli.app + "/" + cli.transport + "/fixed-width-one-session-v1";
    // Add routes only after native execution, fidelity and complete-workload qualification.
    // serve/usb4-auto: tools/minimax/session_cache_gate.sh on both ranks (A == L == B, OFF == baseline, unnamed
    // explicit and automatic ON == OFF with DSpark off/on and both scheduler arms).
    static constexpr std::array<const char*, 1> gpu_qualified_session_routes{"serve/usb4-auto/fixed-width-one-session-v1"};
    const bool qualified = std::find(gpu_qualified_session_routes.begin(),
                                    gpu_qualified_session_routes.end(), c.route) != gpu_qualified_session_routes.end();
    if (env("STRIX_SESSION_MAX_PARKED")) c.max_parked = std::stoull(value("STRIX_SESSION_MAX_PARKED"));
    if (!explicit_cli && (baseline(env) || (env("STRIX_BASELINE") && value("STRIX_BASELINE") != "0" && value("STRIX_BASELINE") != "off"))) c.reason = "baseline";
    else if (!explicit_cli && (mode == "0" || mode == "off")) c.reason = "switch";
    else if (cli.slots < 2) c.reason = "at least two slots required";
    else if (c.max_parked == 0) c.reason = "max parked is zero";
    // Live direct serving is unqualified even with an explicit cache opt-in.
    // Runtime callers supply the negotiated transport, not the request spelling.
    else if (cli.app == "serve" && cli.transport == kRouteGpu)
        c.reason = "route not yet GPU-qualified: " + c.route;
    else if (!explicit_cli && !qualified && mode != "1" && mode != "on")
        c.reason = "route not yet GPU-qualified: " + c.route;
    c.enabled = c.reason.empty();
    return c;
}
SessionConfig session_config(const Cli& cli, const Lookup& env, std::uint64_t available,
                             bool explicit_cli, const std::string& explicit_directory) {
    auto c = session_policy(cli, env, explicit_cli);
    auto value = [&](const char* key, const char* fallback) { const char* v = env(key); return std::string(v ? v : fallback); };
    if (available == UINT64_MAX) {
        available = 0;
#ifdef _WIN32
        MEMORYSTATUSEX m{}; m.dwLength = sizeof(m);
        if (GlobalMemoryStatusEx(&m)) available = m.ullAvailPhys;
#else
        std::ifstream mem("/proc/meminfo"); std::string line;
        while (std::getline(mem, line)) if (line.starts_with("MemAvailable:")) {
            std::istringstream in(line.substr(13)); in >> available; available *= 1024; break;
        }
#endif
    }
    // MemAvailable is read after the model load, so it is what remains for the host; a quarter admits the six-peak
    // capture guard at the default 8256-token context on the two-node serve route (~31 GB available -> ~7.7 GB).
    c.ram_bytes = std::min<std::uint64_t>(8ULL << 30, available / 4);
    if (env("STRIX_SESSION_RAM_BYTES")) c.ram_bytes = std::stoull(value("STRIX_SESSION_RAM_BYTES", "0"));
    auto xdg = value("XDG_CACHE_HOME", "");
    auto home = value("HOME", "");
    if (home.empty()) home = value("USERPROFILE", "");
    const auto default_dir = (xdg.empty() ? home + "/.cache" : xdg) + "/strix-engine/sessions";
    c.directory = value("STRIX_SESSION_DIR", default_dir.c_str());
    if (explicit_cli && !explicit_directory.empty()) c.directory = explicit_directory;
    if (c.reason.empty()) {
        const auto limit = (std::numeric_limits<std::uint64_t>::max() - (64ULL << 20)) / 76804;
        const auto payload = cli.capacity > limit ? std::numeric_limits<std::uint64_t>::max() : cli.capacity * 76804 + (64ULL << 20);
        if (payload > c.ram_bytes / 6 || available < (256ULL << 20) || payload > (available - (256ULL << 20)) / 6)
            c.reason = "budget/MemAvailable below one session capture peak";
    }
    c.enabled = c.reason.empty();
    // Automatic serving parks in host RAM only; the shared disk tier is opt-in (no per-server ownership of a directory).
    const auto disk = value("STRIX_SESSION_DISK", "0");
    c.disk = c.enabled && (explicit_cli || disk == "1" || disk == "on");
    if (c.disk && explicit_directory.empty() && xdg.empty() && home.empty() && !env("STRIX_SESSION_DIR")) { c.disk = false; c.disk_reason = "home directory unavailable"; }
    if (c.disk) {
        std::error_code ec;
        auto ancestor = std::filesystem::absolute(c.directory, ec);
        while (!ancestor.empty() && !std::filesystem::exists(ancestor, ec)) ancestor = ancestor.parent_path();
        const auto space = std::filesystem::space(ancestor, ec);
        c.disk_bytes = ec ? 0 : std::min<std::uint64_t>(64ULL << 30, space.available / 4);
        if (env("STRIX_SESSION_DISK_BYTES")) c.disk_bytes = std::stoull(value("STRIX_SESSION_DISK_BYTES", "0"));
        const auto payload = cli.capacity <= (1ULL << 40) ? cli.capacity * 76804 + (64ULL << 20) : UINT64_MAX;
        if (ec || payload > c.disk_bytes / 2 || payload > space.available / 2) { c.disk = false; c.disk_reason = "disk space below two checkpoints"; }
        if (c.disk) {
            const auto probe = std::filesystem::path(c.directory) / (".probe-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            std::filesystem::create_directories(c.directory, ec);
            bool writable = !ec;
            if (writable) { std::ofstream file(probe); file << "probe"; file.flush(); writable = bool(file); }
            std::filesystem::remove(probe, ec);
            if (!writable || ec) { c.disk = false; c.disk_reason = "directory not creatable/writable"; }
        }
    }
    return c;
}

namespace {
constexpr const char* kPeerGone =
    "a request must fail within a minute when the peer is gone; the CLI waits out an uneven two-rank load";
constexpr HostDefault kHostDefaults[] = {
    // The CRC organisation and the 1 MiB host chunks are items of the table above (apply() exports them for both
    // entry points), so they are not repeated here.
    // The ranks load at different speeds (44 s vs 166 s under page-cache pressure): the first collective after the
    // load must not hit the dist layer's 120 s default.
    {"STRIX_DIST_USB4_TIMEOUT_MS", "600000", "60000", kPeerGone, false},
    {"STRIX_DIST_WAIT_TIMEOUT_MS", "600000", "60000", kPeerGone, false},
    // Prefill MoE table ring: exact, 1K -0.5 %, 8K not slower; qualified with one engine owner only.
    {"STRIX_MM_PF_TABLE_RING", "1", nullptr,
     "qualified with a single engine owner; not qualified with several sessions sharing the engine", true},
};
}  // namespace

std::span<const HostDefault> host_defaults() { return kHostDefaults; }

const char* host_default_value(const HostDefault& row, const std::string& app, const Lookup& env) {
    if (const char* set = env(row.name)) return set;
    if (row.optimization) {
        const char* base = env(kBaselineEnv);
        if (base != nullptr && std::string(base) == "1") return nullptr;
    }
    return app == "serve" ? row.serve : row.rank;
}

void apply_host_defaults(const std::string& app) {
    const Lookup lookup = [](const char* key) { return process_env(key); };
    for (const auto& row : kHostDefaults)
        if (const char* value = host_default_value(row, app, lookup)) (void)export_env(row.name, value, 0);
}

std::string effective_config(const Cli& cli, const Lookup& env) {
    std::ostringstream o;
    o << "app=" << cli.app << "\n"
      << "transport=" << cli.transport << "\n"
      << "slots=" << cli.slots << "\n"
      << "capacity=" << cli.capacity << "\n"
      << "max_rows=" << cli.max_rows << "\n";
    for (const char* r : kRoute) {
        const char* v = env(r);
        o << r << "=" << (v != nullptr ? v : "default") << "\n";
    }
    const char* order_asked = env("STRIX_MMX_PF_ORDER");
    // requested = how the order was asked (unset = "default"); effective resolves the pf_order table item like apply().
    const auto& order = strix::moe_core::resolve_grouped_order(linear_order(env).c_str());
    o << "STRIX_MMX_PF_ORDER.requested=" << (order_asked ? order_asked : "default") << "\n"
      << "STRIX_MMX_PF_ORDER.effective=" << (order.invariant() ? "invariant" : "adaptive") << "\n";
    o << "build.gu4_scalar=" << (STRIX_MM_GU4_SCALAR ? "on" : "off") << "\n";
    for (const Resolved& r : resolve(env)) o << r.item->env << "=" << r.value << "\n";
    {   // FAST prefill profile rows (default OFF): the same resolution the engine, the cache identity and the ranks use.
        const auto pf = prefill_policy(env);
        const auto pr = profile_resolution(env);
        for (const auto& e : pr.entries) o << e.lever->env << "=" << e.value << "\n";
        if (pr.profile != profile::Name::standard) {  // default prints nothing new (legacy text unchanged)
            o << "profile.name=" << profile::name(pr.profile) << "\n" << "profile.version=" << pr.versioned() << "\n"
              << "profile.qualification=" << pr.qualification() << "\n";
            if (pr.is_turbo())
                o << "profile.weights_requested=" << pr.weights.requested_text() << "\n"
                  << "profile.weights=" << pr.weights.identity() << "\n"
                  << "profile.weights_origin=" << pr.weights.origin << "\n"
                  << "profile.weights_reason=" << (pr.weights.reason.empty() ? "-" : pr.weights.reason) << "\n"
                  << "profile.cache_namespace=" << pr.cache_namespace() << "\n";
        }
        const auto id = prefill_identity(env);
        const bool bundle_requested = strix::fast_profile::switch_on(env(kFastBundle.env));
        const auto pipeline_reference = pair_pipeline_choice(8192U);
        o << "prefill.fast_profile=" << (bundle_requested && (pf.msa_flash != 0 || pf.router_wmma) ? kFastBundle.profile : "off") << "\n"
          << "prefill.fast_profile_requested=" << (bundle_requested ? kFastBundle.profile : "off") << "\n"
          << "prefill.msa_flash_requested=" << pf.msa_flash_requested << "\n"
          << "prefill.msa_flash_source=" << pf.msa_flash_asked << "\n"
          << "prefill.msa_flash_layers_requested=" << pf.msa_flash_layers_requested << "\n"
          << "prefill.idx_wmma=" << (pf.idx_wmma ? "1" : "0") << "\n"
          << "prefill.idx_wmma_source=" << pf.idx_wmma_asked << "\n"
          << "prefill.idx_wmma_layers=" << (pf.idx_wmma ? pf.idx_wmma_layers : std::string("none")) << "\n"
          << "prefill.fast_policies=" << (pf.policies.empty() ? "none" : pf.policies) << "\n"
          << "prefill.fast_note=" << (pf.note.empty() ? "none" : pf.note) << "\n"
          << "prefill.tp2_exchange_fmt=" << strix::xport::exchange_format_name(pf.exchange_fmt) << " (" << pf.exchange_fmt_reason << ")\n"
          << "prefill.pair_pipeline=off mode=tensor residency=tensor_split free_memory=pending "
             "reason=layout: tensor-split weights; pipeline execution unimplemented\n"
          << "prefill.pair_pipeline_reference=estimate profile=FAST rows=8192 assumed_free_gb=31.0 assumed_extra_gb=41.6 "
          << strix::pair::effective_line(pipeline_reference) << "\n"
          << "prefill.identity=" << (id.empty() ? "default" : id) << "\n"
          << "profile_fingerprint=" << profile::hex_identity(pr.fingerprint()) << "\n";
    }
    o << "STRIX_MM_PF_ASYNC_MIN_ROWS=" << pf_async_min_rows(env("STRIX_MM_PF_ASYNC_MIN_ROWS")) << "\n";
    const auto prior = dspark::parse_prior(cli.dspark_prior.empty() ? env("STRIX_MM_DSPARK_PRIOR") : cli.dspark_prior.c_str());
    o << "dspark.prior=" << dspark::prior_name(prior) << "\n";
    if (cli.app == "serve") {
        const auto policy = serve_policy(env);
        std::string mode = cli.dspark_mode;
        if (mode.empty()) for (const auto& r : resolve(env))
            if (std::strcmp(r.item->env, "STRIX_MM_SERVE_DSPARK") == 0) mode = r.value;
        const auto requested = dspark::parse_mode(mode);
        o << "serve.dspark_mode=" << dspark::mode_name(requested) << "\n"
          << "serve.dspark_effective=" << (requested == dspark::Mode::on ? "on_if_assets" : requested == dspark::Mode::automatic && prior == dspark::Prior::structural_v1 ? "per_request_if_assets" : "plain") << "\n"
          << "serve.dspark_reason=" << (requested == dspark::Mode::automatic ? (prior == dspark::Prior::unqualified_v1 ? "plain:prior_unqualified" : "auto:per_request") :
                                        requested == dspark::Mode::off ? "plain:off" : "legacy") << "\n"
          << "serve.dspark_inflight=" << dspark::parse_inflight(env("STRIX_MM_SERVE_DSPARK_INFLIGHT")) << "\n"
          << "serve.dspark_prior=" << dspark::prior_name(prior) << "\n";
        o << std::setprecision(std::numeric_limits<double>::max_digits10)
          << "serve.sched=" << (policy.sched == serve::Sched::off ? "off" : "interleave") << "\n"
          << "serve.service_ms=" << policy.service << "\n"
          << "serve.gap_ms=" << policy.gap << "\n"
          << "serve.short_first=" << (policy.short_first ? 1 : 0) << "\n"
          << "serve.nest=" << (policy.nest ? 1 : 0) << "\n";
    }
    const auto session = session_policy(cli, env, false);
    const char* requested = env("STRIX_SESSION_CACHE");
    o << "session_requested=" << (requested ? requested : "auto") << "\n"
      << "session_cache=" << (session.enabled ? "pending" : "off") << "\n"
      << "session_reason=" << (session.enabled ? "runtime admission required" : session.reason) << "\n";
    for (const auto& row : kHostDefaults) {
        const char* value = host_default_value(row, cli.app, env);
        o << "host." << row.name << "=" << (value ? value : "unset") << "\n";
        if (cli.app == "serve" && row.serving_reason != nullptr && env(row.name) == nullptr)
            o << "host." << row.name << ".serving_reason=" << row.serving_reason << "\n";
    }
    o << strix::weight::family_plan_notice(cli.weight_plans);
    return o.str();
}

bool print_or_apply(const std::string& app, int argc, char** argv) {
    std::vector<std::string> args;
    bool print = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], kPrintFlag) == 0) print = true;
        else args.emplace_back(argv[i]);
    }
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--baseline") { (void)export_env(kBaselineEnv, "1", 1); (void)export_env("STRIX_SESSION_CACHE", "0", 1); }
        // Product profile and TURBO weight toggle: the CLI value wins over an inherited variable (CLI > environment).
        if (a == "--profile" && i + 1 < argc && profile::is_name(argv[i + 1])) {
            (void)export_env(kProductProfileEnv, argv[++i], 1);
            continue;
        }
        if (a == profile::turbo::kWeightsOption || a == profile::turbo::kBitsOption || a == profile::turbo::kPackOption) {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            const auto key = a == profile::turbo::kWeightsOption ? profile::turbo::kWeightsEnv :
                a == profile::turbo::kBitsOption ? profile::turbo::kBitsEnv : profile::turbo::kPackEnv;
            (void)export_env(std::string(key).c_str(), argv[++i], 1);
            continue;
        }
        const std::pair<const char*, const char*> flags[] = {
            {"--session-host-budget", "STRIX_SESSION_RAM_BYTES"}, {"--session-cache-dir", "STRIX_SESSION_DIR"},
            {"--session-cache", "STRIX_SESSION_CACHE"}, {"--session-disk", "STRIX_SESSION_DISK"},
            {"--session-ram-bytes", "STRIX_SESSION_RAM_BYTES"}, {"--session-max-parked", "STRIX_SESSION_MAX_PARKED"},
            {"--session-dir", "STRIX_SESSION_DIR"}, {"--session-disk-bytes", "STRIX_SESSION_DISK_BYTES"}};
        for (const auto& flag : flags) if (a == flag.first) {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            (void)export_env(flag.second, argv[++i], 1); break;
        }
    }
    // shared opt-out: DSpark off unless STRIX_MM_SERVE_DSPARK is set (must precede apply(), which exports the default)
    if (app == "serve" && strix::session::speculation_environment_off(process_env(strix::session::speculation_environment)))
        (void)export_env("STRIX_MM_SERVE_DSPARK", "0", 0);
    apply();
    (void)split_frame_policy();  // validate and record the lever on both entry points (rank CLI and serve)
    (void)pf_async_min_rows(process_env("STRIX_MM_PF_ASYNC_MIN_ROWS"));
    (void)effective_config(parse_cli(app, args), process_env);
    if (!print) {  // a weight request this build cannot honour refuses the run (printed, not refused, above)
        profile_resolution(process_env).weights.require_honoured();
        return false;
    }
    std::fputs(effective_config(parse_cli(app, args), process_env).c_str(), stdout);
    if (app == "serve") {
        const bool off = serve_policy(process_env).sched == serve::Sched::off;
        const char* profile = process_env("STRIX_MM_PROFILE");
        const char* identity = process_env("STRIX_MM_PREFILL_IDENTITY");
        const char* reason = off ? "sched_off" : profile != nullptr && std::atoi(profile) != 0 ? "profiling" :
                             identity != nullptr && std::strcmp(identity, "fp32") == 0 ? "fp32_prefill" : "";
        std::printf("serve.effective_sched=%s\nserve.fallback_reason=%s\nserve.max_unit_layers=dynamic\nserve.qualification=pending\n",
                    off ? "off" : *reason == 0 ? "interleave" : "unlimited", reason);
    }
    std::fflush(stdout);
    return true;
}

}  // namespace strix::minimax::defaults
