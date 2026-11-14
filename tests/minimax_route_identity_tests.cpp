// Inventory success is not numerical qualification. --require-identity rejects
// every open gate. --audit-sources (also implied by --require-identity) checks
// source fingerprints, which force a re-audit after any MiniMax source edit, even
// for new selectors that do not use the shared pure policy header.
#include "minimax/engine/route_identity.hpp"
#include "minimax/engine/mm_defaults.hpp"
#include "minimax/rtxpf/fp32_identity.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
namespace route = strix::minimax::route;
namespace fs = std::filesystem;
void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
struct Selector { const char* id; char category; const char* plan_key; const char* gate; const char* first_op; };
constexpr Selector selectors[] = {
#include "minimax_route_identity_registry.inc"
};
struct Source { const char* path; std::uint64_t hash; };
constexpr Source sources[] = {
#include "minimax_route_identity_sources.inc"
};
std::uint64_t fingerprint(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "cannot read audited source: " + path.generic_string());
    std::uint64_t hash = 14695981039346656037ULL;
    char byte = 0;
    while (input.get(byte)) {
        if (byte == '\r') continue;  // git autocrlf is not a route change
        hash = (hash ^ static_cast<unsigned char>(byte)) * 1099511628211ULL;
    }
    require(input.eof(), "source read failed");
    return hash;
}
// Audit scope: every file compiled into, or included by, the MiniMax rank/serve
// targets and the shared session owners that directly implement its request
// lifecycle. HTTP parsing, tokenization, registry, transport and device/core
// libraries remain under their own gates. Recursive rules fail closed on adds.
struct ScopeRule { const char* dir; const char* prefix; bool recursive; const char* why; };
constexpr ScopeRule scope_rules[] = {
    {"src/minimax", "", true, "strix_minimax_{engine,assets,defaults,session,serve,pfx,dspark,moe_opt_hip,model_description}"},
    {"src/moe_core", "", true, "shared MoE device core included by the MiniMax engine kernels"},
    {"src/dspark_common", "", true, "drafter kernels included by strix_minimax_dspark"},
    {"src/xport", "", true, "strix_xport INTERFACE linked by strix_minimax_engine (xfer)"},
    {"include/strix/minimax", "", true, "public headers of the strix_minimax_* libraries"},
    {"include/strix/session", "", true, "shared contracts and storage APIs consumed by MiniMax"},
    {"include/strix/serve", "session_runtime", false, "shared SessionRuntime API consumed by MiniMax"},
    {"src/session", "", true, "shared queue, storage, backend and prefix owners used by MiniMax"},
    {"apps", "minimax_", false, "strix-minimax-rank / strix-minimax-serve mains"},
    {"src", "minimax_", false, "strix_minimax_msa (minimax_msa.cpp, minimax_moe.cpp)"},
    {"src", "hip_minimax_", false, "strix_minimax_msa_hip"},
    {"include/strix", "minimax_", false, "headers of strix_minimax_msa"},
    {"include/strix", "hip_minimax_", false, "headers of strix_minimax_msa_hip"},
    {"src", "bp_codec.", false, "compiled directly into strix-minimax-rank"},
    {"include/strix", "bp_codec.", false, "header of bp_codec.cpp"},
};
constexpr const char* exact_scope_files[] = {
    "cmake/StrixSession.cmake", "include/strix/serve/engine.hpp",
    "src/serve/engine.cpp", "src/serve/session_runtime.cpp"};
void source_coverage(const fs::path& source_root = fs::path(STRIX_ROUTE_SOURCE_DIR)) {
    const fs::path root = source_root;
    std::set<std::string> found, expected;
    found.insert("CMakeLists.txt");  // build-time arithmetic/addressing switches
    for (const auto* path : exact_scope_files) found.insert(path);
    const auto take = [&](const fs::directory_entry& entry, const ScopeRule& rule) {
        if (entry.is_regular_file() && entry.path().filename().generic_string().starts_with(rule.prefix))
            found.insert(entry.path().lexically_relative(root).generic_string());
    };
    for (const auto& rule : scope_rules) {
        require(fs::is_directory(root / rule.dir), std::string("audit scope directory missing: ") + rule.dir);
        if (rule.recursive)
            for (const auto& entry : fs::recursive_directory_iterator(root / rule.dir)) take(entry, rule);
        else
            for (const auto& entry : fs::directory_iterator(root / rule.dir)) take(entry, rule);
    }
    std::string stale;
    for (const auto& source : sources) {
        require(expected.insert(source.path).second, "duplicate source receipt");
        if (!fs::is_regular_file(root / source.path) || fingerprint(root / source.path) != source.hash)
            stale += "\n  stale receipt: " + std::string(source.path);
    }
    std::string drift;
    for (const auto& path : found)
        if (!expected.contains(path)) drift += "\n  untracked in-scope: " + path;
    for (const auto& path : expected)
        if (!found.contains(path)) drift += "\n  receipt outside scope: " + path;
    require(stale.empty() && drift.empty(), "UNCOVERED source receipt/census drift; audit selectors before updating receipt" + stale + drift);
}
const Selector& allow(std::string_view id) {
    for (const auto& selector : selectors) if (id == selector.id) return selector;
    throw std::runtime_error("UNCOVERED selector: " + std::string(id));
}
void gated_difference(std::string_view id, const std::string& op) {
    const auto& entry = allow(id);
    require(std::string_view(entry.plan_key).size() != 0 && std::string_view(entry.gate).size() != 0,
            "missing explicit plan/gate");
    std::cout << (entry.category == 'c' ? "IDENTITY_FAILURE " : "GATED_EQUIVALENCE ")
              << id << " first_divergent_op=" << op << " plan_key=" << entry.plan_key << " gate=" << entry.gate << '\n';
}
using Tuple = std::array<unsigned, 4>;
void compare_tuple(std::string_view id, const Tuple& alone, const Tuple& together) {
    if (alone == together) return;
    // Every physical tactic change needs an explicit entry, even when the
    // arithmetic equivalence has a construction proof or an existing GPU gate.
    const auto& entry = allow(id);
    require(std::string_view(entry.plan_key).size() != 0 && std::string_view(entry.gate).size() != 0,
            "UNCOVERED route tuple difference");
}
struct Row { std::uint32_t slot; std::uint64_t pos; };
void decode_moe_selectors() {
    for (unsigned n : {7U, 8U})
        for (unsigned u : {24U, 25U, 26U, 32U})
            for (unsigned s = 0; s <= u; ++s) {
                // U distinct experts: S degree-one, others degree>=2, total degree 4N.
                // Reused experts can occupy at most N rows (top-four IDs distinct per row).
                if (u > 4 * n || 2 * u - s > 4 * n || s + (u - s) * n < 4 * n) continue;
                for (bool enabled : {false, true}) {
                    require(route::moe_interleave(n, enabled) == (enabled && n == 8), "interleave N7/8");
                    compare_tuple("decode-moe-interleave", {0,0,0,0},
                                  {route::moe_interleave(n, enabled) ? 1U : 0U,0,0,0});
                    for (bool shared : {false, true})
                        for (unsigned slot_rows : {1U, 2U, n})
                            for (unsigned bits : {3U, 5U})
                                for (bool wide : {false, true})
                                    for (unsigned waves : {4U, 8U})
                                        for (unsigned depth : {4U, 8U, 12U}) {
                                            const bool expected = enabled && n == 8 && u >= 25 && s >= 18 &&
                                                !shared && slot_rows == 1 && bits == 3 && !wide && waves == 4 && depth == 4;
                                            const bool got = route::moe_single_reader(enabled, n, u, s, shared,
                                                                                     slot_rows, bits, wide, waves, depth);
                                            require(got == expected, "singleton N/U/S/slot/K/wide/W/depth admission");
                                            compare_tuple("decode-moe-singleton", {0,0,0,0}, {got ? 1U : 0U,0,0,0});
                                        }
                }
            }
    // Down partials + ordered finish: mode 1 only at the measured B4 wide W4 Depth4 regime, mode 2 everywhere.
    for (unsigned mode : {0U, 1U, 2U})
        for (unsigned n = 1; n <= 8; ++n)
            for (bool wide : {false, true})
                for (unsigned waves : {4U, 8U})
                    for (unsigned depth : {4U, 8U, 12U}) {
                        const bool got = route::moe_down_split(mode, n, wide, waves, depth);
                        require(got == (mode == 2 || (mode == 1 && n == 4 && wide && waves == 4 && depth == 4)),
                                "down split N/wide/W/depth admission");
                        compare_tuple("decode-moe-down-split", {0,0,0,0}, {got ? 1U : 0U,0,0,0});
                    }
    require(!route::moe_single_slot(8, 24, 16, false, 1) &&
            route::moe_single_slot(8, 25, 18, false, 1) &&
            !route::moe_single_slot(8, 25, 17, false, 1) &&
            route::moe_single_slot(8, 26, 20, false, 1), "singleton feasible S and threshold boundaries");
}
void decode_and_verify() {
    decode_moe_selectors();
    // Row zero is the same token at the same position in every companion set.
    for (unsigned n = 1; n <= 8; ++n) {
        std::vector<Row> rows;
        for (unsigned r = 0; r < n; ++r) rows.push_back({r, r == 0 ? 1000ULL : 9000ULL + r});
        const auto layout = route::step_layout(std::span<const Row>(rows));
        require(layout.seqs == n && layout.rows_per_seq == 1, "decode layout");
        for (auto& row : rows) row.slot = 7;  // DSpark R=1..8, consecutive positions
        for (unsigned r = 0; r < n; ++r) rows[r].pos = 1000ULL + r;
        const auto verify = route::step_layout(std::span<const Row>(rows));
        require(verify.seqs == 1 && verify.rows_per_seq == n, "verify layout");
        const auto index = route::index_route(n, 0, 2, 0);
        require(index.ring == (n == 1 ? 32U : 16U) && index.rows_per_wave == (n > 2 ? 2U : 0U) &&
                !index.lds && !index.tiled, "index projection route");
        require(route::wide_for(n, 2) == (n >= 2 && n <= 4), "wide decode route");
        for (bool dense : {false, true})
            for (std::uint64_t pos : {std::uint64_t{2047}, std::uint64_t{8190}, std::uint64_t{8191}, std::uint64_t{131071}}) {
                const bool attend = route::attend_n2_sparse(layout.seqs, layout.rows_per_seq, dense, pos);
                require(attend == (!dense && n == 2 && pos >= 8191), "attend N2/W1 sparse >= 8K grid");
                require(!route::attend_n2_sparse(verify.seqs, verify.rows_per_seq, dense, pos),
                        "verify windows keep the default attend");
                compare_tuple("decode-attend-grid", {0,0,0,0}, {attend ? 1U : 0U,0,0,0});
            }
        for (bool enabled : {false, true})
            for (bool dense : {false, true})
                for (std::uint64_t pos : {std::uint64_t{8191}, std::uint64_t{16382}, std::uint64_t{16383}, std::uint64_t{131071}}) {
                    const bool once = route::select_once(enabled, n, dense, pos);
                    require(once == (enabled && !dense && n >= 4 && pos >= 16383), "selection once: sparse >= 4 rows, >= 16K");
                    compare_tuple("decode-select-once", {0,0,0,0}, {once ? 1U : 0U,0,0,0});
                }
        compare_tuple("index-proj", {32,0,0,0}, {index.ring,index.rows_per_wave,0,0});
        compare_tuple("decode-wide", {0,0,0,0}, {route::wide_for(n, 2) ? 1U : 0U,0,0,0});
        for (const auto& shape : std::array<std::array<unsigned, 3>, 8>{{
                 {6144, 4096, 3}, {6144, 256, 1}, {4096, 6144, 1}, {6144, 6144, 2},
                 {6144, 12288, 2}, {12288, 6144, 1}, {6144, 99840, 1}, {1536, 6144, 1}}}) {
            const int chosen = route::scheduled_linear_version(shape[0], shape[1], shape[2], n);
            require(chosen == 4 || chosen == 8 || chosen == 9 || chosen == 10, "uncovered linear version");
            if (chosen != route::scheduled_linear_version(shape[0], shape[1], shape[2], 1))
                require(allow("decode-linears").category == 'a', "missing linear equivalence gate");
        }
    }
    const std::array<Row, 3> mixed{{{0, 1000}, {0, 1001}, {1, 3000}}};
    require(route::step_layout(std::span<const Row>(mixed)).seqs == 0, "mixed-slot fallback");
    gated_difference("decode-layout", "MSA fused attend -> per-row K1-K4 (mixed slots)");
    const std::array<Row, 1> alone{{{0, 2047}}};
    const std::array<Row, 2> companions{{{0, 2047}, {1, 2048}}};
    require(route::step_dense(std::span<const Row>(alone), 2048) &&
            !route::step_dense(std::span<const Row>(companions), 2048), "dense/sparse companion boundary");
    gated_difference("msa-dense", "MSA dense -> sparse entry (row at 2047; companion at 2048)");
    gated_difference("index-proj", "index proj ring32 -> ring16, row-pair scheduling");
    gated_difference("decode-wide", "router_shared/gate_up wide columns at N2..4");
}
void prefill_cuts() {
    // Mode 10 changes physical ownership only; native qualification stays open.
    const auto& inblock = allow("prefill-msa-inblock");
    require(inblock.category == 'a' && std::string_view(inblock.gate).find("UNCOVERED") != std::string_view::npos,
            "in-block native gate must remain explicit until qualified");
    compare_tuple("prefill-msa-inblock", {8, 16, 1, 1}, {10, 1, 0, 0});
    gated_difference("prefill-msa-inblock", "global chunk partials+ticket tail -> register history+in-block ordered combine");
    // The production group builder sees the same row's four expert IDs, while
    // all companion rows either share expert 3 or avoid it.
    for (unsigned n : {1U,2U,4U,8U,9U,15U,16U,17U,31U,32U,33U,63U,64U,65U,127U,128U,129U,255U,256U,257U}) {
        for (bool sharing : {false, true}) {
            std::vector<std::uint32_t> ids(std::size_t{n} * 4);
            for (unsigned r = 0; r < n; ++r) {
                ids[std::size_t{r} * 4] = r == 0 || sharing ? 3U : 5U;
                for (unsigned k = 1; k < 4; ++k) ids[std::size_t{r} * 4 + k] = 20U + k;
            }
            const auto groups = route::expert_rows(ids, n, 128, 4);
            const auto occupancy = static_cast<unsigned>(groups[3].size());
            require(occupancy == (sharing ? n : 1U) && groups[3].front() == 0, "companion group occupancy");
            compare_tuple("prefill-routed-split", {16,0,0,0}, {route::gemm_splits(6144,1536,4,occupancy),0,0,0});
            compare_tuple("prefill-tile", {16,0,0,0}, {route::gemm_tile_rows(6144,1536,4,occupancy),0,0,0});
            compare_tuple("prefill-mt", {1,0,0,0}, {route::mt_of_class(route::mt_class(occupancy)),0,0,0});
            const auto fit = route::mtfit(occupancy);
            require(fit.main * 128 + fit.remainder == occupancy && fit.tail_class < 8, "mtfit coverage");
        }
    }
    bool rejected = false;
    try { (void)route::expert_rows(std::array<std::uint32_t, 1>{128}, 1, 128, 1); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "out-of-range expert must fail closed");
    // Same row stays in the suffix; simulated cache cuts only change the call's
    // occupancy. There is no prefix-cache implementation in the current server.
    for (unsigned boundary : {8U, 16U, 32U, 64U, 128U, 256U, 2048U, 4096U}) {
        for (unsigned count : {boundary - 1, boundary, boundary + 1}) {
            const auto got = route::unit_rows(1, boundary, count);
            require(got == std::min(boundary, count), "unit cut");
            require(route::mt_of_class(route::mt_class(count)) <= 8, "MT bucket");
            for (unsigned bits : {3U, 4U, 5U, 6U, 8U})
                for (const auto& shape : std::array<std::array<unsigned, 2>, 7>{{
                         {6144,1536}, {1536,6144}, {4096,1024}, {5120,2304},
                         {5120,1280}, {1280,32768}, {8192,5120}}}) {
                    const auto splits = route::gemm_splits(shape[0], shape[1], bits, count);
                    const auto tile = route::gemm_tile_rows(shape[0], shape[1], bits, count);
                    require(splits > 0 && splits <= shape[0] / 16 && (tile == 16 || tile == 64), "GEMM tuple");
                }
        }
    }
    for (const char* id : {"prefill-routed-split", "prefill-shared-split"}) {
        const unsigned reference = route::gemm_splits(6144, 1536, 4, 1);
        unsigned first = 0;
        for (unsigned count = 2; count <= 4097; ++count)
            if (route::gemm_splits(6144, 1536, 4, count) != reference) { first = count; break; }
        require(reference == 16 && first == 9 && route::gemm_splits(6144, 1536, 4, first) == 8,
                "split divergence changed; re-audit first operation");
        gated_difference(id, "L3 gate/up WMMA k-tile24: split1 reset vs split0 continuation (group rows1->9, S16->8)");
    }
    require(route::kv_tile_config(2047, -2) == 7 && route::kv_tile_config(2048, -2) == 1 &&
            route::kv_tile_config(2049, -2) == 1, "K/V auto boundary");
    for (unsigned rows : {15U, 16U, 17U}) {
        const unsigned nw = (rows + 15) / 16;
        const auto dense = route::dense_windows(2040, rows, nw, 16, 2048);
        require(dense == 0, "straddling dense window");
    }
    for (std::uint64_t cut : {127ULL,128ULL,129ULL,2047ULL,2048ULL,2049ULL,4095ULL,4096ULL,4097ULL}) {
        // Logical last row is fixed at 8192; each suffix is planned afresh using
        // actual unit/occupancy selectors. Its original plan must be retained.
        const auto remaining = static_cast<std::size_t>(8193ULL - cut);
        const auto rows = route::unit_rows(1, 4096, remaining);
        compare_tuple("prefill-unit-cuts", {4096,0,0,0}, {static_cast<unsigned>(rows),0,0,0});
    }
    require(route::dense_windows(2040, 8, 1, 16, 2048) == 1, "dense tail window");
    for (unsigned budget : {1U, 7U, 60U}) {
        unsigned layer = 0;
        std::vector<unsigned> history;
        while (layer < 60) {
            const unsigned end = route::layer_end(layer, budget, 60);
            while (layer < end) history.push_back(layer++);
        }
        require(history.size() == 60 && history.front() == 0 && history.back() == 59, "resume layer history");
    }
    // Nesting changes retained addresses/ownership, never the original prompt or expert-group shape.
    for (unsigned rows : {1U, 8U, 183U, 512U})
        require(route::unit_rows(1, 4096, rows) == rows, "nested original one-chunk shape");
    compare_tuple("prefill-nest", {0,4096,443,183}, {1,4096,443,183});
    gated_difference("prefill-nest", "activation/cache-table bank and slot binding; original row/group arithmetic retained");
    require(strix::minimax::rtxpf::tile_rows_for(95, 1) == 64 &&
            strix::minimax::rtxpf::tile_rows_for(96, 1) == 128, "FP32 tile policy");
    std::vector<strix::minimax::rtxpf::F32Job> jobs{{nullptr,nullptr,nullptr,6144,1536,4,129,0,0}}, tails;
    std::uint32_t tiles = 0;
    require(strix::minimax::rtxpf::split_tails(jobs, tails, tiles) == 12 && jobs[0].rows == 128 &&
            tails.size() == 1 && tails[0].row_begin == 128 && tiles == 12, "FP32 independent tail chains");
    require(!route::fp32_profile(nullptr) && route::fp32_profile("fp32"), "arithmetic profile selector");
    require(route::append_rows(31) == 8 && route::append_rows(32) == 32 && route::append_rows(33) == 32,
            "deferred append boundary");
    compare_tuple("draft-append", {8,0,0,0}, {static_cast<unsigned>(route::append_rows(32)),0,0,0});
}
void config_receipts() {
    using namespace strix::minimax::defaults;
    const Lookup empty = [](const char*) -> const char* { return nullptr; };
    const Lookup baseline_env = [](const char* key) -> const char* {
        return std::string_view(key) == kBaselineEnv ? "1" : nullptr;
    };
    require(resolve(empty).size() == items().size(), "defaults registry");
    for (const auto& resolved : resolve(empty))
        if (std::string_view(resolved.item->env) == "STRIX_MM_PF_MSA_REWORK")
            require(resolved.value == "10", "in-block mode 10 is the default");
    const auto inblock_env = [](const char* name) -> const char* {
        return std::string_view(name) == "STRIX_MM_PF_MSA_REWORK" ? "10" : nullptr;
    };
    for (const auto& resolved : resolve(inblock_env))
        if (std::string_view(resolved.item->env) == "STRIX_MM_PF_MSA_REWORK")
            require(resolved.value == "10", "explicit in-block mode must survive resolution");
    for (const auto& resolved : resolve(empty)) require(!resolved.value.empty(), "empty effective setting");
    require(effective_config(Cli{}, empty) != effective_config(Cli{}, baseline_env), "baseline receipt");
    std::ifstream receipt(fs::path(STRIX_ROUTE_SOURCE_DIR) / "tests/fixtures/minimax_production_receipt.txt");
    require(receipt.good(), "missing effective-config hardware receipt");
    std::map<std::string, std::string> values;
    // These host scheduling defaults have no hardware receipt yet. Check them independently;
    // the recorded rank/serve receipts must still cover every measured engine setting.
    const std::map<std::string, std::string> pending_policy = {
        {"STRIX_MM_SERVE_SCHED", "interleave"}, {"STRIX_MM_SERVE_SERVICE_MS", "250"},
        {"STRIX_MM_SERVE_SHORT_FIRST", "1"}, {"STRIX_MM_SERVE_NEST", "1"},
        {"STRIX_MM_PF_ASYNC_START", "1"}, {"STRIX_MM_PF_LATE_SRC_FENCE", "1"},
    };
    std::size_t pending_entries = 0;
    for (const auto& resolved : resolve(empty)) {
        const auto pending = pending_policy.find(resolved.item->env);
        if (pending != pending_policy.end()) {
            require(resolved.value == pending->second, "pending serving policy drift: " + pending->first);
            require(std::string_view(resolved.item->receipt).find("pending") != std::string_view::npos,
                    "serving policy needs qualification evidence: " + pending->first);
            ++pending_entries;
        } else values.emplace(resolved.item->env, resolved.value);
    }
    require(pending_entries == pending_policy.size(), "incomplete pending serving policy");
    unsigned receipts = 0, entries = 0;
    for (std::string line; std::getline(receipt, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.starts_with("receipt ")) ++receipts;
        if (!line.starts_with("env ")) continue;
        const auto equal = line.find('=');
        require(equal != std::string::npos, "bad receipt setting");
        const auto key = line.substr(4, equal - 4);
        const auto found = values.find(key);
        require(found != values.end() && found->second == line.substr(equal + 1), "effective-config drift: " + key);
        ++entries;
    }
    require(receipts == 2 && entries == 2 * values.size(), "incomplete rank/serve effective-config receipts");
}
void selector_boundary_proofs() {
    using namespace strix::minimax::defaults;
    const Cli serve = parse_cli("serve", {});
    require(serve.transport == "usb4gpu", "serve direct transport construction default");
    const Lookup explicit_cache = [](const char* key) -> const char* {
        return std::string_view(key) == "STRIX_SESSION_CACHE" ? "on" : nullptr;
    };
    const auto automatic = session_policy(serve, explicit_cache, false);
    const auto explicit_on = session_policy(serve, explicit_cache, true);
    require(!automatic.enabled && !explicit_on.enabled &&
            automatic.route == "serve/usb4gpu/fixed-width-one-session-v1" &&
            automatic.reason.find("route not yet GPU-qualified") == 0 &&
            explicit_on.reason == automatic.reason,
            "direct serve session admission must refuse before storage/native work, including explicit opt-in");

    for (const auto& item : items())
        require(std::string_view(item.env) != "STRIX_MMATTN_INDEX_SPLIT",
                "retired index split selector must not re-enter resolver defaults");
    for (const auto* relative : {"src/minimax/attn/msa_fused.hip", "src/minimax/attn/msa_fused.hpp"}) {
        std::ifstream input(fs::path(STRIX_ROUTE_SOURCE_DIR) / relative, std::ios::binary);
        require(input.good(), std::string("cannot inspect retired selector source: ") + relative);
        const std::string source((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        require(source.find("STRIX_MMATTN_INDEX_SPLIT") == std::string::npos &&
                source.find("index_split(") == std::string::npos &&
                source.find("index_kernel<DecodeView, true>") == std::string::npos,
                "retired index split selector or launch remains in current source");
    }
}
}  // namespace
int main(int argc, char** argv) {
    try {
        const std::string_view mode = argc >= 2 ? std::string_view(argv[1]) : std::string_view();
        require(argc == 1 || (argc == 2 && (mode == "--require-identity" || mode == "--audit-sources")) ||
                (argc == 3 && mode == "--audit-sources"),
                "usage: [--require-identity | --audit-sources [source-root]]");
        // Source fingerprints are an audit aid, not a CI gate: any edit to a MiniMax source would fail them.
        if (argc >= 2) source_coverage(argc == 3 ? fs::path(argv[2]) : fs::path(STRIX_ROUTE_SOURCE_DIR));
        selector_boundary_proofs();
        std::set<std::string_view> ids;
        std::array<unsigned, 3> counts{};
        bool open_gate = false;
        for (const auto& selector : selectors) {
            require(ids.insert(selector.id).second, "duplicate selector");
            require(selector.category >= 'a' && selector.category <= 'c', "unclassified selector");
            require(std::string_view(selector.plan_key).size() != 0 && std::string_view(selector.gate).size() != 0,
                    "UNCOVERED selector has no explicit gate/key");
            ++counts[static_cast<std::size_t>(selector.category - 'a')];
            open_gate |= selector.category == 'c' || std::string_view(selector.gate).find("UNCOVERED") != std::string_view::npos;
            require(selector.category != 'c' || std::string_view(selector.first_op).size() != 0, "failure lacks first operation");
        }
        using strix::minimax::rtxpf::resolve_core_version;
        require(strix::minimax::rtxpf::GroupedCall{}.core_version == 0, "FP32 automatic selection");
        for (const char* env : {static_cast<const char*>(nullptr), "", "31", "invalid", "3x", "4"}) {
            require(resolve_core_version(0, env) == 31, "FP32 default v3.1");
            require(resolve_core_version(3, env) == 3, "FP32 explicit v3");
            require(resolve_core_version(31, env) == 31, "FP32 explicit v3.1");
        }
        require(resolve_core_version(0, "3") == 3 && resolve_core_version(31, "3") == 31,
                "FP32 env rollback and explicit override");
        decode_and_verify();
        prefill_cuts();
        config_receipts();
        for (const auto& selector : selectors)
            if (selector.category == 'c' && std::string_view(selector.id) != "prefill-routed-split" &&
                std::string_view(selector.id) != "prefill-shared-split")
                gated_difference(selector.id, selector.first_op);
        std::cout << "CENSUS inventoried: a=" << counts[0] << " b=" << counts[1] << " c=" << counts[2]
                  << "; open gates are NOT identity qualification\n";
        return mode == "--require-identity" && open_gate ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL CLOSED: " << e.what() << '\n';
        return 1;
    }
}
