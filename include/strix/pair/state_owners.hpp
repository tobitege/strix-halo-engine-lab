#pragma once
// strix::pair::state: ONE copy per owner for the prompt-state hand-off (prefill host -> serving ranks), prefix first.
//
// A family adapter only DESCRIBES its state (strix/pair/state_decls.hpp): pieces with an owner, bytes per row or fixed
// bytes, and how many layers carry them. It never decides what crosses which link. This header (host only, no sockets,
// no device code) plans for every family the same way:
//   owner     per_rank: a distinct slice per rank (MiniMax K/V heads, GLM KDA heads) goes to its rank only.
//             shared:   identical bytes on every rank (MiniMax index keys, GLM DSA, DS41 state) go ONCE to the relay rank,
//                       which forwards them on the pair interconnect (strix/pair/state_relay.hpp), instead of once per
//                       rank over the prefill host's link (one NIC, shared by every destination).
//             stage:    owned by the rank whose layer stage holds the layer (Qwen pair stages); nobody else gets it.
//             replica:  a few bytes every rank needs (position): sent to each rank, never relayed.
//   prefix    before sending, every destination reports what it holds for the request (Holding: identity digest and
//             rows). The agreed boundary is the longest prefix ALL destinations hold, rounded down to the family's
//             block. Row pieces then ship only rows [boundary, P). Fixed pieces (recurrent final state, rings, held
//             activations) always ship in full: the final fixed state is not a prefix of anything.
//   recompute with a suffix of S rows the alternative is to ship only the S token ids and let the ranks prefill the
//             suffix from their cached boundary state. It needs the fixed state AT the boundary on the ranks and a
//             MEASURED recompute rate; an unmeasured rate is never chosen (same rule as deploy/route).
// No choice here changes a bit of the imported state, provided the reused prefix rows are the bits a full send would
// carry: a family declares that with `prefix_rows_invariant` (the producer's row r does not depend on the prompt length),
// a measured property; without it prefix reuse stays off (reason string says so).
//
// Switches (shared resolver, recorded for --print-effective-config; CLI = serve):
//   STRIX_HANDOFF_SHARED_ONCE  shared pieces once + relay       (default per family: see each adapter's decl)
//   STRIX_HANDOFF_PREFIX_FIRST ask the ranks first, ship suffix  (default off: needs the family's invariance receipt)
#include <algorithm>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "strix/lever.hpp"

namespace strix::pair::state {

enum class Owner : std::uint8_t { per_rank, shared, stage, replica };
enum class Extent : std::uint8_t { rows, fixed };

// One component of the state, per layer. Rows stored for P prompt rows: rows (P - rows_less) / row_div (floor), or a
// fixed size. `bytes` is per stored row (rows) or per layer (fixed), per destination slice for per_rank pieces.
struct Piece {
    std::string_view name{};
    std::uint32_t kind = 0;            // family wire component id
    Owner owner = Owner::shared;
    Extent extent = Extent::rows;
    std::uint64_t bytes = 0;
    std::uint32_t layers = 1;          // layers carrying this piece
    std::uint32_t layer0 = 0, layer_stride = 1;  // layer index of entry i = layer0 + i * layer_stride (stage owners)
    std::uint32_t row_div = 1;         // stored rows = floor((P - rows_less) / row_div)
    std::uint32_t rows_less = 0;       // e.g. MTP rows stop one row short
    std::string_view source{};         // consumer file:line that reads it
};

struct Family {
    std::string_view name{};           // "minimax-m3.1", "glm-5.3", ...
    std::string_view rows_of{};        // "P = N - 1" or "N": what the rows argument means
    std::span<const Piece> pieces{};
    std::uint32_t world = 2;           // destination ranks
    std::uint32_t block_rows = 1;      // prefix reuse granularity
    bool prefix_rows_invariant = false;  // measured: producer row r bits do not depend on the prompt length
    bool shared_once_default = false;  // the family's default for STRIX_HANDOFF_SHARED_ONCE (a receipt, see decl)
    std::string_view shared_once_source{};
};

[[nodiscard]] inline std::uint64_t stored_rows(const Piece& p, std::uint64_t rows) noexcept {
    if (p.extent == Extent::fixed) return 1;
    const std::uint64_t r = rows > p.rows_less ? rows - p.rows_less : 0;
    return r / std::max<std::uint32_t>(p.row_div, 1);
}
// Bytes of one layer of `p` for row range [row0, rows) of P = rows (row pieces), or its fixed size.
[[nodiscard]] inline std::uint64_t piece_bytes(const Piece& p, std::uint64_t row0, std::uint64_t rows) noexcept {
    if (p.extent == Extent::fixed) return p.bytes;
    const std::uint64_t hi = stored_rows(p, rows), lo = std::min(stored_rows(p, row0), hi);
    return (hi - lo) * p.bytes;
}

// Which rank owns layer `layer` when stage s starts at stage_first[s] (ascending, stage_first[0] = 0).
[[nodiscard]] inline std::uint32_t stage_of(std::span<const std::uint32_t> stage_first, std::uint32_t layer) noexcept {
    std::uint32_t s = 0;
    for (std::uint32_t i = 0; i < stage_first.size(); ++i)
        if (stage_first[i] <= layer) s = i;
    return s;
}

// What a destination rank reports before the send: the rows it holds for this request's identity (0 = nothing), the
// digest of those rows' token chain (strix/prefix_state_cache.hpp block digests, or the family's own), and whether it
// also holds the fixed state AT that boundary (needed only to recompute the suffix there).
struct Holding {
    std::uint64_t rows = 0;
    std::uint64_t digest = 0;
    bool fixed_at_rows = false;
};

struct Links {
    double host_bytes_per_s = 125e6;    // prefill host egress, shared by every destination (1 GbE payload ~117.6e6)
    double relay_bytes_per_s = 4.4e9;   // pair interconnect (USB4 host path, 35 Gb/s at >= 16 MB pieces)
    double recompute_rows_per_s = -1;   // ranks' suffix prefill rate; < 0 = unmeasured, never chosen
};

struct Options {
    bool shared_once = false;
    bool prefix_first = false;
    std::vector<std::uint32_t> stage_first{};  // stage owners: first layer of each rank's stage (empty: rank 0 owns all)
    std::uint32_t relay_rank = 0;
};

enum class Mode : std::uint8_t { full, suffix, ids };
[[nodiscard]] inline std::string_view mode_name(Mode m) noexcept {
    return m == Mode::full ? "full" : m == Mode::suffix ? "suffix" : "ids";
}

// One piece's traffic: `to` receives it from the prefill host, then forwards it to every rank in `relay_to` (bit mask).
struct Leg {
    std::uint32_t piece = 0;     // index into Family::pieces
    std::uint32_t to = 0;
    std::uint32_t relay_to = 0;  // bit r set: `to` forwards the piece to rank r on the pair link
    std::uint64_t row0 = 0, rows = 0;  // prompt rows covered (row pieces), [row0, rows)
    std::uint64_t bytes = 0;     // over all layers of the piece
};

struct Plan {
    Mode mode = Mode::full;
    std::uint64_t rows = 0, boundary = 0;
    std::vector<Leg> legs;
    std::vector<std::uint64_t> host_bytes_to;  // per destination rank, over the prefill host's link
    std::uint64_t host_bytes = 0;              // sum (one NIC)
    std::uint64_t relay_bytes = 0;             // pair interconnect
    std::uint64_t ids_bytes = 0;               // suffix token ids (Mode::ids)
    // Optimistic wire bound assuming perfect overlap, not completion latency (GLM stages before relaying).
    double host_s = 0, relay_s = 0, recompute_s = 0, predicted_s = 0;
    std::string reason;
};

// The prefix every destination can reuse. `chain[k]` is the sender's digest of its first (k + 1) * block rows (the same
// chain the ranks' caches key on). A rank's holding counts only when its rows are a whole number of blocks and its
// digest equals the sender's at that row count; the boundary is the minimum over ALL ranks (one rank without the prefix
// means no reuse), clipped to the rows being shipped. The boundary's fixed state is usable only when every rank holds
// exactly that boundary and reports the fixed state there.
struct Agreement {
    std::uint64_t boundary = 0;
    bool fixed_at = false;
    std::string reason{};
};
[[nodiscard]] inline Agreement agree(const Family& f, std::uint64_t rows, std::span<const Holding> h,
                                     std::span<const std::uint64_t> chain) {
    Agreement a;
    if (!f.prefix_rows_invariant) {
        a.reason = "prefix: off (family has no prompt-length row invariance receipt)";
        return a;
    }
    if (h.size() != f.world) {
        a.reason = "prefix: off (" + std::to_string(h.size()) + " of " + std::to_string(f.world) + " ranks answered)";
        return a;
    }
    const std::uint64_t blk = std::max<std::uint32_t>(f.block_rows, 1);
    std::uint64_t b = rows / blk * blk;
    bool fixed = true;
    for (std::uint32_t r = 0; r < h.size(); ++r) {
        const Holding& x = h[r];
        const std::uint64_t k = x.rows / blk;
        const bool valid = x.rows % blk == 0 && k > 0 && k <= chain.size() && chain[k - 1] == x.digest;
        const std::uint64_t held = valid ? x.rows : 0;
        if (x.rows != 0 && !valid) a.reason = "prefix: rank " + std::to_string(r) + " digest or block mismatch, ignored";
        b = std::min(b, held);
    }
    for (const Holding& x : h) fixed = fixed && x.fixed_at_rows && x.rows == b;
    a.boundary = b;
    a.fixed_at = b > 0 && fixed;
    if (a.reason.empty()) a.reason = b > 0 ? "prefix: reuse " + std::to_string(b) + " rows" : "prefix: no common prefix";
    return a;
}

namespace detail {
inline void add(Plan& plan, const Family& f, std::uint32_t i, std::uint32_t to, std::uint32_t relay, std::uint64_t row0,
                std::uint64_t layers_bytes) {
    if (layers_bytes == 0) return;
    plan.legs.push_back(Leg{i, to, relay, row0, plan.rows, layers_bytes});
    plan.host_bytes_to[to] += layers_bytes;
    plan.host_bytes += layers_bytes;
    for (std::uint32_t r = 0; r < f.world; ++r)
        if (relay & (1U << r)) plan.relay_bytes += layers_bytes;
}
}  // namespace detail

// Plans the hand-off of `rows` prompt rows (Family::rows_of) to `f.world` ranks.
[[nodiscard]] inline Plan plan(const Family& f, std::uint64_t rows, const Options& o, const Links& links,
                               const Agreement& prefix = {}) {
    Plan p;
    p.rows = rows;
    p.host_bytes_to.assign(f.world, 0);
    const std::uint32_t world = std::max<std::uint32_t>(f.world, 1);
    const std::uint32_t all = world >= 32 ? 0xffffffffU : (1U << world) - 1U;
    const std::uint64_t b = o.prefix_first ? std::min(prefix.boundary, rows) : 0;
    const bool fixed_at = o.prefix_first && prefix.fixed_at;
    p.boundary = b;
    p.mode = b > 0 ? Mode::suffix : Mode::full;
    for (std::uint32_t i = 0; i < f.pieces.size(); ++i) {
        const Piece& x = f.pieces[i];
        const std::uint64_t one = piece_bytes(x, x.extent == Extent::rows ? b : 0, rows);
        switch (x.owner) {
        case Owner::per_rank:
        case Owner::replica:
            for (std::uint32_t r = 0; r < world; ++r) detail::add(p, f, i, r, 0, b, one * x.layers);
            break;
        case Owner::shared:
            if (o.shared_once && world > 1)
                detail::add(p, f, i, o.relay_rank, all & ~(1U << o.relay_rank), b, one * x.layers);
            else
                for (std::uint32_t r = 0; r < world; ++r) detail::add(p, f, i, r, 0, b, one * x.layers);
            break;
        case Owner::stage:
            for (std::uint32_t l = 0; l < x.layers; ++l) {
                const std::uint32_t layer = x.layer0 + l * x.layer_stride;
                const std::uint32_t r = o.stage_first.empty() ? 0 : std::min(stage_of(o.stage_first, layer), world - 1);
                detail::add(p, f, i, r, 0, b, one);
            }
            break;
        }
    }
    p.host_s = static_cast<double>(p.host_bytes) / links.host_bytes_per_s;
    p.relay_s = static_cast<double>(p.relay_bytes) / links.relay_bytes_per_s;
    p.predicted_s = std::max(p.host_s, p.relay_s);
    // Recompute the suffix on the ranks instead: only the ids cross (4 B each, to every rank), the ranks need the
    // boundary's fixed state and a measured rate, and it must beat the suffix send.
    if (b > 0 && fixed_at && links.recompute_rows_per_s > 0) {
        const std::uint64_t s = rows - b;
        const double ids_s = static_cast<double>(4 * s * world) / links.host_bytes_per_s;
        const double rec = static_cast<double>(s) / links.recompute_rows_per_s + ids_s;
        if (rec < p.predicted_s) {
            Plan q;
            q.mode = Mode::ids;
            q.rows = rows;
            q.boundary = b;
            q.host_bytes_to.assign(world, 4 * s);
            q.ids_bytes = q.host_bytes = 4 * s * world;
            q.host_s = ids_s;
            q.recompute_s = static_cast<double>(s) / links.recompute_rows_per_s;
            q.predicted_s = rec;
            q.reason = "ids: suffix recompute " + std::to_string(rec) + " s < send " + std::to_string(p.predicted_s) + " s";
            return q;
        }
    }
    p.reason = std::string(mode_name(p.mode)) + (o.shared_once ? " shared-once" : " per-rank copies") +
               (o.prefix_first ? "; " + prefix.reason : std::string());
    return p;
}

// Bytes over the prefill host's link for `rows` with `prefix` reused rows, the hook a route planner uses as its
// state_bytes(rows, prefix) (deploy/route: RouteProvider::state_bytes).
[[nodiscard]] inline std::uint64_t host_bytes(const Family& f, std::uint64_t rows, std::uint64_t prefix, bool shared_once) {
    Options o;
    o.shared_once = shared_once;
    o.prefix_first = prefix > 0;
    return plan(f, rows, o, Links{}, Agreement{prefix, false, {}}).host_bytes;
}

// ---- shared resolver -------------------------------------------------------------------------------------------
inline constexpr lever::Decl shared_once_lever{"handoff-shared-once", "STRIX_HANDOFF_SHARED_ONCE", lever::Exactness::bitwise,
                                               false, {}, {}, "wave4 MINST1 (bytes identical by construction: one buffer)"};
inline constexpr lever::Decl prefix_first_lever{"handoff-prefix-first", "STRIX_HANDOFF_PREFIX_FIRST", lever::Exactness::bitwise,
                                                false, {}, {}, "needs the family's prompt-length row invariance receipt"};

struct Resolved {
    bool shared_once = false, prefix_first = false;
};

// Last CLI option wins, including replay's alternating mode. Environment resolution still uses cli.
struct SharedOnceMode {
    int cli = -1;
    bool ab = false;
    void set(bool on, bool alternate = false) { cli = on ? 1 : 0; ab = on && alternate; }
};
// One resolver for the CLI and the server of a family: `cli_*` is the command-line value (-1 unset, 0 off, 1 on); it
// wins over the environment, which wins over the family default. Records both switches for --print-effective-config.
[[nodiscard]] inline Resolved resolve(const Family& f, int cli_shared_once = -1, int cli_prefix_first = -1,
                                      const char* env_shared_once = nullptr, const char* env_prefix_first = nullptr,
                                      bool read_env = true) {
    lever::Decl so = shared_once_lever;
    so.default_on = f.shared_once_default;
    const auto pick = [&](const lever::Decl& d, int cli, const char* fixed) {
        if (cli == 0 || cli == 1) {
            envcfg::record(std::string(d.env).c_str(), cli ? "1" : "0", cli ? "1" : "0");
            return cli == 1;
        }
        if (!read_env) return lever::select(d, lever::Inputs{}, fixed).on;
        return lever::select(d, lever::Inputs{}).on;
    };
    Resolved r;
    r.shared_once = pick(so, cli_shared_once, env_shared_once);
    r.prefix_first = pick(prefix_first_lever, cli_prefix_first, env_prefix_first);
    return r;
}

}  // namespace strix::pair::state
