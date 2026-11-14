#include "strix/device/hip_gate.hpp"
#include "strix/deepseek41_attention_state.hpp"
#include "strix/deepseek41_batch_forward.hpp"
#include "strix/deepseek41_prefill.hpp"
#include "strix/deepseek41_prefix_cache.hpp"
#include "strix/ds41_expert_tp.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/dist/open_transport.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

// deepseek41 prefix cache gate: a prompt prefilled after its beginning was cached (DevicePrefill resumed from the
// cached block states: import_state + extend) against the same prompt prefilled fresh, bitwise: the greedy token after
// the prompt, the FNV of the last row's logits and the digest of the full attention state (every ring's live window,
// compressed KV, index keys, carries, tokens, position, next token; the last row's top-k is left out, nothing reads it).
// With --decode G both states are imported into a BatchForward (two slots, the prefill reading its weights in place)
// and decode G greedy tokens side by side, which must agree.
//
//   strix-deepseek41-prefix MODEL.gguf --prompt FILE (comma-separated ids) [--rows N] [--layers L]
//       [--world 1 [--stream-experts] | --transport emulate | --rank R --transport usb4-auto]
//       [--block 256] [--max-rows 4096] [--cache-mb 0] [--warm W1,W2,...] [--decode G] [--repeat-hit K] [--before-tail]
//       [--inrun [--inrun-check]] [--byte-check]
//       [--cache-dir DIR [--disk-mb 0] [--phase all|write|read|fresh] [--no-fresh] [--fork F] [--rewind R]
//        [--corrupt block|snapshot] [--session G]]
//
// Scenarios, each on an emptied cache: "miss" (the prompt through the cache: pieces at the last full block, then the
// tail), and for every W of --warm (W < 0: N + W) a first prompt of W tokens through the cache, then the full prompt
// (a hit at the last boundary <= W cached by the first prompt; multi-turn chat when W = N - suffix). --repeat-hit K
// repeats the hit K times (timing; each hit resumes from the same snapshot). Times: "ttft" = lookup + state assembly +
// import + every prefill piece + the final export, i.e. what a server waits before its first token (without the
// tokenizer and the slot import, printed apart).
//
// --cache-dir DIR: the disk tier of the cache in DIR/rank<R> (write-through; DIR/ref-rank<R>.txt keeps the reference
// outcome). Phases: "write" (= "all": fresh, then through the cache: miss, RAM hit, disk hit after the RAM tier is
// dropped; fork --fork F: the prompt's first F tokens + its reversed tail after the prompt was cached, then the prompt
// again; rewind --rewind R: the first R tokens from the deepest snapshot <= R; every warm W read from disk; session
// --session G: G tokens decoded from the prompt's state in a BatchForward slot, the block boundaries it reaches
// inserted from the slot (session snapshots), then the conversation's last boundary and 40 more tokens through the
// cache against fresh prefills), "read" (a new process on the files: the index rebuilt, optionally after
// --corrupt block|snapshot damaged one file; disk hit, a warm hit from disk and a RAM hit against the reference file,
// decode against its tokens), "fresh" (only the fresh prefill, against the reference file). --no-fresh: the miss is the
// reference (long prompts: one prefill per turn).
// --inrun: the cache's snapshots inside the prefill runs (plan_prefill_runs, DevicePrefill::set_snapshot_points /
// export_state_at), as the serve's --prefix-inrun-snapshots; --inrun-check also compares every in-run snapshot's state
// with a fresh prefill of exactly its tokens (a run ending there), bitwise.
namespace {

namespace ds = strix::deepseek41;
using Clock = std::chrono::steady_clock;

struct Args {
    std::string model, prompt_path, transport, warm = "-64,-256";
    std::string cache_dir, phase = "all", corrupt;
    std::uint64_t disk_mb = 0;
    std::uint32_t fork = 0, rewind = 0, session = 0;
    bool before_tail = false, no_fresh = false, inrun = false, inrun_check = false, byte_check = false;
    std::uint32_t rank = 0, world = 1, layers = 40, rows = 0, block = 256, max_rows = 4096, decode = 0, repeat_hit = 1;
    std::uint64_t cache_mb = 0;
    unsigned connect_timeout = 60;
    bool stream_experts = false, shared = true;
};

double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

std::uint64_t fnv(const std::vector<float>& v) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const float f : v) {
        std::uint32_t b;
        std::memcpy(&b, &f, 4);
        for (int k = 0; k < 4; ++k, b >>= 8U) h = (h ^ (b & 0xffU)) * 1099511628211ULL;
    }
    return h;
}

std::vector<std::uint32_t> read_prompt(const std::string& path, std::uint32_t rows) {
    std::ifstream in(path);
    if (!in) throw std::invalid_argument("cannot read " + path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<std::uint32_t> ids;
    std::stringstream ss(text);
    for (std::string t; std::getline(ss, t, ',');)
        if (!t.empty()) ids.push_back(static_cast<std::uint32_t>(std::stoul(t)));
    if (rows != 0U) {
        if (rows > ids.size()) {  // repeat the prompt up to `rows` tokens (long-context gates from a shorter file)
            const auto base = ids;
            while (ids.size() < rows) ids.insert(ids.end(), base.begin(), base.end());
        }
        ids.resize(rows);
    }
    return ids;
}

std::vector<long> parse_list(const std::string& s) {
    std::vector<long> out;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');)
        if (!t.empty()) out.push_back(std::stol(t));
    return out;
}

struct Outcome {
    std::uint32_t next_token = 0, resume = 0;
    std::uint64_t logits = 0, digest = 0;
    std::vector<std::pair<std::string, std::uint64_t>> hashes;
    ds::AttentionState state;
    std::vector<std::uint8_t> canonical;
    std::vector<float> full_logits;
    bool logits_known = true;
    double seconds = 0.0, lookup_seconds = 0.0, import_seconds = 0.0, export_seconds = 0.0;
    std::vector<std::uint32_t> pieces;
    std::uint64_t disk_bytes = 0;
    double disk_seconds = 0.0;
    bool read_failed = false;
};

class Gate {
public:
    Gate(const Args& a, std::uint32_t rank, strix::dist::Communicator* comm, bool print, std::size_t prompt_rows) : a_(a), print_(print) {
        const auto positions = static_cast<std::uint32_t>(prompt_rows + a.decode + 16U);
        if (a.decode != 0U) {
            ds::BatchForwardOptions bo;
            bo.layers = a.layers;
            bo.max_positions = positions;
            bo.slots = 2;
            bo.rank = rank;
            bo.world = comm ? 2U : 1U;
            bo.communicator = comm;
            bo.tp_attention = comm != nullptr;
            bo.engram_direct_io = true;
            batch_ = std::make_unique<ds::BatchForward>(a.model, bo);
        }
        ds::PrefillOptions o;
        o.layers = a.layers;
        o.max_positions = positions;
        o.max_rows = std::min<std::uint32_t>(a.max_rows, positions);
        o.rank = rank;
        o.world = comm ? 2U : 1U;
        o.communicator = comm;
        o.stream_experts = a.stream_experts;
        min_rows_ = comm ? o.window_rows + 1U : 1U;
        if (batch_ && a.shared) o.shared_weights = batch_->file_form_weights();
        o.expert_tp = batch_ ? batch_->expert_tp() : false;  // the batch forward's expert identity, read in place (as the serve)
        const auto t0 = Clock::now();
        prefill_ = std::make_unique<ds::DevicePrefill>(a.model, o);
        if (print_)
            std::printf("rank %u: prefill weights %.2f GiB loaded + %.2f GiB shared in %.1f s%s\n", rank, static_cast<double>(prefill_->weight_bytes()) / 1073741824.0,
                        static_cast<double>(prefill_->shared_weight_bytes()) / 1073741824.0, since(t0),
                        batch_ ? " (BatchForward holds the shared ones)" : "");
        ds::PrefixCacheOptions co;
        co.block = a.block;
        co.budget_bytes = a.cache_mb << 20U;
        if (!a.cache_dir.empty()) {
            co.dir = rank_dir(a, rank);
            co.disk_budget_bytes = a.disk_mb << 20U;
            co.model_id = ds::prefix_model_id(a.model, a.layers, ds::prefix_namespace_from_env(batch_ && batch_->expert_tp()));
        }
        cache_ = std::make_unique<ds::PrefixCache>(co);
        if (print_ && cache_->has_disk()) {
            const auto st = cache_->stats();
            std::printf("rank %u: disk tier %s: %llu files scanned in %.3f s, %llu ignored; %llu blocks + %llu snapshots = %.1f MB\n", rank,
                        co.dir.string().c_str(), static_cast<unsigned long long>(st.scanned_files), st.scan_seconds,
                        static_cast<unsigned long long>(st.ignored_files), static_cast<unsigned long long>(st.disk_blocks),
                        static_cast<unsigned long long>(st.disk_snapshots), static_cast<double>(st.disk_bytes) / 1e6);
            std::fflush(stdout);
        }
    }

    // The prompt prefilled fresh: one run of max_rows pieces (extend), no cache.
    Outcome fresh(const std::vector<std::uint32_t>& prompt) {
        Outcome o;
        const auto t0 = Clock::now();
        ds::PrefillResult r;
        std::uint32_t at = 0;
        for (const auto end : ds::plan_prefill_pieces(0U, 0U, static_cast<std::uint32_t>(prompt.size()), a_.max_rows, a_.max_rows, min_rows_)) {
            const std::span<const std::uint32_t> piece(prompt.data() + at, end - at);
            r = at == 0U ? prefill_->run(piece) : prefill_->extend(piece);
            at = end;
            o.pieces.push_back(at);
        }
        o.next_token = r.next_token;
        o.logits = fnv(r.logits);
        if (a_.byte_check) o.full_logits = r.logits;
        const auto t1 = Clock::now();
        o.state = prefill_->export_state();
        o.state.tokens = prompt;
        o.state.next_token = r.next_token;
        o.export_seconds = since(t1);
        o.seconds = since(t0);
        finish(o);
        return o;
    }

    // The prompt through the cache: resume from the longest snapshot, run the planned pieces, insert every boundary.
    Outcome cached(const std::vector<std::uint32_t>& prompt) {
        Outcome o;
        const auto t0 = Clock::now();
        const auto N = static_cast<std::uint32_t>(prompt.size());
        auto hit = cache_->lookup(prompt, min_rows_);
        ds::AttentionState loaded;
        if (hit.resume > 0U) {
            try {
                loaded = cache_->load(prompt, hit.resume, hit.key);
                o.disk_bytes = cache_->last_load().disk_bytes;
                o.disk_seconds = cache_->last_load().disk_seconds;
                cache_->use(loaded, hit.key);
            } catch (const ds::PrefixCacheReadError& e) {  // the path goes, the prompt runs as a miss
                std::printf("  disk read failed (%s): forget, miss\n", e.what());
                o.read_failed = true;
                cache_->forget(prompt, hit.resume);
                hit = ds::PrefixLookup{};
            }
        }
        o.resume = hit.resume;
        if (hit.resume == N) {  // the whole prompt is a cached boundary: no prefill
            o.state = std::move(loaded);
            o.next_token = hit.next_token;
            o.logits = hit.logits_fnv;
            o.logits_known = hit.logits_fnv != 0U;
            o.lookup_seconds = since(t0);
            o.seconds = since(t0);
            finish(o);
            return o;
        }
        if (hit.resume > 0U) {
            const auto& s = loaded;
            o.lookup_seconds = since(t0);
            const auto t1 = Clock::now();
            prefill_->import_state(s);
            o.import_seconds = since(t1);
        } else {
            o.lookup_seconds = since(t0);
        }
        std::vector<std::uint32_t> points;
        if (a_.inrun) {
            auto plan = ds::plan_prefill_runs(hit.resume, hit.shared, N, a_.block, a_.max_rows, min_rows_, a_.before_tail);
            o.pieces = std::move(plan.ends);
            points = std::move(plan.points);
        } else {
            o.pieces = ds::plan_prefill_pieces(hit.resume, hit.shared, N, a_.block, a_.max_rows, min_rows_, a_.before_tail);
        }
        ds::PrefillResult r;
        std::uint32_t at = hit.resume;
        for (const auto end : o.pieces) {
            const std::span<const std::uint32_t> piece(prompt.data() + at, end - at);
            std::vector<std::uint32_t> inside;
            for (const auto p : points)
                if (p > at && p < end) inside.push_back(p);
            if (a_.inrun) prefill_->set_snapshot_points(inside);
            r = at == 0U ? prefill_->run(piece) : prefill_->extend(piece);
            at = end;
            for (const auto p : inside) {
                const auto t1 = Clock::now();
                auto s = prefill_->export_state_at(p);
                s.tokens.assign(prompt.begin(), prompt.begin() + p);
                s.next_token = ds::kPrefixUnknownNextToken;
                (void)cache_->insert(s, 0U);
                o.export_seconds += since(t1);
                if (a_.inrun_check && std::none_of(inrun_.begin(), inrun_.end(), [&](const auto& x) { return x.tokens == s.tokens; }))
                    inrun_.push_back(std::move(s));
            }
            if (end % a_.block == 0U) {
                const auto t1 = Clock::now();
                auto s = prefill_->export_state();
                s.tokens.assign(prompt.begin(), prompt.begin() + end);
                s.next_token = r.next_token;
                (void)cache_->insert(s, fnv(r.logits));
                o.export_seconds += since(t1);
                if (end == N) o.state = std::move(s);
            }
        }
        o.next_token = r.next_token;
        o.logits = fnv(r.logits);
        if (a_.byte_check) o.full_logits = r.logits;
        if (o.state.layers.empty()) {
            const auto t1 = Clock::now();
            o.state = prefill_->export_state();
            o.state.tokens = prompt;
            o.state.next_token = r.next_token;
            o.export_seconds += since(t1);
        }
        o.seconds = since(t0);
        finish(o);
        return o;
    }

    // Both states decoded side by side in a BatchForward (slots 0 and 1); returns the two token lists.
    std::pair<std::vector<std::uint32_t>, std::vector<std::uint32_t>> decode(const Outcome& x, const Outcome& y, double& import_seconds) {
        auto& f = *batch_;
        const auto t0 = Clock::now();
        for (std::uint32_t s = 0; s < 2U; ++s) {
            if (!f.reserve(s, f.capacity(s))) throw std::runtime_error("reserve failed");
            f.import_state(s, s == 0U ? x.state : y.state);
        }
        import_seconds = since(t0) / 2.0;
        std::vector<std::uint32_t> tx{x.next_token}, ty{y.next_token};
        for (std::uint32_t g = 1; g < a_.decode; ++g) {
            const std::vector<ds::BatchRow> rows{{0U, tx.back()}, {1U, ty.back()}};
            const auto& t = f.step(rows);
            tx.push_back(t[0]);
            ty.push_back(t[1]);
        }
        return {tx, ty};
    }

    // Session snapshots: `x`'s state into slot 0, `g` tokens decoded greedily; every block boundary the slot reaches is
    // inserted from the slot (only the KV entries the cache lacks). Returns the conversation (prompt + decoded tokens)
    // and sets `last` to the last boundary inserted.
    std::vector<std::uint32_t> session(const Outcome& x, std::uint32_t g, std::uint32_t& last) {
        auto& f = *batch_;
        if (!f.reserve(0U, f.capacity(0U))) throw std::runtime_error("reserve failed");
        f.import_state(0U, x.state);
        std::vector<std::uint32_t> hist = x.state.tokens;
        std::uint32_t next = x.next_token;
        last = 0;
        for (std::uint32_t i = 0; i < g; ++i) {
            const std::vector<ds::BatchRow> rows{{0U, next}};
            hist.push_back(next);
            next = f.step(rows)[0];
            const std::uint32_t p = f.position(0U);
            if (p % a_.block != 0U) continue;
            const auto t0 = Clock::now();
            const std::uint32_t from = cache_->covered(hist, p);
            auto s = f.export_state(0U, from);
            s.tokens.assign(hist.begin(), hist.begin() + p);
            s.next_token = next;
            const bool ok = cache_->insert(s, 0U, from);
            if (ok) last = p;
            if (print_) std::printf("  session: boundary %u from the slot (KV from %u): %s in %.3f s\n", p, from, ok ? "inserted" : "REFUSED", since(t0));
        }
        return hist;
    }

    ds::PrefixCache& cache() { return *cache_; }
    // --inrun-check: the in-run snapshots taken so far (one per distinct token prefix)
    [[nodiscard]] const std::vector<ds::AttentionState>& inrun_states() const { return inrun_; }
    [[nodiscard]] bool has_batch() const { return batch_ != nullptr; }

private:
    void finish(Outcome& o) const {
        if (a_.byte_check) o.canonical = ds::canonical_attention_state_bytes(o.state);
        o.digest = o.state.digest(true);
        o.hashes = o.state.hashes();
    }
public:
    static std::string rank_dir(const Args& a, std::uint32_t rank) { return a.cache_dir + "/rank" + std::to_string(rank); }

private:
    const Args& a_;
    bool print_;
    std::uint32_t min_rows_ = 1;  // world 2 (row split): every run gives each rank a row
    std::unique_ptr<ds::BatchForward> batch_;
    std::unique_ptr<ds::DevicePrefill> prefill_;
    std::unique_ptr<ds::PrefixCache> cache_;
    std::vector<ds::AttentionState> inrun_;
};

std::string list(const std::vector<std::uint32_t>& v) {
    std::string s;
    for (const auto x : v) s += (s.empty() ? "" : ",") + std::to_string(x);
    return s.empty() ? "-" : s;
}

// Compares an outcome with the fresh one; prints one GATE line; returns the number of differences.
int compare(std::uint32_t rank, const std::string& name, const Outcome& ref, const Outcome& o) {
    int bad = 0;
    if (o.next_token != ref.next_token) ++bad;
    if (o.logits_known && ref.logits_known && o.logits != ref.logits) ++bad;
    if (o.digest != ref.digest) ++bad;
    if (!ref.canonical.empty() && !o.canonical.empty() && o.canonical != ref.canonical) ++bad;
    const bool logits_bytes = !ref.full_logits.empty() && !o.full_logits.empty();
    if (logits_bytes && (ref.full_logits.size() != o.full_logits.size() ||
        std::memcmp(ref.full_logits.data(), o.full_logits.data(), ref.full_logits.size() * sizeof(float)) != 0)) ++bad;
    std::size_t diff = 0;
    for (std::size_t i = 0; i < ref.hashes.size() && i < o.hashes.size(); ++i)
        if (ref.hashes[i] != o.hashes[i] && ref.hashes[i].first.rfind("topk", 0) != 0) {
            if (diff < 6) std::printf("  rank %u %s: state %s differs\n", rank, name.c_str(), ref.hashes[i].first.c_str());
            ++diff;
        }
    std::printf("GATE rank %u %-14s resume %6u pieces %-24s token %6u logits %016llx digest %016llx  %s  ttft %.3f s (lookup+state %.3f, import %.3f, "
                "export+insert %.3f)",
                rank, name.c_str(), o.resume, list(o.pieces).c_str(), o.next_token, static_cast<unsigned long long>(o.logits),
                static_cast<unsigned long long>(o.digest), bad != 0 ? "DIFFERENT" : !o.logits_known || !ref.logits_known ? "STATE-ONLY" : "EQUAL", o.seconds, o.lookup_seconds, o.import_seconds,
                o.export_seconds);
    if (!ref.canonical.empty() && !o.canonical.empty())
        std::printf(" canonical-bytes %s, full-logits-bytes %s", o.canonical == ref.canonical ? "EQUAL" : "DIFFERENT",
                    logits_bytes ? "CHECKED" : "UNAVAILABLE (require fresh suffix)");
    if (o.disk_bytes != 0U)
        std::printf(" disk %.1f MB in %.3f s = %.2f GB/s", static_cast<double>(o.disk_bytes) / 1e6, o.disk_seconds,
                    o.disk_seconds > 0.0 ? static_cast<double>(o.disk_bytes) / o.disk_seconds / 1e9 : 0.0);
    std::printf("\n");
    std::fflush(stdout);
    return bad;
}

int run_disk(const Args& a, const std::vector<std::uint32_t>& prompt, std::uint32_t rank, strix::dist::Communicator* comm, bool print);

int run(const Args& a, const std::vector<std::uint32_t>& prompt, std::uint32_t rank, strix::dist::Communicator* comm, bool print) {
    if (!a.cache_dir.empty()) return run_disk(a, prompt, rank, comm, print);
    Gate g(a, rank, comm, print, prompt.size());
    const auto N = static_cast<std::uint32_t>(prompt.size());
    int bad = 0;
    const auto ref = g.fresh(prompt);
    std::printf("GATE rank %u %-14s resume %6u pieces %-24s token %6u logits %016llx digest %016llx  ref    time %.3f s (export %.3f)\n", rank, "fresh",
                0U, list(ref.pieces).c_str(), ref.next_token, static_cast<unsigned long long>(ref.logits), static_cast<unsigned long long>(ref.digest),
                ref.seconds, ref.export_seconds);
    // miss through the cache, then the same prompt again (a hit at its last full block, or a full hit)
    g.cache().clear();
    const auto miss = g.cached(prompt);
    bad += compare(rank, "miss", ref, miss);
    const auto again = g.cached(prompt);
    bad += compare(rank, "same-prompt", ref, again);
    Outcome last_hit = again;
    for (const long w : parse_list(a.warm)) {
        const long W = w < 0 ? static_cast<long>(N) + w : w;
        if (W <= 0 || W >= static_cast<long>(N)) continue;
        g.cache().clear();
        const std::vector<std::uint32_t> first(prompt.begin(), prompt.begin() + W);
        (void)g.cached(first);
        const auto st = g.cache().stats();
        for (std::uint32_t k = 0; k < a.repeat_hit; ++k) {
            const auto hit = g.cached(prompt);
            bad += compare(rank, "warm" + std::to_string(W) + (a.repeat_hit > 1U ? "#" + std::to_string(k) : ""), ref, hit);
            last_hit = hit;
            if (k + 1U < a.repeat_hit) {  // the next repetition resumes from the same snapshot: drop the full prompt's
                g.cache().clear();
                (void)g.cached(first);
            }
        }
        std::printf("  cache after the first prompt (%ld tokens): %llu B in %llu blocks + %llu snapshots = %.0f B per cached token\n", W,
                    static_cast<unsigned long long>(st.bytes), static_cast<unsigned long long>(st.blocks), static_cast<unsigned long long>(st.snapshots),
                    static_cast<double>(st.bytes) / static_cast<double>(W / a.block * a.block));
    }
    // every in-run snapshot against a fresh prefill of its tokens (the state a run ending there exports)
    for (const auto& s : std::vector<ds::AttentionState>(g.inrun_states())) {
        const auto f = g.fresh(s.tokens);
        auto x = s, y = f.state;
        x.next_token = y.next_token = 0;  // the in-run snapshot does not know it
        Outcome ox, oy;
        ox.digest = x.digest(true);
        ox.hashes = x.hashes();
        oy.digest = y.digest(true);
        oy.hashes = y.hashes();
        ox.next_token = oy.next_token;
        ox.logits_known = oy.logits_known = false;
        if (a.byte_check) { ox.canonical = ds::canonical_attention_state_bytes(x); oy.canonical = ds::canonical_attention_state_bytes(y); }
        ox.resume = s.position;
        bad += compare(rank, "inrun" + std::to_string(s.position), oy, ox);
    }
    if (a.byte_check) {
        auto suffix = prompt;
        suffix.insert(suffix.end(), prompt.begin(), prompt.begin() + std::min<std::size_t>(16, prompt.size()));
        const auto suffix_ref = g.fresh(suffix);
        const auto suffix_hit = g.cached(suffix);
        bad += compare(rank, "byte-suffix", suffix_ref, suffix_hit);
    }
    if (a.decode != 0U && g.has_batch()) {
        double import_seconds = 0.0;
        const auto [tx, ty] = g.decode(ref, last_hit, import_seconds);
        const bool equal = tx == ty;
        bad += equal ? 0 : 1;
        std::printf("GATE rank %u decode %u tokens from the fresh and the cached state: %s (slot import %.3f s each); tokens %s\n", rank, a.decode,
                    equal ? "EQUAL" : "DIFFERENT", import_seconds, list(tx).c_str());
        if (!equal) std::printf("  cached: %s\n", list(ty).c_str());
    }
    const auto st = g.cache().stats();
    std::printf("GATE rank %u %s (%d differences); cache %llu lookups %llu hits %llu inserts %llu refused\n", rank, bad == 0 ? "PASS" : "FAIL", bad,
                static_cast<unsigned long long>(st.lookups), static_cast<unsigned long long>(st.hits), static_cast<unsigned long long>(st.inserts),
                static_cast<unsigned long long>(st.refused));
    std::fflush(stdout);
    return bad;
}

// ---- disk tier gate ----
struct Reference {
    std::uint32_t tokens = 0, next_token = 0;
    std::uint64_t logits = 0, digest = 0;
    std::vector<std::uint32_t> decoded;
};

std::string ref_path(const Args& a, std::uint32_t rank) { return a.cache_dir + "/ref-rank" + std::to_string(rank) + ".txt"; }

void save_reference(const std::string& path, const Reference& r) {
    std::ofstream f(path);
    f << r.tokens << ' ' << r.next_token << ' ' << r.logits << ' ' << r.digest << ' ' << list(r.decoded) << '\n';
}

Reference load_reference(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::invalid_argument("no reference file " + path + " (run --phase write first)");
    Reference r;
    std::string dec;
    f >> r.tokens >> r.next_token >> r.logits >> r.digest >> dec;
    if (dec != "-")
        for (const long v : parse_list(dec)) r.decoded.push_back(static_cast<std::uint32_t>(v));
    return r;
}

Outcome as_outcome(const Reference& r) {
    Outcome o;
    o.next_token = r.next_token;
    o.logits = r.logits;
    o.digest = r.digest;
    return o;
}

// Damages one file of the rank's disk tier before it is opened (a payload byte flipped): the deepest snapshot on the
// prompt's chain, or the block in the middle of that chain.
void corrupt_one(const std::string& dir, const std::vector<std::uint32_t>& prompt, std::uint32_t block, const std::string& what) {
    namespace fs = std::filesystem;
    const auto name = [&](const char* prefix, std::uint32_t blocks, const char* ext) {
        char n[64];
        std::snprintf(n, sizeof n, "%s-%016llx.%s", prefix, static_cast<unsigned long long>(ds::prefix_key(prompt, block, blocks)), ext);
        return fs::path(dir) / n;
    };
    const auto D = static_cast<std::uint32_t>(prompt.size() / block);
    std::uint32_t deepest = 0;
    for (std::uint32_t k = D; k >= 1U && deepest == 0U; --k)
        if (fs::exists(name("s", k, "snp"))) deepest = k;
    if (deepest == 0U) throw std::runtime_error("--corrupt: no snapshot of the prompt in " + dir);
    const fs::path victim = what == "snapshot" ? name("s", deepest, "snp") : name("b", std::max<std::uint32_t>(1U, deepest / 2U), "blk");
    std::fstream io(victim, std::ios::in | std::ios::out | std::ios::binary);
    io.seekg(4096);
    char c = 0;
    io.read(&c, 1);
    c = static_cast<char>(c ^ 0x5a);
    io.seekp(4096);
    io.write(&c, 1);
    std::printf("corrupted %s (byte 4096; the prompt's deepest snapshot is at %u)\n", victim.string().c_str(), deepest * block);
}

int run_disk(const Args& a, const std::vector<std::uint32_t>& prompt, std::uint32_t rank, strix::dist::Communicator* comm, bool print) {
    const auto N = static_cast<std::uint32_t>(prompt.size());
    const std::string rpath = ref_path(a, rank);
    if (a.phase == "read" && !a.corrupt.empty()) corrupt_one(Gate::rank_dir(a, rank), prompt, a.block, a.corrupt);
    Gate g(a, rank, comm, print, std::max<std::size_t>(prompt.size() + a.session + 64U, prompt.size()));
    int bad = 0;
    const auto report_cache = [&](const char* when) {
        const auto st = g.cache().stats();
        std::printf("  cache %s: RAM %.1f MB (%llu blocks, %llu snapshots); disk %.1f MB (%llu blocks, %llu snapshots), written %llu files %.1f MB in %.3f s "
                    "(%.2f GB/s), read %llu files %.1f MB in %.3f s, disk hits %llu, read errors %llu\n",
                    when, static_cast<double>(st.bytes) / 1e6, static_cast<unsigned long long>(st.blocks), static_cast<unsigned long long>(st.snapshots),
                    static_cast<double>(st.disk_bytes) / 1e6, static_cast<unsigned long long>(st.disk_blocks), static_cast<unsigned long long>(st.disk_snapshots),
                    static_cast<unsigned long long>(st.written_files), static_cast<double>(st.written_bytes) / 1e6, st.write_seconds,
                    st.write_seconds > 0.0 ? static_cast<double>(st.written_bytes) / st.write_seconds / 1e9 : 0.0,
                    static_cast<unsigned long long>(st.read_files), static_cast<double>(st.read_bytes) / 1e6, st.read_seconds,
                    static_cast<unsigned long long>(st.disk_hits), static_cast<unsigned long long>(st.read_errors));
        std::fflush(stdout);
    };
    const auto decode_one = [&](const Outcome& o) {
        double import_seconds = 0.0;
        auto [tx, ty] = g.decode(o, o, import_seconds);
        std::printf("  decode %u tokens from %s state (slot import %.3f s)\n", a.decode, "this", import_seconds);
        return tx;
    };
    if (a.phase == "fresh") {
        const auto ref = load_reference(rpath);
        if (ref.tokens != N) throw std::invalid_argument("the reference is of another prompt length");
        const auto fr = g.fresh(prompt);
        bad += compare(rank, "fresh-vs-ref", as_outcome(ref), fr);
        std::printf("  fresh prefill %.3f s\n", fr.seconds);
        if (a.decode != 0U && g.has_batch()) {
            const auto tx = decode_one(fr);
            const bool eq = tx == ref.decoded;
            bad += eq ? 0 : 1;
            std::printf("GATE rank %u decode %u tokens from the fresh state vs the reference: %s\n", rank, a.decode, eq ? "EQUAL" : "DIFFERENT");
        }
    } else if (a.phase == "read") {  // a new process on the files of a write phase
        const auto ref = load_reference(rpath);
        if (ref.tokens != N) throw std::invalid_argument("the reference is of another prompt length");
        const auto R = a.byte_check ? g.fresh(prompt) : as_outcome(ref);
        if (a.byte_check) bad += compare(rank, "restart-fresh", as_outcome(ref), R);
        report_cache("after the restart");
        const auto disk = g.cached(prompt);
        bad += compare(rank, "restart-disk", R, disk);
        const auto ram = g.cached(prompt);
        bad += compare(rank, "restart-ram", R, ram);
        if (a.decode != 0U && g.has_batch()) {
            const auto tx = decode_one(disk);
            const bool eq = tx == ref.decoded;
            bad += eq ? 0 : 1;
            std::printf("GATE rank %u decode %u tokens from the restart disk-hit state vs the reference: %s; tokens %s\n", rank, a.decode,
                        eq ? "EQUAL" : "DIFFERENT", list(tx).c_str());
        }
        // a warm hit (multi-turn shape) from disk: drop the RAM tier, the full prompt resumes below its end
        for (const long w : parse_list(a.warm)) {
            const long W = w < 0 ? static_cast<long>(N) + w : w;
            if (W <= 0 || W >= static_cast<long>(N)) continue;
            g.cache().clear_ram();
            const std::vector<std::uint32_t> first(prompt.begin(), prompt.begin() + W);
            const auto turn1 = g.cached(first);  // from disk: the snapshot at or below W, then W's tail (inserted)
            g.cache().clear_ram();
            const auto hit = g.cached(prompt);
            bad += compare(rank, "restart-warm" + std::to_string(W), R, hit);
            std::printf("  (turn 1 of %ld tokens: resume %u from disk, %.3f s)\n", W, turn1.resume, turn1.seconds);
        }
        report_cache("at the end");
    } else {  // write
        Outcome ref;
        if (!a.no_fresh) {
            ref = g.fresh(prompt);
            std::printf("GATE rank %u %-14s resume %6u pieces %-24s token %6u logits %016llx digest %016llx  ref    time %.3f s (export %.3f)\n", rank, "fresh",
                        0U, list(ref.pieces).c_str(), ref.next_token, static_cast<unsigned long long>(ref.logits),
                        static_cast<unsigned long long>(ref.digest), ref.seconds, ref.export_seconds);
        }
        g.cache().clear();
        const auto miss = g.cached(prompt);
        g.cache().flush();
        if (a.no_fresh) {
            ref = miss;
            std::printf("GATE rank %u %-14s resume %6u pieces %-24s token %6u logits %016llx digest %016llx  ref    time %.3f s (export+insert %.3f)\n", rank,
                        "miss(ref)", 0U, list(miss.pieces).c_str(), miss.next_token, static_cast<unsigned long long>(miss.logits),
                        static_cast<unsigned long long>(miss.digest), miss.seconds, miss.export_seconds);
        } else {
            bad += compare(rank, "miss", ref, miss);
        }
        report_cache("after the miss");
        const auto ram = g.cached(prompt);
        bad += compare(rank, "ram-hit", ref, ram);
        g.cache().clear_ram();
        const auto disk = g.cached(prompt);
        bad += compare(rank, "disk-hit", ref, disk);
        Reference saved{N, ref.next_token, ref.logits, ref.digest, {}};
        if (a.decode != 0U && g.has_batch()) {
            double import_seconds = 0.0;
            const auto [tx, ty] = g.decode(ref, disk, import_seconds);
            const bool eq = tx == ty;
            bad += eq ? 0 : 1;
            std::printf("GATE rank %u decode %u tokens from the reference and the disk-hit state: %s (slot import %.3f s each); tokens %s\n", rank, a.decode,
                        eq ? "EQUAL" : "DIFFERENT", import_seconds, list(tx).c_str());
            saved.decoded = tx;
        }
        for (const long w : parse_list(a.warm)) {  // multi-turn shape, turn 2 resumed from disk
            const long W = w < 0 ? static_cast<long>(N) + w : w;
            if (W <= 0 || W >= static_cast<long>(N)) continue;
            g.cache().clear();
            const std::vector<std::uint32_t> first(prompt.begin(), prompt.begin() + W);
            (void)g.cached(first);
            const auto ram_hit = g.cached(prompt);
            bad += compare(rank, "warm" + std::to_string(W) + "-ram", ref, ram_hit);
            g.cache().clear();
            (void)g.cached(first);
            g.cache().clear_ram();
            const auto disk_hit = g.cached(prompt);
            bad += compare(rank, "warm" + std::to_string(W) + "-disk", ref, disk_hit);
        }
        if (a.fork != 0U && a.fork < N) {  // two continuations sharing the first F tokens
            std::vector<std::uint32_t> other(prompt.begin(), prompt.begin() + a.fork);
            other.insert(other.end(), prompt.rbegin(), prompt.rbegin() + (N - a.fork));
            const auto ref_b = g.fresh(other);
            g.cache().clear();
            (void)g.cached(prompt);
            g.cache().clear_ram();
            const auto b = g.cached(other);
            bad += compare(rank, "fork-B", ref_b, b);
            const auto a2 = g.cached(prompt);
            bad += compare(rank, "fork-A-again", ref, a2);
            g.cache().clear_ram();
            const auto b2 = g.cached(other);
            bad += compare(rank, "fork-B-disk", ref_b, b2);
        }
        if (a.rewind != 0U && a.rewind < N) {  // the first R tokens: the deepest snapshot <= R, then the rest re-prefilled
            const std::vector<std::uint32_t> head(prompt.begin(), prompt.begin() + a.rewind);
            const auto ref_r = g.fresh(head);
            g.cache().clear();
            (void)g.cached(prompt);
            g.cache().clear_ram();
            const auto r = g.cached(head);
            bad += compare(rank, "rewind" + std::to_string(a.rewind), ref_r, r);
            std::printf("  rewind to %u: resumed at %u, re-prefilled %u rows in %.3f s (fresh %.3f s)\n", a.rewind, r.resume, a.rewind - r.resume, r.seconds,
                        ref_r.seconds);
        }
        if (a.session != 0U && g.has_batch()) {  // session snapshots taken by the decode
            g.cache().clear();
            const auto base = g.cached(prompt);
            std::uint32_t last = 0;
            const auto conv = g.session(base, a.session, last);
            if (last == 0U) {
                std::printf("GATE rank %u session: no block boundary reached (decode more)\n", rank);
                ++bad;
            } else {
                const std::vector<std::uint32_t> upto(conv.begin(), conv.begin() + last);
                const auto ref_s = g.fresh(upto);
                g.cache().clear_ram();
                auto full = g.cached(upto);
                full.logits_known = false;  // state-only; fresh session-turn suffix below must prove logits
                bad += compare(rank, "session-full", ref_s, full);
                std::vector<std::uint32_t> next_turn = upto;
                next_turn.insert(next_turn.end(), prompt.begin(), prompt.begin() + 40);
                const auto ref_t = g.fresh(next_turn);
                g.cache().clear_ram();
                const auto t = g.cached(next_turn);
                bad += compare(rank, "session-turn", ref_t, t);
                if (a.decode != 0U) {
                    double import_seconds = 0.0;
                    const auto [cold, restored] = g.decode(ref_t, t, import_seconds);
                    bad += cold == restored ? 0 : 1;
                    std::printf("GATE rank %u session suffix decode: %s; tokens %s\n", rank,
                                cold == restored ? "EQUAL" : "DIFFERENT", list(restored).c_str());
                }
            }
        }
        report_cache("at the end");
        g.cache().flush();
        if (a.phase == "write") save_reference(rpath, saved);
    }
    if (a.byte_check && a.phase != "fresh") {
        auto suffix = prompt;
        suffix.insert(suffix.end(), prompt.begin(), prompt.begin() + std::min<std::size_t>(16, prompt.size()));
        const auto cold = g.fresh(suffix);
        const auto restored = g.cached(suffix);
        bad += compare(rank, "disk-byte-suffix", cold, restored);
    }
    const auto st = g.cache().stats();
    std::printf("GATE rank %u %s (%d differences); cache %llu lookups %llu hits %llu inserts %llu refused, disk hits %llu\n", rank, bad == 0 ? "PASS" : "FAIL", bad,
                static_cast<unsigned long long>(st.lookups), static_cast<unsigned long long>(st.hits), static_cast<unsigned long long>(st.inserts),
                static_cast<unsigned long long>(st.refused), static_cast<unsigned long long>(st.disk_hits));
    std::fflush(stdout);
    return bad;
}

}  // namespace

int main(int argc, char** argv) try {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        const auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(s + " needs a value");
            return std::string(argv[++i]);
        };
        const auto u32 = [&] { return static_cast<std::uint32_t>(std::stoul(next())); };
        if (s == "--prompt") a.prompt_path = next();
        else if (s == "--rows") a.rows = u32();
        else if (s == "--layers") a.layers = u32();
        else if (s == "--world") a.world = u32();
        else if (s == "--rank") a.rank = u32(), a.world = 2;
        else if (s == "--transport") a.transport = next(), a.world = 2;
        else if (s == "--stream-experts") a.stream_experts = true;
        else if (s == "--block") a.block = u32();
        else if (s == "--max-rows") a.max_rows = u32();
        else if (s == "--cache-mb") a.cache_mb = std::stoull(next());
        else if (s == "--warm") a.warm = next();
        else if (s == "--decode") a.decode = u32();
        else if (s == "--repeat-hit") a.repeat_hit = std::max<std::uint32_t>(1U, u32());
        else if (s == "--no-shared-weights") a.shared = false;
        else if (s == "--before-tail") a.before_tail = true;
        else if (s == "--inrun") a.inrun = true;
        else if (s == "--byte-check") a.byte_check = true;
        else if (s == "--inrun-check") a.inrun = a.inrun_check = true;
        else if (s == "--cache-dir") a.cache_dir = next();
        else if (s == "--disk-mb") a.disk_mb = std::stoull(next());
        else if (s == "--phase") a.phase = next();
        else if (s == "--no-fresh") a.no_fresh = true;
        else if (s == "--fork") a.fork = u32();
        else if (s == "--rewind") a.rewind = u32();
        else if (s == "--corrupt") a.corrupt = next();
        else if (s == "--session") a.session = u32();
        else if (s == "--connect-timeout") a.connect_timeout = static_cast<unsigned>(std::stoul(next()));
        else if (a.model.empty()) a.model = s;
        else throw std::invalid_argument("unknown argument " + s);
    }
    if (a.model.empty() || a.prompt_path.empty()) throw std::invalid_argument("usage: strix-deepseek41-prefix MODEL.gguf --prompt FILE ...");
    if (a.phase != "all" && a.phase != "write" && a.phase != "read" && a.phase != "fresh") throw std::invalid_argument("--phase all|write|read|fresh");
    if (a.session != 0U && a.decode == 0U) throw std::invalid_argument("--session needs --decode (the batch forward)");
    if (a.stream_experts && a.decode != 0U) throw std::invalid_argument("--decode needs the experts resident (not --stream-experts)");
    // The expert layout as the serve sets it: expert TP only when STRIX_DS41_EXPERT_TP=1, then the dup layout the TP
    // prefill reads in place (else the two-rank default would load expert TP and the prefill a second expert copy).
    strix::ds41::expert_tp::set_decode_default(false);
    if (strix::ds41::expert_tp::enabled_from_env()) strix::ds41::expert_tp::set_packed(0);
    const auto prompt = read_prompt(a.prompt_path, a.rows);
    std::printf("prompt %zu tokens, block %u, max rows %u, layers %u\n", prompt.size(), a.block, a.max_rows, a.layers);
    if (a.world == 1U) return run(a, prompt, 0U, nullptr, true) == 0 ? 0 : 1;
    if (a.transport == "emulate") {
        auto [t0, t1] = strix::dist::memory_pair();
        std::exception_ptr failure;
        int bad1 = 0;
        std::thread peer([&, t = std::move(t1)]() mutable {
            try {
                strix::dist::Communicator comm(strix::dist::Group{1U, 2U, 1U, 0x70726678ULL}, std::move(t), std::size_t{22} << 20U);
                bad1 = run(a, prompt, 1U, &comm, false);
            } catch (...) {
                failure = std::current_exception();
            }
        });
        int bad0 = 0;
        try {
            strix::dist::Communicator comm(strix::dist::Group{0U, 2U, 1U, 0x70726678ULL}, std::move(t0), std::size_t{22} << 20U);
            bad0 = run(a, prompt, 0U, &comm, true);
        } catch (...) {
            peer.join();
            throw;
        }
        peer.join();
        if (failure) std::rethrow_exception(failure);
        return bad0 == 0 && bad1 == 0 ? 0 : 1;
    }
    ::alarm(a.connect_timeout + 30U);
    auto opened = strix::dist::open_pair(a.transport, a.rank, {std::chrono::seconds(a.connect_timeout)});
    ::alarm(0);
    if (!opened.receipt.empty()) std::printf("usb4 topology: %s\n", opened.receipt.c_str());
    strix::dist::Communicator comm(strix::dist::Group{a.rank, 2U, 1U, 0x70726678ULL}, std::move(opened.host), std::move(opened.direct),
                                   std::size_t{22} << 20U);
    std::printf("connected (rank %u of 2)\n", a.rank);  // the pair launchers' watchdog waits for this line
    std::fflush(stdout);
    return run(a, prompt, a.rank, &comm, true) == 0 ? 0 : 1;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
