#include "strix/residency/expert_residency.hpp"

#include "strix/env_flags.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <limits>
#include <stdexcept>

namespace strix::residency {
namespace {
constexpr std::uint64_t kGiB = 1ull << 30;

double parse_gib(const char* what, const std::string& s) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || end == s.c_str() || *end != '\0' || !std::isfinite(v) || v <= 0.0 || v > 1.0e6)
        throw std::invalid_argument(std::string(what) + ": expected a positive number of GiB, got '" + s + "'");
    return v;
}
std::uint64_t gib_to_bytes(double g) { return static_cast<std::uint64_t>(g * double(kGiB)); }

bool parse_mode(std::string_view s, Mode& m) {
    if (s == "auto") m = Mode::automatic;
    else if (s == "full") m = Mode::full;
    else if (s == "tiered") m = Mode::tiered;
    else return false;
    return true;
}
const char* mode_name(Mode m) { return m == Mode::full ? "full" : m == Mode::tiered ? "tiered" : "auto"; }

std::string gib(std::uint64_t b) {
    char buf[48];
    std::snprintf(buf, sizeof buf, "%.2f", double(b) / double(kGiB));
    return buf;
}
std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += char(c); }
        else if (c == '\n') o += "\\n";
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += char(c);
    }
    return o + "\"";
}
std::string num(double v) {
    char buf[48];
    std::snprintf(buf, sizeof buf, "%.6g", v);
    return buf;
}
}  // namespace

std::uint64_t grouped_bytes(std::uint64_t positions, std::uint32_t group, std::uint64_t bytes) {
    if (!group) throw std::invalid_argument("residency: zero storage group");
    const auto groups = positions / group + (positions % group != 0);
    if (bytes && groups > std::numeric_limits<std::uint64_t>::max() / bytes)
        throw std::overflow_error("residency: storage size overflow");
    return groups * bytes;
}

std::uint64_t session_bytes(const SessionBytes& s, std::uint32_t positions) {
    return grouped_bytes(positions, 1, s.per_position) + grouped_bytes(positions, s.group_size, s.per_group) + s.fixed;
}

void validate_expert_tensor(const std::vector<std::uint64_t>& dims, std::uint64_t bytes,
                            std::uint64_t input, std::uint64_t output, std::uint64_t experts,
                            std::uint64_t block_elements, std::uint64_t block_bytes) {
    if (dims != std::vector<std::uint64_t>{input, output, experts} || !input || !output || !experts ||
        !block_elements || !block_bytes || input % block_elements)
        throw std::invalid_argument("residency: expert tensor shape or block geometry mismatch");
    auto expected = input / block_elements;
    for (const auto factor : {output, experts, block_bytes}) {
        if (expected > std::numeric_limits<std::uint64_t>::max() / factor)
            throw std::overflow_error("residency: expert tensor size overflow");
        expected *= factor;
    }
    if (bytes != expected) throw std::invalid_argument("residency: expert tensor byte size mismatch");
}

bool allocation_fits(std::uint64_t budget, std::uint64_t initial_free, std::uint64_t free_now, std::uint64_t bytes) {
    const auto used = initial_free > free_now ? initial_free - free_now : 0;
    return bytes <= free_now && (!budget || (used <= budget && bytes <= budget - used));
}

bool parse_flag(std::string_view arg, const std::function<std::string()>& next, Options& out) {
    if (arg == "--expert-residency") {
        const std::string v = next();
        if (!parse_mode(v, out.mode))
            throw std::invalid_argument("--expert-residency: expected auto|full|tiered, got '" + v + "'");
        out.mode_explicit = true;
        return true;
    }
    if (arg == "--vram-budget-gib") {
        out.vram_budget = gib_to_bytes(parse_gib("--vram-budget-gib", next()));
        if (out.vram_budget == 0) throw std::invalid_argument("--vram-budget-gib: too small");
        return true;
    }
    return false;
}

Options with_environment(Options in) {
    if (!in.mode_explicit && in.mode == Mode::automatic) {
        if (const char* v = envcfg::text("STRIX_EXPERT_RESIDENCY"); v && *v) {
            if (!parse_mode(v, in.mode))
                throw std::invalid_argument(std::string("STRIX_EXPERT_RESIDENCY: expected auto|full|tiered, got '") + v + "'");
        }
    }
    if (const char* v = envcfg::text("STRIX_TIER_HOST_CHECK"); v && v[0] == '0') in.host_check = false;
    if (in.vram_budget == 0) {
        if (const char* v = envcfg::text("STRIX_VRAM_BUDGET_GIB"); v && *v)
            in.vram_budget = gib_to_bytes(parse_gib("STRIX_VRAM_BUDGET_GIB", v));
    }
    return in;
}

BudgetCap budget_cap(const Options& opt, const CardBytes& card) {
    BudgetCap c;
    c.bytes = opt.vram_budget ? opt.vram_budget : card.total;
    c.available = std::min(card.free, c.bytes);
    if (opt.vram_budget) c.source = "residency";
    if (opt.device_cap) {
        const auto used = card.total > card.free ? card.total - card.free : 0;
        const auto available = opt.device_cap > used ? opt.device_cap - used : 0;
        if (available < c.available || (!opt.vram_budget && available == c.available)) {
            c.bytes = opt.device_cap;
            c.available = std::min(card.free, available);
            c.source = "device_profile";
        }
    }
    return c;
}

CardBytes live_budget_view(const CardBytes& physical, std::uint64_t device_cap,
                           std::uint64_t engine_allowance, std::uint64_t engine_used) {
    CardBytes view = physical;
    if (device_cap && device_cap < physical.total) {
        const auto used = physical.total > physical.free ? physical.total - physical.free : 0;
        view.free = used < device_cap ? device_cap - used : 0;
        view.total = device_cap;
    }
    if (engine_allowance) {
        const auto remaining = engine_allowance > engine_used ? engine_allowance - engine_used : 0;
        view.free = std::min(view.free, remaining);
        view.total = std::min(view.total, engine_allowance);
    }
    return view;
}

Decision resolve(const Options& opt, const ModelBytes& m, const CardBytes& card, const SessionBytes& s,
                 const HostBytes& host) {
    Decision d;
    d.positions_requested = s.positions;
    d.positions = s.positions;
    const BudgetCap applied = budget_cap(opt, card);
    const std::uint64_t cap = applied.bytes;
    d.cap = cap;
    d.cap_source = applied.source;
    d.budget = applied.available;
    if (m.expert_bytes.size() != m.layers || m.layers == 0 || m.experts == 0) {
        d.error = "residency: model bytes inconsistent (expert_bytes size != layers, or empty)";
        d.reason = d.error;
        return d;
    }
    const std::uint64_t zero_all = std::accumulate(m.expert_bytes.begin(), m.expert_bytes.end(), std::uint64_t{0});
    const std::uint64_t experts_all = zero_all * m.experts;
    auto sessions = [&](std::uint32_t pos) {
        return std::uint64_t{s.sessions} * session_bytes(s, pos);
    };
    const std::uint64_t fixed_cost = m.core + s.scratch + s.margin;
    const auto scratch = [&](std::uint32_t pos) { return s.position_scratch ? s.position_scratch(pos) : 0ULL; };
    const std::uint64_t need_full = fixed_cost + scratch(s.positions) + experts_all + sessions(s.positions);
    d.session_total = sessions(s.positions);

    const bool fits = need_full <= d.budget;
    if (opt.mode == Mode::full || (opt.mode == Mode::automatic && fits)) {
        d.mode = Mode::full;
        d.slots_per_layer = m.experts;
        d.host_expert_bytes = 0;
        if (fits) {
            d.ok = true;
            d.reason = std::string(opt.mode == Mode::full ? "full requested, " : "") + "fits: " + gib(need_full) +
                       " of " + gib(d.budget) + " GiB";
        } else {
            d.error = "expert-residency full does not fit: needs " + gib(need_full) + " GiB, budget " + gib(d.budget) +
                      " GiB, short " + gib(need_full - d.budget) + " GiB (use auto or tiered)";
            d.reason = d.error;
        }
        return d;
    }

    d.mode = Mode::tiered;
    d.host_expert_bytes = experts_all;
    // Positions: the requested count first, then multiples of 1024 downwards (down to 1024).
    std::vector<std::uint32_t> tries{s.positions};
    for (std::uint32_t p = (s.positions > 1024 ? (s.positions - 1) / 1024 * 1024 : 0); p >= 1024; p -= 1024) {
        tries.push_back(p);
    }
    bool found = false;
    for (std::uint32_t pos : tries) {
        const std::uint64_t used = fixed_cost + scratch(pos) + zero_all + sessions(pos);
        if (used <= d.budget && zero_all <= d.budget - used) {
            d.positions = pos;
            d.session_total = sessions(pos);
            d.expert_pool = d.budget - used;
            found = true;
            break;
        }
    }
    if (!found) {
        const std::uint32_t pos = tries.back();
        const std::uint64_t used = fixed_cost + scratch(pos) + 2 * zero_all + sessions(pos);
        d.positions = pos;
        d.session_total = sessions(pos);
        d.error = "tiered does not fit: core+scratch+margin+zero and resident experts+" + std::to_string(pos) + " positions need " +
                  gib(used) + " GiB, budget " + gib(d.budget) + " GiB";
        d.reason = d.error;
        return d;
    }
    d.slots_per_layer = static_cast<std::uint32_t>(std::min<std::uint64_t>(d.expert_pool / zero_all, m.experts));
    d.reason = "tiered: " + gib(need_full) + " GiB needed, budget " + gib(d.budget) + " GiB, " +
               std::to_string(d.slots_per_layer) + "/" + std::to_string(m.experts) + " slots per layer";
    if (d.positions != d.positions_requested)
        d.reason += ", positions reduced " + std::to_string(d.positions_requested) + " -> " + std::to_string(d.positions);
    // the CPU only reads experts that are not resident (promotions re-read the mapped model file)
    std::uint64_t resident = 0;
    for (const auto b : m.expert_bytes) resident += std::uint64_t{d.slots_per_layer} * b;
    const std::uint64_t host_need = experts_all - std::min(resident, experts_all) + m.host_tables + 4 * kGiB;
    if (host.available < host_need) {
        const std::string shortfall = "host RAM short for tiered experts: need " + gib(host_need) +
                                      " GiB (non-resident experts + tables + 4 GiB), available " + gib(host.available) + " GiB";
        if (opt.host_check) {
            d.error = shortfall;
            d.reason += "; " + d.error;
            return d;
        }
        d.reason += "; " + shortfall + " (STRIX_TIER_HOST_CHECK=0: cold experts page from the model file)";
    }
    d.ok = true;
    return d;
}

std::string decision_json(const Decision& d) {
    std::string o = "{\"expert_residency\":";
    o += json_str(mode_name(d.mode));
    o += ",\"reason\":" + json_str(d.reason);
    o += ",\"vram_budget_bytes\":" + std::to_string(d.budget);
    o += ",\"vram_cap_bytes\":" + std::to_string(d.cap);
    o += ",\"vram_cap_source\":" + json_str(d.cap_source);
    o += ",\"expert_pool_bytes\":" + std::to_string(d.expert_pool);
    o += ",\"slots_per_layer\":" + std::to_string(d.slots_per_layer);
    o += ",\"positions\":" + std::to_string(d.positions);
    o += ",\"positions_requested\":" + std::to_string(d.positions_requested);
    o += ",\"host_expert_bytes\":" + std::to_string(d.host_expert_bytes) + "}";
    return o;
}

std::string receipt_json(const Receipt& r) {
    std::string o = "{\"active_bytes\":" + std::to_string(r.active_bytes);
    o += ",\"hit_bytes\":" + std::to_string(r.hit_bytes);
    o += ",\"byte_hit_rate\":" + num(r.byte_hit_rate());
    o += ",\"promoted_bytes\":" + std::to_string(r.promoted_bytes);
    o += ",\"promotions\":" + std::to_string(r.promotions);
    o += ",\"evictions\":" + std::to_string(r.evictions);
    o += ",\"steps\":" + std::to_string(r.steps);
    o += ",\"cpu_ms\":" + num(r.cpu_ms);
    o += ",\"upload_ms\":" + num(r.upload_ms);
    o += ",\"exchange_ms\":" + num(r.exchange_ms);
    o += ",\"resident_bytes\":" + std::to_string(r.resident_bytes);
    o += ",\"context_filled\":" + std::to_string(r.context_filled) + "}";
    return o;
}

Planner::Planner(const ModelBytes& m, std::uint32_t slots_per_layer, PlannerConfig cfg)
    : layers_(m.layers), experts_(m.experts), slots_(std::min(slots_per_layer, m.experts)), top_k_(m.top_k), cfg_(cfg),
      bytes_(m.expert_bytes) {
    if (bytes_.size() != layers_) throw std::invalid_argument("Planner: expert_bytes size != layers");
    score_.assign(layers_, std::vector<double>(experts_, 0.0));
    table_.assign(layers_, std::vector<std::int32_t>(experts_, -1));
    slot_expert_.assign(layers_, std::vector<std::int32_t>(slots_, -1));
    used_.assign(layers_, std::vector<std::uint8_t>(experts_, 0));
}

void Planner::seed(std::uint32_t layer, const std::uint32_t* counts) {
    double total = 0;
    for (std::uint32_t e = 0; e < experts_; ++e) total += counts[e];
    if (total <= 0) return;
    // Counts -> steady-state decayed score of a per-step use probability (counts / total * top_k, clamped to 1).
    const double scale = 1.0 / (1.0 - cfg_.decay);
    const double k = top_k_ ? double(top_k_) : 1.0;
    for (std::uint32_t e = 0; e < experts_; ++e) score_[layer][e] = std::min(1.0, counts[e] / total * k) * scale;
}

void Planner::observe(std::uint32_t layer, const std::uint32_t* ids, std::uint32_t n) {
    if (ended_) {
        for (auto& u : used_) std::fill(u.begin(), u.end(), std::uint8_t{0});
        ended_ = false;
    }
    for (std::uint32_t i = 0; i < n; ++i)
        if (ids[i] < experts_) used_[layer][ids[i]] = 1;
}

void Planner::end_step() {
    for (std::uint32_t l = 0; l < layers_; ++l)
        for (std::uint32_t e = 0; e < experts_; ++e) {
            score_[l][e] = score_[l][e] * cfg_.decay + (used_[l][e] ? 1.0 : 0.0);
            if (used_[l][e]) {
                rc_.active_bytes += bytes_[l];
                if (table_[l][e] >= 0) rc_.hit_bytes += bytes_[l];
            }
        }
    ++rc_.steps;
    ended_ = true;  // used_ stays valid for plan() until the next observe()
}

std::vector<Move> Planner::plan() const {
    struct Cand {
        Move mv;
        double gain;
        bool free_slot;
    };
    std::vector<Cand> all;
    for (std::uint32_t l = 0; l < layers_; ++l) {
        std::vector<std::uint32_t> cands, victims, free_slots;
        for (std::uint32_t e = 0; e < experts_; ++e)
            if (table_[l][e] < 0 && score_[l][e] > 0.0) cands.push_back(e);
        for (std::uint32_t s = 0; s < slots_; ++s) {
            const std::int32_t e = slot_expert_[l][s];
            if (e < 0) free_slots.push_back(s);
            else if (!used_[l][std::uint32_t(e)]) victims.push_back(std::uint32_t(e));
        }
        std::sort(cands.begin(), cands.end(), [&](std::uint32_t a, std::uint32_t b) {
            return score_[l][a] != score_[l][b] ? score_[l][a] > score_[l][b] : a < b;
        });
        std::sort(victims.begin(), victims.end(), [&](std::uint32_t a, std::uint32_t b) {
            return score_[l][a] != score_[l][b] ? score_[l][a] < score_[l][b] : a < b;
        });
        std::size_t ci = 0;
        for (std::uint32_t s : free_slots) {
            if (ci >= cands.size()) break;
            Move mv;
            mv.layer = l;
            mv.slot = s;
            mv.expert = cands[ci++];
            all.push_back({mv, score_[l][mv.expert], true});
        }
        for (std::size_t vi = 0; vi < victims.size() && ci < cands.size(); ++vi, ++ci) {
            const double gain = score_[l][cands[ci]] - score_[l][victims[vi]];  // uniform bytes within a layer
            if (!(gain > cfg_.margin_reuses)) break;
            Move mv;
            mv.layer = l;
            mv.slot = std::uint32_t(table_[l][victims[vi]]);
            mv.evict = victims[vi];
            mv.expert = cands[ci];
            all.push_back({mv, gain, false});
        }
    }
    std::stable_sort(all.begin(), all.end(), [](const Cand& a, const Cand& b) {
        if (a.free_slot != b.free_slot) return a.free_slot;
        if (a.gain != b.gain) return a.gain > b.gain;
        if (a.mv.layer != b.mv.layer) return a.mv.layer < b.mv.layer;
        return a.mv.expert < b.mv.expert;
    });
    std::vector<Move> out;
    std::uint64_t spent = 0;
    for (const Cand& c : all) {
        const std::uint64_t b = bytes_[c.mv.layer];
        if (spent + b > cfg_.upload_bytes_per_step && !(out.empty() && b > cfg_.upload_bytes_per_step)) continue;
        spent += b;
        out.push_back(c.mv);
    }
    return out;
}

void Planner::commit(const Move& mv) {
    if (mv.evict != Move::kNone) {
        table_[mv.layer][mv.evict] = -1;
        ++rc_.evictions;
    }
    slot_expert_[mv.layer][mv.slot] = std::int32_t(mv.expert);
    table_[mv.layer][mv.expert] = std::int32_t(mv.slot);
    ++rc_.promotions;
    rc_.promoted_bytes += bytes_[mv.layer];
}

void Planner::fill_from_seed() {
    for (std::uint32_t l = 0; l < layers_; ++l) {
        std::vector<std::uint32_t> order(experts_);
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(order.begin(), order.end(),
                         [&](std::uint32_t a, std::uint32_t b) { return score_[l][a] > score_[l][b]; });
        std::fill(table_[l].begin(), table_[l].end(), -1);
        for (std::uint32_t s = 0; s < slots_; ++s) {
            slot_expert_[l][s] = std::int32_t(order[s]);
            table_[l][order[s]] = std::int32_t(s);
        }
    }
}

Receipt Planner::receipt() const {
    Receipt r = rc_;
    r.resident_bytes = 0;
    for (std::uint32_t l = 0; l < layers_; ++l)
        for (std::uint32_t s = 0; s < slots_; ++s)
            if (slot_expert_[l][s] >= 0) r.resident_bytes += bytes_[l];
    return r;
}

}  // namespace strix::residency
