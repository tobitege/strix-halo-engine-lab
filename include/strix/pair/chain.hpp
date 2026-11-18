#pragma once
// Heterogeneous layer chain planner, shared by every engine and backend (model- and vendor-agnostic).
// Devices sit on a line D0 - D1 - ... - Dk-1 (e.g. a discrete card, then the two nodes of a pair); the link
// between neighbours i and i+1 is directed (forward = towards i+1, reverse = towards i). A plan runs one prompt
// through a contiguous run of devices [first, last]: stage s runs layers [cuts[s], cuts[s+1]) of every block of
// `block` rows and the last stage also runs the tail (final norm, head). The one-device run is always a candidate,
// so a slow link makes the planner keep the whole prefill on one device.
//
// Wall time: a flow shop of blocks (the last one partial) over the stage and forward-hop resources in order,
//   done[r][j] = max(done[r-1][j], done[r][j-1]) + t_r(rows_j),
// t_stage = block_fixed_ms + rows x sum(layer ms/row) (+ tail), t_hop = latency_ms + rows x bytes_per_row / bandwidth.
// After the prefill one `decoder` device owns the whole state: per-row state of layers owned elsewhere streams per
// block over the hops on its way (forward hops join the flow shop; reverse hops wait for their producing stage
// and upstream relay, coalescing one message per block/hop). Fixed state crosses once at the end, store-and-forward.
// Costs come from the engine's measured receipts; a device without costs never runs layers and a needed link
// without bandwidth rejects the plan (unknown is not infinite).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::pair::chain {
// Host prediction only: bound resource updates per plan and total search work. Exhaustion never returns a winner.
inline constexpr std::uint64_t max_plan_work = 1'000'000U;
inline constexpr std::uint64_t max_search_work = 20'000'000U;
struct Device {
    std::string name;
    std::vector<double> layer_ms_per_row; // one per model layer; empty: decoder-only device
    double tail_ms_per_row = 0.0;         // final norm / head when this device runs the last layer
    double block_fixed_ms = 0.0;          // per-block launch, sync and staging cost
    double capacity_bytes = 0.0;          // weight budget; checked only when the model declares weight bytes
    // Measured per-row cost multiplier by block rows {rows, factor}, ascending rows (piecewise linear in log2 rows,
    // clamped at both ends; empty = 1). Small units cost more per row (launches, fixed waits, smaller GEMMs).
    std::vector<std::pair<std::uint32_t, double>> unit_scale{};
};
struct Link {                             // between device i and i+1
    double forward_bytes_per_ms = 0.0;    // useful rate towards i+1 (0: no link)
    double reverse_bytes_per_ms = 0.0;    // useful rate towards i (0: no link)
    double latency_ms = 0.0;              // per message (one per block and hop)
};
struct ModelShape {
    std::uint32_t layers = 0;
    double seam_bytes_per_row = 0.0;       // activations crossing a layer cut
    std::vector<double> state_bytes_per_row; // per layer: prefix state growing with rows (KV, indices)
    std::vector<double> state_fixed_bytes;   // per layer: state of fixed size (recurrent matrices, convolutions)
    std::vector<double> weight_bytes;        // per layer, optional (empty: memory unchecked)
};
// Split toggle (one value for CLI and server, resolved here): auto_plan searches everything; device_only keeps the
// prefill on one device that also decodes; handoff runs one device and moves the whole state to another decoder;
// layer_cut uses exactly cuts.size() + 1 stages, a non-zero entry fixing that interior cut and 0 leaving it to the planner.
enum class Split { auto_plan, device_only, handoff, layer_cut };
inline const char* split_name(Split split) {
    switch (split) {
    case Split::device_only: return "device-only";
    case Split::handoff: return "handoff-full";
    case Split::layer_cut: return "layer-cut";
    default: return "auto";
    }
}
struct Request {
    Split split = Split::auto_plan;
    std::vector<std::uint32_t> cuts;       // layer_cut only: interior cuts, 0 = auto
    std::uint64_t rows = 0;
    std::vector<std::uint32_t> blocks;     // candidate rows per block, any order
    std::vector<std::uint32_t> decoders;   // devices allowed to own the state after the prefill
    std::function<bool(std::uint32_t)> legal_cut; // empty: every layer boundary is legal
    std::uint32_t max_stages = 0;          // 0: as many as devices
};
struct Plan {
    bool valid = false;
    std::uint32_t first = 0, last = 0, decoder = 0;
    std::vector<std::uint32_t> cuts;       // stages + 1 entries: 0, ..., layers
    std::uint32_t block = 0;
    std::uint64_t blocks = 0;
    double wall_ms = 0.0;                  // predicted prefill wall including the state move
    double makespan_ms = 0.0;              // flow shop only
    double fixed_state_ms = 0.0;           // end-of-prefill fixed state relay
    std::vector<double> stage_block_ms;    // per stage, one full block
    std::vector<double> hop_block_ms;      // per forward hop first..max(last,decoder)-1, one full block
    std::string bottleneck;                // slowest resource of a full block
    std::string reason;
    std::uint32_t stages() const { return static_cast<std::uint32_t>(cuts.empty() ? 0U : cuts.size() - 1U); }
    double rows_per_s(std::uint64_t rows) const { return wall_ms > 0.0 ? static_cast<double>(rows) * 1000.0 / wall_ms : 0.0; }
};
struct Choice {
    Plan best;
    std::vector<Plan> candidates;          // best plan of every (device run, decoder), valid or with its reject reason
};

namespace detail {
inline double range_sum(const std::vector<double>& values, std::uint32_t begin, std::uint32_t end) {
    double sum = 0.0;
    for (auto layer = begin; layer < end; ++layer) sum += values[layer];
    return sum;
}
inline double unit_factor(const std::vector<std::pair<std::uint32_t, double>>& scale, double rows) {
    if (scale.empty() || !(rows > 0.0)) return 1.0;
    if (rows <= scale.front().first) return scale.front().second;
    for (std::size_t i = 1; i < scale.size(); ++i) {
        if (rows <= scale[i].first) {
            const double a = std::log2(double(scale[i - 1].first)), b = std::log2(double(scale[i].first));
            const double t = b > a ? (std::log2(rows) - a) / (b - a) : 1.0;
            return scale[i - 1].second + t * (scale[i].second - scale[i - 1].second);
        }
    }
    return scale.back().second;
}
inline std::uint32_t owner(const std::vector<std::uint32_t>& cuts, std::uint32_t first, std::uint32_t layer) {
    std::uint32_t stage = 0;
    while (stage + 2U < cuts.size() && layer >= cuts[stage + 1U]) ++stage;
    return first + stage;
}
} // namespace detail

// Evaluates one fully specified plan (first, decoder, cuts, block); fills timing or the reject reason.
inline Plan evaluate(const std::vector<Device>& devices, const std::vector<Link>& links, const ModelShape& model,
    std::uint64_t rows, std::uint32_t first, std::uint32_t decoder, std::vector<std::uint32_t> cuts, std::uint32_t block) {
    Plan plan;
    plan.first = first;
    plan.decoder = decoder;
    plan.cuts = std::move(cuts);
    plan.block = block;
    const auto reject = [&](std::string why) { plan.valid = false; plan.reason = std::move(why); return plan; };
    if (plan.cuts.size() < 2U || rows == 0U || block == 0U) return reject("empty plan");
    if (devices.size() > std::numeric_limits<std::uint32_t>::max() || first >= devices.size() ||
        plan.cuts.size() - 1U > devices.size() - first || decoder >= devices.size() || links.size() < devices.size() - 1U)
        return reject("device or link outside the declaration");
    const auto stages = plan.stages();
    plan.last = first + stages - 1U;
    if (plan.cuts.front() != 0U || plan.cuts.back() != model.layers) return reject("cuts do not cover the model");
    for (std::uint32_t s = 0; s < stages; ++s)
        if (plan.cuts[s + 1U] > model.layers || plan.cuts[s] >= plan.cuts[s + 1U])
            return reject("cuts must be in bounds and strictly increasing");
    const auto has = [&](const std::vector<double>& values) { return values.size() == model.layers; };
    if (!has(model.state_bytes_per_row) || !has(model.state_fixed_bytes)) return reject("model state shape missing");
    for (std::uint32_t s = 0; s < stages; ++s) {
        const auto& device = devices[first + s];
        if (device.layer_ms_per_row.size() != model.layers) return reject("no layer costs for " + device.name);
        if (!model.weight_bytes.empty()) {
            if (model.weight_bytes.size() != model.layers) return reject("model weight shape mismatch");
            const auto need = detail::range_sum(model.weight_bytes, plan.cuts[s], plan.cuts[s + 1U]);
            if (!(device.capacity_bytes > 0.0)) return reject("capacity undeclared for " + device.name);
            if (need > device.capacity_bytes) return reject("weights do not fit " + device.name);
        }
    }
    // Forward resources: stage first, hop first, ..., stage last, then state-only hops up to the decoder.
    const auto end = std::max(plan.last, decoder);
    std::vector<double> forward_state(devices.size(), 0.0), reverse_state(devices.size(), 0.0);
    std::vector<double> forward_fixed(devices.size(), 0.0), reverse_fixed(devices.size(), 0.0);
    for (std::uint32_t layer = 0; layer < model.layers; ++layer) {
        const auto own = detail::owner(plan.cuts, first, layer);
        for (auto hop = own; hop < decoder; ++hop) { forward_state[hop] += model.state_bytes_per_row[layer]; forward_fixed[hop] += model.state_fixed_bytes[layer]; }
        for (auto hop = decoder; hop < own; ++hop) { reverse_state[hop] += model.state_bytes_per_row[layer]; reverse_fixed[hop] += model.state_fixed_bytes[layer]; }
    }
    struct Resource { double fixed = 0.0, per_row = 0.0; std::string name; const std::vector<std::pair<std::uint32_t, double>>* scale = nullptr; };
    const auto row_cost = [](const Resource& r, double n) { return n * r.per_row * (r.scale ? detail::unit_factor(*r.scale, n) : 1.0); };
    std::vector<Resource> resources;
    for (auto device = first; device <= end; ++device) {
        if (device <= plan.last) {
            const auto s = device - first;
            const auto& d = devices[device];
            auto per_row = detail::range_sum(d.layer_ms_per_row, plan.cuts[s], plan.cuts[s + 1U]);
            if (device == plan.last) per_row += d.tail_ms_per_row;
            resources.push_back({d.block_fixed_ms, per_row, "stage " + d.name, &d.unit_scale});
        }
        if (device < end) {
            const auto bytes = (device < plan.last ? model.seam_bytes_per_row : 0.0) + forward_state[device];
            const auto& link = links[device];
            if (bytes > 0.0 && !(link.forward_bytes_per_ms > 0.0))
                return reject("no forward link " + devices[device].name + "->" + devices[device + 1U].name);
            resources.push_back({link.latency_ms, bytes > 0.0 ? bytes / link.forward_bytes_per_ms : 0.0,
                "link " + devices[device].name + "->" + devices[device + 1U].name});
        }
    }
    plan.blocks = rows / block + (rows % block != 0U);
    const auto reverse_hops = plan.last > decoder ? plan.last - decoder : 0U;
    const auto work_per_block = resources.size() + reverse_hops;
    if (plan.blocks > max_plan_work / work_per_block) return reject("plan work limit exceeded");
    for (auto hop = decoder; hop < plan.last; ++hop)
        if ((reverse_state[hop] > 0.0 || reverse_fixed[hop] > 0.0) && !(links[hop].reverse_bytes_per_ms > 0.0))
            return reject("no reverse link " + devices[hop + 1U].name + "->" + devices[hop].name);
    std::vector<double> done(resources.size(), 0.0);
    std::vector<double> reverse_done(devices.size(), 0.0);
    double reverse_ms = 0.0;
    for (std::uint64_t j = 0; j < plan.blocks; ++j) {
        const auto block_rows = static_cast<double>(j + 1U == plan.blocks ? rows - j * block : block);
        double previous = 0.0;
        for (std::size_t r = 0; r < resources.size(); ++r) {
            done[r] = std::max(previous, done[r]) + resources[r].fixed + row_cost(resources[r], block_rows);
            previous = done[r];
        }
        double relay = 0.0;
        for (auto device = plan.last; device > decoder; --device) {
            const auto hop = device - 1U;
            if (reverse_state[hop] == 0.0) continue;
            // A stage's completion is at resource 2*s; devices before first only relay.
            const double produced = device >= first ? done[2U * (device - first)] : 0.0;
            reverse_done[hop] = std::max({reverse_done[hop], relay, produced}) + links[hop].latency_ms +
                block_rows * reverse_state[hop] / links[hop].reverse_bytes_per_ms;
            relay = reverse_done[hop];
        }
        reverse_ms = std::max(reverse_ms, relay);
    }
    plan.makespan_ms = done.back();
    double best_block = -1.0;
    for (const auto& resource : resources) {
        const auto t = resource.fixed + row_cost(resource, static_cast<double>(block));
        (resource.name.starts_with("stage") ? plan.stage_block_ms : plan.hop_block_ms).push_back(t);
        if (t > best_block) { best_block = t; plan.bottleneck = resource.name; }
    }
    double fixed_ms = 0.0;
    for (std::uint32_t hop = 0; hop + 1U < devices.size(); ++hop) {
        if (reverse_fixed[hop] > 0.0) {
            if (!(links[hop].reverse_bytes_per_ms > 0.0))
                return reject("no reverse link " + devices[hop + 1U].name + "->" + devices[hop].name);
            fixed_ms += reverse_fixed[hop] / links[hop].reverse_bytes_per_ms + links[hop].latency_ms;
        }
        if (forward_fixed[hop] > 0.0) {
            if (!(links[hop].forward_bytes_per_ms > 0.0))
                return reject("no forward link " + devices[hop].name + "->" + devices[hop + 1U].name);
            fixed_ms += forward_fixed[hop] / links[hop].forward_bytes_per_ms + links[hop].latency_ms;
        }
    }
    plan.fixed_state_ms = fixed_ms;
    plan.wall_ms = std::max(plan.makespan_ms, reverse_ms) + fixed_ms;
    plan.valid = std::isfinite(plan.wall_ms);
    plan.reason = plan.valid ? "bottleneck " + plan.bottleneck : "non-finite cost";
    return plan;
}

// Best plan over every device run, decoder, legal cut set and block candidate. Ties keep fewer stages, then the
// earlier device run, then the smaller block, so an equal split never displaces the one-device plan.
inline Choice choose(const std::vector<Device>& devices, const std::vector<Link>& links, const ModelShape& model,
    const Request& request) {
    if (devices.empty() || model.layers == 0U || request.rows == 0U || request.blocks.empty() || request.decoders.empty())
        throw std::invalid_argument("chain plan needs devices, a model, rows, blocks and a decoder");
    auto blocks = request.blocks;
    std::sort(blocks.begin(), blocks.end());
    const auto legal = [&](std::uint32_t cut) { return !request.legal_cut || request.legal_cut(cut); };
    const auto cap = request.max_stages == 0U ? static_cast<std::uint32_t>(devices.size()) : request.max_stages;
    Choice choice;
    std::uint64_t work_left = max_search_work;
    const auto charge = [&](std::uint64_t work) {
        if (work > work_left) throw std::invalid_argument("chain search work limit exceeded");
        work_left -= work;
    };
    const auto better = [](const Plan& a, const Plan& b) {
        if (!b.valid) return a.valid;
        if (!a.valid) return false;
        return a.wall_ms < b.wall_ms * (1.0 - 1e-12);
    };
    if (request.split == Split::layer_cut && (request.cuts.empty() || request.cuts.size() + 1U > devices.size()))
        throw std::invalid_argument("layer-cut split needs 1 .. devices-1 interior cuts");
    for (const auto cut : request.cuts)
        if (cut >= model.layers || (cut != 0U && !legal(cut))) throw std::invalid_argument("forced cut is not a legal layer cut");
    const auto wanted = [&](std::uint32_t count, std::uint32_t first, std::uint32_t decoder) {
        switch (request.split) {
        case Split::device_only: return count == 1U && decoder == first;
        case Split::handoff: return count == 1U && decoder != first;
        case Split::layer_cut: return count == request.cuts.size() + 1U;
        default: return true;
        }
    };
    for (std::uint32_t count = 1; count <= std::min<std::uint32_t>(cap, static_cast<std::uint32_t>(devices.size())); ++count) {
        if (count > model.layers) break;
        for (std::uint32_t first = 0; first + count <= devices.size(); ++first) {
            for (const auto decoder : request.decoders) {
                charge(1U);
                if (!wanted(count, first, decoder)) continue;
                Plan run_best;
                run_best.first = first; run_best.last = first + count - 1U; run_best.decoder = decoder;
                run_best.reason = "no legal cut set";
                std::vector<std::uint32_t> cuts(count + 1U, 0U);
                cuts.back() = model.layers;
                const std::function<void(std::uint32_t)> place = [&](std::uint32_t index) {
                    if (index == count) {
                        for (const auto block : blocks) {
                            charge(1U);
                            auto plan = evaluate(devices, links, model, request.rows, first, decoder, cuts, block);
                            if (!plan.stage_block_ms.empty()) charge(plan.blocks * (plan.stage_block_ms.size() + plan.hop_block_ms.size() +
                                (plan.last > decoder ? plan.last - decoder : 0U)));
                            if (better(plan, run_best) || (!run_best.valid && !plan.valid && run_best.reason == "no legal cut set"))
                                run_best = std::move(plan);
                        }
                        return;
                    }
                    const auto forced = request.split == Split::layer_cut ? request.cuts[index - 1U] : 0U;
                    for (auto cut = cuts[index - 1U] + 1U; cut + (count - index) <= model.layers; ++cut) {
                        charge(1U);
                        if (!legal(cut) || (forced != 0U && cut != forced)) continue;
                        cuts[index] = cut;
                        place(index + 1U);
                    }
                };
                place(1U);
                if (better(run_best, choice.best) || (choice.candidates.empty() && !choice.best.valid)) choice.best = run_best;
                choice.candidates.push_back(std::move(run_best));
            }
        }
    }
    if (choice.candidates.empty()) choice.best.reason = std::string("no device run matches split ") + split_name(request.split);
    if (choice.best.valid) choice.best.reason = std::string("chosen (split ") + split_name(request.split) + "): " + choice.best.reason;
    return choice;
}

// Uniform per-layer costs for a device whose receipts give only a whole-model rate (rows/s): a search seed,
// never a qualified cost. Shape follows `reference` (another device's measured per-layer vector) when given.
inline std::vector<double> scaled_costs(double rows_per_s, std::uint32_t layers, std::span<const double> reference = {}) {
    if (!std::isfinite(rows_per_s) || !(rows_per_s > 0.0) || layers == 0U) throw std::invalid_argument("scaled costs need a finite rate and layers");
    const double total_ms = 1000.0 / rows_per_s;
    std::vector<double> costs(layers, total_ms / layers);
    if (reference.size() == layers) {
        double sum = 0.0;
        for (const auto value : reference) sum += value;
        if (sum > 0.0) for (std::uint32_t l = 0; l < layers; ++l) costs[l] = total_ms * reference[l] / sum;
    }
    return costs;
}
} // namespace strix::pair::chain
