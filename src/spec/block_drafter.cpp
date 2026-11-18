#include "strix/spec/block_drafter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace strix::spec {

namespace {
constexpr std::uint32_t chain_steps_cap = 96;  // table size: copy prefix + extension steps
}

BlockDrafter::BlockDrafter(BlockConfig config, ChainDrafter* chain)
    : config_(std::move(config)), chain_(chain), proposer_(config_.proposer), arbiter_(config_.arbiter),
      chain_ms_(chain_steps_cap + 1, 0.0), chain_w_(chain_steps_cap + 1, 0.0), step_ms_(config_.chain_seed_step_ms) {
    if (config_.max_block < 1 || config_.max_block > config_.arbiter.windows.back())
        throw std::invalid_argument("block drafter: max_block must be within the arbiter windows");
    if (config_.chain_max_k < 1 || config_.chain_max_k > ext_.size())
        throw std::invalid_argument("block drafter: chain_max_k in 1..8");
    if (!config_.clock_ms) config_.clock_ms = [] {
        using clock = std::chrono::steady_clock;
        return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
    };
}

double BlockDrafter::now_ms() const { return config_.clock_ms(); }

double BlockDrafter::chain_ms(std::uint32_t steps) const {
    steps = std::clamp<std::uint32_t>(steps, 1, chain_steps_cap);
    if (chain_w_[steps] > 0.0) return chain_ms_[steps];
    for (std::uint32_t j = steps - 1; j >= 1; --j)
        if (chain_w_[j] > 0.0)  // no measured slope yet: proportional to the nearest measured depth
            return step_learned_ ? chain_ms_[j] + step_ms_ * double(steps - j) : chain_ms_[j] * double(steps) / double(j);
    for (std::uint32_t j = steps + 1; j <= chain_steps_cap; ++j)
        if (chain_w_[j] > 0.0)
            return step_learned_ ? std::max(step_ms_, chain_ms_[j] - step_ms_ * double(j - steps)) : chain_ms_[j] * double(steps) / double(j);
    return config_.chain_seed_flush_ms + step_ms_ * double(steps - 1);
}

void BlockDrafter::learn_chain_ms(std::uint32_t steps, double ms) {
    steps = std::clamp<std::uint32_t>(steps, 1, chain_steps_cap);
    if (!(ms > 0.0)) return;
    // marginal step cost from the nearest other measured depth
    for (std::uint32_t d = 1; d < chain_steps_cap; ++d) {
        const std::uint32_t lo = steps > d ? steps - d : 0, hi = steps + d;
        std::uint32_t o = 0;
        if (lo >= 1 && chain_w_[lo] > 0.0) o = lo;
        else if (hi <= chain_steps_cap && chain_w_[hi] > 0.0) o = hi;
        if (o) {
            const double s = (ms - chain_ms_[o]) / (double(steps) - double(o));
            if (s > 0.01) {
                step_ms_ = step_learned_ ? 0.5 * step_ms_ + 0.5 * s : s;
                step_learned_ = true;
            }
            break;
        }
        if (lo < 1 && hi > chain_steps_cap) break;
    }
    auto& w = chain_w_[steps];
    chain_ms_[steps] = w > 0.0 ? (chain_ms_[steps] * w + ms) / (w + 1.0) : ms;
    w = std::min(w + 1.0, 8.0);
}

Block BlockDrafter::next(const StepParams& params) {
    Block b;
    if (params.remaining == 0 || params.max_rows == 0) {
        b.decision = Decision{Source::serial, 0, 0, false, 0.0, 0.0, false};
        return b;
    }
    const std::uint32_t max_rows = std::min(params.max_rows, config_.max_block);
    if (config_.copy) proposer_.propose(proposal_, {}, config_.max_block - 1);
    else proposal_.tokens.clear();
    StepInput in;
    in.lookup = config_.copy ? &proposal_ : nullptr;
    const bool use_chain = chain_ && config_.chain;
    in.model_max_k = use_chain ? std::min(config_.chain_max_k, config_.max_block - 1) : 0;
    if (in.model_max_k > 0) {
        model_ms_.resize(in.model_max_k);
        for (std::uint32_t k = 1; k <= in.model_max_k; ++k) model_ms_[k - 1] = chain_ms(k);
        in.model_draft_ms = model_ms_;
    }
    in.remaining = params.remaining;
    in.max_rows = max_rows;
    in.context = params.context;
    b.decision = arbiter_.decide(in);
    const auto& d = b.decision;
    b.source = d.source;
    if (d.source == Source::serial || d.k == 0) {
        b.source = Source::serial;
        return b;
    }
    if (d.source == Source::lookup) {
        b.drafts.assign(proposal_.tokens.begin(), proposal_.tokens.begin() + std::ptrdiff_t(d.k));
        b.copy = d.k;
        b.bucket = std::min(proposal_.bucket(), lookup_buckets - 1);
        // Hybrid: the copy proposal ended below the window; extend it with the chain when that pays.
        const std::uint32_t n = d.k;
        const std::uint32_t limit = params.remaining - 1;
        const std::uint32_t by_block = config_.max_block - 1 > n ? config_.max_block - 1 - n : 0U;
        const std::uint32_t room = std::min({config_.chain_max_k, by_block, limit > n ? limit - n : 0U,
                                             max_rows - 1 > n ? max_rows - 1 - n : 0U});
        if (config_.hybrid && use_chain && n == proposal_.tokens.size() && room > 0) {
            double full = 1.0;
            for (std::uint32_t j = 1; j <= n; ++j) full *= arbiter_.survival(Source::lookup, b.bucket, j);
            const double base = d.expected_tokens / std::max(d.expected_ms, 1e-9);
            double best = base * config_.hybrid_switch_in, surv = 1.0, gain = 0.0;
            std::uint32_t pick = 0;
            const double v0 = arbiter_.verify_ms(n + 1, params.context);
            for (std::uint32_t m = 1; m <= room; ++m) {
                const auto& e = ext_[m - 1];
                const double s = arbiter_.survival(Source::model, 0, m);
                surv *= (e[0] + 2.0 * s) / (e[0] + e[1] + 2.0);
                gain += full * surv;
                const double ms = d.expected_ms + (arbiter_.verify_ms(n + m + 1, params.context) - v0) + chain_ms(n + m);
                const double score = (d.expected_tokens + gain) / std::max(ms, 1e-9);
                if (score > best) {
                    best = score;
                    pick = m;
                }
            }
            if (pick > 0) {
                const double t0 = now_ms();
                std::vector<std::uint32_t> ext;
                chain_->draft(std::span<const std::uint32_t>(b.drafts.data(), n), pick, ext);
                b.draft_ms = now_ms() - t0;
                learn_chain_ms(n + pick, b.draft_ms);
                b.ext = std::uint32_t(std::min<std::size_t>(ext.size(), pick));
                b.drafts.insert(b.drafts.end(), ext.begin(), ext.begin() + std::ptrdiff_t(b.ext));
            }
        }
    } else {
        const double t0 = now_ms();
        chain_->draft({}, d.k, b.drafts);
        b.draft_ms = now_ms() - t0;
        learn_chain_ms(d.k, b.draft_ms);
        if (b.drafts.size() > d.k) b.drafts.resize(d.k);
        b.ext = std::uint32_t(b.drafts.size());
        if (b.drafts.empty()) b.source = Source::serial;
    }
    return b;
}

void BlockDrafter::observe(const Block& b, std::uint32_t accepted, double verify_ms,
                           std::span<const std::uint32_t> emitted, std::uint64_t context) {
    Outcome o;
    o.source = b.source;
    o.bucket = b.bucket;
    o.proposed = b.source == Source::lookup ? b.copy : std::uint32_t(b.drafts.size());
    o.accepted = std::min(accepted, o.proposed);
    o.rows = b.rows();
    o.verify_ms = verify_ms;
    o.draft_ms = b.source == Source::model ? b.draft_ms : 0.0;
    o.context = context;
    if (b.hybrid()) {
        ++hybrid_blocks;
        hybrid_ext_proposed += b.ext;
        if (accepted >= b.copy)
            for (std::uint32_t j = 1; j <= b.ext && j <= ext_.size(); ++j) {
                if (accepted >= b.copy + j) {
                    ext_[j - 1][0] += 1.0;
                    ++hybrid_ext_accepted;
                } else {
                    ext_[j - 1][1] += 1.0;
                    break;
                }
            }
    }
    stats_.record(o, std::uint32_t(emitted.size()));
    arbiter_.observe(o, emitted);
    proposer_.commit(emitted);
    if (chain_) chain_->accept(emitted);
}

StepRecord BlockSpeculator::step(std::vector<std::uint32_t>& out, const StepParams& params) {
    StepRecord r;
    auto block = drafter_.next(params);
    if (block.decision.rows == 0) return r;
    rows_.assign(1, anchor_);
    rows_.insert(rows_.end(), block.drafts.begin(), block.drafts.end());
    const double t0 = drafter_.now_ms();
    target_.verify(rows_, greedy_);
    if (greedy_.size() != rows_.size()) throw std::runtime_error("block speculator: target returned the wrong row count");
    std::uint32_t accepted = 0;
    while (accepted < block.drafts.size() && block.drafts[accepted] == greedy_[accepted]) ++accepted;
    target_.commit(accepted + 1);
    const double verify_ms = drafter_.now_ms() - t0;
    std::vector<std::uint32_t> emitted(block.drafts.begin(), block.drafts.begin() + accepted);
    emitted.push_back(greedy_[accepted]);
    drafter_.observe(block, accepted, verify_ms, emitted, params.context);
    out.insert(out.end(), emitted.begin(), emitted.end());
    anchor_ = emitted.back();
    r.source = block.source;
    r.rows = block.rows();
    r.proposed = std::uint32_t(block.drafts.size());
    r.accepted = accepted;
    r.copy = block.copy;
    r.ext = block.ext;
    r.emitted = std::uint32_t(emitted.size());
    r.verify_ms = verify_ms;
    r.draft_ms = block.draft_ms;
    return r;
}

}  // namespace strix::spec
