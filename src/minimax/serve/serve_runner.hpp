#pragma once
// Native command execution only. Scheduling and publication belong to Lead's shared runtime.
#include "minimax/serve/minimax_lead.hpp"
#include "strix/minimax/engine.hpp"
#include "minimax/dspark/hip_minimax_dspark.hpp"
#include "minimax/session/runtime.hpp"
#include "strix/env_flags.hpp"
#include "strix/pair/lockstep.hpp"
#include <algorithm>
#include <cstdlib>
namespace strix::minimax::serve::native {
namespace mm = strix::minimax::engine;
namespace sv = strix::serve;
namespace ms = strix::minimax::serve;
namespace ds = strix::minimax::dspark;
// One validating reader for every numeric switch of the serve path.
inline double env_num(const char* name, double def) { return strix::env_number<double>(name, def); }
using ms::lead::Drafts;
void exchange_tokens(mm::Engine&, std::vector<std::uint32_t>&, std::size_t, bool);
struct Runner {
    void send_prompt(std::vector<std::uint32_t>& prompt) { exchange_tokens(eng, prompt, prompt.size(), true); }
    static void synchronize() { ds::DeviceDrafter::device_synchronize(); }
    mm::Engine& eng;
    strix::dist::Communicator& comm;
    void quiesce() { comm.release_transport(nullptr); }
    ds::DeviceDrafter* dd = nullptr;  // DSpark drafter (both ranks), nullptr = none
    std::uint32_t v0 = 0, vn = 0;     // this rank's vocabulary half (drafter Markov steps)

    bool prefix_reuse = false;
    bool conversation = false;  // both ranks voted: resident prefills may restart mid-chunk at a reply-end frontier
    // Validation counters only; the sole token/LCP/eviction index is PrefixCache.
    struct ResidentFrontier { std::uint64_t rows = 0, generation = 0, total = 0; };
    std::vector<ResidentFrontier> resident_frontiers;
    void invalidate_residents() noexcept { for (auto& f : resident_frontiers) f.rows = 0; }
    std::uint64_t local_id = 0, control_round = 0;
    std::uint32_t last_next = 0, contexts = 0;
    // STRIX_MM_SERVE_DSPARK_STAGE=0 (or STRIX_MM_BASELINE=1 when unset) appends every cycle's rows at once (previous form).
    const bool stage = std::getenv("STRIX_MM_SERVE_DSPARK_STAGE") != nullptr ? env_num("STRIX_MM_SERVE_DSPARK_STAGE", 1) != 0
                                                                     : env_num("STRIX_MM_BASELINE", 0) != 1;
    // Typed operator/library activation only; ordinary service construction is OFF.
    std::shared_ptr<ms::paired::Endpoint> state_endpoint;
    std::shared_ptr<ms::paired::Owner> state_owner;
    void state(ms::SessionPlan& p) {
        invalidate_residents();
        if (!state_owner) throw strix::session::UnsupportedOperation("MiniMax paired state OFF");
        if (p.kind == ms::SessionPlan::Kind::capture) p.state=state_owner->capture(p.frontier);
        else {
            const auto checkpoint=state_owner->restore(p.frontier.handle,p.rank,p.state.at(p.rank));
            p.frontier.progress=checkpoint.first;p.frontier.progress_limit=checkpoint.second;
            if(state_endpoint->identity().schema==2){const auto& f=state_endpoint->frontier(p.frontier.handle);
                p.frontier.retained=f.retained;p.frontier.finish=f.finish;}
        }
    }
    bool input_supported(ms::paired::InputProfile profile) const {
        return state_endpoint && (eng.rank()!=0 || state_owner) && !dd && !contexts &&
               !eng.prefill_open() && state_endpoint->input_supported(profile);
    }
    ms::paired::Identity input_identity() const {
        if(!input_supported(ms::paired::InputProfile::plain_greedy_r1))
            throw strix::session::UnsupportedOperation("MiniMax native input identity OFF");
        return state_endpoint->identity();
    }
    // Optional operator witness wraps real writes/readback only. Never installed
    // by service startup and never changes arithmetic or selection.
    std::function<void(std::span<const mm::Row>,std::span<const std::uint32_t>)> input_witness;
    std::vector<std::uint32_t> input(std::span<const mm::Row> rows,
                                    std::span<const strix::session::KvHandle> handles,bool follower=false) {
        invalidate_residents();
        if(!input_supported(ms::paired::InputProfile::plain_greedy_r1))
            throw strix::session::UnsupportedOperation("MiniMax native input OFF/profile");
        std::vector<std::uint32_t> next;
        const auto ok=state_endpoint->input(rows,handles,[&] {
            auto result=eng.step(rows);synchronize();
            if(input_witness)input_witness(rows,result);
            return result;
        },next);
        if(!ok && !follower)throw std::runtime_error("MiniMax paired input failed; full rollback required");
        return next;
    }
    void input_control(const Control& c) {
        std::array<strix::session::KvHandle,8> handles{};
        for(std::size_t i=0;i<c.rows;++i)handles[i]={c.row[i].slot,c.dpos[i],c.parent_task_id};
        (void)input(std::span(c.row,c.rows),std::span(handles.data(),c.rows),true);
    }
    void retire_contexts(std::uint32_t mask) {
        contexts &= ~mask;
        if (dd != nullptr) for (std::uint32_t s = 0; s < eng.slots(); ++s)
            if ((mask >> s) & 1U) dd->drop_staged(s);
    }
    void prefill_begin(std::uint32_t slot, std::span<const std::uint32_t> prompt, bool context,
                       std::uint64_t id, std::uint64_t parent = 0, std::uint64_t generation = 0, std::uint64_t resident = 0) {
        if (prefix_reuse) {
            if (resident_frontiers.empty()) resident_frontiers.resize(eng.slots());
            auto& f = resident_frontiers.at(slot);
            if (resident && ((dd && context) || parent || state_endpoint || !generation || generation <= f.generation ||
                resident > f.rows || (!conversation && resident % eng.prefill_reuse_rows())))
                throw std::runtime_error("MiniMax stale/unqualified resident frontier");
            f.rows = resident; f.generation = generation; f.total = resident + prompt.size();
        } else if (resident) throw std::runtime_error("MiniMax resident vote OFF");
        if (state_endpoint) {
            if (dd || context || parent || eng.prefill_open())
                throw strix::session::UnsupportedOperation("MiniMax state plain quiescent profile required");
            state_endpoint->begin(slot,generation,prompt);
        }
        contexts = (contexts & ~(1U << slot)) | (context ? 1U << slot : 0U);
        if (dd != nullptr) dd->drop_staged(slot);  // a new sequence: the previous one's staged rows are never read
        mm::Engine::PrefillTapSink sink;
        if (dd != nullptr && context)
            sink = [this](const float* taps, std::size_t n, std::uint32_t s, std::uint64_t pos0) { if ((contexts >> s) & 1U) dd->append(s, taps, n, pos0, nullptr); };
        if (id == 0) throw std::runtime_error("prefill begin: missing task ID");
        eng.prefill_begin_task(id, parent, generation, slot, prompt, resident, std::move(sink));
    }
    void prefill_check(std::uint64_t id, std::uint32_t next) const {
        const auto p = eng.prefill_progress();
        const auto result = eng.prefill_result();
        if (p.open || p.id != id || result.id != id || result.next != next || last_next != next) {
            eng.abort("control channel: completion mismatch");
            throw std::runtime_error("control channel: completion identity / token mismatch");
        }
    }
    void require_active(std::uint64_t id, unsigned bank) const {
        if (!eng.prefill_active() || eng.prefill_progress().id != id || eng.prefill_progress().bank != bank)
            throw std::runtime_error("control channel: wrong active task");
    }
    bool prefill_unit(unsigned layers, std::uint32_t* next) try {
        const auto slot = eng.prefill_progress().slot;
        const bool done = eng.prefill_resume(ms::resume_layers(layers), &last_next);
        if (done && prefix_reuse) {
            auto& f = resident_frontiers.at(slot);
            f.rows = conversation ? f.total : f.total - f.total % eng.prefill_reuse_rows();
        }
        if (done && next) *next = last_next;
        if (done && state_endpoint) state_endpoint->selected(eng.prefill_progress().slot,last_next);
        return done;
    } catch (...) { invalidate_residents(); throw; }
    void prefill_cancel(std::uint64_t id) {
        if (id == 0) throw std::runtime_error("prefill cancel: missing task ID");
        eng.prefill_cancel(id);
    }
    void cancel_all() {
        if (eng.prefill_active()) prefill_cancel(eng.prefill_progress().id);
        if (eng.prefill_progress(true).open) prefill_cancel(eng.prefill_progress(true).id);
    }
    // DSpark block + Markov chain (one 8-byte candidate exchange per step) + confidences, as strix-minimax-rank --spec auto.
    Drafts draft(std::uint32_t slot, std::uint32_t anchor, std::uint64_t pos) {
        if (slot >= eng.slots() || ((contexts >> slot) & 1U) == 0) throw std::runtime_error("draft: retired drafter context");
        Drafts out;
        dd->forward_block(slot, eng.embed_row(anchor), eng.embed_row(ds::noise_token), pos, nullptr);
        const float* logits = eng.head_logits(dd->y(), ds::block);
        std::array<std::uint32_t, ds::block> prev{};
        std::uint32_t t = anchor;
        for (std::uint32_t i = 0; i < ds::block; ++i) {
            prev[i] = t;
            const ds::Candidate mine = dd->markov_step_host(t, logits + static_cast<std::size_t>(i) * vn, v0, v0 + vn, nullptr);
            const auto theirs = eng.peer2(__builtin_bit_cast(std::uint32_t, mine.value), mine.id);
            t = ds::combine(mine, ds::Candidate{__builtin_bit_cast(float, theirs[0]), theirs[1]}).id;
            out.tokens[i] = t;
        }
        out.confidence = dd->confidence_host(prev, nullptr);
        return out;
    }
    // One step; then each slot of `append` gets its committed rows appended (its anchor + the drafts the target accepted:
    // the window's contiguous tap rows, before the next call overwrites them). The rows are staged and appended in one
    // batch by the slot's next block pass or after 32 rows (measurement: with the gate mostly shut every 1-row append
    // cost ~3 ms per cycle; batching changes only the drafter context's bits, the emitted tokens are target-greedy).
    std::vector<std::uint32_t> step(std::span<const mm::Row> rows, std::uint32_t append) {
        if (state_endpoint) {
            if (dd || append || eng.prefill_open())
                throw strix::session::UnsupportedOperation("MiniMax state plain quiescent rows required");
            state_endpoint->reserve_step(rows);
        }
        if ((append & ~contexts) != 0) throw std::runtime_error("step: retired drafter context");
        auto g = eng.step(rows);
        if (conversation && prefix_reuse && !resident_frontiers.empty())
            for (const auto& row : rows)  // upper bound of the rows decode wrote; the lead's store holds the exact frontier
                if (row.slot < resident_frontiers.size() && resident_frontiers[row.slot].rows)
                    resident_frontiers[row.slot].rows = std::max<std::uint64_t>(resident_frontiers[row.slot].rows, row.pos + 1);
        if (dd != nullptr && append != 0) {
            const float* taps = eng.last_taps();
            for (std::size_t i0 = 0; i0 < rows.size();) {
                std::size_t i1 = i0 + 1;
                while (i1 < rows.size() && rows[i1].slot == rows[i0].slot) ++i1;
                if ((append >> rows[i0].slot) & 1U) {
                    std::vector<std::uint32_t> d;
                    for (std::size_t i = i0 + 1; i < i1; ++i) d.push_back(rows[i].token);
                    const auto v = ds::verify_greedy(d, std::span<const std::uint32_t>(g.data() + i0, i1 - i0));
                    if (stage) dd->append_deferred(rows[i0].slot, taps + i0 * ds::taps * 6144, v.accepted + 1, rows[i0].pos, nullptr);
                    else dd->append(rows[i0].slot, taps + i0 * ds::taps * 6144, v.accepted + 1, rows[i0].pos, nullptr);
                }
                i0 = i1;
            }
        }
        if (state_endpoint) state_endpoint->stepped(rows,g);
        return g;
    }
};

void follow(mm::Engine&, Runner&, strix::minimax::session::Runtime*);
int resume_check(mm::Engine&, Runner&, const std::string&, unsigned, const std::string&, unsigned);
// Both-rank explicit operator probe. Call instead of starting Lead/HTTP/named
// activity. Identity must be made from the existing compatibility() helper's
// loaded assets/executable/effective-profile record; never from a model label.
int paired_state_probe(mm::Engine&, Runner&, const mm::Options&, const std::string& executable, std::uint64_t budget,
                       std::span<const std::uint32_t> prompt, std::string_view ticket = "state");
} // namespace strix::minimax::serve::native
