#pragma once
#include "strix/hip_check.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/prompt_terminal_diagnostics.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/hip_known_prefix_window.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_mtp_greedy_runtime.hpp"
#include "strix/hip_mtp_target_hidden.hpp"
#include "strix/hip_mtp_window_verifier.hpp"
#include "strix/host_phase_timing.hpp"
#include "strix/serial_request.hpp"
#include "strix/session_state_memory.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::decode_cli {
// Wide copy verify (opt-in, --mtp-copy-wide-rows 32[,16] with --mtp-ngram-draft=copy on the tiled MTP prompt; each
// listed size has its own tiled tail owner, tried largest first, sharing the snapshot and hidden-row buffers).
//
// A cycle whose context drafter has a long, confident, unanimous copy span (HipMtpGreedyRuntime::propose_wide)
// verifies its rows-1 tokens in ONE rows-row forward of the tiled prefill tail owner of that many rows (all-rows
// heads, the decode catalog: no split prefill profile) instead of the 8-row verify window:
//  1. the verify owners' pending deferred GDN rows are applied and their QSA pools invalidated
//     (HipMtpWindowVerifier::flush_pending_window), then the in-place session state is snapshotted: every state
//     component except the position-indexed attention KV and sparse index (cells at or after the committed
//     frontier are never authoritative), device components device to device, host words by copy;
//  2. the tiled owner runs inputs [pending, p1..p_{rows-1}] at the frontier; a step observer copies every row's
//     wide hidden state (the MTP target tap, as the tiled prompt does) and each row's full FP32 logits row is
//     selected on the device (enqueue_hip_greedy_rows, 8 rows per launch: the same finite first-max rule with
//     lowest-token ties and equal zeros); a row with a non-finite logit falls back to the host path, every row
//     downloaded in order and selected with select_greedy_token (the finite first-max rule of every target row
//     and of adopt_prefill_row) until the first row whose selection differs from the next input (or is EOS);
//  3. all rows accepted: rows 0..rows-1 stay committed, the last row is adopted as the pending ticket and the
//     greedy runtime publishes the cycle (finish_wide_cycle: predictor teacher rows from the tapped hidden rows,
//     the context drafter's finish_cycle);
//  4. otherwise (k accepted rows, or an EOS before the last row) the wide cycle is ABANDONED: the snapshot and
//     the host session are restored (the pre-cycle committed frontier exactly), the owners' pools invalidated,
//     and the greedy runtime returns to its pre-proposal state (abandon_wide_cycle: nothing committed, the
//     predictor untouched, the drafter scores the proposal against the target's rows 0..k). cycle() returns
//     nullopt and the caller runs the ordinary cycle, whose copy-first windows re-cover the accepted prefix. No
//     wide cycle is proposed again until the committed frontier passes the rejected row.
// The tiled rows were measured to give full-vocabulary logits bit-identical to serial decode rows at the decode
// frontier (tests/hip_mtp_wide_cost.hpp); STRIX_MTP_WIDE_GATE=1 (tests/hip_mtp_wide_gate.hpp) checks this path
// against the serial oracle. Single requests only (not cohort/batched routes).
class MtpWideCopyCycle {
public:
    // cycles/rows_accepted: committed wide cycles; abandoned: rejected wide forwards, whose rows were all computed
    // and none committed (abandoned_rows_wasted); host wall ms of committed and of abandoned wide cycles.
    // abandon_rows: per abandoned forward, the rows accepted before the rejected one (n - 1 of the emitted rows).
    struct Stats {std::uint64_t cycles=0,rows_accepted=0,abandoned=0,abandoned_rows_wasted=0;double ms_committed=0.0,ms_abandoned=0.0;
        std::vector<std::uint32_t> abandon_rows;};
    // One configured wide size: its tiled owner and its own counters.
    struct Size {std::shared_ptr<HipKnownPrefixWindow> owner;std::uint32_t rows=0;Stats stats;};
    // Diagnostics: called with every downloaded row's index and full logits (rows 0..k, in order).
    using RowLogits=std::function<void(std::uint32_t,std::span<const float>)>;
    // owners: one tiled all-rows owner per wide size, strictly descending rows. They share one state snapshot and
    // one hidden-row buffer sized for the largest.
    MtpWideCopyCycle(std::vector<std::shared_ptr<HipKnownPrefixWindow>> owners,HipMtpTargetHidden& target,
        const ExecutionRuntimeResources& resources)
        :target_(target),resources_(resources) {
        if(owners.empty() || !resources_.session_state_layout || !resources_.session_state_memory)
            throw std::invalid_argument("MTP wide copy verify requires its tiled owners and the session state");
        for(auto& owner:owners) {
            if(!owner || !owner->supports_tiled_prefill() || owner->output_policy()!=KnownWindowOutputPolicy::all_rows ||
               owner->logical_rows()<2U || (!sizes_.empty() && owner->logical_rows()>=sizes_.back().rows))
                throw std::invalid_argument("MTP wide copy verify requires tiled all-rows owners of strictly descending rows");
            require_rotary_producer_identity(owner->rotary_identity(),target_.model().rotary_identity());
            Size size;size.rows=owner->logical_rows();size.owner=std::move(owner);
            sizes_.push_back(std::move(size));
        }
        rows_=sizes_.front().rows;
        const auto& model=target_.model();
        vocab_=model.vocab_size;
        width_=static_cast<std::uint64_t>(model.hidden_size)*model.hc_count*2U;
        const auto& steps=target_.compiled().execution.steps;
        for(std::size_t i=1;i<steps.size();++i)
            if(steps[i].op==SerialOp::final_hyper_mix && steps[i-1U].op==SerialOp::mlp_inject) tap_=i-1U;
        if(tap_==SIZE_MAX) throw std::logic_error("MTP wide copy verify has no wide hidden tap step");
        strix::hip_check(hipMalloc(reinterpret_cast<void**>(&hidden_),static_cast<std::size_t>(width_*rows_)),"MTP wide hidden rows");
        // The in-place components a forward overwrites: everything but the position-indexed KV and sparse index.
        for(const auto& a:resources_.session_state_layout->allocations) {
            if(a.component==SessionStateComponent::attention_kv || a.component==SessionStateComponent::sparse_index) continue;
            Saved s;s.view=resolve_session_state(*resources_.session_state_layout,*resources_.session_state_memory,a.component);
            if(s.view.storage==SessionStateStorage::device)
                strix::hip_check(hipMalloc(reinterpret_cast<void**>(&s.device),static_cast<std::size_t>(s.view.bytes)),"MTP wide state snapshot");
            else s.host.resize(static_cast<std::size_t>(s.view.bytes));
            saved_.push_back(std::move(s));
        }
        logits_.resize(static_cast<std::size_t>(vocab_));
        greedy_bytes_=hip_greedy_rows_scratch_bytes(hip_greedy_rows_max,static_cast<std::uint32_t>(vocab_));
        strix::hip_check(hipMalloc(reinterpret_cast<void**>(&greedy_scratch_),static_cast<std::size_t>(greedy_bytes_)),"MTP wide greedy scratch");
        strix::hip_check(hipHostMalloc(reinterpret_cast<void**>(&replies_),sizeof(HipGreedyRowsReply)*rows_,hipHostMallocMapped|hipHostMallocCoherent),
            "MTP wide greedy replies");
    }
    ~MtpWideCopyCycle() {
        if(hidden_)(void)hipFree(hidden_);
        if(greedy_scratch_)(void)hipFree(greedy_scratch_);
        if(replies_)(void)hipHostFree(replies_);
        for(auto& s:saved_) if(s.device)(void)hipFree(s.device);
    }
    MtpWideCopyCycle(const MtpWideCopyCycle&)=delete;
    MtpWideCopyCycle& operator=(const MtpWideCopyCycle&)=delete;
    // The largest configured size.
    [[nodiscard]] std::uint32_t rows() const noexcept {return rows_;}
    [[nodiscard]] const std::vector<Size>& sizes() const noexcept {return sizes_;}
    // Totals over every size.
    [[nodiscard]] Stats stats() const noexcept {
        Stats t;
        for(const auto& size:sizes_) {
            t.cycles+=size.stats.cycles;t.rows_accepted+=size.stats.rows_accepted;t.abandoned+=size.stats.abandoned;
            t.abandoned_rows_wasted+=size.stats.abandoned_rows_wasted;
            t.ms_committed+=size.stats.ms_committed;t.ms_abandoned+=size.stats.ms_abandoned;
        }
        return t;
    }
    void set_row_logits(RowLogits sink) {row_logits_=std::move(sink);}
    // One committed wide cycle when the greedy runtime proposes one and the target accepts every row, else nullopt
    // with the committed state unchanged (the caller runs the ordinary cycle). Sizes are tried largest first: the
    // first whose rows-1 token proposal succeeds runs. forced (the gate's fixed proposal) selects the size of
    // forced.size() + 1 rows.
    [[nodiscard]] std::optional<HipMtpGreedyCycleResult> cycle(HipMtpGreedyRuntime& greedy,const HipMtpTargetHiddenTicket& initial,
        std::uint32_t budget,std::span<const std::uint32_t> eos,std::span<const std::uint32_t> forced={}) {
        Size* size=nullptr;
        std::vector<std::uint32_t> proposal;
        for(auto& candidate:sizes_) {
            if(!forced.empty() && candidate.rows!=forced.size()+1U) continue;
            proposal=greedy.propose_wide(initial,candidate.rows,budget,eos,forced);
            if(!proposal.empty()){size=&candidate;break;}
        }
        if(!size) {
            if(!forced.empty() && std::none_of(sizes_.begin(),sizes_.end(),[&](const Size& x){return x.rows==forced.size()+1U;}))
                throw std::invalid_argument("MTP forced wide proposal matches no configured wide size");
            return std::nullopt;
        }
        const auto wide_rows=size->rows;
        auto& owner=*size->owner;
        auto& size_stats=size->stats;
        const auto start=std::chrono::steady_clock::now();
        host_phase_mark("w0 wide start",true); // STRIX_CYCLE_TIMING: wide cycle phases
        auto& session=greedy.target_session();
        auto* verifier=greedy.window_verifier();
        try {
            if(proposal.size()+1U!=wide_rows || !verifier) throw std::logic_error("MTP wide proposal does not fill the wide owner");
            std::vector<std::uint32_t> inputs{initial.predicted_token()};
            inputs.insert(inputs.end(),proposal.begin(),proposal.end());
            // External forwards need QSA invalidation even without pending GDN rows.
            verifier->flush_pending_window();
            owner.discard_deferred();
            strix::hip_check(hipDeviceSynchronize(),"MTP wide cycle start fence");
            const auto base=session;
            snapshot();
            host_phase_mark("w1 flush+snapshot");
            const auto result=[&] {
                owner.set_prefill_step_observer(observer(wide_rows));
                try {
                    auto executed=owner.execute(session,std::span<const std::uint32_t>(inputs));
                    owner.set_prefill_step_observer({});
                    return executed;
                } catch(...) {owner.set_prefill_step_observer({});throw;}
            }();
            if(session.health!=SerialSessionHealth::ready || session.next_position!=base.next_position+wide_rows ||
               session.committed_tokens!=base.committed_tokens+wide_rows || result.logits.size()!=wide_rows ||
               result.logit_row_indices.size()!=wide_rows)
                throw std::logic_error("MTP wide forward did not commit every row with its logits");
            strix::hip_check(hipDeviceSynchronize(),"MTP wide forward completion");
            host_phase_mark("w2 wide forward");
            const auto is_eos=[&](std::uint32_t t){return std::find(eos.begin(),eos.end(),t)!=eos.end();};
            // Row r predicts the token after inputs[r]; it is accepted when it equals inputs[r + 1].
            for(std::uint32_t r=0;r<wide_rows;++r)
                if(result.logit_row_indices[r]!=r || result.logits[r].bytes<vocab_*sizeof(float))
                    throw std::logic_error("MTP wide forward logit wide_rows differ from the all-wide_rows policy");
            const auto device_tokens=select_rows(result.logits,wide_rows);
            std::vector<std::uint32_t> emitted;
            const bool digest=prefix_batch_certification_enabled() && prompt_terminal_digest_enabled();
            std::vector<std::optional<ModelContentIdentity>> row_digests;
            for(std::uint32_t r=0;r<wide_rows;++r) {
                // Host logits: the diagnostics sink, the non-finite fallback, and the adopted last row.
                if(digest || row_logits_ || !device_tokens || r+1U==wide_rows) {
                    strix::hip_check(hipMemcpy(logits_.data(),reinterpret_cast<const void*>(result.logits[r].address),
                        static_cast<std::size_t>(vocab_*sizeof(float)),hipMemcpyDeviceToHost),"MTP wide row logits");
                    if(row_logits_) row_logits_(r,logits_);
                    if(digest) row_digests.push_back(raw_fp32_logits_digest(logits_));
                }
                if(device_tokens) {
                    if(row_logits_ && select_greedy_token(logits_,vocab_)!=(*device_tokens)[r])
                        throw std::logic_error("MTP wide device greedy row differs from the host selection");
                    emitted.push_back((*device_tokens)[r]);
                } else emitted.push_back(select_greedy_token(logits_,vocab_));
                if(r+1U==wide_rows || is_eos(emitted.back()) || emitted.back()!=inputs[r+1U]) break;
            }
            const auto n=static_cast<std::uint32_t>(emitted.size());
            host_phase_mark("w3 logits+select");
            if(n<wide_rows) {
                // Abandon: the pre-cycle committed frontier exactly; nothing of this cycle is committed.
                restore();
                session=base;
                owner.discard_deferred();
                // Restoring target bytes also requires unconditional QSA invalidation.
                verifier->flush_pending_window();
                verifier->skip_cycle_sample();
                host_phase_mark("w4 abandon restore");
                greedy.abandon_wide_cycle(emitted);
                host_phase_mark("w5 abandon bookkeeping");
                ++size_stats.abandoned;size_stats.abandoned_rows_wasted+=wide_rows;size_stats.abandon_rows.push_back(n-1U);
                size_stats.ms_abandoned+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                return std::nullopt;
            }
            std::vector<MtpDeviceView> views{initial.view()};
            for(std::uint32_t r=0;r+1U<n;++r) views.push_back(view(r));
            auto final_target=target_.adopt_prefill_row(session,inputs.back(),view(wide_rows-1U),logits_,target_.model().rotary_identity());
            host_phase_mark("w4 adopt");
            verifier->skip_cycle_sample();
            auto out=greedy.finish_wide_cycle(initial,std::move(final_target),views,
                std::span<const std::uint32_t>(inputs).first(n),std::move(emitted),wide_rows-1U,wide_rows,budget,eos);
            out.output_logits_digests=std::move(row_digests);
            host_phase_mark("w5 predictor prime+finish");
            ++size_stats.cycles;size_stats.rows_accepted+=out.accepted_candidates;
            size_stats.ms_committed+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            return out;
        } catch(...) {greedy.fail_cycle();throw;}
    }
private:
    struct Saved {SessionStateAddress view;std::byte* device=nullptr;std::vector<std::byte> host;};
    [[nodiscard]] MtpDeviceView view(std::uint32_t row) const {
        return MtpDeviceView{reinterpret_cast<std::uintptr_t>(hidden_+static_cast<std::size_t>(row)*width_),width_,1U};
    }
    [[nodiscard]] HipKnownPrefixWindow::PrefillStepObserver observer(std::uint32_t wide_rows) {
        return [this,wide_rows](std::size_t step,const std::function<ExecutionDispatchContext(std::uint32_t)>& row) {
            if(step!=tap_) return;
            const auto stream=hip_execution_stream();
            for(std::uint32_t r=0;r<wide_rows;++r) {
                const auto source=resolve_dispatch_write(row(r),0);
                if(source.bytes<width_) throw std::logic_error("MTP wide tap row is undersized");
                strix::hip_check(hipMemcpyAsync(hidden_+static_cast<std::size_t>(r)*width_,reinterpret_cast<const void*>(source.address),
                    static_cast<std::size_t>(width_),hipMemcpyDeviceToDevice,stream),"MTP wide tap copy");
            }
        };
    }
    // Every row's greedy token selected on the device, or nullopt when any row saw a non-finite logit (the caller
    // then selects on the host, which rejects it exactly as the serial path does).
    [[nodiscard]] std::optional<std::vector<std::uint32_t>> select_rows(const std::vector<ExecutionBufferAddress>& logits,
        std::uint32_t wide_rows) {
        const auto stream=hip_execution_stream();
        auto scratch=logits.front();
        scratch.address=reinterpret_cast<std::uintptr_t>(greedy_scratch_);scratch.bytes=greedy_bytes_;
        for(std::uint32_t first=0;first<wide_rows;first+=hip_greedy_rows_max) {
            const auto count=std::min(hip_greedy_rows_max,wide_rows-first);
            enqueue_hip_greedy_rows(std::span<const ExecutionBufferAddress>(logits).subspan(first,count),
                static_cast<std::uint32_t>(vocab_),scratch,replies_+first,stream);
        }
        strix::hip_check(hipStreamSynchronize(static_cast<hipStream_t>(stream)),"MTP wide greedy rows completion");
        std::vector<std::uint32_t> tokens(wide_rows);
        for(std::uint32_t r=0;r<wide_rows;++r) {
            if(replies_[r].status) return std::nullopt;
            tokens[r]=replies_[r].token;
        }
        return tokens;
    }
    void snapshot() {
        const auto stream=static_cast<hipStream_t>(target_.stream());
        for(auto& s:saved_) {
            if(s.device) strix::hip_check(hipMemcpyAsync(s.device,reinterpret_cast<const void*>(s.view.address),static_cast<std::size_t>(s.view.bytes),
                hipMemcpyDeviceToDevice,stream),"MTP wide state snapshot copy");
            else std::memcpy(s.host.data(),reinterpret_cast<const void*>(s.view.address),s.host.size());
        }
        strix::hip_check(hipStreamSynchronize(stream),"MTP wide state snapshot completion");
    }
    void restore() {
        strix::hip_check(hipDeviceSynchronize(),"MTP wide restore fence");
        const auto stream=static_cast<hipStream_t>(target_.stream());
        for(auto& s:saved_) {
            if(s.device) strix::hip_check(hipMemcpyAsync(reinterpret_cast<void*>(s.view.address),s.device,static_cast<std::size_t>(s.view.bytes),
                hipMemcpyDeviceToDevice,stream),"MTP wide state restore copy");
            else std::memcpy(reinterpret_cast<void*>(s.view.address),s.host.data(),s.host.size());
        }
        strix::hip_check(hipStreamSynchronize(stream),"MTP wide state restore completion");
    }
    HipMtpTargetHidden& target_;
    ExecutionRuntimeResources resources_;
    std::vector<Size> sizes_; // descending rows
    std::uint32_t rows_=0;
    std::uint64_t vocab_=0,width_=0;
    std::size_t tap_=SIZE_MAX;
    std::byte* hidden_=nullptr;
    std::vector<Saved> saved_;
    std::vector<float> logits_;
    std::byte* greedy_scratch_=nullptr;
    std::uint64_t greedy_bytes_=0;
    HipGreedyRowsReply* replies_=nullptr;
    RowLogits row_logits_;
};
} // namespace strix::decode_cli
