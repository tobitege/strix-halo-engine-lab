#include "session_engine.hpp"
#include <algorithm>
#include <set>
#include <atomic>
#include <cstdlib>
#include <string_view>

namespace strix::gemma4::engine {
namespace {
// A plan executor, not a resource adapter. All native work goes through these
// grouped plans; no kernels, allocations or per-row forwards are duplicated.
class PlanBackend final : public engine::Backend, public session::Backend {
public:
    explicit PlanBackend(engine::Backend& target) : target_(target) {}
    ForwardResult forward_rows(std::span<const RowPlan> rows) override {
        ForwardPlan plan(*this,rows); execute(plan); return std::move(plan.result);
    }
    PrefillRows prefill_rows(KvTxn& tx,std::span<const std::uint32_t> tokens,bool export_hidden,bool taps) override {
        const PrefillItem item{&tx,tokens}; PrefillPlan plan(*this,{&item,1},export_hidden,taps,false);
        execute(plan); return std::move(plan.result.at(0));
    }
    std::vector<PrefillRows> prefill_rows_batch(std::span<const PrefillItem> items,bool export_hidden,bool taps) override {
        PrefillPlan plan(*this,items,export_hidden,taps,true);execute(plan);return std::move(plan.result);
    }
    std::size_t prefill_chunk_rows() const override {return target_.prefill_chunk_rows();}
    bool prefill_matches_decode() const override {return target_.prefill_matches_decode();}
    bool prefill_batch_supported() const override {return target_.prefill_batch_supported();}
    bool input_continuation_supported() const override {return target_.input_continuation_supported();}
    std::optional<std::uint64_t> native_turn_work_bound() const override {return target_.native_turn_work_bound();}
    std::unique_ptr<StreamState> create_stream() override {return target_.create_stream();}
    std::unique_ptr<StreamState> create_stream(std::uint32_t n) override {return target_.create_stream(n);}
    const ModelConfig& config() const override {return target_.config();}
    strix::serve::Json identity() const override {return target_.identity();}
private:
    engine::Backend& target_;
    void execute_plan(session::ExecutionPlan& p) override {
        if(auto* f=dynamic_cast<ForwardPlan*>(&p)) {
            if(f->rows.empty() || f->rows.size()>8) throw std::invalid_argument("invalid Gemma4 forward plan");
            std::set<std::size_t> outputs;
            for(const auto& row:f->rows) if(!row.active || !row.transaction || row.output_row>=f->rows.size() ||
                !outputs.insert(row.output_row).second) throw std::invalid_argument("invalid Gemma4 forward row binding");
            f->result=target_.forward_rows(f->rows);return;
        }
        if(auto* f=dynamic_cast<PrefillPlan*>(&p)) {
            if(f->items.empty()) throw std::invalid_argument("empty Gemma4 prefill plan");
            for(const auto& item:f->items) if(!item.transaction || item.tokens.empty())
                throw std::invalid_argument("invalid Gemma4 prefill item");
            if(f->batch) f->result=target_.prefill_rows_batch(f->items,f->hidden,f->taps);
            else {
                if(f->items.size()!=1) throw std::invalid_argument("invalid scalar prefill plan");
                f->result.push_back(target_.prefill_rows(*f->items[0].transaction,f->items[0].tokens,f->hidden,f->taps));
            }
            return;
        }
        throw session::UnsupportedOperation("unknown Gemma4 execution plan");
    }
};
void require(bool ok,const char* why) {if(!ok)throw std::invalid_argument(why);}
session::FinishReason finish(Status s) {
    switch(s) {case Status::stopped:return session::FinishReason::stop;
    case Status::length:return session::FinishReason::length;case Status::cancelled:return session::FinishReason::cancelled;
    default:return session::FinishReason::none;}
}
}
struct SessionEngine::Impl {
    struct Lane {
        session::KvHandle handle; Request request; bool started=false;
        std::size_t offset=0;
        // resumed admits a fresh cancellation source on the next bounded call;
        // new_turn keeps stop-policy admission open until the next selection.
        bool input_mode=false, sampler_initialized=false, resumed=false, new_turn=false;
        std::stop_token cancellation;
        std::shared_ptr<const PrefixSnapshot> candidate;
    };
    struct Checkpoint {
        std::shared_ptr<const void> physical;
        bool sampler_initialized, resumed, new_turn;
        std::stop_token cancellation;
    };
    PlanBackend backend;
    Coordinator compute;
    std::mutex& mutex;
    std::vector<Lane> lanes;
    std::atomic_flag in_flight = ATOMIC_FLAG_INIT;
    struct Guard {
        std::atomic_flag& flag;
        explicit Guard(std::atomic_flag& f):flag(f) {
            require(!flag.test_and_set(),"Gemma4 operation already in flight");
        }
        ~Guard(){flag.clear();}
    };
    const bool continuation, has_drafts;
    const bool decode_states;
    const std::size_t chunk;
    // Destroy the store first: its release hooks still need lanes and compute.
    std::unique_ptr<session::KvStore> kv;
    Impl(engine::Backend& b,DraftFactory d,std::shared_ptr<CostModel> costs,std::mutex& m,
         std::size_t slots,std::size_t limit,std::size_t budget)
        : backend(b),compute(backend,d,std::move(costs)),mutex(m),lanes(slots),continuation(b.input_continuation_supported() && !d),has_drafts(bool(d)),decode_states(b.prefill_matches_decode()),chunk(b.prefill_chunk_rows()) {
        require(slots>0 && slots<=8,"Gemma4 slot count exceeds grouped planner capacity");
        compute.set_prefill_limit(limit);compute.set_prefill_budget(budget);
        session::Hooks hooks;
        hooks.partial_restore_supported=true;
        hooks.initialize=[this](auto h){lanes.at(h.slot)=Lane{};lanes[h.slot].handle=h;};
        hooks.release=[this](auto h){auto& l=lane(h);std::lock_guard lock(mutex);if(l.started)compute.release(h.slot);l=Lane{};};
        hooks.reset=[this](auto h){auto& l=lane(h);std::lock_guard lock(mutex);if(l.started)compute.release(h.slot);l=Lane{};l.handle=h;};
        hooks.capture=[this](auto h,std::uint32_t){require(!in_flight.test(),"in-flight checkpoint");auto& l=lane(h);require(l.started,"unstarted lane checkpoint");
            check_frontier(l);std::lock_guard lock(mutex);
            auto physical=compute.capture_lane(h.slot);const auto retained=compute.captured_lane_bytes(physical);
            if(retained && *retained>UINT64_MAX-sizeof(Checkpoint)) throw std::overflow_error("checkpoint size overflow");
            // New continuation snapshots report unknown as zero, never a
            // cache-sized estimate of a full admitted native allocation.
            // Preserve legacy estimate-only accounting for default callers;
            // neither path authorizes complete-turn retention without bounds.
            const auto size=retained ? *retained+sizeof(Checkpoint) :
                continuation ? 0 : bytes(compute.result(h.slot).frontier);
            return session::RankState{std::make_shared<Checkpoint>(Checkpoint{std::move(physical),l.sampler_initialized,l.resumed,l.new_turn,l.cancellation}),size};};
        hooks.view=hooks.capture;
        hooks.restore=[this](auto h,std::uint32_t,const auto& state){auto& l=lane(h);std::lock_guard lock(mutex);
            auto checkpoint=std::static_pointer_cast<const Checkpoint>(state.owner);
            compute.restore_lane(h.slot,checkpoint->physical);l.sampler_initialized=checkpoint->sampler_initialized;l.resumed=checkpoint->resumed;l.new_turn=checkpoint->new_turn;l.cancellation=checkpoint->cancellation;l.started=true;l.request=compute.request(h.slot);l.offset=compute.prompt_offset(h.slot);l.input_mode=compute.input_mode(h.slot);l.candidate.reset();};
        hooks.truncate=[](auto,std::uint32_t,std::uint64_t){throw std::invalid_argument("Gemma4 partial lane rewind unsupported");};
        hooks.capture_prompt=[this](auto h,std::uint32_t){require(!in_flight.test(),"in-flight prompt checkpoint");auto& l=lane(h);require(bool(l.candidate),"unprepared prompt checkpoint");
            check_frontier(l);require(l.candidate->tokens.size()==kv->consumed(h),"stale prompt checkpoint");
            return session::RankState{l.candidate,l.candidate->retained_bytes().value_or(bytes(l.candidate->tokens.size()))};};
        hooks.restore_prompt=[this](auto h,std::span<const session::RankState> ranks){auto& l=lane(h);
            Guard guard(in_flight);
            require(ranks.size()==1 && !l.started,"invalid Gemma4 prompt restart");
            auto prefix=std::static_pointer_cast<const PrefixSnapshot>(ranks[0].owner);
            // The store has already checked the sealed purpose and C++ payload
            // provenance. Validate the NEW request before allocating a stream.
            if(!prefix || !reusable(*prefix) || prefix->tokens.empty() ||
                prefix->tokens.size()>=l.request.prompt.size() || !prefix->state ||
                prefix->state->frontier()!=prefix->tokens.size() ||
                !std::equal(prefix->tokens.begin(),prefix->tokens.end(),l.request.prompt.begin())) return false;
            if(prefix->purpose==PrefixSnapshot::Purpose::complete_turn) {
                const auto* forced=std::getenv("STRIX_G4_DFLASH");
                if(l.request.dflash==DraftMode::on || (l.request.dflash==DraftMode::automatic &&
                    forced && std::string_view(forced)=="on")) return false;
            }
            std::lock_guard lock(mutex);compute.submit_at(h.slot,l.request,std::move(prefix));l.started=true;
            l.offset=compute.result(h.slot).cached_tokens;return l.offset!=0;};
        kv=std::make_unique<session::KvStore>(std::uint32_t(slots),1,std::move(hooks));
    }
    Lane& lane(session::KvHandle h) {auto& l=lanes.at(h.slot);require(l.handle==h,"stale or foreign Gemma4 lease");return l;}
    const Lane& lane(session::KvHandle h) const {const auto& l=lanes.at(h.slot);require(l.handle==h,"stale or foreign Gemma4 lease");return l;}
    void check_frontier(const Lane& l) const {
        const auto& r=compute.result(l.handle.slot);
        const auto pending=(l.input_mode || l.offset==l.request.prompt.size()) ? compute.pending(l.handle.slot) : std::nullopt;
        require(kv->consumed(l.handle)==r.frontier && kv->pending(l.handle)==pending,
            "Gemma4 physical/logical frontier mismatch");
    }
    std::uint64_t bytes(std::uint64_t positions) const {
        const auto& c=backend.config();std::uint64_t total=0;
        for(std::uint32_t i=0;i<c.layer.size();++i) if(c.layer[i].producer==i) {
            const std::uint64_t width=std::uint64_t(c.kv_heads)*c.layer[i].head_dim;
            const auto rows=c.layer[i].attention==AttentionKind::global?positions:c.window;
            total+=rows*(2*width*sizeof(float)+sizeof(std::uint64_t));
        }return total;
    }
    bool reusable(const PrefixSnapshot& p) const {
        if(p.purpose==PrefixSnapshot::Purpose::complete_turn)
            return continuation && backend.input_continuation_supported() &&
                p.identity==backend.identity().dump() && bool(p.retained_bytes());
        return decode_states || (p.prefill_written && chunk && p.tokens.size()%chunk==0);
    }
    void validate_input(Lane& l, std::span<const std::uint32_t> tokens) const {
        require(continuation,"Gemma4 input continuation unsupported for configured backend/drafter");
        const auto consumed=kv->consumed(l.handle); const auto pending=kv->pending(l.handle);
        require(!l.started || (l.input_mode && compute.result(l.handle.slot).status==Status::running),"input needs active continuation lane; resume terminal lanes first");
        require(!l.cancellation.stop_requested() || l.resumed,"cancelled input source requires explicit resume");
        require(l.started || l.request.prompt.empty(),"legacy prepared request cannot enter input mode");
        require(tokens.size()<=backend.config().max_positions-consumed-std::size_t(pending.has_value()),"input exceeds context capacity");
        for (auto id:tokens) require(id<backend.config().vocab,"input token outside vocabulary");
        if(l.started) check_frontier(l);
    }
    Sampling validate_sampler(const Lane& l, const session::SamplerState& s) const {
        require(!l.cancellation.stop_requested() || l.resumed || s.cancellation==l.cancellation,"cancelled source requires explicit resume");
        require(!s.constraint && s.parameters.min_p==0 && s.parameters.seeded,"input continuation needs explicit supported sampling policy");
        require(std::isfinite(s.parameters.temperature) && s.parameters.temperature>=0 &&
            std::isfinite(s.parameters.top_p) && s.parameters.top_p>0 && s.parameters.top_p<=1,"invalid continuation sampling parameters");
        Sampling sampling{float(s.parameters.temperature),float(s.parameters.top_p),s.parameters.top_k,s.parameters.seed,s.rng_draws};
        sampling.validate();
        for(auto id:s.stop_tokens) require(id<backend.config().vocab,"stop token outside vocabulary");
        if(l.sampler_initialized) {
            const auto& old=l.request.sampling;
            require(s.rng_draws==compute.result(l.handle.slot).rng_draws && sampling.seed==old.seed &&
                sampling.temperature==old.temperature && sampling.top_p==old.top_p && sampling.top_k==old.top_k &&
                (l.new_turn || std::equal(s.stop_tokens.begin(),s.stop_tokens.end(),l.request.stop_ids.begin(),l.request.stop_ids.end())),"continuation sampler cursor/policy mismatch");
        }
        return sampling;
    }
    void sync(Lane& l) {
        const auto old=kv->consumed(l.handle);
        if(l.input_mode) kv->append_consumed(l.handle,compute.input_history(l.handle.slot).subspan(std::size_t(old)));
        else {
            const auto ids=compute.history(l.handle.slot,old);
            kv->append_consumed(l.handle,ids);
        }
        kv->set_pending(l.handle,compute.pending(l.handle.slot));
    }
};
SessionEngine::SessionEngine(Backend& b,DraftFactory d,std::shared_ptr<CostModel> costs,std::mutex& m,
    std::size_t slots,std::size_t limit,std::size_t budget) :impl_(std::make_unique<Impl>(b,std::move(d),std::move(costs),m,slots,limit,budget)) {}
SessionEngine::~SessionEngine()=default;
session::KvStore& SessionEngine::store(){require(!impl_->in_flight.test(),"Gemma4 store access during in-flight work");return *impl_->kv;}
void SessionEngine::prepare(session::KvHandle h,Request r){Impl::Guard guard(impl_->in_flight);auto& l=impl_->lane(h);
    (void)impl_->kv->consumed(h);
    require(!l.started && !r.prompt.empty() && r.max_tokens && r.prompt.size()+r.max_tokens<=caps().max_context,"invalid Gemma4 lane request");
    require(!impl_->kv->consumed(h) && !impl_->kv->pending(h),"Gemma4 admission requires fresh metadata");
    r.sampling.validate();for(auto id:r.prompt)require(id<impl_->backend.config().vocab,"prompt token outside vocabulary");
    impl_->kv->reserve_consumed(h,r.prompt.size()+r.max_tokens);l.request=std::move(r);
}
void SessionEngine::start(session::KvHandle h){Impl::Guard guard(impl_->in_flight);auto& l=impl_->lane(h);(void)impl_->kv->consumed(h);if(l.started)return;
    std::lock_guard lock(impl_->mutex);impl_->compute.submit_at(h.slot,l.request);l.started=true;}
std::uint64_t SessionEngine::initial_cursor(session::KvHandle h) const{return impl_->lane(h).request.sampling.draws;}
const Result& SessionEngine::result(session::KvHandle h) const {require(!impl_->in_flight.test(),"in-flight result inspection");require(impl_->lane(h).started,"unstarted Gemma4 lane");return impl_->compute.result(h.slot);}
strix::serve::Json SessionEngine::lane_receipt(session::KvHandle h) const {require(!impl_->in_flight.test(),"in-flight receipt inspection");require(impl_->lane(h).started,"unstarted Gemma4 lane");
    std::lock_guard lock(impl_->mutex);return impl_->compute.receipt(h.slot);}
bool SessionEngine::cache_candidate(session::KvHandle h){Impl::Guard guard(impl_->in_flight);auto& l=impl_->lane(h);std::lock_guard lock(impl_->mutex);
    require(l.started,"unstarted Gemma4 checkpoint");l.candidate.reset();
    auto p=impl_->compute.snapshot(h.slot);if(!p || !impl_->reusable(*p))return false;l.candidate=std::move(p);return true;}
void SessionEngine::commit_pending(session::KvHandle& h) {
    Impl::Guard guard(impl_->in_flight);auto& l=impl_->lane(h);
    (void)impl_->kv->consumed(h); // rejects quarantined handles before physical inspection
    require(impl_->continuation && impl_->backend.input_continuation_supported(),
        "Gemma4 terminal commit unsupported for configured backend/drafter");
    require(l.started && !l.cancellation.stop_requested(),"terminal commit requires uncancelled admitted lane");
    impl_->check_frontier(l);std::lock_guard lock(impl_->mutex);
    const auto& result=impl_->compute.result(h.slot);
    require((result.status==Status::stopped || result.status==Status::length) && result.error.empty() &&
        (!l.input_mode || !impl_->compute.input_remaining(h.slot)),"terminal commit requires successful quiescent turn");
    if(impl_->compute.sealed_turn(h.slot)) {impl_->compute.commit_pending(h.slot);return;}
    const auto anchor=impl_->kv->pending(h);
    impl_->kv->reserve_consumed(h,std::size_t(anchor.has_value()));
    const auto before=l;const auto checkpoint=impl_->compute.capture_lane(h.slot);
    try {
        impl_->compute.commit_pending(h.slot);
        if(anchor) impl_->kv->append_consumed(h,{&*anchor,1});
        l.candidate.reset();
    } catch(...) {
        try {impl_->compute.restore_lane(h.slot,checkpoint);l=before;}
        catch(...) {impl_->kv->invalidate(h);}
        throw;
    }
}
bool SessionEngine::turn_cache_candidate(session::KvHandle h) {
    Impl::Guard guard(impl_->in_flight);auto& l=impl_->lane(h);(void)impl_->kv->consumed(h);
    if(!impl_->continuation || !impl_->backend.input_continuation_supported() || !l.started ||
        l.cancellation.stop_requested()) return false;
    impl_->check_frontier(l);std::lock_guard lock(impl_->mutex);
    if(!impl_->compute.sealed_turn(h.slot)) return false;
    // Construct first: failure cannot discard a previously prepared candidate.
    auto p=impl_->compute.turn_snapshot(h.slot);if(!p) return false;
    l.candidate=std::move(p);return true;
}
std::optional<std::uint64_t> SessionEngine::turn_retained_bound(session::KvHandle h) const {
    require(!impl_->in_flight.test(),"in-flight turn sizing");
    const auto& l=impl_->lane(h);
    if(!impl_->continuation || !impl_->backend.input_continuation_supported() || !l.started ||
        l.cancellation.stop_requested()) return {};
    impl_->check_frontier(l);std::lock_guard lock(impl_->mutex);
    return impl_->compute.turn_retained_bound(h.slot);
}
std::optional<std::uint64_t> SessionEngine::turn_work_bound() const {
    if(impl_->in_flight.test() || !impl_->continuation || !impl_->backend.input_continuation_supported()) return {};
    std::lock_guard lock(impl_->mutex); // native capacity is mutable between passes
    return impl_->compute.turn_work_bound();
}
bool SessionEngine::matches_decode() const{return impl_->decode_states;}
std::size_t SessionEngine::chunk_rows() const{return impl_->chunk;}
void SessionEngine::prefill(std::span<const std::uint32_t> tokens,session::KvHandle& h) {
    Impl::Guard guard(impl_->in_flight);
    auto& l=impl_->lane(h); impl_->validate_input(l,tokens);
    if(tokens.empty()) return;
    require(!l.started || !impl_->compute.input_remaining(h.slot),"input lane has an unfinished bounded suffix");
    impl_->kv->reserve_consumed(h,tokens.size()+std::size_t(impl_->kv->pending(h).has_value()));
    std::lock_guard lock(impl_->mutex);
    const auto before=l;
    std::shared_ptr<const void> checkpoint;
    if(l.started) checkpoint=impl_->compute.capture_lane(h.slot);
    try {
        if(!l.started) {impl_->compute.submit_input_at(h.slot);l.started=true;l.input_mode=true;}
        impl_->compute.stage_input(h.slot,tokens);
        const Coordinator::LaneAdvance lane{h.slot,{}};
        do {
            impl_->compute.advance({&lane,1},true);
            const auto& r=impl_->compute.result(h.slot);
            if(!r.error.empty()) throw std::runtime_error(r.error);
        } while(impl_->compute.input_remaining(h.slot));
        impl_->sync(l); l.candidate.reset();
    } catch(...) {
        try {
            if(checkpoint) impl_->compute.restore_lane(h.slot,checkpoint);
            else if(l.started) impl_->compute.release(h.slot);
            l=before;
        } catch(...) {impl_->kv->invalidate(h);}
        throw;
    }
}
session::StepResult SessionEngine::decode(session::KvHandle& h,const session::SamplerState& s) {
    require(impl_->lane(h).input_mode,"scalar decode requires input continuation mode");
    const session::AdvanceItem item{h,{},s,{}};
    auto result=advance({&item,1});
    if(!result[0].error.empty()) throw std::runtime_error(result[0].error);
    return std::move(result[0].output);
}
void SessionEngine::append_external(std::span<const std::uint32_t> tokens,session::KvHandle& h) {
    const auto& l=impl_->lane(h);
    require(l.started && l.input_mode,"external append requires an admitted input continuation lane");
    prefill(tokens,h);
}
void SessionEngine::resume_lane(session::KvHandle& h) {
    Impl::Guard guard(impl_->in_flight);
    auto& l=impl_->lane(h); (void)impl_->kv->consumed(h);
    require(impl_->continuation && l.started && l.input_mode,"resume requires supported continuation lane");
    impl_->check_frontier(l); std::lock_guard lock(impl_->mutex);
    impl_->compute.resume(h.slot);l.resumed=true;l.new_turn=true;
}
std::span<const float> SessionEngine::retained_logits(session::KvHandle h) const {
    require(!impl_->in_flight.test(),"in-flight logit inspection");
    const auto& l=impl_->lane(h);impl_->check_frontier(l);return impl_->compute.retained_logits(h.slot);
}
std::vector<session::AdvanceResult> SessionEngine::advance(std::span<const session::AdvanceItem> items) {
    Impl::Guard guard(impl_->in_flight);
    for (const auto& item : items) session::validate_row_hints(item.row_hints, false);
    require(!items.empty() && items.size()<=impl_->lanes.size(),"invalid Gemma4 advance width");
    std::set<std::uint32_t> seen;std::vector<Coordinator::LaneAdvance> lanes;
    struct Before {std::size_t ids;std::uint64_t proposed,accepted,frontier;bool input;};
    std::vector<session::AdvanceResult> out(items.size());std::vector<Before> old;
    for(std::size_t i=0;i<items.size();++i){const auto& item=items[i];auto& l=impl_->lane(item.state);
        require(l.started && seen.insert(item.state.slot).second,"invalid or duplicate Gemma4 lane");
        impl_->check_frontier(l);
        const auto& r=impl_->compute.result(item.state.slot);const auto& s=item.sampler;
        require(r.status==Status::running,"terminal Gemma4 lane requires restart");
        if (l.input_mode) {
            require(impl_->continuation,"configured continuation unsupported");
            impl_->validate_sampler(l,s);
            require(!l.resumed || (!s.cancellation.stop_requested() && (!l.cancellation.stop_possible() ||
                (s.cancellation.stop_possible() && s.cancellation!=l.cancellation))),"resume requires fresh cancellation source");
            if (!item.prompt.empty()) {
                impl_->validate_input(l,item.prompt);
                const auto queued=impl_->compute.queued_input(item.state.slot);
                if(!queued.empty()) require(std::equal(item.prompt.begin(),item.prompt.end(),queued.begin(),queued.end()),"bounded suffix mismatch");
            } else {
                require(!impl_->compute.input_remaining(item.state.slot),"decode before suffix completion");
            }
        } else require(s.rng_draws==r.rng_draws && !s.constraint && s.parameters.min_p==0 &&
            float(s.parameters.temperature)==l.request.sampling.temperature && float(s.parameters.top_p)==l.request.sampling.top_p &&
            s.parameters.top_k==l.request.sampling.top_k && s.parameters.seed==l.request.sampling.seed &&
            s.max_tokens==l.request.max_tokens-r.ids.size() &&
            std::equal(s.stop_tokens.begin(),s.stop_tokens.end(),l.request.stop_ids.begin(),l.request.stop_ids.end()) &&
            std::equal(item.prompt.begin(),item.prompt.end(),l.request.prompt.begin()+std::ptrdiff_t(l.offset),l.request.prompt.end()),
            "Gemma4 lane input/cursor/policy mismatch");
        out[i].output.before={impl_->kv->consumed(item.state),impl_->kv->pending(item.state)};
        out[i].output.tokens.reserve(std::min<std::uint32_t>(s.max_tokens,8)); // native row cap bounds output
        old.push_back({r.ids.size(),r.acceptance.attempted,r.acceptance.accepted,r.frontier,!item.prompt.empty()});
        lanes.push_back({item.state.slot,[&item]{return item.sampler.cancellation.stop_requested() || (item.cancelled && item.cancelled());}});
    }
    // No lane is staged until every handle, policy and suffix has passed admission.
    // Complete checkpoints cover configuration and staged suffix allocation too.
    const bool has_input=std::any_of(items.begin(),items.end(),[&](const auto& item){return impl_->lane(item.state).input_mode;});
    std::vector<std::shared_ptr<const void>> checkpoints(has_input ? items.size() : 0);
    std::vector<Impl::Lane> before(has_input ? items.size() : 0);
    for(std::size_t i=0;i<items.size();++i) {
        auto& l=impl_->lane(items[i].state);
        if(l.input_mode) {
            before[i]=l;
            checkpoints[i]=impl_->compute.capture_lane(l.handle.slot);
            impl_->kv->reserve_consumed(l.handle,impl_->backend.config().max_positions-impl_->kv->consumed(l.handle));
        }
    }
    std::lock_guard lock(impl_->mutex);
    try {
        for(const auto& item:items) {
            auto& l=impl_->lane(item.state); l.candidate.reset();
            if(!l.input_mode) continue;
            if(!item.prompt.empty()) {
                if(!impl_->compute.input_remaining(l.handle.slot)) impl_->compute.stage_input(l.handle.slot,item.prompt);
                l.resumed=false;l.cancellation=item.sampler.cancellation;
            } else {
                auto sampling=impl_->validate_sampler(l,item.sampler);
                impl_->compute.configure_decode(l.handle.slot,sampling,item.sampler.stop_tokens,item.sampler.max_tokens);
                l.request=impl_->compute.request(l.handle.slot);l.sampler_initialized=true;l.resumed=false;l.new_turn=false;l.cancellation=item.sampler.cancellation;
            }
        }
        // One planner invocation across ready input and output lanes, including legacy requests.
        impl_->compute.advance(lanes,true);
    } catch(...) {
        for(std::size_t i=0;i<checkpoints.size();++i) if(checkpoints[i]) {
            try {impl_->compute.restore_lane(items[i].state.slot,checkpoints[i]);impl_->lane(items[i].state)=before[i];}
            catch(...) {impl_->kv->invalidate(items[i].state);}
        }
        throw;
    }
    for(std::size_t i=0;i<items.size();++i){auto& l=impl_->lane(items[i].state);
        if (l.input_mode && !impl_->compute.result(l.handle.slot).error.empty()) {
            auto& a=out[i]; a.error=impl_->compute.result(l.handle.slot).error;
            try {impl_->compute.restore_lane(l.handle.slot,checkpoints[i]);l=before[i];}
            catch(...) {impl_->kv->invalidate(items[i].state);}
            a.output.after=a.output.before; a.output.rng_draws=impl_->compute.result(items[i].state.slot).rng_draws;
            continue;
        }
        const auto& r=impl_->compute.result(l.handle.slot);
        auto& a=out[i];auto& o=a.output;
        // Freeze every committed output before a metadata/capture failure in
        // another lane can be reported. No exception hides a peer's commit.
        o.tokens.assign(r.ids.begin()+std::ptrdiff_t(old[i].ids),r.ids.end());
        o.rng_draws=r.rng_draws;o.finish=finish(r.status);
        o.proposed_drafts=std::uint32_t(r.acceptance.attempted-old[i].proposed);
        o.accepted_drafts=std::uint32_t(r.acceptance.accepted-old[i].accepted);
        try {
            impl_->sync(l);
            if(l.input_mode) {
                if(old[i].input) a.input_tokens=std::size_t(r.frontier-old[i].frontier)-
                    std::size_t(o.before.pending.has_value() && r.frontier>old[i].frontier);
                if(items[i].sampler.max_tokens && r.status==Status::length && r.frontier+std::size_t(impl_->compute.pending(l.handle.slot).has_value())>=caps().max_context)
                    o.finish=session::FinishReason::context;
                if(!r.error.empty()) a.error=r.error;
            } else {
                const auto offset=impl_->compute.prompt_offset(l.handle.slot);a.input_tokens=offset-l.offset;l.offset=offset;
                l.cancellation=items[i].sampler.cancellation;
                if(r.status==Status::failed)a.error=r.error;
            }
        } catch(const std::exception& error) {
            a.error=error.what();
            if(l.input_mode) {impl_->kv->invalidate(l.handle);o.after=o.before;continue;}
        }
        o.after={impl_->kv->consumed(l.handle),impl_->kv->pending(l.handle)};
    }return out;
}
session::Capabilities SessionEngine::caps() const {session::Capabilities c;c.max_context=impl_->backend.config().max_positions;
    c.max_slots=std::uint32_t(impl_->lanes.size());c.max_batch=c.max_slots;
    c.prefix_cache=impl_->decode_states || impl_->chunk!=0;c.mixed_advance=true;
    c.external_append=c.lane_resume=impl_->continuation;
    c.turn_commit=impl_->continuation && impl_->backend.input_continuation_supported();
    c.lane_cancellation=true;c.temperature=c.top_p=c.top_k=c.seed=true;c.speculative=impl_->has_drafts;return c;}
void SessionEngine::cancel_lane(session::KvHandle& h){Impl::Guard guard(impl_->in_flight);(void)impl_->kv->consumed(h);require(impl_->lane(h).started,"unstarted lane cancellation");
    std::lock_guard lock(impl_->mutex);impl_->compute.cancel(h.slot);}
strix::profile::ExecutionReceipt SessionEngine::receipt() const {
    strix::profile::ExecutionReceipt r;r.engine="gemma4";r.implementation="shared-session/native-grouped-plan";r.topology="single";
    const auto identity=impl_->backend.identity();
    if(const auto* value=identity.find("identity"))r.identity.arithmetic_contract=value->as_string();
    if(const auto* value=identity.find("prefill_identity"))r.identity.prefill_lineage=value->as_string();
    r.identity.state_format="gemma4-stream-v1";return r;
}
} // namespace strix::gemma4::engine
