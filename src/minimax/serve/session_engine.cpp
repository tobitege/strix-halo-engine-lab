#include "session_engine.hpp"
#include <algorithm>
#include <tuple>

namespace strix::minimax::serve {
namespace ss = strix::session;
SessionEngine::SessionEngine(ss::Backend& b, std::uint32_t slots, std::uint64_t capacity,
                             std::function<void(ss::KvHandle)> release,
                             std::function<engine::PfProgress()> progress, bool paired_state, paired::InputProfile input_profile)
    : backend_(b), bindings_(input_profile!=paired::InputProfile::off && slots>8 ?
        throw std::invalid_argument("MiniMax input slot limit") : slots), capacity_(capacity), release_(std::move(release)), progress_(std::move(progress)),
      paired_state_(paired_state), input_profile_(input_profile) {
    if (!slots || !capacity || capacity == UINT64_MAX || !release_ || !progress_)
        throw std::invalid_argument("invalid MiniMax session limits/hooks");
    if(input_profile_!=paired::InputProfile::off) {
        if(!paired_state_ || slots>8 || input_profile_!=paired::InputProfile::plain_greedy_r1)
            throw std::invalid_argument("MiniMax input requires paired plain greedy R1 profile");
        require_input(); // a typed declaration alone cannot establish support
    }
}
SessionEngine::Binding& SessionEngine::checked(ss::KvHandle h) {
    if (!h.valid() || h.slot >= bindings_.size() || bindings_[h.slot].handle != h)
        throw std::invalid_argument("stale MiniMax session binding");
    return bindings_[h.slot];
}
const SessionEngine::Binding& SessionEngine::checked(ss::KvHandle h) const {
    return const_cast<SessionEngine*>(this)->checked(h);
}
ss::Hooks SessionEngine::storage_hooks() {
    ss::Hooks h;
    h.resident_prefill_supported = !paired_state_ && input_profile_ == paired::InputProfile::off;
    h.initialize = [this](ss::KvHandle k) {
        auto& b = bindings_.at(k.slot);
        if (b.handle.valid()) throw std::logic_error("MiniMax physical slot already leased");
        if(paired_state_)captured_={};
        b = {}; b.handle = k;
    };
    h.reset = [this](ss::KvHandle k) { auto& b = checked(k); if(paired_state_)captured_={}; release_(k); b = {}; b.handle = k; };
    h.release = [this](ss::KvHandle k) {
        auto& b = bindings_.at(k.slot);
        if (!b.handle.valid()) return; // partial initialization cleanup
        (void)checked(k); if(paired_state_)captured_={}; release_(k); b = {};
    };
    h.view = [this](ss::KvHandle k, std::uint32_t rank) { return capture_rank(k,rank,false); };
    h.capture = [this](ss::KvHandle k, std::uint32_t rank) { return capture_rank(k,rank,true); };
    h.restore = [this](ss::KvHandle k, std::uint32_t rank, const ss::RankState& state) {
        auto& b=checked(k);
        if (!paired_state_ || b.open || !store_ || rank>=2)
            throw ss::UnsupportedOperation("MiniMax generic restore unavailable for this profile");
        captured_={};
        SessionPlan p(backend_,SessionPlan::Kind::restore);p.slot=k.slot;p.generation=k.generation;
        p.frontier.handle=k;p.rank=rank;p.state[rank]=state;backend_.execute(p);
        if(rank==1){b.progress=p.frontier.progress;b.progress_limit=p.frontier.progress_limit;
            b.retained=p.frontier.retained;b.finish=p.frontier.finish;}
    };
    h.truncate = [this](ss::KvHandle k, std::uint32_t, std::uint64_t) {
        (void)checked(k); throw ss::UnsupportedOperation("MiniMax generic truncate unavailable");
    };
    return h;
}
ss::RankState SessionEngine::capture_rank(ss::KvHandle k,std::uint32_t rank,bool sequential) {
    const auto& b=checked(k);
    if(!paired_state_ || !store_ || b.open || rank>=2)
        throw ss::UnsupportedOperation("MiniMax paired capture unavailable for this profile");
    // Refuse ANY partial prefill, including another slot, before state work.
    for(const auto& binding:bindings_)if(binding.open)
        throw ss::UnsupportedOperation("MiniMax paired capture during partial prefill");
    paired::Frontier f;f.handle=k;f.tokens=store_->tokens(k);f.pending=store_->pending(k);
    f.progress=b.progress;f.progress_limit=b.progress_limit;
    f.retained=b.retained;f.finish=b.finish;
    if(sequential && rank==1 && captured_[1].owner && captured_frontier_==f) {
        auto out=captured_[1];captured_={};return out;
    }
    captured_={};
    SessionPlan p(backend_,SessionPlan::Kind::capture);p.frontier=f;backend_.execute(p);
    if(!p.state[0].owner || !p.state[1].owner)throw std::runtime_error("MiniMax incomplete paired capture");
    auto out=p.state[rank];
    if(sequential && rank==0){captured_frontier_=std::move(f);captured_=std::move(p.state);}
    return out;
}
void SessionEngine::begin(ss::KvHandle h, std::span<const std::uint32_t> prompt, bool context,
                          std::uint64_t task, std::uint64_t parent, std::uint64_t resident) {
    if(input_profile_!=paired::InputProfile::off)throw ss::UnsupportedOperation("MiniMax input profile excludes legacy planner calls");
    auto& b = checked(h);
    if (!store_ || b.open || prompt.empty() || resident >= capacity_ || prompt.size() + 1 >= capacity_ - resident ||
        store_->consumed(h) != resident || (resident && (context || parent || paired_state_)))
        throw std::invalid_argument("invalid MiniMax prefill plan");
    if(paired_state_)captured_={};
    b.prompt.assign(prompt.begin(), prompt.end());
    store_->reserve_consumed(h, prompt.size());
    b.progress = 0;
    // At most 60 layer boundaries per input row (a conservative finite bound
    // independent of native chunk/group size). No unbounded no-progress loop.
    b.progress_limit = 60 * prompt.size();
    SessionPlan p(backend_, SessionPlan::Kind::begin);
    p.slot = h.slot; p.prompt = prompt; p.context = context;
    p.task = task; p.parent = parent; p.generation = h.generation; p.resident = resident;
    try { backend_.execute(p); b.open = true; }
    catch (...) { store_->invalidate(h); throw; }
}
bool SessionEngine::unit(ss::KvHandle h, unsigned layers, std::uint32_t* next) {
    if(input_profile_!=paired::InputProfile::off)throw ss::UnsupportedOperation("MiniMax input profile excludes legacy planner calls");
    auto& b = checked(h);
    if (!store_) throw std::logic_error("MiniMax store not attached");
    (void)store_->consumed(h);
    if (!b.open || !layers || b.progress >= b.progress_limit)
        throw std::logic_error("MiniMax prefill checkpoint exhausted");
    SessionPlan p(backend_, SessionPlan::Kind::unit); p.layers = layers; p.slot = h.slot; p.generation = h.generation;
    const auto before = progress_();
    if (!before.open || before.slot != h.slot || before.generation != h.generation)
        throw std::logic_error("MiniMax native prefill binding mismatch");
    try { backend_.execute(p); }
    catch (...) { store_->invalidate(h); throw; }
    const auto after = progress_();
    if (!p.done && (after.id != before.id || after.slot != before.slot ||
        after.generation != before.generation || !after.open ||
        std::tie(after.c0,after.chunk,after.layer) <= std::tie(before.c0,before.chunk,before.layer)))
        { store_->invalidate(h); throw std::runtime_error("MiniMax native prefill made no checkpoint progress"); }
    ++b.progress;
    if (p.done) {
        if (p.next.size() != 1) throw std::runtime_error("MiniMax prefill result count");
        store_->append_consumed(h, b.prompt); b.prompt.clear(); b.open = false;
        *next = p.next.front();
    }
    return p.done;
}
std::vector<std::uint32_t> SessionEngine::step(std::span<const engine::Row> rows, std::uint32_t append,
                                              std::span<const ss::KvHandle> handles) {
    if(input_profile_!=paired::InputProfile::off)throw ss::UnsupportedOperation("MiniMax input profile excludes legacy planner calls");
    if (!store_ || rows.empty() || rows.size() > 8 || handles.size() != rows.size())
        throw std::invalid_argument("invalid MiniMax grouped rows/bindings");
    std::uint32_t seen = 0;
    for (std::size_t i=0;i<rows.size();++i) {
        const auto& row = rows[i]; const auto h = handles[i];
        const auto& b = checked(h);
        if (h.slot != row.slot) throw std::invalid_argument("MiniMax row/lease mismatch");
        const auto pos = store_->consumed(h);
        if (b.open || row.pos >= capacity_) throw std::invalid_argument("invalid MiniMax decode row");
        if (i && rows[i-1].slot == row.slot) {
            if (row.pos != rows[i-1].pos + 1) throw std::invalid_argument("MiniMax speculative cursor mismatch");
        } else {
            if (row.slot >= 32 || (seen & (1U << row.slot)) || row.pos != pos || store_->pending(h) != row.token)
                throw std::invalid_argument("MiniMax stale decode cursor");
            seen |= 1U << row.slot;
        }
    }
    if(paired_state_)captured_={};
    SessionPlan p(backend_, SessionPlan::Kind::step); p.rows = rows; p.bindings = handles; p.append = append;
    try { backend_.execute(p); }
    catch (...) {
        for (const auto& row : rows) store_->invalidate(bindings_[row.slot].handle);
        throw;
    }
    if (p.next.size() != rows.size()) throw std::runtime_error("MiniMax grouped result count");
    return p.next;
}
SessionDraft SessionEngine::draft(ss::KvHandle h, std::uint32_t anchor, std::uint64_t pos) {
    (void)checked(h);
    if(paired_state_)throw ss::UnsupportedOperation("MiniMax generic paired profile excludes configured drafters");
    if (!store_ || pos >= capacity_ || store_->consumed(h) != pos || store_->pending(h) != anchor)
        throw std::invalid_argument("MiniMax stale draft cursor");
    SessionPlan p(backend_, SessionPlan::Kind::draft); p.slot = h.slot; p.generation = h.generation; p.anchor = anchor; p.pos = pos;
    try { backend_.execute(p); }
    catch (...) { store_->invalidate(h); throw; }
    return p.drafts;
}
void SessionEngine::commit(ss::KvHandle h, std::uint32_t id) {
    if(input_profile_!=paired::InputProfile::off)throw ss::UnsupportedOperation("MiniMax input profile excludes legacy planner calls");
    auto& b = checked(h);
    if (b.open || !store_) throw std::logic_error("MiniMax commit before prefill completion");
    if(paired_state_)captured_={};
    if (const auto pending = store_->pending(h)) {
        if (store_->consumed(h) >= capacity_) throw std::runtime_error("MiniMax pending anchor exceeds physical rows");
        store_->reserve_consumed(h, 1);
        store_->append_consumed(h, std::span(&*pending, 1));
    }
    // Native lookup may select from the last physical row. Its selected ID is
    // an unconsumed pending anchor, not another physical row. The native
    // planner then terminates with context, as before this extraction.
    if (store_->consumed(h) > capacity_) throw std::runtime_error("MiniMax commit exceeds context");
    store_->set_pending(h, id);
}
std::uint64_t SessionEngine::checkpoint(ss::KvHandle h) const {
    if (!store_) throw std::logic_error("MiniMax store not attached");
    (void)store_->consumed(h); return checked(h).progress;
}
void SessionEngine::observe_named(ss::KvHandle h, std::span<const std::uint32_t> history, std::optional<std::uint32_t> pending) {
    auto& b = checked(h);
    if(paired_state_)throw ss::UnsupportedOperation("MiniMax generic paired profile excludes named activity");
    if (b.open || !store_ || history.size() > capacity_)
        throw std::invalid_argument("invalid MiniMax named frontier");
    const auto old = store_->tokens(h);
    if (old.size() > history.size() || !std::equal(old.begin(),old.end(),history.begin()))
        throw std::invalid_argument("MiniMax named frontier rewound");
    const auto suffix = history.subspan(old.size());
    store_->reserve_consumed(h,suffix.size()); store_->append_consumed(h,suffix); store_->set_pending(h,pending);
}
void SessionEngine::require_input() const {
    if(input_profile_==paired::InputProfile::off)
        throw ss::UnsupportedOperation("MiniMax generic input OFF");
    for(const auto& b:bindings_)if(b.open)
        throw ss::UnsupportedOperation("MiniMax input during partial prefill");
    SessionPlan p(backend_,SessionPlan::Kind::input_admit);p.input_profile=input_profile_;
    backend_.execute(p);
    if(p.native_identity.schema!=2 || p.native_identity.world!=2 || p.native_identity.slots!=bindings_.size() ||
       p.native_identity.capacity!=capacity_ ||
       !std::any_of(p.native_identity.assets_profile.begin(),p.native_identity.assets_profile.end(),[](auto v){return v!=0;}))
        throw std::invalid_argument("MiniMax declared input limits do not match loaded paired endpoint");
}
void SessionEngine::validate_input(ss::KvHandle h,std::span<const std::uint32_t> ids,bool terminal) const {
    const auto& b=checked(h);
    if(!store_)throw std::logic_error("MiniMax store not attached");
    const auto n=store_->consumed(h);const auto pending=store_->pending(h);
    if(b.open || (!terminal && b.finish!=ss::FinishReason::none))
        throw std::invalid_argument("MiniMax terminal lane requires resume");
    if(n>capacity_ || (pending && n>=capacity_ && !ids.empty()) ||
       ids.size()>capacity_-n || (!ids.empty() && pending && ids.size()>capacity_-n-1))
        throw std::invalid_argument("MiniMax input exceeds physical context");
    if(!std::all_of(ids.begin(),ids.end(),[](auto t){return t<200064;}))
        throw std::invalid_argument("MiniMax input token range");
}
void SessionEngine::validate_sampler(const ss::SamplerState& s) {
    const auto& p=s.parameters;
    if(p.temperature!=0 || p.top_p!=1 || p.top_k!=0 || p.min_p!=0 || p.seeded || p.seed!=0 ||
       s.rng_draws || s.constraint || !std::all_of(s.stop_tokens.begin(),s.stop_tokens.end(),[](auto t){return t<200064;}))
        throw std::invalid_argument("MiniMax input supports only native plain greedy policy");
}
void SessionEngine::consume(std::span<InputWork> work,const std::function<void()>& before_rows,bool single_group) {
    // The activated endpoint precharges bounded lane/work metadata. All histories,
    // result storage, old complete checkpoints and native reservations precede
    // writes. KvStore's partial_restore_supported remains FALSE: this rollback
    // imports exact old complete images directly, never truncates a prefix.
    std::vector<ss::StateSnapshot> old(work.size());
    std::vector<engine::Row> rows;rows.reserve(8);
    std::vector<ss::KvHandle> handles;handles.reserve(8);
    std::vector<std::size_t> indices;indices.reserve(8);
    for(std::size_t i=0;i<work.size();++i)if(!work[i].tokens.empty()) {
        store_->reserve_consumed(work[i].handle,work[i].tokens.size());
        old[i]=store_->capture(work[i].handle);
    }
    captured_={};
    if(before_rows)before_rows();
    bool attempted=false;
    try {
        for(std::size_t cycle=0;;++cycle) {
            rows.clear();handles.clear();indices.clear();
            for(std::size_t i=0;i<work.size();++i) {
                const auto h=work[i].handle;
                const auto first=single_group?0:cycle;
                const auto end=single_group?work[i].tokens.size():std::min(work[i].tokens.size(),cycle+1);
                for(auto pos=first;pos<end;++pos) {
                    rows.push_back({h.slot,work[i].tokens[pos],store_->consumed(h)+pos});
                    handles.push_back(h);indices.push_back(i);
                }
            }
            if(rows.empty())break;
            SessionPlan p(backend_,SessionPlan::Kind::input);p.rows=rows;p.bindings=handles;
            attempted=true;backend_.execute(p);
            if(p.next.size()!=rows.size() || !std::all_of(p.next.begin(),p.next.end(),[](auto t){return t<200064;}))
                throw std::runtime_error("MiniMax native input result count/range");
            for(std::size_t j=0;j<indices.size();++j)work[indices[j]].retained=p.next[j];
            if(single_group)break;
        }
    } catch(...) {
        const auto failure=std::current_exception();
        if(attempted)for(std::size_t i=0;i<work.size();++i)if(!work[i].tokens.empty()) {
            bool ok=true;
            for(unsigned rank=0;rank<2;++rank)try {
                SessionPlan p(backend_,SessionPlan::Kind::restore);p.frontier.handle=work[i].handle;
                p.rank=rank;p.state[rank]=old[i].ranks.at(rank);backend_.execute(p);
            } catch(...) {ok=false;}
            if(!ok)store_->invalidate(work[i].handle);
        }
        captured_={};std::rethrow_exception(failure);
    }
    // No fallible allocation/publication remains. Inputs are never outputs.
    for(auto& w:work)if(!w.tokens.empty()) {
        store_->append_consumed(w.handle,w.tokens);checked(w.handle).retained=w.retained;
    }
}
void SessionEngine::prefill(std::span<const std::uint32_t> ids,ss::KvHandle& h) {
    const ss::PrefillItem item{ids,h};prefill_batch(std::span(&item,1));
}
void SessionEngine::append_external(std::span<const std::uint32_t> ids,ss::KvHandle& h) {prefill(ids,h);}
void SessionEngine::prefill_batch(std::span<const ss::PrefillItem> items) {
    require_input();
    if(items.size()>caps().max_batch)throw std::invalid_argument("MiniMax input batch limit");
    std::uint32_t seen=0;
    for(const auto& item:items) {
        validate_input(item.state,item.tokens);
        if(seen&(1U<<item.state.slot))throw std::invalid_argument("MiniMax duplicate input lane");
        seen|=1U<<item.state.slot;
    }
    std::vector<InputWork> work;work.reserve(items.size());
    for(const auto& item:items) {
        InputWork w;w.handle=item.state;
        if(!item.tokens.empty()) {
            w.tokens.reserve(item.tokens.size()+1);
            if(auto pending=store_->pending(item.state))w.tokens.push_back(*pending);
            w.tokens.insert(w.tokens.end(),item.tokens.begin(),item.tokens.end());
        }
        work.push_back(std::move(w));
    }
    consume(work);
}
ss::StepResult SessionEngine::decode(ss::KvHandle& h,const ss::SamplerState& sampler) {
    const ss::DecodeItem item{h,sampler};auto out=decode_batch(std::span(&item,1));return std::move(out[0]);
}
std::vector<ss::StepResult> SessionEngine::decode_batch(std::span<const ss::DecodeItem> items) {
    require_input();
    if(items.size()>caps().max_batch)throw std::invalid_argument("MiniMax input batch limit");
    std::vector<ss::AdvanceItem> input;input.reserve(items.size());
    for(const auto& item:items)input.push_back({item.state,{},item.sampler,{},item.row_hints});
    std::vector<ss::StepResult> result(items.size()); // before any committed output
    auto output=advance(input);
    for(std::size_t i=0;i<output.size();++i){output[i].output.rows=output[i].rows;result[i]=std::move(output[i].output);}return result;
}
std::vector<ss::AdvanceResult> SessionEngine::advance(std::span<const ss::AdvanceItem> items) {
    require_input();
    if(items.size()>caps().max_batch)throw std::invalid_argument("MiniMax input batch limit");
    std::uint32_t seen=0,physical=0;constexpr std::array<std::uint32_t,2> shapes{1,2};
    for(const auto& item:items) {
        validate_input(item.state,item.prompt);validate_sampler(item.sampler);
        const auto hints=ss::resolve_row_hints(item.row_hints,true,shapes);
        const auto pending=store_->pending(item.state);
        const auto required=!item.sampler.max_tokens || item.sampler.cancellation.stop_requested() ? 0U :
            !item.prompt.empty() ? (pending?2U:1U) : pending && store_->consumed(item.state)<capacity_?1U:0U;
        if(hints.physical_row_cap && required>*hints.physical_row_cap)
            throw std::invalid_argument("MiniMax row cap cannot admit pending plus input");
        physical+=required;
        if(seen&(1U<<item.state.slot))throw std::invalid_argument("MiniMax duplicate advance lane");
        seen|=1U<<item.state.slot;
    }
    if(physical>8)throw std::invalid_argument("MiniMax grouped input exceeds eight physical rows");
    std::vector<ss::AdvanceResult> results(items.size());
    std::vector<InputWork> work(items.size());
    std::vector<bool> decoding(items.size(),false);
    for(std::size_t i=0;i<items.size();++i) {
        const auto& item=items[i];const auto h=item.state;auto& r=results[i];auto& o=r.output;
        auto& w=work[i];w.handle=h;w.tokens.reserve(2);o.tokens.reserve(1);
        o.before={store_->consumed(h),store_->pending(h)};o.after=o.before;
        // Call the live cancellation probe at this quiescent boundary. Capture
        // may perform transport, so check again immediately before row work.
        if(item.sampler.cancellation.stop_requested() || (item.cancelled && item.cancelled()))o.finish=ss::FinishReason::cancelled;
        else if(!item.sampler.max_tokens)o.finish=ss::FinishReason::length;
        else if(o.before.consumed_tokens+o.before.pending.has_value()>=capacity_+1)o.finish=ss::FinishReason::context;
        else if(!item.prompt.empty()) {
            if(o.before.pending)w.tokens.push_back(*o.before.pending);
            w.tokens.push_back(item.prompt[0]);r.input_tokens=1;
        } else {
            if(!o.before.pending && !checked(h).retained)
                throw std::invalid_argument("MiniMax decode has no consumed selection boundary");
            decoding[i]=true;
            if(o.before.pending)w.tokens.push_back(*o.before.pending);
            else w.retained=checked(h).retained;
        }
    }
    // No externally visible commit has occurred yet. A completed grouped call
    // wins subsequent cancellation races and all its results remain reportable.
    consume(work,[&] {
        for(std::size_t i=0;i<items.size();++i) {
            const auto& item=items[i];
            if(item.sampler.cancellation.stop_requested() || (item.cancelled && item.cancelled())) {
                work[i].tokens.clear();decoding[i]=false;results[i].input_tokens=0;
                results[i].output.finish=ss::FinishReason::cancelled;
            }
        }
    },true);
    for(std::size_t i=0;i<items.size();++i) {
        const auto& item=items[i];const auto h=item.state;auto& r=results[i];auto& o=r.output;auto& b=checked(h);
        const auto count=work[i].tokens.size();r.rows={count,count,0};
        if(decoding[i]) {
            const auto candidate=work[i].retained;
            // Retained-only first selection has zero physical rows. It never
            // reads transient native logits/hidden/taps from another lane.
            o.tokens.push_back(*candidate);store_->set_pending(h,*candidate);
            if(std::find(item.sampler.stop_tokens.begin(),item.sampler.stop_tokens.end(),*candidate)!=item.sampler.stop_tokens.end())
                o.finish=ss::FinishReason::stop;
            else if(store_->consumed(h)+1>=capacity_+1)o.finish=ss::FinishReason::context;
            else if(item.sampler.max_tokens==1)o.finish=ss::FinishReason::length;
        }
        b.finish=o.finish;o.after={store_->consumed(h),store_->pending(h)};
    }
    return results;
}
void SessionEngine::cancel_lane(ss::KvHandle& h) {
    require_input();validate_input(h,{},true);checked(h).finish=ss::FinishReason::cancelled;captured_={};
}
void SessionEngine::resume_lane(ss::KvHandle& h) {
    require_input();validate_input(h,{},true);checked(h).finish=ss::FinishReason::none;captured_={};
}
void SessionEngine::commit_pending(ss::KvHandle& h) {
    require_input();validate_input(h,{},true);const auto finish=checked(h).finish;
    if(finish==ss::FinishReason::none || finish==ss::FinishReason::cancelled)
        throw std::invalid_argument("MiniMax sealing requires successful terminal lane");
    if(const auto pending=store_->pending(h)) {
        if(store_->consumed(h)>=capacity_)throw std::invalid_argument("MiniMax sealing exceeds physical context");
        InputWork w;w.handle=h;w.tokens.push_back(*pending);consume(std::span(&w,1));
    }
}
strix::profile::ExecutionReceipt SessionEngine::receipt(ss::KvHandle h) const {
    (void)checked(h);if(!store_)throw std::logic_error("MiniMax store not attached");
    const auto n=store_->consumed(h);const auto& b=checked(h);auto r=receipt();
    r.digests={{"lease",std::to_string(h.slot)+":"+std::to_string(h.generation)},
               {"consumed",std::to_string(n)},
               {"pending",store_->pending(h)?std::to_string(*store_->pending(h)):"none"},
               {"retained",b.retained?std::to_string(*b.retained):"none"},
               {"finish",std::to_string(static_cast<unsigned>(b.finish))},{"rng_draws","0"}};
    return r;
}
ss::Capabilities SessionEngine::caps() const {
    // This contract counts consumed + pending. Public native admission and the
    // physical arena retain capacity_; only an unconsumed selection may follow
    // the last physical row. No additional row or slot is allocated/admitted.
    ss::Capabilities c; c.max_context = capacity_ + 1; c.max_slots = static_cast<std::uint32_t>(bindings_.size());
    c.max_batch = std::min(c.max_slots, 8U); c.pair = true;
    c.owner_control = input_profile_==paired::InputProfile::off;
    if(!c.owner_control) {
        c.batch_prefill=true;c.batch_decode=true;c.mixed_advance=true;c.row_hints=true;
        c.external_append=true;c.lane_cancellation=true;c.lane_resume=true;c.request_lane_batch=true;
    }
    return c;
}
strix::profile::ExecutionReceipt SessionEngine::receipt() const {
    strix::profile::ExecutionReceipt r; r.engine = "minimax";
    r.implementation = input_profile_==paired::InputProfile::off ? "shared-session/native-layer-plan" :
                       "shared-session/native-plain-greedy-r1"; r.topology = "pair";
    if(input_profile_!=paired::InputProfile::off) {
        r.identity.arithmetic_contract="native-step/plain-greedy-r1";
        r.identity.state_format="minimax-paired-schema2";
        r.identity.prefill_lineage="teacher-forced-native-step-r1";
    }
    return r;
}
} // namespace strix::minimax::serve
