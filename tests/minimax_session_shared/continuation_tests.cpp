// Always-active checks exercise the production adapter/backend, additive control
// replay and paired transaction with independent mutable native-rank resources.
#define main minimax_r4_retained_suite
#include "paired_state_tests.cpp"
#undef main
#include <limits>

static std::uint32_t oracle_state(std::span<const std::uint32_t> ids) {
    std::uint64_t state=0;
    for(std::size_t pos=0;pos<ids.size();++pos)state=(state*131+ids[pos]+pos+1)%200064;
    return static_cast<std::uint32_t>(state);
}
static std::uint32_t oracle_next(std::span<const std::uint32_t> ids) {return (oracle_state(ids)*17+23)%200064;}
static void boundary(Pair& p,ss::KvHandle h,const std::vector<std::uint32_t>& ids) {
    auto snap=p.store.capture(h);auto image=Pair::image(snap.ranks[0]);
    check(snap.tokens==ids && !snap.pending,"every input consumed, none published");
    check(image->frontier.retained==oracle_next(ids),"retained native candidate independently computed");
    for(unsigned rank=0;rank<2;++rank) {
        std::uint64_t state=0;
        for(std::size_t pos=0;pos<ids.size();++pos) {
            state=(state*131+ids[pos]+pos+1)%200064;
            const auto offset=pos*ps::Endpoint::row_bytes;std::uint32_t actual=0;
            for(unsigned j=0;j<4;++j)actual|=std::uint32_t(image->rows[rank][offset+j])<<(j*8);
            check(actual==state,"actual causal rows, not metadata-only input");
        }
    }
}
static void full_and_mixed() {
    Pair p(64ULL<<20,true,true);auto a=p.store.acquire(),b=p.store.acquire(),c=p.store.acquire();
    auto caps=p.engine.caps();
    check(caps.external_append && caps.mixed_advance && caps.batch_decode && caps.batch_prefill && caps.request_lane_batch &&
          caps.row_hints && caps.lane_cancellation && caps.lane_resume && !caps.owner_control &&
          !caps.turn_commit && !caps.fork_lanes && !caps.shared_prefix_cow && !caps.prefix_cache,"precisely enumerated caps");
    std::vector<std::uint32_t> x{4,8,15},y{16,23};
    std::array<ss::PrefillItem,2> fill{{{x,a},{y,b}}};p.engine.prefill_batch(fill);
    check(p.rank0.input_calls==3 && p.rank0.input_rows==5 && p.rank1.input_calls==3 && p.rank1.input_rows==5,"grouped unequal prefill cycles");
    check(p.rank0.input_trace[0].size()==2 && p.rank0.input_trace[1].size()==2 && p.rank0.input_trace[2].size()==1,"one grouped forward per cycle");
    boundary(p,a,x);boundary(p,b,y);
    // Another lane overwrites native last-call outputs. First selection must use
    // the private candidate at a's exact fully-consumed boundary.
    p.engine.prefill(std::vector<std::uint32_t>{99},c);auto calls=p.rank0.input_calls;
    ss::SamplerState s;s.max_tokens=4;auto first=p.engine.decode(a,s);
    check(first.tokens==std::vector<std::uint32_t>{oracle_next(x)} && first.before==ss::Frontier{3,{}} &&
          first.after==ss::Frontier{3,first.tokens[0]} && first.rows.physical_rows==0 && !first.rng_draws &&
          p.rank0.input_calls==calls,"retained selection, zero forward/RNG");
    auto r=p.engine.decode(b,s);check(r.tokens[0]==oracle_next(y),"cross-lane transient overwrite isolated");
    const auto next_a=first.tokens[0];x.push_back(next_a);
    const auto extra=std::vector<std::uint32_t>{42};
    // a has a pending decode anchor; c is ready to consume supplied input.
    std::array<ss::AdvanceItem,2> mixed{{{a,{},s,{},ss::RowHints{8,0.5}},{c,extra,s,{},{}}}};
    calls=p.rank0.input_calls;auto out=p.engine.advance(mixed);
    check(p.rank0.input_calls==calls+1 && p.rank1.input_calls==calls+1 && p.rank0.input_trace.back().size()==2,"ONE mixed grouped execution");
    check(out[0].output.tokens==std::vector<std::uint32_t>{oracle_next(x)} && out[0].input_tokens==0 && out[0].rows.physical_rows==1 &&
          !out[0].output.rows.physical_rows && out[1].input_tokens==1 && out[1].output.tokens.empty() && out[1].rows.physical_rows==1,
          "input order, actual rows and no discarded candidate output");
    boundary(p,c,{99,42});
    // Consume one previously delivered pending anchor exactly once, then suffix.
    const auto anchor=*p.store.pending(b);std::vector<std::uint32_t> suffix{7,11};
    p.engine.append_external(suffix,b);y.push_back(anchor);y.insert(y.end(),suffix.begin(),suffix.end());boundary(p,b,y);
    calls=p.rank0.input_calls;p.engine.prefill({},b);p.engine.append_external({},b);
    check(p.rank0.input_calls==calls,"empty input no-op");
    auto receipt=p.engine.receipt(a);check(receipt.implementation=="shared-session/native-plain-greedy-r1" &&
          receipt.json()!=p.engine.receipt(b).json(),"per-handle owning receipts");
    p.stop();
}
static void policies_and_lifecycle() {
    Pair p(64ULL<<20,true,true);auto a=p.store.acquire(),b=p.store.acquire();
    p.engine.prefill(std::vector<std::uint32_t>{1,3},a);p.engine.prefill(std::vector<std::uint32_t>{9},b);
    const auto candidate=oracle_next(std::vector<std::uint32_t>{1,3});
    ss::SamplerState s;s.max_tokens=9;s.stop_tokens=std::span(&candidate,1);
    auto result=p.engine.decode(a,s);check(result.finish==ss::FinishReason::stop && result.tokens[0]==candidate,"stop includes first stop only");
    refuses([&]{(void)p.engine.decode(a,s);});refuses([&]{p.engine.prefill(std::vector<std::uint32_t>{5},a);});
    auto calls=p.rank0.input_calls;p.engine.commit_pending(a);p.engine.commit_pending(a);
    check(p.rank0.input_calls==calls+1 && !p.store.pending(a) && p.store.consumed(a)==3,"repeat seal consumes terminal anchor once");
    auto terminal=p.store.capture(a);check(Pair::image(terminal.ranks[0])->frontier.finish==ss::FinishReason::stop,"terminal snapshot policy");
    p.engine.resume_lane(a);boundary(p,a,{1,3,candidate});
    p.engine.cancel_lane(a);auto cancelled=p.store.capture(a);refuses([&]{p.engine.commit_pending(a);});
    p.engine.resume_lane(a);s.stop_tokens={};s.max_tokens=0;calls=p.rank0.input_calls;
    auto zero=p.engine.decode(a,s);check(zero.tokens.empty() && zero.finish==ss::FinishReason::length && zero.before==zero.after &&
        zero.rows.physical_rows==0 && p.rank0.input_calls==calls,"zero output allowance does no physical work");
    p.engine.commit_pending(a);check(p.rank0.input_calls==calls,"already sealed terminal no-op");
    p.engine.resume_lane(a);ss::SamplerState live;live.max_tokens=7;std::stop_source stop;stop.request_stop();
    ss::SamplerState stopped=live;stopped.cancellation=stop.get_token();
    std::array<ss::DecodeItem,2> items{{{a,stopped,{}},{b,live,{}}}};auto batch=p.engine.decode_batch(items);
    check(batch[0].finish==ss::FinishReason::cancelled && batch[0].tokens.empty() && batch[1].tokens.size()==1,"per-lane cancellation preserves peer");
    refuses([&]{p.engine.append_external({},a);});p.engine.resume_lane(a);
    check(p.engine.decode(a,live).tokens.size()==1,"fresh cancellation source resumes same history");
    // Cancellation between capture and the native grouped boundary.
    unsigned probes=0;const std::vector<std::uint32_t> ids{33};
    std::array<ss::AdvanceItem,2> advance{{{a,ids,live,[&]{return ++probes>=2;},{}},{b,{},live,{},{}}}};
    calls=p.rank0.input_calls;auto mixed=p.engine.advance(advance);
    check(mixed[0].output.finish==ss::FinishReason::cancelled && mixed[0].rows.physical_rows==0 &&
          mixed[1].rows.physical_rows==1 && p.rank0.input_calls==calls+1 && p.rank0.input_trace.back().size()==1,
          "boundary cancellation excludes only its row");
    // Fresh destination restores cancellation and retained selection, including
    // exact owner policy metadata; it cannot decode until explicitly resumed.
    p.store.release(b);b=p.store.acquire();p.store.restore(b,cancelled);
    refuses([&]{(void)p.engine.decode(b,live);});p.engine.resume_lane(b);
    check(p.engine.decode(b,live).tokens[0]==oracle_next(std::vector<std::uint32_t>{1,3,candidate}),"restore retained selection and lifecycle");
    p.stop();
}
static void validation() {
    Pair p(64ULL<<20,true,true);auto a=p.store.acquire(),b=p.store.acquire();p.engine.prefill(std::vector<std::uint32_t>{1},a);
    ss::SamplerState s;s.max_tokens=7;
    const auto controls=p.wire.controls,calls=p.rank0.input_calls;
    const auto reject_sampler=[&](ss::SamplerState bad){std::array<ss::DecodeItem,2> batch{{{a,s,{}},{b,bad,{}}}};
        refuses([&]{(void)p.engine.decode_batch(batch);});};
    auto bad=s;bad.parameters.temperature=0.1;reject_sampler(bad);bad=s;bad.parameters.top_p=0.5;reject_sampler(bad);
    bad=s;bad.parameters.top_k=1;reject_sampler(bad);bad=s;bad.parameters.min_p=0.1;reject_sampler(bad);
    bad=s;bad.parameters.seeded=true;reject_sampler(bad);bad=s;bad.parameters.seed=1;reject_sampler(bad);
    bad=s;bad.rng_draws=1;reject_sampler(bad);bad=s;bad.constraint=std::make_shared<strix::serve::ConstraintSpec>();reject_sampler(bad);
    bad=s;bad.parameters.temperature=std::numeric_limits<double>::quiet_NaN();reject_sampler(bad);
    const auto reject_hint=[&](ss::RowHints hint){ss::DecodeItem item{a,s,hint};refuses([&]{(void)p.engine.decode_batch(std::span(&item,1));});};
    reject_hint({0,{}});reject_hint({1,-1});reject_hint({1,std::numeric_limits<double>::infinity()});
    std::array<ss::DecodeItem,2> duplicate{{{a,s,{}},{a,s,{}}}};refuses([&]{(void)p.engine.decode_batch(duplicate);});
    const std::vector<std::uint32_t> ids{2},invalid{200064},large(9,1);
    std::array<ss::PrefillItem,2> late{{{ids,a},{invalid,b}}};refuses([&]{p.engine.prefill_batch(late);});
    std::array<ss::PrefillItem,2> dupfill{{{ids,a},{ids,a}}};refuses([&]{p.engine.prefill_batch(dupfill);});
    refuses([&]{p.engine.prefill(large,b);});refuses([&]{(void)p.engine.decode(b,s);});
    check(p.wire.controls==controls && p.rank0.input_calls==calls,"entire invalid group refuses before controls/rows");
    auto stale=b;p.store.release(b);b=p.store.acquire();refuses([&]{p.engine.prefill(ids,stale);});
    refuses([&]{(void)p.engine.receipt(stale);});
    p.rank0.configured=true;refuses([&]{p.engine.prefill(ids,b);});p.rank0.configured=false;
    p.rank0.named=true;refuses([&]{p.engine.prefill(ids,b);});p.rank0.named=false;
    p.rank0.parked.open=true;refuses([&]{p.engine.prefill(ids,b);});p.rank0.parked.open=false;
    check(p.wire.controls==controls,"unsupported active resources before controls");
    p.stop();
    // An old runner with no optional input methods remains buildable in the
    // existing owner tests. State-only endpoints cannot declare input support.
    Pair typed(64ULL<<20,true,true);
    const auto controls_before=typed.wire.controls;
    for(auto limits:std::array<std::pair<unsigned,std::uint64_t>,3>{{{2,8},{3,7},{UINT32_MAX,8}}})
        refuses([&]{ms::SessionEngine wrong(typed.backend,limits.first,limits.second,[](auto){},[]{return mm::PfProgress{};},true,
                                           ps::InputProfile::plain_greedy_r1);});
    check(typed.wire.controls==controls_before,"declared limits cannot override loaded native identity or allocate unbounded slots");typed.stop();
    Pair state_only;refuses([&]{ms::SessionEngine wrong(state_only.backend,3,8,[](auto){},[]{return mm::PfProgress{};},true,
                                                        ps::InputProfile::plain_greedy_r1);});state_only.stop();
}
static void context_and_empty() {
    Pair p(64ULL<<20,true,true);auto a=p.store.acquire();const std::vector<std::uint32_t> ids(8,3);
    p.engine.prefill(ids,a);boundary(p,a,ids);ss::SamplerState s;s.max_tokens=100;
    const auto calls=p.rank0.input_calls;auto r=p.engine.decode(a,s);
    check(r.tokens.size()==1 && r.finish==ss::FinishReason::context && r.rows.physical_rows==0 &&
          p.rank0.input_calls==calls,"physical capacity and pending allowance distinguished");
    refuses([&]{p.engine.commit_pending(a);});p.engine.resume_lane(a);
    r=p.engine.decode(a,s);check(r.tokens.empty() && r.finish==ss::FinishReason::context && r.before==r.after,"no post-context row/output");
    p.engine.prefill_batch({});auto empty=p.engine.decode_batch({});check(empty.empty(),"explicit empty groups no-op");p.stop();
}
static void failed_rows(unsigned failing_rank,bool rollback_fails) {
    Pair p(64ULL<<20,true,true);auto a=p.store.acquire(),b=p.store.acquire(),c=p.store.acquire();
    p.engine.prefill(std::vector<std::uint32_t>{1,2},a);p.engine.prefill(std::vector<std::uint32_t>{4},b);
    p.engine.prefill(std::vector<std::uint32_t>{8},c);
    auto olda=p.store.capture(a),oldb=p.store.capture(b),bystander=p.store.capture(c);
    auto& failing=failing_rank?p.rank1:p.rank0;failing.fail_input_call=failing.input_calls+2;
    if(rollback_fails)failing.fail_import=2;
    const std::vector<std::uint32_t> x{3,5,7},y{6,9};std::array<ss::PrefillItem,2> group{{{x,a},{y,b}}};
    refuses([&]{p.engine.prefill_batch(group);});
    if(rollback_fails){refuses([&]{(void)p.store.tokens(a);});refuses([&]{(void)p.store.tokens(b);});}
    else {
        for(const auto& entry:std::array<std::pair<ss::KvHandle,ss::StateSnapshot>,2>{{{a,olda},{b,oldb}}}) {
            auto now=p.store.capture(entry.first);check(now.tokens==entry.second.tokens && now.pending==entry.second.pending &&
                Pair::image(now.ranks[0])->rows==Pair::image(entry.second.ranks[0])->rows &&
                Pair::image(now.ranks[0])->frontier==Pair::image(entry.second.ranks[0])->frontier,"complete group/rank rollback after native writes");
        }
        p.engine.prefill_batch(group);boundary(p,a,{1,2,3,5,7});boundary(p,b,{4,6,9});
        refuses([&]{p.store.restore(a,olda);}); // shared restriction remains real
    }
    check(Pair::image(p.store.capture(c).ranks[0])->rows==Pair::image(bystander.ranks[0])->rows,"bystander survives failed group");
    check(olda.sealed() && oldb.sealed(),"immutable owners survive row and rollback faults");p.stop();
}
static void source_lifetime() {
    Pair p(64ULL<<20,true,true);auto source=p.store.acquire(),dest=p.store.acquire();
    p.engine.prefill(std::vector<std::uint32_t>{2,5},source);auto frozen=p.store.capture(source);
    const auto old_handle=source;p.store.release(source);source=p.store.acquire();
    p.engine.prefill(std::vector<std::uint32_t>{77},source);p.store.restore(dest,frozen);
    ss::SamplerState s;s.max_tokens=5;auto r=p.engine.decode(dest,s);
    check(source.generation!=old_handle.generation && r.tokens[0]==oracle_next(std::vector<std::uint32_t>{2,5}),"release/reuse cannot overwrite retained source");
    p.stop();
}
static void result_and_transport_faults() {
    {
        Pair p(64ULL<<20,true,true);auto a=p.store.acquire(),b=p.store.acquire();
        p.engine.prefill(std::vector<std::uint32_t>{2,4},a);p.engine.prefill(std::vector<std::uint32_t>{7},b);
        auto old=p.store.capture(a),other=p.store.capture(b);ss::SamplerState s;s.max_tokens=8;
        (void)p.engine.decode(a,s);old=p.store.capture(a);
        p.runner.input_result_fault=[](auto& next){next.clear();};
        refuses([&]{(void)p.engine.decode(a,s);});p.runner.input_result_fault={};
        auto restored=p.store.capture(a);
        check(restored.tokens==old.tokens && restored.pending==old.pending &&
            Pair::image(restored.ranks[0])->frontier==Pair::image(old.ranks[0])->frontier &&
            Pair::image(restored.ranks[0])->rows==Pair::image(old.ranks[0])->rows,"malformed post-write result fully rolled back before publication");
        check(Pair::image(p.store.capture(b).ranks[0])->rows==Pair::image(other.ranks[0])->rows,"malformed result bystander");p.stop();
    }
    {
        Pair p(64ULL<<20,true,true);auto a=p.store.acquire(),b=p.store.acquire();
        p.engine.prefill(std::vector<std::uint32_t>{3},a);p.engine.prefill(std::vector<std::uint32_t>{8},b);
        auto frozen=p.store.capture(a),other=p.store.capture(b);
        bool armed=false,injected=false;unsigned votes=0;
        p.runner.command_fault=[&](const auto& c){if(c.command==16)armed=true;};
        p.wire.fault=[&](unsigned rank,auto& bytes){if(armed && rank==1 && bytes.size()==1 && ++votes==2){bytes[0]=17;injected=true;}};
        refuses([&]{p.engine.append_external(std::vector<std::uint32_t>{4},a);});
        check(injected,"fault follows actual input writes");refuses([&]{(void)p.store.tokens(a);});
        check(frozen.sealed() && other.sealed() && Pair::image(frozen.ranks[0])->frontier.retained==oracle_next(std::vector<std::uint32_t>{3}),
              "input lost acknowledgement abort preserves immutable owners");
    }
    for(unsigned rank=0;rank<2;++rank) {
        Pair p(64ULL<<20,true,true);auto a=p.store.acquire(),b=p.store.acquire();
        p.engine.prefill(std::vector<std::uint32_t>{2},a);p.engine.prefill(std::vector<std::uint32_t>{5},b);
        ss::SamplerState s;s.max_tokens=1;(void)p.engine.decode(a,s);p.engine.cancel_lane(b);
        auto source=p.store.capture(a),old=p.store.capture(b);
        (rank?p.rank1:p.rank0).fail_import=1;refuses([&]{p.store.restore(b,source);});
        auto now=p.store.capture(b);
        check(now.tokens==old.tokens && now.pending==old.pending && Pair::image(now.ranks[0])->frontier==Pair::image(old.ranks[0])->frontier &&
              Pair::image(now.ranks[0])->rows==Pair::image(old.ranks[0])->rows,"failed schema2 install restores retained candidate/cancellation/ranks");p.stop();
    }
}
static void pending_group_shape() {
    Pair p(64ULL<<20,true,true,5);std::array<ss::KvHandle,5> handles;
    ss::SamplerState s;s.max_tokens=9;const std::vector<std::uint32_t> ids{2},suffix{3};
    for(auto& h:handles){h=p.store.acquire();p.engine.prefill(ids,h);(void)p.engine.decode(h,s);}
    const auto controls=p.wire.controls,calls=p.rank0.input_calls;
    std::vector<ss::AdvanceItem> over;
    for(auto& h:handles)over.push_back({h,suffix,s,{},{}});
    refuses([&]{(void)p.engine.advance(over);});
    check(p.wire.controls==controls && p.rank0.input_calls==calls,"ten physical rows refuse before mutation");
    ss::AdvanceItem cap{handles[0],suffix,s,{},ss::RowHints{1,1}};
    refuses([&]{(void)p.engine.advance(std::span(&cap,1));});
    check(p.wire.controls==controls,"impossible pending suffix cap refuses before controls");
    cap.row_hints=ss::RowHints{2,1.5};const auto anchor=*p.store.pending(handles[0]);
    auto result=p.engine.advance(std::span(&cap,1));
    check(result[0].input_tokens==1 && result[0].output.tokens.empty() && result[0].rows.physical_rows==2 &&
          p.rank0.input_calls==calls+1 && p.rank0.input_trace.back().size()==2,"pending and suffix consume in ONE causal native group");
    boundary(p,handles[0],{2,anchor,3});
    // All eight physical rows fit in one call, attributed once to each lane.
    std::array<ss::AdvanceItem,4> eight{{{handles[1],suffix,s,{},{}},{handles[2],suffix,s,{},{}},
                                      {handles[3],suffix,s,{},{}},{handles[4],suffix,s,{},{}}}};
    auto grouped=p.engine.advance(eight);check(p.rank0.input_calls==calls+2 && p.rank0.input_trace.back().size()==8,"exact eight-row physical bound");
    for(auto& r:grouped)check(r.input_tokens==1 && r.rows.physical_rows==2 && r.rows.useful_rows==2 && r.rows.padding_rows==0,"per-lane physical attribution");
    p.stop();
}
int main()try {
    full_and_mixed();policies_and_lifecycle();validation();context_and_empty();source_lifetime();result_and_transport_faults();pending_group_shape();
    for(unsigned r=0;r<2;++r){failed_rows(r,false);failed_rows(r,true);}
    std::cout<<"continuation checks "<<checks<<" (always active)\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
