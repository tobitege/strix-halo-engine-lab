// Share the loaded tiny-model fixture, fault observers and independent causal
// reference with continuation tests. The existing assertions remain active in
// their own executable; no second target or sampler is implemented here.
#define STRIX_G4_CONTINUATION_TEST_SUPPORT
#include "continuation_tests.cpp"

namespace {
void seal_reference(Reference& ref) {
    if(ref.pending) {ref.consume(*ref.pending);ref.pending.reset();}
}
e::Request request(std::vector<std::uint32_t> ids,const s::SamplerState& p,std::size_t allowance) {
    e::Request r;r.prompt=std::move(ids);r.max_tokens=allowance;r.sampling=policy(p);
    r.stop_ids.assign(p.stop_tokens.begin(),p.stop_tokens.end());r.dflash=e::DraftMode::off;return r;
}
void legacy_run(e::SessionEngine& engine,s::KvHandle h,const e::Request& r,std::size_t offset=0) {
    auto p=sampler(r.sampling.temperature!=0,r.sampling.draws);
    p.parameters.temperature=r.sampling.temperature;p.parameters.top_p=r.sampling.top_p;
    p.parameters.top_k=r.sampling.top_k;p.parameters.seed=r.sampling.seed;p.stop_tokens=r.stop_ids;
    while(engine.result(h).status==e::Status::running) {
        p.max_tokens=std::uint32_t(r.max_tokens-engine.result(h).ids.size());
        s::AdvanceItem item{h,std::span(r.prompt).subspan(offset),p,{}};
        auto out=engine.advance({&item,1}).at(0);CHECK(out.error.empty());
        offset+=out.input_tokens;p.rng_draws=out.output.rng_draws;
    }
}
void compare_committed(e::SessionEngine& engine,s::KvHandle h,const Reference& ref) {
    equivalent(engine,h,ref);CHECK(bits(engine.retained_logits(h),ref.logits));
    const auto snapshot=engine.store().capture(h);
    CHECK(snapshot.sealed() && !snapshot.pending && snapshot.tokens==ref.consumed);
    CHECK(snapshot.consumed==ref.state.frontier());
}
void terminal_matrix() {
    auto model=tiny();Tables tables;
    for(bool legacy:{false,true})for(bool sampled:{false,true})for(bool eos:{false,true})
    for(auto n:{1U,7U,8U,9U,15U,16U,17U,25U}) {
        Observed backend(model,tables.view());std::mutex mutex;e::SessionEngine engine(backend,{}, {},mutex,2,0,0);
        CHECK(engine.caps().turn_commit && !engine.caps().prefix_cache);
        auto h=engine.store().acquire();auto ids=input(n);auto p=sampler(sampled);p.max_tokens=eos?5:3;
        // Explicit thinking/delimiter IDs belong to the consumed transcript,
        // independent of presentation filtering. They use ordinary target math.
        ids.front()=32;ids.back()=33;
        Reference ref(model,tables.view(),policy(p));ref.append(ids);
        const auto first=ref.decode();const std::array stops{first};if(eos)p.stop_tokens=stops;
        if(!eos){ref.decode();ref.decode();}
        const auto r=request(ids,p,p.max_tokens);
        if(legacy) {engine.prepare(h,r);engine.start(h);legacy_run(engine,h,r);}
        else {
            engine.prefill(ids,h);
            do {auto out=engine.decode(h,p);p.rng_draws=out.rng_draws;p.max_tokens=std::uint32_t(r.max_tokens-engine.result(h).ids.size());}
            while(engine.result(h).status==e::Status::running);
        }
        equivalent(engine,h,ref);CHECK(!engine.turn_cache_candidate(h));
        const auto before=engine.result(h);const auto count=backend.trace.size();
        const auto pending=engine.store().pending(h);CHECK(pending);
        engine.commit_pending(h);seal_reference(ref);compare_committed(engine,h,ref);
        const auto& after=engine.result(h);
        CHECK(after.status==(eos?e::Status::stopped:e::Status::length));
        CHECK(after.status==before.status && after.ids==before.ids && after.rng_draws==before.rng_draws);
        CHECK(after.acceptance.emitted==before.acceptance.emitted && after.windows==before.windows);
        CHECK(after.target_rows==before.target_rows+1 && after.target_calls==before.target_calls+1);
        CHECK(backend.trace.size()==count+1 && backend.trace.back().position==before.frontier && backend.trace.back().token==*pending);
        CHECK(engine.turn_cache_candidate(h));const auto saved=engine.store().capture_prompt(h);
        CHECK(saved.sealed() && saved.purpose==s::SnapshotPurpose::prompt_reuse && saved.tokens==ref.consumed);
        CHECK(!engine.cache_candidate(h)); // ordinary Host admission stays OFF
        CHECK(engine.turn_cache_candidate(h));
        const auto receipt=engine.lane_receipt(h).dump();const auto calls=backend.calls;
        engine.commit_pending(h);CHECK(backend.calls==calls && engine.lane_receipt(h).dump()==receipt);
        CHECK(engine.store().capture_prompt(h).tokens==saved.tokens);
        refuses([&]{engine.decode(h,p);});refuses([&]{engine.append_external(ids,h);});
        refuses([&]{engine.prefill({},h);}); // no implicit resume
        if(!legacy) {
            const auto sealed=engine.store().capture(h);
            engine.resume_lane(h);CHECK(!engine.turn_cache_candidate(h));
            // Sealing and resuming are different states: append requires resume.
            engine.append_external(std::span(ids).first(1),h);
            ref.append(std::span(ids).first(1));equivalent(engine,h,ref);
            engine.store().restore(h,sealed);
            const auto restored_calls=backend.calls;engine.commit_pending(h);
            CHECK(backend.calls==restored_calls && engine.result(h).status==before.status);
            CHECK(engine.store().tokens(h)==sealed.tokens && !engine.store().pending(h));
        }
        engine.store().release(h);
    }
}
void empty_and_context() {
    auto model=tiny();Tables tables;Observed backend(model,tables.view());std::mutex mutex;
    e::SessionEngine engine(backend,{}, {},mutex,2,0,0);
    for(auto n:{8U,95U,96U}) {
        auto h=engine.store().acquire();auto ids=input(n);auto p=sampler(true);
        engine.prefill(ids,h);Reference ref(model,tables.view(),policy(p));ref.append(ids);
        p.max_tokens=n==95?1:0;
        auto out=engine.decode(h,p);
        if(n==95) {
            CHECK(out.tokens==std::vector<std::uint32_t>{ref.decode()});
            // Already admitted anchor occupies the last position, no extra
            // output reservation is needed by terminal commit.
            CHECK(engine.store().consumed(h)==95 && engine.store().pending(h));
        } else CHECK(out.tokens.empty());
        const auto before=engine.result(h);const auto calls=backend.calls;
        if(n==8) {backend.enabled=false;refuses([&]{engine.commit_pending(h);});CHECK(backend.calls==calls);backend.enabled=true;}
        engine.commit_pending(h);seal_reference(ref);compare_committed(engine,h,ref);
        CHECK(backend.calls==calls+std::size_t(n==95));
        CHECK(engine.result(h).ids==before.ids && engine.result(h).rng_draws==before.rng_draws && engine.result(h).status==before.status);
        CHECK(engine.store().consumed(h)==(n==95?96:n));engine.commit_pending(h);
        if(n==8) {
            engine.resume_lane(h);p.max_tokens=1;
            const auto resumed_calls=backend.calls;auto selected=engine.decode(h,p);
            CHECK(backend.calls==resumed_calls && selected.tokens==std::vector<std::uint32_t>{ref.decode()});
            engine.commit_pending(h);seal_reference(ref);compare_committed(engine,h,ref);
        }
        engine.store().release(h);
    }
    // Context exit with zero selection from a completely consumed input.
    auto h=engine.store().acquire();engine.prefill(input(96),h);auto p=sampler(true);
    auto out=engine.decode(h,p);CHECK(out.finish==s::FinishReason::context && out.tokens.empty());
    const auto calls=backend.calls;engine.commit_pending(h);CHECK(backend.calls==calls);
    CHECK(engine.turn_cache_candidate(h));
}
// Host fault wrapper exercises unknown native accounting. It does not execute
// or qualify a device target and must never advertise a native scratch bound.
void unknown_native_accounting() {
    auto model=tiny();Tables tables;Observed backend(model,tables.view());backend.native_accounting=true;
    backend.faults=std::make_shared<Faults>();backend.faults->unknown_size=true;
    std::mutex mutex;e::SessionEngine engine(backend,{}, {},mutex,1,0,0);auto h=engine.store().acquire();
    engine.prefill(input(9),h);auto p=sampler();p.max_tokens=1;engine.decode(h,p);engine.commit_pending(h);
    CHECK(engine.caps().turn_commit);CHECK(!engine.turn_work_bound() && !engine.turn_retained_bound(h));
    CHECK(engine.store().capture(h).bytes==0);
    const auto forks=backend.faults->forks;CHECK(!engine.turn_cache_candidate(h));
    CHECK(backend.faults->forks==forks); // refuse before allocating a candidate
}
void prompt_restart() {
    auto model=tiny();Tables tables;Observed backend(model,tables.view());std::mutex mutex;
    e::SessionEngine engine(backend,{}, {},mutex,3,0,0);
    for(bool legacy:{false,true}) {
        auto h=engine.store().acquire();auto p=sampler(true);p.max_tokens=4;auto ids=input(17);
        ids.insert(ids.end(),{32,34,35,33}); // evaluated thinking block + delimiter
        Reference old(model,tables.view(),policy(p));old.append(ids);for(int i=0;i<4;++i)old.decode();
        auto r=request(ids,p,4);
        if(legacy){engine.prepare(h,r);engine.start(h);legacy_run(engine,h,r);}
        else {engine.prefill(ids,h);while(engine.result(h).status==e::Status::running){auto o=engine.decode(h,p);p.rng_draws=o.rng_draws;p.max_tokens=std::uint32_t(4-engine.result(h).ids.size());}}
        engine.commit_pending(h);seal_reference(old);CHECK(engine.turn_cache_candidate(h));
        const auto saved=engine.store().capture_prompt(h);const auto old_bytes=saved.bytes;
        CHECK(saved.tokens==old.consumed && saved.tokens.size()==ids.size()+4);
        const auto prefix=std::static_pointer_cast<const e::PrefixSnapshot>(saved.ranks[0].owner);
        CHECK(prefix->purpose==e::PrefixSnapshot::Purpose::complete_turn);
        CHECK(bits(prefix->logits,old.logits) && bits(prefix->hidden,old.hidden) && bits(prefix->taps,old.taps));
        const auto frozen_digest=prefix->state->digest();const auto snapshot=engine.store().capture(h);
        refuses([&]{engine.store().restore_prompt(h,snapshot);}); // resume purpose cannot be reinterpreted
        engine.store().release(h);auto stale=h;
        h=engine.store().acquire();auto full=saved.tokens;auto suffix=input(9);full.insert(full.end(),suffix.begin(),suffix.end());
        auto fresh=sampler(true,2);fresh.parameters.seed=744;fresh.max_tokens=3;r=request(full,fresh,3);
        engine.prepare(h,r);const auto calls=backend.calls,trace_before=backend.trace.size();
        CHECK(engine.store().restore_prompt(h,saved));engine.start(h);
        CHECK(backend.calls==calls && engine.result(h).ids.empty() && engine.result(h).rng_draws==2);
        CHECK(digest(engine,h)==frozen_digest);
        CHECK(engine.result(h).status==e::Status::running && engine.result(h).cached_tokens==saved.consumed);
        legacy_run(engine,h,r,std::size_t(saved.consumed));
        Reference ref(model,tables.view(),policy(fresh));ref.append(full);for(int i=0;i<3;++i)ref.decode();
        equivalent(engine,h,ref);engine.commit_pending(h);seal_reference(ref);compare_committed(engine,h,ref);
        CHECK(backend.trace.size()==trace_before+suffix.size()+3);
        const auto trace_begin=backend.trace.size()-suffix.size()-3;
        // New suffix rows plus generated anchors only. The final commit adds
        // the last generated anchor; not one cached prefix row is evaluated.
        CHECK(backend.trace[trace_begin].position==saved.consumed);
        CHECK(backend.trace[trace_begin].token==suffix.front());
        CHECK(prefix->state->digest()==frozen_digest && prefix->tokens==saved.tokens && saved.bytes==old_bytes);
        refuses([&]{engine.commit_pending(stale);});engine.store().release(h);
        for(int miss=0;miss<4;++miss) {
            auto next=engine.store().acquire();auto mismatch=saved.tokens;
            if(miss!=0)mismatch.push_back(4);if(miss==1)mismatch.front()=(mismatch.front()+1)%37;
            auto req=request(mismatch,fresh,2);if(miss==3)req.dflash=e::DraftMode::on;engine.prepare(next,req);
            const auto creates=backend.creates,calls_before=backend.calls;
            if(miss==2)backend.enabled=false;
            CHECK(!engine.store().restore_prompt(next,saved));
            CHECK(backend.creates==creates && backend.calls==calls_before && engine.store().consumed(next)==0);
            if(miss==3){engine.store().release(next);continue;}
            backend.enabled=true;engine.start(next);legacy_run(engine,next,req);engine.store().release(next);
        }
        // A forged payload or foreign store never reaches an erased cast.
        auto forged=saved;forged.tokens.front()^=1;auto next=engine.store().acquire();
        refuses([&]{engine.store().restore_prompt(next,forged);});engine.store().release(next);
        std::mutex other_mutex;e::SessionEngine other(backend,{}, {},other_mutex,1,0,0);auto foreign=other.store().acquire();
        refuses([&]{other.store().restore_prompt(foreign,saved);});
    }
}
void eos_and_legacy_capacity() {
    auto model=tiny();Tables tables;
    for(bool legacy:{false,true}) {
        Observed backend(model,tables.view());std::mutex mutex;e::SessionEngine engine(backend,{}, {},mutex,1,0,0);
        auto h=engine.store().acquire();auto ids=input(9);auto p=sampler(true);
        p.parameters.temperature=10;p.parameters.top_k=0;p.parameters.top_p=1;p.max_tokens=4;
        const std::array<std::uint32_t,1> eos{1};p.stop_tokens=eos;
        Reference ref(model,tables.view(),policy(p));ref.append(ids);
        // Find a seeded policy that actually emits EOS=1 using the existing
        // sampler over independent logits. No target row is replayed.
        bool found=false;
        for(std::uint64_t seed=0;seed<10000;++seed) {
            p.parameters.seed=seed;auto draws=p.rng_draws;
            if(e::select_token(ref.logits,policy(p),draws)==1){found=true;break;}
        }
        CHECK(found);ref.sampling=policy(p);CHECK(ref.decode()==1);
        const auto r=request(ids,p,4);
        if(legacy){engine.prepare(h,r);engine.start(h);legacy_run(engine,h,r);}else{engine.prefill(ids,h);engine.decode(h,p);}
        CHECK(engine.result(h).status==e::Status::stopped && engine.store().pending(h)==1);
        engine.commit_pending(h);seal_reference(ref);compare_committed(engine,h,ref);
    }
    for(bool sampled:{false,true}) {
        Observed backend(model,tables.view());std::mutex mutex;e::SessionEngine engine(backend,{}, {},mutex,1,0,0);
        auto h=engine.store().acquire();auto p=sampler(sampled);auto ids=input(95);auto r=request(ids,p,1);
        Reference ref(model,tables.view(),policy(p));ref.append(ids);ref.decode();
        engine.prepare(h,r);engine.start(h);legacy_run(engine,h,r);
        CHECK(engine.store().consumed(h)==95 && engine.store().pending(h));
        engine.commit_pending(h);seal_reference(ref);compare_committed(engine,h,ref);
        CHECK(engine.store().consumed(h)==96);
    }
}
void retained_accounting() {
    auto model=tiny();Tables tables;e::HostBackend backend(model,tables.view(),1);std::mutex mutex;
    e::SessionEngine engine(backend,{}, {},mutex,2,0,0);auto h=engine.store().acquire();
    engine.prefill(input(1),h);const auto small=engine.store().capture(h);
    engine.append_external(input(24),h);const auto large=engine.store().capture(h);
    CHECK(large.bytes>small.bytes+24*sizeof(std::uint32_t));
    std::uint64_t kv_only=0;for(const auto& layer:model.config.layer)
        kv_only+=(layer.attention==g::AttentionKind::global?25:8)*(2*8*sizeof(float)+sizeof(std::uint64_t));
    CHECK(large.bytes>kv_only); // full Slot, final outputs, requests/history and stream metadata
    auto p=sampler();p.max_tokens=0;engine.decode(h,p);engine.commit_pending(h);CHECK(engine.turn_cache_candidate(h));
    auto candidate=engine.store().capture_prompt(h);
    const auto prefix=std::static_pointer_cast<const e::PrefixSnapshot>(candidate.ranks[0].owner);
    CHECK(prefix->retained_bytes()==candidate.bytes && candidate.bytes>kv_only);
    std::cout<<"retained payload bytes: one-row="<<small.bytes<<" twenty-five-row="<<large.bytes
        <<" turn-candidate="<<candidate.bytes<<" KV-only="<<kv_only<<'\n';
    const auto state=backend.create_stream();const auto empty=state->retained_bytes();CHECK(empty);
    auto stream=backend.create_stream();
    for(auto id:input(25)){auto tx=stream->begin();g::RowPlan row{0,1,stream->frontier(),id,tx.get(),0};backend.forward_rows({&row,1});tx->commit(1);}
    CHECK(*stream->retained_bytes()>*empty);
    // Model geometry affects retained allocations at the same frontier.
    auto bigger=tiny();bigger.config.window=16;e::HostBackend wide(bigger,tables.view(),1);
    CHECK(*wide.create_stream()->retained_bytes()>*empty);
    class Unknown final:public e::StreamState {
        std::unique_ptr<g::KvTxn> begin()override{return {};}
        std::uint64_t frontier()const override{return 0;}
        std::string digest()const override{return {};}
    } unknown;
    CHECK(!unknown.retained_bytes());
    // Hold all physical row allocations fixed after ring wrap. Only retained
    // causal history grows; a frontier-only KV formula cannot explain this.
    auto sliding=tiny();for(auto& layer:sliding.config.layer)layer.attention=g::AttentionKind::sliding;
    e::HostBackend rings(sliding,tables.view(),1);std::mutex ring_mutex;
    e::SessionEngine history(rings,{}, {},ring_mutex,1,0,0);auto lane=history.store().acquire();
    history.prefill(input(9),lane);auto zero=sampler();zero.max_tokens=0;history.decode(lane,zero);
    history.commit_pending(lane);CHECK(history.turn_cache_candidate(lane));auto first=history.store().capture_prompt(lane);
    const auto first_payload=std::static_pointer_cast<const e::PrefixSnapshot>(first.ranks[0].owner);
    const auto first_digest=first_payload->state->digest();
    history.resume_lane(lane);history.append_external(input(16),lane);history.decode(lane,zero);
    history.commit_pending(lane);CHECK(history.turn_cache_candidate(lane));auto second=history.store().capture_prompt(lane);
    const auto second_payload=std::static_pointer_cast<const e::PrefixSnapshot>(second.ranks[0].owner);
    CHECK(first_payload->state->retained_bytes()==second_payload->state->retained_bytes());
    CHECK(second.bytes>first.bytes && first.tokens.size()==9 && second.tokens.size()==25);
    CHECK(first_payload->state->digest()==first_digest);
}
void commit_refusal_and_rollback() {
    auto model=tiny();Tables tables;Observed backend(model,tables.view());std::mutex mutex;
    e::SessionEngine engine(backend,{}, {},mutex,3,0,0);auto h=engine.store().acquire(),peer=engine.store().acquire();
    auto ids=input(17);engine.prefill(ids,peer);const auto peer_digest=digest(engine,peer);
    refuses([&]{engine.commit_pending(h);});engine.prefill(ids,h);auto p=sampler(true);p.max_tokens=1;
    refuses([&]{engine.commit_pending(h);});engine.decode(h,p);
    const auto original=engine.store().capture(h);const auto before=engine.result(h);const auto prior=digest(engine,h);
    backend.fail_call=backend.calls+1;refuses([&]{engine.commit_pending(h);});backend.fail_call=0;
    CHECK(engine.store().tokens(h)==original.tokens && engine.store().pending(h)==original.pending);
    CHECK(engine.result(h).ids==before.ids && engine.result(h).rng_draws==before.rng_draws && engine.result(h).status==before.status);
    CHECK(digest(engine,h)==prior && digest(engine,peer)==peer_digest);
    bool reentered=false;backend.on_call=[&]{reentered=true;CHECK(!engine.turn_work_bound());
        refuses([&]{engine.commit_pending(h);});refuses([&]{(void)engine.store();});};
    engine.commit_pending(h);backend.on_call={};CHECK(reentered && !engine.store().pending(h));
    auto stale=h;engine.store().release(h);h=engine.store().acquire();refuses([&]{engine.commit_pending(stale);});
    engine.prefill(ids,h);engine.cancel_lane(h);refuses([&]{engine.commit_pending(h);});engine.store().release(h);
    h=engine.store().acquire();engine.prefill(ids,h);engine.decode(h,p);engine.store().invalidate(h);
    refuses([&]{engine.commit_pending(h);});engine.store().release(h);
    std::mutex other_mutex;e::SessionEngine other(backend,{}, {},other_mutex,1,0,0);auto foreign=other.store().acquire();
    refuses([&]{engine.commit_pending(foreign);});
    // Backend and configured drafter refusal applies even with no pending ID.
    backend.enabled=false;e::SessionEngine disabled(backend,{}, {},mutex,1,0,0);auto d=disabled.store().acquire();
    CHECK(!disabled.caps().turn_commit);refuses([&]{disabled.commit_pending(d);});backend.enabled=true;
    auto alternate=tiny();alternate.identity="unqualified";e::HostBackend wrong(alternate,tables.view(),1);
    e::SessionEngine identity(wrong,{}, {},mutex,1,0,0);auto u=identity.store().acquire();
    CHECK(!identity.caps().turn_commit);refuses([&]{identity.commit_pending(u);});
    e::DraftFactory factory=[]()->std::unique_ptr<e::DraftSession>{throw std::runtime_error("configured factory must not run");};
    e::SessionEngine drafted(backend,factory,{},mutex,1,0,0);auto a=drafted.store().acquire();
    auto req=request(ids,p,1);drafted.prepare(a,req);drafted.start(a);legacy_run(drafted,a,req);
    CHECK(!drafted.caps().turn_commit);const auto calls=backend.calls;refuses([&]{drafted.commit_pending(a);});CHECK(backend.calls==calls);
    // Failed legacy lanes are terminal but never eligible for commit.
    auto f=engine.store().acquire();engine.prepare(f,req);engine.start(f);backend.fail_call=backend.calls+1;
    refuses([&]{legacy_run(engine,f,req);});
    backend.fail_call=0;CHECK(engine.result(f).status==e::Status::failed);refuses([&]{engine.commit_pending(f);});
}
void publication_rollback() {
    auto model=tiny();Tables tables;
    for(bool legacy:{false,true})for(bool unrecoverable:{false,true}) {
        Observed backend(model,tables.view());backend.faults=std::make_shared<Faults>();std::mutex mutex;
        e::SessionEngine engine(backend,{}, {},mutex,2,0,0);auto h=engine.store().acquire(),peer=engine.store().acquire();
        auto ids=input(17);auto p=sampler(true);p.max_tokens=1;auto r=request(ids,p,1);
        engine.prefill(ids,peer);const auto peer_digest=digest(engine,peer);
        if(legacy){engine.prepare(h,r);engine.start(h);legacy_run(engine,h,r);}else{engine.prefill(ids,h);engine.decode(h,p);}
        const auto snapshot=engine.store().capture(h);const auto before=engine.result(h);const auto prior=digest(engine,h);
        backend.faults->fail_commit=backend.faults->commits+1;backend.faults->fail_restore=unrecoverable;
        refuses([&]{engine.commit_pending(h);});CHECK(digest(engine,peer)==peer_digest);
        if(unrecoverable) {refuses([&]{(void)engine.store().consumed(h);});engine.store().release(h);}
        else {
            CHECK(digest(engine,h)==prior && engine.store().pending(h)==snapshot.pending && engine.store().tokens(h)==snapshot.tokens);
            CHECK(engine.result(h).ids==before.ids && engine.result(h).rng_draws==before.rng_draws && engine.result(h).status==before.status);
            // Fail after Coordinator has updated frontier, retained outputs and
            // sealed metadata, before adapter publication. Restore all of it.
            backend.faults->failed=false;backend.faults->fail_digest=true;
            refuses([&]{engine.commit_pending(h);});backend.faults->fail_digest=false;
            CHECK(digest(engine,h)==prior && engine.store().pending(h)==snapshot.pending);
            CHECK(engine.result(h).frontier==before.frontier && engine.result(h).ids==before.ids);
            engine.commit_pending(h);CHECK(engine.turn_cache_candidate(h));const auto prepared=engine.store().capture_prompt(h);
            // Optional capture failure leaves delivered output and the prepared
            // immutable candidate intact. No resume or output replay occurs.
            backend.faults->fail_restore=true;
            refuses([&]{engine.turn_cache_candidate(h);});
            CHECK(engine.store().capture_prompt(h).tokens==prepared.tokens && engine.result(h).ids==before.ids);
            engine.commit_pending(h);engine.store().release(h);
        }
        backend.faults->failed=false;backend.faults->fail_restore=false;
        auto reused=engine.store().acquire();engine.prefill(input(2),reused);CHECK(digest(engine,peer)==peer_digest);
    }
}
}
int main() {
    try {
        std::cout<<"terminal matrix\n";terminal_matrix();
        std::cout<<"empty/context\n";empty_and_context();
        std::cout<<"EOS and legacy capacity\n";eos_and_legacy_capacity();
        std::cout<<"new request prompt restart\n";prompt_restart();
        std::cout<<"retained accounting\n";retained_accounting();
        std::cout<<"unknown native accounting (host wrapper)\n";unknown_native_accounting();
        std::cout<<"refusal/evaluation rollback\n";commit_refusal_and_rollback();
        std::cout<<"publication rollback\n";publication_rollback();
        std::cout<<"real host final-turn commit and complete prompt reuse PASS\n";return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
