#include "gemma4/engine/session_engine.hpp"
#include "gemma4/engine/host_backend.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>

namespace g = strix::gemma4;
namespace e = g::engine;
namespace s = strix::session;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string(#x)+" line "+std::to_string(__LINE__)); } while(false)
template<class F> void refuses(F f) {
    bool caught=false; try { f(); } catch(const std::exception&) {caught=true;} CHECK(caught);
}
namespace {
// Ordinary loaded-family weights and real HostTarget evaluation; no model math
// or synthetic StreamState is implemented by this test.
g::ModelManifest tiny() {
    g::ModelManifest m; auto& c=m.config;
    c.hidden=16;c.layers=2;c.inter=24;c.vocab=37;c.heads=4;c.kv_heads=1;
    c.ple_dim=8;c.window=8;c.max_positions=96;
    c.layer={{g::AttentionKind::sliding,8,0},{g::AttentionKind::global,8,1}};
    std::mt19937 rng(471);
    for(const auto& [name,shape]:g::required_tensors(c)) {
        const auto outputs=shape.size()==2?std::uint32_t(shape[0]):1U, inputs=std::uint32_t(shape.back());
        auto values=std::make_shared<std::vector<float>>(std::size_t(outputs)*inputs);
        for(auto& x:*values) {
            x=float(int(rng()%127)-63)*0.0009765625F;
            if(name.find("norm")!=std::string::npos)x+=0.5F;
            if(name.ends_with("layer_scalar"))x=0.875F;
        }
        m.tensors.push_back({name,shape,{reinterpret_cast<const std::uint8_t*>(values->data()),outputs,inputs,inputs*4ULL,g::WeightFormat::f32},{}});
        m.storage.push_back(values);
    }
    return m;
}
struct Tables {
    std::vector<float> sliding=std::vector<float>(96*256),global=std::vector<float>(96*512);
    Tables(){for(std::size_t i=0;i<sliding.size();i+=2)sliding[i]=1;for(std::size_t i=0;i<global.size();i+=2)global[i]=1;}
    g::RopeTables view()const{return {sliding,global,96,"test-rotation"};}
};
std::vector<std::uint32_t> input(std::size_t n) {
    std::vector<std::uint32_t> ids;for(std::size_t i=0;i<n;++i)ids.push_back(std::uint32_t((i*7+2)%37));return ids;
}
bool bits(std::span<const float> a,std::span<const float> b) {
    return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size_bytes())==0);
}
struct Faults {
    std::size_t commits=0,fail_commit=0;
    bool failed=false,fail_restore=false,fail_digest=false,unknown_size=false;
    std::size_t forks=0;
};
class ObservedTxn final:public g::KvTxn {
public:
    ObservedTxn(std::unique_ptr<g::KvTxn> tx,std::shared_ptr<Faults> faults):physical(std::move(tx)),faults_(std::move(faults)){}
    std::uint64_t frontier()const override{return physical->frontier();}
    void reserve_rows(std::size_t n)override{physical->reserve_rows(n);}
    void commit(std::size_t n)override {
        physical->commit(n);
        if(faults_->fail_digest)faults_->failed=true;
        if(++faults_->commits==faults_->fail_commit){faults_->failed=true;throw std::runtime_error("injected post-commit failure");}
    }
    std::unique_ptr<g::KvTxn> physical;
private:
    std::shared_ptr<Faults> faults_;
};
class ObservedStream final:public e::StreamState {
public:
    ObservedStream(std::unique_ptr<e::StreamState> state,std::shared_ptr<Faults> faults):physical_(std::move(state)),faults_(std::move(faults)){}
    std::unique_ptr<g::KvTxn> begin()override{return std::make_unique<ObservedTxn>(physical_->begin(),faults_);}
    std::uint64_t frontier()const override{return physical_->frontier();}
    std::string digest()const override{
        if(faults_->failed && faults_->fail_digest)throw std::runtime_error("injected final publication failure");
        return physical_->digest();
    }
    std::optional<std::uint64_t> retained_bytes()const override {
        if(faults_->unknown_size)return std::nullopt;
        const auto bytes=physical_->retained_bytes();if(!bytes)return std::nullopt;
        CHECK(*bytes<=UINT64_MAX-sizeof(ObservedStream)-sizeof(Faults));
        return *bytes+sizeof(ObservedStream)+sizeof(Faults);
    }
    std::unique_ptr<e::StreamState> fork(std::uint32_t n)const override {
        ++faults_->forks;
        if(faults_->failed && faults_->fail_restore)throw std::runtime_error("injected rollback copy failure");
        return std::make_unique<ObservedStream>(physical_->fork(n),faults_);
    }
private:
    std::unique_ptr<e::StreamState> physical_;
    std::shared_ptr<Faults> faults_;
};
class Observed final:public e::Backend {
public:
    Observed(const g::ModelManifest& m,g::RopeTables t):host(m,t,1){}
    bool input_continuation_supported()const override{return enabled && host.input_continuation_supported();}
    std::unique_ptr<e::StreamState> create_stream()override{
        ++creates;auto state=host.create_stream();
        if(faults)return std::make_unique<ObservedStream>(std::move(state),faults);
        return state;
    }
    const g::ModelConfig& config()const override{return host.config();}
    strix::serve::Json identity()const override{
        auto j=host.identity();if(native_accounting)j.set("backend","gfx1151-device");return j;
    }
    std::size_t prefill_chunk_rows()const override{return prefill_chunk;}
    e::PrefillRows prefill_rows(g::KvTxn&,std::span<const std::uint32_t>,bool,bool)override {
        ++prefill_calls;throw std::runtime_error("unexpected separate prefill in mixed continuation");
    }
    g::ForwardResult forward_rows(std::span<const g::RowPlan> rows)override {
        ++calls; widths.push_back(rows.size());
        for(const auto& row:rows) trace.push_back({row.stream,row.position,row.token});
        if(on_call)on_call();
        std::vector<g::RowPlan> physical(rows.begin(),rows.end());
        for(auto& row:physical)if(auto* tx=dynamic_cast<ObservedTxn*>(row.transaction))row.transaction=tx->physical.get();
        auto result=host.forward_rows(physical);
        if(device_checked) {
            result.device_checked=true;
            for(std::size_t i=0;i<result.rows;++i) {
                std::uint64_t draws=0;
                result.selected.push_back(e::select_token(std::span(result.logits).subspan(i*result.vocab,result.vocab),{},draws));
                CHECK(draws==0);
            }
        }
        if(fail_call==calls)throw std::runtime_error("injected failure after target evaluation");
        return result;
    }
    struct Row{std::uint64_t slot,position;std::uint32_t token;};
    e::HostBackend host;
    bool enabled=true,device_checked=false,native_accounting=false;
    std::size_t calls=0,creates=0,fail_call=0;
    std::size_t prefill_chunk=0,prefill_calls=0;
    std::vector<std::size_t> widths;
    std::vector<Row> trace;
    std::function<void()> on_call;
    std::shared_ptr<Faults> faults;
};
// Independent serial execution of the exact causal transcript. The family
// sampler is shared deliberately; scheduling/history/retention are independent.
struct Reference {
    g::HostTarget target;
    g::HostStream state;
    std::vector<std::uint32_t> consumed,generated;
    std::optional<std::uint32_t> pending;
    std::vector<float> logits,hidden,taps;
    e::Sampling sampling;
    std::uint64_t draws;
    Reference(const g::ModelManifest& m,g::RopeTables t,e::Sampling policy)
        :target(g::bind_host_weights(m),t,1),state(m.config),sampling(policy),draws(policy.draws){}
    void consume(std::uint32_t id) {
        auto tx=state.begin();g::RowPlan row{0,1,state.frontier(),id,tx.get(),0,true,true,true,true};
        auto result=target.forward_rows({&row,1});tx->commit(1);logits=std::move(result.logits);
        hidden=std::move(result.hidden);taps=std::move(result.taps);consumed.push_back(id);
    }
    void append(std::span<const std::uint32_t> ids) {
        if(ids.empty())return;
        if(pending){consume(*pending);pending.reset();}
        for(auto id:ids)consume(id);
    }
    std::uint32_t decode() {
        if(pending){consume(*pending);pending.reset();}
        auto id=e::select_token(logits,sampling,draws);generated.push_back(id);pending=id;return id;
    }
};
s::SamplerState sampler(bool sampled=false,std::uint64_t cursor=11) {
    s::SamplerState p;p.parameters.seed=931;p.parameters.seeded=true;
    p.parameters.temperature=sampled?0.8:0;p.parameters.top_p=0.75;p.parameters.top_k=7;
    p.rng_draws=cursor;p.max_tokens=8;return p;
}
e::Sampling policy(const s::SamplerState& p) {
    return {float(p.parameters.temperature),float(p.parameters.top_p),p.parameters.top_k,p.parameters.seed,p.rng_draws};
}
std::string digest(e::SessionEngine& engine,s::KvHandle h) {
    return engine.lane_receipt(h).find("state_digest")->as_string();
}
void equivalent(e::SessionEngine& engine,s::KvHandle h,const Reference& ref) {
    CHECK(engine.store().tokens(h)==ref.consumed);CHECK(engine.store().pending(h)==ref.pending);
    CHECK(engine.result(h).ids==ref.generated);CHECK(engine.result(h).rng_draws==ref.draws);
    CHECK(digest(engine,h)==ref.state.digest());
}
void boundaries() {
    auto model=tiny();Tables tables;
    for(bool sampled:{false,true})for(auto n:{1U,7U,8U,9U,15U,16U,17U,25U}) {
        Observed backend(model,tables.view());std::mutex mutex;e::SessionEngine engine(backend,{}, {},mutex,2,0,0);
        auto h=engine.store().acquire();auto ids=input(n);auto p=sampler(sampled);
        Reference ref(model,tables.view(),policy(p));ref.append(ids);
        engine.prefill(ids,h);
        CHECK(engine.result(h).ids.empty());CHECK(engine.result(h).rng_draws==0);
        CHECK(engine.store().consumed(h)==n);CHECK(!engine.store().pending(h));
        CHECK(bits(engine.retained_logits(h),ref.logits));CHECK(digest(engine,h)==ref.state.digest());
        CHECK(backend.trace.size()==n);CHECK(backend.calls==(n+7)/8);
        for(std::size_t i=0;i<n;++i){CHECK(backend.trace[i].position==i);CHECK(backend.trace[i].token==ids[i]);}
        const auto calls=backend.calls;auto result=engine.decode(h,p);
        CHECK(result.tokens==std::vector<std::uint32_t>{ref.decode()});CHECK(backend.calls==calls);
        CHECK(result.before==(s::Frontier{n,{}}));CHECK(result.after==(s::Frontier{n,result.tokens.back()}));
        CHECK(engine.retained_logits(h).empty());equivalent(engine,h,ref);
        for(int i=0;i<3;++i){p.rng_draws=result.rng_draws;result=engine.decode(h,p);CHECK(result.tokens==std::vector<std::uint32_t>{ref.decode()});equivalent(engine,h,ref);}
        CHECK(backend.trace.size()==n+3); // first output uses zero new rows
    }
}
void interleaved() {
    auto model=tiny();Tables tables;Observed backend(model,tables.view());std::mutex mutex;
    e::SessionEngine engine(backend,{}, {},mutex,2,0,0);auto h=engine.store().acquire();auto p=sampler(true);
    Reference ref(model,tables.view(),policy(p));auto ids=input(9);engine.prefill(ids,h);ref.append(ids);
    for(int turn=0;turn<4;++turn) {
        for(int j=0;j<2;++j){auto r=engine.decode(h,p);CHECK(r.tokens==std::vector<std::uint32_t>{ref.decode()});p.rng_draws=r.rng_draws;equivalent(engine,h,ref);}
        const auto calls=backend.calls;const auto generated=engine.result(h).ids;
        const auto checkpoint=engine.store().capture(h);const auto old_digest=digest(engine,h);
        engine.append_external({},h);CHECK(backend.calls==calls);CHECK(engine.store().pending(h)==ref.pending);
        auto suffix=input(std::size_t(turn+1));engine.append_external(suffix,h);ref.append(suffix);
        CHECK(engine.result(h).ids==generated);CHECK(engine.result(h).rng_draws==p.rng_draws);equivalent(engine,h,ref);
        CHECK(bits(engine.retained_logits(h),ref.logits));
        // Restore a checkpoint containing a published pending output and replay
        // its append. This also restores policy initialization and output cursors.
        engine.store().restore(h,checkpoint);CHECK(digest(engine,h)==old_digest);
        CHECK(engine.store().pending(h)==checkpoint.pending);CHECK(engine.result(h).ids==generated);
        engine.append_external(suffix,h);equivalent(engine,h,ref);
    }
    CHECK(!engine.store().supports_fork());
}
void terminal_boundaries() {
    auto model=tiny();Tables tables;Observed backend(model,tables.view());std::mutex mutex;
    e::SessionEngine engine(backend,{}, {},mutex,2,0,0);auto h=engine.store().acquire();auto ids=input(8);engine.prefill(ids,h);
    auto p=sampler(true);p.max_tokens=0;auto calls=backend.calls;const auto before=digest(engine,h);
    auto r=engine.decode(h,p);CHECK(r.tokens.empty());CHECK(r.finish==s::FinishReason::length);CHECK(r.rng_draws==11);
    CHECK(backend.calls==calls);CHECK(digest(engine,h)==before);
    refuses([&]{engine.prefill({},h);});refuses([&]{engine.decode(h,p);});engine.resume_lane(h);
    p.max_tokens=8;
    auto logits=engine.retained_logits(h);auto draws=p.rng_draws;const auto stop=e::select_token(logits,policy(p),draws);
    const std::vector<std::uint32_t> stops{stop};p.stop_tokens=stops;
    r=engine.decode(h,p);CHECK(r.tokens==stops);CHECK(r.finish==s::FinishReason::stop);CHECK(backend.calls==calls);
    const auto pending=engine.store().pending(h);refuses([&]{engine.append_external(ids,h);});refuses([&]{engine.decode(h,p);});
    engine.resume_lane(h);engine.append_external(ids,h);CHECK(!engine.store().pending(h));CHECK(engine.result(h).ids==stops);
    CHECK(engine.store().tokens(h)[8]==*pending);CHECK(engine.result(h).rng_draws==draws);
    // A new turn keeps the sampling policy, including the stop set and cursor.
    p.rng_draws=draws;p.stop_tokens={};std::stop_source cancelled;p.cancellation=cancelled.get_token();
    r=engine.decode(h,p);p.rng_draws=r.rng_draws;CHECK(r.tokens.size()==1);cancelled.request_stop();calls=backend.calls;
    refuses([&]{engine.append_external(ids,h);});
    r=engine.decode(h,p);CHECK(r.tokens.empty());CHECK(r.finish==s::FinishReason::cancelled);CHECK(backend.calls==calls);
    refuses([&]{engine.append_external({},h);});engine.resume_lane(h);
    refuses([&]{engine.decode(h,p);}); // already stopped source cannot restart work
    std::stop_source fresh;p.cancellation=fresh.get_token();
    r=engine.decode(h,p);CHECK(r.tokens.size()==1);
    engine.cancel_lane(h);const auto snapshot=engine.store().capture(h);engine.resume_lane(h);
    CHECK(engine.store().consumed(h)==snapshot.consumed);CHECK(engine.store().pending(h)==snapshot.pending);
    auto full=engine.store().acquire();auto all=input(96);engine.prefill(all,full);auto q=sampler();calls=backend.calls;
    auto end=engine.decode(full,q);CHECK(end.finish==s::FinishReason::context);CHECK(end.tokens.empty());CHECK(backend.calls==calls);
    engine.resume_lane(full);refuses([&]{engine.append_external(ids,full);});
    engine.store().release(full);full=engine.store().acquire();all=input(95);engine.prefill(all,full);
    end=engine.decode(full,q);q.rng_draws=end.rng_draws;CHECK(end.tokens.size()==1);calls=backend.calls;
    const auto anchor=engine.store().pending(full);q.max_tokens=0;end=engine.decode(full,q);
    CHECK(end.tokens.empty() && end.finish==s::FinishReason::length);CHECK(engine.store().pending(full)==anchor);
    engine.resume_lane(full);engine.append_external({},full);CHECK(engine.store().pending(full)==anchor);
    refuses([&]{engine.append_external(ids,full);});q.max_tokens=8;end=engine.decode(full,q);
    CHECK(end.tokens.empty() && end.finish==s::FinishReason::context);CHECK(backend.calls==calls);CHECK(engine.store().pending(full)==anchor);
}
void resumed_bounded_policy() {
    auto model=tiny();Tables tables;Observed backend(model,tables.view());std::mutex mutex;
    e::SessionEngine engine(backend,{}, {},mutex,1,0,0);auto h=engine.store().acquire();
    auto ids=input(7);auto p=sampler(true);Reference ref(model,tables.view(),policy(p));
    engine.prefill(ids,h);ref.append(ids);p.max_tokens=1;
    auto result=engine.decode(h,p);CHECK(result.tokens==std::vector<std::uint32_t>{ref.decode()});
    CHECK(result.finish==s::FinishReason::length);p.rng_draws=result.rng_draws;
    engine.resume_lane(h);std::stop_source fresh;p.cancellation=fresh.get_token();
    const std::vector<std::uint32_t> stops{36};p.stop_tokens=stops;p.max_tokens=8;
    auto suffix=input(9);auto calls=backend.calls;
    const s::AdvanceItem first{h,suffix,p,{}};auto admitted=engine.advance({&first,1});
    CHECK(admitted[0].input_tokens==7 && admitted[0].output.tokens.empty());
    CHECK(backend.calls==calls+1 && engine.result(h).rng_draws==p.rng_draws);
    const auto checkpoint=engine.store().capture(h);
    const s::AdvanceItem tail{h,std::span<const std::uint32_t>(suffix).subspan(7),p,{}};
    admitted=engine.advance({&tail,1});CHECK(admitted[0].input_tokens==2);
    engine.store().restore(h,checkpoint);
    admitted=engine.advance({&tail,1});CHECK(admitted[0].input_tokens==2);
    ref.append(suffix);CHECK(bits(engine.retained_logits(h),ref.logits));
    calls=backend.calls;result=engine.decode(h,p);
    CHECK(backend.calls==calls && result.tokens==std::vector<std::uint32_t>{ref.decode()});
    equivalent(engine,h,ref);
}
void rejection_and_caps() {
    auto model=tiny();Tables tables;Observed backend(model,tables.view());std::mutex mutex;
    e::SessionEngine engine(backend,{}, {},mutex,3,0,0);auto h=engine.store().acquire();auto ids=input(1);engine.prefill(ids,h);
    const auto caps=engine.caps();CHECK(caps.external_append && caps.lane_resume && caps.mixed_advance);
    CHECK(!caps.fork_lanes && !caps.shared_prefix_cow && !caps.request_lane_batch && !caps.batch_decode && !caps.batch_prefill && !caps.row_hints);
    CHECK(!caps.speculative && !caps.prefix_cache);
    refuses([&]{e::SessionEngine too_wide(backend,{}, {},mutex,9,0,0);});
    auto empty=engine.store().acquire();const auto initial_creates=backend.creates;
    engine.prefill({},empty);CHECK(backend.creates==initial_creates && engine.store().consumed(empty)==0);
    refuses([&]{(void)engine.result(empty);});refuses([&]{engine.decode(empty,sampler());});
    e::Request prepared;prepared.prompt=ids;prepared.max_tokens=2;prepared.dflash=e::DraftMode::off;
    engine.prepare(empty,prepared);refuses([&]{engine.prefill(ids,empty);});refuses([&]{engine.prefill({},empty);});
    CHECK(backend.creates==initial_creates);engine.store().release(empty);
    refuses([&]{engine.prefill_batch({});});refuses([&]{(void)engine.decode_batch({});});
    auto p=sampler();auto bad=p;bad.parameters.min_p=0.1;const auto calls=backend.calls;
    refuses([&]{engine.decode(h,bad);});bad=p;bad.parameters.seeded=false;refuses([&]{engine.decode(h,bad);});
    bad=p;bad.constraint=std::make_shared<strix::serve::ConstraintSpec>();refuses([&]{engine.decode(h,bad);});
    bad=p;bad.parameters.top_p=1.000000000001;refuses([&]{engine.decode(h,bad);});
    std::vector<std::uint32_t> invalid{37};refuses([&]{engine.prefill(invalid,h);});CHECK(backend.calls==calls);
    const s::AdvanceItem duplicates[]={{h,{},p,{}},{h,{},p,{}}};refuses([&]{engine.advance(duplicates);});
    s::AdvanceItem hint{h,{},p,{},s::RowHints{1,{}}};refuses([&]{engine.advance({&hint,1});});CHECK(backend.calls==calls);
    auto selected=engine.decode(h,p);auto stale=p;stale.rng_draws=selected.rng_draws+1;
    refuses([&]{engine.decode(h,stale);});stale=p;++stale.parameters.seed;refuses([&]{engine.decode(h,stale);});
    const auto snapshot=engine.store().capture(h);refuses([&]{(void)engine.fork_lane(snapshot);});
    auto foreign=h;++foreign.store;refuses([&]{engine.prefill(ids,foreign);});
    engine.store().release(h);auto replacement=engine.store().acquire();refuses([&]{engine.prefill(ids,h);});
    engine.prefill(ids,replacement);engine.store().invalidate(replacement);refuses([&]{engine.decode(replacement,p);});refuses([&]{engine.append_external(ids,replacement);});
    engine.store().release(replacement);
    // Unsupported effective identity/drafter rejects before creating a stream.
    backend.enabled=false;e::SessionEngine unsupported(backend,{}, {},mutex,1,0,0);auto u=unsupported.store().acquire();
    const auto creates=backend.creates;refuses([&]{unsupported.prefill(ids,u);});CHECK(backend.creates==creates);CHECK(!unsupported.caps().external_append);
    backend.enabled=true;unsigned drafts=0;
    e::SessionEngine drafted(backend,[&]()->std::unique_ptr<e::DraftSession>{++drafts;return nullptr;},{},mutex,1,0,0);
    auto d=drafted.store().acquire();refuses([&]{drafted.prefill(ids,d);});CHECK(drafts==0);CHECK(!drafted.caps().external_append);
    model.identity="unqualified-test-identity";CHECK(!backend.host.input_continuation_supported());
    e::SessionEngine identity(backend,{}, {},mutex,1,0,0);auto bad_identity=identity.store().acquire();
    refuses([&]{identity.prefill(ids,bad_identity);});CHECK(!identity.caps().external_append && !identity.caps().lane_resume);
}
void rollback() {
    auto model=tiny();Tables tables;Observed backend(model,tables.view());std::mutex mutex;
    backend.faults=std::make_shared<Faults>();
    e::SessionEngine engine(backend,{}, {},mutex,3,0,0);auto h=engine.store().acquire();auto ids=input(9);auto p=sampler(true);
    engine.prefill(ids,h);auto output=engine.decode(h,p);p.rng_draws=output.rng_draws;
    const auto old=engine.store().capture(h);const auto old_digest=digest(engine,h);auto suffix=input(17);
    backend.fail_call=backend.calls+2;refuses([&]{engine.append_external(suffix,h);});
    CHECK(engine.store().tokens(h)==old.tokens);CHECK(engine.store().pending(h)==old.pending);CHECK(digest(engine,h)==old_digest);
    CHECK(engine.result(h).ids==output.tokens);CHECK(engine.result(h).rng_draws==p.rng_draws);
    backend.fail_call=0;engine.append_external(suffix,h);CHECK(engine.store().consumed(h)==27);
    // A target failure in decode restores the pending anchor and RNG as well.
    output=engine.decode(h,p);p.rng_draws=output.rng_draws;const auto next=engine.store().capture(h);const auto next_digest=digest(engine,h);
    backend.fail_call=backend.calls+1;refuses([&]{engine.decode(h,p);});CHECK(digest(engine,h)==next_digest);CHECK(engine.store().pending(h)==next.pending);
    CHECK(engine.result(h).rng_draws==p.rng_draws);backend.fail_call=0;
    const auto committed_ids=engine.result(h).ids;
    backend.faults->fail_commit=backend.faults->commits+1;
    // Sampling has drawn and the pending row has physically committed when
    // this failure fires. Neither that draw nor an unpublished output survives.
    refuses([&]{engine.decode(h,p);});CHECK(digest(engine,h)==next_digest);
    CHECK(engine.store().pending(h)==next.pending && engine.result(h).rng_draws==p.rng_draws);
    CHECK(engine.result(h).ids==committed_ids);backend.faults->fail_commit=0;
    Reference oracle(model,tables.view(),policy(p));oracle.append(next.tokens);
    const std::uint32_t anchor=*next.pending;oracle.append({&anchor,1});
    const auto retried=engine.decode(h,p);CHECK(retried.tokens==std::vector<std::uint32_t>{oracle.decode()});
    CHECK(retried.rng_draws==oracle.draws && digest(engine,h)==oracle.state.digest());
    auto empty=engine.store().acquire();backend.fail_call=backend.calls+2;refuses([&]{engine.prefill(suffix,empty);});
    CHECK(engine.store().consumed(empty)==0);CHECK(!engine.store().pending(empty));backend.fail_call=0;engine.prefill(ids,empty);
    // Reentrant calls are refused immediately, before entering a second plan.
    bool attempted=false;backend.on_call=[&]{attempted=true;refuses([&]{engine.append_external(ids,h);});
        refuses([&]{(void)engine.store().capture(h);});refuses([&]{engine.lane_receipt(h);});};
    engine.append_external(ids,empty);CHECK(attempted);backend.on_call={};
}
void mixed_group() {
    auto model=tiny();Tables tables;Observed backend(model,tables.view());std::mutex mutex;
    e::SessionEngine engine(backend,{}, {},mutex,3,0,0);auto a=engine.store().acquire(),b=engine.store().acquire(),c=engine.store().acquire();
    auto ids=input(1);engine.prefill(ids,a);engine.prefill(ids,b);auto p=sampler();
    auto output=engine.decode(b,p);p.rng_draws=output.rng_draws;
    e::Request legacy;legacy.prompt=input(2);legacy.max_tokens=8;legacy.stop_ids.clear();legacy.dflash=e::DraftMode::off;
    engine.prepare(c,legacy);engine.start(c);s::SamplerState old;old.max_tokens=8;
    auto suffix=input(9);s::AdvanceItem items[]={{a,suffix,p,{}},{b,{},p,{}},{c,legacy.prompt,old,{}}};
    const auto calls=backend.calls;auto results=engine.advance(items);
    CHECK(backend.calls==calls+1);CHECK(backend.widths.back()==8);CHECK(results[0].input_tokens==6);CHECK(results[0].output.tokens.empty());
    CHECK(results[1].output.tokens.size()==1);CHECK(results[2].input_tokens==2);CHECK(results[2].output.tokens.empty());
    CHECK(backend.trace[backend.trace.size()-2].slot==b.slot);CHECK(backend.trace.back().slot==c.slot);
    // Remaining bounded suffix joins another single forward; no hidden scalar passes.
    const auto remaining=std::span<const std::uint32_t>(suffix).subspan(results[0].input_tokens);
    auto q=sampler();const s::AdvanceItem tail{a,remaining,q,{}};auto result=engine.advance({&tail,1});
    CHECK(backend.calls==calls+2);CHECK(result[0].input_tokens==3);CHECK(engine.store().consumed(a)==10);
    CHECK(engine.result(a).ids.empty());CHECK(engine.result(a).rng_draws==0);
    // Both retained-first lanes select without a forward; batching APIs still
    // refuse because this is mixed advance, not qualified fork-lane batching.
    auto fresh=engine.store().capture(a);const auto before=backend.calls;
    const s::AdvanceItem first{a,{},q,{}};result=engine.advance({&first,1});CHECK(backend.calls==before);CHECK(result[0].output.tokens.size()==1);
    engine.store().restore(a,fresh);CHECK(!engine.store().pending(a));CHECK(engine.result(a).ids.empty());
}
void mixed_long_prompt() {
    auto model=tiny();Tables tables;Observed backend(model,tables.view());std::mutex mutex;
    e::SessionEngine engine(backend,{}, {},mutex,2,0,0);
    auto legacy=engine.store().acquire(),lane=engine.store().acquire();
    e::Request request;request.prompt=input(17);request.max_tokens=8;
    request.stop_ids.clear();request.dflash=e::DraftMode::off;
    engine.prepare(legacy,request);engine.start(legacy);
    auto prefix=input(1);engine.prefill(prefix,lane);
    // A separately advertised prefill route is a tripwire: a mixed call must
    // use the one existing grouped forward, even for a long legacy prompt.
    backend.prefill_chunk=8;const auto calls=backend.calls;
    auto suffix=input(2);auto p=sampler();s::SamplerState old;old.max_tokens=8;
    const s::AdvanceItem items[]={{legacy,request.prompt,old,{}},{lane,suffix,p,{}}};
    auto results=engine.advance(items);
    CHECK(backend.calls==calls+1 && backend.widths.back()==8 && backend.prefill_calls==0);
    CHECK(results[0].error.empty() && results[1].error.empty());
    CHECK(results[0].input_tokens==7 && results[1].input_tokens==1);
    CHECK(results[0].output.tokens.empty() && results[1].output.tokens.empty());
    Reference legacy_ref(model,tables.view(),{}),lane_ref(model,tables.view(),policy(p));
    legacy_ref.append(std::span<const std::uint32_t>(request.prompt).first(7));
    lane_ref.append(prefix);lane_ref.append(std::span<const std::uint32_t>(suffix).first(1));
    CHECK(digest(engine,legacy)==legacy_ref.state.digest());
    CHECK(digest(engine,lane)==lane_ref.state.digest());
    CHECK(engine.store().tokens(legacy)==legacy_ref.consumed);
    CHECK(engine.store().tokens(lane)==lane_ref.consumed);
    CHECK(engine.result(lane).rng_draws==0);
    CHECK(bits(engine.retained_logits(lane),lane_ref.logits));
}
// Exercise the device-checked result contract without claiming native execution.
// Retained selection has no row in the other lane's device winner array.
void retained_with_device_rows() {
    auto model=tiny();Tables tables;Observed backend(model,tables.view());backend.device_checked=true;
    std::mutex mutex;e::SessionEngine engine(backend,{}, {},mutex,2,0,0);
    auto a=engine.store().acquire(),b=engine.store().acquire();auto ids=input(9);auto p=sampler();
    engine.prefill(ids,a);engine.prefill(ids,b);
    Reference ar(model,tables.view(),policy(p)),br(model,tables.view(),policy(p));ar.append(ids);br.append(ids);
    auto first=engine.decode(a,p);CHECK(first.tokens==std::vector<std::uint32_t>{ar.decode()});
    auto suffix=input(2);engine.append_external(suffix,a);ar.append(suffix);
    auto old=engine.decode(b,p);CHECK(old.tokens==std::vector<std::uint32_t>{br.decode()});
    const s::AdvanceItem items[]={{a,{},p,{}},{b,{},p,{}}};const auto calls=backend.calls;
    auto out=engine.advance(items);CHECK(backend.calls==calls+1 && backend.widths.back()==1);
    CHECK(out[0].output.tokens==std::vector<std::uint32_t>{ar.decode()});
    CHECK(out[1].output.tokens==std::vector<std::uint32_t>{br.decode()});
    equivalent(engine,a,ar);equivalent(engine,b,br);
    // A retained lane after all real rows must not index past the result array.
    engine.append_external(suffix,b);br.append(suffix);
    const s::AdvanceItem reversed[]={{a,{},p,{}},{b,{},p,{}}};out=engine.advance(reversed);
    CHECK(out[0].output.tokens==std::vector<std::uint32_t>{ar.decode()});
    CHECK(out[1].output.tokens==std::vector<std::uint32_t>{br.decode()});
    equivalent(engine,a,ar);equivalent(engine,b,br);
}
void publication_failure() {
    auto model=tiny();Tables tables;Observed backend(model,tables.view());backend.faults=std::make_shared<Faults>();std::mutex mutex;
    e::SessionEngine engine(backend,{}, {},mutex,3,0,0);
    auto legacy=engine.store().acquire(),a=engine.store().acquire(),b=engine.store().acquire();
    e::Request request;request.prompt=input(1);request.max_tokens=8;request.stop_ids.clear();request.dflash=e::DraftMode::off;
    engine.prepare(legacy,request);engine.start(legacy);auto ids=input(9);engine.prefill(ids,a);engine.prefill(ids,b);
    auto before_a=engine.store().capture(a),before_b=engine.store().capture(b);
    const auto digest_a=digest(engine,a),digest_b=digest(engine,b);auto p=sampler();s::SamplerState old;old.max_tokens=8;
    auto suffix=input(3);
    const s::AdvanceItem items[]={{legacy,request.prompt,old,{}},{a,suffix,p,{}},{b,suffix,p,{}}};
    backend.faults->fail_commit=backend.faults->commits+2;
    const auto calls=backend.calls;auto results=engine.advance(items);
    CHECK(backend.calls==calls+1);CHECK(results[0].output.tokens.size()==1); // unrelated committed output remains publishable
    CHECK(!results[1].error.empty() && !results[2].error.empty());CHECK(results[1].output.tokens.empty() && results[2].output.tokens.empty());
    CHECK(engine.store().tokens(a)==before_a.tokens && engine.store().tokens(b)==before_b.tokens);
    CHECK(digest(engine,a)==digest_a && digest(engine,b)==digest_b);
    CHECK(engine.result(a).rng_draws==0 && engine.result(b).rng_draws==0);
    backend.faults->fail_commit=0;engine.append_external(suffix,a);engine.append_external(suffix,b);
    // A later chunk has already overwritten sliding rings when publication
    // throws. Complete rollback must restore tags, values and retained output.
    const auto checkpoint=engine.store().capture(a);const auto checkpoint_digest=digest(engine,a);
    const std::vector<float> logits(engine.retained_logits(a).begin(),engine.retained_logits(a).end());
    backend.faults->fail_commit=backend.faults->commits+2;auto long_suffix=input(17);
    refuses([&]{engine.append_external(long_suffix,a);});CHECK(digest(engine,a)==checkpoint_digest);
    CHECK(engine.store().tokens(a)==checkpoint.tokens);CHECK(bits(engine.retained_logits(a),logits));
    // If complete restoration itself fails, quarantine only that handle. An
    // unrelated committed lane remains usable and keeps its prior output.
    backend.faults->failed=false;backend.faults->fail_restore=true;backend.faults->fail_commit=backend.faults->commits+1;
    refuses([&]{engine.append_external(suffix,a);});refuses([&]{(void)engine.store().tokens(a);});
    backend.faults->fail_restore=false;backend.faults->fail_commit=0;
    engine.append_external(suffix,b);CHECK(engine.store().tokens(b).size()==15);
    CHECK(engine.result(legacy).ids==results[0].output.tokens);engine.store().release(a);
    auto reused=engine.store().acquire();engine.prefill(ids,reused);CHECK(engine.store().consumed(reused)==ids.size());
}
void bounded_pending_and_cancel() {
    auto model=tiny();Tables tables;Observed backend(model,tables.view());std::mutex mutex;
    e::SessionEngine engine(backend,{}, {},mutex,8,0,0);std::vector<s::KvHandle> handles;
    auto ids=input(1);auto p=sampler();
    for(int i=0;i<8;++i){auto h=engine.store().acquire();engine.prefill(ids,h);engine.decode(h,p);handles.push_back(h);}
    const auto published=engine.result(handles[0]).ids;auto suffix=input(2);
    std::vector<s::AdvanceItem> items;
    for(std::size_t i=0;i<handles.size();++i)items.push_back({handles[i],i==0?std::span<const std::uint32_t>(suffix):std::span<const std::uint32_t>{},p,{}});
    auto calls=backend.calls;auto result=engine.advance(items);CHECK(backend.calls==calls+1 && backend.widths.back()==8);
    CHECK(result[0].input_tokens==0);CHECK(engine.store().consumed(handles[0])==2);CHECK(!engine.store().pending(handles[0]));
    CHECK(engine.result(handles[0]).ids==published);
    // Cancel before any remaining suffix row. Preserve the unfinished input and
    // finish it only after explicit resume with a fresh source.
    std::stop_source stopped;auto q=p;q.cancellation=stopped.get_token();stopped.request_stop();
    const s::AdvanceItem cancelled{handles[0],suffix,q,{}};result=engine.advance({&cancelled,1});
    CHECK(result[0].output.finish==s::FinishReason::cancelled);CHECK(result[0].input_tokens==0);CHECK(backend.calls==calls+1);
    refuses([&]{engine.append_external(suffix,handles[0]);});engine.resume_lane(handles[0]);
    refuses([&]{engine.advance({&cancelled,1});});std::stop_source fresh;q.cancellation=fresh.get_token();
    const s::AdvanceItem tail{handles[0],suffix,q,{}};result=engine.advance({&tail,1});CHECK(result[0].input_tokens==2);
    CHECK(backend.calls==calls+2);CHECK(engine.result(handles[0]).ids==published);
    Reference ref(model,tables.view(),policy(p));ref.append(ids);CHECK(ref.decode()==published[0]);ref.append(suffix);
    equivalent(engine,handles[0],ref);CHECK(bits(engine.retained_logits(handles[0]),ref.logits));
}
}
#ifndef STRIX_G4_CONTINUATION_TEST_SUPPORT
int main() {
    try {
        std::cout<<"boundaries\n";boundaries();
        std::cout<<"interleaved\n";interleaved();
        std::cout<<"terminal boundaries\n";terminal_boundaries();
        std::cout<<"resumed bounded policy\n";resumed_bounded_policy();
        std::cout<<"admission and capabilities\n";rejection_and_caps();
        std::cout<<"rollback\n";rollback();
        std::cout<<"mixed group\n";mixed_group();
        std::cout<<"mixed long prompt\n";mixed_long_prompt();
        std::cout<<"retained with device rows\n";retained_with_device_rows();
        std::cout<<"publication failures\n";publication_failure();
        std::cout<<"bounded pending and cancellation\n";bounded_pending_and_cancel();
        std::cout<<"real host continuation: boundaries, serial reference, append, policy, rollback, grouped planner PASS\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
#endif
