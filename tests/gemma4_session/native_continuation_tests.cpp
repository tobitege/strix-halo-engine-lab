#include "gemma4/engine/runtime.hpp"
#include "gemma4/engine/session_engine.hpp"
#include "gemma4/ref/target.hpp"
#include "gemma4/kv/storage.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stop_token>

namespace g = strix::gemma4;
namespace e = g::engine;
namespace s = strix::session;
int gemma4_native_api_turns(e::Backend&,const g::ModelManifest&,g::RopeTables,
    const std::filesystem::path&,const strix::serve::Json&,const std::string&);
#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string(#x)+" line "+std::to_string(__LINE__)); } while(false)
namespace {
template<class F> void refuses(F f) {
    bool caught=false;try {f();}catch(const std::exception&){caught=true;}CHECK(caught);
}
bool bits(std::span<const float> a,std::span<const float> b) {
    return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size_bytes())==0);
}
struct Faults {std::size_t commits=0,fail_commit=0;bool failed=false,fail_restore=false;};
// These wrappers observe the actual device stream and its complete eager copies.
// The independent existing HostTarget is an oracle, never an execution fallback.
class CheckedTxn final:public g::KvTxn {
public:
    CheckedTxn(std::unique_ptr<g::KvTxn> n,std::unique_ptr<g::KvTxn> h,std::shared_ptr<Faults> f)
        :native(std::move(n)),host(std::move(h)),faults(std::move(f)){}
    std::uint64_t frontier()const override {CHECK(native->frontier()==host->frontier());return native->frontier();}
    void reserve_rows(std::size_t n)override {native->reserve_rows(n);host->reserve_rows(n);host_reserved=true;}
    void commit(std::size_t n)override {
        native->commit(n);host->commit(n);
        if(++faults->commits==faults->fail_commit){faults->failed=true;throw std::runtime_error("operator probe post-publication injection");}
    }
    std::unique_ptr<g::KvTxn> native,host;
    std::shared_ptr<Faults> faults;
    bool host_reserved=false;
};
class CheckedState final:public e::StreamState {
public:
    CheckedState(std::unique_ptr<e::StreamState> n,std::unique_ptr<g::HostStream> h,std::shared_ptr<Faults> f)
        :native_(std::move(n)),host_(std::move(h)),faults_(std::move(f)){}
    std::unique_ptr<g::KvTxn> begin()override {return std::make_unique<CheckedTxn>(native_->begin(),host_->begin(),faults_);}
    std::uint64_t frontier()const override {CHECK(native_->frontier()==host_->frontier());return native_->frontier();}
    std::string digest()const override {
        const auto d=native_->digest();CHECK(d==host_->digest());return d;
    }
    std::unique_ptr<e::StreamState> fork(std::uint32_t n)const override {
        if(faults_->failed && faults_->fail_restore)throw std::runtime_error("operator probe rollback-copy injection");
        auto native=native_->fork(n);CHECK(native);auto host=host_->snapshot();
        CHECK(native->frontier()==host->frontier() && native->digest()==host->digest());
        return std::make_unique<CheckedState>(std::move(native),std::move(host),faults_);
    }
    bool kv_view(std::uint32_t l,const g::KvTxn* tx,g::KvView& out)const override {
        const auto* checked=dynamic_cast<const CheckedTxn*>(tx);
        if(tx && !checked)throw std::invalid_argument("foreign probe transaction");
        return native_->kv_view(l,checked?checked->native.get():nullptr,out);
    }
    // Deliberately no retained-byte claim: device turn retention is unsupported.
private:
    std::unique_ptr<e::StreamState> native_;
    std::unique_ptr<g::HostStream> host_;
    std::shared_ptr<Faults> faults_;
};
class Observed final:public e::Backend {
public:
    Observed(e::Backend& b,const g::ModelManifest& m,g::RopeTables t)
        :backend(b),reference(g::bind_host_weights(m),t,1){}
    bool input_continuation_supported()const override{return backend.input_continuation_supported();}
    bool prefill_matches_decode()const override{return backend.prefill_matches_decode();}
    std::size_t prefill_chunk_rows()const override{return backend.prefill_chunk_rows();}
    bool prefill_batch_supported()const override{return backend.prefill_batch_supported();}
    std::unique_ptr<e::StreamState> create_stream()override{return create_stream(config().max_positions);}
    std::unique_ptr<e::StreamState> create_stream(std::uint32_t n)override {
        ++creates;
        return std::make_unique<CheckedState>(backend.create_stream(n),std::make_unique<g::HostStream>(config()),faults);
    }
    const g::ModelConfig& config()const override{return backend.config();}
    strix::serve::Json identity()const override{return backend.identity();}
    g::ForwardResult forward_rows(std::span<const g::RowPlan> rows)override {
        ++calls;total_rows+=rows.size();widths.push_back(rows.size());
        std::vector<g::RowPlan> native(rows.begin(),rows.end());
        for(auto& row:native)row.transaction=checked(row.transaction).native.get();
        std::map<CheckedTxn*,std::size_t> counts;
        for(const auto& row:rows)++counts[&checked(row.transaction)];
        for(auto [tx,count]:counts)if(!tx->host_reserved){tx->host->reserve_rows(count);tx->host_reserved=true;}
        auto out=backend.forward_rows(native);
        CHECK(out.rows==rows.size() && out.vocab==config().vocab && out.tap_width==5*config().hidden);
        const auto wanted=[&](auto field){return std::any_of(rows.begin(),rows.end(),[&](const auto& row){return row.*field;});};
        CHECK(out.logits.size()==out.rows*out.vocab || (!wanted(&g::RowPlan::export_logits) && out.logits.empty()));
        CHECK(out.taps.size()==out.rows*out.tap_width || (!wanted(&g::RowPlan::export_taps) && out.taps.empty()));
        CHECK(!wanted(&g::RowPlan::export_hidden) || out.hidden.size()==out.rows*config().hidden);
        CHECK(out.selected.size()==out.rows || (!out.device_checked && out.selected.empty()));
        // Serial oracle passes deliberately differ from native grouping. No
        // tolerance, quantization or fixture refresh hides a row mismatch.
        for(const auto& row:rows) {
            auto oracle=row;oracle.output_row=0;oracle.transaction=checked(row.transaction).host.get();
            oracle.export_logits=oracle.export_hidden=oracle.export_taps=true;
            const auto expected=reference.forward_rows({&oracle,1});
            oracle_logits[row.stream]=expected.logits;
            if(!out.logits.empty())CHECK(bits(std::span(out.logits).subspan(row.output_row*out.vocab,out.vocab),expected.logits));
            if(row.export_hidden)CHECK(bits(std::span(out.hidden).subspan(row.output_row*config().hidden,config().hidden),expected.hidden));
            if(!out.taps.empty())CHECK(bits(std::span(out.taps).subspan(row.output_row*out.tap_width,out.tap_width),expected.taps));
            std::uint64_t draws=0;
            const auto winner=e::select_token(expected.logits,{},draws);CHECK(draws==0);
            if(!out.selected.empty())CHECK(out.selected.at(row.output_row)==winner);
            trace.push_back({row.stream,row.position,row.token});
        }
        if(calls==fail_call)throw std::runtime_error("operator probe post-evaluation injection");
        return out;
    }
    struct Row {std::uint64_t lane,position;std::uint32_t token;};
    e::Backend& backend;
    g::HostTarget reference;
    std::shared_ptr<Faults> faults=std::make_shared<Faults>();
    std::size_t calls=0,total_rows=0,fail_call=0,creates=0;
    std::vector<std::size_t> widths;
    std::vector<Row> trace;
    std::map<std::uint64_t,std::vector<float>> oracle_logits;
private:
    static CheckedTxn& checked(g::KvTxn* tx) {auto* p=dynamic_cast<CheckedTxn*>(tx);CHECK(p);return *p;}
};
s::SamplerState sampler(bool sampled=false) {
    s::SamplerState p;p.parameters.seeded=true;p.parameters.seed=941;
    p.parameters.temperature=sampled?0.8:0;p.parameters.top_p=0.75;p.parameters.top_k=7;
    p.rng_draws=11;p.max_tokens=8;return p;
}
e::Sampling policy(const s::SamplerState& p) {
    return {float(p.parameters.temperature),float(p.parameters.top_p),p.parameters.top_k,p.parameters.seed,p.rng_draws};
}
std::string digest(e::SessionEngine& engine,s::KvHandle h) {
    return engine.lane_receipt(h).find("state_digest")->as_string();
}
std::vector<std::uint32_t> ids(const std::vector<std::uint32_t>& seed,std::size_t n) {
    std::vector<std::uint32_t> out;out.reserve(n);for(std::size_t i=0;i<n;++i)out.push_back(seed[i%seed.size()]);return out;
}
void stable(e::SessionEngine& engine,s::KvHandle h) {
    const auto r=engine.lane_receipt(h);
    CHECK(std::uint64_t(r.find("frontier")->as_int())==engine.store().consumed(h));
    CHECK(std::uint64_t(r.find("rng_draws")->as_int())==engine.result(h).rng_draws);
    CHECK(!r.find("state_digest")->as_string().empty());
    CHECK(r.find("finish_reason")->as_string()==e::status_name(engine.result(h).status));
    CHECK(std::uint64_t(r.find("target_calls")->as_int())==engine.result(h).target_calls);
    CHECK(std::uint64_t(r.find("target_rows")->as_int())==engine.result(h).target_rows);
    CHECK(!engine.turn_work_bound() && !engine.turn_retained_bound(h) && !engine.turn_cache_candidate(h));
    CHECK(!engine.caps().fork_lanes && !engine.caps().shared_prefix_cow);
}
void lifecycle(Observed& backend) {
    auto a=backend.create_stream(),b=backend.create_stream();auto at=a->begin(),bt=b->begin();
    refuses([&]{a->fork(backend.config().max_positions);});refuses([&]{a->digest();});
    refuses([&]{a->begin();});
    g::KvView view;std::uint32_t producer=0;
    while(producer<backend.config().layers && backend.config().layer[producer].producer!=producer)++producer;
    CHECK(producer<backend.config().layers);
    refuses([&]{a->kv_view(producer,bt.get(),view);});
    at.reset();bt.reset();CHECK(a->frontier()==0 && b->frontier()==0);
    auto copy=a->fork(backend.config().max_positions);const auto original=copy->digest();
    a.reset();CHECK(copy->digest()==original); // independent allocation, no source owner needed
}
void boundary(Observed& backend,const std::vector<std::uint32_t>& seed,std::size_t n) {
    std::mutex mutex;e::SessionEngine engine(backend,{}, {},mutex,2,0,0);
    auto h=engine.store().acquire();const auto prefix=ids(seed,n);auto p=sampler(true);
    const auto calls=backend.calls,rows=backend.total_rows,creates=backend.creates;
    engine.prefill({},h);CHECK(backend.calls==calls && backend.creates==creates);
    engine.prefill(prefix,h);CHECK(engine.store().tokens(h)==prefix && !engine.store().pending(h));
    CHECK(engine.result(h).ids.empty() && engine.result(h).rng_draws==0);
    CHECK(backend.total_rows==rows+n && backend.calls==calls+(n+7)/8);
    for(std::size_t i=0;i<n;++i){const auto& row=backend.trace.at(backend.trace.size()-n+i);CHECK(row.position==i && row.token==prefix[i]);}
    const std::vector<float> logits(engine.retained_logits(h).begin(),engine.retained_logits(h).end());
    auto draws=p.rng_draws;const auto selected=e::select_token(logits,policy(p),draws);
    const auto prefills=backend.calls;auto out=engine.decode(h,p);
    CHECK(out.tokens==std::vector<std::uint32_t>{selected} && out.rng_draws==draws && backend.calls==prefills);
    CHECK(engine.store().pending(h)==selected);p.rng_draws=out.rng_draws;
    out=engine.decode(h,p);
    const auto second=e::select_token(backend.oracle_logits.at(h.slot),policy(p),draws);
    CHECK(out.tokens==std::vector<std::uint32_t>{second} && out.rng_draws==draws);
    p.rng_draws=out.rng_draws;
    const std::vector<std::uint32_t> generated{selected,second};
    const auto checkpoint=engine.store().capture(h);CHECK(checkpoint.bytes==0);const auto receipt=engine.lane_receipt(h).dump();
    engine.append_external({},h);CHECK(engine.lane_receipt(h).dump()==receipt);
    const auto suffix=ids(seed,3);engine.append_external(suffix,h);
    auto causal=prefix;causal.push_back(selected);causal.push_back(second);causal.insert(causal.end(),suffix.begin(),suffix.end());
    CHECK(engine.store().tokens(h)==causal && !engine.store().pending(h));
    CHECK(engine.result(h).ids==generated && engine.result(h).rng_draws==draws);
    const auto expected_digest=digest(engine,h);const std::vector<float> expected(engine.retained_logits(h).begin(),engine.retained_logits(h).end());
    engine.store().restore(h,checkpoint);CHECK(engine.lane_receipt(h).dump()==receipt);
    engine.append_external(suffix,h);CHECK(digest(engine,h)==expected_digest && bits(engine.retained_logits(h),expected));
    stable(engine,h);
    // Complete copy survives releasing/reusing its source and another source mutation.
    const auto saved=engine.store().capture(h);engine.store().release(h);
    auto reused=engine.store().acquire();engine.prefill(ids(seed,1),reused);
    engine.store().restore(reused,saved);CHECK(digest(engine,reused)==expected_digest);
    engine.append_external(ids(seed,2),reused);CHECK(saved.tokens==causal);stable(engine,reused);
}
void terminal(Observed& backend,const std::vector<std::uint32_t>& seed,bool context_only,int selected_mode=-1,int sampled_case=-1) {
    for(bool sampled:{false,true})for(int mode=context_only?2:0;mode<(context_only?4:2);++mode) {
        if((selected_mode>=0 && mode!=selected_mode) || (sampled_case>=0 && int(sampled)!=sampled_case))continue;
        std::mutex mutex;e::SessionEngine engine(backend,{}, {},mutex,1,0,0);
        auto h=engine.store().acquire();auto p=sampler(sampled);
        // Last-context and fully consumed context are separate, bounded tickets.
        auto prefix=ids(seed,mode>=2?backend.config().max_positions-std::uint32_t(mode==2):9);
        engine.prefill(prefix,h);std::vector<std::uint32_t> stop;
        if(mode==1) {auto draws=p.rng_draws;stop.push_back(e::select_token(engine.retained_logits(h),policy(p),draws));p.stop_tokens=stop;}
        p.max_tokens=mode==3?8:1;
        const auto calls=backend.calls;auto out=engine.decode(h,p);CHECK(backend.calls==calls);
        if(mode==3)CHECK(out.finish==s::FinishReason::context && out.tokens.empty());
        else CHECK(out.tokens.size()==1 && out.finish==(mode==1?s::FinishReason::stop:
            mode==2?s::FinishReason::context:s::FinishReason::length));
        const auto before=engine.result(h);const auto pending=engine.store().pending(h);
        engine.commit_pending(h);CHECK(backend.calls==calls+std::size_t(pending.has_value()));
        CHECK(!engine.store().pending(h) && engine.store().consumed(h)==prefix.size()+std::size_t(pending.has_value()));
        CHECK(engine.result(h).ids==before.ids && engine.result(h).rng_draws==before.rng_draws && engine.result(h).status==before.status);
        const auto sealed=engine.lane_receipt(h).dump();engine.commit_pending(h);
        CHECK(engine.lane_receipt(h).dump()==sealed);stable(engine,h);
    }
    if(context_only)return; // context cases are independently bounded tickets
    // A zero-output seal at an arbitrary frontier does no target work.
    std::mutex mutex;e::SessionEngine engine(backend,{}, {},mutex,1,0,0);auto h=engine.store().acquire();
    engine.prefill(ids(seed,9),h);auto p=sampler();p.max_tokens=0;auto out=engine.decode(h,p);CHECK(out.tokens.empty());
    const auto calls=backend.calls;engine.commit_pending(h);CHECK(backend.calls==calls);stable(engine,h);
}
void cancel(Observed& backend,const std::vector<std::uint32_t>& seed) {
    std::mutex mutex;e::SessionEngine engine(backend,{}, {},mutex,1,0,0);auto h=engine.store().acquire();
    engine.prefill(ids(seed,9),h);auto p=sampler(true);auto out=engine.decode(h,p);p.rng_draws=out.rng_draws;
    const auto before=engine.store().capture(h);const auto state=digest(engine,h);const auto calls=backend.calls;
    std::stop_source stopped;p.cancellation=stopped.get_token();stopped.request_stop();
    const auto suffix=ids(seed,3);const s::AdvanceItem item{h,suffix,p,{}};auto result=engine.advance({&item,1});
    CHECK(result[0].output.finish==s::FinishReason::cancelled && result[0].input_tokens==0 && backend.calls==calls);
    CHECK(engine.store().tokens(h)==before.tokens && engine.store().pending(h)==before.pending && digest(engine,h)==state);
    refuses([&]{engine.commit_pending(h);});engine.resume_lane(h);refuses([&]{engine.advance({&item,1});});
    std::stop_source fresh;p.cancellation=fresh.get_token();const s::AdvanceItem resumed{h,suffix,p,{}};
    result=engine.advance({&resumed,1});CHECK(result[0].error.empty() && result[0].input_tokens==suffix.size());
    CHECK(engine.result(h).rng_draws==out.rng_draws && engine.result(h).ids==out.tokens);stable(engine,h);
}
void rollback(Observed& backend,const std::vector<std::uint32_t>& seed,bool wrapped) {
    std::mutex mutex;e::SessionEngine engine(backend,{}, {},mutex,3,0,0);
    auto legacy=engine.store().acquire(),h=engine.store().acquire(),peer=engine.store().acquire();
    auto prefix=ids(seed,wrapped?std::size_t(backend.config().window)+3:9);engine.prefill(prefix,h);engine.prefill(prefix,peer);
    e::Request request;request.prompt=ids(seed,1);request.max_tokens=8;request.stop_ids.clear();request.dflash=e::DraftMode::off;
    engine.prepare(legacy,request);engine.start(legacy);
    const auto receipt=engine.lane_receipt(h).dump(),peer_receipt=engine.lane_receipt(peer).dump();const auto suffix=ids(seed,17);
    auto p=sampler();s::SamplerState old;old.max_tokens=8;
    const s::AdvanceItem group[]={{legacy,request.prompt,old,{}},{h,suffix,p,{}},{peer,suffix,p,{}}};
    backend.faults->fail_commit=backend.faults->commits+2;
    const auto calls=backend.calls;const auto failed=engine.advance(group);
    CHECK(backend.calls==calls+1 && backend.widths.back()==8);
    CHECK(failed[0].output.tokens.size()==1 && !failed[1].error.empty() && !failed[2].error.empty());
    CHECK(engine.result(legacy).ids==failed[0].output.tokens);
    CHECK(engine.lane_receipt(h).dump()==receipt && engine.lane_receipt(peer).dump()==peer_receipt);
    backend.faults->fail_commit=0;backend.faults->failed=false;
    backend.fail_call=backend.calls+2;refuses([&]{engine.append_external(suffix,h);});backend.fail_call=0;
    CHECK(engine.lane_receipt(h).dump()==receipt && engine.lane_receipt(peer).dump()==peer_receipt);
    backend.faults->fail_commit=backend.faults->commits+2;
    refuses([&]{engine.append_external(suffix,h);});CHECK(engine.lane_receipt(h).dump()==receipt);
    backend.faults->fail_commit=0;backend.faults->failed=false;engine.append_external(suffix,h);stable(engine,h);
    backend.faults->fail_commit=backend.faults->commits+1;backend.faults->fail_restore=true;
    refuses([&]{engine.append_external(ids(seed,9),h);});refuses([&]{(void)engine.store().tokens(h);});
    backend.faults->fail_commit=0;backend.faults->fail_restore=false;backend.faults->failed=false;
    CHECK(engine.lane_receipt(peer).dump()==peer_receipt);engine.append_external(ids(seed,9),peer);
    engine.store().release(h);h=engine.store().acquire();engine.prefill(prefix,h);stable(engine,h);
}
void mixed(Observed& backend,const std::vector<std::uint32_t>& seed) {
    std::mutex mutex;e::SessionEngine engine(backend,{}, {},mutex,3,0,0);
    auto a=engine.store().acquire(),b=engine.store().acquire(),c=engine.store().acquire();auto p=sampler();
    auto prefix=ids(seed,9);engine.prefill(prefix,a);engine.prefill(prefix,b);auto first=engine.decode(b,p);
    auto suffix=ids(seed,9);e::Request legacy;legacy.prompt=ids(seed,17);legacy.stop_ids.clear();legacy.max_tokens=8;legacy.dflash=e::DraftMode::off;
    engine.prepare(c,legacy);engine.start(c);s::SamplerState old;old.max_tokens=8;
    const s::AdvanceItem items[]={{a,suffix,p,{}},{b,{},p,{}},{c,legacy.prompt,old,{}}};
    const auto calls=backend.calls;auto out=engine.advance(items);
    CHECK(backend.calls==calls+1 && backend.widths.back()==8);
    CHECK(out[0].error.empty() && out[1].error.empty() && out[2].error.empty());
    CHECK(out[0].output.tokens.empty() && out[0].input_tokens==6 && out[2].input_tokens==1);
    CHECK(out[1].output.tokens.size()==1 && first.tokens.size()==1);
    const auto remaining=std::span(suffix).subspan(out[0].input_tokens);const s::AdvanceItem tail{a,remaining,p,{}};
    auto finished=engine.advance({&tail,1});CHECK(finished[0].input_tokens==3 && finished[0].output.tokens.empty());
    CHECK(engine.result(a).ids.empty() && engine.result(a).rng_draws==0);
    // Retained greedy output shares a call with a native pending-anchor row.
    const auto logits=engine.retained_logits(a);std::uint64_t draws=p.rng_draws;
    const auto expected=e::select_token(logits,policy(p),draws);
    const s::AdvanceItem select[]={{a,{},p,{}},{b,{},p,{}}};const auto before=backend.calls;
    out=engine.advance(select);CHECK(backend.calls==before+1 && backend.widths.back()==1);
    CHECK(out[0].output.tokens==std::vector<std::uint32_t>{expected});stable(engine,a);stable(engine,b);
}
void queue_lifetime(const g::ModelManifest& model,g::RopeTables tables,const std::vector<std::uint32_t>& seed) {
    std::unique_ptr<e::StreamState> saved;std::string digest;
    {
        auto backend=e::make_device_backend(model,tables,false,e::NativeInputContinuation::exact_g4_v1);
        CHECK(backend->input_continuation_supported());Observed observed(*backend,model,tables);
        auto source=observed.create_stream();auto tx=source->begin();tx->reserve_rows(3);
        auto tokens=ids(seed,3);std::vector<g::RowPlan> rows;
        for(std::size_t i=0;i<tokens.size();++i)rows.push_back({0,1,i,tokens[i],tx.get(),i,true,true,true,true});
        observed.forward_rows(rows);tx->commit(3);tx.reset();
        saved=source->fork(model.config.max_positions);digest=saved->digest();source.reset();
        std::cout<<"queue-lifetime native_calls "<<observed.calls<<" native_rows "<<observed.total_rows<<" grouping 3\n";
    } // Destroy the producer target and its queue before reading/copying the snapshot.
    CHECK(saved->digest()==digest && saved->frontier()==3);
    auto copied=saved->fork(model.config.max_positions);saved.reset();
    CHECK(copied->digest()==digest && copied->frontier()==3);
}
void accounting(const g::ModelManifest& model,g::RopeTables tables,const std::vector<std::uint32_t>& seed,
                const strix::serve::Json& identity) {
    const auto* budget=identity.find("max_transient_bytes");CHECK(budget && budget->as_int()>0);
    auto backend=e::make_device_backend(model,tables,false,e::NativeInputContinuation::exact_g4_v1);
    const auto work=backend->native_turn_work_bound();CHECK(work && *work<=std::uint64_t(budget->as_int()));
    auto source=backend->create_stream(48);auto tx=source->begin();
    CHECK(!source->retained_bytes() && !source->native_payload());
    tx->reserve_rows(3);auto tokens=ids(seed,3);std::vector<g::RowPlan> rows;
    for(std::size_t r=0;r<3;++r)rows.push_back({0,1,r,tokens[r],tx.get(),r,true,true,true,true});
    const auto out=backend->forward_rows(rows);
    g::HostStream oracle(model.config);auto htx=oracle.begin();htx->reserve_rows(3);
    g::HostTarget reference(g::bind_host_weights(model),tables,1);
    for(std::size_t r=0;r<3;++r) {
        const g::RowPlan row{0,1,r,tokens[r],htx.get(),0,true,true,true,true};
        const auto expected=reference.forward_rows({&row,1});
        CHECK(bits(std::span(out.logits).subspan(r*model.config.vocab,model.config.vocab),expected.logits));
    }
    tx->commit(3);htx->commit(3);
    CHECK(!source->retained_bytes()); // publication alone is not owner quiescence
    tx.reset();htx.reset();CHECK(source->digest()==oracle.digest());
    auto payload=source->native_payload();CHECK(payload && payload->total()==*source->retained_bytes());
    std::uint64_t arrays=0,overlays=0;
    for(std::uint32_t l=0;l<model.config.layers;++l) if(model.config.layer[l].producer==l) {
        const auto width=std::uint64_t(model.config.kv_heads)*model.config.layer[l].head_dim;
        const auto slots=model.config.layer[l].attention==g::AttentionKind::global?48:model.config.window;
        g::memory::add(arrays,slots,2*width*sizeof(float)+sizeof(std::uint64_t));
        g::memory::add(overlays,3*width,2*sizeof(float));
    }
    const auto descriptors=std::uint64_t(model.config.layers)*sizeof(g::gpu::detail::Publication);
    CHECK(payload->device==arrays+overlays+descriptors && payload->pinned==descriptors);
    auto saved=source->fork(48);const auto copy=saved->native_payload();CHECK(copy);
    CHECK(copy->device==arrays+descriptors && copy->pinned==descriptors);
    const auto digest=saved->digest();source.reset();backend.reset();
    CHECK(saved->digest()==digest && saved->frontier()==3 && saved->native_payload()->total()==copy->total());
    for(int i=0;i<4;++i) {
        auto next=saved->fork(48);CHECK(next->digest()==digest && next->native_payload()->total()==copy->total());
        saved=std::move(next);
    }
    std::cout<<"native exact retained host="<<copy->host<<" device="<<copy->device<<" pinned="<<copy->pinned
        <<" total="<<copy->total()<<" source_spare_overlays="<<overlays<<" preflight_work_bound="<<*work<<'\n';
}
e::Result legacy_reference(const g::ModelManifest& model,g::RopeTables tables,const std::vector<std::uint32_t>& seed) {
    auto backend=e::make_device_backend(model,tables);CHECK(!backend->input_continuation_supported());
    {auto ordinary=backend->create_stream(48);CHECK(!ordinary->retained_bytes());}
    e::Coordinator compute(*backend);e::Request request;request.prompt=ids(seed,25);
    request.sampling=policy(sampler(true));request.max_tokens=3;request.stop_ids.clear();request.dflash=e::DraftMode::off;
    const auto lane=compute.submit(request);compute.run();const auto result=compute.result(lane);
    CHECK(result.status==e::Status::length && result.ids.size()==3 && result.frontier==27);
    CHECK(result.prefill_calls>0 && result.prefill_frontier==24);
    std::cout<<"legacy-reference "<<compute.receipt(lane).dump()<<'\n';return result;
}
void legacy_equivalence(Observed& backend,const std::vector<std::uint32_t>& seed,const e::Result& reference) {
    std::mutex mutex;e::SessionEngine engine(backend,{}, {},mutex,1,0,0);auto h=engine.store().acquire();
    engine.prefill(ids(seed,25),h);auto p=sampler(true);p.max_tokens=3;
    for(std::uint32_t i=0;i<3;++i){auto out=engine.decode(h,p);CHECK(out.tokens.size()==1);p.rng_draws=out.rng_draws;--p.max_tokens;}
    const auto& result=engine.result(h);
    CHECK(result.ids==reference.ids && result.frontier==reference.frontier && result.rng_draws==reference.rng_draws && result.status==reference.status);
    CHECK(digest(engine,h)==reference.digest);stable(engine,h);
}
void activation(const g::ModelManifest& model,g::RopeTables rope) {
    auto ordinary=e::make_device_backend(model,rope,true);
    CHECK(!ordinary->input_continuation_supported() && !ordinary->identity().find("native_input_continuation"));
    auto inspected=e::make_device_backend(model,rope,true,e::NativeInputContinuation::exact_g4_v1);
    CHECK(!inspected->input_continuation_supported());refuses([&]{inspected->create_stream();});
    auto changed=model;changed.identity="unsupported-g4-route";
    refuses([&]{e::make_device_backend(changed,rope,true,e::NativeInputContinuation::exact_g4_v1);});
    changed=model;changed.cache_dtype="unsupported-cache";
    refuses([&]{e::make_device_backend(changed,rope,true,e::NativeInputContinuation::exact_g4_v1);});
    // Typed values outside the enum cannot activate a route either.
    refuses([&]{e::make_device_backend(model,rope,true,static_cast<e::NativeInputContinuation>(99));});
}
}
int main(int argc,char** argv) {
    try {
        std::map<std::string,std::string> args;
        for(int i=1;i<argc;i+=2) {CHECK(i+1<argc);CHECK(args.emplace(argv[i],argv[i+1]).second);}
        CHECK(args.size()==4 && args.contains("--model") && args.contains("--rope") && args.contains("--identity") && args.contains("--case"));
        const auto selected=args.at("--case");
        const bool context_ticket=selected=="context-pending-greedy" || selected=="context-pending-sampled" ||
            selected=="context-full-greedy" || selected=="context-full-sampled";
        CHECK(selected.starts_with("api-") || context_ticket || selected=="cache-refuse" || selected=="accounting" || selected=="queue-lifetime" || selected=="legacy" || selected=="refuse-route" || selected=="activation" || selected=="prefix" || selected=="wrap" || selected=="terminal" || selected=="cancel" || selected=="rollback-wrap" || selected=="rollback" || selected=="mixed");
        for(const auto* name:{"--model","--rope","--identity"})CHECK(std::filesystem::exists(args.at(name)));
        auto model=std::filesystem::path(args.at("--model")).extension()==".gguf"?
            g::load_gguf_model(args.at("--model")):g::load_safetensors_model(args.at("--model"));
        auto rope=g::load_rope_tables(args.at("--rope"));
        std::ifstream input(args.at("--identity"));CHECK(input.good());
        const std::string text{std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
        const auto identity=strix::serve::parse_json_unique(text);
        CHECK(identity.find("identity") && identity.find("identity")->as_string()==g::arithmetic_id);
        CHECK(identity.find("model_hash") && identity.find("model_hash")->as_string()==model.model_hash && !model.model_hash.empty());
        CHECK(identity.find("rope_hash") && identity.find("rope_hash")->as_string()==rope.view.hash && !rope.view.hash.empty());
        CHECK(identity.find("context") && identity.find("prompt_ids"));
        const auto context=identity.find("context")->as_int();CHECK(context>=48 && std::uint64_t(context)<=model.config.max_positions && std::uint64_t(context)<=rope.view.positions);
        model.config.max_positions=std::uint32_t(context);model.rope_hash=rope.view.hash;
        std::vector<std::uint32_t> seed;for(const auto& value:identity.find("prompt_ids")->as_array()) {
            const auto id=value.as_int();CHECK(id>=0 && std::uint64_t(id)<model.config.vocab);seed.push_back(std::uint32_t(id));
        }CHECK(!seed.empty());
        if(selected=="refuse-route") {
            auto off=e::make_device_backend(model,rope.view,true);CHECK(!off->input_continuation_supported());
            refuses([&]{e::make_device_backend(model,rope.view,true,e::NativeInputContinuation::exact_g4_v1);});
            std::cout<<"factory resolved-route refusal only; native execution NOT RUN\n";return 0;
        }
        activation(model,rope.view);
        if(selected=="activation") {std::cout<<"factory inspection/refusal only; native execution NOT RUN\n";return 0;}
        if(selected=="queue-lifetime") {
            queue_lifetime(model,rope.view,seed);std::cout<<"PASS completed copy survives producer queue teardown\n";return 0;
        }
        if(selected=="accounting") {accounting(model,rope.view,seed,identity);return 0;}
        std::optional<e::Result> legacy;
        if(selected=="legacy")legacy=legacy_reference(model,rope.view,seed);
        auto native=e::make_device_backend(model,rope.view,false,e::NativeInputContinuation::exact_g4_v1);
        if(selected=="cache-refuse") {
            CHECK(native->input_continuation_supported() && !native->native_turn_work_bound());
            std::cout<<"native cache-work refusal before stream/copy/seal; resolved identity "<<native->identity().dump()<<'\n';return 0;
        }
        if(selected.starts_with("api-")) {
            CHECK(selected=="api-turn" || selected=="api-sse" || selected=="api-no-store" || selected=="api-mismatch" ||
                selected=="api-pressure" || selected=="api-retained-pressure" || selected=="api-no-retention" || selected=="api-fault" || selected=="api-cancel");
            return gemma4_native_api_turns(*native,model,rope.view,args.at("--identity"),identity,selected);
        }
        CHECK(native->input_continuation_supported());Observed observed(*native,model,rope.view);
        // A configured factory remains unsupported even if it would never draft.
        {std::mutex mutex;e::SessionEngine refused(observed,[]{return std::unique_ptr<e::DraftSession>{};},{},mutex,1,0,0);
         CHECK(!refused.caps().external_append && !refused.caps().turn_commit);auto h=refused.store().acquire();
         refuses([&]{refused.prefill(ids(seed,1),h);});CHECK(observed.calls==0 && observed.creates==0);}
        if(selected=="legacy")legacy_equivalence(observed,seed,*legacy);
        if(selected=="prefix") {lifecycle(observed);boundary(observed,seed,9);}
        if(selected=="wrap") {CHECK(std::uint64_t(model.config.window)+16<model.config.max_positions);boundary(observed,seed,std::size_t(model.config.window)+3);}
        if(selected=="terminal")terminal(observed,seed,false);
        if(context_ticket)terminal(observed,seed,true,selected.starts_with("context-pending")?2:3,selected.ends_with("sampled")?1:0);
        if(selected=="cancel")cancel(observed,seed);
        if(selected=="rollback")rollback(observed,seed,false);
        if(selected=="rollback-wrap") {CHECK(std::uint64_t(model.config.window)+48<model.config.max_positions);rollback(observed,seed,true);}
        if(selected=="mixed")mixed(observed,seed);
        std::cout<<"case "<<selected<<" native_calls "<<observed.calls<<" native_rows "<<observed.total_rows<<" grouping";
        for(auto n:observed.widths)std::cout<<' '<<n;
        std::cout<<"\nidentity "<<native->identity().dump()<<"\nPASS exact device rows/logits/hidden/taps/state and session checks; no performance qualification\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<"FAIL "<<error.what()<<'\n';return 1;}
}
