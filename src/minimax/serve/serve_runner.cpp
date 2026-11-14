#include "serve_runner.hpp"
#include <fstream>
namespace strix::minimax::serve::native {
using ms::Command;
using ms::Control;
using ms::exchange;
constexpr std::size_t kTokensPerMessage = 16384;
std::uint64_t fnv(const void* p, std::size_t bytes, std::uint64_t h = 14695981039346656037ULL) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    for (std::size_t i = 0; i < bytes; ++i) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}
std::string hex(std::uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
    return b;
}
// Prompt tokens of a prefill, rank 0 -> rank 1.
void exchange_tokens(mm::Engine& eng, std::vector<std::uint32_t>& tokens, std::size_t n, bool sender) try {
    if (!sender) tokens.assign(n, 0);
    std::vector<std::uint32_t> in(kTokensPerMessage);
    for (std::size_t c0 = 0; c0 < n; c0 += kTokensPerMessage) {
        const std::size_t m = std::min(kTokensPerMessage, n - c0);
        std::vector<std::uint32_t> out(m, 0);
        if (sender) std::copy_n(tokens.begin() + static_cast<long>(c0), m, out.begin());
        eng.exchange_bytes(out.data(), in.data(), m * 4);
        if (!sender) std::copy_n(in.begin(), m, tokens.begin() + static_cast<long>(c0));
    }
} catch (...) {
    eng.abort("control channel: protocol failure");
    throw;
}

void follow(mm::Engine& eng, Runner& run, strix::minimax::session::Runtime* session) try {
    ms::SessionCounts counts;
    // STRIX_MM_SERVE_DSPARK is off|auto|on|0|1 (ds::parse_mode); the follower starts taps only for an exact "1".
    bool taps = run.dd != nullptr && strix::env_flag_default_off("STRIX_MM_SERVE_DSPARK");
    strix::pair::Lockstep lock(static_cast<std::uint32_t>(eng.rank()), run.control_round);
    for (;;) {
        Control mine;
        mine.plen = eng.prefill_progress().open ? 0 : run.last_next;
        const Control c = exchange(eng, mine, lock);
        if (static_cast<Command>(c.command) == Command::heartbeat) run.quiesce();
        if (static_cast<Command>(c.command) == Command::state) {
            run.invalidate_residents();
            if (!run.state_endpoint) throw std::runtime_error("control channel: paired state OFF");
            run.state_endpoint->follow(c);
            continue;
        }
        if (static_cast<Command>(c.command) == Command::session) {
            const auto body = ms::session_wire(eng, {}, c.plen);
            const auto available = eng.peer(session ? 1U : 0U);
            run.invalidate_residents();
            if (session && available) (void)ms::session_execute(eng, *session, body, counts);
            eng.set_taps(taps);
            continue;
        }
        if (static_cast<Command>(c.command) == Command::taps) taps = (c.pflags & 1U) != 0;
        if (ms::replay_resident_begin(c, eng, run, [&](std::size_t n) {
                std::vector<std::uint32_t> prompt; exchange_tokens(eng,prompt,n,false); return prompt;
            }, run.prefix_reuse, run.conversation)) continue;
        if (!ms::replay_command(c, eng, run, [&](std::size_t n) {
                std::vector<std::uint32_t> prompt;
                exchange_tokens(eng, prompt, n, false);
                return prompt;
            })) return;
    }
} catch (...) {
    eng.abort("control channel: protocol failure");
    throw;
}

int paired_state_probe(mm::Engine& eng, Runner& run, const mm::Options& options, const std::string& executable,
                       std::uint64_t budget, std::span<const std::uint32_t> prompt, std::string_view ticket) try {
    using namespace ms::paired;
    need(budget && budget!=UINT64_MAX && !run.dd && !run.contexts && !eng.prefill_open() && !run.state_endpoint && eng.slots()>=3 &&
         options.slots==eng.slots() && options.capacity==eng.capacity() &&
         !prompt.empty() && prompt.size()+4<eng.capacity(),"MiniMax state probe admission");
    const bool input_mode=ticket!="state";
    need(ticket=="state" || ticket=="input" || ticket=="fault0" || ticket=="fault1","MiniMax operator ticket");
    need(!input_mode || (prompt.size()<=8 && options.max_rows>=8 && prompt.size()+8<eng.capacity()),"MiniMax bounded input ticket");
    auto snapshot=eng.loader_snapshot();
    strix::verify_loader_snapshot(snapshot);
    const auto record=strix::minimax::session::compatibility(
        {{"world","2"},{"capacity",std::to_string(options.capacity)},
         {"slots",std::to_string(options.slots)},{"max_rows",std::to_string(options.max_rows)},
         {"producer",input_mode?"generic-paired-input-plain-greedy-r1-v2":"generic-paired-plain-owner-v1"},{"draft","unconfigured"}},snapshot,executable);
    const auto digest=strix::minimax::session::sha(std::span(
        reinterpret_cast<const std::uint8_t*>(record.data()),record.size()));
    Identity identity;identity.capacity=eng.capacity();identity.slots=eng.slots();identity.schema=input_mode?2:1;
    need(digest.size()==64,"MiniMax compatibility digest");
    for(std::size_t i=0;i<32;++i)identity.assets_profile[i]=static_cast<std::uint8_t>(std::stoul(digest.substr(i*2,2),nullptr,16));
    struct Reference {
        std::vector<mm::Row> rows;
        std::vector<std::uint32_t> next;
        std::vector<float> logits;
        std::vector<Bytes> images;
    };
    std::vector<Reference> reference;
    std::size_t cursor=0, fault_cursor=0;
    bool injected=false;
    std::uint64_t native_calls=0,native_rows=0;
    std::vector<std::uint32_t> x(prompt.begin(),prompt.end());
    std::vector<std::uint32_t> y(prompt.begin(),prompt.begin()+static_cast<std::ptrdiff_t>(std::min<std::size_t>(2,prompt.size())));
    std::uint64_t reference_charge=0;
    if(input_mode) {
        // Cold independent native teacher-forcing: no SessionEngine, frontier
        // metadata, paired Endpoint or Runner selection is used for this arm.
        // Every forward has the SAME explicit row grouping as the tested arm.
        reference_charge=mul(32,add(mul(2*Endpoint::row_bytes,prompt.size()+8),mul(2*200064,4)));
        need(reference_charge<budget && strix::minimax::session::memory_admitted(
            0,reference_charge,budget,strix::minimax::session::available_memory()),"MiniMax reference witness budget");
        // Bind loaded profile, complete fixture and selected ticket on BOTH
        // ranks before the independent cold arm can write any native row.
        std::array<std::uint8_t,96> admission{},peer_admission{};
        std::copy(identity.assets_profile.begin(),identity.assets_profile.end(),admission.begin());
        const auto history_hash=strix::sha256_bytes_fast(std::as_bytes(prompt));
        const auto ticket_hash=strix::sha256_bytes_fast(std::as_bytes(std::span(ticket.data(),ticket.size())));
        std::copy(history_hash.bytes.begin(),history_hash.bytes.end(),admission.begin()+32);
        std::copy(ticket_hash.bytes.begin(),ticket_hash.bytes.end(),admission.begin()+64);
        eng.exchange_bytes(admission.data(),peer_admission.data(),admission.size());
        need(admission==peer_admission,"MiniMax operator paired profile/fixture/ticket mismatch");
        reference.reserve(32);
        const auto cold=[&](std::vector<mm::Row> rows) {
            Reference r;r.rows=std::move(rows);r.images.resize(r.rows.size());
            // Allocate all row readback before this native write.
            for(std::size_t i=0;i<r.rows.size();++i)r.images[i].resize(static_cast<std::size_t>((r.rows[i].pos+1)*Endpoint::row_bytes));
            r.next=eng.step(r.rows);Runner::synchronize();r.logits=eng.last_logits(r.rows.size());
            for(std::size_t i=0;i<r.rows.size();++i)eng.export_rows(r.rows[i].slot,0,r.rows[i].pos+1,r.images[i].data());
            auto next=r.next;reference.push_back(std::move(r));return next;
        };
        (void)cold({mm::Row{2,123,0}});
        std::array<std::uint32_t,2> candidate{};
        for(std::size_t pos=0;pos<x.size();++pos) {
            std::vector<mm::Row> rows{{0,x[pos],pos}};
            if(pos<y.size())rows.push_back({1,y[pos],pos});
            auto next=cold(std::move(rows));candidate[0]=next[0];if(pos<y.size())candidate[1]=next[1];
        }
        fault_cursor=reference.size();
        auto next=cold({mm::Row{0,candidate[0],x.size()},mm::Row{0,7,x.size()+1},mm::Row{1,candidate[1],y.size()}});
        next=cold({mm::Row{0,11,x.size()+2},mm::Row{1,next[2],y.size()+1}});
        (void)cold({mm::Row{0,next[0],x.size()+3}}); // terminal pending commit
        (void)cold({mm::Row{0,91,0}});(void)cold({mm::Row{0,92,1}}); // source release/reuse
    }
    Resources resources{
        [] { Runner::synchronize(); },
        [&] { return !run.dd && !eng.prefill_open(); },
        [&](auto slot,auto n,void* p){eng.export_rows(slot,0,n,p);},
        [&](auto slot,auto n,const void* p){eng.import_rows(slot,0,n,p);},
        [&](const void* out,void* in,std::size_t n){
            try { eng.exchange_bytes(out,in,n); } catch (...) {eng.abort("paired state transport");throw;}
        },
        [&]{eng.abort("paired state protocol");},
        [&](std::uint64_t n){return strix::minimax::session::memory_admitted(
            0,n,budget,strix::minimax::session::available_memory());}};
    run.state_endpoint=std::make_shared<Endpoint>(identity,budget-reference_charge,std::move(resources),input_mode?InputProfile::plain_greedy_r1:InputProfile::off);
    if(input_mode)run.input_witness=[&](std::span<const mm::Row> rows,std::span<const std::uint32_t> next) {
        ++native_calls;native_rows+=rows.size();need(cursor<reference.size(),"MiniMax extra native grouped call");
        const auto& expected=reference[cursor];
        need(rows.size()==expected.rows.size() && std::equal(next.begin(),next.end(),expected.next.begin(),expected.next.end()),"MiniMax complete native candidate IDs");
        for(std::size_t i=0;i<rows.size();++i)need(rows[i].slot==expected.rows[i].slot && rows[i].token==expected.rows[i].token &&
            rows[i].pos==expected.rows[i].pos,"MiniMax native causal row grouping");
        const auto logits=eng.last_logits(rows.size());
        need(logits.size()==expected.logits.size() && !std::memcmp(logits.data(),expected.logits.data(),logits.size()*sizeof(float)),"MiniMax rank raw logits");
        Bytes image;
        for(std::size_t i=0;i<rows.size();++i) {
            image.resize(expected.images[i].size());eng.export_rows(rows[i].slot,0,rows[i].pos+1,image.data());
            need(image==expected.images[i],"MiniMax rank complete native row images");
        }
        std::printf("rank %u input-witness cycle=%zu physical=%zu logits=%s rows=%s next=",eng.rank(),cursor,rows.size(),
            hex(fnv(logits.data(),logits.size()*sizeof(float))).c_str(),hex(fnv(image.data(),image.size())).c_str());
        for(auto token:next)std::printf("%u,",token);std::printf("\n");
        if(ticket!="input" && !injected && cursor==fault_cursor) {
            injected=true;
            if(eng.rank()==(ticket=="fault0"?0U:1U))throw std::runtime_error("operator native fault AFTER row/logit writes");
            return; // BOTH ranks retry the same reference cycle after rollback
        }
        ++cursor;
    };
    const auto check_counts=[&] {
        if(!input_mode)return;
        std::uint64_t expected_rows=0;for(const auto& ref:reference)expected_rows+=ref.rows.size();
        need(native_calls==reference.size()+(ticket=="input"?0:1) &&
             native_rows==expected_rows+(ticket=="input"?0:reference[fault_cursor].rows.size()),"MiniMax actual grouped call/row counts");
    };
    if(eng.rank()==1) {
        follow(eng,run,nullptr);check_counts();
        need(!input_mode || (cursor==reference.size() && (ticket=="input" || injected)),"MiniMax follower witness incomplete");
        std::printf("rank 1 native grouped calls=%llu physical_rows=%llu\n",static_cast<unsigned long long>(native_calls),static_cast<unsigned long long>(native_rows));
        run.input_witness={};run.state_endpoint.reset();return 0;
    }
    // Same controls and follower as production. No second scheduler or queue.
    struct ProbeRunner {
        Runner& run;mm::Engine& eng;strix::pair::Lockstep lock;
        void send(Control c) {(void)exchange(eng,c,lock);}
        void prefill_begin(std::uint32_t slot,std::span<const std::uint32_t> prompt,bool context,
                           std::uint64_t task,std::uint64_t parent,std::uint64_t generation) {
            Control c;c.command=static_cast<std::uint32_t>(Command::prefill);c.pslot=slot;
            c.plen=static_cast<std::uint32_t>(prompt.size());c.task_id=task;c.parent_task_id=parent;
            c.slot_generation=generation;c.pflags=context?1U:0U;send(c);
            std::vector<std::uint32_t> ids(prompt.begin(),prompt.end());run.send_prompt(ids);
            run.prefill_begin(slot,prompt,context,task,parent,generation);
        }
        bool prefill_unit(unsigned layers,std::uint32_t* next) {
            Control c;c.command=static_cast<std::uint32_t>(Command::unit);c.layers=layers;
            c.task_id=eng.prefill_progress().id;send(c);return run.prefill_unit(layers,next);
        }
        std::vector<std::uint32_t> step(std::span<const mm::Row> rows,std::uint32_t append) {
            Control c;c.command=static_cast<std::uint32_t>(Command::step);c.rows=static_cast<std::uint32_t>(rows.size());
            std::copy(rows.begin(),rows.end(),c.row);c.append=append;send(c);return run.step(rows,append);
        }
        Drafts draft(std::uint32_t,std::uint32_t,std::uint64_t) {throw std::runtime_error("probe drafter unsupported");}
        void synchronize(){Runner::synchronize();}
        bool input_supported(InputProfile p)const{return run.input_supported(p);}
        Identity input_identity()const{return run.input_identity();}
        auto input(std::span<const mm::Row> rows,std::span<const strix::session::KvHandle> handles) {
            auto c=ms::input_control(rows,handles);send(c);return run.input(rows,handles);
        }
        void state(ms::SessionPlan& p){run.state(p);}
    } probe{run,eng,strix::pair::Lockstep(static_cast<std::uint32_t>(eng.rank()),run.control_round)};
    run.state_owner=std::make_shared<Owner>(*run.state_endpoint,[&](Control c){probe.send(c);},budget-reference_charge);
    if(input_mode) {
        namespace ss=strix::session;
        ms::SessionBackend<ProbeRunner> backend(probe);
        ms::SessionEngine adapter(backend,eng.slots(),eng.capacity(),[&](auto h){run.state_endpoint->forget(h.slot);},
            [&]{return eng.prefill_progress();},true,InputProfile::plain_greedy_r1);
        ss::KvStore store(eng.slots(),2,adapter.storage_hooks());adapter.attach(store);
        auto a=store.acquire(),b=store.acquire(),c=store.acquire();
        adapter.prefill(std::vector<std::uint32_t>{123},c);auto bystander=store.capture(c);
        std::array<ss::PrefillItem,2> fill{{{x,a},{y,b}}};adapter.prefill_batch(fill);
        const auto check_candidate=[&](auto h,const ss::StepResult& result) {
            const auto snapshot=store.capture(h);const auto image=std::static_pointer_cast<const RankImage>(snapshot.ranks[0].owner)->image;
            need(result.tokens.size()==1 && !result.rng_draws && result.tokens[0]==image->frontier.retained,"MiniMax retained decode frontier");
        };
        ss::SamplerState live;live.max_tokens=8;
        std::array<ss::DecodeItem,2> decode{{{a,live,{}},{b,live,{}}}};
        auto selected=adapter.decode_batch(decode);check_candidate(a,selected[0]);check_candidate(b,selected[1]);
        need(selected[0].tokens[0]==reference[x.size()].next[0] && selected[1].tokens[0]==reference[y.size()].next[1],"MiniMax first complete output IDs");
        need(selected[0].rows.physical_rows==0 && selected[1].rows.physical_rows==0,"MiniMax retained-only zero physical rows");
        auto olda=store.capture(a),oldb=store.capture(b);
        const auto same=[](const ss::StateSnapshot& lhs,const ss::StateSnapshot& rhs) {
            auto l=std::static_pointer_cast<const RankImage>(lhs.ranks[0].owner)->image;
            auto r=std::static_pointer_cast<const RankImage>(rhs.ranks[0].owner)->image;
            return lhs.tokens==rhs.tokens && lhs.pending==rhs.pending && l->rows==r->rows && l->frontier==r->frontier;
        };
        const std::vector<std::uint32_t> suffix{7,11};
        std::array<ss::AdvanceItem,2> mixed{{{a,suffix,live,{},ss::RowHints{2,1.0}},{b,{},live,{},{}}}};
        if(ticket!="input") {
            bool failed=false;try{(void)adapter.advance(mixed);}catch(const std::exception&){failed=true;}
            need(failed && injected,"MiniMax real post-write fault reached");
            need(same(store.capture(a),olda) && same(store.capture(b),oldb),"MiniMax full native paired rollback");
            need(same(store.capture(c),bystander),"MiniMax fault bystander");
        }
        auto out=adapter.advance(mixed);need(out[0].input_tokens==1 && out[0].output.tokens.empty() && out[0].rows.physical_rows==2 &&
            out[1].output.tokens.size()==1 && out[1].output.tokens[0]==reference[fault_cursor].next[2] &&
            !out[1].output.rng_draws,"MiniMax pending anchor and mixed receipt");
        std::array<ss::AdvanceItem,2> tail{{{a,std::span(suffix).subspan(1),live,{},{}},{b,{},live,{},{}}}};
        out=adapter.advance(tail);need(out[0].input_tokens==1 && out[0].output.tokens.empty() &&
            out[1].output.tokens==std::vector<std::uint32_t>{reference[fault_cursor+1].next[1]},"MiniMax suffix input progress and complete output IDs");
        ss::SamplerState zero=live;zero.max_tokens=0;
        const auto no_output=adapter.decode(b,zero);need(no_output.tokens.empty() && no_output.rows.physical_rows==0 &&
            no_output.finish==ss::FinishReason::length,"MiniMax zero output budget");
        ss::SamplerState one;one.max_tokens=1;auto terminal=adapter.decode(a,one);
        check_candidate(a,terminal);need(terminal.tokens[0]==reference[fault_cursor+1].next[0],"MiniMax terminal output ID");adapter.commit_pending(a);const auto before_repeat=cursor;adapter.commit_pending(a);
        need(cursor==before_repeat && !store.pending(a),"MiniMax repeat terminal commit");
        auto frozen=store.capture(a);const auto source=a;store.release(a);a=store.acquire();
        adapter.prefill(std::vector<std::uint32_t>{91,92},a);need(a.generation!=source.generation,"MiniMax source generation reuse");
        store.restore(b,frozen);auto restored=store.capture(b);
        auto fi=std::static_pointer_cast<const RankImage>(frozen.ranks[0].owner)->image;
        auto ri=std::static_pointer_cast<const RankImage>(restored.ranks[0].owner)->image;
        need(fi->rows==ri->rows && fi->frontier.retained==ri->frontier.retained && ri->frontier.finish==ss::FinishReason::length &&
             frozen.tokens==restored.tokens && frozen.pending==restored.pending,"MiniMax full native continuation snapshot restore");
        need(same(store.capture(c),bystander),"MiniMax bystander release/reuse/restore");
        adapter.resume_lane(b);auto resumed=adapter.decode(b,live);check_candidate(b,resumed);
        need(resumed.tokens[0]==reference[fault_cursor+2].next[0],"MiniMax restored complete output ID");
        std::printf("%s\n%s\n",adapter.receipt(a).json().c_str(),adapter.receipt(b).json().c_str());
        need(cursor==reference.size(),"MiniMax missing native grouped calls");check_counts();
        std::printf("rank 0 native grouped calls=%llu physical_rows=%llu rng_draws=0\n",static_cast<unsigned long long>(native_calls),static_cast<unsigned long long>(native_rows));
    } else {
        ms::SessionBackend<ProbeRunner> backend(probe);
        ms::SessionEngine adapter(backend,eng.slots(),eng.capacity(),[&](auto h){run.state_endpoint->forget(h.slot);},
                                 [&]{return eng.prefill_progress();},true);
        strix::session::KvStore store(eng.slots(),2,adapter.storage_hooks());adapter.attach(store);
        auto source=store.acquire(),dest=store.acquire(),bystander=store.acquire();
        std::uint64_t task=0;
        const auto fill=[&](auto h,std::span<const std::uint32_t> ids){adapter.begin(h,ids,false,++task,0);std::uint32_t next=0;
            while(!adapter.unit(h,60,&next)){} adapter.commit(h,next);};
        fill(source,prompt);fill(bystander,prompt);
        auto frozen=store.capture(source),other=store.capture(bystander);
        const auto view=store.view(source,1);
        need(std::static_pointer_cast<const RankImage>(view.owner)->rank==1,"probe peer view");
        store.release(source);source=store.acquire();
        std::vector<std::uint32_t> replacement(prompt.begin(),prompt.end());replacement[0]=(replacement[0]+1)%200064;
        fill(source,replacement);
        store.restore(dest,frozen);auto restored=store.capture(dest);
        need(restored.tokens==frozen.tokens && restored.pending==frozen.pending,"probe canonical restore");
        auto a=std::static_pointer_cast<const RankImage>(frozen.ranks[0].owner)->image;
        auto b=std::static_pointer_cast<const RankImage>(restored.ranks[0].owner)->image;
        need(a->rows==b->rows,"probe both-rank native row restoration");
        const mm::Row row{dest.slot,*store.pending(dest),store.consumed(dest)};
        const auto ids=adapter.step(std::span(&row,1),0,std::span(&dest,1));adapter.commit(dest,ids[0]);
        auto unchanged=store.capture(bystander);
        need(std::static_pointer_cast<const RankImage>(other.ranks[0].owner)->image->rows==
             std::static_pointer_cast<const RankImage>(unchanged.ranks[0].owner)->image->rows,"probe bystander");
    }
    Control bye;bye.command=static_cast<std::uint32_t>(Command::shutdown);probe.send(bye);
    run.control_round=probe.lock.round();run.input_witness={};run.state_owner.reset();run.state_endpoint.reset();return 0;
} catch (...) {eng.abort("paired state operator probe");throw;}

// ---- both ranks, before READY: the resumable prefill vs prefill() with decode steps of other slots in between ----
// Each request has a complete, fixed-length continuation (stop IDs are retained). Compare all IDs, committed states,
// prompt output rows and continuation logits. Include idle bounded resumes and two queued prompts in reversed order.
int resume_check(mm::Engine& eng, Runner& run, const std::string& files, unsigned tokens, const std::string& nested_file, unsigned child_layers) try {
    if (eng.slots() < 3 || tokens == 0) throw std::invalid_argument("resume-check: requires at least 3 slots and positive continuation length");
    struct Result {
        std::array<std::vector<std::uint32_t>, 3> ids;
        std::array<std::uint64_t, 3> state{}, logits{}, out{};
        std::array<std::uint32_t, ds::block> draft{};
        bool operator==(const Result&) const = default;
    };
    const auto receipt = [&](const Result& a, const Result& ref) {
        for (unsigned slot = 0; slot < 3; ++slot) {
            if (ref.ids[slot].empty()) continue;
            std::printf("rank %u resume-check slot=%u ids=%zu/%zu ids_fnv=%s/%s state=%s/%s out=%s/%s logits=%s/%s\n",
                        eng.rank(), slot, a.ids[slot].size(), ref.ids[slot].size(),
                        hex(fnv(a.ids[slot].data(), a.ids[slot].size() * 4)).c_str(),
                        hex(fnv(ref.ids[slot].data(), ref.ids[slot].size() * 4)).c_str(),
                        hex(a.state[slot]).c_str(), hex(ref.state[slot]).c_str(),
                        hex(a.out[slot]).c_str(), hex(ref.out[slot]).c_str(),
                        hex(a.logits[slot]).c_str(), hex(ref.logits[slot]).c_str());
        }
    };
    eng.set_prefill_check_outputs(true);
    int bad = 0;
    bool nested_ran = false;
    std::size_t at = 0;
    while (at <= files.size()) {
        const auto comma = files.find(',', at);
        std::string f = files.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        at = comma == std::string::npos ? files.size() + 1 : comma + 1;
        if (f.empty()) continue;
        std::vector<unsigned> ks{1, 7, 60};
        if (const auto colon = f.rfind(':'); colon != std::string::npos) {
            ks.clear();
            for (std::size_t a = colon + 1; a < f.size();) {
                const auto slash = f.find('/', a);
                ks.push_back(static_cast<unsigned>(std::stoul(f.substr(a, slash == std::string::npos ? std::string::npos : slash - a))));
                a = slash == std::string::npos ? f.size() : slash + 1;
            }
            f.resize(colon);
        }
        std::ifstream in(f, std::ios::binary);
        std::vector<std::uint32_t> p;
        for (std::uint32_t t; in.read(reinterpret_cast<char*>(&t), 4);) p.push_back(t);
        if (p.size() < 80 || p.size() + tokens >= eng.capacity())
            throw std::invalid_argument("resume-check: prompt needs >=80 tokens and room for the complete continuation");
        const std::vector<std::uint32_t> q(p.begin(), p.begin() + 64);
        const bool ctx = run.dd != nullptr && p.size() <= 4096;
        auto hash_logits = [&](Result& r, unsigned slot, std::size_t rows, std::size_t row) {
            const auto logits = eng.last_logits(rows);
            const auto width = logits.size() / rows;
            r.logits[slot] = fnv(logits.data() + row * width, width * sizeof(float),
                                  r.logits[slot] == 0 ? 14695981039346656037ULL : r.logits[slot]);
        };
        // Capture before another prefill, decode or draft can replace the prompt output/head.
        const auto completed = [&](Result& r, unsigned slot, std::uint32_t next) {
            r.ids[slot].push_back(next);
            r.out[slot] = eng.prefill_check_output_fnv();
            hash_logits(r, slot, 1, 0);
        };
        const auto arm = [&](unsigned k, bool idle) {
            Result r;
            std::array<std::uint64_t, 3> pos{p.size(), q.size(), q.size()};
            auto backgrounds = [&] {
                for (unsigned slot = 1; slot <= 2; ++slot) {
                    std::uint32_t next = 0;
                    run.prefill_begin(slot, q, false, ++run.local_id);
                    (void)run.prefill_unit(~0U, &next);
                    completed(r, slot, next);
                }
            };
            auto step = [&] {
                const std::array<mm::Row, 2> rows{mm::Row{1, r.ids[1].back(), pos[1]}, mm::Row{2, r.ids[2].back(), pos[2]}};
                const auto g = run.step(rows, 0);
                for (unsigned slot = 1; slot <= 2; ++slot) {
                    ++pos[slot];
                    r.ids[slot].push_back(g[slot - 1]);
                    hash_logits(r, slot, 2, slot - 1);
                }
            };
            if (!idle) backgrounds();
            if (k != 0 && !idle && tokens > 1) step();
            run.prefill_begin(0, p, ctx, ++run.local_id);
            unsigned units = 0;
            for (bool done = false; !done;) {
                unsigned budget = k == 0 ? ~0U : k;
                if (idle && k != 0) {
                    const ms::LayerModel model;
                    budget = ms::resume_layers(std::min(budget, ms::unit_layers(250, false, 1500, 0, model.fixed_ms,
                        model.layer_ms(std::max<std::uint64_t>(1, eng.prefill_unit_rows())))));
                }
                std::uint32_t next = 0;
                done = run.prefill_unit(budget, &next);
                ++units;
                if (done) completed(r, 0, next);
                if (!idle && k != 0 && r.ids[1].size() < tokens) step();
            }
            if (ctx) r.draft = run.draft(0, r.ids[0].back(), p.size()).tokens;
            if (idle) backgrounds();
            while (r.ids[1].size() < tokens) step();
            for (unsigned i = 1; i < tokens; ++i) {
                const mm::Row row{0, r.ids[0].back(), pos[0]++};
                const auto g = run.step(std::span(&row, 1), 0);
                r.ids[0].push_back(g[0]);
                hash_logits(r, 0, 1, 0);
            }
            for (unsigned slot = 0; slot < 3; ++slot) r.state[slot] = eng.state_fnv(slot, pos[slot]);
            if (idle && k != 0 && *eng.prefill_unlimited_reason() == 0 && units < 2) ++bad;
            std::printf("rank %u resume-check prompt_tokens=%zu k=%u idle=%u units=%u fallback=%s\n",
                        eng.rank(), p.size(), k, idle ? 1U : 0U, units, eng.prefill_unlimited_reason());
            return r;
        };
        const Result ref = arm(0, false);
        for (unsigned k : ks) {
            if (k == 0) throw std::invalid_argument("resume-check: k >= 1");
            for (bool idle : {false, true}) {
                const auto a = arm(k, idle);
                const bool ok = a == ref;
                receipt(a, ref);
                bad += ok ? 0 : 1;
                std::printf("rank %u resume-check k=%u idle=%u continuation=%u all IDs/state/prompt outputs/first+continuation logits/draft: %s\n",
                            eng.rank(), k, idle ? 1U : 0U, tokens, ok ? "IDENTICAL" : "MISMATCH");
            }
        }
        // Two waiting prompts submitted together. Rank 0 selects; the follower receives only the selected slot.
        const auto reorder = [&](bool short_first) {
            Result r;
            sv::GenerationQueue queue;
            if (eng.rank() == 0) {
                sv::GenerationJob long_job, short_job;
                long_job.prompt = p; short_job.prompt = q;
                (void)queue.submit(std::move(long_job)); (void)queue.submit(std::move(short_job));
            }
            for (unsigned i = 0; i < 2; ++i) {
                std::optional<sv::TakenJob> job;
                unsigned slot = 0;
                if (eng.rank() == 0) {
                    job = queue.take(false, short_first ? sv::QueueSelection::short_first : sv::QueueSelection::fifo);
                    slot = job->job.prompt.size() == p.size() ? 0U : 1U;
                }
                slot = eng.agree(slot);
                if (slot != (short_first ? 1 - i : i)) ++bad;
                const auto& prompt = slot == 0 ? p : q;
                run.prefill_begin(slot, prompt, false, ++run.local_id);
                std::uint32_t next = 0;
                (void)run.prefill_unit(~0U, &next);
                completed(r, slot, next);
                std::uint64_t pos = prompt.size();
                for (unsigned j = 1; j < tokens; ++j) {
                    const mm::Row row{slot, r.ids[slot].back(), pos++};
                    r.ids[slot].push_back(run.step(std::span(&row, 1), 0)[0]);
                    hash_logits(r, slot, 1, 0);
                }
                r.state[slot] = eng.state_fnv(slot, pos);
                if (job) queue.finish(*job, {});
            }
            return r;
        };
        const auto fifo = reorder(false);
        const auto shortest = reorder(true);
        receipt(shortest, fifo);
        const bool ordered = shortest == fifo;
        bad += ordered ? 0 : 1;
        std::printf("rank %u resume-check short-first complete continuations/state/prompt outputs/first+continuation logits: %s\n", eng.rank(), ordered ? "IDENTICAL" : "MISMATCH");
        if (nested_file.empty()) continue;
        nested_ran = true;
        // Explicit nested qualification never truncates the original fixture.
        std::ifstream short_in(nested_file, std::ios::binary);
        if (!short_in) throw std::invalid_argument("resume-check nested: cannot open fixture");
        const auto short_prompt = ms::nested_fixture(short_in, eng.nested_rows(), eng.capacity(), tokens);
        if (*eng.prefill_unlimited_reason() != 0)
            throw std::runtime_error("resume-check nested: requires bounded path (NOT a pass)");
        const std::array<std::vector<std::uint32_t>, 3> prompts{p, short_prompt, q};
        const auto nested_arm = [&](unsigned checkpoint, bool background_step) {
            Result r;
            std::array<std::uint64_t, 3> pos{p.size(), short_prompt.size(), q.size()};
            strix::pair::Lockstep lock(static_cast<std::uint32_t>(eng.rank()), run.control_round);
            const auto command = [&](Command op, std::uint64_t id, std::uint64_t parent = 0,
                                     unsigned slot = 0, unsigned layers = 0) {
                Control out;
                if (eng.rank() == 0) {
                    out.command = static_cast<std::uint32_t>(op);
                    out.task_id = id; out.parent_task_id = parent; out.slot_generation = 1;
                    out.pslot = slot; out.layers = layers;
                    out.bank = op == Command::nest ? 1U : eng.prefill_progress().bank;
                    if (op == Command::check) out.plen = run.last_next;
                    if (op == Command::prefill || op == Command::nest) out.plen = static_cast<std::uint32_t>(prompts[slot].size());
                }
                const auto peer = exchange(eng, out, lock);
                run.control_round = lock.round();
                const auto& c = eng.rank() == 0 ? out : peer;
                (void)ms::replay_command(c, eng, run, [&](std::size_t n) {
                    auto prompt = eng.rank() == 0 ? prompts[slot] : std::vector<std::uint32_t>{};
                    exchange_tokens(eng, prompt, n, eng.rank() == 0);
                    return prompt;
                });
            };
            const auto unit = [&](std::uint64_t id, unsigned layers, std::uint32_t* next) {
                command(Command::unit, id, 0, 0, layers);
                *next = run.last_next;
                return !eng.prefill_active();
            };
            const auto begin = [&](unsigned slot, std::uint64_t id, std::uint64_t parent) {
                command(parent == 0 ? Command::prefill : Command::nest, id, parent, slot);
            };
            const auto checked_completion = [&](unsigned slot, std::uint32_t next) {
                command(Command::check, 100 + slot);
                completed(r, slot, next);
            };
            const auto advance = [&](unsigned slot) {
                const mm::Row row{slot, r.ids[slot].back(), pos[slot]++};
                r.ids[slot].push_back(run.step(std::span(&row, 1), 0)[0]);
                hash_logits(r, slot, 1, 0);
            };
            const auto complete = [&](unsigned slot) {
                while (r.ids[slot].size() < tokens) advance(slot);
                r.state[slot] = eng.state_fnv(slot, pos[slot]);
            };
            std::uint32_t next = 0;
            if (checkpoint == 0) {
                for (unsigned slot = 0; slot < 3; ++slot) {
                    begin(slot, 100 + slot, 0);
                    (void)unit(100 + slot, ~0U, &next);
                    checked_completion(slot, next);
                    complete(slot);
                }
            } else {
                begin(2, 102, 0);
                (void)unit(102, ~0U, &next);
                checked_completion(2, next);
                begin(0, 100, 0);
                if (unit(100, checkpoint, &next)) throw std::runtime_error("nested check did not park an incomplete long");
                const auto before = eng.prefill_progress();
                const auto live = eng.prefill_activation_fnv(100);
                if (background_step && tokens > 1) advance(2); // overwrite the shared eight-row save's source
                command(Command::park, 100);
                const auto frozen = eng.prefill_activation_fnv(100);
                if (live != frozen) ++bad;
                begin(1, 101, 100);
                unsigned child_units = 0;
                while (!unit(101, child_layers, &next)) {
                    ++child_units;
                    if (background_step && r.ids[2].size() < tokens) advance(2);
                    if (eng.prefill_activation_fnv(100) != frozen) ++bad;
                }
                if (child_layers < 60 && child_units == 0) ++bad;
                checked_completion(1, next); // consume child output/head before rebinding
                if (eng.prefill_activation_fnv(100) != frozen) ++bad;
                command(Command::resume_parked, 100);
                if (eng.prefill_progress() != before) throw std::runtime_error("nested checkpoint changed");
                (void)unit(100, ~0U, &next);
                checked_completion(0, next);
                for (;;) {
                    std::vector<mm::Row> rows;
                    for (unsigned slot = 0; slot < 3; ++slot)
                        if (r.ids[slot].size() < tokens) rows.push_back({slot, r.ids[slot].back(), pos[slot]++});
                    if (rows.empty()) break;
                    const auto g = run.step(rows, 0);
                    for (std::size_t i = 0; i < rows.size(); ++i) {
                        r.ids[rows[i].slot].push_back(g[i]);
                        hash_logits(r, rows[i].slot, rows.size(), i);
                    }
                }
                for (unsigned slot = 0; slot < 3; ++slot) complete(slot);
            }
            return r;
        };
        if (std::none_of(ks.begin(), ks.end(), [](unsigned k) { return k > 0 && k < 60; }))
            throw std::invalid_argument("resume-check nested: supply at least one park layer in 1..59");
        const auto standalone = nested_arm(0, false);
        for (unsigned checkpoint : ks) {
            if (checkpoint >= 60) continue; // existing k=60 is a unit size, not a valid first-group park layer
            for (bool background_step : {false, true}) {
                const auto nested = nested_arm(checkpoint, background_step);
                receipt(nested, standalone);
                const bool identical = nested == standalone;
                bad += identical ? 0 : 1;
                std::printf("rank %u resume-check nested parent_rows=%zu short_rows=%zu park_layer=%u background_step=%u standalone B1 IDs/state/output/first+continuation logits: %s\n",
                    eng.rank(), p.size(), short_prompt.size(), checkpoint, background_step ? 1U : 0U, identical ? "IDENTICAL" : "MISMATCH");
            }
        }
        std::fflush(stdout);
    }
    if (!nested_file.empty() && !nested_ran) throw std::invalid_argument("resume-check nested: no parent fixtures");
    eng.set_prefill_check_outputs(false);
    return bad + static_cast<int>(eng.peer(static_cast<std::uint32_t>(bad)));
} catch (...) {
    eng.abort("control channel: protocol failure");
    throw;
}

} // namespace strix::minimax::serve::native
