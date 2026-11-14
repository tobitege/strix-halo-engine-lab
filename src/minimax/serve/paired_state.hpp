#pragma once
// Family-owned eager snapshots. The bytes are Engine::export_rows' native
// K/V/index representation, not a second named-session checkpoint codec.
#include "minimax_control.hpp"
#include "minimax_sched.hpp"
#include "strix/session/engine.hpp"
#include "strix/model_content_hash.hpp"
#include <bit>
#include <functional>
#include <limits>

namespace strix::minimax::serve {
namespace paired {
using Bytes = std::vector<std::uint8_t>;
// Library activation is typed and checked against a live native endpoint. Schema
// 1 remains the r4 state-only contract; schema 2 is this single row profile.
enum class InputProfile { off, plain_greedy_r1 };

inline void need(bool ok, const char* why) { if (!ok) throw std::invalid_argument(why); }
inline std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > UINT64_MAX-a) throw std::length_error("MiniMax state size overflow");
    return a+b;
}
inline std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (b && a > UINT64_MAX/b) throw std::length_error("MiniMax state size overflow");
    return a*b;
}
struct Identity {
    // SHA-256 of the existing loader/executable/effective-profile compatibility
    // record. Rank order and physical capacity are independently checked below.
    std::array<std::uint8_t,32> assets_profile{};
    std::uint64_t capacity = 0;
    std::uint32_t slots = 0, world = 2, schema = 1;
    bool operator==(const Identity&) const = default;
};
struct Frontier {
    strix::session::KvHandle handle;
    std::vector<std::uint32_t> tokens;
    std::optional<std::uint32_t> pending;
    std::uint64_t progress = 0, progress_limit = 0; // closed owner planner checkpoint
    std::optional<std::uint32_t> retained; // native greedy candidate at consumed boundary
    strix::session::FinishReason finish = strix::session::FinishReason::none;
    bool operator==(const Frontier&) const = default;
};
// Adapters are synchronous. Broken transport must wake/abort the pair, never
// return an unacknowledged success. No device addresses cross this seam.
struct Resources {
    std::function<void()> synchronize;
    std::function<bool()> supported; // no configured drafter/named/open/parked task
    std::function<void(std::uint32_t,std::uint64_t,void*)> export_rows;
    std::function<void(std::uint32_t,std::uint64_t,const void*)> import_rows;
    std::function<void(const void*,void*,std::size_t)> exchange;
    std::function<void()> abort;
    std::function<bool(std::uint64_t)> admit;
};
struct Ledger { std::uint64_t used = 0, limit = 0; };
struct Image {
    Identity identity;
    Frontier frontier;
    std::array<Bytes,2> rows;
    std::shared_ptr<Ledger> ledger;
    std::uint64_t charge = 0;
    ~Image() { if (ledger) ledger->used -= charge; }
};
struct RankImage { std::shared_ptr<const Image> image; std::uint32_t rank = 0; };
// Trusted erased payloads are accepted only by ownership identity, before cast.
// There is deliberately no cross-store schema/domain compatibility token.
class Owner;
class Endpoint {
public:
    static constexpr std::uint64_t row_bytes = 76800;
    enum class Op : std::uint32_t { capture = 1, prepare = 2, install = 3 };
    Endpoint(Identity identity, std::uint64_t budget, Resources resources, InputProfile profile=InputProfile::off)
        : id_(identity), budget_(budget), io_(std::move(resources)), input_profile_(profile) {
        need(std::endian::native == std::endian::little && ((id_.schema == 1 && profile==InputProfile::off) ||
             (id_.schema == 2 && profile==InputProfile::plain_greedy_r1)) && id_.world == 2 &&
             id_.slots && id_.slots <= 8 && id_.capacity && budget && budget != UINT64_MAX &&
             std::any_of(id_.assets_profile.begin(),id_.assets_profile.end(),[](auto v){return v!=0;}) &&
             io_.synchronize && io_.supported && io_.export_rows && io_.import_rows && io_.exchange && io_.abort && io_.admit,
             "MiniMax state invalid identity/resources/budget");
        need(base_bytes()<=budget_ && io_.admit(base_bytes()),"MiniMax state metadata budget");
        lanes_.resize(id_.slots);
    }
    // Called by the native row producer on EACH rank, only when activated.
    // Metadata allocations precede producer mutation.
    void begin(std::uint32_t slot, std::uint64_t generation, std::span<const std::uint32_t> tokens) {
        need(slot < lanes_.size() && generation && tokens.size() <= id_.capacity,"MiniMax state begin");
        Frontier f; f.handle = {slot,generation,1}; f.tokens.assign(tokens.begin(),tokens.end());
        lanes_[slot] = std::move(f);
    }
    void selected(std::uint32_t slot, std::uint32_t token) { lanes_.at(slot).pending = token; }
    void reserve_step(std::span<const engine::Row> rows) {
        std::uint32_t seen = 0;
        for (const auto& r : rows) {
            need(r.slot < lanes_.size() && !(seen & (1U<<r.slot)),"MiniMax state plain rows only");
            seen |= 1U<<r.slot;
            auto& f = lanes_[r.slot];
            need(f.pending == r.token && r.pos == f.tokens.size() && r.pos < id_.capacity,"MiniMax state row frontier");
            f.tokens.reserve(f.tokens.size()+1);
        }
    }
    void stepped(std::span<const engine::Row> rows, std::span<const std::uint32_t> next) {
        need(rows.size()==next.size(),"MiniMax state row result");
        for (std::size_t i=0;i<rows.size();++i) {
            auto& f=lanes_[rows[i].slot]; f.tokens.push_back(rows[i].token); f.pending=next[i];
        }
    }
    bool input_supported(InputProfile profile) const {
        return profile==input_profile_ && profile==InputProfile::plain_greedy_r1 && id_.schema==2 && io_.supported() && io_.admit(base_bytes());
    }
    const Frontier& frontier(strix::session::KvHandle h) const {
        need(h.slot<lanes_.size() && lanes_[h.slot].handle==h,"MiniMax input lease");
        return lanes_[h.slot];
    }
    // Contiguous teacher rows per lane, ONE native grouped execution on each rank.
    // Both ranks independently reserve/validate, vote BEFORE writes, then vote
    // AFTER synchronization. A recoverable native fault leaves the control loop
    // alive for the owner's full old-pair rollback. Transport faults abort it.
    template<class Execute>
    bool input(std::span<const engine::Row> rows, std::span<const strix::session::KvHandle> handles,
               Execute execute, std::vector<std::uint32_t>& next) try {
        bool ok=input_supported(InputProfile::plain_greedy_r1) && !rows.empty() && rows.size()<=8 && rows.size()==handles.size();
        try {
            need(ok,"MiniMax input unsupported shape/profile");
            std::uint32_t seen=0,lane_rows=0;
            for(std::size_t i=0;i<rows.size();++i) {
                const auto& r=rows[i];const auto h=handles[i];
                need(h.valid() && h.slot==r.slot && h.slot<lanes_.size(),"MiniMax input lease");
                auto& f=lanes_[h.slot];
                const bool continuation=i && rows[i-1].slot==r.slot;
                if(continuation) {
                    need(++lane_rows<=2,"MiniMax input lane shape");
                    need(handles[i-1]==h && r.pos==rows[i-1].pos+1,"MiniMax input causal row order");
                } else {
                    need(!(seen&(1U<<h.slot)) && r.pos==f.tokens.size() &&
                         (!f.pending || f.pending==r.token),"MiniMax input duplicate/frontier");
                    seen|=1U<<h.slot;lane_rows=1;
                }
                need(f.handle==h && r.pos<id_.capacity && r.token<200064,"MiniMax teacher row frontier");
                // The schema-2 metadata base precharges full lane capacity. This
                // also avoids later history allocations inside full-input cycles.
                f.tokens.reserve(static_cast<std::size_t>(id_.capacity));
            }
            next.reserve(rows.size());
        } catch(...) {ok=false;}
        if(!vote(ok))return false;
        try {
            next=execute();io_.synchronize();
            need(next.size()==rows.size() && std::all_of(next.begin(),next.end(),[](auto t){return t<200064;}),"MiniMax input candidate result");
        } catch(...) {ok=false;}
        if(!vote(ok))return false;
        verify_bytes(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(next.data()),next.size()*4));
        for(std::size_t i=0;i<rows.size();++i) {
            auto& f=lanes_[rows[i].slot];f.tokens.push_back(rows[i].token);
            f.pending.reset();f.retained=next[i];f.finish=strix::session::FinishReason::none;
        }
        return true;
    } catch(...) {io_.abort();throw;}
    void forget(std::uint32_t slot) { lanes_.at(slot) = {}; }
    const Identity& identity() const noexcept { return id_; }
    // Used by the real follower loop after its normal round validation.
    void follow(const Control& c) try {
        need(c.command == static_cast<std::uint32_t>(Command::state) && c.task_id == sequence_+1 &&
             c.pslot < lanes_.size() && c.slot_generation && c.pflags >= 1 && c.pflags <= 3,
             "MiniMax state stale/duplicate/invalid command");
        sequence_ = c.task_id;
        const auto op = static_cast<Op>(c.pflags);
        Header local{}, remote{}; io_.exchange(local.data(),remote.data(),remote.size());
        verify_bytes(remote);
        if(op==Op::install) {
            // All history/row allocations were staged by prepare, before rank 0
            // imported. Install carries only the exact prepared header.
            bool ok=io_.supported() && staged_ && remote==header(*staged_) &&
                    staged_->handle.slot==c.pslot && staged_->handle.generation==c.slot_generation;
            if(!vote(ok))return;
            try {
                io_.synchronize();
                if(!staged_rows_.empty())io_.import_rows(c.pslot,staged_->tokens.size(),staged_rows_.data());
                io_.synchronize();lanes_[c.pslot]=std::move(*staged_);staged_.reset();Bytes{}.swap(staged_rows_);
            } catch(...) {ok=false;}
            (void)vote(ok);return;
        }
        // A prior throwing rank-0 attempt may leave an unpublished peer stage.
        // Reclaim that scratch BEFORE admitting/allocating another operation;
        // active lanes and immutable snapshots never refer to this buffer.
        staged_.reset();Bytes{}.swap(staged_rows_);
        auto f = parse(remote,c);
        bool ok = io_.supported();
        Bytes raw;
        Frontier staged;
        if (op == Op::capture) {
            ok = ok && matches(f);
            try { admit(f.tokens.size()); raw.resize(size(f.tokens.size())); } catch (...) { ok=false; }
            if (!vote(ok)) return;
            try { io_.synchronize(); if (!raw.empty()) io_.export_rows(c.pslot,f.tokens.size(),raw.data()); }
            catch (...) { ok=false; }
            if (!vote(ok)) return;
            transfer(raw,false);
            lanes_[c.pslot]=std::move(f);
        } else if (op == Op::prepare) {
            ok=ok && lanes_[c.pslot].handle==f.handle;
            try { admit(f.tokens.size()); raw.resize(size(f.tokens.size())); staged=f;
                if(id_.schema==2)staged.tokens.reserve(static_cast<std::size_t>(id_.capacity)); }
            catch (...) { ok=false; }
            if (!vote(ok)) return;
            transfer(raw,true);
            // Stage ALL metadata before the owner can import rank 0.
            staged_ = std::move(staged); staged_rows_ = std::move(raw);
            (void)vote(true);
        }
    } catch (...) { io_.abort(); throw; }
private:
    friend class Owner;
    using Header = std::array<std::uint8_t,144>;
    Identity id_;
    std::uint64_t budget_, sequence_ = 0;
    Resources io_;
    InputProfile input_profile_ = InputProfile::off;
    std::vector<Frontier> lanes_;
    std::optional<Frontier> staged_;
    Bytes staged_rows_;
    static std::size_t size(std::uint64_t n) {
        auto bytes=mul(n,row_bytes); need(bytes <= SIZE_MAX,"MiniMax state host size");
        return static_cast<std::size_t>(bytes);
    }
    std::uint64_t base_bytes() const { return add(mul(mul(id_.slots,id_.capacity),id_.schema==2?32:8),id_.schema==2?8192:4096); }
    std::uint64_t peak(std::uint64_t n) const { return add(base_bytes(),add(mul(size(n),4),add(mul(n,32),262144))); }
    void admit(std::uint64_t n) const {
        need(n <= id_.capacity && peak(n) <= budget_ && io_.admit(peak(n)),"MiniMax state scratch/resource budget");
    }
    bool matches(const Frontier& f) const {
        const auto& l=lanes_.at(f.handle.slot);
        // A newly leased empty lane has no causal physical rows to export.
        // This is also KvStore's rollback checkpoint before fresh-lane restore.
        if(f.tokens.empty() && !f.pending && !f.retained)return true;
        // Store identity is owner-local. Follower retains the same generation,
        // canonical IDs and selected pending anchor, independently produced.
        return l.handle.generation==f.handle.generation && l.tokens==f.tokens &&
               (id_.schema==2 ? (l.retained==f.retained && (!f.pending || l.pending==f.pending || l.retained==f.pending)) :
                (!f.pending || l.pending==f.pending)); // shared same-length pending clear
    }
    static void put(Header& h,std::size_t off,std::uint64_t v) {
        for (unsigned i=0;i<8;++i) {h[off+i]=static_cast<std::uint8_t>(v);v>>=8;}
    }
    static std::uint64_t get(const Header& h,std::size_t off) {
        std::uint64_t v=0;for(unsigned i=0;i<8;++i)v|=std::uint64_t(h[off+i])<<(8*i);return v;
    }
    Header header(const Frontier& f) const {
        Header h{}; std::copy(id_.assets_profile.begin(),id_.assets_profile.end(),h.begin());
        put(h,32,id_.schema);put(h,40,id_.world);put(h,48,id_.capacity);put(h,56,id_.slots);
        put(h,64,f.handle.store);put(h,72,f.handle.generation);put(h,80,f.handle.slot);
        put(h,88,f.tokens.size());put(h,96,f.pending.has_value());put(h,104,f.pending.value_or(0));
        put(h,112,row_bytes);put(h,120,f.progress);put(h,128,f.progress_limit);
        if(id_.schema==2)put(h,136,(f.retained ? 1ULL | (std::uint64_t(*f.retained)<<1) : 0ULL) |
                                      (std::uint64_t(f.finish)<<32));
        return h;
    }
    Frontier parse(const Header& h,const Control& c) {
        need(std::equal(id_.assets_profile.begin(),id_.assets_profile.end(),h.begin()) &&
             get(h,32)==id_.schema && get(h,40)==id_.world && get(h,48)==id_.capacity &&
             get(h,56)==id_.slots && get(h,72)==c.slot_generation && get(h,80)==c.pslot &&
             get(h,88)<=id_.capacity && get(h,96)<=1 && get(h,104)<200064 &&
             get(h,112)==row_bytes && (id_.schema==2 || get(h,136)==0) && get(h,120)<=get(h,128) && get(h,64),"MiniMax state header identity/length");
        admit(get(h,88));
        Frontier f;f.handle={c.pslot,c.slot_generation,get(h,64)};
        if(get(h,96))f.pending=static_cast<std::uint32_t>(get(h,104));
        f.progress=get(h,120);f.progress_limit=get(h,128);
        if(id_.schema==2) {
            const auto selection=get(h,136);const auto token=(selection & 0xffffffffULL)>>1;
            need((selection>>32)<=4 && token<200064 && ((selection&1) || token==0),"MiniMax input selection header");
            if(selection&1)f.retained=static_cast<std::uint32_t>(token);
            f.finish=static_cast<strix::session::FinishReason>(selection>>32);
            need(!f.retained || !f.tokens.empty() || get(h,88)>0,"MiniMax empty retained boundary");
        }
        // Receive history in fixed bounded chunks; scalars are explicitly LE.
        if(id_.schema==2)f.tokens.reserve(static_cast<std::size_t>(id_.capacity));
        f.tokens.resize(static_cast<std::size_t>(get(h,88)));
        history(std::span(f.tokens));return f;
    }
    template<class T> void history(std::span<T> tokens) {
        std::array<std::uint8_t,65536> out{},in{};
        for(std::size_t p=0;p<tokens.size();p+=16384) {
            auto n=std::min<std::size_t>(16384,tokens.size()-p);
            if constexpr(std::is_const_v<T>)
                for(std::size_t i=0;i<n;++i)for(unsigned j=0;j<4;++j)out[i*4+j]=static_cast<std::uint8_t>(tokens[p+i]>>(j*8));
            io_.exchange(out.data(),in.data(),n*4);
            if constexpr(!std::is_const_v<T>)for(std::size_t i=0;i<n;++i) {
                std::uint32_t t=0;for(unsigned j=0;j<4;++j)t|=std::uint32_t(in[i*4+j])<<(j*8);
                need(t<200064,"MiniMax state token range");tokens[p+i]=t;
            }
        }
        verify_bytes(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(tokens.data()),tokens.size()*4));
    }
    bool vote(bool ok) {
        std::uint8_t out=ok?1:0,in=0;io_.exchange(&out,&in,1);
        if(in>1){io_.abort();throw std::runtime_error("MiniMax state malformed acknowledgement");}
        return ok && in==1;
    }
    void transfer(Bytes& bytes,bool receive) {
        std::array<std::uint8_t,65536> zero{},in{};
        for(std::size_t p=0;p<bytes.size();p+=65536) {
            auto n=std::min<std::size_t>(65536,bytes.size()-p);
            io_.exchange(receive?zero.data():bytes.data()+p,receive?bytes.data()+p:in.data(),n);
        }
        verify_bytes(bytes);
    }
    void send_rows(std::span<const std::uint8_t> bytes) {
        std::array<std::uint8_t,65536> in{};
        for(std::size_t p=0;p<bytes.size();p+=65536)
            io_.exchange(bytes.data()+p,in.data(),std::min<std::size_t>(65536,bytes.size()-p));
        verify_bytes(bytes);
    }
    void verify_bytes(std::span<const std::uint8_t> bytes) {
        const auto hash=strix::sha256_bytes_fast(std::as_bytes(bytes));
        std::array<std::uint8_t,32> peer{};
        io_.exchange(hash.bytes.data(),peer.data(),peer.size());
        if(peer!=hash.bytes){io_.abort();throw std::runtime_error("MiniMax state row transport digest mismatch");}
    }
};

class Owner {
public:
    using Send = std::function<void(Control)>;
    Owner(Endpoint& endpoint,Send send,std::uint64_t budget)
        : e_(endpoint),send_(std::move(send)),ledger_(std::make_shared<Ledger>(Ledger{0,budget})) {
        need(send_ && budget && budget!=UINT64_MAX,"MiniMax state owner budget");
    }
    std::array<strix::session::RankState,2> capture(const Frontier& f) {
        need(f.handle.valid() && f.handle.slot<e_.id_.slots && f.tokens.size()<=e_.id_.capacity &&
             (!f.pending || *f.pending<200064) && (!f.retained || (*f.retained<200064 && !f.tokens.empty())) &&
             static_cast<unsigned>(f.finish)<=4 &&
             (e_.id_.schema==2 || (!f.retained && f.finish==strix::session::FinishReason::none)) &&
             std::all_of(f.tokens.begin(),f.tokens.end(),[](auto t){return t<200064;}),"MiniMax state capture frontier");
        need(e_.io_.supported() && e_.matches(f),"MiniMax state unsupported/noncanonical capture");
        e_.admit(f.tokens.size());
        const auto charge=add(mul(Endpoint::size(f.tokens.size()),2),add(mul(f.tokens.size(),8),4096));
        need(add(ledger_->used,add(charge,e_.peak(f.tokens.size())))<=ledger_->limit,"MiniMax state retained/transport budget");
        std::erase_if(known_,[](const auto& p){return p.expired();});
        known_.shrink_to_fit(); // no retained high-water registry after owner release
        known_.reserve(known_.size()+2);
        auto canonical=f;
        if(e_.id_.schema==2)canonical.tokens.reserve(static_cast<std::size_t>(e_.id_.capacity));
        auto image=std::make_shared<Image>(); image->identity=e_.id_;image->frontier=f;
        for(auto& r:image->rows)r.resize(Endpoint::size(f.tokens.size()));
        // Immutable owner and wrapper allocations also precede export.
        std::array<std::shared_ptr<const RankImage>,2> ranks{
            std::make_shared<const RankImage>(RankImage{image,0}),std::make_shared<const RankImage>(RankImage{image,1})};
        image->ledger=ledger_;image->charge=charge;ledger_->used+=charge;
        request(Endpoint::Op::capture,f);
        if(!e_.vote(true))throw std::runtime_error("MiniMax state peer capture refused");
        bool ok=true;
        try {e_.io_.synchronize();if(!image->rows[0].empty())e_.io_.export_rows(f.handle.slot,f.tokens.size(),image->rows[0].data());}
        catch(...){ok=false;}
        if(!e_.vote(ok))throw std::runtime_error("MiniMax state paired export failed");
        e_.transfer(image->rows[1],true);
        e_.lanes_[f.handle.slot]=std::move(canonical);
        std::array<strix::session::RankState,2> result;
        for(unsigned r=0;r<2;++r) {
            result[r]={ranks[r],charge/2+(r==0?charge%2:0)};
            known_.push_back(ranks[r]);
        }
        return result;
    }
    std::pair<std::uint64_t,std::uint64_t> restore(strix::session::KvHandle h,unsigned rank,const strix::session::RankState& state) {
        need(rank<2 && h.slot<e_.id_.slots && h.generation,"MiniMax state destination");
        need(e_.lanes_[h.slot].handle==h,"MiniMax state stale destination generation");
        std::shared_ptr<const RankImage> payload;
        for(auto it=known_.begin();it!=known_.end();) {
            if(auto p=it->lock()) {
                if(p.get()==state.owner.get() && !p.owner_before(state.owner) && !state.owner.owner_before(p))payload=p;
                ++it;
            } else it=known_.erase(it);
        }
        need(payload && payload->rank==rank && payload->image->identity==e_.id_,"MiniMax state payload provenance/rank/schema");
        auto image=payload->image;
        const auto expected=image->charge/2+(rank==0?image->charge%2:0);
        need(state.bytes==expected && e_.io_.supported(),"MiniMax state payload bytes/profile");
        for(const auto& r:image->rows)need(r.size()==Endpoint::size(image->frontier.tokens.size()),"MiniMax state incomplete pair");
        e_.admit(image->frontier.tokens.size());
        need(add(ledger_->used,e_.peak(image->frontier.tokens.size()))<=ledger_->limit,"MiniMax restore scratch budget");
        if(rank==0) {
            auto f=image->frontier;f.handle=h; // allocate metadata BEFORE either physical import
            if(e_.id_.schema==2)f.tokens.reserve(static_cast<std::size_t>(e_.id_.capacity));
            request(Endpoint::Op::prepare,f);
            if(!e_.vote(true))throw std::runtime_error("MiniMax state peer prepare refused");
            // transfer does not write the immutable image on the sender
            e_.send_rows(image->rows[1]);
            if(!e_.vote(true))throw std::runtime_error("MiniMax state peer staging failed");
            staged_image_=image;staged_handle_=h;
            e_.io_.synchronize();
            if(!image->rows[0].empty())e_.io_.import_rows(h.slot,f.tokens.size(),image->rows[0].data());
            e_.io_.synchronize();e_.lanes_[h.slot]=std::move(f);
        } else {
            need(staged_image_.lock()==image && staged_handle_==h,"MiniMax state restore rank order");
            request(Endpoint::Op::install,e_.lanes_[h.slot]);
            if(!e_.vote(true) || !e_.vote(true))throw std::runtime_error("MiniMax state peer import failed");
            staged_image_.reset();
        }
        return {image->frontier.progress,image->frontier.progress_limit};
    }
private:
    Endpoint& e_;Send send_;std::shared_ptr<Ledger> ledger_;
    std::vector<std::weak_ptr<const RankImage>> known_;
    std::weak_ptr<const Image> staged_image_;
    strix::session::KvHandle staged_handle_;
    void request(Endpoint::Op op,const Frontier& f) try {
        need(e_.sequence_!=UINT64_MAX,"MiniMax state sequence exhausted");
        Control c;c.command=static_cast<std::uint32_t>(Command::state);c.task_id=++e_.sequence_;
        c.pslot=f.handle.slot;c.slot_generation=f.handle.generation;c.pflags=static_cast<std::uint32_t>(op);
        send_(c);
        auto out=e_.header(f);Endpoint::Header in{};e_.io_.exchange(out.data(),in.data(),out.size());
        e_.verify_bytes(out);
        if(op!=Endpoint::Op::install)e_.history(std::span<const std::uint32_t>(f.tokens));
    } catch(...) {e_.io_.abort();throw;}
};
} // namespace paired
} // namespace strix::minimax::serve
