#include "strix/session/recurrent_state.hpp"
#include "strix/session/kv_pages.hpp"
#include "strix/session/prefix_cache.hpp"

#include <array>
#include <iostream>
#include <limits>
#include <map>
#include <type_traits>

using namespace strix::session;
namespace {
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F&& f) {
    try { f(); } catch (const std::exception&) { return; }
    throw std::runtime_error("expected rejection");
}
template<class F> void unsupported(F&& f) {
    try { f(); } catch (const UnsupportedOperation&) { return; }
    throw std::runtime_error("expected explicit unsupported result");
}
struct Dense {
    static inline int live = 0;
    std::array<std::uint64_t,4> matrix{};
    std::array<std::uint64_t,2> conv{};
    Dense() { ++live; }
    Dense(const Dense& d) : matrix(d.matrix), conv(d.conv) { ++live; }
    Dense(Dense&& d) noexcept : matrix(d.matrix), conv(d.conv) { ++live; }
    ~Dense() { --live; }
    bool operator==(const Dense& other) const { return matrix == other.matrix && conv == other.conv; }
};
using Lane = LaneState<Dense>;
static_assert(!std::is_copy_constructible_v<Lane>);
static_assert(std::is_const_v<std::remove_reference_t<decltype(std::declval<Lane>().base()->payload)>>);
struct Page {
    static inline int live = 0;
    std::array<std::uint64_t,4> rows{};
    Page() { ++live; }
    Page(const Page& p) : rows(p.rows) { ++live; }
    ~Page() { --live; }
};
using Pages = KvPages<Page>;
struct Cursors {
    std::uint64_t draft = 0, rng = 0, selection = 0, position = 0;
    std::uint64_t seed = 913, generated = 0, supplied = 0;
    bool terminal = false, busy = false;
    bool operator==(const Cursors&) const = default;
};
struct Image {
    Lane lane;
    Pages::Snapshot pages;
    Cursors cursors;
};
struct Rank {
    Lane lane;
    Pages pages;
    Cursors cursors;
    Rank(Lane state, Pages::Clone clone) : lane(std::move(state)), pages(std::move(clone)) {}
};

// Bounded canonical host-fixture encoding: little-endian scalar fields, never
// pointers, padding, bool object representations or raw struct memory.
class Writer {
public:
    std::vector<std::uint8_t> bytes;
    std::size_t limit;
    explicit Writer(std::size_t bound) : limit(bound) {}
    void room(std::size_t n) {
        if (bytes.size() > limit || n > limit - bytes.size() || n > bytes.max_size() - bytes.size())
            throw std::length_error("serialization size limit");
    }
    void integer(std::uint64_t value) {
        room(8);
        for (unsigned i=0;i<8;++i) bytes.push_back(static_cast<std::uint8_t>(value >> (8*i)));
    }
    void string(const std::string& value) {
        if (value.size() > UINT32_MAX) throw std::length_error("serialization string length");
        integer(value.size()); room(value.size());
        bytes.insert(bytes.end(), value.begin(), value.end());
    }
    void frontier(LaneFrontier f) {
        integer(f.consumed); integer(f.pending.has_value()); integer(f.pending.value_or(0));
    }
};

// A correctness-only adapter, not a production model/backend. It uses the real
// store's sealed hooks and real page COW; no second snapshot store is implemented.
class HostOps final : public RecurrentStateOps {
public:
    LaneTrust trust;
    std::uint32_t ranks;
    std::string base_key;
    std::map<std::pair<std::uint32_t,std::uint32_t>,Rank> physical;
    std::uint64_t clones = 0, allocations = 0, restores = 0, evaluations = 0;
    int fail_eval_rank = -1, fail_dense_rank = -1;
    bool fail_clone = false, fail_allocate = false, fail_capture = false;
    bool fail_fork = false, fail_release = false, fail_restore = false;
    bool fail_rollback = false, rolling_back = false, portable = true;
    std::size_t wire_limit = 65536;
    mutable std::uint64_t dense_clone_calls = 0;
    std::uint64_t fail_dense_clone_at = 0;
    KvStore store; // destroyed before hook resources above

    explicit HostOps(std::uint32_t slots = 8, std::uint32_t rank_count = 2,
                     LaneTrust tokens = {}, std::string base = "origin-0")
        : trust(std::move(tokens)), ranks(rank_count), base_key(std::move(base)), store(slots,rank_count,hooks()) {}
    LaneIdentity identity(std::uint32_t r) const {
        return {1,ranks,r,"host-dense-v1","ordered-host-ranks","fixture-model","integer-reference-v1",base_key};
    }
    LaneFrontier frontier(KvHandle h) const { return {store.consumed(h),store.pending(h)}; }
    Rank& rank(KvHandle h, std::uint32_t r) {
        (void)store.consumed(h);
        return physical.at({h.slot,r});
    }
    auto dense_clone() const {
        return [this](const Dense& d) -> std::unique_ptr<Dense> {
            ++dense_clone_calls;
            if (fail_allocate || dense_clone_calls == fail_dense_clone_at) throw std::bad_alloc();
            return std::make_unique<Dense>(d);
        };
    }
    Pages::Clone page_clone() {
        return [this](const Page& p) {
            if (fail_clone) throw std::bad_alloc();
            auto copy = std::make_unique<Page>(p); ++clones; return copy;
        };
    }
    Hooks hooks() {
        if (!trust.domain) trust.domain = std::make_shared<const int>(0);
        if (!trust.schema) trust.schema = std::make_shared<const int>(0);
        Hooks h;
        h.snapshot_domain = trust.domain; h.snapshot_schema = trust.schema;
        h.copy_on_write_supported = h.partial_restore_supported = true;
        h.initialize = [this](KvHandle k) {
            ++allocations;
            for (std::uint32_t r=0;r<ranks;++r) {
                physical.try_emplace({k.slot,r},Lane::create(identity(r),trust,{},Dense{}),page_clone());
                if (fail_allocate) throw std::bad_alloc();
            }
        };
        h.release = [this](KvHandle k) {
            if (fail_release) throw std::runtime_error("release failure");
            for (std::uint32_t r=0;r<ranks;++r) physical.erase({k.slot,r});
        };
        h.reset = [this](KvHandle k) {
            for (std::uint32_t r=0;r<ranks;++r) {
                physical.erase({k.slot,r});
                physical.try_emplace({k.slot,r},Lane::create(identity(r),trust,{},Dense{}),page_clone());
            }
        };
        h.capture = [this](KvHandle k, std::uint32_t r) {
            if (fail_capture && r == 1) throw std::bad_alloc();
            const auto& s = physical.at({k.slot,r});
            if (s.cursors.busy) throw std::invalid_argument("in-flight capture");
            auto image = std::make_shared<const Image>(Image{
                s.lane.clone(identity(r),trust,dense_clone()),s.pages.capture(),s.cursors});
            return RankState{image,static_cast<std::uint64_t>(sizeof(Dense)+sizeof(Cursors)+s.pages.size()*sizeof(Page))};
        };
        h.view = h.capture;
        auto install = [this](KvHandle k, std::uint32_t r, const RankState& state) {
            // Only KvStore calls this hook, AFTER checking sealed owner/order,
            // origin or BOTH trusted tokens, and rank count. No public cast API.
            const auto& image = *static_cast<const Image*>(state.owner.get());
            image.lane.validate(identity(r),trust);
            auto lane = image.lane.clone(identity(r),trust,dense_clone());
            auto pages = Pages(image.pages,page_clone());
            auto& target = physical.at({k.slot,r});
            target.lane = std::move(lane); target.pages = std::move(pages); target.cursors = image.cursors;
        };
        h.restore = [this,install](KvHandle k, std::uint32_t r, const RankState& state) {
            ++restores; install(k,r,state);
            if (r == 1 && rolling_back && fail_rollback) throw std::runtime_error("rollback rank failure");
            if (r == 1 && fail_restore) {
                fail_restore = false; rolling_back = true;
                throw std::runtime_error("restore changed throwing rank");
            }
        };
        h.fork_rank = [this,install](KvHandle k, std::uint32_t r, const RankState& state) {
            install(k,r,state);
            if (r == 1 && fail_fork) throw std::runtime_error("unpublished fork failure");
        };
        h.truncate = [](KvHandle, std::uint32_t, std::uint64_t) {
            throw UnsupportedOperation("interior rollback requires complete saved checkpoint");
        };
        return h;
    }
    void quiescent(KvHandle h, bool writable = true) const {
        const auto f = frontier(h);
        for (std::uint32_t r=0;r<ranks;++r) {
            const auto& s = physical.at({h.slot,r});
            s.lane.validate(identity(r),trust); s.lane.require_dense_reference();
            if (s.lane.frontier() != f || s.cursors.position != f.consumed || s.cursors.busy ||
                (writable && s.cursors.terminal)) throw std::invalid_argument("lane frontier/turn/quiescence");
        }
    }
    template<class Work> void transaction(KvHandle h, Work&& work) {
        auto saved = snapshot(h);
        try { work(); } catch (...) {
            auto error = std::current_exception();
            try { store.restore(h,saved); } catch (...) { store.invalidate(h); }
            std::rethrow_exception(error);
        }
    }
    StateSnapshot snapshot(KvHandle h) const override {
        quiescent(h,false); return store.capture(h);
    }
    KvHandle fork(const StateSnapshot& saved) override {
        // Store checks all compatibility before allocation and erased hooks.
        if (saved.consumed > 24 || (saved.pending && saved.consumed == 24))
            throw std::length_error("fork capacity");
        return store.fork(saved);
    }
    void evaluate(KvHandle h, std::uint32_t token) {
        for (std::uint32_t r=0;r<ranks;++r) {
            auto& s = rank(h,r);
            const auto i = s.cursors.position;
            // Unsigned fake recurrence: proves orchestration, no floating-point
            // operation-order or real-model equivalence assertion.
            s.lane.update(identity(r),trust,{i+1,{}},dense_clone(),[&](Dense& d) {
                for (std::size_t j=0;j<d.matrix.size();++j)
                    d.matrix[j] = d.matrix[j]*131 + token + r + j;
                d.conv = {d.conv[1],token+r};
            });
            if (i/4 == s.pages.size()) {
                if (fail_allocate) throw std::bad_alloc();
                s.pages.append(std::make_unique<Page>());
            }
            s.pages.write(static_cast<std::size_t>(i/4),[&](Page& p) { p.rows[i%4] = token+1000ULL*r; });
            s.cursors.draft ^= token+17*i; s.cursors.selection += token%7;
            ++s.cursors.position; ++evaluations;
            if (static_cast<int>(r) == fail_eval_rank) throw std::runtime_error("evaluation rank failure");
        }
    }
    void set_anchor(KvHandle h, std::uint32_t token) {
        store.set_pending(h,token);
        for (std::uint32_t r=0;r<ranks;++r)
            rank(h,r).lane.update(identity(r),trust,frontier(h),dense_clone(),[](Dense&) {});
    }
    void append(KvHandle h, LaneFrontier expected, std::span<const std::uint32_t> input) override {
        quiescent(h);
        if (expected != frontier(h)) throw std::invalid_argument("stale append frontier");
        const auto after = validate_lane_append(expected,input,24,256);
        if (input.empty()) return;
        std::vector<std::uint32_t> evaluated;
        evaluated.reserve(input.size()+(expected.pending ? 1U : 0U));
        if (expected.pending) evaluated.push_back(*expected.pending);
        evaluated.insert(evaluated.end(),input.begin(),input.end());
        store.reserve_consumed(h,evaluated.size());
        transaction(h,[&] {
            for (auto token : evaluated) evaluate(h,token);
            store.append_consumed(h,evaluated);
            for (std::uint32_t r=0;r<ranks;++r) rank(h,r).cursors.supplied += input.size();
        });
        check(frontier(h) == after,"append published expected consumed route");
    }
    void rollback(KvHandle h, LaneFrontier target, const StateSnapshot& saved) override {
        quiescent(h,false); validate_lane_checkpoint(saved,target);
        store.restore(h,saved); // store restores attempted ranks or quarantines
    }
    void densify(KvHandle h) override {
        quiescent(h);
        transaction(h,[&] {
            for (std::uint32_t r=0;r<ranks;++r) {
                rank(h,r).lane.densify(identity(r),trust,dense_clone());
                if (static_cast<int>(r) == fail_dense_rank) throw std::runtime_error("densify rank failure");
            }
        });
    }
    LaneSerialization serialize(const StateSnapshot& saved) const override {
        // Validate via actual store BEFORE ANY erased payload interpretation.
        // The existing store has no validation-only API. This fixture uses a
        // temporary complete fork and releases it before return. It therefore
        // needs a spare slot; production encoders may use their own typed path.
        auto& self = const_cast<HostOps&>(*this);
        const auto temporary = self.fork(saved);
        try {
            if (!portable) {
                self.store.release(temporary);
                return {LaneSerializationStatus::unsupported,{},"opaque payload has no host encoding"};
            }
            Writer w(wire_limit);
            w.string("strix-host-lane"); w.integer(1); w.integer(ranks);
            w.frontier({saved.consumed,saved.pending}); w.integer(saved.tokens.size());
            for (auto t : saved.tokens) w.integer(t);
            for (std::uint32_t r=0;r<ranks;++r) {
                const auto& s = self.rank(temporary,r);
                const auto id = identity(r);
                w.integer(id.version); w.integer(id.ranks); w.integer(id.rank);
                w.string(id.schema); w.string(id.topology); w.string(id.model);
                w.string(id.numerical); w.string(id.base);
                w.frontier(s.lane.base()->frontier); w.frontier(s.lane.frontier());
                // Include canonical base contents and lane-local dense state.
                auto dense = [&](const Dense& d) {
                    for (auto x : d.matrix) w.integer(x);
                    for (auto x : d.conv) w.integer(x);
                };
                dense(s.lane.base()->payload); w.integer(s.lane.log()->size());
                dense(s.lane.read(id,trust));
                const auto& c = s.cursors;
                for (auto x : {c.draft,c.rng,c.selection,c.position,c.seed,c.generated,c.supplied}) w.integer(x);
                w.integer(c.terminal); w.integer(c.busy);
                w.integer(s.pages.size());
                for (std::size_t p=0;p<s.pages.size();++p)
                    for (auto x : s.pages.read(p).rows) w.integer(x);
            }
            self.store.release(temporary);
            return {LaneSerializationStatus::supported,std::move(w.bytes),{}};
        } catch (...) {
            self.store.release(temporary); throw;
        }
    }
    // One fixture-only seeded output. Inputs go through append; only this
    // output-selection operation advances RNG/generated counters.
    std::uint32_t generate(KvHandle h) {
        quiescent(h);
        if (frontier(h).consumed+(frontier(h).pending ? 1U : 0U) >= 24)
            throw std::length_error("output capacity");
        store.reserve_consumed(h,1);
        std::uint32_t output = 0;
        transaction(h,[&] {
            if (auto p = store.pending(h)) {
                evaluate(h,*p); store.append_consumed(h,std::span<const std::uint32_t>(&*p,1));
            }
            const auto& s = rank(h,0);
            const auto& c = s.cursors;
            const auto value = s.lane.read(identity(0),trust).matrix[0]+c.draft+c.selection+
                ((c.seed ^ (c.rng*6364136223846793005ULL)) >> 11);
            output = static_cast<std::uint32_t>(value%251);
            set_anchor(h,output);
            for (std::uint32_t r=0;r<ranks;++r) { ++rank(h,r).cursors.rng; ++rank(h,r).cursors.generated; }
        });
        return output; // all result storage is prepared before commit
    }
};

void equal(HostOps& ops, const StateSnapshot& a, const StateSnapshot& b) {
    check(ops.serialize(a).bytes == ops.serialize(b).bytes,"canonical full-state equality");
}
void append(HostOps& ops, KvHandle h, std::initializer_list<std::uint32_t> input) {
    ops.append(h,ops.frontier(h),std::span<const std::uint32_t>(input.begin(),input.size()));
}

void ownership_and_evaluation() {
    HostOps e;
    RecurrentStateOps& api = e;
    auto source = e.store.acquire();
    append(e,source,{1,2,3,4,5,6,7});
    const auto anchor = e.generate(source);
    auto saved = api.snapshot(source);
    check(saved.sealed() && saved.pending == anchor,"complete pending checkpoint");
    const auto original = api.serialize(saved).bytes;
    const auto before_clones = e.clones;
    auto a = api.fork(saved), b = api.fork(saved);
    check(e.clones == before_clones,"fork clones zero attention prefix pages");
    for (std::uint32_t r=0;r<2;++r) {
        auto& x=e.rank(a,r); auto& y=e.rank(b,r); auto& z=e.rank(source,r);
        check(x.lane.base() == y.lane.base() && x.lane.base() == z.lane.base(),"immutable checkpoint shared");
        check(x.lane.log()->empty() && x.lane.has_private_dense(),"empty-log dense fallback");
        check(&x.lane.read(e.identity(r),e.trust) != &y.lane.read(e.identity(r),e.trust),"dense copies independently owned");
        for (std::size_t p=0;p<2;++p)
            check(x.pages.owner(p) == y.pages.owner(p) && x.pages.owner(p) == z.pages.owner(p),"real attention owners shared");
    }
    e.store.release(source);
    const auto old=e.rank(b,0).pages.owner(1);
    const auto rng=e.rank(a,0).cursors.rng, outputs=e.rank(a,0).cursors.generated;
    const auto eval=e.evaluations;
    api.append(a,e.frontier(a),{});
    check(e.evaluations == eval && e.frontier(a).pending == anchor,"empty append preserves pending anchor");
    append(e,a,{91,92});
    check(e.evaluations == eval+6 && e.frontier(a) == LaneFrontier{10,{}},"teacher forcing consumes anchor once on each rank");
    check(e.rank(a,0).cursors.rng == rng && e.rank(a,0).cursors.generated == outputs,"input draws no sampler RNG or output count");
    check(e.rank(a,0).pages.owner(0) == e.rank(b,0).pages.owner(0) &&
          e.rank(a,0).pages.owner(1) != old && e.rank(b,0).pages.read(1).rows[3] == 0,"only touched page detached");
    check(e.rank(a,0).lane.read(e.identity(0),e.trust) != e.rank(b,0).lane.read(e.identity(0),e.trust),"dense sibling independence");
    check(api.serialize(saved).bytes == original,"source release and sibling writes preserve snapshot");

    // Independent scalar oracle includes every consumed token and pending anchor.
    Dense reference;
    for (auto token : {1U,2U,3U,4U,5U,6U,7U,anchor,91U,92U}) {
        for (std::size_t j=0;j<4;++j) reference.matrix[j]=reference.matrix[j]*131+token+j;
        reference.conv={reference.conv[1],token};
    }
    check(e.rank(a,0).lane.read(e.identity(0),e.trust) == reference,"actual dense recurrence evaluation");
    const auto continued=e.snapshot(a);
    const auto c=e.rank(a,0).cursors;
    const auto expected=static_cast<std::uint32_t>((reference.matrix[0]+c.draft+c.selection+
        ((c.seed ^ (c.rng*6364136223846793005ULL)) >> 11))%251);
    check(e.generate(a) == expected,"seeded continuation after input keeps sampler stream");
    api.rollback(a,{continued.consumed,continued.pending},continued);
    const auto predense=api.snapshot(a);
    api.densify(a); equal(e,predense,api.snapshot(a));
    check(e.generate(a) == expected,"rollback and densify preserve continuation");
    api.rollback(a,{saved.consumed,saved.pending},saved);
    equal(e,saved,api.snapshot(a)); equal(e,saved,api.snapshot(b));
    auto bytes=api.serialize(saved);
    check(bytes.status == LaneSerializationStatus::supported && bytes.bytes == original,"deterministic identity encoding");
    e.portable=false;
    auto no=api.serialize(saved);
    check(no.status == LaneSerializationStatus::unsupported && no.bytes.empty() && !no.reason.empty(),"opaque serialization explicitly unsupported");
    e.portable=true;
    auto changed=api.snapshot(a); changed.ranks[0].owner=std::make_shared<int>(17);
    rejects([&] { (void)api.serialize(changed); });
    check(api.serialize(saved).bytes == original,"retained metadata alias cannot substitute payload");
    e.store.release(a); e.store.release(b);
    auto resurrected=api.fork(saved); equal(e,saved,api.snapshot(resurrected));
    e.store.release(resurrected);
}

void compatibility_and_limits() {
    HostOps e;
    auto h=e.store.acquire(); append(e,h,{1,2,3}); auto saved=e.snapshot(h);
    const auto old=e.serialize(saved).bytes;
    for (int i=0;i<7;++i) {
        auto bad=e.identity(0);
        if(i==0) bad.schema="other";
        if(i==1) bad.model="other";
        if(i==2) bad.numerical="other";
        if(i==3) bad.topology="other";
        if(i==4) bad.rank=1;
        if(i==5) bad.base="unknown";
        if(i==6) bad.version=2;
        rejects([&] { (void)e.rank(h,0).lane.read(bad,e.trust); });
    }
    auto wrong=e.trust; wrong.schema=std::make_shared<const int>(0);
    rejects([&] { (void)e.rank(h,0).lane.read(e.identity(0),wrong); });
    wrong=e.trust; wrong.domain=std::make_shared<const int>(0);
    rejects([&] { (void)e.rank(h,0).lane.read(e.identity(0),wrong); });
    // Same address, different owner does not confer interpretation authority.
    wrong=e.trust; wrong.domain=std::shared_ptr<const void>(e.trust.domain.get(),[](const void*) {});
    rejects([&] { (void)e.rank(h,0).lane.read(e.identity(0),wrong); });
    HostOps same(8,2,e.trust), foreign, wrong_rank(8,1,e.trust);
    auto other=e.trust; other.schema=std::make_shared<const int>(0); HostOps wrong_schema(8,2,other);
    auto cross=same.fork(saved); same.store.release(cross);
    const auto allocations=foreign.allocations;
    rejects([&] { (void)foreign.fork(saved); }); rejects([&] { (void)wrong_rank.fork(saved); });
    rejects([&] { (void)wrong_schema.fork(saved); });
    check(foreign.allocations == allocations && wrong_rank.allocations == 0 && wrong_schema.allocations == 0,"foreign compatibility rejected before hooks/allocation");
    for (int i=0;i<5;++i) {
        auto bad=saved;
        if(i==0) bad.ranks[0].owner=std::make_shared<int>(0);
        if(i==1) std::swap(bad.ranks[0],bad.ranks[1]);
        if(i==2) ++bad.bytes;
        if(i==3) bad.pending=99;
        if(i==4) bad.tokens[0]=99;
        rejects([&] { (void)e.fork(bad); }); rejects([&] { e.rollback(h,{bad.consumed,bad.pending},bad); });
    }
    StateSnapshot unsealed; rejects([&] { (void)e.fork(unsealed); });
    rejects([&] { e.rollback(h,{2,{}},saved); });
    unsupported([&] { e.store.truncate(h,2); });
    rejects([&] { append(e,h,{256}); });
    std::vector<std::uint32_t> huge(24,1);
    rejects([&] { e.append(h,e.frontier(h),huge); });
    rejects([&] { e.append(h,{0,{}},{}); });
    e.rank(h,1).cursors.busy=true;
    rejects([&] { append(e,h,{1}); }); rejects([&] { (void)e.snapshot(h); });
    e.rank(h,1).cursors.busy=false; e.rank(h,1).cursors.terminal=true;
    rejects([&] { append(e,h,{1}); }); rejects([&] { e.densify(h); });
    e.rank(h,1).cursors.terminal=false;
    check(e.serialize(e.snapshot(h)).bytes == old,"invalid admission leaves whole state unchanged");
    auto stale=h; e.store.release(h); h=e.store.acquire();
    rejects([&] { append(e,stale,{1}); }); rejects([&] { (void)e.snapshot(stale); });
    rejects([&] { append(foreign,h,{1}); });
    rejects([&] { (void)validate_lane_append({UINT64_MAX,1},{},UINT64_MAX,256); });
    rejects([&] { (void)validate_lane_append({UINT64_MAX,{}},std::array<std::uint32_t,1>{1},UINT64_MAX,256); });
    rejects([&] { (void)validate_lane_append({0,300},{},24,256); });
    Writer bounded(7); rejects([&] { bounded.integer(1); });
    e.wire_limit=8; rejects([&] { (void)e.serialize(saved); }); e.wire_limit=65536;
    auto factors=Lane::create(e.identity(0),e.trust,{},Dense{},{1,2});
    unsupported([&] { (void)factors.read(e.identity(0),e.trust); });
    unsupported([&] { factors.densify(e.identity(0),e.trust,e.dense_clone()); });
    unsupported([&] { (void)factors.clone(e.identity(0),e.trust,e.dense_clone()); });
    auto empty=Lane::create(e.identity(0),e.trust,{},Dense{});
    const auto base=empty.base();
    empty.densify(e.identity(0),e.trust,e.dense_clone());
    check(empty.has_private_dense() && empty.base()==base && base->payload==Dense{},"densify materializes private state without changing base");
    rejects([&] { empty.densify(e.identity(0),e.trust,[](const Dense&) { return std::unique_ptr<Dense>{}; }); });
    check(empty.base()==base,"null clone failure atomicity");
    // Actual serialization includes base identity even with identical history
    // and dense contents. Its bytes do not grant foreign interpretation rights.
    HostOps alternate(8,2,{},"origin-1"); auto alternate_lane=alternate.store.acquire();
    append(alternate,alternate_lane,{1,2,3}); auto alternate_saved=alternate.snapshot(alternate_lane);
    check(alternate.serialize(alternate_saved).bytes != old,"base identity encoded in complete serialization");
    rejects([&] { (void)e.serialize(alternate_saved); });
}

void failures_and_lifetime() {
    HostOps e(8); auto source=e.store.acquire(); append(e,source,{1,2,3});
    auto saved=e.snapshot(source); auto lane=e.fork(saved); const auto baseline=e.serialize(saved).bytes;
    auto unchanged=[&] {
        check(e.serialize(e.snapshot(lane)).bytes==baseline,"failed operation restores every coupled rank");
        check(e.serialize(e.snapshot(source)).bytes==baseline,"unrelated sibling unaffected");
    };
    e.fail_eval_rank=1; rejects([&] { append(e,lane,{9}); }); e.fail_eval_rank=-1; unchanged();
    e.fail_clone=true; rejects([&] { append(e,lane,{9}); }); e.fail_clone=false; unchanged();
    e.fail_capture=true; rejects([&] { append(e,lane,{9}); }); e.fail_capture=false; unchanged();
    e.fail_allocate=true; rejects([&] { append(e,lane,{9}); });
    rejects([&] { (void)e.fork(saved); }); e.fail_allocate=false; unchanged();
    // Two snapshot clones, then rank-0 work, then rank-1 allocation failure.
    e.fail_dense_clone_at=e.dense_clone_calls+4;
    rejects([&] { append(e,lane,{9}); }); e.fail_dense_clone_at=0; unchanged();
    const auto dense_restores=e.restores;
    e.fail_dense_rank=1; rejects([&] { e.densify(lane); }); e.fail_dense_rank=-1;
    check(e.restores-dense_restores==2,"densify restores both attempted ranks"); unchanged();
    // Failure after pending metadata changes publishes no hidden output, and a
    // later evaluation failure must retain a previously committed anchor.
    e.fail_dense_clone_at=e.dense_clone_calls+4;
    rejects([&] { (void)e.generate(lane); }); e.fail_dense_clone_at=0; unchanged();
    const auto output=e.generate(lane); auto pending_saved=e.snapshot(lane);
    e.fail_eval_rank=1; rejects([&] { (void)e.generate(lane); }); e.fail_eval_rank=-1;
    equal(e,pending_saved,e.snapshot(lane));
    check(e.store.pending(lane)==output,"failed generation preserves committed reported anchor");
    e.rollback(lane,{saved.consumed,saved.pending},saved); unchanged();
    e.fail_fork=true; const auto live=e.physical.size();
    rejects([&] { (void)e.fork(saved); }); e.fail_fork=false;
    check(e.physical.size()==live,"failed fork releases unpublished rank payloads"); unchanged();
    append(e,lane,{9}); const auto advanced=e.snapshot(lane);
    const auto restore_start=e.restores; e.fail_restore=true;
    rejects([&] { e.rollback(lane,{saved.consumed,saved.pending},saved); }); e.rolling_back=false;
    check(e.restores-restore_start==4,"restore failure repairs both ranks including throwing rank");
    equal(e,advanced,e.snapshot(lane));
    e.fail_restore=true; e.fail_rollback=true;
    rejects([&] { e.rollback(lane,{saved.consumed,saved.pending},saved); });
    e.fail_rollback=false; e.rolling_back=false;
    rejects([&] { (void)e.snapshot(lane); });
    check(e.serialize(e.snapshot(source)).bytes==baseline,"quarantine confined to affected lane");
    e.store.release(lane); auto replacement=e.store.acquire();
    check(replacement.slot==lane.slot && replacement.generation!=lane.generation,"quarantined release reclaims generation");
    e.store.release(replacement);
    // An evaluation transaction whose restore fails must also quarantine.
    lane=e.fork(saved); e.fail_eval_rank=1; e.fail_restore=true;
    rejects([&] { append(e,lane,{9}); }); e.fail_eval_rank=-1; e.rolling_back=false;
    rejects([&] { (void)e.snapshot(lane); }); e.store.release(lane);

    HostOps capacity(2); auto root=capacity.store.acquire(); append(capacity,root,{1}); auto checkpoint=capacity.snapshot(root);
    capacity.fail_fork=true; capacity.fail_release=true;
    rejects([&] { (void)capacity.fork(checkpoint); }); capacity.fail_fork=false; capacity.fail_release=false;
    const auto count=capacity.allocations;
    rejects([&] { (void)capacity.fork(checkpoint); });
    check(capacity.allocations==count,"failed unpublished release quarantines capacity");
    // Backend teardown owns any unreleased resources; store retries at teardown.

    StateSnapshot held;
    LaneTrust retained;
    {
        HostOps owner; auto h=owner.store.acquire(); append(owner,h,{1,2,3,4,5});
        retained=owner.trust;
        auto entry=std::make_shared<PrefixSnapshot>();
        entry->identity={"fixture-model","integer-reference-v1","host-dense-v1",2};
        entry->state=owner.snapshot(h); entry->attention_frontier=entry->recurrent_frontier=5;
        Policy policy; policy.max_entries=1;
        PrefixCache cache(policy); check(cache.admit(entry),"complete checkpoint cache admission");
        auto hit=cache.lookup(std::array<std::uint32_t,5>{1,2,3,4,5},entry->identity);
        check(bool(hit),"cache holds typed complete rank owners");
        held=hit->state; entry->state.ranks[0].owner=std::make_shared<int>(0);
        check(held.sealed(),"cache freezes mutable metadata alias");
        owner.store.release(h);
    }
    HostOps receiver(8,2,retained); auto resumed=receiver.fork(held);
    equal(receiver,held,receiver.snapshot(resumed));
    check(receiver.rank(resumed,0).lane.log()->empty(),"snapshot survives source/store/cache destruction");
    held={}; append(receiver,resumed,{6}); receiver.store.release(resumed);
}
} // namespace

int main() {
    try {
        ownership_and_evaluation(); compatibility_and_limits(); failures_and_lifetime();
        check(Dense::live==0 && Page::live==0,"all snapshot/dense/page resources released");
        std::cout << "session recurrent host contracts passed\n";
    } catch (const std::exception& error) {
        std::cerr << "session recurrent failure: " << error.what() << '\n'; return 1;
    }
}
