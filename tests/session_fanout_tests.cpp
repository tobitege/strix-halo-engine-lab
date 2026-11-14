#include "strix/session/engine.hpp"
#include "strix/session/kv_pages.hpp"
#include "strix/session/prefix_cache.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>

using namespace strix::session;
namespace {
void require(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
template<class F> void reject(F&& f) {
  try { f(); } catch (const std::exception&) { return; }
  throw std::runtime_error("expected rejection");
}
struct Page {
  static inline int live = 0;
  std::array<std::uint64_t,4> rows{};
  Page() { ++live; }
  Page(const Page& p) : rows(p.rows) { ++live; }
  ~Page() { --live; }
};
using Pages = KvPages<Page>;
struct Cursors {
  std::uint64_t recurrent = 0, draft = 0, rng = 0, policy = 0, cursor = 0;
  std::uint64_t seed = 0, generated = 0, supplied = 0;
  bool terminal = false, busy = false;
  bool operator==(const Cursors&) const = default;
};
struct Image { Pages::Snapshot pages; Cursors cursors; };
struct Rank {
  Pages pages;
  Cursors cursors;
  explicit Rank(Pages::Clone clone) : pages(std::move(clone)) {}
};

// Host fixture only. Real adapters must independently qualify their complete
// payloads, page write barriers, numerical routes and native execution.
class HostEngine final : public Engine {
public:
  std::map<std::pair<std::uint32_t,std::uint32_t>, Rank> physical;
  std::uint32_t rank_count;
  std::uint64_t clones = 0, cloned_bytes = 0, initializes = 0, restores = 0, releases = 0;
  std::uint64_t evaluated_rows = 0;
  std::uint64_t continuation_seed = 0; // explicit new-job policy in this fixture
  bool fail_clone = false, fail_initialize = false, fail_capture = false;
  bool fail_fork = false, fail_release = false, fail_restore = false;
  bool fail_rollback = false, rolling_back = false, fail_evaluate = false;
  Capabilities available;
  KvStore store; // destroyed before captured-hook resources above

  explicit HostEngine(std::uint32_t slots = 12, std::uint32_t ranks = 2, bool cow = true,
      std::shared_ptr<const void> domain = {}, std::shared_ptr<const void> schema = {})
      : rank_count(ranks), store(slots,ranks,hooks(cow,std::move(domain),std::move(schema))) {
    available.max_context = 24; available.max_slots = slots; available.max_batch = slots;
    available.pair = ranks == 2; available.seed = true; available.temperature = true; available.batch_decode = true;
    available.fork_lanes = cow; available.shared_prefix_cow = cow;
    available.request_lane_batch = true; available.lane_cancellation = true;
    available.external_append = true; available.lane_resume = true;
  }
  Rank& rank(KvHandle h, std::uint32_t r) {
    (void)store.consumed(h);
    return physical.at({h.slot,r});
  }
  Pages::Clone cloner() {
    return [this](const Page& p) {
      if (fail_clone) throw std::bad_alloc();
      auto out = std::make_unique<Page>(p);
      ++clones; cloned_bytes += sizeof(p.rows);
      return out;
    };
  }
  Hooks hooks(bool cow, std::shared_ptr<const void> domain, std::shared_ptr<const void> schema) {
    Hooks h;
    h.partial_restore_supported = true;
    h.copy_on_write_supported = cow;
    h.snapshot_domain = std::move(domain); h.snapshot_schema = std::move(schema);
    h.initialize = [this](KvHandle k) {
      ++initializes;
      for (std::uint32_t r=0;r<rank_count;++r) {
        physical.try_emplace({k.slot,r},cloner());
        if (fail_initialize && r == 0) throw std::bad_alloc();
      }
    };
    h.release = [this](KvHandle k) {
      ++releases;
      if (fail_release) throw std::runtime_error("cleanup failure");
      for (std::uint32_t r=0;r<rank_count;++r) physical.erase({k.slot,r});
    };
    h.reset = [this](KvHandle k) {
      for (std::uint32_t r=0;r<rank_count;++r) physical.at({k.slot,r}) = Rank(cloner());
    };
    h.capture = [this](KvHandle k, std::uint32_t r) {
      if (fail_capture && r == 1) throw std::bad_alloc();
      const auto& state = physical.at({k.slot,r});
      auto image = std::make_shared<const Image>(Image{state.pages.capture(),state.cursors});
      return RankState{image, state.pages.size()*sizeof(Page::rows) + sizeof(Cursors)};
    };
    h.view = h.capture;
    h.capture_prompt = h.capture;
    h.restore_prompt = [this](KvHandle k, std::span<const RankState> ranks) {
      std::vector<Rank> prepared;
      for (const auto& rank_state : ranks) {
        const auto& saved=*static_cast<const Image*>(rank_state.owner.get());
        prepared.emplace_back(cloner());
        prepared.back().pages=Pages(saved.pages,cloner());
        auto& c=prepared.back().cursors; c=saved.cursors;
        c.seed=continuation_seed; c.rng=0; c.generated=0;
        c.draft=0; c.policy=0; c.terminal=false; c.busy=false;
      }
      for (std::uint32_t r=0;r<rank_count;++r) physical.at({k.slot,r})=std::move(prepared[r]);
      return true;
    };
    h.restore = [this](KvHandle k, std::uint32_t r, const RankState& state) {
      ++restores;
      // Only store-issued sealed payloads in this adapter's compatible domain
      // reach this cast. Rank order and complete schema were checked by store.
      const auto& image = *static_cast<const Image*>(state.owner.get());
      auto& target = physical.at({k.slot,r});
      target.pages = Pages(image.pages, cloner());
      target.cursors = image.cursors;
      if (r == 1 && rolling_back && fail_rollback) throw std::bad_alloc();
      if (r == 1 && fail_restore) {
        fail_restore = false; rolling_back = true;
        throw std::runtime_error("restore changed throwing rank");
      }
    };
    h.fork_rank = [this](KvHandle k, std::uint32_t r, const RankState& state) {
      const auto& image = *static_cast<const Image*>(state.owner.get());
      auto& target = physical.at({k.slot,r});
      target.pages = Pages(image.pages,cloner()); target.cursors = image.cursors;
      if (r == 1 && fail_fork) throw std::runtime_error("fork rank failure");
    };
    h.truncate = [this](KvHandle k, std::uint32_t r, std::uint64_t prefix) {
      // This fake only uses equal-frontier truncation. Arbitrary recurrent
      // truncation needs a genuine family checkpoint at that frontier.
      if (physical.at({k.slot,r}).cursors.cursor != prefix)
        throw std::invalid_argument("no historical recurrent checkpoint");
    };
    return h;
  }
  void active(KvHandle h) {
    (void)store.consumed(h);
    for (std::uint32_t r=0;r<rank_count;++r) {
      const auto& c = rank(h,r).cursors;
      if (c.busy || c.terminal) throw std::invalid_argument("lane requires quiescence/resume");
    }
  }
  void evaluate(KvHandle h, std::uint32_t token) {
    ++evaluated_rows; // one logical forward row across the participating ranks
    for (std::uint32_t r=0;r<rank_count;++r) {
      auto& s = rank(h,r);
      const auto index = static_cast<std::size_t>(s.cursors.cursor);
      if (index/4 == s.pages.size()) s.pages.append(std::make_unique<Page>());
      s.pages.write(index/4,[&](Page& p) { p.rows[index%4] = token + 1000ULL*r; });
      s.cursors.recurrent = s.cursors.recurrent*131U + token + r;
      s.cursors.draft ^= token + s.cursors.cursor*17U;
      s.cursors.policy += token%7U;
      ++s.cursors.cursor;
      if (r == 1 && fail_evaluate) throw std::runtime_error("evaluation failure");
    }
  }
  void rollback(KvHandle h, const StateSnapshot& saved) {
    try { store.restore(h,saved); } catch (...) { store.invalidate(h); throw; }
  }
  void prefill(std::span<const std::uint32_t> tokens, KvHandle& h) override {
    active(h);
    if (tokens.empty()) return;
    const auto logical = store.consumed(h) + (store.pending(h) ? 1U : 0U);
    if (logical > available.max_context || tokens.size() > available.max_context-logical)
      throw std::invalid_argument("context exhausted");
    for (auto t : tokens) if (t >= 256) throw std::invalid_argument("invalid token");
    std::vector<std::uint32_t> input;
    input.reserve(tokens.size());
    if (auto p = store.pending(h)) input.push_back(*p);
    input.insert(input.end(),tokens.begin(),tokens.end()-1);
    store.reserve_consumed(h,input.size());
    const auto saved = store.capture(h);
    try {
      for (auto t : input) evaluate(h,t);
      store.append_consumed(h,input); store.set_pending(h,tokens.back());
      for (std::uint32_t r=0;r<rank_count;++r) rank(h,r).cursors.supplied += tokens.size();
    } catch (...) {
      auto error = std::current_exception();
      try { rollback(h,saved); } catch (...) {}
      std::rethrow_exception(error);
    }
  }
  void validate_decode(KvHandle h, const SamplerState& s) {
    active(h);
    const auto& c = rank(h,0).cursors;
    if (s.rng_draws != c.rng || (s.parameters.seeded && s.parameters.seed != c.seed) ||
        s.parameters.temperature != (s.parameters.seeded ? 1.0 : 0.0) || s.parameters.top_p != 1 || s.parameters.top_k != 0 ||
        s.parameters.min_p != 0 || s.constraint) throw std::invalid_argument("sampler mismatch");
    if (store.consumed(h) == 0 && !store.pending(h)) throw std::invalid_argument("empty lane");
  }
  StepResult step(KvHandle& h, const SamplerState& sampler) {
    StepResult out;
    out.before = out.after = {store.consumed(h),store.pending(h)};
    out.rng_draws = rank(h,0).cursors.rng;
    if (sampler.cancellation.stop_requested()) {
      cancel_lane(h); out.finish = FinishReason::cancelled; return out;
    }
    const auto logical = store.consumed(h)+(store.pending(h) ? 1U : 0U);
    if (!sampler.max_tokens || logical == available.max_context) {
      out.finish = !sampler.max_tokens ? FinishReason::length : FinishReason::context;
      for (std::uint32_t r=0;r<rank_count;++r) rank(h,r).cursors.terminal = true;
      return out;
    }
    out.tokens.reserve(1); // metadata/result storage precedes physical mutation
    store.reserve_consumed(h,store.pending(h) ? 1U : 0U);
    if (auto pending = store.pending(h)) {
      evaluate(h,*pending);
      store.append_consumed(h,std::span<const std::uint32_t>(&*pending,1));
    }
    const auto& c = rank(h,0).cursors;
    const auto random = sampler.parameters.seeded ? ((c.seed ^ (c.rng*6364136223846793005ULL)) >> 11) : 0;
    const auto token = static_cast<std::uint32_t>((c.recurrent+c.draft+c.policy+random)%251U);
    out.tokens.push_back(token); store.set_pending(h,token);
    for (std::uint32_t r=0;r<rank_count;++r) {
      auto& x = rank(h,r).cursors;
      x.rng += sampler.parameters.seeded ? 1U : 0U; ++x.generated;
    }
    if (std::find(sampler.stop_tokens.begin(),sampler.stop_tokens.end(),token) != sampler.stop_tokens.end()) {
      out.finish = FinishReason::stop;
      for (std::uint32_t r=0;r<rank_count;++r) rank(h,r).cursors.terminal = true;
    }
    out.after = {store.consumed(h),store.pending(h)}; out.rng_draws = rank(h,0).cursors.rng;
    return out;
  }
  StepResult decode(KvHandle& h, const SamplerState& sampler) override {
    validate_decode(h,sampler);
    auto saved = store.capture(h);
    try { return step(h,sampler); } catch (...) {
      auto error = std::current_exception();
      try { rollback(h,saved); } catch (...) {}
      std::rethrow_exception(error);
    }
  }
  std::vector<StepResult> decode_batch(std::span<const DecodeItem> items) override {
    if (!available.batch_decode || !available.request_lane_batch || items.size() > available.max_batch)
      throw std::invalid_argument("unsupported lane batch");
    for (std::size_t i=0;i<items.size();++i) {
      (void)resolve_row_hints(items[i].row_hints, available.row_hints,
                             std::array<std::uint32_t,3>{2,4,8});
      validate_decode(items[i].state,items[i].sampler);
      for (std::size_t j=0;j<i;++j) if (items[j].state == items[i].state)
        throw std::invalid_argument("duplicate batch lane");
    }
    std::vector<StateSnapshot> saved;
    std::vector<StepResult> out;
    saved.reserve(items.size()); out.reserve(items.size());
    for (const auto& item : items) saved.push_back(store.capture(item.state));
    try {
      for (const auto& item : items) {
        const auto start=evaluated_rows;
        out.push_back(step(item.state,item.sampler));
        if (available.row_hints) {
          const auto effective=resolve_row_hints(item.row_hints, true, std::array<std::uint32_t,3>{2,4,8});
          const auto useful=evaluated_rows-start;
          const std::uint64_t physical_rows=effective.physical_row_cap.value_or(2);
          out.back().rows={physical_rows,useful,physical_rows-useful};
        }
      }
      if (available.row_hints && !out.empty()) {
        // Fixture's grouped shape aligns to four. Charge the one group tail
        // only to the first input, independently of accepted output counts.
        std::uint64_t total=0;
        for (const auto& r:out) total+=*r.rows.physical_rows;
        const auto tail=(4-total%4)%4;
        *out[0].rows.physical_rows+=tail; *out[0].rows.padding_rows+=tail;
      }
      return out;
    } catch (...) {
      auto error = std::current_exception();
      for (std::size_t i=0;i<items.size();++i) {
        try { rollback(items[i].state,saved[i]); } catch (...) {}
      }
      std::rethrow_exception(error);
    }
  }
  KvHandle fork_lane(const StateSnapshot& s) override {
    if (!available.fork_lanes || !available.shared_prefix_cow ||
        s.consumed > available.max_context || (s.pending && s.consumed == available.max_context))
      throw std::invalid_argument("unsupported fork/context");
    return store.fork(s);
  }
  void cancel_lane(KvHandle& h) override {
    if (!available.lane_cancellation) throw std::invalid_argument("unsupported cancel");
    active(h);
    for (std::uint32_t r=0;r<rank_count;++r) rank(h,r).cursors.terminal = true;
  }
  void resume_lane(KvHandle& h) override {
    if (!available.lane_resume) throw std::invalid_argument("unsupported resume");
    (void)store.consumed(h);
    for (std::uint32_t r=0;r<rank_count;++r)
      if (rank(h,r).cursors.busy) throw std::invalid_argument("in-flight resume");
    for (std::uint32_t r=0;r<rank_count;++r) rank(h,r).cursors.terminal = false;
  }
  void append_external(std::span<const std::uint32_t> tokens, KvHandle& h) override {
    if (!available.external_append) throw std::invalid_argument("unsupported append");
    prefill(tokens,h); // same evaluation/anchor path, zero target RNG draws
  }
  Capabilities caps() const override { return available; }
  strix::profile::ExecutionReceipt receipt() const override { return {}; }
};

struct Reference {
  std::vector<std::uint32_t> tokens;
  std::optional<std::uint32_t> pending;
  std::array<Cursors,2> state{};
  std::array<std::vector<std::uint64_t>,2> rows;
  void consume(std::uint32_t token) {
    for (std::uint32_t r=0;r<2;++r) {
      auto& c = state[r];
      rows[r].push_back(token+1000ULL*r);
      c.recurrent = 131*c.recurrent+token+r;
      c.draft = c.draft ^ (token+17*c.cursor);
      c.policy = c.policy+token%7;
      c.cursor = c.cursor+1;
    }
    tokens.push_back(token); pending.reset();
  }
  void append(std::span<const std::uint32_t> suffix) {
    if (suffix.empty()) return;
    if (pending) consume(*pending);
    for (std::size_t i=0;i+1<suffix.size();++i) consume(suffix[i]);
    pending = suffix.back();
    for (auto& c : state) c.supplied += suffix.size();
  }
  std::uint32_t decode(bool seeded) {
    if (pending) consume(*pending);
    const auto& c = state[0];
    std::uint64_t random = 0;
    if (seeded) random = (c.seed ^ (6364136223846793005ULL*c.rng)) >> 11;
    const auto next = static_cast<std::uint32_t>((c.recurrent+c.draft+c.policy+random)%251);
    pending = next;
    for (auto& x : state) { x.rng += seeded ? 1U : 0U; ++x.generated; }
    return next;
  }
};
void matches(HostEngine& engine, KvHandle h, const Reference& ref) {
  require(engine.store.tokens(h) == ref.tokens && engine.store.pending(h) == ref.pending,"logical history equality");
  for (std::uint32_t r=0;r<2;++r) {
    const auto& physical = engine.rank(h,r);
    require(physical.cursors == ref.state[r],"full recurrent/draft/RNG/policy/cursor digest");
    require(physical.pages.size() == (ref.rows[r].size()+3)/4,"physical page count");
    for (std::size_t p=0;p<physical.pages.size();++p)
      for (std::size_t j=0;j<4;++j) {
        const auto i=p*4+j;
        require(physical.pages.read(p).rows[j] == (i<ref.rows[r].size() ? ref.rows[r][i] : 0),"full physical page digest");
      }
  }
}
const Image& image(const StateSnapshot& s, std::uint32_t r) {
  return *static_cast<const Image*>(s.ranks.at(r).owner.get());
}
void snapshot_equal(const StateSnapshot& a, const StateSnapshot& b) {
  require(a.tokens == b.tokens && a.pending == b.pending && a.consumed == b.consumed,"snapshot metadata equality");
  for (std::uint32_t r=0;r<a.ranks.size();++r) {
    const auto& x=image(a,r); const auto& y=image(b,r);
    require(x.cursors == y.cursors && x.pages.size() == y.pages.size(),"snapshot cursor digest");
    for (std::size_t p=0;p<x.pages.size();++p) require(x.pages.read(p).rows == y.pages.read(p).rows,"snapshot byte digest");
  }
}
void sharing_and_continuation() {
  HostEngine e;
  auto source=e.store.acquire();
  const std::array<std::uint32_t,8> prompt{1,2,3,4,5,6,7,8};
  e.prefill(prompt,source);
  Reference root; root.append(prompt); matches(e,source,root);
  auto snapshot=e.store.capture(source);
  require(snapshot.sealed() && snapshot.pending == 8U,"sealed pending-anchor checkpoint");
  const auto clone_start=e.clones;
  auto a=e.fork_lane(snapshot), b=e.fork_lane(snapshot);
  require(e.clones == clone_start && e.cloned_bytes == 0,"zero prefix bytes cloned at fork");
  for (std::uint32_t r=0;r<2;++r) for (std::size_t p=0;p<2;++p) {
    const auto owner=image(snapshot,r).pages.owner(p);
    require(e.rank(source,r).pages.owner(p)==owner && e.rank(a,r).pages.owner(p)==owner &&
        e.rank(b,r).pages.owner(p)==owner,"real prefix owner sharing");
  }
  // Direct page-writer qualification avoids transaction checkpoints that would
  // legitimately add an owner again between independent decode calls.
  for (std::uint32_t r=0;r<2;++r) {
    auto& pages=e.rank(a,r).pages;
    pages.write(1,[](Page& p) { p.rows[3]=88; });
    const auto detached=pages.owner(1); const auto count=e.clones;
    pages.write(1,[](Page& p) { p.rows[3]=89; });
    require(e.clones==count && pages.owner(1)==detached,"subsequent lane-local writes do not reclone");
    require(pages.owner(0)==image(snapshot,r).pages.owner(0),"untouched page remains shared");
    require(pages.owner(1)!=image(snapshot,r).pages.owner(1),"partial tail write detaches touched page");
    require(image(snapshot,r).pages.read(1).rows[3]==0 && e.rank(source,r).pages.read(1).rows[3]==0 &&
        e.rank(b,r).pages.read(1).rows[3]==0,"source snapshot sibling bytes immutable");
  }
  require(e.clones-clone_start==2 && e.cloned_bytes==2*sizeof(Page::rows),"exactly one touched page cloned per rank");
  e.store.restore(a,snapshot);
  Reference ra=root, rb=root;
  for (std::uint32_t r=0;r<2;++r) {
    auto& ca=e.rank(a,r).cursors; ca.recurrent+=13; ca.draft=17+r; ca.policy=41+r; ca.rng=3; ca.seed=987;
    auto& cb=e.rank(b,r).cursors; cb.recurrent+=29; cb.draft=81+r; cb.policy=12+r; cb.rng=9; cb.seed=765;
    ra.state[r]=ca; rb.state[r]=cb;
  }
  SamplerState sa,sb; sa.max_tokens=sb.max_tokens=8;
  sa.parameters.seeded=sb.parameters.seeded=true; sa.parameters.seed=987; sb.parameters.seed=765;
  sa.parameters.temperature=sb.parameters.temperature=1;
  for (int round=0;round<3;++round) {
    sa.rng_draws=ra.state[0].rng; sb.rng_draws=rb.state[0].rng;
    const std::array<DecodeItem,2> items{{{b,sb},{a,sa}}};
    const auto clones_before=e.clones;
    const auto out=e.decode_batch(items);
    if (round==1) require(e.clones==clones_before,"appending fresh pages clones zero existing bytes");
    require(out.size()==2 && out[0].tokens==std::vector<std::uint32_t>{rb.decode(true)} &&
        out[1].tokens==std::vector<std::uint32_t>{ra.decode(true)},"ordered seeded batch equals independent reference");
    matches(e,a,ra); matches(e,b,rb);
  }
  require(e.rank(a,0).pages.size()==3 && e.rank(a,0).pages.owner(0)==image(snapshot,0).pages.owner(0),
      "fresh appended page preserves untouched prefix owner");
  require(e.rank(a,0).pages.owner(2)!=e.rank(b,0).pages.owner(2),"new suffix pages are lane-owned");
  matches(e,source,root); snapshot_equal(snapshot,e.store.capture(source));
  std::stop_source stop; stop.request_stop(); sa.cancellation=stop.get_token();
  sa.rng_draws=ra.state[0].rng; sb.rng_draws=rb.state[0].rng;
  const std::array<DecodeItem,2> cancel_items{{{b,sb},{a,sa}}};
  auto out=e.decode_batch(cancel_items);
  require(out[0].tokens==std::vector<std::uint32_t>{rb.decode(true)} &&
      out[1].finish==FinishReason::cancelled && out[1].tokens.empty(),"one lane cancellation preserves sibling commit");
  for (auto& c : ra.state) c.terminal=true;
  matches(e,a,ra); matches(e,b,rb);
  const std::array<std::uint32_t,2> external{91,92};
  reject([&] { e.append_external(external,a); });
  reject([&] { (void)e.decode(a,sa); });
  e.resume_lane(a); for (auto& c : ra.state) c.terminal=false;
  const auto rng=ra.state[0].rng;
  e.append_external(external,a); ra.append(external); matches(e,a,ra);
  require(e.rank(a,0).cursors.rng==rng,"external inputs draw no RNG and are not generated outputs");
  sa.cancellation={}; sa.rng_draws=rng;
  require(e.decode(a,sa).tokens==std::vector<std::uint32_t>{ra.decode(true)},"continued external append output");
  matches(e,a,ra); matches(e,b,rb); matches(e,source,root);

  e.store.release(source); e.store.release(a); e.store.release(b);
  auto again=e.fork_lane(snapshot); matches(e,again,root);
  auto stale=again; e.store.release(again); again=e.fork_lane(snapshot);
  require(again.slot==stale.slot && again.generation!=stale.generation,"cleanup reuses slot with new generation");
  reject([&] { (void)e.decode(stale,{}); });
  // Snapshot is now the only original prefix owner other than this new lane.
  // Write barrier must still detach, even after every old mutable lane died.
  const auto old_owner=image(snapshot,0).pages.owner(1);
  e.append_external(external,again);
  require(e.rank(again,0).pages.owner(1)!=old_owner && image(snapshot,0).pages.read(1).rows[3]==0,
      "last surviving snapshot remains immutable");
}

void validation_and_failures() {
  HostEngine e(5);
  auto source=e.store.acquire(); const std::array<std::uint32_t,4> prompt{1,2,3,4}; e.prefill(prompt,source);
  auto snapshot=e.store.capture(source); auto lane=e.fork_lane(snapshot);
  const auto before=e.store.capture(lane); const auto calls=e.initializes; const auto restores=e.restores;
  StateSnapshot forged; forged.tokens=snapshot.tokens; forged.consumed=snapshot.consumed;
  forged.pending=snapshot.pending; forged.bytes=snapshot.bytes;
  forged.ranks={{std::make_shared<int>(7),snapshot.ranks[0].bytes},{std::make_shared<int>(8),snapshot.ranks[1].bytes}};
  reject([&] { e.store.restore(lane,forged); }); reject([&] { (void)e.fork_lane(forged); });
  for (int which=0;which<6;++which) {
    auto bad=snapshot;
    if (which==0) bad.tokens[0]=99;
    if (which==1) bad.ranks.pop_back();
    if (which==2) ++bad.bytes;
    if (which==3) std::swap(bad.ranks[0],bad.ranks[1]);
    if (which==4) bad.ranks[0].owner=std::make_shared<int>(0);
    if (which==5) bad.pending=99;
    reject([&] { e.store.restore(lane,bad); }); reject([&] { (void)e.fork_lane(bad); });
  }
  HostEngine foreign;
  reject([&] { (void)foreign.fork_lane(snapshot); });
  auto foreign_handle=foreign.store.acquire();
  reject([&] { (void)e.decode(foreign_handle,{}); });
  foreign.store.release(foreign_handle);
  require(e.initializes==calls && e.restores==restores && foreign.initializes==1,"invalid erased metadata rejected before hooks");
  snapshot_equal(before,e.store.capture(lane));
  auto domain=std::make_shared<const int>(1), schema=std::make_shared<const int>(2);
  HostEngine compatible_a(3,2,true,domain,schema), compatible_b(3,2,true,domain,schema);
  auto ca=compatible_a.store.acquire(); compatible_a.prefill(prompt,ca); auto cross=compatible_a.store.capture(ca);
  auto cb=compatible_b.fork_lane(cross); snapshot_equal(cross,compatible_b.store.capture(cb));
  HostEngine wrong_schema(3,2,true,domain,std::make_shared<const int>(2));
  HostEngine wrong_domain(3,2,true,std::make_shared<const int>(1),schema);
  HostEngine wrong_rank(3,1,true,domain,schema);
  reject([&] { (void)wrong_schema.fork_lane(cross); }); reject([&] { (void)wrong_domain.fork_lane(cross); });
  reject([&] { (void)wrong_rank.fork_lane(cross); });
  require(wrong_schema.initializes==0 && wrong_domain.initializes==0 && wrong_rank.initializes==0,"schema/domain/rank mismatch precedes allocation");
  HostEngine unsupported(2,2,false); require(!unsupported.store.supports_fork(),"COW fails closed");
  reject([&] { (void)unsupported.store.fork(snapshot); });
  auto incomplete=e.hooks(true,{},{}); incomplete.fork_rank={};
  KvStore missing_fork_hook(1,2,std::move(incomplete));
  require(!missing_fork_hook.supports_fork(),"capability alone cannot advertise sharing");
  reject([&] { (void)missing_fork_hook.fork(snapshot); });
  e.available.shared_prefix_cow=false; reject([&] { (void)e.fork_lane(snapshot); }); e.available.shared_prefix_cow=true;
  e.available.fork_lanes=false; reject([&] { (void)e.fork_lane(snapshot); }); e.available.fork_lanes=true;
  e.available.external_append=false; reject([&] { e.append_external(prompt,lane); }); e.available.external_append=true;
  SamplerState s; s.max_tokens=8;
  std::array<DecodeItem,2> duplicate{{{lane,s},{lane,s}}};
  reject([&] { (void)e.decode_batch(duplicate); });
  std::array<DecodeItem,2> distinct{{{source,s},{lane,s}}};
  e.available.request_lane_batch=false; reject([&] { (void)e.decode_batch(distinct); }); e.available.request_lane_batch=true;
  e.available.batch_decode=false; reject([&] { (void)e.decode_batch(distinct); }); e.available.batch_decode=true;
  std::vector<std::uint32_t> excessive(25,1); reject([&] { e.append_external(excessive,lane); });
  const std::array<std::uint32_t,1> invalid{256}; reject([&] { e.append_external(invalid,lane); });
  e.rank(lane,1).cursors.busy=true;
  reject([&] { e.append_external(prompt,lane); }); reject([&] { e.cancel_lane(lane); });
  reject([&] { e.resume_lane(lane); }); e.rank(lane,1).cursors.busy=false;
  e.available.lane_cancellation=false; reject([&] { e.cancel_lane(lane); }); e.available.lane_cancellation=true;
  e.available.lane_resume=false; reject([&] { e.resume_lane(lane); }); e.available.lane_resume=true;
  e.resume_lane(lane); // explicit new-turn boundary after rejected cancellation
  s.rng_draws=1; reject([&] { (void)e.decode(lane,s); }); s.rng_draws=0;
  snapshot_equal(before,e.store.capture(lane));

  e.fail_initialize=true; reject([&] { (void)e.fork_lane(snapshot); }); e.fail_initialize=false;
  e.fail_capture=true; reject([&] { e.append_external(prompt,lane); }); e.fail_capture=false;
  const auto shared_owner=e.rank(lane,0).pages.owner(0); const auto clone_count=e.clones;
  e.fail_clone=true; reject([&] { e.append_external(prompt,lane); }); e.fail_clone=false;
  require(e.rank(lane,0).pages.owner(0)==shared_owner && e.clones==clone_count,"clone failure preserves shared owner");
  e.fail_evaluate=true; reject([&] { e.append_external(prompt,lane); });
  reject([&] { (void)e.decode_batch(distinct); }); e.fail_evaluate=false;
  snapshot_equal(before,e.store.capture(lane)); snapshot_equal(snapshot,e.store.capture(source));
  // Fail after an earlier lane committed to verify batch-wide unreported work
  // is restored, rather than merely testing a failure in the first item.
  auto busy_saved=e.store.capture(lane);
  e.rank(lane,1).cursors.cursor=24; // force out-of-range page access in second lane
  reject([&] { (void)e.decode_batch(distinct); });
  snapshot_equal(snapshot,e.store.capture(source)); e.store.restore(lane,busy_saved);

  e.fail_fork=true; reject([&] { (void)e.fork_lane(snapshot); }); e.fail_fork=false;
  snapshot_equal(before,e.store.capture(lane)); snapshot_equal(snapshot,e.store.capture(source));
  auto reused=e.fork_lane(snapshot); e.store.release(reused);
  auto destination=e.store.acquire(); e.prefill(std::array<std::uint32_t,2>{9,10},destination);
  auto saved=e.store.capture(destination); const auto rollback_start=e.restores;
  e.fail_restore=true; reject([&] { e.store.restore(destination,snapshot); }); e.rolling_back=false;
  require(e.restores-rollback_start==4,"restore and rollback include both attempted ranks");
  snapshot_equal(saved,e.store.capture(destination));
  e.fail_restore=true; e.fail_rollback=true;
  reject([&] { e.store.restore(destination,snapshot); }); e.rolling_back=false; e.fail_rollback=false;
  reject([&] { (void)e.store.capture(destination); });
  snapshot_equal(before,e.store.capture(lane)); // only affected lane poisoned
  e.store.release(destination); auto replacement=e.store.acquire();
  require(replacement.slot==destination.slot && replacement.generation!=destination.generation,"poisoned cleanup generation reuse");
  e.store.release(replacement);
  e.fail_fork=true; e.fail_release=true;
  reject([&] { (void)e.fork_lane(snapshot); }); e.fail_fork=false; e.fail_release=false;
  auto remaining1=e.store.acquire(), remaining2=e.store.acquire();
  const auto alloc_count=e.initializes;
  reject([&] { (void)e.fork_lane(snapshot); });
  require(e.initializes==alloc_count,"exhaustion has no physical allocation side effects");
  snapshot_equal(snapshot,e.store.capture(source)); snapshot_equal(before,e.store.capture(lane));
  e.store.release(remaining1); e.store.release(remaining2);

  HostEngine full(2);
  auto full_lane=full.store.acquire();
  std::vector<std::uint32_t> context(24,7); full.prefill(context,full_lane);
  const auto full_before=full.store.capture(full_lane);
  reject([&] { full.append_external(prompt,full_lane); });
  auto context_result=full.decode(full_lane,s);
  require(context_result.finish==FinishReason::context && context_result.tokens.empty() &&
      context_result.before==context_result.after && context_result.rng_draws==0,"actual store context exhaustion preserves frontier");
  full.resume_lane(full_lane);
  snapshot_equal(full_before,full.store.capture(full_lane));
  full.available.max_context=23;
  const auto full_allocations=full.initializes;
  reject([&] { (void)full.fork_lane(full_before); });
  require(full.initializes==full_allocations,"snapshot context capacity checked before slot allocation");
}

void cache_and_lifetime() {
  const auto live_start=Page::live;
  StateSnapshot held;
  {
    HostEngine e(2);
    auto source=e.store.acquire(); e.prefill(std::array<std::uint32_t,5>{1,2,3,4,5},source);
    e.store.truncate(source,e.store.consumed(source)); // discard pending, safe consumed checkpoint
    auto snapshot=std::make_shared<PrefixSnapshot>();
    snapshot->identity={"host-model","host-profile","host-pages-v1",2};
    snapshot->state=e.store.capture(source);
    snapshot->attention_frontier=snapshot->recurrent_frontier=snapshot->state.consumed;
    Policy policy; policy.max_entries=1;
    PrefixCache cache(policy); require(cache.admit(snapshot),"sealed cache admission");
    const auto identity=snapshot->identity;
    std::array<std::uint32_t,4> prompt{1,2,3,4};
    auto hit=cache.lookup(prompt,identity); require(hit && hit->state.sealed(),"cache retains provenance");
    snapshot->state.ranks[0].owner=std::make_shared<int>(1); // caller alias cannot mutate frozen entry
    require(hit->state.sealed(),"frozen seal metadata survives caller alias");
    held=hit->state;
    auto sibling=e.fork_lane(held); e.store.release(source);
    e.store.set_pending(sibling,9);
    e.prefill(std::array<std::uint32_t,1>{10},sibling);
    e.store.truncate(sibling,e.store.consumed(sibling));
    auto next=std::make_shared<PrefixSnapshot>(); next->identity=identity; next->state=e.store.capture(sibling);
    next->attention_frontier=next->recurrent_frontier=next->state.consumed;
    require(cache.admit(next) && cache.size()==1,"cache eviction");
    e.store.release(sibling);
    auto restored=e.fork_lane(held);
    snapshot_equal(held,e.store.capture(restored));
  }
  require(Page::live>live_start && held.sealed(),"snapshot pages survive store destruction and cache eviction");
  held=StateSnapshot{};
  require(Page::live==live_start,"shared pages reclaimed after last snapshot owner");
}

void boundaries_and_rows() {
  HostEngine e;
  auto source=e.store.acquire(); e.prefill(std::array<std::uint32_t,2>{1,2},source);
  // Materialize the final input with the existing physical path; no selection
  // and no metadata-only checkpoint. This is the fully consumed PLAN boundary.
  const auto last=*e.store.pending(source);
  e.store.reserve_consumed(source,1); e.evaluate(source,last);
  e.store.append_consumed(source,std::span(&last,1));
  const auto plan=e.store.capture(source);
  require(plan.consumed==2 && !plan.pending && image(plan,0).cursors.generated==0 &&
          image(plan,0).cursors.rng==0,"fully consumed boundary selects no token");
  auto lane=e.fork_lane(plan); snapshot_equal(plan,e.store.capture(lane));
  SamplerState s;
  SamplerState random; random.parameters.seeded=true; random.parameters.seed=123; random.parameters.temperature=1;
  for (std::uint32_t r=0;r<2;++r) e.rank(lane,r).cursors.seed=123;
  const auto output=e.decode(lane,random); random.rng_draws=output.rng_draws;
  const auto published=e.store.capture(lane);
  require(published.pending==output.tokens.back(),"published output remains pending");
  auto resumed=e.fork_lane(published); snapshot_equal(published,e.store.capture(resumed));
  require(e.rank(resumed,0).cursors.generated==1 && e.rank(resumed,0).cursors.rng==1 &&
          e.rank(resumed,0).cursors.seed==123,"resume fork preserves selection cursors");
  // Prompt-only reuse is a distinct explicit continuation, not resume-fork.
  // Use the fully consumed checkpoint; never drop a published pending output.
  auto prompt_source=e.fork_lane(published);
  const auto anchor=*e.store.pending(prompt_source);
  e.store.reserve_consumed(prompt_source,1); e.evaluate(prompt_source,anchor);
  e.store.append_consumed(prompt_source,std::span(&anchor,1));
  const auto prompt_full=e.store.capture(prompt_source);
  const auto prompt=e.store.capture_prompt(prompt_source); auto fresh=e.store.acquire();
  e.continuation_seed=456;
  reject([&]{(void)e.fork_lane(prompt);});
  require(e.store.restore_prompt(fresh,prompt) && e.rank(fresh,0).cursors.seed==456 &&
          e.rank(fresh,0).cursors.rng==0 && e.rank(fresh,1).cursors.rng==0 &&
          e.rank(fresh,0).cursors.generated==0 && !e.store.pending(fresh),"prompt reuse initializes explicit new policy");
  require(e.rank(prompt_source,0).cursors.rng==1 &&
          e.rank(fresh,0).cursors.recurrent==e.rank(prompt_source,0).cursors.recurrent,
          "prompt continuation resets selection while retaining complete consumed state");
  snapshot_equal(prompt_full,e.store.capture(prompt_source));
  snapshot_equal(plan,e.store.capture(source)); snapshot_equal(published,e.store.capture(lane));
  const auto unchanged=e.store.capture(source); const auto work=e.evaluated_rows;
  std::array<DecodeItem,2> items{{{source,s},{resumed,random,RowHints{2,{}}}}};
  reject([&]{(void)e.decode_batch(items);});
  snapshot_equal(unchanged,e.store.capture(source)); snapshot_equal(published,e.store.capture(resumed));
  require(e.evaluated_rows==work,"unsupported hint in later item rejects whole group before mutation");
  e.available.row_hints=true;
  items[1].row_hints=RowHints{0,{}};
  reject([&]{(void)e.decode_batch(items);});
  require(e.evaluated_rows==work,"zero cap refuses without work");
  items[0].row_hints=RowHints{3,0.25}; items[1].row_hints=RowHints{7,0.5};
  const auto results=e.decode_batch(items);
  require(results[0].rows.physical_rows==4ULL && results[0].rows.useful_rows==0ULL &&
          results[0].rows.padding_rows==4ULL && results[0].tokens.size()==1,
          "retained logits yield output without physical useful work; group tail charged once");
  require(results[1].rows.physical_rows==4ULL && results[1].rows.useful_rows==1ULL &&
          results[1].rows.padding_rows==3ULL && results[1].tokens.size()==1,
          "pending input evaluation differs from accepted output and padding");
  require(e.evaluated_rows-work==1,"useful accounting comes from evaluated rows");
}
} // namespace
int main() {
  try {
    sharing_and_continuation(); validation_and_failures(); cache_and_lifetime(); boundaries_and_rows();
    require(Page::live==0,"all physical pages released");
    std::cout << "session fanout host contracts passed\n";
  } catch (const std::exception& e) {
    std::cerr << "session fanout failure: " << e.what() << '\n'; return 1;
  }
}
