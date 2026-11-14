#include "strix/session/kv.hpp"

#include <array>
#include <map>
#include <stdexcept>
#include <type_traits>

using namespace strix::session;
static_assert(!std::is_copy_constructible_v<KvStore> && !std::is_copy_assignable_v<KvStore>);
static_assert(!std::is_move_constructible_v<KvStore> && !std::is_move_assignable_v<KvStore>);
static_assert(std::is_aggregate_v<StateSnapshot> && std::is_aggregate_v<Hooks>);

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void reject(F&& call) {
  bool threw = false;
  try { call(); } catch (...) { threw = true; }
  require(threw, "expected rejection");
}
struct Payload { int attention = 0; int recurrent = 0; int opaque_cursor = 0; };
}

int main() {
  std::map<std::pair<std::uint32_t, std::uint32_t>, Payload> physical;
  bool fail_restore = false;
  bool fail_rollback = false;
  bool fail_truncate = false;
  Hooks hooks;
  hooks.partial_restore_supported = true;
  hooks.initialize = [&](KvHandle h) { for (std::uint32_t r=0;r<2;++r) physical[{h.slot,r}] = {}; };
  hooks.reset = [&](KvHandle h) { for (std::uint32_t r=0;r<2;++r) physical[{h.slot,r}] = {}; };
  hooks.release = [&](KvHandle h) { for (std::uint32_t r=0;r<2;++r) physical.erase({h.slot,r}); };
  hooks.view = [&](KvHandle h, std::uint32_t r) { return RankState{std::make_shared<Payload>(physical.at({h.slot,r})), sizeof(Payload)}; };
  hooks.capture = hooks.view;
  hooks.restore = [&](KvHandle h, std::uint32_t r, const RankState& value) {
    auto p = *static_cast<const Payload*>(value.owner.get());
    if (fail_restore && r == 1 && p.attention == 20) { physical[{h.slot,r}] = p; throw std::runtime_error("restore changed then failed"); }
    if (fail_rollback && p.attention == 1) throw std::runtime_error("rollback failed");
    physical[{h.slot,r}] = p;
  };
  hooks.truncate = [&](KvHandle h, std::uint32_t r, std::uint64_t n) {
    physical[{h.slot,r}].attention = static_cast<int>(n);
    physical[{h.slot,r}].recurrent = static_cast<int>(n);
    if (fail_truncate && r == 1) throw std::runtime_error("truncate changed then failed");
  };

  KvStore store(2, 2, hooks);
  KvStore foreign_store(1, 2, hooks);
  auto a = store.acquire();
  auto b = store.acquire();
  require(a.slot != b.slot, "two live slots");
  physical[{a.slot,0}] = {11,12,13};
  physical[{a.slot,1}] = {21,22,23};
  physical[{b.slot,0}] = {31,32,33};
  physical[{b.slot,1}] = {41,42,43};
  std::array<std::uint32_t,2> consumed{4,5};
  store.append_consumed(a, consumed);
  store.set_pending(a, 9);
  auto snapshot = store.capture(a);
  require(snapshot.pending == std::uint32_t{9} && snapshot.tokens == std::vector<std::uint32_t>({4,5}), "pending snapshot");
  physical[{a.slot,0}] = {1,2,3};
  physical[{a.slot,1}] = {4,5,6};
  store.restore(a, snapshot);
  require(physical[{a.slot,0}].attention == 11 && physical[{a.slot,1}].opaque_cursor == 23, "rank restore");
  require(physical[{b.slot,0}].attention == 31, "slot isolation");

  physical[{a.slot,0}] = {10,10,10};
  physical[{a.slot,1}] = {20,20,20};
  auto target = store.capture(a);
  store.restore(a, snapshot);
  fail_restore = true;
  reject([&] { store.restore(a,target); });
  require(physical[{a.slot,0}].attention == 11 && physical[{a.slot,1}].attention == 21, "throwing rank rollback");
  require(store.consumed(a) == 2 && store.pending(a) == std::uint32_t{9}, "metadata unchanged on failure");
  fail_restore = false;

  fail_truncate = true;
  reject([&] { store.truncate(a,1); });
  require(physical[{a.slot,0}].attention == 11 && physical[{a.slot,1}].attention == 21, "truncate rollback");
  fail_truncate = false;
  store.truncate(a,2);
  require(store.consumed(a) == 2 && !store.pending(a), "equal consumed truncation removes anchor");
  require(physical[{a.slot,0}].attention == 11, "equal consumed truncation preserves committed physical state");
  store.set_pending(a,9);
  reject([&] { store.append_consumed(a, consumed); });
  reject([&] { store.reserve_consumed(a, UINT64_MAX); });
  require(store.consumed(a) == 2 && store.pending(a) == 9U, "failed metadata append preserves frontier");
  store.truncate(a,1);
  require(store.tokens(a) == std::vector<std::uint32_t>({4}) && !store.pending(a), "truncate metadata");

  auto malformed = snapshot;
  malformed.bytes++;
  reject([&] { store.restore(a,malformed); });
  malformed = snapshot;
  malformed.ranks[0].owner.reset();
  reject([&] { store.restore(a,malformed); });
  malformed = snapshot;
  malformed.ranks[0].bytes = UINT64_MAX;
  malformed.ranks[1].bytes = 2;
  reject([&] { store.restore(a,malformed); });
  reject([&] { (void)store.view(a,2); });
  reject([&] { (void)store.view(KvHandle{},0); });
  reject([&] { store.release(foreign_store.acquire()); });
  physical[{a.slot,0}] = {1,1,1};
  physical[{a.slot,1}] = {1,1,1};

  fail_restore = true;
  fail_rollback = true;
  reject([&] { store.restore(a,target); });
  fail_restore = false;
  fail_rollback = false;
  reject([&] { (void)store.capture(a); });
  store.release(a); // poisoned handles may be cleaned
  auto reused = store.acquire();
  require(reused.slot == a.slot && reused.generation != a.generation, "safe generation reuse");
  reject([&] { (void)store.consumed(a); });
  store.invalidate(reused);
  reject([&] { store.append_consumed(reused, consumed); });
  store.release(reused);
  store.reset(b);
  require(store.consumed(b) == 0, "reset metadata");
  store.release(b);
  reject([&] { (void)store.capture(b); });
  // Prompt snapshots have distinct sealed provenance. They deliberately omit
  // the former job's pending/RNG cursor and require a new sampler on restore.
  {
    int cursor = 19;
    Hooks prompt_hooks;
    prompt_hooks.initialize = prompt_hooks.reset = prompt_hooks.release = [](KvHandle) {};
    prompt_hooks.view = prompt_hooks.capture = [&](KvHandle,std::uint32_t) {
      return RankState{std::make_shared<int>(cursor),sizeof(int)};
    };
    prompt_hooks.restore = [&](KvHandle,std::uint32_t,const RankState& rank) {cursor=*static_cast<const int*>(rank.owner.get());};
    prompt_hooks.truncate = [](KvHandle,std::uint32_t,std::uint64_t) {};
    prompt_hooks.capture_prompt = [](KvHandle,std::uint32_t) {return RankState{std::make_shared<int>(0),sizeof(int)};};
    unsigned restarts=0; bool decline=false, failure=false;
    prompt_hooks.restore_prompt = [&](KvHandle,std::span<const RankState> ranks) {
      ++restarts;require(ranks.size()==1,"prompt rank count");
      if(failure)throw std::runtime_error("prompt restore failure");
      if(decline)return false;
      cursor=3;return true; // the new job starts at its OWN cursor
    };
    KvStore prompts(2,1,prompt_hooks);auto source=prompts.acquire();
    prompts.append_consumed(source,consumed);prompts.set_pending(source,9);
    const auto full=prompts.capture(source);const auto prompt=prompts.capture_prompt(source);
    require(full.sealed() && prompt.sealed() && full.pending==9U && !prompt.pending &&
        prompt.purpose==SnapshotPurpose::prompt_reuse,"distinct sealed prompt purpose");
    auto forged=full;forged.pending.reset();reject([&]{prompts.restore_prompt(source,forged);});
    forged=prompt;forged.purpose=SnapshotPurpose::resume;require(!forged.sealed(),"purpose is sealed");
    auto destination=prompts.acquire();reject([&]{prompts.restore(destination,prompt);});
    decline=true;require(!prompts.restore_prompt(destination,prompt) && !prompts.consumed(destination),"decline preserves fresh lane");
    decline=false;require(prompts.restore_prompt(destination,prompt) && cursor==3 &&
        prompts.consumed(destination)==2 && !prompts.pending(destination),"prompt reuse restarts sampler");
    reject([&]{prompts.restore_prompt(destination,prompt);});
    prompts.reset(destination);failure=true;reject([&]{prompts.restore_prompt(destination,prompt);});
    reject([&]{(void)prompts.capture(destination);});require(restarts==3,"unproven snapshots never reach prompt hooks");
  }
}
