#include "strix/session/prefix_cache.hpp"

#include <stdexcept>

using namespace strix::session;
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
std::shared_ptr<PrefixSnapshot> make(std::vector<std::uint32_t> tokens, int marker = 1) {
  auto snapshot = std::make_shared<PrefixSnapshot>();
  snapshot->identity = {"model", "profile", "state-v1", 2};
  snapshot->state.tokens = std::move(tokens);
  snapshot->state.consumed = snapshot->state.tokens.size();
  snapshot->state.ranks = {{std::make_shared<int>(marker),4},{std::make_shared<int>(marker+1),5}};
  snapshot->state.bytes = 9;
  snapshot->attention_frontier = snapshot->state.consumed;
  snapshot->recurrent_frontier = snapshot->state.consumed;
  return snapshot;
}
}
int main() {
  PrefixCache cache(Policy{true,true,false,4,1000,EvictionPolicy::lru});
  auto short_meta = make({1,2});
  auto long_meta = make({1,2,3,4});
  require(cache.admit(short_meta) && cache.admit(long_meta), "admission");
  short_meta->state.tokens[0] = 99;
  short_meta->identity.model = "mutated";
  std::uint32_t prompt[]{1,2,3,9};
  auto hit = cache.lookup(prompt, {"model","profile","state-v1",2});
  require(hit && hit->state.tokens.size() == 2, "longest safe matching candidate");
  require(hit->state.tokens[0] == 1, "metadata frozen at admission");
  require(!cache.lookup(prompt, {"model","profile","state-v2",2}), "format identity");
  require(!cache.lookup(prompt, {"model","profile","state-v1",1}), "topology identity");
  require(!cache.lookup(prompt, {"","profile","state-v1",2}), "unknown identity");
  auto held = hit;
  auto invalid = make({8});
  invalid->state.pending = 7;
  require(!cache.admit(invalid), "pending snapshot refusal");
  invalid = make({8});
  invalid->recurrent_frontier = 0;
  require(!cache.admit(invalid), "recurrent mismatch refusal");
  invalid = make({8});
  invalid->state.ranks[1].owner.reset();
  require(!cache.admit(invalid), "null rank refusal");
  require(held->state.tokens[0] == 1 && *static_cast<const int*>(held->state.ranks[0].owner.get()) == 1,
          "lookup ownership survives metadata mutation");

  PrefixCache lru(Policy{true,true,false,2,1000,EvictionPolicy::lru});
  require(lru.admit(make({1})) && lru.admit(make({2})), "lru setup");
  std::uint32_t one[]{1};
  require(bool(lru.lookup(one,{"model","profile","state-v1",2})), "touch LRU entry");
  require(lru.admit(make({3})), "LRU admission");
  std::uint32_t two[]{2};
  require(!lru.lookup(two,{"model","profile","state-v1",2}), "LRU eviction");
  require(bool(lru.lookup(one,{"model","profile","state-v1",2})), "LRU retained touched");

  PrefixCache fifo(Policy{true,true,false,2,1000,EvictionPolicy::fifo});
  require(fifo.admit(make({1})) && fifo.admit(make({2})), "fifo setup");
  require(bool(fifo.lookup(one,{"model","profile","state-v1",2})), "FIFO read");
  require(fifo.admit(make({3})), "FIFO admission");
  require(!fifo.lookup(one,{"model","profile","state-v1",2}) &&
          bool(fifo.lookup(two,{"model","profile","state-v1",2})), "FIFO oldest eviction");

  PrefixCache bounded(Policy{true,true,false,2,40,EvictionPolicy::lru});
  require(bounded.admit(make({5})), "bounded setup");
  std::uint32_t five[]{5};
  auto before = bounded.lookup(five,{"model","profile","state-v1",2});
  require(!bounded.admit(make(std::vector<std::uint32_t>(100,7))), "oversized refusal");
  require(bounded.size() == 1 && before && bounded.bytes() == 13, "refusal preserves existing entry");
  PrefixCache exact(Policy{true,true,true,3,1000,EvictionPolicy::lru});
  require(exact.admit(make({1,2})), "leave-one setup");
  std::uint32_t exact_prompt[]{1,2};
  require(!exact.lookup(exact_prompt,{"model","profile","state-v1",2}), "leave one token boundary");
  require(bool(exact.lookup(prompt,{"model","profile","state-v1",2})), "leave one token hit");
  PrefixCache off(Policy{false,true,false,3,1000,EvictionPolicy::lru});
  PrefixCache no_store(Policy{true,false,false,3,1000,EvictionPolicy::lru});
  require(!off.admit(make({1})) && !no_store.admit(make({1})), "disabled admission");
  require(!off.lookup(prompt,{"model","profile","state-v1",2}), "disabled lookup");

  require(!bounded.admit(make({5},9)) && bounded.bytes() == 13, "default duplicate refusal");
  Policy replacement{true,true,false,2,1000,EvictionPolicy::fifo};
  replacement.replace_duplicates = true;
  PrefixCache refresh(replacement);
  require(refresh.admit(make({1})) && refresh.admit(make({2})), "replacement setup");
  auto larger = make({1},40);
  larger->state.ranks[0].bytes = 20;
  larger->state.bytes = 25;
  require(refresh.admit(larger) && refresh.bytes() == 42 && refresh.size() == 2, "replacement byte accounting");
  larger->state.ranks[0].bytes = 99;
  require(refresh.admit(make({3})), "replacement refresh order");
  require(!refresh.lookup(two,{"model","profile","state-v1",2}), "FIFO replacement refreshes insertion");
  auto refreshed = refresh.lookup(one,{"model","profile","state-v1",2});
  require(refreshed && refreshed->state.bytes == 25 &&
      *static_cast<const int*>(refreshed->state.ranks[0].owner.get()) == 40, "replacement metadata frozen");
  replacement.eviction = EvictionPolicy::lru;
  PrefixCache refresh_lru(replacement);
  require(refresh_lru.admit(make({1})) && refresh_lru.admit(make({2})) &&
      refresh_lru.admit(make({1},7)) && refresh_lru.admit(make({3})), "LRU replacement setup");
  require(!refresh_lru.lookup(two,{"model","profile","state-v1",2}) &&
      bool(refresh_lru.lookup(one,{"model","profile","state-v1",2})), "replacement refreshes LRU");

  Policy oversize{true,true,false,2,10,EvictionPolicy::lru};
  oversize.retain_one_oversize = true;
  PrefixCache sole(oversize);
  require(sole.admit(make({1})) && sole.bytes() == 13 && sole.size() == 1, "sole oversize opt-in");
  require(sole.admit(make({2,3})) && sole.bytes() == 17 && sole.size() == 1, "oversize replaces previous retention");
  auto small = make({4});
  small->state.ranks[0].bytes = small->state.ranks[1].bytes = 1;
  small->state.bytes = 2;
  require(sole.admit(small) && sole.bytes() == 6 && sole.size() == 1, "bounded admission evicts old oversize");
  small->state.pending = 5;
  require(!sole.admit(small), "retention does not weaken safe checkpoints");
  Policy physical_only;physical_only.max_bytes=0;physical_only.retain_one_oversize=true;
  physical_only.replace_duplicates=true;physical_only.charge_token_bytes=false;
  PrefixCache resident(physical_only);
  require(resident.admit(make({1,2,3})) && resident.bytes()==9,"physical-only byte contract and zero-cap sole retention");
  require(resident.admit(make({1,2,3},8)) && resident.size()==1 && resident.bytes()==9,"physical duplicate replacement");
  NamedStateCache named(10,2);
  require(named.put("a",{1,2,3}) && named.put("b",{4,5,6}),"named retention setup");
  named.touch("a");
  require(named.put("c",{7,8,9}) && named.find("a") && !named.find("b") && named.evictions()==1,
          "named retention uses deterministic LRU");
  require(!named.put("a",std::vector<std::uint8_t>(11)) && named.find("a") && named.bytes()==6,
          "oversize refusal preserves the previous named checkpoint");
  require(named.evict_until([](auto n){return n<=3;},"a") && named.find("a") && !named.find("c"),
          "restore peak eviction protects the target checkpoint");
  require(!named.evict_until([](auto){return false;},"a") && named.find("a"),
          "an impossible protected peak fails closed");
  require(named.put("a",{8}) && named.bytes()==1 && named.entries().size()==1,
          "named replacement updates charged bytes");
  NamedStateCache disabled(10,0);
  require(!disabled.put("a",{1}) && disabled.entries().empty(),"zero named entries retains the off policy");
}
