#include "minimax/engine/deployment.hpp"
#include "gemma4/engine/deployment.hpp"
#include "strix/serve/json.hpp"
#include <cstring>
#include <fstream>
#include <iostream>

namespace {
using namespace strix::deploy;
constexpr std::uint64_t GiB = 1ULL<<30;
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
template<class F> void rejects(F f) {
    bool threw = false; try { f(); } catch (const std::exception&) { threw = true; }
    require(threw,"expected refusal");
}
DeviceFacts device(std::uint64_t gib, bool unified = true) {
    DeviceFacts d; d.gfx_target = unified ? "gfx1151" : "gfx1201"; d.cu_count = 40;
    d.mem_total_bytes = gib*GiB; d.mem_free_bytes = gib*GiB; d.unified = unified;
    d.source = "host-test"; return d;
}
struct PairTransport {
    std::uint64_t peer_gib = 128;
    bool mismatch = false;
    void exchange(std::span<const std::byte> in, std::span<std::byte> out) {
        require(in.size() == out.size(),"wire sizes"); std::memcpy(out.data(),in.data(),in.size());
        if (in.size() == 16) { if (mismatch) out[0] ^= std::byte{1}; return; }
        std::uint64_t rank; std::memcpy(&rank,in.data()+8,8); rank = 1-rank;
        std::memcpy(out.data()+8,&rank,8);
        const auto memory = peer_gib*GiB;
        std::memcpy(out.data()+32,&memory,8); std::memcpy(out.data()+40,&memory,8);
    }
};
void minimax_tests() {
    namespace mm = strix::minimax::engine;
    mm::DeploymentModel m; m.weights = 80*GiB;
    mm::FootprintProvider provider(m);
    Spec s; s.nodes = 2; s.link = "usb4-auto"; s.context = 8192; s.slots = 8;
    auto p = resolve(s,{device(128),device(128)},provider);
    require(!p.refused && p.chosen.format.prefill == "exl3","MiniMax plain EXL3 default");
    require(p.ranks[0].kv_bytes == 60ULL*640*2*8192*8,"MiniMax native FP16 K/V/index byte count");
    require(p.ranks[0].footprint.staging == (768ULL<<20),"MiniMax staging reservation");
    // Without the FAST_I4 runtime hook only executable formats are advertised.
    for (const auto& f : provider.formats("tp2")) require(f.prefill == "exl3","no unexecutable INT4 candidate");
    auto pinned_i4 = s; pinned_i4.format.prefill = "int4-stored";
    require(resolve(pinned_i4,{device(256),device(256)},provider).refused,"INT4 pin refuses without runtime");
    // INT4 hook: the same provider with the runtime declared present.
    mm::FootprintProvider provider_i4(m,4096,"minimax-m3.1-flash",0,true);
    auto large = resolve(s,{device(256),device(256)},provider_i4);
    require(!large.refused && large.chosen.format.resident_layers == 57 && !large.chosen.format.jit_slots,"all INT4 resident");
    auto small = resolve(s,{device(128),device(128)},provider_i4);
    require(!small.refused && small.chosen.format.jit_slots == 2 && !small.chosen.format.resident_layers,"small pair two-slot JIT");
    auto mixed = resolve(s,{device(256),device(128)},provider_i4);
    require(!mixed.refused && mixed.chosen.format == small.chosen.format,"mixed pair uses smaller budget");
    auto reverse = resolve(s,{device(128),device(256)},provider_i4);
    require(reverse.chosen.format == mixed.chosen.format,"rank ordering cannot change residency");
    auto alias = s; alias.format.prefill = "int4-jit";
    require(resolve(alias,{device(128),device(128)},provider_i4).chosen.format.jit_slots == 2,"design JIT format attribution");
    require(resolve(s,{device(32,false),device(32,false)},provider_i4).refused,"discrete memory refuses oversized target");
    s.format.resident_layers = 57; s.format.total_layers = 57; s.pins["format.resident_layers"] = "57";
    require(resolve(s,{device(256),device(128)},provider_i4).refused,"resident pin never falls back");
    s.pins.clear(); s.format = {}; s.context = 131072;
    require(resolve(s,{device(128),device(128)},provider).refused,"KV-first pinned long context refusal");
    s.context = 8192; s.topology = "single";
    require(resolve(s,{device(256)},provider).refused,"MiniMax requires pair");
    s.topology = "tp2"; s.nodes = 2;
    require(resolve(s,{device(256)},provider).refused,"never invent remote facts");
    PairTransport t;
    const Exchange wire = [&](auto out, auto in) { t.exchange(out,in); };
    auto facts = exchange_pair_facts(0,device(256),wire);
    require(facts[0].mem_total_bytes == 256*GiB && facts[1].mem_total_bytes == 128*GiB,"fresh asymmetric facts exchange");
    auto live = resolve(s,facts,provider); agree_plan(live,wire);
    t.mismatch = true; rejects([&] { agree_plan(live,wire); });
    t.mismatch = false;
    require(resolve_pair(s,provider,0,device(256),wire).hash() == live.hash(),"resolve_pair agrees");
    rejects([] { checked_mul(UINT64_MAX,2); });
    const auto json = strix::serve::parse_json(large.to_json());
    require(json.find("deployment_id")->as_string() == large.hash(),"canonical HTTP plan id");
}
void gemma_tests() {
    namespace g = strix::gemma4::engine;
    g::DeploymentModel m; m.weights = 6*GiB; m.largest_tensor = GiB;
    g::FootprintProvider provider(m);
    Spec s; s.context = 8192; s.slots = 4;
    auto small = resolve(s,{device(64)},provider);
    auto large = resolve(s,{device(192)},provider);
    auto discrete = resolve(s,{device(32,false)},provider);
    require(!small.refused && !large.refused && !discrete.refused,"Gemma unified and discrete fits");
    require(small.chosen.topology == "single" && discrete.chosen.format.decode == "q4_k_m","Gemma single Q4_K_M default");
    require(small.ranks[0].footprint.kv_per_token == 4ULL*(2*2*512*4+8),"global producer KV bytes only");
    auto one = s; one.slots = 1;
    auto p1 = resolve(one,{device(64)},provider);
    require(p1.ranks[0].kv_bytes < small.ranks[0].kv_bytes,"slots affect rings and global KV");
    auto shorter = s; shorter.context = 4096;
    require(resolve(shorter,{device(64)},provider).ranks[0].kv_bytes < small.ranks[0].kv_bytes,"context affects global KV");
    s.context = 65537;
    require(resolve(s,{device(192)},provider).refused,"native context limit");
    s.context = 8192; s.topology = "tp2";
    require(resolve(s,{device(64),device(192)},provider).refused,"absent Gemma pair route refuses");
    s.topology = "replica"; s.nodes = 2;
    auto replicas = resolve(s,{device(64),device(32,false)},g::FootprintProvider(m,true));
    require(!replicas.refused && replicas.ranks.size() == 2,"replicas each reserve complete weights");
    require(replicas.ranks[0].footprint.weights == replicas.ranks[1].footprint.weights,"replica weights not sharded");
    auto automatic = one; automatic.context = 65536; automatic.slots = 0;
    const auto auto_slots = resolve(automatic,{device(40,false)},provider);
    require(!auto_slots.refused && auto_slots.chosen.slots < 8,"unified resolver selects admitted slot count");
    auto no_cache = resolve(one,{device(64)},g::FootprintProvider(m,false,0));
    require(no_cache.ranks[0].required_bytes+4*GiB == p1.ranks[0].required_bytes,"prefix OFF restores full cache cap");
    s.format.decode = "exl3";
    require(resolve(s,{device(64),device(64)},g::FootprintProvider(m,true)).refused,"unsupported format pin refusal");
}
void metadata_tests(const std::filesystem::path& dir) {
    std::filesystem::create_directories(dir);
    const auto file = dir/"model.safetensors";
    // Sparse payload: the reader must only consume the JSON header.
    const std::string header = R"({"language_model.model.embed_tokens.weight":{"dtype":"BF16","shape":[1,6144],"data_offsets":[0,12288]},"lm_head.trellis":{"dtype":"I16","shape":[1,1,16],"data_offsets":[12288,12320]}})";
    std::ofstream f(file,std::ios::binary); const auto n = std::uint64_t(header.size());
    for (unsigned i = 0; i < 8; ++i) f.put(static_cast<char>((n>>(8*i))&255));
    f << header; f.seekp(static_cast<std::streamoff>(8+n+12319)); f.put('\0'); f.close();
    const auto model = strix::minimax::engine::deployment_model(dir);
    require(model.weights > 2*GiB,"MiniMax header-only sizing");
    std::ofstream bad(file,std::ios::binary|std::ios::trunc); bad << "broken"; bad.close();
    rejects([&] { (void)strix::minimax::engine::deployment_model(dir); });
    rejects([&] { (void)strix::gemma4::engine::deployment_model(file); });
    std::filesystem::remove(file); std::filesystem::remove(dir);
}
void gguf_header_test() {
    const auto path = std::filesystem::path("engine-deploy-fixture.gguf");
    std::ofstream f(path,std::ios::binary);
    const auto u32 = [&](std::uint32_t n) { for (unsigned i=0;i<4;++i) f.put(static_cast<char>((n>>(8*i))&255)); };
    const auto u64 = [&](std::uint64_t n) { for (unsigned i=0;i<8;++i) f.put(static_cast<char>((n>>(8*i))&255)); };
    const auto str = [&](const std::string& s) { u64(s.size()); f << s; };
    f << "GGUF"; u32(3); u64(1); u64(8);
    str("general.architecture"); u32(8); str("gemma4");
    const std::pair<const char*,std::uint32_t> fields[] = {
        {"general.file_type",15},{"gemma4.embedding_length",2560},{"gemma4.block_count",42},
        {"gemma4.feed_forward_length",10240},{"gemma4.attention.head_count_kv",2},
        {"gemma4.attention.sliding_window",512},{"gemma4.context_length",65536}};
    for (const auto& [key,n] : fields) { str(key); u32(4); u32(n); }
    str("token_embd.weight"); u32(2); u64(256); u64(1); u32(12); u64(0);
    while (static_cast<std::uint64_t>(f.tellp())%32) f.put('\0');
    for (unsigned i=0;i<144;++i) f.put('\0');
    f.close();
    const auto m = strix::gemma4::engine::deployment_model(path);
    require(m.weights == 144 && m.largest_tensor == 144 && m.hidden == 2560,"GGUF metadata geometry, no payload load");
    std::filesystem::remove(path);
}
} // namespace
int main() {
    try {
        minimax_tests(); gemma_tests(); metadata_tests("engine-deploy-fixture"); gguf_header_test();
        std::cout << "MiniMax/Gemma deployment host tests passed\n"; return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
