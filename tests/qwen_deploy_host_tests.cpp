#include "../apps/qwen_deploy_provider.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
namespace dp = strix::deploy;
static void check(bool b, const char* m) { if (!b) throw std::runtime_error(m); }
static dp::DeviceFacts device(unsigned gib) {
    dp::DeviceFacts d; d.gfx_target="gfx1201"; d.cu_count=64;
    d.mem_total_bytes=d.mem_free_bytes=std::uint64_t{gib}<<30; d.source="fixture"; return d;
}
int main() {
    try {
        constexpr std::uint64_t gib = 1ULL<<30;
        auto node = device(128); node.mem_total_bytes = node.mem_free_bytes = 1172*gib/10;
        dp::Spec automatic; automatic.context = 8192;
        dp::Footprint http; http.weights=80*gib; http.kv_total=gib; http.workspace=9*gib;
        http.elastic_extra.emplace_back("session_ram",8*gib);
        strix::qwen::deployment::Provider batch(http,8192,4,"single","default",{},2*gib,gib,gib);
        const auto four=dp::resolve(automatic,{node},batch);
        check(!four.refused && four.chosen.slots==4,"KV-first admits native HTTP stacks before session RAM");
        check(four.ranks[0].kv_bytes==4*gib,"only persistent state is classified as KV; scratch and metadata stay separate");
        check(four.ranks[0].footprint.extra.back().first=="session_ram" &&
              four.ranks[0].required_bytes<=four.ranks[0].budget_bytes,"elastic session preserves safety margin");
        auto tight=node; tight.mem_free_bytes=100*gib;
        const auto fewer=dp::resolve(automatic,{tight},batch);
        check(!fewer.refused && fewer.chosen.slots==1 && fewer.ranks[0].footprint.extra.back().second==gib,
              "session shrinks after slot search, never hides KV overcommit");
        automatic.slots=4;
        check(dp::resolve(automatic,{tight},batch).refused,"pinned slots never shrink");
        automatic.slots=0;
        dp::Footprint cli; cli.weights=103*gib; cli.kv_total=gib; cli.workspace=gib;
        cli.elastic_extra.emplace_back("session_ram",8*gib);
        strix::qwen::deployment::Provider cli_auto(cli,8192,1,"single","default");
        const auto cli_fit=dp::resolve(automatic,{node},cli_auto);
        check(!cli_fit.refused && cli_fit.ranks[0].footprint.extra.back().second==node.mem_free_bytes-113*gib,
              "default CLI session budget fits remaining measured bytes");
        cli.elastic_extra.clear(); cli.extra.emplace_back("session_ram",8*gib);
        strix::qwen::deployment::Provider cli_pin(cli,8192,1,"single","default");
        check(dp::resolve(automatic,{node},cli_pin).refused,"explicit session budget never shrinks");
        dp::Footprint f; f.weights=20ULL<<30; f.kv_total=2ULL<<30; f.workspace=1ULL<<30;
        strix::qwen::deployment::Provider single(f,8192,1,"single","default");
        dp::Spec s; s.context=8192; s.slots=1;
        auto p=dp::resolve(s,{device(32)},single);
        check(!p.refused && p.ranks[0].required_bytes==23ULL<<30,"single owner inventory exactly once");
        check(dp::resolve(s,{device(24)},single).refused,"free bytes/margin arithmetic");
        strix::qwen::deployment::Provider replicas(f,8192,1,"replica","default");
        s.nodes=2;
        auto r=dp::resolve(s,{device(32),device(64)},replicas);
        check(!r.refused && r.ranks.size()==2,"two independent replicas");
        check(r.ranks[0].required_bytes==r.ranks[1].required_bytes,"full replica on each node");
        s.nodes=1;
        check(dp::resolve(s,{device(64)},replicas).refused,"replica binding requires two nodes");
        s.nodes=2;
        strix::qwen::deployment::Provider pipeline(f,8192,1,"pipeline","default");
        check(dp::resolve(s,{device(32),device(64)},pipeline).refused,"pipeline requires link");
        s.link="usb4auto";
        auto pp=dp::resolve(s,{device(32),device(64)},pipeline);
        check(!pp.refused && pp.ranks[0].footprint.weights==f.weights,"pipeline full model continuation");
        check(pp.ranks[0].footprint.staging==32ULL<<20,"pipeline handoff staging");
        s.context=16384;
        check(dp::resolve(s,{device(64),device(64)},pipeline).refused,"context cannot drift from compiler owners");
        s.context=8192; s.slots=2;
        check(dp::resolve(s,{device(64),device(64)},pipeline).refused,"unsupported slots pin refuses");
        s.slots=1; s.profile="quality";
        check(dp::resolve(s,{device(64),device(64)},pipeline).refused,"profile pin refuses");
        s.profile="auto"; s.format.decode="int4";
        check(dp::resolve(s,{device(64),device(64)},pipeline).refused,"cannot claim another GGUF encoding");
        s.format.decode="auto";
        unsigned calls=0;
        const auto echo_peer=[&](std::span<const std::byte> out,std::span<std::byte> in) {
            std::memcpy(in.data(),out.data(),out.size());
            if (!calls++) { std::uint64_t rank=1; std::memcpy(in.data()+8,&rank,8); }
        };
        const auto agreed=dp::resolve_pair(s,pipeline,0,device(32),echo_peer);
        check(!agreed.refused && calls==2,"live facts and plan agreement");
        calls=0; bool mismatch=false;
        try {
            (void)dp::resolve_pair(s,pipeline,0,device(32),[&](auto out,auto in) {
                echo_peer(out,in); if(calls==2) in[0]^=std::byte{1};
            });
        } catch(const std::runtime_error&) { mismatch=true; }
        check(mismatch,"peer plan mismatch refuses");
        check(dp::inspection_devices(dp::Spec{},0,device(32)).size()==1,"no fabricated remote memory");
        std::cout<<"Qwen deployment host checks passed\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
