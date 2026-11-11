#include "../apps/glm_deploy_owners.hpp"
#include "glm/xfer/glm_seqpar.hpp"
#include <iostream>
#include <map>
#include <stdexcept>

namespace dp = strix::deploy;
namespace gd = strix::glm::deployment;
static void check(bool b, const char* message) { if (!b) throw std::runtime_error(message); }
static dp::DeviceFacts device(unsigned gib, bool unified = true) {
    dp::DeviceFacts d; d.gfx_target = "gfx1151"; d.cu_count = 40;
    d.mem_total_bytes = d.mem_free_bytes = std::uint64_t{gib} << 30; d.unified = unified; d.source = "fixture";
    return d;
}
int main() {
    try {
        namespace gx = strix::glm::xfer;
        check(gx::prefill_pipeline(2,nullptr)==gx::prefill_pipeline(1,"2"), "default and explicit pipeline 2 resolve identically");
        check(gx::prefill_pipeline(2,"1")==1,"pipeline 1 off switch");
        bool bad_pipeline=false;
        try { (void)gx::prefill_pipeline(1,"3"); } catch(const std::invalid_argument&) { bad_pipeline=true; }
        check(bad_pipeline,"invalid explicit pipeline refuses before device work");
        const unsigned rows=8192, depth=4;
        const auto capacity=gx::GlmSchedule::row_bytes((rows+1U)/2U);
        const auto stride=(capacity+sizeof(gx::Trailer)+4095U)&~std::size_t{4095};
        const auto pipeline_bytes=gx::seqpar_workspace_bytes(rows,depth,capacity);
        check(pipeline_bytes==2*depth*stride+gx::GlmSchedule::row_bytes(rows)+(8U<<20U),
              "pipeline bounds both rings and full normalized buffer once");
        check(gx::seqpar_workspace_bytes(rows,8,capacity)>pipeline_bytes,"channel depth changes pipeline bound");
        bool bad_depth=false, big_stride=false;
        try { (void)gx::seqpar_workspace_bytes(rows,1,capacity); } catch(const std::invalid_argument&) { bad_depth=true; }
        try { (void)gx::seqpar_workspace_bytes(rows,4,SIZE_MAX); } catch(const std::overflow_error&) { big_stride=true; }
        check(bad_depth && big_stride,"pipeline bound rejects invalid depth and overflow");
        gd::Options o; o.slots = 4;
        const std::array<gd::Owners, 2> weights{{{70ULL<<30, 1ULL<<30}, {71ULL<<30, 1ULL<<30}}};
        const auto workspace = [](std::uint32_t, bool shard) { return (shard ? 3ULL : 2ULL)<<30; };
        gd::Provider provider(weights, o, workspace);
        dp::Spec s; s.nodes = 2; s.link = "usb4-auto"; s.context = 8192; s.slots = 1;
        auto p = dp::resolve(s, {device(128),device(128)}, provider);
        check(!p.refused && p.chosen.topology == "tp2" && p.chosen.format.prefill == "exl3", "EXL3 TP2 fit");
        check(p.ranks[0].footprint.weights != p.ranks[1].footprint.weights, "rank owners preserved");
        check(p.ranks[0].kv_bytes == 8192ULL*16*1664 + 30ULL*(32*128*128*4+3*4096*3*4), "replicated KV/state formula");
        check(p.ranks[0].after_kv_bytes == p.ranks[0].budget_bytes-p.ranks[0].kv_bytes, "KV first");
        check(dp::resolve(s, {device(128)}, provider).refused, "missing peer refuses");
        check(dp::resolve(s, {device(128),device(64)}, provider).refused, "mixed pair limited by smaller rank");
        check(dp::resolve(s, {device(32,false),device(32,false)}, provider).refused, "discrete bytes, no named tier");
        s.format.prefill = "int4-stored";
        const auto i4 = dp::resolve(s, {device(192),device(192)}, provider);
        check(i4.refused && i4.refusal.find("no supported candidate") != std::string::npos, "absent INT4 runtime advertises no INT4 candidate");
        s.format.prefill = "auto";
        o.mtp = false; o.kv_shard = "1";
        gd::Provider shard(weights,o,workspace);
        const auto sh = dp::resolve(s,{device(128),device(128)},shard);
        check(!sh.refused && sh.ranks[0].kv_bytes < p.ranks[0].kv_bytes, "sharded tokens and recurrent state");
        o.kv_shard = "auto";
        gd::Provider automatic(weights,o,workspace);
        const auto au = dp::resolve(s,{device(128),device(128)},automatic);
        check(!au.refused && au.ranks[0].kv_bytes > sh.ranks[0].kv_bytes, "auto reserves replicated upper bound");
        s.context=262144;
        const auto longctx=dp::resolve(s,{device(128),device(128)},provider);
        check(!longctx.refused && longctx.ranks[0].kv_bytes > p.ranks[0].kv_bytes,"long context reserves requested KV");
        s.context=4000000;
        check(dp::resolve(s,{device(128),device(128)},provider).refused,"KV never shrinks to fit");
        s.context=8192;
        s.slots = 2;
        check(dp::resolve(s,{device(128),device(128)},shard).refused,"shard requires one slot");
        s.slots=1; s.context=8193;
        check(dp::resolve(s,{device(128),device(128)},provider).refused,"unaligned context refuses");
        struct Tensor { std::uint64_t bytes; };
        std::map<std::string, Tensor> tensors{{"model.language_model.embed_tokens.weight",{1ULL<<30}}};
        for (int e = 0; e < 16; ++e) {
            const auto pre = "model.language_model.layers.3.mlp.experts." + std::to_string(e) + ".gate_proj.";
            tensors[pre + "trellis"] = {8ULL<<20}; tensors[pre + "suh"] = {8192}; tensors[pre + "svh"] = {4096};
        }
        const auto fast = gd::header_owners(tensors,true,{true,true,false}), ref = gd::header_owners(tensors,true,{false,true,false});
        const auto kept = gd::header_owners(tensors,true,{false,true,true});
        check(kept.weights > ref.weights, "--raw-linears adds the raw arena copy");
        check(ref.weights > fast.weights && fast.staging >= (1ULL<<30), "header bounds include prepared route and staging");
        tensors.at("model.language_model.layers.3.mlp.experts.0.gate_proj.trellis").bytes=UINT64_MAX;
        bool overflow=false;
        try { (void)gd::header_owners(tensors,true,{true,true,false}); } catch(const std::overflow_error&) { overflow=true; }
        check(overflow,"header overflow refuses");
        gd::Options pending=o; pending.profile="fast";
        gd::Provider selected(weights,pending,workspace);
        s.context=8192;s.profile="quality";
        check(dp::resolve(s,{device(192),device(192)},selected).refused,"profile pin honored or refused");
        std::cout << "GLM deployment host checks passed\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
