#pragma once
#include "qwen_effective_config.hpp"
#include "strix/hip_mtp_fast_draft.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/q51_known_rows.hpp"
#include "strix/mtp_payload_geometry.hpp"
#include "strix/hip_mtp_draft_head.hpp"
#include "strix/mtp_target_checkpoint_plan.hpp"
namespace strix::decode_cli {
// Setup, calibration, prompt, decode, and drain. Unspecified owners live in all phases.
inline constexpr unsigned mtp_setup=1U,mtp_calibration=2U,mtp_prompt=4U,mtp_decode=8U,mtp_drain=16U,mtp_all=31U;
struct YarnMtpGeometry {
    gguf::FileInfo sidecar;
    std::vector<std::pair<std::uint32_t,std::uint64_t>> window_arenas;
    std::uint64_t target_tensor_count=0;
    std::uint64_t page_bytes=4096;
};
struct YarnMtpOwners {
    std::map<std::string,std::uint64_t> bytes,caps,actuals;
    std::map<std::string,unsigned> phases;
    std::uint64_t device=0,post_setup_peak=0;
    std::string refusal;
};
inline std::uint64_t mtp_q4_manifest_bytes(const gguf::FileInfo& info) {
    std::uint64_t bytes=0;
    for(const auto& t:info.tensors) {
        const auto at=t.name.find('.',4U);
        if(at==std::string::npos) continue;
        const auto suffix=std::string_view(t.name).substr(at+1U);
        if(suffix!="attn_q.weight" && suffix!="attn_k.weight" && suffix!="attn_v.weight" &&
           suffix!="attn_output.weight" && suffix!="ffn_gate_exps.weight" && suffix!="ffn_up_exps.weight" &&
           suffix!="ffn_down_exps.weight" && suffix!="ffn_gate_shexp.weight" && suffix!="ffn_up_shexp.weight" &&
           suffix!="ffn_down_shexp.weight" && suffix!="nextn.eh_proj.weight") continue;
        if(t.ggml_type!=8U || t.byte_size%34U) throw std::invalid_argument("YaRN MTP Q4 source is not complete Q8 blocks");
        bytes=prefill_checked_add(bytes,t.byte_size/34U*18U);
    }
    if(!bytes) throw std::invalid_argument("YaRN MTP Q4 manifest has no converted tensors");
    return bytes;
}
inline YarnMtpOwners yarn_mtp_owners(const Options& o,const ModelContract& m,const YarnMtpGeometry& geometry) {
    YarnMtpOwners plan;
    if(!o.mtp.enabled) return plan;
    using mtp_payload::add;using mtp_payload::mul;using mtp_payload::align;
    auto& b=plan.bytes;const auto& mtp=o.mtp;
    const bool dump=!mtp.golden_dump.empty(),chain=mtp.draft_chain && !dump;
    const auto bounded=[&](const std::string& name,std::uint64_t actual,std::uint64_t cap,unsigned phases) {
        b[name]=actual;plan.actuals[name]=actual;plan.caps[name]=cap;plan.phases[name]=phases;
        if(actual>cap && plan.refusal.empty()) plan.refusal="YaRN MTP "+name+" actual="+std::to_string(actual)+" exceeds cap="+std::to_string(cap);
    };
    const auto common=mtp_all^mtp_setup;
    const auto cache=mul(o.context_tokens,2304U);
    const auto runtime=mtp_payload::runtime(m,o.context_tokens);
    bounded("mtp_predictor_runtime",runtime.retained,mtp_payload::runtime_budget(m,o.context_tokens,mtp.runtime_bytes,mtp.runtime_automatic),common);
    // Split cache/workspace for inspection without introducing a second charge.
    b.erase("mtp_predictor_runtime");plan.phases.erase("mtp_predictor_runtime");
    b["mtp_predictor_kv_raw"]=cache;b["mtp_predictor_workspace_alignment"]=runtime.retained-cache;
    plan.phases["mtp_predictor_kv_raw"]=plan.phases["mtp_predictor_workspace_alignment"]=common;
    const auto& info=geometry.sidecar;
    if(!info.file_size || info.file_size!=mtp.sidecar_bytes) throw std::invalid_argument("YaRN MTP sidecar size changed before admission");
    auto sidecar=align(info.file_size,geometry.page_bytes);
    if(mtp.sidecar_residency=="copy") {
        std::vector<std::pair<std::uint64_t,std::uint64_t>> ranges;
        for(const auto& t:info.tensors) ranges.emplace_back(t.absolute_offset,add(t.absolute_offset,t.byte_size));
        std::sort(ranges.begin(),ranges.end());
        std::uint64_t end=0;
        for(const auto& [first,last]:ranges) { if(last>info.file_size) throw std::invalid_argument("MTP sidecar tensor outside file");
            sidecar=add(sidecar,last>std::max(first,end)?last-std::max(first,end):0U);end=std::max(end,last); }
    }
    bounded("mtp_sidecar_retained",sidecar,mtp.sidecar_retained_bytes,mtp_all);
    // Register aliases the mapping; ordinary copy has no selective-copy staging.
    bounded("mtp_sidecar_setup_peak",sidecar,mtp.sidecar_setup_bytes,mtp_setup);
    b.erase("mtp_sidecar_setup_peak");plan.phases.erase("mtp_sidecar_setup_peak");
    b["mtp_sidecar_setup_overlap"]=0;plan.phases["mtp_sidecar_setup_overlap"]=mtp_setup;
    bounded("mtp_sidecar_header",info.data_offset,mtp.header_bytes,mtp_setup);
    SerialSessionStateLayoutRequest sr;sr.context_tokens=o.context_tokens;
    const auto state=build_serial_session_state_layout(m,sr);
    const auto calibration=mtp_tiled_prompt(o)?mtp_calibration_rows(o):o.tokens.size()-1U;
    if(!calibration || calibration>4096U) throw std::invalid_argument("MTP calibration exceeds golden row bound");
    const auto tail=o.known_prefix_tail_capture_rows?o.known_prefix_tail_capture_rows:
        o.known_prefix_tail_rows.empty()?o.known_prefix_prefill_rows:o.known_prefix_tail_rows.back();
    const auto rows=mtp_tiled_prompt(o)?std::max<std::uint64_t>({calibration+1U,tail+1U,mtp.candidate_depth+3U}):
        std::max<std::uint64_t>(o.tokens.size(),mtp.candidate_depth+3U);
    bounded("mtp_target_hidden_captures",mtp_payload::target_capture(m,rows,o.deferred_moe_status?&state:nullptr),mtp.target_capture_bytes,common);
    const auto width=mul(m.hidden_size,m.hc_count);
    const auto file=mtp_payload::golden_file(calibration,width,m.vocab_size,o.rotary.mode()==RotaryMode::yarn?288U:184U);
    bounded("mtp_golden_file",file,mtp.golden_max_bytes,mtp_calibration);
    b.erase("mtp_golden_file");plan.phases.erase("mtp_golden_file");
    b["mtp_golden_host"]=mtp_payload::golden_host(calibration,width,m.vocab_size);plan.phases["mtp_golden_host"]=mtp_calibration;
    // Encoded file storage and replay scratch do not overlap. Golden rows coexist with either.
    const auto replay=dump?0U:mtp_payload::replay_host(calibration,width,m.vocab_size);
    bounded("mtp_replay_host",replay,mtp.replay_host_bytes,mtp_calibration);
    const auto receipts=dump?0U:mul(calibration,sizeof(HipMtpForwardReplayRowReceipt)+2U*sizeof(std::uint32_t));
    b["mtp_replay_receipts"]=receipts;plan.phases["mtp_replay_receipts"]=common;
    b["mtp_replay_host"]-=receipts;
    b["mtp_golden_codec_overlap"]=file>replay?file-replay:0U;plan.phases["mtp_golden_codec_overlap"]=mtp_calibration;
    b["mtp_q4_shared_weights"]=chain && !mtp.draft_exact && o.yarn_mtp_q4_all?mtp_q4_manifest_bytes(info):0U;
    b["mtp_fast_workspace_hc"]=chain && !mtp.draft_exact?add(mul(hip_mtp_fast_moe_workspace_floats,4U),mul(mul(m.hidden_size,m.hc_count),m.hc_count*8U)):0U;
    // Subset capacity depends on supplied IDs and configured context vocabulary, never C.
    std::uint64_t head_capacity=m.vocab_size,base_host_capacity=0;
    const bool full=mtp.draft_vocab.empty() && !mtp.draft_context_vocab;
    if(chain && !full) {
        std::uint64_t base=0;
        if(!mtp.draft_vocab.empty()) {
            std::string chosen;
            const auto spec=HipMtpDraftHead::resolve_vocabulary_spec(mtp.draft_vocab,o.tokens,static_cast<std::uint32_t>(m.vocab_size),chosen);
            const auto ids=HipMtpDraftHead::read_token_file(spec,static_cast<std::uint32_t>(m.vocab_size));
            base=ids.size();base_host_capacity=ids.capacity();
        }
        head_capacity=add(base,mtp.draft_context_vocab);
        if(!head_capacity || head_capacity>m.vocab_size) throw std::invalid_argument("MTP draft head capacity exceeds vocabulary");
    }
    // Native max_slots=16, Partial=24, Reply=200. Full head borrows target weights.
    auto head=add(mul(head_capacity,4U),add(mul((head_capacity+255U)/256U,24U),264U));
    if(!full) head=add(head,add(mul(base_host_capacity,4U),add(m.vocab_size,mul(head_capacity,add(mul(m.hidden_size/32U,mtp.draft_q4?18U:34U),8U)))));
    const auto top2=o.yarn_mtp_draft_top2;
    if(top2) head=add(head,mul(head_capacity,64U));
    b["mtp_draft_head"]=chain?head:0U;
    for(const auto* name:{"mtp_q4_shared_weights","mtp_fast_workspace_hc","mtp_draft_head"}) plan.phases[name]=mtp_prompt|mtp_decode|mtp_drain;
    std::uint64_t windows=0,arenas=0,unions=0,captures=0;
    const auto expected=mtp.window_rows_set.empty()?std::vector<std::uint32_t>{mtp.window_rows}:mtp.window_rows_set;
    if(!dump && mtp.window_rows && geometry.window_arenas.size()!=expected.size()) throw std::invalid_argument("MTP window geometry missing actual compiled owners");
    if(!dump) for(const auto& [r,arena]:geometry.window_arenas) {
        if(std::find(expected.begin(),expected.end(),r)==expected.end()) throw std::invalid_argument("MTP window geometry differs from row set");
        const auto ub=mtp.window_expert_union?q51_known_rows::required_scratch_bytes(r,static_cast<std::uint32_t>(m.num_experts),static_cast<std::uint32_t>(m.num_experts_per_token)):0U;
        arenas=add(arenas,add(arena,8U));unions=add(unions,ub);
        const auto capture=mtp_payload::window_capture(m,r,mtp.window_history_intervals,geometry.target_tensor_count);
        bounded("mtp_window_capture_r"+std::to_string(r),capture,mtp.window_capture_bytes,common);
        b.erase("mtp_window_capture_r"+std::to_string(r));
        captures=add(captures,mul(4U,capture));
        // Separate native greedy rows partials and mapped replies, coexisting with selectors.
        if(o.yarn_mtp_greedy_rows) windows=add(windows,add(mul(r,mtp_payload::selector(m.vocab_size)-64U),mul(r,8U)));
    }
    const auto checkpoint=!dump && mtp.window_rows && !mtp.window_prefix?
        plan_hip_mtp_target_window_rollback(m,state,0,mtp.window_rows,true).reservation.retained_bytes:0U;
    bounded("mtp_window_arena",add(arenas,unions),mtp.window_arena_bytes,common);
    bounded("mtp_window_union",unions,mtp.window_expert_union_bytes,common);
    bounded("mtp_window_checkpoint",checkpoint,mtp.window_checkpoint_bytes,common);
    for(const auto* name:{"mtp_window_arena","mtp_window_union","mtp_window_checkpoint"}) b.erase(name);
    windows=add(windows,add(add(arenas,unions),add(captures,checkpoint)));
    const auto m1=qsa_mtp_pool_bytes(o,m);
    bounded("mtp_window_arenas_captures_checkpoints",add(windows,m1),mtp.window_total_bytes,common);
    b["mtp_window_arenas_captures_checkpoints"]=windows; // M1 already belongs to target state owners.
    const auto owners=dump?0U:geometry.window_arenas.size();
    const auto key=mul(m.gdn_key_heads,m.gdn_key_head_dim),value=mul(m.gdn_value_heads,m.gdn_value_head_dim);
    const auto conv=add(mul(key,2U),value);
    const auto saved_row=align(add(align(mul(conv,2U)),mul(add(add(key,value),mul(m.gdn_value_heads,2U)),4U)));
    const auto gdn=add(mul(mul(saved_row,16U),m.gdn_layers),256U);
    std::uint64_t external=0;
    if(!dump) for(const auto& [r,arena]:geometry.window_arenas) {
        (void)arena;
        const auto csr=q51_known_rows::required_scratch_bytes(r,static_cast<std::uint32_t>(m.num_experts),static_cast<std::uint32_t>(m.num_experts_per_token));
        const auto ple=add(mul(mul(r,PleHashContract::maximum_heads),1024U),
            add(mul(mul(r,add(mul(m.hidden_size,add(m.hc_count,1U)),m.hc_count)),4U),
                add(ple_conv_state_bytes_per_session(m),mul(r,sizeof(PleHashResult)))));
        external=add(external,add(gdn,add(csr,ple)));
    }
    b["mtp_window_external_ple_csr_gdn"]=external;
    b["mtp_tapped_hidden_pipeline"]=dump?0U:mul(mul(o.known_prefix_prefill_rows,width),o.yarn_mtp_pipeline?4U:2U);
    plan.phases["mtp_tapped_hidden_pipeline"]=mtp_prompt;
    b["mtp_held_hidden"]=mtp_tiled_prompt(o)?mul(width,2U):0U;plan.phases["mtp_held_hidden"]=common;
    const auto blocks=o.context_tokens/4U;
    b["mtp_historical_window_pools"]=qsa_mtp_pool_route(o)?0U:mul(owners,mul(mul(blocks,128U),16U*4U));
    b["mtp_historical_predictor_pool"]=qsa_mtp_pool_route(o)?0U:mul(mul(blocks,128U),4U);
    for(const auto& [name,bytes]:b) { (void)bytes;if(!plan.phases.contains(name)) plan.phases[name]=common; }
    if(mtp_prompt_phase_separation(o))
        for(const auto* name:{"mtp_window_arenas_captures_checkpoints","mtp_window_external_ple_csr_gdn","mtp_historical_window_pools"})
            plan.phases[name]=mtp_decode|mtp_drain;
    for(const auto phase:{mtp_setup,mtp_calibration,mtp_prompt,mtp_decode,mtp_drain}) {
        std::uint64_t total=0,device=0;
        for(const auto& [name,bytes]:b) {
            const auto mask=plan.phases.contains(name)?plan.phases.at(name):common;
            if(!(mask&phase)) continue;
            total=add(total,bytes);
            if(name!="mtp_replay_receipts" && name!="mtp_golden_host" && name!="mtp_replay_host" && name!="mtp_golden_codec_overlap" && name!="mtp_sidecar_header") device=add(device,bytes);
        }
        if(phase!=mtp_setup) plan.post_setup_peak=std::max(plan.post_setup_peak,total);
        plan.device=std::max(plan.device,device);
    }
    return plan;
}
} // namespace strix::decode_cli
