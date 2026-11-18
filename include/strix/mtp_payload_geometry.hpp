#pragma once
#include "strix/execution_buffers.hpp"
#include "strix/indexed_attention_state.hpp"
#include "strix/indexed_attention_workspace.hpp"
#include "strix/mtp_forward_payload.hpp"
#include "strix/mtp_target_window_payload.hpp"
#include <array>
#include <stdexcept>

namespace strix::mtp_payload {
inline std::uint64_t add(std::uint64_t a,std::uint64_t b) {
    if(b>UINT64_MAX-a) throw std::overflow_error("MTP payload addition overflow");
    return a+b;
}
inline std::uint64_t mul(std::uint64_t a,std::uint64_t b) {
    if(a && b>UINT64_MAX/a) throw std::overflow_error("MTP payload multiplication overflow");
    return a*b;
}
inline std::uint64_t align(std::uint64_t n,std::uint64_t alignment=256U) {
    if(!alignment) throw std::invalid_argument("MTP payload alignment is zero");
    return mul(add(n,alignment-1U)/alignment,alignment);
}
inline std::uint64_t selector(std::uint64_t vocabulary) {
    if(!vocabulary || vocabulary>UINT32_MAX) throw std::invalid_argument("MTP selector vocabulary differs");
    return add(mul((vocabulary-1U)/256U+1U,16U),64U);
}
struct Runtime {
    // HipMtpBuffer order; retired roles have zero storage and zero padding.
    std::array<std::uint64_t,24> sizes{},offsets{};
    std::uint64_t pool=0,reader=0,retained=0;
};
inline Runtime runtime(const ModelContract& m,std::uint64_t capacity) {
    if(!capacity || capacity>m.effective_max_positions()) throw std::invalid_argument("MTP runtime capacity differs");
    const auto buffers=build_serial_execution_buffer_plan(m,{});
    const auto attention=build_indexed_attention_workspace_layout(m);
    auto predictor=m;predictor.attention_kv_storage=AttentionKvStorage::bf16;
    const auto state=build_indexed_attention_state_layout(predictor);
    const auto wide=mul(mul(m.hidden_size,m.hc_count),2U);
    const auto input=mul(mul(m.hidden_size,add(m.hc_count,1U)),4U);
    Runtime p;
    p.sizes={wide,wide,0U,mul(m.hidden_size,2U),input,input,wide,wide,
        buffers.hidden_bytes,buffers.hidden_bytes,buffers.hidden_bytes,buffers.injection_weight_bytes,
        buffers.route_index_bytes,buffers.route_weight_bytes,buffers.hyper_workspace_bytes,attention.total_bytes,
        buffers.mlp_intermediate_bytes,mul(m.num_experts,4U),buffers.logits_bytes,
        mul(capacity,state.kv_row_bytes),mul(capacity,state.raw_index_row_bytes),0U,0U,8U};
    for(std::size_t i=0;i<p.sizes.size();++i) if(p.sizes[i]) {
        p.pool=align(p.pool);p.offsets[i]=p.pool;p.pool=add(p.pool,p.sizes[i]);
    }
    p.reader=selector(m.vocab_size);p.retained=add(add(p.pool,255U),p.reader);
    return p;
}
// Only automatically selected budgets grow with context. Explicit budgets remain ceilings;
// admission and the native owner reject a retained payload exceeding the returned ceiling.
// Invalid capacities keep the declared value for the owner's own refusal.
inline std::uint64_t runtime_budget(const ModelContract& m,std::uint64_t capacity,std::uint64_t declared,
    bool automatic=false) {
    if(!automatic || !capacity || capacity>m.effective_max_positions()) return declared;
    const auto need=runtime(m,capacity).retained;
    return need>declared?need:declared;
}
inline std::uint64_t target_capture(const ModelContract& m,std::uint64_t rows,
    const SerialSessionStateLayout* deferred) {
    if(!rows) throw std::invalid_argument("MTP target capture rows are zero");
    const auto extra=deferred?add(263U,add(deferred->host_total_bytes,deferred->host_alignment-1U)):0U;
    return add(extra,add(selector(m.vocab_size),mul(rows,add(mul(mul(m.hidden_size,m.hc_count),2U),255U))));
}
inline std::uint64_t window_capture(const ModelContract& m,std::uint64_t rows,
    std::uint64_t history,std::uint64_t tensors) {
    if(!rows || rows>8U || !history) throw std::invalid_argument("MTP window capture geometry differs");
    auto bytes=add(add(mul(mul(mul(m.hidden_size,m.hc_count),2U),rows),255U),mul(rows,selector(m.vocab_size)));
    bytes=add(bytes,mul(history,2U*sizeof(ArithmeticRowInterval)));
    // MtpDeviceView consists of three address/size/generation words.
    bytes=add(bytes,mul(rows,2U*sizeof(std::uint32_t)+sizeof(unsigned char)+2U*sizeof(ExecutionBufferAddress)+24U));
    return add(add(bytes,sizeof(MtpTargetWindowLease)),mul(tensors,sizeof(RuntimeTensorBinding)));
}
inline std::uint64_t golden_file(std::uint64_t rows,std::uint64_t width,std::uint64_t vocabulary,
    std::uint64_t header) {
    return add(header,mul(rows,add(24U,add(mul(width,4U),mul(vocabulary,4U)))));
}
inline std::uint64_t golden_host(std::uint64_t rows,std::uint64_t width,std::uint64_t vocabulary) {
    return add(sizeof(HipMtpForwardGolden),mul(rows,add(sizeof(HipMtpForwardGoldenRow),add(mul(width,4U),mul(vocabulary,4U)))));
}
inline std::uint64_t replay_host(std::uint64_t rows,std::uint64_t width,std::uint64_t vocabulary) {
    const auto row=add(mul(width,4U),mul(vocabulary,4U));
    return add(add(row,mul(rows,add(sizeof(HipMtpForwardReplayRowReceipt)+2U*sizeof(std::uint32_t),row))),mul(add(rows,1U),192U));
}
} // namespace strix::mtp_payload
