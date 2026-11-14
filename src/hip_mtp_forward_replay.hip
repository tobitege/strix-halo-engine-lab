#include "strix/hip_check.hpp"
#include "strix/hip_mtp_forward_replay.hpp"
#include "strix/mtp_payload_geometry.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/mtp_forward_golden_file.hpp"
#include "strix/hip_launch_context.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace strix {
namespace {
constexpr std::uint64_t reference_limit=4096U;
static_assert(sizeof(float)==4U && std::numeric_limits<float>::is_iec559);
void require(bool v,const char* m){if(!v)throw std::invalid_argument(m);}
std::uint64_t add(std::uint64_t a,std::uint64_t b){if(b>UINT64_MAX-a)throw std::overflow_error("MTP replay payload overflow");return a+b;}
std::uint64_t mul(std::uint64_t a,std::uint64_t b){if(b && a>UINT64_MAX/b)throw std::overflow_error("MTP replay payload overflow");return a*b;}
using strix::hip_check;
void append(std::vector<std::byte>& out,std::uint64_t v){for(unsigned shift=0;shift<64U;shift+=8U)out.push_back(static_cast<std::byte>((v>>shift)&255U));}
void append(std::vector<std::byte>& out,const ModelContentIdentity& d){for(auto v:d.bytes)out.push_back(static_cast<std::byte>(v));}
template<class T>ModelContentIdentity digest(const std::vector<T>& values){return sha256_bytes(std::as_bytes(std::span<const T>(values)));}
float decode(std::uint16_t v){return std::bit_cast<float>(static_cast<std::uint32_t>(v)<<16U);}
std::uint32_t firstmax(std::span<const float> row){
    require(!row.empty() && row.size()<=UINT32_MAX,"MTP replay vocabulary is invalid");
    std::uint32_t best=0;
    for(std::size_t i=0;i<row.size();++i){require(std::isfinite(row[i]),"MTP replay logits must be entirely finite");if(row[i]>row[best])best=static_cast<std::uint32_t>(i);}
    return best;
}
void read(MtpDeviceView v,void* destination,std::uint64_t bytes,hipStream_t stream){
    require(v.address && v.allocation_generation && v.bytes>=bytes && bytes<=std::numeric_limits<std::uintptr_t>::max()-v.address,
        "MTP replay output view is invalid");
    hip_check(hipMemcpyAsync(destination,reinterpret_cast<const void*>(v.address),static_cast<std::size_t>(bytes),hipMemcpyDeviceToHost,stream));
}
}
ModelContentIdentity hip_mtp_forward_golden_payload_identity(const HipMtpForwardGolden& g){
    require(std::endian::native==std::endian::little,"MTP golden payload requires little-endian IEEE storage");
    require(g.rows.size()<=reference_limit,"MTP golden row count exceeds bounded framing");
    std::vector<std::byte> frame;frame.reserve(static_cast<std::size_t>(mul(add(g.rows.size(),1U),192U)));
    append(frame,0x4d54505245504c31ULL);append(frame,g.independent_source_content);append(frame,g.target_content);append(frame,g.sidecar_content);
    append(frame,g.independent_reference_identity);append(frame,g.structural_plan_identity);append(frame,g.semantic_identity);append(frame,g.numerical_profile_identity);append(frame,g.rows.size());
    if(g.rotary_factor) {
        append(frame,UINT64_C(0x594d5450474f4c33));
        append(frame,g.rotary_factor);append(frame,g.rotary_descriptor_sha256);append(frame,g.rotary_table_sha256);
        append(frame,g.context_capacity);append(frame,g.calibration_begin);append(frame,g.calibration_end);
        append(frame,g.target_prefill_profile_identity);append(frame,g.target_decode_profile_identity);
        append(frame,g.prompt_prefill_profile_identity);
    }
    for(const auto& row:g.rows){append(frame,row.source_position);append(frame,row.source_input_token);append(frame,row.next_input_token);append(frame,row.selected_token);
        append(frame,row.target_hidden_bf16.size());append(frame,digest(row.target_hidden_bf16));append(frame,row.next_hidden_bf16.size());append(frame,digest(row.next_hidden_bf16));
        append(frame,row.full_logits.size());append(frame,digest(row.full_logits));}
    return sha256_bytes(frame);
}
HipMtpNativeForwardReplayResult replay_hip_mtp_forward(HipMtpRuntime& runtime,std::span<const HipMtpReplayRow> inputs,
    const HipMtpForwardGolden& g,const HipMtpForwardReplayRequest& request){
    const auto& owner=runtime.resources();owner.validate();owner.require_idle();
    require(current_hip_launch_context()==nullptr,"MTP replay must not borrow an active launch scope");
    hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
    hip_check(hipStreamIsCapturing(static_cast<hipStream_t>(owner.stream()),&capture));
    require(capture==hipStreamCaptureStatusNone,"MTP replay must not enter active capture");
    hip_check(hipPeekAtLastError());
    require(!runtime.failed() && runtime.next_source_position()==0U,"MTP full replay initial slice requires a fresh native predictor");
    require(request.policy.minimum_steps>=2U && request.maximum_rows && request.maximum_rows<=reference_limit && inputs.size()>=request.policy.minimum_steps &&
        inputs.size()<=request.maximum_rows && inputs.size()==g.rows.size() && !inputs.empty(),"MTP replay row count is invalid");
    require(g.independent_source_content.valid() && g.independent_reference_identity && request.expected_golden_payload.valid() &&
        g.target_content==owner.shared_target().target_model_content_identity &&
        g.sidecar_content==owner.shared_target().sidecar_model_content_identity && g.structural_plan_identity==owner.plan().plan_identity &&
        g.semantic_identity==owner.plan().forward_semantics.contract_identity &&
        g.numerical_profile_identity==runtime.numerical_profile_identity(),"MTP independent golden source/model/plan/profile binding mismatch");
    if(g.rotary_factor) require(request.target_prefill_profile_identity && request.target_decode_profile_identity &&
        request.prompt_prefill_profile_identity,"YaRN MTP replay requires the actual target profile pair");
    validate_mtp_forward_golden_rotary(g,owner.model().rotary_identity(),owner.context_capacity(),
        request.prompt_prefill_profile_identity,request.target_prefill_profile_identity,request.target_decode_profile_identity);
    // Validate the declared policy before any device copy, not after mutation.
    (void)assess_mtp_forward_replay(request.policy,{});
    const auto width=mul(owner.model().hidden_size,owner.model().hc_count),vocabulary=owner.model().vocab_size;
    require(width && width<=SIZE_MAX/2U && vocabulary && vocabulary<=UINT32_MAX && vocabulary<=SIZE_MAX/4U,
        "MTP replay canonical capture geometry is invalid");
    const auto budget=mtp_payload::replay_host(inputs.size(),width,vocabulary);
    require(request.host_payload_budget_bytes && budget<=request.host_payload_budget_bytes,"MTP replay exceeds explicit host payload budget");
    for(std::size_t i=0;i<inputs.size();++i){const auto& row=g.rows[i];const auto& input=inputs[i];
        require(input.hidden.valid() && input.hidden.same_producer(inputs[0].hidden) && input.hidden.source_position()==i &&
            row.source_position==i && input.hidden.input_token()==row.source_input_token && input.next_input_token==row.next_input_token &&
            row.source_input_token<vocabulary && row.next_input_token<vocabulary && i+1U<owner.context_capacity(),
            "MTP replay ticket/seed/source ordering or capacity mismatch");
        require(input.hidden.target_content_identity()==g.target_content && input.hidden.committed_session().next_position==i+1U &&
            input.hidden.committed_session().health==SerialSessionHealth::ready && !input.hidden.committed_session().arithmetic_lineage.tainted,
            "MTP replay target ticket is not its actual completed source row");
        require(row.target_hidden_bf16.size()==width && row.next_hidden_bf16.size()==width && row.full_logits.size()==vocabulary &&
            firstmax(row.full_logits)==row.selected_token,"MTP independent golden row is incomplete or has invalid first-max");
        for(auto v:row.target_hidden_bf16)require(std::isfinite(decode(v)),"MTP golden target hidden must be finite");
        for(auto v:row.next_hidden_bf16)require(std::isfinite(decode(v)),"MTP golden predictor hidden must be finite");
    }
    const auto payload=hip_mtp_forward_golden_payload_identity(g);require(payload==request.expected_golden_payload,"MTP full golden payload digest differs");
    HipMtpNativeForwardReplayResult result;result.golden_payload=payload;result.semantic_identity=g.semantic_identity;
    result.numerical_profile_identity=g.numerical_profile_identity;
    auto& observation=result.observation;observation.native_arithmetic_profile_identity=result.numerical_profile_identity;
    observation.proposal_identity=owner.plan().proposal_identity;
    observation.plan_identity=owner.plan().plan_identity;observation.independent_reference_identity=g.independent_reference_identity;
    observation.reference_tokens.reserve(inputs.size());observation.engine_tokens.reserve(inputs.size());result.rows.reserve(inputs.size());
    std::vector<std::uint16_t> target_hidden(static_cast<std::size_t>(width)),actual_hidden(static_cast<std::size_t>(width));
    std::vector<float> actual_logits(static_cast<std::size_t>(vocabulary));const auto stream=static_cast<hipStream_t>(owner.stream());
    try{
        // Immutable actual target snapshots are all read/checked BEFORE any
        // predictor input is consumed; target hidden is never a self-oracle.
        for(std::size_t i=0;i<inputs.size();++i){read(inputs[i].hidden.view(),target_hidden.data(),mul(width,2U),stream);hip_check(hipStreamSynchronize(stream));
            require(target_hidden==g.rows[i].target_hidden_bf16,"actual completed target hidden differs from independent input golden");}
        for(std::size_t i=0;i<inputs.size();++i){
            if(request.safe_boundary) request.safe_boundary();
            const auto actual=runtime.execute_diagnostic(inputs[i].hidden,inputs[i].next_input_token);
            require(actual.device_completed && actual.completed_steps==owner.plan().steps.size() && actual.source_position==i &&
                actual.seed_position==i+1U && actual.numerical_profile_identity==g.numerical_profile_identity &&
                actual.wide_state==owner.view(HipMtpBuffer::wide_current) && actual.logits==owner.view(HipMtpBuffer::logits),
                "MTP diagnostic publication is not the admitted complete native row");
            read(actual.wide_state,actual_hidden.data(),mul(width,2U),stream);read(actual.logits,actual_logits.data(),mul(vocabulary,4U),stream);
            hip_check(hipStreamSynchronize(stream));const auto selected=firstmax(actual_logits);require(selected==actual.token,"MTP device first-max differs from full CPU first-max");
            for(std::size_t j=0;j<actual_hidden.size();++j){const auto v=decode(actual_hidden[j]);require(std::isfinite(v),"MTP native wide hidden is nonfinite");
                observation.maximum_hidden_abs_error=std::max(observation.maximum_hidden_abs_error,std::abs(static_cast<double>(v)-decode(g.rows[i].next_hidden_bf16[j])));}
            for(std::size_t j=0;j<actual_logits.size();++j)observation.maximum_logit_abs_error=std::max(observation.maximum_logit_abs_error,
                std::abs(static_cast<double>(actual_logits[j])-g.rows[i].full_logits[j]));
            observation.reference_tokens.push_back(g.rows[i].selected_token);observation.engine_tokens.push_back(selected);
            result.rows.push_back({actual.source_position,actual.seed_position,selected,actual.completed_steps,digest(actual_hidden),digest(actual_logits)});
        }
        observation.hidden_values_compared=mul(width,inputs.size());observation.logit_values_compared=mul(vocabulary,inputs.size());
        result.assessment=assess_mtp_forward_replay(request.policy,observation);return result;
    }catch(...){const_cast<HipMtpResources&>(owner).fail();throw;}
}
HipMtpForwardGolden capture_hip_mtp_forward_golden(HipMtpRuntime& runtime,std::span<const HipMtpReplayRow> inputs,
    const ModelContentIdentity& source,std::uint64_t reference,std::function<void()> safe_boundary){
    const auto& owner=runtime.resources();owner.validate();owner.require_idle();
    require(current_hip_launch_context()==nullptr,"MTP golden capture must not borrow an active launch scope");
    require(!runtime.failed() && runtime.next_source_position()==0U,"MTP golden capture requires a fresh native predictor");
    require(source.valid() && reference && !inputs.empty() && inputs.size()<=reference_limit,"MTP golden capture pins or row count are invalid");
    const auto width=mul(owner.model().hidden_size,owner.model().hc_count),vocabulary=owner.model().vocab_size;
    require(width && width<=SIZE_MAX/2U && vocabulary && vocabulary<=UINT32_MAX && vocabulary<=SIZE_MAX/4U,
        "MTP golden capture geometry is invalid");
    HipMtpForwardGolden g;g.independent_source_content=source;g.independent_reference_identity=reference;
    g.target_content=owner.shared_target().target_model_content_identity;g.sidecar_content=owner.shared_target().sidecar_model_content_identity;
    g.structural_plan_identity=owner.plan().plan_identity;g.semantic_identity=owner.plan().forward_semantics.contract_identity;
    g.numerical_profile_identity=runtime.numerical_profile_identity();g.rows.resize(inputs.size());
    if(owner.model().rotary_identity().descriptor().mode()==RotaryMode::yarn) {
        const auto rotary=owner.model().rotary_identity();
        g.rotary_factor=rotary.descriptor().factor();g.rotary_descriptor_sha256=rotary.descriptor().sha256();
        g.rotary_table_sha256=rotary.table_sha256();g.context_capacity=owner.context_capacity();
        g.calibration_end=inputs.size();
    }
    for(std::size_t i=0;i<inputs.size();++i){const auto& input=inputs[i];
        require(input.hidden.valid() && input.hidden.same_producer(inputs[0].hidden) && input.hidden.source_position()==i &&
            input.hidden.input_token()<vocabulary && input.next_input_token<vocabulary && i+1U<owner.context_capacity() &&
            input.hidden.target_content_identity()==g.target_content && input.hidden.committed_session().next_position==i+1U &&
            input.hidden.committed_session().health==SerialSessionHealth::ready && !input.hidden.committed_session().arithmetic_lineage.tainted,
            "MTP golden capture target ticket is not its actual completed source row");}
    const auto stream=static_cast<hipStream_t>(owner.stream());
    try{
        for(std::size_t i=0;i<inputs.size();++i){auto& row=g.rows[i];row.target_hidden_bf16.resize(static_cast<std::size_t>(width));
            read(inputs[i].hidden.view(),row.target_hidden_bf16.data(),mul(width,2U),stream);}
        hip_check(hipStreamSynchronize(stream));
        for(std::size_t i=0;i<inputs.size();++i){auto& row=g.rows[i];
            if(safe_boundary) safe_boundary();
            const auto actual=runtime.execute_diagnostic(inputs[i].hidden,inputs[i].next_input_token);
            require(actual.device_completed && actual.completed_steps==owner.plan().steps.size() && actual.source_position==i &&
                actual.seed_position==i+1U && actual.numerical_profile_identity==g.numerical_profile_identity,
                "MTP golden capture publication is not the complete native row");
            row.source_position=i;row.source_input_token=inputs[i].hidden.input_token();row.next_input_token=inputs[i].next_input_token;
            row.next_hidden_bf16.resize(static_cast<std::size_t>(width));row.full_logits.resize(static_cast<std::size_t>(vocabulary));
            read(actual.wide_state,row.next_hidden_bf16.data(),mul(width,2U),stream);read(actual.logits,row.full_logits.data(),mul(vocabulary,4U),stream);
            hip_check(hipStreamSynchronize(stream));row.selected_token=firstmax(row.full_logits);
            require(row.selected_token==actual.token,"MTP device first-max differs from full CPU first-max");
            for(auto v:row.next_hidden_bf16)require(std::isfinite(decode(v)),"MTP native wide hidden is nonfinite");
            for(auto v:row.target_hidden_bf16)require(std::isfinite(decode(v)),"MTP target hidden is nonfinite");}
        return g;
    }catch(...){const_cast<HipMtpResources&>(owner).fail();throw;}
}
} // namespace strix
