#include "strix/distributed_greedy_control.hpp"
#include <deque>
#include <iostream>
#include <stdexcept>
#include <utility>

using namespace strix;
namespace {
int failures=0;
void expect(bool good,const char* why) {if(!good) {++failures;std::cerr<<why<<'\n';}}
template<class F> bool rejects(F&& action) {try {action();} catch(const std::exception&) {return true;}return false;}
DistributedGreedyRequestIdentity request(std::uint64_t maximum=2) {
    DistributedGreedyRequestIdentity r;r.model_content_identity.bytes.fill(1);r.prompt_sha256.bytes.fill(2);
    r.serial_execution_identity=11;r.kernel_plan_identity=12;r.placement_plan_identity=13;
    r.request_id=21;r.connection_epoch=22;r.initial_binding_identity=31;r.initial_generation=41;
    r.prompt_tokens=1;r.vocab_size=8;r.context_capacity=4;r.max_new_tokens=maximum;
    r.max_target_inputs=maximum?maximum:1;r.eos_token_id=7;r.boundary_dtype_id=8;return r;
}
DistributedGreedyMessage first(DistributedGreedyControlState& state,const DistributedGreedyRequestIdentity& r) {
    auto m=state.make_message(DistributedGreedyMessageKind::input_command,0);
    m.generation=r.initial_generation;m.binding_identity=r.initial_binding_identity;
    m.history_identity=r.initial_history_identity;m.input_token=1;m.output_intent=DistributedGreedyOutputIntent::finite_firstmax;return m;
}
DistributedGreedyMessage result(DistributedGreedyControlState& state,const DistributedGreedyMessage& input,
    std::uint32_t selected=3,std::uint32_t bits=0x80000000U) {
    auto m=state.make_message(DistributedGreedyMessageKind::target_result,1);
    m.generation=input.generation;m.binding_identity=input.binding_identity;m.decision_identity=77+input.position;
    m.position=input.position+1;m.committed_tokens=input.committed_tokens+1;m.history_identity=88+input.position;
    m.output_intent=input.output_intent;
    if(input.output_intent==DistributedGreedyOutputIntent::finite_firstmax) {
        m.selected_present=true;m.selected_token=selected;m.selected_logit_bits=bits;
    }
    return m;
}
DistributedGreedyMessage finish(DistributedGreedyControlState& state,const DistributedGreedyMessage& last,std::uint64_t emissions) {
    auto m=state.make_message(DistributedGreedyMessageKind::finish,0);
    m.generation=last.generation;m.binding_identity=last.binding_identity;m.decision_identity=last.decision_identity;
    m.position=last.position;m.committed_tokens=last.committed_tokens;m.history_identity=last.history_identity;
    m.emitted_tokens=emissions;m.pending_present=emissions!=0;m.pending_token=emissions?last.selected_token:0;
    m.finish_reason=DistributedGreedyFinishReason::max_new_tokens;return m;
}
class Stream final:public DistributedControlByteStream {
public:
    std::deque<std::vector<std::byte>> incoming,sent;
    bool fail_send=false;unsigned sends=0;
    void send(std::span<const std::byte> bytes) override {++sends;
        if(fail_send) throw std::runtime_error("injected control loss");
        sent.emplace_back(bytes.begin(),bytes.end());}
    std::vector<std::byte> receive() override {if(incoming.empty()) throw std::runtime_error("missing frame");auto b=std::move(incoming.front());incoming.pop_front();return b;}
};
}
int main() {
    auto r=request();validate_distributed_greedy_request_identity(r);
    for(unsigned field=0;field<6;++field) {
        auto bad=r;
        if(!field) bad.prompt_sha256={};
        if(field==1) bad.max_target_inputs=0;
        if(field==2) bad.max_target_inputs=1;
        if(field==3) bad.prompt_tokens=5;
        if(field==4) bad.eos_token_id=8;
        if(field==5) bad.initial_generation=UINT64_MAX;
        expect(rejects([&]{validate_distributed_greedy_request_identity(bad);}),"request identity/budget/overflow admission rejects before targets");
    }
    {
        DistributedGreedyControlState state(r);auto input=first(state,r);
        const auto encoded=encode_distributed_greedy_message(input);const auto decoded=decode_distributed_greedy_message(encoded);
        expect(encode_distributed_greedy_message(decoded)==encoded && inspect_distributed_control_frame_kind(encoded)==DistributedControlFrameKind::greedy_control,
            "fixed payloadv1 roundtrips in dedicated outerkind6");
        auto short_frame=encoded;short_frame.pop_back();expect(rejects([&]{(void)decode_distributed_greedy_message(short_frame);}),"truncation rejects");
        auto extra=encoded;extra.push_back(std::byte{});expect(rejects([&]{(void)decode_distributed_greedy_message(extra);}),"trailing bytes reject");
        auto reserved=encoded;reserved[10]=std::byte{1};expect(rejects([&]{(void)decode_distributed_greedy_message(reserved);}),"nonzero outer reserved bits reject");
        auto kind5=encoded;kind5[8]=std::byte{5};expect(rejects([&]{(void)decode_distributed_greedy_message(kind5);}),"greedy never reinterprets kind5 commits");
        auto digest=encoded;digest[29]=std::byte{1};expect(rejects([&]{(void)decode_distributed_greedy_message(digest);}),"digest byte encoding is canonical");
        auto boolean=encoded;boolean[16U+348U]=std::byte{2};
        expect(rejects([&]{(void)decode_distributed_greedy_message(boolean);}),"payloadv1 boolean cannot carry integer2");
        expect(rejects([&]{(void)encode_distributed_greedy_message(input,{2,65537,64});}),"greedy frame never increases64KiB limit");
    }
    {
        DistributedGreedyControlState state(r);auto input=first(state,r);state.observe(input);
        auto reply=result(state,input);state.observe(reply);
        auto next=state.make_message(DistributedGreedyMessageKind::input_command,0);
        next.generation=reply.generation+1;next.binding_identity=32;next.position=reply.position;
        next.committed_tokens=reply.committed_tokens;next.history_identity=reply.history_identity;
        next.previous_binding_identity=reply.binding_identity;next.previous_decision_identity=reply.decision_identity;
        next.input_token=reply.selected_token;next.output_intent=DistributedGreedyOutputIntent::finite_firstmax;
        state.observe(next);auto second=result(state,next);state.observe(second);
        auto stopped=finish(state,second,2);state.observe(stopped);
        auto confirmed=stopped;confirmed.kind=DistributedGreedyMessageKind::finished;confirmed.sender_node=1;confirmed.receiver_node=0;confirmed.sequence=3;
        state.observe(confirmed);expect(state.finished() && !state.failed(),"two targets and pending-last-token finish without consuming bonus");
        expect(rejects([&]{state.observe(next);}),"finished transcript cannot submit another pending token");
    }
    for(unsigned bad=0;bad<5;++bad) {
        DistributedGreedyControlState state(r);auto input=first(state,r);state.observe(input);auto reply=result(state,input);
        if(!bad) ++reply.sequence;
        if(bad==1) ++reply.binding_identity;
        if(bad==2) ++reply.generation;
        if(bad==3) reply.output_intent=DistributedGreedyOutputIntent::state_only;
        if(bad==4) {++reply.request.request_id;reply.request_identity=distributed_greedy_request_identity(reply.request);}
        expect(rejects([&]{state.observe(reply);}) && state.failed(),"wrong result sequence/binding/generation/intent/request is permanently failed");
        expect(rejects([&]{state.observe(input);}),"uncertain result does not permit retry");
    }
    for(const auto bits:{0x00000000U,0x80000000U,0x00000001U,0x80000001U,0x7f7fffffU}) {
        DistributedGreedyControlState state(r);auto input=first(state,r);state.observe(input);auto reply=result(state,input,0,bits);
        validate_distributed_greedy_message(reply);expect(decode_distributed_greedy_message(encode_distributed_greedy_message(reply)).selected_token==0,
            "finite signedzero/subnormal/extreme bits and successful token0 remain exact");
    }
    for(const auto bits:{0x7f800000U,0xff800000U,0x7fc00001U,0x7f800001U}) {
        DistributedGreedyControlState state(r);auto input=first(state,r);state.observe(input);auto reply=result(state,input,0,bits);
        expect(rejects([&]{validate_distributed_greedy_message(reply);}),"Inf/qNaN/sNaN cannot be a successful finite reply");
    }
    {
        const auto zero=request(0);DistributedGreedyControlState state(zero);auto input=first(state,zero);
        state.observe(input);auto reply=result(state,input);state.observe(reply);
        auto stopped=finish(state,reply,0);state.observe(stopped);
        auto ack=stopped;ack.kind=DistributedGreedyMessageKind::finished;ack.sender_node=1;ack.receiver_node=0;ack.sequence=2;
        state.observe(ack);expect(state.finished() && stopped.position==zero.prompt_tokens && !stopped.pending_present,
            "maxnew0 consumes full prefix and discards final selection without emitted/pending tokens");
    }
    {
        DistributedGreedyControlState state(request(1));auto input=first(state,request(1));state.observe(input);
        auto reply=result(state,input,7,0x3f800000U);state.observe(reply);auto stopped=finish(state,reply,1);
        stopped.finish_reason=DistributedGreedyFinishReason::eos;state.observe(stopped);
        auto wrong=stopped;wrong.kind=DistributedGreedyMessageKind::finished;wrong.sender_node=1;wrong.receiver_node=0;wrong.sequence=2;wrong.pending_token=6;
        expect(rejects([&]{state.observe(wrong);}),"finish ACK cannot change pending EOS or consumed frontier");
    }
    {
        DistributedTokenBinding binding;binding.serial_execution_identity=r.serial_execution_identity;
        binding.kernel_plan_identity=r.kernel_plan_identity;binding.placement_plan_identity=r.placement_plan_identity;
        binding.model_content_identity=r.model_content_identity;binding.request_id=r.request_id;
        binding.connection_epoch=r.connection_epoch;binding.generation=r.initial_generation;binding.boundary_dtype_id=r.boundary_dtype_id;
        binding.input_token=1;auto actual=r;actual.initial_binding_identity=distributed_token_binding_identity(binding);
        DistributedGreedyControlState state(actual);auto input=first(state,actual);state.observe(input);auto reply=result(state,input);
        DistributedTokenCommit commit;commit.binding_identity=reply.binding_identity;commit.decision_identity=reply.decision_identity;
        commit.next_position=reply.position;commit.committed_tokens=reply.committed_tokens;commit.history_identity=reply.history_identity;
        validate_distributed_greedy_target_commit(reply,binding,commit);
        ++commit.decision_identity;
        expect(rejects([&]{validate_distributed_greedy_target_commit(reply,binding,commit);}),
            "reply metadata must match caller's actual decision, not merely a finite scalar");
    }
    {
        DistributedGreedyControlState state(r);auto input=first(state,r);state.observe(input);auto reply=result(state,input);state.observe(reply);
        auto next=state.make_message(DistributedGreedyMessageKind::input_command,0);
        next.generation=reply.generation+1;next.binding_identity=32;next.position=reply.position;
        next.committed_tokens=reply.committed_tokens;next.history_identity=reply.history_identity;
        next.previous_binding_identity=reply.binding_identity;next.previous_decision_identity=reply.decision_identity;
        next.input_token=4;next.output_intent=DistributedGreedyOutputIntent::finite_firstmax;
        expect(rejects([&]{state.observe(next);}),"next input cannot consume a token other than the accepted pending result");
    }
    {
        DistributedGreedyControlState state(r);auto input=first(state,r);state.observe(input);auto reply=result(state,input);state.observe(reply);
        auto cancelled=finish(state,reply,0);cancelled.finish_reason=DistributedGreedyFinishReason::cancelled;
        state.observe(cancelled);expect(!state.failed(),"post-submit cancellation can retain consumed state without emitting a bonus");
    }
    {
        Stream stream;FramedDistributedGreedyWire wire(stream,r,0);DistributedGreedyControlState source(r);auto input=first(source,r);
        // No actual network/device is exercised; injected stream loss is sticky.
        stream.fail_send=true;expect(rejects([&]{wire.send(input);}) && wire.failed(),"lost command is uncertain, never silently retried");
        expect(rejects([&]{wire.send(input);}) && stream.sends==1,"failed wire does not send a second command");
    }
    {
        DistributedGreedyControlState state(r);auto error=state.make_message(DistributedGreedyMessageKind::failure,1);
        error.failure=DistributedGreedyFailure::uncertain;state.observe(error);
        expect(state.failed(),"explicit uncertain error is separate from successful token0");
        error.selected_present=true;expect(rejects([&]{validate_distributed_greedy_message(error);}),"failure cannot smuggle successful selection");
    }
    {
        auto prefix=request(1);prefix.prompt_tokens=3;prefix.max_target_inputs=3;
        DistributedGreedyControlState state(prefix);auto input=first(state,prefix);
        expect(rejects([&]{state.validate(input);}),"early known-prefix input cannot request selected-output intent");
        input.output_intent=DistributedGreedyOutputIntent::state_only;state.observe(input);
        auto reply=result(state,input);state.observe(reply);
        auto premature=finish(state,reply,1);
        expect(rejects([&]{state.validate(premature);}),"prefix-only completed state cannot manufacture an emitted token");
    }
    {
        DistributedGreedyControlState state(r);auto input=first(state,r);state.observe(input);
        auto reply=result(state,input);state.observe(reply);auto inflated=finish(state,reply,2);
        expect(rejects([&]{state.validate(inflated);}),"finish count cannot exceed completed selected targets");
    }
    {
        class PrefixBackend final:public SerialGreedyStepBackend {
        public:
            std::uint64_t calls=0;
            std::vector<float> execute(std::uint32_t,std::uint64_t position) override {
                expect(position==calls,"zero-emission cursor still consumes positions causally");++calls;
                return {0.0F,1.0F,2.0F,3.0F,4.0F,5.0F,6.0F,7.0F};
            }
        } backend;
        const std::array<std::uint32_t,3> prompt{1,2,3};
        const auto zero=execute_serial_greedy_token_request({prompt,8,4,0,7},backend);
        expect(backend.calls==3 && zero.steps_executed==3 && zero.final_position==3 && zero.generated_tokens.empty(),
            "source-authoritative zero-emission serial policy consumes full prefix without bonus");
    }
    {
        Stream coordinator_stream,terminal_stream;
        FramedDistributedGreedyWire coordinator(coordinator_stream,r,0),terminal(terminal_stream,r,1);
        DistributedGreedyControlState builder(r);const auto input=first(builder,r);coordinator.send(input);
        terminal_stream.incoming.push_back(coordinator_stream.sent.back());const auto received=terminal.receive();
        builder.observe(received);const auto reply=result(builder,received);terminal.send(reply);
        coordinator_stream.incoming.push_back(terminal_stream.sent.back());
        const auto observed=coordinator.receive();
        expect(observed.selected_token==reply.selected_token && observed.decision_identity==reply.decision_identity &&
            !coordinator.failed() && !terminal.failed(),"dedicated duplex kind6 preserves one command/result transcript");
    }
    return failures?1:0;
}
