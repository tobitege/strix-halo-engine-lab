#include "strix/known_prompt_chunks.hpp"
#include "strix/known_prefill_limits.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
int failures=0;
void expect(bool condition,const char* text) {
    if(!condition) {std::cerr<<"FAIL: "<<text<<'\n';++failures;}
}
class Rows final : public strix::KnownPromptWindowBackend, public strix::SerialGreedyStepBackend {
public:
    std::uint32_t vocabulary=16, width=2;
    bool tiled=false;
    bool supports_tiled_prefill() const noexcept override {return tiled;}
    strix::KnownWindowOutputPolicy policy=strix::KnownWindowOutputPolicy::all_rows;
    strix::KnownWindowOutputPolicy output_policy() const noexcept override {return policy;}
    std::uint32_t logical_rows() const noexcept override {return width;}
    unsigned windows=0,serials=0;
    unsigned state_windows=0,state_serials=0;
    bool fail_partial_window=false,bad_extent=false,nonfinite=false,poisoned=false;
    std::vector<std::pair<std::uint64_t,std::uint32_t>> inputs;
    std::vector<float> values(std::uint32_t input) const {
        std::vector<float> result(vocabulary,-1.0F);result[(input+1U)%vocabulary]=1;
        if(bad_extent) result.pop_back();
        if(nonfinite) result[0]=std::numeric_limits<float>::infinity();
        return result;
    }
    std::vector<float> execute(std::uint32_t token,std::uint64_t position) override {
        ++serials;inputs.emplace_back(position,token);return values(token);
    }
    std::vector<float> execute_rows(std::span<const std::uint32_t> tokens,std::uint64_t position) override {
        if(policy==strix::KnownWindowOutputPolicy::state_only)throw std::logic_error("state-only mock cannot produce requested heads");
        if(tokens.size()!=width) throw std::invalid_argument("recording backend width mismatch");
        if(width==2U) return execute_window({tokens[0],tokens[1]},position);
        ++windows;
        for(std::size_t row=0;row<tokens.size();++row) {
            inputs.emplace_back(position+row,tokens[row]);
            if(fail_partial_window && row==1U) {poisoned=true;throw std::runtime_error("partial wide failure");}
        }
        return values(tokens.back());
    }
    std::vector<float> execute_window(std::array<std::uint32_t,2> tokens,std::uint64_t position) override {
        ++windows;inputs.emplace_back(position,tokens[0]);
        if(fail_partial_window) {poisoned=true;throw std::runtime_error("partial target window failure");}
        inputs.emplace_back(position+1U,tokens[1]);return values(tokens[1]);
    }
    void execute_state_rows(std::span<const std::uint32_t> tokens,std::uint64_t position) override {
        // A tiled backend also runs headless blocks (more prompt follows) under a head policy.
        const bool state_policy=policy==strix::KnownWindowOutputPolicy::state_only;
        if((!state_policy&&!tiled)||tokens.size()!=width)
            throw std::invalid_argument("mock state-only window policy/width differs");
        ++windows;if(state_policy) ++state_windows;
        for(std::size_t row=0;row<tokens.size();++row){inputs.emplace_back(position+row,tokens[row]);
            if(fail_partial_window&&row==1U){poisoned=true;throw std::runtime_error("partial state-only failure");}}
    }
    void execute_state(std::uint32_t token,std::uint64_t position) override {
        ++state_serials;inputs.emplace_back(position,token);
    }
};
// Tiled mock: complete blocks with or without head, and serial rows, recorded in order.
class Tiled final : public strix::KnownPromptWindowBackend {
public:
    Tiled(std::uint32_t rows,std::vector<std::pair<char,std::uint64_t>>& trace):width(rows),log(trace) {}
    std::uint32_t width;
    std::vector<std::pair<char,std::uint64_t>>& log;
    bool supports_tiled_prefill() const noexcept override {return true;}
    strix::KnownWindowOutputPolicy output_policy() const noexcept override {return strix::KnownWindowOutputPolicy::terminal_row;}
    std::uint32_t logical_rows() const noexcept override {return width;}
    std::vector<float> execute_window(std::array<std::uint32_t,2>,std::uint64_t) override {throw std::logic_error("no pairs");}
    std::vector<float> execute_rows(std::span<const std::uint32_t> tokens,std::uint64_t position) override {
        if(tokens.size()!=width) throw std::invalid_argument("tiled mock width mismatch");
        log.emplace_back(width==32U?'H':'h',position);
        std::vector<float> result(16,-1.0F);result[(tokens.back()+1U)%16U]=1;return result;
    }
    void execute_state_rows(std::span<const std::uint32_t> tokens,std::uint64_t position) override {
        if(tokens.size()!=width) throw std::invalid_argument("tiled mock width mismatch");
        log.emplace_back(width==32U?'S':'s',position);
    }
};
} // namespace
int main() {
    {
        Rows backend;
        backend.width=16;backend.tiled=true;backend.policy=strix::KnownWindowOutputPolicy::terminal_row;
        const std::vector<std::uint32_t> prompt(32,1U);
        strix::KnownPromptChunkRequest request{prompt,0,16,64,3,15};
        request.logical_rows=16;request.tiled_prefill=true;
        request.output_policy=strix::KnownWindowOutputPolicy::terminal_row;
        unsigned selections=0;
        request.emission_hook=[&](std::uint64_t step,std::uint64_t position,std::uint32_t argmax,std::span<const float> values) {
            expect(step==selections && position==31U+step && argmax==(step?5U:2U) && values[argmax]==1.0F,
                "hook sees terminal prefill and each decode row without intermediate prompt rows");
            ++selections;return step?15U:4U;
        };
        strix::KnownPromptChunkCursor cursor(request,backend,backend);
        while(cursor.advance()) {}
        const auto& result=cursor.finish_greedy();
        expect(result.generated_tokens==std::vector<std::uint32_t>({4,15}) && selections==2 &&
            backend.windows==2 && backend.serials==1 && backend.inputs.back()==std::pair<std::uint64_t,std::uint32_t>{32,4},
            "forcing preserves tiled prefill and consumes the emitted decode token");
    }
    {
        Rows backend;
        backend.width=16;backend.tiled=true;backend.policy=strix::KnownWindowOutputPolicy::terminal_row;
        const std::vector<std::uint32_t> prompt(32,1U);
        strix::KnownPromptChunkRequest request{prompt,0,16,64,3,15};
        request.logical_rows=16;request.tiled_prefill=true;
        request.output_policy=strix::KnownWindowOutputPolicy::terminal_row;
        unsigned selections=0;
        request.emission_hook=[&](std::uint64_t,std::uint64_t,std::uint32_t,std::span<const float>)->std::uint32_t {
            ++selections;
            throw std::runtime_error("capture write failed");
        };
        bool rejected=false;
        try {
            strix::KnownPromptChunkCursor cursor(request,backend,backend);
            while(cursor.advance()) {}
            (void)cursor.finish_greedy();
        } catch(const std::runtime_error&) { rejected=true; }
        expect(rejected && selections==1U && backend.serials==0U,
            "measurement failure propagates from terminal prefill before decode consumes a token");
    }
    // Same canonical pair/tail and existing generation policy, with no prompt
    // recomputation when the new cursor yields between chunks.
    for(const std::uint64_t start:{0U,2U}) for(const std::size_t count:{1U,2U,3U,4U,5U}) {
        const std::vector<std::uint32_t> all{1,2,3,4,5};
        const std::span<const std::uint32_t> prompt(all.data(),count);
        Rows ordinary,chunked;
        const auto expected=strix::execute_known_prompt_greedy_request({prompt,start,16,32,3,15,{}},ordinary,ordinary);
        strix::KnownPromptChunkCursor cursor({prompt,start,16,32,3,15},chunked,chunked);
        std::size_t completed=0;
        while(cursor.advance()) {
            ++completed;
            const auto trace=chunked.inputs;
            const auto frontier=cursor.frontier();
            const std::vector<float> retained(frontier.logits.begin(),frontier.logits.end());
            expect(frontier.valid && frontier.state_position==start+chunked.inputs.size() && !frontier.selection_emitted &&
                cursor.progress().emitted_tokens==0 && chunked.inputs==trace &&
                std::vector<float>(cursor.frontier().logits.begin(),cursor.frontier().logits.end())==retained,
                "yield retains completed logits and does not emit or recompute known inputs");
        }
        expect(completed==(count+1U)/2U && cursor.progress().windows_executed==count/2U &&
            cursor.progress().serial_tail_steps==count%2U,"each advance is one real pair or final odd row");
        const auto& actual=cursor.finish_greedy();
        expect(expected.generation && actual.generated_tokens==expected.generation->generated_tokens &&
            actual.final_position==expected.generation->final_position && actual.stop_reason==expected.generation->stop_reason &&
            chunked.inputs==ordinary.inputs,"chunked route delegates identical serial generation and token/input ordering");
        expect(cursor.progress().consumed_prompt_tokens==count && cursor.progress().consumed_generation_tokens==actual.steps_executed &&
            cursor.progress().emitted_tokens==actual.generated_tokens.size() &&
            cursor.progress().bonus_pending==actual.generated_tokens.back() &&
            cursor.frontier().unconsumed_selection==actual.generated_tokens.back(),
            "prompt inputs, generation inputs, emissions and unconsumed bonus remain distinct");
        expect(!actual.steps_executed || cursor.frontier().logits.empty(),"old prompt logits never masquerade as new generation frontier");
        bool rejected=false;
        try {(void)cursor.finish_greedy();} catch(const std::logic_error&) {rejected=true;}
        expect(rejected,"finished generation cannot execute or emit twice");
    }
    {
        Rows rows;const std::vector<std::uint32_t> prompt{1,2};
        strix::KnownPromptChunkCursor cursor({prompt,2,16,8,1,15},rows,rows);
        cursor.cancel();
        expect(!cursor.advance() && rows.inputs.empty() && cursor.progress().next_position==2 &&
            cursor.progress().cancelled && !cursor.frontier().valid,"cancel before first chunk neither consumes nor fabricates a frontier");
    }
    {
        Rows rows;const std::vector<std::uint32_t> prompt{1,2,3};
        strix::KnownPromptChunkCursor cursor({prompt,0,16,8,0,15},rows,rows);
        expect(cursor.advance() && cursor.progress().consumed_prompt_tokens==2,"first pair completes atomically");
        const auto retained=cursor.frontier().logits.data();cursor.cancel();
        expect(!cursor.advance() && rows.inputs.size()==2 && cursor.progress().cancelled &&
            cursor.frontier().logits.data()==retained,"cancel before odd tail is sticky and retains quiescent frontier storage");
        bool rejected=false;try {(void)cursor.finish_greedy();} catch(const std::logic_error&) {rejected=true;}
        expect(rejected && rows.inputs.size()==2,"cancelled prefill cannot start generation");
    }
    {
        Rows rows;const std::vector<std::uint32_t> prompt{2};
        strix::KnownPromptChunkCursor cursor({prompt,0,16,4,4,3},rows,rows);
        (void)cursor.advance();const auto& result=cursor.finish_greedy();
        expect(result.generated_tokens==std::vector<std::uint32_t>{3} && rows.inputs.size()==1 &&
            cursor.progress().bonus_pending==3U && cursor.progress().consumed_generation_tokens==0 &&
            result.stop_reason==strix::SerialRequestStopReason::eos,"first generated EOS remains emitted but unconsumed");
    }
    {
        Rows rows;const std::vector<std::uint32_t> prompt{1,2};
        strix::KnownPromptChunkCursor cursor({prompt,2,16,4,4,15},rows,rows);
        (void)cursor.advance();const auto& result=cursor.finish_greedy();
        expect(result.generated_tokens==std::vector<std::uint32_t>{3} && result.final_position==4 && rows.inputs.size()==2 &&
            result.stop_reason==strix::SerialRequestStopReason::context_capacity,"exact-capacity prompt emits pending selection without another input");
    }
    {
        Rows rows;const std::vector<std::uint32_t> prompt{1,15,2};
        strix::KnownPromptChunkCursor cursor({prompt,0,16,4,0,15},rows,rows);
        while(cursor.advance()) {}
        const auto& result=cursor.finish_greedy();
        expect(rows.inputs.size()==3 && result.generated_tokens.empty() && !cursor.progress().bonus_pending &&
            cursor.progress().next_position==3 && !cursor.frontier().logits.empty(),
            "interior known EOS and maxnew0 still consume complete prompt without publishing a bonus");
    }
    {
        const std::vector<std::uint32_t> good{1,2},bad{1,16};
        const std::vector<strix::KnownPromptChunkRequest> invalid{{{},0,16,8,1,15},{bad,0,16,8,1,15},
            {good,9,16,8,1,15},{good,7,16,8,1,15},{good,0,0,8,1,0},{good,0,16,8,1,16},
            {good,std::numeric_limits<std::uint64_t>::max()-1U,16,std::numeric_limits<std::uint64_t>::max(),1,15}};
        for(const auto& request:invalid) {
            Rows rows;bool rejected=false;
            try {strix::KnownPromptChunkCursor cursor(request,rows,rows);} catch(const std::invalid_argument&) {rejected=true;}
            expect(rejected && rows.inputs.empty(),"entire malformed suffix rejects before any target input");
        }
    }
    for(const unsigned failure:{0U,1U,2U}) {
        Rows rows;const std::vector<std::uint32_t> prompt{1,2};
        rows.fail_partial_window=failure==0;rows.bad_extent=failure==1;rows.nonfinite=failure==2;
        strix::KnownPromptChunkCursor cursor({prompt,0,16,8,1,15},rows,rows);
        bool threw=false;try {(void)cursor.advance();} catch(const std::exception&) {threw=true;}
        const auto calls=rows.windows;bool blocked=false;
        try {(void)cursor.advance();} catch(const std::logic_error&) {blocked=true;}
        expect(threw && blocked && cursor.progress().failed && rows.windows==calls && cursor.frontier().logits.empty() && !cursor.frontier().valid,
            "partial/native/output failure forbids resume and invalidates retained frontier without fake rollback");
        if(failure==0) expect(rows.poisoned && rows.inputs.size()==1 && cursor.progress().consumed_prompt_tokens==0,
            "failed-call counters do not pretend partially consumed opaque target state was restored");
    }
    for(const std::uint32_t width:{4U,8U}) {
        for(const std::size_t count:{1U,3U,4U,7U,8U,9U,17U}) {
            std::vector<std::uint32_t> prompt;
            for(std::size_t i=0;i<count;++i) prompt.push_back(static_cast<std::uint32_t>(i%14U)+1U);
            Rows target;target.width=width;
            strix::KnownPromptChunkCursor cursor({prompt,2,16,64,1,15,width},target,target);
            std::size_t chunks=0;
            while(cursor.advance()) {
                ++chunks;
                const auto saved=target.inputs;
                const auto frontier=cursor.frontier();
                expect(frontier.valid && !frontier.logits.empty() && frontier.state_position==2U+saved.size() &&
                    target.inputs==saved,"wide chunk pause keeps its full frontier and does not replay known tokens");
            }
            expect(target.windows==count/width && target.serials==count%width &&
                chunks==count/width+count%width,"wide cursor invokes only full configured R and explicit unpadded R1 suffix");
            for(std::size_t row=0;row<count;++row)
                expect(target.inputs[row]==std::pair<std::uint64_t,std::uint32_t>{2U+row,prompt[row]},
                    "wide and suffix calls preserve every absolute known position and token");
            const auto& result=cursor.finish_greedy();
            expect(result.steps_executed==0U && result.generated_tokens.size()==1U &&
                result.generated_tokens[0]==(prompt.back()+1U)%16U && target.inputs.size()==count,
                "wide frontier emits the terminal selection without consuming a predicted second token");
        }
        Rows mismatch;const std::vector<std::uint32_t> prompt(width,1U);
        bool rejected=false;
        try {strix::KnownPromptChunkCursor cursor({prompt,0,16,64,1,15,width},mismatch,mismatch);}
        catch(const std::invalid_argument&) {rejected=true;}
        expect(rejected && mismatch.inputs.empty(),"wide request rejects an R2-only backend before any input");
        Rows failing;failing.width=width;failing.fail_partial_window=true;
        strix::KnownPromptChunkCursor cursor({prompt,0,16,64,1,15,width},failing,failing);
        rejected=false;try {(void)cursor.advance();} catch(const std::runtime_error&) {rejected=true;}
        const auto count=failing.inputs.size();bool sticky=false;
        try {(void)cursor.advance();} catch(const std::logic_error&) {sticky=true;}
        expect(rejected && sticky && failing.poisoned && failing.inputs.size()==count &&
            cursor.progress().consumed_prompt_tokens==0U && !cursor.frontier().valid,
            "wide partial failure is sticky without pretending an opaque target rollback");
    }
    using Policy=strix::KnownWindowOutputPolicy;
    for(const unsigned width:{2U,4U,8U})for(const auto policy:{Policy::terminal_row,Policy::state_only}) {
        for(const unsigned count:{1U,width,width+1U,2U*width,2U*width+1U}) {
            std::vector<std::uint32_t> prompt;for(unsigned i=0;i<count;++i)prompt.push_back(i%12U+1U);
            Rows target;target.width=width;target.policy=policy;
            strix::KnownPromptChunkCursor cursor({prompt,3U,16U,64U,1U,15U,width,policy},target,target);
            unsigned advances=0U;
            while(cursor.advance()) {
                ++advances;const auto before=target.inputs;const auto frontier=cursor.frontier();
                expect(frontier.valid&&frontier.state_position==3U+target.inputs.size()&&target.inputs==before,
                    "output-policy yield retains real causal state without recomputing known inputs");
                if(policy==Policy::state_only&&!cursor.progress().prefill_complete)
                    expect(frontier.logits.empty()&&!frontier.unconsumed_selection&&cursor.progress().emitted_tokens==0U,
                        "state-only chunk never exposes stale logits or fabricates a generation frontier");
            }
            const auto state_windows=policy==Policy::state_only?(count-1U)/width:0U;
            const auto full_windows=policy==Policy::terminal_row?count/width:0U;
            const auto tail=count-(state_windows+full_windows)*width;
            expect(target.state_windows==state_windows&&target.windows==state_windows+full_windows&&
                target.state_serials==(policy==Policy::state_only?tail-1U:0U)&&
                target.serials==(policy==Policy::state_only?1U:tail)&&
                cursor.progress().state_only_windows==state_windows&&advances==state_windows+full_windows+tail,
                "state-only uses only nonterminal full windows and an explicit bounded R1 terminal tail");
            expect(!cursor.frontier().logits.empty()&&cursor.frontier().unconsumed_selection==(prompt.back()+1U)%16U,
                "every prompt length and divisibility ends at an actual full-output terminal frontier");
            for(unsigned row=0;row<count;++row)expect(target.inputs[row]==std::pair<std::uint64_t,std::uint32_t>{3U+row,prompt[row]},
                "head omission consumes every known token once at its original absolute position");
            const auto before=target.inputs;const auto& result=cursor.finish_greedy();
            expect(result.generated_tokens==std::vector<std::uint32_t>{(prompt.back()+1U)%16U}&&
                result.steps_executed==0U&&target.inputs==before&&cursor.progress().bonus_pending==result.generated_tokens.back(),
                "terminal selection remains emitted and unconsumed after state-only or terminal-row prefill");
        }
        Rows target;target.width=width;target.policy=policy;const std::vector<std::uint32_t> prompt(width+1U,1U);
        strix::KnownPromptChunkCursor cancelled({prompt,0U,16U,64U,1U,15U,width,policy},target,target);
        expect(cancelled.advance(),"first output-policy chunk completes before cancellation");const auto calls=target.inputs;
        cancelled.cancel();expect(!cancelled.advance()&&target.inputs==calls&&cancelled.frontier().valid,
            "cancel after output-policy chunk preserves state and never consumes its remaining terminal input");
        if(policy==Policy::state_only)expect(cancelled.frontier().logits.empty()&&!cancelled.frontier().unconsumed_selection,
            "cancelled state-only frontier remains honestly without logits or an emitted bonus");
    }
    {
        Rows target;target.policy=Policy::state_only;target.fail_partial_window=true;const std::vector<std::uint32_t> prompt{1U,2U,3U};
        strix::KnownPromptChunkCursor cursor({prompt,0U,16U,16U,1U,15U,2U,Policy::state_only},target,target);
        bool failed=false;try{(void)cursor.advance();}catch(const std::runtime_error&){failed=true;}
        const auto calls=target.inputs;bool sticky=false;try{(void)cursor.advance();}catch(const std::logic_error&){sticky=true;}
        expect(failed&&sticky&&target.poisoned&&calls.size()==2U&&target.inputs==calls&&!cursor.frontier().valid&&
            cursor.progress().consumed_prompt_tokens==0U,"state-only backend failure cannot resume or pretend a partial native call was rolled back");
    }
    for(const unsigned width:{2U,4U,8U}) for(const unsigned count:{1U,width,width+1U,2U*width}) {
        Rows target;target.width=width;target.policy=Policy::state_only;
        const std::vector<std::uint32_t> prompt(count,15U); // Known EOS is still input.
        strix::KnownPromptChunkCursor cursor({prompt,3U,16U,3U+count,0U,15U,width,Policy::state_only},target,target);
        while(cursor.advance()) {
            expect(cursor.frontier().valid && cursor.frontier().logits.empty() &&
                !cursor.frontier().unconsumed_selection && !cursor.frontier().selection_emitted,
                "state-only zero-emission request never creates a logit or selection frontier");
        }
        expect(target.windows==count/width && target.state_windows==count/width &&
            target.state_serials==count%width && target.serials==0U && target.inputs.size()==count,
            "zero-emission request uses every complete window and omits ALL serial heads");
        const auto before=target.inputs;const auto& result=cursor.finish_greedy();
        expect(result.generated_tokens.empty() && result.steps_executed==0U && result.final_position==3U+count &&
            result.stop_reason==strix::SerialRequestStopReason::max_new_tokens && target.inputs==before &&
            cursor.progress().generation_complete && !cursor.progress().bonus_pending,
            "zero-emission finish neither consumes a bonus nor runs a terminal model input");
        bool rejected=false;try{(void)cursor.finish_greedy();}catch(const std::logic_error&){rejected=true;}
        expect(rejected && target.inputs==before,"zero-emission finish remains one-shot");
    }
    for(const auto policy:{Policy::terminal_row,Policy::state_only,static_cast<Policy>(255U)}) {
        Rows target;const std::vector<std::uint32_t> prompt{1U,2U,3U};bool rejected=false;
        try{strix::KnownPromptChunkCursor cursor({prompt,0U,16U,16U,1U,15U,2U,policy},target,target);}
        catch(const std::invalid_argument&){rejected=true;}
        expect(rejected&&target.inputs.empty(),"unknown or mismatched output-policy backend rejects before consuming any input");
    }
    for(const unsigned width:{16U,17U,32U,1024U,2048U,4096U,8192U,16384U})
    for(const unsigned count:{1U,width,width+1U,2U*width})
    for(const auto policy:{Policy::all_rows,Policy::terminal_row,Policy::state_only}) {
        Rows target;target.tiled=true;target.width=width;target.policy=policy;
        std::vector<std::uint32_t> prompt;for(unsigned row=0;row<count;++row)prompt.push_back(row%14U+1U);
        const auto budget=policy==Policy::state_only?0U:1U;
        strix::KnownPromptChunkCursor cursor({prompt,5U,16U,5U+count,budget,15U,width,policy,true},target,target);
        while(cursor.advance()) {}
        expect(target.windows==count/width && target.inputs.size()==count &&
            cursor.progress().serial_tail_steps==count%width && cursor.progress().consumed_prompt_tokens==count,
            "explicit T prefill uses only complete unpadded windows and exact serial suffixes");
        for(unsigned row=0;row<count;++row)
            expect(target.inputs[row]==std::pair<std::uint64_t,std::uint32_t>{5U+row,prompt[row]},
                "T prefill cursor preserves each absolute input/position without window replay");
        const auto before=target.inputs;const auto& result=cursor.finish_greedy();
        expect(result.final_position==5U+count && result.steps_executed==0U && target.inputs==before,
            "tiled terminal frontier never consumes an emitted bonus at exact capacity");
        if(policy==Policy::state_only)
            expect(result.generated_tokens.empty() && cursor.frontier().logits.empty() &&
                !cursor.frontier().unconsumed_selection && target.serials==0U,
                "state-only T zero-emission prefill has no fabricated logit or selection");
        else expect(result.generated_tokens==std::vector<std::uint32_t>{(prompt.back()+1U)%16U},
            "T full/terminal output emits the actual final known-input selection once");
    }
    for(const unsigned invalid:{0U,1U,2U,3U,4U,5U}) {
        const std::vector<std::uint32_t> prompt{1U,2U,3U};Rows target;target.tiled=true;target.width=16U;
        strix::KnownPromptChunkRequest request{prompt,0U,16U,64U,1U,15U,16U,Policy::terminal_row,true};
        if(invalid==0U) target.tiled=false;
        if(invalid==1U) request.tiled_prefill=false;
        if(invalid==2U) request.logical_rows=target.width=15U;
        if(invalid==3U) request.logical_rows=target.width=strix::known_prefill_max_rows+1U;
        if(invalid==4U) request.logical_rows=target.width=8U;
        if(invalid==5U) request.output_policy=target.policy=Policy::state_only;
        else target.policy=Policy::terminal_row;
        bool rejected=false;try{strix::KnownPromptChunkCursor cursor(request,target,target);}
        catch(const std::invalid_argument&){rejected=true;}
        expect(rejected && target.inputs.empty(),
            "unsupported/nonexplicit/oversized/state-only-generation T rejects before target work");
    }
    if(!failures) std::cout<<"known prompt chunk tests passed\n";
    {
        // Tail backend: complete T=32 blocks, then complete 16-row tail blocks, then serial rows.
        std::vector<std::pair<char,std::uint64_t>> trace;
        Tiled main_blocks(32,trace),tail_blocks(16,trace);Rows serial;
        std::vector<std::uint32_t> prompt(55);for(std::uint32_t i=0;i<55U;++i) prompt[i]=i%15U;
        strix::KnownPromptChunkRequest request{prompt,0,16,64,2,15};
        request.logical_rows=32;request.tiled_prefill=true;request.output_policy=strix::KnownWindowOutputPolicy::terminal_row;
        strix::KnownPromptChunkCursor cursor(request,main_blocks,serial);
        cursor.set_tail_backend(tail_blocks);
        while(cursor.advance()) {}
        const bool headless=[]{const char* v=std::getenv("STRIX_PREFILL_HEADLESS");return !v||v[0]!='0';}();
        const std::vector<std::pair<char,std::uint64_t>> expected{{headless?'S':'H',0},{headless?'s':'h',32}};
        expect(trace==expected && serial.serials==7U && cursor.progress().windows_executed==2U &&
            cursor.progress().tail_windows==1U && cursor.progress().serial_tail_steps==7U,
            "tail backend runs complete tail blocks headless before the serial rows");
        (void)cursor.finish_greedy();
        {   // Pipeline frames come from the cold plan: T block + tail block; the 7 serial rows are not frames.
            const auto before=trace.size();
            strix::KnownPromptChunkCursor planned(request,main_blocks,serial);planned.set_tail_backend(tail_blocks);
            const auto plan=strix::known_prompt_pipeline_plan(planned.cold_plan());
            expect(plan.units==2U && plan.rows==48U && plan.serial_rows==7U &&
                plan.unit_rows==std::vector<std::uint64_t>{32U,16U} && trace.size()==before,
                "pipeline plan counts tiled units only and leaves serial rows out");
            std::vector<std::uint32_t> on_grain(48,3U);
            strix::KnownPromptChunkRequest grain{on_grain,0,16,64,1,15};
            grain.logical_rows=32;grain.tiled_prefill=true;grain.output_policy=strix::KnownWindowOutputPolicy::terminal_row;
            strix::KnownPromptChunkCursor exact_plan(grain,main_blocks,serial);exact_plan.set_tail_backend(tail_blocks);
            const auto whole=strix::known_prompt_pipeline_plan(exact_plan.cold_plan());
            expect(whole.units==2U && whole.serial_rows==0U && whole.rows==48U && whole.identity==plan.identity,
                "a prefix on the 16-row grain is frames only; the same frames give the same plan identity");
        }
        // A prompt ending exactly on a tail block keeps that block's head as the terminal frontier.
        trace.clear();
        std::vector<std::uint32_t> exact(48,1U);
        strix::KnownPromptChunkRequest second{exact,0,16,64,1,15};
        second.logical_rows=32;second.tiled_prefill=true;second.output_policy=strix::KnownWindowOutputPolicy::terminal_row;
        strix::KnownPromptChunkCursor ends(second,main_blocks,serial);ends.set_tail_backend(tail_blocks);
        while(ends.advance()) {}
        const std::vector<std::pair<char,std::uint64_t>> exact_trace{{headless?'S':'H',0},{'h',32}};
        expect(trace==exact_trace && ends.frontier().unconsumed_selection.has_value(),"final tail block produces the terminal head");
        // Two tail levels: the largest tail that fits, then the next.
        trace.clear();
        std::vector<std::uint32_t> two(109,2U);
        strix::KnownPromptChunkRequest third{two,0,16,128,1,15};
        third.logical_rows=64;third.tiled_prefill=true;third.output_policy=strix::KnownWindowOutputPolicy::terminal_row;
        Tiled main64(64,trace),tail24(24,trace);Rows serial_two;
        strix::KnownPromptChunkCursor levels(third,main64,serial_two);
        levels.set_tail_backend(tail24);levels.set_tail_backend(tail_blocks);
        while(levels.advance()) {}
        const std::vector<std::pair<char,std::uint64_t>> two_trace{{headless?'s':'h',0},{headless?'s':'h',64},{headless?'s':'h',88}};
        expect(trace==two_trace && serial_two.serials==5U && levels.progress().tail_windows==2U,"tails are tried largest first");
        if(trace!=two_trace) for(const auto& [kind,position]:trace) std::cerr<<kind<<position<<' ';
        bool order_rejected=false;
        try {strix::KnownPromptChunkCursor bad_order(third,main64,serial_two);bad_order.set_tail_backend(tail_blocks);bad_order.set_tail_backend(tail24);}
        catch(const std::invalid_argument&) {order_rejected=true;}
        expect(order_rejected,"tails must be descending");
        bool rejected=false;
        try {Tiled wide(32,trace);strix::KnownPromptChunkCursor bad(second,main_blocks,serial);bad.set_tail_backend(wide);}
        catch(const std::invalid_argument&) {rejected=true;}
        expect(rejected,"a tail backend must be smaller than T");
    }
    return failures?1:0;
}
