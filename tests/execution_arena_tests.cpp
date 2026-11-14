#include "strix/execution_arena.hpp"
#include "../src/core/row_layout.hpp"
#include "strix/ctx_exact_memory.hpp"
#include "strix/indexed_attention_workspace.hpp"

#include "strix/model_contract.hpp"
#include "strix/known_prefill_limits.hpp"

#include <algorithm>
#include <limits>

#include <iostream>
#include <string_view>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    const auto model =
        strix::ModelContract::qwen38_flash_next();
    const auto execution =
        strix::build_serial_decode_execution_plan(model);
    const auto io =
        strix::build_serial_execution_io_plan(execution);

    strix::ExecutionBufferRequest request;
    request.logical_rows = 1;

    const auto sizes =
        strix::build_serial_execution_buffer_plan(
            model, request);

    const auto exact =
        strix::build_serial_execution_arena_plan(
            model,
            request,
            execution,
            io,
            1);

    expect(exact.total_bytes ==
               sizes.peak_with_materialized_logits_bytes,
           "liveness allocator reproduces the analytical serial peak");
    expect(exact.total_bytes <
               exact.naive_nonaliased_bytes,
           "liveness allocator aliases dead buffer generations");

    const auto aligned =
        strix::build_serial_execution_arena_plan(
            model,
            request,
            execution,
            io,
            256);
    expect(aligned.total_bytes >= exact.total_bytes &&
               aligned.total_bytes % 256ULL == 0,
           "aligned arena rounds the physical allocation safely");
    expect(aligned.total_bytes <
               aligned.naive_nonaliased_bytes,
           "aligned arena still reuses dead lifetimes");




    const auto indexed_it =
        std::find_if(
            io.steps.begin(),
            io.steps.end(),
            [](const auto& step) {
                return step.op ==
                    strix::SerialOp::indexed_attention;
            });
    expect(
        indexed_it != io.steps.end() &&
            indexed_it->temporaries.size() == 1,
        "arena fixture exposes indexed-attention temporary");
    if (indexed_it != io.steps.end() &&
        !indexed_it->temporaries.empty()) {
        const auto* temporary =
            strix::find_arena_allocation(
                aligned,
                indexed_it->temporaries[0]);
        expect(
            temporary != nullptr &&
                temporary->first_step ==
                    temporary->last_step,
            "indexed-attention temporary has one-step lifetime");
        expect(
            temporary != nullptr &&
                temporary->bytes ==
                    sizes.indexed_attention_workspace_bytes,
            "indexed-attention temporary uses analytical workspace size");
    }

    const auto ple_it =
        std::find_if(
            io.steps.begin(),
            io.steps.end(),
            [](const auto& step) {
                return step.op ==
                    strix::SerialOp::ple;
            });
    expect(
        ple_it != io.steps.end() &&
            ple_it->temporaries.size() == 1,
        "arena fixture exposes PLE temporary");
    if (ple_it != io.steps.end() &&
        !ple_it->temporaries.empty()) {
        const auto* temporary =
            strix::find_arena_allocation(
                aligned,
                ple_it->temporaries[0]);
        expect(
            temporary != nullptr &&
                temporary->first_step ==
                    temporary->last_step,
            "PLE temporary has one-step lifetime");
        expect(
            temporary != nullptr &&
                temporary->bytes ==
                    sizes.ple_workspace_bytes,
            "PLE temporary uses canonical workspace size");
    }

    const auto shared_it =
        std::find_if(
            io.steps.begin(),
            io.steps.end(),
            [](const auto& step) {
                return step.op ==
                    strix::SerialOp::shared_expert;
            });
    expect(
        shared_it != io.steps.end() &&
            shared_it->temporaries.size() == 1,
        "arena fixture exposes shared-expert temporary");
    if (shared_it != io.steps.end() &&
        !shared_it->temporaries.empty()) {
        const auto* temporary =
            strix::find_arena_allocation(
                aligned,
                shared_it->temporaries[0]);
        expect(
            temporary != nullptr &&
                temporary->first_step ==
                    temporary->last_step,
            "temporary workspace has one-step lifetime");
        expect(
            temporary != nullptr &&
                temporary->bytes ==
                    sizes.mlp_intermediate_bytes,
            "temporary workspace uses analytical MLP size");
    }

    const auto& lm_io = io.steps.back();
    expect(lm_io.op == strix::SerialOp::lm_head &&
               lm_io.reads.size() == 1 &&
               lm_io.writes.size() == 1,
           "LM head IO fixture is simple");

    const auto* final_hidden =
        strix::find_arena_allocation(
            aligned,
            lm_io.reads[0]);
    const auto* logits =
        strix::find_arena_allocation(
            aligned,
            lm_io.writes[0]);
    expect(final_hidden != nullptr &&
               logits != nullptr,
           "arena resolves versioned LM-head buffers");
    if (final_hidden != nullptr &&
        logits != nullptr) {
        const auto hidden_end =
            final_hidden->offset +
            final_hidden->bytes;
        const auto logits_end =
            logits->offset +
            logits->bytes;
        expect(hidden_end <= logits->offset ||
                   logits_end <= final_hidden->offset,
               "LM-head input and output do not alias while simultaneously live");
    }

    request.logical_rows = 8;
    const auto rows_eight =
        strix::build_serial_execution_arena_plan(
            model,
            request,
            execution,
            io,
            256);
    expect(rows_eight.total_bytes >
               aligned.total_bytes,
           "arena scales with logical row class");

    bool invalid_alignment = false;
    try {
        static_cast<void>(
            strix::build_serial_execution_arena_plan(
                model,
                request,
                execution,
                io,
                0));
    } catch (const std::invalid_argument&) {
        invalid_alignment = true;
    }
    expect(invalid_alignment,
           "zero arena alignment is rejected");


    const auto window = strix::build_known_window_arena(aligned);
    expect(window.row_stride.size() == window.physical.allocations.size(),
        "known window retains a stride for every actual serial allocation");
    for (std::size_t index = 0; index < window.physical.allocations.size(); ++index) {
        const auto& physical = window.physical.allocations[index];
        const auto& first = window.row_view[0].allocations[index];
        const auto& second = window.row_view[1].allocations[index];
        const auto* original = strix::find_arena_allocation(aligned, physical.use);
        expect(original && first.bytes == original->bytes && second.bytes == original->bytes &&
            second.offset == first.offset + window.row_stride[index] &&
            first.offset + first.bytes <= second.offset &&
            second.offset + second.bytes <= physical.offset + physical.bytes,
            "known window preserves exact R1 payloads in disjoint aligned row slices");
        expect(window.row_view[0].total_bytes == window.physical.total_bytes &&
            window.row_view[1].total_bytes == window.physical.total_bytes,
            "row views bind the complete physical allocation extent");
        for (std::size_t previous = 0; previous < index; ++previous) {
            const auto& other = window.physical.allocations[previous];
            if (physical.first_step <= other.last_step && other.first_step <= physical.last_step)
                expect(physical.offset + physical.bytes <= other.offset ||
                    other.offset + other.bytes <= physical.offset,
                    "known window never aliases live two-row allocation regions");
        }
    }
    auto duplicate = aligned; duplicate.allocations.push_back(duplicate.allocations.front());
    bool rejected_window = false;
    try { (void)strix::build_known_window_arena(duplicate); }
    catch (const std::invalid_argument&) { rejected_window = true; }
    expect(rejected_window, "known window rejects duplicated serial uses");
    auto malformed = aligned; malformed.allocations.front().first_step = malformed.allocations.front().last_step + 1U;
    rejected_window = false;
    try { (void)strix::build_known_window_arena(malformed); }
    catch (const std::invalid_argument&) { rejected_window = true; }
    expect(rejected_window, "known window rejects malformed source lifetimes");
    strix::ExecutionArenaPlan huge;
    huge.alignment = 1; huge.total_bytes = std::numeric_limits<std::uint64_t>::max();
    auto huge_allocation = aligned.allocations.front();
    huge_allocation.offset = 0; huge_allocation.bytes = std::numeric_limits<std::uint64_t>::max() / 2U + 1U;
    huge.allocations.push_back(huge_allocation);
    rejected_window = false;
    try { (void)strix::build_known_window_arena(huge,1); }
    catch (const std::overflow_error&) { rejected_window = true; }
    expect(rejected_window, "known window rejects doubled row extent overflow");

    {
        // The scoped index answers every lookup as the linear search does: first allocation of a
        // duplicated use, nullptr for an absent one, and a plan that grows inside the scope.
        auto lookup = aligned; lookup.allocations.push_back(lookup.allocations.front());
        auto absent = lookup.allocations.front().use; absent.generation = std::numeric_limits<std::uint64_t>::max();
        std::vector<const strix::ExecutionArenaAllocation*> linear;
        for (const auto& allocation : lookup.allocations) linear.push_back(strix::find_arena_allocation(lookup, allocation.use));
        bool same = strix::find_arena_allocation(lookup, absent) == nullptr && linear.back() == &lookup.allocations.front();
        {
            const strix::ArenaLookupScope scope;
            for (std::size_t i = 0; i < lookup.allocations.size(); ++i)
                same = same && strix::find_arena_allocation(lookup, lookup.allocations[i].use) == linear[i];
            same = same && strix::find_arena_allocation(lookup, absent) == nullptr;
            auto grown = lookup.allocations.back(); grown.use = absent;
            lookup.allocations.push_back(grown);
            same = same && strix::find_arena_allocation(lookup, absent) == &lookup.allocations.back();
        }
        expect(same, "scoped arena lookup index matches the linear search");

        // A persistent owner index answers its plans like the search in every scope opened with it;
        // a plan whose storage, size or indexed hit changed falls back to the search, and plans
        // outside the index are indexed per scope.
        lookup.allocations.pop_back();
        std::vector<const strix::ExecutionArenaAllocation*> expected;
        for (const auto& allocation : lookup.allocations) expected.push_back(strix::find_arena_allocation(lookup, allocation.use));
        auto other = aligned;
        const strix::ExecutionArenaPlan* const owned[] = {&lookup};
        const strix::ArenaLookupIndex index(owned);
        bool persistent = expected.back() == &lookup.allocations.front();
        for (int round = 0; round < 2; ++round) {
            const strix::ArenaLookupScope scope(index);
            for (std::size_t i = 0; i < lookup.allocations.size(); ++i)
                persistent = persistent && strix::find_arena_allocation(lookup, lookup.allocations[i].use) == expected[i];
            persistent = persistent && strix::find_arena_allocation(lookup, absent) == nullptr;
            for (const auto& allocation : other.allocations)
                persistent = persistent && strix::find_arena_allocation(other, allocation.use) == &allocation;
        }
        {
            const strix::ArenaLookupScope scope(index);
            const auto duplicated = lookup.allocations.front().use;
            lookup.allocations.front().use = absent; // indexed hit no longer holds its use
            persistent = persistent && strix::find_arena_allocation(lookup, duplicated) == &lookup.allocations.back();
            lookup.allocations.front().use = duplicated;
            auto grown = lookup.allocations.back(); grown.use = absent;
            lookup.allocations.push_back(grown); // size (and maybe storage) changed
            persistent = persistent && strix::find_arena_allocation(lookup, absent) == &lookup.allocations.back() &&
                strix::find_arena_allocation(lookup, duplicated) == &lookup.allocations.front();
        }
        expect(persistent, "persistent arena lookup index matches the linear search");
    }

    request.logical_rows=1;request.parallel_moe_router=true;
    const auto fast_io=strix::build_serial_execution_io_plan(execution,true);
    request.q4k_a8g32_gate_up=true;request.grouped_exact_q51_down=true;request.q51_slot_parallel_down=true;
    request.a8_down_q51=true;request.a8_down_q80=true;request.q5k_a8g32_gate_up=true;request.dense_a8=true;
    for(auto maximum:{262144ULL,524288ULL,1048576ULL}) {
        auto geometry_model=model;geometry_model.max_position_embeddings=maximum;
        const auto full=strix::build_indexed_attention_workspace_layout(geometry_model);
        const auto compact=strix::build_indexed_attention_workspace_layout(geometry_model,256,false,1,true);
        const auto source=strix::build_serial_execution_arena_plan(geometry_model,request,execution,fast_io,256);
        expect(full.total_bytes-compact.total_bytes==full.block_scores_bytes,"score-free layout removes the score allocation");
        for(auto rows:{16U,32U,64U,128U,256U,512U,1024U,2048U,4096U,8192U}) {
            const auto old=strix::build_known_prefill_arena(source,256,rows);
            const auto off=strix::build_known_prefill_arena(source,256,rows,0);
            const auto ring=strix::build_known_prefill_arena(source,256,rows,full.block_scores_bytes);
            expect(old.physical.total_bytes==off.physical.total_bytes && old.row_stride==off.row_stride,
                "score ring off retains original bytes and strides");
            expect(ring.physical.total_bytes==998400ULL*rows,"logits still bound score-free uniform arena at every maximum");
            const auto terminal=strix::build_known_prefill_arena(source,256,rows,full.block_scores_bytes,true);
            const auto legacy_terminal=strix::build_known_prefill_arena(source,256,rows,0,true);
            // The capacity port describes the tight layout; the shared row layout policy may only add its step to
            // the strides that were page multiples (two stream slots and the MLP intermediate here).
            constexpr strix::core::RowLayoutPolicy tight_rows{4096, 0, 2, 16};
            const auto tight_terminal=strix::build_known_prefill_arena(source,256,rows,full.block_scores_bytes,true,true,0,&tight_rows);
            expect(tight_terminal.physical.total_bytes==324352ULL*rows+993280ULL,"score-free headless arena matches independent capacity port");
            expect(terminal.physical.total_bytes>=tight_terminal.physical.total_bytes &&
                terminal.physical.total_bytes<=(324352ULL+3U*256U)*rows+993280ULL,"row layout policy adds at most its step per moved stride");
            if(rows==8192U) std::cout<<"headless arena bytes/row tight "<<(tight_terminal.physical.total_bytes-993280ULL)/rows
                <<" policy "<<(terminal.physical.total_bytes-993280ULL)/rows<<'\n';
            expect(terminal.row_stride.back()==0 && terminal.row_view.front().allocations.back().bytes==0 &&
                terminal.row_view.back().allocations.back().bytes==993280U,"only the terminal row exposes the single logits stripe");
            const auto lookup=strix::ArenaLookupIndex::uniform_views(terminal.row_view);
            const strix::ArenaLookupScope scope(lookup);
            for(const auto& view:terminal.row_view) {
                const auto* row_logits=strix::find_arena_allocation(view,terminal.physical.allocations.back().use);
                expect(row_logits && row_logits->bytes==view.allocations.back().bytes,"mixed extents retain correct indexed row lookup");
            }
            std::cout<<"I12 M="<<maximum<<" T="<<rows<<" arena="<<terminal.physical.total_bytes
                <<" terminal_only="<<legacy_terminal.physical.total_bytes<<" stripe="<<993280U<<'\n';
            std::cout<<"I1 M="<<maximum<<" T="<<rows<<" old="<<old.physical.total_bytes
                <<" score_free="<<ring.physical.total_bytes<<" ring="<<strix::QsaScoreRingGeometry(maximum).total_bytes()<<'\n';
        }
    }
    for(auto capacity:{64U,128U,256U,512U,1024U}) for(auto rows:{1U,16U,63U,64U,65U,128U,129U,8192U,16384U}) {
        strix::QsaScoreRingLifetime ring;
        std::array<bool,2> recorded{},completed{};
        std::uint32_t group=0,waits=0;
        for(std::uint32_t first=0;first<rows;first+=capacity,++group) {
            const auto slot=ring.acquire([&](std::uint32_t i){
                expect(recorded[i]&&!completed[i],"slot wait observes consumer event pending after enqueue return");
                completed[i]=true;++waits;
            });
            expect(slot==group%2U && (!recorded[slot]||completed[slot]),"no reuse before consumer completion, including wraparound");
            expect(std::min(capacity,rows-first)>0U,"last partial group has bounded live rows");
            recorded[slot]=false;completed[slot]=false;
            ring.consumed(slot,[&](std::uint32_t i){recorded[i]=true;});
        }
        expect(waits==(group>2U?group-2U:0U),"two slot ring waits exactly on wrapped consumers");
    }
    {
        strix::QsaScoreRingLifetime ring;
        const auto first=ring.acquire([](std::uint32_t){throw std::logic_error("unexpected wait");});
        ring.consumed(first,[](std::uint32_t){});
        const auto second=ring.acquire([](std::uint32_t){throw std::logic_error("unexpected wait");});
        ring.consumed(second,[](std::uint32_t){});
        bool refused=false;
        try {(void)ring.acquire([](std::uint32_t){throw std::runtime_error("consumer event is not complete");});}
        catch(const std::runtime_error&){refused=true;}
        expect(refused,"enqueue return cannot release a pending slot");
        const auto resumed=ring.acquire([&](std::uint32_t slot){expect(slot==first,"failed wait preserves the pending consumer lease");});
        expect(resumed==first,"slot becomes reusable only after its consumer wait succeeds");
        ring.consumed(resumed,[](std::uint32_t){});
    }
    for(auto item:{"STRIX_QSA_SCORE_RING","STRIX_TERMINAL_LOGITS_ARENA"}) {
        expect(strix::ctx_exact_memory_enabled([](const char*)->const char*{return nullptr;},item),"exact memory defaults on");
        expect(!strix::ctx_exact_memory_enabled([](const char*)->const char*{return "0";},item),"exact memory off switches restore legacy policy");
        expect(!strix::ctx_exact_memory_enabled([](const char* name)->const char*{
            return std::string_view(name)=="STRIX_CTX_EXACT_MEMORY"?"0":nullptr;},item),"master off disables both independent policies");
        expect(!strix::ctx_exact_memory_enabled([&](const char* name)->const char*{
            return std::string_view(name)==item?"0":nullptr;},item),"individual off switch disables its policy");
        expect(!strix::ctx_exact_memory_enabled([](const char* name)->const char*{
            return std::string_view(name)=="STRIX_QPF2K_DEFAULTS"?"0":nullptr;},item),"STRIX_QPF2K_DEFAULTS=0 disables compact memory");
        expect(!strix::ctx_exact_memory_enabled([](const char* name)->const char*{
            return std::string_view(name)=="STRIX_BASELINE"?"1":nullptr;},item),"STRIX_BASELINE=1 disables compact memory");
        strix::qpf2k::baseline=true;
        expect(!strix::ctx_exact_memory_enabled([](const char*)->const char*{return nullptr;},item),"--baseline disables compact memory");
        strix::qpf2k::baseline=false;
        {
            const strix::qpf2k::ModelScope gsq(true);
            expect(!strix::ctx_exact_memory_enabled([](const char*)->const char*{return nullptr;},item),"GSQ checkpoints keep the legacy layout");
        }
        expect(strix::ctx_exact_memory_enabled([](const char*)->const char*{return nullptr;},item),"policy restored after the suppressions");
    }
    {
        const auto large = strix::build_known_prefill_arena(aligned, aligned.alignment, 16384U);
        expect(large.row_view.size() == 16384U && large.physical.total_bytes > UINT32_MAX,
            "T16384 arena preserves 64-bit extents beyond 4 GiB");
        for (std::size_t i = 0; i < aligned.allocations.size(); ++i) {
            const auto& last = large.row_view.back().allocations[i];
            const auto& physical = large.physical.allocations[i];
            expect(last.offset == physical.offset + 16383U * large.row_stride[i] &&
                last.offset + last.bytes <= physical.offset + physical.bytes && last.bytes == aligned.allocations[i].bytes,
                "T16384 final row keeps exact serial payload and aligned stripe bounds");
        }
        for (const auto rows : {16385U,32768U}) {
            bool rejected = false;
            try { (void)strix::build_known_prefill_arena(aligned, aligned.alignment, rows); }
            catch (const std::invalid_argument&) { rejected = true; }
            expect(rejected, "arena rejects rows beyond the T16384 capacity contract");
        }
    }
    if (failures == 0) {
        std::cout << "execution arena tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
