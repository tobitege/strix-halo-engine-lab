# Test-only observation of the real queue. No production target includes this.
include_guard(GLOBAL)

function(strix_replay_count text needle result)
    set(rest "${text}")
    set(count 0)
    string(LENGTH "${needle}" size)
    while(TRUE)
        string(FIND "${rest}" "${needle}" position)
        if(position EQUAL -1)
            break()
        endif()
        math(EXPR count "${count}+1")
        math(EXPR position "${position}+${size}")
        string(SUBSTRING "${rest}" ${position} -1 rest)
    endwhile()
    set(${result} ${count} PARENT_SCOPE)
endfunction()

function(strix_replay_observe_queue source result)
    strix_replay_count("${source}" "ready_.notify_one();\n    return ticket;" submissions)
    strix_replay_count("${source}" "return ticket->publish(tokens) && keep;" current)
    strix_replay_count("${source}" "return ticket->publish(tokens);" archived)
    strix_replay_count("${source}" "ticket->publish(tokens)" publications)
    if(NOT submissions EQUAL 1 OR
       NOT ((current EQUAL 1 AND archived EQUAL 0 AND publications EQUAL 1) OR
            (current EQUAL 0 AND archived EQUAL 2 AND publications EQUAL 2)))
        message(FATAL_ERROR "Replay observation seam changed: submissions=${submissions}, current=${current}, archived=${archived}, publications=${publications}")
    endif()
    if(source MATCHES "minimax_replay_(submitted|tokens|publish)")
        message(FATAL_ERROR "Replay queue is already instrumented")
    endif()
    # Both archived lambdas call the same helper; the current live parser still
    # runs first, publishes even on parser stop, and returns publish && keep.
    string(REPLACE "ticket->publish(tokens)" "minimax_replay_publish(ticket, tokens)" source "${source}")
    string(REPLACE "ready_.notify_one();\n    return ticket;"
        "ready_.notify_one();\n    minimax_replay_submitted();\n    return ticket;" source "${source}")
    # Match the existing FakeClock, without changing queue layout or scheduling.
    string(REPLACE "std::chrono::steady_clock::now()" "GenerationQueue::test_now()" source "${source}")
    set(preamble [=[#include "strix/serve/engine.hpp"
void minimax_replay_submitted();
void minimax_replay_tokens(strix::serve::GenerationTicket*, std::span<const std::uint32_t>);
namespace {
bool minimax_replay_publish(const std::shared_ptr<strix::serve::GenerationTicket>& ticket,
    std::span<const std::uint32_t> tokens) {
    minimax_replay_tokens(ticket.get(), tokens);
    return ticket->publish(tokens);
}
}
]=])
    set(${result} "${preamble}${source}" PARENT_SCOPE)
endfunction()
