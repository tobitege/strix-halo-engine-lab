cmake_minimum_required(VERSION 3.24)
project(strix_checkpoint_publication LANGUAGES CXX)
include(CTest)
find_package(Threads REQUIRED)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
# Independent host closure, actual production TUs, each compiled exactly once.
add_library(checkpoint_host STATIC
    "${REPO_ROOT}/src/session/checkpoint_io.cpp"
    "${REPO_ROOT}/src/session_state_cache.cpp"
    "${REPO_ROOT}/src/prefix_state_cache.cpp"
    "${REPO_ROOT}/src/model_content_hash.cpp"
    "${REPO_ROOT}/src/model_runtime_binding.cpp"
    "${REPO_ROOT}/src/rotary_descriptor.cpp"
    "${REPO_ROOT}/src/session/prefix_cache.cpp"
    "${REPO_ROOT}/src/session/kv.cpp"
    "${REPO_ROOT}/src/session/backend_cpu.cpp"
    "${REPO_ROOT}/src/minimax/session/session_cache.cpp"
    "${REPO_ROOT}/src/minimax/model_assets.cpp"
    "${REPO_ROOT}/src/minimax/model_description.cpp"
    "${REPO_ROOT}/src/model_description.cpp"
    "${REPO_ROOT}/src/serve/json.cpp"
    "${REPO_ROOT}/src/prompt_terminal_diagnostics.cpp")
target_include_directories(checkpoint_host PUBLIC "${REPO_ROOT}/include" "${REPO_ROOT}/src")
target_compile_features(checkpoint_host PUBLIC cxx_std_20)
target_link_libraries(checkpoint_host PUBLIC Threads::Threads)
if(MSVC)
    target_compile_options(checkpoint_host PUBLIC /EHsc /utf-8 /W4)
    target_compile_definitions(checkpoint_host PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    target_compile_options(checkpoint_host PUBLIC -Wall -Wextra -Wpedantic)
endif()
foreach(test session_checkpoint_io session_state_cache minimax_session_cache session_cache_owner)
    add_executable(${test}_tests "${REPO_ROOT}/tests/${test}_tests.cpp")
    target_link_libraries(${test}_tests PRIVATE checkpoint_host)
    add_test(NAME ${test} COMMAND ${test}_tests)
    set_tests_properties(${test} PROPERTIES TIMEOUT 120 LABELS "cpu;session")
endforeach()
if(NOT WIN32)
    target_sources(minimax_session_cache_tests PRIVATE "${REPO_ROOT}/src/minimax/engine/mm_pack.cpp")
endif()

set(CHECKPOINT_ENTRY_FIXTURES "" CACHE PATH "Private independently captured entry codec outputs")
if(CHECKPOINT_ENTRY_FIXTURES)
    add_test(NAME session_checkpoint_entry COMMAND session_checkpoint_io_tests "${CHECKPOINT_ENTRY_FIXTURES}")
    set_tests_properties(session_checkpoint_entry PROPERTIES TIMEOUT 120 LABELS "cpu;session")
endif()
