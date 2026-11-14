cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(strix_minimax_session_shared_host LANGUAGES CXX)
    include(CTest)
endif()
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
option(MINIMAX_SESSION_BASELINE "Capture from independently frozen original sources" OFF)
# Both paths verify fixture integrity; only the independent baseline build can
# contain capture code. Reproduction output must never replace expectations.
file(READ "${CMAKE_CURRENT_LIST_DIR}/fixtures/baseline.json" frozen_manifest)
string(JSON fixture_count LENGTH "${frozen_manifest}" fixture_sha256)
math(EXPR fixture_last "${fixture_count}-1")
foreach(i RANGE ${fixture_last})
    string(JSON fixture MEMBER "${frozen_manifest}" fixture_sha256 ${i})
    string(JSON expected GET "${frozen_manifest}" fixture_sha256 "${fixture}")
    file(SHA256 "${CMAKE_CURRENT_LIST_DIR}/fixtures/${fixture}" actual)
    if(NOT actual STREQUAL expected)
        message(FATAL_ERROR "Frozen fixture hash mismatch: ${fixture}")
    endif()
endforeach()
set(MINIMAX_BASELINE_SOURCE_ROOT "${REPO_ROOT}/build/minimax-session-baseline-p5/original" CACHE PATH "Original source snapshot")
# The pre-migration source closure is an optional local archive. With it, the original scheduler is compiled
# beside the current one; without it, the current sources are still checked against the frozen fixtures.
if(EXISTS "${MINIMAX_BASELINE_SOURCE_ROOT}/src/serve/engine.cpp")
    set(MINIMAX_ORIGINAL_SOURCES ON)
else()
    set(MINIMAX_ORIGINAL_SOURCES OFF)
    if(MINIMAX_SESSION_BASELINE)
        message(FATAL_ERROR "MINIMAX_SESSION_BASELINE needs the archived sources at MINIMAX_BASELINE_SOURCE_ROOT")
    endif()
    message(STATUS "MiniMax archived original sources absent: original-vs-current targets skipped")
endif()
if(MINIMAX_SESSION_BASELINE)
    set(MM_SOURCE_ROOT "${MINIMAX_BASELINE_SOURCE_ROOT}")
    file(READ "${CMAKE_CURRENT_LIST_DIR}/fixtures/baseline.json" baseline)
    string(JSON count LENGTH "${baseline}" sha256_raw)
    math(EXPR last "${count}-1")
    foreach(i RANGE ${last})
        string(JSON source MEMBER "${baseline}" sha256_raw ${i})
        string(JSON expected GET "${baseline}" sha256_raw "${source}")
        file(SHA256 "${MM_SOURCE_ROOT}/${source}" actual)
        if(NOT actual STREQUAL expected)
            message(FATAL_ERROR "Original source hash mismatch: ${source}")
        endif()
    endforeach()
else()
    set(MM_SOURCE_ROOT "${REPO_ROOT}")
endif()
# These three literal test-only seams observe, without replacing, production
# queue publication/submission and the actual production Lead test scenarios.
# Generated copies live only in the authorized build tree. Neither build can
# change a production scheduling or API operation.
file(READ "${MM_SOURCE_ROOT}/tests/minimax_lead_tests.cpp" lead_test)
string(REPLACE "return {engine.controls, queue.work};" "minimax_replay_scenario(*this);\n        return {engine.controls, queue.work};" lead_test "${lead_test}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/lead_fixture.inc" "${lead_test}")
file(READ "${MM_SOURCE_ROOT}/src/serve/engine.cpp" queue_source)
include("${REPO_ROOT}/cmake/StrixReplayObservation.cmake")
strix_replay_count("${lead_test}" "minimax_replay_scenario(*this);" scenario_hooks)
if(NOT scenario_hooks EQUAL 1)
    message(FATAL_ERROR "Replay scenario observation seam changed: ${scenario_hooks}")
endif()
strix_replay_observe_queue("${queue_source}" queue_fixture)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/queue_fixture.cpp" "${queue_fixture}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${MM_SOURCE_ROOT}/src/serve/engine.cpp" "${MM_SOURCE_ROOT}/tests/minimax_lead_tests.cpp")
if(MINIMAX_ORIGINAL_SOURCES)
add_test(NAME minimax-replay-observation
    COMMAND "${CMAKE_COMMAND}" "-DREPO_ROOT=${REPO_ROOT}"
        "-DARCHIVED_ROOT=${MINIMAX_BASELINE_SOURCE_ROOT}"
        "-DWORK=${CMAKE_CURRENT_BINARY_DIR}/observation-regression"
        -P "${CMAKE_CURRENT_LIST_DIR}/replay_observation_tests.cmake")
set_tests_properties(minimax-replay-observation PROPERTIES TIMEOUT 30 LABELS "cpu;session;minimax")
execute_process(COMMAND "${CMAKE_COMMAND}" "-DREPO_ROOT=${REPO_ROOT}"
    "-DARCHIVED_ROOT=${MINIMAX_BASELINE_SOURCE_ROOT}"
    "-DWORK=${CMAKE_CURRENT_BINARY_DIR}/observation-regression"
    -P "${CMAKE_CURRENT_LIST_DIR}/replay_observation_tests.cmake"
    RESULT_VARIABLE observation_status)
if(NOT observation_status EQUAL 0)
    message(FATAL_ERROR "Replay observation regression failed at configure time")
endif()
endif()
find_package(Threads REQUIRED)
add_library(minimax_replay_family STATIC
    "${CMAKE_CURRENT_BINARY_DIR}/queue_fixture.cpp"
    "${MM_SOURCE_ROOT}/src/serve/json.cpp"
    "${MM_SOURCE_ROOT}/src/serve/http_server.cpp"
    "${MM_SOURCE_ROOT}/src/serve/deployment.cpp"
    "${MM_SOURCE_ROOT}/src/serve/request_fields.cpp"
    "${MM_SOURCE_ROOT}/src/serve/json_constraint.cpp"
    "${MM_SOURCE_ROOT}/src/serve/bpe_tokenizer.cpp"
    "${MM_SOURCE_ROOT}/src/serve/chat_template.cpp"
    "${MM_SOURCE_ROOT}/src/minimax/serve/minimax_api.cpp"
    "${MM_SOURCE_ROOT}/src/minimax/serve/minimax_chat.cpp"
    "${MM_SOURCE_ROOT}/src/minimax/serve/minimax_tokenizer.cpp"
    "${MM_SOURCE_ROOT}/src/minimax/serve/minimax_observe.cpp"
    "${MM_SOURCE_ROOT}/src/minimax/dspark/minimax_dspark_host.cpp"
    "${MM_SOURCE_ROOT}/src/lookup/index.cpp")
target_include_directories(minimax_replay_family PUBLIC "${MM_SOURCE_ROOT}/include" "${MM_SOURCE_ROOT}/src")
target_compile_features(minimax_replay_family PUBLIC cxx_std_20)
target_compile_definitions(minimax_replay_family PUBLIC STRIX_SERVE_QUEUE_TEST=1)
target_link_libraries(minimax_replay_family PUBLIC Threads::Threads)
if(WIN32)
    target_compile_definitions(minimax_replay_family PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
    target_link_libraries(minimax_replay_family PUBLIC ws2_32)
endif()
if(MSVC)
    target_compile_options(minimax_replay_family PUBLIC /EHsc /utf-8)
    target_compile_definitions(minimax_replay_family PUBLIC __builtin_rintf=rintf)
endif()
add_executable(minimax_session_replay replay_tests.cpp)
target_link_libraries(minimax_session_replay PRIVATE minimax_replay_family)
target_include_directories(minimax_session_replay PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(minimax_session_replay PRIVATE MM_DIAGNOSTICS="${CMAKE_CURRENT_BINARY_DIR}" MM_FIXTURES="${CMAKE_CURRENT_LIST_DIR}/fixtures"
    $<$<BOOL:${MINIMAX_SESSION_BASELINE}>:MM_CAPTURE_BASELINE=1>)
add_test(NAME minimax-session-replay COMMAND minimax_session_replay)
set_tests_properties(minimax-session-replay PROPERTIES TIMEOUT 120 LABELS "cpu;session;minimax")

if(NOT MINIMAX_SESSION_BASELINE)
    include("${REPO_ROOT}/cmake/StrixStateCache.cmake")
    target_link_libraries(minimax_replay_family PUBLIC strix_state_cache strix_session_owner)
    target_sources(minimax_replay_family PRIVATE
        "${REPO_ROOT}/src/serve/api.cpp"
        "${REPO_ROOT}/src/serve/media.cpp"
        "${REPO_ROOT}/src/serve/constraint_selection.cpp"
        "${REPO_ROOT}/src/serve/deepseek_chat.cpp"
        "${REPO_ROOT}/src/serve/glm_chat.cpp"
        "${REPO_ROOT}/src/serve/health.cpp"
        "${REPO_ROOT}/src/preflight.cpp"
        "${REPO_ROOT}/src/linux_uma_observation.cpp"
        "${REPO_ROOT}/src/uma_owner_inventory.cpp"
        "${REPO_ROOT}/src/minimax/serve/session_engine.cpp")
endif()

# Compile the archived original source/header closure, including its own queue.
# Never describe a current-source translation unit as an original comparison.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/capture_noop.cpp"
    "#include \"strix/serve/engine.hpp\"\nvoid minimax_replay_submitted() {}\nvoid minimax_replay_tokens(strix::serve::GenerationTicket*,std::span<const std::uint32_t>) {}\n")
if(MINIMAX_ORIGINAL_SOURCES)
file(READ "${CMAKE_CURRENT_LIST_DIR}/fixtures/baseline.json" baseline)
string(JSON original_count LENGTH "${baseline}" sha256_raw)
math(EXPR original_last "${original_count}-1")
foreach(i RANGE ${original_last})
    string(JSON source MEMBER "${baseline}" sha256_raw ${i})
    string(JSON expected GET "${baseline}" sha256_raw "${source}")
    file(SHA256 "${MINIMAX_BASELINE_SOURCE_ROOT}/${source}" actual)
    if(NOT actual STREQUAL expected)
        message(FATAL_ERROR "Original source hash mismatch: ${source}")
    endif()
endforeach()
add_library(minimax_original_family STATIC)
foreach(source serve/engine serve/json serve/http_server serve/deployment serve/request_fields
        serve/json_constraint serve/bpe_tokenizer serve/chat_template
        minimax/serve/minimax_api minimax/serve/minimax_chat minimax/serve/minimax_tokenizer
        minimax/serve/minimax_observe minimax/dspark/minimax_dspark_host lookup/index)
    target_sources(minimax_original_family PRIVATE "${MINIMAX_BASELINE_SOURCE_ROOT}/src/${source}.cpp")
endforeach()
target_include_directories(minimax_original_family PUBLIC
    "${MINIMAX_BASELINE_SOURCE_ROOT}/include" "${MINIMAX_BASELINE_SOURCE_ROOT}/src")
target_compile_features(minimax_original_family PUBLIC cxx_std_20)
target_compile_definitions(minimax_original_family PUBLIC STRIX_SERVE_QUEUE_TEST=1)
target_link_libraries(minimax_original_family PUBLIC Threads::Threads)
if(WIN32)
    target_compile_definitions(minimax_original_family PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
    target_link_libraries(minimax_original_family PUBLIC ws2_32)
endif()
if(MSVC)
    target_compile_options(minimax_original_family PUBLIC /EHsc /utf-8 /UNDEBUG)
    target_compile_definitions(minimax_original_family PUBLIC __builtin_rintf=rintf)
    target_compile_options(minimax_replay_family PUBLIC /UNDEBUG)
else()
    target_compile_options(minimax_original_family PUBLIC -UNDEBUG)
    target_compile_options(minimax_replay_family PUBLIC -UNDEBUG)
endif()
add_executable(minimax_original_lead_tests "${MINIMAX_BASELINE_SOURCE_ROOT}/tests/minimax_lead_tests.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/capture_noop.cpp")
target_link_libraries(minimax_original_lead_tests PRIVATE minimax_original_family)
add_test(NAME minimax-original-lead COMMAND minimax_original_lead_tests)
endif()

if(NOT MINIMAX_SESSION_BASELINE)
    add_executable(minimax_paired_state_tests paired_state_tests.cpp "${CMAKE_CURRENT_BINARY_DIR}/capture_noop.cpp")
    target_link_libraries(minimax_paired_state_tests PRIVATE minimax_replay_family)
    add_test(NAME minimax-paired-state COMMAND minimax_paired_state_tests)
    set_tests_properties(minimax-paired-state PROPERTIES TIMEOUT 90 LABELS "cpu;session;minimax")
    add_executable(minimax_session_owner session_tests.cpp "${CMAKE_CURRENT_BINARY_DIR}/capture_noop.cpp")
    target_link_libraries(minimax_session_owner PRIVATE minimax_replay_family)
    add_test(NAME minimax-session-owner COMMAND minimax_session_owner)
    set_tests_properties(minimax-session-owner PROPERTIES TIMEOUT 120 LABELS "cpu;session;minimax")
    add_library(minimax_owner_cache STATIC
        "${REPO_ROOT}/src/minimax/session/session_cache.cpp"
        "${REPO_ROOT}/src/minimax/model_assets.cpp"
        "${REPO_ROOT}/src/minimax/model_description.cpp"
        "${REPO_ROOT}/src/model_description.cpp"
        )
    target_link_libraries(minimax_owner_cache PUBLIC minimax_replay_family)
    # Reuse the existing model-content SHA provider for state transport checks.
    # No second codec/hash implementation or duplicate production source owner.
    target_link_libraries(minimax_paired_state_tests PRIVATE minimax_owner_cache)
    add_executable(minimax_owner_cache_tests "${REPO_ROOT}/tests/minimax_session_cache_tests.cpp"
        "${CMAKE_CURRENT_BINARY_DIR}/capture_noop.cpp")
    target_link_libraries(minimax_owner_cache_tests PRIVATE minimax_owner_cache)
    if(NOT WIN32)
        target_sources(minimax_owner_cache_tests PRIVATE "${REPO_ROOT}/src/minimax/engine/mm_pack.cpp")
    endif()
    add_test(NAME minimax-owner-cache COMMAND minimax_owner_cache_tests)
    set_tests_properties(minimax-owner-cache PROPERTIES TIMEOUT 180 LABELS "cpu;session;minimax")
    add_executable(minimax_owner_api_tests "${REPO_ROOT}/tests/minimax_session_api_tests.cpp"
        "${CMAKE_CURRENT_BINARY_DIR}/capture_noop.cpp")
    target_link_libraries(minimax_owner_api_tests PRIVATE minimax_replay_family)
    add_test(NAME minimax-owner-api COMMAND minimax_owner_api_tests)
endif()

if(NOT MINIMAX_SESSION_BASELINE)
    if(COMMAND strix_session_add_minimax_continuation)
        strix_session_add_minimax_continuation()
    else()
        add_executable(minimax_continuation_tests continuation_tests.cpp "${CMAKE_CURRENT_BINARY_DIR}/capture_noop.cpp")
        target_link_libraries(minimax_continuation_tests PRIVATE minimax_replay_family minimax_owner_cache)
        add_test(NAME minimax-continuation COMMAND minimax_continuation_tests)
        set_tests_properties(minimax-continuation PROPERTIES TIMEOUT 90 LABELS "cpu;session;minimax")
    endif()
endif()

if(NOT MINIMAX_SESSION_BASELINE)
    if(COMMAND strix_session_add_minimax_prefix_reuse)
        strix_session_add_minimax_prefix_reuse()
    else()
        add_executable(minimax_prefix_reuse_tests prefix_reuse_tests.cpp "${CMAKE_CURRENT_BINARY_DIR}/capture_noop.cpp")
        target_link_libraries(minimax_prefix_reuse_tests PRIVATE minimax_replay_family minimax_owner_cache)
        add_test(NAME minimax-prefix-reuse COMMAND minimax_prefix_reuse_tests)
        set_tests_properties(minimax-prefix-reuse PROPERTIES TIMEOUT 90 LABELS "cpu;session;minimax")
    endif()
endif()

if(NOT MINIMAX_SESSION_BASELINE)
    if(NOT TARGET strix_session)
        include("${REPO_ROOT}/cmake/StrixSession.cmake")
    endif()
    if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/generated/strix/build_info.hpp")
        set(STRIX_BUILD_JSON "{\"format\":\"strix-build\",\"revision\":1,\"status\":\"unavailable\"}")
        configure_file("${REPO_ROOT}/include/strix/build_info.hpp.in"
            "${CMAKE_CURRENT_BINARY_DIR}/generated/strix/build_info.hpp" @ONLY)
    endif()
    target_include_directories(minimax_replay_family PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/shared_api_tests.cpp" AND NOT TARGET minimax_shared_api_tests)
        add_executable(minimax_shared_api_tests shared_api_tests.cpp "${CMAKE_CURRENT_BINARY_DIR}/capture_noop.cpp")
        target_link_libraries(minimax_shared_api_tests PRIVATE minimax_replay_family minimax_owner_cache)
        add_test(NAME minimax-shared-api COMMAND minimax_shared_api_tests)
        set_tests_properties(minimax-shared-api PROPERTIES TIMEOUT 120 LABELS "cpu;session;minimax;api")
    endif()
endif()
