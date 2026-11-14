cmake_minimum_required(VERSION 3.24)
project(strix_minimax_shared_api_host LANGUAGES CXX)
include(CTest)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
# Reuse the real shared runtime and the production family/service closure.
# Register the independent project's bounded host selection explicitly.
set(BUILD_TESTING OFF)
add_subdirectory("${REPO_ROOT}/tests/session_runtime" shared)
set(BUILD_TESTING ON)
foreach(test runtime http_replay registry turn_cache)
    add_executable(session_${test}_r7 "${REPO_ROOT}/tests/session_${test}_tests.cpp")
    target_link_libraries(session_${test}_r7 PRIVATE strix_session_shell)
    add_test(NAME session_${test} COMMAND session_${test}_r7 "${REPO_ROOT}")
    set_tests_properties(session_${test} PROPERTIES TIMEOUT 120)
endforeach()
set(_strix_session_root "${REPO_ROOT}")
    add_library(strix_lookup STATIC "${REPO_ROOT}/src/lookup/index.cpp")
    target_include_directories(strix_lookup PUBLIC "${REPO_ROOT}/include")
    target_compile_features(strix_lookup PUBLIC cxx_std_20)
    add_library(minimax_service_host STATIC
        "${REPO_ROOT}/src/minimax/engine/mm_defaults.cpp"
        "${REPO_ROOT}/src/minimax/model_description.cpp"
        "${REPO_ROOT}/src/minimax/model_graph.cpp"
        "${REPO_ROOT}/src/model_graph/model_graph.cpp"
        "${REPO_ROOT}/src/model_graph/plan.cpp"
        "${REPO_ROOT}/src/minimax/engine/deployment.cpp"
        "${REPO_ROOT}/src/deploy/device.cpp"
        "${REPO_ROOT}/src/deploy/spec.cpp"
        "${REPO_ROOT}/src/deploy/plan.cpp"
        "${REPO_ROOT}/src/minimax/serve/minimax_api.cpp"
        "${REPO_ROOT}/src/minimax/serve/minimax_observe.cpp"
        "${REPO_ROOT}/src/minimax/serve/minimax_chat.cpp"
        "${REPO_ROOT}/src/minimax/serve/minimax_tokenizer.cpp")
    target_include_directories(minimax_service_host PUBLIC "${REPO_ROOT}/include" "${REPO_ROOT}/src")
    target_compile_features(minimax_service_host PUBLIC cxx_std_20)
    target_link_libraries(minimax_service_host PUBLIC strix_session_runtime strix_lookup)
    add_library(strix_minimax_session_service STATIC
        "${REPO_ROOT}/src/minimax/serve/session_service.cpp"
        "${REPO_ROOT}/src/minimax/serve/serve_runtime.cpp")
    target_link_libraries(strix_minimax_session_service PUBLIC minimax_service_host)
    target_compile_definitions(strix_minimax_session_service PRIVATE STRIX_SERVE_REGISTRY_ROOT="${REPO_ROOT}")
    if(EXISTS "${REPO_ROOT}/src/minimax/serve/session_engine.cpp")
        target_sources(strix_minimax_session_service PRIVATE "${REPO_ROOT}/src/minimax/serve/session_engine.cpp")
    endif()
    if(MSVC)
        target_compile_definitions(strix_minimax_session_service PUBLIC __builtin_rintf=rintf)
        target_compile_definitions(strix_minimax_session_service PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    strix_session_warnings(strix_minimax_session_service)
    if(STRIX_LEGACY_MAINS)
        # Host compatibility smoke only; this is not the native legacy target.
        add_executable(minimax_service_compat "${REPO_ROOT}/apps/minimax_serve_main.cpp")
        target_link_libraries(minimax_service_compat PRIVATE strix_minimax_session_service strix_session_shell)
        add_test(NAME minimax-service-help COMMAND minimax_service_compat --help)
        add_test(NAME minimax-service-effective COMMAND minimax_service_compat --print-effective-config)
    endif()
    if(MSVC)
        target_compile_options(minimax_service_host PRIVATE /EHsc /utf-8)
    endif()
add_executable(minimax_service_tests "${REPO_ROOT}/tests/minimax_service/service_tests.cpp")
target_link_libraries(minimax_service_tests PRIVATE strix_minimax_session_service strix_session_shell)
target_compile_definitions(minimax_service_tests PRIVATE STRIX_EXPECT_MINIMAX_SERVICE=1)
add_test(NAME minimax-service COMMAND minimax_service_tests "${REPO_ROOT}")
set_tests_properties(minimax-service PROPERTIES TIMEOUT 120)
# The production factory is linked by the shared shell's service registration.
target_compile_definitions(session_registry_r7 PRIVATE STRIX_EXPECT_MINIMAX_SERVICE=1)
# The independent replay keeps its own instrumented production queue. Linking
# the shared protocol library supplies ApiService without copying api.cpp.
add_subdirectory("${REPO_ROOT}/tests/minimax_session_shared" replay)
target_link_libraries(minimax_replay_family PUBLIC strix_serve strix_session)
if(NOT TARGET minimax_shared_api_tests)
    add_executable(minimax_shared_api_tests "${REPO_ROOT}/tests/minimax_session_shared/shared_api_tests.cpp")
    target_link_libraries(minimax_shared_api_tests PRIVATE minimax_replay_family minimax_owner_cache)
    target_sources(minimax_shared_api_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/replay/capture_noop.cpp")
    add_test(NAME minimax-shared-api COMMAND minimax_shared_api_tests)
    set_tests_properties(minimax-shared-api PROPERTIES TIMEOUT 120 LABELS "cpu;session;minimax")
endif()
foreach(pair "minimax_api;minimax-api" "minimax_observe;minimax-observe")
    list(GET pair 0 source)
    list(GET pair 1 name)
    add_executable(${source}_r7 "${REPO_ROOT}/tests/${source}_tests.cpp")
    target_link_libraries(${source}_r7 PRIVATE minimax_service_host)
    add_test(NAME ${name} COMMAND ${source}_r7)
    set_tests_properties(${name} PROPERTIES TIMEOUT 120)
    if(source STREQUAL "minimax_api")
        # The retained real socket assertions take about 185 s in MSVC Debug.
        # This is a HOST test allowance, separate from native operator tickets.
        set_tests_properties(${name} PROPERTIES TIMEOUT 300)
    endif()
endforeach()
if(MSVC)
    target_compile_options(minimax_shared_api_tests PRIVATE /EHsc /utf-8 /UNDEBUG)
    target_compile_definitions(minimax_shared_api_tests PRIVATE __builtin_rintf=rintf)
else()
    target_compile_options(minimax_shared_api_tests PRIVATE -UNDEBUG)
endif()


# Independent archived-source witnesses never replace frozen expectations.
get_target_property(original_headers minimax_original_family INTERFACE_INCLUDE_DIRECTORIES)
list(GET original_headers 0 original_include)
get_filename_component(original_root "${original_include}" DIRECTORY)
if(EXISTS "${original_root}/tests/minimax_api_tests.cpp")
    add_executable(minimax_original_api_r7 "${original_root}/tests/minimax_api_tests.cpp")
    target_link_libraries(minimax_original_api_r7 PRIVATE minimax_original_family)
    add_test(NAME minimax-original-api COMMAND minimax_original_api_r7)
    set_tests_properties(minimax-original-api PROPERTIES TIMEOUT 120)
endif()
set(MINIMAX_WIRE_CAPTURE_SOURCE "" CACHE FILEPATH "Optional host wire witness source")
if(MINIMAX_WIRE_CAPTURE_SOURCE)
    add_executable(minimax_wire_current "${MINIMAX_WIRE_CAPTURE_SOURCE}")
    target_link_libraries(minimax_wire_current PRIVATE minimax_service_host)
    add_executable(minimax_wire_original "${MINIMAX_WIRE_CAPTURE_SOURCE}")
    target_link_libraries(minimax_wire_original PRIVATE minimax_original_family)
endif()
