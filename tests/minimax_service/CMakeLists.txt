cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(strix_minimax_service_host LANGUAGES CXX)
    include(CTest)
    get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
    add_subdirectory("${REPO_ROOT}/tests/session_runtime" shared)
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
else()
    set(REPO_ROOT "${PROJECT_SOURCE_DIR}")
endif()
add_executable(minimax_service_tests service_tests.cpp)
target_link_libraries(minimax_service_tests PRIVATE strix_minimax_session_service strix_session_shell)
target_compile_definitions(minimax_service_tests PRIVATE STRIX_EXPECT_MINIMAX_SERVICE=1)
strix_session_warnings(minimax_service_tests)
if(NOT MSVC)
    # Request fixtures name only the fields they exercise.
    target_compile_options(minimax_service_tests PRIVATE -Wno-missing-field-initializers)
endif()
add_test(NAME minimax-service COMMAND minimax_service_tests "${REPO_ROOT}")
set_tests_properties(minimax-service PROPERTIES LABELS "cpu;session;minimax" TIMEOUT 60)
