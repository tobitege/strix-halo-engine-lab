cmake_minimum_required(VERSION 3.20)
project(strix_route_policy_host LANGUAGES CXX)
enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Threads REQUIRED)
get_filename_component(DEFAULT_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(STRIX_SOURCE_ROOT "${DEFAULT_SOURCE_ROOT}" CACHE PATH "Source tree under test")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/../route_policy_host_extract.py"
    "${CMAKE_CURRENT_LIST_DIR}/../route_policy_host_tests.cpp.in"
    "${STRIX_SOURCE_ROOT}/src/glm/prefill/glm_pf_exact.hip"
    "${STRIX_SOURCE_ROOT}/src/glm/prefill/glm_pf_exact.hpp"
    "${STRIX_SOURCE_ROOT}/src/glm/prefill/glm_pf_dense_policy.hpp"
    "${STRIX_SOURCE_ROOT}/src/glm/engine/engine.hip"
    "${STRIX_SOURCE_ROOT}/tests/hip_glm_pf_wmma_tests.hip"
    "${STRIX_SOURCE_ROOT}/src/minimax/engine/engine.hip"
    "${STRIX_SOURCE_ROOT}/src/minimax/moe/mmx_prefill_iu4.inc"
    "${STRIX_SOURCE_ROOT}/apps/hip_decode_token_main.cpp"
    "${STRIX_SOURCE_ROOT}/apps/hip_decode_options.hpp")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/../route_policy_host_extract.py"
    --source "${STRIX_SOURCE_ROOT}" --output "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND_ERROR_IS_FATAL ANY)
add_executable(route-policy-host "${CMAKE_CURRENT_BINARY_DIR}/route_policy_host.cpp")
add_executable(route-receipt-host "${STRIX_SOURCE_ROOT}/tests/route_receipt_tests.cpp"
    "${STRIX_SOURCE_ROOT}/src/serve/json.cpp")
foreach(target route-policy-host route-receipt-host)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/stubs"
        "${STRIX_SOURCE_ROOT}/include" "${STRIX_SOURCE_ROOT}/src" "${STRIX_SOURCE_ROOT}")
    target_link_libraries(${target} PRIVATE Threads::Threads)
    if(MSVC)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(${target} PRIVATE /W4 /WX /permissive- /utf-8)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endforeach()
foreach(case minimax glm_equivalent qwen glm_bypass glm_serve_chunk glm_schedule glm_reference glm_dense_baseline)
    add_test(NAME route-policy-${case} COMMAND route-policy-host ${case})
endforeach()
add_test(NAME route-receipt-host COMMAND route-receipt-host)
