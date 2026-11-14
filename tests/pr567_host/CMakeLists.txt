cmake_minimum_required(VERSION 3.20)
project(strix_pr567_host LANGUAGES CXX)
enable_testing()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(STRIX_ROOT "../.." ABSOLUTE)
add_executable(strix-row-layout-tests
    "${STRIX_ROOT}/tests/row_layout_tests.cpp"
    "${STRIX_ROOT}/src/execution_arena.cpp"
    "${STRIX_ROOT}/src/execution_buffers.cpp"
    "${STRIX_ROOT}/src/execution_io.cpp"
    "${STRIX_ROOT}/src/ple_workspace.cpp"
    "${STRIX_ROOT}/src/indexed_attention_workspace.cpp"
    "${STRIX_ROOT}/src/model_contract.cpp"
    "${STRIX_ROOT}/src/qsa_selection_plan.cpp"
    "${STRIX_ROOT}/src/q6p_registry.cpp"
    "${STRIX_ROOT}/src/moe_q4k_a8_runtime.cpp")
target_include_directories(strix-row-layout-tests PRIVATE "${STRIX_ROOT}/include")
if(MSVC)
    target_compile_definitions(strix-row-layout-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(strix-row-layout-tests PRIVATE /W4 /WX /Gy)
    target_link_options(strix-row-layout-tests PRIVATE /OPT:REF)
else()
    target_compile_options(strix-row-layout-tests PRIVATE -Wall -Wextra -Wpedantic -Werror -ffunction-sections -fdata-sections)
    target_link_options(strix-row-layout-tests PRIVATE -Wl,--gc-sections)
endif()
add_test(NAME strix-row-layout-tests COMMAND strix-row-layout-tests)
