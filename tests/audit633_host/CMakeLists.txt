cmake_minimum_required(VERSION 3.20)
project(audit633_host LANGUAGES CXX)
enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
get_filename_component(STRIX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(AUDIT633_REVISION "" CACHE STRING "Optional local Git revision for regression demonstration")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/extract.py"
    "${STRIX_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}" "${AUDIT633_REVISION}"
    COMMAND_ERROR_IS_FATAL ANY)
add_executable(audit633-host regression.cpp
    "${STRIX_ROOT}/src/model_contract.cpp"
    "${STRIX_ROOT}/src/execution_buffers.cpp"
    "${STRIX_ROOT}/src/indexed_attention_state.cpp"
    "${STRIX_ROOT}/src/indexed_attention_workspace.cpp"
    "${STRIX_ROOT}/src/ple_workspace.cpp"
    "${STRIX_ROOT}/src/moe_q4k_a8_runtime.cpp"
    "${STRIX_ROOT}/src/q6p_registry.cpp"
    "${STRIX_ROOT}/src/qsa_selection_plan.cpp")
target_include_directories(audit633-host PRIVATE "${CMAKE_CURRENT_BINARY_DIR}" "${STRIX_ROOT}/include")
target_compile_features(audit633-host PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(audit633-host PRIVATE /W4 /WX /permissive- /Gy)
    target_compile_definitions(audit633-host PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_link_options(audit633-host PRIVATE /OPT:REF)
else()
    target_compile_options(audit633-host PRIVATE -Wall -Wextra -Werror -ffunction-sections -fdata-sections)
    target_link_options(audit633-host PRIVATE -Wl,--gc-sections)
endif()
foreach(case implicit-context explicit-context explicit-yarn native-boundary runtime-ceiling runtime-automatic)
    add_test(NAME audit633-${case} COMMAND audit633-host ${case})
endforeach()
