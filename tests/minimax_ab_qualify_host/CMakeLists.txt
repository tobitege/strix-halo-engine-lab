cmake_minimum_required(VERSION 3.20)
project(minimax_ab_qualify_host LANGUAGES CXX)
enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(NOT DEFINED STRIX_ROOT)
    get_filename_component(STRIX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/extract.py"
    "${STRIX_ROOT}/apps/minimax_rank_main.cpp"
    "${STRIX_ROOT}/tests/hip_mmx_moe_bench.hip"
    "${STRIX_ROOT}/src/minimax/engine/engine.hip")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/extract.py"
    "${STRIX_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}" COMMAND_ERROR_IS_FATAL ANY)
add_executable(minimax-ab-qualify-host regression.cpp)
target_compile_features(minimax-ab-qualify-host PRIVATE cxx_std_20)
target_include_directories(minimax-ab-qualify-host PRIVATE "${CMAKE_CURRENT_BINARY_DIR}" "${STRIX_ROOT}/include")
if(MSVC)
    target_compile_options(minimax-ab-qualify-host PRIVATE /W4 /WX /permissive-)
else()
    target_compile_options(minimax-ab-qualify-host PRIVATE -Wall -Wextra -Werror)
endif()
foreach(case covered fallback stale contaminated partial ordinary groups)
    set(expected "")
    if(case STREQUAL "covered")
        set(expected "0 3 0 3 0 3 -> COVERED")
    elseif(case MATCHES "^(fallback|stale|contaminated|partial)$")
        set(expected "UNQUALIFIED")
    endif()
    add_test(NAME minimax-ab-qualify-${case} COMMAND "${CMAKE_COMMAND}"
        "-DEXECUTABLE=$<TARGET_FILE:minimax-ab-qualify-host>"
        "-DCASE=${case}" "-DEXPECTED=${expected}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/run_case.cmake")
endforeach()
