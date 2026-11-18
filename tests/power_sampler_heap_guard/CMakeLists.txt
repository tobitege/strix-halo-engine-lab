cmake_minimum_required(VERSION 3.20)
project(power_sampler_heap_guard NONE)
enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(STRIX_TEST_CXX "" CACHE FILEPATH "C++ compiler for preprocessing the heap guard")
if(NOT STRIX_TEST_CXX)
    message(FATAL_ERROR "Set STRIX_TEST_CXX to a C++ compiler executable")
endif()
add_test(NAME power_sampler_heap_guard
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/../power_sampler_heap_guard_test.py"
            --compiler "${STRIX_TEST_CXX}")
