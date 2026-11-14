cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(strix_ds41_prefix_checkpoint_io LANGUAGES CXX)
    include(CTest)
    get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
    include("${REPO_ROOT}/cmake/StrixStateCache.cmake")
else()
    set(REPO_ROOT "${PROJECT_SOURCE_DIR}")
endif()

# Same production Writer/codec; only a worker-thread observation boundary is enabled.
# No accelerator, native model, fake cache, or alternate IO implementation.
add_library(ds41_prefix_io_fixture STATIC
    "${REPO_ROOT}/src/deepseek41_prefix_cache.cpp"
    "${REPO_ROOT}/src/deepseek41_attention_state.cpp")
target_link_libraries(ds41_prefix_io_fixture PUBLIC strix_state_cache)
target_compile_features(ds41_prefix_io_fixture PUBLIC cxx_std_20)
target_compile_definitions(ds41_prefix_io_fixture PUBLIC STRIX_DS41_PREFIX_IO_TESTING)
if(MSVC)
    target_compile_options(ds41_prefix_io_fixture PUBLIC /EHsc /utf-8 /W4)
else()
    target_compile_options(ds41_prefix_io_fixture PUBLIC -Wall -Wextra -Wpedantic)
endif()
add_executable(ds41_prefix_checkpoint_io_tests io_tests.cpp)
target_link_libraries(ds41_prefix_checkpoint_io_tests PRIVATE ds41_prefix_io_fixture)
add_test(NAME ds41_prefix_checkpoint_io COMMAND ds41_prefix_checkpoint_io_tests
    test "${CMAKE_CURRENT_BINARY_DIR}/artifacts")
set_tests_properties(ds41_prefix_checkpoint_io PROPERTIES TIMEOUT 120 LABELS "cpu;session;checkpoint")

# Checks must execute with NDEBUG, including an independently checked failing arm.
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/check-sentinel-$<CONFIG>.cmake" CONTENT
"execute_process(COMMAND \"$<TARGET_FILE:ds41_prefix_checkpoint_io_tests>\" --fail-sentinel RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 10)
if(NOT rc EQUAL 1 OR NOT err STREQUAL \"DS41 prefix IO FAIL: deliberate active-check sentinel\\n\")
  message(FATAL_ERROR \"sentinel rc=\${rc} out=\${out} err=\${err}\")
endif()
")
add_test(NAME ds41_prefix_checkpoint_io_checks COMMAND "${CMAKE_COMMAND}" -P
    "${CMAKE_CURRENT_BINARY_DIR}/check-sentinel-$<CONFIG>.cmake")
set_tests_properties(ds41_prefix_checkpoint_io_checks PROPERTIES TIMEOUT 15 LABELS "cpu;session;checkpoint")
