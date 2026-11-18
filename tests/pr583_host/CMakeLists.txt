cmake_minimum_required(VERSION 3.24)
project(strix_pr583_host LANGUAGES CXX)
include(CTest)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Threads REQUIRED)
get_filename_component(STRIX_ROOT "../.." ABSOLUTE)
add_subdirectory(../ds41_step_digest_host step_profile)
set(COMM "${STRIX_ROOT}/include/strix/dist/communicator.hpp")
set(FORWARD "${STRIX_ROOT}/src/hip_deepseek41_batch_forward.hip")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/admission.inc" "${CMAKE_CURRENT_BINARY_DIR}/startup.inc"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/source_contract.py"
            "${COMM}" "${FORWARD}" "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${COMM}" "${FORWARD}" source_contract.py VERBATIM)
add_executable(pr583_admission_tests admission_tests.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/admission.inc" "${CMAKE_CURRENT_BINARY_DIR}/startup.inc")
target_include_directories(pr583_admission_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(pr583_admission_tests PRIVATE cxx_std_20)
target_link_libraries(pr583_admission_tests PRIVATE Threads::Threads)
if(MSVC)
    target_compile_options(pr583_admission_tests PRIVATE /W4 /WX /EHsc)
else()
    target_compile_options(pr583_admission_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
add_test(NAME pr583_admission_source_contract
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/source_contract.py" "${COMM}" "${FORWARD}")
add_test(NAME pr583_scratch_and_pair_agreement COMMAND pr583_admission_tests pair)
add_test(NAME pr583_chunk_route_admission COMMAND pr583_admission_tests chunks)
set_tests_properties(pr583_scratch_and_pair_agreement pr583_chunk_route_admission PROPERTIES TIMEOUT 20)
