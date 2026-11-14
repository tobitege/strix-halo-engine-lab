cmake_minimum_required(VERSION 3.24)
project(strix_ds41_step_digest_host LANGUAGES CXX)
include(CTest)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Threads REQUIRED)
set(SERVER "${CMAKE_CURRENT_SOURCE_DIR}/../../apps/deepseek41_serve_main.cpp")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/step_digest.hpp"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/source_contract.py"
            "${SERVER}" "${CMAKE_CURRENT_BINARY_DIR}/step_digest.hpp"
    DEPENDS "${SERVER}" source_contract.py VERBATIM)
add_executable(ds41_step_digest_tests step_digest_tests.cpp "${CMAKE_CURRENT_BINARY_DIR}/step_digest.hpp")
target_include_directories(ds41_step_digest_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(ds41_step_digest_tests PRIVATE cxx_std_20)
target_link_libraries(ds41_step_digest_tests PRIVATE Threads::Threads)
if(MSVC)
    target_compile_options(ds41_step_digest_tests PRIVATE /W4 /WX /EHsc)
else()
    target_compile_options(ds41_step_digest_tests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ds41_step_digest_source_contract
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/source_contract.py" "${SERVER}")
add_test(NAME ds41_step_digest_emulated_pair COMMAND ds41_step_digest_tests)
