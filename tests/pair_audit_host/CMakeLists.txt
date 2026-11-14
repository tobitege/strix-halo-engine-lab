cmake_minimum_required(VERSION 3.20)
project(pair_audit_host LANGUAGES CXX)
enable_testing()
find_package(Threads REQUIRED)
get_filename_component(root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

function(pair_test target source)
    add_executable(${target} "${source}" "${root}/src/serve/json.cpp")
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_include_directories(${target} PRIVATE "${root}/include" "${root}/src")
    target_link_libraries(${target} PRIVATE Threads::Threads)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX /EHsc /wd4996)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -Wshadow -Wsign-conversion)
    endif()
endfunction()

pair_test(pair "${root}/tests/qwen_prefill_pair_tests.cpp")
add_test(NAME qwen-prefill-pair COMMAND pair)
pair_test(staging "${root}/tests/qwen_prefill_staging_crc_tests.cpp")
target_include_directories(staging BEFORE PRIVATE "${root}/tests/qwen_prefill_staging_shim")
add_test(NAME qwen-prefill-staging-crc COMMAND staging)

# Prove the negative test rejects accepted corruption and unrelated transport failures.
# Only generated build-tree copies are mutated; the production headers stay intact.
file(READ "${root}/include/strix/pair/pipe.hpp" pipe)
foreach(kind IN ITEMS accepted-corruption declined-without-guard)
    set(mutant "${pipe}")
    if(kind STREQUAL "accepted-corruption")
        foreach(scope IN ITEMS piece whole)
            string(REPLACE "throw std::runtime_error(\"pair prefill seam ${scope} CRC mismatch\");"
                "(void)crc;" mutant "${mutant}")
        endforeach()
    else()
        string(REPLACE "if (device && !moved) throw std::logic_error(\"pair prefill device-checked piece was not moved by the transport\");"
            "(void)device;" mutant "${mutant}")
    endif()
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${kind}/strix/pair/pipe.hpp" "${mutant}")
    pair_test(${kind} "${root}/tests/qwen_prefill_pair_tests.cpp")
    target_include_directories(${kind} BEFORE PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/${kind}")
    add_test(NAME rejects-${kind} COMMAND ${kind} --device-checked-only)
    set_tests_properties(rejects-${kind} PROPERTIES WILL_FAIL TRUE)
endforeach()

file(READ "${root}/tests/qwen_prefill_pair_tests.cpp" pair_source)
string(REPLACE "if (mode == 1U && offset == 0U) wire[size / 2U] ^= std::byte{1};"
    "if (mode == 1U && offset == 0U) throw std::runtime_error(\"fake transport timed out\");"
    timeout_source "${pair_source}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/timeout.cpp" "${timeout_source}")
pair_test(unrelated-timeout "${CMAKE_CURRENT_BINARY_DIR}/timeout.cpp")
add_test(NAME rejects-unrelated-timeout COMMAND unrelated-timeout --device-checked-only)
set_tests_properties(rejects-unrelated-timeout PROPERTIES WILL_FAIL TRUE)
set_tests_properties(qwen-prefill-pair qwen-prefill-staging-crc rejects-accepted-corruption
    rejects-declined-without-guard rejects-unrelated-timeout PROPERTIES TIMEOUT 120)
