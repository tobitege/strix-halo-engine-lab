cmake_minimum_required(VERSION 3.20)
project(strix_pr528_host_tests LANGUAGES CXX)
enable_testing()
find_package(Threads REQUIRED)
get_filename_component(STRIX_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

add_executable(host-lane "${STRIX_ROOT}/tests/host_lane_tests.cpp")
add_executable(incremental-tokenizer "${STRIX_ROOT}/tests/incremental_tokenizer_tests.cpp"
    "${STRIX_ROOT}/src/serve/bpe_tokenizer.cpp" "${STRIX_ROOT}/src/serve/json.cpp"
    "${STRIX_ROOT}/src/serve/prompt_stream.cpp")
add_executable(bpe-cache "${STRIX_ROOT}/tests/bpe_cache_tests.cpp" "${STRIX_ROOT}/src/serve/json.cpp")
foreach(target host-lane incremental-tokenizer bpe-cache)
    target_include_directories(${target} PRIVATE "${STRIX_ROOT}/src" "${STRIX_ROOT}/include")
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_link_libraries(${target} PRIVATE Threads::Threads)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX /utf-8 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES TIMEOUT 300)
endforeach()
foreach(rounds 0 -1)
    add_test(NAME benchmark-rounds-${rounds} COMMAND "${CMAKE_COMMAND}"
        "-DPROGRAM=$<TARGET_FILE:incremental-tokenizer>" "-DROUNDS=${rounds}"
        -P "${CMAKE_CURRENT_LIST_DIR}/invalid_rounds.cmake")
endforeach()
