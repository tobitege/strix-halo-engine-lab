cmake_minimum_required(VERSION 3.20)
project(strix_token_memo_host LANGUAGES CXX)
enable_testing()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(STRIX_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
find_package(Threads REQUIRED)
if(MSVC)
    add_compile_options(/W4 /WX /utf-8)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()
add_library(tokenizer_host STATIC
    "${STRIX_ROOT}/src/serve/bpe_tokenizer.cpp"
    "${STRIX_ROOT}/src/serve/json.cpp"
    "${STRIX_ROOT}/src/serve/chat_template.cpp"
    "${STRIX_ROOT}/src/serve/deepseek_chat.cpp"
    "${STRIX_ROOT}/src/serve/glm_chat.cpp"
    "${STRIX_ROOT}/src/minimax/serve/minimax_tokenizer.cpp"
    "${STRIX_ROOT}/src/minimax/serve/minimax_chat.cpp")
target_include_directories(tokenizer_host PUBLIC "${STRIX_ROOT}/include" "${STRIX_ROOT}/src")
target_link_libraries(tokenizer_host PUBLIC Threads::Threads)
add_library(token_memo_host STATIC "${STRIX_ROOT}/src/serve/token_memo.cpp")
target_link_libraries(token_memo_host PUBLIC tokenizer_host)
add_executable(strix-token-memo-tests "${STRIX_ROOT}/tests/token_memo_tests.cpp")
target_link_libraries(strix-token-memo-tests PRIVATE token_memo_host)
add_test(NAME token_memo_properties COMMAND strix-token-memo-tests)
add_executable(strix-minimax-tokenizer-tests "${STRIX_ROOT}/tests/minimax_tokenizer_tests.cpp")
target_link_libraries(strix-minimax-tokenizer-tests PRIVATE token_memo_host)
target_compile_definitions(strix-minimax-tokenizer-tests PRIVATE
    STRIX_MINIMAX_MEMO_FIXTURE="${STRIX_ROOT}/tests/minimax_tokenizer_memo.json")
add_test(NAME minimax_synthetic_memo COMMAND strix-minimax-tokenizer-tests
    "${STRIX_ROOT}/src/minimax/serve/fixtures/minimax_tokenizer_fixtures.json")

# Optional private snapshot of the unfixed headers/source, to demonstrate regression sensitivity.
set(TOKEN_MEMO_ORIGINAL_ROOT "" CACHE PATH "Unfixed token memo source/header snapshot")
if(TOKEN_MEMO_ORIGINAL_ROOT)
    add_executable(token_memo_original "${STRIX_ROOT}/tests/token_memo_tests.cpp"
        "${TOKEN_MEMO_ORIGINAL_ROOT}/src/serve/token_memo.cpp")
    target_include_directories(token_memo_original BEFORE PRIVATE "${TOKEN_MEMO_ORIGINAL_ROOT}/include")
    target_link_libraries(token_memo_original PRIVATE tokenizer_host)
    foreach(regression lifetime capacity)
        add_test(NAME original_rejects_${regression} COMMAND token_memo_original --${regression}-regression)
        set_tests_properties(original_rejects_${regression} PROPERTIES WILL_FAIL TRUE)
    endforeach()
endif()
