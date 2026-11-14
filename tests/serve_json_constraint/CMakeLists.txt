cmake_minimum_required(VERSION 3.24)
project(strix_json_constraint_host LANGUAGES CXX)
include(CTest)
find_package(Threads REQUIRED)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_executable(strix-serve-json-constraint-tests
    "${REPO_ROOT}/tests/serve_json_constraint_tests.cpp"
    "${REPO_ROOT}/src/serve/json_constraint.cpp"
    "${REPO_ROOT}/src/serve/json.cpp"
    "${REPO_ROOT}/src/serve/bpe_tokenizer.cpp")
target_include_directories(strix-serve-json-constraint-tests PRIVATE "${REPO_ROOT}/include")
target_compile_features(strix-serve-json-constraint-tests PRIVATE cxx_std_20)
target_link_libraries(strix-serve-json-constraint-tests PRIVATE Threads::Threads)
if(MSVC)
    target_compile_options(strix-serve-json-constraint-tests PRIVATE /W4 /WX /permissive-)
    target_compile_definitions(strix-serve-json-constraint-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(strix-serve-json-constraint-tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
add_test(NAME strix-serve-json-constraint-tests COMMAND strix-serve-json-constraint-tests)
set_tests_properties(strix-serve-json-constraint-tests PROPERTIES TIMEOUT 180)
