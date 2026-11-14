cmake_minimum_required(VERSION 3.20)
project(session_recurrent_host LANGUAGES CXX)
enable_testing()
get_filename_component(STRIX_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
include("${STRIX_ROOT}/cmake/StrixStateCache.cmake")
add_library(strix_session STATIC
    "${STRIX_ROOT}/src/session/kv.cpp"
    "${STRIX_ROOT}/src/session/prefix_cache.cpp")
target_include_directories(strix_session PUBLIC "${STRIX_ROOT}/include")
target_compile_features(strix_session PUBLIC cxx_std_20)
target_link_libraries(strix_session PUBLIC strix_state_cache)
add_executable(session_recurrent "${STRIX_ROOT}/tests/session_recurrent_tests.cpp")
target_link_libraries(session_recurrent PRIVATE strix_session)
add_test(NAME session_recurrent COMMAND session_recurrent)
