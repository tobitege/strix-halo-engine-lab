cmake_minimum_required(VERSION 3.24)
project(strix_session_storage LANGUAGES CXX)
include(CTest)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
include("${REPO_ROOT}/cmake/StrixSession.cmake")
# Keep the historical independently buildable storage archive target.
add_library(strix_session_storage INTERFACE)
target_link_libraries(strix_session_storage INTERFACE strix_session)
if(BUILD_TESTING)
    # Retain the historical CTest entry names as compatibility aliases.
    add_test(NAME session_kv_tests COMMAND session_kv_tests)
    add_test(NAME session_prefix_cache_tests COMMAND session_prefix_cache_tests)
    set_tests_properties(session_kv_tests session_prefix_cache_tests
        PROPERTIES LABELS "cpu;session" TIMEOUT 30)
endif()
