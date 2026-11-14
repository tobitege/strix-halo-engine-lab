# Existing host cache/hash closure, shared by core, serving and standalone adopters.
include_guard(GLOBAL)
get_filename_component(_strix_state_cache_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
find_package(Threads REQUIRED)
add_library(strix_state_cache STATIC
    "${_strix_state_cache_root}/src/prefix_state_cache.cpp"
    "${_strix_state_cache_root}/src/session_state_cache.cpp"
    "${_strix_state_cache_root}/src/rotary_descriptor.cpp"
    "${_strix_state_cache_root}/src/prompt_terminal_diagnostics.cpp"
    "${_strix_state_cache_root}/src/model_content_hash.cpp"
    "${_strix_state_cache_root}/src/model_runtime_binding.cpp")
# The common writer is supplied with real family adapters. A missing concurrent
# source remains pending; never synthesize a replacement or an empty test.
if(EXISTS "${_strix_state_cache_root}/src/session/checkpoint_io.cpp")
    target_sources(strix_state_cache PRIVATE "${_strix_state_cache_root}/src/session/checkpoint_io.cpp")
else()
    message(STATUS "Common checkpoint IO source pending; combined-source verification required")
endif()
target_include_directories(strix_state_cache PUBLIC "${_strix_state_cache_root}/include")
target_compile_features(strix_state_cache PUBLIC cxx_std_20)
target_link_libraries(strix_state_cache PUBLIC Threads::Threads)
# Session compatibility serializes the existing rotary identity. Keep its
# strict arithmetic when this minimal closure is configured without the root.
if(MSVC)
    set_source_files_properties("${_strix_state_cache_root}/src/rotary_descriptor.cpp"
        PROPERTIES COMPILE_OPTIONS "/fp:strict")
else()
    set_source_files_properties("${_strix_state_cache_root}/src/rotary_descriptor.cpp"
        PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-frounding-math")
endif()
if(MSVC)
    target_compile_options(strix_state_cache PRIVATE /EHsc /utf-8 /W4)
    target_compile_definitions(strix_state_cache PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    target_compile_options(strix_state_cache PRIVATE -Wall -Wextra -Wpedantic)
endif()
