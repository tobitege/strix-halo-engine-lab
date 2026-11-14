# Shared vision preprocessing (host only): image decode, resize policies and resamplers, normalize, patchify.
# The decoder is a vendored single-header library compiled into one translation unit; no system package is needed.
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/StrixStateCache.cmake")
get_filename_component(_strix_vision_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
add_library(strix_vision_preprocess STATIC
    "${_strix_vision_root}/src/vision/decode.cpp"
    "${_strix_vision_root}/src/vision/resize.cpp"
    "${_strix_vision_root}/src/vision/patchify.cpp")
target_include_directories(strix_vision_preprocess PUBLIC "${_strix_vision_root}/include")
# SYSTEM: the vendored decoder is not held to the project warning set.
target_include_directories(strix_vision_preprocess SYSTEM PRIVATE "${_strix_vision_root}/third_party")
target_link_libraries(strix_vision_preprocess PUBLIC strix_state_cache)
target_compile_features(strix_vision_preprocess PUBLIC cxx_std_20)
if(COMMAND strix_target_warnings)
    strix_target_warnings(strix_vision_preprocess)
endif()
if(MSVC)
    target_compile_options(strix_vision_preprocess PRIVATE /EHsc /utf-8)
else()
    # The fp32 resampler and normalization are a bit-exact contract (fixtures, cache digests): never fuse a
    # multiply and an add, whatever the target instruction set.
    target_compile_options(strix_vision_preprocess PRIVATE -ffp-contract=off -fno-fast-math)
endif()

if(BUILD_TESTING)
    add_executable(strix-vision-preprocess-tests "${_strix_vision_root}/tests/vision/preprocess_tests.cpp")
    target_link_libraries(strix-vision-preprocess-tests PRIVATE strix_vision_preprocess)
    if(COMMAND strix_target_warnings)
        strix_target_warnings(strix-vision-preprocess-tests)
    endif()
    if(MSVC)
        target_compile_options(strix-vision-preprocess-tests PRIVATE /EHsc /utf-8)
    else()
        # The test restates fp32 expressions of the library; it must evaluate them the same way.
        target_compile_options(strix-vision-preprocess-tests PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME vision-preprocess
        COMMAND strix-vision-preprocess-tests "${_strix_vision_root}/tests/vision/fixtures")
    set_tests_properties(vision-preprocess PROPERTIES TIMEOUT 120)
endif()
