# Shared vision encoder runtime (include/strix/vision, src/vision): host library with the FP32 reference tower, the
# device plan and the embed cache; the HIP runner where the HIP runtime is built.
add_library(strix_vision_encoder STATIC
    "${CMAKE_SOURCE_DIR}/src/vision/reference.cpp"
    "${CMAKE_SOURCE_DIR}/src/vision/embed_cache.cpp")
target_include_directories(strix_vision_encoder PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_compile_features(strix_vision_encoder PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(strix_vision_encoder PUBLIC Threads::Threads)
if(MSVC)
    target_compile_definitions(strix_vision_encoder PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
strix_target_warnings(strix_vision_encoder)

strix_servespec_host_test(NAME strix-vision-encoder-tests
    SOURCES tests/vision_encoder_tests.cpp
    LIBS strix_vision_encoder)
strix_servespec_host_test(NAME strix-dense-fixture-checks-tests
    SOURCES tests/dense_fixture_checks_tests.cpp)
if(MSVC AND TARGET strix-vision-encoder-tests)
    target_compile_definitions(strix-vision-encoder-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

if(STRIX_ENABLE_HIP_RUNTIME AND TARGET strix_hip_runtime)
    add_library(strix_vision_encoder_hip STATIC "${CMAKE_SOURCE_DIR}/src/vision/encoder.hip")
    set_target_properties(strix_vision_encoder_hip PROPERTIES
        HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
    target_compile_options(strix_vision_encoder_hip PRIVATE
        -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee -Wall -Wextra)
    target_include_directories(strix_vision_encoder_hip PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(strix_vision_encoder_hip PUBLIC strix_vision_encoder strix_hip_runtime)
    if(BUILD_TESTING)
        # Device self-test (synthetic towers against the host reference). Needs a GPU, so it is not a CTest test.
        add_executable(strix-vision-encoder-hip-tests "${CMAKE_SOURCE_DIR}/tests/vision_encoder_hip_tests.cpp")
        target_link_libraries(strix-vision-encoder-hip-tests PRIVATE strix_vision_encoder_hip)
        if(MSVC)
            target_compile_definitions(strix-vision-encoder-hip-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
        endif()
        strix_target_warnings(strix-vision-encoder-hip-tests)
        # Device self-test of the shared dense FP16 core the GEMM is built on (column widths, idle-wave skip, padding).
        add_executable(strix-f16-dense-tall-hip-tests "${CMAKE_SOURCE_DIR}/tests/hip_f16_dense_tall_tests.hip")
        set_target_properties(strix-f16-dense-tall-hip-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_compile_options(strix-f16-dense-tall-hip-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee -Wall -Wextra)
        target_include_directories(strix-f16-dense-tall-hip-tests PRIVATE "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/include")
        target_link_libraries(strix-f16-dense-tall-hip-tests PRIVATE strix_hip_runtime)
    endif()
endif()
