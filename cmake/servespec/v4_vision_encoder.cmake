# Shared vision encoder runtime (include/strix/vision, src/vision): host library with the FP32 reference tower, the
# device plan and the embed cache; the HIP runner where the HIP runtime is built.
add_library(strix_vision_encoder STATIC
    "${CMAKE_SOURCE_DIR}/src/vision/reference.cpp"
    "${CMAKE_SOURCE_DIR}/src/vision/projector.cpp"
    "${CMAKE_SOURCE_DIR}/src/vision/embed_cache.cpp")
if(NOT MSVC)
    # Projector activation: no fused multiply-add (rows are cache content; the device kernel keeps the same order).
    set_source_files_properties("${CMAKE_SOURCE_DIR}/src/vision/projector.cpp" PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math")
endif()
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
if(TARGET strix-vision-encoder-tests)
    foreach(policy IN ITEMS default off on)
        if(policy STREQUAL "default")
            set(projector_env "--unset=STRIX_VISION_PROJECT_DEVICE")
            set(projector_expected on)
        elseif(policy STREQUAL "off")
            set(projector_env "STRIX_VISION_PROJECT_DEVICE=0")
            set(projector_expected off)
        else()
            set(projector_env "STRIX_VISION_PROJECT_DEVICE=1")
            set(projector_expected on)
        endif()
        add_test(NAME vision-projector-policy-${policy} COMMAND ${CMAKE_COMMAND} -E env "${projector_env}"
            $<TARGET_FILE:strix-vision-encoder-tests> --project-device ${projector_expected})
    endforeach()
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
        # Device self-test of the shared dense flash core's one- and two-term operands against the host model of its
        # products and order and against exact attention (small residuals, cancellation, tails; two_term_model.hpp).
        # Run strix-dense-flash-terms-hip-tests explicitly and record its result; host CTest does not run it.
        # Causal single-split evidence only: no cyclic placement, split-K merge, production packing or image fidelity.
        add_executable(strix-dense-flash-terms-hip-tests "${CMAKE_SOURCE_DIR}/tests/hip_dense_flash_terms_tests.hip")
        set_target_properties(strix-dense-flash-terms-hip-tests PROPERTIES
            HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
        target_compile_options(strix-dense-flash-terms-hip-tests PRIVATE
            -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee -Wall -Wextra)
        target_include_directories(strix-dense-flash-terms-hip-tests PRIVATE "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/include")
        target_link_libraries(strix-dense-flash-terms-hip-tests PRIVATE strix_hip_runtime)
    endif()
endif()
