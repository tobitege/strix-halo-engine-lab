# Provided embedding rows in a prefill run: the shared splice input (include/strix/session/embed_splice.hpp, header
# only), its device kernel (strix_embed_splice_hip) and the DeepSeek V4.1 prefill that takes it.
if(STRIX_ENABLE_HIP_RUNTIME AND TARGET strix_ds41_device_prefill)
    add_library(strix_embed_splice_hip STATIC "${CMAKE_SOURCE_DIR}/src/session/embed_splice.hip")
    set_target_properties(strix_embed_splice_hip PROPERTIES
        HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
    target_compile_options(strix_embed_splice_hip PRIVATE
        -mno-wavefrontsize64 -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee -Wall -Wextra)
    target_include_directories(strix_embed_splice_hip PUBLIC "${CMAKE_SOURCE_DIR}/include" "${STRIX_HIP_INCLUDE_DIR}")
    target_link_libraries(strix_ds41_device_prefill PUBLIC strix_embed_splice_hip)
endif()

strix_servespec_host_test(NAME session-embed-splice
    SOURCES tests/session_embed_splice_tests.cpp src/session/embed_rows.cpp)
if(TARGET session-embed-splice)
    set_tests_properties(session-embed-splice PROPERTIES LABELS "cpu;session")
endif()

strix_servespec_host_test(NAME deepseek41-vision-splice
    SOURCES tests/deepseek41_vision_splice_tests.cpp src/session/embed_rows.cpp
    LIBS strix_deepseek41_vision)
if(TARGET deepseek41-vision-splice)
    set_tests_properties(deepseek41-vision-splice PROPERTIES LABELS "cpu;session")
endif()
