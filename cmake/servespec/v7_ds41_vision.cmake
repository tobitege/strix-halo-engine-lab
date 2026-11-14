# DeepSeek V4.1 image tower on the shared vision stack: the GGUF weight source every GGUF tower shares
# (strix_vision_gguf), the family adapter (strix_deepseek41_vision), its host test and the real-file gate tool.
if(TARGET strix_vision_encoder AND TARGET strix_core)
    add_library(strix_vision_gguf STATIC "${CMAKE_SOURCE_DIR}/src/vision/gguf_weights.cpp")
    target_link_libraries(strix_vision_gguf PUBLIC strix_vision_encoder strix_core)
    strix_target_warnings(strix_vision_gguf)
endif()

if(TARGET strix_vision_gguf AND TARGET strix_vision_preprocess AND TARGET strix_family_deepseek41)
    add_library(strix_deepseek41_vision STATIC "${CMAKE_SOURCE_DIR}/src/deepseek41/vision_adapter.cpp")
    target_link_libraries(strix_deepseek41_vision PUBLIC
        strix_family_deepseek41 strix_vision_gguf strix_vision_preprocess)
    if(MSVC)
        target_compile_definitions(strix_deepseek41_vision PRIVATE _CRT_SECURE_NO_WARNINGS)
    else()
        # Rows are cache content: no fused multiply-add in the projector's activation.
        target_compile_options(strix_deepseek41_vision PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    strix_target_warnings(strix_deepseek41_vision)
endif()

strix_servespec_host_test(NAME deepseek41-vision
    SOURCES tests/deepseek41_vision_tests.cpp
    LIBS strix_deepseek41_vision)
if(TARGET deepseek41-vision AND NOT MSVC)
    # The test recomputes normalised pixels and compares them exactly with the preprocessing library, which is built
    # without fused multiply-add.
    target_compile_options(deepseek41-vision PRIVATE -ffp-contract=off -fno-fast-math)
endif()

if(BUILD_TESTING)
    # Resume state of the reference tool (tools/vision/ds41_ref.py): needs numpy, as the tool does.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import numpy"
            RESULT_VARIABLE strix_ds41_ref_numpy OUTPUT_QUIET ERROR_QUIET)
        if(strix_ds41_ref_numpy EQUAL 0)
            add_test(NAME deepseek41-vision-ref COMMAND "${Python3_EXECUTABLE}" -B
                "${CMAKE_SOURCE_DIR}/tests/deepseek41_vision_ref_tests.py")
        endif()
    endif()
endif()

if(BUILD_TESTING AND TARGET strix_deepseek41_vision)
    # Real-file gate (needs the admitted vision file and the dumps of tools/vision/ds41_ref.py): not a CTest test.
    add_executable(strix-deepseek41-vision-gate "${CMAKE_SOURCE_DIR}/tests/deepseek41_vision_gate.cpp")
    target_link_libraries(strix-deepseek41-vision-gate PRIVATE strix_deepseek41_vision)
    strix_target_warnings(strix-deepseek41-vision-gate)
    if(TARGET strix_vision_encoder_hip)
        add_executable(strix-deepseek41-vision-gate-hip "${CMAKE_SOURCE_DIR}/tests/deepseek41_vision_gate.cpp")
        target_compile_definitions(strix-deepseek41-vision-gate-hip PRIVATE STRIX_DS41_VISION_GATE_HIP=1)
        target_link_libraries(strix-deepseek41-vision-gate-hip PRIVATE strix_deepseek41_vision strix_vision_encoder_hip)
        strix_target_warnings(strix-deepseek41-vision-gate-hip)
    endif()
endif()
