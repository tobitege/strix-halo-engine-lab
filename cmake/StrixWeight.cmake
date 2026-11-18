include_guard(GLOBAL)
get_filename_component(STRIX_WEIGHT_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
add_library(strix-weight STATIC
    ${STRIX_WEIGHT_ROOT}/src/weight/host_q8.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/source.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/admission.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/model_binding.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/readers/gguf.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/readers/safetensors.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/readers/pack.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/adapters/qwen.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/adapters/ds41.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/adapters/minimax.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/adapters/glm.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/adapters/gemma4.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/adapters/h3.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/adapters/vision.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/resolve.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/cache.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/image_identity.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/decision_table.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/runtime.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/family.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/levers.cpp
    ${STRIX_WEIGHT_ROOT}/src/cpu/weight/runtime.cpp
    ${STRIX_WEIGHT_ROOT}/src/hip/weight/runtime.cpp
    ${STRIX_WEIGHT_ROOT}/src/cuda/weight/runtime.cpp
    ${STRIX_WEIGHT_ROOT}/src/intel/weight/runtime.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/types.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/contracts.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/registry.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/reference.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/formats/plain.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/formats/ggml.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/formats/gsq.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/formats/ds41.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/formats/packed.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/formats/fp8.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/formats/exl3.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/formats/channel_quant.cpp
    ${STRIX_WEIGHT_ROOT}/src/weight/formats/nvfp4.cpp
    # Existing numerical implementations remain the only scalar delegates.
    ${STRIX_WEIGHT_ROOT}/src/ggml_dequant.cpp
    ${STRIX_WEIGHT_ROOT}/src/gguf_reader.cpp
    ${STRIX_WEIGHT_ROOT}/src/iq4_nl.cpp
    ${STRIX_WEIGHT_ROOT}/src/ds41_formats.cpp
    ${STRIX_WEIGHT_ROOT}/src/ds41_moe.cpp
    ${STRIX_WEIGHT_ROOT}/src/ds41_expert_tp.cpp
    ${STRIX_WEIGHT_ROOT}/src/ds41_packed.cpp
    ${STRIX_WEIGHT_ROOT}/src/ds41_native_fp8.cpp
    ${STRIX_WEIGHT_ROOT}/src/q6p_registry.cpp
    ${STRIX_WEIGHT_ROOT}/src/cuda/core/nvfp4_reference.cpp
    ${STRIX_WEIGHT_ROOT}/src/gemma4/exl3/host.cpp
    ${STRIX_WEIGHT_ROOT}/src/h3/te/te_sha256.cpp
    ${STRIX_WEIGHT_ROOT}/src/serve/json.cpp)
target_include_directories(strix-weight PUBLIC ${STRIX_WEIGHT_ROOT}/include PRIVATE ${STRIX_WEIGHT_ROOT}/src)
target_compile_features(strix-weight PUBLIC cxx_std_20)
if(MSVC)
    target_compile_options(strix-weight PRIVATE /fp:strict /D_CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(strix-weight PRIVATE -ffp-contract=off)
endif()

# Legacy loader targets may be declared by vendor-only early-return branches or
# nested family modules. Attach the shared implementation after all declarations.
function(strix_weight_link_consumers directory)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
        if(target STREQUAL "strix-weight")
            continue()
        endif()
        get_target_property(sources ${target} SOURCES)
        string(JOIN ";" source_list ${sources})
        if(source_list MATCHES "(ds41_native_fp8|gguf_reader|qwen_model_description|ds41_prefill_driver|qwen_prefill_driver|glm_rank_plan|glm_model_description|mm_pack|src[/\\\\](gemma4|h3|vision)[/\\\\])" OR target MATCHES "^(strix[-_]h3|strix[-_]gemma4)")
            get_target_property(kind ${target} TYPE)
            if(NOT kind STREQUAL "INTERFACE_LIBRARY" AND NOT kind STREQUAL "UTILITY")
                set_property(TARGET ${target} APPEND PROPERTY LINK_LIBRARIES strix-weight)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_LINK_LIBRARIES strix-weight)
            endif()
        endif()
    endforeach()
    get_property(children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(child IN LISTS children)
        strix_weight_link_consumers("${child}")
    endforeach()
endfunction()
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL strix_weight_link_consumers "${CMAKE_SOURCE_DIR}")

find_package(Threads REQUIRED)
target_link_libraries(strix-weight PUBLIC Threads::Threads)

option(STRIX_WFMT_NATIVE_HIP_RUNTIME "Compile native HIP image runtime callbacks" OFF)
option(STRIX_WFMT_NATIVE_CUDA_RUNTIME "Compile native CUDA image runtime callbacks" OFF)
if(STRIX_WFMT_NATIVE_HIP_RUNTIME)
    find_package(hip CONFIG REQUIRED)
    target_link_libraries(strix-weight PRIVATE hip::host)
    target_compile_definitions(strix-weight PRIVATE STRIX_WFMT_HIP_RUNTIME=1)
endif()
if(STRIX_WFMT_NATIVE_CUDA_RUNTIME)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(strix-weight PRIVATE CUDA::cudart)
    target_compile_definitions(strix-weight PRIVATE STRIX_WFMT_CUDA_RUNTIME=1)
endif()
