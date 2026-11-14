enable_language(CUDA)
find_package(Threads REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/EHsc> $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
endif()
add_library(strix_qwen_cuda STATIC
    src/cuda/qwen_prefill_driver.cu src/cuda/qwen_hc.cu src/cuda/qwen_dense.cu
    src/cuda/qwen_gdn.cu src/cuda/qwen_qsa.cu src/cuda/qwen_moe.cu
    src/cuda/qwen_plemtp.cu src/cuda/qwen_mtp_draft.cu src/cuda/qwen_transcode.cu src/cuda/qwen_amd_math_tables.cu src/gguf_reader.cpp)
target_include_directories(strix_qwen_cuda PUBLIC include src/cuda)
# Every Qwen CUDA binary resolves its device through the start gate (strix-devices-v1) before its first allocation.
include(${CMAKE_CURRENT_LIST_DIR}/StrixDevice.cmake)
target_link_libraries(strix_qwen_cuda PUBLIC strix_device)
target_compile_options(strix_qwen_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(strix_qwen_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()
# Handoff library and the serving front end of the node (OpenAI/Anthropic API, SSE, GGUF BPE tokenizer, Qwen chat
# template) for the RTX engine: strix_qwen_handoff and strix_serve_win, with the libraries they call linked by name.
include(${CMAKE_CURRENT_LIST_DIR}/StrixServeStandalone.cmake)
strix_add_standalone_serve(strix_serve_win strix_qwen_handoff)
include(${CMAKE_CURRENT_LIST_DIR}/StrixBuildInfo.cmake)  # generated/strix/build_info.hpp for src/preflight.cpp
add_executable(strix-cuda-qwen-prefill-service apps/cuda_qwen_prefill_service_main.cu)
target_link_libraries(strix-cuda-qwen-prefill-service PRIVATE strix_qwen_cuda strix_qwen_handoff)
add_executable(qwen_prefill_e2e tests/cuda/qwen_prefill_e2e.cu)
target_link_libraries(qwen_prefill_e2e PRIVATE strix_qwen_cuda strix_qwen_handoff)
add_executable(qwen_prefill_schedule_test tests/cuda/qwen_prefill_schedule_test.cpp)
target_include_directories(qwen_prefill_schedule_test PRIVATE include)
target_link_libraries(qwen_prefill_schedule_test PRIVATE Threads::Threads)
add_executable(qwen_handoff_codec_test tests/cuda/qwen_handoff_codec_test.cpp)
target_link_libraries(qwen_handoff_codec_test PRIVATE strix_qwen_handoff)
enable_testing()
add_test(NAME qwen_prefill_schedule COMMAND qwen_prefill_schedule_test)
add_test(NAME qwen_handoff_codec COMMAND qwen_handoff_codec_test)
add_executable(qwen_decode_e2e tests/cuda/qwen_decode_e2e.cu)
target_link_libraries(qwen_decode_e2e PRIVATE strix_qwen_cuda strix_qwen_handoff)
add_executable(qwen_decode_bench tests/cuda/qwen_decode_bench.cu)
target_link_libraries(qwen_decode_bench PRIVATE strix_qwen_cuda)
add_executable(strix-cuda-qwen-serve apps/cuda_qwen_serve_main.cu)
target_link_libraries(strix-cuda-qwen-serve PRIVATE strix_qwen_cuda strix_serve_win)
if(MSVC)
    target_compile_options(strix-cuda-qwen-serve PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()
add_executable(qwen_quant_host_test tests/cuda/qwen_quant_host_test.cpp src/ggml_dequant.cpp src/iq4_nl.cpp src/gguf_reader.cpp)
target_include_directories(qwen_quant_host_test PRIVATE include)
add_executable(qwen_quant_device_test tests/cuda/qwen_quant_device_test.cu src/ggml_dequant.cpp src/iq4_nl.cpp)
target_link_libraries(qwen_quant_device_test PRIVATE strix_qwen_cuda)
target_compile_options(qwen_quant_device_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
# Multi-session decode: cohort window packing (host test) and the cohort gate/bench (GPU).
add_executable(qwen_cohort_plan_test tests/cuda/qwen_cohort_plan_test.cpp)
target_include_directories(qwen_cohort_plan_test PRIVATE include)
add_test(NAME qwen_cohort_plan COMMAND qwen_cohort_plan_test)
add_executable(qwen_cohort_e2e tests/cuda/qwen_cohort_e2e.cu)
target_link_libraries(qwen_cohort_e2e PRIVATE strix_qwen_cuda strix_qwen_handoff)
# Multi-segment QSA core == per-segment layer_core on random data (device, no model): qwen_qsa_multi_test <tables dir>
add_executable(qwen_qsa_multi_test tests/cuda/qwen_qsa_multi_test.cu)
target_link_libraries(qwen_qsa_multi_test PRIVATE strix_qwen_cuda)
target_compile_options(qwen_qsa_multi_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

# Prefix storage and compatibility planning are host-only; no CUDA initialization in their tests.
add_library(strix_qwen_prefix STATIC src/prefix_state_cache.cpp src/cuda/qwen_prefix_identity.cpp)
target_include_directories(strix_qwen_prefix PUBLIC include)
target_link_libraries(strix_qwen_prefix PUBLIC strix_qwen_handoff)
target_link_libraries(strix-cuda-qwen-serve PRIVATE strix_qwen_prefix)
add_executable(qwen_prefix_cache_host_test tests/cuda/qwen_prefix_cache_host_test.cpp)
target_link_libraries(qwen_prefix_cache_host_test PRIVATE strix_qwen_prefix)
target_compile_definitions(qwen_prefix_cache_host_test PRIVATE STRIX_PREFIX_TEST_SOURCE="${CMAKE_SOURCE_DIR}")
add_test(NAME qwen_prefix_cache_host COMMAND qwen_prefix_cache_host_test)
add_executable(qwen_prefix_cache_e2e tests/cuda/qwen_prefix_cache_e2e.cu)
target_link_libraries(qwen_prefix_cache_e2e PRIVATE strix_qwen_cuda strix_qwen_prefix)
if(MSVC)
    target_compile_options(qwen_prefix_cache_e2e PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

# Explicit unqualified same-pack storage candidate. The operator target uses
# the actual production CUDA down/merge implementations from strix_qwen_cuda.
add_executable(qwen_route_bf16_operator tests/cuda/qwen_moe_tests.cu)
target_include_directories(qwen_route_bf16_operator PRIVATE tests/cuda)
target_link_libraries(qwen_route_bf16_operator PRIVATE strix_qwen_cuda)
target_compile_options(qwen_route_bf16_operator PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
add_executable(qwen_route_bf16_e2e tests/cuda/qwen_route_bf16_e2e.cu)
target_link_libraries(qwen_route_bf16_e2e PRIVATE strix_qwen_cuda)
target_compile_options(qwen_route_bf16_e2e PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    foreach(target qwen_route_bf16_operator qwen_route_bf16_e2e)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
    endforeach()
else()
    # Existing Qwen sources use the Windows spelling of 64-bit stdio.
    # POSIX aliases preserve the byte offsets and do not alter GPU arithmetic.
    target_compile_definitions(strix_qwen_cuda PUBLIC _fseeki64=fseeko _ftelli64=ftello)
    foreach(target qwen_route_bf16_operator qwen_route_bf16_e2e)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>)
    endforeach()
endif()

# Armed diagnostic: actual production Q8 proposer, BF16 target and verification.
add_executable(qwen_bf16_mtp_e2e tests/cuda/qwen_bf16_mtp_e2e.cu)
target_link_libraries(qwen_bf16_mtp_e2e PRIVATE strix_qwen_cuda)
target_compile_options(qwen_bf16_mtp_e2e PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(qwen_bf16_mtp_e2e PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
else()
    target_compile_options(qwen_bf16_mtp_e2e PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>)
endif()
