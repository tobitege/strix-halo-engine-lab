# Shared CUDA core (model-agnostic kernels: dense rows, norms, rope, decode attention, GDN decode).
# Usable from the full build (-DSTRIX_CUDA_CORE=ON) and standalone (-DSTRIX_CUDA_CORE_ONLY=ON, CUDA language only).
if(TARGET strix_cuda_core)
    return()
endif()
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/EHsc> $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
endif()
get_filename_component(_cc_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
# NVFP4 (TURBO weights) needs sm_120a PTX (cvt.e2m1x2): own object library, built for 120a only.
add_library(strix_cuda_core_fp4 OBJECT ${_cc_root}/src/cuda_core/dense/nvfp4.cu)
set_target_properties(strix_cuda_core_fp4 PROPERTIES CUDA_ARCHITECTURES "120a-real")
target_include_directories(strix_cuda_core_fp4 PUBLIC ${_cc_root}/include)
target_compile_options(strix_cuda_core_fp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(strix_cuda_core_fp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()
add_library(strix_cuda_core STATIC
    $<TARGET_OBJECTS:strix_cuda_core_fp4>
    ${_cc_root}/src/cuda_core/dense/dense.cu
    ${_cc_root}/src/cuda_core/dense/dense_mma.cu
    ${_cc_root}/src/cuda_core/dense/dense_big.cu
    ${_cc_root}/src/cuda_core/norms/norms.cu
    ${_cc_root}/src/cuda_core/norms/rope.cu
    ${_cc_root}/src/cuda_core/attention/attention.cu
    ${_cc_root}/src/cuda_core/attention/attention_flash.cu
    ${_cc_root}/src/cuda_core/attention/attention_flash8.cu
    ${_cc_root}/src/cuda_core/gdn/gdn.cu)
target_include_directories(strix_cuda_core PUBLIC ${_cc_root}/include)
# Identity: no FMA contraction, IEEE division and sqrt, no flush-to-zero (as the existing Qwen CUDA build).
target_compile_options(strix_cuda_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(strix_cuda_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/StrixDevice.cmake)
# qwen35 CUDA engine (reusable library: forward_rows / commit / MTP) + the decode CLI.
add_library(strix_qwen35_cuda STATIC ${_cc_root}/src/qwen35/cuda/engine.cu ${_cc_root}/src/gguf_reader.cpp
    ${_cc_root}/src/qwen35/config.cpp ${_cc_root}/src/qwen35/family.cpp ${_cc_root}/src/model_family_registry.cpp)
target_link_libraries(strix_qwen35_cuda PUBLIC strix_cuda_core)
target_include_directories(strix_qwen35_cuda PUBLIC ${_cc_root}/include)
target_compile_options(strix_qwen35_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(strix_qwen35_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()
add_executable(strix-cuda-qwen35-decode ${_cc_root}/apps/cuda_qwen35_decode_main.cu)
target_link_libraries(strix-cuda-qwen35-decode PRIVATE strix_qwen35_cuda strix_device)
target_compile_options(strix-cuda-qwen35-decode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(strix-cuda-qwen35-decode PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

add_executable(cuda_core_test ${_cc_root}/tests/cuda_core/cuda_core_test.cu)
target_link_libraries(cuda_core_test PRIVATE strix_cuda_core)
target_compile_options(cuda_core_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(cuda_core_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

add_executable(qwen35_cuda_engine_test ${_cc_root}/tests/qwen35/cuda_engine_test.cu)
target_link_libraries(qwen35_cuda_engine_test PRIVATE strix_qwen35_cuda)
target_compile_options(qwen35_cuda_engine_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(qwen35_cuda_engine_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/Qwen35Spec.cmake)
add_executable(cuda_core_nvfp4_test ${_cc_root}/tests/cuda_core/nvfp4_test.cu)
target_link_libraries(cuda_core_nvfp4_test PRIVATE strix_cuda_core)
target_compile_options(cuda_core_nvfp4_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(cuda_core_nvfp4_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

# Long-context prefill attention bench + pipe identity check (scripts: --check-only is the ctest form).
add_executable(cuda_core_flash_bench ${_cc_root}/tests/cuda_core/flash_bench.cu)
target_link_libraries(cuda_core_flash_bench PRIVATE strix_cuda_core)
if(MSVC)
    target_compile_options(cuda_core_flash_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()
add_executable(cuda_core_dense_big_test ${_cc_root}/tests/cuda_core/dense_big_test.cu)
target_link_libraries(cuda_core_dense_big_test PRIVATE strix_cuda_core)
target_compile_options(cuda_core_dense_big_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true;--ftz=false>)
if(MSVC)
    target_compile_options(cuda_core_dense_big_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc,/Zc:preprocessor,/utf-8>)
endif()

# Host-only: converted-pack GGUF writer round trip (no GPU).
add_executable(gguf_pack_writer_test ${_cc_root}/tests/gguf_pack_writer_test.cpp ${_cc_root}/src/gguf_reader.cpp)
target_include_directories(gguf_pack_writer_test PRIVATE ${_cc_root}/include)

if(BUILD_TESTING)
    add_test(NAME gguf_pack_writer_test COMMAND gguf_pack_writer_test)
    foreach(t cuda_core_test qwen35_cuda_engine_test cuda_core_nvfp4_test cuda_core_dense_big_test)
        add_test(NAME ${t} COMMAND ${t})
        set_tests_properties(${t} PROPERTIES LABELS "gpu;cuda")
    endforeach()
    add_test(NAME cuda_core_flash_bench COMMAND cuda_core_flash_bench --check-only)  # flash pipes bitwise identical, flash8 finite
    set_tests_properties(cuda_core_flash_bench PROPERTIES LABELS "gpu;cuda")
endif()

# FAST8 needs a feature-specific image and must not select a generic SM120
# image whose MMA helper traps. Other device generations are rejected at runtime.
get_target_property(_cc_archs strix_cuda_core CUDA_ARCHITECTURES)
if(_cc_archs MATCHES "(^|;)120a(-real)?(;|$)" AND NOT _cc_archs MATCHES "(^|;)120(-real|-virtual)?(;|$)")
    target_compile_definitions(strix_cuda_core PRIVATE STRIX_CUDA_FAST8_SM120A=1)
endif()
