# Host-only fixtures extracted from production CUDA control flow.
set(production_sources
    "${PROJECT_SOURCE_DIR}/src/cuda/models/qwen/qwen_prefill_driver.cu"
    "${PROJECT_SOURCE_DIR}/src/cuda/models/qwen/qwen_qsa.cu")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/production.inc"
    COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/extract_host.py"
        "${CMAKE_CURRENT_BINARY_DIR}/production.inc"
    DEPENDS extract_host.py ${production_sources}
    VERBATIM)
add_executable(strix-qwen-cuda-graph-host-tests host.cpp "${CMAKE_CURRENT_BINARY_DIR}/production.inc")
target_compile_features(strix-qwen-cuda-graph-host-tests PRIVATE cxx_std_20)
target_include_directories(strix-qwen-cuda-graph-host-tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/include")
if(MSVC)
    target_compile_options(strix-qwen-cuda-graph-host-tests PRIVATE /W4 /WX /permissive-)
else()
    target_compile_options(strix-qwen-cuda-graph-host-tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
foreach(case capture pool)
    add_test(NAME strix-qwen-cuda-graph-${case}-tests COMMAND strix-qwen-cuda-graph-host-tests ${case})
endforeach()
add_test(NAME strix-qwen-serve-bench-kit-tests
    COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/test_benchmark.py")
