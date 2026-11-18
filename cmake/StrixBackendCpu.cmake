# CPU backend (src/cpu): layer-2 backend of the device rule with ISA specialisations under
# src/cpu/arch/<isa>. Opt-in; nothing else in the build depends on it.
if(TARGET strix_backend_cpu)
    return()
endif()
get_filename_component(_strix_cpu_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
include(${CMAKE_CURRENT_LIST_DIR}/StrixWeight.cmake)
add_library(strix_backend_cpu STATIC
    ${_strix_cpu_root}/src/cpu/core/reference.cpp
    ${_strix_cpu_root}/src/cpu/core/select.cpp
    ${_strix_cpu_root}/src/cpu/core/routed_expert.cpp
    ${_strix_cpu_root}/src/cpu/weight/registry.cpp
    ${_strix_cpu_root}/src/cpu/arch/avx2/routed_expert_avx2.cpp
    ${_strix_cpu_root}/src/cpu/arch/avx512/kernels.cpp
    ${_strix_cpu_root}/src/cpu/arch/neon/kernels.cpp)
target_include_directories(strix_backend_cpu PUBLIC ${_strix_cpu_root}/src ${_strix_cpu_root}/include)
target_link_libraries(strix_backend_cpu PUBLIC strix-weight)
target_compile_features(strix_backend_cpu PUBLIC cxx_std_20)
if(STRIX_WFMT_CAPTURE)
    target_compile_definitions(strix_backend_cpu PUBLIC STRIX_WFMT_CAPTURE=1)
endif()
strix_target_warnings(strix_backend_cpu)
if(MSVC)
    target_compile_options(strix_backend_cpu PRIVATE /fp:precise)
else()
    target_compile_options(strix_backend_cpu PRIVATE -ffp-contract=off)
endif()
# AVX2 row kernels of the host routed expert: only this TU is compiled for AVX2 (runtime-selected by caps().avx2); the
# float chain uses separate mul/add/sub intrinsics, never FMA. AVX-VNNI is used when cpuid reports it.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|X86_64|x64")
    if(MSVC)
        set_source_files_properties(${_strix_cpu_root}/src/cpu/arch/avx2/routed_expert_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(${_strix_cpu_root}/src/cpu/arch/avx2/routed_expert_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mf16c;-mavxvnni")
    endif()
endif()
if(BUILD_TESTING)
    find_package(Threads REQUIRED)
    # The ds41 Q8_0 / A8 host reference TU only (the rest of strix_ds41_formats is not needed).
    add_executable(strix-cpu-backend-host-tests ${_strix_cpu_root}/tests/cpu_backend_host_tests.cpp ${_strix_cpu_root}/src/ds41_formats.cpp)
    target_link_libraries(strix-cpu-backend-host-tests PRIVATE strix_backend_cpu Threads::Threads)
    target_include_directories(strix-cpu-backend-host-tests PRIVATE ${_strix_cpu_root}/include)
    if(NOT MSVC)
        target_compile_options(strix-cpu-backend-host-tests PRIVATE -ffp-contract=off)
    endif()
    strix_target_warnings(strix-cpu-backend-host-tests)
    add_test(NAME strix-cpu-backend-host-tests COMMAND strix-cpu-backend-host-tests)

    # Host routed expert executor: bitwise vs the scalar reference and the verbatim CUDA-test oracle.
    # STRIX_EXP_TABLES=<dir> selects the real gfx1151 probe tables; otherwise synthetic tables are generated.
    add_executable(strix-cpu-routed-expert-tests ${_strix_cpu_root}/tests/cpu/routed_expert_host_tests.cpp)
    target_link_libraries(strix-cpu-routed-expert-tests PRIVATE strix_backend_cpu Threads::Threads)
    strix_target_warnings(strix-cpu-routed-expert-tests)
    add_test(NAME strix-cpu-routed-expert-tests COMMAND strix-cpu-routed-expert-tests)
endif()

# Microbench of the executor (not a test): scripts/bench/residency/cpu_expert_bench.cpp
add_executable(strix-cpu-expert-bench ${_strix_cpu_root}/scripts/bench/residency/cpu_expert_bench.cpp)
target_link_libraries(strix-cpu-expert-bench PRIVATE strix_backend_cpu)
find_package(Threads REQUIRED)
target_link_libraries(strix-cpu-expert-bench PRIVATE Threads::Threads)
strix_target_warnings(strix-cpu-expert-bench)
