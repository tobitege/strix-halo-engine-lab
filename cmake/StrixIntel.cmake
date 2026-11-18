# Intel GPU backend (SYCL / oneAPI over Level Zero), opt-in with STRIX_BACKEND_INTEL (default OFF).
# Host part (always built when the option is on): the xpu-smi device listing, the backend interface with its no-SYCL
# stubs, the plain C++ reference kernel and their host tests. Device part: only when the C++ compiler is the oneAPI
# DPC++ compiler (icpx / icx, CMAKE_CXX_COMPILER_ID IntelLLVM) and -fsycl compiles a SYCL program; otherwise it stays
# off with a status line. Nothing here touches the HIP or CUDA configuration.
if(TARGET strix_intel_core)
    return()
endif()
include(${CMAKE_CURRENT_LIST_DIR}/StrixDevice.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/StrixWeight.cmake)
set(_strix_intel_root "${CMAKE_CURRENT_LIST_DIR}/..")

set(STRIX_INTEL_SYCL OFF)
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "-fsycl")
    check_cxx_source_compiles("#include <sycl/sycl.hpp>\nint main() { sycl::queue q; return 0; }" STRIX_INTEL_SYCL_COMPILES)
    unset(CMAKE_REQUIRED_FLAGS)
    if(STRIX_INTEL_SYCL_COMPILES)
        set(STRIX_INTEL_SYCL ON)
    endif()
endif()
if(STRIX_INTEL_SYCL)
    message(STATUS "Intel backend: SYCL device code enabled (${CMAKE_CXX_COMPILER})")
else()
    find_program(STRIX_ICPX_PROGRAM NAMES icpx icx)
    if(STRIX_ICPX_PROGRAM)
        message(STATUS "Intel backend: host only; oneAPI compiler found at ${STRIX_ICPX_PROGRAM}: configure with CMAKE_CXX_COMPILER set to it for SYCL")
    else()
        message(STATUS "Intel backend: host only (no oneAPI DPC++ compiler)")
    endif()
endif()

add_library(strix_device_intel STATIC ${_strix_intel_root}/src/device/xpu_smi.cpp)
target_link_libraries(strix_device_intel PUBLIC strix_device)
target_compile_features(strix_device_intel PUBLIC cxx_std_20)
if(MSVC)
    target_compile_definitions(strix_device_intel PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

add_library(strix_intel_core STATIC ${_strix_intel_root}/src/intel/core/backend.cpp
    ${_strix_intel_root}/src/intel/weight/registry.cpp)
target_include_directories(strix_intel_core PUBLIC ${_strix_intel_root}/src ${_strix_intel_root}/include)
target_compile_features(strix_intel_core PUBLIC cxx_std_20)
target_link_libraries(strix_intel_core PUBLIC strix_device_intel strix-weight)
if(STRIX_INTEL_SYCL)
    target_compile_definitions(strix_intel_core PUBLIC STRIX_HAVE_SYCL=1)
    target_compile_options(strix_intel_core PUBLIC -fsycl)
    target_link_options(strix_intel_core PUBLIC -fsycl)
endif()
foreach(_t strix_device_intel strix_intel_core)
    if(COMMAND strix_target_warnings)
        strix_target_warnings(${_t})
    endif()
endforeach()

if(BUILD_TESTING)
    add_executable(strix-intel-backend-host-tests ${_strix_intel_root}/tests/intel_backend_host_tests.cpp)
    target_link_libraries(strix-intel-backend-host-tests PRIVATE strix_intel_core)
    if(COMMAND strix_target_warnings)
        strix_target_warnings(strix-intel-backend-host-tests)
    endif()
    add_test(NAME strix-intel-backend-host-tests COMMAND strix-intel-backend-host-tests)
    add_test(NAME strix-intel-dense-boundaries COMMAND strix-intel-backend-host-tests dense)
    add_test(NAME strix-intel-memory-boundaries COMMAND strix-intel-backend-host-tests memory)

    # Compile the real exception boundary with a host-only SYCL fault double, never -fsycl.
    add_executable(strix-intel-sycl-boundary-tests
        ${_strix_intel_root}/tests/intel_backend_sycl_failures.cpp
        ${_strix_intel_root}/tests/wfmt/intel_queue.cpp
        ${_strix_intel_root}/src/intel/weight/registry.cpp
        ${_strix_intel_root}/src/intel/core/backend.cpp)
    target_include_directories(strix-intel-sycl-boundary-tests BEFORE PRIVATE
        ${_strix_intel_root}/tests/intel_sycl_stub ${_strix_intel_root}/src ${_strix_intel_root}/include)
    target_compile_definitions(strix-intel-sycl-boundary-tests PRIVATE STRIX_HAVE_SYCL=1)
    target_compile_features(strix-intel-sycl-boundary-tests PRIVATE cxx_std_20)
    target_link_libraries(strix-intel-sycl-boundary-tests PRIVATE strix_device_intel strix-weight)
    if(COMMAND strix_target_warnings)
        strix_target_warnings(strix-intel-sycl-boundary-tests)
    endif()
    foreach(_case async sync open queues)
        add_test(NAME strix-intel-sycl-${_case}-failures COMMAND strix-intel-sycl-boundary-tests ${_case})
    endforeach()
endif()
