cmake_minimum_required(VERSION 3.20)
project(strix_ds41_host_pin_driver_host LANGUAGES CXX)
# Host tests of the DS41 RTX host-pin route: the real whole-layer planning route extracted from
# ds41_prefill_driver.cu (no CUDA, no model) and the prefill matrix kit's failure handling.
# Built from the top-level test tree; also configurable standalone.
enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(STRIX_HOST_PIN_BASH bash)
get_filename_component(STRIX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${STRIX_ROOT}/src/cuda/models/ds41/ds41_prefill_driver.cu")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/extract_driver.py"
    "${STRIX_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}" COMMAND_ERROR_IS_FATAL ANY)
add_executable(strix-ds41-host-pin-driver-tests driver_tests.cpp)
target_compile_features(strix-ds41-host-pin-driver-tests PRIVATE cxx_std_20)
target_include_directories(strix-ds41-host-pin-driver-tests PRIVATE "${STRIX_ROOT}/include" "${CMAKE_CURRENT_BINARY_DIR}")
if(MSVC)
    target_compile_options(strix-ds41-host-pin-driver-tests PRIVATE /W4 /WX /EHsc)
else()
    target_compile_options(strix-ds41-host-pin-driver-tests PRIVATE -Wall -Wextra -Wshadow -Wconversion -Werror)
endif()
add_test(NAME strix-ds41-host-pin-driver-tests COMMAND strix-ds41-host-pin-driver-tests)
if(STRIX_HOST_PIN_BASH)
    add_test(NAME strix-ds41-prefill-matrix-kit-tests COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/matrix_tests.py"
        "${STRIX_ROOT}" "${STRIX_HOST_PIN_BASH}" "${CMAKE_CURRENT_BINARY_DIR}/matrix-scratch")
    set_tests_properties(strix-ds41-prefill-matrix-kit-tests PROPERTIES TIMEOUT 90)
endif()
