# Small standalone host fixture: no engine/library build or device dependencies.
cmake_minimum_required(VERSION 3.20)
project(dspark_plan_host_tests LANGUAGES CXX)
enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
get_filename_component(STRIX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(PLAN_SOURCE "${STRIX_ROOT}/src/hip_deepseek41_batch_dspark.hip" CACHE FILEPATH "Production plan source")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/dspark_plan_host_tests.cpp"
    COMMAND "${Python3_EXECUTABLE}" "${STRIX_ROOT}/tests/dspark_plan_host_extract.py"
        --source "${PLAN_SOURCE}" --template "${STRIX_ROOT}/tests/dspark_plan_host_tests.cpp.in"
        --output "${CMAKE_CURRENT_BINARY_DIR}/dspark_plan_host_tests.cpp"
    DEPENDS "${PLAN_SOURCE}" "${STRIX_ROOT}/tests/dspark_plan_host_extract.py"
        "${STRIX_ROOT}/tests/dspark_plan_host_tests.cpp.in" VERBATIM)
add_executable(dspark-plan-host-tests "${CMAKE_CURRENT_BINARY_DIR}/dspark_plan_host_tests.cpp")
if(TARGET strix_spec)
    target_link_libraries(dspark-plan-host-tests PRIVATE strix_spec)
else()
    target_sources(dspark-plan-host-tests PRIVATE
        "${STRIX_ROOT}/src/spec/drafting.cpp" "${STRIX_ROOT}/src/lookup/index.cpp")
endif()
target_include_directories(dspark-plan-host-tests PRIVATE "${STRIX_ROOT}/include")
target_compile_features(dspark-plan-host-tests PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(dspark-plan-host-tests PRIVATE /W4 /WX /permissive-)
else()
    target_compile_options(dspark-plan-host-tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
add_test(NAME dspark-arbiter-serial COMMAND dspark-plan-host-tests serial)
add_test(NAME dspark-arbiter-probe COMMAND dspark-plan-host-tests probe)
add_test(NAME dspark-parked-lone-first COMMAND dspark-plan-host-tests lone-first)
add_test(NAME dspark-parked-multi-first COMMAND dspark-plan-host-tests multi-first)
