# Lever governance: the typed lever declaration (include/strix/lever.hpp) and the census of STRIX_* switches.
# Host-only; included by the full build and by the standalone CUDA Qwen configuration. The census test fails when a switch is read with a raw getenv that is not in
# tools/env_switch_baseline.txt, or when one boolean switch has two governed defaults.
if(NOT BUILD_TESTING OR TARGET strix-lever-tests)
    return()
endif()
add_executable(strix-lever-tests ${CMAKE_CURRENT_LIST_DIR}/../tests/lever_tests.cpp)
target_include_directories(strix-lever-tests PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../include)
target_compile_features(strix-lever-tests PRIVATE cxx_std_20)
if(MSVC)
    target_compile_definitions(strix-lever-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
if(COMMAND strix_target_warnings)
    strix_target_warnings(strix-lever-tests)
endif()
add_test(NAME strix-lever-tests COMMAND strix-lever-tests)
set_tests_properties(strix-lever-tests PROPERTIES LABELS "cpu" TIMEOUT 30)
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME strix-env-switch-census COMMAND "${Python3_EXECUTABLE}" -B
        "${CMAKE_CURRENT_LIST_DIR}/../tools/env_switch_census.py" --check --root "${CMAKE_CURRENT_LIST_DIR}/..")
    set_tests_properties(strix-env-switch-census PROPERTIES LABELS "cpu" TIMEOUT 120)
    add_test(NAME strix-env-switch-census-self-test COMMAND "${Python3_EXECUTABLE}" -B
        "${CMAKE_CURRENT_LIST_DIR}/../tools/env_switch_census.py" --self-test)
    set_tests_properties(strix-env-switch-census-self-test PROPERTIES LABELS "cpu" TIMEOUT 30)
    add_test(NAME strix-env-switch-census-cli-tests COMMAND "${Python3_EXECUTABLE}" -B
        "${CMAKE_CURRENT_LIST_DIR}/../tests/env_switch_census_tests.py")
    set_tests_properties(strix-env-switch-census-cli-tests PROPERTIES LABELS "cpu" TIMEOUT 30)
endif()
