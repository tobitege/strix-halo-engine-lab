cmake_minimum_required(VERSION 3.24)
project(StrixPackagingRegressions NONE)
enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(STRIX_TEST_BASH bash REQUIRED)
if(WIN32)
    find_program(STRIX_TEST_POWERSHELL NAMES powershell pwsh REQUIRED)
endif()
set(STRIX_PACKAGE_REGRESSION_BASE "" CACHE STRING "Optional original revision for regression sensitivity checks")
set(_versions fixed)
if(STRIX_PACKAGE_REGRESSION_BASE)
    list(APPEND _versions original)
endif()
foreach(finding RANGE 1 10)
    if(NOT WIN32 AND finding MATCHES "^(1|4|7|9|10)$")
        continue()
    endif()
    foreach(version IN LISTS _versions)
        add_test(NAME package-685-${finding}-${version}
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/regressions.py"
                --root "${CMAKE_CURRENT_LIST_DIR}/../.." --work "${CMAKE_CURRENT_BINARY_DIR}"
                --bash "${STRIX_TEST_BASH}" --powershell "${STRIX_TEST_POWERSHELL}"
                --cmake "${CMAKE_COMMAND}" --finding ${finding} --version ${version}
                --original-ref "${STRIX_PACKAGE_REGRESSION_BASE}")
        set_tests_properties(package-685-${finding}-${version} PROPERTIES TIMEOUT 90 LABELS "cpu;packaging")
    endforeach()
endforeach()
