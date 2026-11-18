cmake_minimum_required(VERSION 3.24)
project(strix_profile_host_checks LANGUAGES CXX)

# Isolated host gate: no device language, model assets or platform transport library.
# The full repository build and original engine gates remain separate requirements.
include(CTest)
get_filename_component(STRIX_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

function(profile_host_test target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE "${STRIX_ROOT}/include" "${STRIX_ROOT}/src" "${STRIX_ROOT}/apps")
    target_compile_features(${target} PRIVATE cxx_std_20)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX /utf-8)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS=)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
    endif()
    add_test(NAME ${target} COMMAND ${target} ${ARG_TEST_ARGS})
    set_tests_properties(${target} PROPERTIES LABELS "cpu;profile" TIMEOUT 60)
endfunction()

profile_host_test(strix-engine-profile-common-tests "${STRIX_ROOT}/tests/engine_profile_common_tests.cpp")
profile_host_test(strix-qwen-allon-tests "${STRIX_ROOT}/tests/qwen_allon_tests.cpp")
profile_host_test(strix-glm5-rank-defaults-common-cpu-tests "${STRIX_ROOT}/tests/glm_rank_defaults_common_cpu_tests.cpp")

set(ARG_TEST_ARGS "${STRIX_ROOT}/tests/fixtures/minimax_production_receipt.txt")
profile_host_test(strix-minimax-defaults-tests "${STRIX_ROOT}/tests/minimax_defaults_tests.cpp"
    "${STRIX_ROOT}/src/minimax/engine/mm_defaults.cpp")
unset(ARG_TEST_ARGS)
profile_host_test(strix-minimax-route-identity-tests "${STRIX_ROOT}/tests/minimax_route_identity_tests.cpp"
    "${STRIX_ROOT}/src/minimax/engine/mm_defaults.cpp")
add_test(NAME strix-minimax-route-source-audit COMMAND strix-minimax-route-identity-tests --audit-sources)
set_tests_properties(strix-minimax-route-source-audit PROPERTIES LABELS "cpu;profile" TIMEOUT 60)
target_compile_definitions(strix-minimax-route-identity-tests PRIVATE STRIX_ROUTE_SOURCE_DIR="${STRIX_ROOT}")

profile_host_test(strix-deepseek41-pair-defaults-tests "${STRIX_ROOT}/tests/deepseek41_pair_defaults_tests.cpp"
    "${STRIX_ROOT}/src/deepseek41_pair_defaults.cpp")
add_test(NAME strix-deepseek41-dspark-inspection
    COMMAND "${CMAKE_COMMAND}" "-DTEST_EXE=$<TARGET_FILE:strix-deepseek41-pair-defaults-tests>"
        -P "${CMAKE_CURRENT_LIST_DIR}/dspark_inspection.cmake")
profile_host_test(strix-ds41-rtx-profile-tests "${STRIX_ROOT}/tests/ds41_rtx_profile_tests.cpp"
    "${STRIX_ROOT}/src/ds41_rtx_profile.cpp")
profile_host_test(strix-gemma4-profile-tests "${STRIX_ROOT}/tests/gemma4_profile/profile_tests.cpp"
    "${STRIX_ROOT}/src/gemma4/engine/profile.cpp")
profile_host_test(strix-gemma4-profile-migration-tests "${STRIX_ROOT}/tests/gemma4_profile/migration_tests.cpp"
    "${STRIX_ROOT}/src/gemma4/engine/profile.cpp")

add_executable(strix-optimization-table "${STRIX_ROOT}/tools/models/optimization_table.cpp"
    "${STRIX_ROOT}/src/minimax/engine/mm_defaults.cpp" "${STRIX_ROOT}/src/deepseek41_pair_defaults.cpp")
target_include_directories(strix-optimization-table PRIVATE "${STRIX_ROOT}/include" "${STRIX_ROOT}/src" "${STRIX_ROOT}/apps")
target_compile_features(strix-optimization-table PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(strix-optimization-table PRIVATE /W4 /WX /utf-8)
    target_compile_definitions(strix-optimization-table PRIVATE _CRT_SECURE_NO_WARNINGS=)
else()
    target_compile_options(strix-optimization-table PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
endif()
add_test(NAME strix-optimization-table-current COMMAND strix-optimization-table --check "${STRIX_ROOT}/docs/engine/optimization-table.md")
set_tests_properties(strix-optimization-table-current PROPERTIES LABELS "cpu;profile" TIMEOUT 60)

# Exercise the real shared cluster resolvers together without device toolchains.
profile_host_test(strix-cluster-serve-cli-parity-tests "${STRIX_ROOT}/tests/serve_cli_parity_tests.cpp"
    "${STRIX_ROOT}/src/minimax/engine/mm_defaults.cpp" "${STRIX_ROOT}/src/deepseek41_pair_defaults.cpp")
target_compile_definitions(strix-cluster-serve-cli-parity-tests PRIVATE
    STRIX_PARITY_GLM=1 STRIX_PARITY_MINIMAX=1 STRIX_PARITY_DS41=1)
