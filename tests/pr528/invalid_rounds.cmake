execute_process(COMMAND "${PROGRAM}" --bench-concurrent synthetic 100 "${ROUNDS}"
    RESULT_VARIABLE result ERROR_VARIABLE error OUTPUT_QUIET TIMEOUT 10)
if(NOT result STREQUAL "1" OR NOT error MATCHES "ROUNDS must be positive")
    message(FATAL_ERROR "Expected argument error, got ${result}: ${error}")
endif()
