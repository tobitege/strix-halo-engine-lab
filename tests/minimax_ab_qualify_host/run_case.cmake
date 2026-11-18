execute_process(COMMAND "${EXECUTABLE}" "${CASE}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE output)
if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "${CASE} exited with ${result}:\n${output}")
endif()
if(NOT "${EXPECTED}" STREQUAL "" AND NOT "${output}" MATCHES "${EXPECTED}")
    message(FATAL_ERROR "${CASE} missing expected receipt '${EXPECTED}':\n${output}")
endif()
message("${output}")
