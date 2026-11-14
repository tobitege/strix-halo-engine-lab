# No old scheduler is recreated or extracted. Both fixtures include the actual
# templated lead/follower. This script checks the always-active negative sentinel.
execute_process(COMMAND "${BINARY}" --negative RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 10)
if(NOT rc EQUAL 43 OR NOT err STREQUAL "GLM owner negative sentinel\n" OR NOT out STREQUAL "")
    message(FATAL_ERROR "GLM negative sentinel mismatch: rc=${rc}, stdout=${out}, stderr=${err}")
endif()
