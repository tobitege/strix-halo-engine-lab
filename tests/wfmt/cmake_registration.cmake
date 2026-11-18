string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
set(build "${BINARY_ROOT}/registration-${suffix}")
foreach(pass fresh reconfigure)
    execute_process(COMMAND "${CMAKE_COMMAND}" -S "${CMAKE_CURRENT_LIST_DIR}/registration"
        -B "${build}" -G "${GENERATOR}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "${pass} registration failed: ${output}\n${error}")
    endif()
endforeach()
