# Check diagnostics as well as status: an eventual invalid plan is not a parsing rejection.
function(reject option value)
    execute_process(COMMAND "${PLANNER}" --rows 1 --card-rate 4320 --node-rate 1600
        --link-mbps 118 --decoder card "${option}" "${value}"
        RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT "${status}" STREQUAL "1" OR NOT error MATCHES "bad value for ${option}")
        message(FATAL_ERROR "${option} '${value}': expected parse rejection, got ${status}: ${error}")
    endif()
endfunction()
foreach(value IN ITEMS inf 1e100 18446744073709551616 1.5 -1 "" 1x)
    reject(--rows "${value}")
endforeach()
foreach(value IN ITEMS 4294967296 1.5 inf "" "1," ",1")
    reject(--cuts "${value}")
endforeach()
foreach(value IN ITEMS 4294967296 1.5 inf "" "1," ",1")
    reject(--blocks "${value}")
endforeach()
foreach(option IN ITEMS --card-rate --node-rate --link-mbps --pair-mbps --latency-ms --card-fixed-ms --node-fixed-ms)
    foreach(value IN ITEMS inf nan 1e1000 "" -1 1x)
        reject("${option}" "${value}")
    endforeach()
endforeach()
# Valid uint64 input must reach the work guard without a floating conversion.
execute_process(COMMAND "${PLANNER}" --rows 18446744073709551615 --card-rate 4320
    --node-rate 1600 --split device-only --blocks 2 --decoder card
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT "${status}" STREQUAL "2" OR NOT output MATCHES "plan work limit exceeded")
    message(FATAL_ERROR "uint64 max did not reach planning guard: ${status}: ${error}")
endif()
