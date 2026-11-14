# Host session resources and storage are independent of accelerator languages
# and strix_core. Standalone test projects retain their own entry points.
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/StrixStateCache.cmake")
get_filename_component(_strix_session_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
# Host replay projects retain a configure-time copy of this production source.
# Regenerate that copy after queue API changes instead of linking a stale copy
# against current session headers. No frozen expectations are regenerated.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_strix_session_root}/src/serve/engine.cpp")
function(strix_session_warnings target)
    if(COMMAND strix_target_warnings)
        strix_target_warnings(${target})
    elseif(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()
    if(MSVC)
        target_compile_options(${target} PRIVATE /EHsc /permissive-)
    endif()
endfunction()

# Host fixtures are written densely; keep their style out of the error set
# while every other warning stays an error.
function(strix_session_test_style target)
    if(NOT MSVC)
        target_compile_options(${target} PRIVATE -Wno-misleading-indentation -Wno-unused-function)
    endif()
endfunction()

add_library(strix_session STATIC
    "${_strix_session_root}/src/session/backend_cpu.cpp"
    "${_strix_session_root}/src/session/embed_rows.cpp"
    "${_strix_session_root}/src/session/kv.cpp"
    "${_strix_session_root}/src/session/prefix_cache.cpp"
    "${_strix_session_root}/src/session/parallel_job.cpp")
target_include_directories(strix_session PUBLIC "${_strix_session_root}/include")
target_compile_features(strix_session PUBLIC cxx_std_20)
target_link_libraries(strix_session PUBLIC strix_state_cache)
set_target_properties(strix_session PROPERTIES CXX_EXTENSIONS OFF)
strix_session_warnings(strix_session)
# The execution owner has no dependency on an HTTP implementation. Replay's
# observed queue and production's queue both link this same host owner archive.
add_library(strix_session_owner STATIC "${_strix_session_root}/src/serve/session_runtime.cpp"
    "${_strix_session_root}/src/serve/serial_session_owner.cpp")
target_link_libraries(strix_session_owner PUBLIC strix_session)
strix_session_warnings(strix_session_owner)

# Central host contract/test registration. Add future cost/config (PR2),
# protocol (PR3) and runner (PR5) implementations to strix_session above and
# their real tests to this list when present. Missing future APIs are pending;
# do not register placeholder executables or empty passing tests.
function(strix_session_add_contract_tests)
    if(NOT BUILD_TESTING)
        return()
    endif()
    foreach(test engine_contract backend kv prefix_cache recurrent fanout parallel_job cache_owner)
        add_executable(session_${test}_tests "${_strix_session_root}/tests/session_${test}_tests.cpp")
        target_link_libraries(session_${test}_tests PRIVATE strix_session)
        strix_session_warnings(session_${test}_tests)
        strix_session_test_style(session_${test}_tests)
        if(MSVC)
            target_compile_options(session_${test}_tests PRIVATE /UNDEBUG)
        else()
            target_compile_options(session_${test}_tests PRIVATE -UNDEBUG)
        endif()
        add_test(NAME session_${test} COMMAND session_${test}_tests)
        set_tests_properties(session_${test} PROPERTIES LABELS "cpu;session" TIMEOUT 30)
    endforeach()
endfunction()
strix_session_add_contract_tests()

# Actual MiniMax continuation target; also used by the standalone host project.
function(strix_session_add_minimax_continuation)
    add_executable(minimax_continuation_tests
        "${_strix_session_root}/tests/minimax_session_shared/continuation_tests.cpp"
        "${CMAKE_CURRENT_BINARY_DIR}/capture_noop.cpp")
    target_link_libraries(minimax_continuation_tests PRIVATE minimax_replay_family minimax_owner_cache)
    add_test(NAME minimax-continuation COMMAND minimax_continuation_tests)
    set_tests_properties(minimax-continuation PROPERTIES TIMEOUT 90 LABELS "cpu;session;minimax")
endfunction()

# Resident prefix target uses the production owner/control/adapter seams.
function(strix_session_add_minimax_prefix_reuse)
    add_executable(minimax_prefix_reuse_tests
        "${_strix_session_root}/tests/minimax_session_shared/prefix_reuse_tests.cpp"
        "${CMAKE_CURRENT_BINARY_DIR}/capture_noop.cpp")
    target_link_libraries(minimax_prefix_reuse_tests PRIVATE minimax_replay_family minimax_owner_cache)
    add_test(NAME minimax-prefix-reuse COMMAND minimax_prefix_reuse_tests)
    set_tests_properties(minimax-prefix-reuse PROPERTIES TIMEOUT 90 LABELS "cpu;session;minimax")
endfunction()

# Called once the existing production HTTP and registry targets exist. Keeping
# this separate leaves the session resources independent of serving/core and
# keeps the earlier minimal accelerator configure independent of the shell.
function(strix_session_add_runtime)
    add_library(strix_qwen_http_session_owner STATIC "${_strix_session_root}/src/qwen/http_session_owner.cpp")
    target_include_directories(strix_qwen_http_session_owner PUBLIC "${_strix_session_root}/src")
    target_link_libraries(strix_qwen_http_session_owner PUBLIC strix_session_owner)
    strix_session_warnings(strix_qwen_http_session_owner)
    add_library(strix_session_runtime INTERFACE)
    target_link_libraries(strix_session_runtime INTERFACE strix_session_owner strix_serve strix_model_registry)
    add_library(strix_session_shell STATIC "${_strix_session_root}/src/serve/run.cpp")
    target_link_libraries(strix_session_shell PUBLIC strix_session_runtime)
    strix_session_warnings(strix_session_shell)
    add_executable(strix-serve "${_strix_session_root}/apps/strix_serve_main.cpp")
    target_link_libraries(strix-serve PRIVATE strix_session_shell)
    strix_session_warnings(strix-serve)
    if(BUILD_TESTING)
        foreach(test runtime http_replay registry turn_cache)
            add_executable(session_${test}_tests "${_strix_session_root}/tests/session_${test}_tests.cpp")
            target_link_libraries(session_${test}_tests PRIVATE strix_session_shell)
            strix_session_warnings(session_${test}_tests)
            strix_session_test_style(session_${test}_tests)
            add_test(NAME session_${test} COMMAND session_${test}_tests "${_strix_session_root}")
            set_tests_properties(session_${test} PROPERTIES LABELS "cpu;session" TIMEOUT 30)
        endforeach()
    endif()
    if(BUILD_TESTING AND NOT TARGET qwen_session_owner_tests)
        add_subdirectory("${_strix_session_root}/tests/qwen_session_owner" "${CMAKE_BINARY_DIR}/qwen-session-owner")
    endif()
    if(BUILD_TESTING AND NOT TARGET serial_session_owner_tests)
        add_subdirectory("${_strix_session_root}/tests/serial_session_owner" "${CMAKE_BINARY_DIR}/serial-session-owner")
    endif()
    if(BUILD_TESTING AND STRIX_FAMILY_MINIMAX)
        add_subdirectory("${_strix_session_root}/tests/minimax_session_shared"
            "${CMAKE_BINARY_DIR}/minimax-session-shared")
    endif()
    if(BUILD_TESTING AND NOT TARGET http_shutdown_tests)
        add_subdirectory("${_strix_session_root}/tests/http_shutdown" "${CMAKE_BINARY_DIR}/http-shutdown")
    endif()
endfunction()

# Explicit optional archive reference without a runtime/family dependency cycle.
function(strix_session_link_services)
    if(TARGET strix_minimax_session)
        set(checkpoint_minimax strix_minimax_session)
    elseif(TARGET minimax_owner_cache)
        set(checkpoint_minimax minimax_owner_cache)
    endif()
    if(BUILD_TESTING AND checkpoint_minimax
       AND EXISTS "${_strix_session_root}/tests/session_checkpoint_io_tests.cpp"
       AND EXISTS "${_strix_session_root}/src/session/checkpoint_io.cpp"
       AND NOT TARGET session_checkpoint_io_tests)
        add_executable(session_checkpoint_io_tests "${_strix_session_root}/tests/session_checkpoint_io_tests.cpp")
        target_link_libraries(session_checkpoint_io_tests PRIVATE strix_state_cache ${checkpoint_minimax})
        strix_session_warnings(session_checkpoint_io_tests)
        if(MSVC)
            target_compile_options(session_checkpoint_io_tests PRIVATE /UNDEBUG)
        else()
            target_compile_options(session_checkpoint_io_tests PRIVATE -UNDEBUG -Wno-missing-field-initializers)
        endif()
        set(STRIX_CHECKPOINT_REFERENCE_DIR "" CACHE PATH "Optional independently archived host checkpoint reference files")
        add_test(NAME session_checkpoint_io COMMAND session_checkpoint_io_tests "${STRIX_CHECKPOINT_REFERENCE_DIR}")
        set_tests_properties(session_checkpoint_io PROPERTIES TIMEOUT 120 LABELS "cpu;session;checkpoint")
    elseif(BUILD_TESTING AND checkpoint_minimax AND NOT TARGET session_checkpoint_io_tests)
        message(STATUS "Checkpoint adapter test source pending; no placeholder test registered")
    endif()
    if(BUILD_TESTING AND TARGET minimax_replay_family AND NOT MINIMAX_SESSION_BASELINE
       AND EXISTS "${_strix_session_root}/tests/minimax_session_shared/shared_api_tests.cpp"
       AND NOT TARGET minimax_shared_api_tests)
        add_executable(minimax_shared_api_tests
            "${_strix_session_root}/tests/minimax_session_shared/shared_api_tests.cpp"
            "${CMAKE_BINARY_DIR}/minimax-session-shared/capture_noop.cpp")
        target_link_libraries(minimax_shared_api_tests PRIVATE minimax_replay_family minimax_owner_cache)
        add_test(NAME minimax-shared-api COMMAND minimax_shared_api_tests)
        set_tests_properties(minimax-shared-api PROPERTIES TIMEOUT 120 LABELS "cpu;session;minimax;api")
    endif()
    if(TARGET strix_minimax_session_service AND TARGET strix_session_shell)
        target_compile_definitions(strix_session_shell PRIVATE STRIX_SERVE_MINIMAX=1)
        target_link_libraries(strix_session_shell PRIVATE strix_minimax_session_service)
        if(TARGET session_registry_tests)
            target_compile_definitions(session_registry_tests PRIVATE STRIX_EXPECT_MINIMAX_SERVICE=1)
        endif()
    endif()
    if(TARGET strix_gemma4_session_service AND TARGET strix_session_shell)
        target_compile_definitions(strix_session_shell PRIVATE STRIX_SERVE_GEMMA4=1)
        target_link_libraries(strix_session_shell PRIVATE strix_gemma4_session_service)
        if(TARGET session_registry_tests)
            target_compile_definitions(session_registry_tests PRIVATE STRIX_EXPECT_GEMMA4_SERVICE=1)
        endif()
    endif()
    # Explicit native operator target only: no accelerator language is enabled
    # here, no GPU test enters CTest, and ordinary/legacy construction stays OFF.
    if(BUILD_TESTING AND TARGET strix_minimax_engine AND TARGET strix_minimax_session_service)
        add_executable(minimax-native-paired-state EXCLUDE_FROM_ALL
            "${_strix_session_root}/tests/minimax_session_shared/native_paired_state_tests.cpp")
        target_link_libraries(minimax-native-paired-state PRIVATE strix_minimax_session_service
            strix_minimax_engine strix_minimax_dspark strix_dist)
        target_compile_features(minimax-native-paired-state PRIVATE cxx_std_20)
        strix_session_warnings(minimax-native-paired-state)
    endif()
    # A device-backed service makes HIP the link language of every shell
    # consumer; these targets were declared before the HIP target list.
    if(STRIX_ENABLE_HIP_RUNTIME AND DEFINED STRIX_HIP_ARCHS
       AND (TARGET strix_gemma4_session_service OR TARGET strix_minimax_session_service))
        foreach(_t strix_session_shell strix-serve strix-skeleton-serve skeleton_engine_tests
                session_runtime_tests session_http_replay_tests session_registry_tests session_turn_cache_tests
                minimax_service_tests minimax_service_compat minimax-native-paired-state)
            if(TARGET ${_t})
                set_target_properties(${_t} PROPERTIES HIP_ARCHITECTURES "${STRIX_HIP_ARCHS}")
            endif()
        endforeach()
    endif()
endfunction()
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL strix_session_link_services)

# The runtime paths and optional existing launcher target are defined later by
# the root build. No dependency in the reverse direction through strix_core.
function(strix_session_add_hip)
    if(NOT STRIX_ENABLE_HIP_RUNTIME OR STRIX_CUDA_QWEN_ONLY)
        return()
    endif()
    add_library(strix_session_hip STATIC "${_strix_session_root}/src/session/backend_hip.cpp")
    target_link_libraries(strix_session_hip PUBLIC strix_session PRIVATE "${STRIX_HIP_RUNTIME_LIBRARY}")
    target_include_directories(strix_session_hip SYSTEM PRIVATE "${STRIX_HIP_INCLUDE_DIR}")
    target_compile_definitions(strix_session_hip PRIVATE __HIP_PLATFORM_AMD__=1)
    target_compile_features(strix_session_hip PUBLIC cxx_std_20)
    if(TARGET strix_ds41_trellis_hip)
        target_link_libraries(strix_session_hip PRIVATE strix_ds41_trellis_hip)
        target_compile_definitions(strix_session_hip PRIVATE STRIX_SESSION_HIP_ADD=1)
    endif()
    if(COMMAND strix_device_add_hip_gate)
        strix_device_add_hip_gate()
        target_link_libraries(strix_session_hip PUBLIC strix_device_hip)
        target_compile_definitions(strix_session_hip PRIVATE STRIX_SESSION_DEVICE_GATE=1)
    endif()
    strix_session_warnings(strix_session_hip)
endfunction()
cmake_language(DEFER CALL strix_session_add_hip)
