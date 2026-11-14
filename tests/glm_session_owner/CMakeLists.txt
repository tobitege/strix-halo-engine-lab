get_filename_component(GLM_OWNER_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
add_library(glm_owner_native_fixture STATIC
    "${GLM_OWNER_ROOT}/src/glm/host/glm_host.cpp"
    "${GLM_OWNER_ROOT}/src/glm/spec/glm_spec_host.cpp"
    "${GLM_OWNER_ROOT}/src/glm/spec/glm_spec_merge.cpp")
target_include_directories(glm_owner_native_fixture PUBLIC "${GLM_OWNER_ROOT}/src" "${GLM_OWNER_ROOT}/include")
target_compile_features(glm_owner_native_fixture PUBLIC cxx_std_20)
add_executable(glm_session_owner_tests owner_tests.cpp)
add_executable(glm_owner_existing_cpu_tests "${GLM_OWNER_ROOT}/tests/glm_serve_cpu_tests.cpp")
foreach(target glm_session_owner_tests glm_owner_existing_cpu_tests)
    target_link_libraries(${target} PRIVATE glm_owner_native_fixture strix_serve strix_glm_session_owner)
endforeach()

# An explicit host-only compatibility include can be supplied for MSVC's absent
# compiler builtins. It is never propagated to the product or device targets.
set(GLM_OWNER_MSVC_COMPAT "" CACHE FILEPATH "Optional non-strict MSVC host compatibility include")
foreach(target glm_owner_native_fixture glm_session_owner_tests glm_owner_existing_cpu_tests)
    strix_session_warnings(${target})
    if(WIN32)
        target_compile_definitions(${target} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    endif()
    if(MSVC)
        target_compile_options(${target} PRIVATE /utf-8)
        if(GLM_OWNER_MSVC_COMPAT)
            target_compile_options(${target} PRIVATE "/FI${GLM_OWNER_MSVC_COMPAT}")
        endif()
    endif()
endforeach()
add_test(NAME glm-owner-replay COMMAND glm_session_owner_tests)
add_test(NAME glm-owner-lifetime COMMAND glm_session_owner_tests --lifetime)
add_test(NAME glm-owner-http COMMAND glm_session_owner_tests --http "${CMAKE_CURRENT_BINARY_DIR}/http.raw")
add_test(NAME glm-owner-existing-cpu COMMAND glm_owner_existing_cpu_tests)
add_test(NAME glm-owner-checks COMMAND "${CMAKE_COMMAND}"
    "-DBINARY=$<TARGET_FILE:glm_session_owner_tests>" -P "${CMAKE_CURRENT_LIST_DIR}/extract_reference.cmake")
set_tests_properties(glm-owner-replay glm-owner-checks glm-owner-lifetime glm-owner-http PROPERTIES TIMEOUT 30 LABELS "cpu;session;glm-owner")
set_tests_properties(glm-owner-existing-cpu PROPERTIES TIMEOUT 80 LABELS "cpu;session;glm-owner;existing")
