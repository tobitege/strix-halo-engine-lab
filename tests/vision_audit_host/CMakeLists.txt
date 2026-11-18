cmake_minimum_required(VERSION 3.20)
project(vision_audit_host LANGUAGES CXX)
enable_testing()
get_filename_component(STRIX_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(STRIX_AUDIT_PROJECTOR_SOURCE "${STRIX_ROOT}/src/vision/projector.cpp" CACHE FILEPATH "Projector source under test")
add_executable(vision-audit-host ../vision_encoder_tests.cpp
    "${STRIX_ROOT}/src/vision/reference.cpp"
    "${STRIX_AUDIT_PROJECTOR_SOURCE}"
    "${STRIX_ROOT}/src/vision/embed_cache.cpp")
target_include_directories(vision-audit-host PRIVATE "${STRIX_ROOT}/include" "${STRIX_ROOT}/src/vision")
target_compile_features(vision-audit-host PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(vision-audit-host PRIVATE /W4 /WX)
    target_compile_definitions(vision-audit-host PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(vision-audit-host PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -ffp-contract=off -fno-fast-math)
endif()
find_package(Threads REQUIRED)
target_link_libraries(vision-audit-host PRIVATE Threads::Threads)
add_test(NAME vision-audit-host COMMAND vision-audit-host)
add_test(NAME vision-projector-policy-default COMMAND ${CMAKE_COMMAND} -E env
    --unset=STRIX_VISION_PROJECT_DEVICE $<TARGET_FILE:vision-audit-host> --project-device off)
add_test(NAME vision-projector-policy-off COMMAND ${CMAKE_COMMAND} -E env
    STRIX_VISION_PROJECT_DEVICE=0 $<TARGET_FILE:vision-audit-host> --project-device off)
add_test(NAME vision-projector-policy-on COMMAND ${CMAKE_COMMAND} -E env
    STRIX_VISION_PROJECT_DEVICE=1 $<TARGET_FILE:vision-audit-host> --project-device on)
