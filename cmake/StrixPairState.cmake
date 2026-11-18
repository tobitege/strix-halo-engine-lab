# Prompt-state hand-off planner shared by every family (include/strix/pair/state_owners.hpp, state_decls.hpp,
# state_relay.hpp): one copy per owner, relay inside the pair, prefix first. Host only, header only.
if(NOT BUILD_TESTING OR TARGET strix-pair-state-owners-tests)
    return()
endif()
add_executable(strix-pair-state-owners-tests ${CMAKE_CURRENT_LIST_DIR}/../tests/pair_state_owners_tests.cpp)
target_include_directories(strix-pair-state-owners-tests PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../include)
target_compile_features(strix-pair-state-owners-tests PRIVATE cxx_std_20)
if(MSVC)
    target_compile_definitions(strix-pair-state-owners-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
if(COMMAND strix_target_warnings)
    strix_target_warnings(strix-pair-state-owners-tests)
endif()
add_test(NAME strix-pair-state-owners-tests COMMAND strix-pair-state-owners-tests)
set_tests_properties(strix-pair-state-owners-tests PROPERTIES LABELS "cpu" TIMEOUT 30)

# Pin actual serialized traffic, separately from the declaration's tensor capacity estimates.
add_executable(pair_state_serialized_tests
    ${CMAKE_CURRENT_LIST_DIR}/../tests/pair_state_serialized_tests.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../src/deepseek41_attention_state.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../src/ds41_attention.cpp)
target_include_directories(pair_state_serialized_tests PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../include)
target_compile_features(pair_state_serialized_tests PRIVATE cxx_std_20)
if(MSVC)
    target_compile_definitions(pair_state_serialized_tests PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
if(COMMAND strix_target_warnings)
    strix_target_warnings(pair_state_serialized_tests)
endif()
add_test(NAME pair-state-ds41-serialized COMMAND pair_state_serialized_tests)
set_tests_properties(pair-state-ds41-serialized PROPERTIES LABELS "cpu" TIMEOUT 30)
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME pair-state-probe-median COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_LIST_DIR}/../tests/owners_probe_tests.py")
    set_tests_properties(pair-state-probe-median PROPERTIES LABELS "cpu" TIMEOUT 30)
endif()
