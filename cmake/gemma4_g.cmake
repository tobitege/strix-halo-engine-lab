# Feature exports share target arithmetic and reuse the portable product adapter.
set(gemma4_template_sources src/gemma4/serve/chat.cpp src/gemma4/serve/chat.hpp
    src/gemma4/serve/adapter.cpp src/gemma4/serve/adapter.hpp)
set(gemma4_template_hashes "")
foreach(source IN LISTS gemma4_template_sources)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${source}" source_text)
    string(REPLACE "\r\n" "\n" source_text "${source_text}")
    string(SHA256 source_hash "${source_text}")
    string(APPEND gemma4_template_hashes "${source_hash}\n")
endforeach()
string(SHA256 gemma4_template_hash "${gemma4_template_hashes}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${gemma4_template_sources})
add_library(strix_gemma4_features STATIC src/gemma4/features/features.cpp src/gemma4/features/npz.cpp)
target_link_libraries(strix_gemma4_features PUBLIC strix_gemma4_ref)
target_compile_definitions(strix_gemma4_features PRIVATE STRIX_G4_FEATURE_TEMPLATE_HASH="${gemma4_template_hash}")
strix_target_warnings(strix_gemma4_features)
target_link_libraries(strix_gemma4_engine PUBLIC strix_gemma4_features)
add_executable(strix-gemma4-features src/gemma4/features/main.cpp)
target_link_libraries(strix-gemma4-features PRIVATE strix_gemma4_engine)
strix_target_warnings(strix-gemma4-features)
if(STRIX_FAMILY_GEMMA4 AND BUILD_TESTING)
    foreach(test continuation turn_commit turn_cache_integration native_memory)
        if(test STREQUAL "turn_cache_integration")
            set(test_stem turn-cache)
        else()
            string(REPLACE "_" "-" test_stem "${test}")
        endif()
        add_executable(strix-gemma4-${test_stem}-tests tests/gemma4_session/${test}_tests.cpp)
        target_link_libraries(strix-gemma4-${test_stem}-tests PRIVATE strix_gemma4_engine)
        strix_target_warnings(strix-gemma4-${test_stem}-tests)
        strix_session_test_style(strix-gemma4-${test_stem}-tests)
        if(test STREQUAL "turn_cache_integration")
            add_test(NAME gemma4-session-${test_stem} COMMAND strix-gemma4-${test_stem}-tests
                "${CMAKE_CURRENT_BINARY_DIR}/gemma4-turn-cache-fixture")
        else()
            add_test(NAME gemma4-session-${test_stem} COMMAND strix-gemma4-${test_stem}-tests)
        endif()
        set_tests_properties(gemma4-session-${test_stem} PROPERTIES LABELS "cpu;session;gemma4" TIMEOUT 120)
    endforeach()
    add_executable(strix-gemma4-shared-api-tests tests/gemma4_session/shared_api_tests.cpp)
    target_link_libraries(strix-gemma4-shared-api-tests PRIVATE strix_gemma4_engine)
    strix_target_warnings(strix-gemma4-shared-api-tests)
    strix_session_test_style(strix-gemma4-shared-api-tests)
    add_test(NAME gemma4-session-shared-api COMMAND strix-gemma4-shared-api-tests
        "${CMAKE_CURRENT_BINARY_DIR}/gemma4-shared-api-fixture")
    set_tests_properties(gemma4-session-shared-api PROPERTIES LABELS "cpu;gemma4;session;api" TIMEOUT 180)
    add_executable(strix-gemma4-session-tests tests/gemma4_session/session_tests.cpp tests/gemma4_session/replay_tests.cpp)
    target_link_libraries(strix-gemma4-session-tests PRIVATE strix_gemma4_session_service strix_session_shell)
    target_compile_definitions(strix-gemma4-session-tests PRIVATE
        G4_SESSION_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
        G4_SESSION_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/tests/gemma4_session/fixtures")
    strix_target_warnings(strix-gemma4-session-tests)
    strix_session_test_style(strix-gemma4-session-tests)
    if(Python3_Interpreter_FOUND)
        add_test(NAME gemma4-session-tokenizer COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/gemma4_serve/make_deployment_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/gemma4-session-tokenizer.json)
        set_tests_properties(gemma4-session-tokenizer PROPERTIES FIXTURES_SETUP gemma4_session LABELS "cpu;gemma4;session")
        add_test(NAME gemma4-session COMMAND strix-gemma4-session-tests ${CMAKE_CURRENT_BINARY_DIR}/gemma4-session-tokenizer.json)
        set_tests_properties(gemma4-session PROPERTIES FIXTURES_REQUIRED gemma4_session LABELS "cpu;gemma4;session" TIMEOUT 120)
    endif()
    add_executable(strix-gemma4-features-tests tests/gemma4_features/features_tests.cpp)
    target_link_libraries(strix-gemma4-features-tests PRIVATE strix_gemma4_engine)
    strix_target_warnings(strix-gemma4-features-tests)
    add_test(NAME gemma4-features-host COMMAND strix-gemma4-features-tests)
    set_tests_properties(gemma4-features-host PROPERTIES TIMEOUT 60)
    if(STRIX_GEMMA4_TOKENIZER_JSON)
        add_test(NAME gemma4-features-http COMMAND strix-gemma4-features-tests --tokenizer "${STRIX_GEMMA4_TOKENIZER_JSON}")
        set_tests_properties(gemma4-features-http PROPERTIES TIMEOUT 60)
    endif()
    add_executable(strix-gemma4-serve-batch-tests tests/gemma4_serve_batch/serve_batch_tests.cpp)
    target_link_libraries(strix-gemma4-serve-batch-tests PRIVATE strix_gemma4_engine)
    strix_target_warnings(strix-gemma4-serve-batch-tests)
    if(STRIX_GEMMA4_TOKENIZER_JSON)
        add_test(NAME gemma4-serve-batch COMMAND strix-gemma4-serve-batch-tests --tokenizer "${STRIX_GEMMA4_TOKENIZER_JSON}")
        set_tests_properties(gemma4-serve-batch PROPERTIES TIMEOUT 120)
    endif()
    if(STRIX_GEMMA4_DEVICE_BACKEND)
        add_test(NAME gemma4-features-device COMMAND strix-gemma4-features-tests --device)
        set_tests_properties(gemma4-features-device PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4")
    endif()
endif()

# Opt-in operator qualification uses real loaded assets and the existing backend.
# CPU object compilation is not registered or reported as a native test run.
if(STRIX_FAMILY_GEMMA4 AND BUILD_TESTING)
    if(STRIX_GEMMA4_DEVICE_BACKEND)
        add_executable(strix-gemma4-native-continuation-tests tests/gemma4_session/native_continuation_tests.cpp
            tests/gemma4_session/turn_cache_integration_tests.cpp)
        target_compile_definitions(strix-gemma4-native-continuation-tests PRIVATE STRIX_G4_NATIVE_TURN_PROBE=1)
        target_link_libraries(strix-gemma4-native-continuation-tests PRIVATE strix_gemma4_engine)
        strix_target_warnings(strix-gemma4-native-continuation-tests)
        option(STRIX_GEMMA4_NATIVE_CONTINUATION_TESTS "Register loaded-device continuation operator tickets" OFF)
        set(STRIX_GEMMA4_NATIVE_MODEL "" CACHE FILEPATH "Exact Gemma4 model asset")
        set(STRIX_GEMMA4_NATIVE_ROPE "" CACHE PATH "Exact Gemma4 rotary tables")
        set(STRIX_GEMMA4_NATIVE_IDENTITY "" CACHE FILEPATH "Operator hash/context/prompt identity JSON")
        if(STRIX_GEMMA4_NATIVE_CONTINUATION_TESTS)
            foreach(asset MODEL ROPE IDENTITY)
                if(NOT STRIX_GEMMA4_NATIVE_${asset} OR NOT EXISTS "${STRIX_GEMMA4_NATIVE_${asset}}")
                    message(FATAL_ERROR "Native continuation registration requires STRIX_GEMMA4_NATIVE_${asset}")
                endif()
            endforeach()
            foreach(ticket activation queue-lifetime legacy prefix wrap terminal context-pending-greedy context-pending-sampled
                    context-full-greedy context-full-sampled cancel rollback rollback-wrap mixed)
                add_test(NAME gemma4-native-continuation-${ticket} COMMAND strix-gemma4-native-continuation-tests
                    --model "${STRIX_GEMMA4_NATIVE_MODEL}" --rope "${STRIX_GEMMA4_NATIVE_ROPE}"
                    --identity "${STRIX_GEMMA4_NATIVE_IDENTITY}" --case ${ticket})
                set_tests_properties(gemma4-native-continuation-${ticket} PROPERTIES
                    LABELS "gemma4;native;operator" TIMEOUT 120
                    ENVIRONMENT "STRIX_G4_IDENTITY=g4-v1;STRIX_G4_PREFILL_EXACT_MANYROW=1;STRIX_G4_PREFILL_IDENTITY=exact;STRIX_G4_EXL3_PREFILL=exact;STRIX_G4_DECODE_ATTN=exact;STRIX_G4_KV_QUEUE=1")
                if(NOT ticket STREQUAL "activation")
                    set_property(TEST gemma4-native-continuation-${ticket} APPEND PROPERTY LABELS gpu)
                endif()
            endforeach()
            add_test(NAME gemma4-native-continuation-queue-lifetime-no-queue COMMAND strix-gemma4-native-continuation-tests
                --model "${STRIX_GEMMA4_NATIVE_MODEL}" --rope "${STRIX_GEMMA4_NATIVE_ROPE}"
                --identity "${STRIX_GEMMA4_NATIVE_IDENTITY}" --case queue-lifetime)
            set_tests_properties(gemma4-native-continuation-queue-lifetime-no-queue PROPERTIES
                LABELS "gemma4;native;operator;gpu" TIMEOUT 120
                ENVIRONMENT "STRIX_G4_IDENTITY=g4-v1;STRIX_G4_PREFILL_EXACT_MANYROW=1;STRIX_G4_PREFILL_IDENTITY=exact;STRIX_G4_EXL3_PREFILL=exact;STRIX_G4_DECODE_ATTN=exact;STRIX_G4_KV_QUEUE=0")
            foreach(route a8 wmma attention)
                add_test(NAME gemma4-native-continuation-refuse-${route} COMMAND strix-gemma4-native-continuation-tests
                    --model "${STRIX_GEMMA4_NATIVE_MODEL}" --rope "${STRIX_GEMMA4_NATIVE_ROPE}"
                    --identity "${STRIX_GEMMA4_NATIVE_IDENTITY}" --case refuse-route)
                set_tests_properties(gemma4-native-continuation-refuse-${route} PROPERTIES
                    LABELS "gemma4;native;operator;inspection" TIMEOUT 120)
            endforeach()
            foreach(ticket turn sse no-store mismatch pressure retained-pressure no-retention fault cancel)
                add_test(NAME gemma4-native-api-${ticket} COMMAND strix-gemma4-native-continuation-tests
                    --model "${STRIX_GEMMA4_NATIVE_MODEL}" --rope "${STRIX_GEMMA4_NATIVE_ROPE}"
                    --identity "${STRIX_GEMMA4_NATIVE_IDENTITY}" --case api-${ticket})
                set_tests_properties(gemma4-native-api-${ticket} PROPERTIES LABELS "gemma4;native;operator;gpu"
                    TIMEOUT 120 ENVIRONMENT "STRIX_G4_IDENTITY=g4-v1;STRIX_G4_PREFILL_EXACT_MANYROW=0;STRIX_G4_PREFILL_IDENTITY=exact;STRIX_G4_DECODE_ATTN=exact;STRIX_G4_GRAPHS=0;STRIX_G4_CHUNK_DECODE=0;STRIX_G4_KV_QUEUE=1")
            endforeach()
            add_test(NAME gemma4-native-accounting COMMAND strix-gemma4-native-continuation-tests
                --model "${STRIX_GEMMA4_NATIVE_MODEL}" --rope "${STRIX_GEMMA4_NATIVE_ROPE}"
                --identity "${STRIX_GEMMA4_NATIVE_IDENTITY}" --case accounting)
            set_tests_properties(gemma4-native-accounting PROPERTIES LABELS "gemma4;native;operator;gpu"
                TIMEOUT 120 ENVIRONMENT "STRIX_G4_IDENTITY=g4-v1;STRIX_G4_PREFILL_EXACT_MANYROW=0;STRIX_G4_DECODE_ATTN=exact;STRIX_G4_GRAPHS=0;STRIX_G4_CHUNK_DECODE=0;STRIX_G4_KV_QUEUE=1;STRIX_G4_KV_PUBLISH=1;STRIX_G4_KV_POOL=1")
            foreach(mode graph prefill chunk)
                add_test(NAME gemma4-native-cache-refuse-${mode} COMMAND strix-gemma4-native-continuation-tests
                    --model "${STRIX_GEMMA4_NATIVE_MODEL}" --rope "${STRIX_GEMMA4_NATIVE_ROPE}"
                    --identity "${STRIX_GEMMA4_NATIVE_IDENTITY}" --case cache-refuse)
                set_tests_properties(gemma4-native-cache-refuse-${mode} PROPERTIES LABELS "gemma4;native;operator;gpu"
                    TIMEOUT 120 ENVIRONMENT "STRIX_G4_IDENTITY=g4-v1;STRIX_G4_PREFILL_IDENTITY=exact;STRIX_G4_DECODE_ATTN=exact;STRIX_G4_PREFILL_EXACT_MANYROW=0;STRIX_G4_GRAPHS=0;STRIX_G4_CHUNK_DECODE=0")
            endforeach()
            set_tests_properties(gemma4-native-cache-refuse-graph PROPERTIES ENVIRONMENT
                "STRIX_G4_IDENTITY=g4-v1;STRIX_G4_PREFILL_IDENTITY=exact;STRIX_G4_DECODE_ATTN=exact;STRIX_G4_PREFILL_EXACT_MANYROW=0;STRIX_G4_GRAPHS=1;STRIX_G4_CHUNK_DECODE=0")
            set_tests_properties(gemma4-native-cache-refuse-prefill PROPERTIES ENVIRONMENT
                "STRIX_G4_IDENTITY=g4-v1;STRIX_G4_PREFILL_IDENTITY=exact;STRIX_G4_DECODE_ATTN=exact;STRIX_G4_PREFILL_EXACT_MANYROW=1;STRIX_G4_GRAPHS=0;STRIX_G4_CHUNK_DECODE=0")
            set_tests_properties(gemma4-native-cache-refuse-chunk PROPERTIES ENVIRONMENT
                "STRIX_G4_IDENTITY=g4-v1;STRIX_G4_PREFILL_IDENTITY=exact;STRIX_G4_DECODE_ATTN=exact;STRIX_G4_PREFILL_EXACT_MANYROW=0;STRIX_G4_GRAPHS=0;STRIX_G4_CHUNK_DECODE=1")
            set_tests_properties(gemma4-native-continuation-refuse-a8 PROPERTIES
                ENVIRONMENT "STRIX_G4_IDENTITY=g4-a8-v1;STRIX_G4_PREFILL_IDENTITY=exact;STRIX_G4_EXL3_PREFILL=exact;STRIX_G4_DECODE_ATTN=exact")
            set_tests_properties(gemma4-native-continuation-refuse-wmma PROPERTIES
                ENVIRONMENT "STRIX_G4_IDENTITY=g4-v1;STRIX_G4_PREFILL_EXACT_MANYROW=1;STRIX_G4_PREFILL_IDENTITY=v1;STRIX_G4_EXL3_PREFILL=exact;STRIX_G4_DECODE_ATTN=exact")
            set_tests_properties(gemma4-native-continuation-refuse-attention PROPERTIES
                ENVIRONMENT "STRIX_G4_IDENTITY=g4-v1;STRIX_G4_PREFILL_IDENTITY=exact;STRIX_G4_EXL3_PREFILL=exact;STRIX_G4_DECODE_ATTN=g4-dattn-f1")
        else()
            message(STATUS "Gemma4 native continuation operator tests not registered (explicit assets/activation required)")
        endif()
    else()
        add_library(strix_gemma4_native_continuation_host_compile OBJECT EXCLUDE_FROM_ALL
            tests/gemma4_session/native_continuation_tests.cpp tests/gemma4_session/turn_cache_integration_tests.cpp)
        target_link_libraries(strix_gemma4_native_continuation_host_compile PRIVATE strix_gemma4_engine)
        target_compile_definitions(strix_gemma4_native_continuation_host_compile PRIVATE STRIX_GEMMA4_DEVICE_BACKEND=1 STRIX_G4_NATIVE_TURN_PROBE=1)
        strix_target_warnings(strix_gemma4_native_continuation_host_compile)
    endif()
endif()
