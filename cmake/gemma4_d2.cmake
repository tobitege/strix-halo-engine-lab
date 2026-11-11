# Coordinator and host product path; device implementations plug into engine::Backend.
option(STRIX_GEMMA4_DEVICE_BACKEND "Build the Linux/HIP Gemma 4 device backend adapter" OFF)
add_library(strix_gemma4_engine STATIC
    src/gemma4/engine/coordinator.cpp
    src/gemma4/engine/host_backend.cpp
    src/gemma4/engine/chat_runner.cpp
    src/gemma4/engine/api.cpp
    src/serve/deployment.cpp
    src/gemma4/engine/runtime.cpp)
target_link_libraries(strix_gemma4_engine PUBLIC strix_gemma4_ref strix_gemma4_dflash strix_gemma4_eagle strix_engine_deploy)
strix_target_warnings(strix_gemma4_engine)
if(MSVC)
    target_compile_definitions(strix_gemma4_engine PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
if(WIN32)
    # Portable FIFO queue; the socket transport remains Linux-only.
    target_link_libraries(strix_gemma4_engine PUBLIC strix_gemma4_serve)
    target_sources(strix_gemma4_engine PRIVATE src/serve/engine.cpp)
else()
    # Compile the existing family adapter and shared primitives unchanged. The
    # full shared ApiService also pulls unrelated engine-core implementations.
    target_sources(strix_gemma4_engine PRIVATE
        src/gemma4/serve/tokenizer.cpp src/gemma4/serve/chat.cpp
        src/gemma4/serve/parse.cpp src/gemma4/serve/adapter.cpp
        src/serve/json.cpp src/serve/bpe_tokenizer.cpp src/serve/chat_template.cpp
        src/serve/engine.cpp src/serve/http_server.cpp)
endif()
if(STRIX_GEMMA4_DEVICE_BACKEND)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT STRIX_ENABLE_HIP_RUNTIME OR NOT TARGET strix_gemma4_device)
        message(FATAL_ERROR "STRIX_GEMMA4_DEVICE_BACKEND requires Linux, HIP and strix_gemma4_device")
    endif()
    target_sources(strix_gemma4_engine PRIVATE src/gemma4/engine/device_backend.cpp)
    target_link_libraries(strix_gemma4_engine PUBLIC strix_gemma4_device strix_gemma4_eagle_device)
    target_compile_definitions(strix_gemma4_engine PUBLIC STRIX_GEMMA4_DEVICE_BACKEND=1)
endif()
add_executable(strix-gemma4-decode apps/gemma4_decode_main.cpp)
target_link_libraries(strix-gemma4-decode PRIVATE strix_gemma4_engine)
strix_target_warnings(strix-gemma4-decode)
if(NOT WIN32)
    add_executable(strix-gemma4-serve apps/gemma4_serve_main.cpp)
    target_link_libraries(strix-gemma4-serve PRIVATE strix_gemma4_engine)
    strix_target_warnings(strix-gemma4-serve)
endif()
if(BUILD_TESTING)
    if(WIN32)
        # The socket server is native-only; still compile its deployment binding
        # in host CI without linking or allocating an accelerator.
        add_library(strix-gemma4-deploy-cli-compile OBJECT apps/gemma4_serve_main.cpp)
        target_link_libraries(strix-gemma4-deploy-cli-compile PRIVATE strix_gemma4_engine)
        strix_target_warnings(strix-gemma4-deploy-cli-compile)
    endif()
    add_executable(strix-gemma4-integration-tests tests/gemma4_integration/integration.cpp)
    target_link_libraries(strix-gemma4-integration-tests PRIVATE strix_gemma4_engine)
    strix_target_warnings(strix-gemma4-integration-tests)
    add_test(NAME gemma4-integration-host COMMAND strix-gemma4-integration-tests)
    add_test(NAME gemma4-integration-off COMMAND strix-gemma4-integration-tests --off)
    set_tests_properties(gemma4-integration-host gemma4-integration-off PROPERTIES TIMEOUT 60)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        set(deployment_tokenizer "${CMAKE_CURRENT_BINARY_DIR}/gemma4-deployment-tokenizer.json")
        add_test(NAME gemma4-deployment-fixture COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/gemma4_serve/make_deployment_fixture.py ${deployment_tokenizer})
        set_tests_properties(gemma4-deployment-fixture PROPERTIES FIXTURES_SETUP gemma4_deployment TIMEOUT 60)
        add_test(NAME gemma4-deployment-http-host COMMAND strix-gemma4-integration-tests --chat ${deployment_tokenizer})
        set_tests_properties(gemma4-deployment-http-host PROPERTIES FIXTURES_REQUIRED gemma4_deployment TIMEOUT 60)
    endif()
    set_tests_properties(gemma4-integration-off PROPERTIES ENVIRONMENT "STRIX_G4_DFLASH=0")
    if(STRIX_GEMMA4_DEVICE_BACKEND)
        add_test(NAME gemma4-integration-device COMMAND strix-gemma4-integration-tests --device)
        set_tests_properties(gemma4-integration-device PROPERTIES TIMEOUT 60 LABELS "gpu;gemma4")
    endif()
    if(STRIX_GEMMA4_TOKENIZER_JSON)
        add_test(NAME gemma4-integration-chat COMMAND strix-gemma4-integration-tests --chat "${STRIX_GEMMA4_TOKENIZER_JSON}")
        set_tests_properties(gemma4-integration-chat PROPERTIES TIMEOUT 60)
    endif()
endif()
