# The serving front end of the standalone CUDA Qwen configuration (STRIX_CUDA_QWEN_ONLY): the API library and the
# handoff library under it, host sources only. One function, so the full host build links the same graph in a test
# (strix-serve-standalone-link-tests) and a caller added to src/serve/api.cpp without its library fails there, not
# only on a machine with a CUDA toolchain.
include(${CMAKE_CURRENT_LIST_DIR}/StrixPower.cmake)
function(strix_add_standalone_serve serve_target handoff_target)
    set(_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/..")
    find_package(Threads REQUIRED)
    add_library(${handoff_target} STATIC ${_root}/src/cuda/qwen_handoff.cpp ${_root}/src/bp_codec.cpp
        ${_root}/src/serve/json.cpp ${_root}/src/model_content_hash.cpp ${_root}/src/model_runtime_binding.cpp)
    target_include_directories(${handoff_target} PUBLIC ${_root}/include ${_root}/src/cuda)
    target_compile_features(${handoff_target} PUBLIC cxx_std_20)
    target_link_libraries(${handoff_target} PUBLIC Threads::Threads)
    if(WIN32)
        target_link_libraries(${handoff_target} PUBLIC ws2_32)
    endif()
    add_library(${serve_target} STATIC
        ${_root}/src/serve/json_constraint.cpp ${_root}/src/serve/constraint_selection.cpp ${_root}/src/serve/bpe_tokenizer.cpp
        ${_root}/src/serve/chat_template.cpp ${_root}/src/serve/deepseek_chat.cpp ${_root}/src/serve/engine.cpp
        ${_root}/src/serve/api.cpp ${_root}/src/serve/media.cpp ${_root}/src/serve/request_fields.cpp ${_root}/src/serve/health.cpp
        ${_root}/src/serve/http_server.cpp ${_root}/src/serve/token_memo.cpp
        # Called by src/serve/api.cpp (deployment routes, preflight view of the status).
        ${_root}/src/serve/deployment.cpp ${_root}/src/preflight.cpp ${_root}/src/linux_uma_observation.cpp
        ${_root}/src/uma_owner_inventory.cpp)
    # generated/: strix/build_info.hpp (cmake/StrixBuildInfo.cmake), read by src/preflight.cpp.
    target_include_directories(${serve_target} PUBLIC ${_root}/include PRIVATE ${_root}/src/serve "${CMAKE_BINARY_DIR}/generated")
    target_compile_features(${serve_target} PUBLIC cxx_std_20)
    # src/serve/api.cpp calls the shared power sampler: linked by name, this graph has no HIP gate to carry it.
    target_link_libraries(${serve_target} PUBLIC ${handoff_target} strix_power strix_session)  # api.cpp builds the per-request EmbedTable
    if(MSVC)
        target_compile_definitions(${serve_target} PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_definitions(${handoff_target} PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
endfunction()
