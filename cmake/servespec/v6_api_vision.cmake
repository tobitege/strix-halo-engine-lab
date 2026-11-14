# Media input through the API front doors (serve::media -> placeholder -> virtual ids -> GenerationJob::embeds).
# src/serve/api.cpp builds the per-request EmbedTable, so the serve library needs the session library that owns
# src/session/embed_rows.cpp.
if(TARGET strix_serve AND TARGET strix_session)
    target_link_libraries(strix_serve PUBLIC strix_session)
endif()

# Fake adapter on the example engine: the path end to end without a tower.
strix_servespec_host_test(NAME serve-api-vision
    SOURCES tests/serve_api_vision_tests.cpp
    LIBS strix_session_shell
    TIMEOUT 60)
if(TARGET serve-api-vision)
    set_tests_properties(serve-api-vision PROPERTIES LABELS "cpu;session")
    if(MSVC)
        target_compile_definitions(serve-api-vision PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
endif()
