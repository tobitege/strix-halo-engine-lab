# Embedding rows in the token stream (virtual ids, interner, per-request table, placeholder expansion).
# The source itself is part of strix_session (cmake/StrixSession.cmake).
strix_servespec_host_test(NAME session-embed-rows
    SOURCES tests/session_embed_rows_tests.cpp src/session/embed_rows.cpp src/serve/prefix_directory.cpp)
if(TARGET session-embed-rows)
    set_tests_properties(session-embed-rows PROPERTIES LABELS "cpu;session")
    if(MSVC)
        target_compile_definitions(session-embed-rows PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
endif()
