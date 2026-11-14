# Conversation keep: retention policy, index and byte ledger for kept lane states (shared by family adapters).
strix_servespec_serve_sources(src/serve/conversation_keep.cpp)
strix_servespec_host_test(NAME strix-conversation-keep-tests
    SOURCES tests/conversation_keep_tests.cpp src/serve/conversation_keep.cpp src/serve/prefix_directory.cpp)
if(MSVC AND TARGET strix-conversation-keep-tests)
    target_compile_definitions(strix-conversation-keep-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
