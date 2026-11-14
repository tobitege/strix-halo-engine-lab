# Prompt join rule (nest at a service point or join the next group forward): header-only rule and its host test.
strix_servespec_host_test(NAME strix-prompt-join-tests
    SOURCES tests/prompt_join_tests.cpp
    LIBS strix_serve)
