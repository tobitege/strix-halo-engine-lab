# Prompt compression before prefill, selection layer: shared source (strix_serve) and its host test.
strix_servespec_serve_sources(src/serve/prompt_compress.cpp)
strix_servespec_host_test(NAME strix-prompt-compress-tests
    SOURCES tests/prompt_compress_tests.cpp
    LIBS strix_serve)
