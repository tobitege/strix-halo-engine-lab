# Prompt scorer (LOSSY prefill compression, opt-in per request, default OFF): CPU reference of the scorer
# math and the loaders of its weight files. Host only.
strix_servespec_serve_sources(src/pflash/scorer_cpu.cpp)
strix_servespec_host_test(NAME strix-prompt-scorer-tests
    SOURCES tests/prompt_scorer_tests.cpp src/pflash/scorer_cpu.cpp src/pflash/probe_gguf.cpp src/gguf_reader.cpp)
if(MSVC AND TARGET strix-prompt-scorer-tests)
    target_compile_definitions(strix-prompt-scorer-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
