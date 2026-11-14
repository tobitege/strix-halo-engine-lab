# Drafter CPU reference of the LOSSY prefill compression (opt-in per request, default OFF): blocks 0-14 and
# the block-15 QK mass, checked against goldens of tools/pflash_ref.py. Host only; nothing calls it yet.
strix_servespec_serve_sources(src/pflash/drafter_cpu.cpp)
strix_servespec_host_test(NAME strix-pflash-drafter-tests
    SOURCES tests/pflash_drafter_tests.cpp src/pflash/drafter_cpu.cpp src/pflash/scorer_cpu.cpp src/gguf_reader.cpp)
if(MSVC AND TARGET strix-pflash-drafter-tests)
    target_compile_definitions(strix-pflash-drafter-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
