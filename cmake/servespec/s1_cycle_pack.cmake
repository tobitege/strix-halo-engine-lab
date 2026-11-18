# Shared one-completion-per-cycle pack (header part; the device gather is src/session/cycle_pack.hip in strix_hip_runtime).
strix_servespec_host_test(NAME strix-cycle-pack-tests
    SOURCES tests/cycle_pack_tests.cpp)
if(TARGET strix-cycle-pack-tests AND MSVC)
    target_compile_definitions(strix-cycle-pack-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
