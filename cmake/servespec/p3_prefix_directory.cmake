# Prefix directory: index of sealed and in-flight prompt prefixes for exact-frontier forks.
strix_servespec_serve_sources(src/serve/prefix_directory.cpp)
strix_servespec_host_test(NAME strix-prefix-directory-tests
    SOURCES tests/prefix_directory_tests.cpp src/serve/prefix_directory.cpp)
if(MSVC AND TARGET strix-prefix-directory-tests)
    target_compile_definitions(strix-prefix-directory-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
