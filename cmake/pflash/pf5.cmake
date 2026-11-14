# Prompt compression before prefill, region adapters: one per chat family plus the shared walker, and their host tests.
strix_servespec_serve_sources(src/serve/pflash_regions.cpp src/serve/pflash_regions_qwen.cpp
    src/serve/pflash_regions_glm.cpp src/serve/pflash_regions_deepseek.cpp src/serve/pflash_regions_minimax.cpp)
foreach(strix_pflash_family qwen glm deepseek)
    strix_servespec_host_test(NAME strix-pflash-regions-${strix_pflash_family}-tests
        SOURCES tests/pflash_regions_${strix_pflash_family}_tests.cpp
        LIBS strix_serve)
endforeach()
# The MiniMax renderer is one host source on top of strix_serve; compiling it here runs the test in every host build.
strix_servespec_host_test(NAME strix-pflash-regions-minimax-tests
    SOURCES tests/pflash_regions_minimax_tests.cpp src/minimax/serve/minimax_chat.cpp
    LIBS strix_serve)
if(TARGET strix-pflash-regions-minimax-tests)
    target_include_directories(strix-pflash-regions-minimax-tests PRIVATE "${CMAKE_SOURCE_DIR}/src")
endif()
