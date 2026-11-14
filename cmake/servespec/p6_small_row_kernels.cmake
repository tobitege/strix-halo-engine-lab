# Small-row kernels (2..8 rows): host reference of the small dense A8 tile layout, and GPU gates of the three
# opt-in levers. The GPU tests reuse existing test binaries, so they exist only where those are built.
strix_servespec_host_test(NAME strix-dense-a8-small-reference-tests
    SOURCES tests/dense_a8_small_reference_tests.cpp)
if(TARGET strix-dense-a8-small-reference-tests)
    if(MSVC)
        target_compile_options(strix-dense-a8-small-reference-tests PRIVATE /fp:strict)
    else()
        target_compile_options(strix-dense-a8-small-reference-tests PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
endif()
if(BUILD_TESTING AND TARGET strix-hip-dense-a8-wmma-tests)
    add_test(NAME strix-hip-dense-a8-wmma-small-tests
        COMMAND ${CMAKE_COMMAND} -E env STRIX_DENSE_A8_WMMA_SMALL=1 STRIX_DENSE_A8_ENGINE_ONLY=1
            $<TARGET_FILE:strix-hip-dense-a8-wmma-tests>)
    set_tests_properties(strix-hip-dense-a8-wmma-small-tests PROPERTIES LABELS "gpu" RUN_SERIAL TRUE)
endif()
if(BUILD_TESTING AND TARGET strix-hip-gdn-deferred-window-tests)
    add_test(NAME strix-hip-gdn-anyseg-tests
        COMMAND ${CMAKE_COMMAND} -E env STRIX_COHORT_GDN_ANYSEG=1 $<TARGET_FILE:strix-hip-gdn-deferred-window-tests>)
    set_tests_properties(strix-hip-gdn-anyseg-tests PROPERTIES LABELS "gpu" RUN_SERIAL TRUE)
endif()
if(BUILD_TESTING AND TARGET strix-hip-known-window-tests)
    # One QSA cohort family per test (0..3 = dense * 2 + FP32 indexer): the dense families take about 25 s each.
    foreach(strix_p6_family 0 1 2 3)
        add_test(NAME strix-hip-qsa-cohort-part2-f${strix_p6_family}-tests
            COMMAND ${CMAKE_COMMAND} -E env STRIX_COHORT_QSA2_MERGED=1
                $<TARGET_FILE:strix-hip-known-window-tests> --qsa-cohort ${strix_p6_family})
        set_tests_properties(strix-hip-qsa-cohort-part2-f${strix_p6_family}-tests PROPERTIES
            LABELS "gpu" RUN_SERIAL TRUE TIMEOUT 120)
    endforeach()
endif()
