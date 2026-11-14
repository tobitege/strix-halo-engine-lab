# Step planner (decode first, bounded prompt slices): shared planner (strix_serve) and its host test.
strix_servespec_serve_sources(src/serve/step_planner.cpp)
strix_servespec_host_test(NAME strix-step-planner-tests
    SOURCES tests/step_planner_tests.cpp
    LIBS strix_serve)
