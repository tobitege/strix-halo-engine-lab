# Row budget across lanes: shared planner (strix_serve) and its host test.
# The planner is a pure host source with no dependencies. It lives in strix_core (which strix_serve links) so the
# MTP window controller of strix_core can call it too.
if(TARGET strix_core)
    target_sources(strix_core PRIVATE ${CMAKE_SOURCE_DIR}/src/serve/row_budget.cpp)
else()
    strix_servespec_serve_sources(src/serve/row_budget.cpp)
endif()
strix_servespec_host_test(NAME strix-row-budget-tests
    SOURCES tests/row_budget_tests.cpp
    LIBS strix_serve)
