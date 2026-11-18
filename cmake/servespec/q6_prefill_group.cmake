# Prefill group rule (several short prompts admitted together run as one device prefill pass): header-only rule
# and its host test.
strix_servespec_host_test(NAME strix-prefill-group-tests
    SOURCES tests/prefill_group_tests.cpp)
strix_servespec_host_test(NAME strix-prefill-group-regression-tests
    SOURCES tests/prefill_group_regression_tests.cpp)
