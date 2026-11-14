# Cycle receipt and ledger (header only) and the serving load harness.
strix_servespec_host_test(NAME strix-cycle-receipt-tests
    SOURCES tests/cycle_receipt_tests.cpp
    LIBS strix_serve_json)
strix_servespec_python_test(NAME strix-serve-bench-tests SCRIPT tests/serve_bench_tests.py)
