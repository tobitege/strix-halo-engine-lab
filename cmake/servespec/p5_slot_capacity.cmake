# Slot capacity: shared accounting (header only), capacity report and memory census.
strix_servespec_host_test(NAME strix-slot-capacity-tests
    SOURCES tests/slot_capacity_tests.cpp)
if(MSVC AND TARGET strix-slot-capacity-tests)
    target_compile_definitions(strix-slot-capacity-tests PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
