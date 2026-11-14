# Two-replica request router: placement policy, blocking HTTP client, proxy front end and the strix-router binary.
strix_servespec_serve_sources(
    src/serve/replica_router.cpp
    src/serve/http_client.cpp
    src/serve/replica_proxy.cpp)

if(TARGET strix_serve)
    add_executable(strix-router "${CMAKE_SOURCE_DIR}/apps/strix_router_main.cpp")
    target_link_libraries(strix-router PRIVATE strix_serve)
    if(WIN32)
        target_link_libraries(strix-router PRIVATE ws2_32)
    endif()
    strix_target_warnings(strix-router)
endif()

strix_servespec_host_test(NAME strix-replica-router-tests
    SOURCES tests/replica_router_tests.cpp
    LIBS strix_serve)
strix_servespec_host_test(NAME strix-replica-proxy-tests
    SOURCES tests/replica_proxy_tests.cpp
    LIBS strix_serve
    TIMEOUT 120)
if(WIN32 AND TARGET strix-replica-proxy-tests)
    target_link_libraries(strix-replica-proxy-tests PRIVATE ws2_32)
endif()
