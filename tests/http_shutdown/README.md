# Shared HTTP shutdown fixture

The standalone project compiles `src/serve/http_server.cpp` directly. Integrated
registration in `cmake/StrixSession.cmake` links the actual shared serving library.
No queue, runtime or server algorithm is substituted.

Run each case as a separate bounded subprocess through CTest:

```sh
cmake -S tests/http_shutdown -B build/http-shutdown
cmake --build build/http-shutdown --config Release
ctest --test-dir build/http-shutdown -C Release -R '^http-shutdown-' --no-tests=error --timeout 8 --output-on-failure
```

The cases cover accepted idle sockets, keep-alive after an exact response,
partial headers and bodies, a non-reading peer during a large stream, concurrent
accept/stop, concurrent repeated stop, maximum-connection and post-stop refusal, ordinary and
SSE response bytes, and request deadlines. Acceptance counts, response completion
and writer progress provide phase handshakes. Peers stay open until successful
stop, accept join, zero active connections and actual server destruction.
Production defaults are used for interruption cases. The separate `deadline`
case uses a one-second request deadline solely to check the 408 contract.

Checks throw in every build configuration. The negative sentinel requires
exactly exit 1 and its exact diagnostic. Each CTest case has an eight-second
process deadline, including cases that could otherwise hang during destruction.
An external runner must also bound process-tree teardown. A timeout is a failure.
A host run on one OS does not qualify socket behavior or stress on another OS.
