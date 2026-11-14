# GLM request-owner host fixture

Host-only tests of the GLM serving lead on the shared session runtime
(`strix::serve::SessionRuntime` in owner-control mode). They include the real
templated lead / follower with a native double; no device is needed.

- `glm-owner-replay`: request traces (plain, chunked prefill, MTP, G2, import,
  prefix cache, cancellation, drain) through the runtime-owned queue.
- `glm-owner-lifetime`: owner failures settle every taken request and release
  every logical lease before the native engine goes away.
- `glm-owner-http`: the same lead behind `ApiService` and the HTTP front end.
- `glm-owner-existing-cpu`: the existing `tests/glm_serve_cpu_tests.cpp` linked
  against the same owner library.
- `glm-owner-checks`: negative sentinel (a failing check must fail the binary).

Run: `ctest -L glm-owner --output-on-failure`.

Device exactness (token ids, logits, prefix-cache restore) is not covered here;
see `docs/engine/session-glm-owner-adoption.md`.
