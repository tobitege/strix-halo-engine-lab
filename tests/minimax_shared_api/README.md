# MiniMax shared API host project

This independent CXX-only project reuses `tests/session_runtime`, the production
MiniMax frontend/service sources and the existing owner/replay suites. It links
the actual shared `ApiService`; no copied API or fake frontend is substituted.
The central `minimax_shared_api_tests` target uses
`tests/minimax_session_shared/shared_api_tests.cpp`.

```sh
cmake -S tests/minimax_shared_api -B build/minimax-api-r7/host-on \
  -DSTRIX_LEGACY_MAINS=ON -DBUILD_TESTING=ON \
  -DSTRIX_ENABLE_HIP_RUNTIME=OFF -DSTRIX_ENABLE_CUDA_TRELLIS=OFF \
  -DSTRIX_ENABLE_CUDA_DS41=OFF
cmake --build build/minimax-api-r7/host-on --config Debug
ctest --test-dir build/minimax-api-r7/host-on -C Debug --output-on-failure
```

Repeat in a fresh directory for legacy OFF and run both Debug and Release.
Assertions in the shared API tests stay active in both configurations. The test
creates its synthetic tokenizer asset only in its build working directory and
loads it through the real family tokenizer. It exercises prompt equality,
actual family rendering/decoding, live SSE before terminal while the production
owner is paused, EOS IDs/usage, receipt cleanup, queued cancellation, queue/slot
reuse, observation, metadata routes, session validation and draining. Existing
scripted API tests independently cover execution errors, exact timing order and
scheduler callback error bytes. Callback-only nonstream generation is checked
without vocabulary metadata or a streaming callback; a streaming request then
refuses before queue work.

The nested project uses the same checked test-only observation transform as the
root build, in `cmake/StrixReplayObservation.cmake`. No independent queue patch
is applied. Configure-time checks and `minimax-replay-observation` exercise the
actual current and archived queue sources and reject changed/missing seams.
The transform preserves the publication expression, live parser and frozen
fixtures/normalization. Original-source
reference tests remain separate; failures are failures and must be reported.
Native numerical, transport, capacity and performance qualification is outside
these host tests. A combined fresh root configuration with both families enabled
and the prescribed central registration is an additional integration gate.
