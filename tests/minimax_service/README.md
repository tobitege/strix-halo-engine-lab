# MiniMax extracted service host checks

This project builds the actual `session_service.cpp` factory, validation,
registry integration, MinimaxApi, Observer, session routes and shared shell.
`SessionAssetLoader` replaces only the owned native resources and Runner work
with deterministic host assets. It never installs an unrelated fake service.
Default host construction explicitly refuses native MiniMax execution.
The host fixture uses the common runtime; native Lead/Runner replay is covered
separately by [the owner project](../minimax_session_shared/README.md).

```sh
cmake -S tests/minimax_service -B build/minimax-service-p6/standalone -DSTRIX_LEGACY_MAINS=ON
cmake --build build/minimax-service-p6/standalone --config Debug --parallel 4
ctest --test-dir build/minimax-service-p6/standalone -C Debug --output-on-failure
cmake --build build/minimax-service-p6/standalone --config Release --parallel 4
ctest --test-dir build/minimax-service-p6/standalone -C Release --output-on-failure
```

Repeat with a separate build directory and `STRIX_LEGACY_MAINS=OFF`. Root
`STRIX_FAMILY_MINIMAX=ON`, `BUILD_TESTING=ON` registers `minimax_service_tests`
as `minimax-service`. The host compatibility smoke compiles the thin main only
in this standalone project, when legacy mains are ON. It is not a build of the
native legacy target, which is absent in the root with HIP OFF.

Assertions remain active under NDEBUG. Coverage includes actual registry ID,
name, family and description selection; explicit archive installation; schema
refusal; help/effective-config/graph/plan/deployment inspection before loader
construction; malformed common inputs; API greedy sampling notes, token IDs,
metadata/admin/session routes; leader HTTP and follower no-listener roles;
startup exit, setup/handler/listener/owner/follower failures and destruction.
Listener/owner failures exercise the production shell, not only its test seam.
Gemma4 frozen replay and shared regression remain separate required suites.

No assets are downloaded and no accelerator work runs. Host fixtures do not
establish paired native startup votes, physical KV/draft/RNG equivalence, native
COW/fork/append, model fidelity or unchanged capacity/throughput. Preserve the
archived original sources and three frozen fixtures. Historical hello-byte
failures and source-receipt failures remain failures; this project does not
regenerate expectations, receipts or selector tables. Combined owner review,
source-selector review and all existing device qualification gates remain open.
