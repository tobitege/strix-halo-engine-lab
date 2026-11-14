# Checkpoint publication host tests

This independent C++20 project builds actual production checkpoint codecs,
publication IO, pooled retention and the Qwen/MiniMax named-session adapters.
It needs no accelerator compiler, device execution, model or tokenizer assets.
It does not include existing root or standalone build modules. Each production
translation unit is compiled once into its host closure.

```sh
cmake -S tests/session_checkpoint_io -B build/checkpoint-publication-r8/host
cmake --build build/checkpoint-publication-r8/host --config Debug
ctest --test-dir build/checkpoint-publication-r8/host -C Debug --output-on-failure
cmake --build build/checkpoint-publication-r8/host --config Release
ctest --test-dir build/checkpoint-publication-r8/host -C Release --output-on-failure
```

`session_checkpoint_io_tests` uses throwing checks in Debug and Release. It
checks real Qwen named-session parking/disk restart, MiniMax paired publication,
resume and erase, generic rank metadata, compatibility/frontier refusals, exact
old-codec hashes, prefix legacy/certificate/durability formats, bounded corrupt
frames, allocation/read/write/sync/replace faults, short writes, persistent locks,
concurrent publication, interrupted temporary cleanup and failed-rank manifest
agreement. The thread-local fault seams have inert production defaults.

Fixed hashes were captured using independently archived entry sources before
their production codecs were edited. Private entry binary files can additionally
be supplied with `-DCHECKPOINT_ENTRY_FIXTURES=ENTRY_DIR`. That directory contains
`qwen0.bin`, `qwen1.bin`, `minimax0.bin` and `minimax1.bin`; its sibling
`frozen-prefix/prefix0`, `prefix1` and `prefix2` holds the original segmented
files. This enables the separate `session_checkpoint_entry` exact-byte and
cross-decode test. With no private fixture directory, that subcase prints NOT RUN;
fixed entry hashes are still checked. Never regenerate historical expectations
using the migrated encoder.

Platform checks are distinct from deterministic fault tests. Windows executes
real locking and replacement of an existing file. POSIX inode/flock and directory
fsync checks run only on POSIX; they are NOT RUN on Windows. Native state import,
machine-crash recovery, accelerator compilation, model fidelity, capacity and
performance remain separate NOT RUN operator gates.

The project also registers the original `session_state_cache_tests`,
`minimax_session_cache_tests` and `session_cache_owner_tests`, preserving their
assertions. On the current Windows host the first retains its unsupported
unbuffered-write fault assertion. The MiniMax suite reaches its historical
900000-byte replacement case and refuses rather than deleting its last good
generation. That assertion conflicts with protected replacement and remains a
reported failure; later cases in that executable are unrun. The new tests
separately exercise both families' lock wrappers and failed-rank preservation.

Existing build integration belongs to the common host closure: link
`session/checkpoint_io.cpp`, `session_state_cache.cpp` and its rotary dependency
once below both families and PrefixStateCache. The new test also needs the real
MiniMax session/assets/description, shared JSON and prompt-terminal diagnostics
providers because it calls the actual Qwen HttpSessionStore. Do not add another
copy of these translation units to a linked archive. The independent project
does not certify the root legacy ON/OFF combined matrix or native Runtime TU.
