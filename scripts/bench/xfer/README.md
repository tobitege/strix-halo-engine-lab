# Hand-off transport kit (docs/xfer.md)

No host names here: endpoints and files are arguments.

- `strix-xfer-probe` (CMake target, host only):
  - `listen PORT`: the receiver.
  - `send HOST:PORT --sizes 64K,1M,16M --streams 1,2,4 --codecs identity,bp-v1 --data random|bf16|fp32|file:PATH --width W`:
    piece-size sweep, one JSON line per session with MB/s, % of line rate, and CPU s/GB on both ends.
  - `send ... --schedule FILE [--data4 SRC]`: releases each piece at its producer time and prints the tail after the
    last ready piece.
  - `plan --schedule FILE`: the shared cost model for every link class.
  - `codec --data ...`: bp-v1 ratio and MB/s.
- `schedule.py qwen|ds41|uniform ...`: writes piece schedules ("READY_MS BYTES WIDTH" lines) from a family's state
  layout and a measured prefill time.
- `state_ratio.py STATE.rtxs [--out DIR]`: zstd-1 plain vs byte-plane ratio per region of a real Qwen state (Python
  >= 3.14). `--out` writes the regions for `strix-xfer-probe codec --data file:...`.
- `copy_probe.cpp`: H2D/D2H pageable vs pinned, alone and beside a busy kernel. One source for CUDA (nvcc) and HIP
  (hipcc).

Order for a two-host run: start the sender first (it retries the connection for up to 15 min), then the listener
inside the GPU turn of the node. Launching a turn over ssh can block until it ends, so launch it in the background.

Probe sizes accept positive finite bytes with an optional single K/M/G suffix (binary units,
case insensitive); fractional bytes truncate and quantities below one byte are rejected.
Plan thread limits are 0..256 (0 uses the caller). Predictions require host/workload calibration;
shared TCP results are not family HTTP service qualification. Keep comparative inputs/results
outside Git. State exports validate metadata and payload checksum first and require fresh output
filenames; existing files and unsafe component identifiers are refused. Copy event intervals
measure device time separately from host wall time; interval overlap may include submission gaps.

Host regressions only (no engine build): configure `tests/handoff_host`, build Release, then run
`ctest --test-dir BUILD_DIR -C Release --output-on-failure`. The copy timing regression uses a
mock device timeline; native CUDA/HIP timing and family-service qualification remain separate.
