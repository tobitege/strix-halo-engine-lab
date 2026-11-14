# Integrated USB4 multi-slot engine: build and test entrypoint

This is the current entrypoint for the accumulated USB4 work on `main`.
Use **`strix-gpu-queue-visibility`**, built from **`lab/gpu_queue`**. It defaults
to **eight outstanding operations per direction**, supports depths 2..8, and
can submit eight TX plus eight RX operations in each full-duplex batch window.
The older `strix-direct-gpu-commission` and `strix-gpu-visibility` executables
remain C1 compatibility/diagnostic routes; their one-operation limit is not the
limit of this multi-slot adapter.

The current matching driver is `hyukudan/strix-halo-usb4-driver` `main`, with
all nine patches applied to pinned Linux
`72d3fcf802c45d00b300f25b848a93c3a2bd7c7e`. No previous ZIP or unmerged PR is
needed for this source integration. Raw driver `overlay/` remains the original
capture and must not be mistaken for the applied nine-patch result.

## Software contract

`LinuxGpuQueue` is the reusable engine-side adapter for `QUEUE` / `PREPARE_SLOT`.
`run_gpu_queue_session()` coordinates its two-host transfer and recovery windows.
The native executable binds them to the existing HIP pool allocator, DMA-BUF
export owner, retained import lease, per-slot execution memory/streams/events,
GPU byte verifier and authenticated loopback control transport.

For depth D and R ring frames, each slot owns
`floor((R-D)/D) * 4096` payload bytes. The final D frames store separate terminal
records. Payload tails, physical offsets, complete event batches, generation,
handle and sequence are validated. RX is posted before peer readiness permits
TX. Producers finish before submission. EAGAIN preserves the same operation;
a later sequence cannot overtake a prepared predecessor. Mutating ioctls are
not blindly replayed after an ambiguous interruption or reply.

Descriptor and pool identity (one `fstat` and two `fcntl` per check, plus the pool
owner's check) are verified at session transitions: open, QUERY/QUEUE, IMPORT, ARM,
STATS, STOP, RELEASE, RECOVER and group reset. On the hot path (PREPARE, SUBMIT, EVENT,
RETIRE) of the communicator's device route the identity is re-verified at most every
100 ms, on the next hot operation, and before any ioctl error other than `EAGAIN`/`EINTR`
is reported, so a replaced descriptor is named as such; EVENT is not asked while no
submitted operation is incomplete. The fence, handle, sequence, direction and every reply
field of each operation are checked on every call. `STRIX_DIST_GPU_QUEUE_LEAN=0` restores
the identity check before and after every hot ioctl. `GpuQueueExpected::lean` selects the
mode per queue and `identity_interval` the re-check interval; the adapter's own default
remains the per-call check.

Lean admission uses the engine's `gpu-lean` transport capability in
`options.transport_capabilities`, not a driver-advertised lean feature. The
driver's QUERY/QUEUE capability checks still apply. Lean assumes exclusive
descriptor ownership: another owner closing, replacing or changing descriptor
flags between checks can affect an ioctl before detection. An error-triggered
check cannot undo that ioctl; the strict-mode escape hatch remains available.

This implementation uses **completed batch windows**, not a sliding/adaptive
slot-refill scheduler. Both imported allocations drain controller access before
GPU consumers run, avoiding a deadlock on allocation-wide implicit reservation
fences while another producer still needs the GPU. It is a serialized-owner API.
It does not automatically replace the model dispatcher's production transport,
turn a physical link into multiple links, or grant production qualification.

## GPU visibility and end-to-end correctness

On the receive path, the GPU verifies every byte of **raw RX before any acquire,
unpack or terminal copy**. The ordinary acquire/unpack copies then run, and the
actual execution destination is verified separately. GPU producer/consumer
completion and authenticated peer credit are distinct from local controller
completion. A negative control changes one byte only in the independent unpacked
execution allocation, requires exact detection, restores it and verifies again.
An imported or DMA-posted RX allocation is never modified for that control.

The pattern binds epoch, sequence, seed, byte position and a mixed sender UUID.
This is error/replay detection, not cryptographic authentication. Per-invocation
result identities, complete byte coverage and mismatch counts are checked.
Only fixed-size GPU result records and terminal metadata cross to CPU; the CPU
does not verify the payload by downloading it. System-scope events are explicitly
requested. These choices implement a testable synchronization contract; they do
not by themselves prove the target GPU/exporter/importer/cache behavior.

The acquire and unpack GPU-to-GPU copies remain. This is **not** a qualified
zero-copy GPU RDMA implementation. Native HIP compilation and physical execution
on both hosts have not been certified in this publication.

## Recovery and retained ownership

`--recover-every N` coordinates live-session recovery between completed windows.
Local controller STOP and independent GPU quiescence must pass. Both peers agree,
RECOVER releases imports, and a strictly larger epoch is imported and armed.
Handles remain increasing within an open file. Old operations/imports fail.

`--controller-reset` adds the explicit administrative controller group reset at
those boundaries. The driver accepts up to 16 owner-held GPU FDs in its library
API, checks all of their identities and quiescence, and independently checks the
entire NHI ring and tunnel inventory. The executable itself owns one GPU FD per
host; several sessions must be coordinated using
`LinuxGpuQueue::reset_controller_group()`, not guessed from other processes.

`--include-network --independent-control` permits temporary USB4NET down/up.
Only interfaces on that controller that were previously up are restored. This
can lose packets. **Control must use a physically independent link/controller**;
an SSH tunnel on the USB4NET interface being stopped is not independent. Unknown
or non-participating consumers veto reset. There is no forced reset, automatic
reset loop, failed-window replay or recreation of a lost control socket.

A timeout, process exit or missing device is not a memory-release proof.
Unproved DMA/GPU quiescence and ambiguous release/reset outcomes retain the exact
owners. The driver now retires proven-safe original contexts even with old FDs
open; it never reassigns old HopIDs to a replacement connection. See
[cooperative recovery and retirement](usb4-controller-recovery.md).

## 1. Build from the integrated source

Use current `main` in both repositories. The driver's
`docs/INSTALL-AND-TEST.md` describes reconstructing the nine-patch kernel,
full offline compilation and preparing a host-specific installation. Its
`tools/validate_native.py` can build kernel/modules for GPU-on/off configurations
and the HIP target locally without Actions. It does not install or run them.
Keep a known-good boot entry and the IOMMU enabled. Rebuild compatible consumers;
do not mix new Thunderbolt/USB4NET modules with old internal headers.

From this engine checkout, with the ROCm development toolchain installed:

```sh
cmake -S lab/gpu_queue -B build/gpu-queue -G Ninja \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON -DBUILD_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build/gpu-queue --target strix-gpu-queue-visibility -j2
build/gpu-queue/strix-gpu-queue-visibility --help
```

Adjust only the actual ROCm compiler path as required by the host. No SDK, kernel,
firmware or driver is downloaded or installed by this target. Root CMake and
unrelated MTP/runtime changes have been preserved.

## 2. Prepare the actual endpoints

Deliberately configure the intended `gpu_direct` route and interrupt mode, match
ring/HopID geometry, and exclude ConfigFS changes through close. Opening an
unknown stock/copied-mode stream is not a safe compatibility probe. The CLI
requires `--configured-gpu-direct` as an operator acknowledgment; it is not an
automatic ConfigFS lock. Nonblocking, close-on-exec FD identity, actual module
build ID/kernel release and GPU UUID/runtime/architecture are checked throughout.

Collect the local observations with the existing driver/GPU observation tools
as described in `lab/direct_gpu_stream/COMMISSIONING.md`. The native queue accepts
only the provisioned extension, exact observed `gfx1151` identity and the updated
canonical UAPI. Use a fresh increasing peer-agreed epoch, never invented UUIDs
or module build IDs. A per-open observer instance is not reused by the runner.

Control uses the existing loopback-only adapter. Establish authenticated SSH
forwarding separately. The sender listens on its local loopback port; the
receiver connects to its local forwarded port. Local/peer endpoint IDs reverse
between hosts. No script creates or exposes an unauthenticated LAN listener.

## 3. Configuration-only check and real two-host run

On each host set these variables to actual observations/shared settings. ROLE
is sender on A and receiver on B; CONTROL_MODE is listen on A and connect on B.
Local kernel/module/GPU values may differ but must match each host's actual
identity. Campaign, run, link label, request, seed, epoch and geometry agree.

```sh
args=(
  --experimental-commissioning --configured-gpu-direct
  --campaign-id "$CAMPAIGN" --run-id "$RUN"
  --dma-device "$DMA_DEVICE" --local-endpoint "$LOCAL_ENDPOINT"
  --peer-endpoint "$PEER_ENDPOINT" --physical-link "$PHYSICAL_LINK"
  --kernel-release "$EXPECTED_KERNEL" --driver-build-id "$EXPECTED_MODULE_BUILD_ID"
  --gpu-device 0 --gpu-uuid "$GPU_UUID"
  --gpu-runtime-version "$GPU_RUNTIME_VERSION" --gpu-driver-version "$GPU_DRIVER_VERSION"
  --epoch "$FRESH_EPOCH" --request-id "$REQUEST_ID"
  --role "$ROLE" --control-mode "$CONTROL_MODE" --control-port "$LOOPBACK_PORT"
  --payload-bytes 4097 --ring-frames 32 --queue-depth 8 --windows 4
  --full-duplex --timeout-ms 30000
)
build/gpu-queue/strix-gpu-queue-visibility "${args[@]}" --check-config
# The following command performs actual allocation, GPU work and USB4 DMA.
build/gpu-queue/strix-gpu-queue-visibility "${args[@]}" > "$PRIVATE_RECEIPT_PATH"
```

With R=32 and D=8, each slot has 12,288 bytes of payload capacity; 4,097 exercises
a partial final frame. Match the driver's actual configured ring size; changing
only a CLI value does not change the hardware configuration. Start with a one-way
run by omitting `--full-duplex`, then use a new epoch for duplex/reuse. Recovery
and reset tests are separate explicit campaigns after ordinary integrity passes.

Keep hardware identities, configurations and receipts outside Git. Compare the
two completed receipts on one machine:

```sh
python3 tools/qualification/gpu_queue_evidence.py /private/A.json /private/B.json
```

The summary verifies supplied record structure/coverage, not authenticity or
production admission. It retains `authenticity_verified=false` and
`production_qualified=false`. Software peak-outstanding counts are not measured
physical overlap; elapsed time includes verification and coordination. This is
not a raw-link or model-throughput benchmark. Do not add TX and RX to inflate a
single-direction link rate.

## 4. Reproduce portable tests without copying driver fixtures

Materialize the driver from its current `main` and the genuine pinned Linux base.
Then run the engine tests with that source and the driver's canonical headers:

```sh
export USB4_SOURCE_TREE=/absolute/path/materialized-nine-patch-driver
export USB4_DRIVER_REPO=/absolute/path/strix-halo-usb4-driver
python3 -m unittest discover -s tests -p 'test_*gpu*.py' -v
```

The driver and engine UAPI are compared byte-for-byte. Kernel source bodies,
queue/protocol code and GPU verifier bodies are exercised with explicitly
substituted kernel/HIP/MMIO/DMA APIs, sanitizers, 2,048 slot reuses and bounded
local-socket peer processes. All **101 engine checks passed with GCC and Clang**
again using canonical driver tests. The cumulative delivery additionally passed
81 driver and 12 package checks per compiler. Bundle-only context tests and
original repository tests have separate counts and are not native coverage.

The publication includes every delivered source change and the prior PR27/28
compatibility routes, but no native build result, installed binary, hardware
visibility result, 40-Gb/s negotiation fix or model qualification is manufactured.
