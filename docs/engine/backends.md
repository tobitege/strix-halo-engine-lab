# Session execution backends

`session::Backend` separates execution resources from model semantics. Its two
entry points serve different levels of work:

- `execute(ExecutionPlan&)` preserves a family's complete grouped forward plan.
  The owner check, synchronous completion boundary and failure contract remain
  unchanged. There is no scalar-forward fallback.
- Resource methods and `dispatch(Queue&, Operation)` let an executor implement
  that plan using portable buffers, copies and registered batch operations.
  Registry lookup occurs once per operation, outside element and kernel loops.

The public headers are `include/strix/session/backend.hpp`, `backend_cpu.hpp`
and `backend_hip.hpp`. They contain no native device types or SDK includes.
The shared host implementation and CPU reference live in
`src/session/backend_cpu.cpp`; native HIP details live in `backend_hip.cpp`.
Gemma4 serving now executes its native grouped plans through this seam. Its
existing target retains allocations, streams, kernels and numerical routes;
resource adoption is separate and remains unqualified.

## Resources and ownership

`Buffer`, `Queue` and `Event` are opaque, move-only RAII handles. Default,
moved-from and reset handles are invalid. `reset()` releases ownership; native
destruction is nonthrowing. Resource state retains a unique backend provenance
token, independent of the backend's address. A different instance of the same
backend is still a different owner. Address reuse cannot revive an old handle.

`BufferSlice(buffer, byte_offset, byte_count)` is a weak, non-owning view.
The owning buffer must remain alive until the call begins; the backend pins
validated resources through completion or error drain. A slice of a released
buffer is rejected, as are foreign resources. Moving the buffer preserves its
slices. Handles can be destroyed after their backend: cleanup retains its own
implementation state and never calls a destroyed backend. Using those handles
through another backend is rejected.

All calls and handle destruction for an instance must be externally serialized.
This interface does not provide concurrent submission or a thread-safe allocator.
Do not destroy or move a backend during a call, or reset its bound native device
while resources are alive. Plans and engine/state bindings
borrow their backend and must not outlive it; resource handles have the separate
RAII cleanup lifetime described above. A plan must retain the owning handles
for every slice it will use, rather than storing slices alone.

Allocation requires positive bytes and nonzero power-of-two alignment.
Alignment reservation overflow is rejected. Slice validation uses subtraction
to check bounds without overflowing offset plus length. Copy sizes must match
exactly. Device-to-device copies conservatively reject overlapping ranges,
including exact self-copy; disjoint ranges of one buffer are allowed. Empty
copies use valid handles and in-bounds empty slices. Uninitialized allocation
contents are unspecified; allocation does not imply zeroing.

`upload`, `download` and `copy` implement host-to-device, device-to-host and
device-to-device transfers. CPU device storage is host memory, with the same
ownership and byte checks. Host spans must name valid readable/writable storage
for the duration of the call. The caller may reuse or destroy that storage as
soon as the call returns or throws. There is no pointer-as-integer resource API,
native pointer import, borrowed production allocation or implicit default queue.

## Completion, events and failure

Every successful resource call completes synchronously. Calls on a queue are
ordered; `record(queue)` returns a new immutable event for the completed queue
prefix. `wait(other_queue, event)` establishes that dependency on another queue
of the same owner. Events survive destruction of their source queue. There are
no unrecorded events or event re-recording. `synchronize(queue)` is an explicit
completion/error boundary. These semantics allow a future implementation to
overlap work inside one operation or grouped plan, without retaining borrowed
host spans across calls. This API currently exposes no asynchronous submission.

Unknown operation keys, malformed bindings, invalid ownership, byte bounds,
aliasing and alignment are refused before execution, preserving destinations.
`UnsupportedOperation` derives from `std::invalid_argument`. Other validation
errors use `invalid_argument`, `out_of_range`, `overflow_error` or `length_error`
as appropriate. Native execution errors use `runtime_error`; partial device
writes are possible after execution starts, so affected outputs are tentative.

After an execution/completion error, the backend drains submitted work before
unwinding borrowed resources, and the queue remains poisoned. Reusing it is
rejected. HIP retains submitted buffers and owned pinned staging through drain.
If native completion cannot be established, it poisons the backend, quarantines
the affected native storage, and deliberately leaks that storage rather than
freeing it while device work might still use it. It makes no successful
completion claim in that case. Host copy staging ensures the device never
retains a caller's host span, including after failed drain. Explicitly destroy
the failed adapter and recover the device outside this interface.

`execute(plan)` retains its existing contract: reject unknown plan schemas
before side effects, drain all internal queues/exchanges on failure, and leave
the engine responsible for restoring or invalidating tentative model state.
The resource queue drain does not implement logical KV rollback.

## Memory admission

`memory_budget()` reports policy limit, reserved bytes, available admission
bytes and optional observed free/total bytes. Reserved accounting includes
alignment padding and lasts until resource destruction. It is not an exact
allocator metadata or process working-set measurement. A probe is a snapshot,
never a guarantee that the next allocation fits physically.

CPU uses a caller-selected limit (64 MiB by default). It charges requested bytes
plus normalized alignment minus one, and reports no physical memory probe:
`probe_known=false`, observed values zero. Available bytes describe only the
remaining policy allowance. HIP requires an explicit limit, charges the full
overallocated device extent, and admits at most the minimum of remaining policy
allowance and `hipMemGetInfo` free bytes. An unknown HIP probe admits zero bytes;
a poisoned HIP backend also reports zero available. Temporary pinned copy
staging is separate host memory, not device budget capacity. Neither backend
changes context size, cache policy, production reservations or device placement.

## Operation registry and adapters

An `OperationKey` is the exact tuple `(operation, dtype, quantization, backend)`.
Names are case-sensitive strings, with `none` explicitly identifying an
unquantized format. `supports(key)` and `query(key)` perform no device work or
resource writes.
Unknown tuples return false/no schema, including unknown formats and keys for
another backend. No dtype conversion, dequantization or CPU fallback is inferred.

`query` returns input count, element size, alignment and maximum element count.
`Operation` binds input slices, one output slice and an element count. Every
slice must have exactly `elements * element_bytes` bytes. Multiplication
overflow, insufficient declared buffer alignment and misaligned offsets are
rejected. Outputs must be disjoint from all inputs. Inputs may share read-only
storage. Zero-element operations still validate their resources and schema.

Backend implementations call protected `register_operation` during construction.
It rejects empty/duplicate/foreign keys and invalid schemas. First resource use,
plan execution, dispatch or capability query seals registration; subsequent
registration throws `logic_error`. The executor receives
validated, pinned buffer bindings and a queue state; native extraction happens
inside its implementation file.
More complex operations should add explicit validated descriptors/schema support
rather than reinterpret these flat-vector bindings as model-specific geometry.

| Implementation | Resources | Registered arithmetic | Family plans |
|---|---|---|---|
| CPU | Aligned host buffers, synchronous copies/queues/events, policy budget | `add/f32/none/cpu`, two contiguous input vectors | Unsupported by the base reference adapter |
| HIP | Device allocation/free, pinned copy staging, native streams/events and memory probe | `add/f32/none/hip` only when the existing launcher target is linked; maximum 1,048,576 elements | Unsupported by the base resource adapter |
| Plan-only backend | Unsupported by default | Empty registry | Existing complete-plan implementations remain compatible |

HIP's conditional add adapter calls the existing
`strix::ds41::enqueue_add_f32` launcher declared in `ds41_shexp_tp.hpp` and
implemented in `src/hip_ds41_shexp_tp.hip`. It does not copy kernel arithmetic.
`strix_session_hip` links `strix_ds41_trellis_hip` only when that target exists;
otherwise the HIP arithmetic registry is empty while native resources remain
implemented. CPU add is a correctness reference, not a loader or general model
execution backend. Neither adapter advertises other arithmetic or quantization.

Gemma4 is the first production family plan consumer. `ForwardPlan` contains the
existing causal/independent RowPlans, transaction bindings, export flags and
owning grouped result. `PrefillPlan` contains complete many-row segments, hidden/
tap requirements and result storage. Neither schema includes accelerator SDK
types. The plan-only executor checks schema/owner and delegates to the existing
loaded `engine::Backend`/`Target` for one grouped call. It does not dispatch an
unrelated resource arithmetic operation or copy kernels into shared code.

The adapter preserves the existing prefill-batch unsupported fallback using the
same already selected chunks; native grouped target decode is never scalarized.
Completion and physical transaction semantics remain those of the target.
The shared session owner validates committed per-lane progress and publishes
surviving lanes on partial failure. Ticket cancellation is observed at quiescent
chunk/collective boundaries, never mid-kernel. See the
[serving ownership contract](session-layer.md).

Resource methods on this Gemma4 plan executor are unsupported. Native resource
adoption, complete native state/sampler/drafter snapshots and device/performance
qualification are separate gates. Full snapshots fail closed if a physical
stream or drafter cannot freeze its complete state. Prompt-purpose reuse uses
fresh sampling/drafter initialization, distinct from a full lane resume.
Deep-copy StreamState forks do not advertise COW. Qwen, DeepSeek and GLM
serving retain their existing routes. MiniMax uses a separate native plan
adapter described below.

## Additional devices

New implementations supply the same resource hooks and register only operations
they implement. Device selection belongs to adapter construction, not a central
device switch in family semantics or serving. Shared plan schemas and model
state policy remain unchanged when adding a backend executor for those schemas.

The existing CUDA paths are separate from this adapter. `cmake/QwenCuda.cmake`
selects the CUDA language and existing Qwen libraries; the root build also has
the DeepSeek CUDA path and its limited build configuration. In
`src/cuda/qwen_prefill_driver.cu`, device arenas use `cudaMalloc`, copy staging
uses `cudaMallocHost`, queues use `cudaStreamCreateWithFlags`, events use
`cudaEventCreate`, and uploads use `cudaMemcpyAsync`. A future CUDA adapter maps
those allocations, streams, events, copies and memory observations into private
resource states, then binds the existing whole-operation launch APIs such as
the Qwen dense/attention launchers. Its public header needs no CUDA types.
No CUDA implementation or rerouting is introduced here.

A Vulkan implementation can own buffer/memory allocations, staging mappings,
command submissions and fence/timeline dependencies behind these handles.
An Intel SYCL/Level Zero implementation can own context allocations, queues,
copy submissions and events similarly. Both must preserve host-span lifetime,
explicit unsupported results, batch boundaries and drain behavior. CPU proves
the contract without a device SDK; it does not prove those device implementations
or allow unimplemented operations to be reported as supported.

## Build and qualification

The normal root build includes `cmake/StrixSession.cmake` outside HIP guards.
`strix_session` contains the CPU resource implementation plus the stable
`src/session/kv.cpp` and `prefix_cache.cpp` storage sources, exports `include/`
and C++20, and does not depend on `strix_core`. With `BUILD_TESTING=ON` it registers
`session_engine_contract`, `session_backend`, `session_kv` and
`session_prefix_cache`. It does not include standalone projects as subdirectories.
The early CUDA-only root path also includes this host module.
`strix_session_hip` is added only with `STRIX_ENABLE_HIP_RUNTIME`, using the
existing runtime include/library settings; no host test requires it.

Standalone host entry points remain `tests/session_backend` and
`tests/session_contract`. The backend and grouped-plan checks stay active under
`NDEBUG`. Host coverage includes offset copies, queue/event ordering, expired
and foreign resources, bounds and overflow refusals, destination preservation,
budget accounting, typed registry execution and error drain/poisoning.

HIP is unbuilt and unexecuted in this change's host validation. Native follow-up
must compile/link both launcher-present and launcher-absent configurations;
exercise native allocation/alignment, all copy directions, multiple queues and
events, owner rejection, probe failures, launch failures, drain and cleanup;
and compare adopted family token/logit/state results against the same numerical
identity before device qualification. CPU tests establish neither model
fidelity nor cross-device exactness or performance. Existing family qualification
and complete-workload gates remain required.

MiniMax's extracted service owns the native resources and Runner, while
`Lead::runtime()` supplies the shared owner. `SessionBackend`/`SessionPlan` in
`src/minimax/serve/session_engine.*` retain begin/unit/step/draft operation
boundaries; the extracted Runner executes the existing native calls. Its service
factory validates registry metadata and refuses execution in host-only builds.
The CPU add reference never substitutes for MiniMax arithmetic. Service host
fakes establish factory/HTTP/lifecycle behavior only. Combined owner review,
source-receipt review, paired physical state and device execution gates remain
open. No native resource portability, lane COW or serialization support is
claimed. See [serving ownership](session-layer.md) and the
[reference scope](../../tests/minimax_session_shared/README.md).
