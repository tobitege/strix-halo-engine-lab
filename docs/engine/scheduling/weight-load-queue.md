# Weight load queue

One shared queue loads model weights (MiniMax and DeepSeek V4.1 use it today; GLM, Qwen and Gemma still
have their own loaders and move to it next): file ranges are read by a bounded set of workers into
per-ticket staging and copied to device memory on a copy stream of their own, so reading the next tensor
overlaps the upload (and any CPU conversion) of the previous ones. Engines describe what to load; they do
not own read loops, bounce buffers or staging buffers.

Sources: `src/core/weight_load_queue.hpp` (host only), `src/core/weight_file_source.hpp`,
`src/core/hip_weight_uploader.hpp`, `src/core/hip_range_weight_loader.hpp`.
Test: `strix-weight-load-queue-tests`.

## Parts

| Part | What it is |
| --- | --- |
| Manifest | What one ticket does: file ranges to read, an optional CPU conversion, and the staging ranges that leave the ticket (copied to a device address, or kept for a host consumer). It is built once and never changed by the queue. |
| Read workers | `STRIX_LOAD_WORKERS` threads (default 4) that read and convert tickets in parallel. They exist only while a load is open. They block in file reads and in the staging budget, which is why they are not host-lane jobs. |
| Staging | One buffer per ticket from a pool, bounded by a resident byte budget that the engine passes in (`STRIX_LOAD_STAGING_MIB` overrides it). A buffer returns to the pool when the ticket's upload event completed. Unbuffered reads land in the staging at the file's alignment; there is no shared bounce buffer. |
| Uploader | The HIP uploader owns a non-blocking stream and records one event per ticket. Nothing is submitted to the legacy stream. |
| Byte-rate governor | Reads draw from the process byte budget (`src/core/byte_budget.hpp`), so a load that runs while a decode is active is throttled like any other background byte stream. |

## Waiting

A submit never waits. There are three places that do:

- `wait(ticket)`: the first consumer of one tensor needs its bytes in device memory.
- `take(ticket)`: a host consumer needs the staged bytes of a held ticket (for example a library call that
  uploads and converts synchronously). Held tickets are taken in submit order.
- `publish()`: end of load. Every ticket is read and one stream wait covers every copy.

Tickets get their staging in submit order. A later ticket can therefore never take the staging an earlier
one needs, and a consumer that takes held tickets in order always makes progress, whatever the budget.

## Failure

A failed read or upload is stored on its ticket and rethrown by `wait`, `take` and `publish`. After the
first failure the remaining tickets are cancelled instead of read. If a copy was submitted but its event
could not be recorded, the uploader waits for the stream before it gives the staging back. Destroying the
queue joins the workers and waits for the copy stream, so no copy outlives its staging.

- Bookkeeping memory is taken at `submit` (and, in the HIP uploader, before the first copy is enqueued):
  an allocation failure fails that call and can never return staging that a copy still reads.
- A failed upload wait while a worker waits for staging is a ticket failure like any other.
- `cancel()` is the non-throwing exit for destructors and failure paths: unread tickets fail, the workers
  go idle and submitted copies complete; it does not wait for consumers of held tickets. `publish()` is
  the fallible step and belongs on the success path.
- A worker thread that cannot start makes the constructor throw after joining the started ones.

## Adapters

- File range to device pointer (`HipRangeWeightLoader`): `load(offset, bytes, device)` queues, `ready()`
  is the one wait. Ranges are cut into pieces and small consecutive ranges share one ticket
  (`WeightRangeTickets`, host only): boundaries follow the aligned staging a ticket takes, so any budget
  of at least 1 MiB loads any range; a smaller budget is refused before loading. DeepSeek
  V4.1 (prefill, batch forward and device forward) loads through it.
- MiniMax builds one manifest per EXL3 slice (reads, tile gather, three parts). Raw device slices are
  uploaded by the queue; prepared linears use held tickets, so the next slices are read while the
  current one is prepared.

## Identity check

With `STRIX_LOAD_HASH_OUT=<file>` a load writes one content-hash line per loaded tensor or slice (device
read-back for device copies, staged bytes for prepared linears). Two loaders are byte-identical when
their sorted hash files are equal.
