# GLM serving request ownership

`strix-glm5-serve` keeps its command line, defaults, routes and wire protocol. What
changed is who owns a request between the HTTP front end and the native pair.

## Before / after

Before, `Lead::run()` was a private request loop: it took jobs from the
`GenerationQueue`, held the tickets, published tokens and finished the jobs itself.

Now `Lead::loop()` enters `strix::serve::SessionRuntime::run()` (the shared owner
used by the other engines) through `GlmSessionOwner`:

- The runtime owns queue reservations, taken requests, logical leases, checked
  ticket callbacks and terminal accounting.
- The lead keeps one bounded `control_turn()` per runtime step, in the original
  order: prefix clear; cancellation of active / prefilling / importing requests;
  failed-import promotion; one admission; idle heartbeat; grouped plain or MTP / G2
  decode; import heartbeat and poll; one prefill chunk.
- Active, prefilling and importing records hold a generation-checked handle and a
  borrowed view of the request. A logical handle never addresses native state:
  `SlotTable`, slot pressure and the slot reserved by a waiting import stay in the
  lead.
- The lockstep control channel, the planner, calibration, speculative rounds, seeds
  and last rows are unchanged family code.

The adapter (`src/glm/serve/glm_session_owner.{hpp,cpp}`) reports `owner_control`
with the real context and slot limits. Generic prefill / decode, fork, append and
retention are not advertised, and the shared retaining cache is not constructed.

## Shared runtime additions used here

`SessionRuntime` gained an owner-control surface for native protocols:
`reserve_fifo`, `hold_reservation` / `promote` / `refuse_pending`, `inspect`,
`ticket_cancelled`, `notify_admitted`, `publish_progress`, `complete_native`, and
the hooks `native_validate`, `native_drained`, `drain_native_on_shutdown`,
`disable_native_cache`. They are additive; engines that do not set them keep their
behaviour.

## Shutdown and failures

A normal stop drains native work and keeps the shutdown-message order. An escaping
failure settles every active / prefilling / importing request, releases the logical
leases, retires host decoder and seed work and joins the import receivers before
the engine is destroyed. After a fatal protocol failure no further rank message is
sent to a peer in an unknown state.

## Not in this change

The exact prompt-prefix cache (`glm_serve_prefix_cache.hpp`) stays family code. Its
entries are committed in lockstep on both ranks with ids carried on the control
wire, so moving it onto the shared `PrefixStateCache` changes the rank protocol and
is a separate change with its own pair exactness gate. The RTX import stager is
unchanged for the same reason.

## Tests

Host: `ctest -L glm-owner` (see `tests/glm_session_owner/README.md`) and the
existing `strix-glm5-serve-cpu-tests`. Device: same greedy requests against the
previous and the new server (single request, two-turn conversation, prefix-cache
restore), token ids and text equal.
