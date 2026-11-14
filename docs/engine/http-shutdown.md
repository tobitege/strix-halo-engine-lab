# HTTP shutdown boundary

`HttpServer::stop()` stops admission and interrupts connection socket waits. It is
idempotent and may run concurrently with the accept loop or another stop call.
The caller joins its `serve()` thread before destroying the server. The server
destructor then drains every connection handler; objects captured by handlers
must remain alive through that drain. A handler doing application work still
needs its application's normal cancellation or queue shutdown signal.

Accepted descriptors become registered under the admission mutex before a
connection thread starts. Registration persists through graceful draining and
final close. Stop only shuts down registered sockets. The connection thread
closes and unregisters under the same mutex, so a later stop cannot shut down a
recycled descriptor. Thread-start failure follows the same retirement path.
The listener descriptor is pinned by a count of in-progress polls while readiness
polling runs outside the admission mutex. Accept checks stop under that mutex,
preventing admission after stop. Stop waits for existing polls to retire within
their 200 ms slice, then closes the listener before returning. Repeated stops
serialize the final close, preserving connection refusal and avoiding descriptor
reuse. Stop never waits behind a polling accept loop holding the mutex.

Socket descriptors are nonblocking on Windows and POSIX. Receive and send use
readiness waits with at most 50 ms slices, checking stop and retaining the
configured deadlines. This handles systems where a successful socket shutdown
does not wake an already blocked receive. Request header/body deadlines, the
idle timeout, write progress timeout, body/header budgets, connection limits,
HTTP response framing and SSE bytes keep their existing contracts. Graceful
close retains its one-second/one-MiB bound and skips draining during stop.
All these operations execute on host threads.

The [shutdown fixture](../../tests/http_shutdown/README.md) compiles the actual
shared server. These are host lifecycle checks. They establish no native numerical, capacity
or complete-workload performance qualification.
