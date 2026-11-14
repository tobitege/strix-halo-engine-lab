# strix-router: one engine per node, one address for clients

When a model fits in the memory of a single node, run one full engine on each node and put `strix-router` in front.
Each node keeps its own memory bandwidth and its own prefill capacity; the link between nodes carries only HTTP.
Models that need both memories (pair-only placements) cannot use it.

The router is a separate binary. The engines are not changed and answer exactly as they do when addressed directly.

## Start

```sh
# on each node: the usual server, e.g. on port 8080
# on the node clients talk to:
strix-router --listen 0.0.0.0:8090 \
    --replica http://NODE_A:8080 --replica http://NODE_B:8080
```

Clients use `http://ROUTER:8090` with the same routes, bodies and credentials as before (headers pass through).

## Placement

1. A named session (`/v1/sessions/<id>/...`, header `X-Session-Id`, or body field `session_id`) returns to the
   replica that holds it.
2. Otherwise the replica that already saw the longest leading part of the prompt (hashes of 1 KiB blocks), unless it
   is more than one slot busier than the other.
3. Otherwise the least busy replica (`slots_busy + queue_depth` from each replica's `GET /api/status`).

A replica that stops answering, or reports `failed` or `draining`, receives nothing new. A request is moved to
another replica only when its connection was refused, before any byte was sent; after that a failure is returned to
the client (502, or a stream that ends without its terminator) and never retried, so nothing is generated twice.

## Check

- `GET /health`: 200 while at least one replica is healthy.
- `GET /router/status`: per replica health, capacity, requests in flight, and how requests were placed
  (`by_session`, `by_prefix`, `by_load`).
- Every forwarded response carries `X-Strix-Replica: <index>`.

## Options

`--poll-ms` (status poll period, default 250), `--connect-timeout-ms` (2000), `--idle-timeout-s` (0 = none),
`--block-bytes` (1024), `--affinity-entries` (4096), `--max-connections` (256), `--status-target` (`/api/status`).

## Roll back

Stop `strix-router` and point clients at one node's server again. Nothing on the nodes was changed.

## Limits

No TLS and no authentication of its own: bind it to a trusted interface, as with the servers. Responses are passed
through, not inspected, so per-token speed of a replica is not used for placement yet.
