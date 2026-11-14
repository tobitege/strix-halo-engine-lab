# Route receipt

A measurement is only worth something when it ran on the route that was meant. The route receipt says, for every
process and every served request, which kernel family actually ran each product, which levers were requested and
which took effect, and why anything was declined. The code is one header,
[include/strix/route_receipt.hpp](../../include/strix/route_receipt.hpp), with no device dependency.

## What it records

- **Launches and rows** per product (`dense`, `attention`, `index`, `router`, `moe`, `moe_gate_up`, `moe_down`,
  `head`, `core`) and kernel family (`exact`, `f16_wmma`, `iu4`, `fma_rows`, `fallback`).
- **Bypasses**: a call that asked for one family and ran another, with the family it wanted, the reason of the
  deciding branch, the number of calls and their rows.
- **Layers**: runs of consecutive layers that ran the same families for the same reasons, with the visits per reason.
- **Levers**: requested value, effective value, reason code and, where the selector has one, its own reason text.
- **Switches**: every governed switch that was set explicitly (the registry of `strix/env_flags.hpp`), and how many
  ran on their defaults.
- **Fingerprint**: a hash of the effective value of every numerical lever.

Snapshots are approximate process-window telemetry, not coherent transactions. Independent relaxed loads can
observe rows, calls and layer visits at different instants during launches. Concurrent request windows overlap;
their deltas are not isolated per-request counts. Exact totals are checked only after producers have stopped.
Each hook defines whether its count represents a logical call or a physical launch; these units are not interchangeable.
Current MiniMax MoE and GLM dense hooks count logical dispatch calls, which may enqueue multiple physical kernels.

Reason codes: `selected`, `disabled`, `wrong_route`, `layer_mask`, `short_call`, `format`, `shape`, `dependency`,
`assets`, `memory`, `build`, `qualification`, `nested`, `no_match`, `peer_disagreement`, `resource_limit`,
`launch_error`.

## Cost

A launch site pays one branch and two relaxed atomic increments on the host (four inside a layer scope). There is
no device work, no mutex on the launch path and no change to any output. Levers are recorded once at start-up.

## Where it is reported

- **Every binary that binds a device** prints one line on stderr when it exits:
  `strix-route-receipt {"schema":"strix-route-receipt-v1",...}`. stdout is unchanged.
- **The server** returns the block of the request window as `route` in the per-response receipt object and under
  `timings.route` where the envelope has timings (counters and fingerprint; a window on a server with concurrent
  requests also holds their launches), and the process totals with levers and switches
  under `route` in `/api/status`, beside the power view and under the same access rule.
- **Two ranks** compare the fingerprint before any product runs. A difference is an error: the MiniMax ranks
  exchange it after the model load, the GLM ranks carry it in the hello. Each rank reports its own counters.

## Adoption

| Engine | Family lines | Levers |
|---|---|---|
| MiniMax | MoE prefill per layer: `iu4` or `exact`, with the reason a layer keeps the exact operator (`disabled`, `short_call`, `layer_mask`); scope `gate_up` splits into `moe_gate_up` and `moe_down` | Effective IU4 enable, scope, row floor, kept and A8 layers; MSA flash mode/layers; prefill identity; MoE kernels |
| GLM | Dense prefill linears: `f16_wmma` or `exact`, with a bypass line when the WMMA identity is on and a call does not take it (`short_call` below the row floor); dense rows: `fma_rows` or `exact` | prefill mode, identity, WMMA identities, X route and its row floor |
| Qwen | none yet | entry point (CLI or HTTP), automatic prefill with the selector's reason, tile, resolved governed prefill identity/hash and budget/cap, numerical profile pair and GSQ owner |
| DeepSeek, Gemma | none yet | the shared default: explicitly set switches |

## Adding a site

```cpp
#include "strix/route_receipt.hpp"
strix::route::launched(strix::route::Product::dense, strix::route::Family::f16_wmma, rows);   // after the enqueue
strix::route::bypass(strix::route::Product::dense, strix::route::Family::f16_wmma,
                     strix::route::Reason::short_call, rows);                                 // at the deciding branch
strix::route::lever("engine.lever", requested, effective, strix::route::Reason::selected);    // once, at start-up
```

A lever whose value changes the arithmetic keeps `numeric = true` (the default) and so enters the fingerprint; an
informative lever (an entry point, a chunk size) passes `false`.
Prefer shared dispatch boundaries for new hooks where multiple engines use the same dispatch. Current family
adapters remain in place; moving them requires native selector-to-counter validation.

## Switch

`STRIX_ROUTE_RECEIPT=0` turns the counters and the printed block off. The lever list and the rank comparison stay.

## Tests

`strix-route-receipt-tests` (host): counters, the off switch, layer runs and request deltas, lever replacement,
fingerprint properties, the rank comparison (equal, different, a peer without it) and concurrent increments.
The concurrent test verifies totals after joining; it does not qualify snapshot coherence or native selector-to-receipt agreement.

The standalone host policy fixtures compile the production resolver and receipt bodies with device types and
launches excluded. They cover MiniMax policy disagreement, equivalent GLM settings, Qwen option identities and
GLM forced-route bypasses. They do not establish native engine agreement. Run only these host targets with:

```sh
cmake -S tests/route_policy_host -B artifacts/route-policy-host
cmake --build artifacts/route-policy-host --config Release
ctest --test-dir artifacts/route-policy-host -C Release --output-on-failure
```
