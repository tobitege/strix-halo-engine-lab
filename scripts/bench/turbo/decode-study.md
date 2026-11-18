# Decode anomaly study kit

The collector and HTTP client are source controlled. Scratch directories hold outputs. The deployment supplies one decoder binary, a model, sealed MTP assets, a server adapter, and the existing CPU/GPU lane scripts. No server recipe or private helper is imported implicitly.

## Deployment adapter contract

`TURBODEC_SERVER_ADAPTER` is an executable Bash recipe called as:

```sh
bash "$TURBODEC_SERVER_ADAPTER" "$TURBODEC_BIN" PORT STDERR_PATH [engine arguments...]
```

It must honor `TURBODEC_MODEL`, append the supplied arguments unchanged, apply the space-separated `SRVENV` assignments, respect the `SRVTIME` limit, and place the optional `WRAP` command before the engine. It owns the common numerical, MTP, golden, context, tokenizer and serving arguments. Declare those arguments and all asset hashes in the deployment receipt. E and T must use the same recipe and binary; only the product/weight switches differ.

The adapter must support a graceful stop and keep the process inside the ticket's cleanup boundary. Readiness uses the engine's listening message or `GET /v1/models`.

## Required paths and entry points

Set these to deployment inputs, then run from the repository:

```sh
kit="$PWD/scripts/bench/turbo"
export TURBODEC_ROOT="$(mktemp -d)"       # outputs only
export TURBODEC_BIN=/build/decoder
export TURBODEC_MODEL=/models/target.gguf
export TURBODEC_SERVER_ADAPTER=/deploy/server-adapter.sh
export TURBODEC_LAB=/deploy/lane-scripts
bash "$kit/decode-study-packed.sh" study 3 build-label
```

The packed runner performs CPU file pre-cache before each quiet `-time` ticket. Each ticket contains E then T, two sequential C1 requests per process, eager slots=2, MTP enabled by the adapter, and a 120-second GPU limit with an internal 115-second budget. It stops on an invalid arm. The default code prompt is the public 16-lane inventory fixture; the first lane reproduces the study's code request. `TURBODEC_CODE_PROMPT_MODULE` overrides it.

`decode-study-ticket.sh TAG BUILD_LABEL ARMS [ENV...]` also runs directly inside an existing GPU ticket. It gets the binary/model/adapter from the required variables above. `TURBODEC_PROMPT_MODULE` selects a Python prompt-data module exposing `PREFIX` and `LANES`; its default is the public four-class module. The client is `decode-study-client.py PORT OUT.jsonl ARMS MAXTOK` and loads no model itself.

`decode-study-control-ticket.sh` adds native-TURBO C1/C1 and two passes of four classes. `TURBODEC_CLASS_PROMPT_MODULE` defaults to code, prose, arithmetic, and JSON-instruction prompts. The JSON case is not a native tool-calling API fixture.

`decode-study-trace-ticket.sh` captures E/T kernel and HIP API traces with live host marks. It requires ROCprofv3. Optional same-server C8 is bounded, needs at least eight prompt lanes, and writes separate results. Never use traced throughput as qualification throughput.

## Accounting and outputs

The first request is cold and the last is warm. The primary decoder rate is the sum of actual receipt-emitted tokens divided by summed receipt cycle milliseconds. E is receipt-emitted tokens per completed pass. A completion token produced by prefill is retained separately.

The public client also derives cycle emissions as API accepted+cycles and checks `completion_tokens - cycle_emitted_tokens == 1` per lane. It reports this check, true cycle E, and explicitly named legacy client E. If the check fails, use the scoped receipts; do not invent a denominator. Accepted/proposed remains the API statistic and is not renamed acceptance among verified rows.

`decode-study-phase-summary.py` preserves physical/useful/padding rows and depth counts, attributes cycles with paired wall/monotonic anchors and client spans, and labels process-global exit marks separately. `decode-study-trace.py` uses paired monotonic/boottime anchors, scopes C1 and C8 separately, and groups actual kernels/API calls/live marks by request and physical row count. A zero-selected trace fails its scope check.

Sampled busy-clock work includes the busy factor and uses the cycle-emitted denominator in the receipt analyzer. It is a sampled estimate, not a hardware instruction counter. Layout bytes divided by time are payload-rate estimates; they are not measured DRAM traffic.

The study's packed adapters set the existing `STRIX_PLE_REWARM=0` on both arms to fit the startup budget. Record that workaround, readiness time, cold/warm prompt times and memory/fault state. This kit does not promote a runtime default or change model arithmetic.
