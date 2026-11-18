# Devices: declare a known card or a cluster in one line

[Español](devices.es.md)

Every GPU binary binds its card through the device declaration (`strix-devices-v1`, see
[deployment.md](deployment.md)). A device entry can name a **known device model**; every route then sizes itself to
that model: how much memory it plans for, which weights stay in VRAM, which go to pinned host memory or are read from
the model file, how many positions and session slots fit, and the minimum it needs when nothing fits.

## One line

On the command line of any route:

```
--device-profile rtx-5090-32
```

(or `STRIX_DEVICE_PROFILE=rtx-5090-32`). In the declaration file:

```toml
schema = "strix-devices-v1"

[devices.main]
uuid = "GPU-..."              # the card's identity, as before
profile = "rtx-5090-32"       # vendor comes from the profile
roles = ["replica"]

[devices.host]
kind = "cpu"
memory_gib = 64               # host RAM the routes may plan for (optional)
```

Any profile value can be overridden in the entry: `memory_gib = 30` (a smaller budget), `bandwidth_gbytes_ps`,
`units`. `profile = "custom"` takes every value from the entry (`memory_gib` required).

A **cluster of identical cards** is one entry without an identity:

```toml
[devices.pair]
profile = "rtx-5090-32"
count = 2
interconnect = "pcie5x16"     # pcie5x16, pcie4x16, pcie4x4, oculink4x4, usb4, 1gbe ... 100gbe
roles = ["replica"]
```

Such an entry is *planned*: the planner sizes it and predicts its decode rate; the gate never binds it (each card a
process runs on is still declared with its own identity). No catalogue card has NVLink, so `nvlink` is refused.

## Which budget wins

One resolver (`include/strix/device/budget.hpp`) serves every route, the CLI and the server alike:

1. the route's own flags (`--vram-total-gib`, `--ram-total-gib` and their environment variables), `none` = no cap;
2. the entry's `memory_gib` (host: the cpu unit's `memory_gib`);
3. the profile's memory;
4. nothing: the route uses what the card has free, as without a profile.

A declared budget larger than the card is refused at start. A budget smaller than the card is an emulation: the route
behaves as on the smaller device (free memory = the budget minus what is already on the card). The device receipt
line and `/api/status` show `profile=` and the budget only when one is declared.

| Route | What the budget sizes |
|---|---|
| DeepSeek V4.1 Flash, standalone RTX | whole-device caps of the tier planner: arena in VRAM, pinned host tier, the rest from the file |
| GLM-5.3 Flash, RTX producer | the same caps (`--device-profile` on the producer) |
| Qwen, RTX serve | weights stay in VRAM: refuses before the load when they do not fit; session slots from the budget |
| HIP routes planned by `--deploy-check` (DeepSeek serve, MiniMax, Gemma 4) | the device facts the deployment plan reads |

## Catalogue

Vendor public specifications unless the `measured` column lists the field. Peaks are dense TFLOPS / TOPS (FP16 and
FP8 at the FP16-accumulate rate; GeForce cards run BF16 at half that rate); 0 = no tensor path or not stated. Memory
for NVIDIA cards is the total the runtime reports. The data file is [data/devices/profiles.toml](../data/devices/profiles.toml)
(`strix-device-plan --toml`).

| profile | arch | memory GiB | GB/s | units | FP32 | FP16 | BF16 | FP8 | FP4 | INT8 | host link | measured |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| rtx-pro-6000-blackwell-96 | sm_120 | 95.59 | 1792 | 188 SM | 125.0 | 500.0 | 500.0 | 1000.0 | 2000.0 | 1000.0 | pcie5x16 | memory_bytes, units |
| rtx-5090-32 | sm_120 | 31.84 | 1792 | 170 SM | 104.8 | 419.0 | 209.5 | 838.0 | 1676.0 | 838.0 | pcie5x16 | - |
| rtx-5080-16 | sm_120 | 15.92 | 960 | 84 SM | 56.3 | 225.1 | 112.6 | 450.2 | 900.5 | 450.2 | pcie5x16 | - |
| rtx-4090-24 | sm_89 | 23.99 | 1008 | 128 SM | 82.6 | 330.3 | 165.2 | 660.6 | 0 | 660.6 | pcie4x16 | - |
| strix-halo-128 | gfx1151 | 117.19 | 256 | 40 CU | 29.7 | 59.4 | 59.4 | 0 | 0 | 59.4 | integrated | memory_bytes |
| dgx-spark-gb10-128 | sm_121 | 128.00 | 273 | 48 SM | 0 | 125.0 | 125.0 | 250.0 | 500.0 | 250.0 | nvlink-c2c | - |
| rx-9070-xt-16 | gfx1201 | 16.00 | 640 | 64 CU | 48.7 | 195.0 | 195.0 | 389.0 | 0 | 389.0 | pcie5x16 | - |

The RTX 5090 has the RTX PRO 6000's SM generation and memory speed with 170 of its 188 SMs: compute-bound phases run
about 10 % slower than on the PRO at equal memory; memory-bound decode at equal residency is close.

## Planning without a GPU

`strix-device-plan` prints the budget and the memory plan for a profile or a declaration, from a route's needs file
(`strix-route-needs-v1`):

```
strix-device-plan --list
strix-device-plan --profile rtx-5090-32 --needs scripts/bench/rtx/needs/qwen-serve.toml
strix-device-plan --devices scripts/bench/rtx/devices/2x-rtx-5090.toml --needs scripts/bench/rtx/needs/qwen-serve.toml
```

Exit 0 = fits, 3 = refused (the line names the minimum). For `count = N` shapes it adds a bandwidth-model prediction
(each card reads its share of the per-token bytes at the profile's bandwidth times the route's measured efficiency,
spilled bytes cross the host link, plus N-1 hops on the interconnect). It is a prediction: no shape is measured.

Qwen serve (weights resident, measured needs): RTX PRO 6000 fits 3 slots at 34,816 positions; it needs at least
91.24 GiB, so one RTX 5090 or 5080 refuses, and so does 2x RTX 5090 (46.62 GiB per card for an even layer split).

## Emulating a smaller card

On a large card, `--device-profile rtx-5090-32` makes a route plan for 31.84 GiB. The kit
[scripts/bench/rtx/devprof-emulate.sh](../scripts/bench/rtx/devprof-emulate.sh) runs one route command per profile on
one card pinned by UUID and keeps the budget, plan and result lines of each run. Measured with the equivalent
whole-device flags on one RTX PRO 6000 (24 GiB VRAM + 32 GiB RAM): DeepSeek V4.1 Flash standalone 6.55 tok/s
(6.75 with DSpark), GLM-5.3 Flash decode 0.67 tok/s warm; most experts are read from disk at that budget
([discrete-gpu.md](discrete-gpu.md)).
