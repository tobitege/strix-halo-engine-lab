# Strix Halo engine lab

[English](README.md) · [Español](README.es.md)

This is a C++/HIP inference engine for AMD Strix Halo: Ryzen AI Max+ 395, Radeon 8060S (`gfx1151`), 128 GB of unified
memory, Linux and ROCm. It runs on one node, or on two nodes linked by USB4. An RTX PRO 6000 adds optional routes.
It runs Qwen3.8-Flash-Next, DeepSeek V4.1 Flash, MiniMax M3.1 Flash and GLM-5.3 Flash in production.
Nemotron 3.5 Bolt E4B, MiniMax H3 (audio + video) and MiniMax-H3-Image (text to image) are opt-in builds.
The default path is exact: its tokens are identical to the declared reference path.
How we measure and what "exact" covers: [methodology](docs/methodology.md).

## Results

Shipped defaults, greedy decoding, one request. The numbers are workload-specific. Conditions, repeat counts and
off switches are in [performance](docs/performance.md) and [measured gains](docs/measured-gains.md).

| Model | Hardware | Decode | Prompt reading |
|---|---|---|---|
| Qwen3.8-Flash-Next | 1 node | 62.44 tok/s (MTP, 1K code) | 1,695 tok/s at 4K (exact) |
| Qwen3.8-Flash-Next | 2 nodes (two replicas, server) | 158.1 tok/s total with 8 streams (68.2 with 1) | - |
| Qwen3.8-Flash-Next | 2 nodes (pipelined prefill, opt-in) | - | ~2,660 tok/s at 32K, ~2,800 at 128K (exact) |
| DeepSeek V4.1 Flash | pair | 70.6 tok/s (DSpark, code) | 1,057 tok/s at 8.4K, 1,172 at 32.5K (exact) |
| MiniMax M3.1 Flash | pair | 23.10 tok/s (1K) | 596 tok/s at 8K, 529 at 32K (exact) |
| GLM-5.3 Flash | pair | 44.5 tok/s (MTP, code) | 541 tok/s at 8K, 529 at 32K (exact) |
| Nemotron 3.5 Bolt E4B | 1 node | 91.9 tok/s (C1) | 2,780 tok/s at 8K, 2,336 at 32K (`fast`) |
| Nemotron 3.5 Bolt E4B | 2 nodes (pipelined prefill, opt-in) | - | 4,804 tok/s at 8K, 3,965 at 32K (`fast`, same bytes as 1 node) |
| MiniMax H3 | pair | 5 s 1344x768 clip with audio in 405.5 s | - |
| MiniMax-H3-Image (int8, 6 steps) | 1 node | 1024x1024 image in 33.6 s, 2048x2048 in 52 s | - |
| MiniMax-H3-Image (int8, 6 steps) | pair | 1024x1024 image in 22.4 s, 2048x2048 in 44.6 s | - |

<a id="hardware"></a>

## Different hardware

The deployment planner budgets measured device memory for the requested KV context,
workspace and supported weight formats/topologies before loading. See
[deployment on different hardware](docs/deployment.md) for inspection, TOML examples,
orchestrator APIs and current integration limits, including pending INT4 runtimes.

## Quick start

One node. You need the ROCm toolchain, the model GGUF and a `prompt.ids` file from the tokenizer.

```bash
git clone https://github.com/hyukudan/strix-halo-engine-lab.git
cd strix-halo-engine-lab
export PATH=/opt/rocm/core-10.1/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/bin/amdclang++ \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON \
  -DSTRIX_ENABLE_EXPERIMENTAL_Q4K_A8_RUNTIME=ON \
  -DSTRIX_BUILD_EXPERIMENTAL_Q4K_A8_PROBE=ON \
  -DSTRIX_USE_OPENSSL_SHA256=ON -DBUILD_TESTING=ON
ninja -C build strix-hip-decode-token
./build/strix-hip-decode-token Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf $(cat prompt.ids) --generate 256
```

The run prints a JSON receipt with token IDs and timing. Add `--print-effective-config` to see the resolved
configuration before any device work. With the [drafter bundle](docs/engine/speculative/mtp-native-cli.md#default-mtp-drafter-bundle)
beside the model this command decodes with exact MTP speculation (the MTP figure above); without it, or with
`--no-mtp` / `STRIX_MTP=0`, it decodes the same tokens one at a time.

Pair (DeepSeek). Two nodes with 128 GB each and two USB4 cables. Run on both nodes, with `--rank 1` and the other
address on the second one:

```bash
scripts/pair/ds41-pair init --rank 0 --peer user@node-b --model-dir /path/to/models
scripts/pair/ds41-pair check && scripts/pair/ds41-pair driver && scripts/pair/ds41-pair build && scripts/pair/ds41-pair fetch
scripts/pair/ds41-pair run --both    # rank 0 only
scripts/pair/ds41-pair smoke         # rank 0 only
```

The script prints root steps and never runs them silently. Details: [pair setup](docs/pair-setup.md).
Every engine's build and launch command: [getting started](docs/getting-started/README.md).

<a id="models"></a>

## Model matrix

`default` is exact. `fast` adds qualified bit-changing options. `quality` uses a higher-precision pack where one exists.

| Model | Single / pair | Exact / FAST | Memory |
|---|---|---|---|
| Qwen3.8-Flash-Next (UD-Q4_K_XL) | single (2 replicas behind a router) | exact default; `fast` | 111 GB weights on a 128 GB node |
| DeepSeek V4.1 Flash (2-bit experts, 6.5-bit trellis dense) | pair over USB4 | exact default; `fast` | 2 x 128 GB |
| MiniMax M3.1 Flash (EXL3 3.06 bpw) | pair TP2 | exact | 2 x 128 GB |
| GLM-5.3 Flash (EXL3 4.05 bpw) | pair TP2 | exact; opt-in WMMA prefill | 2 x 128 GB |
| Nemotron 3.5 Bolt E4B (Q4_K_M; EXL3 for `quality`) | single, opt-in build; opt-in pair prompt prefill (`STRIX_G4_PAIR=1`, same bytes) | exact default; `fast`; `quality` | 128 GB node |
| MiniMax H3 (audio + video) | single or pair, opt-in build | dense reference; FAST levers each with its own flag | 128 GB per node |
| MiniMax-H3-Image (int8 or W4A8) | single or pair (`strix-h3-image`), opt-in build | GPU VAE; `--parallel sequence` for one image, `request` for several | 128 GB per node |

<a id="discrete-gpu"></a>

Optional RTX PRO 6000 (96 GB) routes: DeepSeek standalone, DeepSeek RTX + pair, Qwen prefill offload, Qwen serve and
GLM on one card (draft). See [using a discrete GPU](docs/discrete-gpu.md).

Every switch, its default and when other hardware wants a different value: [toggles](docs/toggles.md).

## How it works

- One shared kernel core: GGUF and EXL3 readers, quantized matrix kernels, attention, the memory planner and the
  receipts. The core knows no model.
- Each model family is a module with its own contract, layer walk, plan and gates
  ([model families](docs/models/model-families.md)).
- The engine picks kernels by the number of rows in flight. One-row decode reads the smallest weights.
  Wide verify and prefill use the matrix-core and integer paths ([regime rule](docs/toggles.md#the-regime-rule)).
- Speculative decoding (MTP, DSpark) keeps the exact contract.
- Two nodes run tensor parallel (TP2) over two direct USB4 cables: about 37 Gb/s per direction and 18.4 µs for a
  20 KB exchange ([pair](docs/pair.md)).
- Every default has a measured gain and an off switch ([optimizations](docs/optimizations.md)).
- Adding a model: [docs/engine/adding-a-model.md](docs/engine/adding-a-model.md).

## Documentation

| Topic | Docs |
|---|---|
| Start | [getting started](docs/getting-started/README.md) · [engine reference](docs/getting-started/engine-reference.md) · [reproduce](docs/reproduce.md) |
| Models | [Qwen](docs/models/qwen3.8/README.md) · [DeepSeek](docs/models/deepseek-v4.1/README.md) · [MiniMax](docs/models/minimax-m3.1/README.md) · [GLM](docs/models/glm-5.3/README.md) · [Nemotron](docs/models/nemotron-3.5-bolt-e4b/README.md) · [MiniMax H3](docs/models/minimax-h3/README.md) · [families](docs/models/model-families.md) |
| Pair and serving | [pair](docs/pair.md) · [pair setup](docs/pair-setup.md) · [serving](docs/serving.md) · [discrete GPU](docs/discrete-gpu.md) |
| Switches | [toggles](docs/toggles.md) · [optimizations](docs/optimizations.md) ([JSON](docs/optimizations.json)) |
| Numbers | [performance](docs/performance.md) · [measured gains](docs/measured-gains.md) · [methodology](docs/methodology.md) |
| Internals | [architecture](docs/architecture/README.md) · [decision register](docs/decisions/decision-register.md) · [implementation status](docs/reference/implementation-status.md) · [sources](docs/reference/sources-ledger.md) |
| Index | [full documentation index](docs/README.md) · [AGENTS.md](AGENTS.md) |

<a id="related-repositories"></a>

## Related repositories

- [strix-halo-usb4-driver](https://github.com/hyukudan/strix-halo-usb4-driver): the Linux patch series that turns
  the Strix Halo USB4 controllers into the direct link between the two nodes.
- [strix-trellis](https://github.com/hyukudan/strix-trellis): the trellis-quantization (EXL3-style) library
  used for DeepSeek's dense weights and MiniMax inference.
- [TensorMeld](https://github.com/hyukudan/tensormeld): placement and routing across nodes and GPUs; its router
  serves the Qwen replicas.
- [tierstream](https://github.com/hyukudan/tierstream): tiered expert streaming across VRAM, pinned host memory
  and NVMe used by the standalone DeepSeek RTX route.

<a id="license"></a>

## License

The engine source is licensed under the **[Apache License 2.0](LICENSE)**, except the Linux kernel
module under `lab/direct_gpu_stream/kernel/`, which is **GPL-2.0-only** as Linux kernel code must be (its license
texts are kept in that directory). Code adopted from other projects keeps its original license and notices. Model
weights are distributed under their own licenses. **Built with MiniMax M3.1-Flash**; its model license
is non-commercial by default. Preserve model and conversion notices and review the applicable terms. Third-party references used for formats and model semantics are
listed with their licenses in [docs/reference/sources-ledger.md](docs/reference/sources-ledger.md).
