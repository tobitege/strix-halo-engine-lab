# Windows: native build of the Qwen single-node engine

[Getting started](README.md) · [Engine reference](engine-reference.md) · [Target platform](../architecture/01-target-platform.md)

Linux is the primary runtime. This page describes an experimental native Windows build of
`strix-hip-decode-token` (Qwen3.8-Flash-Next, one node, `gfx1151`): what it needs, how it differs from Linux
and what is not available. Nothing here changes the Linux build.

The CI builds it too (`.github/workflows/windows.yml`): the engine and the host tests of the code this build
touches, on a hosted runner without a GPU. No kernel runs there and no model is loaded; those checks are local.

## What you need

- Windows 11 on a Strix Halo APU (`gfx1151`).
- GPU memory set aside in the firmware (BIOS) that holds the model, its state and the workspaces. The engine
  uses only this dedicated memory. The UD-Q4_K_XL model needs about 84 GB of it at its peak.
- A ROCm SDK for Windows with the HIP compiler (tested: TheRock ROCm SDK 7.14.1 for `gfx1151`).
- Visual Studio 2022 Build Tools with the C++ toolset, CMake and Ninja. The C++ library of Visual Studio 2026
  (MSVC 14.5x) does not compile with the HIP headers of this SDK; the build script selects 2022.

## Build

```bat
set ROCM_SDK=C:\path\to\_rocm_sdk_devel
scripts\windows\build-hip.bat all
```

The script configures `build-hip` with the SDK's `clang++` inside the Visual Studio environment, builds
`strix-hip-decode-token` and copies the SDK's HIP runtime DLLs next to it. Windows loads a program's DLLs
from its own folder first; without the copy it finds the display driver's HIP runtime in `System32`, which is
another version and fails at the first kernel launch.

Tests use their own folder:

```bat
set STRIX_BUILD_DIR=C:\path\to\build-hip-tests
set STRIX_EXTRA_CMAKE=-DBUILD_TESTING=ON
scripts\windows\build-hip.bat all strix-process-restart-tests strix-mapped-model-file-tests
```

Two settings differ from the Linux build, both set by the script:

- `STRIX_WARNINGS_AS_ERRORS=OFF`. On Linux the C++ sources are compiled with GCC; clang reports about 50
  more warnings in them (sign conversions, integer to floating point conversions, unused names). They were
  read one by one: none is a truncation through Windows' 32-bit `long`. The number parsing with `strtoul`
  and `stoul` on the Qwen path was checked separately; the one place that depended on a 64-bit `long`
  (`carried_pre_injected`) now uses `strtoull`.
- `_CRT_SECURE_NO_WARNINGS`. The Windows C runtime marks `getenv`, `fopen` and similar as deprecated.

## Memory: two pools

A Linux Strix Halo node has one memory pool for the host and the GPU. Under Windows the firmware sets the
GPU's dedicated memory aside, and Windows sees and allocates only the rest as RAM. The engine treats them as
two pools:

- **GPU pool:** the adapter's dedicated memory (its size from DXGI, its use from the adapter's
  `Dedicated Usage` counter). The model, the state and the workspaces are priced against it.
- **Host pool:** the RAM Windows reports as available. Metadata copies, native host pools, HTTP slot and
  session memory and the session and prefix cache defaults are priced against it. The automatic CLI prefill
  keeps 1 GiB of it free.

The model always goes into the GPU pool. The default residency registers the mapped model file, which would
keep the model in host RAM, so under Windows the copied arena (`STRIX_MODEL_COPY_ALL`) is the only form of
the default residency: unset and `0` mean `1`, and `fine` or `uc` select only the arena's memory type. The
file is copied in 32 MiB pieces, and each piece leaves the working set after its copy.

`--residency copy` is a different mode: it copies the model in one piece and keeps the mapped pages in the
working set until the copy is complete (23 GB of RAM on the 111 GB file in one measurement). Use the default.

These parts price host and device memory as one pool and decline under Windows:

- YaRN admission (long contexts beyond the model's native positions).
- Automatic prefill tiles for the HTTP server. Serving needs an explicit tiled configuration.
- PLE reclaim, rolling PLE and the resident PLE fallback.
- The late fit of the automatic prefix cache (it relies on cgroup and `/proc` observations).

## Other differences from Linux

- **Inspection initializes HIP.** On Linux `--print-effective-config` reads sysfs and exits before any
  device work. Windows has no such files: the device facts come from the HIP runtime (device properties
  only, no allocation and no kernel).
- **The eager code loading restart is a child process.** Windows has no `exec`. The program runs again with
  the same command line as a child in a job object and the first process waits and ends with the child's
  exit code. Ending the first process ends the child.
- **No pair runtime.** The two-node transport (USB4, TCP, GPU-direct) is built on Linux only. Pair prefill
  and the deployment pipeline refuse where they would open a link.
- **Model file identity.** The file is identified by volume serial number, file index, size, last write time
  and creation time. NTFS completes the last write time when the writing handle closes, so a same-size write
  through a handle that is still open is not seen.

## Measured on one machine

Ryzen AI Max+ PRO 495, Radeon 8065S, 160 GiB dedicated GPU memory, 31.7 GB RAM visible to Windows,
UD-Q4_K_XL (111 GB), CLI runs under a RAM watchdog, 2026-11-11. These are local measurements, not
qualified receipts.

| | Value |
|---|---|
| Full-file hash at first load | 811 MB/s |
| Lowest available RAM during load and run | 19.6 GB |
| Peak working set of the engine | 1.8 GB |
| Peak dedicated GPU memory | 84 GB |
| Prefill, 14,981 prompt tokens | about 19 s (about 790 tokens/s) |
| Decode | 45 to 63 ms per step |
