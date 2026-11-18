# Install the release packages (0.1.0)

[Español](install.es.md)

Strix 0.1.0 ships as packages split by backend. A host build gives `strix-core`, `strix-tools` and `strix-docs`;
a HIP build gives `strix-hip`; a CUDA build gives `strix-cuda` (Linux) or one self-contained ZIP (Windows).
Packages contain the engine programs, the model registry (`share/strix/models`), the serve configuration, the
license notices and the user documentation. They never contain weights, mmproj files, draft models, tokenizers
extracted from models, lab code, artifacts or captures: models are yours to place (e.g. `/var/lib/strix/models`).

## Support matrix

**Qualified** means tuning and numerics were measured on that hardware (`qualified` in
`include/strix/hip/arch.hpp` for HIP); the published figures and exactness claims hold only there.
**Unqualified** means the code compiles for that target through the capability layer; nothing about its speed or
its numerics has been measured. `share/strix/support/build-<package>.txt` in every package lists the targets it
was compiled for and the status of each one.

| OS | Backend | Architecture | Status | Package |
|---|---|---|---|---|
| Linux x86_64 | HIP / ROCm | gfx1151 (Strix Halo, RDNA 3.5) | qualified | `strix-hip` DEB / RPM / tar.gz |
| Linux x86_64 | HIP / ROCm | gfx1200, gfx1201 (RDNA 4) | unqualified; does not compile yet, see [other AMD targets](#other-amd-targets) | none in 0.1.0 |
| Linux x86_64 | HIP / ROCm | gfx942, gfx950 (CDNA 3 / CDNA 4) | unqualified; does not compile yet, see [other AMD targets](#other-amd-targets) | none in 0.1.0 |
| Windows 11 x64 | CUDA | sm_120 (RTX PRO 6000 Blackwell) | qualified | `strix-cuda-qwen-*.zip`, `strix-cuda-ds41-*.zip` |
| Windows 11 x64 | CUDA | other `CMAKE_CUDA_ARCHITECTURES` (the DeepSeek backend names sm_89 and sm_86) | unqualified | build with `package.cmd ... ARCHS` |
| Linux x86_64 | CUDA | sm_120 and others | unqualified: CMake configures it, never measured | `package.sh ... cuda-qwen` |
| Linux, Windows | host only | x86_64 | no GPU code | `strix-core`, `strix-tools`, `strix-docs` |
| Linux | CPU backend (`STRIX_BACKEND_CPU`), Intel SYCL (`STRIX_BACKEND_INTEL`) | x86_64 / Intel GPU | scaffolds, opt-in source builds; no packaged program | none |

Two-node (TP2 / pair) serving needs two machines and the transport of [pair setup](pair-setup.md); it is not a
requirement of one-node packages, and no kernel module is shipped.

### Other AMD targets

`STRIX_HIP_ARCHS` (CMake) selects the HIP targets of every HIP program; `packaging/package.sh BUILD hip --hip-archs
"gfx1151;gfx1201"` builds one `strix-hip` for several targets, and its description and `build-hip.txt` mark each
one qualified or unqualified.

Measured for 0.1.0 (`--hip-archs "gfx1151;gfx1200;gfx1201;gfx942;gfx950"`): the build stops for gfx1200 and gfx942
in ten HIP sources that call RDNA 3 instructions without an architecture guard: the `wmma_*_w32` builtins (need
`wmma-256b-insts`, 75 errors), `permlanex16` (`gfx10-insts`, 28) and `sudot4` (`dot8-insts`, 20), for example
`src/hip_ds41_rn_wide.hip:110`, `src/core/integer_operand.hpp:60`, `src/moe_core/hip_ds41_prefill_variants.inc:219`,
`src/hip_q8_rows_gemv.hpp`, `src/core/f16_dense_tall.hpp`, `src/moe_core/dense_a8_small.inc`. Those call sites
have to select their instruction through the capability layer (`strix/hip_arch_caps.hpp`) before an RDNA 4 or
CDNA package can be built; until then `strix-hip` is gfx1151 only.

## Runtime dependencies

| Package | Needs | Tested with (not a declared minimum) |
|---|---|---|
| `strix-core` | glibc, libstdc++ | Ubuntu 26.04 |
| `strix-tools` | `strix-bitcalc-cmp`: Python >= 3.8 and NumPy | Python 3 of the distribution |
| `strix-hip` | ROCm HIP runtime (`libamdhip64.so`), `amdgpu` kernel driver with KFD (`/dev/kfd`, `/dev/dri/renderD*`), membership of `render` and `video` (the service gets both) | Ubuntu 26.04, kernel 7.4.0-rc1, HIP 7.16 (Ubuntu ROCm packages) |
| `strix-cuda` (Windows) | NVIDIA driver with CUDA 13.0 support; the MSVC C++ runtime ships in the ZIP and the CUDA runtime is linked statically | Windows 11, driver 617.14, CUDA 13.0 toolkit for the build |

The HIP DMA-BUF export path needs HIP 7 or newer; older headers make that path report "not supported". Memory per
model and the host settings that help it are in [memory budget](operations/memory-budget.md); the packages never
change kernel, GTT, IOMMU or boot settings.

DEB executable ABI dependencies come from `dpkg-shlibdeps`, including versioned C/C++ runtimes and HIP SONAME
providers. `strix-hip` additionally declares `libamdhip64-7 (>= 7.1.0) | hip-runtime-amd (>= 7.1.0)`
(override this additional constraint with `-DSTRIX_PACKAGE_DEB_HIP_DEPENDS=...`). Build-host ROCm package
requirements are retained; cross-distribution portability is not established. RPM requirements come from
`rpmbuild`'s automatic library scan (`libamdhip64.so.N`), so they resolve on any RPM distribution that ships ROCm.

## Install on Linux

Debian / Ubuntu:

```sh
sudo apt install ./strix-core_0.1.0-1_amd64.deb ./strix-tools_0.1.0-1_amd64.deb \
  ./strix-docs_0.1.0-1_all.deb ./strix-hip_0.1.0-1_amd64.deb
```

Fedora / RHEL / openSUSE (with a ROCm repository):

```sh
sudo dnf install ./strix-core-0.1.0-1.x86_64.rpm ./strix-tools-0.1.0-1.x86_64.rpm \
  ./strix-docs-0.1.0-1.noarch.rpm ./strix-hip-0.1.0-1.x86_64.rpm
```

tar.gz (no package manager): each archive mirrors the root layout (`usr/`, `etc/`). Extract into a directory and
run the programs from `<dir>/usr/bin`, or extract into `/` as root; then create the `strix` account and run
`systemctl daemon-reload` yourself (the DEB / RPM hooks do that).

Files: programs in `/usr/bin`, data in `/usr/share/strix`, notices in `/usr/share/doc/strix-*`, configuration in
`/etc/strix`, state in `/var/lib/strix`, unit `/usr/lib/systemd/system/strix.service`.

### The service

The packages create the `strix` system account and install `strix.service`, but never enable or start it.

1. Put the model where `strix` can read it, e.g. `/var/lib/strix/models/`.
2. Set `STRIX_MODEL` (first GGUF part) and optional `STRIX_SERVE_ARGS` in `/etc/strix/strix.env`.
3. Set HTTP options in `STRIX_SERVE_ARGS` (for example `--http-port 9000`). The unit supplies token `1` for
   CLI bootstrap and `--http-serve`; requests supply their own prompts. Defaults are `127.0.0.1:8080`.
   `serve.toml` is a reserved example and is not read by this service.
4. `sudo systemctl enable --now strix`, then `journalctl -u strix -f`; `curl http://127.0.0.1:8080/health`.

Upgrades keep both files (DEB conffiles, RPM `%config(noreplace)`). Removal stops the service and keeps
`/etc/strix`, `/var/lib/strix` and the account.

## Install on Windows (CUDA)

1. Extract `strix-cuda-qwen-0.1.0-windows-x86_64.zip` (or the DeepSeek `strix-cuda-ds41-*` ZIP).
2. `powershell -ExecutionPolicy Bypass -File install.ps1 -AddToPath` copies the programs to
   `%LOCALAPPDATA%\Programs\Strix` (no administrator rights; `-Prefix` changes it) and writes `serve.args`,
   `serve.toml` and `devices.example.toml` to `%ProgramData%\Strix` unless they exist (never overwritten).
3. Declare the GPU: copy `devices.example.toml` to `devices.toml` and put the UUID from `nvidia-smi -L`; the CUDA
   servers refuse an undeclared card ([declare the GPU](discrete-gpu.md#declare-the-gpu-to-use)).
4. Put the model options in `serve.args` (one option per line, see the comments in the file and
   [Qwen RTX serve](models/qwen3.8/rtx-serve.md)), then run `strix-start.ps1 -Console`.
5. To start it at logon: `install.ps1 -RegisterTask` registers the "Strix Server" scheduled task (not started
   now). The servers are console programs, so a scheduled task runs them without a service wrapper; output goes to
   `%ProgramData%\Strix\logs\strix-serve.log`.

`uninstall.ps1` removes recorded, unmodified program files and empty owned directories. It keeps configuration
(also inside the prefix), added models, pre-existing content and modified files. Installations without an
ownership manifest are left untouched. Installation refuses collisions with unowned or modified files.
`strix-start.ps1` passes `--serve-config serve.toml` to a server whose `--help` lists that option; the CUDA servers
take their options from `serve.args` until they bind the shared serve-config expander.

## Smoke test after installing

| Platform | Command |
|---|---|
| Linux, installed | `/usr/share/strix/support/smoke-core.sh --require-health` (also `smoke-hip.sh`; add `--model FILE.gguf` for the effective-config check) |
| Linux, extracted tar.gz | `bash usr/share/strix/support/smoke-core.sh --root DIR` |
| Windows | `powershell -ExecutionPolicy Bypass -File smoke.ps1`; Qwen: `-ServeArgs` checks HTTP health. DeepSeek: `-EffectiveConfigArgs` runs `--print-config`; its spool service has no HTTP health endpoint. Unsupported explicit checks fail. |

Checks: every `strix-*` program loads (no missing library or DLL) and answers `--help`; `strix-skeleton-serve`
answers `GET /health` = 200 without weights; with a model, `strix-hip-decode-token --print-effective-config`
resolves it (Linux) or the CUDA server answers `/health` = 200 (Windows). One PASS / FAIL / SKIP line per check;
exit code 1 on any FAIL.

## Build the packages

Version: `project(... VERSION 0.1.0)` in `CMakeLists.txt`. On a commit tagged `v0.1.0` the package release is `1`
(`0.1.0-1`); any other commit gets `0.git<sha>` (sorts before the tagged release). `--require-tag` (Linux) or
`-DSTRIX_PACKAGE_REQUIRE_TAG=ON` refuses an untagged or mismatching tree.

Linux (CMake >= 3.24, Ninja, a C++20 compiler, `dpkg-deb` for DEB, `rpmbuild` for RPM; HIP SDK for `hip`):

```sh
packaging/package.sh build/pkg-host host                          # strix-core, strix-tools, strix-docs
packaging/package.sh build/pkg-hip hip --hip-archs "gfx1151"      # strix-hip
packaging/package.sh build/pkg-cuda cuda-qwen --cuda-archs 120    # strix-cuda (Linux, unqualified)
```

Only the packaged programs are built. Each run writes `configure.log`, `build.log` and `cpack.log` in the build
directory and lists the packages (`.deb`, `.rpm`, `.tar.gz`, each with a `.sha256`). `--rpath /opt/rocm/lib`
records the ROCm library folder in the HIP programs when the ROCm install is not in the loader path.

Windows (Visual Studio 2022 x64, CUDA 13.0 in `CUDA_PATH`, CMake >= 3.24, Ninja):

```bat
packaging\windows\package.cmd build-pkg-qwen qwen 120
packaging\windows\package.cmd build-pkg-ds41 ds41 120 C:\src\tierstream
```

The ZIP is always built; CPack adds an NSIS installer when `makensis` is on `PATH`, or a WiX MSI when `candle` is.

CI: `.github/workflows/packages.yml` runs on `v*` tags and on pull requests that touch `packaging/`: the host job
builds DEB / RPM / tar.gz, checks their contents, installs the DEBs and runs the smoke test; the HIP job builds
`strix-hip` in the pinned ROCm container and smoke-tests the extracted archive. Hosted runners have no CUDA
toolkit and no GPU: the Windows ZIP is built with `package.cmd` on a CUDA machine, and the `/health` check with a
model runs on the target hardware.

## What each package contains

| Package | Programs (when the build mode creates them) | Other files |
|---|---|---|
| `strix-core` | `strix-serve`, `strix-skeleton-serve`, `strix-memory-plan`, `strix-model-describe`, `strix-deepseek41-expert-server` | model registry, skeleton example registry |
| `strix-tools` | `strix-inspect-gguf`, `strix-tokenize`, `strix-model-manifest`, `strix-deepseek41-inspect`, `strix-deepseek41-model-describe`, `strix-ds41-prepare-load`, `strix-gemma4-tokenize`, `strix-bitcalc-cmp` | |
| `strix-docs` | | README and `docs/` (English and Spanish) |
| `strix-hip` | `strix-hip-decode-token`, `strix-deepseek41-rank`, `-serve`, `-prefill`, `-prefix`, `-batch`, `-pipeline-front`, `-dspark-rank`, `strix-ds41-prefill-pair`, and the MiniMax / GLM / Nemotron programs when their families are built | `/etc/strix/serve.toml`, `/etc/strix/strix.env`, `strix.service`, sysusers entry |
| `strix-cuda` | `strix-cuda-qwen-serve`, `strix-cuda-qwen-prefill-service` (Qwen build); `strix-cuda-ds41-replica`, `-decode`, `-prefill`, `-prefill-service`, `-pipeline-front`, `-multi-front` (DeepSeek build) | Windows: MSVC runtime DLLs, `serve.args`, `serve.toml`, `devices.example.toml`, install / start / smoke scripts |

The lists live in `packaging/StrixPackaging.cmake`; the support matrix in `packaging/StrixSupportMatrix.cmake`
(`strix-package-support-tests` fails when it disagrees with `arch.hpp`).

## Package build record

| Package | Built on | Result |
|---|---|---|
| `strix-cuda-qwen-0.1.0-windows-x86_64.zip` (29 MB) | Windows 11, MSVC 14.44, CUDA 13.0, sm_120 | `--help` of both programs PASS; installed with a Qwen3.8-Flash-Next model and a declared RTX PRO 6000: weights 82.5 GB uploaded in 97 s, `GET /health` = 200; uninstall kept the configuration |
| `strix-core`, `strix-tools` DEB (amd64), `strix-docs` DEB (all), tar.gz | Ubuntu 26.04, GCC, host tree | Depends from `dpkg-shlibdeps` (`libc6 (>= 2.43)`, `libstdc++6 (>= 14)`; tools add `python3`, `python3-numpy`); extracted tar.gz smoke: 13 PASS (5 programs have no `--help` yet: `strix-deepseek41-expert-server`, `strix-inspect-gguf`, `strix-memory-plan`, `strix-model-describe`, `strix-model-manifest`), `strix-skeleton-serve` `/health` = 200; `strix-package-support-tests` passed |
| `strix-hip` DEB (amd64, 309 MB installed) + tar.gz, gfx1151 | Ubuntu 26.04, Ubuntu ROCm packages (HIP 7.16), 75 min cold build | Depends (measured before the tested-range change) `libamdhip64-7 (>= 7.1.0)` + the build host ROCm runtime package, conffiles `/etc/strix/serve.toml` and `strix.env`, postinst / prerm / postrm, unit and sysusers entry; every program resolves `libamdhip64.so.7`; description and `build-hip.txt`: `gfx1151: qualified` |
| RPM | not built on a node (no `rpmbuild` there) | built and listed (`rpm -qlp`) by the CI host job |
