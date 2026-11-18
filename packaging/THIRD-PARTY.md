# Third-party notices (Strix 0.1.0 packages)

Strix is licensed under the Apache License 2.0 (`LICENSE`). The packages also contain, or are built from, the
following third-party code; each full licence text is in `LICENSES/`.

| Component | Where | Licence | Text |
|---|---|---|---|
| ISA-L CRC32C (adapted) | compiled into the programs (CRC32C checks) | BSD 3-Clause | `LICENSES/ISA-L-CRC32C.txt` |
| stb_image (`third_party/stb/stb_image.h`) | compiled into the programs that decode images | MIT or public domain (choice of the user) | `LICENSES/stb_image.txt` |
| strix-trellis kernels | compiled into the HIP programs that use trellis weights | Apache-2.0 | `LICENSES/strix-trellis-Apache-2.0.txt` |
| Microsoft Visual C++ runtime (`msvcp140*.dll`, `vcruntime140*.dll`, `concrt140.dll`) | Windows ZIP only, next to the programs | Visual Studio redistributable files (REDIST list of the Visual Studio licence) | Microsoft Visual Studio licence terms |
| NVIDIA CUDA runtime | linked statically into the Windows CUDA programs; no NVIDIA DLL is shipped | NVIDIA CUDA Toolkit EULA (static runtime) | the NVIDIA driver is installed by the user |

Not included in any package: model weights, mmproj files, draft models, vocabularies or tokenizers extracted from
models, logits references. The HIP and CUDA runtimes are dependencies of the packages, not part of them.
