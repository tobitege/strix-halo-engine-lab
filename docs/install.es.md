# Instalar los paquetes de la versión (0.1.0)

[English](install.md)

Strix 0.1.0 se distribuye en paquetes separados por backend. Una compilación host da `strix-core`, `strix-tools` y
`strix-docs`; una compilación HIP da `strix-hip`; una CUDA da `strix-cuda` (Linux) o un ZIP autocontenido (Windows).
Los paquetes llevan los programas del motor, el registro de modelos (`share/strix/models`), la configuración de
servicio, los avisos de licencia y la documentación de usuario. Nunca llevan pesos, mmproj, modelos borrador,
tokenizadores extraídos de modelos, código de lab, artefactos ni capturas: los modelos los coloca el usuario
(p. ej. `/var/lib/strix/models`).

## Matriz de soporte

**Cualificado**: el ajuste y la numérica se midieron en ese hardware (`qualified` en `include/strix/hip/arch.hpp`
para HIP); las cifras publicadas y la exactitud solo valen ahí. **No cualificado**: el código compila para ese
destino a través de la capa de capacidades; no se ha medido ni su velocidad ni su numérica.
`share/strix/support/build-<paquete>.txt` de cada paquete lista los destinos compilados y el estado de cada uno.

| SO | Backend | Arquitectura | Estado | Paquete |
|---|---|---|---|---|
| Linux x86_64 | HIP / ROCm | gfx1151 (Strix Halo, RDNA 3.5) | cualificado | `strix-hip` DEB / RPM / tar.gz |
| Linux x86_64 | HIP / ROCm | gfx1200, gfx1201 (RDNA 4) | no cualificado; aún no compila, ver [otros destinos AMD](#otros-destinos-amd) | ninguno en 0.1.0 |
| Linux x86_64 | HIP / ROCm | gfx942, gfx950 (CDNA 3 / CDNA 4) | no cualificado; aún no compila, ver [otros destinos AMD](#otros-destinos-amd) | ninguno en 0.1.0 |
| Windows 11 x64 | CUDA | sm_120 (RTX PRO 6000 Blackwell) | cualificado | `strix-cuda-qwen-*.zip`, `strix-cuda-ds41-*.zip` |
| Windows 11 x64 | CUDA | otros `CMAKE_CUDA_ARCHITECTURES` (el backend DeepSeek nombra sm_89 y sm_86) | no cualificado | compilar con `package.cmd ... ARCHS` |
| Linux x86_64 | CUDA | sm_120 y otros | no cualificado: CMake lo configura, nunca medido | `package.sh ... cuda-qwen` |
| Linux, Windows | solo host | x86_64 | sin código GPU | `strix-core`, `strix-tools`, `strix-docs` |
| Linux | backend CPU (`STRIX_BACKEND_CPU`), Intel SYCL (`STRIX_BACKEND_INTEL`) | x86_64 / GPU Intel | andamiajes, compilación opcional desde fuente; sin programa empaquetado | ninguno |

El servicio con dos nodos (TP2 / par) necesita dos máquinas y el transporte de [configurar el par](pair-setup.es.md);
no lo necesitan los paquetes de un nodo, y no se distribuye ningún módulo de kernel.

### Otros destinos AMD

`STRIX_HIP_ARCHS` (CMake) elige los destinos HIP de todos los programas HIP; `packaging/package.sh BUILD hip
--hip-archs "gfx1151;gfx1201"` construye un `strix-hip` para varios destinos, y su descripción y `build-hip.txt`
marcan cada uno como cualificado o no.

Medido para 0.1.0 (`--hip-archs "gfx1151;gfx1200;gfx1201;gfx942;gfx950"`): la compilación se detiene para gfx1200 y
gfx942 en diez fuentes HIP que llaman a instrucciones RDNA 3 sin guarda de arquitectura: los builtins `wmma_*_w32`
(necesitan `wmma-256b-insts`, 75 errores), `permlanex16` (`gfx10-insts`, 28) y `sudot4` (`dot8-insts`, 20), por
ejemplo `src/hip_ds41_rn_wide.hip:110`, `src/core/integer_operand.hpp:60`,
`src/moe_core/hip_ds41_prefill_variants.inc:219`, `src/hip_q8_rows_gemv.hpp`, `src/core/f16_dense_tall.hpp`,
`src/moe_core/dense_a8_small.inc`. Esas llamadas deben elegir su instrucción mediante la capa de capacidades
(`strix/hip_arch_caps.hpp`) antes de poder construir un paquete RDNA 4 o CDNA; hasta entonces `strix-hip` es solo
gfx1151.

## Dependencias en ejecución

| Paquete | Necesita | Probado con (no es un mínimo declarado) |
|---|---|---|
| `strix-core` | glibc, libstdc++ | Ubuntu 26.04 |
| `strix-tools` | `strix-bitcalc-cmp`: Python >= 3.8 y NumPy | Python 3 de la distribución |
| `strix-hip` | runtime HIP de ROCm (`libamdhip64.so`), driver `amdgpu` con KFD (`/dev/kfd`, `/dev/dri/renderD*`), grupos `render` y `video` (el servicio los recibe) | Ubuntu 26.04, kernel 7.4.0-rc1, HIP 7.16 (paquetes ROCm de Ubuntu) |
| `strix-cuda` (Windows) | driver NVIDIA con soporte CUDA 13.0; el runtime C++ de MSVC va en el ZIP y el runtime CUDA está enlazado estático | Windows 11, driver 617.14, toolkit CUDA 13.0 para compilar |

La ruta de exportación DMA-BUF de HIP necesita HIP 7 o posterior; con cabeceras anteriores esa ruta responde "no
soportado". La memoria por modelo y los ajustes del host que la ayudan están en
[presupuesto de memoria](operations/memory-budget.md); los paquetes nunca cambian kernel, GTT, IOMMU ni arranque.

Las dependencias ABI de los ejecutables DEB salen de `dpkg-shlibdeps`, incluidos los runtimes C/C++ con versión
y los proveedores del SONAME HIP. `strix-hip` añade `libamdhip64-7 (>= 7.1.0) | hip-runtime-amd (>= 7.1.0)`
(esta restricción adicional se cambia con `-DSTRIX_PACKAGE_DEB_HIP_DEPENDS=...`). Se conservan los requisitos
ROCm del host; no se establece portabilidad entre distribuciones. Los requisitos RPM salen del
análisis automático de bibliotecas de `rpmbuild` (`libamdhip64.so.N`) y se resuelven en cualquier distribución RPM
con ROCm.

## Instalar en Linux

Debian / Ubuntu:

```sh
sudo apt install ./strix-core_0.1.0-1_amd64.deb ./strix-tools_0.1.0-1_amd64.deb \
  ./strix-docs_0.1.0-1_all.deb ./strix-hip_0.1.0-1_amd64.deb
```

Fedora / RHEL / openSUSE (con un repositorio ROCm):

```sh
sudo dnf install ./strix-core-0.1.0-1.x86_64.rpm ./strix-tools-0.1.0-1.x86_64.rpm \
  ./strix-docs-0.1.0-1.noarch.rpm ./strix-hip-0.1.0-1.x86_64.rpm
```

tar.gz (sin gestor de paquetes): cada archivo reproduce la estructura raíz (`usr/`, `etc/`). Extraer en un
directorio y ejecutar desde `<dir>/usr/bin`, o extraer en `/` como root; después crear la cuenta `strix` y
ejecutar `systemctl daemon-reload` a mano (en DEB / RPM lo hacen los scripts del paquete).

Ficheros: programas en `/usr/bin`, datos en `/usr/share/strix`, avisos en `/usr/share/doc/strix-*`, configuración
en `/etc/strix`, estado en `/var/lib/strix`, unidad `/usr/lib/systemd/system/strix.service`.

### El servicio

Los paquetes crean la cuenta de sistema `strix` e instalan `strix.service`, pero nunca lo habilitan ni lo arrancan.

1. Colocar el modelo donde `strix` pueda leerlo, p. ej. `/var/lib/strix/models/`.
2. Fijar `STRIX_MODEL` (primera parte GGUF) y, si hace falta, `STRIX_SERVE_ARGS` en `/etc/strix/strix.env`.
3. Fijar las opciones HTTP en `STRIX_SERVE_ARGS` (por ejemplo `--http-port 9000`). La unidad pasa el token `1`
   para iniciar el CLI y `--http-serve`; cada petición aporta su propio prompt. Por defecto: `127.0.0.1:8080`.
   `serve.toml` es un ejemplo reservado y este servicio no lo lee.
4. `sudo systemctl enable --now strix`, luego `journalctl -u strix -f`; `curl http://127.0.0.1:8080/health`.

Las actualizaciones conservan ambos ficheros (conffiles en DEB, `%config(noreplace)` en RPM). Al desinstalar se
para el servicio y se conservan `/etc/strix`, `/var/lib/strix` y la cuenta.

## Instalar en Windows (CUDA)

1. Extraer `strix-cuda-qwen-0.1.0-windows-x86_64.zip` (o el ZIP DeepSeek `strix-cuda-ds41-*`).
2. `powershell -ExecutionPolicy Bypass -File install.ps1 -AddToPath` copia los programas a
   `%LOCALAPPDATA%\Programs\Strix` (sin administrador; `-Prefix` lo cambia) y escribe `serve.args`, `serve.toml`
   y `devices.example.toml` en `%ProgramData%\Strix` si no existen (nunca se sobrescriben).
3. Declarar la GPU: copiar `devices.example.toml` a `devices.toml` con el UUID de `nvidia-smi -L`; los servidores
   CUDA rechazan una tarjeta no declarada ([declarar la GPU](discrete-gpu.es.md)).
4. Poner las opciones del modelo en `serve.args` (una opción por línea, ver los comentarios del fichero y
   [Qwen en RTX](models/qwen3.8/rtx-serve.md)), y ejecutar `strix-start.ps1 -Console`.
5. Para arrancar al iniciar sesión: `install.ps1 -RegisterTask` registra la tarea programada "Strix Server" (no la
   arranca ahora). Los servidores son programas de consola, así que una tarea programada los ejecuta sin envoltorio
   de servicio; la salida va a `%ProgramData%\Strix\logs\strix-serve.log`.

`uninstall.ps1` elimina los ficheros registrados sin modificar y los directorios propios vacíos. Conserva la
configuración (también dentro del prefijo), modelos añadidos, contenido previo y ficheros modificados. Sin
manifiesto de propiedad no elimina nada. La instalación rechaza colisiones con ficheros ajenos o modificados.
`strix-start.ps1` pasa `--serve-config serve.toml` a un servidor cuyo `--help` liste esa opción; los servidores
CUDA toman sus opciones de `serve.args` hasta que enlacen el expansor compartido de serve-config.

## Prueba tras instalar

| Plataforma | Orden |
|---|---|
| Linux, instalado | `/usr/share/strix/support/smoke-core.sh --require-health` (también `smoke-hip.sh`; `--model FICHERO.gguf` añade la comprobación de configuración efectiva) |
| Linux, tar.gz extraído | `bash usr/share/strix/support/smoke-core.sh --root DIR` |
| Windows | `powershell -ExecutionPolicy Bypass -File smoke.ps1`; Qwen: `-ServeArgs` comprueba HTTP. DeepSeek: `-EffectiveConfigArgs` ejecuta `--print-config`; su servicio de cola no ofrece HTTP. Las comprobaciones explícitas no compatibles fallan. |

Comprueba: cada programa `strix-*` carga (sin biblioteca o DLL ausente) y responde a `--help`;
`strix-skeleton-serve` responde `GET /health` = 200 sin pesos; con un modelo, `strix-hip-decode-token
--print-effective-config` lo resuelve (Linux) o el servidor CUDA responde `/health` = 200 (Windows). Una línea
PASS / FAIL / SKIP por comprobación; código de salida 1 si hay algún FAIL.

## Construir los paquetes

Versión: `project(... VERSION 0.1.0)` en `CMakeLists.txt`. En un commit con la etiqueta `v0.1.0` la revisión del
paquete es `1` (`0.1.0-1`); cualquier otro commit recibe `0.git<sha>` (ordena antes de la versión etiquetada).
`--require-tag` (Linux) o `-DSTRIX_PACKAGE_REQUIRE_TAG=ON` rechaza un árbol sin etiqueta o con otra versión.

Linux (CMake >= 3.24, Ninja, compilador C++20, `dpkg-deb` para DEB, `rpmbuild` para RPM; SDK HIP para `hip`):

```sh
packaging/package.sh build/pkg-host host                          # strix-core, strix-tools, strix-docs
packaging/package.sh build/pkg-hip hip --hip-archs "gfx1151"      # strix-hip
packaging/package.sh build/pkg-cuda cuda-qwen --cuda-archs 120    # strix-cuda (Linux, no cualificado)
```

Solo se compilan los programas empaquetados. Cada ejecución escribe `configure.log`, `build.log` y `cpack.log` en
el directorio de compilación y lista los paquetes (`.deb`, `.rpm`, `.tar.gz`, cada uno con su `.sha256`).
`--rpath /opt/rocm/lib` graba la carpeta de bibliotecas ROCm en los programas HIP cuando ROCm no está en la ruta
del cargador.

Windows (Visual Studio 2022 x64, CUDA 13.0 en `CUDA_PATH`, CMake >= 3.24, Ninja):

```bat
packaging\windows\package.cmd build-pkg-qwen qwen 120
packaging\windows\package.cmd build-pkg-ds41 ds41 120 C:\src\tierstream
```

El ZIP se construye siempre; CPack añade un instalador NSIS si `makensis` está en el `PATH`, o un MSI WiX si está
`candle`.

CI: `.github/workflows/packages.yml` se ejecuta con etiquetas `v*` y en pull requests que tocan `packaging/`: el
trabajo host construye DEB / RPM / tar.gz, revisa su contenido, instala los DEB y ejecuta la prueba; el trabajo
HIP construye `strix-hip` en el contenedor ROCm fijado por digest y prueba el archivo extraído. Los runners
alojados no tienen toolkit CUDA ni GPU: el ZIP de Windows se construye con `package.cmd` en una máquina CUDA, y la
comprobación de `/health` con un modelo se hace en el hardware de destino.

## Contenido de cada paquete

| Paquete | Programas (si el modo de compilación los crea) | Otros ficheros |
|---|---|---|
| `strix-core` | `strix-serve`, `strix-skeleton-serve`, `strix-memory-plan`, `strix-model-describe`, `strix-deepseek41-expert-server` | registro de modelos, registro del ejemplo skeleton |
| `strix-tools` | `strix-inspect-gguf`, `strix-tokenize`, `strix-model-manifest`, `strix-deepseek41-inspect`, `strix-deepseek41-model-describe`, `strix-ds41-prepare-load`, `strix-gemma4-tokenize`, `strix-bitcalc-cmp` | |
| `strix-docs` | | README y `docs/` (inglés y español) |
| `strix-hip` | `strix-hip-decode-token`, `strix-deepseek41-rank`, `-serve`, `-prefill`, `-prefix`, `-batch`, `-pipeline-front`, `-dspark-rank`, `strix-ds41-prefill-pair`, y los programas MiniMax / GLM / Nemotron si se compilan sus familias | `/etc/strix/serve.toml`, `/etc/strix/strix.env`, `strix.service`, entrada sysusers |
| `strix-cuda` | `strix-cuda-qwen-serve`, `strix-cuda-qwen-prefill-service` (Qwen); `strix-cuda-ds41-replica`, `-decode`, `-prefill`, `-prefill-service`, `-pipeline-front`, `-multi-front` (DeepSeek) | Windows: DLL del runtime MSVC, `serve.args`, `serve.toml`, `devices.example.toml`, scripts de instalación / arranque / prueba |

Las listas están en `packaging/StrixPackaging.cmake`; la matriz de soporte en `packaging/StrixSupportMatrix.cmake`
(`strix-package-support-tests` falla si no coincide con `arch.hpp`).

## Registro de compilación de paquetes

| Paquete | Compilado en | Resultado |
|---|---|---|
| `strix-cuda-qwen-0.1.0-windows-x86_64.zip` (29 MB) | Windows 11, MSVC 14.44, CUDA 13.0, sm_120 | `--help` de ambos programas PASS; instalado con un modelo Qwen3.8-Flash-Next y una RTX PRO 6000 declarada: pesos 82,5 GB subidos en 97 s, `GET /health` = 200; la desinstalación conservó la configuración |
| DEB `strix-core`, `strix-tools` (amd64), `strix-docs` (all), tar.gz | Ubuntu 26.04, GCC, árbol host | Depends de `dpkg-shlibdeps` (`libc6 (>= 2.43)`, `libstdc++6 (>= 14)`; tools añade `python3`, `python3-numpy`); prueba del tar.gz extraído: 13 PASS (5 programas aún sin `--help`: `strix-deepseek41-expert-server`, `strix-inspect-gguf`, `strix-memory-plan`, `strix-model-describe`, `strix-model-manifest`), `/health` de `strix-skeleton-serve` = 200; `strix-package-support-tests` pasa |
| DEB `strix-hip` (amd64, 309 MB instalado) + tar.gz, gfx1151 | Ubuntu 26.04, paquetes ROCm de Ubuntu (HIP 7.16), 75 min en frío | Depends `libamdhip64-7 (>= 7.1.0)` + el paquete del runtime ROCm, conffiles `/etc/strix/serve.toml` y `strix.env`, postinst / prerm / postrm, unidad y entrada sysusers; todos los programas resuelven `libamdhip64.so.7`; descripción y `build-hip.txt`: `gfx1151: qualified` |
| RPM | no construido en un nodo (sin `rpmbuild`) | lo construye y lista (`rpm -qlp`) el trabajo host del CI |
