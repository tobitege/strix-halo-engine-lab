# Uso de una GPU discreta (opcional)

[English](discrete-gpu.md) · [Español](discrete-gpu.es.md) · [README principal](../README.es.md)

Todos los motores funcionan solo con Strix Halo; ese es el valor por defecto y nada de esta página lo cambia. Una GPU
discreta de NVIDIA añade rutas aparte y opcionales. Cada ruta es su propia compilación y su propio comando, con su
propia identidad declarada y sus propios controles. Si no la compila ni la inicia, los valores por defecto de Strix
Halo no cambian.

## Qué se ha probado

| | Probado |
|---|---|
| Tarjeta | RTX PRO 6000 Blackwell Workstation Edition, 96 GB (`sm_120`) |
| SO / controlador | Windows 11 (WDDM), controlador NVIDIA 617.14 |
| Herramientas | CUDA 13.0, Visual Studio 2022 (MSVC x64), CMake 3.24 o posterior, Ninja |
| Presupuestos menores | DeepSeek independiente y GLM funcionan exactos con `--vram-total-gib 24 --ram-total-gib 32`; DeepSeek también con 32 + 32 |
| No probado | Otras tarjetas, hosts CUDA con Linux, presupuestos por debajo de los indicados |

El planificador de DeepSeek independiente rechaza presupuestos por debajo de unos 18.8 GiB de VRAM con
`--max-positions 2560` e indica el mínimo que necesita.

## Rutas de un vistazo

| Ruta | Qué se ejecuta dónde | Medido | Cuándo no ayuda |
|---|---|---|---|
| DeepSeek V4.1 Flash, RTX independiente | El modelo completo en la tarjeta: expertos enrutados Q2, densos Q8_0, caché de expertos en VRAM + memoria fijada del host, objetivo y borrador DSpark en la misma GPU | Tarjeta completa: 82.3 / 87.7 / 94.4 tok/s (código / inglés / español). 24 GiB de VRAM + 32 GiB de RAM, un NVMe, prompt de 96 tokens: 6.55 tok/s sin borrador, 6.75 tok/s con DSpark, con 4 lecturas de archivo en vuelo (5.45 tok/s con 1) | Con 24 GiB de VRAM + 32 GiB de RAM la mayoría de los expertos se leen del disco: 6.55 tok/s frente a 82-94 tok/s con la tarjeta completa |
| DeepSeek V4.1 Flash, RTX + par | La RTX ejecuta las capas 0-21 con CUDA; los dos nodos Strix Halo ejecutan las capas 22-39 y la cabeza, por 1 GbE | Decode por lotes 60.1 / 97.5 / 137.2 tok/s con 2 / 4 / 8 filas; lectura del prompt en la RTX 3.64 / 4.39 / 12.20 s con 2K / 8K / 32K | La ubicación de expertos calientes empeora las ventanas DSpark de 4 o más |
| Qwen3.8-Flash-Next, descarga del prefill | La RTX lee el prompt (prefill CUDA exacto) y envía el estado de atención y recurrente al nodo capa a capa; el nodo decodifica | TTFT 1.36 / 1.92 / 2.94 / 4.95 / 8.99 s con 1K / 4K / 8K / 16K / 32K, frente a 1.80 / 5.20 / 10.46 / 21.43 / 44.02 s en el mismo servidor sin la RTX | Por debajo de 1K tokens el enrutador mantiene los prompts en el nodo; con 1K el prefill de un nodo (1.12 s) es más rápido que la descarga (1.36 s) |
| Qwen3.8-Flash-Next, servido desde una RTX | Todo el modelo en una RTX PRO 6000: prefill CUDA exacto y decode MTP greedy exacto detrás de las API del nodo (`strix-cuda-qwen-serve`), batching continuo sobre 4 slots | Salida greedy exacta con cualquier número de slots; aún no se publica una cifra de rendimiento | Requiere la tarjeta de 96 GB: una sesión nunca se desborda a la memoria del sistema y un prompt más largo que el contexto se rechaza |
| GLM-5.3 Flash, una RTX (borrador) | Tronco, capas densas y expertos compartidos en la tarjeta; expertos enrutados servidos por la caché de [tierstream](https://github.com/hyukudan/tierstream) entre VRAM, RAM fijada y disco | Tarjeta completa: el solapamiento del prefill deja el TTFT en 0.56× con 8K y 0.66× con 32K; decode 6.0 / 6.2 tok/s en caliente con 1K / 8K. 24 GiB de VRAM + 32 GiB de RAM: TTFT 60.3 s con 1K, 62.4 s con 8K; decode 0.67 tok/s en caliente | Con presupuestos pequeños la mayoría de los expertos se leen del disco, así que el decode es lento |

Cada fila es exacta frente a su referencia declarada: los ids de token (y, donde se indica, los logits y el estado)
son iguales a los de la ejecución de referencia. Detalles y condiciones:
[DeepSeek independiente](models/deepseek-v4.1/README.es.md#deepseek-standalone-rtx-results),
[DeepSeek RTX + par](models/deepseek-v4.1/README.es.md#deepseek-rtx-pipeline),
[descarga del prefill de Qwen](models/qwen3.8/README.es.md#qwen-prefill-offload),
[Qwen servido desde una RTX](models/qwen3.8/rtx-serve.md),
[GLM en una RTX](models/glm-5.3/README.es.md#glm-rtx-results).

## Declare la GPU que se usa

Los binarios CUDA de DeepSeek y Qwen solo arrancan en una tarjeta que usted haya declarado. Cada uno lee un archivo
`strix-devices-v1` antes de su primera reserva de memoria, identifica las tarjetas por UUID (nunca por índice) y se
niega a arrancar, con una línea que enumera cada tarjeta y el motivo, cuando la tarjeta no está declarada, está marcada
como `excluded`, coincide con una política de denegación, o cuando hay más de una tarjeta declarada utilizable y no se
ha elegido ninguna. Una tarjeta no declarada es una tarjeta excluida.

```toml
schema = "strix-devices-v1"

[devices.main]                       # the table name is the device id
vendor = "nvidia"
uuid = "GPU-<uuid from nvidia-smi -L>"
roles = ["replica", "prefill"]       # no roles = no work
attach = { kind = "pcie", pcie_gen = 4, pcie_width = 16 }

[devices.other]                      # a card these binaries must never use
vendor = "nvidia"
uuid = "GPU-<uuid>"
excluded = true
reason = "kept for the desktop"

[device_policies.never-other]        # deny by name: still holds if the UUID changes
name_contains = "<part of the card name>"
excluded = true
```

Indique el archivo con `--devices FILE`, con `STRIX_DEVICES`, o guárdelo como `%APPDATA%\strix\devices.toml`
(`~/.config/strix/devices.toml` en Linux). Si hay varias tarjetas utilizables, elija una con `--device-id ID` o
`STRIX_DEVICE`. El binario restringe entonces su propio proceso a esa tarjeta (fija `CUDA_VISIBLE_DEVICES` al UUID
antes de iniciar CUDA y comprueba el resultado), de modo que ningún hilo puede alcanzar otra tarjeta, e imprime la
tarjeta que ha tomado:

```
device: id=main vendor=nvidia uuid=GPU-... pci=0000:01:00.0 name="..." arch=sm_120 ordinal=0 attach=pcie gen4 x16 roles=replica,prefill declared=devices.toml
```

`strix-cuda-qwen-serve` también la muestra en `device` dentro de `/api/status`.

Los binarios AMD (HIP) de un nodo pasan por la misma comprobación y aceptan las mismas opciones, con el dispositivo
identificado por su dirección PCI (`pci = "0000:c5:00.0"`, `vendor = "amd"`). Sin declaración, la única GPU integrada
del nodo sigue funcionando y la línea indica `declared=implicit`; una GPU AMD discreta o una segunda GPU siempre
necesitan el archivo.

Las herramientas de GLM de esta página aún no leen la declaración. Para ellas, y para cualquier otro programa CUDA,
seleccione la tarjeta por UUID en la consola:

```powershell
nvidia-smi -L                      # lists each GPU with its UUID
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
$env:CUDA_VISIBLE_DEVICES = "GPU-<uuid of the card to use>"
```

### Otras unidades: la NPU y la CPU

El mismo archivo puede declarar las dos unidades que no son GPU. Nada las usa si no se declaran: no hay NPU ni CPU
implícita, y un archivo antiguo sin estas entradas se comporta exactamente igual que antes.

```toml
[devices.npu]
kind = "npu"                         # gpu (por defecto) | npu | cpu
vendor = "amd"
pci = "0000:c5:00.1"                 # identidad, como en una GPU
arch = "aie2p"
roles = ["aux"]                      # solo modelos auxiliares (puntuación, visión, embeddings)
driver = "amdxdna"                   # las tres claves siguientes son obligatorias para conceder aux
firmware = "<versión del firmware>"
corun_qualification = "<id del recibo de ejecución conjunta GPU + NPU>"

[devices.host]
kind = "cpu"                         # sin identidad; como máximo una unidad cpu
cores = "2-15"                       # núcleos que puede fijar el carril del host
roles = ["host", "aux"]              # host: tokenizador, decodificación de imagen, muestreo, trabajo de fondo
byte_cap_gbytes_ps = 10.0            # techo del tráfico de memoria de fondo mientras una sesión decodifica
```

Reglas que aplican el analizador y la puerta de arranque:

- `decode`, `prefill`, `pair`, `draft`, `replica`, `front` y `experts` son roles de GPU. Una NPU solo admite `aux`; una
  CPU solo admite `host` y `aux`; una GPU también puede llevar `aux`.
- El rol `aux` de una NPU queda rechazado hasta que estén presentes `driver`, `firmware` y `corun_qualification`. El
  archivo se carga igualmente; la unidad simplemente no recibe trabajo.
- `cores` y `byte_cap_gbytes_ps` pertenecen a la unidad cpu, y las tres claves de cualificación a una NPU; en cualquier
  otro sitio son errores.
- La línea de arranque y el recibo listan las unidades declaradas después de la GPU, por ejemplo
  `units=cpu:host[host,aux cores=2-15 byte_cap=10GB/s],npu:npu[aux(refused)]`.

Declarar una unidad solo registra lo que tiene la máquina y lo que puede hacer. Hoy ningún motor envía trabajo a una NPU.

## DeepSeek V4.1 Flash, RTX independiente

Compile primero la biblioteca CUDA de tierstream y después, desde la raíz del repositorio con una copia de tierstream
en `TIERSTREAM`:

```powershell
cmake -S . -B build-rtx -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl `
  -DSTRIX_CUDA_DS41_ONLY=ON -DSTRIX_ENABLE_CUDA_DS41=ON `
  -DSTRIX_FAMILY_DEEPSEEK41=ON -DSTRIX_ENABLE_HIP_RUNTIME=OFF `
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRIX_TIERSTREAM_DIR=TIERSTREAM
cmake --build build-rtx --target strix-cuda-ds41-replica
```

Ejecución, el proceso persistente, presupuestos menores y todas las opciones:
[guía de la RTX independiente](models/deepseek-v4.1/rtx-standalone.md).

## DeepSeek V4.1 Flash, RTX + par

Requiere el par de dos nodos, una compilación CUDA (`STRIX_ENABLE_CUDA_DS41`) en el equipo de la GPU y un enlace de
1-2.5 GbE entre ellos. El puerto TCP de la canalización no tiene autenticación: manténgalo en una red privada.
Contrato de lanzamiento de las etapas: [etapas de la canalización](models/deepseek-v4.1/pipeline-stages.md).

## Qwen3.8-Flash-Next, descarga del prefill

El lado del nodo usa `--rtx-state-import` con `--http-serve` (sin `--http-batch`); con la opción desactivada nada
cambia. Compilación y protocolo del servicio: [servicio de prefill CUDA de Qwen](engine/serving/qwen-cuda-prefill.md) e
[importación de estado de la RTX](rtx-state-import.md).

## Qwen3.8-Flash-Next, servido desde una RTX

`strix-cuda-qwen-serve` sirve el modelo desde una RTX PRO 6000 sin nodo: API de OpenAI y Anthropic, SSE, llamadas a
herramientas y salida greedy exacta con cualquier número de slots. Compilación para Windows con
`-DSTRIX_CUDA_QWEN_ONLY=ON`; las compilaciones de Strix Halo no compilan estos ficheros. Compilación, opciones,
valores predeterminados e interruptores: [servicio Qwen en la RTX](models/qwen3.8/rtx-serve.md).

Una caché exacta de prefijos del prompt está activada por defecto (`--prefix-cache off` o
`STRIX_QRTX_PREFIX_CACHE=0` la desactiva): los turnos siguientes de un agente restauran su prefijo en caché y solo
hacen prefill de los tokens nuevos. [Caché de prefijos del prompt](models/qwen3.8/rtx-serve.md#prompt-prefix-cache).

## GLM-5.3 Flash en una RTX (borrador)

Compilación y después la ejecución medida (exactitud y velocidad; un servidor HTTP para esta ruta aún no se ha probado
de extremo a extremo):

```powershell
pwsh src/glm/rtxpf/producer/build.ps1 -BuildDir BUILD_DIR -Fresh -Parallel 2 `
    -TierstreamSource TIERSTREAM_DIR [-CutlassDir CUTLASS_DIR]
python -B src/glm/rtxpf/producer/decode_qualification_window.py --build BUILD_DIR `
    --model MODEL_DIR --prompt-dir PROMPT_DIR --unit-file OUT_DIR/glm-units.bin `
    --unit-manifest-sha256 MANIFEST_SHA256 --full-card --allow-gpu
```

Para una tarjeta menor sustituya `--full-card` por `--vram-total-gib 24 --ram-total-gib 32` (o sus propios límites).
El archivo de unidades, todas las opciones y las filas medidas: [guía de GLM RTX](../src/glm/rtxpf/README.md).

El servicio de GLM (`glm_native_service`) acepta además `--dgpu auto|<uuid>`: comprueba mediante NVML, antes de
cualquier llamada CUDA, que la tarjeta es una RTX PRO `sm_120` (`auto` elige la tarjeta válida con más memoria) y la
fija; `--dgpu off` se rechaza (esta ruta no tiene camino por CPU). El arnés de la ejecución medida de arriba selecciona
la tarjeta con `CUDA_VISIBLE_DEVICES`, como se describe en [Declare la GPU que se usa](#declare-la-gpu-que-se-usa).

## Solución de problemas

- **Un plan no cabe.** Los planificadores de DeepSeek y GLM lo rechazan antes de cargar e indican la memoria que
  necesitan; reduzca el contexto (`--max-positions`) o aumente los límites. GLM: `--plan-budget` muestra el reparto
  sin CUDA.
- **Compruebe primero la configuración.** DeepSeek independiente: `--print-config` imprime la configuración resuelta y
  termina antes de cualquier llamada CUDA. GLM: `--print-effective-config` imprime cada opción, su valor y su origen
  sin cargar el modelo.
- **Se usa la tarjeta equivocada.** Establezca `CUDA_DEVICE_ORDER=PCI_BUS_ID` y el UUID como arriba.
- **Las velocidades son menores que en la tabla.** Otros programas que usan la misma GPU (grabación de pantalla,
  captura u otros trabajos de cálculo) reducen todas las cifras; las mediciones de arriba se tomaron con la tarjeta sin
  otra carga.
- **Para quedarse con los valores por defecto de Strix Halo,** basta con no compilar ni iniciar estas rutas. Cada ruta
  tiene además sus propios interruptores de desactivación en su guía.
