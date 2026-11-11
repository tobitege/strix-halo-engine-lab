# Strix Halo engine lab

[English](README.md) · [Español](README.es.md)

Es un motor de inferencia en C++/HIP para AMD Strix Halo: Ryzen AI Max+ 395, Radeon 8060S (`gfx1151`), 128 GB de memoria
unificada, Linux y ROCm. Funciona en un nodo, o en dos nodos unidos por USB4. Una RTX PRO 6000 añade rutas opcionales.
Ejecuta Qwen3.8-Flash-Next, DeepSeek V4.1 Flash, MiniMax M3.1 Flash y GLM-5.3 Flash en producción.
Nemotron 3.5 Bolt E4B, MiniMax H3 (audio + vídeo) y MiniMax-H3-Image (texto a imagen) son compilaciones opcionales.
La ruta por defecto es exacta: sus tokens son idénticos a los de la ruta de referencia declarada.
Cómo medimos y qué cubre "exacto": [metodología](docs/methodology.es.md).

## Resultados

Valores por defecto publicados, decodificación greedy, una petición. Los números dependen de la carga. Condiciones,
repeticiones e interruptores de apagado en [rendimiento](docs/performance.md) y [ganancias medidas](docs/measured-gains.es.md).

| Modelo | Hardware | Decode | Lectura del prompt |
|---|---|---|---|
| Qwen3.8-Flash-Next | 1 nodo | 62.44 tok/s (MTP, código 1K) | 1,695 tok/s a 4K (exacto) |
| Qwen3.8-Flash-Next | 2 nodos (dos réplicas, servidor) | 158.1 tok/s en total con 8 flujos (68.2 con 1) | - |
| Qwen3.8-Flash-Next | 2 nodos (prefill en pipeline, opcional) | - | ~2,660 tok/s a 32K, ~2,800 a 128K (exacto) |
| DeepSeek V4.1 Flash | par | 70.6 tok/s (DSpark, código) | 1,057 tok/s a 8.4K, 1,172 a 32.5K (exacto) |
| MiniMax M3.1 Flash | par | 23.10 tok/s (1K) | 596 tok/s a 8K, 529 a 32K (exacto) |
| GLM-5.3 Flash | par | 44.5 tok/s (MTP, código) | 541 tok/s a 8K, 529 a 32K (exacto) |
| Nemotron 3.5 Bolt E4B | 1 nodo | 91.9 tok/s (C1) | 2,780 tok/s a 8K, 2,336 a 32K (`fast`) |
| MiniMax H3 | par | clip de 5 s 1344x768 con audio en 405.5 s | - |
| MiniMax-H3-Image (int8, 6 pasos) | 1 nodo | imagen 1024x1024 en 33.6 s, 2048x2048 en 52 s | - |
| MiniMax-H3-Image (int8, 6 pasos) | par | imagen 1024x1024 en 22.4 s, 2048x2048 en 44.6 s | - |

<a id="hardware"></a>

## Otro hardware

El planificador reserva la memoria medida del dispositivo para KV del contexto
solicitado, workspace y formatos/topologías compatibles antes de cargar. Consulte
[despliegue en otro hardware](docs/deployment.es.md) para inspección, ejemplos TOML,
API de orquestación y límites actuales, incluidos los runtimes INT4 pendientes.

## Inicio rápido

Un nodo. Necesita el toolchain de ROCm, el GGUF del modelo y un fichero `prompt.ids` generado con el tokenizador.

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

La ejecución imprime un recibo JSON con los IDs de token y los tiempos. Añada `--print-effective-config` para ver la
configuración resuelta antes de cualquier trabajo en el dispositivo.

Par (DeepSeek). Dos nodos de 128 GB y dos cables USB4. Ejecute en ambos nodos, con `--rank 1` y la otra dirección en
el segundo:

```bash
scripts/pair/ds41-pair init --rank 0 --peer user@node-b --model-dir /path/to/models
scripts/pair/ds41-pair check && scripts/pair/ds41-pair driver && scripts/pair/ds41-pair build && scripts/pair/ds41-pair fetch
scripts/pair/ds41-pair run --both    # solo rank 0
scripts/pair/ds41-pair smoke         # solo rank 0
```

El script imprime los pasos de root y nunca los ejecuta en silencio. Detalles: [configuración del par](docs/pair-setup.es.md).
La compilación y el lanzamiento de cada motor: [primeros pasos](docs/getting-started/README.es.md).

<a id="models"></a>

## Matriz de modelos

`default` es exacto. `fast` añade opciones cualificadas que cambian bits. `quality` usa un pack de mayor precisión
donde existe.

| Modelo | Uno / par | Exacto / FAST | Memoria |
|---|---|---|---|
| Qwen3.8-Flash-Next (UD-Q4_K_XL) | uno (2 réplicas detrás de un router) | exacto por defecto; `fast` | 111 GB de pesos en un nodo de 128 GB |
| DeepSeek V4.1 Flash (expertos de 2 bits, denso trellis de 6.5 bits) | par por USB4 | exacto por defecto; `fast` | 2 x 128 GB |
| MiniMax M3.1 Flash (EXL3 3.06 bpw) | par TP2 | exacto | 2 x 128 GB |
| GLM-5.3 Flash (EXL3 4.05 bpw) | par TP2 | exacto; prefill WMMA opcional | 2 x 128 GB |
| Nemotron 3.5 Bolt E4B (Q4_K_M; EXL3 para `quality`) | uno, compilación opcional | exacto por defecto; `fast`; `quality` | nodo de 128 GB |
| MiniMax H3 (audio + vídeo) | uno o par, compilación opcional | referencia densa; palancas FAST, cada una con su flag | 128 GB por nodo |
| MiniMax-H3-Image (int8 o W4A8) | uno o par (`strix-h3-image`), compilación opcional | VAE en GPU; `--parallel sequence` para una imagen, `request` para varias | 128 GB por nodo |

<a id="discrete-gpu"></a>

Rutas opcionales con RTX PRO 6000 (96 GB): DeepSeek independiente, DeepSeek RTX + par, descarga del prefill de Qwen,
servicio de Qwen y GLM en una tarjeta (borrador). Vea [uso de una GPU discreta](docs/discrete-gpu.es.md).

Cada interruptor, su valor por defecto y cuándo otro hardware quiere otro valor: [toggles](docs/toggles.es.md).

## Cómo funciona

- Un núcleo de kernels compartido: lectores GGUF y EXL3, kernels de matrices cuantizadas, atención, el planificador de
  memoria y los recibos. El núcleo no conoce ningún modelo.
- Cada familia de modelos es un módulo con su propio contrato, recorrido de capas, plan y compuertas
  ([familias de modelos](docs/models/model-families.md)).
- El motor elige kernels según el número de filas en vuelo. El decode de una fila lee los pesos más pequeños.
  La verificación ancha y el prefill usan las rutas de matrix cores y de enteros ([regla de régimen](docs/toggles.es.md#la-regla-de-régimen)).
- La decodificación especulativa (MTP, DSpark) mantiene el contrato exacto.
- Dos nodos ejecutan paralelismo de tensores (TP2) por dos cables USB4 directos: unos 37 Gb/s por sentido y 18.4 µs
  por un intercambio de 20 KB ([par](docs/pair.es.md)).
- Cada valor por defecto tiene una ganancia medida y un interruptor de apagado ([optimizaciones](docs/optimizations.md)).
- Añadir un modelo: [docs/engine/adding-a-model.es.md](docs/engine/adding-a-model.es.md).

## Documentación

| Tema | Documentos |
|---|---|
| Inicio | [primeros pasos](docs/getting-started/README.es.md) · [referencia del motor](docs/getting-started/engine-reference.md) · [reproducir](docs/reproduce.md) |
| Modelos | [Qwen](docs/models/qwen3.8/README.es.md) · [DeepSeek](docs/models/deepseek-v4.1/README.es.md) · [MiniMax](docs/models/minimax-m3.1/README.es.md) · [GLM](docs/models/glm-5.3/README.es.md) · [Nemotron](docs/models/nemotron-3.5-bolt-e4b/README.es.md) · [MiniMax H3](docs/models/minimax-h3/README.es.md) · [familias](docs/models/model-families.md) |
| Par y servicio | [par](docs/pair.es.md) · [configuración del par](docs/pair-setup.es.md) · [servicio](docs/serving.es.md) · [GPU discreta](docs/discrete-gpu.es.md) |
| Interruptores | [toggles](docs/toggles.es.md) · [optimizaciones](docs/optimizations.md) ([JSON](docs/optimizations.json)) |
| Números | [rendimiento](docs/performance.md) · [ganancias medidas](docs/measured-gains.es.md) · [metodología](docs/methodology.es.md) |
| Internos | [arquitectura](docs/architecture/README.md) · [registro de decisiones](docs/decisions/decision-register.md) · [estado de implementación](docs/reference/implementation-status.md) · [fuentes](docs/reference/sources-ledger.md) |
| Índice | [índice completo de la documentación](docs/README.md) · [AGENTS.md](AGENTS.md) |

<a id="related-repositories"></a>

## Repositorios relacionados

- [strix-halo-usb4-driver](https://github.com/hyukudan/strix-halo-usb4-driver): la serie de parches de Linux que convierte
  los controladores USB4 de Strix Halo en el enlace directo entre los dos nodos.
- [strix-trellis](https://github.com/hyukudan/strix-trellis): la biblioteca de cuantización trellis (estilo EXL3)
  usada para los pesos densos de DeepSeek y la inferencia de MiniMax.
- [TensorMeld](https://github.com/hyukudan/tensormeld): colocación y enrutamiento entre nodos y GPU; su router
  sirve las réplicas de Qwen.
- [tierstream](https://github.com/hyukudan/tierstream): streaming de expertos por niveles entre VRAM, memoria de host fijada
  y NVMe, usado por la ruta de DeepSeek en RTX independiente.

<a id="license"></a>

## Licencia

El código fuente del motor se distribuye bajo la **[Apache License 2.0](LICENSE)**, excepto el módulo del kernel de Linux
en `lab/direct_gpu_stream/kernel/`, que es **GPL-2.0-only** como debe serlo el código del kernel de Linux (sus textos
de licencia se conservan en ese directorio). El código adoptado de otros proyectos conserva su licencia y avisos originales. Los pesos
de los modelos se distribuyen bajo sus propias licencias. **Built with MiniMax M3.1-Flash**; la licencia de su modelo
es no comercial por defecto. Conserve los avisos del modelo y de la conversión y revise los términos aplicables. Las referencias de terceros usadas para los formatos y la semántica de los modelos
se enumeran con sus licencias en [docs/reference/sources-ledger.md](docs/reference/sources-ledger.md).
