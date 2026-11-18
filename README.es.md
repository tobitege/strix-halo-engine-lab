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
| Qwen3.8-Flash-Next | 1 nodo | 62.44 tok/s (MTP, código 1K) | 1,885 tok/s a 8K, 1,845-1,856 a 32K, 1,755-1,762 a 128K (exacto); 2,008 a 8K (`fast`, opcional; calificación de calidad en curso) |
| Qwen3.8-Flash-Next | 2 nodos (dos réplicas, servidor) | 158.1 tok/s en total con 8 flujos (68.2 con 1) | - |
| Qwen3.8-Flash-Next | 2 nodos (prefill en pipeline, opcional) | - | 2,968 tok/s a 32K (exacto), 3,191 (`fast`, opcional; calificación de calidad en curso); por debajo de 28,672 filas de prompt el par lee el prompt en un nodo |
| DeepSeek V4.1 Flash | par | 70.6 tok/s (DSpark, código) | 1,165 tok/s a 8K, 1,285 a 32K (exacto); 1,377 / 1,546 con el perfil FAST de prefill (opcional; [puerta de calidad relativa al suelo superada](docs/models/deepseek-v4.1/README.es.md#puerta-de-calidad-del-prefill-fast)) |
| MiniMax M3.1 Flash | par | 23.10 tok/s (1K) | 669 tok/s a 8K, 529 a 32K (exacto) |
| GLM-5.3 Flash | par | 44.5 tok/s (MTP, código) | 664 tok/s a 8K (exacto, segundo prefill en caliente), 642 a 32K (exacto); `fast` (opcional; calificación de calidad en curso) a 8K: 957 en caliente, 918 en proceso nuevo |
| Nemotron 3.5 Bolt E4B | 1 nodo | 91.9 tok/s (C1) | 2,780 tok/s a 8K, 2,336 a 32K (`fast`) |
| Nemotron 3.5 Bolt E4B | 2 nodos (prefill en pipeline, opcional) | - | 4,804 tok/s a 8K, 3,965 a 32K (`fast`, mismos bytes que 1 nodo) |
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

| Modelo | Uno / par | Exacto / FAST | Memoria | Pesos |
|---|---|---|---|---|
| Qwen3.8-Flash-Next (UD-Q4_K_XL) | uno (2 réplicas detrás de un router) | exacto por defecto; `fast` | 111 GB de pesos en un nodo de 128 GB | [GGUF UD-Q4_K_XL](https://huggingface.co/hyukudan/Qwen3.8-Flash-Next-GGUF-mirror) (réplica de `unsloth`, leído tal cual se publica); [cabeza de draft MTP Q8_0 destilada](https://huggingface.co/hyukudan/Qwen3.8-Flash-Next-MTP-distillG-Q8_0) |
| DeepSeek V4.1 Flash (expertos de 2 bits, denso trellis de 6.5 bits) | par por USB4 | exacto por defecto; `fast` | 2 x 128 GB | [GGUF de expertos](https://huggingface.co/hyukudan/deepseek-v4.1-flash-gguf-mirror), [capa densa](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-dense-U6), [packs trellis](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3) |
| MiniMax M3.1 Flash (EXL3 3.06 bpw) | par TP2 | exacto | 2 x 128 GB | [EXL3 3.06 bpw](https://huggingface.co/hyukudan/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw-mirror) (réplica de `Mia-AiLab`, con nuestro draft EXL3); [origen del draft](https://huggingface.co/hyukudan/MiniMax-M3.1-Flash-DFlash2-mirror) |
| GLM-5.3 Flash (EXL3 4.05 bpw) | par TP2 | exacto; prefill WMMA opcional | 2 x 128 GB | [EXL3 4.05 bpw](https://huggingface.co/hyukudan/GLM-5.3-Flash-exl3-mirror) (réplica de `turboderp`) |
| Nemotron 3.5 Bolt E4B (Q4_K_M; EXL3 para `quality`) | uno, compilación opcional; prefill del prompt en pareja opcional (`STRIX_G4_PAIR=1`, mismos bytes) | exacto por defecto; `fast`; `quality` | nodo de 128 GB | GGUF Q4_K_M oficial (no lo replicamos); [pack EXL3](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-EXL3); [drafter MTP](https://huggingface.co/hyukudan/Nemotron-3.5-Bolt-E4B-MTP-Drafter) |
| MiniMax H3 (audio + vídeo) | uno o par, compilación opcional | referencia densa; palancas FAST, cada una con su flag | 128 GB por nodo | no publicado por nosotros; origen [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/54472221d27c695fca9621b0c3ed52f06c4af707) ([ficheros](docs/models/minimax-h3/README.es.md#instalación-y-ejecución)) |
| MiniMax-H3-Image (int8 o W4A8) | uno o par (`strix-h3-image`), compilación opcional | VAE en GPU; `--parallel sequence` para una imagen, `request` para varias | 128 GB por nodo | no publicado por nosotros; origen [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/0bb149f51e3156ed4b90dd45d5620346bfe1df8f) |

Nuestros pesos en Hugging Face: [huggingface.co/hyukudan](https://huggingface.co/hyukudan). Las réplicas conservan los ficheros de origen; revisiones, nombres de fichero y licencias: [modelos y procedencia](docs/reproduce.md#models-and-provenance).

<a id="discrete-gpu"></a>

Rutas opcionales con RTX PRO 6000 (96 GB): DeepSeek independiente, DeepSeek RTX + par, descarga del prefill de Qwen,
servicio de Qwen y GLM en una tarjeta (borrador). Vea [uso de una GPU discreta](docs/discrete-gpu.es.md).

Cada interruptor, su valor por defecto y cuándo otro hardware quiere otro valor: [toggles](docs/toggles.es.md).

## Entrada de imágenes

DeepSeek V4.1 en el par de dos nodos acepta imágenes en chat completions al estilo OpenAI. Añada una parte `image_url`
a un mensaje de usuario. Está desactivada por defecto: arranque el servidor con `--vision <GGUF de visión>` en el rank 0,
junto con `--prefill-path device`.

```bash
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}]}'
```

Medido en el par de dos nodos, DeepSeek V4.1 Q2, un PNG de 37x23 ampliado a 1550 parches (200 filas de imagen + 21
tokens de texto): primera petición 1,46-1,87 s hasta el primer token (codificación de la imagen 0,93-1,07 s, trama de
2,0 MB al rank 1 en 4-5 ms, prefill del LM unos 0,46 s); la misma imagen otra vez 0,53-0,61 s (acierto de la caché de
embeddings); decode 39-41 tok/s.

Límites: solo PNG y JPEG; se rechazan las imágenes que superan el presupuesto de filas, el vídeo y las peticiones con
imagen y `session_id`. Los demás modelos aún no admiten imágenes. Detalles: [entrada de imágenes](docs/engine/vision.es.md).

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
- Mapa de los núcleos compartidos y qué modelo usa cada uno: [docs/engine/shared-cores.es.md](docs/engine/shared-cores.es.md).

## Documentación

| Tema | Documentos |
|---|---|
| Inicio | [primeros pasos](docs/getting-started/README.es.md) · [referencia del motor](docs/getting-started/engine-reference.md) · [reproducir](docs/reproduce.md) |
| Modelos | [Qwen](docs/models/qwen3.8/README.es.md) · [DeepSeek](docs/models/deepseek-v4.1/README.es.md) · [MiniMax](docs/models/minimax-m3.1/README.es.md) · [GLM](docs/models/glm-5.3/README.es.md) · [Nemotron](docs/models/nemotron-3.5-bolt-e4b/README.es.md) · [MiniMax H3](docs/models/minimax-h3/README.es.md) · [familias](docs/models/model-families.md) |
| Par y servicio | [par](docs/pair.es.md) · [configuración del par](docs/pair-setup.es.md) · [servicio](docs/serving.es.md) · [GPU discreta](docs/discrete-gpu.es.md) |
| Interruptores | [toggles](docs/toggles.es.md) · [optimizaciones](docs/optimizations.md) ([JSON](docs/optimizations.json)) |
| Números | [rendimiento](docs/performance.md) · [ganancias medidas](docs/measured-gains.es.md) · [metodología](docs/methodology.es.md) |
| Internos | [arquitectura](docs/architecture/README.md) · [núcleos compartidos](docs/engine/shared-cores.es.md) · [registro de decisiones](docs/decisions/decision-register.md) · [estado de implementación](docs/reference/implementation-status.md) · [fuentes](docs/reference/sources-ledger.md) |
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
