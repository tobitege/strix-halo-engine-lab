# Kits de medida: qué mide cada uno y cómo se ejecuta

[English](benchmarking.md) · [Español](benchmarking.es.md) · [README principal](../README.es.md)

El benchmark público (`scripts/bench/bench.sh`, [reproduce.md](reproduce.md)) da una tabla por motor. Los kits de
esta página son las medidas del día a día del laboratorio: un brazo contra otro, en orden ABAB estricto, con la
validez de cada ejecución comprobada. Están en el repositorio para que sobrevivan a cualquier limpieza de las
máquinas que los ejecutan. Ninguno nombra un host, un usuario ni una ruta privada: binarios, modelos y prompts
se pasan como argumentos, y el segundo nodo de un par es el alias ssh de `KIT_PEER`.

## Ejecutor común (`scripts/bench/common/`)

| Fichero | Función |
|---|---|
| `ticket.sh OUT KIT [ARGS]` | La única orden que ejecuta un ticket de la cola de GPU. Presupuesto = `GPU_TURN_MAX` (120 s por defecto) menos `KIT_MARGIN` (5 s); los brazos que ya no caben se saltan. Termina con `OUT/summary.txt` (60 líneas como mucho) y `OUT/.done`, así que quien llama espera una vez y lee un solo fichero. |
| `runlib.sh` | Muestreador, precalentado de la caché de páginas, comprobación del binario en los dos nodos, presupuesto, `kit_run`, `kit_pair_arm` (rango 1 en el par, rango 0 aquí). |
| `run1.sh DIR CAP -- CMD` | Un proceso medido con su recibo; también se ejecuta en el par para el rango 1. |
| `receipt.py` | Convierte un directorio de ejecución en `receipt.json` y una línea de resumen (reglas abajo). |
| `abab.py` | `order` imprime el orden ABAB estricto; `summary OUT [--metric M]` da medianas por brazo, deltas por ronda y comprobaciones de identidad. |
| `serve-decode.sh` | Arranca un servidor (y su rango 1 en el par), espera a que esté listo, ejecuta `tools/serve_bench.py` por clase de decodificación y lo para. `--client-cmd` ejecuta cualquier otro cliente en su lugar. |
| `serve_receipts.py` | Convierte los resultados servidos en recibos (`agg_C<n>`, `e2e_C<n>`, `req_C<n>`, `ttft_C<n>`). |
| `forwards_ab.py` | Ciclos de decode estables por (carriles, pasadas del modelo) desde los recibos de ciclo: el A/B en proceso de `STRIX_SERVE_JOINT_GROUP_LANES=ab`. |
| `prompt_ids.py` | Ficheros de prompt de cualquier longitud (`p<N>.ids`, `.csv` o `.u32`). Cada uno es el texto semilla tokenizado una vez, repetido y truncado, con un sha256 por prompt. |

Cada directorio de ejecución guarda `cmd.txt`, `stdout`, `stdout.ts`, `stderr`, `stderr.ts` (la hora de llegada de cada línea),
`samples.tsv`, `klog.txt` y `meta.json`, más `receipt.json` una vez resumido.

**Validez** (una ejecución que falla cualquier comprobación se lista con sus motivos y queda fuera de todas las medianas):

- el código de salida no es 0;
- el `evicted_ms` de kfd de los procesos de la ejecución creció dentro de la ventana medida. Si crece durante la carga del modelo solo es un aviso; `--evict-whole-run` lo vuelve fatal, como hacen los kits de decodificación;
- `MemAvailable` bajó de `KIT_MIN_AVAIL_GB` (2 por defecto);
- faltan líneas `--require` o aparecen líneas `--reject` (por defecto: `what()`, `timed out` y `declined` fuera de las líneas de política de caché).

Los eventos del log del kernel (amdgpu, kfd, thunderbolt, oom, reset) se imprimen como avisos, o cuentan como fallo con `--strict-klog`.

**Caliente y frío.** Cuando un proceso ejecuta dos o más prefills, el primero se informa como frío y el último como
caliente. Las dos cifras nunca se mezclan. **Ciclos de GPU por token** = ocupación × reloj × ventana / tokens, con la
potencia y el reloj integrados sobre la ventana medida. En un host NVIDIA el muestreador lee `nvidia-smi`, y la
memoria del host y las comprobaciones de kfd figuran como no muestreadas.

**Brazos**: se escriben `'NOMBRE|ARGS CLI EXTRA|ENV=V ENV2=V'`. Las variables `STRIX_*` heredadas se eliminan, así que
un brazo solo ve las palancas que nombra. Para tener varias rondas repartidas en tickets, da a todos los tickets el
mismo `OUT`: cada resumen cubre todos los recibos de `OUT`, y `abab.py` empareja los brazos por `--round`.

**Ejecutar un kit**, aquí en un nodo a través de su cola (compila antes, en su propia orden, nunca en la misma shell
que el ticket):

```
<cola>/gpu-turn.sh ETIQUETA scripts/bench/common/ticket.sh OUT <kit> [args del kit]
```

## Kits

| Kit | Mide | Orden (dentro de `ticket.sh OUT ...`) |
|---|---|---|
| `qwen/length-matrix` | Prefill de Qwen, longitud × modo × perfil: `full` (un nodo), `pair` (dos rangos), `full2` (el mismo brazo en los dos nodos a la vez). Precalentado opcional de la caché de páginas (`--warm none` lo omite). Contexto largo (YaRN, de 262K a 1M) con un brazo con `--context N`. | `qwen/length-matrix --bin strix-hip-decode-token --model Q.gguf --ids DIR --sizes 8K,32K,128K,256K --modes full,pair --rounds 3` (brazos por defecto `exact`, `fast`) |
| `qwen/serve-decode` | Decodificación servida de Qwen, clases prosa / edición / json con 1, 4 y 8 chats; un brazo por ticket. | `qwen/serve-decode --bin B --model Q.gguf --arm 'wide\|--mtp-ngram-draft=wide\|' --round 0` |
| `ds41/prefill-pair` | Prefill del par DeepSeek V4.1, longitud × filas × brazo, frío y caliente en un proceso (`--ttft-timing --repeat 2`); el FNV de los logits de los dos rangos debe coincidir. | `ds41/prefill-pair --bin strix-deepseek41-prefill --model M --ids DIR --sizes 8K,32K --arm 'exact\|\|'` |
| `ds41/serve-decode` | Decodificación servida del par DeepSeek V4.1, clases prosa / edición / json con 1, 4 y 8 chats (recibos de ciclo del servidor, extremo a extremo, TTFT). | `ds41/serve-decode --bin-dir DIR --model M --arm 'ctl\|\|' --round 0` (luego `abab.py summary OUT --metric e2e_C8`) |
| `ds41/dense-forms` | Formas de decodificación de pesos (FP32 escalar / FP16 empaquetada) de la política densa compartida `dense-q8a8-f16-v1` en las formas por trozo del rango 0 de DS41: puerta exhaustiva del stager, identidad de bits de cada forma, tiempos de kernel ABAB estricto y censo por capa-trozo. | `ds41/dense-forms --bin strix-hip-ds41-fast-dense-tests [--rows 1024,680]` |
| `glm/prefill-pair` | Prefill del par GLM, exacto contra FAST, varios prompts en una carga (`--pf-ttft`), el primer prompt en frío y el último en caliente; acuerdo entre rangos en el FNV de logits y de estado. | `glm/prefill-pair --bin strix-glm5-rank --model DIR --prompts p8192.u32,p8192.u32` |
| `minimax/fast-gate` | Puerta de calidad FAST de MiniMax con la regla absoluta dNLL admitida: referencia y candidato, ambos rangos, luego seal y compare de `tools/strix-gate/fast_logit_gate.py` (`--dnll-rule abs`). `--rows 0` es la ruta de tiempos (la misma ejecución sin capturar filas de logits dentro del prefill cronometrado; las cifras de prefill salen de ella, no de una ejecución con volcado). Antes de lanzar, el kit comprueba que el rango par puede leer cada entrada (modelo, sesiones, campos de ruta de los argumentos del brazo) y se detiene sin tiempo de GPU si falta alguna. | GPU: `minimax/fast-gate run --bin strix-minimax-rank --model DIR --arm 'cand\|\|STRIX_X=1' --sessions s8k=p.u32`; CPU: `fast-gate.sh seal ...`, `fast-gate.sh compare --ref ref --cand cand --dnll-rule abs` |
| `minimax/serve-decode` | Decodificación servida del par MiniMax por clase con 1 y 8 peticiones, más el tiempo de arranque del servidor: listo es la línea `READY rank 0` del propio servidor y las líneas `startup:` de cada rango quedan en `startup.txt`. `--start-only` para al estar listo. | `minimax/serve-decode --bin strix-minimax-serve --model DIR --arm 'ctl\|\|' --round 0` |
| unidad: proyección de índice MiniMax | Slab exacto contra FAST `minimax-index-proj-wmma-v1` (directa, mosaicos x16, diagnósticos), ABAB estricto de 3 pares con 1.024 / 2.048 / 4.096 / 8.192 filas, subconjunto fijo de las primeras 12 (4.096/8.192 filas) o 30 (1.024/2.048 filas) de 60 capas; tiempos x16-only sujetos a igualdad completa de bytes con WMMA directa; cifras x16-only históricas retiradas hasta repetir la medición, primera llamada en frío, `evicted_ms` de kfd. | `STRIX_MMATTN_BENCH=wmma STRIX_MMATTN_WMMA_SERIES=x16 index-proj-tests` (línea de compilación en `src/minimax/attn/index_proj_tests.hip`) |
| `vision/battery` | Batería de contenido de visión: una respuesta greedy por imagen sintética, puntuada a partir de los píxeles (`score.py`): colores presentes o ausentes, patrón, orientación, regla de la paleta. | `vision/battery --bin-dir DIR --model M --vision V.gguf --images IMGDIR`, luego `python3 scripts/bench/vision/score.py IMGDIR OUT/*/battery.json` |
| `rtx/ds41-rtx-pair-batch` | Decodificación por lotes RTX + par de DeepSeek V4.1, C2 / C4 / C8: FRONT en la RTX sobre `[0, cut)` con `strix-cuda-ds41-multi-front`, TAIL en el par con `strix-deepseek41-batch --hetero-listen`, 2 grupos en vuelo; `--mode both` compara cada petición con su ejecución solo en el par. | ticket del nodo: `rtx/ds41-rtx-pair-batch tail --bin strix-deepseek41-batch --model M --prompts a.csv,...,h.csv --token-file TOK --n 2,4,8 --generate 48`; host RTX: `scripts/bench/rtx/ds41-rtx-pair-batch.sh front --bin strix-cuda-ds41-multi-front --model M --stage NODO:7841 --token-file TOK` |
| `rtx/serve-route` | Rutas RTX servidas (Qwen entero en una RTX, o cualquier servidor RTX compatible con OpenAI), clases de decodificación con 1, 4 y 8 chats. | `scripts/bench/rtx/serve-route.sh --route qwen --bin strix-cuda-qwen-serve --model Q.gguf -- --tables T --rope-table R --mtp-sidecar S --draft-vocab V` |
| `rtx/gpu_guard` | Guarda de 1 s que falla en cerrado para cualquier ejecución RTX (Windows o Linux): resuelve la GPU por nombre a su UUID, registra la telemetría de `nvidia-smi` (temperatura, margen hasta el límite de reducción, potencia, límite, memoria, relojes, PCIe gen/ancho) y los eventos WHEA / nvlddmkm / TDR de Windows; al primer evento, silencio de telemetría, margen térmico < 5 C, reintento PCIe, patrón de pérdida CUDA o límite de tiempo, mata la ejecución, escribe un cerrojo STOP y sale con código distinto de cero; un hijo fallido es un fallo. Solo lectura sobre la máquina. Todos los brazos/agentes deben compartir el mismo `--latch` explícito (o `GPU_GUARD_LATCH`); omitirlo desactiva STOP entre ejecuciones. Los eventos de Windows cubren toda la máquina, incluidos otros dispositivos; Linux no tiene cobertura del registro de eventos. Los sensores térmicos y contadores de reintentos son obligatorios antes del lanzamiento y durante la vigilancia; las mediciones opcionales incluyen recuentos de muestras en el resumen. Las cargas Linux deben conservar su grupo de procesos. | `python scripts/bench/rtx/gpu_guard.py --gpu-name "RTX PRO 6000" --out DIR --latch FICHERO_STOP --timeout-s 300 -- CMD ...` (resumen en `DIR/guard.json`) |
| `logprobs/partials` | Kernels de parciales de logprobs (HIP o CUDA): parciales del dispositivo frente a la referencia FP64 y al orden de carriles del host, y el coste del kernel por paso con 1 / 8 / 32 / 128 filas de un tramo de 64.640 entradas (el coste activado de puntuar). | `bash scripts/bench/logprobs/partials.sh --bin strix-hip-logprob-partials-tests --rows "1 8 32 128"` |
| `pfab` | A/B de prefill en par con una sola carga: `pf_pair_ab.sh --arms B:-/C:K=V+K=V --reps N` corre todos los brazos de un ABAB estricto dentro de un único par de procesos (`--pf-ttft-arms` / `--pf-ttft-reps` de la app de rango, `include/strix/env_arms.hpp` compartido; solo interruptores leídos en cada llamada), primer prefill frío, el resto caliente; `pf_clk.py` imprime por prefill el brazo, segundos, tok/s, ocupación, sclk, ciclos GPU por token, `evicted_ms` de KFD, MemAvailable mínima y el hash de logits. Para tickets bundle de 360 s. | `scripts/bench/pfab/pf_pair_ab.sh --bin BIN --model DIR --prompts A.u32 --out DIR --peer HOST --mode fast --arms B:STRIX_GLM_PF_MLA_WMMA=0/C:- --reps 4` |
| `kfdevict` | Evicciones de colas KFD sin root: sonda dentro del proceso (eventos de evicción propios con su causa, restauración, asignaciones userptr), muestreador del nodo de `evicted_ms` frente a contadores de reclaim / compactación y ventanas de GPU ocupada, generador de presión de memoria controlada y micro A/B de respaldos de memoria host. | `scripts/bench/kfdevict/README.md` |

El servicio de prefill DeepSeek en la RTX (`strix-cuda-ds41-prefill-service`) se mide con
`tools/ds41_prefill_service_client.py`. Las rutas en sí se describen en [discrete-gpu.es.md](discrete-gpu.es.md).

Los prompts de las clases de decodificación están en `scripts/bench/prompts/classes/`. La aceptación de los borradores
cambia un orden de magnitud entre prosa, una edición de código y CSV a JSON, así que la decodificación servida se
informa siempre por clase.

Barra de calidad TURBO: `tools/qualification/turbo_bar/` ([README](../tools/qualification/turbo_bar/README.md)) mide
E (ruta exacta) frente a R (un pack menor de los mismos pesos con los mismos núcleos) y T (TURBO): KL, top-1 y
dNLL sobre logits completos, tareas puntuadas, recuperación, intervalos pareados del 95 % (INCONCLUSO si no
resuelven) y una frontera velocidad/calidad. `rpack_kquant.py` produce R; `models.json` declara E/R/T por modelo.

## Comprobaciones

`scripts/bench/tests/run.sh` comprueba la sintaxis de todos los kits, pasa shellcheck si está instalado y ejecuta
`scripts/bench/tests/test_runner.py`, pruebas sin conexión de los recibos, las reglas de validez, el resumen ABAB,
los recibos servidos y los formatos de prompt.

Las puntuaciones de visión son heurísticas de palabras para imágenes sintéticas, no una medida general de comprensión visual. Las comparaciones ABAB requieren `--base ARM` explícito dentro de un grupo de cargas comparables.
