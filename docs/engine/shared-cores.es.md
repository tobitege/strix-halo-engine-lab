# Núcleos compartidos: un mapa del código

*[English](shared-cores.md)*

El motor construye cada mecanismo una sola vez, en un hogar compartido, y una familia de modelos solo lo enlaza:
elige una política, aporta un adaptador o una clase de traits pequeña, o lee una fila de tabla. Esta página asocia
los hogares compartidos con el código y dice qué motor enlaza cada uno hoy. Es un mapa, no un contrato: las reglas
de diseño están en [docs/architecture](../architecture/README.md) y los interruptores en
[toggles](../toggles.es.md).

Motores citados: Qwen (Qwen 3.8, nivel superior de `src/` e `include/strix/qwen_*`), DeepSeek V4.1 (`*ds41*`,
`*deepseek41*`, `src/deepseek41/`), GLM-5.3 (`src/glm/`), MiniMax M3.1 (`src/minimax/`), Gemma 4 / Nemotron
(`src/gemma4/`) y la torre de visión (`src/vision/`). "Usuarios" lista los ficheros que incluyen la cabecera
compartida en la rama principal; es un punto de partida para leer, no un grafo de llamadas completo.

## Resumen

| Componente | Hogar | Ficheros / tipos principales | Qué enlaza un motor | Usuarios hoy | Documentación |
|---|---|---|---|---|---|
| Núcleo denso FP16 alto | `src/core` | `f16_dense_tall.hpp` (`ColumnTile`, `Plan`, `tail_policy`) | una política con nombre (abajo) | prefill KDA de GLM, lanzador GDN de Qwen, codificador de visión | [diseño de kernels](../architecture/05-kernel-design.md) |
| Política densa FAST Q8_0 x A8 | `src/core` | `f16_dense_q8a8.hpp` (`View`, `supported`, `enqueue_parts`), política `dense-q8a8-f16-v1` | una `View` de sus pesos y filas | DeepSeek V4.1 (palanca `ds41-dense-f16-v1`) | [toggles: perfiles](../toggles.es.md#perfiles-exacto-frente-a-fast) |
| Núcleo de prefill WMMA en mosaico | `src/core` | `tiled_core.hpp` (`GroupedArgs`, `NoAdapter`, `NoChain`), `wmma_tiled.hpp/.hip` | argumentos de plantilla `Adapter` y `Chain` | DeepSeek V4.1 (`src/ds41_integer_operand.hpp`), MoE de prefill de GLM y MiniMax (vía `moe_i4_chain.hpp`, `wmma_tiled.hpp`) | [prefill en mosaico](prefill/tiled-prefill.md) |
| Bucles principales enteros | `src/core` | `integer_mainloop.inc`, `integer_grouped_mainloop.inc`, `integer_grouped_schedule.hpp`, `integer_plan.hpp`, `integer_operand.hpp` | una política de operando y banderas de plan (`Partial`, `SplitStage`) | DeepSeek V4.1 (`src/ds41_dense_adapter.hpp`, `src/hip_ds41_prefill.hip`) | [cuantización y numérica](../architecture/06-quantization-and-numerics.md) |
| GEMV de pocas filas | `src/core` | `gemv_rows.hpp` con `gemv_rows_direct.inc` / `gemv_rows_staged.inc`; `gemv_policy.hpp` (`GemvKernel`, `GemvSelection`) | operando, suma de carril y epílogo del barrido | Qwen (`src/hip_q8_rows_gemv.hpp`), Gemma 4 (`src/gemma4/kernels/linear.hip`) | [diseño de kernels](../architecture/05-kernel-design.md) |
| Disposición de filas / regla de paso | `src/core` | `row_layout.hpp` (`RowLayoutPolicy`) | el paso de fila de sus búferes por filas | arena de ejecución compartida; ventana de prefijo conocido, lanzadores MoE y de experto compartido de Qwen | [disposición de filas](prefill/row-layout.md) |
| Códec trellis EXL3 | `src/core` | `trellis_format.hpp` (formato, referencia en host), `trellis_codec.hpp` (lector de dispositivo) | pesos trellis empaquetados leídos con el códec | Gemma 4 (`src/gemma4/exl3/`), MiniMax (`src/minimax/moe/`, `src/moe_core/exl3_trellis.hpp`), núcleo en mosaico | [códec trellis](kernels/trellis-codec.md) |
| Perfiles de motor | `include/strix` | `engine_profile.hpp` (`Name`, `Lever`, `Engine`, `Status`) y un `*_profile_table.hpp` por motor | una fila de tabla por palanca | Qwen (`profiles/qwen.hpp`, `qwen_profile_table.hpp`), DeepSeek V4.1 (`deepseek41_pair_profile_table.hpp`, `profiles/ds41_rtx.hpp`), GLM (`glm_pair_profile_table.hpp`), MiniMax (`src/minimax/engine/mm_profile_table.hpp`), Gemma 4 (`profiles/gemma4.hpp`) | [toggles: perfiles](../toggles.es.md#perfiles-exacto-frente-a-fast) |
| Ayudante del paquete FAST | `include/strix` | `fast_profile.hpp` (`Member`, `Bundle`, `lever()`, `expand()`) | un interruptor de paquete y sus palancas miembro | DeepSeek V4.1 (`ds41_fast_profile.hpp`), MiniMax (`mm_profile_table.hpp`) | [toggles](../toggles.es.md) |
| Declaraciones de palancas | `include/strix` | `lever.hpp` (`Decl`, `Regime`, `Choice`, `select`) | una fila `Decl`: interruptor, exactitud, valor por defecto, régimen | reglas compartidas del servidor (`serve/cache_budget_rule.hpp`, `serve/queued_prefix_protection.hpp`) | [regla de régimen](../toggles.es.md#la-regla-de-régimen) |
| Núcleo MoE agrupado de prefill | `src/moe_core` | `grouped_core.hpp` (`GroupedCore`), `grouped_policy.hpp` (`GroupedPrefillPolicy<Xor64, PipeTail>`, `GroupedLdsRing`), `grouped_f16.hpp` | decodificación de pesos, disposición de expertos, transformación de entrada, epílogo, traits, política | GLM (`src/glm/prefill/glm_moe_traits.hpp`), MiniMax (`src/minimax/moe/mmx_prefill_core.hpp`), Qwen (`src/hip_moe_q4k_a8_known_rows.hip`, política de almacenamiento) | [prefill agrupado de MiniMax](../minimax-grouped-prefill.md), [organizaciones MoE enrutadas](prefill/routed-moe-organizations.md) |
| Planificaciones de decode MoE | `src/moe_core` | `decode.hpp`, `decode_schedule.hpp`, `grouped_integer_decode.hpp` | traits: cadenas K, materialización de divisiones, epílogo | GLM (`src/glm/moe/glm_moe.hip`), MiniMax (`src/minimax/moe/mmx_moe.hip`, `mmx_moe_traits.hpp`), DeepSeek V4.1 (decode entero) | [diseño de kernels](../architecture/05-kernel-design.md) |
| Ayudantes del enrutador | `src/moe_core` | `route.hpp` (selección serie en k pasadas), `route_prune.hpp` (`RoutePrune` FAST), `device_plan.hpp` (planificador de rutas en dispositivo sin sincronización), `one_split_prefill.hpp` | una política de puntuación / elección | `route.hpp`: Qwen, DeepSeek V4.1, MiniMax; `route_prune.hpp`: Qwen; `device_plan.hpp`: GLM, MiniMax, Qwen; `one_split_prefill.hpp`: GLM | [poda de rutas MoE](kernels/moe-route-prune.md) |
| Flash dispersa, división de claves | `src/attn_core` | `sparse_flash_keysplit.hpp` (`KeySplit<H, D>`, `Split = true` operandos de dos términos) | listas de claves, escalas, máscaras (política) | MSA de MiniMax (`src/minimax/attn/msa_flash_policy.hpp`, `msa_fused.hip`) | [MSA de MiniMax](../minimax-msa.md) |
| Flash dispersa | `src/attn_core` | `sparse_flash_wmma.hpp` | listas de claves por fila | prefill de GLM, atención de DeepSeek V4.1 | [diseño de kernels](../architecture/05-kernel-design.md) |
| Prefill flash densa | `src/attn_core` | `dense_flash_prefill.hpp` (GQA causal FAST) | geometría de cabezas y tipo de operando | Gemma 4 (`src/gemma4/prefill/attention.hip`), codificador de visión | `src/gemma4/prefill/README.md` |
| Flash por bandas y seleccionada | `src/attn_core` | `banded_flash_prefill.hpp` (claves emparejadas), `selected_flash_prefill.hpp` (claves seleccionadas) | decodificación de claves, acceso a filas, máscaras, permutación de filas de valores | DeepSeek V4.1 (bandas), Qwen (seleccionada, `src/hip_indexed_attention_launcher.hip`) | [diseño de kernels](../architecture/05-kernel-design.md) |
| Puntuaciones de índice y selección | `src/attn_core` | `index_score_wmma.hpp` (FAST), `index_score_band.hpp` (top-k exacto con puntuaciones aproximadas), `index_score_mode.hpp`, `select.hpp` (selección radix exacta) | geometría del indexador, política de selección | GLM (`src/glm/prefill/glm_prefill.hip`, `glm_dsa.hip`), MiniMax (`msa_fused.hip`) | [index score WMMA](prefill/index-score-wmma.md) |
| Piezas de flash | `src/attn_core` | `flash_core.hpp`, `flash_tile.hpp`, `online_softmax.hpp`, `masks.hpp`, `wmma_frag.hpp` | (usadas por los núcleos de arriba) | todos los núcleos flash | - |
| Transporte del par | `src/xport` | `channel.hpp` (staging y progreso TP2), `transport_backend.hpp`, `staging.hpp` (núcleo de tickets / préstamos), `framed_exchange.hpp`, `row_fold.hpp` | ABI de cable en `Traits` | GLM (`src/glm/xfer/`), DeepSeek V4.1 (`src/hip_deepseek41_prefill_xport.hpp`), MiniMax (`src/minimax/xfer/`, `src/minimax/pfx/`) | [par](../pair.es.md), [communicator](transport/communicator.md), [plegado de filas del intercambio](decode/exchange-row-fold.es.md) |
| Políticas de la ruta del par | `include/strix/pair` | `link.hpp`, `pipe.hpp`, `balance.hpp`, `regime.hpp`, `prefill_group.hpp`, `lockstep.hpp`, `producer_publish.hpp`, `embed_frame.hpp` | número de capas, corte y tramas del motor | `link`/`pipe`: Qwen, Gemma 4; `balance`: Qwen; `regime`: Gemma 4; `prefill_group`: MiniMax; `lockstep`: GLM, DeepSeek V4.1, MiniMax; `producer_publish`: GLM; `embed_frame`: DeepSeek V4.1 | [prefill del par](prefill/pair-prefill.es.md), [multinodo](../architecture/12-multi-node-and-usb4.md) |
| Capa de sesión | `src/session`, `include/strix/session` | `engine.hpp`, `kv.hpp`, `backend.hpp`, `prefix_cache.hpp`, `parallel_job.hpp`, `embed_rows.hpp`, `embed_splice.hpp` | campos de identidad de un prefijo, filas de embedding de un prompt | caché de prefijos: caché de sesión de MiniMax, runtime de servidor compartido; splice de embeddings: DeepSeek V4.1 | [capa de sesión](session-layer.es.md), [caché de estado de prefijos](serving/prefix-state-cache.md) |
| Puertas de entrada del servidor | `src/serve`, `include/strix/serve` | `api.hpp`, `http_server.hpp`, `media.hpp`, `run.hpp`, `session_runtime.hpp` | un adaptador (recursos, motor, cola, manejador) | todos los motores servidos | [servidor](../serving.es.md), [interno del servidor](serving/serving.md) |
| Ayudantes de planificación del servidor | `include/strix/serve` | `prefill_interleave.hpp`, `prefill_group.hpp`, `token_memo.hpp`, `cycle_receipt.hpp` (`RunReceipt`), `row_budget.hpp` | llamadas de plan desde el planificador | intercalado: servidor de GLM; grupo de prefill: servidor de DeepSeek V4.1; memo de tokens: API de MiniMax, API compartida; recibo: servidor de DeepSeek V4.1, runtime MTP de Qwen, API compartida | [token memo](serving/token-memo.md), [cycle receipt](serving/cycle-receipt.md), [row budget](serving/row-budget.md) |
| Compuerta de paridad servidor / CLI | `include/strix` | `serve_cli_parity.hpp`, test `tests/serve_cli_parity_tests.cpp` | la configuración clave=valor de ambas entradas y las diferencias nombradas | GLM (`apps/glm_rank_defaults.hpp`) | - |
| Runtime de visión | `src/vision`, `include/strix/vision` | `preprocess.hpp`, `encoder.hpp` / `encoder.hip`, `api_media.hpp`, `embed_cache.hpp` (`EmbedCache`), `adapter.hpp` (`VisionAdapter`) | un `VisionAdapter` | DeepSeek V4.1 (`src/deepseek41/vision_adapter.hpp`, `apps/deepseek41_serve_main.cpp`) | [entrada de imágenes](vision.es.md), [adaptadores de visión](vision-adapters.md) |
| Borradores especulativos | `src/spec`, `include/strix/spec` | `drafting.hpp` (`Source`, `Proposal`, `ContextProposer`, `ArbiterConfig`), `drafting.cpp` | fuentes de borrador (serial, lookup, model) | Gemma 4 (`src/gemma4/engine/lookup.hpp`) | [decodificación especulativa](../architecture/09-speculative-decoding.md), [prompt lookup](speculative/prompt-lookup.md) |
| Compuerta de calidad | `tools/strix-gate` | `gate.py` (ejecutor A/B acotado), `fast_logit_gate.py` (compuerta FAST de logits, `--dnll-rule floor`) | una descripción de trabajo por motor | toda palanca FAST; la regla floor solo para el prefill FAST de DeepSeek V4.1 | [README de la compuerta](../../tools/strix-gate/README.md), [metodología](../methodology.es.md) |

## Los componentes

**Núcleo denso FP16 alto** ([`src/core/f16_dense_tall.hpp`](../../src/core/f16_dense_tall.hpp)). Una GEMM densa
FP16 para lotes de filas de prefill sobre WMMA. Es de clase FAST: las activaciones se redondean a FP16, una cadena
WMMA por mosaico desde +0 con k ascendente. Encima viven políticas con nombre. La política `dense-q8a8-f16-v1`
([`f16_dense_q8a8.hpp`](../../src/core/f16_dense_q8a8.hpp), espacio de nombres `strix::core::f16_dense_tall::q8a8`)
calcula pesos Q8_0 x activaciones A8 como una cadena FP16 por salida en lugar de los productos enteros exactos;
DeepSeek V4.1 la enlaza (ejemplo abajo).
Sus stagers decodifican dos pesos por operación FP16 empaquetada (`kWPacked`, `STRIX_F16_Q8A8_WPACK`, por defecto on, mismos
bits; el kit `ds41/dense-forms` mide ambas formas y comprueba los bits). El rasgo opcional `kColumnFast` pone los bloques de columnas en `grid.x`:
los CTA de un bloque de filas comparten x en L2 (mismos bits). La proyección de índice FAST de MiniMax
`minimax-index-proj-wmma-v1` ([`src/minimax/attn/index_proj.hpp`](../../src/minimax/attn/index_proj.hpp)) enlaza el
núcleo con `stage()` leyendo sus pesos empaquetados y, desde 2.048 filas, x redondeada antes en mosaicos FP16 por
k. Esa forma x16 es compartida (`X16Cols<Base>`, `enqueue_x16_tiles`, `kX16Rows` en el núcleo; mismos bits que la ruta
x FP32): MiniMax la usa en las proyecciones de índice (`STRIX_MM_PF_IDX_WMMA=1`, capas `STRIX_MM_PF_IDX_WMMA_LAYERS`,
por defecto 40-59, opcional hasta su gate), GLM en los grupos WMMA de x de `kda_wmma::dense_group` (KDA f_a | g_a | b;
`STRIX_GLM_PF_X16=0` = forma directa) y en sus proyecciones del indexador FAST `glm-pf-idxproj-wmma-v1`
(`STRIX_GLM_PF_IDXPROJ_WMMA=1`, opcional hasta su gate). Kit: `STRIX_MMATTN_BENCH=wmma` en `src/minimax/attn/index_proj_tests.hip` (ABAB estricto, series `x16` / `diag` /
`orders` en `STRIX_MMATTN_WMMA_SERIES`; línea de compilación en la cabecera del archivo).

**Núcleo en mosaico y bucles enteros** ([`src/core/tiled_core.hpp`](../../src/core/tiled_core.hpp)). El núcleo
WMMA de prefill de la biblioteca, solo cabecera para que los motores lo instancien con sus adaptadores en lugar de
mantener copias privadas. Sus bucles enteros (`integer_mainloop.inc`, `integer_grouped_mainloop.inc`) y la
planificación de registros / LDS (`integer_grouped_schedule.hpp`) son compartidos; `integer_plan.hpp` guarda las
banderas del plan.

**GEMV de pocas filas** ([`src/core/gemv_rows.hpp`](../../src/core/gemv_rows.hpp)). La GEMV de decode para 1 a 16
filas de activación que comparten un barrido sobre los pesos. Dos barridos se comparten como cuerpos textuales
(`.inc`); un kernel incluye uno y enlaza su operando, su suma de carril y su epílogo. `gemv_policy.hpp` elige la
forma del kernel.

**Disposición de filas** ([`src/core/row_layout.hpp`](../../src/core/row_layout.hpp)). El paso físico de fila de
los búferes de dispositivo por filas. Un kernel que recorre muchas filas a la vez toca direcciones separadas por un
paso, así que la regla del paso se fija aquí una vez ([disposición de filas](prefill/row-layout.md)).

**Trellis EXL3** ([`trellis_format.hpp`](../../src/core/trellis_format.hpp),
[`trellis_codec.hpp`](../../src/core/trellis_codec.hpp)). El formato de pesos empaquetados (C++ puro, lectores en
host) y la única fuente de operandos en dispositivo por la que todo motor lee pesos trellis
([códec trellis](kernels/trellis-codec.md)).

**Perfiles de motor y paquete FAST**
([`include/strix/engine_profile.hpp`](../../include/strix/engine_profile.hpp),
[`include/strix/fast_profile.hpp`](../../include/strix/fast_profile.hpp)). Un interruptor `--profile
default|fast|quality` elige el valor de cada palanca; cada palanca es una fila del `*_profile_table.hpp` del motor.
`fast_profile.hpp` es el paquete FAST de prefill compartido: un interruptor (por ejemplo
`STRIX_DS41_FAST_PREFILL=1`, `STRIX_MM_FAST_PREFILL=1`) elige palancas miembro con nombre, y una variable miembro
fijada gana al paquete. Sus dos lectores, `lever()` en el lado del dispositivo y `expand()` en el resolvedor del
host, comparten las mismas reglas. [`include/strix/lever.hpp`](../../include/strix/lever.hpp) declara una palanca
como datos (`Decl`: interruptor, exactitud, valor por defecto, regla por régimen) con `select` como única regla de
decisión.

**Núcleo MoE agrupado** ([`src/moe_core/grouped_core.hpp`](../../src/moe_core/grouped_core.hpp)). El núcleo de
prefill de expertos agrupados. No contiene política de lanzamiento: los adaptadores conservan sus firmas de kernel y
su política de filas. `GroupedPrefillPolicy<Xor64, PipeTail>`
([`grouped_policy.hpp`](../../src/moe_core/grouped_policy.hpp)) elige el almacenamiento LDS (XOR64 o con relleno) y
la política de filas de cola `kPipeTail`: los bloques parciales del ítem segmentado se ejecutan en `item_pipe` con
una guarda de fila en el almacenamiento, bit a bit, en lugar del núcleo escalar.

**Decode y enrutado MoE** ([`src/moe_core/decode.hpp`](../../src/moe_core/decode.hpp),
[`route.hpp`](../../src/moe_core/route.hpp), [`route_prune.hpp`](../../src/moe_core/route_prune.hpp),
[`device_plan.hpp`](../../src/moe_core/device_plan.hpp)). Las planificaciones de decode comparten una familia; los
traits conservan las cadenas K, la materialización de divisiones y los epílogos propios del motor. `route.hpp` es la
selección serie en k pasadas (la política es dueña de las transformaciones de puntuación, los empates y la
normalización). `route_prune.hpp` es la poda FAST que pone a cero las asignaciones despreciables antes de construir
el plan de rutas. `device_plan.hpp` es el planificador de rutas en dispositivo sin sincronización y su cola de
ítems.

**Núcleos de atención** ([`src/attn_core`](../../src/attn_core)). `sparse_flash_keysplit.hpp` es el núcleo FAST de
atención dispersa en forma de división de claves (listas de claves por fila hechas de segmentos; con `Split = true`
las consultas y probabilidades van en dos términos, hi + lo; los cuatro estados por wave se fusionan una vez, en
orden de wave). `sparse_flash_wmma.hpp` es la forma con listas de claves por fila, `dense_flash_prefill.hpp` el
prefill FAST denso causal GQA, `selected_flash_prefill.hpp` la planificación de claves seleccionadas,
`banded_flash_prefill.hpp` la de claves emparejadas. `index_score_wmma.hpp` es el núcleo FAST de puntuaciones del
indexador e `index_score_band.hpp` mantiene un top-k exacto sobre puntuaciones aproximadas. Todos comparten
`flash_tile.hpp`, `flash_core.hpp` y `online_softmax.hpp`.

**Transporte del par** ([`src/xport`](../../src/xport), [`include/strix/pair`](../../include/strix/pair)).
`channel.hpp` es el motor compartido de staging y progreso TP2; la ABI de cable propia del motor vive en sus
`Traits`. `staging.hpp` es el núcleo de tickets / préstamos y `framed_exchange.hpp` el entramado de un intercambio
comprobado. `include/strix/pair` guarda las políticas de la ruta: preparación del enlace (`link.hpp`), la costura
de tubería en dos etapas (`pipe.hpp`), el equilibrio de etapas (`balance.hpp`), la elección de modo por petición
(`regime.hpp`), el grupo de prefill (`prefill_group.hpp`), el canal de control en paso (`lockstep.hpp`) y la trama
de embeddings de medios (`embed_frame.hpp`, `pair::EmbedFrame`).

**Capa de sesión** ([`include/strix/session`](../../include/strix/session), [`src/session`](../../src/session)).
Entrada de modelo confirmada, instantáneas de KV, la caché de prefijos (los adaptadores aportan la identidad de
pesos, perfil numérico, formato de estado y topología de rangos; los campos desconocidos se rechazan), filas de
embedding en el flujo de tokens (`embed_rows.hpp`) y su inserción en una ejecución de prefill (`embed_splice.hpp`,
`embed_splice.hip`).

**Capa de servidor** ([`include/strix/serve`](../../include/strix/serve), [`src/serve`](../../src/serve)). Las
puertas de entrada de la API (`api.hpp`, `http_server.hpp`), un normalizador de partes de medios (`media.hpp`), la
forma del adaptador de un motor servido (`run.hpp`), el intercalado de prefill entre rondas de decode
(`prefill_interleave.hpp`), qué prompts cortos comparten una pasada de prefill (`prefill_group.hpp`), la memoria de
codificaciones de prompts (`token_memo.hpp`) y los recibos por ciclo y por ejecución (`cycle_receipt.hpp`,
`RunReceipt`). `include/strix/serve_cli_parity.hpp` es la compuerta de paridad: una familia resuelve una petición
por el servidor y por la CLI y `compare()` falla ante cualquier diferencia que no tenga nombre.

**Runtime de visión** ([`include/strix/vision`](../../include/strix/vision), [`src/vision`](../../src/vision)).
Preprocesado, un codificador ViT guiado por datos, el puente con la capa de medios de la API y una LRU de filas de
imagen proyectadas (`EmbedCache`). Una familia aporta un `VisionAdapter` ([adaptadores de visión](vision-adapters.md),
en inglés).

**Borradores especulativos** ([`include/strix/spec/drafting.hpp`](../../include/strix/spec/drafting.hpp)).
Propuestas de contexto, un árbitro y una línea de estadísticas para todo motor; las fuentes son `serial`, `lookup` y
`model`.

**Compuerta de calidad** ([`tools/strix-gate`](../../tools/strix-gate/README.md)). `gate.py` ejecuta trabajos A/B
acotados y compara la evidencia; `fast_logit_gate.py` es la compuerta FAST de logits. Su `--dnll-rule floor` fija el
límite de dNLL respecto a ejecuciones nulas y solo se permite para el prefill FAST de DeepSeek V4.1.

## Cómo enlaza un motor un núcleo

Ejemplo: DeepSeek V4.1 enlaza la política densa FAST Q8_0 x A8.

1. La política es una fila del paquete FAST del motor:
   [`include/strix/ds41_fast_profile.hpp:18`](../../include/strix/ds41_fast_profile.hpp) declara el miembro
   `STRIX_DS41_FAST_DENSE` = `1` como palanca `ds41-dense-f16-v1`, así que `STRIX_DS41_FAST_PREFILL=1` (o el
   miembro solo) la activa con las reglas compartidas de `fast_profile.hpp`.
2. El motor incluye la política compartida:
   [`src/hip_ds41_layer.hip:2`](../../src/hip_ds41_layer.hip) `#include "core/f16_dense_q8a8.hpp"`.
3. [`src/hip_ds41_layer.hip:3227`](../../src/hip_ds41_layer.hip) documenta la palanca; `fast_dense_on()`
   (línea 3234) la lee con `fast_profile::dense_on()`.
4. `enqueue_fast_dense` (línea 3238) rellena una `q8a8::View` (línea 3241) desde la `Q8A8DensePrefillView` propia
   del motor: pesos, códigos, deltas, bytes por fila, filas, pasos.
5. Línea 3247: las llamadas por debajo del mínimo de filas, o con formas que la política no acepta
   (`q8a8::supported`), devuelven false y conservan los kernels exactos. Línea 3248: `q8a8::enqueue_parts` lanza el
   kernel compartido.

El motor no añade ningún kernel: asocia sus búferes a la vista de la política y conserva la ruta exacta como
respaldo. MiniMax enlaza el núcleo agrupado del mismo modo:
[`src/minimax/moe/mmx_prefill_core.hpp:16`](../../src/minimax/moe/mmx_prefill_core.hpp) instancia `GroupedCore`
con su propia decodificación de pesos, disposición, transformación de entrada, epílogo y traits, y
`GroupedPrefillPolicy<false, Tail>` para la política de filas de cola (apagada por defecto).

## Añadir un nuevo usuario de un núcleo

Enlace el núcleo existente mediante su política, adaptador o traits; no copie un kernel dentro de un motor. Un modelo
nuevo de una familia existente solo necesita su descripción y su fila de registro
([añadir un modelo](adding-a-model.es.md)); una arquitectura nueva se describe en su sección de arquitectura nueva.
Un valor por defecto sigue siendo bit a bit; todo lo que cambia bits se publica bajo una política FAST con nombre y
pasa la [compuerta de calidad](../../tools/strix-gate/README.md)
([toggles: perfiles](../toggles.es.md#perfiles-exacto-frente-a-fast)).
