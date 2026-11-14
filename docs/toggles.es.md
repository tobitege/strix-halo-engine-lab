# Toggles

[English](toggles.md)

Esta página lista los interruptores visibles para el usuario: flags de CLI, variables de entorno `STRIX_*` y opciones de CMake.
Dice qué hace cada uno, cuál es el valor por defecto y cuándo otro hardware querría otro valor.
No repite ganancias medidas. Ganancias, precondiciones y recibos están en
[optimizations.md](optimizations.md) y en el archivo legible por máquina [optimizations.json](optimizations.json).
Los números por modelo están en [performance.md](performance.md) y [measured-gains.md](measured-gains.md).

Los valores por defecto están ajustados para un objetivo: gfx1151 (Ryzen AI Max+ 395) con 128 GB de memoria unificada,
y un par de estas máquinas sobre USB4. En ese objetivo el comando simple ya ejecuta la ruta de producción medida.
Solo necesitas esta página para inspeccionar una ruta, depurar o llevar el motor a otro hardware.

## Reglas comunes a todos los toggles

- Un argumento en la línea de comandos gana a un valor por defecto. Una variable ya fijada en el entorno nunca se cambia.
- `--print-effective-config` imprime el entorno y los argumentos resueltos antes de tocar el dispositivo.
- `=0` (o el flag `--no-*` indicado) apaga un elemento que está activo por defecto.
- Si falta una precondición (pack, biblioteca, memoria), el elemento se descarta y se imprime una línea en stderr.
- Las apps de par necesitan el mismo valor en los dos ranks.

## Perfiles: exacto frente a FAST

| Perfil | Qué hace |
|---|---|
| `--profile default` | Ruta de producción actual; no exporta cambios de perfil. Es el valor por defecto. |
| `--profile fast` | Añade solo opciones cualificadas que cambian bits. Las filas aún no cualificadas siguen apagadas (ver `--allow-pending`). |
| `--profile quality` | Un pack de mayor precisión donde existe (Nemotron EXL3). Si no, igual que `default`. |

Existen perfiles con nombre para Qwen, el par DeepSeek, Gemma4/Nemotron y `strix-glm5-rank`.
MiniMax no tiene perfiles de producto con nombre: `--profile NAME:PROMPT.u32:STEPS` ejecuta una sesión con tiempos.
GLM rank acepta `--profile default|fast|quality` (también `--profile=NAME`); `--profile` sin valor sigue activando
los tiempos de kernels y se puede combinar con un perfil con nombre. En GLM, `fast` solo activa `STRIX_GLM_PF_WMMA=1`;
un valor explícito del entorno prevalece y `--baseline` / `STRIX_GLM_BASELINE=1` veta la adición del perfil.
`quality` usa los ajustes por defecto y lo avisa. Ambos ranks deben acordar el valor efectivo de WMMA;
siguen vigentes las guardas de arranque existentes.

Véanse los [perfiles Qwen](optimizations.md#qwen-product-profiles),
[perfiles del par DeepSeek](optimizations.md#product-profiles-pair) y la [fila FAST de GLM](optimizations.md#glm-pf-wmma-x).
FAST cambia la numérica. Úsalo cuando el rendimiento importe más que una salida idéntica bit a bit.

| Interruptor | Motor | Por defecto | Qué hace |
|---|---|---|---|
| `STRIX_QPF2K_MOE_ROUTE_BF16=1`, `--prefill-moe-route-bf16` | qwen | off (on en `fast`) | Enrutado MoE en BF16 durante el prefill. |
| `STRIX_QPF2K_GDN_FP16_CHUNK=1`, `--gdn-fp16-chunk` | qwen | off (on en `fast`) | Chunk GDN en FP16 durante el prefill. |
| `--prefill-exact-fp16` | qwen | off | Fuerza la ruta de prefill FP16 exacta. |
| `STRIX_DS41_FAST_MOE=1` | deepseek | off (`fast`) | Epílogo MoE relajado. |
| `STRIX_GLM_PF_WMMA=1` | glm | off (on con `--profile fast` en rank) | Prefill WMMA. Identidad FAST opcional; `=0` la desactiva incluso en `fast`. |

## La regla de régimen

El motor elige una familia de kernels por el número de filas en vuelo, no por modelo.

- **Decode a una fila** está limitado por el ancho de banda de memoria. Cada peso se lee una vez por token.
  El motor usa los pesos más pequeños que puede leer: packs trellis/EXL3 (`STRIX_DS41_TRELLIS`, los packs EXL3 de MiniMax, GLM y Nemotron `quality`).
- **Verify ancho y decode por lotes** (MTP, DSpark, varios slots) multiplican las filas por lectura de peso.
  Pasado un umbral de filas ganan las rutas de matrix cores y enteras. Los umbrales son toggles:
  `STRIX_INTPREFILL_DENSE_ROWS` (qwen, 512), `STRIX_QPF2K_DENSE_MIN_ROWS` (qwen, 1024 o 512 según el sitio),
  `STRIX_GLM_PF_WMMA_ROWS` (glm, 64), `STRIX_MM_PF_ASYNC_MIN_ROWS` (minimax, 2048),
  `STRIX_GLM_PF_EXACT_X_MIN_ROWS` (glm, 64; 32 with FAST WMMA).
- **Prefill** está limitado por cómputo. Usa las rutas enteras y de matrix cores, y el perfil FAST añade las que cambian bits.

El punto de cruce depende de la relación entre cómputo y ancho de banda. En gfx1151 esa relación es fija, así que los umbrales son fijos.
En otro hardware, muévelos:

| El hardware cambia en | Efecto | Qué probar |
|---|---|---|
| Más ancho de banda (GPU discreta, HBM) | El decode a una fila es más barato; los packs pequeños importan menos | Packs de más precisión; desactivar trellis con `=0` y comparar |
| Más rendimiento WMMA / INT4 | Las rutas enteras ganan con menos filas | Bajar los umbrales `*_MIN_ROWS` / `*_DENSE_ROWS` |
| Menos rendimiento WMMA / INT4 | Las rutas enteras solo ganan con muchas filas | Subir los umbrales, o `=0` para mantener la ruta exacta |
| Menos memoria | KV, caché de prefijos y modelo draft compiten con los pesos | Presupuestos de caché de prefijos más pequeños, `--slots`, `--prefill-rows N` |
| Interconexión más rápida o más lenta | Cambia cuándo compensan el solapamiento y el reparto | Los interruptores de transporte del par, abajo |

## Flags de CLI por área

Las tablas listan los flags seleccionados. La lista completa de variables de entorno sigue en el
[índice de entorno](#índice-de-entorno).

### Memoria y KV

| Flag | Motor | Por defecto | Qué hace | Cuándo lo quiere otro hardware |
|---|---|---|---|---|
| `--residency register\|copy` | qwen | `register` | Registra los pesos en sitio en vez de copiarlos. Necesita un `RLIMIT_MEMLOCK` alto. | `copy` cuando la memoria fijada es limitada o en una GPU discreta con VRAM propia. |
| `--serve-memory-budget-bytes` | qwen | auto | Limita la memoria del servidor. | Más bajo en máquinas con menos de 128 GB. |
| `--known-prefix-*` | qwen | auto | Presupuestos de la caché de prefijos conocidos. | Más bajo con menos memoria. |
| `--prefix-cache on\|off` | qwen-rtx | on | Caché de prefijos. | Off cuando la VRAM va justa. |
| `--no-prefix-cache`, `--no-prefix-disk`, `--prefix-cache-mb`, `--prefix-cache-dir` | deepseek serve | on | Caché de prefijos en RAM y en disco. | `--prefix-cache-mb` menor con menos memoria; `--prefix-cache-dir` en almacenamiento rápido. |
| `--capacity`, `--slots` | minimax / deepseek serve | según modelo | Contexto por slot y número de slots. | Menos slots o menos capacidad con menos memoria. |
| `--prefill-rows auto\|N` | deepseek serve | `auto` | Tamaño de pieza de prefill que cabe en los dos ranks tras pesos y drafter. | Fija `N` cuando auto se niega, o para cambiar memoria por velocidad de prefill. |

### Prefill

| Flag | Motor | Por defecto | Qué hace | Cuándo lo quiere otro hardware |
|---|---|---|---|---|
| `--block-schedule default\|tail-large` | qwen-rtx | `tail-large` | Orden de bloques en los kernels grandes. | `default` en GPUs con otro número de CUs. |
| `--prefill auto` | glm | `auto` | Elige la ruta de prefill. | - |
| `--prefill-chunk N` | glm | 512 | Tamaño de chunk de prefill en filas. | Mayor con más cómputo; menor con menos memoria. |
| `--prefill-path device` | deepseek serve | `device` | Prefill en la GPU. | - |

### Decode, MTP y DSpark

| Flag | Motor | Por defecto | Qué hace | Cuándo lo quiere otro hardware |
|---|---|---|---|---|
| `--mtp-bundle`, `--rope-yarn 2\|4 --context C` | qwen | - | Bundle de draft MTP; YaRN para contexto largo. | - |
| `--mtp 0\|1` | glm | resuelto (ver `--print-effective-config`) | Apaga o enciende el draft MTP. | `0` cuando falta memoria. |
| `--identity r1b\|r1` | glm | `r1b` | Identidad de decode. | `r1` para comparar. |
| `--dspark`, `--no-dspark`, `--dspark-vocab auto\|full` | deepseek | on si se encuentra el draft | Draft especulativo DSpark. | `--no-dspark` cuando falta memoria. |
| `--dspark-ranks` | deepseek | 2 | Ranks que ejecutan el draft. | - |
| `--dspark-max-inflight` | deepseek | 4 | Peticiones en vuelo antes de apagar el draft. `0` = siempre draft. | Más alto con más cómputo por fila. |
| `--dspark-mode off\|auto\|on`, `--dspark-prior` | minimax | `auto` (`STRIX_MM_SERVE_DSPARK`) | Draft DSpark. | `off` cuando el draft no cabe. |

### Transporte del par

| Flag | Motor | Por defecto | Qué hace | Cuándo lo quiere otro hardware |
|---|---|---|---|---|
| `--transport usb4auto\|usb4gpu` (prefill: `usb4-auto`\|`usb4-gpu`) | deepseek pair | `usb4auto` | Transporte USB4. `usb4gpu` usa streams GPU-direct cuando el driver los ofrece; si no, la ruta por host. | Otras interconexiones necesitan su propio transporte. |
| `--peer-timeout` | deepseek serve | 120 | Segundos de espera al peer. | Más alto en enlaces lentos. |
| `--transport usb4gpu\|usb4auto\|usb4-auto\|tcp`, `STRIX_QWEN_PF_PIPE_TRANSPORT` | qwen pair prefill | `usb4gpu` | Los payloads grandes del seam usan el router del communicator; control/CRC/ACK siguen por host. `usb4auto` o `usb4-auto` fuerza host. Ambos ranks deben coincidir. Gates nativos bit a bit/TTFT pendientes. | [Evidencia de rutas](coord/transport-routing.md). |
| `--transport usb4gpu\|usb4-auto` | minimax serve | `usb4gpu` | GPU-direct con router para mensajes grandes; host para ≤64 KiB y fallback sin streams directos. Mantiene XFER de decode; PFX solo en rank. `usb4-auto` fuerza host y conserva la ruta cualificada de caché de sesiones. | [Regímenes y límites](coord/transport-routing.md). |
| `--tp-attention`, `--no-tp-attention` | deepseek | on | Reparte la atención entre los dos ranks. | Off cuando el enlace es lento frente al cómputo. |
| `--engram-direct`, `--engram-mapped` | deepseek | `direct` | Acceso a la tabla engram. | `mapped` cuando falta memoria. |
| `--decode-xch xfer\|comm`, `STRIX_GLM_DECODE_XCH` | glm | `xfer` (`comm` en baseline) | Modo de intercambio en decode. | Compara ambos en otra interconexión. |
| `--wb-defer`, `--wb-stream`, `--ce-pacing` (`--no-*`) | deepseek-rtx | on | Write-back y ritmo del copy engine. También `STRIX_DS41_WB_STREAM`, `STRIX_DS41_CE_PACING`. | Off en enlaces donde el copy engine no es el cuello de botella. |
| `STRIX_DS41_OB_SPLIT=K` | deepseek | off | Parte el intercambio o_b en K trozos que se solapan con el cómputo. Exacto. Más lento en la ruta por host. | Prueba en una interconexión más rápida o con GPU-direct. |

Montaje del par: [pair-setup.es.md](pair-setup.es.md).

### Exacto frente a FAST

| Flag | Motor | Por defecto | Qué hace |
|---|---|---|---|
| `--profile default\|fast\|quality` | qwen, par deepseek, gemma4/nemotron, glm rank | `default` | Ver [Perfiles](#perfiles-exacto-frente-a-fast). |
| `--profile` (sin valor) | glm rank | off | Tiempos de kernels; se conserva junto a los perfiles con nombre. |
| `--profile NAME:PROMPT.u32:STEPS` | minimax rank | off | Sesión con tiempos, no un perfil de producto. |
| `--allow-pending` | apps de par | off | Deja que `fast` aplique filas aún no cualificadas. |
| `--print-effective-config` | todos | off | Imprime la configuración resuelta y sale antes de tocar el dispositivo. |
| `STRIX_DS41_BASELINE=1` | deepseek | off | Apaga todos los defaults de producción. Ejecuta el comando simple como antes. |

### Núcleo compartido e INT4 (opt-in)

Todos están apagados por defecto. Apagados, cada motor ejecuta los mismos kernels y lanzamientos que antes de
llegar el núcleo en mosaico compartido (`src/core`), con los mismos bits.

| Interruptor | Motor | Por defecto | Qué hace |
|---|---|---|---|
| `STRIX_DS41_PF_CORE=0\|moe\|dense\|1` | deepseek | por forma | Qué productos del prefill corren en el núcleo compartido en lugar de los kernels de DeepSeek: `moe` = gate/up y down de expertos, `dense` = los productos densos Q8_0 x A8, `1` = ambos, `0` = ninguno. Mismos bits en todos los casos. Sin definir: el núcleo compartido donde el fixture de rango lo mide más rápido, el gate/up (-0,6..-2,2 %) y el down (-6,0..-7,2 %) Q8_0 del experto compartido (-2,8..-3,9 % en conjunto), y solo en lotes de 16 filas con `STRIX_DS41_FAST_MOE` desactivado y `STRIX_DS41_PREFILL_STAGE` distinto de `1`, y el gate/up IQ2_XXS (-7,1..-8,2 % con 2048 / 4096 / 8192 filas) y el down Q2_K (-2,4..-3,4 %) enrutados, que corren ahí sin el tile de pesos en LDS para que dos workgroups compartan un WGP; los productos densos siguen en los kernels de DeepSeek. Se resuelve una vez por ejecución; cualquier otro valor, incluido el vacío, se rechaza. |
| `STRIX_DS41_FAST_I4=1` | deepseek | off | Prefill borrador con pérdida: los pesos GGUF de los expertos se decodifican a INT4 en el kernel y la entrada A8 se recuantiza a A4. Corre en el núcleo compartido. `STRIX_DS41_FAST_I4_SCOPE=all\|gu` elige gate/up/down (por defecto) o solo gate/up. Sin cualificar en calidad. |
| `STRIX_GLM_MODEL_FORMAT=fast_i4` | glm | `exl3` | Formato con pérdida: los expertos enrutados gate/up de las capas MoE tardías se guardan en INT4 (prefill W4A4, decode W4A16) y se libera su copia EXL3. Rechaza las identidades exactas. Frente a FAST en el par: KL de los logits de prefill 0,0145, top-1 95,4 %, NLL -0,002 (umbral 0,02 / 93 % / +0,01); tokens de decode con teacher forcing KL 0,0098, top-1 94,8 %; prefill 8K -2,0 %, 32K -3,2 %; velocidad de decode normal y MTP sin cambios. |
| `STRIX_GLM_FAST_I4_EXL3_LAYERS=<lista>` | glm | `0-14` | Con `fast_i4`: capas MoE (ordinales, rangos `a-b` o `none`) que conservan los expertos enrutados EXL3. Una capa INT4 temprana cuesta un escalón de calidad que satura, las tardías son baratas; INT4 en todas las capas no pasa el umbral de calidad. |
| `STRIX_GLM_FAST_I4_SCOPE=gate_up\|all` | glm | `gate_up` | Con `fast_i4`: qué matrices enrutadas van en INT4. `all` añade la proyección down (prefill 8K en caliente -3,6 % frente a FAST; menos margen de calidad). |
| `STRIX_GLM_FAST_I4_SCALE=fast_full\|legacy` | glm | `fast_full` | Con `fast_i4`: regla de escala de los pesos en la conversión de carga. `fast_full` usa todo el rango del nibble [-8, 7] a 0,9 de la escala sin recorte (error de pesos 0,0095 frente a 0,0115 con `legacy`, mismo tiempo de conversión). |
| `STRIX_GLM_FAST_I4_PAIRED=0` | glm | activado | Con `fast_i4` y ámbito `gate_up`: desactiva el kernel emparejado gate/up (un kernel lee los pesos INT4 de gate y up y su epílogo SwiGLU escribe la entrada de down) y ejecuta las llamadas separadas de gate, up y activación. Mismos bits de salida en ambos casos. |
| `STRIX_GLM_I4_CHAIN_GSAVE=1` | glm | desactivado | Con `fast_i4`, ámbito `all` y `STRIX_MOE_I4_CHAIN=1`: el GEMM emparejado gate/up guarda los valores de gate de la primera pasada en las filas gate/up de la llamada en lugar de en registros, de modo que gate/up y down comparten un plan de teselas y una lista de teselas. Mismos bits de salida. Desactivado por defecto: con las formas de este modelo, la variante con registros (gate/up en teselas de 64 filas, down en teselas de 128) midió un 3-7 % menos de tiempo por llamada. `STRIX_MOE_I4_CHAIN_GSAVE=0` fuerza la variante con registros en todos los motores. |
| `STRIX_GLM_PF_SCORE_SEQ=1`, `STRIX_GLM_PF_SCORE_BLOCK=<1..8>` | glm | off, `1` | Arnés de calidad (`STRIX_GLM_PF_SCORE_LAST`): un fichero de puntuación por llamada de prefill y filas por llamada de la cabeza. Solo medición. |
| `STRIX_GLM_PF_STREAMS=1` | glm | off | Un solo stream de cómputo para el prefill seqpar TP2. Mismos bits; no fue más rápido en las primeras muestras. |
| `STRIX_MMX_MODEL_FORMAT=fast_i4` | minimax | `exl3` | Prefill borrador con pérdida: cada capa MoE se convierte a INT4 durante el prefill en dos buffers reutilizables (+3,6 GiB); el decode sigue en EXL3. Ver [prefill agrupado de MiniMax](minimax-grouped-prefill.md#draft-fast_i4). |
| `STRIX_DS41_PREFILL_LEASE=1`, `STRIX_DS41_ENGRAM_DEVICE_READY=1`, `STRIX_DS41_SEED_EARLY=1` | deepseek | off | Opciones de estructura del prefill (intercambio con lease, Engram listo por capa, semilla temprana del draft). Mismos bits; sin ganancia medida todavía. |

### Solo depuración

Estos flags no cambian la salida. Añaden comprobaciones, trazas o tiempos. No los uses para benchmarks.

| Flag | Motor | Qué hace |
|---|---|---|
| `--timing` | varios | Tiempos por etapa. |
| `--verify-trace` | apps de decode | Traza los pasos de verify. |
| `--xb-check`, `--decode-check` | varios | Compara con una ruta de referencia. |
| `--resume-check*` | serve | Comprueba el estado tras reanudar. |
| `--selftest`, `--idcheck` | varios | Autotests y comprobaciones de identidad. |

Las variables de entorno cuyo nombre contiene `TRACE`, `DUMP`, `DEBUG`, `LOG`, `PROBE`, `TIMING`, `CHECK` o `STATS` son solo de depuración.

## Opciones de CMake

| Opción | Por defecto | Qué hace | Cuándo lo quiere otro hardware |
|---|---|---|---|
| `STRIX_HIP_ARCHS` | `gfx1151` | Arquitecturas GPU a compilar, separadas por `;`. | Pon tu arquitectura. Los kernels están ajustados para gfx1151. |
| `STRIX_FAMILY_DEEPSEEK41` | ON | Compila la familia DeepSeek V4.1. | - |
| `STRIX_FAMILY_GLM5` | OFF | Compila el motor GLM-5.3 Flash. | - |
| `STRIX_FAMILY_MINIMAX` | OFF | Compila los módulos MiniMax. | - |
| `STRIX_FAMILY_GEMMA4` | OFF | Compila la familia Nemotron 3.5 Bolt. | - |
| `STRIX_FAMILY_H3` | OFF | Compila la familia H3. | - |
| `STRIX_DS41_TRELLIS`, `STRIX_DS41_TRELLIS_LIBRARY`, `STRIX_DS41_TRELLIS_INCLUDE_DIR` | OFF | Enlaza la biblioteca trellis para los packs densos de DeepSeek. Sin ella los packs se omiten. | Más útil cuando el ancho de banda limita el decode. |
| `STRIX_DS41_PREFILL_FP16_LT` | OFF | Backend de prefill FP16 denso (con `STRIX_DS41_PREFILL_FP16_DENSE=1`). | GPUs con matrix cores FP16 rápidos. |
| `STRIX_DPP_REDUCE` | OFF | Intercambios DPP exactos en las reducciones protegidas de Qwen. | Prueba ON en GPUs RDNA con DPP; déjalo OFF en otras. |
| `STRIX_MINIMAX_GU4_SCALAR` | ON | Direccionamiento exacto de expertos MoE gu4 de MiniMax. OFF compila el lector anterior. | - |

## Índice de entorno

Generado desde [optimizations.json](optimizations.json). El área se deduce del nombre.
Las "rutas de kernel" son elecciones por kernel: cada una tiene un default exacto y un interruptor `=0` para apagarla.
Sigue el enlace para ver la ganancia, las precondiciones y el interruptor de apagado. Los defaults van abreviados; el enlace tiene el texto completo.

### Memoria y KV (12)

| Toggle | Motor | Por defecto | Detalles |
|---|---|---|---|
| `STRIX_DS41_ENGRAM_PAYLOAD_CACHE` | deepseek | auto | [ds-prefill](optimizations.md#ds-prefill) |
| `STRIX_DS41_PREFIX_DIAGNOSTICS` | deepseek | ON: 512 MiB/rank, block 256, reserve ... | [deepseek-pair-agent-prefix-cache](engine/serving/prefix-state-cache.md#deepseek-pair-agent-prefix-cache) |
| `STRIX_PREFIX_CACHE` | qwen | ON on the qualified single tiled-MTP ... | [certified-mtp-subset](engine/serving/prefix-state-cache.md#certified-mtp-subset) |
| `STRIX_PREFIX_CACHE_DISK` | qwen | ON: joining wherever the prefix cache... | [certified-restart-and-in-flight-joining](engine/serving/prefix-state-cache.md#certified-restart-and-in-flight-joining) |
| `STRIX_PREFIX_CACHE_DURABLE` | qwen | ON: joining wherever the prefix cache... | [certified-restart-and-in-flight-joining](engine/serving/prefix-state-cache.md#certified-restart-and-in-flight-joining) |
| `STRIX_PREFIX_CACHE_INFLIGHT` | qwen | ON: joining wherever the prefix cache... | [certified-restart-and-in-flight-joining](engine/serving/prefix-state-cache.md#certified-restart-and-in-flight-joining) |
| `STRIX_PREFIX_CACHE_RETENTION` | qwen | ON on the qualified single tiled-MTP ... | [prefix-retention-policy](engine/serving/prefix-state-cache.md#prefix-retention-policy) |
| `STRIX_PREFIX_CACHE_STAGING_BYTES` | qwen | ON on the qualified single tiled-MTP ... | [certified-mtp-subset](engine/serving/prefix-state-cache.md#certified-mtp-subset) |
| `STRIX_PREFIX_FAIR_RECLAIM_IDLE` | qwen | ON on the qualified single tiled-MTP ... | [prefix-retention-policy](engine/serving/prefix-state-cache.md#prefix-retention-policy) |
| `STRIX_QRTX_PREFIX_CACHE` | qwen-rtx | ON | [qwen-rtx-prefix-cache](optimizations.md#qwen-rtx-prefix-cache) |
| `STRIX_QSA_CACHE_ONLY` | qwen | 1 | [qwen-qsa-cache-only](optimizations.md#qwen-qsa-cache-only) |
| `STRIX_PREFILL_FINAL_CONE` | qwen | 1 | [qwen-prefill-final-cone](optimizations.md#qwen-prefill-final-cone) |
| `STRIX_HOST_WAIT` | qwen | block | [host-wait-policy](optimizations.md#host-wait-policy) |
| `STRIX_SESSION_CACHE` | minimax | on (two-node serve route) | [mm-sessions](optimizations.md#mm-sessions) |

### Prefill (97)

| Toggle | Motor | Por defecto | Detalles |
|---|---|---|---|
| `STRIX_DS41_ATTN_LIVE_CHUNKS` | deepseek | auto | [ds-attention](optimizations.md#ds-attention) |
| `STRIX_DS41_ENGRAM_F16_PF` | deepseek | on | [ds-engram-f16-pf](optimizations.md#ds-engram-f16-pf) |
| `STRIX_DS41_MOE_PREFILL` | deepseek | ON | [ds-moe7](optimizations.md#ds-moe7) |
| `STRIX_DS41_PREFILL_FINAL_KV` | deepseek | auto | [ds-prefill](optimizations.md#ds-prefill) |
| `STRIX_DS41_PREFILL_HC_COLLAPSE` | deepseek | on (with STRIX_DS41_PREFILL_HC_FUSE) | [ds-prefill-hc-collapse](optimizations.md#ds-prefill-hc-collapse) |
| `STRIX_DS41_PREFILL_HC_FUSE` | deepseek | on | [ds-prefill-hc-fuse](optimizations.md#ds-prefill-hc-fuse) |
| `STRIX_DS41_PREFILL_MOE_BAND` | deepseek | on | [ds-moe-v2](optimizations.md#ds-moe-v2) |
| `STRIX_DS41_PREFILL_MOE_V2` | deepseek | on | [ds-moe-v2](optimizations.md#ds-moe-v2) |
| `STRIX_DS41_PREFILL_MOE_V2_Q8` | deepseek | on | [ds-moe-v2](optimizations.md#ds-moe-v2) |
| `STRIX_DS41_PREFILL_O_A8_FUSED` | deepseek | on | [ds-o-a8-fused](optimizations.md#ds-o-a8-fused) |
| `STRIX_DS41_PREFILL_O_B_ONE` | deepseek | on | [ds-prefill-o-b-one](optimizations.md#ds-prefill-o-b-one) |
| `STRIX_DS41_PREFILL_O_LOW_A8_FUSED` | deepseek | on | [ds-prefill-o-low-a8](optimizations.md#ds-prefill-o-low-a8) |
| `STRIX_DS41_PREFILL_ROUTER_V2` | deepseek | on | [ds-router-v2](optimizations.md#ds-router-v2) |
| `STRIX_GLM_PF_ATTN_QWIN` | glm | auto | [glm-pf-attn-qwin](optimizations.md#glm-pf-attn-qwin) |
| `STRIX_GLM_PF_AUX_COMBINE` | glm | 1 | [glm-pf-aux-combine](optimizations.md#glm-pf-aux-combine) |
| `STRIX_GLM_PF_DENSE_FORM` | glm | automático (0) | [glm-pf-dense-rows-v4](optimizations.md#glm-pf-dense-rows-v4) |
| `STRIX_GLM_PF_DENSE_ROWS` | glm | 4 | [glm-pf-dense-rows-v4](optimizations.md#glm-pf-dense-rows-v4) |
| `STRIX_GLM_PF_HC_EXPAND_ROWS` | glm | 2 | [glm-pf-hc-expand-rows](optimizations.md#glm-pf-hc-expand-rows) |
| `STRIX_GLM_PF_HC_FUSE` | glm | 1 | [glm-pf-hc-fuse](optimizations.md#glm-pf-hc-fuse) |
| `STRIX_GLM_PF_HC_HOIST` | glm | 1, 1 | [glm-pf-sumexp](optimizations.md#glm-pf-sumexp) |
| `STRIX_GLM_PF_IDX` | glm | 3 / 2 | [glm-pf-indexer-v2](optimizations.md#glm-pf-indexer-v2) |
| `STRIX_GLM_PF_IN_TOKEN` | glm | 1 | [glm-pf-in-token](optimizations.md#glm-pf-in-token) |
| `STRIX_GLM_PF_KDA_PACK` | glm | 0 | [glm-pf-kda-pack](optimizations.md#glm-pf-kda-pack) |
| `STRIX_GLM_PF_KDA_WMMA` | glm | on in FAST prefill (STRIX_GLM_PF_WMMA=1); off in exact mode | [glm-pf-kda-wmma](optimizations.md#glm-pf-kda-wmma) |
| `STRIX_GLM_PF_KDA_SCAN` | glm | 5 | [glm-pf-kda-scan-v5](optimizations.md#glm-pf-kda-scan-v5) |
| `STRIX_GLM_PF_MIXNORM` | glm | 1 | [glm-pf-mixnorm](optimizations.md#glm-pf-mixnorm) |
| `STRIX_GLM_PF_MLA_GEMM` | glm | 3 (v3); 2 selects v2 | [glm-pf-mla-gemm-v2](optimizations.md#glm-pf-mla-gemm-v2) |
| `STRIX_GLM_PF_OBAR` | glm | 10 | [glm-pf-obar-v5](optimizations.md#glm-pf-obar-v5) |
| `STRIX_GLM_PF_PIPELINE` | glm | 2 / 1 | [prefill-pipeline-and-async-exchange-start](optimizations.md#prefill-pipeline-and-async-exchange-start) |
| `STRIX_GLM_PF_SCORES` | glm | 7 | [glm-pf-scores-v3](optimizations.md#glm-pf-scores-v3) |
| `STRIX_GLM_PF_SCRATCH` | glm | auto | [glm-pf-scratch](optimizations.md#glm-pf-scratch) |
| `STRIX_GLM_PF_SELECT` | glm | 3 / 2 | [glm-pf-indexer-v2](optimizations.md#glm-pf-indexer-v2) |
| `STRIX_GLM_PF_SEQPAR` | glm | auto | [glm-pf-seqpar](optimizations.md#glm-pf-seqpar) |
| `STRIX_GLM_PF_STAGED` | glm | 1 | [glm-pf-staged](optimizations.md#glm-pf-staged) |
| `STRIX_GLM_PF_SUMEXP` | glm | 1, 1 | [glm-pf-sumexp](optimizations.md#glm-pf-sumexp) |
| `STRIX_GLM_PF_WMMA` | glm | off (opt-in FAST identity) | [glm-pf-wmma-x](optimizations.md#glm-pf-wmma-x) |
| `STRIX_GLM_PF_WMMA_LIVE_FRAG` | glm | 1 | [glm-pf-wmma-live-fragments](optimizations.md#glm-pf-wmma-live-fragments) |
| `STRIX_INTPREFILL_DENSE_ROWS` | qwen | 512 | [qwen-intprefill-dense-rows](optimizations.md#qwen-intprefill-dense-rows) |
| `STRIX_MMX_PREFILL_GATHER_TOK` | minimax | 1 | [mm-pf-gather-tok](optimizations.md#mm-pf-gather-tok) |
| `STRIX_MMX_PREFILL_MTFIT` | minimax | 1 | [mm-pf-mtfit](optimizations.md#mm-pf-mtfit) |
| `STRIX_MMX_PREFILL_PIPE` | minimax | 1 | [mm-pf-moe-pipe](optimizations.md#mm-pf-moe-pipe) |
| `STRIX_MMX_PREFILL_TAIL_STREAM` | minimax | 1 | [mm-pf-tail-stream](optimizations.md#mm-pf-tail-stream) |
| `STRIX_MM_PFX` | minimax | on | [mm-rank-pfx](optimizations.md#mm-rank-pfx) |
| `STRIX_MM_PFX_CHUNK` | minimax | 4194304 | [mm-pfx-chunk](optimizations.md#mm-pfx-chunk) |
| `STRIX_MM_PFX_JOINED` | minimax | 1 | [mm-pfx-joined](optimizations.md#mm-pfx-joined) |
| `STRIX_MM_PF_ASYNC_START` | minimax | ON | [minimax-grouped-prefill.md](minimax-grouped-prefill.md) |
| `STRIX_MM_PF_ATTEND4` | minimax | 1 | [mm-pf-attend4](optimizations.md#mm-pf-attend4) |
| `STRIX_MM_PF_DENSE_O` | minimax | 1 | [mm-pf-dense-o](optimizations.md#mm-pf-dense-o) |
| `STRIX_MM_PF_DENSE_QKV` | minimax | 1 | [mm-pf-dense-qkv](optimizations.md#mm-pf-dense-qkv) |
| `STRIX_MM_PF_IDX_ROWS` | minimax | 64 | [mm-pf-idx-rows](optimizations.md#mm-pf-idx-rows) |
| `STRIX_MM_PF_IDX_TILE` | minimax | 1 | [mm-pf-idx-tile](optimizations.md#mm-pf-idx-tile) |
| `STRIX_MM_PF_INTERLEAVE` | minimax | XFER on, PFX loaded, interleave on | [mm-routes](optimizations.md#mm-routes) |
| `STRIX_MM_PF_KV_TILE` | minimax | auto | [mm-pf-kv-tile](optimizations.md#mm-pf-kv-tile) |
| `STRIX_MM_PF_LATE_SRC_FENCE` | minimax | ON | [minimax-grouped-prefill.md](minimax-grouped-prefill.md) |
| `STRIX_MM_PF_MSA_BATCH` | minimax | 1 | [mm-pf-msa-batch](optimizations.md#mm-pf-msa-batch) |
| `STRIX_MM_PF_MSA_REWORK` | minimax | 5 | [mm-pf-msa-rework](optimizations.md#mm-pf-msa-rework) |
| `STRIX_MM_PF_MSA_SCRATCH_MB` | minimax | 1152 | [mm-pf-msa-scratch](optimizations.md#mm-pf-msa-scratch) |
| `STRIX_MM_PF_ROUTER_BATCH` | minimax | 1 | [mm-pf-router-batch](optimizations.md#mm-pf-router-batch) |
| `STRIX_MM_PF_ROUTER_ROWS` | minimax | 32 | [mm-pf-router-v2](optimizations.md#mm-pf-router-v2) |
| `STRIX_MM_PF_TABLE_RING` | minimax | 1 in strix-minimax-rank; 0 in strix-m... | [mm-pf-table-ring](optimizations.md#mm-pf-table-ring) |
| `STRIX_MM_PF_TX_UNCACHED` | minimax | 1 | [mm-pf-tx-uncached](optimizations.md#mm-pf-tx-uncached) |
| `STRIX_PF_P1` | minimax | ON (1) | [mm-pf-moe-pipe](optimizations.md#mm-pf-moe-pipe) |
| `STRIX_PREFILL_HC_OUTPUT_WMMA` | qwen | ON | [bit-changing-prefill-options](optimizations.md#bit-changing-prefill-options) |
| `STRIX_PREFILL_INDEX_PROJ_WMMA` | qwen | OFF; opt in with STRIX_PREFILL_INDEX_... | [bit-changing-prefill-options](optimizations.md#bit-changing-prefill-options) |
| `STRIX_PREFILL_INDEX_SCORE_WMMA` | qwen | OFF; opt in with STRIX_PREFILL_INDEX_... | [engine/prefill/index-score-wmma.md](engine/prefill/index-score-wmma.md) |
| `STRIX_PREFILL_MOE_MERGE_VEC` | qwen | 1 | [qwen-prefill-moe-merge-vec](optimizations.md#qwen-prefill-moe-merge-vec) |
| `STRIX_PREFILL_PIPE` | qwen | 255 | [qwen-prefill-pipe](optimizations.md#qwen-prefill-pipe) |
| `STRIX_PREFILL_PLE_BATCH` | qwen | 1 | [qwen-ple-batch](optimizations.md#qwen-ple-batch) |
| `STRIX_PREFILL_ROUTER_SELECT` | qwen | ON | [qwen-router](optimizations.md#qwen-router) |
| `STRIX_QPF2K_COMPACT_IMAGES` | qwen | ON for eligible full-T score-ring owners | [exact-compact-qsa-attention-output](optimizations.md#exact-compact-qsa-attention-output) |
| `STRIX_QPF2K_DEFAULTS` | qwen | ON | [qwen-defaults](optimizations.md#qwen-defaults) |
| `STRIX_QPF2K_DENSE` | qwen | 1 | [qwen-dense](optimizations.md#qwen-dense) |
| `STRIX_QPF2K_EARLY_SETUP` | qwen | ON for eligible fresh plain-command f... | [early-tiled-prefill-geometry-setup](optimizations.md#early-tiled-prefill-geometry-setup) |
| `STRIX_QPF2K_GDN` | qwen | 1 | [qwen-gdn](optimizations.md#qwen-gdn) |
| `STRIX_QPF2K_GDN_CHUNK16` | qwen | auto | [qwen-gdn-chunk16](optimizations.md#qwen-gdn-chunk16) |
| `STRIX_QPF2K_GDN_COOP_LAUNCH` | qwen | 0 | [qwen-gdn](optimizations.md#qwen-gdn) |
| `STRIX_QPF2K_GDN_GATES_BATCH` | qwen | 1 | [qwen-gdn](optimizations.md#qwen-gdn) |
| `STRIX_QPF2K_GDN_PARTS` | qwen | opt-in (bit 128 off; normal mask 55) | [qwen-gdn-witness](optimizations.md#qwen-gdn-witness) |
| `STRIX_QPF2K_HC` | qwen | 1 | [qwen-hc](optimizations.md#qwen-hc) |
| `STRIX_QPF2K_HC_LOWBAND` | qwen | ON (3) through decoder qpf2k defaults | [hc-low-band-layout](optimizations.md#hc-low-band-layout) |
| `STRIX_QPF2K_HC_MASK` | qwen | 14 | [qwen-hc](optimizations.md#qwen-hc) |
| `STRIX_QPF2K_HC_OUTPUT3` | qwen | 1 | [qwen-hc](optimizations.md#qwen-hc) |
| `STRIX_QPF2K_HC_OUTPUT3_BAND` | qwen | 16 | [qwen-hc](optimizations.md#qwen-hc) |
| `STRIX_QPF2K_HC_RECOMPUTE_NORM` | qwen | ON with the WMMA HC output (auto); =1 ... | [hc-normalized-recompute](optimizations.md#hc-normalized-recompute) |
| `STRIX_QPF2K_HC_UPCOPY` | qwen | auto with HC output v3 | [qwen-hc](optimizations.md#qwen-hc) |
| `STRIX_QPF2K_INDEX_BATCH` | qwen | ON | [exact-indexer-projection-batching](optimizations.md#exact-indexer-projection-batching) |
| `STRIX_QPF2K_MOE_DA16` | qwen | 1 | [qwen-moe](optimizations.md#qwen-moe) |
| `STRIX_QPF2K_MOE_DIRECT_B` | qwen | OFF; explicit candidate only | [qwen-moe-compressed-planes](optimizations.md#qwen-moe-compressed-planes) |
| `STRIX_QPF2K_MOE_HDR` | qwen | 1 | [qwen-moe-header-reuse](optimizations.md#qwen-moe-header-reuse) |
| `STRIX_QPF2K_MOE_MC` | qwen | 1 (mode 1; mode 2 needs an explicit v... | [qwen-moe-merge-commit](optimizations.md#qwen-moe-merge-commit) |
| `STRIX_QPF2K_MOE_ROUTE` | qwen | 14 | [qwen-moe](optimizations.md#qwen-moe) |
| `STRIX_QPF2K_MOE_XOR64` | qwen | 1 | [qwen-moe-xor-rows](optimizations.md#qwen-moe-xor-rows) |
| `STRIX_QPF2K_MOE_DA16_ISSUE` | qwen | auto | [qwen-moe-da16-issue](optimizations.md#qwen-moe-da16-issue) |
| `STRIX_QPF2K_QSA` | qwen | 1 | [qwen-qsa](optimizations.md#qwen-qsa) |
| `STRIX_QPF2K_QSA_MASK` | qwen | 0x2257 | [qwen-qsa](optimizations.md#qwen-qsa) |
| `STRIX_QPF2K_SCHED` | qwen | 14 | [qwen-schedule](optimizations.md#qwen-schedule) |
| `STRIX_QPF2K_SCORE_ROWS` | qwen | auto (32 rows above 4096 blocks, 16 o... | [qwen-qsa-score-rows](optimizations.md#qwen-qsa-score-rows) |
| `STRIX_QPF2K_SHARED_PRE` | qwen | 1 | [qwen-shared-pre](optimizations.md#qwen-shared-pre) |
| `STRIX_QPF2K_T16K` | qwen | ON from 32768 prompt tokens | [automatic-t16384-prefill](optimizations.md#automatic-t16384-prefill) |
| `STRIX_QPF2K_WMMA_PIN` | qwen | 1 | [qwen-dense](optimizations.md#qwen-dense) |
| `STRIX_SERVE_PREFILL` | qwen | ON: admitted T8192; configured prefix... | [default-prompt-prefill](engine/serving/serving.md#default-prompt-prefill) |
| `STRIX_SERVE_ROUTE` | qwen | ON: server tile and prefill arithmetic from the single-request route rule; `0` restores T8192 with the resolved arithmetic | [default-prompt-prefill](engine/serving/serving.md#default-prompt-prefill) |
| `STRIX_SERVE_PREFILL_ORG` | qwen | ON: an admitted server owner takes the prefill organization of the single-request route (row-loop, attention stage, table and expert header/merge batching); `0` keeps the startup items only | [default-prompt-prefill](engine/serving/serving.md#default-prompt-prefill) |
| `STRIX_SERVE_T16K` | qwen | ON: T16384 owner when the context admits a 32,768-token request and every resident slot fits; `0` keeps T8192 | [default-prompt-prefill](engine/serving/serving.md#default-prompt-prefill) |
| `STRIX_SERVE_FP16_MTP` | qwen | ON: explicit-sidecar MTP servers keep the automatic FP16 prefill arithmetic with the unqualified fast options off; `0` declines it | [default-prompt-prefill](engine/serving/serving.md#default-prompt-prefill) |

### Decode, MTP y DSpark (22)

| Toggle | Motor | Por defecto | Detalles |
|---|---|---|---|
| `STRIX_BATCH_TOOL_DRAFT` | qwen | ON (batched decoding) | [qwen-tool-draft](optimizations.md#qwen-tool-draft) |
| `STRIX_DS41_DECODE_PREFETCH` | deepseek | auto | [ds-prefetch](optimizations.md#ds-prefetch) |
| `STRIX_DS41_DEFER_DECODER` | deepseek | on | [ds-defer-decoder](optimizations.md#ds-defer-decoder) |
| `STRIX_DS41_DSPARK_CALIB` | deepseek | auto | [ds-warm](optimizations.md#ds-warm) |
| `STRIX_DS41_R1FUSE_ROUTES` | deepseek | on | [ds-exact-decode-launches](optimizations.md#ds-exact-decode-launches) |
| `STRIX_GLM_R1B_DENSE` | glm | 7 | [glm-r1b-dense-auto](optimizations.md#glm-r1b-dense-auto) |
| `STRIX_GLM_R1B_HC` | glm | 4 | [glm-r1b-hc-v4](optimizations.md#glm-r1b-hc-v4) |
| `STRIX_GLM_SPEC_KDA_TREE` | glm | on (Xfer verify exchange, KDA tree 2,... | [glm-mtp-spec](optimizations.md#glm-mtp-spec) |
| `STRIX_MM_ATTEND4_DECODE` | minimax | 1 | [mm-attend4-decode](optimizations.md#mm-attend4-decode) |
| `STRIX_MM_DRAFT_LIN` | minimax | opt | [mm-draft-lin](optimizations.md#mm-draft-lin) |
| `STRIX_MM_DSPARK_ATTN` | minimax | 2 | [mm-dspark-attn](optimizations.md#mm-dspark-attn) |
| `STRIX_MM_SERVE_DSPARK` | minimax | auto | [mm-serve-dspark](optimizations.md#mm-serve-dspark) |
| `STRIX_MM_SERVE_DSPARK_STAGE` | minimax | 1 | [mm-serve-dspark-stage](optimizations.md#mm-serve-dspark-stage) |
| `STRIX_MOE_R1_GLUE_FUSE` | qwen | 1 (eager plain one-row decode) | [qwen-plain-decode-step](optimizations.md#qwen-plain-decode-step) |
| `STRIX_QSA_DECODE_ATTN_V2` | qwen | 1 (eager plain one-row decode) | [qwen-plain-decode-step](optimizations.md#qwen-plain-decode-step) |
| `STRIX_QSA_DECODE_FUSED_MAXIMA` | qwen | 1 (eager plain one-row decode) | [qwen-plain-decode-step](optimizations.md#qwen-plain-decode-step) |
| `STRIX_QSA_DECODE_POOL` | qwen | 1 (plain append-only serial decode) | [qwen-qsa-decode-pool](optimizations.md#qwen-qsa-decode-pool) |
| `STRIX_QSA_DECODE_SELECT` | qwen | 1 | [qwen-qsa-decode-select](optimizations.md#qwen-qsa-decode-select) |
| `STRIX_QSA_DECODE_VALUE_V2` | qwen | 1 (eager plain one-row decode) | [qwen-plain-decode-step](optimizations.md#qwen-plain-decode-step) |
| `STRIX_QWEN_DECODE_FUSE` | qwen | OFF (opt-in: STRIX_QWEN_DECODE_FUSE=1) | [qwen-decode-fuse](optimizations.md#qwen-decode-fuse) |
| `STRIX_QWEN_YARN_MTP` | qwen | table-backed QSA MTP pool and rolling... | [qwen-yarn-mtp](optimizations.md#qwen-yarn-mtp) |
| `STRIX_TOOL_DRAFT_PREFIX_GATE` | qwen | ON (batched decoding) | [qwen-tool-draft](optimizations.md#qwen-tool-draft) |

### Transporte del par (25)

| Toggle | Motor | Por defecto | Detalles |
|---|---|---|---|
| `STRIX_DIST_ASYNC_START` | glm | 2 / 1 | [prefill-pipeline-and-async-exchange-start](optimizations.md#prefill-pipeline-and-async-exchange-start) |
| `STRIX_DIST_CRC_DEFER` | deepseek | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_DIST_CRC_PARALLEL` | deepseek, glm, minimax | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_DIST_HOST_CHUNK_BYTES` | deepseek, glm, minimax | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_DIST_HOST_CRC_OVERLAP` | deepseek, glm, minimax | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_DIST_STAGED_READY_BYTES` | deepseek | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_DIST_STAGING` | deepseek | auto | [ds-transport](optimizations.md#ds-transport) |
| `STRIX_DIST_USB4_EAGER` | deepseek | auto | [ds-transport](optimizations.md#ds-transport) |
| `STRIX_DS41_CE_PACING` | deepseek-rtx | ON | [rtx-ce-pacing](optimizations.md#rtx-ce-pacing) |
| `STRIX_DS41_ENGRAM_TP` | deepseek | on for steps of up to 8 rows | [ds-engram-tp](optimizations.md#ds-engram-tp) |
| `STRIX_DS41_EXPERT_TP` | deepseek | ON | [ds-core](optimizations.md#ds-core) |
| `STRIX_DS41_F16_PAIR` | deepseek | ON | [ds-core](optimizations.md#ds-core) |
| `STRIX_DS41_TP_DOWN` | deepseek | ON | [ds-core](optimizations.md#ds-core) |
| `STRIX_DS41_TRELLIS_QAKV_PAIR` | deepseek | auto | [ds-trellis](optimizations.md#ds-trellis) |
| `STRIX_DS41_WB_STREAM` | deepseek-rtx | ON | [rtx-wb-stream](optimizations.md#rtx-wb-stream) |
| `STRIX_GLM_MTP_SPEC` | glm | on (Xfer verify exchange, KDA tree 2,... | [glm-mtp-spec](optimizations.md#glm-mtp-spec) |
| `STRIX_GLM_SPEC_XCH` | glm | on (Xfer verify exchange, KDA tree 2,... | [glm-mtp-spec](optimizations.md#glm-mtp-spec) |
| `STRIX_HTTP_EAGER_SLOTS` | qwen | AUTO capacity; at most 4 eager slots;... | [batched-decoding](engine/serving/serving.md#batched-decoding) |
| `STRIX_MMX_LINEAR_V11` | minimax | 1 | [mm-linear-v11](optimizations.md#mm-linear-v11) |
| `STRIX_MMX_PREFILL_PIPE_PAIR` | minimax | auto | [mm-pf-pipe-pair](optimizations.md#mm-pf-pipe-pair) |
| `STRIX_MM_XFER` | minimax | XFER on, PFX loaded, interleave on | [mm-routes](optimizations.md#mm-routes) |
| `STRIX_MTP_JOINT_PLAN` | qwen | calibrated-c3 when STRIX_MTP_BATCH_JO... | [joint-plan-rule](engine/scheduling/joint-serving-windows.md#joint-plan-rule) |
| `STRIX_MTP_JOINT_RULE_FILE` | qwen | calibrated-c3 when STRIX_MTP_BATCH_JO... | [joint-plan-rule](engine/scheduling/joint-serving-windows.md#joint-plan-rule) |
| `STRIX_MTP_PRIME_ROWS` | qwen | ON (STRIX_MTP_PRIME_ROWS=0: desactivar) | [qwen-mtp-prime-rows](optimizations.md#qwen-mtp-prime-rows) |
| `STRIX_QSA_MTP_POOL` | qwen | 1 (eligible native MTP CLI, the singl... | [qwen-qsa-mtp-pool](optimizations.md#qwen-qsa-mtp-pool) |

### Exacto frente a FAST (9)

| Toggle | Motor | Por defecto | Detalles |
|---|---|---|---|
| `STRIX_DS41_COMPRESS_FAST` | deepseek | on | [ds-exact-decode-launches](optimizations.md#ds-exact-decode-launches) |
| `STRIX_DS41_FAST_MOE_DOWN_ROWS` | deepseek | 32 (with STRIX_DS41_FAST_MOE=1) | [ds-fast-moe-down-rows](optimizations.md#ds-fast-moe-down-rows) |
| `STRIX_DS41_FAST_MOE_ORDER` | deepseek | on (with STRIX_DS41_FAST_MOE=1) | [ds-fast-moe-order](optimizations.md#ds-fast-moe-order) |
| `STRIX_DS41_ROUTES_FAST` | deepseek | ON | [ds-core](optimizations.md#ds-core) |
| `STRIX_GLM_PF_EXACTMOE_V` | glm | auto | [glm-pf-exact-moe-auto](optimizations.md#glm-pf-exact-moe-auto) |
| `STRIX_GLM_PF_EXACT_X` | glm | auto | [glm-pf-exact-x](optimizations.md#glm-pf-exact-x) |
| `STRIX_QPF2K_MOE_ROUTE_BF16` | qwen | OFF | [qwen-moe-route-bf16](optimizations.md#qwen-moe-route-bf16) |
| `STRIX_QSA_DECODE_POOL_BF16` | qwen | 1 (plain append-only serial decode un... | [qwen-qsa-decode-pool](optimizations.md#qwen-qsa-decode-pool) |
| `STRIX_QWEN_ROUTER_BF16` | qwen | OFF (opt-in: STRIX_QWEN_ROUTER_BF16=1) | [qwen-exact-bf16-router-weights](optimizations.md#qwen-exact-bf16-router-weights) |

### Rutas de kernel (121)

| Toggle | Motor | Por defecto | Detalles |
|---|---|---|---|
| `STRIX_BASELINE` | qwen | fast path | [qwen-tiled](optimizations.md#qwen-tiled) |
| `STRIX_BATCH_CANCEL_POINTS` | qwen | 250 ms; 2 layers; 2048 rows; 2048 row... | [batched-decoding](engine/serving/serving.md#batched-decoding) |
| `STRIX_BATCH_NEST_ROWS` | qwen | 250 ms; 2 layers; 2048 rows; 2048 row... | [batched-decoding](engine/serving/serving.md#batched-decoding) |
| `STRIX_BATCH_SERVICE_LOOKAHEAD` | qwen | 250 ms; 2 layers; 2048 rows; 2048 row... | [batched-decoding](engine/serving/serving.md#batched-decoding) |
| `STRIX_BATCH_SERVICE_MS` | qwen | 250 ms; 2 layers; 2048 rows; 2048 row... | [batched-decoding](engine/serving/serving.md#batched-decoding) |
| `STRIX_BATCH_SHORT_FIRST_ROWS` | qwen | 250 ms; 2 layers; 2048 rows; 2048 row... | [batched-decoding](engine/serving/serving.md#batched-decoding) |
| `STRIX_DISPATCH_FUSE` | qwen | 1 | [qwen-dispatch-fuse](optimizations.md#qwen-dispatch-fuse) |
| `STRIX_DS41_ATTN_LDS_SWIZZLE` | deepseek | on | [ds-attn-swizzle](optimizations.md#ds-attn-swizzle) |
| `STRIX_DS41_ATTN_LEAN` | deepseek | 3 | [ds-attn-lean](optimizations.md#ds-attn-lean) |
| `STRIX_DS41_ATTN_SEG_HPW` | deepseek | auto | [ds-index](optimizations.md#ds-index) |
| `STRIX_DS41_ATTN_TILE_F16` | deepseek | auto | [ds-attention](optimizations.md#ds-attention) |
| `STRIX_DS41_ATTN_WAVE_CAP` | deepseek | on | [ds-attn-wave-cap](optimizations.md#ds-attn-wave-cap) |
| `STRIX_DS41_ATTN_WINDOW_FOLD` | deepseek | auto | [ds-attention](optimizations.md#ds-attention) |
| `STRIX_DS41_BASELINE` | deepseek | fast path | [ds-packs](optimizations.md#ds-packs) |
| `STRIX_DS41_BATCH_PREFETCH` | deepseek | auto | [ds-prefetch](optimizations.md#ds-prefetch) |
| `STRIX_DS41_BATCH_SEGMENTS` | deepseek | ON | [ds-core](optimizations.md#ds-core) |
| `STRIX_DS41_BATCH_SHARED` | deepseek | auto | [ds-overlap](optimizations.md#ds-overlap) |
| `STRIX_DS41_CB4` | deepseek | auto | [ds-packs](optimizations.md#ds-packs) |
| `STRIX_DS41_DENSE_SELECT` | deepseek | auto | [ds-dense](optimizations.md#ds-dense) |
| `STRIX_DS41_ENGRAM_LOOKAHEAD` | deepseek | on | [ds-engram-lookahead](optimizations.md#ds-engram-lookahead) |
| `STRIX_DS41_ENGRAM_SPLIT_READ` | deepseek | on | [ds-engram-split-read](optimizations.md#ds-engram-split-read) |
| `STRIX_DS41_EXPERT_SLOTS` | deepseek | ON | [ds-core](optimizations.md#ds-core) |
| `STRIX_DS41_IDX_CAND_FILTER` | deepseek | auto | [ds-index](optimizations.md#ds-index) |
| `STRIX_DS41_IDX_DIRECT` | deepseek | auto | [ds-index](optimizations.md#ds-index) |
| `STRIX_DS41_LATE_HC_LAYERS` | deepseek | auto | [ds-overlap](optimizations.md#ds-overlap) |
| `STRIX_DS41_MOE_READY` | deepseek | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_DS41_NORM_ILP` | deepseek | on | [ds-exact-decode-launches](optimizations.md#ds-exact-decode-launches) |
| `STRIX_DS41_OVERLAY` | deepseek | auto | [ds-packs](optimizations.md#ds-packs) |
| `STRIX_DS41_PREFETCH_ROW_MB` | deepseek | auto | [ds-prefetch](optimizations.md#ds-prefetch) |
| `STRIX_DS41_PREFETCH_SIDE` | deepseek | auto | [ds-prefetch](optimizations.md#ds-prefetch) |
| `STRIX_DS41_PREPARED_LOAD` | deepseek | ON: discover matching local pack | [ds-prepared-load](optimizations.md#ds-prepared-load) |
| `STRIX_DS41_PREPARED_ROOT` | deepseek | ON: discover matching local pack | [ds-prepared-load](optimizations.md#ds-prepared-load) |
| `STRIX_DS41_READY_LEAN` | deepseek | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_DS41_READY_READS_ONLY` | deepseek | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_DS41_RTX_BASELINE` | deepseek-rtx | fast path | [rtx-kernels](optimizations.md#rtx-kernels) |
| `STRIX_DS41_SERVE_SCHED` | deepseek | auto | [ds-scheduling](optimizations.md#ds-scheduling) |
| `STRIX_DS41_SERVE_SERVICE_MS` | deepseek | 250 ms soft target; short-first on | [deepseek-v41-pair-scheduler](engine/serving/serving.md#deepseek-v41-pair-scheduler) |
| `STRIX_DS41_SERVE_SHORT_FIRST` | deepseek | 250 ms soft target; short-first on | [deepseek-v41-pair-scheduler](engine/serving/serving.md#deepseek-v41-pair-scheduler) |
| `STRIX_DS41_SERVE_WARM` | deepseek | auto | [ds-warm](optimizations.md#ds-warm) |
| `STRIX_DS41_SHARED_DOWN_FUSED` | deepseek | auto | [ds-shared-down](optimizations.md#ds-shared-down) |
| `STRIX_DS41_SHARED_GROUPED` | deepseek | ON | [ds-shared-grouped](optimizations.md#ds-shared-grouped) |
| `STRIX_DS41_TOUCH_BATCH` | deepseek | on | [ds-exact-decode-launches](optimizations.md#ds-exact-decode-launches) |
| `STRIX_DS41_TRELLIS` | deepseek | auto | [ds-packs-trellis](optimizations.md#ds-packs-trellis) |
| `STRIX_DS41_TRELLIS_DENSE_FULL` | deepseek | auto | [ds-packs-trellis](optimizations.md#ds-packs-trellis) |
| `STRIX_DS41_TRELLIS_HEADOUT` | deepseek | auto | [ds-packs-trellis](optimizations.md#ds-packs-trellis) |
| `STRIX_DS41_TRELLIS_MIX` | deepseek | auto | [ds-packs-trellis](optimizations.md#ds-packs-trellis) |
| `STRIX_DS41_TRELLIS_OA_GROUP` | deepseek | auto | [ds-trellis](optimizations.md#ds-trellis) |
| `STRIX_DS41_TRELLIS_OB_READY` | deepseek | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_DS41_TRELLIS_ROW_VIEWS` | deepseek | on | [ds-exact-decode-launches](optimizations.md#ds-exact-decode-launches) |
| `STRIX_DS41_TRELLIS_SITES` | deepseek | auto | [ds-packs-trellis](optimizations.md#ds-packs-trellis) |
| `STRIX_EAGER_CODE_LOAD` | qwen | ON | [eager-code-object-loading](optimizations.md#eager-code-object-loading) |
| `STRIX_EMBED_BUILD_INFO` | qwen | ON | [configured-build-evidence](operations/startup-preflight.md#configured-build-evidence) |
| `STRIX_GLM_ATTN` | glm | fast | [glm-attn-fast](optimizations.md#glm-attn-fast) |
| `STRIX_GLM_BASELINE` | glm | fast path | [glm-identity](optimizations.md#glm-identity) |
| `STRIX_GLM_BOUNDARY` | glm | 1 | [glm-xfer-boundary-fused](optimizations.md#glm-xfer-boundary-fused) |
| `STRIX_GLM_HC_NORM` | glm | 1 (set by the rank app) | [glm-hc-norm-fused](optimizations.md#glm-hc-norm-fused) |
| `STRIX_GLM_HC_PRE_V` | glm | 7 | [glm-hc-pre-v7](optimizations.md#glm-hc-pre-v7) |
| `STRIX_GLM_HC_SITE` | glm | 1 (engine default) | [glm-hc-site](optimizations.md#glm-hc-site) |
| `STRIX_GLM_KDA_LANES` | glm | on (2ct at one row, cqst at 2..8 rows) | [glm-kda-tree](optimizations.md#glm-kda-tree) |
| `STRIX_GLM_KDA_TREE` | glm | on (2ct at one row, cqst at 2..8 rows) | [glm-kda-tree](optimizations.md#glm-kda-tree) |
| `STRIX_GLM_MAX_ROWS_AUTO` | glm | 8192 for two-rank decode prompt prefi... | [glm-max-rows-auto](optimizations.md#glm-max-rows-auto) |
| `STRIX_GLM_MOE` | glm | fast | [glm-moe-fast](optimizations.md#glm-moe-fast) |
| `STRIX_GLM_POST` | glm | fast | [glm-post-exchange-fast](optimizations.md#glm-post-exchange-fast) |
| `STRIX_GLM_SERVE_G2` | glm | on (with MTP) | [glm-serve-g2](optimizations.md#glm-serve-g2) |
| `STRIX_GLM_STEP_FLAG` | glm | 1 (engine default) | [glm-step-flag](optimizations.md#glm-step-flag) |
| `STRIX_GLM_TIERB` | glm | fused | [glm-attn-tierb-fused](optimizations.md#glm-attn-tierb-fused) |
| `STRIX_LM_HEAD_COHORT_LEAN` | qwen | 1 | [qwen-lm-head-cohort-lean](optimizations.md#qwen-lm-head-cohort-lean) |
| `STRIX_LM_HEAD_ROWS_LEAN` | qwen | 2 | [qwen-lm-head-rows-lean](optimizations.md#qwen-lm-head-rows-lean) |
| `STRIX_MINIMAX_GU4_SCALAR` | minimax | ON (build time) | [mm-gu4](optimizations.md#mm-gu4) |
| `STRIX_MMX_CRC` | minimax | deferred | [mm-xfer-crc-deferred](optimizations.md#mm-xfer-crc-deferred) |
| `STRIX_MMX_MOE_DOWN_SPLIT` | minimax | 1 | [mm-moe-down-split](optimizations.md#mm-moe-down-split) |
| `STRIX_MMX_MOE_GU_INTERLEAVE` | minimax | 1 | [mm-moe-gu-interleave](optimizations.md#mm-moe-gu-interleave) |
| `STRIX_MMX_MOE_PIPE` | minimax | 2 | [mm-moe-pipe](optimizations.md#mm-moe-pipe) |
| `STRIX_MMX_MOE_SINGLE` | minimax | 1 | [mm-moe-single](optimizations.md#mm-moe-single) |
| `STRIX_MMX_MOE_TIGHT_GRID` | minimax | 1 | [mm-moe-tight-grid](optimizations.md#mm-moe-tight-grid) |
| `STRIX_MMX_MOE_WIDE` | minimax | 2 | [mm-moe-wide](optimizations.md#mm-moe-wide) |
| `STRIX_MM_ATTEND_DIET` | minimax | 1 | [mm-attend-diet](optimizations.md#mm-attend-diet) |
| `STRIX_MM_BASELINE` | minimax | fast path | [mm-moe-wide](optimizations.md#mm-moe-wide) |
| `STRIX_MM_INDEX_CONC` | minimax | 1 | [mm-index-conc](optimizations.md#mm-index-conc) |
| `STRIX_MM_LOOKUP` | minimax | 1 | [mm-serve-lookup](optimizations.md#mm-serve-lookup) |
| `STRIX_MM_MARKOV` | minimax | cand, C 4096, H 8192, top-C v2 | [mm-markov-cand](optimizations.md#mm-markov-cand) |
| `STRIX_MM_MARKOV_C` | minimax | cand, C 4096, H 8192, top-C v2 | [mm-markov-cand](optimizations.md#mm-markov-cand) |
| `STRIX_MM_MARKOV_H` | minimax | cand, C 4096, H 8192, top-C v2 | [mm-markov-cand](optimizations.md#mm-markov-cand) |
| `STRIX_MM_MARKOV_TOPC` | minimax | cand, C 4096, H 8192, top-C v2 | [mm-markov-cand](optimizations.md#mm-markov-cand) |
| `STRIX_MM_SELECT_ONCE` | minimax | 1 | [mm-select-once](optimizations.md#mm-select-once) |
| `STRIX_MM_SERVE_GAP_MS` | minimax | interleave for pre-existing decoders;... | [minimax-serve-strix-minimax-serve](engine/serving/serving.md#minimax-serve-strix-minimax-serve) |
| `STRIX_MM_SERVE_NEST` | minimax | ON; GPU measured | [minimax-serve-strix-minimax-serve](engine/serving/serving.md#minimax-serve-strix-minimax-serve) |
| `STRIX_MM_SERVE_SCHED` | minimax | interleave for pre-existing decoders;... | [minimax-serve-strix-minimax-serve](engine/serving/serving.md#minimax-serve-strix-minimax-serve) |
| `STRIX_MM_SERVE_SERVICE_MS` | minimax | interleave for pre-existing decoders;... | [minimax-serve-strix-minimax-serve](engine/serving/serving.md#minimax-serve-strix-minimax-serve) |
| `STRIX_MM_SERVE_SHORT_FIRST` | minimax | interleave for pre-existing decoders;... | [minimax-serve-strix-minimax-serve](engine/serving/serving.md#minimax-serve-strix-minimax-serve) |
| `STRIX_MM_SERVE_TIMELINE` | minimax | interleave for pre-existing decoders;... | [minimax-serve-strix-minimax-serve](engine/serving/serving.md#minimax-serve-strix-minimax-serve) |
| `STRIX_MM_XB_INPLACE` | minimax | 1 | [mm-xb-inplace](optimizations.md#mm-xb-inplace) |
| `STRIX_MM_XB_POSTILP` | minimax | 2 | [mm-xb-postilp](optimizations.md#mm-xb-postilp) |
| `STRIX_MM_XB_PUBLISH` | minimax | 1 | [mm-xb-publish](optimizations.md#mm-xb-publish) |
| `STRIX_MM_XB_WAIT` | minimax | 1 | [mm-xb-wait](optimizations.md#mm-xb-wait) |
| `STRIX_MM_XB_WARM` | minimax | 8 | [mm-xb-warm](optimizations.md#mm-xb-warm) |
| `STRIX_MM_XB_WARM_O` | minimax | 8 | [mm-xb-warm-o](optimizations.md#mm-xb-warm-o) |
| `STRIX_MODEL_REGISTER_MLOCK` | qwen | ON when RLIMIT_MEMLOCK covers the reg... | [registered-page-lock](optimizations.md#registered-page-lock) |
| `STRIX_PLE_PREISSUE` | qwen | ON | [qwen-ple-preissue](optimizations.md#qwen-ple-preissue) |
| `STRIX_PLE_RECLAIMABLE` | qwen | auto | [qwen-l1](optimizations.md#qwen-l1) |
| `STRIX_PLE_REWARM` | qwen | ON (bounded) | [serving-ple-table-re-warm](optimizations.md#serving-ple-table-re-warm) |
| `STRIX_PLE_ROLLING` | qwen | table-backed QSA MTP pool and rolling... | [qwen-yarn-mtp](optimizations.md#qwen-yarn-mtp) |
| `STRIX_PLE_STAGE_SERVE` | qwen | ON | [prompt-group-ple-gather-staging](optimizations.md#prompt-group-ple-gather-staging) |
| `STRIX_PLE_WARM_PARALLEL` | qwen | ON | [qwen-ple-warm-parallel](optimizations.md#qwen-ple-warm-parallel) |
| `STRIX_PLE_WINDOW_WARM` | qwen | ON | [qwen-ple-warm](optimizations.md#qwen-ple-warm) |
| `STRIX_PREFLIGHT` | qwen | ON | [operations/startup-preflight.md](operations/startup-preflight.md) |
| `STRIX_PROMPT_BATCH_AUTO` | qwen | ON | [prompt-group-size-by-load](optimizations.md#prompt-group-size-by-load) |
| `STRIX_PROMPT_BATCH_AUTO_QUEUE` | qwen | ON | [prompt-group-size-by-load](optimizations.md#prompt-group-size-by-load) |
| `STRIX_QSA_ATTEND_BATCH` | qwen | 1 | [qwen-qsa-attend-batch](optimizations.md#qwen-qsa-attend-batch) |
| `STRIX_QSA_INDEX_ROWS` | qwen | 1 | [qwen-qsa-index-rows](optimizations.md#qwen-qsa-index-rows) |
| `STRIX_QSA_NORM_GROUP` | qwen | 1 | [qwen-qsa-norm-group](optimizations.md#qwen-qsa-norm-group) |
| `STRIX_QSA_SCORE_PARK` | qwen | 1 | [qwen-qsa-score-park](optimizations.md#qwen-qsa-score-park) |
| `STRIX_QSA_SCORE_ROWS64` | qwen | 1 (from 65,536 blocks; per-group 64-r... | [qwen-qsa-score-rows64](optimizations.md#qwen-qsa-score-rows64) |
| `STRIX_QSA_UNION_MERGE` | qwen | 1 (from 49,152 blocks; range form below) | [qwen-qsa-union-merge](optimizations.md#qwen-qsa-union-merge) |
| `STRIX_QSA_WINDOW_ATTN_V2` | qwen | 1 | [qwen-qsa-window-grouped-score](optimizations.md#qwen-qsa-window-grouped-score) |
| `STRIX_QWEN_ALLON` | qwen | OFF (opt-in: STRIX_QWEN_ALLON=1 / exa... | [qwen-all-on-composition-profile](optimizations.md#qwen-all-on-composition-profile) |
| `STRIX_QWEN_TAIL_LARGE` | qwen | tail-large | [qwen-rtx-tail-large](optimizations.md#qwen-rtx-tail-large) |
| `STRIX_RELEASE_EVENTS` | qwen | ON | [operations/startup-preflight.md](operations/startup-preflight.md) |
| `STRIX_ROUTER_STAGE_ACT` | qwen | ON | [qwen-router](optimizations.md#qwen-router) |
| `STRIX_SERVE_HEALTH_RECORD` | qwen | ON | [operations/service-health.md](operations/service-health.md) |
| `STRIX_SERVE_OBSERVE_EXPIRY_S` | deepseek | 60 seconds | [ds-observation](optimizations.md#ds-observation) |

### Solo depuración (2)

| Toggle | Motor | Por defecto | Detalles |
|---|---|---|---|
| `STRIX_DS41_OB_READY_EPILOGUE` | deepseek | auto | [ds-ready](optimizations.md#ds-ready) |
| `STRIX_SERVE_BACKEND_PROBE` | qwen | ON | [operations/service-health.md](operations/service-health.md) |
