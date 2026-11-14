# MiniMax M3.1 Flash en el par USB4

[English](README.md) · [Español](README.es.md) · [README principal](../../../README.es.md)

MiniMax M3.1 Flash, pesos trellis EXL3 a 3.06 bpw, en dos nodos Strix Halo (TP2 sobre USB4): prefill y decode
exactos, servicio DSpark con admisión automática, modo 10 de prefill MSA, servicio con plazos y anidado. Production; la
puerta de calidad de ocho textos superada. Esta página conserva todas las mediciones publicadas con su compilación y sus
condiciones y el comando de lanzamiento. Cómo se leen las tablas y qué cubre "exacto":
[metodología](../../methodology.es.md). Instrucciones de compilación (biblioteca trellis incluida):
[primeros pasos](../../getting-started/README.es.md); el par en sí: [dos nodos por USB4](../../pair.es.md).

FAST_I4 opcional en borrador (`STRIX_MMX_MODEL_FORMAT=fast_i4`): el INT4 se produce por capa durante el prefill
en dos búferes reutilizables; memoria +3.6 GiB; el contexto nunca se reduce; decode sin cambios. Los pesos EXL3
siguen residentes para cada capa de decode y fila MTP. `STRIX_MMX_FAST_I4_SCOPE=all|gate_up` selecciona INT4 para
todas las proyecciones de expertos (predeterminado) o solo gate/up. Por defecto INT4 se ejecuta en las capas MoE
38-59 y las capas 3-37 siguen en EXL3 (`STRIX_MMX_FAST_I4_EXL3_LAYERS`, `none` = todas las capas), y los prompts de
menos de 4096 filas hacen el prefill en EXL3 (`STRIX_MMX_FAST_I4_MIN_ROWS`): a 2K, KL 0.0139, coincidencia top-1
96.5 %, dNLL +0.006 frente al formato exacto en cuatro prompts de chat. La conversión en serie es la predeterminada;
`STRIX_MMX_FAST_I4_PREFETCH=1` activa la precarga ordenada por eventos, pendiente de cualificación de tiempos. Es una
identidad numérica con pérdida pendiente de las puertas nativas de tiempo y calidad; no hereda la cualificación de la
ruta exacta de producción. Con el alcance `all`, la cadena MoE INT4 fusionada se ejecuta una vez por chunk de 4096
filas por defecto (`STRIX_MOE_I4_CHAIN=0` selecciona las llamadas sin fusionar, mismos bits): prefill 8K 12.12 s
frente a 12.34 s sin fusionar (−1.7 %, ambos órdenes); 32K 55.23 s frente a 55.74 s (−0.9 %, seis pares, ambos órdenes); anclas iguales,
los ranks coinciden. [Contrato de implementación](../../minimax-grouped-prefill.md#draft-fast_i4).

<a id="minimax-pair"></a>

## Ejecución

Ejecute una vez por nodo con `R` establecido en 0 o 1 y assets EXL3 coincidentes:

```sh
./build/strix-minimax-rank MODEL_DIR R usb4-auto OUTPUT_DIR
```

El [resolutor](../../../src/minimax/engine/mm_defaults.cpp) activa el conjunto de producción de decode/prefill/intercambio,
incluido el modo de prefill MSA 10 (bucle de chunks dentro del bloque con lista de bloques y parciales en LDS, sin
parciales globales; `STRIX_MM_PF_MSA_REWORK=8` restablece el modo anterior, valores válidos de 0 a 10). Identidad FAST opcional: `STRIX_MM_PF_MSA_FLASH=1` (operandos BF16 de un término) o
`2` (dos términos) ejecuta la atención MSA del prefill de las capas 40-59 en el núcleo flash compartido de WMMA
dividido por claves (`STRIX_MM_PF_MSA_FLASH_LAYERS` elige las capas; ver
[mm-pf-msa-flash](../../optimizations.md#mm-pf-msa-flash) para la puerta de calidad por conjunto de capas). `strix-minimax-serve` usa la admisión automática estructural de DSpark con assets
locales compatibles; las peticiones elegibles sin contención pueden usar draft, mientras que las peticiones en competencia/agrupadas o no admitidas
se quedan en simple. Véase [admisión y cualificación](../../engine/serving/dspark-auto-qualification.md).
Una petición de sesión con nombre que no cabe en el contexto devuelve HTTP 400 `context_length_exceeded` (la
sesión sigue aparcada); la generación se detiene al final del contexto con el motivo de fin `length`. Véase
[caché de sesiones](../../minimax/session-cache.md).
Las sesiones de conversación están activadas por defecto en la ruta de servicio de dos nodos: una petición con id
de `session` aparca su estado en la RAM del host justo después de la respuesta y se reanuda desde él en el turno
siguiente (turno 2 de una conversación de 6,989 tokens, prompt de 6,989 tokens restaurado entero: respuesta completa
3.84–5.16 s en lugar de 15.20–16.16 s, de los que el prefill simple son 12.51–13.46 s, 519–559 tok/s; salida idéntica).
El guardado y la reanudación calculan el hash de cada sección del punto de control una sola vez, en paralelo: la
respuesta con sesión del turno 2 tarda 2.78 s en lugar de 4.66 s (mediana, −40.7 %, IC 95 % −42.0 a −39.5 %, 20
ejecuciones emparejadas, salida y bytes del punto de control idénticos). Presupuesto min(8 GiB, 25 % de la memoria libre tras cargar el modelo),
como máximo 8 sesiones aparcadas; el nivel de disco es opcional (`--session-disk on` / `STRIX_SESSION_DISK=1`).
Se desactiva con `--session-cache off` / `STRIX_SESSION_CACHE=0` o con los interruptores de baseline. Las
peticiones con nombre decodifican sin el drafter DSpark (mantiene el prefill agrupado; los ids son los mismos);
`STRIX_MM_SESSION_DRAFT=1` lo restablece.

Los pesos de proyección del indexador se cargan según la ruta. La carga conserva solo las disposiciones que usa
la ruta resuelta por `STRIX_MM_INDEX_PROJ` (`attn` predeterminada, `tiled`, `off`) y `STRIX_MM_INDEX_DECODE`
(predeterminado 1) para decode/verificación (1–8 filas) y prompt (más de 8 filas): la ruta predeterminada `attn`
carga solo la disposición empaquetada; `tiled` carga la disposición FP16 por bloques (decode, con
`STRIX_MM_INDEX_DECODE=1`) y la FP16 traspuesta; `off` carga la FP32 traspuesta. Las salidas de prefill y decode
son idénticas bit a bit a las obtenidas cargando todas las disposiciones. Con la disposición empaquetada, las filas
del prompt usan el kernel slab (activaciones como operandos escalares, la etapa de pesos FP16 ampliada una vez en
LDS): los mismos bits que el kernel tiled anterior en 0,67-0,70x de su tiempo por llamada (banco unitario, 1024-8192
filas). Las demás rutas siguen funcionando:
defina ambas variables antes de que el motor cargue. Seleccionar una ruta cuya disposición no se cargó se rechaza
con un error que pide recargar; un conjunto de disposiciones que no cabe en la memoria del dispositivo hace fallar la carga.

Las capacidades del transporte USB4 son opcionales y están todas desactivadas en MiniMax: `STRIX_MM_USB4_CAPS`
acepta una lista separada por comas de `streams`, `joint`, `ready`, `deferred-crc`, `chunks`, `crc-overlap`,
`gpu-lean` (tabla de streams, calibración conjunta, disponibilidad escalonada, CRC diferido, chunk predeterminado
del host, solapamiento de CRC predeterminado del host, cola de GPU ligera). Sin definir o vacía conserva el
emparejamiento anterior. La selección viaja en el campo de versión hello/plan del emparejamiento cubierto por CRC,
sin trama ni byte adicionales; ambos ranks deben definir la misma lista o el emparejamiento se rechaza, los nombres
desconocidos o vacíos se rechazan y la variable requiere transporte USB4. Se aplica a `strix-minimax-rank` y a la
ruta de servicio. Con `STRIX_XPORT_TIMED_WAIT=1`, las esperas del host imprimen
`xport host_wait owner=... wait=... class=... calls=... ns=...` en stderr. Sin definir o con cualquier valor
distinto de `1`, el informe de espera temporizada del host permanece desactivado; solo se activa con el valor
exacto `1`. El contador del intercambio de prefill
antes llamado `exposed_wait` ahora es `reuse_dependencies`: cuenta aristas de eventos de reutilización de búferes,
no tiempo transcurrido.

<a id="how-fast"></a>

## Resultados medidos

Fuentes: [registros de rendimiento](../../performance.md) y [pruebas actuales de optimización](../../optimizations.md#minimax).

Todas las filas: pesos trellis EXL3 a 3.06 bpw, dos nodos Strix Halo TP2 sobre USB4, referencia greedy.
El motor está en Production; los conjuntos de optimización cualificados siguientes son públicos y están activados por defecto.
Los [cambios de planificación del prefill agrupado en el host](../../minimax-grouped-prefill.md) están activados por defecto para
las llamadas agrupadas en las que cada miembro tiene al menos 2048 filas. Prompts equilibrados de 7940 tokens midieron
un tiempo de prefill un 2.3–2.6% menor con IDs de token completos idénticos; las colas pequeñas desequilibradas usan la ruta ordinaria.

| Modo | Prompt / contexto | Resultado medido | Build / exactitud |
|---|---|---|---|
| Decode greedy | prompts de benchmark de 1K / 8K, en frío | **43.30 / 45.06 ms/token** (23.10 / 22.19 tok/s); 44.57 / 46.10 ms/token en `8cc30ac0` (256 tokens) | árbol público del PR #59; los ranks coinciden, IDs de referencia iguales |
| Prefill combinado | prompt de benchmark de 8K | **14.04 s**, 583 tok/s (brazos alternos en un proceso, 13.85–14.29 s); **45–50 %** del suelo de kernel calculado de 6.3–7.0 s; 17.731 s en `e0bad3d1` (línea base 21.874 s) | #334 `c225cdaa2`; estado y filas finales de ambos ranks iguales a las referencias |
| Prefill combinado nuevo | prompt de benchmark de 32K | **63.48 s**, 516 tok/s (tres ejecuciones por brazo, en ambos órdenes); **41–46 %** del suelo de kernel calculado de 26–29 s; 91.508 s en `e0bad3d1` | #334 `c225cdaa2`; hash de resultado igual en las seis ejecuciones, los ranks coinciden |
| Prefill combinado, mismo proceso | 1K / 8K, brazos alternos | **2.49–2.52 / 13.85–14.29 s**; en `561de319`: 2.73–2.77 / 20.3–20.8 s, 0.848–0.853× / 0.783–0.797× el tiempo de la línea base | build del #334; comprobaciones de estado/fila final iguales en ambos ranks |
| Batch, proceso nuevo por brazo | N=1 / N=8, benchmark de 1K | **43.92 / 130.24 ms/step** (N8: **61.4 tok/s agregados**, calculado) | `8cc30ac0`; cada fila es igual a ejecutarse sola; la invariancia de batch se cumple |
| DSpark | JSON / Python / prosa | **59.96 / 36.93 / 27.05 tok/s**; 54.4 / 34.1 / 24.1 tok/s en `8cc30ac0` | PR #70 `fbd43423`; IDs iguales a greedy |
| Servido, C1 / C2 / C4 / C8 | 8 prompts distintos de ~190 tokens, 256 tokens | **22.0 / 32.2 / 44.5 / 55.7 tok/s** agregados (repetición 22.1 / 32.5 / 45.1 / 56.1) | main `e6ce78571`; los IDs de cada petición son iguales a los de la petición sola |

Los tipos de prompt no se especifican para los benchmarks de prefill y de batch. Las filas de prefill nuevo son ejecuciones únicas;
el resultado del primer uso a 1K fue de **3.509 s**. La ejecución combinada alcanzó después su límite de tiempo en otro brazo; solo se citan
las mediciones completadas. Los nuevos valores predeterminados de prefill en `561de319` incluyen proyecciones densas por tiles, atención/router por batch
y tiles de filas MoE de ajuste exacto. Preservan la identidad de producción.

Los valores predeterminados de límites de intercambio en `8cc30ac0` están cualificados a 1K/8K/16K (16K con repetición sintética): las repeticiones
emparejadas ahorran **4.20 / 2.82 / 3.09 ms/token**. En el A/B de batch con proceso nuevo, N1/N8 bajan de **48.22 / 134.03**
a **43.92 / 130.24 ms/step**; la ralentización anterior de N8 se debió a la limitación térmica/de potencia de la GPU bajo
carga sostenida (unos 96 °C, reloj de shaders −20 % en ~100 s), no al motor. Las filas anteriores son
cifras de arranque en frío con procesos nuevos; con servicio sostenido, el decode medido es alrededor de un 5 % y el prefill
alrededor de un 15 % más lento con esta refrigeración (29 peticiones secuenciales: 44.9 → 47.2 ms/step). Esto no justifica una afirmación general de velocidad de servicio. Los valores predeterminados selectivos posteriores de MoE/atención figuran en el [catálogo](../../optimizations.md#minimax).

**Gate de calidad: superado** en la evaluación prerregistrada de ocho textos, incluidas dos ejecuciones TP2 independientes
e idénticas byte a byte y todas las comprobaciones de prompt/régimen. La KL media respecto a la referencia exacta es **0.1449** frente a
**0.1425** de la aritmética de CPU de producción. La concordancia top-1 es **0.47 puntos porcentuales menor** que la referencia
de CPU, dentro del margen prerregistrado de 2 puntos; la precisión del siguiente token difiere en **+0.01 puntos**. Esta
cualificación está condicionada a esos ocho textos; sustituye al fallo anterior con dos prompts.

## Todos los valores predeterminados incorporados de MiniMax

- Bit 0 de `STRIX_PF_P1`: decodificación MUL1 empaquetada en los kernels MoE PAIR/PIPE del prefill, activada por defecto (`=0` vuelve a la decodificación escalar); salida idéntica bit a bit; TTFT 8K −1,3 %. Se lee una vez en el host, así que hay que reiniciar ambos rangos.

El registro de cambios de cada palanca de MiniMax activada por defecto, con su ganancia medida, su evidencia de
exactitud y su interruptor de desactivación, está en [ganancias medidas](../../measured-gains.es.md); el catálogo
completo es [docs/optimizations.md](../../optimizations.md#minimax). Notas relacionadas:
[planificación agrupada del prefill en el host](../../minimax-grouped-prefill.md), [prefill MSA](../../minimax-msa.md),
[caché de sesión](../../minimax/session-cache.md),
[admisión y cualificación de DSpark](../../engine/serving/dspark-auto-qualification.md). Servicio:
[servicio](../../serving.es.md).
