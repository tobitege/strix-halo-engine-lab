# Añadir un motor

*English: [adding-an-engine.md](adding-an-engine.md)*

Una familia de modelos se añade escribiendo lo que es específico del modelo y
adaptándolo a las capas compartidas. El servicio, las sesiones, las cachés y la
planificación no se escriben de nuevo. Esta página indica qué implementa una
familia, qué recibe y cómo se mantienen separados los backends. Hay un ejemplo
completo que compila en `examples/skeleton_engine/`.

Contexto: [engine/session-layer.es.md](engine/session-layer.es.md) (contratos),
[engine/backends.md](engine/backends.md) (frontera Backend),
[models/model-families.md](models/model-families.md) (reglas de estructura).

## El ejemplo

`examples/skeleton_engine/skeleton_engine.hpp` es una familia en un solo fichero.
Su «modelo» es un contador, de modo que el fichero contiene solo el código de
adaptación que toda familia necesita. Se compila como `strix-skeleton-serve` y lo
prueba `skeleton_engine` (`tests/skeleton_engine_tests.cpp`), que envía una petición
de completado y otra de chat por el camino de producción `strix::serve::run()` con
el backend de referencia en CPU.

```
cmake --build build --target strix-skeleton-serve skeleton_engine_tests
ctest --test-dir build -R skeleton_engine
build/strix-skeleton-serve --registry-root examples/skeleton_engine \
    --registry models/registry.toml --model skeleton --port 8080
curl -s localhost:8080/v1/completions -d '{"model":"skeleton","prompt":"hi","max_tokens":4}'
```

## Qué implementa una familia

| Pieza | Dónde | En el ejemplo |
|---|---|---|
| Descripción del modelo | `models/<modelo>.toml` y una fila en `models/registry.toml` (`name`, `family`, `description`; una fila de producción indica además su opción de CMake, etiqueta de perfil, ficheros de valores por defecto y adaptador) | `examples/skeleton_engine/models/` |
| Adaptador del modelo | Una clase derivada de `strix::session::Engine` (`include/strix/session/engine.hpp`) | `skeleton::Engine` |
| Ganchos de estado | `strix::session::Hooks` para `KvStore`: initialize, reset, release, capture, view, restore, truncate | `Engine::hooks()` |
| Tokenizador | `serve::BpeTokenizer` o cualquier `serve::ApiTokenizer` (`include/strix/serve/tokenizer.hpp`) | `ByteTokenizer` |
| Plantilla y analizador | `ApiConfig::chat_renderer`, `output_parser` opcional, `stop_tokens` (`include/strix/serve/api.hpp`); las plantillas compartidas están en `include/strix/serve/chat_template.hpp` | la lambda de `Service` |
| Servicio | Un `serve::SessionService` dueño del backend, el motor, la cola, `SessionRuntime` y `ApiService` (`include/strix/serve/run.hpp`) | `skeleton::Service` |
| Registro | `install_session_service(ServiceFactories&)` con el valor `family` del registro, referenciado desde `install_service_factories` en `src/serve/run.cpp` tras la opción de CMake de la familia | `install_session_service` |

### El contrato Engine

Obligatorio:

- `prefill(tokens, state)`: consume los tokens en la ranura. La llamada debe
  consumir toda su entrada, para que una instantánea posterior quede exactamente en
  esa frontera.
- `decode(state, sampler)`: produce uno o más tokens. El más reciente queda
  *pendiente* (producido, aún no consumido como fila). El resultado indica la
  frontera anterior y posterior y el motivo de fin. Un paso es terminal cuando llega
  a un token de parada (`stop`), agota `sampler.max_tokens` (`length`), llena el
  contexto (`context`) o se cancela; el runtime rechaza un resultado que incumpla
  estas reglas.
- `caps()`: contexto, ranuras, anchura de lote y qué funciones opcionales existen.
- `receipt()`: el recibo de ejecución que acompaña a cada respuesta.

Opcional, cada elemento anunciado en `caps()`: `prefill_batch` y `decode_batch`
(varias sesiones en una pasada), `advance` (filas de prefill y decode mezcladas en
un ciclo, con `RowHints`), `fork_lane`, `cancel_lane`, `resume_lane`,
`append_external` (carriles de un trabajo paralelo), `commit_pending` y
`RecurrentStateOps` (`include/strix/session/recurrent_state.hpp`) para modelos cuyo
estado no es solo KV de atención.

### Operaciones del Backend

Una familia pide a un `session::Backend` (`include/strix/session/backend.hpp`):

- búferes (`allocate`), colas (`create_queue`), eventos (`record`, `wait`,
  `synchronize`);
- copias entre host y dispositivo (`upload`, `download`, `copy`) y
  `memory_budget()`;
- núcleos mediante el registro de operaciones: `supports(key)` y `query(key)`
  responden si existe una operación para un tipo de dato y formato, `dispatch`
  ejecuta una y un `ExecutionPlan` ejecuta una secuencia preparada. Una operación
  inexistente se notifica como `UnsupportedOperation`, nunca con una alternativa
  silenciosa.

El despacho es por operación y en el host. No hay llamadas virtuales dentro de un
núcleo.

## Qué recibe una familia

Lo aportan las capas de sesión y servicio cuando existen las piezas anteriores:

- **Servicio**: `strix-serve --model <nombre>`; la interfaz compatible con OpenAI
  (`/v1/completions`, `/v1/chat/completions`, `/v1/messages`, `/v1/responses`,
  `/tokenize`, streaming, herramientas, salida estructurada), salud, estado,
  métricas, drenaje y autenticación.
- **Sesiones y ranuras**: el dueño de las peticiones (`serve::SessionRuntime`)
  asigna ranuras, planifica prefill y decode, agrupa sesiones hasta la anchura
  declarada, gestiona cancelación y condiciones de parada y comprueba cada resultado
  de decode.
- **Caché de prefijos**: restauración del prefijo común más largo y la instantánea
  de fin de turno que hace que el turno siguiente solo procese el mensaje nuevo. Es
  una fachada sobre el único `PrefixStateCache`, así que hay un solo presupuesto de
  bytes y un solo escritor duradero (`include/strix/session/prefix_cache.hpp`,
  `checkpoint_io.hpp`).
- **Presupuesto de filas**: `RowHints` y `EffectiveRows` llevan al motor, y de
  vuelta, el presupuesto de filas por ciclo de una cohorte; una familia que
  implementa `advance` recibe cohortes mixtas de prefill y decode del planificador
  compartido.
- **Trabajos paralelos**: bifurcación, cancelación por carril y carriles en lote
  (`include/strix/session/parallel_job.hpp`).

Bibliotecas compartidas que una familia llama en lugar de copiar (véase «Shared
code first» en `AGENTS.md`):

- **Canalizaciones de dos nodos y transportes**: `include/strix/pair` (enlace,
  lockstep, pipe, balance) e `include/strix/dist` (selección de transporte, copia
  directa entre GPU y copia por host).
- **Núcleos**: `src/core` (planes lineales, enteros y empaquetados), `src/moe_core`
  (enrutado MoE y GEMM agrupado), `src/attn_core` (teselas de atención flash,
  máscaras, softmax en línea).
- **Borradores**: `include/strix/spec` y las cabeceras del controlador y el
  verificador de especulación. Una familia que aporta una fuente de borradores la
  declara en `caps()` y en los contadores de aceptados y propuestos de `StepResult`;
  el borrador está activo por defecto donde la familia lo ha cualificado.

## Backends

| Backend | Ficheros | Se compila cuando | Se elige |
|---|---|---|---|
| Referencia en CPU | `include/strix/session/backend_cpu.hpp`, `src/session/backend_cpu.cpp` | siempre (parte de `strix_session`) | el servicio construye `session::CpuBackend` |
| HIP | `include/strix/session/backend_hip.hpp`, `src/session/backend_hip.cpp` | `STRIX_ENABLE_HIP_RUNTIME=ON` (`strix_session_hip`) | el servicio construye el backend HIP |

El servicio es el único lugar que nombra un backend: en el ejemplo es el miembro
`ss::CpuBackend backend_;`, y el motor recibe un `session::Backend&`. Una familia
que sigue este patrón no necesita ninguna cabecera de acelerador en su adaptador.
`backend_hip.hpp` no expone ningún tipo de HIP; el runtime de HIP solo lo incluye
`src/session/backend_hip.cpp`.

### Qué se filtra todavía

La frontera cubre búferes, colas, eventos, copias, presupuesto de memoria y despacho
de operaciones. Hoy es la frontera de almacenamiento y planificación, todavía no la
de los núcleos:

- Los núcleos de producción de todas las familias son unidades de traducción HIP
  (`.hip`) que reciben directamente flujos y punteros de dispositivo de HIP. Sus
  operaciones aún no están en el registro de operaciones del Backend, así que una
  familia no puede ejecutar su pasada real en otro backend cambiando el backend
  construido.
- Los núcleos compartidos (`src/core`, `src/moe_core`, `src/attn_core`) y los
  transportes directos entre GPU incluyen directamente el runtime de HIP.
- Los adaptadores de sesión de las familias migradas usan `session::Backend` para
  el estado de sesión, mientras que sus motores conservan tipos de HIP para ejecutar
  el modelo.
- El camino CUDA es un conjunto de fuentes aparte (`src/cuda`,
  `cmake/QwenCuda.cmake`) que todavía no implementa `session::Backend`;
  `engine/backends.md` lo describe.
- Las familias no migradas tienen su propio bucle de peticiones y sus cachés y no
  usan la frontera.

El backend de CPU es una referencia de corrección para los contratos y las pruebas
de host. No es un camino de inferencia de producción.

## Lista de comprobación

1. `models/<modelo>.toml`, fila en el registro, opción de CMake
   `STRIX_FAMILY_<NOMBRE>`.
2. Código del modelo en `src/<familia>/`: tensores, numérica y núcleos realmente
   específicos del modelo. Reutilizar `src/core`, `src/moe_core`, `src/attn_core`.
3. Adaptador `session::Engine` y ganchos de `KvStore`; declarar las capacidades con
   exactitud.
4. `SessionService` e `install_session_service`; sin `main()` y sin bucle de
   peticiones.
5. Prueba de host con el backend de CPU o un motor guionizado, al estilo de
   `tests/skeleton_engine_tests.cpp`; puerta de exactitud en dispositivo para los
   núcleos reales.
6. Documentación en `docs/models/` y una fila en la tabla de familias.
