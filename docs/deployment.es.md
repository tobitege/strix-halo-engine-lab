# Despliegue en otro hardware

[English](deployment.md) · [Interruptores individuales](toggles.md)

El despliegue se calcula a partir de medidas del dispositivo y de los recursos
del modelo, sin clases de hardware ni tablas basadas en la RAM instalada. Compile
para el destino GPU real: admitir un presupuesto de memoria no añade soporte de
kernels para otra arquitectura.

## Disponibilidad

Comprobado en los dos nodos gfx1151 del laboratorio (117,2 GiB de memoria utilizable
por el dispositivo en cada uno): compilación nativa estricta de cada aplicación de la
tabla, pruebas de host, `--deploy-check` en ambos rangos (sin abrir el dispositivo ni
conectar con el par) y con tamaños de memoria simulados. La última columna indica lo
que ha mostrado una ejecución servida con `--deploy auto` en esos nodos. Siguen siendo
necesarias la cualificación completa de tokens, logits y estado y la medida de picos
de memoria en otro hardware; una inspección admitida no es esa cualificación.

| Motor | Opciones de despliegue | Requisitos de arranque con `--deploy` | Comprobado en los nodos |
|---|---|---|---|
| Qwen | Codificación nativa; un nodo, dos réplicas, prefill explícito en pipeline de dos nodos | Ruta eager tiled registrada; HTTP necesita los argumentos del servidor MTP tiled. Los slots no fijados usan los límites nativos por pila de estado, scratch, metadatos y bridge; `--http-batch N` limita la búsqueda (`0` conserva el runner único). La RAM de sesión predeterminada se ajusta al resto tras reservar slots y propietarios obligatorios; un `--session-ram-bytes` explícito no se reduce. Pipeline solo para prefill CLI con continuación en rango 0, no HTTP. YaRN y las rutas copy/graph se rechazan. | Servido (`--deploy auto`, HTTP): plan, `deployment_id`, capacidad y drenado según lo especificado; 24 tokens greedy idénticos a los del mismo arranque sin `--deploy`. El adaptador anterior forzaba 1 slot donde el arranque normal servía 4; el nuevo cálculo tiene pruebas host y espera una nueva comprobación en los nodos. |
| Nemotron | Q4_K_M; un nodo o réplicas independientes | `--draft off`. No están ligados el dimensionamiento EXL3/con borradores ni la ejecución en par. La cota de paneles preparados del prefill es conservadora. | Servido (`--deploy auto`, un nodo): 8 slots con contexto 65.536, igual que el arranque normal; plan, `deployment_id`, capacidad y drenado según lo especificado; 24 tokens greedy idénticos sin `--deploy`. |
| DeepSeek | Codificación nativa; TP2 | `--cache-profile off`, `--no-dspark`, sin RAM de sesión ni etapa heterogénea, trellis desactivado (`STRIX_DS41_TRELLIS=0`: su workspace no está dimensionado). Las estimaciones de prefill no admitidas se rechazan. | Par servido (`--deploy auto`): un plan acordado (84,6 GiB requeridos por rango; la carga indica 79,8 GiB de pesos y 0,88 GiB propios del prefill); capacidad y drenado según lo especificado; 24 tokens greedy idénticos sin `--deploy`. Esa compilación aún no publicaba el plan en `GET /v1/deployment` ni en `/health`; corregido después (la compilación estricta pasa), comprobación servida pendiente. |
| GLM | EXL3; TP2 | Solo el perfil default; `--prefix-cache-gib 0`. El valor predeterminado y `STRIX_GLM_PF_PIPELINE=2` explícito comparten el límite de scratch de filas completas, anillos de intercambio y buffer coherente de filas normalizadas. Se rechazan importación de prefijos, intercambio externo y pilas truncadas. | Par servido (`--deploy auto`): un plan acordado (94,0 GiB requeridos por rango; el plan de carga del motor son 81,3 GiB físicos); plan, `deployment_id`, capacidad y drenado según lo especificado; 24 tokens greedy idénticos sin `--deploy`. |
| MiniMax | EXL3; TP2 | `--dspark-mode off` (si no, DSpark se detecta junto al modelo), `STRIX_SESSION_CACHE=0` (control de entorno) y sin guardar/restaurar sesiones; `--max-rows` como máximo 4096. | Par servido (`--deploy auto`): ambos rangos acordaron un plan (100,0 GiB requeridos por rango frente a 109,2 GiB de presupuesto; la carga indica 90,3 GiB usados); plan, `deployment_id`, capacidad y drenado según lo especificado; 24 tokens greedy idénticos sin `--deploy`. |

No se ofrecen formatos INT4 (FAST_I4): sus runtimes no están en esta compilación, así
que ningún plan puede elegirlos y fijar `format.prefill = "int4-stored"` o `"int4-jit"`
se rechaza con "no supported candidate". Los proveedores mantienen un punto de
extensión (`deploy::int4_moe_formats`) que añade esos candidatos cuando un motor
enlaza su runtime INT4 y sus controles. `profile.allow_pending` no habilita un
runtime ausente.
El prefill desagregado forma parte del vocabulario del esquema,
pero estas adaptaciones no lo ejecutan. Los comandos sin despliegue mantienen sus
valores predeterminados y contratos de admisión propios.

## Inspección antes de cargar

Añada estas opciones al comando habitual del motor, conservando los recursos de
modelo/estado, el perfil y los argumentos de rango/transporte obligatorios:

```sh
--deploy auto --deploy-check
--deploy deployment.toml --deploy-check
```

`--deploy auto` busca entre las opciones del proveedor. `--deploy FILE` restringe
la búsqueda mediante TOML. `--deploy-check` imprime el plan y termina antes de
cargar los datos de pesos, construir los recursos del dispositivo o conectar el
par; los metadatos deben seguir siendo legibles. La inspección del par necesita
por tanto un inventario medido del otro nodo. Revise `refused`, `refusal`, los
motivos de los candidatos, `requested`, `chosen` y los presupuestos por rango.
No ejecute un plan rechazado. Quite `--deploy-check` para ejecutar una configuración
admitida, conservando los demás argumentos.

El sondeo observa el destino gfx, las CUs, GTT unificada o VRAM dedicada, los bytes
utilizables totales y los libres actuales. Los datos ausentes o ambiguos quedan
desconocidos y no justifican una admisión. Los nodos proceden del inventario medido;
el enlace procede del transporte de lanzamiento y del fichero, sin inventar un
nodo remoto ni medir automáticamente el ancho de banda. Los pares intercambian
medidas recientes en orden de rango y acuerdan el hash antes de cargar. Las réplicas
son procesos independientes gestionados por el orquestador; el despliegue no crea
procesos ni configura la red.

Para cada candidato y rango se reserva, por este orden:

1. KV y otro estado persistente para el contexto y los slots solicitados.
2. Workspace para el prefill/decode admitido.
3. Pesos, conversiones de formato, staging y otros recursos identificados.

El presupuesto es la memoria utilizable libre medida menos el margen de seguridad.
En memoria unificada siguen vigentes las comprobaciones nativas de host/GTT; la
RAM no es un segundo presupuesto que sumar a GTT. Todos los rangos deben caber.
Nunca se reducen el contexto ni los slots fijados. Los slots no fijados se buscan
desde el máximo del proveedor hacia abajo. El orden de topologías/formatos depende
del proveedor; no es un optimizador universal de rendimiento. Las comprobaciones
del asignador nativo siguen siendo determinantes. Qwen comprueba además los slots
planificados contra la admisión nativa medida y rechaza el servicio si ya no caben;
repita la inspección con la presión de memoria actual. La RAM de sesión automática
se ajusta tras los slots KV y puede quedar en cero; el plan y la ejecución usan
la misma cantidad resuelta.

## Contrato TOML

El esquema de entrada es `strix-deploy-v1`; el JSON de salida usa
`strix-deployment-v1`. Son restricciones, no conversiones de modelo ni ajustes
que establezcan cualificación numérica.

| Campo | Efecto y causas de rechazo |
|---|---|
| `schema` | Cadena exacta obligatoria. Se rechazan claves desconocidas, duplicados, tipos incorrectos y alias en conflicto. |
| `model` | Identificador opcional del proveedor; otro identificador se rechaza. Omítalo para usar el proveedor activo. No es una ruta de modelo. |
| `hardware.nodes`, `hardware.link` | Exigen el número de observaciones y un transporte compatible con el lanzamiento. Un nodo puede usar `auto`; varios nodos requieren enlace explícito. No se inventan datos remotos. |
| `topology.mode` | `auto`, `single`, `replica` (alias `replicas`), `tp2`, `pipeline`, `disaggregated`. Solo se ejecutan las opciones anunciadas por el proveedor. TP2 exige dos nodos medidos. |
| `topology.ranks` | Alias del número de nodos; debe coincidir si también se especifica ese número. No inicia procesos. |
| `memory.context` | Capacidad positiva en tokens; se aplica al contexto/capacidad del motor. Si KV no cabe, se rechaza en vez de recortar contexto. |
| `memory.slots` | Número positivo explícito que nunca se reduce. Omítalo para buscar, sujeto al adaptador y a las restricciones explícitas de CLI. |
| `memory.policy` | Solo está implementado `kv-first`; se rechaza `formats-first`. |
| `memory.margin_gib` | Reserva finita no negativa convertida a bytes. Si supera la memoria libre, no cabe. |
| `format.decode`, `format.prefill` | `auto` o un nombre exacto del proveedor. Se rechazan combinaciones no admitidas: Qwen/DeepSeek usan actualmente `native`, Nemotron `q4_k_m`, GLM/MiniMax `exl3`. |
| `format.resident_layers`, `format.total_layers`, `format.jit_slots` | Fijan la colocación exacta, incluido cero. Las capas residentes no pueden superar el total; la tupla completa debe coincidir con un candidato anunciado. No se inventan colocaciones parciales arbitrarias. |
| `profile.name`, `profile.allow_pending` | El perfil debe coincidir con el seleccionado en el lanzamiento. `allow_pending` solo acepta `false`; no cualifica una ruta provisional. |
| `devices` | Array opcional de medidas independientes por nodo, en orden de rango para pares. Datos ausentes, incoherentes o insuficientes impiden la inspección. La ejecución del par los sustituye por medidas recientes. |
| `pins` | Tabla opcional de cadenas que repite selecciones explícitas canónicas con valores idénticos. No permite sobrescribirlas; las diferencias se rechazan. Normalmente se omite. |

Se aceptan los alias raíz `nodes`, `link`, `context`, `slots`, y `topology`/`profile`
como cadenas. No combine un campo raíz con su equivalente anidado. Este esquema
no tiene claves `hardware.gfx`, `min_gtt_gib` ni `format.verify_min_rows`. Los datos
del dispositivo describen observaciones, no clases de hardware.

### Un fichero por motor

Estos contextos, slots y márgenes son ejemplos de política, no mediciones de
capacidad ni rendimiento. Cada fichero debe pasar inspección con sus recursos y
argumentos reales. Los ejemplos multinodo requieren además el inventario indicado
más abajo.

Qwen, un nodo (`qwen.toml`):

```toml
schema = "strix-deploy-v1"
model = "qwen3.8"
[hardware]
nodes = 1
link = "none"
[topology]
mode = "single"
[memory]
policy = "kv-first"
context = 8192
slots = 1
margin_gib = 8
[format]
decode = "native"
prefill = "native"
```

Réplicas de Nemotron (`nemotron.toml`):

```toml
schema = "strix-deploy-v1"
model = "nemotron-bolt-e4b"
[hardware]
nodes = 2
link = "tcp"
[topology]
mode = "replica"
[memory]
context = 8192
slots = 1
[format]
decode = "q4_k_m"
prefill = "q4_k_m"
```

Inicie cada réplica por separado con `--deploy-node 0` o `--deploy-node 1` y
`--draft off`; se vuelve a medir el dispositivo local en ese índice al ejecutar.
El campo de enlace no convierte las réplicas en un par tensor-paralelo.

Par DeepSeek (`deepseek.toml`; se infiere el identificador del modelo activo):

```toml
schema = "strix-deploy-v1"
[hardware]
nodes = 2
link = "usb4auto"
[topology]
mode = "tp2"
[memory]
context = 8192
slots = 1
[format]
decode = "native"
prefill = "native"
```

Par GLM (`glm.toml`):

```toml
schema = "strix-deploy-v1"
model = "glm-5.3"
[hardware]
nodes = 2
link = "usb4-auto"
[topology]
mode = "tp2"
[memory]
context = 8192
slots = 1
[format]
decode = "exl3"
prefill = "exl3"
```

Par MiniMax (`minimax.toml`):

```toml
schema = "strix-deploy-v1"
model = "minimax-m3.1-flash"
[hardware]
nodes = 2
link = "usb4-auto"
[topology]
mode = "tp2"
[memory]
context = 8192
slots = 1
[format]
decode = "exl3"
prefill = "exl3"
```

Use la escritura del transporte que acepte el motor seleccionado. Para TP2 mantenga
un proceso por rango y la configuración habitual de enlace/transporte. Para
inspeccionar sin conexión, añada una tabla `[[devices]]` por nodo. Incluya los seis
campos: `gfx_target` (cadena), `cu_count` (entero), `mem_total_bytes` y
`mem_free_bytes` (enteros), `unified` (booleano) y `source` (cadena no vacía que
describa la procedencia de la medida). Copie observaciones reales; no duplique el
dispositivo local como sustituto de otro nodo sin medir. Los bytes libres no
pueden superar el total. Mantenga privados los inventarios y recibos operativos.

## Formato según el régimen de ejecución

La regla prevista distingue reutilización de filas, no nombres de hardware. En
decode con `R = 1` suele dominar el tráfico de pesos: EXL3 compacto puede minimizar
los bytes leídos. En verificación ancha o trabajo por lotes, INT4 puede ganar si
`rows >= R*`: el umbral del motor/kernel a partir del cual reutilizar trabajo
matricial compensa el almacenamiento y la conversión adicionales. Prefill usa INT4
donde el runtime y el perfil están disponibles y cualificados. `R*` no es una
constante universal ni un campo TOML.

Una vez integrados y cualificados los runtimes INT4, su dimensionamiento y sus
controles, el presupuesto restante podrá alojar todos los paquetes INT4
(`int4-stored`), paquetes por capa preparados al usarlos (`int4-jit`) o un conjunto
parcial residente admitido más staging JIT. También hay que contabilizar fuentes
EXL3 retenidas, escalas/índices, workspace de conversión y picos simultáneos de
carga. Más memoria puede permitir INT4 totalmente residente **después** de reservar
el mismo KV; no aumenta automáticamente un contexto fijado ni cambia la identidad
de decode. Actualmente las adaptaciones GLM/MiniMax rechazan esas selecciones INT4
en vez de presentarlas como ejecución EXL3 equivalente.

| Cambio de hardware | Consecuencia para el plan |
|---|---|
| Más memoria utilizable/libre | Puede admitir más slots no fijados o un formato residente disponible; cada rango debe caber. |
| Más ancho de banda de memoria | Cambia el coste de decode y el umbral INT4; exige medidas locales, no escalar rendimiento publicado. El número de CUs no basta. |
| GPU discreta | Presupuesto sobre VRAM libre, contando transferencias host/dispositivo y staging. Más RAM no convierte los pesos en residentes en VRAM. Siguen vigentes soporte de kernels y admisión nativa. |
| Enlace más rápido | Puede reducir el coste de colectivas o transferencias y desplazar el umbral útil de pipeline/desagregación. No habilita topologías ausentes ni suma presupuestos entre nodos. |

Un nodo evita comunicación. Las réplicas conservan un modelo completo por réplica
y atienden peticiones independientes. TP2 reparte los recursos compatibles entre
dos rangos, pero conserva otros replicados. El pipeline Qwen retiene el modelo
completo en cada rango y cuenta el staging de transferencia. El prefill desagregado
separaría procesamiento del prompt y decode y necesitaría transferencia de estado
compatible; sigue pendiente aquí. La selección automática solo considera las
opciones implementadas del proveedor, no todos estos diseños para cada motor.

## Contrato HTTP para orquestadores

Use la autenticación configurada en el servidor para estas rutas.

| Interfaz | Contrato |
|---|---|
| `GET /v1/deployment` | JSON inmutable del plan activo, con `deployment_id`; 404 si no hay callback de plan activo. No se inventa un plan predeterminado. |
| `GET /health` | Incluye `deployment_id` cuando está conectado. Durante drenaje responde HTTP 503. |
| `GET /api/status` | `capacity` incluye números `slots_total`, `slots_busy`, `queue_depth` y `kv_bytes_free` anulable. DS41 con pool KV compartido publica los bytes de arena sin asignar de todos los rangos libres (incluida la holgura), mediante una copia del scheduler actualizada al reservar/liberar; no es la reserva contigua máxima ni memoria libre del dispositivo. DS41 sin pool, Qwen, GLM, MiniMax y Nemotron publican `null`: el KV pertenece a slots fijos o streams individuales, sin contador de bytes libres de un asignador compartido. La reutilización KV/overlay por stream de Nemotron no es un pool compartido de admisión. |
| `POST /admin/drain` | Cierra admisión de forma idempotente y devuelve `{"status":"draining","drained":false}` hasta completar el trabajo contabilizado, después `drained:true`. Status también expone `drained` tras comenzar el drenaje. |

La admisión de nuevas generaciones tras el drenaje devuelve 503; una cola llena
normal sigue devolviendo 429. Las peticiones en ejecución terminan. La cola común
conserva el trabajo aceptado; DeepSeek mantiene su cancelación previa de trabajo
encolado/diferido. Drenar no termina el proceso. Deje de enviar tráfico, solicite
drenaje, espere a que termine y permita que el gestor de procesos detenga o
sustituya el servidor.

`deployment_id` es el hash canónico del plan (16 dígitos hexadecimales en minúscula),
no un identificador único de proceso/reinicio ni un certificado de fidelidad.
Planes idénticos pueden mantener el ID tras reiniciar; volver a medir puede
cambiarlo. Úselo junto con salud y ciclo de vida del proceso. `/v1/capabilities`
sigue describiendo las políticas API ejecutadas; deployment describe colocación
y presupuestos.

Los controles individuales están en [toggles](toggles.md). Al validar un cambio,
mantenga modelo, perfil, contexto, slots e identidad numérica; compare todos los
IDs de token y logits/estado antes de medir la carga completa.
