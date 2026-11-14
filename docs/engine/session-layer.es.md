# GuÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­a breve de sesiones compartidas

Esta guÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­a describe la implementaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n actual de sesiones y servicio compartidos.
No implica que todas las familias ni todos los backends estÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©n migrados.

## Responsabilidades

La familia conserva tokenizaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n, plantilla de chat, polÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­ticas de parada,
aritmÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©tica y planificaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n de cÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³mputo. `session::Engine` ofrece prefill/decode,
capacidades efectivas y recibos; puede aÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â±adir avance mixto o hooks de carriles
cuando su adaptador los implementa. Gemma4 usa avance mixto con su planificaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n
agrupada nativa, sin convertirla en llamadas escalares.

`SessionRuntime` compartido posee la admisiÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n de `GenerationQueue`, los leases de
`KvStore`, la admisiÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n de prefijos, el recuento y publicaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n de resultados y el
cierre de solicitudes. El planificador de cada familia mantiene su agrupaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n,
aritmÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©tica y semÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡ntica. VÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©anse el [contrato en inglÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©s](session-layer.md), los
[backends](backends.md) y el contrato de [estado recurrente](recurrent-state.md).

## SelecciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n, inspecciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n y ejecuciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n

`models/registry.toml` y `models/*.toml` describen modelos seleccionables; los
metadatos no implementan kernels ni garantizan que exista un adaptador ejecutable.
El registro descriptivo y el registro de admisiÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n GGUF (`ModelFamily`) son
distintos. `--model` acepta un ID/nombre del registro o un alias de familia si
selecciona una ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Âºnica entrada.

Ejemplos con placeholders relativos para los assets locales:

```sh
strix-serve --list-models
strix-serve --model gemma4 --inspect
strix-serve --model gemma4 --family-inspect
strix-serve --model nemotron_3_5_bolt_e4b -- MODEL.gguf --rope-tables TABLE_DIR --tokenizer tokenizer.json
```

`--inspect` carga registro y descripciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n, e informa si hay una fÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡brica compartida
enlazada. No carga assets de la familia ni inicializa dispositivos. En cambio,
`--family-inspect` valida con la fÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡brica enlazada y usa su inspecciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n diferida;
tambiÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©n termina antes de construir el servicio o iniciar trabajo de dispositivo.
La inspecciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n de familia depende de una fÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡brica compilada. Una familia conocida
sin adaptador compartido no puede ejecutarse mediante este shell.

En el comando de servicio, `--` separa opciones del shell de los argumentos de
Gemma4. Se requieren un checkpoint y assets compatibles. El ejemplo indica la
forma del comando; no promete que cualquier archivo con esas extensiones sea
admisible.

## ConstrucciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n y replay de host

El replay de Gemma4 puede compilarse como proyecto host independiente, sin SDK de
acelerador ni descarga de tokenizer oficial:

```sh
cmake -S tests/gemma4_session -B build/gemma4-session-p5
cmake --build build/gemma4-session-p5 --config Debug
ctest --test-dir build/gemma4-session-p5 -C Debug --output-on-failure
cmake --build build/gemma4-session-p5 --config Release
ctest --test-dir build/gemma4-session-p5 -C Release --output-on-failure
```

El proyecto crea el fixture sintÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©tico que necesita el replay. Ese fixture sirve
para contratos y control de flujo; no acredita fidelidad del modelo. El backend
CPU es una referencia ejecutable para pruebas de recursos/operaciones, no un
ejecutor capaz de cargar cualquier modelo.

El replay congelado compara casos sintÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©ticos con una captura independiente de la
implementaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n original. Las dos aplicaciones de servicio actuales usan el
cÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³digo migrado; no son dos lados independientes de una comparaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n antigua/nueva.
Las suites host independientes Debug y Release pasaron 15 pruebas cada una; la
suite Debug raÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­z pasÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³ 17, con dos omisiones por falta de tokenizer oficial. Esto
demuestra ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Âºnicamente los casos host descritos, no paridad GPU, fidelidad con
modelos reales, rendimiento nativo ni portabilidad universal.

## Estado y lÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­mites de adopciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n

- Gemma4 es la primera fÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡brica de servicio real conectada a `SessionRuntime`.
  El shell compartido y el ejecutable legado de Gemma4 usan el mismo propietario
  compartido.
- MiniMax dispone de fÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡brica real enlazada explÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­citamente, preparaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n host,
  Runner y recursos nativos extraÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­dos. La entrada de compatibilidad usa el mismo
  shell y conserva `--model-id`. El lÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­der devuelve `Lead::runtime()`; el shell
  inicia un ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Âºnico propietario. Rank 1 ejecuta `follow()` sin runtime de lÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­der
  ni listener. La auditorÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­a combinada de fuentes, selectores y valores por
  defecto incluye ahora los contratos compartidos de sesiÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n, los propietarios
  de cola/almacenamiento, el runtime y su registro de compilaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n. Quedan filas
  obsoletas del recibo sin resolver y el control sigue cerrÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡ndose ante ellas.
  Esta auditorÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­a no califica comportamiento numÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©rico ni
  ejecuciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n nativa.
- Qwen, DeepSeek, GLM y otras entradas de servicio/rank conservan sus rutas
  actuales. Unificar los ejecutables rank no forma parte de esta etapa.
- `STRIX_LEGACY_MAINS` estÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡ activado por defecto para las entradas de
  compatibilidad Gemma4/MiniMax. Desactivarlo conserva bibliotecas de familia,
  servidor nuevo y pruebas; no significa que todos los ejecutables se hayan
  unificado.
- La selecciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n TOML no prueba admisiÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n GGUF, y la fÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡brica de servicio no migra
  los registros principales de cada familia.

## Estado, cachÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â© y snapshots

`KvStore` administra handles con generaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n y estado por rank. Cada snapshot
sellado vincula payload, dominio/esquema, orden de ranks, frontera y propÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³sito.
Un snapshot de reutilizaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n de prompt contiene filas consumidas coherentes y
contexto oculto admitido; al restaurarlo se crea un sampler y drafter nuevos.
Un snapshot completo de reanudaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n preserva el token pendiente y cursores de
RNG, draft, selecciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n y polÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­tica en todos los ranks. No se debe convertir uno en
otro modificando metadatos.

Truncar la longitud de atenciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n no restaura por sÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­ solo recurrencia, draft ni
cursores. `PrefixCache` exige identidad compatible del modelo, perfil numÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©rico,
formato de estado y topologÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­a de ranks; la coincidencia de tokens no autoriza a
interpretar payloads opacos. Los fallos de validaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n/restauraciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n deben rechazar
la reutilizaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n antes de exponer estado parcial.

La interfaz `RecurrentStateOps` describe snapshot, fork, append, rollback,
densify y serializaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n. La referencia host de carriles usa COW real de pÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡ginas
`KvPages`, pero los logs de factores no vacÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­os y la matemÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡tica productiva de
importaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n/serializaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n no estÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡n implementados. Gemma4 no anuncia fork de
carriles, COW de prefijo compartido, append externo ni reanudaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n de carriles.
Una copia de estado no debe etiquetarse como COW nativo. La captura fÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­sica
completa puede rechazarse si no se puede clonar todo el estado del drafter/rank.

## Exactitud y rendimiento nativos

El replay host no sustituye la calificaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n del dispositivo. Antes de afirmar
exactitud o ausencia de regresiÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n hay que compilar y ejecutar en el hardware
objetivo y comparar IDs completos, logits sin procesar, estado fÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­sico y recibos
por fase bajo la misma identidad numÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©rica, perfil, contexto y capacidad. TambiÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©n
se deben ejercitar los fallos de cachÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©/drafter, cancelaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n y reutilizaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n de
slots, y medir cargas completas intercaladas. Hasta pasar esas barreras, no se
afirma paridad acelerador ni ausencia de ralentizaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n.

## Servicio MiniMax

```sh
strix-serve --model minimax --inspect
strix-serve --model minimax -- MODEL_DIR --print-effective-config
strix-serve --model minimax -- MODEL_DIR --rank 0 --transport usb4-auto
strix-serve --model minimax -- MODEL_DIR --rank 1 --transport usb4-auto
```

`session_service.*` conserva MinimaxApi, Observer, sesiones, mÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©tricas y admin;
`serve_runtime.*` retiene recursos y votos de ambos ranks antes del listener;
`serve_runner.*` conserva comandos, orden de confirmaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n y pruebas de resume.
Las salidas de sesiÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n CLI/check-only usan el resultado de arranque tipado.
READY muestra el puerto enlazado. Los fallos cierran tareas/cola/HTTP y unen el
propietario antes de destruir recursos. Temperatura positiva conserva greedy y
su nota. Los modelos y tokenizers deben estar presentes; no hay descargas.

La fÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡brica host rechaza ejecuciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n nativa si el motor no estÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡ compilado. Las
[pruebas de servicio](../../tests/minimax_service/README.md) usan esa fÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡brica y
el servicio real con recursos sintÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©ticos. No certifican ejecuciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n acelerada ni
sustituyen replay del propietario, revisiÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n de recibos o pruebas de ambos ranks.
No se admite COW nativo, fork/resume, append externo ni captura genÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©rica de
estado emparejado. Los resultados host no acreditan fidelidad del modelo ni
rendimiento nativo.
Con HIP desactivado no existe el ejecutable legado nativo, incluso con
`STRIX_LEGACY_MAINS=ON`; OFF conserva la biblioteca y las pruebas host.

El censo MiniMax incluye fuentes recursivas de la familia, bibliotecas MoE,
drafter y transporte enlazadas, contratos compartidos de sesiÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n y la API e
implementaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n de `GenerationQueue`, propietarios de almacenamiento/backend/
prefijo y la implementaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n de `SessionRuntime`,
el registro de CMake raÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­z y `cmake/StrixSession.cmake`. Un archivo nuevo en ese
ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡mbito queda sin revisar y falla cerrado. HTTP, tokenizaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n/registro y el
transporte mantienen sus controles separados; Gemma4 y sus pruebas anidadas
pertenecen a otro control. Siete huellas antiguas de dispositivo/defaults/core
estÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡n identificadas en el historial, pero esos archivos cambiaron despuÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©s. Los
cambios actuales aÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Âºn no se han contrastado con sus gates de selectores y las
filas siguen sin resolver y fallan cerrado. El censo no califica exactitud ni
ejecuciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n GPU.

La prueba ejecutable del observador compila el wrapper `Borrowed` extraÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­do y el
helper de controles con transporte falso. Comprueba votos de inicio, exclusiÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n
de controles ordinarios, IDs, tamaÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â±os de logits/estado, rechazo de sesiÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n antes
del follower y finalizaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n de una solicitud. La ruta histÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³rica se ejecuta si
el objeto estÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡ disponible; su ausencia no se considera equivalencia.

## RetenciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n al completar un turno

La polÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­tica de biblioteca `completed_turns` estÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡ desactivada por defecto. Usa
el PrefixCache existente, identidad de conversaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n/plantilla/topologÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­a, LRU
con RAM finita y nombres protegidos dentro del mismo lÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­mite. La operaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n
`commit_pending` consume el ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Âºltimo token publicado sin seleccionar otro ni
avanzar RNG; la restauraciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n inicia el muestreo de la nueva peticiÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n. Los fallos
de cachÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â© conservan la respuesta y su recibo. VÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©ase el [contrato](conversation-cache.md).
`turn_checkpoint` admite antes del commit. El hook opcional final `turn_snapshot`
prepara el candidato despuÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©s del commit y de validar la frontera, antes de
capturarlo. `false` rechaza la cachÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©; una excepciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n cuenta como fallo de cachÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©.
Se comprueba cancelaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n despuÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©s del commit, despuÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©s de preparar y antes de
admitir. La ausencia del hook conserva los productores anteriores. Un lÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­mite de
bytes cero se rechaza antes del trabajo; uno positivo insuficiente falla tras
capturar, sin cambiar la respuesta ni el recibo.

La suite compartida independiente pasa 11/11 en Debug/Release con legacy ON/OFF.
Las pruebas reales Gemma de continuaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n, commit y cachÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â© estÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡n registradas contra
el motor de producciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n. Su integraciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n raÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â­z y anidada sigue pendiente/Not Run:
la generaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n falla mientras falta la fuente obligatoria de integraciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n. Las
pruebas simuladas no certifican estado nativo, COW, sesiones distribuidas,
rendimiento ni fidelidad numÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â©rica.

## Estado MiniMax acotado

La [ruta de estado por pareja](minimax-paired-state.md), desactivada por defecto,
conecta capture/restore y vistas del rango real mediante planes neutrales y el
follower de producciÃƒÆ’Ã‚Â³n. El perfil explÃƒÆ’Ã‚Â­cito excluye drafter configurado, lookup,
actividad con nombre y prefill parcial/aparcado. Conserva copias host inmutables
y usa el rollback por rango de KvStore. El servicio ordinario sigue desactivado
para esta ruta. No habilita COW fÃƒÆ’Ã‚Â­sico, continuaciÃƒÆ’Ã‚Â³n genÃƒÆ’Ã‚Â©rica, cachÃƒÆ’Ã‚Â© de turnos ni
calificaciÃƒÆ’Ã‚Â³n numÃƒÆ’Ã‚Â©rica nativa. La sonda nativa estÃƒÆ’Ã‚Â¡ implementada, sin compilar ni
ejecutar en esta tarea. Gemma4 ya tiene integraciÃƒÆ’Ã‚Â³n host de turnos completos;
su validaciÃƒÆ’Ã‚Â³n independiente y los controles nativos siguen separados.


## Estado final y orden de migracion

La meta completa exige UN propietario de peticiones y slots, UN `ApiService`
con `/v1/messages`, `/v1/responses` y `/tokenize` para todas las familias, y UN
cache/checkpoint/escritor durable con payloads del pool existente. El orden es
Gemma4, MiniMax, Qwen, DS41, GLM. Dos launchers migrados no completan la meta.
Las implementaciones adicionales siguen pendientes. El
[inventario exacto de eliminaciones](session-layer.md#consolidation-end-state-and-migration-order)
indica los archivos por familia y los gates de aritmetica, alineacion y formatos
en disco. `PrefixCache` y `NamedStateCache` ya delegan la retencion RAM al almacen/pool
existente; cada familia elimina su antigua politica e IO al migrar. Backends de
recursos, estado recurrente, parallel jobs y paginas quedan retenidos hasta tener
adoptante de produccion, conservando los contratos fanout congelados.

MiniMax negocia `STRIX_MM_SERVE_PREFIX=1` en ambos rangos y la misma alineacion.
El unico indice residente vive en `PrefixStateCache`; `session::PrefixCache` valida y consulta, con leases de
`SessionRuntime`/`KvStore`; no es un snapshot inmutable ni una copia host.
Reutiliza el mayor prefijo de slots libres, alineado, dejando el ultimo token.
Drafter/nesting quedan frios; decode no amplia la frontera segura. Cancelacion
parcial conserva solo el prefijo seguro anterior. Admin, reset, poison y nueva
generacion invalidan antes de mutar; no-store no retiene otra entrada. Los
perfiles genericos de estado/input siguen separados y desactivados por defecto.
Los gates nativos de compilacion, IDs/logits/estado por rango, capacidad y
rendimiento completo siguen pendientes; las pruebas host no los sustituyen.


La consolidacion RAM elimina las listas/mapas propietarios y los bucles de
victimas de ambas fachadas. Gemma4 usa el propietario comun desde su
`SessionRuntime`; MiniMax inyecta la misma instancia en los prefijos residentes
y checkpoints codificados con nombre. Un presupuesto de servicio contiene las
cuotas; leases y reservas siguen contabilizados despues de expulsar una entrada.
Los snapshots de dispositivo conservan su tipo y propiedad inmutable; un
certificado residente cobra solo metadatos. Los buffers host usan el pool sin
una segunda copia. Las excepciones antiguas de indice/oversize son explicitas y
no pueden eludir el limite estricto de turnos completos, desactivados por defecto.

Los adaptadores de sesiones Qwen/MiniMax y prefijos del pool usan el IO host
comun para lock, lectura y publicacion. El sobre compartido conserva codecs y
formatos antiguos; Gemma4 sigue con estado RAM tipado. La biblioteca comun es
independiente de HTTP/core y cada TU tiene un solo propietario. Las pruebas se
registran solo con fuentes y adaptadores reales; faltantes quedan pendientes.
Ver [almacenamiento de checkpoints](checkpoint-storage.md). Aun se migran y eliminan
politicas/IO en orden Gemma4 -> MiniMax -> Qwen -> DS41 -> GLM. La meta de UN
propietario/frontend/cache/checkpoint/escritor no esta completa; los componentes
sin adoptante quedan HELD y los gates nativos siguen pendientes.
