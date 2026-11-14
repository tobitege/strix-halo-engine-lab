# strix-router: un motor por nodo, una sola dirección para los clientes

Cuando un modelo cabe en la memoria de un solo nodo, ejecuta un motor completo en cada nodo y pon `strix-router`
delante. Cada nodo conserva su ancho de banda de memoria y su capacidad de prefill; el enlace entre nodos solo
transporta HTTP. Los modelos que necesitan las dos memorias (colocaciones solo en pareja) no pueden usarlo.

El router es un binario aparte. Los motores no cambian y responden exactamente igual que al llamarlos directamente.

## Arranque

```sh
# en cada nodo: el servidor habitual, por ejemplo en el puerto 8080
# en el nodo al que hablan los clientes:
strix-router --listen 0.0.0.0:8090 \
    --replica http://NODO_A:8080 --replica http://NODO_B:8080
```

Los clientes usan `http://ROUTER:8090` con las mismas rutas, cuerpos y credenciales (las cabeceras se reenvían).

## Colocación

1. Una sesión con nombre (`/v1/sessions/<id>/...`, cabecera `X-Session-Id` o campo `session_id` del cuerpo) vuelve a
   la réplica que la tiene.
2. Si no, la réplica que ya vio el tramo inicial más largo del prompt (hashes de bloques de 1 KiB), salvo que esté
   más de un slot más ocupada que la otra.
3. Si no, la réplica menos ocupada (`slots_busy + queue_depth` del `GET /api/status` de cada réplica).

Una réplica que deja de responder, o que informa `failed` o `draining`, no recibe nada nuevo. Una petición solo se
mueve a otra réplica cuando la conexión fue rechazada, antes de enviar un solo byte; después, un fallo se devuelve al
cliente (502, o un stream que acaba sin su terminador) y nunca se reintenta, así que nada se genera dos veces.

## Comprobación

- `GET /health`: 200 mientras haya al menos una réplica sana.
- `GET /router/status`: por réplica, salud, capacidad, peticiones en curso y cómo se colocaron
  (`by_session`, `by_prefix`, `by_load`).
- Cada respuesta reenviada lleva `X-Strix-Replica: <índice>`.

## Opciones

`--poll-ms` (periodo de sondeo del estado, 250 por defecto), `--connect-timeout-ms` (2000), `--idle-timeout-s`
(0 = sin límite), `--block-bytes` (1024), `--affinity-entries` (4096), `--max-connections` (256),
`--status-target` (`/api/status`).

## Vuelta atrás

Para `strix-router` y apunta los clientes de nuevo al servidor de un nodo. En los nodos no se cambió nada.

## Límites

Sin TLS ni autenticación propia: enlázalo a una interfaz de confianza, igual que los servidores. Las respuestas se
reenvían sin inspeccionarlas, así que la velocidad por token de cada réplica aún no se usa para colocar.
