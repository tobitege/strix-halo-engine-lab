# Entrada de imágenes

La entrada de imágenes permite que una petición de chat lleve imágenes. Hoy un solo modelo las admite: DeepSeek V4.1
en el par de dos nodos. Un [runtime de visión compartido](shared-cores.es.md#los-componentes) sirve a todas las familias; cada modelo solo aporta un adaptador
(la interfaz del adaptador está en [vision-adapters.md](vision-adapters.md), en inglés).

| Modelo | Entrada de imágenes |
|---|---|
| DeepSeek V4.1 | sí, en el par, con `--vision` |
| Qwen, GLM, MiniMax, Gemma | todavía no |

## Uso

Arranque el servidor con `--vision <GGUF de visión>` en el rank 0, `--prefill-path device` y un tamaño explícito
`--prefill-rows N`. Se rechaza visión con `--deploy`, `--deploy-check` o `--prefill-rows auto` hasta incluir en la admisión
los bytes fuente retenidos, los picos de dispositivo, las filas de caché/trama y las copias temporales. El tamaño manual
debe dejar espacio para estas asignaciones en ambos ranks. La bandera está desactivada
por defecto; sin ella no se ejecuta código de visión y el rank 1 no guarda nada de él. Envíe un chat completion al estilo
OpenAI con una parte `image_url` (URL `data:`):

```bash
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}]}'
```

## Ruta de datos

1. **Puerta de medios** (`ApiMediaBridge`, `include/strix/vision/api_media.hpp`). Decodifica PNG y JPEG bajo
   `DecodeLimits` (`image.hpp`). WebP y GIF se rechazan con un 400 ("send PNG or JPEG").
2. **Preprocesado** (`preprocess.hpp`, `ResizeRule`). El plan de redimensionado del modelo original: se mantiene la
   proporción sobre un lienzo gris, los lados son múltiplos de 14, los píxeles son BF16 y los parches se fusionan 3x3
   con celdas parciales rellenas de ceros. El tramo (`embed_span.hpp`, `SpanLayout`) es de como máximo 1024 filas,
   contando los fines de fila y los marcadores de inicio y fin.
3. **Runtime de codificador compartido** (`Encoder`, `encoder.hpp`; `src/vision/encoder.hip`). Una torre ViT dirigida
   por datos para todas las familias. La aritmética HIP es `vision-f16-v1`: paneles FP16 con escala potencia de dos y
   acumulación FP32. DeepSeek usa activaciones divididas hi+lo.
4. **Proyector** (el aligner), en el adaptador (`VisionAdapter`, `adapter.hpp`; DeepSeek:
   `deepseek41/vision_adapter.hpp`).
5. **Inserción**. Las filas de embedding entran en el flujo de tokens como ids virtuales. Las filas de imagen llevan la
   marca de imagen del router. En Engram, las posiciones de imagen no participan en los n-gramas y la puerta de Engram
   vale 0 allí, como en el modelo original.
6. **Trama del par** (`include/strix/pair/embed_frame.hpp`). El rank 0 codifica; por cada imagen nueva envía al rank 1
   una trama: manifiesto, filas BF16 y un CRC32C.
7. **Caché de embeddings** (`EmbedCache`, `embed_cache.hpp`). Clave = identidad del adaptador + aritmética + hash de la
   imagen. LRU, 256 MiB por defecto (`STRIX_VISION_CACHE_MIB`; `0` desactiva el almacenamiento). Vive solo mientras
   dura el proceso y nunca se escribe a disco. Los prompts con imagen nunca entran en la caché de prefijos ni en las
   sesiones.

## Medido

En el par de dos nodos, DeepSeek V4.1 Q2, un PNG de 37x23 ampliado a 1550 parches (200 filas de imagen + 21 tokens de
texto):

| Caso | Tiempo hasta el primer token | Detalle |
|---|---|---|
| Primera petición (imagen sin caché) | 1,46-1,87 s | codificación 0,93-1,07 s; trama al rank 1 de 2,0 MB en 4-5 ms; prefill del LM unos 0,46 s |
| Misma imagen otra vez (acierto de caché) | 0,53-0,61 s | sin codificar; trama de 80 bytes |

Decode: 39-41 tok/s.

## Interruptores

| Interruptor | Por defecto | Significado |
|---|---|---|
| `--vision FILE` | desactivado | Bandera del servidor, solo rank 0. Requiere `--prefill-path device`. |
| `STRIX_VISION_RESIDENT` | 1 | `1`: la torre se sube al arrancar; `0`: con la primera imagen. |
| `STRIX_VISION_CACHE_MIB` | 256 | Tamaño de la caché de embeddings; `0` desactiva el almacenamiento. |

Véase [toggles](../toggles.es.md).

## Límites

- Solo PNG y JPEG.
- Se rechazan las imágenes que superan el presupuesto de filas.
- Se rechazan las peticiones con imagen y `session_id`.
- Se rechaza el vídeo.
- Hasta ahora un solo adaptador de imagen (DeepSeek V4.1); los demás modelos aún no admiten imágenes.
- Requiere la ruta de prefill en dispositivo.
- La codificación se serializa con el preprocesamiento; la planificación peer/split y el solapamiento con el prefill del prefijo de texto quedan pendientes.
- Las posiciones de imagen tienen puertas Engram a cero. Las pruebas sintéticas no cualifican la fidelidad de imágenes servidas;
  la evidencia debe identificar la revisión probada y cubrir esa aritmética en la carga completa de imagen.
