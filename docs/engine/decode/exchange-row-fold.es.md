# Plegado de fila del intercambio (suma en CPU del all-reduce de una fila)

En la decodificación en pareja cada subcapa termina con un all-reduce pequeño: cada rango publica su fila parcial, el
hilo de progreso del módulo de intercambio la mueve por el enlace y un kernel posterior suma los dos parciales antes
de la norma / expansión. Con el plegado, el propio hilo de progreso suma los dos parciales (primero el del rango 0)
justo al llegar el dato del par y antes de publicar la bandera de recepción; el kernel lee una sola ranura plegada.

- Código compartido: `src/xport/row_fold.hpp`, `Channel::fold_next()`, `Channel::can_fold()`,
  `Channel::last_folded()`. Los motores solo declaran la operación antes del envío.
- Exactitud: una suma binary32 por elemento en ambos lados; los resultados son idénticos bit a bit (pruebas
  `strix-xport-row-fold-tests` y `strix-hip-glm5-post-exchange-rows-tests`).
- Solo la suma del par puede pasar a la CPU: la norma (MiniMax) y la expansión (GLM) necesitan estado residente en el
  dispositivo, así que el lanzamiento del kernel se mantiene.

## Uso y vuelta atrás

- Activar: `STRIX_MMX_FOLD=1` (MiniMax) o `STRIX_GLMX_FOLD=1` (GLM); el valor es el máximo de filas plegadas.
- Por defecto: 0 (apagado). Vuelta atrás: quitar la variable o ponerla a 0. El formato de cable no cambia.
- Diagnóstico: `STRIX_MMX_RX_TRACE=ruta` / `STRIX_GLMX_RX_TRACE=ruta` escribe los tiempos de cada intercambio en
  `ruta.<rango>` al salir.

## Medición (pareja de dos nodos, una fila por paso)

- Reparto del kernel posterior: MiniMax 48,3 us = 28,1 us de espera al par + 20,2 us tras la llegada; GLM 24,5 us =
  21,2 us + 3,3 us. El dato del par nunca llegó antes de empezar el kernel: la mayor parte es latencia del enlace.
- A/B en el mismo proceso, mismos tokens, logits y estado: MiniMax 8K neutro (+0,03 ms/token); GLM de -0,03 a
  -0,05 ms/token, en el límite del ruido. Por eso queda apagado por defecto.
