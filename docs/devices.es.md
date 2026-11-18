# Dispositivos: declarar una tarjeta conocida o un clúster en una línea

[English](devices.md)

Cada binario de GPU enlaza su tarjeta mediante la declaración de dispositivos (`strix-devices-v1`, ver
[deployment.es.md](deployment.es.md)). Una entrada puede nombrar un **modelo de dispositivo conocido**; cada ruta se
dimensiona entonces a ese modelo: cuánta memoria planifica, qué pesos quedan en VRAM, cuáles van a memoria fijada del
host o se leen del fichero del modelo, cuántas posiciones y huecos de sesión caben, y el mínimo que necesita cuando
nada cabe.

## Una línea

En la línea de órdenes de cualquier ruta:

```
--device-profile rtx-5090-32
```

(o `STRIX_DEVICE_PROFILE=rtx-5090-32`). En el fichero de declaración:

```toml
schema = "strix-devices-v1"

[devices.main]
uuid = "GPU-..."              # la identidad de la tarjeta, como antes
profile = "rtx-5090-32"       # el fabricante sale del perfil
roles = ["replica"]

[devices.host]
kind = "cpu"
memory_gib = 64               # RAM del host que las rutas pueden planificar (opcional)
```

Cualquier valor del perfil se puede sustituir en la entrada: `memory_gib = 30` (un presupuesto menor),
`bandwidth_gbytes_ps`, `units`. `profile = "custom"` toma todos los valores de la entrada (`memory_gib` obligatorio).

Un **clúster de tarjetas iguales** es una entrada sin identidad:

```toml
[devices.pair]
profile = "rtx-5090-32"
count = 2
interconnect = "pcie5x16"     # pcie5x16, pcie4x16, pcie4x4, oculink4x4, usb4, 1gbe ... 100gbe
roles = ["replica"]
```

Esa entrada es *planificada*: el planificador la dimensiona y predice su velocidad de decodificación; la puerta nunca
la enlaza (cada tarjeta en la que corre un proceso se sigue declarando con su propia identidad). Ninguna tarjeta del
catálogo tiene NVLink, así que `nvlink` se rechaza.

## Qué presupuesto gana

Un único resolvedor (`include/strix/device/budget.hpp`) sirve a todas las rutas, igual a la CLI que al servidor:

1. las opciones propias de la ruta (`--vram-total-gib`, `--ram-total-gib` y sus variables de entorno), `none` = sin límite;
2. el `memory_gib` de la entrada (host: el `memory_gib` de la unidad cpu);
3. la memoria del perfil;
4. nada: la ruta usa lo que la tarjeta tiene libre, como sin perfil.

Un presupuesto declarado mayor que la tarjeta se rechaza al arrancar. Uno menor es una emulación: la ruta se comporta
como en el dispositivo menor (memoria libre = el presupuesto menos lo que ya ocupa la tarjeta). La línea de recibo del
dispositivo y `/api/status` muestran `profile=` y el presupuesto solo cuando se declara uno.

| Ruta | Qué dimensiona el presupuesto |
|---|---|
| DeepSeek V4.1 Flash, RTX independiente | los límites de dispositivo completo del planificador de niveles: arena en VRAM, nivel fijado en host, el resto del fichero |
| GLM-5.3 Flash, productor RTX | los mismos límites (`--device-profile` en el productor) |
| Qwen, servidor RTX | los pesos quedan en VRAM: rechaza antes de cargar si no caben; huecos de sesión según el presupuesto |
| Rutas HIP planificadas con `--deploy-check` (DeepSeek serve, MiniMax, Gemma 4) | los datos del dispositivo que lee el plan de despliegue |

## Catálogo

Especificaciones públicas del fabricante salvo los campos que lista la columna `measured`. Los picos son TFLOPS / TOPS
densos (FP16 y FP8 a la tasa de acumulación FP16; las GeForce ejecutan BF16 a la mitad); 0 = sin camino tensorial o no
declarado. La memoria de las tarjetas NVIDIA es el total que informa el runtime. El fichero de datos es
[data/devices/profiles.toml](../data/devices/profiles.toml) (`strix-device-plan --toml`).

| perfil | arch | memoria GiB | GB/s | unidades | FP32 | FP16 | BF16 | FP8 | FP4 | INT8 | enlace host | medido |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| rtx-pro-6000-blackwell-96 | sm_120 | 95.59 | 1792 | 188 SM | 125.0 | 500.0 | 500.0 | 1000.0 | 2000.0 | 1000.0 | pcie5x16 | memory_bytes, units |
| rtx-5090-32 | sm_120 | 31.84 | 1792 | 170 SM | 104.8 | 419.0 | 209.5 | 838.0 | 1676.0 | 838.0 | pcie5x16 | - |
| rtx-5080-16 | sm_120 | 15.92 | 960 | 84 SM | 56.3 | 225.1 | 112.6 | 450.2 | 900.5 | 450.2 | pcie5x16 | - |
| rtx-4090-24 | sm_89 | 23.99 | 1008 | 128 SM | 82.6 | 330.3 | 165.2 | 660.6 | 0 | 660.6 | pcie4x16 | - |
| strix-halo-128 | gfx1151 | 117.19 | 256 | 40 CU | 29.7 | 59.4 | 59.4 | 0 | 0 | 59.4 | integrated | memory_bytes |
| dgx-spark-gb10-128 | sm_121 | 128.00 | 273 | 48 SM | 0 | 125.0 | 125.0 | 250.0 | 500.0 | 250.0 | nvlink-c2c | - |
| rx-9070-xt-16 | gfx1201 | 16.00 | 640 | 64 CU | 48.7 | 195.0 | 195.0 | 389.0 | 0 | 389.0 | pcie5x16 | - |

La RTX 5090 tiene la generación de SM y la velocidad de memoria de la RTX PRO 6000 con 170 de sus 188 SM: las fases
limitadas por cómputo van un 10 % más lentas que en la PRO a igual memoria; la decodificación limitada por memoria, a
igual residencia, queda cerca.

## Planificar sin GPU

`strix-device-plan` imprime el presupuesto y el plan de memoria de un perfil o de una declaración, a partir del
fichero de necesidades de una ruta (`strix-route-needs-v1`):

```
strix-device-plan --list
strix-device-plan --profile rtx-5090-32 --needs scripts/bench/rtx/needs/qwen-serve.toml
strix-device-plan --devices scripts/bench/rtx/devices/2x-rtx-5090.toml --needs scripts/bench/rtx/needs/qwen-serve.toml
```

Salida 0 = cabe, 3 = rechazado (la línea dice el mínimo). Para formas `count = N` añade una predicción por modelo de
ancho de banda (cada tarjeta lee su parte de los bytes por token al ancho de banda del perfil por la eficiencia medida
de la ruta, los bytes desbordados cruzan el enlace con el host, más N-1 saltos por la interconexión). Es una
predicción: ninguna forma está medida.

Servidor Qwen (pesos residentes, necesidades medidas): la RTX PRO 6000 admite 3 huecos a 34.816 posiciones; necesita
al menos 91,24 GiB, así que una RTX 5090 o 5080 rechaza, y también 2x RTX 5090 (46,62 GiB por tarjeta con un reparto
de capas a partes iguales).

## Emular una tarjeta menor

En una tarjeta grande, `--device-profile rtx-5090-32` hace que una ruta planifique para 31,84 GiB. El kit
[scripts/bench/rtx/devprof-emulate.sh](../scripts/bench/rtx/devprof-emulate.sh) ejecuta una orden de ruta por perfil en
una tarjeta fijada por UUID y guarda las líneas de presupuesto, plan y resultado de cada ejecución. Medido con las
opciones equivalentes de dispositivo completo en una RTX PRO 6000 (24 GiB de VRAM + 32 GiB de RAM): DeepSeek V4.1 Flash
independiente 6,55 tok/s (6,75 con DSpark), decodificación de GLM-5.3 Flash 0,67 tok/s en caliente; con ese
presupuesto la mayoría de expertos se leen del disco ([discrete-gpu.es.md](discrete-gpu.es.md)).
