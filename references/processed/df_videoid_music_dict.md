# Diccionario de Datos

## Descripción
Esta tabla contiene información sobre segmentos de videos de YouTube, identificados por su ID único (`YTID`), junto con los tiempos de inicio y fin de cada segmento. Fue extraido del directorio bal_train, el cual contiene archivos tipo tfrecords.

## Campos

| Nombre del Campo    | Tipo de Dato | Descripción                                                                 |
|---------------------|--------------|-----------------------------------------------------------------------------|
| **YTID**            | `object`     | Identificador único del video de YouTube.                                   |
| **start_seconds**   | `float`      | Tiempo de inicio del segmento del video, en segundos.                       |
| **end_seconds**     | `float`      | Tiempo de finalización del segmento del video, en segundos.                 |

## Ejemplo de Datos

| YTID         | start_seconds | end_seconds |
|--------------|---------------|-------------|
| --aE2O5G5WE  | 0.0           | 10.0        |
| -0mG4W5Hlq8  | 270.0         | 280.0       |
| -0SdAVK79lg  | 30.0          | 40.0        |
| -1TLtjPtnms  | 10.0          | 20.0        |
| -5xOcMJpTUk  | 70.0          | 80.0        |
| ...          | ...           | ...         |
| _vwBe9ZXWXE  | 10.0          | 20.0        |
| _WD9mbwAcrQ  | 130.0         | 140.0       |
| _WS68gpLg7U  | 180.0         | 190.0       |
| _zQTlTCqMzs  | 130.0         | 140.0      