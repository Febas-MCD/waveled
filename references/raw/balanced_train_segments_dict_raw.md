# Diccionario de Datos: balanced_train_segments.csv RAW

## Descripción
Esta tabla contiene información sobre segmentos de videos de YouTube, identificados por su ID, junto con los intervalos de tiempo y las etiquetas asociadas.

## Campos

| Nombre del Campo    | Tipo de Dato | Descripción                                                                 |
|---------------------|--------------|-----------------------------------------------------------------------------|
| **YTID**            | `object`     | Identificador único del video de YouTube.                                   |
| **start_seconds**   | `object`     | Tiempo de inicio del segmento del video en segundos.                        |
| **end_seconds**     | `object`     | Tiempo de finalización del segmento del video en segundos.                  |
| **positive_labels** | `object`     | Lista de etiquetas asociadas al segmento del video, separadas por comas.    |

## Ejemplo de Datos

| YTID         | start_seconds | end_seconds | positive_labels                     |
|--------------|---------------|-------------|-------------------------------------|
| --PJHxphWEs  | 30.000        | 40.000      | "/m/09x0r,/t/dd00088"               |
| --ZhevVpy1s  | 50.000        | 60.000      | "/m/012xff"                         |
| --aE2O5G5WE  | 0.000         | 10.000      | "/m/03fwl,/m/04rlf,/m/09x0r"        |
| --aO5cdqSAg  | 30.000        | 40.000      | "/t/dd00003,/t/dd00005"              |
| --aaILOrkII  | 200.000       | 210.000     | "/m/032s66,/m/073cg4"                |

## Notas
- **YTID**: Es un identificador único para cada video de YouTube.
- **start_seconds** y **end_seconds**: Definen el intervalo de tiempo dentro del video al que se aplican las etiquetas.
- **positive_labels**: Las etiquetas están en un formato específico, donde cada etiqueta puede representar una categoría, objeto, o concepto relacionado con el contenido del video.