# Diccionario de Datos: ontology.json

## Descripción
Esta tabla contiene información sobre categorías, incluyendo su nombre, descripción, ejemplos positivos, relaciones con otras categorías y restricciones.

## Campos

| Nombre del Campo    | Tipo de Dato | Descripción                                                                 |
|---------------------|--------------|-----------------------------------------------------------------------------|
| **id**              | `object`     | Identificador único de la categoría.                                        |
| **name**            | `object`     | Nombre de la categoría.                                                     |
| **description**     | `object`     | Descripción detallada de la categoría.                                      |
| **citation_uri**    | `object`     | Enlace a una fuente de referencia (por ejemplo, Wikipedia).                 |
| **positive_examples**| `object`    | Lista de ejemplos positivos (enlaces a videos de YouTube con marcas de tiempo). |
| **child_ids**       | `object`     | Lista de identificadores de categorías hijas relacionadas.                  |
| **restrictions**    | `object`     | Lista de restricciones aplicables a la categoría (por ejemplo, "blacklist" o "abstract"). |


## Notas
- **id**: Identificador único de la categoría.
- **name**: Nombre de la categoría.
- **description**: Breve descripción del significado o alcance de la categoría.
- **citation_uri**: Enlace a una fuente externa que proporciona más información sobre la categoría.
- **positive_examples**: Lista de ejemplos concretos (enlaces a videos de YouTube con marcas de tiempo) que ilustran la categoría.
- **child_ids**: Lista de identificadores de categorías hijas, que representan subcategorías o relaciones jerárquicas.
- **restrictions**: Lista de restricciones aplicables a la categoría, como "blacklist" (lista negra) o "abstract" (categoría abstracta).