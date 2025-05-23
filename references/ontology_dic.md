# Data Dictionary: ontology.json

## Description
This table contains information about categories, including their name, description, positive examples, relationships with other categories and restrictions.

## Fields

| Field Name           | Data Type   | Description                                                                 |
|----------------------|-------------|-----------------------------------------------------------------------------|
| **id**               | `object`    | Unique identifier of the category.                                          |
| **name**             | `object`    | Category name.                                                              |
| **description**      | `object`    | Detailed description of the category.                                       |
| **citation_uri**     | `object`    | Link to a reference source (e.g. Wikipedia).                                |
| **positive_examples**| `object`    | List of positive examples (links to YouTube videos with timestamps).        |
| **child_ids**        | `object`    | List of identifiers of related child categories.                            |
| **restrictions**     | `object`    | List of restrictions applicable to the category (e.g., “blacklist”).        |


## Notes
- **id**: Unique identifier of the category.
- **name**: Category name.
- **description**: Brief description of the meaning or scope of the category.
- **citation_uri**: Link to an external source that provides more information about the category.
- **positive_examples**: List of concrete examples (links to timestamped YouTube videos) illustrating the category.
- **child_ids**: List of child category identifiers, representing subcategories or hierarchical relationships.
- **restrictions**: List of restrictions applicable to the category, such as “blacklist” or “abstract”.