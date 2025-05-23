# Data Dictionary: balanced_train_segments.csv

## Description
This table contains information about segments of YouTube videos, identified by their ID, along with associated time intervals and tags.

## Fields

| Field Name          | Data Type    | Description                                                                 |
|---------------------|--------------|-----------------------------------------------------------------------------|
| **YTID**            | `object`     | Unique identifier of the YouTube video.                                     |
| **start_seconds**   | `float`      | Start time of the video segment in seconds.                                 |
| **end_seconds**     | `float`      | End time of the video segment in seconds.                                   |
| **positive_labels** | `object`     | List of tags associated with the video segment, separated by commas.        |

## Example Data

| YTID         | start_seconds | end_seconds | positive_labels                     |
|--------------|---------------|-------------|-------------------------------------|
| --PJHxphWEs  | 30.000        | 40.000      | "/m/09x0r,/t/dd00088"               |
| --ZhevVpy1s  | 50.000        | 60.000      | "/m/012xff"                         |
| --aE2O5G5WE  | 0.000         | 10.000      | "/m/03fwl,/m/04rlf,/m/09x0r"        |
| --aO5cdqSAg  | 30.000        | 40.000      | "/t/dd00003,/t/dd00005"              |
| --aaILOrkII  | 200.000       | 210.000     | "/m/032s66,/m/073cg4"                |

## Notes
- **YTID**: Unique identifier for each YouTube video.
- **start_seconds**-**end_seconds**: Time interval within the video to which the tags are applied.
- **positive_labels**: The tags are in a specific format, where each tag can represent a category, object, or concept related to the video content.