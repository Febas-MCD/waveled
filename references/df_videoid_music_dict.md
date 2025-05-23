# Data Dictionary

## Description
This table contains information about segments of YouTube videos, identified by their unique ID (`YTID`), along with the start and end times of each segment. It was extracted from the bal_train directory, which contains tfrecords files.

## Fields

| Field Name          | Data Type    | Description                                                                 |
|---------------------|--------------|-----------------------------------------------------------------------------|
| **YTID**            | `object`     | Unique identifier of the YouTube video.                                     |
| **start_seconds**   | `float`      | Start time of the video segment, in seconds.                                |
| **end_seconds**     | `float`      | End time of the video segment, in seconds.                                  |

## Example Data

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