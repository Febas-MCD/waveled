# Sound waves labeling project (WAVELED)

## Objective

WaveLed is a machine learning-based project for detecting and classifying sound waves. In its initial phase, it identifies whether a sound is music or not; future stages aim to classify musical genres and detect general sound events. This project is ideal for research, accessibility tools, smart environments, and audio-based applications.

## Project Organization

------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## Usage Instructions

1. Clone the repository: `git clone [https://github.com/FelSeb-Team/waveled.git]`
2. Create environment: `make create_environment`
3. Install the dependencies: `make requirements`
4. Create the necessary folders: `make folders`
5. Run the Makefile Data Downloader: `make download_data`
6. (Windows only): manually extract "waveled\data\raw\features.tar.gz" with a software that can handle non case-sensitive files (such as .7zip), move the resulting folders 'bal_train, eval, unbal_train' into the '/raw' path, and erase the used .tar.gz file

### Available Commands
To list all available commands in the Makefile, run: `make help`

## Data Sources

The data used in this project comes from:

- [AudioSet](https://research.google.com/audioset/): AudioSet by Google.

--------

## Project Status
🚧 This project is in an early development phase. Only the binary classification (music vs non-music) model is currently implemented.

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request with your changes

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/FelSeb-Team/waveled/blob/ffaecf9d1b275e3cb878cafda6f2012726d9cc20/LICENSE) file for details.

## Acknowledgments
Thanks to the team for their collaborative efforts in this project and to Google for providing the datasets used in our analysis.

--------
