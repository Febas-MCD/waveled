# Sound waves labeling project (WAVELED)

WaveLed is a machine learning-based project for detecting and classifying sound waves. In its initial phase, it identifies whether a sound is music or not; future stages aim to classify musical genres and detect general sound events. This project is ideal for research, accessibility tools, smart environments, and audio-based applications.

---

## 📂 Project Structure


    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.  
    │   ├── __init__.py    <- Makes src a Python module  
    │   ├── data           <- Scripts to download or generate data  
    │   │   ├── data_downloader.py  
    │   │   └── make_dataset.py  
    │   ├── features       <- Scripts to turn raw data into features for modeling  
    │   │   └── build_features.py  
    │   ├── models         <- Scripts to train models and then use trained models to make predictions  
    │   │   ├── predict_dagshub.py  
    │   │   ├── predict_model.py   
    │   │   ├── train_model.py  
    │   │   └── upload_model.py  
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations  
    │       └── visualize.py 
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

---

## 🚀 Quick Start

### Installation

1. Clone the repository: `git clone [https://github.com/FelSeb-Team/waveled.git]`
2. Create environment: `make create_environment`
3. Install the dependencies: `make requirements`
4. Create the necessary folders: `make folders`
5. Run the Makefile Data Downloader: `make download_data`
6. (Windows only): manually extract "waveled\data\raw\features.tar.gz" with a software that can handle non case-sensitive files (such as .7zip), move the resulting folders 'bal_train, eval, unbal_train' into the '/raw' path, and erase the used .tar.gz file

### Available Commands
To list all available commands in the Makefile, run: `make help`

---

## 🤖 Using Pre-trained Models

**Repository URL:** [https://dagshub.com/felytz/waveled.mlflow](https://dagshub.com/felytz/waveled.mlflow)

1. Clone the Repository: `git clone https://dagshub.com/felytz/waveled.mlflow.git`
2. Set Up the Environment: `pip install -r requirements.txt`
3. Explore Available Models: 
To browse available models and runs, explore the mlruns/ directory manually, or run: `mlflow models list -r ./mlruns`
4. Load a Model in Python:
```
import mlflow.pyfunc

# Replace with actual experiment ID and run ID
model_path = "mlruns/<experiment_id>/<run_id>/artifacts/model"
model = mlflow.pyfunc.load_model(model_path)
```

---

## 📊 Data Sources

The data used in this project comes from:

- [AudioSet](https://research.google.com/audioset/): AudioSet by Google.

---

## 🏗️ Project Status
This project is in an early development phase. Only the binary classification (music vs non-music) model is currently implemented.

---

## 🤝 Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.

---

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/FelSeb-Team/waveled/blob/ffaecf9d1b275e3cb878cafda6f2012726d9cc20/LICENSE) file for details.

---

## 🙏 Acknowledgments
Thanks to the team for their collaborative efforts in this project and to Google for providing the datasets used in our analysis.

---
