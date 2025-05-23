.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3 create_environment test_environment help

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = waveled
PYTHON_INTERPRETER = python

# Detect OS
ifeq ($(OS),Windows_NT)
	SYSTEM = Windows
	CONDA_DETECT_CMD = where conda
	VENV_ACTIVATE_CMD = .\\venv\\Scripts\\activate
else
	SYSTEM = Unix
	CONDA_DETECT_CMD = which conda
	VENV_ACTIVATE_CMD = source venv/bin/activate
endif

# Detect Conda
ifeq (,$(shell $(CONDA_DETECT_CMD)))
	HAS_CONDA = False
else
	HAS_CONDA = True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Download raw data
download_data:
	@echo "Downloading raw data from the web..."
	$(PYTHON_INTERPRETER) src/data/data_downloader.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3 -y
else
	conda create --name $(PROJECT_NAME) python=2.7 -y
endif
ifeq ($(SYSTEM),Windows)
	@echo ">>> New conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
else
	@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
endif
else
	@echo ">>> Creating virtual environment using $(PYTHON_INTERPRETER)."
	$(PYTHON_INTERPRETER) -m venv venv
	@echo ">>> New virtualenv created. Activate with:\n$(VENV_ACTIVATE_CMD)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

## Train model locally
local_model:
    python src/models/train_model.py

upload_model:
    python src/models/upload_model.py

local_pred:
    python src/models/predict_model.py

dh_pred:
    python src/models/predict_dagshub.py