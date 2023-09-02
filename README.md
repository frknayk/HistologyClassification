# Classification Task using PyTorch and MLflow

This repository contains a structured project template for developing vision-based AI models in PyTorch. The focus is on classification, and it includes tools for ensuring reproducibility, CI/CD pipelines, and easy model development.

## Features

- Organized directory structure for datasets, models, scripts, and more.
- Configuration using YAML files for datasets and hyperparameters.
- MLflow integration for experiment tracking.
- Seamless model architecture selection based on YAML configuration.
- Conda environment setup for consistent dependencies.

## Prerequisites

- Conda (Miniconda or Anaconda) installed.

## Setup

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd classification
    ```

2. Create a Conda environment:
```bash
    conda env create -f environment.yaml
    conda activate classification-env
```
3. Update dataset and model configuration files in the dataset_configs/ directory.

4. Train the model
```bash
    python scripts/train.py
```

5. Explore MLflow UI to track and compare experiments
```bash
    mlflow ui
```

## Directory Structure

- data/: Dataset files or links to datasets.
- models/: Model architectures and related utilities.
- scripts/: Training, evaluation, and inference scripts.
- dataset_configs/: Configuration YAML files for datasets.
- environment.yaml: Conda environment specification.
- requirements.txt: Additional Python package requirements.
- README.md: Project overview and setup instructions.

## Experiment Tracking
MLflow is integrated to track experiments, parameters, metrics, and artifacts.
Experiment information can be visualized using the MLflow UI.

## Usage Guide
For detailed instructions on how to use the repository, dataset configuration, and hyperparameter tuning, refer to the usage guide.