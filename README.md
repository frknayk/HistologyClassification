# Histology Image Classification

## Overview

The Histology Data Classification Project is a machine learning project designed to automatically classify histology images into different categories based on their underlying tissue characteristics.
Histology, a branch of biology, involves the microscopic examination of tissue samples to study their cellular structures and diagnose diseases.
In this project, we leverage the power of machine learning to assist in the categorization of histological regions into specific classes.

## Project Goals

The primary goals of this project are as follows:

```text
1. Automated Histology Classification: Develop a machine learning model capable of classifying histology images into distinct categories or classes. 
These classes often represent different tissue types, including Benign, In Situ, Invasive, and Normal histological regions.

2. Enhanced Medical Diagnosis: Enable accurate and efficient diagnosis by automating the classification of histological samples. 
This can assist pathologists and medical professionals in identifying abnormal or cancerous tissues.

3. Data Augmentation and Preprocessing: Implement data augmentation techniques and preprocessing methods to improve model generalization and robustness. 
This involves techniques like random rotations, flips, and resizing of images.

4. Transfer Learning: Utilize transfer learning with pre-trained deep learning models, such as ResNet, to leverage their feature extraction capabilities and fine-tune them for histology classification.

5. Performance Metrics: Employ various performance metrics, including accuracy, precision, recall, F1 score, ROC curves, and precision-recall curves, to evaluate and measure the model's effectiveness.

6. Experiment Tracking: Monitor and visualize model training and performance using tools like Weights and Biases (wandb) to gain insights into the training process and make informed decisions.
```

## Installation

`Prerequisites`: `Conda` (Miniconda or Anaconda) installed.

Follow these steps to set up your project environment using Conda:

1. Clone this repository:

```bash
git clone https://github.com/frknayk/HistologyClassification.git
cd HistologyClassification
```

2. Install project as python-package, necessary for path management

```bash
pip install -e .
```

3. Create a Conda environment and install the required dependencies:

```bash
conda env create -f environment.yml
conda activate env_histology
```

## Weights and Biases (wandb) Setup

1. Install wandb:

```bash
pip install wandb
```

2. Log in to wandb:

```bash
wandb login
```

Follow the prompts to authorize wandb.

## Simple Usage Guide

1. Prepare your dataset and organize it as needed.

2. Modify the configuration parameters in config.yaml to customize your experiment settings.

3. Run the training script:

```bash
python train.py --config config.yaml
```

4. Monitor and visualize your experiment in the wandb dashboard:

```text
Access the wandb dashboard at https://wandb.ai/.
Explore training metrics, logs, and visualizations in real-time.
```

## Training Guide

1. Data Preparation:

- Organize your dataset into appropriate directories.
- Implement data augmentation and transformation if needed in `dataset_loader.py`

2. Configuration:

- Modify the parameters in config.yaml to specify your model, data paths, batch size, and other hyperparameters.

3. Training:

- Use the `train_classifier.py` script to start training:

```bash
python histologyai/train_classifier.py --config config.yaml
```

- Running all experiment configs together: Put all configuration files under `configs/` folder then call `run_all_experiments()` function from `train_classifier.py` script.

```bash
python histologyai/train_classifier.py --run_all True
```

- Or you can directly run the training script in the root path after modifying the config files inside the script:

```bash
python train_all.py
```

4. Explore W&B UI to track and compare experiments

```bash
https://wandb.ai/YourWandbUserName/Histology
```

5. Evaluation:

- Evaluate your trained model on a separate test dataset using the `evaluate.py` script.

## Dataset

The project relies on a dataset containing histology images. These images are organized into four main classes:

```text
- Benign: Histological regions categorized as non-cancerous or benign tissues.
- In Situ: Represents tissues where malignant cells are present as a tumor but have not metastasized, or invaded beyond the layer or tissue type where it arose
- Invasive: Histological regions where abnormal cells have spread to nearby tissues.
- Normal: Normal, healthy tissue samples used for reference and comparison.
```

The dataset is divided into `training`, `validation`, and `test` sets to facilitate model training and evaluation.
Each class is represented by a dedicated folder containing images of the corresponding histological regions.

## Impact

The Histology Data Classification Project has the potential to make a significant impact in the field of medical diagnosis and histopathology.

By automating the classification of histological images, it can:

```text
- Assist pathologists in making faster and more accurate diagnoses.
- Identify abnormal or cancerous tissues at an earlier stage.
- Improve the overall efficiency of histological analysis.
- The project's outcomes may contribute to advancements in medical research and healthcare, ultimately benefiting patients and medical professionals
```

## Features

- Organized directory structure for datasets, models, scripts, and more.
- Configuration using YAML files for datasets and hyperparameters.
- W&B integration for experiment tracking.
- Seamless model architecture selection based on YAML configuration.
- Conda environment setup for consistent dependencies.

## Directory Structure

- data/: Dataset files or links to datasets.
- models/: Model architectures and related utilities.
- configs/: Configuration YAML files for datasets.
- logs/ : Consists wandb/ and checkpoints/ folder belonging to experiment runs.
- environment.yaml: Conda environment specification.
- requirements.txt: Additional Python package requirements.
- README.md: Project overview and setup instructions.


## TODO-List

- Add different normalization techniques for histology image classification
