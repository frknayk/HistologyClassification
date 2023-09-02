
# Usage Guide for Classification Task

This guide provides step-by-step instructions on how to use the classification project template for developing vision-based AI models in PyTorch. The project includes organized directory structures, MLflow integration, and seamless model architecture selection.

## Prerequisites

Conda (Miniconda or Anaconda) installed.

## Configuration

Open the `configs/`` directory and update the dataset configuration YAML files according to your dataset.

Update the `model`` section in each dataset configuration YAML file to specify the desired model architecture. For example:

```yaml
model:
  architecture: resnet18
```

## Training

1. Train the model using the specified dataset configuration:

```bash
python histologyai/train_classifier.py
```

2. MLflow will automatically track the experiment, log parameters, metrics, and artifacts.


## Experiment Tracking

0. Create Weights&Biases account if you already not. 
If you already have an account, please login from a terminal/console by following command:
```bash
    wandb login
```

1. Explore W&B UI to track and compare experiments
```bash
    https://wandb.ai/YourWandbUserName/Histology
```


## Additional Tasks

1. For inference, run:

```bash
python scripts/infer.py
```

## Customization

Modify the model architecture files in the `models/ `directory to add new model architectures.
Adjust the training, evaluation, and inference scripts to suit your specific requirements.

