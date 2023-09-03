import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import os
import yaml

from histologyai.utils import create_module

config_folder = os.path.join(os.path.dirname(__file__), 'configs')
# Load the configuration file
config_file_path = os.path.join(config_folder, 'hyperparameter_tuning.yaml')
with open(config_file_path, 'r') as config_file:
    hyperparameters = yaml.safe_load(config_file)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def objective(trial):
    # Generate the model.
    config = DefaultMunch.fromDict(hyperparameters)
    class_name = trial.suggest_categorical("architecture", config.model.architectures)
    model = create_module(class_name, config.data.num_classes)
    model.to(device)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Choose the batch size
    batch_size = trial.suggest_float("batch_size", 1e-5, 1e-1, log=True)

    # Get the FashionMNIST dataset.
    dataset = ImageClassificationDataset(self.config)
    train_loader, val_loader, test_loader, classes = dataset.create_data_loaders(
                batch_size_train=self.config.training.batch_size,
                batch_size_eval=self.config.evaluation.batch_size)

    train_loader, valid_loader = get_mnist()

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

# Define the objective function for hyperparameter optimization
def objective(trial):
    # Sample hyperparameters from the search space defined in the YAML file
    sampled_hyperparameters = {}
    for param, config in hyperparameters.items():
        if 'choice' in config:
            if config['type'] == 'float':
                sampled_hyperparameters[param] = trial.suggest_float(param, *config['choice'])
            elif config['type'] == 'int':
                sampled_hyperparameters[param] = trial.suggest_int(param, *config['choice'])
            else:
                sampled_hyperparameters[param] = trial.suggest_categorical(param, config['choice'])

    # Sample a model architecture to use
    model = create_module(sampled_hyperparameters["architecture"],sampled_hyperparameters["data"]["num_classes"])
    model.to(device)

    # Train and evaluate the model using the sampled hyperparameters
    accuracy = train_and_evaluate(model, sampled_hyperparameters)  # Replace with your training function

    return -accuracy  # Optimize for accuracy, so we negate it


# Create an Optuna study
study = optuna.create_study(direction='maximize')

# Start the optimization
study.optimize(objective, n_trials=100)  # You can adjust the number of trials

# Print the best hyperparameters and result
best_params = study.best_params
best_result = study.best_value
print(f"Best Hyperparameters: {best_params}")
print(f"Best Result (Accuracy): {best_result}")