import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
import os

# Define a function to create data loaders
def create_data_loaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

# Define a custom CNN model
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        # Define your custom CNN architecture
        # Example:
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 112 * 112, 512)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Forward pass of your custom CNN
        # Example:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return x

# Define a function to train and evaluate a model with given hyperparameters
def train_evaluate_model(trial, model, train_loader, val_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        trial.report(accuracy, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return accuracy

# Define the Optuna objective function
def objective(trial):
    # Define hyperparameters to search
    num_epochs = trial.suggest_int('num_epochs', 5, 20)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    
    # Create data loaders for the dataset
    batch_size = 32
    train_loader, val_loader = create_data_loaders('Histology_Dataset/Train', batch_size)
    
    # Create and train the custom CNN model
    custom_model = CustomCNN(num_classes=4)  # Adjust the number of classes
    accuracy_custom = train_evaluate_model(trial, custom_model, train_loader, val_loader, num_epochs, learning_rate)
    
    # Create and train the ResNet-18 model
    resnet_model = models.resnet18(pretrained=True)
    num_features = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_features, 4)  # Assuming 4 classes
    accuracy_resnet = train_evaluate_model(trial, resnet_model, train_loader, val_loader, num_epochs, learning_rate)
    
    return (accuracy_custom + accuracy_resnet) / 2  # Average accuracy of both models

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)  # Adjust the number of trials as needed
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
