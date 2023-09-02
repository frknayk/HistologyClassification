from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch
from sklearn.model_selection import train_test_split

torch.manual_seed(17)

# Define data transformations including augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.RandomRotation(degrees=(0, 180)),
    transforms.TenCrop(224*2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
val_transform = transforms.Compose([
    transforms.Resize((224*2, 224*2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_transform = transforms.Compose([
    transforms.Resize((224*2, 224*2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def create_data_loaders(
        path_dataset='data/Histology_Dataset/Train/',
        batch_size_train=32,
        batch_size_eval=32,
        use_test=True):
    if use_test:
        return create_data_loaders_with_test(path_dataset, batch_size_train, batch_size_eval)
    return create_data_loaders_without_test(path_dataset, batch_size_train, batch_size_eval)


def create_data_loaders_with_test(
        path_dataset='data/Histology_Dataset/Train/',
        batch_size_train=32,
        batch_size_eval=32):
    # Load entire dataset using ImageFolder
    full_dataset = ImageFolder(root=path_dataset)
    # Split dataset into train, validation, and test sets
    train_ratio = 0.8
    val_ratio = 0.1
    # Split dataset into train, validation, and test sets
    train_size = int(train_ratio * len(full_dataset))
    val_size = int(val_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    # Create dataset with seed for deterministic training
    generator1 = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset,
        [train_size, val_size, test_size], generator=generator1)
    # Apply transformations to datasets
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = test_transform
    # Create data loaders for train, validation, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_eval)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval)
    return train_loader, val_loader, test_loader, full_dataset.classes

def create_data_loaders_without_test(
        path_dataset='data/Histology_Dataset/Train/',
        batch_size_train=32,
        batch_size_eval=32):
    # Load entire dataset using ImageFolder
    full_dataset = ImageFolder(root=path_dataset)
    # Split dataset into train, validation, and test sets
    train_size = int(0.9 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    generator1 = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, 
        [train_size, val_size], generator=generator1)
    # Apply transformations to datasets
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    # Create data loaders for train, validation, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_eval)
    return train_loader,val_loader, None, full_dataset.classes