import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

torch.manual_seed(17)

class ImageClassificationDataset:
    def __init__(self, config):
        self.config = config.data
        self._create_transforms()

    def _create_transforms(self):
        data_transforms = {
            "train": transforms.Compose([
                transforms.RandomHorizontalFlip(p=self.config.transform.RandomHorizontalFlip),
                transforms.RandomVerticalFlip(p=self.config.transform.RandomVerticalFlip),
                transforms.RandomRotation(degrees=self.config.transform.RandomRotation),
                transforms.TenCrop(self.config.transform.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.transform.normalization_params.train.mean,
                                     std=self.config.transform.normalization_params.train.std),
            ]),
            "validation": transforms.Compose([
                transforms.Resize((self.config.transform.TenCrop,self.config.transform.TenCrop)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.transform.normalization_params.train.mean,
                                     std=self.config.transform.normalization_params.train.std),
            ]),
            "test": transforms.Compose([
                transforms.Resize((self.config.transform.resize,self.config.transform.resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.transform.normalization_params.train.mean,
                                     std=self.config.transform.normalization_params.train.std),
            ]),
        }
        self.data_transforms = data_transforms


    def create_data_loaders(self, batch_size_train, batch_size_eval):
        # Load entire dataset using ImageFolder
        full_dataset = ImageFolder(root=self.config.path_train_dataset)
        # Split dataset into train, validation, and test sets
        train_ratio = self.config.split_ratio.train_ratio
        val_ratio = self.config.split_ratio.val_ratio
        # Split dataset into train, validation, and test sets
        train_size = int(train_ratio * len(full_dataset))
        val_size = int(val_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        # Create dataset with seed for deterministic training
        generator = torch.Generator().manual_seed(self.config.random_seed)
        train_dataset, val_dataset, test_dataset = random_split(full_dataset,
            [train_size, val_size, test_size], generator=generator)
        # Apply transformations to datasets
        self._create_transforms()
        train_dataset.dataset.transform = self.data_transforms["train"]
        val_dataset.dataset.transform = self.data_transforms["validation"]
        test_dataset.dataset.transform = self.data_transforms["test"]
        # Create data loaders for train, validation, and test datasets
        train_loader = DataLoader(train_dataset, 
            batch_size=batch_size_train, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_eval)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_eval)
        return train_loader, val_loader, test_loader, full_dataset.classes
        
