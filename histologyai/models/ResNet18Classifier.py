
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        # Add Batch Normalization
        self.model = nn.Sequential(
            self.resnet,
            nn.BatchNorm1d(4)  # Applying BatchNorm to the output of the classifier
        )

    def forward(self, x):
        return self.model(x)
