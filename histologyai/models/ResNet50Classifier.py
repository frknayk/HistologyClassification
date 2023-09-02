
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Classifier, self).__init__()
        model_conv = models.resnet50(weights='IMAGENET1K_V1')
        for param in model_conv.parameters():
            param.requires_grad = False
        num_features = model_conv.fc.in_features
        model_conv.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        self.model = nn.Sequential(
            model_conv,
            nn.BatchNorm1d(4)  # Applying BatchNorm to the output of the classifier
        )
        # self.model = model_conv

    def forward(self, x):
        return self.model(x)
