import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

class DeepClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DeepClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc=nn.Sequential(
                nn.Linear(1024*14*14,128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(128,128),
                nn.Dropout(0.4),
                nn.Linear(128, num_classes)
        )
        
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return x

# model = DeepClassifier(4)
# dummy_input = torch.randn(1, 3, 224, 224)
# output = model.features(dummy_input)
# print(output.shape)