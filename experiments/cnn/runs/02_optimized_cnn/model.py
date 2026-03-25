import torch
import torch.nn as nn


class CNN_02_Optimized(nn.Module):
    """
    Optimized CNN for tomato leaf disease classification.

    Changes compared to baseline:
    - Batch Normalization after each Conv2d
    - Dropout added to convolutional blocks and classifier
    - Architecture depth preserved
    """

    def __init__(self, num_classes: int):
        super(CNN_02_Optimized, self).__init__()

        # Block 1: (3, 224, 224) -> (16, 112, 112)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25)
        )

        # Block 2: -> (32, 56, 56)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25)
        )

        # Block 3: -> (64, 28, 28)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x
