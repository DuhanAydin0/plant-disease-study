import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Simple CNN baseline for tomato leaf disease classification.

    Architecture:
    - 3 convolutional blocks: Conv2d -> ReLU -> MaxPool
    - Fully connected classifier

    Note:
    - No softmax is used here.
    - CrossEntropyLoss will handle logits internally.
    """

    def __init__(self, num_classes: int):
        super(BaselineCNN, self).__init__()

        # Block 1: input (3, 224, 224)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # -> (16, 112, 112)
        )

        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # -> (32, 56, 56)
        )

        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # -> (64, 28, 28)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x
