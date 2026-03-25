import torch
import torch.nn as nn
from torchvision import models


# =========================================================
# Model1 + Model2 backend için
# =========================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, image_size: int = 224):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (image_size // 8) * (image_size // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================================================
# Global CNN (03_all_dataset) için
# =========================================================
class CNN_03_All_Dataset(nn.Module):
    """
    CNN for all dataset.

    Changes in 03:
    - Convolutional Dropout REMOVED
    - Classifier Dropout preserved
    - Architecture depth and width preserved
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # Block 1: (3, 224, 224) -> (16, 112, 112)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Block 2: -> (32, 56, 56)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Block 3: -> (64, 28, 28)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Classifier (unchanged)
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


# =========================================================
# Transfer Learning / Feature Extractor (bonus)
# =========================================================
def build_resnet18_feature_extractor() -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    m = models.resnet18(weights=weights)
    m.fc = nn.Identity()
    m.eval()
    return m
