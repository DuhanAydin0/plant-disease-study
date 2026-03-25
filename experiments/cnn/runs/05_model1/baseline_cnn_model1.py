
# IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path



# CONFIG
DATA_DIR = Path("data/processed/model1_plant")

BATCH_SIZE = 32
EPOCHS = 15 # önce 30 epoch ile denedim. val loss ve val acc sıçramalarına göre overfitting riski olan epoch numarasından bir öncekini seçerek yeniden train ettim
#best val check point veya early stopp kullanmak yerine best epoch değerini manuel analiz ettim.
LR = 1e-3
IMAGE_SIZE = 224
RANDOM_SEED = 42
NUM_WORKERS = 4   # macOS + spawn için main() içinde kullandım aksi takdride hata verdi. YENİ BİLGİ !!

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"



torch.manual_seed(RANDOM_SEED)



# MODEL
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
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
            nn.Linear(128 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# TRAIN / VALIDATION FUNCTIONS
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / num_batches
    accuracy = correct / total

    return avg_loss, accuracy


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            num_batches += 1

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / num_batches
    accuracy = correct / total

    return avg_loss, accuracy


# MAIN
def main():

    # TRANSFORMS
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])


    # DATASETS
    train_dataset = datasets.ImageFolder(
        DATA_DIR / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        DATA_DIR / "val",
        transform=val_transform
    )

    num_classes = len(train_dataset.classes)
    print(f"Classes ({num_classes}): {train_dataset.classes}")


    # DATALOADERS
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )


    # MODEL / LOSS / OPTIMIZER
    model = SimpleCNN(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\nUsing device: {DEVICE}\n")


    # TRAINING LOOP

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion
        )

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )


    # SAVE MODEL

    save_path = Path("/Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/05_model1/model1_baseline_cnn.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": train_dataset.class_to_idx
        },
        save_path
    )

    print(f"\n Model saved to: {save_path}")



if __name__ == "__main__":
    main()
