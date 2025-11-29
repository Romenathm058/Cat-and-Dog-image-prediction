import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm


# ----- SAFE DATA LOADER (SKIPS CORRUPTED IMAGES) -----

class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception:
            # Skip the corrupted image by loading next image
            return self.__getitem__((index + 1) % len(self))


# ----- DATA LOADING -----

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

train_data = SafeImageFolder("data/train", transform=transform)
val_data = SafeImageFolder("data/val", transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)


# ----- SIMPLE CNN MODEL -----

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(32 * 32 * 32, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
)

device = torch.device("cpu")
model.to(device)


# ----- TRAINING -----

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    progress = tqdm(train_loader)

    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_description(f"Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} Finished. Total Loss: {total_loss:.4f}")


# ----- VALIDATION -----

model.eval()
correct = 0
total = 0

print("\nValidating...")
with torch.no_grad():
    for images, labels in tqdm(val_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nValidation Accuracy: {100 * correct / total:.2f}%")
torch.save(model.state_dict(), "cat_dog_model.pth")
print("Model saved as cat_dog_model.pth")
