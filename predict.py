import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# ----- SAME MODEL ARCHITECTURE -----
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

model.load_state_dict(torch.load("cat_dog_model.pth", map_location="cpu"))
model.eval()

# ----- IMAGE TRANSFORM -----
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# ----- PREDICT FUNCTION -----
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    classes = ["cat", "dog"]
    print("Prediction:", classes[predicted.item()])

# TEST IMAGE:
predict_image("test.jpg")   # <-- Replace with your own image path