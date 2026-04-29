import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import os

# ===== SAME MODEL ARCHITECTURE AS TRAINING =====
class LuminaCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# ===== LOAD CLASSES =====
classes = ["asteroids", "galaxy", "nebula", "stars", "unidentified_objects"]


# ===== LOAD MODEL =====
model = LuminaCNN(len(classes))
if os.path.exists("cosmera_model.pth"):
    model.load_state_dict(torch.load("cosmera_model.pth", map_location="cpu"))
model.eval()


# ===== IMAGE TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ===== LOAD TEST IMAGE =====
if os.path.exists("test.jpg"):
    img = Image.open("test.jpg").convert("RGB")
    img_tensor = transform(img).unsqueeze(0)


    # ===== PREDICT =====
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    print(f"Prediction: {classes[pred.item()]} ({confidence.item()*100:.2f}%)")
else:
    print("test.jpg not found.")