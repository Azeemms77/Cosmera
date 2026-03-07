import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn

# ===== SAME MODEL ARCHITECTURE AS TRAINING =====
class CosmicCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# ===== LOAD CLASSES (same order as training) =====
classes = ["asteroids", "galaxy", "nebula", "stars"]


# ===== LOAD MODEL =====
model = CosmicCNN(len(classes))
model.load_state_dict(torch.load("cosmera_model.pth"))
model.eval()


# ===== IMAGE TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# ===== LOAD TEST IMAGE =====
img = Image.open("test.jpg").convert("RGB")
img = transform(img).unsqueeze(0)


# ===== PREDICT =====
with torch.no_grad():
    outputs = model(img)
    _, pred = torch.max(outputs, 1)

print("Prediction:", classes[pred.item()])