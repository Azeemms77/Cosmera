import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Image settings
IMG_SIZE = 256
BATCH_SIZE = 16 # Reduced batch size to save GPU/CPU memory at 256x256
EPOCHS = 30 # Increased epochs for better learning

# Transformations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Added data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard normalization
])

# Load dataset
dataset = datasets.ImageFolder("dataset", transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Class names
classes = dataset.classes
print("Classes identified from dataset folder:", classes)

# Model Architecture (LuminaCNN)
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
            nn.Linear(64 * 32 * 32, 512), # Increased size and capacity for HD
            nn.ReLU(),
            nn.Dropout(0.5), # Added dropout for better generalization
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = LuminaCNN(len(classes))

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print(f"Starting training on {len(dataset)} images across {len(classes)} classes...")

# Attempt to load existing weights (if architecture matches)
if os.path.exists("cosmera_model.pth"):
    try:
        model.load_state_dict(torch.load("cosmera_model.pth", map_location="cpu"))
        print("Loaded existing model weights to continue training.")
    except:
        print("Existing model weights not compatible (likely due to class change). Starting fresh.")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(preds.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Save after every epoch so progress is never lost
    torch.save(model.state_dict(), "cosmera_model.pth")
    print(f"Progress saved to cosmera_model.pth after Epoch {epoch+1}")

# Also save class names for reference
with open("classes.json", "w") as f:
    import json
    json.dump(classes, f)

print("Training session finalized.")
