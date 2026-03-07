from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import io
import os
import httpx
from fastapi.responses import StreamingResponse, FileResponse
import json
import uvicorn


app = FastAPI(title="Lumina AI")

@app.get("/")
async def get_index():
    return FileResponse("index.html")


# CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Lumina Model Architecture =====
class LuminaCNN(nn.Module):
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

# ===== Classes =====
classes = ["asteroids", "galaxy", "nebula", "stars"]

# ===== Load Lumina Model =====
model = LuminaCNN(len(classes))
if os.path.exists("cosmera_model.pth"):
    model.load_state_dict(torch.load("cosmera_model.pth", map_location="cpu"))
model.eval()

# ===== Image Transform =====
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ===== Prediction Endpoint =====
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and transform image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    conf = confidence.item()
    label = classes[pred.item()]


    return {
        "ai": "Lumina",
        "prediction": label,
        "confidence": round(conf, 2)
    }




@app.post("/chat")
async def chat(message: dict):
    user_msg = message.get("message", "")

    prompt = f"""
You are Lumina, an astronomy learning assistant inside the Cosmera platform.

Rules:
- Only answer questions related to astronomy, space, planets, stars, galaxies, nebulae, asteroids, cosmology, or space science.
- If a question is unrelated, politely say you can only discuss astronomy topics.
- Keep explanations clear for students.
- Be friendly but concise.
- Make the response short and to the point.

User question: {user_msg}
"""

    async def generate():
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3",
                        "prompt": prompt,
                        "stream": True
                    },
                    timeout=30
                ) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if line:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
        except Exception as e:
            print(f"Ollama Error: {e}")
            yield f"I'm having trouble connecting to my brain (Ollama) right now. Error: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")

# ===== Serve Static Files =====
# This allows the HTML files and the logo to be served by FastAPI
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)