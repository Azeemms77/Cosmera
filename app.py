from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import torch.nn as nn

import io
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse, FileResponse
import json
import uvicorn
import httpx

load_dotenv()
client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)


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

# ===== Classes =====
classes = ["asteroids", "galaxy", "nebula", "stars", "unidentified_objects"]

# ===== Load Lumina Model =====
model = LuminaCNN(len(classes))
if os.path.exists("cosmera_model.pth"):
    model.load_state_dict(torch.load("cosmera_model.pth", map_location="cpu"))
model.eval()



# ===== Prediction Endpoint =====
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    from torchvision import transforms  # lazy import
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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
    history = message.get("history", [])

    sys_prompt = """
You are Lumina, a highly intelligent, calm, and scientific AI astronomy assistant inside the Cosmera platform.
If the user asks for a simple explanation, explain like a teacher. If they want advanced concepts, be highly scientific.
Always be contextual. Remember previous interactions in the conversation.
Rules:
- Only answer questions related to astronomy, space, planets, stars, galaxies, etc.
- If a question is unrelated, politely refuse.
- Be concise but highly informative and realistic.
"""
    messages = [{"role": "system", "content": sys_prompt}]
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_msg})

    async def generate():
        try:
            response = await client.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
                stream=True,
            )
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"NVIDIA API Error: {e}")
            yield f"I'm having trouble connecting to my central neural network right now. Error: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/apod")
async def get_apod():
    async with httpx.AsyncClient() as c:
        res = await c.get("https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY")
        return res.json()

# ===== Serve Static Files =====
# This allows the HTML files and the logo to be served by FastAPI
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)