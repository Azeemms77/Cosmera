# Cosmera 🌌
Intelligence beyond the stars.

Cosmera is an immersive, AI-powered astronomy platform. It can classify cosmic objects (Galaxies, Nebulae, Stars, Asteroids) using a custom-trained PyTorch Convolutional Neural Network (CNN). It also features **Lumina**, a highly intelligent, context-aware AI astronomy assistant powered by the **NVIDIA NIM API** (Llama 3).

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Azeemms77/Cosmera.git
cd Cosmera
```

### 2. Install Dependencies
Make sure you have Python installed. We recommend setting up a virtual environment (`.venv`), then run:
```bash
pip install -r requirements.txt
```
*(Note for cloud deployments: The requirements are optimized for CPU environments using `torch==...cpu` to save limits).*

### 3. Set Up Lumina AI (NVIDIA API)
Cosmera's assistant uses the cutting-edge NVIDIA NIM API for fast, intelligent reasoning instead of relying on heavy local hardware constraints.
1. Create a `.env` file in the root directory.
2. Add your NVIDIA API key inside the file:
   ```env
   NVIDIA_API_KEY=your_api_key_here
   ```

### 4. Run the Application
Start the FastAPI server:
```bash
python app.py
```
Or use uvicorn directly:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Access the Platform
Open your browser and navigate to:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

## 🌟 Intelligent Features
- **Cosmic Object Identification**: Upload a space image to instantly classify it via our custom PyTorch model.
- **Dynamic AI Explanations**: Once an object is identified, Lumina automatically analyzes it and generates a precise scientific explanation and formation summary.
- **Context-Aware Memory**: The Lumina Assistant doesn't just chat—it remembers! You can ask follow-up questions about objects you just scanned.
- **Advanced UI Features**: Enjoy realistic AI typing animations, feedback rating widgets, and a glassmorphic design that reacts to your interaction.
- **Cross-Analysis Comparison**: Upload two cosmic data signatures side-by-side to trigger an AI-driven scientific comparison (works best with astronomical images).

---
Developed by **Azeem**
