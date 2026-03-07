# Cosmera 🌌
Intelligence beyond the stars.

Cosmera is an AI-powered astronomy platform that can identify cosmic objects (Galaxies, Nebulae, Stars, Asteroids) using a custom-trained CNN and features a local AI assistant named **Lumina**.

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Azeemms77/Cosmera.git
cd Cosmera
```

### 2. Install Dependencies
Make sure you have Python installed. Then run:
```bash
pip install -r requirements.txt
```

### 3. Set Up Lumina AI (Local LLM)
The chat assistant requires **Ollama** and the `llama3` model:
1. Download and install [Ollama](https://ollama.com/).
2. Run the model in your terminal:
   ```bash
   ollama run llama3
   ```

### 4. Run the Application
Start the FastAPI server:
```bash
python app.py
```
Or use uvicorn directly:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

### 5. Access the Platform
Open your browser and go to:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

## 🛠 Features
- **Cosmic Object Identification**: Upload a space image to classify it.
- **Lumina Assistant**: Streamed chat interface for astronomy questions.
- **Interactive UI**: Modern, glassmorphic design for exploring the cosmos.

---
Developed by **Azeem**
