import requests

r = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": "Say hello in one sentence",
        "stream": False
    }
)

print(r.json())