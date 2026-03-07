import requests
try:
    r = requests.get("http://127.0.0.1:8001/", timeout=2)
    print(f"Status: {r.status_code}")
    print(f"Content length: {len(r.text)}")
except Exception as e:
    print(f"Error: {e}")
