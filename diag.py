import requests
import os

def test_server():
    print("Testing server on http://localhost:8001...")
    try:
        r = requests.get("http://localhost:8001/", timeout=5)
        print(f"Index status: {r.status_code}")
        
        r = requests.get("http://localhost:8001/cosmera-logo.png", timeout=5)
        print(f"Logo status: {r.status_code}, Size: {len(r.content)} bytes")
        
        # Test predict with a dummy file
        files = {'file': ('test.jpg', open('test.jpg', 'rb'), 'image/jpeg')}
        r = requests.post("http://localhost:8001/predict", files=files, timeout=10)
        print(f"Predict status: {r.status_code}, Response: {r.json()}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_server()
