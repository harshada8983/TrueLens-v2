"""Usage:  python test_api.py <BACKEND_URL> <image_or_video_file>
Example: python test_api.py https://harshada8983-truelens-api.hf.space photo.jpg
(pip install requests)"""
import sys
import requests

if len(sys.argv) != 3:
    sys.exit(__doc__)
base, path = sys.argv[1].rstrip("/"), sys.argv[2]
is_video = path.lower().endswith((".mp4", ".mov", ".webm", ".avi"))
route = "/predict-video" if is_video else "/predict"
with open(path, "rb") as f:
    r = requests.post(base + route, files={"file": f}, timeout=300)
print("HTTP", r.status_code)
print(r.json())
