"""TrueLens API - image + video deepfake detection.

Routes (frontend ke purane routes jaise hi):
  GET  /                health check
  POST /predict         image
  POST /predict-video   video

Score = P(manipulated). Face na mile to random answer nahi, saaf error (422).
"""
import json
import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

log = logging.getLogger("truelens")
MODEL_DIR = Path(__file__).parent / "model"
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", 10))
MAX_VIDEO_MB = int(os.getenv("MAX_VIDEO_MB", 50))
# Production me chaho to: ALLOWED_ORIGINS=https://true-lens-seven.vercel.app
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

state, lock = {}, threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    import keras
    from mtcnn import MTCNN

    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    state["cfg"] = cfg
    state["model"] = keras.saving.load_model(str(MODEL_DIR / cfg["model_file"]), compile=False)
    state["detector"] = MTCNN()
    state["model"].predict(np.zeros((1, 224, 224, 3), "float32"), verbose=0)  # warm-up
    log.info("Model ready, threshold=%s", cfg["threshold"])
    yield


app = FastAPI(title="TrueLens Deepfake Detector API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])


# ---------- pipeline (training script ke crop_face jaisa hi) ----------
def crop_face(bgr, margin=0.25, min_conf=0.95, max_side=640):
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(bgr, (int(w * s), int(h * s))) if s < 1 else bgr
    try:
        with lock:
            faces = state["detector"].detect_faces(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    except (ValueError, tf.errors.InvalidArgumentError):
        return None  # mtcnn 1.0.0 bug: no-face image par [] ki jagah error deta hai
    if not faces:
        return None
    f = max(faces, key=lambda d: d["box"][2] * d["box"][3])
    if f["confidence"] < min_conf:
        return None
    x, y, bw, bh = [v / s for v in f["box"]]
    cx, cy, side = x + bw / 2, y + bh / 2, max(bw, bh) * (1 + 2 * margin)
    x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    x1, y1 = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
    if x1 - x0 < 40 or y1 - y0 < 40:
        return None
    face = cv2.resize(bgr[y0:y1, x0:x1], (224, 224))
    return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)


def score_faces(faces):
    with lock:
        p = state["model"].predict(np.stack(faces).astype("float32"), batch_size=16, verbose=0)
    return p.ravel()


def verdict(score):
    t, m = state["cfg"]["threshold"], state["cfg"]["uncertain_margin"]
    if score >= t + m:
        return "MANIPULATED"
    if score <= t - m:
        return "REAL"
    return "UNCERTAIN"


def confidence_pct(score, v):
    c = score if v == "MANIPULATED" else 1 - score if v == "REAL" else 1 - abs(score - 0.5) * 2
    return round(float(c) * 100, 2)


def read_limited(upload: UploadFile, max_mb: int) -> bytes:
    data = upload.file.read(max_mb * 1024 * 1024 + 1)
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {max_mb}MB limit.")
    if not data:
        raise HTTPException(400, "Empty file.")
    return data


# ---------- routes ----------
@app.get("/")
def health_check():
    return {"status": "TrueLens API is online. Engine Ready." if "model" in state else "loading"}


@app.post("/predict")
def analyze_image(file: UploadFile = File(...)):
    img = cv2.imdecode(np.frombuffer(read_limited(file, MAX_IMAGE_MB), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image file.")
    face = crop_face(img)
    if face is None:
        raise HTTPException(422, "No clear face detected in the image. Upload a photo with a visible face.")
    score = float(score_faces([face])[0])
    v = verdict(score)
    return {
        "status": "success",
        "prediction": v,
        "confidence_percentage": confidence_pct(score, v),
        "raw_score": round(score, 4),
        "logs": [
            "Image decoded successfully.",
            "Face detected (MTCNN) and cropped to 224x224.",
            f"EfficientNetB0 inference complete. Manipulation score: {score:.4f}",
            f"Decision threshold: {state['cfg']['threshold']:.3f}",
            f"Final verdict: {v}",
        ],
    }


@app.post("/predict-video")
def analyze_video(file: UploadFile = File(...)):
    data = read_limited(file, MAX_VIDEO_MB)
    cfg = state["cfg"]
    suffix = Path(file.filename or "v.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise HTTPException(400, "Invalid or unsupported video file.")
        n = cfg["video_frames"]
        idxs = np.linspace(0, total - 1, n + 2)[1:-1].astype(int) if total > n else np.arange(total)
        faces, skipped = [], 0
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            face = crop_face(fr) if ok else None
            if face is None:
                skipped += 1
            else:
                faces.append(face)
        cap.release()
    finally:
        os.unlink(path)

    if len(faces) < cfg["min_face_frames"]:
        raise HTTPException(422, f"A face was found in only {len(faces)} sampled frames "
                                 f"(need at least {cfg['min_face_frames']}). Cannot give a reliable result.")
    scores = score_faces(faces)
    agg = float(np.median(scores))
    v = verdict(agg)
    ratio = float(np.mean(scores >= cfg["threshold"]))
    return {
        "status": "success",
        "prediction": v,
        "confidence_percentage": confidence_pct(agg, v),
        "raw_score": round(agg, 4),
        "frames_analyzed": len(faces),
        "frames_skipped": skipped,
        "manipulated_frame_ratio": round(ratio, 4),
        "logs": [
            f"Sampled {len(idxs)} frames evenly across the video.",
            f"Face found in {len(faces)} frames, {skipped} skipped (no clear face).",
            f"Per-frame scores: min {scores.min():.3f} / median {agg:.3f} / max {scores.max():.3f}",
            f"{ratio * 100:.0f}% of frames scored above the decision threshold.",
            f"Aggregation: median of frame scores -> {agg:.4f}",
            f"Final verdict: {v}",
        ],
    }
