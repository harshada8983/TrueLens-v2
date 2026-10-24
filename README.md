# TrueLens v2: Deepfake Detection for Images & Videos

TrueLens is a full-stack web app that checks whether a face in an **image or video** is **REAL**, **MANIPULATED**, or **UNCERTAIN**. A React frontend talks to a FastAPI backend that runs a fine-tuned EfficientNetB0 classifier behind an MTCNN face-detection stage.

> This is v2 of [TrueLens](https://github.com/harshada8983/TrueLens): the model was retrained from scratch with a leakage-free evaluation, the video pipeline was rebuilt, and the backend was containerised for permanent hosting instead of a local tunnel.

<!-- Live demo: add link here after deployment -->

---

## Results

All numbers come from a held-out test split the model never saw during training or threshold selection. For FaceForensics++ the split is done **by video ID**, so the same person never appears in both train and test.

| Mode | Test data | Accuracy | AUC | F1 |
|---|---|---|---|---|
| Image (StyleGAN faces) | 639 faces | **83.3%** | 0.931 | 0.844 |
| Video, per frame (FaceForensics++) | 577 frames | 86.3% | 0.925 | 0.857 |
| Video, per video (median of frames) | 147 videos | **89.1%** | **0.946** | n/a |

Aggregating frame scores into one video-level verdict improves accuracy from 86.3% to 89.1%.

**Training data:** 12,263 face crops (9,938 train / 1,109 val / 1,216 test), balanced between real and fake, drawn from FaceForensics++ (video frames) and the 140k Real and Fake Faces dataset (images).

### Honest limitations

- Test sets are small (639 images, 147 videos), so treat the numbers as indicative (about 83% and 89%), not exact.
- The model has **not** been evaluated on an unseen dataset such as Celeb-DF, so accuracy on deepfakes made with other tools may be lower.
- The image classifier was trained on StyleGAN-generated faces; fakes from newer generators can be missed. This is why the app returns **UNCERTAIN** for scores near the decision threshold instead of forcing a guess.

---

## How it works

```
Upload ──> Decode / downscale (OpenCV) ──> MTCNN face detection + crop (224x224)
       ──> EfficientNetB0 ──> P(manipulated) ──> threshold ──> REAL / UNCERTAIN / MANIPULATED
```

**Image:** detect the largest face, crop with 25% margin, classify. If no confident face is found, the API returns an error rather than a random answer.

**Video:** sample 24 frames evenly across the clip, detect and crop a face in each, score every frame, then take the **median** score (robust to a few bad frames). At least 5 frames must contain a face, otherwise the request is rejected.

**Decision rule:** the threshold (0.503) was chosen on the validation set using Youden's J. Scores within ±0.15 of the threshold are reported as UNCERTAIN.

**Model:** EfficientNetB0 (ImageNet weights) with a global-average-pooling, dropout and sigmoid head. Trained in two phases: head only (4 epochs, lr 1e-3), then the top 80 layers fine-tuned with BatchNorm frozen (up to 14 epochs, lr 1e-5, early stopping on validation AUC). Augmentation: flips, brightness/contrast, random JPEG compression and random down-scaling to mimic social-media quality.

---

## Tech stack

| Layer | Tools |
|---|---|
| Model | TensorFlow 2.20 / Keras 3, EfficientNetB0 |
| Face detection | MTCNN |
| Backend | FastAPI, Uvicorn, OpenCV |
| Frontend | React, Vite, Tailwind CSS, Axios |
| Training | Kaggle GPU notebook |
| Deployment | Docker (Hugging Face Spaces), frontend on Vercel |

---

## API

| Method | Route | Input | Limit |
|---|---|---|---|
| GET | `/` | none | health check |
| POST | `/predict` | image (`file`) | 10 MB |
| POST | `/predict-video` | video (`file`) | 50 MB |

Example video response:

```json
{
  "status": "success",
  "prediction": "MANIPULATED",
  "confidence_percentage": 87.4,
  "raw_score": 0.874,
  "frames_analyzed": 22,
  "frames_skipped": 2,
  "manipulated_frame_ratio": 0.82,
  "logs": ["Sampled 24 frames evenly across the video.", "..."]
}
```

Errors (no face found, file too large, invalid file) are returned as HTTP 4xx with a readable `detail` message.

Quick test: `python test_api.py <BACKEND_URL> photo.jpg`

---

## Project structure

```
TrueLens-v2/
├── backend/              FastAPI app, inference pipeline, requirements
│   ├── main.py
│   └── model/            truelens_model.keras + config.json (not stored in git)
├── frontend/             React + Vite UI
├── training/             Kaggle notebook and script used to train the model
├── deploy/huggingface/   Dockerfile for Hugging Face Spaces
└── test_api.py           Small client for testing the API
```

---

## Run locally

**Backend** (Python 3.11 or 3.12)

```bash
cd backend
pip install -r requirements.txt
# place truelens_model.keras and config.json (produced by the training notebook) in backend/model/
uvicorn main:app --port 8000
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

Set `VITE_API_URL` (for example `http://localhost:8000`) to point the UI at your backend.

**Retrain the model:** open `training/TrueLens-Training.ipynb` on Kaggle with a GPU, add the FaceForensics++ frames dataset and the 140k Real and Fake Faces dataset, and run all cells. It produces the model, `config.json` (threshold and settings) and `metrics.json`.

---

## Author

Built by **Harshada**. Feedback and issues are welcome.
