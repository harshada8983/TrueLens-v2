# =====================================================================
# TrueLens - training script for Kaggle (GPU T4/P100, Internet ON)
# Run cell by cell (each "# %%" block = one notebook cell).
#
# Output (in /kaggle/working/model_out):
#   truelens_model.keras, config.json, metrics.json
# Copy the first two into  TrueLens/backend/model/  in your repo.
# =====================================================================

# %% 0. Setup
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mtcnn==1.0.0", "scikit-learn", "pandas", "tqdm"])

import os, re, json, random, hashlib
from pathlib import Path
import numpy as np, pandas as pd, cv2
import tensorflow as tf, keras
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix

print("TF", tf.__version__, "| Keras", keras.__version__)
print(">>> Pin this same TensorFlow version in backend/requirements.txt <<<")

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
WORK = Path("/kaggle/working")
CROPS = WORK / "crops"; OUT = WORK / "model_out"
CROPS.mkdir(exist_ok=True, parents=True); OUT.mkdir(exist_ok=True, parents=True)

# %% 1. CONFIG - dataset paths (already filled in for the attached Kaggle inputs)
# role="train"     -> split into train/val/test (identity-wise)
# role="test_only" -> never used for training; only for an honest generalisation test
SOURCES = [
    # Kaggle "Faceforensic++" (extracted_frames/real|fake/<video_folder>/frames) - video-wise folders
    dict(name="ffpp", root="/kaggle/input/EDIT-FFPP-DATASET/extracted_frames", kind="frames", role="train",
         max_per_class=1000, frames_real=6, frames_fake=6, group_by_id=True),
    # 140k Real and Fake Faces (StyleGAN) - for the image mode
    dict(name="faces140k", root="/kaggle/input/EDIT-140K-DATASET", kind="image", role="train",
         max_per_class=6000, frames_real=1, frames_fake=1, group_by_id=False),
    # If you get an unseen dataset (e.g. Celeb-DF), add it here with role="test_only":
    # dict(name="celebdf", root="...", kind="video", role="test_only", max_per_class=300,
    #      frames_real=10, frames_fake=10, group_by_id=False),
]
# kind: "frames" = folder-per-video (real/fake/<video>/*.png), "video" = .mp4 files, "image" = single images.
# If a path ever breaks, list /kaggle/input and put the real path here.

REAL_TOKENS = {"real", "original", "original_sequences", "youtube", "celeb-real", "youtube-real",
               "original_videos", "pristine"}
FAKE_TOKENS = {"fake", "deepfakes", "face2face", "faceswap", "neuraltextures", "faceshifter",
               "manipulated_sequences", "celeb-synthesis", "synthesis", "manipulated"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def label_from_path(rel: Path):
    """Folder naam se label: 0=real, 1=fake. Pehla matching folder jeetta hai."""
    for tok in rel.parts[:-1]:
        t = tok.lower()
        if t in FAKE_TOKENS: return 1
        if t in REAL_TOKENS: return 0
    return None


# %% 2. Face extraction (IMPORTANT: backend/main.py uses exactly the same logic)
from mtcnn import MTCNN
detector = MTCNN()
SIZE = 224


def crop_face(bgr, margin=0.25, min_conf=0.95, max_side=640):
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(bgr, (int(w * s), int(h * s))) if s < 1 else bgr
    try:
        faces = detector.detect_faces(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    except (ValueError, tf.errors.InvalidArgumentError):
        return None  # mtcnn 1.0.0 bug: no-face frame par [] ki jagah error deta hai
    if not faces: return None
    f = max(faces, key=lambda d: d["box"][2] * d["box"][3])
    if f["confidence"] < min_conf: return None
    x, y, bw, bh = [v / s for v in f["box"]]
    cx, cy, side = x + bw / 2, y + bh / 2, max(bw, bh) * (1 + 2 * margin)
    x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    x1, y1 = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
    if x1 - x0 < 40 or y1 - y0 < 40: return None
    return cv2.resize(bgr[y0:y1, x0:x1], (SIZE, SIZE))


def sample_frames(path, n):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        for i in np.linspace(0, total - 1, n + 2)[1:-1].astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok: frames.append(fr)
    cap.release()
    return frames


def split_of(source, group):
    h = int(hashlib.md5(f"{source}:{group}".encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def group_key(path: Path, by_id):
    if by_id:  # FF++: 123.mp4 (real) / 123_456.mp4 (fake) -> pehla id
        m = re.match(r"(\d+)", path.stem)
        if m: return m.group(1)
    return path.stem


def process_source(src):
    root = Path(src["root"])
    if src["kind"] == "frames" and (root / "extracted_frames").is_dir():
        root = root / "extracted_frames"      # root ek level upar ho to bhi chale
    print(f"[{src['name']}] listing files... (pehli baar kuch minute lag sakte hain, rokna mat)", flush=True)
    by_label, skipped = {0: [], 1: []}, 0
    if src["kind"] == "frames":          # har video ka folder = ek item
        # sirf video folders list karo (real/<video>, fake/<video>) - saari files ginna slow hai
        items = [d for c in sorted(root.iterdir()) if c.is_dir() for d in sorted(c.iterdir()) if d.is_dir()]
        labelled = [(d, label_from_path(d.relative_to(root) / "x")) for d in items]
    else:
        exts = VIDEO_EXTS if src["kind"] == "video" else IMG_EXTS
        labelled = [(p, label_from_path(p.relative_to(root))) for p in root.rglob("*") if p.suffix.lower() in exts]
    for p, lab in labelled:
        if lab is None: skipped += 1
        else: by_label[lab].append(p)
    rng = random.Random(SEED)
    for lab in by_label:
        by_label[lab].sort(); rng.shuffle(by_label[lab]); by_label[lab] = by_label[lab][: src["max_per_class"]]
    print(f"[{src['name']}] real={len(by_label[0])} fake={len(by_label[1])} unlabeled_skipped={skipped}")
    assert by_label[0] and by_label[1], "Labels not detected - check the root path / folder names"
    for lab in (0, 1): print("   example", lab, by_label[lab][0])

    out_dir = CROPS / src["name"]; out_dir.mkdir(exist_ok=True, parents=True)
    rows = []
    for lab, plist in by_label.items():
        n = src["frames_real"] if lab == 0 else src["frames_fake"]
        for p in tqdm(plist, desc=f"{src['name']} label={lab}"):
            vid = str(p.relative_to(root))
            if src["kind"] == "video":
                frames = sample_frames(p, n)
            elif src["kind"] == "frames":
                imgs = sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXTS)
                sel = [imgs[i] for i in np.linspace(0, len(imgs) - 1, min(n, len(imgs))).astype(int)] if imgs else []
                frames = [cv2.imread(str(q)) for q in sel]
            else:
                frames = [cv2.imread(str(p))]
            for k, fr in enumerate(frames):
                if fr is None: continue
                face = crop_face(fr)
                small = max(fr.shape[:2]) <= 320   # already-cropped small images
                if face is None and (src["kind"] == "image" or (src["kind"] == "frames" and small)):
                    face = cv2.resize(fr, (SIZE, SIZE))
                if face is None: continue
                fp = out_dir / f"{hashlib.md5(vid.encode()).hexdigest()[:12]}_{k}.jpg"
                cv2.imwrite(str(fp), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
                split = "ext_test" if src["role"] == "test_only" else split_of(src["name"], group_key(p, src["group_by_id"]))
                rows.append(dict(path=str(fp), label=lab, source=src["name"], video=f"{src['name']}:{vid}", split=split))
    return rows


# Time bachane ke liye chhoti sampling (zarurat ho to badha sakte ho)
for _s in SOURCES:
    if _s["name"] == "ffpp": _s.update(max_per_class=800, frames_real=4, frames_fake=4)
    if _s["name"] == "faces140k": _s.update(max_per_class=3000)

CSV = WORK / "crops.csv"
if CSV.exists():
    df = pd.read_csv(CSV)
else:
    all_rows = []
    for s in SOURCES: all_rows += process_source(s)
    df = pd.DataFrame(all_rows); df.to_csv(CSV, index=False)
print(df.groupby(["source", "split", "label"]).size().unstack(fill_value=0))

# %% 3. Data pipeline
AUTO = tf.data.AUTOTUNE
BATCH = 32


def augment(img):  # img uint8 224x224x3
    img = tf.image.random_flip_left_right(img)
    # random low-resolution (blur jaisa effect)
    s = tf.random.uniform([], 96, 224, dtype=tf.int32)
    img = tf.cond(tf.random.uniform([]) < 0.3,
                  lambda: tf.cast(tf.image.resize(tf.image.resize(img, (s, s)), (SIZE, SIZE)), tf.uint8), lambda: img)
    # random JPEG compression (social-media jaisa)
    img = tf.cond(tf.random.uniform([]) < 0.5, lambda: tf.image.random_jpeg_quality(img, 40, 95), lambda: img)
    img = tf.image.random_brightness(tf.cast(img, tf.float32), 25.0)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    return tf.clip_by_value(img, 0.0, 255.0)


def make_ds(d, train):
    ds = tf.data.Dataset.from_tensor_slices((d.path.values, d.label.values.astype("float32")))
    if train: ds = ds.shuffle(min(len(d), 20000), seed=SEED)
    def load(p, y):
        img = tf.io.decode_jpeg(tf.io.read_file(p), channels=3); img.set_shape([SIZE, SIZE, 3])
        return img, y
    ds = ds.map(load, AUTO)
    ds = ds.map(lambda x, y: (augment(x), y), AUTO) if train else ds.map(lambda x, y: (tf.cast(x, tf.float32), y), AUTO)
    return ds.batch(BATCH).prefetch(AUTO)


tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
ext = df[df.split == "ext_test"]
n0, n1 = (tr.label == 0).sum(), (tr.label == 1).sum()
class_weight = {0: len(tr) / (2 * n0), 1: len(tr) / (2 * n1)}
print("class_weight", class_weight)

# %% 4. Model (output = P(fake))
L = keras.layers
base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(SIZE, SIZE, 3))
base.trainable = False
inp = keras.Input((SIZE, SIZE, 3))
x = base(inp, training=False)           # input 0-255 (rescaling is inside the model)
x = L.GlobalAveragePooling2D()(x)
x = L.Dropout(0.4)(x)
out = L.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)


def compile_(lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc"), "accuracy"])


train_ds, val_ds = make_ds(tr, True), make_ds(va, False)
cb = lambda: [keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
              keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=1)]

# Phase 1: train only the classification head
compile_(1e-3)
model.fit(train_ds, validation_data=val_ds, epochs=4, class_weight=class_weight, callbacks=cb())

# Phase 2: fine-tune the top layers (BatchNorm frozen)
base.trainable = True
for layer in base.layers[:-80]: layer.trainable = False
for layer in base.layers:
    if isinstance(layer, L.BatchNormalization): layer.trainable = False
compile_(1e-5)
model.fit(train_ds, validation_data=val_ds, epochs=14, class_weight=class_weight, callbacks=cb())

# %% 5. Threshold + honest evaluation
def predict(d):
    return model.predict(make_ds(d, False), verbose=0).ravel()


pv = predict(va)
fpr, tpr, thr = roc_curve(va.label.values, pv)
THRESH = float(thr[np.argmax(tpr - fpr)])
print("threshold (val Youden):", round(THRESH, 3))

metrics = {"threshold": THRESH, "tf": tf.__version__, "keras": keras.__version__}


def evaluate(d, name):
    if len(d) == 0: return
    for src, g in d.groupby("source"):
        p = predict(g); y = g.label.values
        if len(set(y)) < 2: continue
        g = g.assign(p=p)
        v = g.groupby("video").agg(p=("p", "median"), label=("label", "first"))
        m = dict(frames=int(len(g)), frame_auc=float(roc_auc_score(y, p)),
                 frame_acc=float(((p >= THRESH) == y).mean()), frame_f1=float(f1_score(y, p >= THRESH)),
                 videos=int(len(v)), video_auc=float(roc_auc_score(v.label, v.p)) if v.label.nunique() > 1 else None,
                 video_acc=float(((v.p >= THRESH) == v.label).mean()),
                 confusion=confusion_matrix(y, p >= THRESH).tolist())
        metrics[f"{name}/{src}"] = m
        print(name, src, json.dumps(m, indent=1))


evaluate(te, "test")        # same-distribution test (identity-wise unseen)
evaluate(ext, "ext_test")   # unseen dataset -> real generalisation number

# %% 6. Save
model.save(OUT / "truelens_model.keras")
cfg = dict(model_file="truelens_model.keras", input_size=SIZE, threshold=THRESH,
           uncertain_margin=0.15, video_aggregate="median", video_frames=24, min_face_frames=5,
           tf_version=tf.__version__, keras_version=keras.__version__)
(OUT / "config.json").write_text(json.dumps(cfg, indent=2))
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("Saved ->", OUT, "| copy these 2 files to backend/model/:", "truelens_model.keras, config.json")
