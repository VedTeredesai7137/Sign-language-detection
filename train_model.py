import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
import string

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# =====================================================
# CONFIGURATION (WSL PATHS)
# =====================================================

DATA_DIR = "/mnt/d/VED/Coding/ML/Computer Vision/CVE/Sign Convention/SLD/data2"

MODEL_DIR = "models"
MP_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")

# 🔁 NEW MODEL NAME (A–Z)
OUTPUT_MODEL_PATH = os.path.join(MODEL_DIR, "signs_AZ_rf_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# LABELS (A–Z ONLY)
# =====================================================

LABELS = list(string.ascii_uppercase)  # ['A', 'B', ..., 'Z']

print(f"📌 Total classes: {len(LABELS)}")
print(LABELS)

# =====================================================
# MEDIAPIPE HAND LANDMARKER (TASKS API)
# =====================================================

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
)

landmarker = HandLandmarker.create_from_options(options)

# =====================================================
# DATA EXTRACTION
# =====================================================

X = []
y = []

print("\n🔍 Extracting hand landmarks (MediaPipe Tasks API)\n")

for label in LABELS:
    label_path = os.path.join(DATA_DIR, label)

    if not os.path.isdir(label_path):
        print(f"⚠️ Folder missing, skipping: {label}")
        continue

    image_files = sorted(os.listdir(label_path))

    if len(image_files) == 0:
        print(f"⚠️ Empty folder: {label}")
        continue

    for img_name in tqdm(image_files, desc=f"Processing {label}", leave=False):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb
        )

        result = landmarker.detect(mp_image)

        if not result.hand_landmarks:
            continue

        landmarks = result.hand_landmarks[0]

        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])

        if len(features) != 63:
            continue

        X.append(features)
        y.append(label)

landmarker.close()

# =====================================================
# DATA VALIDATION
# =====================================================

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"\n✅ Total usable samples: {len(X)}")

if len(X) == 0:
    raise RuntimeError("❌ No landmarks detected. Dataset is unusable.")

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# RANDOM FOREST (A–Z)
# =====================================================

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\n🧠 Training RandomForest (A–Z)...\n")
model.fit(X_train, y_train)

# =====================================================
# EVALUATION
# =====================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Validation Accuracy: {accuracy * 100:.2f}%\n")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# =====================================================
# SAVE MODEL
# =====================================================

with open(OUTPUT_MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"\n💾 Model saved at: {OUTPUT_MODEL_PATH}")
