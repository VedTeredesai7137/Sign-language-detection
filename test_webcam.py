import cv2
import pickle
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# ===================== PATHS =====================
MODEL_DIR = "models"
MP_MODEL_PATH = f"{MODEL_DIR}/hand_landmarker.task"
CLASSIFIER_PATH = f"{MODEL_DIR}/signs_all_rf_model.pkl"

# ===================== LOAD MODEL =====================
with open(CLASSIFIER_PATH, "rb") as f:
    model = pickle.load(f)

print("✅ Loaded trained RandomForest model")

# ===================== MEDIAPIPE SETUP =====================
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
)

landmarker = HandLandmarker.create_from_options(options)

# ===================== WEBCAM =====================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("❌ Webcam not accessible from WSL")

print("🎥 Webcam started. Press 'q' to quit.")

# ===================== LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    prediction_text = "No hand"

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        feature_vector = []
        for lm in landmarks:
            feature_vector.extend([lm.x, lm.y, lm.z])

        feature_vector = np.array(feature_vector).reshape(1, -1)

        prediction = model.predict(feature_vector)[0]
        confidence = np.max(model.predict_proba(feature_vector))

        prediction_text = f"{prediction} ({confidence*100:.1f}%)"

        # Draw landmarks
        for lm in landmarks:
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    cv2.putText(
        frame,
        prediction_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3,
    )

    cv2.imshow("Sign Language Detection (A/B/C)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ===================== CLEANUP =====================
cap.release()
cv2.destroyAllWindows()
landmarker.close()
