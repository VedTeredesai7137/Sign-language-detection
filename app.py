from flask import Flask, render_template, Response
import cv2
import pickle
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

app = Flask(__name__)

# ===============================
# LOAD MODEL
# ===============================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
with open(os.path.join(MODEL_DIR, "signs_all_rf_model.pkl"), "rb") as f:
    model = pickle.load(f)

LABELS = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]

# ===============================
# MEDIAPIPE HAND LANDMARKER
# ===============================
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "hand_landmarker.task")),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
)
landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# ===============================
# VIDEO STREAM
# ===============================
def generate_frames():
    last_pred = "Detecting..."

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            features = []
            for lm in landmarks:
                features.extend([lm.x, lm.y, lm.z])

            features = np.array(features).reshape(1, -1)
            last_pred = model.predict(features)[0]

        # ✅ OVERLAY TEXT (THIS IS WHY IT NOW SHOWS)
        cv2.rectangle(frame, (20, 20), (360, 90), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Detected: {last_pred}",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 255, 0),
            4
        )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

# ===============================
# ROUTES
# ===============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(debug=True)
