from flask import Flask, render_template, Response, send_from_directory, request, jsonify
import cv2
import pickle
import numpy as np
import os
import base64
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

app = Flask(__name__)

# ===============================
# CONFIGURATION
# ===============================
MODE = os.getenv("MODE", "local").lower()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
MODEL_FILE = "signs_all_rf_model.pkl"
HAND_LANDMARKER_FILE = "hand_landmarker.task"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
HAND_LANDMARKER_PATH = os.path.join(MODEL_DIR, HAND_LANDMARKER_FILE)

GOOGLE_DRIVE_FILE_ID = "1znuoAf-V8JNrF8DatR0f8p0tnNxtXGiJ"

# ===============================
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# ===============================
def download_model_from_gdrive():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print(f"📥 Downloading model from Google Drive...")
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            print(f"✅ Model downloaded successfully to {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            raise
    else:
        print(f"✅ Model already exists at {MODEL_PATH}")

download_model_from_gdrive()

# ===============================
# LOAD MODEL
# ===============================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(HAND_LANDMARKER_PATH):
    raise FileNotFoundError(f"Hand landmarker file not found: {HAND_LANDMARKER_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

LABELS = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]

# ===============================
# MEDIAPIPE HAND LANDMARKER (Initialized once)
# ===============================
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
)
landmarker = HandLandmarker.create_from_options(options)

print(f"✅ Running in {MODE.upper()} mode")

# ===============================
# LOCAL MODE: WEBCAM INITIALIZATION
# ===============================
cap = None
if MODE == "local":
    try:
        for camera_idx in [0, 1]:
            cap = cv2.VideoCapture(camera_idx)
            if cap.isOpened():
                print(f"✅ Webcam opened successfully (camera {camera_idx})")
                break
            cap.release()
        
        if cap is None or not cap.isOpened():
            print("⚠️ Warning: Webcam not accessible. Video stream will not work.")
            cap = None
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize webcam: {e}")
        cap = None

# ===============================
# LOCAL MODE: VIDEO STREAM GENERATOR
# ===============================
def generate_frames():
    if cap is None:
        while True:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "Webcam not available",
                (150, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            _, buffer = cv2.imencode(".jpg", placeholder)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    last_pred = "Detecting..."
    while True:
        success, frame = cap.read()
        if not success:
            continue

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

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

# ===============================
# CLOUD MODE: PREDICTION FUNCTION
# ===============================
def predict_from_image_data(image_data):
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None, 0.0
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        
        result = landmarker.detect(mp_image)
        
        if not result.hand_landmarks:
            return None, 0.0
        
        landmarks = result.hand_landmarks[0]
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])
        
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]
        confidence = float(np.max(model.predict_proba(features)))
        
        return prediction, confidence
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, 0.0

# ===============================
# ROUTES
# ===============================
@app.route("/")
def index():
    return render_template("index.html", mode=MODE)

@app.route("/video")
def video():
    if MODE != "local":
        return jsonify({"error": "Video stream only available in local mode"}), 404
    
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/predict", methods=["POST"])
def predict():
    if MODE != "cloud":
        return jsonify({"error": "Predict endpoint only available in cloud mode"}), 404
    
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        image_data = data["image"]
        prediction, confidence = predict_from_image_data(image_data)
        
        if prediction is None:
            return jsonify({
                "prediction": None,
                "confidence": 0.0,
                "status": "no_hand_detected"
            }), 200
        
        return jsonify({
            "prediction": str(prediction),
            "confidence": confidence,
            "status": "success"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/public/<path:filename>")
def public_files(filename):
    return send_from_directory(PUBLIC_DIR, filename)

# ===============================
# CLEANUP
# ===============================
def cleanup():
    if cap is not None:
        cap.release()
    if landmarker is not None:
        landmarker.close()

import atexit
atexit.register(cleanup)

# ===============================
# MAIN (Gunicorn compatible)
# ===============================
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    app.run(host=host, port=port, debug=debug)
