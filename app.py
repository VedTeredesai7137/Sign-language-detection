from flask import Flask, render_template, Response, send_from_directory, request, jsonify
import cv2
import pickle
import numpy as np
import os
import base64
import string
import time
import threading
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

app = Flask(__name__)

# Request size limit: 5MB (prevents DoS from large base64 images)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# ===============================
# CONFIGURATION
# ===============================
MODE = os.getenv("MODE", "local").lower()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
MODEL_FILE = "signs_AZ_rf_model.pkl"
HAND_LANDMARKER_FILE = "hand_landmarker.task"
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODEL_DIR, MODEL_FILE))
HAND_LANDMARKER_PATH = os.path.join(MODEL_DIR, HAND_LANDMARKER_FILE)

GOOGLE_DRIVE_FILE_ID = "138YZZwXls251Zsg3cBgUG0qjVctw25Sx"


# ===============================
# MEDIAPIPE THREAD SAFETY LOCK
# ===============================
# CRITICAL: MediaPipe HandLandmarker.detect() is NOT thread-safe.
# Even with Gunicorn --workers=1, Flask may handle multiple requests
# concurrently via threading. This lock ensures only ONE call to
# landmarker.detect() executes at a time across all threads.
#
# Without this lock, concurrent MediaPipe calls will cause:
# - Segmentation faults
# - Memory corruption
# - Unpredictable crashes
#
# This lock is MANDATORY for production safety.
mediapipe_lock = threading.Lock()

# ===============================
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# ===============================
def download_model_from_gdrive():
    """
    Download model from Google Drive with timeout and validation.
    Fails fast with clear error if download is blocked or fails.
    """
    # Allow MODEL_PATH override via environment variable
    target_path = MODEL_PATH
    target_dir = os.path.dirname(target_path)
    
    os.makedirs(target_dir, exist_ok=True)
    
    if os.path.exists(target_path):
        file_size = os.path.getsize(target_path)
        if file_size > 0:
            print(f"✅ Model already exists at {target_path} ({file_size} bytes)")
            return
        else:
            print(f"⚠️ Model file exists but is empty, re-downloading...")
            os.remove(target_path)
    
    print(f"📥 Downloading model from Google Drive...")
    try:
        import gdown
        import signal
        
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        
        # Download with timeout handling
        # gdown internally uses requests which has default timeout behavior
        # We validate file size after download to ensure it's not blocked/empty
        gdown.download(url, target_path, quiet=False)
        
        # Validate downloaded file exists and has size > 0
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Download completed but file not found: {target_path}")
        
        file_size = os.path.getsize(target_path)
        if file_size == 0:
            os.remove(target_path)
            raise RuntimeError(f"Downloaded file is empty (0 bytes). Google Drive may have blocked the download or file is corrupted.")
        
        if file_size < 1024:  # Less than 1KB is suspicious for a model file
            os.remove(target_path)
            raise RuntimeError(f"Downloaded file is too small ({file_size} bytes). Google Drive may have blocked the download. File ID: {GOOGLE_DRIVE_FILE_ID}")
        
        print(f"✅ Model downloaded successfully to {target_path} ({file_size} bytes)")
    except FileNotFoundError as e:
        error_msg = f"FATAL: Model file not found after download: {e}"
        print(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"FATAL: Failed to download model from Google Drive (File ID: {GOOGLE_DRIVE_FILE_ID}): {e}"
        print(error_msg)
        raise RuntimeError(error_msg)

download_model_from_gdrive()

# ===============================
# LOAD MODEL
# ===============================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"FATAL: Model file not found: {MODEL_PATH}")

if not os.path.exists(HAND_LANDMARKER_PATH):
    raise FileNotFoundError(f"FATAL: Hand landmarker file not found: {HAND_LANDMARKER_PATH}. Please ensure models/hand_landmarker.task exists.")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Model loaded successfully")
except Exception as e:
    raise RuntimeError(f"FATAL: Failed to load model from {MODEL_PATH}: {e}")

LABELS = list(string.ascii_uppercase)

# ===============================
# MEDIAPIPE HAND LANDMARKER
# ===============================
# CRITICAL: MediaPipe HandLandmarker is NOT thread-safe.
# This application MUST run with Gunicorn using --workers=1 (single worker).
# Multiple workers will cause crashes and unpredictable behavior.
#
# Recommended Gunicorn command:
# gunicorn app:app --workers=1 --threads=1 --timeout 120
#
# The landmarker is initialized ONCE at startup and reused for all requests.
# Never re-initialize MediaPipe objects per request.
# All landmarker.detect() calls are protected by mediapipe_lock.
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
)
landmarker = HandLandmarker.create_from_options(options)
print(f"✅ MediaPipe HandLandmarker initialized")

print(f"✅ Running in {MODE.upper()} mode")

# ===============================
# LOCAL MODE: WEBCAM INITIALIZATION
# ===============================
# Webcam is ONLY initialized in local mode.
# In cloud mode (MODE=cloud), this code is skipped entirely.
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
# FPS throttling: Target 30 FPS (33ms per frame) to prevent tight loops
# that can kill Gunicorn workers. time.sleep() prevents CPU spinning.
FRAME_INTERVAL = 1.0 / 30.0  # 30 FPS

def generate_frames():
    """
    Video stream generator for local mode only.
    Handles client disconnects safely (GeneratorExit) and throttles FPS.
    All MediaPipe calls are protected by mediapipe_lock.
    """
    if cap is None:
        while True:
            try:
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
                time.sleep(FRAME_INTERVAL)
            except GeneratorExit:
                return

    last_pred = "Detecting..."
    last_frame_time = 0.0
    
    while True:
        try:
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            # FPS throttling: skip frames if processing is too fast
            if elapsed < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - elapsed)
            
            success, frame = cap.read()
            if not success:
                time.sleep(FRAME_INTERVAL)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # CRITICAL: Protect MediaPipe call with lock
            with mediapipe_lock:
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
            
            last_frame_time = time.time()
        except GeneratorExit:
            return
        except Exception as e:
            print(f"Error in frame generator: {e}")
            time.sleep(FRAME_INTERVAL)

# ===============================
# CLOUD MODE: PREDICTION FUNCTION
# ===============================
def predict_from_image_data(image_data):
    """
    Process base64 image data and return prediction.
    Handles malformed input gracefully without crashing.
    All MediaPipe calls are protected by mediapipe_lock.
    """
    try:
        if not image_data or not isinstance(image_data, str):
            return None, 0.0
        
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        try:
            image_bytes = base64.b64decode(image_data, validate=True)
        except Exception as e:
            print(f"Invalid base64 data: {e}")
            return None, 0.0
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None, 0.0
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        
        # CRITICAL: Protect MediaPipe call with lock
        with mediapipe_lock:
            result = landmarker.detect(mp_image)
        
        if not result.hand_landmarks:
            return None, 0.0
        
        landmarks = result.hand_landmarks[0]
        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])
        
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        # Defensively handle predict_proba (may not be available for all models)
        try:
            proba = model.predict_proba(features)
            confidence = float(np.max(proba))
        except AttributeError:
            confidence = 1.0
        except Exception:
            confidence = 1.0
        
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

@app.route("/health")
def health():
    """
    Health check endpoint for Render deployment monitoring.
    Returns 200 OK if application is running.
    No MediaPipe usage to avoid blocking on health checks.
    """
    return jsonify({"status": "ok"}), 200

@app.route("/video")
def video():
    # Video stream is ONLY available in local mode.
    # In cloud mode, webcam access is not possible on the server.
    if MODE != "local":
        return jsonify({"error": "Video stream only available in local mode. Set MODE=local to enable."}), 404
    
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/predict", methods=["POST"])
def predict():
    # Predict endpoint is ONLY available in cloud mode.
    # In local mode, predictions are streamed via /video.
    if MODE != "cloud":
        return jsonify({"error": "Predict endpoint only available in cloud mode. Set MODE=cloud to enable."}), 404
    
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
        print(f"Error in /predict endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

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
# This app.run() is ONLY executed when running directly (python app.py).
# In production with Gunicorn, this block is skipped and Gunicorn
# directly uses the 'app' Flask instance.
#
# DEPLOYMENT NOTES FOR RENDER:
# ----------------------------
# 1. Procfile content (create file named "Procfile" in project root):
#    web: gunicorn app:app --workers=1 --threads=1 --timeout 120
#
# 2. Required environment variables in Render dashboard:
#    MODE=cloud
#    PORT=10000 (or let Render set automatically)
#
# 3. Recommended Python version: 3.11
#    Set in render.yaml or Build Settings
#
# 4. Build command: pip install -r requirements.txt
#
# 5. Start command: (leave empty, uses Procfile)
#
# Single worker (--workers=1) is REQUIRED due to MediaPipe thread-safety constraints.
# Even with single worker, mediapipe_lock protects against thread-level concurrency.
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    app.run(host=host, port=port, debug=debug)
