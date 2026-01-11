import pickle
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

MODEL_DIR = "models"
MP_MODEL_PATH = f"{MODEL_DIR}/hand_landmarker.task"
CLASSIFIER_PATH = f"{MODEL_DIR}/signs_all_rf_model.pkl"


class SignPredictor:
    def __init__(self):
        # Load classifier
        with open(CLASSIFIER_PATH, "rb") as f:
            self.model = pickle.load(f)

        # MediaPipe Hand Landmarker
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
        )

        self.landmarker = HandLandmarker.create_from_options(options)

    def predict(self, frame_bgr):
        rgb = frame_bgr[:, :, ::-1]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None, 0.0, []

        landmarks = result.hand_landmarks[0]

        features = []
        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])

        if len(features) != 63:
            return None, 0.0, []

        features = np.array(features).reshape(1, -1)

        prediction = self.model.predict(features)[0]
        confidence = np.max(self.model.predict_proba(features))

        return prediction, confidence, landmarks

    def close(self):
        self.landmarker.close()
