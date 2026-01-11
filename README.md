# SignVision AI - Sign Language Detection

SignVision AI is a robust machine learning application designed to detect and classify sign language gestures (0-9 and A-Z) using computer vision. It leverages **MediaPipe** for hand landmark extraction and a **Random Forest Classifier** for accurate prediction. The application is built with **Flask** and allows for both local real-time webcam inference and cloud-based API prediction.

## 🚀 Features

-   **Real-time Detection**: Detects hand gestures via webcam in the browser (Local Mode).
-   **API Prediction**: RESTful API endpoint for image-based predictions (Cloud Mode).
-   **Robust Model**: Uses MediaPipe for reliable hand tracking and Random Forest for classification.
-   **Auto-Model Download**: Automatically fetches the trained model from Google Drive if not present.
-   **Thread-Safe**: Implements locking mechanisms to ensure MediaPipe runs safely in multi-threaded environments.
-   **Docker Ready**: Fully containerized for easy deployment.

## 🛠️ Tech Stack

-   **Language**: Python 3.11
-   **Framework**: Flask
-   **ML/CV**: MediaPipe, OpenCV, Scikit-learn, NumPy
-   **Deployment**: Docker, Gunicorn

## 🐳 Docker Usage (Quick Start)

To quickly get the application up and running using Docker, follow these commands:

### 1. Pull the Image
```bash
docker pull vedteredesai/signvision-ai:v1
```

### 2. Run the Container
Run the container in detached mode, mapping port 5000. Set `MODE=cloud` for API usage or `MODE=local` (note: webcam passthrough to Docker on Windows/Mac can be complex) for local testing if configured.

```bash
docker run -d --name signvision-app -p 5000:5000 -e MODE=cloud vedteredesai/signvision-ai:v1
```

### 3. Access the App
-   **Health Check**: `http://localhost:5000/health`
-   **Web Interface**: `http://localhost:5000/`

## 📦 Local Installation & Development

If you prefer to run the source code directly:

### Prerequisites
-   Python 3.10+
-   Git

### 1. Clone the Repository
```bash
git clone <repository-url>
cd SLD
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
You can run the app in `local` mode (default) to use your webcam.

```bash
# Windows (PowerShell)
$env:MODE="local"; python app.py

# Linux/Mac
export MODE=local
python app.py
```
Open your browser to `http://localhost:5000`.

## 🧠 Model Training

The project includes scripts to train the model from scratch if you have a dataset.

1.  **Prepare Data**: Organize images in `data/` folder with subfolders named `0-9` and `A-Z`.
2.  **Train**:
    ```bash
    python train_model.py
    ```
    This will extract landmarks and save the trained model to `models/signs_all_rf_model.pkl`.

## 🌐 API Endpoints

### `GET /health`
Returns the status of the service.
-   **Response**: `{"status": "ok"}`

### `POST /predict` (Cloud Mode Only)
Accepts a base64 encoded image and returns the predicted sign.

-   **Headers**: `Content-Type: application/json`
-   **Body**:
    ```json
    {
      "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    }
    ```
-   **Response**:
    ```json
    {
      "prediction": "A",
      "confidence": 0.95,
      "status": "success"
    }
    ```

## 📁 Project Structure

```
SLD/
├── app.py                 # Main Flask application
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
├── train_model.py         # Script to train the Random Forest model
├── infernece/             # Inference logic module
├── templates/             # HTML templates for the web UI
├── models/                # Directory for saved models
└── public/                # Static assets
```

## ⚠️ Important Notes

-   **MediaPipe Thread Safety**: MediaPipe's `HandLandmarker` is not thread-safe. The application uses a `threading.Lock` to strictly serialize access to the landmarker. When deploying with Gunicorn, **you must use `--workers=1`** to avoid concurrency issues that lead to crashes.
