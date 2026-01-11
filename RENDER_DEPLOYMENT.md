# Render Deployment Checklist

## Prerequisites
- Python 3.11 (recommended)
- Git repository with all code
- Google Drive model file accessible (File ID: 1znuoAf-V8JNrF8DatR0f8p0tnNxtXGiJ)
- MediaPipe hand_landmarker.task file in models/ directory

## Required Files
- [x] `Procfile` (contains: `web: gunicorn app:app --workers=1 --threads=1 --timeout 120`)
- [x] `requirements.txt` (all dependencies including gunicorn)
- [x] `app.py` (main Flask application)
- [x] `templates/index.html` (frontend template)
- [x] `public/` directory with A.jpg and C.jpg
- [x] `models/hand_landmarker.task` (MediaPipe model file)

## Environment Variables (Set in Render Dashboard)
1. `MODE=cloud` (REQUIRED - enables cloud mode)
2. `PORT` (optional - Render sets automatically)
3. `MODEL_PATH` (optional - override model file path)

## Deployment Steps

### 1. Create New Web Service
- Go to Render Dashboard → New → Web Service
- Connect your Git repository
- Select repository and branch

### 2. Configure Build Settings
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: (leave empty, uses Procfile)

### 3. Configure Environment Variables
- Add `MODE=cloud` (REQUIRED)
- Optional: `MODEL_PATH` if using custom path

### 4. Configure Runtime Settings
- **Python Version**: 3.11 (recommended)
- **Instance Type**: Choose based on your needs (minimum: 512MB RAM)

### 5. Deploy
- Click "Create Web Service"
- Wait for build to complete
- Monitor logs for:
  - ✅ Model downloaded successfully
  - ✅ MediaPipe HandLandmarker initialized
  - ✅ Running in CLOUD mode

### 6. Verify Deployment
- Visit `/health` endpoint → should return `{"status": "ok"}`
- Visit `/` endpoint → should load frontend
- Test `/predict` endpoint with base64 image data

## Important Notes

### Gunicorn Configuration
- **Single Worker Required**: MediaPipe is NOT thread-safe
- Command: `gunicorn app:app --workers=1 --threads=1 --timeout 120`
- Do NOT increase workers - will cause crashes

### Model Download
- Model downloads automatically on first startup from Google Drive
- Download occurs during build/startup phase
- If download fails, service will not start (fail-fast behavior)
- Model file ID: `1znuoAf-V8JNrF8DatR0f8p0tnNxtXGiJ`

### File Size Limits
- Request size limit: 5MB (base64 images)
- Health check: No MediaPipe usage (fast response)

### Troubleshooting
- **Service won't start**: Check logs for model download errors
- **503 errors**: Check if MediaPipe model file exists
- **Memory issues**: Increase instance size if needed
- **Timeout errors**: Already configured with 120s timeout

## Testing Locally (Before Deployment)
1. Set `MODE=cloud` environment variable
2. Run: `gunicorn app:app --workers=1 --threads=1 --timeout 120`
3. Test `/health` endpoint
4. Test `/predict` endpoint with browser client
5. Verify no webcam initialization (cloud mode)

