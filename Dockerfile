FROM python:3.11-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install minimal system dependencies required by OpenCV headless & MediaPipe
# Note: libgl1-mesa-glx is not available in Debian Trixie and not needed for opencv-python-headless
# libxrender-dev changed to libxrender1 for runtime (dev packages not needed)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app.py .
COPY templates/ ./templates/
COPY public/ ./public/
COPY models/ ./models/

# Expose port (default 5000, can be overridden via PORT env var)
EXPOSE 5000

# Required runtime environment variables
ENV FLASK_ENV=production
ENV MODE=cloud
ENV PORT=5000
ENV HOST=0.0.0.0

# Gunicorn command matches Procfile settings (single worker for MediaPipe thread-safety)
# Using $PORT env var for flexibility, with 5000 as fallback
CMD ["sh", "-c", "gunicorn app:app --workers=1 --threads=1 --timeout=120 --bind=0.0.0.0:${PORT:-5000}"]
