FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY templates/ ./templates/
COPY public/ ./public/
COPY infernece/ ./infernece/

# Create models directory (model will be downloaded on startup)
RUN mkdir -p models

# Copy hand_landmarker.task (required for MediaPipe)
# This file should be included in the repository or built into the image
COPY models/hand_landmarker.task ./models/

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV USE_WEBCAM=false
ENV PORT=5000
ENV HOST=0.0.0.0

# Run the application
CMD ["python", "app.py"]
