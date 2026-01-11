FROM python:3.11-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install minimal system dependencies required by OpenCV & MediaPipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
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

# Ensure models directory exists (for downloaded ML model)
RUN mkdir -p models

# Expose port (Render uses $PORT)
EXPOSE 5000

# Required runtime environment variables
ENV FLASK_ENV=production
ENV MODE=cloud
ENV PORT=5000
ENV HOST=0.0.0.0

# Gunicorn is mandatory for production
CMD ["gunicorn", "app:app", "--workers=1", "--threads=1", "--timeout=120", "--bind=0.0.0.0:5000"]
