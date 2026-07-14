# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    cmake \
    build-essential \
    pkg-config \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements-api.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY api_inference.py .
COPY salmonn_sqa/ ./salmonn_sqa/

# Create directory for models (can be mounted as volume)
RUN mkdir -p /app/salmonn_sqa/models /app/salmonn_sqa/ckpt

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the API server
CMD ["python3", "api_inference.py", "--host", "0.0.0.0", "--port", "8000"]
