# Epsilon AI Inference Service for RunPod
# Created by Neural Operations & Holdings LLC

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY runpod/requirements-runpod.txt ./requirements-runpod.txt
RUN pip install --no-cache-dir -r requirements-runpod.txt

# Copy application code
# Note: Build context should be project root, so paths are relative to that
COPY services/python-services/model_config.py ./model_config.py
COPY services/python-services/inference_service.py ./inference_service.py

# Create models directory
RUN mkdir -p /app/models

# Expose port
EXPOSE 8005

# Start inference service
CMD ["python", "-m", "uvicorn", "inference_service:app", "--host", "0.0.0.0", "--port", "8005"]

