#!/bin/bash
# Deploy Epsilon AI to RunPod
# Created by Neural Operations & Holdings LLC

set -e

echo "=========================================="
echo "Epsilon AI RunPod Deployment"
echo "Created by Neural Operations & Holdings LLC"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    exit 1
fi

# Get image name from user or use default
IMAGE_NAME=${1:-"epsilon-inference:latest"}
REGISTRY=${2:-"runpod.io"}

FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}"

echo ""
echo "Building Docker image..."
echo "Note: Run this from the project root directory"
docker build -t ${FULL_IMAGE_NAME} -f runpod/Dockerfile .

echo ""
echo "Image built successfully!"
echo ""
echo "Next steps:"
echo "1. Push image to RunPod:"
echo "   docker push ${FULL_IMAGE_NAME}"
echo ""
echo "2. Create RunPod template with image: ${FULL_IMAGE_NAME}"
echo ""
echo "3. Deploy pod with GPU (RTX 4090 or A100)"
echo ""
echo "4. Get public URL and set INFERENCE_URL in Render"

