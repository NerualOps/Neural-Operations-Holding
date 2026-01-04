#!/bin/bash
# Deploy model artifact to Render
# This script downloads the model artifact and prepares it for Render deployment

set -e

MODEL_ARTIFACT_URL="${MODEL_ARTIFACT_URL:-}"
MODEL_DIR="${EPSILON_MODEL_DIR:-services/python-services/models/latest}"

if [ -z "$MODEL_ARTIFACT_URL" ]; then
    echo "ERROR: MODEL_ARTIFACT_URL environment variable is required"
    echo "Set MODEL_ARTIFACT_URL to the URL of your model artifact (zip file)"
    exit 1
fi

echo "Deploying model artifact to Render..."
echo "Artifact URL: $MODEL_ARTIFACT_URL"
echo "Target directory: $MODEL_DIR"

# Create model directory
mkdir -p "$MODEL_DIR"

# Download artifact
echo "Downloading model artifact..."
if command -v curl &> /dev/null; then
    curl -L -o /tmp/model_artifact.zip "$MODEL_ARTIFACT_URL"
elif command -v wget &> /dev/null; then
    wget -O /tmp/model_artifact.zip "$MODEL_ARTIFACT_URL"
else
    echo "ERROR: Neither curl nor wget is available"
    exit 1
fi

# Extract artifact
echo "Extracting model artifact..."
unzip -q -o /tmp/model_artifact.zip -d "$MODEL_DIR"

# Verify model files exist
if [ ! -f "$MODEL_DIR/model.pt" ]; then
    echo "ERROR: model.pt not found in artifact"
    exit 1
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: config.json not found in artifact"
    exit 1
fi

if [ ! -f "$MODEL_DIR/tokenizer.json" ]; then
    echo "ERROR: tokenizer.json not found in artifact"
    exit 1
fi

echo "✓ Model artifact deployed successfully"
echo "Model directory: $MODEL_DIR"
ls -lh "$MODEL_DIR"

# Cleanup
rm -f /tmp/model_artifact.zip

