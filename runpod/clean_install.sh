#!/bin/bash
# Clean Install Script for RunPod
# Wipes everything and reinstalls with correct versions
# CRITICAL: Requires Python 3.11 for Triton compatibility (Python 3.12 has Triton issues)
# Created by Neural Operations & Holdings LLC

set -e

echo "=========================================="
echo "Epsilon AI - Clean Install Script"
echo "Wiping and reinstalling with correct versions"
echo "CRITICAL: Python 3.11 required for Triton"
echo "=========================================="

# Check Python version - MUST be 3.11 for Triton
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
if [ "$PYTHON_VERSION" != "3.11" ]; then
    echo "ERROR: Python 3.11 is required for Triton compatibility. Current: $(python --version)"
    echo "Python 3.12 has known Triton issues. Please use a Python 3.11 environment."
    exit 1
fi
echo "✓ Python version check passed: $(python --version)"

# 1) Stop any running services
echo ""
echo "[1/7] Stopping services..."
pkill -9 -f "python -m uvicorn" 2>/dev/null || true
pkill -9 -f "uvicorn inference_service" 2>/dev/null || true
sleep 2
echo "✓ Services stopped"

# 2) Clean Python packages (keep system Python and essential packages)
echo ""
echo "[2/7] Cleaning Python packages..."
# Don't uninstall everything - keep pip, setuptools, wheel
pip freeze | grep -v "^#" | grep -v "^pip=" | grep -v "^setuptools=" | grep -v "^wheel=" | xargs pip uninstall -y 2>/dev/null || true
echo "✓ Packages uninstalled"

# 3) Clean model cache and downloads
echo ""
echo "[3/7] Cleaning model cache..."
rm -rf /workspace/models
rm -rf /workspace/app/models
rm -rf ~/.cache/huggingface
rm -rf /root/.cache/huggingface
rm -rf /workspace/.cache
rm -rf /workspace/app/offload
echo "✓ Cache cleared"

# 4) Clean app directory (keep only what we need)
echo ""
echo "[4/7] Cleaning app directory..."
cd /workspace/app
rm -f *.log *.pid
rm -rf __pycache__ .pytest_cache
echo "✓ App directory cleaned"

# 5) Install PyTorch with CUDA (from PyTorch index)
echo ""
echo "[5/7] Installing PyTorch 2.8.0 with CUDA 12.8..."
pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
echo "✓ PyTorch installed"

# 5b) Install uvicorn/fastapi FIRST with exact versions to avoid conflicts
echo ""
echo "[5b/7] Installing web framework with exact versions..."
pip install --no-cache-dir --force-reinstall "uvicorn[standard]==0.32.0" "fastapi==0.115.0" "click>=8.1.0,<9.0.0"
echo "✓ Web framework installed"

# 5c) Install transformers from GitHub (CRITICAL: supports gpt_oss architecture)
echo ""
echo "[5c/7] Installing transformers from GitHub (supports gpt_oss)..."
pip uninstall -y transformers 2>/dev/null || true
pip install --no-cache-dir -U git+https://github.com/huggingface/transformers.git
pip install --no-cache-dir -U accelerate huggingface_hub safetensors
echo "✓ Transformers installed from GitHub"

# 6) Install exact versions from requirements (skip transformers - already installed)
echo ""
echo "[6/7] Installing remaining package versions..."
cd /workspace
if [ -f "runpod/requirements-runpod.txt" ]; then
    # Install requirements but skip transformers line (already installed from GitHub)
    grep -v "^git+https://github.com/huggingface/transformers.git" runpod/requirements-runpod.txt | grep -v "^#.*transformers" | pip install --no-cache-dir -r /dev/stdin || true
else
    # Fallback: install from GitHub
    wget -q https://raw.githubusercontent.com/NerualOps/Neural-Operations-Holding/main/runpod/requirements-runpod.txt -O /tmp/requirements.txt
    grep -v "^git+https://github.com/huggingface/transformers.git" /tmp/requirements.txt | grep -v "^#.*transformers" | pip install --no-cache-dir -r /dev/stdin || true
fi

# Install optional packages (non-critical)
echo "Installing optional packages..."
pip install --no-cache-dir "hf-transfer>=0.1.9" 2>/dev/null || echo "Warning: hf-transfer failed (optional, continuing...)"

# Verify critical packages are installed
python -c "import uvicorn" || (echo "ERROR: uvicorn not installed!" && pip install --no-cache-dir "uvicorn[standard]==0.32.0")
python -c "import fastapi" || (echo "ERROR: fastapi not installed!" && pip install --no-cache-dir "fastapi==0.115.0")
echo "✓ Packages installed"

# 7) Verify critical versions
echo ""
echo "[7/7] Verifying versions..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || (echo "ERROR: PyTorch not installed!" && exit 1)
python -c "import triton; print(f'Triton: {triton.__version__}')" || (echo "ERROR: Triton not installed!" && exit 1)
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" || (echo "ERROR: Transformers not installed!" && exit 1)
python -c "import uvicorn; print(f'Uvicorn: {uvicorn.__version__}')" || (echo "ERROR: Uvicorn not installed!" && exit 1)
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')" || (echo "ERROR: FastAPI not installed!" && exit 1)
python -c "import torch; assert '2.8.0' in torch.__version__, f'PyTorch version mismatch: {torch.__version__}'"
python -c "import triton; assert triton.__version__ == '3.4.0', f'Triton version mismatch: {triton.__version__} (required: 3.4.0)'"
echo "✓ Version verification passed"

echo ""
echo "=========================================="
echo "Clean install complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download inference_service.py:"
echo "   cd /workspace/app"
echo "   wget https://raw.githubusercontent.com/NerualOps/Neural-Operations-Holding/main/services/python-services/inference_service.py -O inference_service.py"
echo ""
echo "2. Download model_config.py:"
echo "   wget https://raw.githubusercontent.com/NerualOps/Neural-Operations-Holding/main/services/python-services/model_config.py -O model_config.py"
echo ""
echo "3. Start service:"
echo "   nohup python -m uvicorn inference_service:app --host 0.0.0.0 --port 8005 > inference.log 2>&1 &"
echo "   tail -f inference.log"
echo ""

