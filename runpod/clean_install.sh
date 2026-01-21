#!/bin/bash
# Clean Install Script for RunPod
# Wipes everything and reinstalls with correct versions
# Created by Neural Operations & Holdings LLC

set -e

echo "=========================================="
echo "Epsilon AI - Clean Install Script"
echo "Wiping and reinstalling with correct versions"
echo "=========================================="

# 1) Stop any running services
echo ""
echo "[1/7] Stopping services..."
pkill -9 -f "python -m uvicorn" 2>/dev/null || true
pkill -9 -f "uvicorn inference_service" 2>/dev/null || true
sleep 2
echo "✓ Services stopped"

# 2) Clean Python packages (keep system Python)
echo ""
echo "[2/7] Cleaning Python packages..."
pip freeze | grep -v "^#" | xargs pip uninstall -y 2>/dev/null || true
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

# 6) Install exact versions from requirements
echo ""
echo "[6/7] Installing exact package versions..."
cd /workspace
if [ -f "runpod/requirements-runpod.txt" ]; then
    pip install --no-cache-dir -r runpod/requirements-runpod.txt
else
    # Fallback: install from GitHub
    wget -q https://raw.githubusercontent.com/NerualOps/Neural-Operations-Holding/main/runpod/requirements-runpod.txt -O /tmp/requirements.txt
    pip install --no-cache-dir -r /tmp/requirements.txt
fi
echo "✓ Packages installed"

# 7) Verify critical versions
echo ""
echo "[7/7] Verifying versions..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import triton; print(f'Triton: {triton.__version__}')" || (echo "ERROR: Triton not installed!" && exit 1)
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
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

