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

# Check current Python version and switch to 3.11
CURRENT_PYTHON=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
if [ "$CURRENT_PYTHON" != "3.11" ]; then
    echo "Current Python version: $(python3 --version)"
    echo "Uninstalling Python 3.12 and installing Python 3.11..."
    apt-get update -qq
    apt-get remove -y python3.12 python3.12-dev python3.12-venv python3.12-minimal 2>/dev/null || true
    apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
    update-alternatives --set python3 /usr/bin/python3.11
    echo "✓ Python 3.11 installed and set as default"
fi

# Verify Python 3.11 is available
if command -v python3.11 &> /dev/null || python3.11 --version &> /dev/null; then
    PYTHON_CMD=python3.11
    PIP_CMD=pip3.11
else
    PYTHON_CMD=python3
    PIP_CMD=pip3
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
if [ "$PYTHON_VERSION" != "3.11" ]; then
    echo "ERROR: Python 3.11 is required for Triton compatibility. Current: $($PYTHON_CMD --version)"
    echo "Python 3.12 has known Triton issues. Please use a Python 3.11 environment."
    exit 1
fi
echo "✓ Python version check passed: $($PYTHON_CMD --version) (using $PYTHON_CMD)"

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
$PIP_CMD freeze | grep -v "^#" | grep -v "^pip=" | grep -v "^setuptools=" | grep -v "^wheel=" | xargs $PIP_CMD uninstall -y 2>/dev/null || true
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
# Ensure /workspace exists first
mkdir -p /workspace
# Create /workspace/app
mkdir -p /workspace/app
# Verify it was created
if [ ! -d "/workspace/app" ]; then
    echo "ERROR: Failed to create /workspace/app directory"
    exit 1
fi
cd /workspace/app
rm -f *.log *.pid 2>/dev/null || true
rm -rf __pycache__ .pytest_cache 2>/dev/null || true
echo "✓ App directory cleaned"

# 5) Install PyTorch with CUDA (from PyTorch index)
echo ""
echo "[5/7] Installing PyTorch 2.8.0 with CUDA 12.8..."
$PIP_CMD install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
echo "✓ PyTorch installed"

# 5b) Install uvicorn/fastapi FIRST with exact versions to avoid conflicts
echo ""
echo "[5b/7] Installing web framework with exact versions..."
$PIP_CMD install --no-cache-dir --force-reinstall "uvicorn[standard]==0.32.0" "fastapi==0.115.0" "click>=8.1.0,<9.0.0"
echo "✓ Web framework installed"

# 5c) Install transformers from GitHub (CRITICAL: supports gpt_oss architecture)
echo ""
echo "[5c/7] Installing transformers from GitHub (supports gpt_oss)..."
$PIP_CMD uninstall -y transformers huggingface_hub accelerate safetensors 2>/dev/null || true
# Retry logic for network issues
TRANSFORMERS_INSTALLED=false
for i in {1..3}; do
    if $PIP_CMD install --no-cache-dir -U git+https://github.com/huggingface/transformers.git; then
        TRANSFORMERS_INSTALLED=true
        break
    else
        echo "Transformers install attempt $i/3 failed, retrying in 5 seconds..."
        sleep 5
    fi
done
if [ "$TRANSFORMERS_INSTALLED" = false ]; then
    echo "ERROR: Failed to install transformers from GitHub after 3 attempts!"
    exit 1
fi
$PIP_CMD install --no-cache-dir -U "huggingface_hub>=0.27.0" "accelerate>=0.35.0" "safetensors>=0.4.0"
echo "✓ Transformers installed from GitHub"

# 6) Install exact versions from requirements (skip transformers - already installed)
echo ""
echo "[6/7] Installing remaining package versions..."
cd /workspace
if [ -f "runpod/requirements-runpod.txt" ]; then
    REQUIREMENTS_FILE="runpod/requirements-runpod.txt"
else
    # Fallback: download from GitHub
    echo "Downloading requirements from GitHub..."
    wget -q https://raw.githubusercontent.com/NerualOps/Neural-Operations-Holding/main/runpod/requirements-runpod.txt -O /tmp/requirements.txt
    REQUIREMENTS_FILE="/tmp/requirements.txt"
fi

# Install requirements but skip transformers line (already installed from GitHub)
# Also skip comments and empty lines
TEMP_REQ=$(mktemp)
grep -v "^git+https://github.com/huggingface/transformers.git" "$REQUIREMENTS_FILE" | \
    grep -v "^#.*transformers" | \
    grep -v "^#" | \
    grep -v "^$" > "$TEMP_REQ"

# Install all requirements at once (more reliable than one-by-one)
$PIP_CMD install --no-cache-dir -r "$TEMP_REQ" || {
    echo "Warning: Some packages failed, trying individual installs..."
    while read -r line; do
        if [ -n "$line" ]; then
            echo "Installing: $line"
            $PIP_CMD install --no-cache-dir "$line" || echo "Warning: Failed to install $line"
        fi
    done < "$TEMP_REQ"
}
rm -f "$TEMP_REQ"

# CRITICAL: Install BitsAndBytes for 4-bit quantization (required for Python 3.12+ and fallback)
# Note: BitsAndBytes may show warnings about CUDA binary or triton.ops, but basic 4-bit quantization will still work
echo "Installing BitsAndBytes for 4-bit quantization..."
BITSANDBYTES_INSTALLED=false
for i in {1..3}; do
    if $PIP_CMD install --no-cache-dir "bitsandbytes==0.44.0"; then
        BITSANDBYTES_INSTALLED=true
        break
    else
        echo "BitsAndBytes install attempt $i/3 failed, retrying in 5 seconds..."
        sleep 5
    fi
done
if [ "$BITSANDBYTES_INSTALLED" = false ]; then
    echo "ERROR: Failed to install BitsAndBytes after 3 attempts!"
    exit 1
fi
# Patch BitsAndBytes to handle missing triton.ops
# NOTE: triton.ops is only used for performance optimization, NOT for core 4-bit quantization
# Core 4-bit quantization uses CUDA kernels (libbitsandbytes_cuda128.so) which work independently
echo "Patching BitsAndBytes to handle triton.ops import error..."
$PYTHON_CMD -c "
import sys
from pathlib import Path

# Find bitsandbytes installation
try:
    import site
    site_packages = site.getsitepackages()[0] if site.getsitepackages() else '/usr/local/lib/python3.11/dist-packages'
except:
    site_packages = '/usr/local/lib/python3.11/dist-packages'

triton_file = Path(site_packages) / 'bitsandbytes' / 'nn' / 'triton_based_modules.py'
if triton_file.exists():
    content = triton_file.read_text()
    # Make triton.ops import optional - this is only for performance tuning, not core functionality
    if 'from triton.ops.matmul_perf_model import' in content:
        # Check if already patched
        if 'try:' in content and 'from triton.ops.matmul_perf_model import' in content.split('try:')[1] if 'try:' in content else '':
            print('✓ BitsAndBytes already patched')
        else:
            # Patch the import to be optional
            lines = content.split('\n')
            new_lines = []
            patched = False
            for i, line in enumerate(lines):
                if 'from triton.ops.matmul_perf_model import' in line and not patched:
                    new_lines.append('try:')
                    new_lines.append('    ' + line)
                    new_lines.append('except ImportError:')
                    new_lines.append('    # triton.ops is optional - only used for performance optimization')
                    new_lines.append('    # Core 4-bit quantization works without it')
                    new_lines.append('    early_config_prune = None')
                    new_lines.append('    estimate_matmul_time = None')
                    patched = True
                else:
                    new_lines.append(line)
            if patched:
                triton_file.write_text('\n'.join(new_lines))
                print('✓ Patched BitsAndBytes triton.ops import (4-bit quantization will still work)')
            else:
                print('✓ Could not find import line to patch')
    else:
        print('✓ BitsAndBytes doesn\'t need patching')
else:
    print('WARNING: Could not find BitsAndBytes triton_based_modules.py to patch')
" || echo "Warning: Could not patch BitsAndBytes (may still work)"

# Test import and verify 4-bit quantization config can be created
$PYTHON_CMD -c "
import sys
try:
    import bitsandbytes
    from bitsandbytes import BitsAndBytesConfig
    # Test that we can create a 4-bit config (this is what we actually use)
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=None,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    print(f'✓ BitsAndBytes version: {bitsandbytes.__version__}')
    print('✓ BitsAndBytes 4-bit quantization config works correctly')
    sys.exit(0)
except ImportError as e:
    error_msg = str(e)
    if 'triton.ops' in error_msg:
        print('ERROR: BitsAndBytes import still failing after patch')
        sys.exit(1)
    else:
        print(f'ERROR: BitsAndBytes import failed: {e}')
        sys.exit(1)
except Exception as e:
    print(f'ERROR: BitsAndBytes test failed: {e}')
    sys.exit(1)
" || (echo "ERROR: BitsAndBytes installation verification failed!" && exit 1)

# Install optional packages (non-critical)
echo "Installing optional packages..."
$PIP_CMD install --no-cache-dir "hf-transfer>=0.1.9" 2>/dev/null || echo "Warning: hf-transfer failed (optional, continuing...)"

# Verify critical packages are installed
echo "Verifying critical packages..."
$PYTHON_CMD -c "import uvicorn" || (echo "ERROR: uvicorn not installed!" && $PIP_CMD install --no-cache-dir "uvicorn[standard]==0.32.0")
$PYTHON_CMD -c "import fastapi" || (echo "ERROR: fastapi not installed!" && $PIP_CMD install --no-cache-dir "fastapi==0.115.0")
$PYTHON_CMD -c "import pydantic" || (echo "ERROR: pydantic not installed!" && $PIP_CMD install --no-cache-dir "pydantic==2.9.2")
$PYTHON_CMD -c "import filelock" || (echo "ERROR: filelock not installed!" && $PIP_CMD install --no-cache-dir "filelock==3.16.1")
$PYTHON_CMD -c "import psutil" || (echo "ERROR: psutil not installed!" && $PIP_CMD install --no-cache-dir "psutil==6.1.0")
$PYTHON_CMD -c "import tqdm" || (echo "ERROR: tqdm not installed!" && $PIP_CMD install --no-cache-dir "tqdm>=4.66.0,<5.0.0")
echo "✓ Packages installed"

# 7) Verify critical versions and 4-bit dependencies
echo ""
echo "[7/7] Verifying versions and 4-bit dependencies..."
$PYTHON_CMD -c "import torch; print(f'PyTorch: {torch.__version__}')" || (echo "ERROR: PyTorch not installed!" && exit 1)
$PYTHON_CMD -c "import triton; print(f'Triton: {triton.__version__}')" || (echo "ERROR: Triton not installed!" && exit 1)
$PYTHON_CMD -c "import transformers; print(f'Transformers: {transformers.__version__}')" || (echo "ERROR: Transformers not installed!" && exit 1)
$PYTHON_CMD -c "import bitsandbytes; print(f'BitsAndBytes: {bitsandbytes.__version__}')" || (echo "ERROR: BitsAndBytes not installed!" && exit 1)
$PYTHON_CMD -c "import uvicorn; print(f'Uvicorn: {uvicorn.__version__}')" || (echo "ERROR: Uvicorn not installed!" && exit 1)
$PYTHON_CMD -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')" || (echo "ERROR: FastAPI not installed!" && exit 1)
$PYTHON_CMD -c "import torch; assert '2.8.0' in torch.__version__, f'PyTorch version mismatch: {torch.__version__}'"
$PYTHON_CMD -c "import triton; assert triton.__version__ == '3.4.0', f'Triton version mismatch: {triton.__version__} (required: 3.4.0)'"
$PYTHON_CMD -c "import bitsandbytes; assert '0.44' in bitsandbytes.__version__, f'BitsAndBytes version mismatch: {bitsandbytes.__version__}'"
echo "✓ Version verification passed"
echo "✓ 4-bit quantization dependencies verified (Triton for MXFP4, BitsAndBytes for fallback)"

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
echo "   nohup $PYTHON_CMD -m uvicorn inference_service:app --host 0.0.0.0 --port 8005 > inference.log 2>&1 &"
echo "   tail -f inference.log"
echo ""

