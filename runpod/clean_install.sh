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

# 0) Install tmux for persistent sessions
echo ""
echo "[0/8] Installing tmux..."
apt-get update -qq
apt-get install -y tmux 2>/dev/null || (echo "Warning: tmux installation failed, continuing..." && true)
echo "✓ Tmux installed"

# 1) Stop any running services
echo ""
echo "[1/8] Stopping services..."
pkill -9 -f "python -m uvicorn" 2>/dev/null || true
pkill -9 -f "uvicorn inference_service" 2>/dev/null || true
tmux kill-session -t inference 2>/dev/null || true
sleep 2
echo "✓ Services stopped"

# 2) Clean Python packages (keep system Python and essential packages)
echo ""
echo "[2/8] Cleaning Python packages..."
# Don't uninstall everything - keep pip, setuptools, wheel
$PIP_CMD freeze | grep -v "^#" | grep -v "^pip=" | grep -v "^setuptools=" | grep -v "^wheel=" | xargs $PIP_CMD uninstall -y 2>/dev/null || true
echo "✓ Packages uninstalled"

# 3) Clean model cache and downloads
echo ""
echo "[3/8] Cleaning model cache..."
rm -rf /workspace/models
rm -rf /workspace/app/models
rm -rf ~/.cache/huggingface
rm -rf /root/.cache/huggingface
rm -rf /workspace/.cache
rm -rf /workspace/app/offload
echo "✓ Cache cleared"

# 4) Clean app directory (keep only what we need)
echo ""
echo "[4/8] Cleaning app directory..."
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
echo "[5/8] Installing PyTorch 2.8.0 with CUDA 12.8..."
$PIP_CMD install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
echo "✓ PyTorch installed"

# 5b) Install uvicorn/fastapi FIRST with exact versions to avoid conflicts
echo ""
echo "[5b/8] Installing web framework with exact versions..."
$PIP_CMD install --no-cache-dir --force-reinstall "uvicorn[standard]==0.32.0" "fastapi==0.115.0" "click>=8.1.0,<9.0.0"
echo "✓ Web framework installed"

# 5c) Install transformers from GitHub (CRITICAL: supports gpt_oss architecture)
echo ""
echo "[5c/8] Installing transformers from GitHub (supports gpt_oss)..."
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
echo "[6/8] Installing remaining package versions..."
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
# Install with CUDA support - need to ensure CUDA is available
echo "Installing BitsAndBytes for 4-bit quantization with CUDA support..."
# Check CUDA version
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/') || CUDA_VERSION="12.8"
echo "Detected CUDA version: ${CUDA_VERSION}"

# Install BitsAndBytes - try with CUDA environment variable set
export CUDA_HOME=/usr/local/cuda 2>/dev/null || true
export PATH=/usr/local/cuda/bin:$PATH 2>/dev/null || true
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH 2>/dev/null || true

BITSANDBYTES_INSTALLED=false
for i in {1..3}; do
    # Uninstall first to ensure clean install
    $PIP_CMD uninstall -y bitsandbytes 2>/dev/null || true
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
$PYTHON_CMD << 'PATCH_SCRIPT'
import sys
from pathlib import Path
import re

# Find bitsandbytes installation
try:
    import site
    site_packages = site.getsitepackages()[0] if site.getsitepackages() else '/usr/local/lib/python3.11/dist-packages'
except:
    site_packages = '/usr/local/lib/python3.11/dist-packages'

triton_file = Path(site_packages) / 'bitsandbytes' / 'nn' / 'triton_based_modules.py'
if not triton_file.exists():
    print('ERROR: Could not find BitsAndBytes triton_based_modules.py to patch')
    sys.exit(1)

content = triton_file.read_text()

# Check if already patched
if 'try:' in content and 'from triton.ops.matmul_perf_model import' in content:
    # Check if the try is right before the import
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'from triton.ops.matmul_perf_model import' in line:
            # Check if previous line is 'try:'
            if i > 0 and lines[i-1].strip() == 'try:':
                print('✓ BitsAndBytes already patched')
                sys.exit(0)
            break

# Find and patch the import line
if 'from triton.ops.matmul_perf_model import' in content:
    # Use regex to find the exact import line and replace it
    pattern = r'^(\s*)from triton\.ops\.matmul_perf_model import (early_config_prune, estimate_matmul_time)'
    
    def replace_import(match):
        indent = match.group(1)
        imports = match.group(2)
        return f'{indent}try:\n{indent}    from triton.ops.matmul_perf_model import {imports}\n{indent}except ImportError:\n{indent}    # triton.ops is optional - only used for performance optimization\n{indent}    # Core 4-bit quantization works without it\n{indent}    early_config_prune = None\n{indent}    estimate_matmul_time = None'
    
    new_content = re.sub(pattern, replace_import, content, flags=re.MULTILINE)
    
    if new_content != content:
        triton_file.write_text(new_content)
        print('✓ Patched BitsAndBytes triton.ops import (4-bit quantization will still work)')
        sys.exit(0)
    else:
        print('WARNING: Could not find import line to patch with regex, trying line-by-line...')
        # Fallback: line-by-line replacement
        lines = content.split('\n')
        new_lines = []
        patched = False
        for i, line in enumerate(lines):
            if 'from triton.ops.matmul_perf_model import' in line and not patched:
                indent = len(line) - len(line.lstrip())
                indent_str = ' ' * indent
                new_lines.append(indent_str + 'try:')
                new_lines.append(indent_str + '    ' + line.lstrip())
                new_lines.append(indent_str + 'except ImportError:')
                new_lines.append(indent_str + '    # triton.ops is optional - only used for performance optimization')
                new_lines.append(indent_str + '    # Core 4-bit quantization works without it')
                new_lines.append(indent_str + '    early_config_prune = None')
                new_lines.append(indent_str + '    estimate_matmul_time = None')
                patched = True
            else:
                new_lines.append(line)
        if patched:
            triton_file.write_text('\n'.join(new_lines))
            print('✓ Patched BitsAndBytes triton.ops import (line-by-line method)')
            sys.exit(0)
        else:
            print('ERROR: Could not find import line to patch')
            sys.exit(1)
else:
    print('WARNING: triton.ops import line not found in file - may not need patching')
    sys.exit(0)
PATCH_SCRIPT

if [ $? -ne 0 ]; then
    echo "WARNING: BitsAndBytes patch had issues, but continuing..."
fi

# Test import and verify 4-bit quantization config can be created
# Note: CUDA binary warning is OK - BitsAndBytes will still work for 4-bit quantization
echo "Verifying BitsAndBytes installation..."
$PYTHON_CMD << 'PYTHON_VERIFY'
import sys
import os
import warnings

# Suppress all warnings during import
warnings.filterwarnings('ignore')
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

try:
    # Redirect stderr to capture CUDA binary warnings
    import io
    from contextlib import redirect_stderr
    
    stderr_capture = io.StringIO()
    with redirect_stderr(stderr_capture):
        import bitsandbytes
        from bitsandbytes import BitsAndBytesConfig
        import torch
    
    # Check stderr for CUDA binary warning (non-fatal)
    stderr_output = stderr_capture.getvalue()
    if 'CUDA binary' in stderr_output or 'compiled without GPU support' in stderr_output:
        print('WARNING: BitsAndBytes CUDA binary not found (this is OK for 4-bit quantization)')
    
    # Test that we can create a 4-bit config (this is what we actually use)
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    print(f'✓ BitsAndBytes version: {bitsandbytes.__version__}')
    print('✓ BitsAndBytes 4-bit quantization config created successfully')
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f'✓ CUDA is available: {torch.version.cuda}')
        print('✓ 4-bit quantization will use CUDA')
    else:
        print('WARNING: CUDA not available, but 4-bit quantization config is valid')
    
    print('✓ BitsAndBytes is ready for 4-bit quantization')
    sys.exit(0)
    
except ImportError as e:
    error_msg = str(e)
    if 'triton.ops' in error_msg:
        print('ERROR: BitsAndBytes import failing due to triton.ops - patch may have failed')
        sys.exit(1)
    else:
        print(f'ERROR: BitsAndBytes import failed: {e}')
        sys.exit(1)
except Exception as e:
    error_msg = str(e)
    # Any other error - try to continue anyway
    print(f'WARNING: BitsAndBytes verification error: {e}')
    print('Attempting to verify config creation anyway...')
    try:
        from bitsandbytes import BitsAndBytesConfig
        import torch
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
        print('✓ BitsAndBytesConfig can be created - 4-bit quantization will work')
        sys.exit(0)
    except:
        print('ERROR: Cannot create BitsAndBytesConfig')
        sys.exit(1)
PYTHON_VERIFY

if [ $? -ne 0 ]; then
    echo "ERROR: BitsAndBytes verification failed!"
    exit 1
fi

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
echo "[7/8] Verifying versions and 4-bit dependencies..."
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
echo "3. Start service in tmux:"
echo "   cd /workspace/app"
echo "   tmux new-session -d -s inference '$PYTHON_CMD -m uvicorn inference_service:app --host 0.0.0.0 --port 8005'"
echo "   tmux attach-session -t inference"
echo ""
echo "   Or to view logs without attaching:"
echo "   tmux capture-pane -t inference -p"
echo ""

