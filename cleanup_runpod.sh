#!/bin/bash
# Complete cleanup script for RunPod - clears all caches and temporary files

echo "=== RUNPOD COMPLETE CLEANUP ==="
echo ""

# Step 1: Stop all Python processes
echo "[1/8] Stopping all Python processes..."
pkill -9 -f uvicorn
pkill -9 -f python
sleep 2

# Step 2: Clear GPU memory
echo "[2/8] Clearing GPU memory..."
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    import gc
    gc.collect()
    print("GPU memory cleared")
EOF

# Step 3: Remove entire Hugging Face cache
echo "[3/8] Removing Hugging Face cache..."
rm -rf ~/.cache/huggingface
echo "Hugging Face cache cleared"

# Step 4: Remove model directory
echo "[4/8] Removing model directory..."
rm -rf /workspace/models/epsilon-20b
rm -rf /workspace/models
echo "Model directory cleared"

# Step 5: Clear Python cache
echo "[5/8] Clearing Python cache..."
find /workspace -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find /workspace -type f -name "*.pyc" -delete 2>/dev/null
find /workspace -type f -name "*.pyo" -delete 2>/dev/null
echo "Python cache cleared"

# Step 6: Clear pip cache
echo "[6/8] Clearing pip cache..."
pip cache purge 2>/dev/null || true
rm -rf ~/.cache/pip
echo "Pip cache cleared"

# Step 7: Clear system temp files
echo "[7/8] Clearing temporary files..."
rm -rf /tmp/*
rm -rf /workspace/tmp
rm -rf /workspace/*.log
rm -rf /workspace/.cursor/debug.log
echo "Temporary files cleared"

# Step 8: Show disk usage
echo "[8/8] Current disk usage:"
df -h /workspace
echo ""
du -sh /workspace/* 2>/dev/null | sort -h | tail -10
echo ""

echo "=== CLEANUP COMPLETE ==="
echo "Ready to reinstall and download model"

