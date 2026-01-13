# RunPod Deployment Guide
Created by Neural Operations & Holdings LLC

## Overview
Deploy Epsilon AI 20B model to RunPod GPU. No Docker needed - deploy directly and setup via SSH.

## Step 1: Deploy Pod on RunPod

On the pod configuration screen:

1. **Pod Template**: `Runpod Pytorch 2.8.0` (default)
2. **Expose HTTP Port**: `8005`
3. **Environment Variables** (add these):
   ```
   PORT=8005
   ```
   **Optional** (only if you want to use Supabase for model storage):
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_supabase_service_key
   ```
   **Note**: If Supabase credentials are not set, the model will download directly from Hugging Face automatically (recommended - faster and simpler).
4. **SSH Terminal Access**: ✅ Enable
5. **Container Disk**: 80GB (you have this)
6. Click **Deploy Pod**

Wait 1-2 minutes for pod to start.

## Step 2: Get Your Public URL

After deployment, copy your **Public URL**:
- Format: `https://xxxxx-8005.proxy.runpod.net`
- Save this - you'll need it for Render

## Step 3: SSH Into Pod

1. Click **Connect** → **SSH** on your pod
2. Or copy the SSH command from RunPod

## Step 4: Setup Service (Copy-Paste These Commands)

Once SSH'd in, run these commands:

```bash
# Install dependencies
pip install fastapi uvicorn transformers accelerate torch supabase python-dotenv psutil bitsandbytes pydantic

# Create app directory
mkdir -p /workspace/app
cd /workspace/app

# Create model_config.py
cat > model_config.py << 'EOF'
"""
Epsilon AI Model Configuration
Created by Neural Operations & Holdings LLC
"""
HF_MODEL_ID = "openai/gpt-oss-20b"
MODEL_NAME = "Epsilon AI 20B"
COMPANY_NAME = "Neural Operations & Holdings LLC"
EOF
```

## Step 5: Upload inference_service.py

**Option A: RunPod File Manager**
1. Go to pod → **Files** tab
2. Navigate to `/workspace/app/`
3. Upload `services/python-services/inference_service.py`

**Option B: Copy-Paste via SSH**
1. Open your local `services/python-services/inference_service.py`
2. Copy all content
3. In SSH: `nano inference_service.py`
4. Paste (right-click), save (Ctrl+X, Y, Enter)

## Step 6: Run the Service

```bash
cd /workspace/app
python -m uvicorn inference_service:app --host 0.0.0.0 --port 8005
```

Service will:
- Download model from Hugging Face automatically (first time, ~10-15 min for 40GB)
- Load model into GPU memory
- Start serving on port 8005

## Step 7: Keep Service Running (Background)

Press `Ctrl+Z` to stop, then:

```bash
# Install screen
apt-get update && apt-get install -y screen

# Start in background
screen -S inference
python -m uvicorn inference_service:app --host 0.0.0.0 --port 8005

# Detach: Ctrl+A, then D
# Reattach later: screen -r inference
```

## Step 8: Test Your Service

Visit: `https://xxxxx-8005.proxy.runpod.net/health`

Should return: `{"status":"ok","model_loaded":true}`

## Step 9: Connect Render

1. Go to Render dashboard → Your service → Environment
2. Add: `INFERENCE_URL=https://xxxxx-8005.proxy.runpod.net`
3. Save and redeploy

Done! Your model is now running on RunPod GPU.

## Troubleshooting

**Service stops when SSH closes?**
- Use `screen` (Step 7) to keep it running

**Model not loading?**
- Model downloads from Hugging Face automatically - first download takes 10-15 minutes
- Check pod logs in RunPod console for download progress
- Ensure you have enough disk space (80GB recommended)
- If using Supabase, check credentials are correct

**Can't connect from Render?**
- Verify public URL is correct
- Check pod is running (not stopped)
- Test URL directly: `curl https://xxxxx-8005.proxy.runpod.net/health`

## Cost
- A40 (48GB VRAM): $0.40/hour
- Monthly (24/7): ~$290/month
- Pay-per-second billing
- **Note**: A40 has 48GB VRAM - perfect for running 20B model at full precision without quantization!

