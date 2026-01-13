# Epsilon AI Model Setup Guide

## Overview
Epsilon AI uses a 20B parameter language model with Harmony response format.
Created by Neural Operations & Holdings LLC.

## Model Information
- **Model**: Epsilon AI 20B
- **Format**: Harmony response format (automatically applied via transformers chat template)
- **Identity**: Epsilon AI - Created by Neural Operations & Holdings LLC
- **Size**: ~40GB (will be downloaded automatically)

## Setup Steps

### 1. Install Dependencies
```powershell
cd services\python-services
pip install -r requirements_gptoss.txt
```

### 2. Download Model (Optional - will auto-download on first use)
```powershell
python ml_local\scripts\download_gptoss_model.py
```

**Note**: The model is ~40GB. It will be downloaded automatically when the inference service starts if not already cached.

### 3. Update Environment Variables
The inference service will automatically use the Epsilon AI model. You can override with:
```powershell
$env:EPSILON_MODEL_ID="openai/gpt-oss-20b"
$env:EPSILON_MODEL_DIR="services\python-services\models\epsilon-20b"
```

### 4. Start Inference Service
The service will automatically download and load the model on first startup:
```powershell
cd services\python-services
python -m uvicorn inference_service:app --host 127.0.0.1 --port 8005
```

## Important Notes

### Memory Requirements
- **20B model**: Needs ~40GB RAM/VRAM for full precision
- **2GB GPU**: Will automatically use CPU mode
- For better performance, consider:
  - Using quantization (8-bit or 4-bit)
  - Using a cloud GPU service

### Harmony Format
The model uses Harmony response format automatically. You don't need to format prompts manually - just pass the user message and the chat template handles it.

### Model Identity
The model has been configured to identify as "Epsilon AI - Created by Neural Operations & Holdings LLC" when asked about its name or origin.

## Fine-Tuning
Once the model is working, you can fine-tune it using:
```powershell
python ml_local\scripts\finetune_sales_expert.py
```

The fine-tuning script has been updated to work with the Epsilon AI model.

## Production Deployment

The model is ready for production use. The inference service will:
1. Automatically download the model on first startup
2. Cache it locally for future use
3. Handle CPU/GPU automatically based on available resources
4. Use Harmony format for all responses

## Troubleshooting

### Model won't load
- Check disk space (need ~40GB free)
- Check internet connection (first download is large)
- Check logs in `services/python-services/` for errors

### Out of memory errors
- The model will automatically fall back to CPU if GPU memory is insufficient
- Consider using quantization for lower memory usage
- May need to use a cloud service for inference

### Slow generation
- Normal for CPU inference with 20B model
- Consider using a GPU service or quantization
- The model is optimized for quality over speed

