"""
Epsilon AI Inference Service
Uses the Epsilon AI model with Harmony response format
Created by Neural Operations & Holdings LLC
"""
import os
import sys
import io
from pathlib import Path
from typing import Optional, List
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json

app = FastAPI(title="Epsilon AI Inference Service")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and tokenizer
model = None
tokenizer = None
pipe = None
model_metadata = None

# Import model configuration
from model_config import HF_MODEL_ID, MODEL_NAME, COMPANY_NAME

# Model configuration
MODEL_ID = os.getenv('EPSILON_MODEL_ID', HF_MODEL_ID)
MODEL_DIR = os.getenv('EPSILON_MODEL_DIR', str(Path(__file__).parent / 'models' / 'epsilon-20b'))


def load_model():
    """Load Epsilon AI model using transformers"""
    global model, tokenizer, pipe, model_metadata
    
    print(f"[INFERENCE SERVICE] Loading Epsilon AI model: {MODEL_ID}", flush=True)
    
    try:
        # Check if MODEL_ID is a local path or Hugging Face ID
        model_path = Path(MODEL_ID) if os.path.exists(MODEL_ID) else MODEL_ID
        use_local = os.path.exists(str(model_path))
        
        if use_local:
            print(f"[INFERENCE SERVICE] Loading from local path: {model_path}", flush=True)
            load_kwargs = {"local_files_only": True}
        else:
            print(f"[INFERENCE SERVICE] Loading from Hugging Face: {MODEL_ID}", flush=True)
            load_kwargs = {"cache_dir": MODEL_DIR}
        
        # Load tokenizer
        print(f"[INFERENCE SERVICE] Loading tokenizer...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            **load_kwargs
        )
        
        # Load model with appropriate settings
        print(f"[INFERENCE SERVICE] Loading model (this may take a while)...", flush=True)
        
        # Check available RAM
        import psutil
        available_ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"[INFERENCE SERVICE] Available RAM: {available_ram_gb:.2f} GB", flush=True)
        
        # 20B model memory requirements:
        # - Full precision (float32): ~80GB
        # - Half precision (float16): ~40GB  
        # - 8-bit quantization: ~20GB
        # - 4-bit quantization: ~10GB (model supports MXFP4 natively)
        
        device_map = "auto"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        use_quantization = False
        quantization_config = None
        
        # Check if we need quantization based on available RAM
        if available_ram_gb < 16:
            print(f"[INFERENCE SERVICE] Low RAM detected ({available_ram_gb:.2f} GB)", flush=True)
            
            # Try 4-bit quantization (requires CUDA/bitsandbytes)
            if torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"  # NormalFloat4
                    )
                    use_quantization = True
                    print(f"[INFERENCE SERVICE] 4-bit quantization enabled (reduces memory to ~10GB)", flush=True)
                except ImportError:
                    print(f"[INFERENCE SERVICE] WARNING: bitsandbytes not available for GPU quantization", flush=True)
            else:
                # CPU-only: Use CPU offloading to disk (slower but works with limited RAM)
                print(f"[INFERENCE SERVICE] CPU-only detected, using disk offloading for memory efficiency", flush=True)
                device_map = "sequential"  # Load layers sequentially to reduce peak memory
                # Note: This will be slower but should work with 8GB RAM
                print(f"[INFERENCE SERVICE] WARNING: Model will be slow on CPU with limited RAM", flush=True)
        
        # Check GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[INFERENCE SERVICE] GPU memory: {gpu_memory:.2f} GB", flush=True)
            
            if gpu_memory < 40 and not use_quantization:
                print(f"[INFERENCE SERVICE] GPU memory insufficient for full model, using CPU", flush=True)
                device_map = "cpu"
                torch_dtype = torch.float32
        
        # Load model with quantization if needed
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            **load_kwargs
        }
        
        if use_quantization and quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            **model_kwargs
        )
        
        # Create pipeline for easier generation with harmony format
        print(f"[INFERENCE SERVICE] Creating pipeline...", flush=True)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        
        model_metadata = {
            "model_id": MODEL_ID,
            "model_name": MODEL_NAME,
            "framework": "transformers",
            "harmony_format": True,
            "epsilon_identity": f"Epsilon AI - Created by {COMPANY_NAME}"
        }
        
        print(f"[INFERENCE SERVICE] Model loaded successfully!", flush=True)
        print(f"[INFERENCE SERVICE] Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
        
    except Exception as e:
        import traceback
        print(f"[INFERENCE SERVICE] ERROR: Failed to load model: {e}", flush=True)
        print(f"[INFERENCE SERVICE] Traceback: {traceback.format_exc()}", flush=True)
        raise


async def bootstrap_model_from_supabase():
    """Bootstrap model from Supabase if not found locally"""
    try:
        from supabase import create_client, Client
        import zipfile
        import shutil
        
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        
        if not supabase_url or not supabase_key:
            print("[INFERENCE SERVICE] Supabase credentials not set - will download from Hugging Face", flush=True)
            return False
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Get latest approved model
        response = supabase.table('epsilon_model_deployments').select(
            'id, model_id, storage_path, version'
        ).eq('status', 'approved').order(
            'deployed_at', desc=True
        ).limit(1).execute()
        
        if not response.data or len(response.data) == 0:
            print("[INFERENCE SERVICE] No approved production model found in Supabase", flush=True)
            return False
        
        deployment = response.data[0]
        storage_path = deployment.get('storage_path')
        
        if not storage_path:
            print("[INFERENCE SERVICE] Approved deployment has no storage_path", flush=True)
            return False
        
        # Download model zip
        print(f"[INFERENCE SERVICE] Downloading model from Supabase: {storage_path}", flush=True)
        
        # Handle chunked files or regular zip
        zip_data = None
        if storage_path.endswith('.metadata.json'):
            # Chunked file - download metadata and reassemble
            print(f"[INFERENCE SERVICE] Detected chunked model file, downloading chunks...", flush=True)
            metadata_response = supabase.storage.from_('epsilon-models').download(storage_path)
            if not metadata_response:
                print("[INFERENCE SERVICE] Failed to download chunk metadata", flush=True)
                return False
            
            chunk_metadata = json.loads(metadata_response.decode('utf-8'))
            
            chunks = []
            for chunk_info in sorted(chunk_metadata['chunks'], key=lambda x: x['index']):
                print(f"[INFERENCE SERVICE] Downloading chunk {chunk_info['index'] + 1}/{len(chunk_metadata['chunks'])}...", flush=True)
                chunk_response = supabase.storage.from_('epsilon-models').download(chunk_info['path'])
                if not chunk_response:
                    print(f"[INFERENCE SERVICE] Failed to download chunk {chunk_info['index']}", flush=True)
                    return False
                chunks.append(chunk_response)
            
            zip_data = b''.join(chunks)
            print(f"[INFERENCE SERVICE] Reassembled {len(chunk_metadata['chunks'])} chunks ({len(zip_data) / 1024 / 1024:.2f} MB)", flush=True)
        else:
            # Regular single file
            zip_data = supabase.storage.from_('epsilon-models').download(storage_path)
        
        if not zip_data:
            print("[INFERENCE SERVICE] Failed to download model artifact", flush=True)
            return False
        
        # Extract to models/latest/
        models_dir = Path(__file__).parent / 'models' / 'latest'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear existing files
        if models_dir.exists():
            shutil.rmtree(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract zip
        print(f"[INFERENCE SERVICE] Extracting model to {models_dir}...", flush=True)
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zipf:
            zipf.extractall(models_dir)
        
        print(f"[INFERENCE SERVICE] Model extracted successfully", flush=True)
        
        # Update MODEL_DIR to point to extracted model
        global MODEL_DIR
        MODEL_DIR = str(models_dir / 'epsilon-20b')
        
        return True
        
    except Exception as e:
        import traceback
        print(f"[INFERENCE SERVICE] Bootstrap from Supabase failed: {e}", flush=True)
        print(f"[INFERENCE SERVICE] Traceback: {traceback.format_exc()}", flush=True)
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, tokenizer, pipe, MODEL_DIR
    
    print(f"[INFERENCE SERVICE] Starting up...", flush=True)
    
    # Try to bootstrap from Supabase first
    bootstrap_success = await bootstrap_model_from_supabase()
    
    if bootstrap_success:
        print(f"[INFERENCE SERVICE] Bootstrap successful, model directory: {MODEL_DIR}", flush=True)
        # Update MODEL_ID to point to local directory
        global MODEL_ID
        MODEL_ID = MODEL_DIR
    
    # Try to load model (will download from Hugging Face if needed)
    try:
        load_model()
        if model is not None and tokenizer is not None:
            print(f"[INFERENCE SERVICE] Model loaded successfully", flush=True)
        else:
            print(f"[INFERENCE SERVICE] ERROR: Model or tokenizer is None after load_model()", flush=True)
    except Exception as e:
        import traceback
        print(f"[INFERENCE SERVICE] ERROR: Failed to load model: {e}", flush=True)
        print(f"[INFERENCE SERVICE] Traceback: {traceback.format_exc()}", flush=True)
        print(f"[INFERENCE SERVICE] Service will start but /generate will fail until model is loaded", flush=True)
        if not bootstrap_success:
            print(f"[INFERENCE SERVICE] The model will be downloaded automatically from Hugging Face on first request", flush=True)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    repetition_penalty: float = 1.3


class GenerateResponse(BaseModel):
    text: str
    model_id: Optional[str] = None
    tokens: dict


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": model is not None and tokenizer is not None,
        "model_id": MODEL_ID,
        "model_dir": MODEL_DIR
    }


@app.get("/model-info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_id": MODEL_ID,
        "metadata": model_metadata,
        "parameters": sum(p.numel() for p in model.parameters()) if model else None,
        "harmony_format": True,
        "epsilon_identity": f"Epsilon AI - Created by {COMPANY_NAME}"
    }
    
    return info


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text using Epsilon AI with Harmony format"""
    
    if model is None or tokenizer is None or pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health endpoint.")
    
    try:
        # Format prompt using Harmony format (chat template)
        # The model expects messages in a specific format
        messages = [
            {"role": "user", "content": request.prompt}
        ]
        
        # Generate using pipeline (handles harmony format automatically)
        outputs = pipe(
            messages,  # Pass messages directly - pipeline handles harmony format
            max_new_tokens=min(request.max_new_tokens, 256),
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            do_sample=True,
            return_full_text=False
        )
        
        # Extract generated text
        generated_text = outputs[0]["generated_text"]
        
        # Clean up response
        generated_text = generated_text.strip()
        
        # Remove any "Epsilon:" prefixes if they appear (harmony format handles this)
        if generated_text.startswith("Epsilon:"):
            generated_text = generated_text[8:].strip()
        
        # Calculate tokens
        prompt_tokens = len(tokenizer.encode(request.prompt))
        completion_tokens = len(tokenizer.encode(generated_text))
        
        model_id = model_metadata.get("model_id") if model_metadata else MODEL_ID
        
        return GenerateResponse(
            text=generated_text,
            model_id=model_id,
            tokens={"prompt": prompt_tokens, "completion": completion_tokens}
        )
        
    except Exception as e:
        import traceback
        print(f"[INFERENCE SERVICE] Generation error: {e}", flush=True)
        print(f"[INFERENCE SERVICE] Traceback: {traceback.format_exc()}", flush=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


class ReloadModelRequest(BaseModel):
    model_dir: Optional[str] = None


@app.post("/reload-model")
async def reload_model(request: ReloadModelRequest):
    """Reload model"""
    global MODEL_DIR, model, tokenizer, pipe
    
    try:
        if request.model_dir:
            MODEL_DIR = request.model_dir
        
        load_model()
        
        if model is not None and tokenizer is not None:
            return {
                "status": "ok", 
                "message": f"Model reloaded from {MODEL_ID}",
                "model_loaded": True
            }
        else:
            raise HTTPException(status_code=500, detail="Model or tokenizer is None after reload")
    except Exception as e:
        import traceback
        error_detail = f"Failed to reload model: {str(e)}\n{traceback.format_exc()}"
        print(f"[INFERENCE SERVICE] {error_detail}", flush=True)
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8005))
    uvicorn.run(app, host="0.0.0.0", port=port)
