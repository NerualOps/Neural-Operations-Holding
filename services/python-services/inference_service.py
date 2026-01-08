"""
Epsilon AI Inference Service
FastAPI service for model inference (NO TRAINING)
"""
import os
import sys
from pathlib import Path
from typing import Optional, List
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tokenizers import Tokenizer
import json

# Import from shared transformer core (no dependency on ml_local/)
from epsilon_transformer_core import EpsilonTransformerLM, TransformerConfig

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
model_config = None
model_metadata = None


def load_model(model_dir: str):
    """Load model from exported artifact directory"""
    global model, tokenizer, model_config, model_metadata
    
    # Initialize tokenizer to None if not set
    if 'tokenizer' not in globals():
        tokenizer = None
    
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load config
    config_path = model_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    model_config = TransformerConfig.from_dict(config_dict)
    
    # Load tokenizer
    tokenizer_path = model_dir / 'tokenizer.json'
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Load model weights first to check actual dimensions
    model_path = model_dir / 'model.pt'
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # If checkpoint is a dict with 'model' or 'state_dict', extract it
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Infer actual model dimensions from weights
    if 'token_embedding.weight' in state_dict:
        actual_vocab_size = state_dict['token_embedding.weight'].shape[0]
        actual_d_model = state_dict['token_embedding.weight'].shape[1]
        
        # Infer n_heads from attention weights
        actual_n_heads = model_config.n_heads
        if 'blocks.0.attn.q_proj.weight' in state_dict:
            q_weight = state_dict['blocks.0.attn.q_proj.weight']
            # q_proj should be (d_model, d_model), and we can infer head_dim from RoPE if present
            if 'blocks.0.attn.rope.cos_cached' in state_dict:
                rope_dim = state_dict['blocks.0.attn.rope.cos_cached'].shape[-1]
                # rope_dim should equal head_dim, and head_dim = d_model / n_heads
                inferred_n_heads = actual_d_model // rope_dim
                if inferred_n_heads > 0 and inferred_n_heads <= 32:
                    actual_n_heads = inferred_n_heads
                    print(f"[INFERENCE SERVICE] Inferred n_heads={actual_n_heads} from RoPE dimension")
        
        # Update config if needed
        if model_config.vocab_size != actual_vocab_size:
            print(f"[INFERENCE SERVICE] WARNING: Config vocab_size ({model_config.vocab_size}) doesn't match model ({actual_vocab_size}). Updating config.")
            model_config.vocab_size = actual_vocab_size
        
        if model_config.d_model != actual_d_model:
            print(f"[INFERENCE SERVICE] WARNING: Config d_model ({model_config.d_model}) doesn't match model ({actual_d_model}). Updating config.")
            model_config.d_model = actual_d_model
        
        if model_config.n_heads != actual_n_heads:
            print(f"[INFERENCE SERVICE] WARNING: Config n_heads ({model_config.n_heads}) doesn't match model ({actual_n_heads}). Updating config.")
            model_config.n_heads = actual_n_heads
    
    # Create model with potentially updated config
    model = EpsilonTransformerLM(model_config)
    
    # Load weights with strict=False to handle minor mismatches
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"[INFERENCE SERVICE] Model weights loaded successfully (strict mode)")
    except RuntimeError as e:
        print(f"[INFERENCE SERVICE] WARNING: Strict loading failed: {e}")
        print(f"[INFERENCE SERVICE] Attempting non-strict loading...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"[INFERENCE SERVICE] Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"[INFERENCE SERVICE] Unexpected keys: {unexpected_keys[:5]}...")
    
    model.eval()
    
    # Load metadata if available
    metadata_path = model_dir / 'run_meta.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            model_metadata = json.load(f)
    else:
        model_metadata = None
    
    print(f"[INFERENCE SERVICE] Model loaded from {model_dir}")
    print(f"[INFERENCE SERVICE] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return True


async def bootstrap_model_from_supabase():
    """On startup, download approved production model from Supabase if available"""
    try:
        from supabase import create_client, Client
        
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        
        if not supabase_url or not supabase_key:
            print("[INFERENCE SERVICE] Supabase credentials not set - skipping model bootstrap", flush=True)
            return False
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Get latest approved model (auto-uploaded models are auto-approved)
        response = supabase.table('epsilon_model_deployments').select(
            'id, model_id, storage_path, version'
        ).eq('status', 'approved').order(
            'approved_at', desc=True
        ).limit(1).execute()
        
        if not response.data or len(response.data) == 0:
            print("[INFERENCE SERVICE] No approved production model found in Supabase", flush=True)
            return False
        
        deployment = response.data[0]
        storage_path = deployment.get('storage_path')
        
        if not storage_path:
            print("[INFERENCE SERVICE] Approved deployment has no storage_path", flush=True)
            return False
        
        # Handle chunked files (metadata.json) or regular zip files
        zip_data = None
        
        if storage_path.endswith('.metadata.json'):
            # Chunked file - download metadata first, then reassemble chunks
            print(f"[INFERENCE SERVICE] Detected chunked model file, downloading chunks...", flush=True)
            
            metadata_response = supabase.storage.from_('epsilon-models').download(storage_path)
            if not metadata_response:
                print("[INFERENCE SERVICE] Failed to download chunk metadata", flush=True)
                return False
            
            import json
            chunk_metadata = json.loads(metadata_response.decode('utf-8'))
            
            # Download and reassemble chunks
            chunks = []
            for chunk_info in sorted(chunk_metadata['chunks'], key=lambda x: x['index']):
                print(f"[INFERENCE SERVICE] Downloading chunk {chunk_info['index'] + 1}/{len(chunk_metadata['chunks'])}...", flush=True)
                chunk_response = supabase.storage.from_('epsilon-models').download(chunk_info['path'])
                if not chunk_response:
                    print(f"[INFERENCE SERVICE] Failed to download chunk {chunk_info['index']}", flush=True)
                    return False
                chunks.append(chunk_response)
            
            # Combine chunks
            zip_data = b''.join(chunks)
            print(f"[INFERENCE SERVICE] Reassembled {len(chunk_metadata['chunks'])} chunks ({len(zip_data) / 1024 / 1024:.2f} MB)", flush=True)
        else:
            # Regular single file
            print(f"[INFERENCE SERVICE] Downloading model artifact from Supabase: {storage_path}", flush=True)
            zip_data = supabase.storage.from_('epsilon-models').download(storage_path)
        
        if not zip_data:
            print("[INFERENCE SERVICE] Failed to download model artifact", flush=True)
            return False
        
        # Extract to models/latest/
        models_dir = Path(__file__).parent / 'models' / 'latest'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        from zipfile import ZipFile
        import io
        
        if isinstance(zip_data, bytes):
            zip_buffer = io.BytesIO(zip_data)
        else:
            zip_buffer = io.BytesIO(zip_data)
        
        with ZipFile(zip_buffer, 'r') as zip_ref:
            zip_ref.extractall(models_dir)
        
        # Rename model file if needed (handle different naming conventions)
        model_files = list(models_dir.glob('*.pt'))
        if model_files and not (models_dir / 'model.pt').exists():
            model_files[0].rename(models_dir / 'model.pt')
            print(f"[INFERENCE SERVICE] Renamed {model_files[0].name} to model.pt", flush=True)
        
        print(f"[INFERENCE SERVICE] Model artifact extracted to {models_dir}", flush=True)
        
        # Verify required files exist
        required_files = ['model.pt', 'config.json', 'tokenizer.json']
        missing_files = [f for f in required_files if not (models_dir / f).exists()]
        if missing_files:
            print(f"[INFERENCE SERVICE] WARNING: Missing required files after extraction: {missing_files}", flush=True)
            # List what we actually have
            existing_files = list(models_dir.glob('*'))
            print(f"[INFERENCE SERVICE] Files found: {[f.name for f in existing_files]}", flush=True)
            return False
        
        print(f"[INFERENCE SERVICE] All required files verified", flush=True)
        return True
        
    except Exception as e:
        import traceback
        print(f"[INFERENCE SERVICE] Model bootstrap from Supabase failed: {e}", flush=True)
        print(f"[INFERENCE SERVICE] Traceback: {traceback.format_exc()}", flush=True)
        return False


# Load model on startup
MODEL_DIR = os.getenv('EPSILON_MODEL_DIR', str(Path(__file__).parent / 'models' / 'latest'))

# Bootstrap and load model on FastAPI startup (not at module level to avoid event loop issues)
@app.on_event("startup")
async def startup_event():
    """Bootstrap model from Supabase and load it on startup"""
    global MODEL_DIR, model, tokenizer
    
    print(f"[INFERENCE SERVICE] Starting up...", flush=True)
    
    # Try to bootstrap from Supabase
    bootstrap_success = await bootstrap_model_from_supabase()
    
    if bootstrap_success:
        MODEL_DIR = str(Path(__file__).parent / 'models' / 'latest')
        print(f"[INFERENCE SERVICE] Bootstrap successful, model directory: {MODEL_DIR}", flush=True)
    
    # Load model if directory exists
    if os.path.exists(MODEL_DIR):
        try:
            load_model(MODEL_DIR)
            if model is not None and tokenizer is not None:
                print(f"[INFERENCE SERVICE] Model loaded successfully", flush=True)
            else:
                print(f"[INFERENCE SERVICE] ERROR: Model or tokenizer is None after load_model()", flush=True)
        except Exception as e:
            import traceback
            print(f"[INFERENCE SERVICE] ERROR: Failed to load model: {e}", flush=True)
            print(f"[INFERENCE SERVICE] Traceback: {traceback.format_exc()}", flush=True)
            print(f"[INFERENCE SERVICE] Service will start but /generate will fail until model is loaded", flush=True)
    else:
        print(f"[INFERENCE SERVICE] WARNING: Model directory not found: {MODEL_DIR}", flush=True)
        print(f"[INFERENCE SERVICE] Bootstrap success was: {bootstrap_success}", flush=True)
        print(f"[INFERENCE SERVICE] Set EPSILON_MODEL_DIR environment variable or place model in models/latest", flush=True)
        print(f"[INFERENCE SERVICE] You can trigger a reload via POST /reload-model to bootstrap from Supabase", flush=True)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None  # Changed from stop_sequences to stop


class GenerateResponse(BaseModel):
    text: str
    model_id: Optional[str] = None
    tokens: dict  # {"prompt": int, "completion": int}


@app.get("/")
@app.head("/")
async def root():
    """Root endpoint for health checks"""
    return {
        "status": "ok",
        "service": "Epsilon AI Inference Service",
        "model_loaded": model is not None,
        "model_dir": MODEL_DIR
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_dir": MODEL_DIR
    }


@app.get("/model-info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "config": model_config.to_dict() if model_config else None,
        "metadata": model_metadata,
        "parameters": sum(p.numel() for p in model.parameters()) if model else None
    }
    
    return info


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from prompt"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health endpoint.")
    
    try:
        # Encode prompt
        encoded = tokenizer.encode(request.prompt)
        prompt_token_count = len(encoded.ids)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long)
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=50  # Default top_k
            )
        
        # Decode generated tokens (only new tokens)
        generated_ids = generated[0][prompt_token_count:].cpu().tolist()
        generated_text = tokenizer.decode(generated_ids)
        
        # Apply stop sequences if provided
        if request.stop:
            for stop_seq in request.stop:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
                    break
        
        # Extract model_id from metadata
        model_id = None
        if model_metadata:
            model_id = model_metadata.get('model_id') or model_metadata.get('git_commit', 'unknown')
        
        return GenerateResponse(
            text=generated_text,
            model_id=model_id,
            tokens={
                "prompt": prompt_token_count,
                "completion": len(generated_ids)
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


class ReloadModelRequest(BaseModel):
    model_dir: Optional[str] = None


@app.post("/reload-model")
async def reload_model(request: ReloadModelRequest):
    """Reload model from directory or bootstrap from Supabase (admin endpoint)"""
    global MODEL_DIR, model, tokenizer
    
    # If no model_dir provided and model not loaded, try to bootstrap from Supabase
    if not request.model_dir and (model is None or tokenizer is None):
        print("[INFERENCE SERVICE] Model not loaded, attempting to bootstrap from Supabase...", flush=True)
        bootstrap_success = await bootstrap_model_from_supabase()
        if bootstrap_success:
            MODEL_DIR = str(Path(__file__).parent / 'models' / 'latest')
            print(f"[INFERENCE SERVICE] Bootstrap successful, model directory: {MODEL_DIR}", flush=True)
        else:
            raise HTTPException(status_code=503, detail="Failed to bootstrap model from Supabase. No model available.")
    
    if request.model_dir:
        MODEL_DIR = request.model_dir
    
    try:
        load_model(MODEL_DIR)
        if model is not None and tokenizer is not None:
            return {
                "status": "ok", 
                "message": f"Model reloaded from {MODEL_DIR}",
                "model_loaded": True
            }
        else:
            raise HTTPException(status_code=500, detail="Model or tokenizer is None after reload")
    except Exception as e:
        import traceback
        error_detail = f"Failed to reload model: {str(e)}\n{traceback.format_exc()}"
        print(f"[INFERENCE SERVICE] Reload error: {error_detail}", flush=True)
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8005))
    uvicorn.run(app, host="0.0.0.0", port=port)

