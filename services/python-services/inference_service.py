"""
Epsilon AI Inference Service
Uses the Epsilon AI model with Harmony response format
Created by Neural Operations & Holdings LLC
"""
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
from typing import Optional, List
import re
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from filelock import FileLock
from huggingface_hub import snapshot_download

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
model_metadata = None

# Import model configuration
try:
    from model_config import HF_MODEL_ID, MODEL_NAME, COMPANY_NAME
except ImportError:
    # Fallback if model_config.py is not in the same directory
    import sys
    from pathlib import Path
    config_path = Path(__file__).parent
    if str(config_path) not in sys.path:
        sys.path.insert(0, str(config_path))
    from model_config import HF_MODEL_ID, MODEL_NAME, COMPANY_NAME

# Model configuration
MODEL_ID = os.getenv('EPSILON_MODEL_ID', HF_MODEL_ID)
# Use /workspace for model storage (usually has more space than /root)
MODEL_DIR = os.getenv('EPSILON_MODEL_DIR', str(Path('/workspace/models/epsilon-20b')))


def load_model():
    """Load Epsilon AI model using transformers - optimized for GPU"""
    global model, tokenizer, model_metadata
    
    # Use file lock to prevent concurrent downloads from multiple workers
    lock_dir = Path(__file__).parent / '.cursor'
    lock_dir.mkdir(exist_ok=True)
    lock_path = str(lock_dir / "hf_download.lock")
    
    print(f"[INFERENCE SERVICE] Loading Epsilon AI model: {MODEL_ID}", flush=True)
    
    try:
        # Check disk space before downloading
        import shutil
        disk_usage = shutil.disk_usage(MODEL_DIR if Path(MODEL_DIR).exists() else Path(__file__).parent)
        free_gb = disk_usage.free / (1024**3)
        print(f"[INFERENCE SERVICE] Available disk space: {free_gb:.2f} GB", flush=True)
        if free_gb < 50:
            print(f"[INFERENCE SERVICE] WARNING: Low disk space ({free_gb:.2f} GB). Model requires ~40GB.", flush=True)
        
        # Clear CUDA cache aggressively before loading
        if torch.cuda.is_available():
            # Clear all GPU memory
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            # Check current GPU memory usage
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free = total - reserved
            
            print(f"[INFERENCE SERVICE] GPU memory - Total: {total:.2f} GB, Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Free: {free:.2f} GB", flush=True)
            
            # If GPU is mostly full, something else is using it
            if reserved > total * 0.9:
                print(f"[INFERENCE SERVICE] WARNING: GPU memory is {reserved:.2f} GB / {total:.2f} GB reserved. Clearing...", flush=True)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()
                torch.cuda.empty_cache()
        
        # Clear Hugging Face cache in home directory (but keep MODEL_DIR cache for faster re-downloads)
        hub_dir = Path.home() / ".cache" / "huggingface"
        if hub_dir.exists():
            print(f"[INFERENCE SERVICE] Clearing Hugging Face cache in home directory to free disk space...", flush=True)
            import shutil
            try:
                # Get size before deletion
                cache_size = sum(f.stat().st_size for f in hub_dir.rglob('*') if f.is_file()) / (1024**3)
                shutil.rmtree(hub_dir)
                print(f"[INFERENCE SERVICE] Cleared {cache_size:.2f} GB of Hugging Face cache", flush=True)
            except Exception as e:
                print(f"[INFERENCE SERVICE] Warning: Could not clear cache: {e}", flush=True)
        
        # Also remove .cache folder from MODEL_DIR if it exists (created by snapshot_download)
        model_cache_dir = Path(MODEL_DIR) / ".cache"
        if model_cache_dir.exists():
            print(f"[INFERENCE SERVICE] Removing .cache folder from model directory...", flush=True)
            import shutil
            try:
                shutil.rmtree(model_cache_dir)
                print(f"[INFERENCE SERVICE] Removed .cache folder from {MODEL_DIR}", flush=True)
            except Exception as e:
                print(f"[INFERENCE SERVICE] Warning: Could not remove model cache: {e}", flush=True)
        
        # Acquire lock to prevent concurrent downloads
        with FileLock(lock_path, timeout=60 * 60):  # 1 hour timeout
            # 1) Download snapshot to a stable local directory ONCE
            Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
            
            # Check if model already exists locally - count safetensors files and verify sizes
            safetensors_files = list(Path(MODEL_DIR).glob("*.safetensors"))
            if safetensors_files and len(safetensors_files) >= 2:
                # Verify files are complete (each should be ~4-5GB)
                complete_files = [f for f in safetensors_files if f.stat().st_size > 4 * 1024 * 1024 * 1024]  # > 4GB
                if len(complete_files) >= 2:
                    total_size = sum(f.stat().st_size for f in complete_files) / (1024**3)
                    print(f"[INFERENCE SERVICE] Model files found locally ({len(complete_files)} complete safetensors, {total_size:.2f} GB), skipping download...", flush=True)
                    local_path = MODEL_DIR
                else:
                    print(f"[INFERENCE SERVICE] Found {len(safetensors_files)} safetensors but only {len(complete_files)} are complete, re-downloading...", flush=True)
                    # Remove ALL files in MODEL_DIR to start fresh (incomplete files)
                    import shutil
                    try:
                        shutil.rmtree(MODEL_DIR)
                        MODEL_DIR.mkdir(parents=True, exist_ok=True)
                        print(f"[INFERENCE SERVICE] Cleared incomplete model directory", flush=True)
                    except Exception as e:
                        print(f"[INFERENCE SERVICE] Warning: Could not clear model directory: {e}", flush=True)
                    local_path = snapshot_download(
                        repo_id=MODEL_ID,
                        local_dir=MODEL_DIR,
                        local_dir_use_symlinks=False,
                        max_workers=1,
                        ignore_patterns=[".cache/**"],  # Ignore cache folder
                    )
                    print(f"[INFERENCE SERVICE] Model snapshot downloaded to: {local_path}", flush=True)
            else:
                print(f"[INFERENCE SERVICE] Downloading model snapshot to local directory (this may take 10-15 minutes)...", flush=True)
                local_path = snapshot_download(
                    repo_id=MODEL_ID,
                    local_dir=MODEL_DIR,
                    local_dir_use_symlinks=False,
                    max_workers=1,  # Single worker to prevent concurrent downloads
                    ignore_patterns=[".cache/**"],  # Ignore cache folder
                )
                print(f"[INFERENCE SERVICE] Model snapshot downloaded to: {local_path}", flush=True)
        
        print(f"[INFERENCE SERVICE] Loading tokenizer from local path: {local_path}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            local_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        print(f"[INFERENCE SERVICE] Loading model on GPU from local path: {local_path}", flush=True)
        
        # Clear any existing model from memory first
        global model
        if model is not None:
            print(f"[INFERENCE SERVICE] Clearing existing model from memory...", flush=True)
            del model
            model = None
        
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free = total - reserved
            print(f"[INFERENCE SERVICE] GPU memory before load - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Free: {free:.2f} GB, Total: {total:.2f} GB", flush=True)
            
            if free < 1.0:
                print(f"[INFERENCE SERVICE] Warning: Low free memory ({free:.2f} GB), attempting aggressive cleanup...", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                free = total - (torch.cuda.memory_reserved(0) / (1024**3))
                print(f"[INFERENCE SERVICE] After cleanup - Free: {free:.2f} GB", flush=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True,
            max_memory={0: "42GB"}
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
        
        # Log tokenizer special tokens for Harmony format debugging
        print(f"[INFERENCE SERVICE] EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})", flush=True)
        if hasattr(tokenizer, 'additional_special_tokens') and tokenizer.additional_special_tokens:
            print(f"[INFERENCE SERVICE] Additional special tokens: {tokenizer.additional_special_tokens}", flush=True)
        if hasattr(tokenizer, 'special_tokens_map'):
            print(f"[INFERENCE SERVICE] Special tokens map: {tokenizer.special_tokens_map}", flush=True)
        
    except Exception as e:
        import traceback
        print(f"[INFERENCE SERVICE] ERROR: Failed to load model: {e}", flush=True)
        print(f"[INFERENCE SERVICE] Traceback: {traceback.format_exc()}", flush=True)
        # Don't raise - allow service to start even if model loading fails
        # The /generate endpoint will return 503 if model is not loaded
        model = None
        tokenizer = None
        model_metadata = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, tokenizer
    
    print(f"[INFERENCE SERVICE] Starting up...", flush=True)
    
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
    global model  # Allow modification of global variables
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health endpoint.")
    
    try:
        # Harmony format: Use chat template exactly as model was trained
        # The model was trained with harmony_format, so we must use the tokenizer's chat template
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            # Harmony format uses system + user messages, chat template handles formatting
            messages = [
                {"role": "system", "content": "You are Epsilon AI, created by Neural Operations & Holdings LLC."},
                {"role": "user", "content": request.prompt}
            ]
            # add_generation_prompt=True ensures proper harmony format response generation
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print(f"[INFERENCE SERVICE] Harmony format prompt (first 200 chars): {formatted_prompt[:200]}", flush=True)
        else:
            # Fallback if chat template not available - use harmony-style format
            formatted_prompt = f"User: {request.prompt}\nEpsilon AI: "
            print(f"[INFERENCE SERVICE] Using fallback format (no chat template): {formatted_prompt[:200]}", flush=True)
        
        # Harmony format stopping criteria - stop at markers like "assistantfinal"
        # Only use specific boundary markers, not generic words like "analysis" or "final" that can appear in normal responses
        harmony_stop_markers = [
            "assistantfinal",
            "Assistantfinal",
            "ASSISTANTFINAL",
            "\nassistantfinal",
            "\nAssistantfinal",
            "assistant_final",
            "Assistant_final",
            "ASSISTANT_FINAL",
            "\nassistant_final",
            "\nAssistant_final",
            "<final>",
            "</final>"
        ]
        
        # Markdown-style markers with strict line-boundary matching (regex patterns)
        harmony_stop_regex = [
            r"(^|\n)##\s*final\b",
            r"(^|\n)###\s*final\b"
        ]
        
        # Check for end-of-turn tokens (many chat models use special EOT tokens)
        eot_token_id = None
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            eot_token_id = tokenizer.eos_token_id
        if hasattr(tokenizer, 'additional_special_tokens') and tokenizer.additional_special_tokens:
            for special_token in tokenizer.additional_special_tokens:
                if 'eot' in special_token.lower() or 'end_of_turn' in special_token.lower():
                    eot_token_id = tokenizer.convert_tokens_to_ids(special_token)
                    print(f"[INFERENCE SERVICE] Found EOT token: {special_token} (ID: {eot_token_id})", flush=True)
                    break
        
        # Use standard temperature (0.7) for natural responses
        gen_temperature = request.temperature if request.temperature > 0 else 0.7
        
        # Tokenize input properly
        tokenized = tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        prompt_len_tokens = input_ids.shape[1]
        
        # Get device from model (handle device_map="auto" correctly)
        # Prefer first CUDA device for sharded models, fallback to CPU if needed
        device = None
        if hasattr(model, 'device'):
            device = model.device
        elif hasattr(model, 'hf_device_map') and model.hf_device_map:
            # For sharded models, pick the first CUDA device
            for d in model.hf_device_map.values():
                if isinstance(d, str) and d.startswith("cuda"):
                    device = torch.device(d)
                    break
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Create stopping criteria class for Harmony format markers
        class HarmonyStoppingCriteria(StoppingCriteria):
            def __init__(self, prompt_len_tokens: int, tokenizer, markers, regex_patterns, window_tokens: int = 80):
                self.prompt_len = prompt_len_tokens
                self.tok = tokenizer
                self.markers = [m.lower() for m in markers]
                self.regex_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in regex_patterns]
                self.window = window_tokens
            
            def __call__(self, input_ids, scores, **kwargs):
                # Get only generated tokens (after prompt)
                gen_ids = input_ids[0, self.prompt_len:]
                if gen_ids.numel() == 0:
                    return False
                
                # Only decode the last window of tokens (efficient)
                tail = gen_ids[-self.window:] if gen_ids.shape[0] > self.window else gen_ids
                text = self.tok.decode(tail, skip_special_tokens=False).lower()
                
                # Check for simple string markers
                if any(m in text for m in self.markers):
                    return True
                
                # Check for regex patterns (markdown markers at line boundaries)
                for pattern in self.regex_patterns:
                    if pattern.search(text):
                        return True
                
                return False
        
        # Create stopping criteria
        # max_new_tokens already provides a safety limit, so stopping criteria is the primary control
        stopping_criteria = StoppingCriteriaList([
            HarmonyStoppingCriteria(prompt_len_tokens, tokenizer, harmony_stop_markers, harmony_stop_regex)
        ])
        
        # Add EOT token to eos_token_id if found
        eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        if eot_token_id is not None and eot_token_id != eos_token_id:
            # Use EOT as the primary stop token
            eos_token_id = eot_token_id
            print(f"[INFERENCE SERVICE] Using EOT token ID {eot_token_id} as eos_token_id", flush=True)
        
        try:
            # Use model.generate() directly with stopping criteria for proper Harmony format handling
            # max_new_tokens provides a safety limit (belt-and-suspenders) in case stopping criteria never triggers
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=min(request.max_new_tokens, 512),  # Safety limit: prevents infinite generation
                    temperature=gen_temperature,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id,
                    eos_token_id=eos_token_id,
                    stopping_criteria=stopping_criteria  # Primary stopping mechanism
                )
            
            # Decode only the newly generated tokens (not the prompt)
            generated_tokens = generated_ids[0, prompt_len_tokens:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"[INFERENCE SERVICE] Generated text (first 300 chars): {generated_text[:300]}", flush=True)
        except RuntimeError as e:
            if "dtype" in str(e).lower() or "scalar type" in str(e).lower():
                print(f"[INFERENCE SERVICE] Dtype mismatch detected, attempting to convert model to float16...", flush=True)
                if hasattr(model, 'to'):
                    try:
                        model = model.to(torch.float16)
                        # Retry generation with converted model
                        with torch.no_grad():
                            generated_ids = model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=min(request.max_new_tokens, 512),
                                temperature=gen_temperature,
                                top_p=request.top_p,
                                repetition_penalty=request.repetition_penalty,
                                do_sample=True,
                                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id,
                                eos_token_id=eos_token_id,
                                stopping_criteria=stopping_criteria
                            )
                        generated_tokens = generated_ids[0, prompt_len_tokens:]
                        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        print(f"[INFERENCE SERVICE] Generated text after dtype conversion (first 300 chars): {generated_text[:300]}", flush=True)
                    except Exception as conv_error:
                        print(f"[INFERENCE SERVICE] Failed to convert model dtype: {conv_error}", flush=True)
                        raise e  # Re-raise original error
                else:
                    raise e
            else:
                raise e
        
        generated_text = generated_text.strip()
        
        # Harmony format cleanup: Extract only the final response after markers
        # The stopping criteria should have stopped generation, but if markers appear, extract final section
        # Only use specific boundary markers, not generic words that can appear in normal responses
        harmony_markers = [
            "assistantfinal", "Assistantfinal", "ASSISTANTFINAL",
            "assistant_final", "Assistant_final", "ASSISTANT_FINAL",
            "<final>", "</final>"
        ]
        text_lower = generated_text.lower()
        
        # If any Harmony marker exists, extract everything after it
        for marker in harmony_markers:
            marker_pos = text_lower.find(marker.lower())
            if marker_pos != -1:
                # Extract text after marker
                after_marker = generated_text[marker_pos + len(marker):].strip()
                # Remove any leading punctuation/whitespace
                after_marker = after_marker.lstrip('.,;:!? \n\t')
                if len(after_marker) > 0:
                    generated_text = after_marker
                    print(f"[INFERENCE SERVICE] Extracted response after Harmony marker '{marker}'", flush=True)
                    break
        
        # Check for regex patterns (markdown markers at line boundaries only)
        harmony_marker_regex = [
            re.compile(r"(^|\n)##\s*final\b", re.IGNORECASE),
            re.compile(r"(^|\n)###\s*final\b", re.IGNORECASE)
        ]
        for pattern in harmony_marker_regex:
            match = pattern.search(generated_text)
            if match:
                # Extract text after the matched marker
                after_marker = generated_text[match.end():].strip()
                after_marker = after_marker.lstrip('.,;:!? \n\t')
                if len(after_marker) > 0:
                    generated_text = after_marker
                    print(f"[INFERENCE SERVICE] Extracted response after Harmony regex marker", flush=True)
                    break
        
        # Remove any remaining analysis prefixes at the start (defensive)
        analysis_prefixes = ["This is Epsilon AI", "We have", "The user", "User says", "They want", "We should", "We need"]
        text_lower = generated_text.lower().strip()
        for prefix in analysis_prefixes:
            if text_lower.startswith(prefix.lower()):
                # Find common response starters
                response_starters = ["Hi", "Hello", "Hey", "I'm", "I am", "Sure", "Yes", "No", "Here", "I", "Let"]
                for starter in response_starters:
                    pos = generated_text.find(starter)
                    if pos > 0 and pos < 300:
                        generated_text = generated_text[pos:].strip()
                        print(f"[INFERENCE SERVICE] Cleaned analysis prefix, response starts at pos {pos}", flush=True)
                        break
                break
        
        # Calculate tokens
        prompt_tokens = len(tokenizer.encode(request.prompt))
        completion_tokens = len(tokenizer.encode(generated_text)) if generated_text else 0
        
        model_id = model_metadata.get("model_id") if model_metadata else MODEL_ID
        
        return GenerateResponse(
            text=generated_text,
            model_id=model_id,
            tokens={"prompt": prompt_tokens, "completion": completion_tokens}
        )
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"[INFERENCE SERVICE] Generation error: {error_msg}", flush=True)
        print(f"[INFERENCE SERVICE] Traceback: {traceback_str}", flush=True)
        
        # Provide more helpful error messages
        if "dtype" in error_msg.lower() or "scalar type" in error_msg.lower():
            error_detail = f"Model dtype mismatch. Please ensure model and pipeline use compatible dtypes. Original error: {error_msg}"
        elif "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
            error_detail = f"GPU memory error. Try reducing max_new_tokens. Original error: {error_msg}"
        else:
            error_detail = f"Generation failed: {error_msg}"
        
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/reload-model")
async def reload_model():
    """Reload model"""
    global model, tokenizer
    
    try:
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
        print(f"[INFERENCE SERVICE] {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8005))
    uvicorn.run(app, host="0.0.0.0", port=port)
