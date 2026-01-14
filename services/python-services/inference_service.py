"""
Epsilon AI Inference Service
Uses the Epsilon AI model with Harmony response format
Created by Neural Operations & Holdings LLC
"""
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
from typing import Optional, List
import torch
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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
pipe = None
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
    global model, tokenizer, pipe, model_metadata
    
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
        global model, pipe
        if model is not None:
            print(f"[INFERENCE SERVICE] Clearing existing model from memory...", flush=True)
            del model
            del pipe
            model = None
            pipe = None
        
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
        
        print(f"[INFERENCE SERVICE] Creating pipeline...", flush=True)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            dtype=torch.float16,
            device_map="auto",
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


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, tokenizer, pipe
    
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
    global model, pipe  # Allow modification of global variables
    
    if model is None or tokenizer is None or pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health endpoint.")
    
    try:
        # Format prompt with concise system instruction for Epsilon AI identity
        # Keep it simple to prevent the model from generating analysis/thinking text
        system_instruction = "You are Epsilon AI, created by Neural Operations & Holdings LLC. Respond naturally and directly to the user. Do not explain your reasoning or show your thinking process."
        
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": request.prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = f"System: {system_instruction}\n\nUser: {request.prompt}\nEpsilon AI:"
        try:
            # Prepare stop sequences to prevent analysis text generation
            stop_sequences = []
            if request.stop and len(request.stop) > 0:
                stop_sequences.extend(request.stop)
            # Add stop sequences that prevent analysis text
            analysis_stops = ["We have to", "We should", "We need to", "So we can", "per developer", "Ensure no", "Ok."]
            stop_sequences.extend(analysis_stops)
            
            # Tokenize stop sequences
            stop_token_ids = []
            for stop_seq in stop_sequences:
                stop_tokens = tokenizer.encode(stop_seq, add_special_tokens=False)
                if len(stop_tokens) > 0:
                    stop_token_ids.append(stop_tokens[0])  # Use first token of stop sequence
            
            outputs = pipe(
                formatted_prompt,
                max_new_tokens=min(request.max_new_tokens, 512),  # Increased limit
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                do_sample=True,
                return_full_text=False,
                pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
                # Use eos_token_id as stop if no custom stops, otherwise use stop_token_ids
                eos_token_id=tokenizer.eos_token_id if not stop_token_ids else None
            )
            
        except RuntimeError as e:
            if "dtype" in str(e).lower() or "scalar type" in str(e).lower():
                print(f"[INFERENCE SERVICE] Dtype mismatch detected, attempting to convert model to float16...", flush=True)
                if hasattr(model, 'to'):
                    try:
                        model = model.to(torch.float16)
                        pipe = pipeline(
                            "text-generation",
                            model=model,
                            tokenizer=tokenizer,
                            dtype=torch.float16,
                            device_map="auto",
                        )
                        outputs = pipe(
                            formatted_prompt,
                            max_new_tokens=min(request.max_new_tokens, 512),
                            temperature=request.temperature,
                            top_p=request.top_p,
                            repetition_penalty=request.repetition_penalty,
                            do_sample=True,
                            return_full_text=False,
                            pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
                        )
                    except Exception as conv_error:
                        print(f"[INFERENCE SERVICE] Failed to convert model dtype: {conv_error}", flush=True)
                        raise e  # Re-raise original error
                else:
                    raise e
            else:
                raise e
        
        # Extract generated text - handle different output formats
        if isinstance(outputs, list) and len(outputs) > 0:
            if isinstance(outputs[0], dict):
                generated_text = outputs[0].get("generated_text", "").strip()
            else:
                generated_text = str(outputs[0]).strip()
        else:
            generated_text = str(outputs).strip()
        
        analysis_markers = ["Ok.", "Okay.", "So,", "Therefore,", "Thus,", "Hey there!", "Hi!", "Hello!"]
        for marker in analysis_markers:
            if marker in generated_text:
                marker_pos = generated_text.find(marker)
                if marker_pos != -1:
                    after_marker = generated_text[marker_pos + len(marker):].strip()
                    if len(after_marker) > 10 and after_marker[0].isalpha():
                        generated_text = after_marker
                        break
        
        if "User says" in generated_text or "They want" in generated_text or "We need respond" in generated_text:
            sentences = generated_text.split('.')
            filtered_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if any(pattern in sentence.lower() for pattern in ['user says', 'they want', 'we need', 'just respond', 'no explanation', 'provide', 'as per instructions', 'developer role', 'system instruction', 'hierarchy', 'conflict', 'potential confusion', 'what do they', 'likely they']):
                    continue
                filtered_sentences.append(sentence)
            if filtered_sentences:
                generated_text = '. '.join(filtered_sentences).strip()
        generated_text = re.sub(r'analysis\s*', '', generated_text, flags=re.IGNORECASE)
        # Remove "assistantfinal" prefix
        generated_text = re.sub(r'assistantfinal\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'User says\s*"[^"]*"\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'User says[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'The user says\s*"[^"]*"\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'The user says[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'They want[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'Just respond[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'No explanation[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'Provide[^.]*answer[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'We need respond as per instructions[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'as per instructions[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'from developer role[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'Also we have system instruction[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'but dev overrides[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'The hierarchy[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'if there\'s conflict[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'we follow higher priority[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'Here there is no explicit contradiction[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'So I should answer[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'Provide concise info[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'maybe mention[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'However developer says[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'There\'s potential confusion[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'what do they expect[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'Likely they\'d want[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        # Remove "We should respond" analysis patterns
        generated_text = re.sub(r'We should respond[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        # Remove "We need to" analysis patterns
        generated_text = re.sub(r'We need to[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        # Remove "We have to" analysis patterns
        generated_text = re.sub(r'We have to[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        # Remove instruction patterns like "The instruction:"
        generated_text = re.sub(r'The instruction:[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        # Remove "So reply with" patterns
        generated_text = re.sub(r'So reply with[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        # Remove "So we can say" patterns
        generated_text = re.sub(r'So we can say[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        # Remove "per developer instruction" patterns
        generated_text = re.sub(r'per developer instruction[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        # Remove "Ensure no references" patterns
        generated_text = re.sub(r'Ensure no references[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        # Remove "Ok\." patterns (analysis conclusion)
        generated_text = re.sub(r'Ok\.\s*', '', generated_text, flags=re.IGNORECASE)
        # Remove "Let's" analysis patterns
        generated_text = re.sub(r"Let's[^.]*\.\s*", '', generated_text, flags=re.IGNORECASE)
        # Remove entire analysis blocks that start with "We have to respond" or similar
        # This catches multi-sentence analysis blocks - be very aggressive
        generated_text = re.sub(r'We (have to|should|need to) respond[^.]*\.\s*(We (should|can|need to))?[^.]*\.\s*(So (we can say|reply with))?[^.]*\.\s*(Ensure|Ok\.)?[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        # Remove "We have to respond as" patterns (common analysis start)
        generated_text = re.sub(r'We have to respond as[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        # Remove "We should greet them" patterns
        generated_text = re.sub(r'We should greet them[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        # Remove "introduce ourselves per developer instruction" patterns
        generated_text = re.sub(r'introduce ourselves per developer instruction[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        # Remove "Identify as Epsilon AI" analysis patterns
        generated_text = re.sub(r'Identify as Epsilon AI[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        # Remove "never mention other AIs" analysis patterns
        generated_text = re.sub(r'never mention other AIs[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        # Remove "So we can say something like" patterns (common analysis)
        generated_text = re.sub(r'So we can say something like[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        # Remove quoted examples in analysis (e.g., "Hello! I'm Epsilon AI...")
        generated_text = re.sub(r'"[^"]*Epsilon AI[^"]*"[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        # Remove analysis blocks that mention "per developer instruction" or "developer instruction"
        generated_text = re.sub(r'[^.]*per developer instruction[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'[^.]*developer instruction[^.]*\.\s*', '', generated_text, flags=re.IGNORECASE)
        
        # Remove any "Epsilon:" or "Epsilon AI:" prefixes if they appear
        generated_text = re.sub(r'^Epsilon\s*(AI)?:\s*', '', generated_text, flags=re.IGNORECASE)
        
        # Remove the original prompt if it appears in the output
        if formatted_prompt in generated_text:
            generated_text = generated_text.replace(formatted_prompt, "").strip()
        
        # Remove mentions of ChatGPT/OpenAI and replace with Epsilon AI
        generated_text = re.sub(r'\bChatGPT\b', 'Epsilon AI', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'\bChat-GPT\b', 'Epsilon AI', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'\bOpenAI\b', 'Neural Operations & Holdings LLC', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'\bGPT\b(?=\s+architecture)', 'Epsilon', generated_text, flags=re.IGNORECASE)
        
        # Clean up any extra whitespace
        generated_text = re.sub(r'\s+', ' ', generated_text).strip()
        
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
    global model, tokenizer, pipe
    
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
