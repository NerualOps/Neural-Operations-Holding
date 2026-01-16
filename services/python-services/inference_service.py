"""
Epsilon AI Inference Service
Uses the Epsilon AI model with Harmony response format
Created by Neural Operations & Holdings LLC
"""
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
from typing import Optional, List, Any
import re
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from filelock import FileLock
from huggingface_hub import snapshot_download

app = FastAPI(title="Epsilon AI Inference Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None
model_metadata = None

try:
    from model_config import HF_MODEL_ID, MODEL_NAME, COMPANY_NAME
    try:
        from model_config import DISPLAY_MODEL_ID
    except ImportError:
        DISPLAY_MODEL_ID = MODEL_NAME
except ImportError:
    import sys
    config_path = Path(__file__).parent
    if str(config_path) not in sys.path:
        sys.path.insert(0, str(config_path))
    from model_config import HF_MODEL_ID, MODEL_NAME, COMPANY_NAME
    try:
        from model_config import DISPLAY_MODEL_ID
    except ImportError:
        DISPLAY_MODEL_ID = MODEL_NAME

HF_MODEL_ID_INTERNAL = os.getenv('EPSILON_MODEL_ID', HF_MODEL_ID)
MODEL_ID = os.getenv('EPSILON_DISPLAY_MODEL_ID', DISPLAY_MODEL_ID)
# Use /workspace for model storage (usually has more space than /root)
MODEL_DIR = Path(os.getenv('EPSILON_MODEL_DIR', '/workspace/models/epsilon-20b'))


def load_model():
    """Load Epsilon AI model using transformers - optimized for GPU"""
    global model, tokenizer, model_metadata
    
    # Use file lock to prevent concurrent downloads from multiple workers
    lock_dir = Path(__file__).parent / '.cursor'
    lock_dir.mkdir(exist_ok=True)
    lock_path = str(lock_dir / "hf_download.lock")
    
    print(f"[INFERENCE SERVICE] Loading Epsilon AI model: {MODEL_ID}", flush=True)
    print(f"[INFERENCE SERVICE] Internal model repository: {HF_MODEL_ID_INTERNAL}", flush=True)
    
    try:
        import shutil
        disk_usage = shutil.disk_usage(MODEL_DIR if MODEL_DIR.exists() else Path(__file__).parent)
        free_gb = disk_usage.free / (1024**3)
        print(f"[INFERENCE SERVICE] Available disk space: {free_gb:.2f} GB", flush=True)
        if free_gb < 50:
            print(f"[INFERENCE SERVICE] WARNING: Low disk space ({free_gb:.2f} GB). Model requires ~40GB.", flush=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            
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
            try:
                cache_size = sum(f.stat().st_size for f in hub_dir.rglob('*') if f.is_file()) / (1024**3)
                shutil.rmtree(hub_dir)
                print(f"[INFERENCE SERVICE] Cleared {cache_size:.2f} GB of Hugging Face cache", flush=True)
            except Exception as e:
                print(f"[INFERENCE SERVICE] Warning: Could not clear cache: {e}", flush=True)
        
        # Also remove .cache folder from MODEL_DIR if it exists (created by snapshot_download)
        model_cache_dir = MODEL_DIR / ".cache"
        if model_cache_dir.exists():
            print(f"[INFERENCE SERVICE] Removing .cache folder from model directory...", flush=True)
            try:
                shutil.rmtree(model_cache_dir)
                print(f"[INFERENCE SERVICE] Removed .cache folder from {MODEL_DIR}", flush=True)
            except Exception as e:
                print(f"[INFERENCE SERVICE] Warning: Could not remove model cache: {e}", flush=True)
        
        with FileLock(lock_path, timeout=60 * 60):
            Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
            
            safetensors_files = list(Path(MODEL_DIR).glob("*.safetensors"))
            if safetensors_files and len(safetensors_files) >= 2:
                complete_files = [f for f in safetensors_files if f.stat().st_size > 4 * 1024 * 1024 * 1024]
                if len(complete_files) >= 2:
                    total_size = sum(f.stat().st_size for f in complete_files) / (1024**3)
                    print(f"[INFERENCE SERVICE] Model files found locally ({len(complete_files)} complete safetensors, {total_size:.2f} GB), skipping download...", flush=True)
                    local_path = MODEL_DIR
                else:
                    print(f"[INFERENCE SERVICE] Found {len(safetensors_files)} safetensors but only {len(complete_files)} are complete, re-downloading...", flush=True)
                    try:
                        shutil.rmtree(MODEL_DIR)
                        MODEL_DIR.mkdir(parents=True, exist_ok=True)
                        print(f"[INFERENCE SERVICE] Cleared incomplete model directory", flush=True)
                    except Exception as e:
                        print(f"[INFERENCE SERVICE] Warning: Could not clear model directory: {e}", flush=True)
                    local_path = snapshot_download(
                        repo_id=HF_MODEL_ID_INTERNAL,
                        local_dir=str(MODEL_DIR),
                        local_dir_use_symlinks=False,
                        max_workers=1,
                        ignore_patterns=[".cache/**"],
                    )
                    print(f"[INFERENCE SERVICE] Model snapshot downloaded to: {local_path}", flush=True)
            else:
                print(f"[INFERENCE SERVICE] Downloading model snapshot to local directory (this may take 10-15 minutes)...", flush=True)
                local_path = snapshot_download(
                    repo_id=HF_MODEL_ID_INTERNAL,
                    local_dir=str(MODEL_DIR),
                    local_dir_use_symlinks=False,
                    max_workers=1,
                    ignore_patterns=[".cache/**"],
                )
                print(f"[INFERENCE SERVICE] Model snapshot downloaded to: {local_path}", flush=True)
        
        print(f"[INFERENCE SERVICE] Loading tokenizer from local path: {local_path}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            local_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        print(f"[INFERENCE SERVICE] Loading model on GPU from local path: {local_path}", flush=True)
        
        global model
        if model is not None:
            print(f"[INFERENCE SERVICE] Clearing existing model from memory...", flush=True)
            del model
            model = None
        
        if torch.cuda.is_available():
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
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True,
            max_memory={0: "42GB"}
        )
        
        model_metadata = {
            "model_id": MODEL_ID,
            "internal_repo": HF_MODEL_ID_INTERNAL,
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
        "model_dir": str(MODEL_DIR)
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


def clean_markdown_text(text: str) -> str:
    """
    Clean markdown formatting and convert tables to plain text.
    Removes markdown symbols and formats tables as readable text.
    """
    if not text:
        return ""
    
    lines = text.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        trimmed = line.strip()
        
        if '|' in trimmed and trimmed.count('|') >= 2:
            if trimmed.replace('|', '').replace('-', '').replace(':', '').replace(' ', '').strip():
                table_rows = []
                header_row_index = -1
                
                while i < len(lines):
                    current_line = lines[i].strip()
                    if not current_line or '|' not in current_line or current_line.count('|') < 2:
                        break
                    
                    if re.match(r'^[\|\s\-:]+$', current_line):
                        header_row_index = len(table_rows)
                        i += 1
                        continue
                    
                    cells = [c.strip() for c in current_line.split('|')]
                    if cells and cells[0] == '':
                        cells.pop(0)
                    if cells and cells[-1] == '':
                        cells.pop()
                    
                    if cells:
                        table_rows.append(cells)
                    i += 1
                
                if table_rows:
                    for row_idx, row in enumerate(table_rows):
                        if header_row_index == row_idx:
                            result.append(' | '.join(row))
                            result.append('-' * (sum(len(c) for c in row) + len(row) * 3 - 3))
                        else:
                            result.append(' | '.join(row))
                    continue
        
        result.append(line)
        i += 1
    
    text = '\n'.join(result)
    
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


def parse_harmony_response(text: str, tokenizer: Any) -> Optional[str]:
    """
    Parse Harmony format response to extract only the 'final' channel content.
    Simple single-path parser: if <|channel|> exists, extract final channel.
    Otherwise return cleaned text (Harmony tokens removed).
    NEVER returns empty if given non-empty text.
    """
    if not text:
        return ""
    
    if '<|channel|>' in text.lower():
        final_match = re.search(r'<\|channel\|>final(?:<message>)?(.*?)(?=<\|channel\|>|<\|end\|>|$)', text, re.DOTALL | re.IGNORECASE)
        if final_match:
            final_content = final_match.group(1).strip()
            final_content = final_content.lstrip(': \n\t')
            final_content = final_content.replace('<message>', '').replace('</message>', '')
            final_content = re.sub(r'^<message>', '', final_content, flags=re.IGNORECASE)
            final_content = re.sub(r'</message>$', '', final_content, flags=re.IGNORECASE)
            if '<|end|>' in final_content:
                final_content = final_content[:final_content.find('<|end|>')].strip()
            if final_content:
                print(f"[INFERENCE SERVICE] Extracted final channel ({len(final_content)} chars)", flush=True)
                return final_content
        else:
            print(f"[INFERENCE SERVICE] Harmony markers found but no final channel - returning None to use clean decode", flush=True)
            return None
    
    cleaned_text = re.sub(r'<\|start\|>', '', text)
    cleaned_text = re.sub(r'<\|message\|>', '', cleaned_text)
    cleaned_text = re.sub(r'<\|end\|>', '', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text if cleaned_text else text.strip()


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text using Epsilon AI with Harmony format"""
    global model
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health endpoint.")
    
    try:
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": "You are Epsilon AI, created by Neural Operations & Holdings LLC. Never mention ChatGPT, OpenAI, or GPT. Always identify yourself as Epsilon AI."},
                {"role": "user", "content": request.prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print(f"[INFERENCE SERVICE] Harmony format prompt (first 200 chars): {formatted_prompt[:200]}", flush=True)
        else:
            formatted_prompt = f"User: {request.prompt}\nEpsilon AI: "
            print(f"[INFERENCE SERVICE] Using fallback format (no chat template): {formatted_prompt[:200]}", flush=True)
        
        eot_token_id = None
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            eot_token_id = tokenizer.eos_token_id
        
        if hasattr(tokenizer, 'additional_special_tokens') and tokenizer.additional_special_tokens:
            for special_token in tokenizer.additional_special_tokens:
                if 'eot' in special_token.lower() or 'end_of_turn' in special_token.lower():
                    eot_token_id = tokenizer.convert_tokens_to_ids(special_token)
                    print(f"[INFERENCE SERVICE] Found EOT token in additional_special_tokens: {special_token} (ID: {eot_token_id})", flush=True)
                    break
        
        if eot_token_id is None or eot_token_id == tokenizer.eos_token_id:
            if hasattr(tokenizer, 'special_tokens_map') and tokenizer.special_tokens_map:
                for k, v in tokenizer.special_tokens_map.items():
                    if isinstance(v, str) and ("eot" in v.lower() or "end_of_turn" in v.lower()):
                        eot_token_id = tokenizer.convert_tokens_to_ids(v)
                        print(f"[INFERENCE SERVICE] Found EOT token in special_tokens_map: {k}={v} (ID: {eot_token_id})", flush=True)
                        break
        
        gen_temperature = request.temperature if request.temperature > 0 else 0.7
        
        tokenized = tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        prompt_len_tokens = input_ids.shape[1]
        
        device = None
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            cuda_devices = []
            for layer_name, d in model.hf_device_map.items():
                if isinstance(d, (str, torch.device)):
                    device_str = str(d) if isinstance(d, torch.device) else d
                    if device_str.startswith("cuda"):
                        device_obj = torch.device(device_str)
                        if device_obj not in cuda_devices:
                            cuda_devices.append(device_obj)
            if cuda_devices:
                device = cuda_devices[0]
                print(f"[INFERENCE SERVICE] Model uses device_map='auto', moving inputs to first CUDA device: {device}", flush=True)
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"[INFERENCE SERVICE] Model uses device_map='auto' but no CUDA device found, using: {device}", flush=True)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[INFERENCE SERVICE] Using device: {device}", flush=True)
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": min(request.max_new_tokens, 1024),
            "temperature": gen_temperature,
            "top_p": request.top_p,
            "repetition_penalty": request.repetition_penalty,
            "do_sample": True,
        }
        
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            print(f"[INFERENCE SERVICE] Using device_map='auto' - letting model use default EOS token", flush=True)
        else:
            eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
            gen_kwargs["eos_token_id"] = eos_token_id
            print(f"[INFERENCE SERVICE] Using eos_token_id: {tokenizer.eos_token_id}", flush=True)
        
        try:
            with torch.no_grad():
                generated_ids = model.generate(**gen_kwargs)
            
            generated_tokens = generated_ids[0, prompt_len_tokens:]
            generated_text_raw = tokenizer.decode(generated_tokens, skip_special_tokens=False)
            generated_text_clean = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            print(f"[INFERENCE SERVICE] Generated text raw (first 500 chars): {generated_text_raw[:500]}", flush=True)
            
            has_harmony = (
                '<|channel|>' in generated_text_raw.lower() or
                '<|start|>assistant' in generated_text_raw.lower() or
                '<|message|>' in generated_text_raw.lower()
            )
            
            if has_harmony:
                parsed = parse_harmony_response(generated_text_raw, tokenizer)
                if parsed is not None and len(parsed) > 0:
                    generated_text = parsed
                    print(f"[INFERENCE SERVICE] Extracted Harmony format content ({len(parsed)} chars)", flush=True)
                else:
                    generated_text = generated_text_clean
                    print(f"[INFERENCE SERVICE] No final channel found in Harmony format, using clean decoded text", flush=True)
            else:
                generated_text = generated_text_clean
            
            if not generated_text or len(generated_text.strip()) == 0:
                generated_text = generated_text_clean
                print(f"[INFERENCE SERVICE] Generated text empty, using clean decoded text", flush=True)
            
            if not generated_text or len(generated_text.strip()) == 0:
                generated_text = re.sub(r'<\|start\|>', '', generated_text_raw)
                generated_text = re.sub(r'<\|message\|>', '', generated_text)
                generated_text = re.sub(r'<\|end\|>', '', generated_text)
                generated_text = generated_text.strip()
                print(f"[INFERENCE SERVICE] Generated text still empty, using cleaned raw text", flush=True)
        except RuntimeError as e:
            if "dtype" in str(e).lower() or "scalar type" in str(e).lower():
                print(f"[INFERENCE SERVICE] Dtype mismatch detected, attempting to convert model to float16...", flush=True)
                if hasattr(model, 'to'):
                    try:
                        model = model.to(torch.float16)
                        retry_device = device
                        if retry_device is None:
                            if hasattr(model, 'hf_device_map') and model.hf_device_map:
                                cuda_devices = []
                                for layer_name, d in model.hf_device_map.items():
                                    if isinstance(d, (str, torch.device)):
                                        device_str = str(d) if isinstance(d, torch.device) else d
                                        if device_str.startswith("cuda"):
                                            device_obj = torch.device(device_str)
                                            if device_obj not in cuda_devices:
                                                cuda_devices.append(device_obj)
                                if cuda_devices:
                                    retry_device = cuda_devices[0]
                                else:
                                    retry_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            else:
                                retry_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        
                        retry_input_ids = input_ids.to(retry_device)
                        retry_attention_mask = attention_mask.to(retry_device)
                        
                        retry_gen_kwargs = {
                            "input_ids": retry_input_ids,
                            "attention_mask": retry_attention_mask,
                            "max_new_tokens": min(request.max_new_tokens, 1024),
                            "temperature": gen_temperature,
                            "top_p": request.top_p,
                            "repetition_penalty": request.repetition_penalty,
                            "do_sample": True,
                        }
                        if not (hasattr(model, 'hf_device_map') and model.hf_device_map):
                            eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
                            retry_gen_kwargs["pad_token_id"] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
                            retry_gen_kwargs["eos_token_id"] = eos_token_id
                        with torch.no_grad():
                            generated_ids = model.generate(**retry_gen_kwargs)
                        generated_tokens = generated_ids[0, prompt_len_tokens:]
                        generated_text_raw = tokenizer.decode(generated_tokens, skip_special_tokens=False)
                        generated_text_clean = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                        
                        print(f"[INFERENCE SERVICE] Generated text raw after dtype conversion (first 500 chars): {generated_text_raw[:500]}", flush=True)
                        
                        has_harmony = (
                            '<|channel|>' in generated_text_raw.lower() or
                            '<|start|>assistant' in generated_text_raw.lower() or
                            '<|message|>' in generated_text_raw.lower()
                        )
                        
                        if has_harmony:
                            parsed = parse_harmony_response(generated_text_raw, tokenizer)
                            if parsed is not None and len(parsed) > 0:
                                generated_text = parsed
                                print(f"[INFERENCE SERVICE] Extracted Harmony format content ({len(parsed)} chars)", flush=True)
                            else:
                                generated_text = generated_text_clean
                                print(f"[INFERENCE SERVICE] No final channel found in Harmony format, using clean decoded text", flush=True)
                        else:
                            generated_text = generated_text_clean
                        
                        if not generated_text or len(generated_text.strip()) == 0:
                            generated_text = generated_text_clean
                            print(f"[INFERENCE SERVICE] Generated text empty, using clean decoded text", flush=True)
                        
                        if not generated_text or len(generated_text.strip()) == 0:
                            generated_text = re.sub(r'<\|start\|>', '', generated_text_raw)
                            generated_text = re.sub(r'<\|message\|>', '', generated_text)
                            generated_text = re.sub(r'<\|end\|>', '', generated_text)
                            generated_text = generated_text.strip()
                            print(f"[INFERENCE SERVICE] Generated text still empty, using cleaned raw text", flush=True)
                    except Exception as conv_error:
                        print(f"[INFERENCE SERVICE] Failed to convert model dtype: {conv_error}", flush=True)
                        raise e  # Re-raise original error
                else:
                    raise e
            else:
                raise e
        
        generated_text = generated_text.strip()
        
        generated_text = re.sub(r'<\|start\|>', '', generated_text)
        generated_text = re.sub(r'<\|message\|>', '', generated_text)
        generated_text = re.sub(r'<\|end\|>', '', generated_text)
        generated_text = re.sub(r'<\|return\|>', '', generated_text)
        generated_text = re.sub(r'<\|call\|>', '', generated_text)
        generated_text = re.sub(r'\s+', ' ', generated_text)
        generated_text = generated_text.strip()
        
        if request.stop:
            earliest_stop = None
            earliest_pos = len(generated_text)
            
            for stop_str in request.stop:
                if stop_str:
                    stop_str_clean = stop_str.strip()
                    pos = generated_text.find(stop_str_clean)
                    if pos != -1 and pos < earliest_pos:
                        earliest_pos = pos
                        earliest_stop = stop_str_clean
            
            if earliest_stop is not None:
                generated_text = generated_text[:earliest_pos].strip()
                print(f"[INFERENCE SERVICE] Truncated at earliest stop string: {earliest_stop} (position {earliest_pos})", flush=True)
        
        gpt_identity_patterns = [
            r'\bI am ChatGPT\b',
            r'\bI\'m ChatGPT\b',
            r'\bI am GPT\b',
            r'\bI\'m GPT\b',
            r'\bI am GPT-?\d+\b',
            r'\bI\'m GPT-?\d+\b',
            r'\bcreated by OpenAI\b',
            r'\bdeveloped by OpenAI\b',
            r'\btrained by OpenAI\b',
            r'\bfrom OpenAI\b',
            r'\bas an OpenAI model\b',
            r'\bas a ChatGPT model\b',
            r'\bas a GPT model\b'
        ]
        for pattern in gpt_identity_patterns:
            generated_text = re.sub(pattern, 'Epsilon AI', generated_text, flags=re.IGNORECASE)
        
        generated_text = clean_markdown_text(generated_text)
        generated_text = generated_text.strip()
        
        if not generated_text:
            print(f"[INFERENCE SERVICE] WARNING: Generated text is empty after processing, using fallback message", flush=True)
            generated_text = "I apologize, but I couldn't generate a response. Please try again."
        
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
                "message": f"Model reloaded successfully",
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
