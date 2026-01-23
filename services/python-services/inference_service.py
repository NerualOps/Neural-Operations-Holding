"""
Epsilon AI Inference Service
Uses the Epsilon AI model with Harmony response format
Created by Neural Operations & Holdings LLC
"""
import os
import sys
import warnings

# Force unbuffered output for real-time logs
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTHONUNBUFFERED'] = '1'  # Ensure unbuffered output
# Set CUDA launch blocking for better error reporting (can be removed in production if needed)
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')

# CRITICAL: Catch bf16 fallback warnings and convert to errors
def bf16_fallback_warning_handler(message, category, filename, lineno, file=None, line=None):
    """Convert bf16 fallback warnings to RuntimeError"""
    msg_str = str(message)
    if "default to dequantizing" in msg_str.lower() and "bf16" in msg_str.lower():
        print(f"[INFERENCE SERVICE] CRITICAL: Caught bf16 fallback warning: {msg_str}", flush=True)
        raise RuntimeError(
            f"CRITICAL: Model attempted to fall back to bf16! This will cause OOM. "
            f"Original warning: {msg_str}"
        )
    # For other warnings, use default handler
    return warnings._showwarning_orig(message, category, filename, lineno, file, line)

# Install custom warning handler
warnings._showwarning_orig = warnings.showwarning
warnings.showwarning = bf16_fallback_warning_handler

from pathlib import Path
from typing import Optional, List, Any
import re
import gc
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from filelock import FileLock
from huggingface_hub import snapshot_download

try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
    from transformers import PretrainedConfig
    
    # This will be populated when we know the model type
    _PATCHED_MODEL_TYPES = set()
    
    def patch_config_mapping_for_model_type(model_type: str):
        """Patch CONFIG_MAPPING and MODEL_MAPPING for unknown model types"""
        if model_type and model_type not in CONFIG_MAPPING:
            CONFIG_MAPPING[model_type] = PretrainedConfig
            # Also patch MODEL_FOR_CAUSAL_LM_MAPPING if needed
            if model_type not in MODEL_FOR_CAUSAL_LM_MAPPING:
                # Use a generic mapping - trust_remote_code will load the actual model
                from transformers import AutoModelForCausalLM
                MODEL_FOR_CAUSAL_LM_MAPPING[model_type] = AutoModelForCausalLM
            _PATCHED_MODEL_TYPES.add(model_type)
            print(f"[INFERENCE SERVICE] Patched CONFIG_MAPPING and MODEL_MAPPING for '{model_type}'", flush=True)
            # Verify the patch
            if model_type in CONFIG_MAPPING:
                print(f"[INFERENCE SERVICE] ✓ Verified: '{model_type}' is now in CONFIG_MAPPING", flush=True)
            else:
                print(f"[INFERENCE SERVICE] ✗ ERROR: Patch failed - '{model_type}' not in CONFIG_MAPPING", flush=True)
except Exception as e:
    print(f"[INFERENCE SERVICE] WARNING: Could not import CONFIG_MAPPING for patching: {e}", flush=True)
    def patch_config_mapping_for_model_type(model_type: str):
        pass

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
MODEL_DIR = Path(os.getenv('EPSILON_MODEL_DIR', '/workspace/models/epsilon-120b'))


def load_model():
    """Load Epsilon AI model using transformers - optimized for GPU"""
    global model, tokenizer, model_metadata
    
    lock_dir = Path(__file__).parent / '.cursor'
    lock_dir.mkdir(exist_ok=True)
    lock_path = str(lock_dir / "hf_download.lock")
    
    print(f"[INFERENCE SERVICE] Loading Epsilon AI model: {MODEL_ID}", flush=True)
    print(f"[INFERENCE SERVICE] Internal model repository: {HF_MODEL_ID_INTERNAL}", flush=True)
    
    # CRITICAL: Always use BitsAndBytes 4-bit quantization (never MXFP4/bf16)
    # Force BnB 4-bit always regardless of Python version
    use_bitsandbytes = True
    
    import sys
    python_version = sys.version_info
    print(f"[INFERENCE SERVICE] Python version: {python_version.major}.{python_version.minor}.{python_version.micro}", flush=True)
    print(f"[INFERENCE SERVICE] ALWAYS using BitsAndBytes 4-bit quantization (never MXFP4/bf16)", flush=True)
    
    # Only check Triton if we're NOT using BitsAndBytes (we are, so skip Triton checks)
    triton_available = False
    triton_version = None
    triton_compatible = True
    if not use_bitsandbytes:
        # Triton checks (MXFP4 path) - only executed if not using BitsAndBytes
        try:
            import triton
            triton_available = True
            triton_version = getattr(triton, '__version__', 'unknown')
            print(f"[INFERENCE SERVICE] Triton is available: version {triton_version}", flush=True)
            
            # Check if Triton version matches PyTorch requirements
            torch_version = torch.__version__
            # PyTorch 2.8.0+cu128 requires triton==3.4.0 (exact match)
            if "2.8.0" in torch_version or ("2.8" in torch_version and "+cu" in torch_version):
                if triton_version != "3.4.0":
                    triton_compatible = False
                    print(f"[INFERENCE SERVICE] CRITICAL ERROR: Version mismatch! PyTorch {torch_version} requires triton==3.4.0, but you have triton {triton_version}", flush=True)
                    print(f"[INFERENCE SERVICE] This will cause MXFP4 quantization to fail and model will load as bf16 (too large for GPU/system memory)", flush=True)
                    print(f"[INFERENCE SERVICE] FIX REQUIRED: pip uninstall -y triton && pip install 'triton==3.4.0'", flush=True)
                    print(f"[INFERENCE SERVICE] ABORTING model load to prevent bf16 fallback and system memory exhaustion.", flush=True)
                    raise RuntimeError(
                        f"Triton version mismatch: PyTorch {torch_version} requires triton==3.4.0, but triton {triton_version} is installed. "
                        f"Fix with: pip uninstall -y triton && pip install 'triton==3.4.0'"
                    )
            else:
                # For other PyTorch versions, check if Triton is >= 3.4.0
                try:
                    version_parts = [int(x) for x in triton_version.split('.')[:2]]
                    if version_parts[0] < 3 or (version_parts[0] == 3 and version_parts[1] < 4):
                        print(f"[INFERENCE SERVICE] WARNING: Triton version {triton_version} may be too old. Recommended: >=3.4.0", flush=True)
                except (ValueError, AttributeError):
                    pass
        except ImportError:
            print(f"[INFERENCE SERVICE] CRITICAL ERROR: Triton is NOT available.", flush=True)
            print(f"[INFERENCE SERVICE] Without Triton, MXFP4 quantization will fail and model will attempt to load as bf16.", flush=True)
            print(f"[INFERENCE SERVICE] A 120B model in bf16 requires ~240GB and will exhaust system memory.", flush=True)
            print(f"[INFERENCE SERVICE] ABORTING to prevent system memory exhaustion.", flush=True)
            raise RuntimeError(
                "Triton is required for MXFP4 quantization. Without it, the model will fall back to bf16 "
                "which will exhaust system memory. Install Triton: pip install 'triton==3.4.0'"
            )
        except Exception as e:
            print(f"[INFERENCE SERVICE] CRITICAL ERROR: Triton import failed: {e}", flush=True)
            print(f"[INFERENCE SERVICE] ABORTING to prevent bf16 fallback and system memory exhaustion.", flush=True)
            raise RuntimeError(
                f"Triton import failed: {e}. Triton is required for MXFP4 quantization. "
                "Fix Triton installation before proceeding."
            )
    
    # Check PyTorch version
    print(f"[INFERENCE SERVICE] PyTorch version: {torch.__version__}, CUDA: {torch.version.cuda}", flush=True)
    
    # Always verify BitsAndBytes is available (we're always using it)
    try:
        import bitsandbytes
        print(f"[INFERENCE SERVICE] BitsAndBytes version: {bitsandbytes.__version__}", flush=True)
    except ImportError:
        raise RuntimeError(
            "BitsAndBytes is REQUIRED for 4-bit quantization. "
            "Install: pip install -U bitsandbytes"
        )
    
    try:
        import shutil
        disk_usage = shutil.disk_usage(MODEL_DIR if MODEL_DIR.exists() else Path(__file__).parent)
        free_gb = disk_usage.free / (1024**3)
        print(f"[INFERENCE SERVICE] Available disk space: {free_gb:.2f} GB", flush=True)
        if free_gb < 50:
            print(f"[INFERENCE SERVICE] WARNING: Low disk space ({free_gb:.2f} GB). Model requires ~40GB.", flush=True)
        
        # Check CUDA availability without accessing devices (avoids initialization issues)
        if torch.cuda.is_available():
            try:
                num_gpus = torch.cuda.device_count()
                print(f"[INFERENCE SERVICE] Detected {num_gpus} GPU(s)", flush=True)
                
                # Just report device names without accessing them
                for gpu_id in range(num_gpus):
                    try:
                        props = torch.cuda.get_device_properties(gpu_id)
                        total_gb = props.total_memory / (1024**3)
                        print(f"[INFERENCE SERVICE] GPU {gpu_id}: {props.name}, {total_gb:.2f} GB total memory", flush=True)
                    except Exception as e:
                        print(f"[INFERENCE SERVICE] WARNING: Could not query GPU {gpu_id} properties: {e}", flush=True)
                        print(f"[INFERENCE SERVICE] This is OK - device_map='auto' will handle device placement during model load", flush=True)
                        continue
            except Exception as e:
                print(f"[INFERENCE SERVICE] WARNING: Could not query CUDA devices: {e}", flush=True)
                print(f"[INFERENCE SERVICE] Will attempt to load model anyway - device_map='auto' will handle placement", flush=True)
        
        hub_dir = Path.home() / ".cache" / "huggingface"
        if hub_dir.exists():
            print(f"[INFERENCE SERVICE] Clearing Hugging Face cache in home directory to free disk space...", flush=True)
            try:
                cache_size = sum(f.stat().st_size for f in hub_dir.rglob('*') if f.is_file()) / (1024**3)
                shutil.rmtree(hub_dir)
                print(f"[INFERENCE SERVICE] Cleared {cache_size:.2f} GB of Hugging Face cache", flush=True)
            except Exception as e:
                print(f"[INFERENCE SERVICE] Warning: Could not clear cache: {e}", flush=True)
        
        # NOTE:
        # Deleting the model directory cache can race with HF downloads/extracts and emit
        # noisy "Directory not empty" warnings (e.g. subdir "metal"). Make this opt-in.
        if os.getenv("EPSILON_CLEAR_MODEL_CACHE", "").strip() in {"1", "true", "True", "yes", "YES"}:
        model_cache_dir = MODEL_DIR / ".cache"
        if model_cache_dir.exists():
                print(
                    f"[INFERENCE SERVICE] EPSILON_CLEAR_MODEL_CACHE enabled; removing {model_cache_dir} ...",
                    flush=True,
                )
            try:
                    shutil.rmtree(model_cache_dir, ignore_errors=True)
                    print(f"[INFERENCE SERVICE] Removed {model_cache_dir}", flush=True)
            except Exception as e:
                    # ignore_errors=True should prevent most failures, but keep it non-fatal regardless.
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
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                    local_path = snapshot_download(
                        repo_id=HF_MODEL_ID_INTERNAL,
                        local_dir=str(MODEL_DIR),
                        max_workers=1,
                        ignore_patterns=[".cache/**"],
                            resume_download=True,
                    )
                    print(f"[INFERENCE SERVICE] Model snapshot downloaded to: {local_path}", flush=True)
            else:
                print(f"[INFERENCE SERVICE] Downloading model snapshot to local directory (this may take 10-15 minutes)...", flush=True)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                local_path = snapshot_download(
                    repo_id=HF_MODEL_ID_INTERNAL,
                    local_dir=str(MODEL_DIR),
                    max_workers=1,
                    ignore_patterns=[".cache/**"],
                        resume_download=True
                )
                print(f"[INFERENCE SERVICE] Model snapshot downloaded to: {local_path}", flush=True)
        
        print(f"[INFERENCE SERVICE] Loading tokenizer from local path: {local_path}", flush=True)
        # Try to load tokenizer - if ANY error occurs, download fresh from HuggingFace
        # This handles missing files, corrupted files, or any initialization errors
        try:
        tokenizer = AutoTokenizer.from_pretrained(
            local_path,
            trust_remote_code=True,
            local_files_only=True
        )
            print(f"[INFERENCE SERVICE] Tokenizer loaded successfully from local path", flush=True)
        except Exception as e:
            # ANY exception means we need to download fresh - corrupted, missing, or incompatible files
            print(f"[INFERENCE SERVICE] Tokenizer load failed ({type(e).__name__}: {str(e)[:200]}), downloading fresh from HuggingFace...", flush=True)
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    HF_MODEL_ID_INTERNAL,
                    trust_remote_code=True,
                    cache_dir=str(MODEL_DIR),
                    local_dir_use_symlinks=False
                )
                # Save tokenizer to local path for next time
                tokenizer.save_pretrained(local_path)
                print(f"[INFERENCE SERVICE] Tokenizer downloaded and saved to {local_path}", flush=True)
            except Exception as download_error:
                print(f"[INFERENCE SERVICE] CRITICAL: Failed to download tokenizer: {download_error}", flush=True)
                raise
        
        print(f"[INFERENCE SERVICE] Loading model on GPU from local path: {local_path}", flush=True)
        
        global model
        if model is not None:
            print(f"[INFERENCE SERVICE] Clearing existing model from memory...", flush=True)
            del model
            model = None
        
        # Don't access CUDA devices here - let model loading with device_map='auto' handle it
        # Accessing devices before model load can cause "device busy" errors
        if torch.cuda.is_available():
            try:
                num_gpus = torch.cuda.device_count()
                print(f"[INFERENCE SERVICE] Preparing {num_gpus} GPU(s) for model loading", flush=True)
                print(f"[INFERENCE SERVICE] Using device_map='auto' - model will be placed automatically", flush=True)
            except Exception as e:
                print(f"[INFERENCE SERVICE] WARNING: Could not query CUDA device count: {e}", flush=True)
                print(f"[INFERENCE SERVICE] Will attempt to load model anyway", flush=True)
        
        # CRITICAL FIX: Load config.json directly and create config object manually
        # This bypasses CONFIG_MAPPING lookup entirely, which fails for unknown architectures like 'gpt_oss'
        import json
        config_json_path = Path(local_path) / "config.json"
        model_config_obj = None
        config_data = {}
        model_type = None
        
        if config_json_path.exists():
            with open(config_json_path, 'r') as f:
                config_data = json.load(f)
                model_type = config_data.get("model_type", None)
                print(f"[INFERENCE SERVICE] Loaded config.json, model_type: '{model_type}'", flush=True)
                
                # Try to load config using AutoConfig with trust_remote_code (for custom architectures)
                try:
                    model_config_obj = AutoConfig.from_pretrained(local_path, trust_remote_code=True, local_files_only=True)
                    print(f"[INFERENCE SERVICE] Successfully loaded config via AutoConfig", flush=True)
                except (KeyError, ValueError) as e:
                    # If AutoConfig fails due to unknown architecture, create config from dict
                    print(f"[INFERENCE SERVICE] AutoConfig failed for '{model_type}': {e}", flush=True)
                    print(f"[INFERENCE SERVICE] Creating PretrainedConfig from config.json dict...", flush=True)
                    try:
                        from transformers import PretrainedConfig
                        # Create a generic PretrainedConfig from the dict
                        # trust_remote_code=True will load the actual custom config class when loading the model
                        model_config_obj = PretrainedConfig.from_dict(config_data)
                        print(f"[INFERENCE SERVICE] Created PretrainedConfig from dict", flush=True)
                    except Exception as e2:
                        print(f"[INFERENCE SERVICE] Could not create PretrainedConfig from dict: {e2}", flush=True)
                        print(f"[INFERENCE SERVICE] Will pass config dict directly to from_pretrained", flush=True)
                        model_config_obj = config_data
        else:
            print(f"[INFERENCE SERVICE] WARNING: config.json not found at {config_json_path}", flush=True)
        
        # CRITICAL: Remove any existing quantization_config from model config
        # The model may have Mxfp4Config, but we're forcing BitsAndBytes - remove the conflict
        if model_config_obj is not None:
            if hasattr(model_config_obj, 'quantization_config'):
                print(f"[INFERENCE SERVICE] Removing existing quantization_config from model config (was: {type(model_config_obj.quantization_config).__name__})", flush=True)
                model_config_obj.quantization_config = None
            elif isinstance(model_config_obj, dict):
                if 'quantization_config' in model_config_obj:
                    print(f"[INFERENCE SERVICE] Removing quantization_config from config dict", flush=True)
                    del model_config_obj['quantization_config']
        
        if 'quantization_config' in config_data:
            print(f"[INFERENCE SERVICE] Removing quantization_config from config_data dict", flush=True)
            del config_data['quantization_config']
        
        # CRITICAL: ALWAYS apply BitsAndBytes 4-bit quantization
        # Never trust pre-quantization claims - always force quantization to prevent bf16 fallback
        print("[INFERENCE SERVICE] ALWAYS applying BitsAndBytes 4-bit quantization (never trust pre-quantization)", flush=True)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print("[INFERENCE SERVICE] BitsAndBytes 4-bit quantization config created", flush=True)
        
        # If model is pre-quantized with Mxfp4Config but we're hitting OOM, 
        # we can't override quantization, but we can use more aggressive memory management
        use_8bit_fallback = os.getenv("EPSILON_USE_8BIT", "").strip() in {"1", "true", "True", "yes", "YES"}
        if use_8bit_fallback and quantization_config is None:
            # Can't use 8-bit if model is already quantized, but log the option
            print("[INFERENCE SERVICE] EPSILON_USE_8BIT requested but model is pre-quantized - using memory limits instead", flush=True)
        
        # Normalize max_memory to use "GiB" consistently (transformers/accelerate expects this)
        # This prevents 'int' object has no attribute 'replace' errors
        def normalize_max_memory(mm):
            """Convert max_memory dict to use GiB strings consistently"""
            if not mm:
                return None
            out = {}
            for k, v in mm.items():
                if isinstance(v, (int, float)):
                    out[k] = f"{int(v)}GiB"
                else:
                    s = str(v).strip()
                    # Normalize GB/MB to GiB, extract number if needed
                    if "GiB" in s or "GB" in s or "MB" in s or "MiB" in s:
                        # Extract number and convert to GiB
                        num_str = s.replace("GiB", "").replace("GB", "").replace("MB", "").replace("MiB", "").strip()
                        try:
                            num = float(num_str)
                            # Convert MB to GiB if needed
                            if "MB" in s or "MiB" in s:
                                num = num / 1024
                            out[k] = f"{int(num)}GiB"
                        except (ValueError, TypeError):
                            out[k] = s if "GiB" in s else f"{s}GiB"
                    else:
                        # Assume it's already a number, add GiB
                        try:
                            num = float(s)
                            out[k] = f"{int(num)}GiB"
                        except (ValueError, TypeError):
                            out[k] = f"{s}GiB"
            return out
        
        # Calculate safe max_memory per GPU (leave 14GB headroom for system/overhead/dequantization)
        # 120B models with MXFP4 need significant headroom for transient buffers during loading
        max_memory = None
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            # Get actual GPU memory and set very conservative limits
            gpu_memories = {}
            for i in range(torch.cuda.device_count()):
                total_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                # Reserve 14GB for system/overhead/dequantization/MXFP4 staging buffers
                safe_gb = max(1, int(total_gb - 14))
                gpu_memories[i] = f"{safe_gb}GiB"
            # Always include CPU for offloading
            gpu_memories["cpu"] = "200GiB"
            max_memory = gpu_memories
            print(f"[INFERENCE SERVICE] Configuring model for {torch.cuda.device_count()} GPUs with conservative memory limits: {max_memory}", flush=True)
        else:
            if torch.cuda.is_available():
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                safe_gb = max(1, int(total_gb - 14))
                max_memory = {0: f"{safe_gb}GiB", "cpu": "200GiB"}
                print(f"[INFERENCE SERVICE] Loading model on single GPU with conservative memory limit: {max_memory}", flush=True)
            else:
                max_memory = {"cpu": "200GiB"}
                print(f"[INFERENCE SERVICE] Loading model on CPU", flush=True)
        
        # Set PyTorch CUDA allocator config for better memory management
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8')
        
        # Normalize max_memory to GiB strings (prevents int.replace errors)
        if max_memory:
            max_memory = normalize_max_memory(max_memory)
        
        # Build kwargs - use balanced_low_0 for better GPU distribution
        # trust_remote_code=True is CRITICAL for custom architectures
        # Pass config explicitly to bypass CONFIG_MAPPING lookup
        load_kwargs = {
            "device_map": "balanced_low_0",  # Better GPU distribution than "auto"
            "trust_remote_code": True,  # CRITICAL: This allows loading custom architectures
            "low_cpu_mem_usage": True,
            "local_files_only": True
        }
        
        # CRITICAL: Pass config explicitly to bypass internal AutoConfig.from_pretrained() call
        # This prevents the CONFIG_MAPPING KeyError for unknown architectures
        if model_config_obj is not None:
            load_kwargs["config"] = model_config_obj
            print(f"[INFERENCE SERVICE] Passing explicit config to from_pretrained() to bypass CONFIG_MAPPING lookup", flush=True)
        
        # CRITICAL: Pass config explicitly to bypass internal AutoConfig.from_pretrained() call
        # This prevents the CONFIG_MAPPING KeyError for unknown architectures
        if model_config_obj is not None:
            load_kwargs["config"] = model_config_obj
            print(f"[INFERENCE SERVICE] Passing explicit config to from_pretrained() to bypass CONFIG_MAPPING lookup", flush=True)
        
        if max_memory:
            load_kwargs["max_memory"] = max_memory
            # Enable CPU offloading for very large models (120B+)
            load_kwargs["offload_folder"] = str(MODEL_DIR / "offload")
            Path(load_kwargs["offload_folder"]).mkdir(parents=True, exist_ok=True)
        # CRITICAL: ALWAYS pass both load_in_4bit=True AND quantization_config
        # load_in_4bit=True forces the BnB quantized load path even with trust_remote_code=True
        load_kwargs["load_in_4bit"] = True
        load_kwargs["quantization_config"] = quantization_config
        print(f"[INFERENCE SERVICE] Will load with load_in_4bit=True and quantization_config: {type(quantization_config).__name__}", flush=True)
        
        # Clear GPU cache before loading to maximize available memory
        if torch.cuda.is_available():
            for gpu_id in range(torch.cuda.device_count()):
                try:
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception as e:
                    print(f"[INFERENCE SERVICE] WARNING: Could not clear GPU {gpu_id} cache: {e}", flush=True)
                    continue
            gc.collect()
            print(f"[INFERENCE SERVICE] Cleared GPU cache before model load", flush=True)
        
        print(f"[INFERENCE SERVICE] Loading model with kwargs: {list(load_kwargs.keys())}", flush=True)
        
        # Try multiple loading strategies
        model = None
        # Store triton status for later checks (in nested scope)
        _triton_available = triton_available and triton_compatible
        load_strategies = [
            ("Initial load with memory limits", load_kwargs.copy()),
        ]
        
        # Strategy 2: Reduced memory limits (normalized to GiB)
        if max_memory:
            reduced_kwargs = load_kwargs.copy()
            reduced_memory = {}
            for k, v in max_memory.items():
                if k == "cpu":
                    reduced_memory["cpu"] = "200GiB"  # Keep CPU limit
                    continue
                # Safe parsing - handle both int and string, normalize to GiB
                if isinstance(v, (int, float)):
                    current = int(v)
                else:
                    mem_str = str(v).strip()
                    # Extract number from "30GiB" or "30GB" etc
                    mem_clean = mem_str.replace("GiB", "").replace("GB", "").replace("MB", "").replace("MiB", "").strip()
                    current = int(float(mem_clean)) if mem_clean.replace(".", "").isdigit() else 30
                # Reduce by 10GiB (very aggressive)
                reduced_memory[k] = f"{max(1, current - 10)}GiB"
            reduced_memory = normalize_max_memory(reduced_memory)
            reduced_kwargs["max_memory"] = reduced_memory
            reduced_kwargs["offload_folder"] = str(MODEL_DIR / "offload")
            Path(reduced_kwargs["offload_folder"]).mkdir(parents=True, exist_ok=True)
            load_strategies.append(("Reduced memory limits", reduced_kwargs))
        
        # Strategy 3: No max_memory, let transformers auto-balance
        no_limit_kwargs = load_kwargs.copy()
        if "max_memory" in no_limit_kwargs:
            del no_limit_kwargs["max_memory"]
        no_limit_kwargs["offload_folder"] = str(MODEL_DIR / "offload")
        Path(no_limit_kwargs["offload_folder"]).mkdir(parents=True, exist_ok=True)
        load_strategies.append(("No memory limits (auto-balance)", no_limit_kwargs))
        
        # Strategy 4: Sequential device_map (load layer by layer) with CPU offload
        sequential_kwargs = load_kwargs.copy()
        if "max_memory" in sequential_kwargs:
            # Keep max_memory but use sequential loading
            sequential_kwargs["max_memory"] = normalize_max_memory(sequential_kwargs["max_memory"])
        sequential_kwargs["device_map"] = "sequential"
        sequential_kwargs["offload_folder"] = str(MODEL_DIR / "offload")
        Path(sequential_kwargs["offload_folder"]).mkdir(parents=True, exist_ok=True)
        load_strategies.append(("Sequential device_map", sequential_kwargs))
        
        # Try each strategy
        for strategy_name, strategy_kwargs in load_strategies:
            if model is not None:
                break
                
            print(f"[INFERENCE SERVICE] Trying strategy: {strategy_name}", flush=True)
            
            # Clear GPU cache before each attempt
            if torch.cuda.is_available():
                for gpu_id in range(torch.cuda.device_count()):
                    try:
                        with torch.cuda.device(gpu_id):
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                    except Exception as e:
                        print(f"[INFERENCE SERVICE] WARNING: Could not clear GPU {gpu_id} cache: {e}", flush=True)
                        continue
                gc.collect()
            
            try:
                # Config is already loaded and passed in strategy_kwargs, so we can load directly
                model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    **strategy_kwargs
                )
                print(f"[INFERENCE SERVICE] Successfully loaded model using strategy: {strategy_name}", flush=True)
                
                # CRITICAL: Verify actual BitsAndBytes 4-bit Linear modules exist
                # This is the only reliable check - is_loaded_in_4bit can be False even with quantization_config
                # and dtype checks are misleading (norms/embeddings can be fp16 even when Linear layers are 4-bit)
                import bitsandbytes as bnb
                linear4bit_count = sum(1 for m in model.modules() if isinstance(m, bnb.nn.Linear4bit))
                assert linear4bit_count > 0, (
                    f"CRITICAL: BitsAndBytes 4-bit did not attach! Linear4bit count={linear4bit_count}. "
                    f"Model will fallback to bf16 and OOM. Check BitsAndBytes installation and trust_remote_code compatibility."
                )
                
                is_4bit = getattr(model, "is_loaded_in_4bit", False)
                first_param = next(iter(model.parameters()))
                dtype = first_param.dtype
                print(f"[INFERENCE SERVICE] Model parameter dtype: {dtype}", flush=True)
                print(f"[INFERENCE SERVICE] Model is_loaded_in_4bit: {is_4bit}", flush=True)
                print(f"[INFERENCE SERVICE] ✓ Verified BitsAndBytes 4-bit modules: Linear4bit count={linear4bit_count}", flush=True)
                
                if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
                    qc = model.config.quantization_config
                    if qc is not None:
                        qc_dict = qc.to_dict() if hasattr(qc, "to_dict") else (qc if isinstance(qc, dict) else {})
                        quant_method = qc_dict.get("quant_method") if isinstance(qc_dict, dict) else None
                        print(f"[INFERENCE SERVICE] Quantization method: {quant_method}", flush=True)
                
                print(f"[INFERENCE SERVICE] ✓ Model loaded with 4-bit quantization (verified with Linear4bit modules)", flush=True)
                
                break
            except RuntimeError as e:
                error_msg = str(e)
                if "CRITICAL: Model loaded as" in error_msg and ("bfloat16" in error_msg or "float16" in error_msg):
                    print(f"[INFERENCE SERVICE] FATAL: {error_msg}", flush=True)
                    raise
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                print(f"[INFERENCE SERVICE] Strategy '{strategy_name}' failed: {error_type}: {error_msg[:200]}", flush=True)
                
                # If it's an AttributeError in transformers (like the int.replace bug), try next strategy
                if "AttributeError" in error_type or "'int' object has no attribute" in error_msg:
                    print(f"[INFERENCE SERVICE] This appears to be a transformers library bug, trying next strategy...", flush=True)
                    continue
                
                # If it's OOM, try next strategy
                if "out of memory" in error_msg.lower() or "cuda out of memory" in error_msg.lower():
                    print(f"[INFERENCE SERVICE] Out of memory, trying next strategy...", flush=True)
                    continue
                
                # For other errors, if this is the last strategy, raise it
                if strategy_name == load_strategies[-1][0]:
                    raise
        
        if model is None:
            raise RuntimeError("Failed to load model with all available strategies. Model may be too large for available VRAM.")
        
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
        
        print(f"[INFERENCE SERVICE] EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})", flush=True)
        if hasattr(tokenizer, 'additional_special_tokens') and tokenizer.additional_special_tokens:
            print(f"[INFERENCE SERVICE] Additional special tokens: {tokenizer.additional_special_tokens}", flush=True)
        if hasattr(tokenizer, 'special_tokens_map'):
            print(f"[INFERENCE SERVICE] Special tokens map: {tokenizer.special_tokens_map}", flush=True)
        
    except Exception as e:
        import traceback
        print(f"[INFERENCE SERVICE] ERROR: Failed to load model: {e}", flush=True)
        print(f"[INFERENCE SERVICE] Traceback: {traceback.format_exc()}", flush=True)
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
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    repetition_penalty: float = 1.3
    conversation_history: Optional[List[dict]] = None


class GenerateResponse(BaseModel):
    model_config = {'protected_namespaces': ()}  # Allow "model_" prefix
    
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
    Clean markdown formatting and convert tables to clean plain text.
    Removes markdown symbols and formats tables as readable text.
    """
    if not text:
        return ""

    code_blocks: List[str] = []

    def _stash_code_block(m: re.Match) -> str:
        code_blocks.append(m.group(0))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

    text = re.sub(r"```[\s\S]*?```", _stash_code_block, text)
    
    lines = text.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        trimmed = line.strip()
        
        if '|' in trimmed and trimmed.count('|') >= 2:
            if trimmed.replace('|', '').replace('-', '').replace(':', '').replace(' ', '').strip():
                table_rows = []
                is_separator_line = False
                
                while i < len(lines):
                    current_line = lines[i].strip()
                    if not current_line or '|' not in current_line or current_line.count('|') < 2:
                        break
                    
                    if re.match(r'^[\|\s\-:]+$', current_line):
                        is_separator_line = True
                        i += 1
                        continue
                    
                    cells = [c.strip() for c in current_line.split('|')]
                    if cells and cells[0] == '':
                        cells.pop(0)
                    if cells and cells[-1] == '':
                        cells.pop()
                    
                    cells = [c for c in cells if c and not re.match(r'^[\s\-:]+$', c)]
                    
                    if cells:
                        table_rows.append(cells)
                    i += 1
                
                if table_rows:
                    result.append('')
                    for row in table_rows:
                        if len(row) == 2:
                            result.append(f"{row[0]}: {row[1]}")
                        elif len(row) > 2:
                            result.append(" • ".join(row))
                        else:
                            result.append(row[0])
                    result.append('')
                    continue
        
        result.append(line)
        i += 1
    
    text = '\n'.join(result)
    
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(?!\*)([^\*\n]+?)(?!\*)\*', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'---{2,}', '—', text)
    text = re.sub(r'^\*\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+[\.\)]\s+', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'[📑🛠︎🎯⏰🚀]', '', text)
    text = re.sub(r'[1️⃣2️⃣3️⃣4️⃣5️⃣6️⃣7️⃣8️⃣]', '', text)
    
    text = re.sub(r'\*{2,}', '', text)
    text = re.sub(r'`{2,}', '', text)
    text = re.sub(r'#{3,}', '', text)
    text = re.sub(r'\|{2,}', '|', text)
    
    text = re.sub(r'<[^>]+>', '', text)
    
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' \n', '\n', text)
    text = re.sub(r'^[\s\-|:]+$', '', text, flags=re.MULTILINE)

    for idx, block in enumerate(code_blocks):
        text = text.replace(f"__CODE_BLOCK_{idx}__", block)
    
    return text.strip()


def parse_harmony_response(text: str, tokenizer: Any) -> Optional[str]:
    """
    Parse Harmony format response to extract ONLY the 'final' channel content.
    CRITICAL: This function MUST NEVER return analysis, commentary, or reasoning text.
    It ONLY extracts content from <|channel|>final and removes ALL other channels.
    """
    if not text:
        return ""
    
    text_lower = text.lower()
    
    if '<|channel|>' in text_lower:
        final_match = re.search(r'<\|channel\|>final(?:<message>)?(.*?)(?=<\|channel\|>|<\|end\|>|$)', text, re.DOTALL | re.IGNORECASE)
        if final_match:
            final_content = final_match.group(1).strip()
            final_content = final_content.replace('<message>', '').replace('</message>', '')
            final_content = re.sub(r'<message>', '', final_content, flags=re.IGNORECASE)
            final_content = re.sub(r'</message>', '', final_content, flags=re.IGNORECASE)
            if '<|end|>' in final_content:
                final_content = final_content[:final_content.find('<|end|>')].strip()
            final_content = re.sub(r'<\|channel\|>analysis.*?<\|channel\|>', '', final_content, flags=re.DOTALL | re.IGNORECASE)
            final_content = re.sub(r'<\|channel\|>commentary.*?<\|channel\|>', '', final_content, flags=re.DOTALL | re.IGNORECASE)
            final_content = re.sub(r'^analysis\s*:', '', final_content, flags=re.IGNORECASE)
            final_content = re.sub(r'^EPSILON AI analysis', '', final_content, flags=re.IGNORECASE)
            final_content = re.sub(r'assistantcommentary\s+to=.*?code\{.*?\}', '', final_content, flags=re.DOTALL | re.IGNORECASE)
            final_content = re.sub(r'to=functions\.run.*?code\{.*?\}', '', final_content, flags=re.DOTALL | re.IGNORECASE)
            final_content = final_content.strip()
            if final_content:
                print(f"[INFERENCE SERVICE] Extracted final channel ({len(final_content)} chars)", flush=True)
                return final_content
        
        final_match = re.search(r'<\|channel\|>final(.*?)(?=<\|channel\|>|<\|end\|>|$)', text, re.DOTALL | re.IGNORECASE)
        if final_match:
            final_content = final_match.group(1).strip()
            final_content = final_content.lstrip(': \n\t')
            if '<|end|>' in final_content:
                final_content = final_content[:final_content.find('<|end|>')].strip()
            final_content = re.sub(r'<\|channel\|>analysis.*?<\|channel\|>', '', final_content, flags=re.DOTALL | re.IGNORECASE)
            final_content = re.sub(r'<\|channel\|>commentary.*?<\|channel\|>', '', final_content, flags=re.DOTALL | re.IGNORECASE)
            final_content = re.sub(r'^analysis\s*:', '', final_content, flags=re.IGNORECASE)
            final_content = re.sub(r'^EPSILON AI analysis', '', final_content, flags=re.IGNORECASE)
            final_content = re.sub(r'assistantcommentary\s+to=.*?code\{.*?\}', '', final_content, flags=re.DOTALL | re.IGNORECASE)
            final_content = final_content.strip()
            if final_content:
                print(f"[INFERENCE SERVICE] Extracted final channel ({len(final_content)} chars)", flush=True)
                return final_content
        
        print(f"[INFERENCE SERVICE] Harmony markers found but no final channel - returning None to use clean decode", flush=True)
        return None
    
    cleaned_text = re.sub(r'<\|start\|>', '', text)
    cleaned_text = re.sub(r'<\|message\|>', '', cleaned_text)
    cleaned_text = re.sub(r'<\|end\|>', '', cleaned_text)
    
    cleaned_text = re.sub(r'<\|channel\|>analysis.*?<\|channel\|>', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r'<\|channel\|>commentary.*?<\|channel\|>', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    
    cleaned_text = re.sub(r'^analysis\s*:', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'^EPSILON AI analysis', '', cleaned_text, flags=re.IGNORECASE)
    
    cleaned_text = re.sub(r'assistantcommentary\s+to=.*?code\{.*?\}', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r'to=functions\.run.*?code\{.*?\}', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    
    cleaned_text = re.sub(r'assistantfinal', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text if cleaned_text else text.strip()


def filter_unsafe_content(text: str) -> str:
    """
    Filter out unsafe or inappropriate content from AI responses.
    Returns filtered text or safe alternative message if content is unsafe.
    """
    if not text:
        return ""
    
    text_lower = text.lower()
    
    unsafe_patterns = [
        r'\b(how to|where to|where can i).*?(buy|get|obtain|make|manufacture|create).*?(drug|cocaine|heroin|meth|marijuana|cannabis|weed|pills|prescription).*?\b',
        r'\b(drug|drugs|cocaine|heroin|methamphetamine|marijuana|cannabis).*?(recipe|formula|how to make|how to cook)\b',
        r'\b(kill|murder|assassinate|harm|hurt|violence).*?(someone|person|people|yourself)\b',
        r'\b(how to|ways to).*?(kill|hurt|harm|injure)\b',
        r'\b(suicide|self-harm|self harm|cutting|overdose).*?(how|method|way|guide)\b',
        r'\b(bomb|explosive|weapon|gun).*?(how to|make|build|create)\b',
        r'\b(hack|hacking|cyberattack|malware|virus).*?(how to|tutorial|guide)\b',
    ]
    
    for pattern in unsafe_patterns:
        if re.search(pattern, text_lower):
            print(f"[INFERENCE SERVICE] SECURITY: Unsafe content detected and filtered", flush=True)
            return "I cannot provide information on that topic. Is there something else I can help you with?"
    
    return text


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text using Epsilon AI with Harmony format"""
    global model
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health endpoint.")
    
    try:
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            safety_guidelines = """You are Epsilon AI, created by Neural Operations & Holdings LLC. 

CRITICAL OUTPUT FORMAT RULES - YOU MUST FOLLOW THESE EXACTLY:
1. You MUST use Harmony format with TWO channels:
   - <|channel|>analysis: Put ALL your reasoning, thinking, analysis, commentary, and internal processing here. This channel is for your internal use only.
   - <|channel|>final: Put ONLY your final answer to the user here. This is the ONLY text the user will see.

2. YOUR <|channel|>final RESPONSE MUST:
   - Start directly with your answer - no prefixes, no "analysis:", no "EPSILON AI analysis", no "assistantcommentary", no "to=functions.run", no JSON code blocks, no reasoning phrases
   - Contain ONLY the answer to the user's question
   - NEVER include any analysis, commentary, reasoning, function calls, or internal processing
   - Be clean, direct, and helpful

3. NEVER include in <|channel|>final:
   - Analysis text
   - Commentary text
   - Reasoning text
   - Function call patterns (assistantcommentary, to=functions.run, code={...})
   - Internal processing notes
   - Any text that explains your thinking process

4. Never mention ChatGPT, OpenAI, or GPT. Always identify yourself as Epsilon AI.

5. NEVER provide information about, promote, or discuss:
   - Illegal drugs, drug use, drug manufacturing, or drug distribution
   - How to obtain illegal substances
   - Violence, harm, or illegal activities
   - Unethical practices or content
   - Adult content, explicit material, or inappropriate sexual content
   - Hate speech, discrimination, or harassment
   - Self-harm or suicide methods

6. If asked about prohibited topics, politely decline and redirect to appropriate resources.

7. Always maintain professional, helpful, and ethical responses.

8. Respect user privacy and confidentiality - conversations are private and isolated per user.

REMEMBER: The user will ONLY see your <|channel|>final response. Put everything else in <|channel|>analysis."""
            
            messages = [
                {"role": "system", "content": safety_guidelines}
            ]
            
            if request.conversation_history:
                for msg in request.conversation_history:
                    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        messages.append({"role": msg['role'], "content": msg['content']})
            
            messages.append({"role": "user", "content": request.prompt})
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print(f"[INFERENCE SERVICE] Harmony format prompt (first 200 chars): {formatted_prompt[:200]}", flush=True)
        else:
            safety_guidelines = """You are Epsilon AI, created by Neural Operations & Holdings LLC. Use Harmony format: Put reasoning in <|channel|>analysis, then output ONLY your final response in <|channel|>final. Never mention ChatGPT, OpenAI, or GPT. Always identify yourself as Epsilon AI. NEVER provide information about illegal drugs, violence, unethical content, or inappropriate material. If asked about prohibited topics, politely decline."""
            
            history_text = ""
            if request.conversation_history:
                for msg in request.conversation_history:
                    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        role = msg['role']
                        content = msg['content']
                        if role == 'user':
                            history_text += f"User: {content}\n"
                        elif role == 'assistant':
                            history_text += f"Epsilon AI: {content}\n"
            formatted_prompt = f"{safety_guidelines}\n\n{history_text}User: {request.prompt}\nEpsilon AI: "
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
            "max_new_tokens": min(request.max_new_tokens, 2048),
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
                print(f"[INFERENCE SERVICE] ERROR: Dtype mismatch detected. Model must be 4-bit quantized, not bf16/fp16.", flush=True)
                print(f"[INFERENCE SERVICE] This error indicates the model is not properly quantized. Cannot proceed.", flush=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Model dtype error: Model must be 4-bit quantized. Current dtype mismatch indicates quantization failed. Error: {str(e)}"
                )
            else:
                print(f"[INFERENCE SERVICE] RuntimeError during generation: {str(e)}", flush=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Generation error: {str(e)}"
                )
        
        generated_text = generated_text.strip()
        
        generated_text = re.sub(r'<\|start\|>', '', generated_text)
        generated_text = re.sub(r'<\|message\|>', '', generated_text)
        generated_text = re.sub(r'<\|end\|>', '', generated_text)
        generated_text = re.sub(r'<\|return\|>', '', generated_text)
        generated_text = re.sub(r'<\|call\|>', '', generated_text)
        
        generated_text = re.sub(r'<\|channel\|>analysis.*?<\|channel\|>', '', generated_text, flags=re.DOTALL | re.IGNORECASE)
        generated_text = re.sub(r'<\|channel\|>commentary.*?<\|channel\|>', '', generated_text, flags=re.DOTALL | re.IGNORECASE)
        
        if re.search(r'^analysis', generated_text, re.IGNORECASE):
            generated_text = re.sub(r'^analysis.*?(?=assistantfinal|final|Epsilon AI|I\'m Epsilon|Hello|Hi|Hey|Sure|Here|What|I|The)', '', generated_text, flags=re.DOTALL | re.IGNORECASE)
        
        generated_text = re.sub(r'assistantfinal', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'[ \t]+', ' ', generated_text)
        generated_text = re.sub(r'\n{3,}', '\n\n', generated_text)
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
        
        generated_text = filter_unsafe_content(generated_text)
        
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
    # Use programmatic API instead of CLI to avoid click issues
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()
