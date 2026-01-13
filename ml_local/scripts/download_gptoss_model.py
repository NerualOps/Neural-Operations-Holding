"""
Download and setup Epsilon AI model from Hugging Face
Created by Neural Operations & Holdings LLC
"""
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import model config
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'services' / 'python-services'))
from model_config import HF_MODEL_ID, MODEL_NAME, COMPANY_NAME

def main():
    print("=" * 70)
    print(f"Downloading {MODEL_NAME}")
    print(f"Created by {COMPANY_NAME}")
    print("=" * 70)
    
    model_id = HF_MODEL_ID
    model_dir = Path(__file__).parent.parent.parent / 'services' / 'python-services' / 'models' / 'epsilon-20b'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nModel ID: {model_id}")
    print(f"Download directory: {model_dir}")
    print(f"\nThis will download ~40GB of model files.")
    print(f"Make sure you have enough disk space and a stable internet connection.")
    print(f"\nStarting download...\n")
    
    try:
        # Download tokenizer
        print("[1/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=str(model_dir)
        )
        print("✓ Tokenizer downloaded")
        
        # Download model (this will take a while)
        print("\n[2/2] Downloading model (this may take 30+ minutes)...")
        print("   The model is ~40GB, so this will take time depending on your connection.")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=str(model_dir),
            low_cpu_mem_usage=True
        )
        
        print("✓ Model downloaded")
        
        print("\n" + "=" * 70)
        print("Download Complete!")
        print("=" * 70)
        print(f"\nModel saved to: {model_dir}")
        print("\nNext steps:")
        print("1. The inference service will automatically use this model")
        print("2. Restart the inference service")
        
    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

