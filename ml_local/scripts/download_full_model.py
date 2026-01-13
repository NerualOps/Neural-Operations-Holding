"""
Download the full Epsilon AI model with all weights and save locally
Created by Neural Operations & Holdings LLC

This script downloads files directly to disk without loading into memory,
which is essential for large models like the 20B parameter model.
"""
import os
import sys
import shutil
from pathlib import Path

# Check for required dependencies
try:
    from huggingface_hub import snapshot_download  # type: ignore
except ImportError:
    print("ERROR: huggingface_hub is not installed.")
    print("Please install it with: pip install huggingface_hub")
    sys.exit(1)

# Import model config
config_path = Path(__file__).parent.parent.parent / 'services' / 'python-services'
sys.path.insert(0, str(config_path))

try:
    from model_config import HF_MODEL_ID, MODEL_NAME, COMPANY_NAME  # type: ignore
except ImportError as e:
    print(f"ERROR: Could not import model_config: {e}")
    print(f"Expected config file at: {config_path / 'model_config.py'}")
    sys.exit(1)

def cleanup_old_files(base_dir: Path):
    """Remove old GPT-2 files and corrupt/incomplete downloads"""
    print("\n[Cleaning up old files...]")
    
    # Directories to check and clean
    dirs_to_clean = [
        base_dir / 'epsilon-20b-full',  # Current download target
        base_dir / 'epsilon-20b',  # Alternative location
        base_dir / 'gpt2',  # Old GPT-2 files
        base_dir / 'pretrained',  # Old pretrained models
    ]
    
    # Also check for GPT-2 in Hugging Face cache
    hf_cache_base = Path.home() / '.cache' / 'huggingface' / 'hub'
    gpt2_cache_patterns = [
        'models--gpt2',
        'models--openai--gpt2',
    ]
    
    cleaned_count = 0
    cleaned_size = 0
    
    # Clean local model directories
    for dir_path in dirs_to_clean:
        if dir_path.exists():
            print(f"   Removing: {dir_path}")
            try:
                # Calculate size before deletion
                total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                cleaned_size += total_size
                
                # Remove directory
                shutil.rmtree(dir_path)
                cleaned_count += 1
                print(f"   [OK] Removed {total_size / (1024**3):.2f} GB")
            except Exception as e:
                print(f"   [WARNING] Could not remove {dir_path}: {e}")
    
    # Clean Hugging Face cache GPT-2 files (optional - be careful)
    if hf_cache_base.exists():
        for pattern in gpt2_cache_patterns:
            cache_dir = hf_cache_base / pattern
            if cache_dir.exists():
                print(f"   Found GPT-2 cache: {cache_dir}")
                print(f"   [INFO] Hugging Face cache GPT-2 files found but not removed")
                print(f"   [INFO] You can manually remove them if needed")
    
    if cleaned_count > 0:
        print(f"\n[OK] Cleaned up {cleaned_count} directories ({cleaned_size / (1024**3):.2f} GB freed)")
    else:
        print(f"[OK] No old files found to clean")
    
    print()

def main():
    print("=" * 70)
    print(f"Downloading Full {MODEL_NAME}")
    print(f"Created by {COMPANY_NAME}")
    print("=" * 70)
    
    # Base models directory
    base_dir = Path(__file__).parent.parent.parent / 'services' / 'python-services' / 'models'
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up old files first
    cleanup_old_files(base_dir)
    
    # Create output directory for full model (fresh start)
    output_dir = base_dir / 'epsilon-20b-full'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Model ID: {HF_MODEL_ID}")
    print(f"Output directory: {output_dir}")
    print(f"\nThis will download ~40GB of model files.")
    print(f"Make sure you have enough disk space and a stable internet connection.")
    print(f"\nDownloading directly to disk (no memory loading)...")
    print(f"This method is safer for large models and will resume if interrupted.\n")
    
    try:
        # Download all model files directly to disk using snapshot_download
        # This downloads files without loading into memory, which is essential for 20B models
        print("[Downloading all model files...]")
        print("   This includes: model weights, tokenizer, config files")
        print("   Progress will be shown below...\n")
        
        downloaded_path = snapshot_download(
            repo_id=HF_MODEL_ID,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,  # Copy files, don't symlink
            resume_download=True  # Resume if interrupted
        )
        
        print("\n[OK] All files downloaded!")
        
        # Verify download by checking file sizes
        print("\n[Verifying download...]")
        weight_files = list(output_dir.rglob('*.safetensors')) + list(output_dir.rglob('*.bin'))
        config_files = list(output_dir.rglob('*.json'))
        tokenizer_files = list(output_dir.rglob('tokenizer*'))
        
        print(f"   Found {len(weight_files)} weight files")
        print(f"   Found {len(config_files)} config files")
        print(f"   Found {len(tokenizer_files)} tokenizer files")
        
        # Calculate total size
        all_files = list(output_dir.rglob('*'))
        all_files = [f for f in all_files if f.is_file()]
        total_size = sum(f.stat().st_size for f in all_files)
        total_size_gb = total_size / (1024**3)
        
        print(f"\n{'='*70}")
        print(f"Download Complete!")
        print(f"{'='*70}")
        print(f"Total files: {len(all_files)}")
        print(f"Total size: {total_size_gb:.2f} GB")
        print(f"Model saved to: {output_dir}")
        print(f"\nNext step: Run upload_model_to_supabase.py to upload to production")
        print(f"{'='*70}")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n\nDownload interrupted by user.")
        print(f"Progress has been saved. Run the script again to resume.")
        return False
    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nNote: Partial downloads are saved. Run the script again to resume.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

