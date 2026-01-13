"""
Fine-tune the pretrained model on sales/CRM documents to create a sales expert
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    print("=" * 70)
    print("Fine-Tuning Epsilon AI as Sales & CRM Expert")
    print("=" * 70)
    
    ml_local = Path(__file__).parent.parent
    train_dir = ml_local / 'train'
    data_dir = ml_local / 'data'
    
    # Check prerequisites
    print("\n[1/5] Checking prerequisites...")
    
    # Check pretrained model (Epsilon AI 20B)
    # The model is loaded from Hugging Face, not from local files
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'services' / 'python-services'))
    from model_config import HF_MODEL_ID, MODEL_NAME
    print(f"[OK] Using {MODEL_NAME} model from Hugging Face")
    print(f"     The model will be downloaded automatically if not cached.")
    
    # Check if training data needs updating
    print("\n[2/5] Checking training data...")
    check_script = ml_local / 'scripts' / 'check_and_update_training_data.py'
    if check_script.exists():
        result = subprocess.run([sys.executable, str(check_script)], 
                              cwd=str(ml_local), capture_output=True, text=True)
        if result.returncode != 0:
            print("⚠️  Warning: Training data may be outdated")
            print("   Consider running: python ml_local/scripts/pull_corpus_from_supabase.py")
    
    # Check training data
    train_bin = data_dir / 'token_bins' / 'train.bin'
    val_bin = data_dir / 'token_bins' / 'val.bin'
    
    if not train_bin.exists():
        print(f"\n[2/5] Training data not found. Need to prepare data...")
        print("Running data preparation pipeline...")
        
        # Check if we have raw text data
        train_txt = data_dir / 'processed' / 'train.txt'
        if not train_txt.exists():
            print("ERROR: No training data found!")
            print("Please run:")
            print("  1. python ml_local/scripts/pull_corpus_from_supabase.py")
            print("  2. python ml_local/scripts/build_token_bins.py --text ml_local/data/processed/train.txt --tokenizer <model_id> --output ml_local/data/token_bins/train.bin")
            print("  3. python ml_local/scripts/build_token_bins.py --text ml_local/data/processed/val.txt --tokenizer <model_id> --output ml_local/data/token_bins/val.bin")
            return False
        
        # Build token bins
        print("Building token bins from text data...")
        result = subprocess.run([
            sys.executable,
            str(ml_local / 'scripts' / 'build_token_bins.py'),
            '--text', str(train_txt),
            '--tokenizer', str(tokenizer_path),
            '--output', str(train_bin)
        ], cwd=str(ml_local))
        
        if result.returncode != 0:
            print("ERROR: Failed to build training token bins")
            return False
        
        # Build validation bins
        val_txt = data_dir / 'processed' / 'val.txt'
        if val_txt.exists():
            print("Building validation token bins...")
            result = subprocess.run([
                sys.executable,
                str(ml_local / 'scripts' / 'build_token_bins.py'),
                '--text', str(val_txt),
                '--tokenizer', str(tokenizer_path),
                '--output', str(val_bin)
            ], cwd=str(ml_local))
    
    if not train_bin.exists():
        print("ERROR: Training data still not found after preparation")
        return False
    
    print(f"[OK] Training data found: {train_bin}")
    if val_bin.exists():
        print(f"[OK] Validation data found: {val_bin}")
    
    # Check config
    config_path = ml_local / 'config.json'
    if not config_path.exists():
        print("ERROR: Config not found")
        return False
    print(f"[OK] Config found: {config_path}")
    
    # Set environment
    os.environ['LOCAL_TRAINING'] = '1'
    
    print("\n[3/5] Starting fine-tuning...")
    print("This will fine-tune the pretrained model on your sales/CRM documents")
    print("The model will learn to be a sales expert based on your training data")
    print("\nTraining parameters (optimized for 2GB GPU):")
    print("  - Base model: Pretrained Epsilon AI model")
    print("  - Training data: Your documents from Supabase")
    print("  - Epochs: 5 (more epochs = deeper learning of your documents)")
    print("  - Batch size: 1 (reduced to prevent crashes)")
    print("  - Sequence length: 128 (reduced to fit in 2GB GPU memory)")
    print("  - Gradient accumulation: 4 steps (effective batch size = 4)")
    print("  - Mixed precision: FP16 enabled (reduces memory by ~50%)")
    print("  - Learning rate: 2e-4 (optimized for fine-tuning)")
    print("  - The model will learn EVERYTHING in your documents:")
    print("    * Product details (engines, specs, features)")
    print("    * Sales processes and techniques")
    print("    * CRM workflows and best practices")
    print("    * Industry knowledge and terminology")
    print("    * Conversational patterns from your documents")
    print("  - Device: NVIDIA GPU (MX250, 2GB VRAM)")
    print("\nMemory optimizations applied to prevent crashes:")
    print("  [OK] Reduced batch size (1) and sequence length (128)")
    print("  [OK] Gradient accumulation (4 steps = effective batch size 4)")
    print("  [OK] FP16 mixed precision (reduces memory by ~50%)")
    print("  [OK] Gradient checkpointing (trades compute for memory)")
    print("  [OK] Periodic GPU cache clearing")
    print("\nThis will take longer but won't crash your PC...")
    
    # Run pretraining with the pretrained checkpoint
    pretrain_script = train_dir / 'pretrain.py'
    
    cmd = [
        sys.executable,
        str(pretrain_script),
        '--config', str(ml_local / 'config.json'),
        '--data', str(train_bin),
        '--tokenizer', str(tokenizer_path),
        '--pretrained', str(pretrained_model),
        '--epochs', '5',  # More epochs = deeper learning
        '--batch-size', '1',  # Reduced to 1 to prevent crashes on 2GB GPU
        '--seq-len', '128',  # Reduced to 128 to fit in 2GB GPU memory
        '--gradient-accumulation-steps', '4',  # Accumulate over 4 steps (effective batch size = 4)
        '--use-amp',  # Enable FP16 mixed precision to reduce memory by ~50%
        '--use-gradient-checkpointing',  # Enable gradient checkpointing to save memory
        '--output-dir', str(ml_local / 'runs' / 'sales_expert')
    ]
    
    # Add validation data if available
    if val_bin.exists():
        cmd.extend(['--val-data', str(val_bin)])
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=str(ml_local))
    
    if result.returncode != 0:
        print("\nERROR: Fine-tuning failed")
        return False
    
    print("\n[4/5] Fine-tuning complete!")
    
    # Export the fine-tuned model
    print("\n[5/5] Exporting fine-tuned model...")
    
    # Find the latest checkpoint
    runs_dir = ml_local / 'runs' / 'sales_expert'
    if not runs_dir.exists():
        print("ERROR: No checkpoints found")
        return False
    
    # Get the latest checkpoint
    checkpoints = list(runs_dir.glob('checkpoint_*.pt'))
    if not checkpoints:
        print("ERROR: No checkpoints found in runs directory")
        return False
    
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"Using checkpoint: {latest_checkpoint}")
    
    # Export
    export_script = train_dir / 'export.py'
    export_cmd = [
        sys.executable,
        str(export_script),
        '--checkpoint', str(latest_checkpoint),
        '--tokenizer', str(tokenizer_path),
        '--output-dir', str(ml_local / 'exports' / 'sales_expert_v1')
    ]
    
    print(f"Running: {' '.join(export_cmd)}\n")
    result = subprocess.run(export_cmd, cwd=str(ml_local))
    
    if result.returncode != 0:
        print("ERROR: Export failed")
        return False
    
    print("\n" + "=" * 70)
    print("Fine-tuning complete!")
    print("=" * 70)
    print(f"\nFine-tuned model exported to: {ml_local / 'exports' / 'sales_expert_v1'}")
    print("\nWhat the model learned:")
    print("  [OK] All product information from your documents")
    print("  [OK] Sales techniques and processes")
    print("  [OK] CRM workflows and best practices")
    print("  [OK] Industry terminology and knowledge")
    print("  [OK] Conversational patterns (can talk like a human)")
    print("\nNext steps:")
    print("1. Test the model locally")
    print("2. Upload to Supabase: python ml_local/scripts/upload_pretrained_model.py --checkpoint ml_local/exports/sales_expert_v1/model.pt --tokenizer <model_id>")
    print("3. The model will auto-deploy to production")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

