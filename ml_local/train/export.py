"""
Export trained model for inference
Creates an inference-ready artifact
"""
import os
import sys
import json
import argparse
import torch
import shutil
from pathlib import Path
from datetime import datetime
import subprocess

# Import from shared transformer core
# Add services/python-services to path to access epsilon_transformer_core
services_path = Path(__file__).parent.parent.parent / 'services' / 'python-services'
sys.path.insert(0, str(services_path))

from epsilon_transformer_core import EpsilonTransformerLM, TransformerConfig


def get_git_commit_hash():
    """Get current git commit hash"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except:
        return 'unknown'


def export_model(checkpoint_path: str, tokenizer_path: str, output_dir: str, dataset_hash: str = None):
    """
    Export model to inference-ready format
    
    Args:
        checkpoint_path: Path to training checkpoint
        tokenizer_path: Path to tokenizer.json
        output_dir: Output directory for export
        dataset_hash: Optional hash of training dataset
    """
    checkpoint_path = Path(checkpoint_path)
    tokenizer_path = Path(tokenizer_path)
    output_dir = Path(output_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting model from checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    
    # Load checkpoint
    device = torch.device('cpu')  # Export to CPU for inference
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config
    config = TransformerConfig.from_dict(checkpoint['config'])
    
    # Create model and load weights
    model = EpsilonTransformerLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    # Save model state dict
    model_path = output_dir / 'model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")
    
    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Config saved: {config_path}")
    
    # Copy tokenizer
    tokenizer_output = output_dir / 'tokenizer.json'
    shutil.copy2(tokenizer_path, tokenizer_output)
    print(f"Tokenizer copied: {tokenizer_output}")
    
    # Generate model_id from timestamp
    model_id = f"run_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}"
    
    # Create metadata with required fields
    metadata = {
        'model_id': model_id,
        'git_commit': get_git_commit_hash(),
        'dataset_hash': dataset_hash or 'unknown',
        'trained_steps': checkpoint.get('step', checkpoint.get('global_step', 'unknown')),
        'created_at': datetime.now().isoformat(),
        # Additional metadata
        'checkpoint_path': str(checkpoint_path),
        'epoch': checkpoint.get('epoch', 'unknown'),
        'loss': checkpoint.get('loss', 'unknown'),
        'model_params': sum(p.numel() for p in model.parameters()),
        'config': config.to_dict()
    }
    
    metadata_path = output_dir / 'run_meta.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")
    
    # Create special tokens file (if needed)
    special_tokens = {
        'pad_token': '<pad>',
        'unk_token': '<unk>',
        'bos_token': '<bos>',
        'eos_token': '<eos>'
    }
    special_tokens_path = output_dir / 'special_tokens.json'
    with open(special_tokens_path, 'w') as f:
        json.dump(special_tokens, f, indent=2)
    print(f"Special tokens saved: {special_tokens_path}")
    
    print(f"\n{'='*60}")
    print(f"Export complete!")
    print(f"Artifact directory: {output_dir}")
    print(f"Model parameters: {metadata['model_params']:,}")
    print(f"{'='*60}\n")
    
    # Auto-upload if requested
    if os.getenv('AUTO_UPLOAD_AFTER_EXPORT') == '1':
        print(f"\n{'='*60}")
        print(f"Auto-uploading model to Supabase...")
        print(f"{'='*60}\n")
        
        # Import and run auto-upload
        auto_upload_script = Path(__file__).parent.parent / 'scripts' / 'auto_upload_model.py'
        if auto_upload_script.exists():
            import subprocess
            result = subprocess.run([
                sys.executable,
                str(auto_upload_script),
                '--checkpoint', str(checkpoint_path),
                '--tokenizer', str(tokenizer_path),
                '--description', f'Exported model from {checkpoint_path.name}'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(result.stdout)
                print(f"\n{'='*60}")
                print(f"Model auto-uploaded and approved!")
                print(f"{'='*60}\n")
            else:
                print(f"WARNING: Auto-upload failed:")
                print(result.stderr)
                print(f"\nYou can manually upload using:")
                print(f"python ml_local/scripts/auto_upload_model.py --checkpoint {checkpoint_path} --tokenizer {tokenizer_path}")
        else:
            print(f"WARNING: Auto-upload script not found. Export complete but not uploaded.")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Export model for inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer.json')
    parser.add_argument('--output', type=str, default='../services/python-services/models/latest', help='Output directory')
    parser.add_argument('--dataset-hash', type=str, help='Dataset hash for metadata')
    
    args = parser.parse_args()
    
    export_model(args.checkpoint, args.tokenizer, args.output, args.dataset_hash)


if __name__ == '__main__':
    main()

