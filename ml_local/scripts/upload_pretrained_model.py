"""
Upload an Epsilon AI model checkpoint to Supabase Storage
"""
import os
import sys
import json
import argparse
import zipfile
import tempfile
import torch
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# LOCAL-ONLY ENFORCEMENT
if os.getenv('LOCAL_TRAINING') != '1':
    print("ERROR: This script can only run locally. Set LOCAL_TRAINING=1")
    sys.exit(1)


def upload_pretrained_model(checkpoint_path: str, tokenizer_path: str = None, 
                           description: str = None):
    """
    Upload an Epsilon AI model to Supabase Storage
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        tokenizer_path: Optional path to tokenizer.json
        description: Optional description of the model
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Get Supabase credentials
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
    
    if not supabase_url or not supabase_key:
        raise ValueError(
            "Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env"
        )
    
    supabase: Client = create_client(supabase_url, supabase_key)
    
    print(f"{'='*60}")
    print(f"Uploading Epsilon AI Model to Supabase")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Create zip file with checkpoint and extract config/tokenizer
    zip_path = checkpoint_path.with_suffix('.zip')
    print(f"[1/3] Creating zip archive...")
    
    # Load checkpoint to extract config
    # Try to load checkpoint - handle different formats
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        # If direct load fails, try loading as state dict only
        print(f"  [WARN] Failed to load as full checkpoint: {e}")
        print(f"  [INFO] Attempting to load as state dict only...")
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            # If it's just a state dict, we need config from elsewhere
            # Check if config.json exists nearby
            config_path = checkpoint_path.parent / 'config.json'
            if not config_path.exists():
                config_path = Path('ml_local/config.json')
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                checkpoint = {
                    'model_state_dict': state_dict,
                    'config': config_dict
                }
                print(f"  [OK] Loaded state dict and config from {config_path}")
            else:
                raise ValueError(f"Checkpoint is state dict only and no config.json found. Please provide config.json or use a full checkpoint with 'config' key.")
        except Exception as e2:
            raise ValueError(f"Cannot load checkpoint file: {e2}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add model weights (save as model.pt for inference)
        model_state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Save model state dict to temp file, then add to zip
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                torch.save(model_state_dict, tmp.name)
                zipf.write(tmp.name, 'model.pt')
                tmp_path = tmp.name
        finally:
            # Clean up temp file after adding to zip
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    print(f"  [WARN] Failed to delete temp file: {e}")
        
        print(f"  [OK] Added model.pt")
        
        # Extract and add config.json
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            # Ensure config is a dict, not a TransformerConfig object
            if hasattr(config_dict, 'to_dict'):
                config_dict = config_dict.to_dict()
            elif not isinstance(config_dict, dict):
                raise ValueError(f"Config must be a dict, got {type(config_dict)}")
            config_json = json.dumps(config_dict, indent=2)
            zipf.writestr('config.json', config_json)
            print(f"  [OK] Added config.json")
        else:
            raise ValueError("Checkpoint missing 'config' - cannot create model artifact")
        
        # Add tokenizer (required for inference)
        tokenizer_found = False
        if tokenizer_path:
            tokenizer_path = Path(tokenizer_path)
            if tokenizer_path.exists():
                zipf.write(tokenizer_path, 'tokenizer.json')
                print(f"  [OK] Added tokenizer.json")
                tokenizer_found = True
        
        if not tokenizer_found:
            # Try to find tokenizer in common locations
            common_tokenizer_paths = [
                Path(__file__).parent.parent / 'pretrained_models' / 'epsilon-20b',
                Path(__file__).parent.parent.parent / 'services' / 'python-services' / 'models' / 'epsilon-20b'
            ]
            for tp in common_tokenizer_paths:
                if tp.exists():
                    zipf.write(tp, 'tokenizer.json')
                    print(f"  [OK] Added tokenizer.json from {tp}")
                    tokenizer_found = True
                    break
        
        if not tokenizer_found:
            raise ValueError("tokenizer.json is required but not found. Please provide --tokenizer path")
        
        # Add metadata
        metadata = {
            'checkpoint_file': checkpoint_path.name,
            'upload_date': datetime.now().isoformat(),
            'description': description or 'Epsilon AI model',
            'license': 'Proprietary',
            'source': 'Epsilon AI'
        }
        
        metadata_str = json.dumps(metadata, indent=2)
        zipf.writestr('metadata.json', metadata_str)
        print(f"  [OK] Added metadata.json")
    
    print(f"[OK] Created zip: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB)")
    
    # Upload to Supabase Storage (with chunking for large files)
    print(f"\n[2/3] Uploading to Supabase Storage...")
    file_size_mb = zip_path.stat().st_size / 1024 / 1024
    chunk_size_mb = 40  # 40MB chunks (under Supabase limit)
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    
    storage_path = f"pretrained_models/{checkpoint_path.stem}.zip"
    
    try:
        # Check if file needs chunking (Supabase limit is usually 50MB)
        if file_size_mb > 45:
            print(f"File is large ({file_size_mb:.2f} MB). Splitting into chunks...")
            
            # Split file into chunks
            chunk_paths = []
            chunk_metadata = {
                'original_file': checkpoint_path.stem + '.zip',
                'total_size': zip_path.stat().st_size,
                'chunk_size': chunk_size_bytes,
                'chunks': []
            }
            
            with open(zip_path, 'rb') as f:
                chunk_index = 0
                while True:
                    chunk_data = f.read(chunk_size_bytes)
                    if not chunk_data:
                        break
                    
                    chunk_filename = f"{checkpoint_path.stem}.zip.chunk{chunk_index:03d}"
                    chunk_storage_path = f"pretrained_models/{chunk_filename}"
                    
                    # Upload chunk
                    print(f"  Uploading chunk {chunk_index + 1} ({len(chunk_data) / 1024 / 1024:.2f} MB)...")
                    response = supabase.storage.from_('epsilon-models').upload(
                        path=chunk_storage_path,
                        file=chunk_data,
                        file_options={
                            'content-type': 'application/octet-stream',
                            'upsert': 'true'
                        }
                    )
                    
                    chunk_metadata['chunks'].append({
                        'index': chunk_index,
                        'path': chunk_storage_path,
                        'size': len(chunk_data)
                    })
                    
                    chunk_paths.append(chunk_storage_path)
                    chunk_index += 1
            
            # Save chunk metadata
            metadata_path = f"pretrained_models/{checkpoint_path.stem}.zip.metadata.json"
            metadata_json = json.dumps(chunk_metadata, indent=2)
            
            response = supabase.storage.from_('epsilon-models').upload(
                path=metadata_path,
                file=metadata_json.encode('utf-8'),
                file_options={
                    'content-type': 'application/json',
                    'upsert': 'true'
                }
            )
            
            print(f"[OK] Uploaded {chunk_index} chunks + metadata")
            print(f"  Main storage path: {metadata_path} (use this for download)")
            storage_path = metadata_path  # Use metadata path as main reference
            
        else:
            # Upload as single file
            print(f"Uploading as single file ({file_size_mb:.2f} MB)...")
            with open(zip_path, 'rb') as f:
                file_data = f.read()
                
            response = supabase.storage.from_('epsilon-models').upload(
                path=storage_path,
                file=file_data,
                file_options={
                    'content-type': 'application/zip',
                    'upsert': 'true'
                }
            )
            
            print(f"[OK] Uploaded to: {storage_path}")
            
    except Exception as e:
        print(f"ERROR: Upload failed: {e}")
        raise
    
    # Validate model quality (6 safety checks)
    print(f"\n[3/4] Validating model quality...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from validate_model_quality import validate_model
    
    # Find config.json (might be in zip or checkpoint)
    config_path = None
    if 'config' in checkpoint:
        # Save config temporarily for validation
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(checkpoint['config'] if isinstance(checkpoint['config'], dict) else checkpoint['config'].__dict__, tmp, indent=2)
            config_path = tmp.name
    
    validation_passed, validation_results = validate_model(
        str(checkpoint_path),
        str(tokenizer_path) if tokenizer_path else None,
        config_path
    )
    
    if config_path and Path(config_path).exists():
        try:
            Path(config_path).unlink()
        except:
            pass
    
    if not validation_passed:
        print(f"ERROR: Model failed validation checks!")
        print(f"  Passed: {validation_results['checks_passed']}/{validation_results['total_checks']}")
        for check_name, check_result in validation_results['details'].items():
            status = "PASS" if check_result.get('passed') else "FAIL"
            reason = check_result.get('reason', '')
            print(f"  {check_name}: {status} {reason}")
        raise ValueError("Model did not pass quality validation checks. Cannot auto-approve.")
    
    print(f"✓ All {validation_results['checks_passed']}/{validation_results['total_checks']} validation checks passed!")
    
    # Create deployment record in database (auto-approved)
    print(f"\n[4/4] Creating deployment record (auto-approved)...")
    
    try:
        file_size_mb = zip_path.stat().st_size / 1024 / 1024
        is_chunked = file_size_mb > 45
        
        # Generate a unique model_id
        model_id = f"pretrained-{checkpoint_path.stem}-{int(datetime.now().timestamp())}"
        
        deployment_data = {
            'model_id': model_id,
            'storage_path': storage_path,
            'status': 'approved',  # Auto-approved after passing validation
            'quality_score': validation_results['checks_passed'] / validation_results['total_checks'],  # Quality score based on checks
            'stats': {
                'file_size_mb': file_size_mb,
                'upload_date': datetime.now().isoformat(),
                'source': 'Epsilon AI',
                'chunked': is_chunked,
                'description': description or 'Epsilon AI model',
                'license': 'Proprietary',
                'model_type': 'pretrained',
                'checkpoint_name': checkpoint_path.stem,
                'validation_results': validation_results['details']
            },
            'version': '1.0.0',
            'temperature': 0.9,
            'deployed_at': datetime.now().isoformat(),
            'deployed_by': 'auto-upload-script',
            'approved_by': 'auto-validation-system',
            'approved_at': datetime.now().isoformat()
        }
        
        result = supabase.table('epsilon_model_deployments').insert(deployment_data).execute()
        
        if result.data:
            deploy_id = result.data[0]['id']
            print(f"[OK] Created deployment record: {deploy_id}")
        else:
            print(f"⚠ Created deployment but no ID returned")
            deploy_id = None
    except Exception as e:
        print(f"WARNING: Failed to create deployment record: {e}")
        print("  Model uploaded to storage but not registered in database")
        deploy_id = None
    
    print(f"\n{'='*60}")
    print(f"Upload Complete!")
    print(f"{'='*60}")
    print(f"Storage path: {storage_path}")
    if deploy_id:
        print(f"Deployment ID: {deploy_id}")
    print(f"\nModel is now live in production (auto-approved after passing {validation_results['checks_passed']} validation checks)!")
    print(f"{'='*60}")
    
    # Clean up local zip file after successful upload
    if zip_path.exists():
        try:
            zip_path.unlink()
            print(f"\n[CLEANUP] Removed local zip file: {zip_path.name}")
        except Exception as e:
            print(f"\n[WARN] Failed to remove local zip file: {e}")
    
    return storage_path, deploy_id


def main():
    parser = argparse.ArgumentParser(
        description='Upload Epsilon AI model to Supabase Storage'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to .pt checkpoint file'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='ml_local/tokenizer/tokenizer.json',
        help='Path to tokenizer.json (default: ml_local/tokenizer/tokenizer.json)'
    )
    parser.add_argument(
        '--description',
        type=str,
        help='Optional description of the model'
    )
    
    args = parser.parse_args()
    
    upload_pretrained_model(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        description=args.description
    )


if __name__ == '__main__':
    main()

