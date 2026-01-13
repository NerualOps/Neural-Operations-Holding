"""
Upload already-downloaded Epsilon AI model to Supabase for production
Created by Neural Operations & Holdings LLC
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import zipfile
from dotenv import load_dotenv
from supabase import create_client, Client

# Import model config
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'services' / 'python-services'))
from model_config import HF_MODEL_ID, MODEL_NAME, COMPANY_NAME

load_dotenv()

def upload_to_supabase():
    """Upload the model to Supabase Storage"""
    print("=" * 70)
    print(f"Uploading {MODEL_NAME} to Supabase")
    print(f"Created by {COMPANY_NAME}")
    print("=" * 70)
    
    # Check for full model directory first (saved with save_pretrained)
    full_model_dir = Path(__file__).parent.parent.parent / 'services' / 'python-services' / 'models' / 'epsilon-20b-full'
    
    if full_model_dir.exists():
        print(f"\nFound full model directory: {full_model_dir}")
        model_dir = full_model_dir
    else:
        # Fall back to Hugging Face cache
        model_dir = Path(__file__).parent.parent.parent / 'services' / 'python-services' / 'models' / 'epsilon-20b'
        cache_base = model_dir / f"models--{HF_MODEL_ID.replace('/', '--')}"
        
        if not cache_base.exists():
            print(f"ERROR: Model not found!")
            print(f"Please run download_full_model.py first to download the full model")
            return False
        
        # Find the snapshot directory
        snapshots_dir = cache_base / "snapshots"
        if not snapshots_dir.exists():
            print(f"ERROR: Snapshots directory not found at {snapshots_dir}")
            return False
        
        # Get the first (and usually only) snapshot
        snapshot_dirs = list(snapshots_dir.iterdir())
        if not snapshot_dirs:
            print(f"ERROR: No snapshots found in {snapshots_dir}")
            return False
        
        model_dir = snapshot_dirs[0]
        print(f"\nFound model snapshot: {model_dir.name}")
    
    print(f"Model path: {model_dir}")
    
    # Get Supabase credentials
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
    
    if not supabase_url or not supabase_key:
        print("ERROR: Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env")
        return False
    
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # Create zip file with model files
    print("\n[1/3] Creating zip archive...")
    zip_path = Path(__file__).parent.parent.parent / 'services' / 'python-services' / 'models' / 'epsilon-20b-production.zip'
    
    # Create zip with all model files
    model_files = []
    total_size = 0
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in model_dir.rglob('*'):
            if file_path.is_file():
                # Include all files - safetensors, bin, json, etc.
                rel_path = file_path.relative_to(model_dir)
                file_size = file_path.stat().st_size
                total_size += file_size
                zipf.write(file_path, f"epsilon-20b/{rel_path}")
                model_files.append((rel_path, file_size))
                if len(model_files) <= 20:
                    size_mb = file_size / (1024 * 1024)
                    if size_mb > 1:
                        print(f"  Added: {rel_path} ({size_mb:.2f} MB)")
                    else:
                        print(f"  Added: {rel_path}")
        
        if len(model_files) > 20:
            print(f"  ... and {len(model_files) - 20} more files")
        
        # Show summary
        total_size_gb = total_size / (1024**3)
        print(f"\n  Total files: {len(model_files)}")
        print(f"  Total size: {total_size_gb:.2f} GB")
        
        # List large files
        large_files = sorted([f for f in model_files if f[1] > 100 * 1024 * 1024], key=lambda x: x[1], reverse=True)
        if large_files:
            print(f"\n  Large files (>100MB):")
            for rel_path, size in large_files[:10]:
                size_gb = size / (1024**3)
                print(f"    {rel_path}: {size_gb:.2f} GB")
    
    file_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Created zip archive: {zip_path.name} ({file_size_mb:.2f} MB)")
    
    # Upload to Supabase
    print(f"\n[2/3] Uploading to Supabase Storage...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    storage_path = f"pretrained_models/epsilon-20b-production-{timestamp}.zip"
    
    # Check if file is too large for single upload (Supabase limit is usually 50MB)
    if file_size_mb > 50:
        print(f"  File is large ({file_size_mb:.2f} MB), using chunked upload...")
        # Chunk the file
        chunk_size = 50 * 1024 * 1024  # 50MB chunks
        chunks = []
        
        with open(zip_path, 'rb') as f:
            chunk_index = 0
            while True:
                chunk_data = f.read(chunk_size)
                if not chunk_data:
                    break
                
                chunk_path = f"pretrained_models/epsilon-20b-production-{timestamp}-chunk-{chunk_index}.zip"
                response = supabase.storage.from_('epsilon-models').upload(
                    path=chunk_path,
                    file=chunk_data,
                    file_options={
                        'content-type': 'application/zip',
                        'upsert': 'true'
                    }
                )
                chunks.append({"index": chunk_index, "path": chunk_path})
                chunk_index += 1
                print(f"  Uploaded chunk {chunk_index}/{len(chunks)}")
        
        # Save chunk metadata
        metadata_path = f"pretrained_models/epsilon-20b-production-{timestamp}.zip.metadata.json"
        chunk_metadata = {
            "model_name": MODEL_NAME,
            "company": COMPANY_NAME,
            "chunks": chunks,
            "total_size_mb": file_size_mb,
            "created_at": datetime.now().isoformat()
        }
        metadata_json = json.dumps(chunk_metadata, indent=2)
        
        response = supabase.storage.from_('epsilon-models').upload(
            path=metadata_path,
            file=metadata_json.encode('utf-8'),
            file_options={
                'content-type': 'application/json',
                'upsert': 'true'
            }
        )
        
        print(f"[OK] Uploaded {len(chunks)} chunks + metadata")
        storage_path = metadata_path
    else:
        # Single file upload
        print(f"  Uploading as single file ({file_size_mb:.2f} MB)...")
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
    
    # Create deployment record
    print(f"\n[3/3] Creating deployment record...")
    
    model_id = f"epsilon-20b-{timestamp}"
    
    # Count parameters (approximate - 20B model)
    param_count = 20_000_000_000
    
    deployment_data = {
        "model_id": model_id,
        "storage_path": storage_path,
        "version": "1.0.0",
        "status": "approved",  # Auto-approve production model
        "learning_description": f"{MODEL_NAME} - Production model uploaded by script",
        "stats": {
            "parameters": param_count,
            "model_name": MODEL_NAME,
            "company": COMPANY_NAME,
            "file_size_mb": file_size_mb
        },
        "temperature": 0.7,  # Default temperature
        "deployed_at": datetime.now().isoformat(),
        "deployed_by": "upload_model_to_supabase_script"
    }
    
    result = supabase.table('epsilon_model_deployments').insert(deployment_data).execute()
    
    if result.data:
        print(f"[OK] Deployment record created: {model_id}")
        print(f"     Status: approved (ready for production)")
    else:
        print(f"[WARN] Deployment record may not have been created")
    
    print(f"\n{'='*70}")
    print(f"Upload Complete!")
    print(f"{'='*70}")
    print(f"Storage path: {storage_path}")
    print(f"Model ID: {model_id}")
    print(f"Status: approved (ready for production)")
    print(f"\nModel is now available in production!")
    print(f"{'='*70}")
    
    # Clean up local zip
    try:
        zip_path.unlink()
        print(f"\n[CLEANUP] Removed local zip file")
    except Exception as e:
        print(f"\n[CLEANUP] Could not remove zip file: {e}")
    
    return True

if __name__ == "__main__":
    success = upload_to_supabase()
    sys.exit(0 if success else 1)

