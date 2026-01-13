"""
Download Epsilon AI model and upload to Supabase for production
Created by Neural Operations & Holdings LLC
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import zipfile
import tempfile
from dotenv import load_dotenv
from supabase import create_client, Client

# Import model config
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'services' / 'python-services'))
from model_config import HF_MODEL_ID, MODEL_NAME, COMPANY_NAME

load_dotenv()

def download_model():
    """Download the model from Hugging Face"""
    print("=" * 70)
    print(f"Downloading {MODEL_NAME}")
    print(f"Created by {COMPANY_NAME}")
    print("=" * 70)
    
    model_dir = Path(__file__).parent.parent.parent / 'services' / 'python-services' / 'models' / 'epsilon-20b'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nModel ID: {HF_MODEL_ID}")
    print(f"Download directory: {model_dir}")
    print(f"\nThis will download ~40GB of model files.")
    print(f"Make sure you have enough disk space and a stable internet connection.")
    print(f"\nStarting download...\n")
    
    try:
        # Download tokenizer
        print("[1/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL_ID,
            trust_remote_code=True,
            cache_dir=str(model_dir)
        )
        print("[OK] Tokenizer downloaded")
        
        # Download model (this will take a while)
        print("\n[2/2] Downloading model (this may take 30+ minutes)...")
        print("   The model is ~40GB, so this will take time depending on your connection.")
        
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            trust_remote_code=True,
            cache_dir=str(model_dir),
            low_cpu_mem_usage=True
        )
        
        print("[OK] Model downloaded")
        return model_dir, tokenizer, model
        
    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def upload_to_supabase(model_dir, tokenizer, model):
    """Upload the model to Supabase Storage"""
    print("\n" + "=" * 70)
    print("Uploading Model to Supabase")
    print("=" * 70)
    
    # Get Supabase credentials
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
    
    if not supabase_url or not supabase_key:
        print("ERROR: Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env")
        return False
    
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # Create zip file with model files
    print("\n[1/3] Creating zip archive...")
    zip_path = model_dir / 'epsilon-20b-production.zip'
    
    # Find the actual model files in Hugging Face cache
    # The cache structure is: cache_dir/models--org--model/snapshots/hash/
    cache_base = model_dir / f"models--{HF_MODEL_ID.replace('/', '--')}"
    if not cache_base.exists():
        print(f"ERROR: Model cache not found at {cache_base}")
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
    
    snapshot_dir = snapshot_dirs[0]
    print(f"  Found model snapshot: {snapshot_dir.name}")
    
    # Create zip with all model files
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all files from snapshot directory
        model_files = []
        for file_path in snapshot_dir.rglob('*'):
            if file_path.is_file():
                # Skip very large files that aren't needed (like .safetensors.index.json if we have the main files)
                if file_path.suffix == '.index.json':
                    continue
                rel_path = file_path.relative_to(snapshot_dir)
                zipf.write(file_path, f"epsilon-20b/{rel_path}")
                model_files.append(rel_path)
                if len(model_files) <= 10:  # Print first 10 files
                    print(f"  Added: {rel_path}")
        
        if len(model_files) > 10:
            print(f"  ... and {len(model_files) - 10} more files")
        print(f"  Total files: {len(model_files)}")
    
    file_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Created zip archive: {zip_path.name} ({file_size_mb:.2f} MB)")
    
    # Upload to Supabase
    print(f"\n[2/3] Uploading to Supabase Storage...")
    storage_path = f"pretrained_models/epsilon-20b-production-{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    
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
                
                chunk_path = f"pretrained_models/epsilon-20b-production-{datetime.now().strftime('%Y%m%d_%H%M%S')}-chunk-{chunk_index}.zip"
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
        metadata_path = f"pretrained_models/epsilon-20b-production-{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip.metadata.json"
        chunk_metadata = {
            "model_name": MODEL_NAME,
            "company": COMPANY_NAME,
            "chunks": chunks,
            "total_size": file_size_mb,
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
    
    model_id = f"epsilon-20b-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    deployment_data = {
        "model_id": model_id,
        "storage_path": storage_path,
        "version": "1.0.0",
        "status": "approved",  # Auto-approve production model
        "description": f"{MODEL_NAME} - Production model uploaded by script",
        "stats": {
            "parameters": sum(p.numel() for p in model.parameters()),
            "model_name": MODEL_NAME,
            "company": COMPANY_NAME
        },
        "deployed_at": datetime.now().isoformat(),
        "deployed_by": "download_and_upload_script"
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
    
    return True

def main():
    # Download model
    model_dir, tokenizer, model = download_model()
    if model_dir is None:
        return False
    
    # Upload to Supabase
    success = upload_to_supabase(model_dir, tokenizer, model)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

