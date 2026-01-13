"""
Quick setup verification script
Run this to check if everything is installed correctly
"""
import sys
import os
from pathlib import Path

print("=" * 60)
print("Training Setup Verification")
print("=" * 60)

# Check Python version
print(f"\n[OK] Python version: {sys.version}")
if sys.version_info < (3, 8):
    print("  [WARN] Python 3.8+ required")
    sys.exit(1)

# Check required packages
required_packages = {
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'tqdm': 'tqdm',
    'yaml': 'PyYAML',
    'tokenizers': 'tokenizers',
    'datasets': 'datasets',
    'supabase': 'supabase',
    'dotenv': 'python-dotenv'
}

print("\nChecking required packages:")
all_ok = True
for module, name in required_packages.items():
    try:
        if module == 'yaml':
            import yaml
        elif module == 'dotenv':
            from dotenv import load_dotenv
        else:
            __import__(module)
        print(f"  [OK] {name}")
    except ImportError:
        print(f"  [FAIL] {name} - NOT INSTALLED")
        all_ok = False

# Check environment variables
print("\nChecking environment variables:")
env_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY', 'LOCAL_TRAINING']
env_ok = True

# Try loading from .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"  ✓ Loaded .env from {env_path}")
    else:
        print(f"  ⚠ .env file not found at {env_path}")
except:
    pass

for var in env_vars:
    value = os.getenv(var)
    if value:
        # Mask sensitive values
        if 'KEY' in var or 'SECRET' in var:
            display_value = value[:10] + '...' if len(value) > 10 else '***'
        else:
            display_value = value
        print(f"  [OK] {var} = {display_value}")
    else:
        print(f"  [FAIL] {var} - NOT SET")
        env_ok = False

# Check directory structure
print("\nChecking directory structure:")
dirs_to_check = [
    ('ml_local', Path(__file__).parent),
    ('data/processed', Path(__file__).parent / 'data' / 'processed'),
    ('tokenizer', Path(__file__).parent / 'tokenizer'),
    ('runs', Path(__file__).parent / 'runs'),
    ('transformer core', Path(__file__).parent.parent / 'services' / 'python-services' / 'epsilon_transformer_core')
]

for name, path in dirs_to_check:
    if path.exists():
        print(f"  [OK] {name}/")
    else:
        print(f"  [WARN] {name}/ - will be created when needed")

# Check transformer core
print("\nChecking transformer core:")
transformer_path = Path(__file__).parent.parent / 'services' / 'python-services' / 'epsilon_transformer_core'
if transformer_path.exists():
    try:
        sys.path.insert(0, str(transformer_path.parent))
        from epsilon_transformer_core import EpsilonTransformerLM, TransformerConfig
        print("  [OK] epsilon_transformer_core imports successfully")
    except ImportError as e:
        print(f"  [FAIL] Failed to import transformer core: {e}")
        all_ok = False
else:
    print(f"  [FAIL] Transformer core not found at {transformer_path}")
    all_ok = False

# Final summary
print("\n" + "=" * 60)
if all_ok and env_ok:
    print("[SUCCESS] All checks passed! You're ready to train.")
    print("\nNext steps:")
    print("  1. python scripts/pull_corpus_from_supabase.py")
    print("  2. python scripts/train_tokenizer.py --corpus data/processed/train.txt")
    print("  3. python scripts/build_token_bins.py --text data/processed/train.txt --tokenizer tokenizer/tokenizer.json --output data/processed/train.bin")
    print("  4. python train/pretrain.py --config config.json --data data/processed/train.bin --tokenizer tokenizer/tokenizer.json")
else:
    print("[FAIL] Some checks failed. Please fix the issues above.")
    if not all_ok:
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
    if not env_ok:
        print("\nTo set environment variables:")
        print("  Create .env file in project root with SUPABASE_URL and SUPABASE_SERVICE_KEY")
print("=" * 60)

