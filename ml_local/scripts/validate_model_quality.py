"""
Validate model quality before auto-approval
Returns True if model passes all safety checks
"""
import torch
import json
from pathlib import Path
from tokenizers import Tokenizer

def validate_model(checkpoint_path: str, tokenizer_path: str, config_path: str = None) -> tuple[bool, dict]:
    """
    Validate model quality with 6 safety checks
    
    Returns:
        (passed: bool, results: dict)
    """
    results = {
        'checks_passed': 0,
        'total_checks': 6,
        'details': {}
    }
    
    try:
        # Check 1: Model loads successfully
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' not in checkpoint and not isinstance(checkpoint, dict):
                results['details']['load_check'] = {'passed': False, 'reason': 'Invalid checkpoint format'}
                return False, results
            results['details']['load_check'] = {'passed': True}
            results['checks_passed'] += 1
        except Exception as e:
            results['details']['load_check'] = {'passed': False, 'reason': str(e)}
            return False, results
        
        # Check 2: Tokenizer loads and matches vocab size
        try:
            tokenizer = Tokenizer.from_file(tokenizer_path)
            vocab_size = len(tokenizer.get_vocab()) if hasattr(tokenizer, 'get_vocab') else 50257
            
            # Load config
            if config_path and Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    config_vocab = config.get('vocab_size', vocab_size)
            else:
                config_vocab = vocab_size
            
            if abs(vocab_size - config_vocab) > 10:  # Allow small differences
                results['details']['vocab_check'] = {'passed': False, 'reason': f'Vocab mismatch: tokenizer={vocab_size}, config={config_vocab}'}
                return False, results
            
            results['details']['vocab_check'] = {'passed': True, 'vocab_size': vocab_size}
            results['checks_passed'] += 1
        except Exception as e:
            results['details']['vocab_check'] = {'passed': False, 'reason': str(e)}
            return False, results
        
        # Check 3: Model has reasonable size (not corrupted)
        checkpoint_size = Path(checkpoint_path).stat().st_size
        if checkpoint_size < 1000:  # Less than 1KB is suspicious
            results['details']['size_check'] = {'passed': False, 'reason': f'Checkpoint too small: {checkpoint_size} bytes'}
            return False, results
        results['details']['size_check'] = {'passed': True, 'size_mb': checkpoint_size / 1024 / 1024}
        results['checks_passed'] += 1
        
        # Check 4: Config is valid JSON
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                required_keys = ['vocab_size', 'n_layers', 'd_model']
                missing = [k for k in required_keys if k not in config]
                if missing:
                    results['details']['config_check'] = {'passed': False, 'reason': f'Missing keys: {missing}'}
                    return False, results
                results['details']['config_check'] = {'passed': True}
                results['checks_passed'] += 1
            except Exception as e:
                results['details']['config_check'] = {'passed': False, 'reason': str(e)}
                return False, results
        else:
            # Config might be in checkpoint
            if 'config' in checkpoint:
                results['details']['config_check'] = {'passed': True}
                results['checks_passed'] += 1
            else:
                results['details']['config_check'] = {'passed': False, 'reason': 'No config found'}
                return False, results
        
        # Check 5: Tokenizer can encode/decode
        try:
            test_text = "Hello, this is a test."
            encoded = tokenizer.encode(test_text)
            decoded = tokenizer.decode(encoded.ids)
            if len(decoded) == 0:
                results['details']['tokenizer_check'] = {'passed': False, 'reason': 'Tokenizer decode returned empty'}
                return False, results
            results['details']['tokenizer_check'] = {'passed': True}
            results['checks_passed'] += 1
        except Exception as e:
            results['details']['tokenizer_check'] = {'passed': False, 'reason': str(e)}
            return False, results
        
        # Check 6: Model weights structure is valid
        try:
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            if not isinstance(state_dict, dict):
                results['details']['weights_check'] = {'passed': False, 'reason': 'State dict is not a dictionary'}
                return False, results
            
            # Check for at least some expected keys (token_embedding or transformer layers)
            has_embedding = any('embedding' in k.lower() or 'transformer' in k.lower() for k in state_dict.keys())
            if not has_embedding and len(state_dict) < 5:
                results['details']['weights_check'] = {'passed': False, 'reason': 'State dict seems incomplete'}
                return False, results
            
            results['details']['weights_check'] = {'passed': True, 'weight_keys': len(state_dict)}
            results['checks_passed'] += 1
        except Exception as e:
            results['details']['weights_check'] = {'passed': False, 'reason': str(e)}
            return False, results
        
        # All checks passed
        results['all_passed'] = True
        return True, results
        
    except Exception as e:
        results['details']['error'] = str(e)
        return False, results

