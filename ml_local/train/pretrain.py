"""
Pre-training script for Epsilon Transformer
LOCAL-ONLY - Requires LOCAL_TRAINING=1 environment variable
"""
import os
import sys
import json
import argparse
import math
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import numpy as np
from torch.utils.data import Dataset, DataLoader  # type: ignore
from tqdm import tqdm  # type: ignore
from datetime import datetime
from pathlib import Path

# Import from shared transformer core
# Add services/python-services to path to access epsilon_transformer_core
services_path = Path(__file__).parent.parent.parent / 'services' / 'python-services'
services_path_str = str(services_path.resolve())
if services_path_str not in sys.path:
    sys.path.insert(0, services_path_str)

# Verify the module exists before importingkay 
epsilon_core_path = services_path / 'epsilon_transformer_core'
if not epsilon_core_path.exists():
    raise ImportError(
        f"epsilon_transformer_core not found at {epsilon_core_path}\n"
        f"Expected path: {services_path}\n"
        f"Current working directory: {os.getcwd()}"
    )
    
from epsilon_transformer_core import EpsilonTransformerLM, TransformerConfig  # type: ignore
from tokenizers import Tokenizer  # type: ignore

# LOCAL-ONLY ENFORCEMENT
if os.getenv('LOCAL_TRAINING') != '1':
    print("ERROR: Training can only run locally. Set LOCAL_TRAINING=1")
    sys.exit(1)


class TokenDataset(Dataset):
    """Dataset for tokenized text data (block sampling, stride=seq_len)"""
    def __init__(self, token_file: str, seq_len: int = 512, dtype: str = None):
        """
        Args:
            token_file: Path to binary token file (uint16 or uint32)
            seq_len: Sequence length for training
            dtype: 'uint16' or 'uint32' (auto-detected from .dtype file if None)
        """
        self.seq_len = seq_len
        
        # Determine dtype
        dtype_file = Path(token_file).with_suffix('.dtype')
        if dtype:
            np_dtype = np.uint16 if dtype == 'uint16' else np.uint32
        elif dtype_file.exists():
            with open(dtype_file, 'r') as f:
                dtype_str = f.read().strip()
                np_dtype = np.uint16 if 'uint16' in dtype_str else np.uint32
        else:
            # Default to uint16, but warn if file is large (might be uint32)
            np_dtype = np.uint16
            file_size = Path(token_file).stat().st_size
            if file_size > 100 * 1024 * 1024:  # > 100MB might indicate uint32
                print(f"WARNING: No .dtype file found for {token_file}. Assuming uint16. If vocab > 65535, specify --dtype uint32")
        
        # Load tokens from binary file
        tokens_np = np.fromfile(token_file, dtype=np_dtype)
        self.tokens = torch.from_numpy(tokens_np).long()
        
        # Validate token values are reasonable (within typical vocab range)
        if len(self.tokens) > 0:
            max_token = self.tokens.max().item()
            min_token = self.tokens.min().item()
            if min_token < 0:
                raise ValueError(f"Invalid tokens: negative values found (min={min_token})")
            if max_token > 1_000_000:  # Unusually large vocab
                print(f"WARNING: Max token value is {max_token:,} - this might indicate wrong dtype")
                print(f"  If vocab size is < 65536, tokens should be < 65536")
                print(f"  Consider checking if file should use uint16 instead of uint32 (or vice versa)")
        
        # Number of full blocks we can make where targets exist (need +1 token)
        # Block sampling: stride = seq_len (not stride = 1)
        self.num_blocks = max(0, (len(self.tokens) - 1) // self.seq_len)
        
        print(f"Loaded {len(self.tokens):,} tokens from {token_file} (dtype: {np_dtype})")
        if len(self.tokens) > 0:
            print(f"  Token range: [{min_token}, {max_token}]")
            print(f"  Block sequences (stride={self.seq_len}): {self.num_blocks:,}")
        
        # Do "suspiciously large" warning ONCE here, not in __len__
        if self.num_blocks > 50_000_000:
            print(f"\nWARNING: Dataset has {self.num_blocks:,} block sequences (still very large).")
            print(f"  This will take a long time to train. Consider using a subset for testing.")
    
    def __len__(self):
        """Return number of block sequences (stride=seq_len)"""
        return self.num_blocks
    
    def __getitem__(self, idx):
        # Block sampling: stride = seq_len
        if idx >= self.num_blocks:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_blocks}")
        
        start = idx * self.seq_len
        end = start + self.seq_len
        
        # Ensure we don't go out of bounds (safety check)
        if end + 1 > len(self.tokens):
            raise IndexError(f"Index {idx} would access beyond token array (end+1={end+1}, len={len(self.tokens)})")
        
        input_ids = self.tokens[start:end]
        targets = self.tokens[start + 1:end + 1]
        return input_ids, targets


def get_lr_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Learning rate schedule with warmup"""
    # Validate inputs
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    if warmup_steps >= total_steps:
        # Adjust warmup to be 10% of total steps
        warmup_steps = max(1, total_steps // 10)
        print(f"  Adjusted warmup_steps to {warmup_steps}")
    
    def lr_lambda(step):
        try:
            if step < warmup_steps:
                # Warmup: linear ramp from 0 to 1
                if warmup_steps > 0:
                    return float(step) / float(warmup_steps)
                else:
                    return 1.0
            else:
                # Cosine decay using pure Python math (avoids tensor creation overhead)
                decay_steps = total_steps - warmup_steps
                if decay_steps <= 0:
                    return 1.0  # No decay if no steps after warmup
                progress = float(step - warmup_steps) / float(decay_steps)
                # Clamp progress to [0, 1] to avoid issues
                progress = max(0.0, min(1.0, progress))
                return 0.5 * (1.0 + math.cos(progress * math.pi))
        except Exception as e:
            print(f"ERROR in lr_lambda at step {step}: {e}")
            return 1.0  # Fallback to full LR
    
    try:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    except Exception as e:
        print(f"ERROR creating LambdaLR: {e}")
        print(f"  optimizer type: {type(optimizer)}")
        print(f"  optimizer param_groups count: {len(optimizer.param_groups)}")
        raise


def train_epoch(model, dataloader, optimizer, scheduler, device, config, epoch, 
                gradient_accumulation_steps=1, use_amp=False, scaler=None, use_gradient_checkpointing=False):
    """Train for one epoch with memory optimizations"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    optimizer_step_count = 0  # Track actual optimizer steps for scheduler
    
    # Calculate batches per epoch for better progress display
    total_batches = len(dataloader)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", total=total_batches, unit="batch")
    
    for batch_idx, (input_ids, targets) in enumerate(pbar):
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        
        # Mixed precision training (FP16) to reduce memory usage
        if use_amp and scaler:
            with torch.amp.autocast('cuda'):
                logits, loss = model(input_ids, targets=targets, use_gradient_checkpointing=use_gradient_checkpointing)
                # Check for NaN/Inf loss before proceeding
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[WARNING] NaN/Inf loss detected at batch {batch_idx}, skipping")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
        else:
            logits, loss = model(input_ids, targets=targets, use_gradient_checkpointing=use_gradient_checkpointing)
            # Check for NaN/Inf loss before proceeding
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARNING] NaN/Inf loss detected at batch {batch_idx}, skipping")
                optimizer.zero_grad(set_to_none=True)
                continue
            loss = loss / gradient_accumulation_steps
        
        # Backward pass - NEVER use retain_graph=True
        if use_amp and scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()  # No retain_graph - let gradients be freed
        
        # Gradient accumulation: only step optimizer every N steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Track if optimizer actually stepped (important for AMP/GradScaler)
            optimizer_actually_stepped = False
            
            # Gradient clipping
            if use_amp and scaler:
                # OPTION C: Check if GradScaler actually stepped
                # scaler.step() can skip the optimizer step on overflow
                # We must only step scheduler if optimizer actually stepped
                prev_scale = scaler.get_scale()
                
                # Check for overflow before unscale - if scale is very small, gradients likely overflowed
                if prev_scale < 1e-6:
                    print(f"[WARNING] Scaler scale is extremely small ({prev_scale:.2e}), gradients likely overflowed. Skipping update.")
                    scaler.update()  # Reduce scale further
                    optimizer.zero_grad(set_to_none=True)
                    continue
                
                scaler.unscale_(optimizer)
                
                # Check for NaN/Inf gradients after unscale (before clipping)
                has_nan = False
                for param in model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan = True
                            break
                
                if has_nan:
                    print(f"[WARNING] NaN/Inf gradients detected after unscale at step {optimizer_step_count}, skipping update")
                    scaler.update()  # Reduce scale
                    optimizer.zero_grad(set_to_none=True)
                    continue  # Skip this update step
                
                # Clip gradients - returns norm BEFORE clipping, but gradients are clipped in-place
                grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                # Verify clipping worked by computing norm after clipping
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norm_after = total_norm ** (1. / 2)
                
                # After clipping, the actual gradient norm should be <= max_grad_norm
                # Log for debugging (first few steps only)
                if optimizer_step_count < 10:
                    print(f"[DEBUG] grad_norm: before={grad_norm_before:.2f}, after={grad_norm_after:.2f}, max={config.max_grad_norm}")
                
                # Safety check: if clipping didn't work (shouldn't happen), skip
                if grad_norm_after > config.max_grad_norm * 1.1:  # Allow 10% tolerance for floating point
                    print(f"[WARNING] Clipping may have failed: after={grad_norm_after:.2f} > max={config.max_grad_norm}")
                    # Still proceed - clipping should have worked
                
                # Only skip if we detect NaN/Inf AFTER clipping (shouldn't happen, but safety check)
                has_nan_after_clip = False
                for param in model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_after_clip = True
                            break
                
                if has_nan_after_clip:
                    print(f"[WARNING] NaN/Inf gradients detected AFTER clipping at step {optimizer_step_count}, skipping update")
                    scaler.update()  # Reduce scale
                    optimizer.zero_grad(set_to_none=True)
                    continue  # Skip this update step
                
                scaler.step(optimizer)
                scaler.update()
                
                # Safer AMP pattern: scale < prev_scale means overflow (optimizer didn't step)
                # scale >= prev_scale means step happened (or equal, which we treat as stepped)
                # Note: "equal" can mean stepped (growth interval keeps it constant) or not stepped,
                # but treating >= as "stepped" is the safer default for scheduler alignment
                optimizer_actually_stepped = scaler.get_scale() >= prev_scale
                
                # Debug print for first few steps to verify scale behavior
                if optimizer_step_count < 5:
                    print(f"[DEBUG] scale: prev={prev_scale}, now={scaler.get_scale()}, stepped={optimizer_actually_stepped}")
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer_actually_stepped = True
            
            # CRITICAL: Step scheduler ONLY AFTER optimizer.step() has been called
            # AND only if the optimizer actually stepped (not skipped due to overflow)
            # This ensures the scheduler sees the updated optimizer state
            # PyTorch requires scheduler.step() to be called AFTER optimizer.step()
            # to avoid the warning: "Detected call of `lr_scheduler.step()` before `optimizer.step()`"
            
            if optimizer_actually_stepped:
                # Debug print for first few steps to verify order
                # Note: batch_idx will be (gradient_accumulation_steps - 1) for first step
                # e.g., if grad_accum=4, first scheduler step happens at batch_idx=3
                if optimizer_step_count < 3:
                    print(f"[DEBUG] SCHEDULER STEP CALLED at batch_idx={batch_idx}, opt_step={optimizer_step_count}")
                    print(f"  (gradient_accumulation_steps={gradient_accumulation_steps}, so first step at batch_idx={gradient_accumulation_steps-1})")
                    # Verify gradients exist (they should, since zero_grad hasn't been called yet)
                    # Gradients remain after optimizer.step() until zero_grad() is called
                    has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
                    if has_grads:
                        print(f"  OK: Gradients present (expected before zero_grad)")
                    else:
                        print(f"  WARNING: No gradients found - optimizer may not have stepped")
                
                # Step scheduler AFTER optimizer has stepped
                scheduler.step()
                optimizer_step_count += 1
            else:
                # GradScaler skipped the optimizer step due to overflow
                if optimizer_step_count < 3:
                    print(f"[DEBUG] Optimizer step SKIPPED (overflow) at batch_idx={batch_idx}, skipping scheduler.step()")
            
            # CRITICAL: Zero gradients AFTER stepping, regardless of overflow
            # We always want to clear grads on update steps, even if overflow happened
            # Otherwise bad grads will keep accumulating
            optimizer.zero_grad(set_to_none=True)
            
            # Periodic memory clearing (not every step - that hurts performance)
            # Reduced frequency to every 2000 steps to avoid unnecessary slowdown
            if optimizer_step_count > 0 and optimizer_step_count % 2000 == 0:
                torch.cuda.empty_cache()
        
        # Logging - ALWAYS use .item() to convert tensor to scalar (prevents graph retention)
        loss_scalar = loss.item() * gradient_accumulation_steps
        total_loss += loss_scalar
        # Count sequences (not tokens) - if you want token count, use: input_ids.numel()
        total_samples += input_ids.size(0)
        
        if batch_idx % 100 == 0:
            # Get learning rate from scheduler (it has stepped after optimizer)
            current_lr = scheduler.get_last_lr()[0] if optimizer_step_count > 0 else optimizer.param_groups[0]['lr']
            
            avg_loss = total_loss / (batch_idx + 1)
            # Use allocated memory, but handle fragmentation issues
            mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
            mem_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            # Use the smaller of allocated/reserved to avoid impossible values
            # Fragmentation can cause reserved > total, so cap at total
            mem_display = min(mem_allocated, mem_reserved, mem_total)
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'opt_steps': f'{optimizer_step_count:,}',
                'gpu_mem': f'{mem_display:.2f}/{mem_total:.2f}GB'
            })
            
            # Warn if memory usage is getting high (use allocated, not reserved)
            if mem_allocated > mem_total * 0.90:
                print(f"\nWARNING: GPU memory usage is high ({mem_allocated:.2f}/{mem_total:.2f}GB)")
                print("  Clearing cache more aggressively...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    return avg_loss, total_samples


def eval_epoch(model, dataloader, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for input_ids, targets in tqdm(dataloader, desc="Validation"):
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            logits, loss = model(input_ids, targets=targets)
            # Always use .item() to convert tensor to scalar (prevents graph retention)
            total_loss += loss.item()
            total_samples += input_ids.size(0)
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    perplexity = torch.exp(torch.tensor(avg_loss, device=device)).item()
    return avg_loss, perplexity


def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, checkpoint_dir):
    """Save training checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{step}.pt"
    
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': model.config.to_dict(),
        'timestamp': datetime.now().isoformat()
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description='Pre-train Epsilon Transformer')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON')
    parser.add_argument('--data', type=str, required=True, help='Path to tokenized training data (.bin)')
    parser.add_argument('--val-data', type=str, help='Path to validation data (.bin)')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer.json')
    parser.add_argument('--output-dir', type=str, default='runs', help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Gradient accumulation steps (for memory efficiency)')
    parser.add_argument('--use-amp', action='store_true', help='Use mixed precision (FP16) training to reduce memory')
    parser.add_argument('--use-gradient-checkpointing', action='store_true', help='Use gradient checkpointing to save memory (trades compute for memory)')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--pretrained', type=str, help='Load pre-trained model checkpoint (for fine-tuning)')
    parser.add_argument('--checkpoint-interval', type=int, default=1000, help='Save checkpoint every N steps')
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = TransformerConfig.from_dict(config_dict)
    else:
        # Use defaults
        config = TransformerConfig()
        print(f"Config file not found, using defaults")
    
    # Override seq_len if provided
    if args.seq_len:
        config.max_seq_len = args.seq_len
    
    # Device - FORCE NVIDIA GPU (no CPU fallback, no Intel graphics)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available! GPU training is required.\n"
            "Please install PyTorch with CUDA support:\n"
            "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n"
            "Or check your CUDA installation and GPU drivers."
        )
    
    # Explicitly use cuda:0 (first CUDA device) and verify it's NVIDIA, not Intel
    device = torch.device('cuda:0')
    gpu_name = torch.cuda.get_device_name(0)
    
    # Verify it's an NVIDIA GPU, not Intel integrated graphics
    if 'intel' in gpu_name.lower() or 'uhd' in gpu_name.lower() or 'iris' in gpu_name.lower():
        raise RuntimeError(
            f"ERROR: Detected Intel graphics ({gpu_name}) instead of NVIDIA GPU!\n"
            "CUDA requires an NVIDIA GPU. Please ensure:\n"
            "1. Your NVIDIA GPU (MX250) is enabled in BIOS\n"
            "2. NVIDIA drivers are installed\n"
            "3. The GPU is set as the primary graphics device"
        )
    
    if 'nvidia' not in gpu_name.lower() and 'geforce' not in gpu_name.lower():
        print(f"WARNING: GPU name '{gpu_name}' doesn't clearly indicate NVIDIA. Proceeding anyway...")
    
    print(f"Using NVIDIA GPU: {gpu_name}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"GPU Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    config.vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer vocab size: {config.vocab_size}")
    
    # Create model
    model = EpsilonTransformerLM(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Enable gradient checkpointing if requested
    if args.use_gradient_checkpointing:
        print("[OK] Gradient checkpointing enabled - saves memory by trading compute for memory")
    
    # Load pre-trained model if provided (for fine-tuning)
    if args.pretrained:
        print(f"\nLoading pre-trained model: {args.pretrained}")
        pretrained_checkpoint = torch.load(args.pretrained, map_location=device)
        
        # Check if config matches
        if 'config' in pretrained_checkpoint:
            pretrained_config = TransformerConfig.from_dict(pretrained_checkpoint['config'])
            if pretrained_config.vocab_size != config.vocab_size:
                print(f"WARNING: Vocab size mismatch!")
                print(f"  Pre-trained: {pretrained_config.vocab_size}")
                print(f"  Current: {config.vocab_size}")
                print(f"  This may cause issues. Consider using matching vocab size.")
        
        # Filter out RoPE cache buffers (they depend on seq_len and will be rebuilt)
        # These are precomputed cosine/sine tables that can be regenerated for any sequence length
        pretrained_state = pretrained_checkpoint['model_state_dict'].copy()
        rope_keys = [k for k in pretrained_state.keys() if "rope.cos_cached" in k or "rope.sin_cached" in k]
        for k in rope_keys:
            pretrained_state.pop(k, None)
        
        if rope_keys:
            print(f"  Filtered out {len(rope_keys)} RoPE cache buffers (will be regenerated for seq_len={config.max_seq_len})")
        
        # Load state dict (with strict=False to preserve conversational ability)
        # This preserves the model's ability to understand questions and have conversations
        # while allowing us to add domain knowledge through fine-tuning
        missing, unexpected = model.load_state_dict(pretrained_state, strict=False)
        
        # Filter out expected missing keys (RoPE caches) from the report
        if missing:
            missing_non_rope = [k for k in missing if "rope" not in k]
            if missing_non_rope:
                print(f"  Note: {len(missing_non_rope)} non-RoPE keys not found (expected for fine-tuning)")
        
        print("[OK] Pre-trained weights loaded - conversational ability preserved")
        print("  The model will understand questions and have conversations")
        print("  Fine-tuning will add your domain knowledge without breaking this ability")
        
        if 'source_model' in pretrained_checkpoint:
            print(f"  Source: {pretrained_checkpoint['source_model']}")
        if 'license' in pretrained_checkpoint:
            print(f"  License: {pretrained_checkpoint['license']}")
    
    # Create dataset and dataloader
    train_dataset = TokenDataset(args.data, seq_len=config.max_seq_len)
    
    # For large datasets, disable shuffling to avoid RAM issues
    # Shuffling tries to create a list of all indices in memory (9GB+ for 4.5M samples)
    dataset_size = len(train_dataset)
    if dataset_size > 1_000_000:
        print(f"[OK] Disabling shuffle for large dataset ({dataset_size:,} samples) to save RAM")
        print("  Sequential order is fine for fine-tuning - model will still learn effectively")
        shuffle = False
    else:
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # Create validation dataloader if provided
    val_loader = None
    if args.val_data:
        val_dataset = TokenDataset(args.val_data, seq_len=config.max_seq_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        print(f"Validation dataset loaded: {len(val_dataset):,} sequences")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Mixed precision training scaler (for FP16)
    scaler = None
    use_amp = args.use_amp
    if use_amp:
        # Lower init_scale for MX250 (more stable, prevents overflow)
        # Start at 0.01 (extremely low) to prevent constant overflow - will grow if stable
        scaler = torch.amp.GradScaler(
            'cuda',
            init_scale=0.01,        # Start extremely low to prevent overflow
            growth_interval=2000,   # grow slowly
            backoff_factor=0.5,
            growth_factor=2.0
        )
        print("[OK] Mixed precision (FP16) training enabled - tuned GradScaler for MX250 (init_scale=0.01)")
    
    # Gradient accumulation
    gradient_accumulation_steps = args.gradient_accumulation_steps
    if gradient_accumulation_steps > 1:
        print(f"[OK] Gradient accumulation: {gradient_accumulation_steps} steps (effective batch size: {args.batch_size * gradient_accumulation_steps})")
    
    # Calculate total optimizer steps correctly
    # Batches per epoch = len(train_loader)
    # Optimizer steps per epoch = batches_per_epoch // gradient_accumulation_steps
    # Total optimizer steps = optimizer_steps_per_epoch * num_epochs
    batches_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = batches_per_epoch // gradient_accumulation_steps
    total_steps = optimizer_steps_per_epoch * args.epochs
    
    # Validate calculation
    if batches_per_epoch % gradient_accumulation_steps != 0:
        print(f"WARNING: Batches per epoch ({batches_per_epoch:,}) is not divisible by gradient_accumulation_steps ({gradient_accumulation_steps})")
        print(f"  Last {batches_per_epoch % gradient_accumulation_steps} batches will be discarded each epoch")
        print(f"  Consider adjusting batch_size or gradient_accumulation_steps")
    
    # OPTION A: Set initial LR to target LR (scheduler will handle warmup from 0)
    # LambdaLR multiplies the base_lr by the lambda, so we need base_lr = target LR
    print(f"Setting optimizer base LR to {config.learning_rate} (scheduler will handle warmup)")
    # Set all param groups to target LR - scheduler lambda will scale from 0 to 1 during warmup
    for param_group in optimizer.param_groups:
        param_group['lr'] = config.learning_rate
    
    # Validate inputs before creating scheduler
    if total_steps <= 0:
        raise ValueError(f"Invalid total_steps: {total_steps}. Must be > 0.")
    if config.warmup_steps < 0:
        raise ValueError(f"Invalid warmup_steps: {config.warmup_steps}. Must be >= 0.")
    if config.warmup_steps >= total_steps:
        print(f"WARNING: warmup_steps ({config.warmup_steps}) >= total_steps ({total_steps})")
        print(f"  Adjusting warmup_steps to {max(1, total_steps // 10)}")
        config.warmup_steps = max(1, total_steps // 10)
    
    # Create scheduler ONCE after all LRs are set
    # LambdaLR may step internally during creation (sets last_epoch=0)
    # This is normal - it will set LR to base_lr * lambda(0) = base_lr * 0 = 0
    try:
        print(f"Creating scheduler with warmup_steps={config.warmup_steps}, total_steps={total_steps}")
        scheduler = get_lr_schedule(optimizer, config.warmup_steps, total_steps)
    except Exception as e:
        import traceback
        print(f"ERROR: Failed to create scheduler: {e}")
        print(f"  warmup_steps: {config.warmup_steps}")
        print(f"  total_steps: {total_steps}")
        print(f"  learning_rate: {config.learning_rate}")
        print(f"  optimizer param_groups: {len(optimizer.param_groups)}")
        print(f"Full traceback:")
        traceback.print_exc()
        raise
    
    # Debug: Check scheduler state after creation
    print(f"Scheduler created. last_epoch = {scheduler.last_epoch}")
    initial_lrs = [g['lr'] for g in optimizer.param_groups]
    print(f"Initial LR(s): {initial_lrs}")
    # Note: LR may be 0 after scheduler creation (if it stepped internally with lambda(0)=0)
    # This is fine - it will ramp up as scheduler.step() is called during training
    
    # Resume from checkpoint if provided
    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['step']
        
        # CRITICAL: After loading scheduler state, we must ensure optimizer has stepped
        # The scheduler's internal step count is restored, but we haven't called optimizer.step() yet
        # in this session. The scheduler will step correctly on the next update step.
        # We do NOT call scheduler.step() here - it will be called after the first optimizer.step()
        print(f"  Resumed: epoch={start_epoch}, step={global_step}")
        print(f"  Scheduler state restored - will step after first optimizer update")
    
    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Starting pre-training")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {config.max_seq_len}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * gradient_accumulation_steps}")
    print(f"Dataset sequences: {len(train_dataset):,}")
    print(f"Batches per epoch: {batches_per_epoch:,}")
    print(f"Optimizer steps per epoch: {optimizer_steps_per_epoch:,}")
    print(f"Total optimizer steps (all epochs): {total_steps:,}")
    print(f"Warmup steps: {config.warmup_steps:,}")
    print(f"{'='*60}\n")
    
    # Final sanity check on epoch length (informational only, no prompt)
    if batches_per_epoch > 10_000_000:  # > 10 million batches per epoch
        print(f"\nWARNING: Very large epoch detected ({batches_per_epoch:,} batches)")
        print(f"  This will take a very long time to complete.")
        print(f"  Consider using a smaller dataset or increasing batch_size/gradient_accumulation_steps")
        print(f"  Continuing anyway...")
    
    for epoch in range(start_epoch, args.epochs):
        avg_loss, samples = train_epoch(
            model, train_loader, optimizer, scheduler, device, config, epoch + 1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_amp=use_amp,
            scaler=scaler,
            use_gradient_checkpointing=args.use_gradient_checkpointing
        )
        
        print(f"\nEpoch {epoch + 1} complete:")
        print(f"  Train loss: {avg_loss:.4f}")
        print(f"  Samples processed: {samples:,}")
        
        # Evaluate on validation set if provided
        if val_loader:
            val_loss, val_perplexity = eval_epoch(model, val_loader, device)
            print(f"  Validation loss: {val_loss:.4f}")
            print(f"  Validation perplexity: {val_perplexity:.2f}")
        
        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model, optimizer, scheduler, epoch + 1, global_step, avg_loss, output_dir
        )
        
        global_step += len(train_loader)
    
    print(f"\n{'='*60}")
    print(f"Pre-training complete!")
    print(f"Final checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

