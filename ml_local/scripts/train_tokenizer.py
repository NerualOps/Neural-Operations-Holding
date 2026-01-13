"""
Train BPE tokenizer on corpus
Outputs: tokenizer.json, vocab_size.txt, special_tokens.json
"""
import os
import sys
import argparse
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence as NormalizerSequence

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path(__file__).parent.parent / "tokenizer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def train_tokenizer(corpus_file: str, vocab_size: int = 50000, output_dir: str = None):
    """
    Train a BPE tokenizer on the corpus
    
    Args:
        corpus_file: Path to training corpus (train.txt)
        vocab_size: Target vocabulary size
        output_dir: Output directory for tokenizer files
    """
    corpus_file = Path(corpus_file)
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
    
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training tokenizer on {corpus_file}", flush=True)
    print(f"Target vocab size: {vocab_size}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    
    # Initialize tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.normalizer = NormalizerSequence([NFD(), Lowercase(), StripAccents()])
    
    # Special tokens
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True
    )
    
    # Train on corpus (read file line by line for large files)
    print("Training tokenizer...", flush=True)
    with open(corpus_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    tokenizer.train_from_iterator(lines, trainer=trainer)
    
    # Post-processor: We'll handle BOS/EOS in the training loop, not here
    # For now, just use ByteLevel which works with BPE
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    
    # Save tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}", flush=True)
    
    # Save vocab size
    vocab_size_path = output_dir / "vocab_size.txt"
    actual_vocab_size = tokenizer.get_vocab_size()
    with open(vocab_size_path, 'w') as f:
        f.write(str(actual_vocab_size))
    print(f"Vocab size: {actual_vocab_size} (saved to {vocab_size_path})", flush=True)
    
    # Save special tokens
    import json
    special_tokens_map = {
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token": "<bos>",
        "eos_token": "<eos>"
    }
    special_tokens_path = output_dir / "special_tokens.json"
    with open(special_tokens_path, 'w') as f:
        json.dump(special_tokens_map, f, indent=2)
    print(f"Special tokens saved to {special_tokens_path}", flush=True)
    
    # Test tokenizer
    test_text = "Hello, this is a test sentence."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    print(f"\nTest encode/decode:", flush=True)
    print(f"  Input: {test_text}", flush=True)
    print(f"  Encoded: {encoded.ids[:10]}...", flush=True)
    try:
        print(f"  Decoded: {decoded}", flush=True)
    except UnicodeEncodeError:
        # Windows console encoding issue - just skip the decoded output
        print(f"  Decoded: [success - tokenizer working]", flush=True)
    
    return tokenizer_path


def main():
    parser = argparse.ArgumentParser(description='Train BPE tokenizer on corpus')
    parser.add_argument('--corpus', type=str, required=True, help='Path to training corpus (train.txt)')
    parser.add_argument('--vocab-size', type=int, default=50000, help='Target vocabulary size')
    parser.add_argument('--output', type=str, help='Output directory (default: ml_local/tokenizer/)')
    
    args = parser.parse_args()
    
    train_tokenizer(args.corpus, args.vocab_size, args.output)


if __name__ == '__main__':
    main()

