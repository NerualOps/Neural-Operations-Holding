"""
Convert text corpus to binary token files (train.bin, val.bin)
Supports uint16 (vocab < 65536) or uint32 (vocab >= 65536)
"""
import os
import sys
import argparse
import numpy as np
import logging
import gc
from pathlib import Path
from datetime import datetime
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"


def determine_dtype(vocab_size: int) -> np.dtype:
    """Determine appropriate dtype based on vocab size"""
    if vocab_size < 65536:
        return np.uint16
    else:
        return np.uint32


def encode_corpus_to_tokens(text_file: str, tokenizer_path: str, output_file: str, dtype: str = None):
    """
    Encode text corpus to binary token file
    
    Args:
        text_file: Path to text corpus file
        tokenizer_path: Path to tokenizer.json
        output_file: Output binary token file path
        dtype: 'uint16' or 'uint32' (auto-detected if None)
    """
    text_file = Path(text_file)
    tokenizer_path = Path(tokenizer_path)
    output_file = Path(output_file)
    
    if not text_file.exists():
        raise FileNotFoundError(f"Text file not found: {text_file}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    
    # Determine dtype
    if dtype:
        if dtype == 'uint16':
            np_dtype = np.uint16
        elif dtype == 'uint32':
            np_dtype = np.uint32
        else:
            raise ValueError(f"Invalid dtype: {dtype}. Must be 'uint16' or 'uint32'")
    else:
        np_dtype = determine_dtype(vocab_size)
    
    # Setup logging
    log_file = output_file.with_suffix('.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting token encoding: {text_file} -> {output_file}")
    logger.info(f"Vocab size: {vocab_size}, Using dtype: {np_dtype}")
    logger.info(f"Log file: {log_file}")
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Process and write in extremely small chunks to avoid memory issues
    total_tokens = 0
    batch_tokens = []
    batch_size = 2000  # Process only 2k tokens at a time before writing (very small!)
    last_log_time = datetime.now()
    log_interval_seconds = 30  # Log every 30 seconds
    
    # Open output file in binary append mode (we'll write incrementally)
    # First, delete if exists to start fresh
    if output_file.exists():
        logger.info(f"Removing existing output file: {output_file}")
        output_file.unlink()
    
    start_time = datetime.now()
    logger.info(f"Processing started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        with open(text_file, 'r', encoding='utf-8') as f, open(output_file, 'ab') as out_f:
            # Process line-by-line to minimize memory usage
            line_count = 0
            doc_count = 0
            buffer_line = ""
            
            for line in f:
                line_count += 1
                buffer_line += line
                
                # Process when we hit a document separator or buffer gets too large
                if '---DOCUMENT_SEPARATOR---' in buffer_line or len(buffer_line) > 64 * 1024:  # 64KB max buffer
                    # Split on separator if present
                    if '---DOCUMENT_SEPARATOR---' in buffer_line:
                        parts = buffer_line.split('---DOCUMENT_SEPARATOR---')
                        # Process all but the last part (which may be incomplete)
                        for doc in parts[:-1]:
                            if doc.strip():
                                try:
                                    encoded = tokenizer.encode(doc.strip())
                                    batch_tokens.extend(encoded.ids)
                                    total_tokens += len(encoded.ids)
                                    doc_count += 1
                                    del encoded
                                    gc.collect()  # Clean up after each document
                                except Exception as e:
                                    logger.error(f"Error encoding document {doc_count}: {str(e)}")
                                    raise
                        
                        # Keep the last part (might be incomplete)
                        buffer_line = parts[-1] if parts else ""
                    else:
                        # Buffer too large, process first 32KB
                        try:
                            encoded = tokenizer.encode(buffer_line[:32*1024])
                            batch_tokens.extend(encoded.ids)
                            total_tokens += len(encoded.ids)
                            buffer_line = buffer_line[32*1024:]
                            del encoded
                        except Exception as e:
                            logger.error(f"Error encoding large buffer: {str(e)}")
                            raise
                    
                    # Write batch when it reaches threshold (very small batches)
                    if len(batch_tokens) >= batch_size:
                        try:
                            tokens_array = np.array(batch_tokens, dtype=np_dtype)
                            tokens_array.tofile(out_f)
                            del tokens_array
                            batch_tokens = []
                            gc.collect()  # Force garbage collection after each write
                        except Exception as e:
                            logger.error(f"Error writing batch at {total_tokens:,} tokens: {str(e)}")
                            raise
                        
                        # Log progress periodically
                        current_time = datetime.now()
                        elapsed = (current_time - last_log_time).total_seconds()
                        if elapsed >= log_interval_seconds:
                            elapsed_total = (current_time - start_time).total_seconds()
                            rate = total_tokens / elapsed_total if elapsed_total > 0 else 0
                            logger.info(f"Progress: {total_tokens:,} tokens processed | "
                                      f"Rate: {rate:,.0f} tokens/sec | "
                                      f"Elapsed: {elapsed_total/60:.1f} min | "
                                      f"Lines read: {line_count:,} | "
                                      f"Docs processed: {doc_count:,}")
                            last_log_time = current_time
                        else:
                            print(f"  Processed {total_tokens:,} tokens...", end='\r')
            
            # Process remaining buffer
            if buffer_line.strip():
                try:
                    logger.info(f"Processing final buffer ({len(buffer_line):,} chars)")
                    encoded = tokenizer.encode(buffer_line.strip())
                    batch_tokens.extend(encoded.ids)
                    total_tokens += len(encoded.ids)
                    doc_count += 1
                    del encoded
                except Exception as e:
                    logger.error(f"Error encoding final buffer: {str(e)}")
                    raise
            
            # Write remaining tokens
            if batch_tokens:
                try:
                    logger.info(f"Writing final batch of {len(batch_tokens):,} tokens")
                    tokens_array = np.array(batch_tokens, dtype=np_dtype)
                    tokens_array.tofile(out_f)
                    del tokens_array
                    batch_tokens = []
                    gc.collect()  # Final cleanup
                except Exception as e:
                    logger.error(f"Error writing final batch: {str(e)}")
                    raise
                    
    except Exception as e:
        logger.error(f"FATAL ERROR during processing: {str(e)}", exc_info=True)
        logger.error(f"Failed at {total_tokens:,} tokens processed")
        logger.error(f"Output file may be incomplete: {output_file}")
        if output_file.exists():
            file_size = output_file.stat().st_size / (1024 * 1024)
            logger.error(f"Incomplete file size: {file_size:.2f} MB")
        raise
    
    end_time = datetime.now()
    elapsed_total = (end_time - start_time).total_seconds()
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    rate = total_tokens / elapsed_total if elapsed_total > 0 else 0
    
    logger.info(f"Processing completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Output file size: {file_size_mb:.2f} MB")
    logger.info(f"Processing time: {elapsed_total/60:.2f} minutes ({elapsed_total:.1f} seconds)")
    logger.info(f"Average rate: {rate:,.0f} tokens/second")
    print(f"\nSaved {total_tokens:,} tokens to {output_file} ({file_size_mb:.2f} MB)")
    
    # Save dtype info for training script
    dtype_info_file = output_file.with_suffix('.dtype')
    with open(dtype_info_file, 'w') as f:
        f.write(str(np_dtype))
    logger.info(f"Dtype info saved to {dtype_info_file}")
    print(f"Dtype info saved to {dtype_info_file}")
    
    logger.info("Token encoding completed successfully!")
    
    return output_file, np_dtype


def main():
    parser = argparse.ArgumentParser(description='Convert text corpus to binary token files')
    parser.add_argument('--text', type=str, required=True, help='Path to text corpus file (train.txt or val.txt)')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer.json')
    parser.add_argument('--output', type=str, required=True, help='Output binary token file (train.bin or val.bin)')
    parser.add_argument('--dtype', type=str, choices=['uint16', 'uint32'], help='Force dtype (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    encode_corpus_to_tokens(args.text, args.tokenizer, args.output, args.dtype)


if __name__ == '__main__':
    main()

