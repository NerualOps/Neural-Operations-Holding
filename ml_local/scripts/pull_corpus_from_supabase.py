"""
Pull training corpus from Supabase
Pulls from both knowledge_documents and doc_chunks tables
Splits train/val at document level to avoid leakage
"""
import os
import sys
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Tuple
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = OUTPUT_DIR / "train.txt"
VAL_FILE = OUTPUT_DIR / "val.txt"
MANIFEST_FILE = OUTPUT_DIR / "corpus_manifest.json"


def get_document_hash(doc_id: str, chunks: List[str]) -> str:
    """Generate deterministic hash for a document"""
    content = f"{doc_id}:{''.join(chunks)}"
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def fetch_documents() -> List[Dict]:
    """Fetch all knowledge documents with their metadata"""
    print("Fetching knowledge documents from Supabase...", flush=True)
    
    # First, try to fetch documents with preferred statuses
    try:
        response = supabase.table("knowledge_documents").select(
            "id, title, content, is_chunked, total_chunks, learning_status, "
            "learning_category, document_type, doc_type, tags, created_at"
        ).in_("learning_status", ["learned", "processing"]).execute()
        
        if response.data and len(response.data) > 0:
            print(f"Found {len(response.data)} documents with learning_status='learned' or 'processing'", flush=True)
            return response.data
    except Exception as e:
        print(f"Warning: Error fetching preferred documents: {e}", flush=True)
    
    # If no preferred documents, check what statuses exist
    print("No documents with learning_status='learned' or 'processing' found.", flush=True)
    print("Checking for documents with any status...", flush=True)
    
    try:
        # Fetch all documents to see what we have
        response = supabase.table("knowledge_documents").select(
            "id, title, content, is_chunked, total_chunks, learning_status, "
            "learning_category, document_type, doc_type, tags, created_at"
        ).execute()
        
        if not response.data or len(response.data) == 0:
            print("ERROR: No documents found in knowledge_documents table at all.", flush=True)
            return []
        
        # Show status breakdown
        status_counts = {}
        for doc in response.data:
            status = doc.get("learning_status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\nFound {len(response.data)} total documents with the following statuses:", flush=True)
        for status, count in status_counts.items():
            print(f"  - {status}: {count} documents", flush=True)
        
        # Use all documents for training (user can filter later if needed)
        print(f"\nUsing all {len(response.data)} documents for training...", flush=True)
        return response.data
        
    except Exception as e:
        print(f"Error fetching documents: {e}", flush=True)
        return []


def fetch_conversations(limit: int = 1000) -> List[Dict]:
    """
    Fetch conversations from epsilon_conversations table
    These will be formatted as conversational training data
    """
    print(f"Fetching conversations from epsilon_conversations...", flush=True)
    
    try:
        response = supabase.table("epsilon_conversations").select(
            "id, session_id, user_message, epsilon_response, created_at"
        ).order("created_at", desc=True).limit(limit).execute()
        
        if not response.data or len(response.data) == 0:
            print(f"  No conversations found in epsilon_conversations table", flush=True)
            return []
        
        print(f"  Found {len(response.data)} conversations", flush=True)
        return response.data
        
    except Exception as e:
        print(f"  Warning: Error fetching conversations: {e}", flush=True)
        return []


def format_conversation_for_training(conv: Dict) -> str:
    """
    Format a conversation as training text
    Format: User: [message]\nEpsilon: [response]\n\n
    """
    user_msg = conv.get("user_message", "").strip()
    epsilon_msg = conv.get("epsilon_response", "").strip()
    
    if not user_msg or not epsilon_msg:
        return ""
    
    # Format as conversational exchange
    formatted = f"User: {user_msg}\nEpsilon: {epsilon_msg}\n\n"
    return formatted


def fetch_all_chunks_bulk(document_ids: List[str]) -> Dict[str, List[str]]:
    """
    Fetch ALL chunks for ALL documents, one document at a time with pagination
    This avoids timeouts by processing smaller queries
    Returns a dict mapping document_id -> list of chunk texts (ordered by chunk_index)
    """
    if not document_ids:
        return {}
    
    print(f"Fetching chunks for {len(document_ids)} documents (one at a time to avoid timeouts)...", flush=True)
    
    chunks_by_doc = {}
    total_fetched = 0
    
    # Process one document at a time to avoid timeouts
    for doc_idx, doc_id in enumerate(document_ids, 1):
        print(f"\n  [{doc_idx}/{len(document_ids)}] Fetching chunks for document {doc_id[:8]}...", flush=True)
        
        doc_chunks = []
        page_size = 10  # Fetch 10 chunks at a time (chunks are very large, so smaller batches)
        offset = 0
        max_retries = 5  # More retries for large chunks
        
        # Fetch chunks for this document in pages
        response = None
        error_msg = None
        done = False
        
        while not done:
            for attempt in range(max_retries):
                try:
                    response = supabase.table("doc_chunks").select(
                        "chunk_text, chunk_index"
                    ).eq("document_id", doc_id).order("chunk_index", desc=False).range(offset, offset + page_size - 1).execute()
                    
                    if not response.data:
                        done = True
                        break
                    
                    doc_chunks.extend(response.data)
                    total_fetched += len(response.data)
                    
                    if len(doc_chunks) % 50 == 0 or len(response.data) < page_size:
                        print(f"    Fetched {len(doc_chunks):,} chunks...", flush=True)
                    
                    # If we got fewer than page_size, we're done with this document
                    if len(response.data) < page_size:
                        done = True
                        break
                    
                    offset += page_size
                    time.sleep(0.2)  # Longer delay between pages to avoid overwhelming the server
                    break  # Success, exit retry loop
                    
                except KeyboardInterrupt:
                    print(f"\n    ⚠ Interrupted by user at offset {offset:,}", flush=True)
                    print(f"    Progress saved: {len(doc_chunks):,} chunks fetched so far", flush=True)
                    raise  # Re-raise to allow user to handle
                    
                except Exception as e:
                    error_msg = str(e)
                    if ("timeout" in error_msg.lower() or "57014" in error_msg or "read" in error_msg.lower()) and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s, 8s, 10s
                        print(f"    Timeout at offset {offset:,} (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...", flush=True)
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"    Error fetching chunks: {error_msg[:200]}", flush=True)
                        if attempt == max_retries - 1:
                            print(f"    Failed after {max_retries} attempts at offset {offset:,}", flush=True)
                            print(f"    Progress: {len(doc_chunks):,} chunks fetched before failure", flush=True)
                            done = True
                        break
        
        # Sort chunks by chunk_index and extract text
        if doc_chunks:
            sorted_chunks = sorted(doc_chunks, key=lambda x: x.get("chunk_index", 0))
            chunks_by_doc[doc_id] = [chunk["chunk_text"] for chunk in sorted_chunks if chunk.get("chunk_text")]
            print(f"    ✓ Successfully fetched {len(chunks_by_doc[doc_id]):,} chunks", flush=True)
        else:
            chunks_by_doc[doc_id] = []
            print(f"    ⚠ No chunks fetched for this document", flush=True)
    
    print(f"\n{'='*60}", flush=True)
    print(f"Successfully fetched chunks for {sum(1 for v in chunks_by_doc.values() if v)}/{len(document_ids)} documents", flush=True)
    print(f"Total chunks fetched: {sum(len(v) for v in chunks_by_doc.values()):,}", flush=True)
    print(f"{'='*60}", flush=True)
    
    return chunks_by_doc


def fetch_chunks_for_document(doc_id: str, max_retries: int = 3) -> List[str]:
    """Legacy function - kept for compatibility, but use fetch_all_chunks_bulk instead"""
    # This is now a wrapper that calls the bulk function
    chunks_dict = fetch_all_chunks_bulk([doc_id])
    return chunks_dict.get(doc_id, [])


def reconstruct_document_text(doc: Dict, chunks: List[str]) -> str:
    """Reconstruct full document text from chunks or use content field"""
    if doc.get("is_chunked") and chunks:
        # Reconstruct from chunks
        return "\n\n".join(chunks)
    elif doc.get("content"):
        # Use direct content
        return doc["content"]
    else:
        return ""


def split_train_val(documents: List[Dict], val_ratio: float = 0.05) -> Tuple[List[Dict], List[Dict]]:
    """
    Split documents into train/val at document level
    Uses deterministic hash of doc_id for consistent splits
    """
    # Sort by doc_id for deterministic split
    sorted_docs = sorted(documents, key=lambda x: x["id"])
    
    # Use hash of doc_id to determine split (consistent across runs)
    val_count = max(1, int(len(sorted_docs) * val_ratio))
    
    # Simple modulo split for consistency
    train_docs = []
    val_docs = []
    
    for i, doc in enumerate(sorted_docs):
        if i % (len(sorted_docs) // val_count + 1) == 0 and len(val_docs) < val_count:
            val_docs.append(doc)
        else:
            train_docs.append(doc)
    
    return train_docs, val_docs


def write_corpus_file(documents: List[Dict], output_file: Path, chunks_by_doc: Dict[str, List[str]] = None, conversations: List[Dict] = None) -> List[Dict]:
    """Write documents and conversations to corpus file and return manifest entries"""
    manifest_entries = []
    
    if chunks_by_doc is None:
        chunks_by_doc = {}
    
    if conversations is None:
        conversations = []
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # CRITICAL: Add Epsilon identity training at the START of training data
        # This ensures the model learns its identity first and most strongly
        identity_file = Path(__file__).parent.parent / 'data' / 'instructions' / 'epsilon_identity.txt'
        if identity_file.exists():
            print(f"  Adding Epsilon identity training data...", flush=True)
            with open(identity_file, 'r', encoding='utf-8') as identity_f:
                identity_text = identity_f.read().strip()
                # Add identity data multiple times at the start for strong learning
                for _ in range(3):  # Repeat 3 times for emphasis
                    f.write(identity_text)
                    f.write("\n\n---DOCUMENT_SEPARATOR---\n\n")
            print(f"  ✓ Epsilon identity data added (repeated 3x for emphasis)", flush=True)
        else:
            print(f"  Warning: Identity file not found: {identity_file}", flush=True)
        for doc in documents:
            doc_id = doc["id"]
            
            # Get chunks from pre-fetched bulk data
            chunks = chunks_by_doc.get(doc_id, [])
            
            # If document is chunked but we don't have chunks, try to use content field
            if doc.get("is_chunked") and not chunks and doc.get("content"):
                print(f"  Warning: Chunks not found for {doc_id[:8]}..., using content field instead", flush=True)
            
            # Reconstruct text
            text = reconstruct_document_text(doc, chunks)
            
            if not text or len(text.strip()) < 100:
                print(f"  Skipping document {doc_id[:8]}... (too short or empty)", flush=True)
                continue  # Skip empty or very short documents
            
            # Write document text (one document per line, with separator)
            f.write(text.strip())
            f.write("\n\n---DOCUMENT_SEPARATOR---\n\n")
            
            # Create manifest entry
            doc_hash = get_document_hash(doc_id, chunks if chunks else [text])
            manifest_entries.append({
                "doc_id": doc_id,
                "title": doc.get("title", "Untitled"),
                "learning_category": doc.get("learning_category"),
                "document_type": doc.get("document_type"),
                "doc_type": doc.get("doc_type"),
                "tags": doc.get("tags"),
                "learning_status": doc.get("learning_status"),
                "is_chunked": doc.get("is_chunked", False),
                "chunk_count": len(chunks) if chunks else 0,
                "text_length": len(text),
                "hash": doc_hash,
                "created_at": doc.get("created_at")
            })
        
        # Add conversations after documents for conversational training
        if conversations:
            print(f"  Adding {len(conversations)} conversations...", flush=True)
            conv_count = 0
            for conv in conversations:
                formatted_conv = format_conversation_for_training(conv)
                if formatted_conv:
                    f.write(formatted_conv)
                    conv_count += 1
            print(f"  ✓ Added {conv_count} formatted conversations", flush=True)
    
    return manifest_entries


def main():
    print("=" * 60, flush=True)
    print("Pulling training corpus from Supabase", flush=True)
    print("=" * 60, flush=True)
    
    # Fetch documents
    documents = fetch_documents()
    
    if not documents:
        print("ERROR: No documents found. Please upload documents first.", flush=True)
        sys.exit(1)
    
    # Identify chunked documents and show their chunk counts
    chunked_docs = [doc for doc in documents if doc.get("is_chunked")]
    chunked_count = len(chunked_docs)
    print(f"\nFound {chunked_count} chunked documents:", flush=True)
    total_chunks_expected = 0
    for doc in chunked_docs:
        chunk_count = doc.get("total_chunks", 0)
        total_chunks_expected += chunk_count
        print(f"  - {doc['id'][:8]}... ({doc.get('title', 'Untitled')[:50]}): {chunk_count:,} chunks", flush=True)
    print(f"Total chunks to fetch: {total_chunks_expected:,}", flush=True)
    
    chunked_doc_ids = [doc["id"] for doc in chunked_docs]
    
    # Fetch ALL chunks in bulk (one big query instead of many small ones)
    chunks_by_doc = {}
    if chunked_doc_ids:
        chunks_by_doc = fetch_all_chunks_bulk(chunked_doc_ids)
    
    # Split train/val at document level
    print("\nSplitting into train/val sets (document-level split)...", flush=True)
    train_docs, val_docs = split_train_val(documents, val_ratio=0.05)
    print(f"  Train documents: {len(train_docs)}", flush=True)
    print(f"  Val documents: {len(val_docs)}", flush=True)
    
    # Fetch conversations for conversational training
    print("\nFetching conversations for conversational training...", flush=True)
    conversations = fetch_conversations(limit=2000)  # Get up to 2000 conversations
    
    if conversations:
        print(f"  Found {len(conversations)} conversations", flush=True)
        # Split conversations into train/val (80/20 split)
        conv_split_idx = int(len(conversations) * 0.8)
        train_convs = conversations[:conv_split_idx]
        val_convs = conversations[conv_split_idx:]
        print(f"  Train conversations: {len(train_convs)}", flush=True)
        print(f"  Val conversations: {len(val_convs)}", flush=True)
    else:
        print("  No conversations found - training will use documents only", flush=True)
        train_convs = []
        val_convs = []
    
    # Write corpus files (chunks are already fetched)
    print("\nWriting corpus files...", flush=True)
    train_manifest = write_corpus_file(train_docs, TRAIN_FILE, chunks_by_doc, train_convs)
    val_manifest = write_corpus_file(val_docs, VAL_FILE, chunks_by_doc, val_convs)
    
    # Calculate total chunks from manifest entries
    total_chunks = sum(entry.get("chunk_count", 0) for entry in train_manifest + val_manifest)
    
    # Create manifest
    manifest = {
        "created_at": str(Path(__file__).stat().st_mtime),
        "total_documents": len(documents),
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "total_chunks": total_chunks,
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "corpus_hash": hashlib.sha256(
            (str(len(train_manifest)) + str(len(val_manifest))).encode()
        ).hexdigest()[:16]
    }
    
    with open(MANIFEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    
    # Calculate file sizes
    train_size = TRAIN_FILE.stat().st_size / (1024 * 1024)  # MB
    val_size = VAL_FILE.stat().st_size / (1024 * 1024)  # MB
    
    print("\n" + "=" * 60, flush=True)
    print("Corpus pull complete!", flush=True)
    print("=" * 60, flush=True)
    print(f"Train file: {TRAIN_FILE} ({train_size:.2f} MB)", flush=True)
    print(f"Val file: {VAL_FILE} ({val_size:.2f} MB)", flush=True)
    print(f"Manifest: {MANIFEST_FILE}", flush=True)
    print(f"Total documents: {len(documents)}", flush=True)
    print(f"Train/Val split: {len(train_docs)}/{len(val_docs)}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()

