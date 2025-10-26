"""
Incremental document addition - only processes NEW files.
Usage: python add_documents.py --new_files "path/to/doc1.pdf" "path/to/doc2.pdf"
"""

import os, json, yaml, hashlib
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import argparse

# Import functions from your existing scripts
import sys
sys.path.append(str(Path(__file__).parent))

from step1_clean_data import partition, clean_text, SUPPORTED_EXT

def get_file_hash(filepath):
    """Generate unique hash for file to track what's been processed."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_processed_hashes(jsonl_path):
    """Get set of already processed file hashes."""
    if not Path(jsonl_path).exists():
        return set()
    
    hashes = set()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            if 'file_hash' in rec:
                hashes.add(rec['file_hash'])
    return hashes

def get_next_id(jsonl_path):
    """Get next available ID for Qdrant."""
    if not Path(jsonl_path).exists():
        return 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        count = sum(1 for _ in f)
    return count

def process_new_files(new_files, cfg):
    """Process only new files and add to existing collection."""
    
    jsonl_path = Path(cfg['processed_dir']) / "clean.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check which files are actually new
    processed_hashes = load_processed_hashes(jsonl_path)
    next_id = get_next_id(jsonl_path)
    
    files_to_process = []
    for fp in new_files:
        fp = Path(fp)
        if not fp.exists():
            print(f"[WARN] File not found: {fp}")
            continue
        
        file_hash = get_file_hash(fp)
        if file_hash in processed_hashes:
            print(f"[SKIP] Already processed: {fp.name}")
            continue
        
        if fp.suffix.lower() not in SUPPORTED_EXT:
            print(f"[SKIP] Unsupported format: {fp.name}")
            continue
        
        files_to_process.append((fp, file_hash))
    
    if not files_to_process:
        print("\n✅ No new files to process!")
        return
    
    print(f"\n📄 Processing {len(files_to_process)} new files...")
    
    # Process and append to JSONL
    new_records = []
    with open(jsonl_path, 'a', encoding='utf-8') as fout:
        for fp, file_hash in tqdm(files_to_process, desc="Extracting"):
            try:
                # Extract text
                elements = partition(filename=str(fp))
                text = "\n".join(
                    [el.text for el in elements if getattr(el, "text", None)]
                ).strip()
                
                if not text:
                    print(f"[WARN] No text extracted from {fp.name}")
                    continue
                
                # Clean text
                text = clean_text(text)
                
                rec = {
                    "source": str(fp),
                    "filename": fp.name,
                    "suffix": fp.suffix.lower(),
                    "text": text,
                    "file_hash": file_hash,
                }
                
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                new_records.append(rec)
                
            except Exception as e:
                print(f"[ERROR] Failed to process {fp.name}: {e}")
    
    print(f"✅ Extracted {len(new_records)} new documents")
    
    # Generate embeddings and add to Qdrant
    if new_records:
        print("\n🔢 Generating embeddings...")
        
        model = SentenceTransformer(cfg['embedding']['model'])
        client = QdrantClient(host=cfg['qdrant']['host'], port=cfg['qdrant']['port'])
        collection = cfg['qdrant']['collection']
        
        # Embed new documents
        texts = [rec['text'] for rec in new_records]
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=cfg['embedding'].get('normalize_embeddings', True),
            batch_size=cfg['embedding'].get('batch_size', 32)
        )
        
        # Upload to Qdrant
        points = [
            PointStruct(
                id=next_id + i,
                vector=embeddings[i].tolist(),
                payload=new_records[i]
            )
            for i in range(len(new_records))
        ]
        
        client.upsert(collection_name=collection, points=points)
        print(f"✅ Added {len(points)} vectors to Qdrant collection '{collection}'")
        print(f"\n🎉 Done! Your knowledge base now has {next_id + len(points)} documents")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_files", nargs="+", help="Paths to new PDF/DOCX files")
    ap.add_argument("--scan_raw", action="store_true", help="Scan raw folder for new files")
    ap.add_argument("--config", default="config.yaml", help="Config file path")
    args = ap.parse_args()
    
    cfg = yaml.safe_load(open(args.config))
    
    # Determine which files to process
    if args.scan_raw:
        # Scan raw folder for all files
        raw_dir = Path(cfg['raw_dir'])
        if not raw_dir.exists():
            print(f"❌ Raw folder not found: {raw_dir}")
            exit(1)
        
        files = [str(p) for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]
        if not files:
            print(f"No supported files found in {raw_dir}")
            exit(0)
        print(f"Found {len(files)} files in raw folder")
    elif args.new_files:
        files = args.new_files
    else:
        print("❌ Must specify either --new_files or --scan_raw")
        exit(1)
    
    process_new_files(files, cfg)
