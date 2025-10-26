"""
Improved data cleaning with document-type-aware chunking.
Usage: python step1_clean_data_improved.py --config config.yaml
"""

import os, re, json, yaml
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from unstructured.partition.auto import partition

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_chunking_config(file_ext: str, config: dict) -> Optional[dict]:
    """Determine chunking strategy based on file extension."""
    chunking_cfg = config['document_processing']['chunking']
    
    for doc_type, settings in chunking_cfg.items():
        if file_ext in settings.get('extensions', []):
            return settings
    
    # Default: treat as medium document
    return chunking_cfg['medium_documents']

def chunk_text(text: str, chunk_config: dict) -> List[str]:
    """
    Split text into chunks based on strategy.
    Strategies:
    - 'none': Return text as-is
    - 'recursive': Split by paragraphs, then sentences
    - 'fixed': Fixed-size character chunks
    - 'sentence': Split by sentences
    """
    strategy = chunk_config.get('strategy', 'none')
    
    if strategy == 'none':
        return [text]
    
    chunk_size = chunk_config.get('chunk_size', 1000)
    chunk_overlap = chunk_config.get('chunk_overlap', 200)
    min_chunk_size = chunk_config.get('min_chunk_size', 100)
    
    if strategy == 'recursive':
        return chunk_recursive(text, chunk_size, chunk_overlap, min_chunk_size)
    elif strategy == 'fixed':
        return chunk_fixed(text, chunk_size, chunk_overlap)
    elif strategy == 'sentence':
        return chunk_by_sentence(text, chunk_size, chunk_overlap, min_chunk_size)
    else:
        return [text]

def chunk_recursive(text: str, size: int, overlap: int, min_size: int) -> List[str]:
    """Split by paragraphs first, then by sentences if needed."""
    chunks = []
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(current_chunk) + len(para) <= size:
            current_chunk += para + "\n\n"
        else:
            if len(current_chunk) >= min_size:
                chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + para + "\n\n"
            else:
                current_chunk = para + "\n\n"
    
    if len(current_chunk) >= min_size:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]

def chunk_fixed(text: str, size: int, overlap: int) -> List[str]:
    """Fixed-size character chunks with overlap."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += size - overlap
    
    return chunks

def chunk_by_sentence(text: str, size: int, overlap: int, min_size: int) -> List[str]:
    """Chunk by sentences, respecting size limits."""
    # Simple sentence split (can be improved with spaCy/NLTK)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sent in sentences:
        if len(current_chunk) + len(sent) <= size:
            current_chunk += sent + " "
        else:
            if len(current_chunk) >= min_size:
                chunks.append(current_chunk.strip())
            current_chunk = sent + " "
    
    if len(current_chunk) >= min_size:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text]

def clean_text(text: str, cleaning_config: dict) -> str:
    """Remove boilerplate and normalize whitespace."""
    patterns = cleaning_config.get('custom_patterns', [])
    
    for pat in patterns:
        text = re.sub(pat, "", text)
    
    if cleaning_config.get('normalize_whitespace', True):
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        
        max_newlines = cleaning_config.get('max_consecutive_newlines', 2)
        text = re.sub(f"\\n{{{max_newlines+1},}}", "\n" * max_newlines, text)
    
    return text.strip()

def iter_files(root_dir: str, supported_exts: List[str]) -> List[Path]:
    """Recursively find supported files."""
    root = Path(root_dir)
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in supported_exts]

def extract_and_clean(config: dict):
    """Extract text, clean, chunk, and write to JSONL."""
    raw_dir = config['raw_dir']
    processed_dir = config['processed_dir']
    out_jsonl = os.path.join(processed_dir, "clean.jsonl")
    
    supported_exts = config['document_processing']['supported_extensions']
    cleaning_cfg = config['document_processing']['cleaning']
    
    files = iter_files(raw_dir, supported_exts)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    
    total_chunks = 0
    
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for fp in tqdm(files, desc="Processing documents"):
            try:
                # Extract text
                elements = partition(filename=str(fp))
                text = "\n".join(
                    [el.text for el in elements if getattr(el, "text", None)]
                ).strip()
                
                if not text:
                    continue
                
                # Clean text
                text = clean_text(text, cleaning_cfg)
                
                # Get chunking strategy for this file type
                chunk_config = get_chunking_config(fp.suffix.lower(), config)
                chunks = chunk_text(text, chunk_config)
                
                # Write each chunk as separate record
                for idx, chunk in enumerate(chunks):
                    rec = {
                        "source": str(fp),
                        "filename": fp.name,
                        "suffix": fp.suffix.lower(),
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "text": chunk,
                        "char_count": len(chunk),
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_chunks += 1
                
            except Exception as e:
                print(f"[WARN] Failed to process {fp}: {e}")
    
    print(f"\n✅ Processed {len(files)} files into {total_chunks} chunks")
    print(f"📄 Output: {out_jsonl}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = ap.parse_args()
    
    config = load_config(args.config)
    extract_and_clean(config)
