#How to execute:
#python step3_create_embeddings_qdrant.py

import os, json, yaml, numpy as np, logging
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams

# Setup logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"step3_embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_jsonl(path):
    return [json.loads(line) for line in open(path, "r", encoding="utf-8")]

def create_embeddings(cfg):
    data_path = os.path.join(cfg["processed_dir"], "clean.jsonl")
    data = load_jsonl(data_path)
    logger.info(f"Loaded {len(data)} docs from {data_path}")

    # --- Load embedding model ---
    device = cfg.get("embedding", {}).get("device", "cpu")
    logger.info(f"Using device: {device}")
    
    model = SentenceTransformer(cfg["embedding"]["model"], device=device)
    batch_size = cfg.get("embedding", {}).get("batch_size", 32)
    normalize = cfg.get("embedding", {}).get("normalize_embeddings", True)

    texts = [d["text"] for d in data]
    ids = list(range(len(texts)))

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        vecs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=normalize)
        embeddings.append(vecs)
    embeddings = np.vstack(embeddings)
    logger.info(f"✅ Generated {len(embeddings)} embeddings, dim={embeddings.shape[1]}")

    # --- Connect to Qdrant ---
    client = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])
    coll = cfg["qdrant"]["collection"]
    
    logger.info(f"Recreating collection: {coll}")
    client.recreate_collection(
        collection_name=coll,
        vectors_config=VectorParams(size=embeddings.shape[1], distance="Cosine")
    )

    # --- Upload to Qdrant ---
    points = [
        PointStruct(id=i, vector=embeddings[i], payload=data[i])
        for i in range(len(data))
    ]
    client.upsert(collection_name=coll, points=points)
    logger.info(f"✅ Uploaded {len(points)} vectors to Qdrant collection '{coll}'")

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    create_embeddings(cfg)