from qdrant_client import QdrantClient
import yaml
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Load config ---
cfg = yaml.safe_load(open("config.yaml"))
collection = cfg["qdrant"]["collection"]

# --- Connect to Qdrant ---
client = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"])

# --- Verify collection exists ---
collections = client.get_collections()
print("\n📚 Collections available in Qdrant:")
for c in collections.collections:
    print(" -", c.name)
print()

# --- Count points in your collection ---
count = client.count(collection_name=collection).count
print(f"✅ Collection '{collection}' contains {count} vectors\n")

# --- Try a simple semantic search ---
query = "what does the valuation tool do?"
print(f"🔍 Query: {query}")
model = SentenceTransformer(cfg["embedding"]["model"])
q_vec = model.encode(query, normalize_embeddings=True)

results = client.search(
    collection_name=collection,
    query_vector=q_vec.tolist(),
    limit=3
)

print("\nTop 3 retrieved documents:\n")
for i, r in enumerate(results, 1):
    print(f"{i}. Score: {r.score:.3f}")
    print(f"   Source: {r.payload.get('source')}")
    print(f"   Snippet: {r.payload.get('text', '')[:200]!r}")
    print()