"""
Simple RAG Query Engine using Ollama only.
"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import yaml
import requests
import logging
import os
from datetime import datetime
from typing import List, Dict

# Setup logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"query_engine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGQueryEngine:
    def __init__(self, config_path="config.yaml"):
        self.cfg = yaml.safe_load(open(config_path))
        
        # Embedding model for retrieval
        logger.info(f"Loading embedding model: {self.cfg['embedding']['model']}...")
        self.embedding_model = SentenceTransformer(self.cfg['embedding']['model'])
        
        # Qdrant client
        self.client = QdrantClient(
            host=self.cfg['qdrant']['host'], 
            port=self.cfg['qdrant']['port']
        )
        self.collection = self.cfg['qdrant']['collection']
        
        # Query settings
        self.top_k = self.cfg['query']['top_k']
        self.score_threshold = self.cfg['query']['score_threshold']
        
        # Ollama settings
        self.ollama_host = self.cfg['llm']['host']
        self.ollama_model = self.cfg['llm']['model']
        self.temperature = self.cfg['llm']['temperature']
        self.max_tokens = self.cfg['llm']['max_tokens']
        
        logger.info(f"Connected to Ollama at: {self.ollama_host}")
        logger.info(f"Using model: {self.ollama_model}")
    
    def retrieve(self, query: str, filters: dict = None) -> List[Dict]:
        """Step 1: Retrieve relevant chunks from vector DB."""
        q_vec = self.embedding_model.encode(query, normalize_embeddings=True)
        
        results = self.client.search(
            collection_name=self.collection,
            query_vector=q_vec.tolist(),
            limit=self.top_k,
            score_threshold=self.score_threshold,
            query_filter=filters
        )
        
        return [
            {
                "text": r.payload['text'],
                "source": r.payload['source'],
                "score": r.score,
                "chunk_index": r.payload.get('chunk_index', 0),
                "filename": r.payload.get('filename', 'unknown')
            }
            for r in results
        ]
    
    def build_context(self, chunks: List[Dict]) -> str:
        """Step 2: Combine retrieved chunks into context."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Document {i}: {chunk['filename']}]\n{chunk['text']}\n"
            )
        context = "\n".join(context_parts)
        return context
    
    def query(self, question: str, filters: dict = None) -> Dict:
        """
        Complete RAG query: retrieve + generate answer with Ollama.
        
        Args:
            question: User's question
            filters: Optional metadata filters (e.g., {"suffix": ".pdf"})
        
        Returns:
            Dict with 'answer' and 'sources'
        """
        # Retrieve relevant chunks
        chunks = self.retrieve(question, filters)
        
        if not chunks:
            return {
                "answer": "No relevant documents found for your question.",
                "sources": []
            }
        
        # Build context from chunks
        context = self.build_context(chunks)
        
        # Create prompt for Ollama
        prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the provided documents.

Documents:
{context}

Question: {question}

Instructions:
- Answer based only on the documents above
- If the answer is not in the documents, say "I cannot find this information in the provided documents"
- Be concise and specific
- Do NOT add document citations in your answer

Answer:"""
        
        # Call Ollama
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            answer = response.json()['response']
        except Exception as e:
            return {
                "answer": f"Error calling Ollama: {e}",
                "sources": []
            }
        
        return {
            "answer": answer,
            "sources": [
                {
                    "file": c['filename'],
                    "score": round(c['score'], 3),
                    "chunk": c['chunk_index']
                }
                for c in chunks
            ]
        }


# ========================================
# Usage Examples
# ========================================

if __name__ == "__main__":
    import sys
    
    engine = RAGQueryEngine()
    
    # Get question from command line argument or use default
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "what does the valuation tool do?"
    
    print(f"Question: {question}\n")
    print("Retrieving relevant documents...")
    
    result = engine.query(question)
    
    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(result['answer'])
    
    print("\n" + "="*60)
    print("SOURCES:")
    print("="*60)
    for i, src in enumerate(result['sources'], 1):
        print(f"{i}. {src['file']} (similarity: {src['score']}, chunk: {src['chunk']})")
