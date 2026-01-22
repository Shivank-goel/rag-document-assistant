from sentence_transformers import SentenceTransformer
from typing import List

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(chunks: List[str]):
    return model.encode(chunks)