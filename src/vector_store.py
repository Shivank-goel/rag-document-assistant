import faiss
import numpy as np
from typing import List

def create_faiss_index(embeddings: List[List[float]]):
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index

def search_index(index, query_embedding, top_k:int = 3):
    distances, indicies = index.search(
        query_embedding.reshape(1, -1).astype("float32"), top_k             
    )
    return indicies[0], distances[0]