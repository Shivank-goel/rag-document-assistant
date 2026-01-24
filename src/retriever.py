from typing import List
from embeddings import embed_text
from vector_store import search_index

class Retriever:
    def __init__(self, index, chunks: List[str]):
        self.index = index
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = embed_text([query])[0]
        indices, distances = search_index(self.index, query_embedding, top_k)
        return [self.chunks[idx] for idx in indices]    