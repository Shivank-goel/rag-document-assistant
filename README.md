I built an end-to-end Retrieval-Augmented Generation system. Documents are ingested, cleaned, chunked, and converted into embeddings, which are stored in a FAISS vector database.
When a user submits a query, I embed the query, retrieve the most relevant chunks via semantic similarity, and pass them as grounded context to an instruction-tuned LLM.
I intentionally used a smaller model to validate grounding and prevent hallucination, and exposed the entire pipeline through a FastAPI service.
