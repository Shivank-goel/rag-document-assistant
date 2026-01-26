import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from ingest import load_raw_document
from utils import clean_text, chunk_text
from embeddings import embed_text
from vector_store import create_faiss_index
from retriever import Retriever
from generator import Generator

app = FastAPI(title="RAG Document Assistant")

# ---- Startup: build pipeline once ----

raw_text = load_raw_document("data/raw/sample.txt")
cleaned_text = clean_text(raw_text)
chunks = chunk_text(cleaned_text)

embeddings = embed_text(chunks)
index = create_faiss_index(embeddings)

retriever = Retriever(index=index, chunks=chunks)
generator = Generator()


# ---- Request / Response Schemas ----

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]


# ---- API Endpoint ----

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    contexts = retriever.retrieve(request.query, top_k=request.top_k)
    answer = generator.generate(request.query, contexts)

    return {
        "answer": answer,
        "contexts": contexts
    }
