# RAG Document Assistant — Architecture

## Overview

The RAG (Retrieval-Augmented Generation) Document Assistant is a question-answering system that combines **document retrieval** with **text generation** to provide accurate, context-grounded answers. Instead of relying solely on a language model's internal knowledge, it retrieves relevant document chunks and uses them as context for generating responses.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        RAG Document Assistant                        │
│                                                                      │
│  ┌─────────────────────── INGESTION PIPELINE ──────────────────────┐ │
│  │                                                                  │ │
│  │  ┌──────────┐   ┌────────────┐   ┌───────────┐   ┌───────────┐ │ │
│  │  │  Raw     │──▶│  Text      │──▶│  Text     │──▶│  Embedding│ │ │
│  │  │  Document│   │  Cleaning  │   │  Chunking │   │  Generation│ │ │
│  │  │(sample   │   │(utils.py)  │   │(utils.py) │   │(embeddings│ │ │
│  │  │  .txt)   │   │            │   │           │   │  .py)     │ │ │
│  │  └──────────┘   └────────────┘   └───────────┘   └─────┬─────┘ │ │
│  │                                                         │       │ │
│  │                                                         ▼       │ │
│  │                                                  ┌────────────┐ │ │
│  │                                                  │   FAISS    │ │ │
│  │                                                  │   Vector   │ │ │
│  │                                                  │   Store    │ │ │
│  │                                                  │(vector_    │ │ │
│  │                                                  │ store.py)  │ │ │
│  │                                                  └────────────┘ │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌─────────────────────── QUERY PIPELINE ──────────────────────────┐ │
│  │                                                                  │ │
│  │  ┌──────────┐   ┌────────────┐   ┌───────────┐   ┌───────────┐ │ │
│  │  │  User    │──▶│  Query     │──▶│  Context  │──▶│  Answer   │ │ │
│  │  │  Query   │   │  Embedding │   │  Retrieval│   │  Generation│ │ │
│  │  │(FastAPI) │   │(embeddings │   │(retriever │   │(generator │ │ │
│  │  │          │   │  .py)      │   │  .py)     │   │  .py)     │ │ │
│  │  └──────────┘   └────────────┘   └───────────┘   └───────────┘ │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
rag-document-assistant/
├── README.md                  # Project overview and setup instructions
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── data/
│   ├── raw/
│   │   └── sample.txt         # Raw input documents
│   └── processed/             # (Future) Processed/cleaned documents
├── docs/
│   └── architecture.md        # This file
└── src/
    ├── __init__.py             # Package initializer
    ├── app.py                  # FastAPI application & API endpoints
    ├── ingest.py               # Document loading from disk
    ├── utils.py                # Text cleaning & chunking utilities
    ├── embeddings.py           # Text-to-vector embedding generation
    ├── vector_store.py         # FAISS index creation & search
    ├── retriever.py            # Retrieves relevant chunks for a query
    ├── generator.py            # LLM-based answer generation
    └── query.py                # Sample query payload
```

---

## Component Details

### 1. Document Ingestion — `ingest.py`

**Purpose**: Loads raw text documents from disk.

- Uses `Path(__file__).parent.parent` to resolve file paths relative to the project root (not the current working directory)
- Returns `Optional[str]` — `None` on failure
- Handles `FileNotFoundError` and other exceptions gracefully

```
data/raw/sample.txt → load_raw_document() → raw text (str)
```

---

### 2. Text Processing — `utils.py`

**Purpose**: Cleans and splits text into manageable chunks.

#### `clean_text(text: str) -> str`
- Strips leading/trailing whitespace from each line
- Removes empty lines
- Joins cleaned lines back together

#### `chunk_text(text: str, chunk_size=500, overlap=50) -> List[str]`
- Splits text into fixed-size chunks (default: 500 characters)
- Uses overlapping windows (default: 50 characters) to preserve context across chunk boundaries
- Overlap ensures no information is lost at chunk edges

```
raw text → clean_text() → cleaned text → chunk_text() → ["chunk1", "chunk2", ...]
```

---

### 3. Embedding Generation — `embeddings.py`

**Purpose**: Converts text chunks into 384-dimensional numerical vectors.

- Uses the **`all-MiniLM-L6-v2`** model from `sentence-transformers`
- Each chunk becomes a vector of 384 floating-point numbers
- Semantically similar text produces vectors that are close in vector space

```
["chunk1", "chunk2"] → embed_text() → [[0.1, 0.2, ...], [0.4, 0.5, ...]]
                                          384 dimensions each
```

---

### 4. Vector Store — `vector_store.py`

**Purpose**: Stores embeddings in a FAISS index for fast similarity search.

#### `create_faiss_index(embeddings) -> faiss.Index`
- Creates a **Flat L2 index** (exhaustive search using Euclidean distance)
- Converts embeddings to `float32` for FAISS compatibility and performance
- Stores all chunk embeddings in the index

#### `search_index(index, query_embedding, top_k=3) -> (indices, distances)`
- Searches the FAISS index for the `top_k` nearest neighbors
- Returns the indices and distances of the closest chunks
- Lower distance = higher similarity

```
embeddings (float32) → FAISS IndexFlatL2 → searchable vector index
query vector → search_index() → [index_2, index_0, index_4], [0.1, 0.3, 0.5]
```

---

### 5. Retriever — `retriever.py`

**Purpose**: Bridges the query and the vector store; returns relevant text chunks.

- Embeds the user query using the same embedding model
- Searches the FAISS index for the top-k most similar chunks
- Maps the returned indices back to the original text chunks

```
"How does RAG work?" → embed → search FAISS → indices → text chunks
```

---

### 6. Generator — `generator.py`

**Purpose**: Generates natural language answers using an LLM with retrieved context.

- Uses **`google/flan-t5-small`** via HuggingFace `transformers` pipeline
- Trims contexts to a max of 1500 characters to stay within model token limits
- Constructs a prompt with explicit rules:
  - Answer using ONLY the provided context
  - Say "I don't know" if the answer isn't in the context
  - Explain in 2–4 clear sentences
  - No external knowledge

```
query + retrieved contexts → prompt template → flan-t5-small → answer
```

---

### 7. API Layer — `app.py`

**Purpose**: Exposes the RAG system as a REST API using FastAPI.

#### Startup
On server start, the full ingestion pipeline runs once:
1. Load document → Clean → Chunk → Embed → Create FAISS index
2. Initialize `Retriever` and `Generator` objects

#### Endpoints

| Method | Path     | Description                                 |
|--------|----------|---------------------------------------------|
| POST   | `/query` | Accepts a question, returns an answer + contexts |

#### Request Schema — `QueryRequest`
```json
{
  "query": "How does RAG reduce hallucinations?",
  "top_k": 3
}
```

#### Response Schema — `QueryResponse`
```json
{
  "answer": "RAG reduces hallucinations by...",
  "contexts": ["context chunk 1...", "context chunk 2...", "context chunk 3..."]
}
```

---

## Data Flow — End to End

```
                         INGESTION (runs once at startup)
                         ================================

  ┌────────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ sample.txt │───▶│  clean   │───▶│  chunk   │───▶│  embed   │───▶│  FAISS   │
  │ (raw doc)  │    │  text    │    │  text    │    │  text    │    │  index   │
  └────────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                         │                               │
                                         │          stored in memory     │
                                         ▼                               ▼
                                      chunks[]                     index object


                         QUERY (runs per request)
                         ========================

  ┌────────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ User Query │───▶│  embed   │───▶│  search  │───▶│ retrieve │───▶│ generate │
  │ (string)   │    │  query   │    │  FAISS   │    │ chunks   │    │  answer  │
  └────────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                                         │
                                                                         ▼
                                                                   JSON response
                                                                  {answer, contexts}
```

---

## Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| Language         | Python 3.9+                         |
| Web Framework    | FastAPI                             |
| Embeddings       | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store     | FAISS (faiss-cpu)                   |
| LLM              | google/flan-t5-small (HuggingFace)  |
| Server           | Uvicorn                             |

---

## How to Run

```bash
# From the project root
cd rag-document-assistant

# Install dependencies
pip3 install -r requirements.txt

# Start the server
cd src
python3 -m uvicorn app:app --reload

# API available at: http://127.0.0.1:8000
# Swagger docs at:  http://127.0.0.1:8000/docs
```

---

## Future Improvements

- [ ] Support for PDF and DOCX file ingestion
- [ ] Persistent vector store (save/load FAISS index to disk)
- [ ] Configurable chunk size and overlap via API or config file
- [ ] Streaming responses for long answers
- [ ] Support for multiple documents
- [ ] Authentication for the API
- [ ] Evaluation metrics (faithfulness, relevance)
- [ ] Upgrade to a larger LLM (flan-t5-base, flan-t5-large)
