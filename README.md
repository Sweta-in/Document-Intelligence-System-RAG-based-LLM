# RAG Document Intelligence Pipeline

A **verification-first** Retrieval-Augmented Generation system that ingests documents (PDF, DOCX, images), builds a FAISS vector index, and answers queries using a strict **retrieve → verify → extract → validate** pipeline.

> **Core principle:** The LLM is **never** a source of truth. It is used _only_ for structured JSON extraction from evidence that has already been retrieved and verified.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        POST /upload                              │
│  File → Loader (PDF/DOCX/Image) → Chunker → Embedder → FAISS   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                        POST /query                               │
│                                                                  │
│  Query → Embed → FAISS Search ─┐                                │
│                                │                                 │
│                    ┌───────────▼───────────┐                     │
│                    │  VERIFICATION GATE    │                     │
│                    │  • score threshold    │                     │
│                    │  • min chunk count    │                     │
│                    │  • keyword overlap    │                     │
│                    └───────────┬───────────┘                     │
│                                │                                 │
│                    PASS ───────┤──────── FAIL                    │
│                    │                      │                       │
│           ┌────────▼────────┐    ┌───────▼──────┐               │
│           │  LLM Extraction │    │  REFUSAL     │               │
│           │  (JSON only)    │    │  response    │               │
│           └────────┬────────┘    └──────────────┘               │
│                    │                                             │
│           ┌────────▼────────┐                                    │
│           │  Pydantic       │                                    │
│           │  Validation     │                                    │
│           │  + cross-field  │                                    │
│           └────────┬────────┘                                    │
│                    │                                             │
│           ┌────────▼────────┐                                    │
│           │  Hallucination  │                                    │
│           │  Check          │                                    │
│           └────────┬────────┘                                    │
│                    │                                             │
│                    ▼                                             │
│           Final JSON Response                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
project/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # FastAPI routes: POST /upload, POST /query, GET /health
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py          # Load PDF, DOCX, image → raw text + metadata
│   │   └── chunker.py         # Semantic chunking with configurable size/overlap
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embedder.py        # Generate embeddings (sentence-transformers)
│   │   └── faiss_store.py     # FAISS index: add, search, persist
│   ├── llm/
│   │   ├── __init__.py
│   │   └── extractor.py       # LLM call — strict JSON schema only
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── schemas.py         # Pydantic models for all I/O
│   │   └── validator.py       # Schema + cross-field validation logic
│   ├── verification/
│   │   ├── __init__.py
│   │   └── filter.py          # Pre-LLM gate: score threshold, field presence, relevance
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── hooks.py           # Precision@K, hallucination detection
│   └── utils/
│       ├── __init__.py
│       └── logger.py          # Structured JSON logging
├── main.py                    # FastAPI app init + lifespan
├── config.py                  # All config via env vars
├── requirements.txt
├── Dockerfile
├── .env.example
├── .dockerignore
└── .gitignore
```

---

## Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Vector Store | FAISS (faiss-cpu) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | OpenAI API (gpt-4o-mini default) |
| PDF Extraction | PyMuPDF (fitz) |
| DOCX Extraction | python-docx |
| Image OCR | Pillow + pytesseract (optional) |
| Validation | Pydantic v2 |
| Logging | structlog (JSON) |
| Containerisation | Docker |

---

## Quick Start

### 1. Clone & configure

```bash
git clone <repo-url>
cd RAGDocumentIntelligence

# Create your .env from the template
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Run locally (Python)

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Linux/macOS)
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

The server starts at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 3. Run with Docker

```bash
# Build the image
docker build -t rag-doc-intel .

# Run the container
docker run -d \
  --name rag-doc-intel \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  rag-doc-intel
```

---

## API Endpoints

### `GET /health` — Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "index_size": 42,
  "embedding_model": "all-MiniLM-L6-v2"
}
```

### `POST /upload` — Upload & Ingest Document

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/document.pdf"
```

**Response (201):**
```json
{
  "document_id": "doc_a1b2c3d4e5f6",
  "filename": "document.pdf",
  "total_chunks": 27,
  "message": "Document ingested and indexed successfully."
}
```

### `POST /query` — Query the Document Store

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the company revenue for Q3 2024?"}'
```

**Success response:**
```json
{
  "answer": "The company reported Q3 2024 revenue of $4.2 billion, representing a 12% year-over-year increase.",
  "confidence": "high",
  "validation_passed": true,
  "evidence_chunks": [
    {
      "chunk_id": "doc_a1b2c3_3_0_f8e2a1",
      "text": "In Q3 2024, total consolidated revenue reached $4.2 billion...",
      "score": 0.87,
      "source": "annual_report_2024.pdf"
    },
    {
      "chunk_id": "doc_a1b2c3_5_1_c3d4e5",
      "text": "Year-over-year revenue growth was 12%, driven by...",
      "score": 0.79,
      "source": "annual_report_2024.pdf"
    }
  ],
  "refusal_reason": null
}
```

**Refusal response (insufficient evidence):**
```json
{
  "answer": null,
  "confidence": "low",
  "validation_passed": false,
  "evidence_chunks": [],
  "refusal_reason": "Insufficient evidence: only 0 chunk(s) above similarity threshold 0.35 (need ≥ 1)"
}
```

---

## Configuration

All configuration is via environment variables (or `.env` file):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key (required for /query) |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model identifier |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CHUNK_SIZE` | `512` | Target chunk size in characters |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.35` | Minimum cosine similarity to pass verification |
| `FAISS_INDEX_PATH` | `./data/faiss_index` | Directory for persisted FAISS index |
| `UPLOAD_DIR` | `./data/uploads` | Directory for uploaded files |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Verification-First Philosophy

1. **Retrieval** — FAISS finds the top-K most similar chunks.
2. **Verification gate** — Before any LLM call:
   - Are enough chunks above the similarity threshold?
   - Do the chunks share keyword overlap with the query?
   - If **no** → the system **refuses** immediately. No LLM call is made.
3. **LLM extraction** — The LLM receives **only** the verified chunks and must return **strict JSON** with `answer`, `confidence`, and `reasoning_steps`.
4. **Validation** — Pydantic v2 validates the JSON schema **and** cross-field invariants:
   - `answer=null` → `confidence` cannot be `"high"`
   - `confidence="high"` → `reasoning_steps` must be non-empty
5. **Hallucination check** — Token overlap verifies the answer is grounded in evidence.
6. **Response** — Only after all checks pass is the final response returned.

---

## Sample End-to-End Flow

```bash
# 1. Check health
curl http://localhost:8000/health

# 2. Upload a PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@sample_report.pdf"

# 3. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What were the key findings in the report?"}'
```

---

## Logging

All pipeline steps emit structured JSON logs to stdout:

```json
{"event": "retrieval_results", "query": "...", "results_count": 5, "scores": [0.87, 0.79, ...], "timestamp": "..."}
{"event": "verification_passed", "chunks_passed": 3, "top_score": 0.87, "timestamp": "..."}
{"event": "llm_call_complete", "model": "gpt-4o-mini", "usage": {"prompt_tokens": 1024, "completion_tokens": 128}, "timestamp": "..."}
{"event": "validation_passed", "confidence": "high", "timestamp": "..."}
{"event": "hallucination_check", "is_suspicious": false, "grounding_ratio": 0.92, "timestamp": "..."}
```

---

## License

MIT
