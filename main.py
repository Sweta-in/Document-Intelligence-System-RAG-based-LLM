"""
Application entry point — FastAPI app with lifespan context manager.
Loads FAISS index on startup, mounts the API router.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import settings
from app.utils.logger import setup_logging, get_logger
from app.retrieval.embedder import get_embedding_dimension
from app.retrieval.faiss_store import FAISSStore
from app.verification.filter import VerificationFilter
from app.api.routes import router, set_faiss_store, set_verification_filter

# ── Logging ──────────────────────────────────────────────────────────────────
setup_logging(settings.LOG_LEVEL)
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown lifecycle:
      - Startup: load embedding model, initialise FAISS store, verification filter
      - Shutdown: persist FAISS index
    """
    logger.info("startup_begin", config={
        "CHUNK_SIZE": settings.CHUNK_SIZE,
        "CHUNK_OVERLAP": settings.CHUNK_OVERLAP,
        "TOP_K": settings.TOP_K,
        "SIMILARITY_THRESHOLD": settings.SIMILARITY_THRESHOLD,
        "LLM_MODEL": settings.LLM_MODEL,
        "EMBEDDING_MODEL": settings.EMBEDDING_MODEL,
        "FAISS_INDEX_PATH": settings.FAISS_INDEX_PATH,
    })

    # ── Embedding dimension ─────────────────────────────────────────
    dim = get_embedding_dimension(settings.EMBEDDING_MODEL)
    logger.info("embedding_dimension", dim=dim)

    # ── FAISS store ─────────────────────────────────────────────────
    store = FAISSStore(dimension=dim, index_dir=settings.FAISS_INDEX_PATH)
    set_faiss_store(store)
    logger.info("faiss_store_ready", vectors=store.size)

    # ── Verification filter ─────────────────────────────────────────
    vf = VerificationFilter(
        similarity_threshold=settings.SIMILARITY_THRESHOLD,
        min_chunks=1,
        min_keyword_overlap=0.15,
    )
    set_verification_filter(vf)

    logger.info("startup_complete")

    yield  # ── Application runs here ──

    # ── Shutdown ────────────────────────────────────────────────────
    logger.info("shutdown_begin")
    store.persist()
    logger.info("shutdown_complete")


# ── App construction ─────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Document Intelligence",
    description=(
        "Verification-first RAG pipeline: ingest documents, retrieve evidence, "
        "verify sufficiency, extract structured answers via LLM, and validate outputs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="", tags=["RAG Pipeline"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level=settings.LOG_LEVEL.lower(),
    )
