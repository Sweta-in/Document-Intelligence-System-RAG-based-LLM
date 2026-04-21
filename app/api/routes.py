"""
FastAPI routes — POST /upload, POST /query, GET /health.
Full error handling with appropriate HTTP status codes.
"""

from __future__ import annotations

import shutil
import traceback
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import ValidationError

from config import settings
from app.ingestion.loader import detect_file_type, load_document
from app.ingestion.chunker import chunk_documents
from app.retrieval.embedder import embed_texts
from app.retrieval.faiss_store import FAISSStore
from app.verification.filter import VerificationFilter
from app.llm.extractor import extract_structured
from app.validation.validator import validate_llm_output
from app.validation.schemas import (
    EvidenceChunk,
    FinalResponse,
    HealthResponse,
    QueryRequest,
    UploadResponse,
)
from app.evaluation.hooks import detect_hallucination
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# ── Shared state (injected by main.py lifespan) ─────────────────────────────
# These are set during application startup via the lifespan context manager.
_faiss_store: FAISSStore | None = None
_verification_filter: VerificationFilter | None = None


def set_faiss_store(store: FAISSStore) -> None:
    global _faiss_store
    _faiss_store = store


def set_verification_filter(vf: VerificationFilter) -> None:
    global _verification_filter
    _verification_filter = vf


def _get_store() -> FAISSStore:
    if _faiss_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FAISS store not initialised — service is starting up.",
        )
    return _faiss_store


def _get_filter() -> VerificationFilter:
    if _verification_filter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Verification filter not initialised.",
        )
    return _verification_filter


# ─────────────────────────────────────────────────────────────────────────────
# POST /upload
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
    description="Accepts PDF, DOCX, or image files. Extracts text, chunks, embeds, and indexes.",
)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document → extract → chunk → embed → index."""

    # ── Validate file type ──────────────────────────────────────────
    filename = file.filename or "unknown"
    file_type = detect_file_type(filename)
    if file_type is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {filename}. Supported: pdf, docx, png, jpg, jpeg, tiff, bmp, webp",
        )

    # ── Save upload to disk ─────────────────────────────────────────
    upload_dir = settings.upload_dir
    dest = upload_dir / filename
    try:
        with open(dest, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
    except Exception as exc:
        logger.error("upload_save_failed", filename=filename, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {exc}",
        )
    finally:
        await file.close()

    # ── Ingest ──────────────────────────────────────────────────────
    try:
        pages = load_document(str(dest), file_type)
        if not pages:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No text could be extracted from the document.",
            )

        chunks = chunk_documents(
            pages,
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP,
        )
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Chunking produced zero chunks — document may be too short.",
            )

        # ── Embed & index ───────────────────────────────────────────
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts, model_name=settings.EMBEDDING_MODEL)

        store = _get_store()
        store.add_chunks(chunks, embeddings)

        document_id = pages[0].get("document_id", "unknown")

        logger.info(
            "upload_complete",
            document_id=document_id,
            filename=filename,
            pages=len(pages),
            chunks=len(chunks),
        )

        return UploadResponse(
            document_id=document_id,
            filename=filename,
            total_chunks=len(chunks),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "upload_pipeline_failed",
            filename=filename,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion pipeline failed: {exc}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# POST /query
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=FinalResponse,
    summary="Query the document store",
    description="Runs retrieval → verification → LLM extraction → validation pipeline.",
)
async def query_documents(request: QueryRequest):
    """
    Full RAG pipeline:
      1. Embed query
      2. FAISS search
      3. Verification gate (pre-LLM)
      4. LLM structured extraction
      5. Pydantic validation (post-LLM)
      6. Hallucination check
      7. Return structured response
    """
    query = request.query
    store = _get_store()
    vf = _get_filter()

    # ── Step 1: Embed query ─────────────────────────────────────────
    try:
        query_embedding = embed_texts(
            [query], model_name=settings.EMBEDDING_MODEL
        )
    except Exception as exc:
        logger.error("query_embedding_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to embed query: {exc}",
        )

    # ── Step 2: FAISS search ────────────────────────────────────────
    if store.size == 0:
        return FinalResponse(
            answer=None,
            confidence="low",
            validation_passed=False,
            evidence_chunks=[],
            refusal_reason="No documents have been indexed yet. Please upload documents first.",
        )

    results = store.search(query_embedding[0], top_k=settings.TOP_K)

    logger.info(
        "retrieval_results",
        query=query,
        results_count=len(results),
        chunk_ids=[r.chunk_id for r in results],
        scores=[round(r.score, 4) for r in results],
    )

    # ── Step 3: Verification gate ───────────────────────────────────
    verification = vf.check(query, results)

    logger.info(
        "verification_result",
        passed=verification.passed,
        reason=verification.reason,
        chunks_after_filter=len(verification.filtered_chunks),
    )

    if not verification.passed:
        return FinalResponse(
            answer=None,
            confidence="low",
            validation_passed=False,
            evidence_chunks=[],
            refusal_reason=verification.reason,
        )

    # ── Step 4: LLM structured extraction ───────────────────────────
    filtered = verification.filtered_chunks

    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY not configured. Cannot call LLM.",
        )

    try:
        raw_output = extract_structured(
            query=query,
            chunks=filtered,
            api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL,
        )
    except ValueError as exc:
        logger.error("llm_extraction_failed", error=str(exc))
        return FinalResponse(
            answer=None,
            confidence="low",
            validation_passed=False,
            evidence_chunks=[
                EvidenceChunk(
                    chunk_id=c.chunk_id,
                    text=c.text[:300],
                    score=c.score,
                    source=c.source,
                )
                for c in filtered
            ],
            refusal_reason=f"LLM returned invalid output: {exc}",
        )
    except Exception as exc:
        logger.error("llm_call_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM API call failed: {exc}",
        )

    # ── Step 5: Validation (BEFORE any explanation) ─────────────────
    try:
        validated = validate_llm_output(raw_output)
    except ValidationError as exc:
        logger.error(
            "output_validation_failed",
            errors=exc.errors(),
            raw=raw_output,
        )
        return FinalResponse(
            answer=None,
            confidence="low",
            validation_passed=False,
            evidence_chunks=[
                EvidenceChunk(
                    chunk_id=c.chunk_id,
                    text=c.text[:300],
                    score=c.score,
                    source=c.source,
                )
                for c in filtered
            ],
            refusal_reason=f"LLM output failed validation: {exc.errors()}",
        )

    # ── Step 6: Hallucination check ─────────────────────────────────
    hall_result = detect_hallucination(
        validated.answer,
        [{"text": c.text} for c in filtered],
    )

    if hall_result["is_suspicious"]:
        logger.warning(
            "hallucination_detected",
            grounding_ratio=hall_result["grounding_ratio"],
            ungrounded=hall_result["ungrounded_tokens"],
        )
        # Downgrade confidence if hallucination is suspected
        confidence = "low"
    else:
        confidence = validated.confidence

    # ── Step 7: Build final response ────────────────────────────────
    evidence = [
        EvidenceChunk(
            chunk_id=c.chunk_id,
            text=c.text[:500],
            score=c.score,
            source=c.source,
        )
        for c in filtered
    ]

    final = FinalResponse(
        answer=validated.answer,
        confidence=confidence,
        validation_passed=True,
        evidence_chunks=evidence,
        refusal_reason=None,
    )

    logger.info(
        "query_complete",
        query=query,
        answer_present=validated.answer is not None,
        confidence=confidence,
        validation_passed=True,
        hallucination_suspicious=hall_result["is_suspicious"],
        evidence_count=len(evidence),
    )

    return final


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health():
    """Return service status and index statistics."""
    store = _faiss_store
    return HealthResponse(
        status="healthy",
        index_size=store.size if store else 0,
        embedding_model=settings.EMBEDDING_MODEL,
    )
