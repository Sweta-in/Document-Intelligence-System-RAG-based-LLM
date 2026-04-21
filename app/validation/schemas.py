"""Pydantic v2 schemas for every I/O boundary in the pipeline."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ── Document / Chunk models ─────────────────────────────────────────────────

class DocumentPage(BaseModel):
    """A single page / section extracted from an uploaded document."""
    text: str
    page_number: int
    source: str
    document_id: str


class Chunk(BaseModel):
    """A semantically-bounded text chunk with provenance metadata."""
    chunk_id: str
    text: str
    metadata: dict = Field(default_factory=dict)


# ── Retrieval models ────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    """A single FAISS search hit."""
    chunk_id: str
    text: str
    score: float
    source: str = ""
    metadata: dict = Field(default_factory=dict)


# ── Verification gate ───────────────────────────────────────────────────────

class VerificationResult(BaseModel):
    """Output of the pre-LLM verification gate."""
    passed: bool
    reason: str
    filtered_chunks: List[SearchResult] = Field(default_factory=list)


# ── LLM extraction ──────────────────────────────────────────────────────────

class LLMExtractionOutput(BaseModel):
    """Validated, structured output returned by the LLM."""
    answer: Optional[str] = None
    confidence: Literal["high", "medium", "low"]
    reasoning_steps: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def cross_field_checks(self) -> "LLMExtractionOutput":
        # If no answer was extracted, confidence must not be high
        if self.answer is None and self.confidence == "high":
            raise ValueError(
                "Confidence cannot be 'high' when answer is null — "
                "the model should not be highly confident about nothing."
            )
        # If confidence is high, reasoning steps must exist
        if self.confidence == "high" and not self.reasoning_steps:
            raise ValueError(
                "Confidence is 'high' but no reasoning_steps were provided — "
                "high-confidence answers require explicit justification."
            )
        return self


# ── Final API response ──────────────────────────────────────────────────────

class EvidenceChunk(BaseModel):
    """A chunk included in the final response as supporting evidence."""
    chunk_id: str
    text: str
    score: float
    source: str = ""


class FinalResponse(BaseModel):
    """The contract returned by POST /query."""
    answer: Optional[str] = None
    confidence: Literal["high", "medium", "low"] = "low"
    validation_passed: bool = False
    evidence_chunks: List[EvidenceChunk] = Field(default_factory=list)
    refusal_reason: Optional[str] = None


# ── Request / response DTOs ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Inbound query request body."""
    query: str = Field(..., min_length=1, max_length=2000)


class UploadResponse(BaseModel):
    """Response after a successful document upload + indexing."""
    document_id: str
    filename: str
    total_chunks: int
    message: str = "Document ingested and indexed successfully."


class HealthResponse(BaseModel):
    """Health-check response."""
    status: str = "healthy"
    index_size: int = 0
    embedding_model: str = ""
