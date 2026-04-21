"""
Semantic chunker — splits extracted document pages into overlapping chunks
that respect sentence boundaries where possible.
"""

from __future__ import annotations

import re
import uuid
from typing import Dict, List

from app.validation.schemas import Chunk
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Sentence-ending pattern (handles ., !, ? followed by space or end-of-string)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_into_sentences(text: str) -> List[str]:
    """Best-effort sentence splitting."""
    sentences = _SENTENCE_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _build_chunks_from_sentences(
    sentences: List[str],
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """
    Greedily pack sentences into chunks of ≤ chunk_size characters,
    then prepend `overlap` characters from the previous chunk's tail.
    """
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        sent_len = len(sentence)

        # If a single sentence exceeds chunk_size, hard-split it
        if sent_len > chunk_size:
            # Flush current buffer first
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            # Hard split the oversized sentence
            for i in range(0, sent_len, chunk_size - overlap):
                chunks.append(sentence[i : i + chunk_size])
            continue

        if current_len + sent_len + (1 if current else 0) > chunk_size:
            # Flush current buffer
            chunks.append(" ".join(current))
            current = []
            current_len = 0

        current.append(sentence)
        current_len += sent_len + (1 if len(current) > 1 else 0)

    if current:
        chunks.append(" ".join(current))

    # Apply overlap: prepend tail of previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + " " + chunks[i])
        chunks = overlapped

    return chunks


def chunk_documents(
    docs: List[Dict],
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[Chunk]:
    """
    Split a list of document page dicts into semantically-bounded Chunk objects.

    Args:
        docs: List of dicts with keys {text, page_number, source, document_id}.
        chunk_size: Target maximum characters per chunk.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List[Chunk] with unique IDs and full metadata provenance.
    """
    all_chunks: List[Chunk] = []

    for doc in docs:
        text = doc.get("text", "")
        if not text.strip():
            continue

        sentences = _split_into_sentences(text)
        if not sentences:
            # Fallback: treat entire text as one "sentence"
            sentences = [text]

        raw_chunks = _build_chunks_from_sentences(sentences, chunk_size, overlap)

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{doc.get('document_id', 'unknown')}_{doc.get('page_number', 0)}_{idx}_{uuid.uuid4().hex[:6]}"
            all_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata={
                        "source": doc.get("source", ""),
                        "document_id": doc.get("document_id", ""),
                        "page_number": doc.get("page_number", 0),
                        "chunk_index": idx,
                    },
                )
            )

    logger.info(
        "chunking_complete",
        input_pages=len(docs),
        output_chunks=len(all_chunks),
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return all_chunks
