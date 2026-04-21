"""
Evaluation hooks — lightweight retrieval quality and hallucination metrics.
These run inline (no external dependencies) and log results for observability.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

from app.utils.logger import get_logger

logger = get_logger(__name__)


def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """
    Compute Precision@K — the fraction of the top-K retrieved items that
    are relevant.

    Args:
        retrieved_ids: Ordered list of chunk IDs returned by the retriever.
        relevant_ids: Ground-truth relevant chunk IDs.
        k: Cutoff rank.

    Returns:
        Precision@K as a float in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_set: Set[str] = set(relevant_ids)
    hits = sum(1 for cid in top_k if cid in relevant_set)
    precision = hits / k

    logger.info(
        "precision_at_k",
        k=k,
        hits=hits,
        precision=precision,
        retrieved_count=len(retrieved_ids),
        relevant_count=len(relevant_ids),
    )
    return precision


def detect_hallucination(
    answer: str | None,
    chunks: List[dict],
) -> Dict:
    """
    Lightweight hallucination detector — checks whether key tokens in the
    answer are grounded in the evidence chunks.

    Strategy:
      - Extract significant tokens (≥4 chars, excluding stopwords) from the answer.
      - Check how many of those tokens appear in at least one chunk.
      - If grounding ratio is below threshold, flag as potentially hallucinated.

    Args:
        answer: The LLM-generated answer string (may be None).
        chunks: List of chunk dicts (must have a "text" key each).

    Returns:
        Dict with keys:
          - is_suspicious (bool)
          - grounding_ratio (float)
          - ungrounded_tokens (List[str])
          - total_answer_tokens (int)
    """
    if not answer:
        return {
            "is_suspicious": False,
            "grounding_ratio": 1.0,
            "ungrounded_tokens": [],
            "total_answer_tokens": 0,
        }

    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all",
        "can", "her", "was", "one", "our", "out", "has", "had",
        "this", "that", "with", "from", "have", "been", "will",
        "they", "what", "when", "where", "which", "who", "how",
        "does", "did", "its", "into", "than", "then", "them",
        "some", "such", "each", "other", "about", "more", "also",
        "would", "could", "should", "there", "their", "these",
        "those", "being", "only", "very", "just", "most", "much",
        "answer", "based", "according", "provided", "context",
    }

    # Extract significant tokens from answer
    answer_tokens = set(
        t for t in re.findall(r"\b[a-zA-Z]{4,}\b", answer.lower())
        if t not in stopwords
    )

    if not answer_tokens:
        return {
            "is_suspicious": False,
            "grounding_ratio": 1.0,
            "ungrounded_tokens": [],
            "total_answer_tokens": 0,
        }

    # Build combined chunk vocabulary
    chunk_text = " ".join(c.get("text", "") for c in chunks).lower()
    chunk_tokens = set(re.findall(r"\b[a-zA-Z]{4,}\b", chunk_text))

    # Find ungrounded tokens
    grounded = answer_tokens & chunk_tokens
    ungrounded = sorted(answer_tokens - chunk_tokens)
    ratio = len(grounded) / len(answer_tokens)

    SUSPICION_THRESHOLD = 0.5
    is_suspicious = ratio < SUSPICION_THRESHOLD

    result = {
        "is_suspicious": is_suspicious,
        "grounding_ratio": round(ratio, 4),
        "ungrounded_tokens": ungrounded[:20],  # Cap for logging
        "total_answer_tokens": len(answer_tokens),
    }

    logger.info(
        "hallucination_check",
        is_suspicious=is_suspicious,
        grounding_ratio=ratio,
        ungrounded_count=len(ungrounded),
    )
    return result
