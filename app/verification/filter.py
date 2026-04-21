"""
Pre-LLM verification gate — decides whether retrieved evidence is sufficient
before spending tokens on an LLM call.
"""

from __future__ import annotations

import re
from typing import List, Set

from app.validation.schemas import SearchResult, VerificationResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _extract_keywords(text: str) -> Set[str]:
    """Extract lowercased alphanumeric tokens ≥ 3 chars as keywords."""
    tokens = re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())
    # Remove very common English stopwords
    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all",
        "can", "her", "was", "one", "our", "out", "has", "had",
        "this", "that", "with", "from", "have", "been", "will",
        "they", "what", "when", "where", "which", "who", "how",
        "does", "did", "its", "into", "than", "then", "them",
        "some", "such", "each", "other", "about", "more", "also",
    }
    return set(tokens) - stopwords


class VerificationFilter:
    """
    Pre-LLM gate that checks whether retrieved chunks are sufficient to
    answer the query.  If the gate fails, the system returns a refusal
    response immediately — no LLM call is made.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.35,
        min_chunks: int = 1,
        min_keyword_overlap: float = 0.15,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.min_chunks = min_chunks
        self.min_keyword_overlap = min_keyword_overlap

    def check(
        self,
        query: str,
        retrieved_chunks: List[SearchResult],
    ) -> VerificationResult:
        """
        Run all verification checks and return a structured result.

        Checks applied (in order):
          1. Minimum chunk count after score-based filtering
          2. At least one chunk above the similarity threshold
          3. Keyword overlap between query and retrieved text

        Returns:
            VerificationResult with passed flag, reason, and filtered chunks.
        """
        # ── Step 1: filter by similarity score ──────────────────────
        above_threshold = [
            c for c in retrieved_chunks if c.score >= self.similarity_threshold
        ]

        if len(above_threshold) < self.min_chunks:
            reason = (
                f"Insufficient evidence: only {len(above_threshold)} chunk(s) "
                f"above similarity threshold {self.similarity_threshold} "
                f"(need ≥ {self.min_chunks})"
            )
            logger.warning(
                "verification_failed_score",
                reason=reason,
                scores=[c.score for c in retrieved_chunks],
            )
            return VerificationResult(
                passed=False, reason=reason, filtered_chunks=[]
            )

        # ── Step 2: keyword overlap ─────────────────────────────────
        query_kw = _extract_keywords(query)
        if query_kw:
            combined_text = " ".join(c.text for c in above_threshold)
            chunk_kw = _extract_keywords(combined_text)
            overlap = query_kw & chunk_kw
            ratio = len(overlap) / len(query_kw) if query_kw else 0.0

            if ratio < self.min_keyword_overlap:
                reason = (
                    f"Low keyword overlap ({ratio:.2%}) between query and "
                    f"retrieved chunks (threshold: {self.min_keyword_overlap:.0%}). "
                    f"Query keywords: {sorted(query_kw)[:10]}, "
                    f"Matched: {sorted(overlap)[:10]}"
                )
                logger.warning(
                    "verification_failed_keywords",
                    reason=reason,
                    overlap_ratio=ratio,
                )
                return VerificationResult(
                    passed=False, reason=reason, filtered_chunks=[]
                )

        # ── All checks passed ───────────────────────────────────────
        logger.info(
            "verification_passed",
            chunks_passed=len(above_threshold),
            top_score=above_threshold[0].score if above_threshold else 0.0,
        )
        return VerificationResult(
            passed=True,
            reason="Verification passed",
            filtered_chunks=above_threshold,
        )
