"""
LLM structured extractor — calls the LLM with a strict JSON-only system prompt.
The LLM is NEVER treated as a source of truth; it merely structures the
information already present in the retrieved chunks.
"""

from __future__ import annotations

import json
from typing import List

from openai import OpenAI

from app.validation.schemas import SearchResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise extraction engine. Use ONLY the provided context.\n"
    "Return a JSON object with keys: answer, confidence (high/medium/low), reasoning_steps (list).\n"
    "If the answer cannot be determined from context, set answer to null.\n"
    "Return ONLY valid JSON. No explanation. No markdown. No preamble."
)


def _build_context_block(chunks: List[SearchResult]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts: List[str] = []
    for i, chunk in enumerate(chunks, 1):
        source_label = chunk.source or chunk.metadata.get("source", "unknown")
        parts.append(
            f"[{i}] (source: {source_label}, score: {chunk.score:.3f})\n{chunk.text}"
        )
    return "\n\n".join(parts)


def extract_structured(
    query: str,
    chunks: List[SearchResult],
    api_key: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Call the LLM to produce a strict JSON extraction from the provided context.

    Args:
        query: The user's original question.
        chunks: Pre-verified, filtered evidence chunks.
        api_key: OpenAI API key.
        model: Model identifier (e.g., gpt-4o-mini).

    Returns:
        Raw dict parsed from the LLM's JSON response.

    Raises:
        ValueError: If the LLM response is not valid JSON.
        openai.OpenAIError: On API communication failures.
    """
    context = _build_context_block(chunks)

    user_message = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Respond with ONLY a JSON object. No markdown fences, no explanation."
    )

    logger.info(
        "llm_call_start",
        model=model,
        query_length=len(query),
        context_chunks=len(chunks),
    )

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content.strip()

    logger.info(
        "llm_call_complete",
        model=model,
        response_length=len(raw_text),
        usage={
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    )

    # ── Strict JSON parse ───────────────────────────────────────────
    # Strip markdown code fences if the model disobeys instructions
    cleaned = raw_text
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(
            "llm_invalid_json",
            raw_response=raw_text[:500],
            error=str(exc),
        )
        raise ValueError(
            f"LLM returned invalid JSON: {exc}. Raw response: {raw_text[:200]}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"LLM returned {type(parsed).__name__} instead of a JSON object"
        )

    logger.info(
        "llm_extraction_parsed",
        has_answer=parsed.get("answer") is not None,
        confidence=parsed.get("confidence", "unknown"),
    )
    return parsed
