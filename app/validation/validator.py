"""
Output validator — applies Pydantic schema parsing AND cross-field business rules
to the raw LLM output dict.  Validation happens BEFORE any explanation is generated.
"""

from __future__ import annotations

from pydantic import ValidationError

from app.validation.schemas import LLMExtractionOutput
from app.utils.logger import get_logger

logger = get_logger(__name__)


def validate_llm_output(raw: dict) -> LLMExtractionOutput:
    """
    Parse and validate the raw LLM JSON against the LLMExtractionOutput schema.

    Applies:
      1. Pydantic type coercion + field validation
      2. Cross-field invariants via model_validator on the schema:
         - answer=null → confidence ≠ "high"
         - confidence="high" → reasoning_steps must be non-empty

    Args:
        raw: The dict returned by extractor.extract_structured().

    Returns:
        Validated LLMExtractionOutput instance.

    Raises:
        pydantic.ValidationError: On schema / cross-field failures.
    """
    # ── Normalise common LLM quirks before parsing ──────────────────
    normalised = dict(raw)

    # The LLM might return "None" or "N/A" as strings instead of null
    answer_val = normalised.get("answer")
    if isinstance(answer_val, str) and answer_val.strip().lower() in (
        "none", "n/a", "null", "unknown", "not found", "not available",
        "cannot be determined", "insufficient information",
    ):
        normalised["answer"] = None

    # Ensure confidence is lowercased
    if "confidence" in normalised and isinstance(normalised["confidence"], str):
        normalised["confidence"] = normalised["confidence"].strip().lower()

    # Ensure reasoning_steps is a list
    steps = normalised.get("reasoning_steps")
    if steps is None:
        normalised["reasoning_steps"] = []
    elif isinstance(steps, str):
        normalised["reasoning_steps"] = [steps]

    # ── Parse with Pydantic (includes cross-field validators) ───────
    try:
        validated = LLMExtractionOutput.model_validate(normalised)
        logger.info(
            "validation_passed",
            answer_present=validated.answer is not None,
            confidence=validated.confidence,
            reasoning_steps_count=len(validated.reasoning_steps),
        )
        return validated

    except ValidationError as exc:
        logger.error(
            "validation_failed",
            errors=exc.errors(),
            raw_input=raw,
        )
        raise
