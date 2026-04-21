"""
Embedding generator — wraps sentence-transformers for batch text embedding.
Model is loaded once and reused for the process lifetime.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    """Load the sentence-transformer model exactly once."""
    from sentence_transformers import SentenceTransformer

    logger.info("loading_embedding_model", model=model_name)
    model = SentenceTransformer(model_name)
    logger.info(
        "embedding_model_loaded",
        model=model_name,
        embedding_dim=model.get_sentence_embedding_dimension(),
    )
    return model


def get_embedding_dimension(model_name: str) -> int:
    """Return the dimensionality of the embedding model."""
    model = _load_model(model_name)
    return model.get_sentence_embedding_dimension()


def embed_texts(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: Input strings to embed.
        model_name: HuggingFace sentence-transformers model identifier.
        batch_size: Batch size for encoding.
        normalize: Whether to L2-normalize embeddings (required for cosine sim in FAISS).

    Returns:
        np.ndarray of shape (len(texts), embedding_dim), dtype float32.
    """
    if not texts:
        logger.warning("embed_texts_empty_input")
        return np.array([], dtype=np.float32).reshape(0, 0)

    model = _load_model(model_name)

    logger.info(
        "embedding_texts",
        count=len(texts),
        model=model_name,
        batch_size=batch_size,
    )

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    result = embeddings.astype(np.float32)
    logger.info(
        "embedding_complete",
        shape=list(result.shape),
    )
    return result
