"""
FAISS vector store — manages the index lifecycle: add, search, persist, load.
Uses FAISS IndexFlatIP (inner product on L2-normalised vectors ≡ cosine similarity).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np

from app.validation.schemas import Chunk, SearchResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

_INDEX_FILENAME = "index.faiss"
_META_FILENAME = "metadata.json"


class FAISSStore:
    """Thread-safe FAISS index with chunk metadata persistence."""

    def __init__(self, dimension: int, index_dir: str | Path) -> None:
        self._dimension = dimension
        self._index_dir = Path(index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._metadata: Dict[int, dict] = {}  # faiss-id → chunk dict
        self._next_id: int = 0

        # Try to load existing index
        loaded = self._try_load()
        if not loaded:
            self._index = faiss.IndexFlatIP(dimension)
            logger.info("faiss_new_index_created", dimension=dimension)

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self._index.ntotal

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
    ) -> int:
        """
        Add chunks and their embeddings to the index.

        Args:
            chunks: Chunk objects with chunk_id, text, metadata.
            embeddings: np.ndarray of shape (len(chunks), dimension).

        Returns:
            Updated total number of vectors in the index.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings"
            )

        embeddings = embeddings.astype(np.float32)

        with self._lock:
            start_id = self._next_id
            for i, chunk in enumerate(chunks):
                fid = start_id + i
                self._metadata[fid] = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.metadata.get("source", ""),
                    "metadata": chunk.metadata,
                }
            self._next_id = start_id + len(chunks)
            self._index.add(embeddings)

        logger.info(
            "faiss_chunks_added",
            added=len(chunks),
            total=self._index.ntotal,
        )

        # Persist after every write
        self.persist()
        return self._index.ntotal

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Search the index for the nearest neighbours.

        Args:
            query_embedding: np.ndarray of shape (1, dimension) or (dimension,).
            top_k: Maximum number of results to return.

        Returns:
            List[SearchResult] sorted by descending similarity.
        """
        if self._index.ntotal == 0:
            logger.warning("faiss_search_empty_index")
            return []

        query = query_embedding.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        effective_k = min(top_k, self._index.ntotal)

        with self._lock:
            scores, ids = self._index.search(query, effective_k)

        results: List[SearchResult] = []
        for score, fid in zip(scores[0], ids[0]):
            if fid == -1:
                continue
            meta = self._metadata.get(int(fid), {})
            results.append(
                SearchResult(
                    chunk_id=meta.get("chunk_id", ""),
                    text=meta.get("text", ""),
                    score=float(score),
                    source=meta.get("source", ""),
                    metadata=meta.get("metadata", {}),
                )
            )

        logger.info(
            "faiss_search_complete",
            top_k=top_k,
            results_returned=len(results),
            top_score=results[0].score if results else None,
        )
        return results

    def persist(self) -> None:
        """Save index + metadata to disk."""
        index_path = self._index_dir / _INDEX_FILENAME
        meta_path = self._index_dir / _META_FILENAME

        with self._lock:
            faiss.write_index(self._index, str(index_path))
            # JSON keys must be strings
            serialisable = {str(k): v for k, v in self._metadata.items()}
            meta_path.write_text(
                json.dumps(serialisable, ensure_ascii=False), encoding="utf-8"
            )

        logger.info(
            "faiss_persisted",
            index_path=str(index_path),
            vectors=self._index.ntotal,
        )

    # ── Private ────────────────────────────────────────────────────────

    def _try_load(self) -> bool:
        """Load index + metadata from disk if files exist."""
        index_path = self._index_dir / _INDEX_FILENAME
        meta_path = self._index_dir / _META_FILENAME

        if not index_path.exists() or not meta_path.exists():
            return False

        try:
            self._index = faiss.read_index(str(index_path))
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            self._metadata = {int(k): v for k, v in raw.items()}
            self._next_id = max(self._metadata.keys(), default=-1) + 1
            logger.info(
                "faiss_loaded_from_disk",
                vectors=self._index.ntotal,
                metadata_entries=len(self._metadata),
            )
            return True
        except Exception as exc:
            logger.error("faiss_load_failed", error=str(exc))
            return False
