"""
Central configuration — all values loaded from environment variables / .env file.
Uses Pydantic v2 BaseSettings for typed, validated config with env-var binding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings bound to environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"

    # ── Embeddings ───────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── Chunking ─────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # ── Retrieval ────────────────────────────────────────────────────────
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.35

    # ── FAISS persistence ────────────────────────────────────────────────
    FAISS_INDEX_PATH: str = "./data/faiss_index"

    # ── File uploads ─────────────────────────────────────────────────────
    UPLOAD_DIR: str = "./data/uploads"

    # ── Logging ──────────────────────────────────────────────────────────
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ── Derived helpers ──────────────────────────────────────────────────
    @property
    def faiss_index_dir(self) -> Path:
        p = Path(self.FAISS_INDEX_PATH)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def upload_dir(self) -> Path:
        p = Path(self.UPLOAD_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p


# Singleton — import this everywhere
settings = Settings()
