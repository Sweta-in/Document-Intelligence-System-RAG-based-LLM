"""
Document loader — extracts raw text + metadata from PDF, DOCX, and image files.
Gracefully falls back if pytesseract / Tesseract OCR is not installed.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Dict, List

from app.utils.logger import get_logger

logger = get_logger(__name__)


def _normalize_text(text: str) -> str:
    """Strip excessive whitespace, fix common encoding artefacts."""
    # Replace common unicode issues
    text = text.replace("\u00a0", " ")  # non-breaking space
    text = text.replace("\ufeff", "")   # BOM
    text = text.replace("\x00", "")     # null bytes
    # Collapse runs of whitespace (preserve single newlines for structure)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _generate_document_id(file_path: str) -> str:
    """Deterministic document ID from file path content hash."""
    path = Path(file_path)
    if path.exists():
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        return f"doc_{content_hash}"
    return f"doc_{uuid.uuid4().hex[:12]}"


# ── PDF via PyMuPDF ──────────────────────────────────────────────────────────

def _load_pdf(file_path: str, document_id: str) -> List[Dict]:
    import fitz  # PyMuPDF

    pages: List[Dict] = []
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        raw = page.get_text("text")
        text = _normalize_text(raw)
        if text:
            pages.append(
                {
                    "text": text,
                    "page_number": page_num + 1,
                    "source": Path(file_path).name,
                    "document_id": document_id,
                }
            )
    doc.close()
    logger.info(
        "pdf_loaded", file=file_path, pages_extracted=len(pages)
    )
    return pages


# ── DOCX via python-docx ────────────────────────────────────────────────────

def _load_docx(file_path: str, document_id: str) -> List[Dict]:
    from docx import Document

    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        logger.warning("docx_empty", file=file_path)
        return []

    # Group paragraphs into logical "pages" of ~20 paragraphs each
    page_size = 20
    pages: List[Dict] = []
    for i in range(0, len(paragraphs), page_size):
        chunk_paragraphs = paragraphs[i : i + page_size]
        text = _normalize_text("\n".join(chunk_paragraphs))
        if text:
            pages.append(
                {
                    "text": text,
                    "page_number": (i // page_size) + 1,
                    "source": Path(file_path).name,
                    "document_id": document_id,
                }
            )
    logger.info(
        "docx_loaded", file=file_path, sections_extracted=len(pages)
    )
    return pages


# ── Image via pytesseract (graceful fallback) ────────────────────────────────

def _load_image(file_path: str, document_id: str) -> List[Dict]:
    try:
        from PIL import Image
        import pytesseract

        img = Image.open(file_path)
        raw = pytesseract.image_to_string(img)
        text = _normalize_text(raw)
        if not text:
            logger.warning("image_ocr_empty", file=file_path)
            return []
        logger.info("image_ocr_success", file=file_path, chars=len(text))
        return [
            {
                "text": text,
                "page_number": 1,
                "source": Path(file_path).name,
                "document_id": document_id,
            }
        ]
    except ImportError:
        logger.warning(
            "ocr_unavailable",
            detail="pytesseract not installed — skipping image OCR",
            file=file_path,
        )
        return []
    except Exception as exc:
        logger.error(
            "image_ocr_failed",
            file=file_path,
            error=str(exc),
        )
        return []


# ── Public API ───────────────────────────────────────────────────────────────

_LOADERS = {
    "pdf": _load_pdf,
    "docx": _load_docx,
    "png": _load_image,
    "jpg": _load_image,
    "jpeg": _load_image,
    "tiff": _load_image,
    "bmp": _load_image,
    "webp": _load_image,
}

SUPPORTED_TYPES = set(_LOADERS.keys())


def detect_file_type(filename: str) -> str | None:
    """Return normalised extension or None if unsupported."""
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext if ext in SUPPORTED_TYPES else None


def load_document(file_path: str, file_type: str) -> List[Dict]:
    """
    Load a document and return a list of page dicts:
        [{text, page_number, source, document_id}, ...]

    Args:
        file_path: Absolute or relative path to the file.
        file_type: One of 'pdf', 'docx', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'webp'.

    Returns:
        List of extracted page dictionaries.

    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If file_path does not exist.
    """
    ft = file_type.lower().lstrip(".")
    if ft not in _LOADERS:
        raise ValueError(f"Unsupported file type: {file_type!r}. Supported: {SUPPORTED_TYPES}")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    document_id = _generate_document_id(file_path)
    loader = _LOADERS[ft]

    logger.info(
        "loading_document",
        file=file_path,
        file_type=ft,
        document_id=document_id,
    )
    return loader(file_path, document_id)
