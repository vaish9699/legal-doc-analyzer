"""Document loading and chunking."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader


@dataclass
class Chunk:
    """A chunk of text from a source document."""
    text: str
    source: str       # filename
    chunk_id: int     # 0-indexed position within the document


def load_document(path: Path) -> str:
    """Load text from a .txt or .pdf file."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks by word count.

    Overlap preserves context across chunk boundaries — important for legal
    clauses that may span boundaries.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    stride = chunk_size - overlap
    if stride <= 0:
        raise ValueError("chunk_size must be larger than overlap")

    for start in range(0, len(words), stride):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break

    return chunks


def ingest_directory(docs_dir: Path) -> Iterator[Chunk]:
    """Yield Chunks for every supported file in docs_dir."""
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    supported = {".txt", ".pdf"}
    files = sorted(p for p in docs_dir.iterdir() if p.suffix.lower() in supported)

    if not files:
        raise ValueError(f"No .txt or .pdf files found in {docs_dir}")

    for path in files:
        try:
            text = load_document(path)
        except Exception as exc:
            print(f"  ! Skipping {path.name}: {exc}")
            continue

        for i, chunk_text_str in enumerate(chunk_text(text)):
            yield Chunk(text=chunk_text_str, source=path.name, chunk_id=i)
