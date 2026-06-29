"""Embedding and FAISS index management."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .ingest import Chunk

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # MiniLM-L6 produces 384-dim vectors


class VectorStore:
    """FAISS-backed vector store for chunks."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Embed all chunks and build the index."""
        if not chunks:
            raise ValueError("No chunks to index")

        texts = [c.text for c in chunks]
        print(f"Embedding {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine similarity via inner product
        ).astype(np.float32)

        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(embeddings)
        self.chunks = chunks

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        """Return top-k chunks for a query."""
        if self.index is None:
            raise RuntimeError("Index not built. Run ingest first.")

        query_emb = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        scores, indices = self.index.search(query_emb, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, index_dir: Path) -> None:
        """Persist index + chunks to disk."""
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "faiss.index"))
        with open(index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(index_dir / "meta.json", "w") as f:
            json.dump(
                {"model": EMBEDDING_MODEL, "num_chunks": len(self.chunks)},
                f,
                indent=2,
            )

    def load(self, index_dir: Path) -> None:
        """Load index + chunks from disk."""
        if not index_dir.exists():
            raise FileNotFoundError(
                f"Index directory not found: {index_dir}. Run 'ingest' first."
            )
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        with open(index_dir / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
