"""Tests for chunking logic."""
from src.ingest import chunk_text


def test_chunk_text_basic():
    text = " ".join(["word"] * 1000)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) >= 2
    assert all(len(c.split()) <= 500 for c in chunks)


def test_chunk_text_overlap():
    """Adjacent chunks should share `overlap` words."""
    words = [f"w{i}" for i in range(600)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()
    # last 50 words of chunk 0 should equal first 50 words of chunk 1
    assert first_chunk_words[-50:] == second_chunk_words[:50]


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_short():
    text = "only a few words here"
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert chunks == [text]


def test_invalid_overlap():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=10, overlap=10)
