"""Answer generation using Claude with retrieved context."""
from __future__ import annotations

import os
from dataclasses import dataclass

from anthropic import Anthropic

from .ingest import Chunk

SYSTEM_PROMPT = """You are a legal document assistant. Answer questions strictly based on the provided document excerpts.

Rules:
- Use ONLY information from the provided excerpts. If the answer is not present, say "I could not find this in the provided documents."
- Cite excerpts using the [N] markers (e.g., "The notice period is 30 days [1].").
- Be concise and precise. Legal answers should not speculate.
- If excerpts contradict each other, point this out."""


@dataclass
class Answer:
    text: str
    sources: list[Chunk]


def build_context(chunks: list[Chunk]) -> str:
    """Format retrieved chunks as numbered context for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] Source: {chunk.source} (chunk {chunk.chunk_id})\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def answer_query(
    query: str,
    retrieved: list[tuple[Chunk, float]],
    model: str = "claude-sonnet-4-5",
) -> Answer:
    """Call Claude with the query and retrieved context."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Export it or add it to .env"
        )

    client = Anthropic(api_key=api_key)
    chunks = [c for c, _ in retrieved]
    context = build_context(chunks)

    user_message = (
        f"Document excerpts:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer based only on the excerpts above. Cite using [N]."
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer_text = "".join(
        block.text for block in response.content if block.type == "text"
    )
    return Answer(text=answer_text, sources=chunks)
