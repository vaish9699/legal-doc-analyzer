"""Command-line interface."""
from __future__ import annotations

from pathlib import Path

import click
from dotenv import load_dotenv

from .answer import answer_query
from .embed import VectorStore
from .ingest import ingest_directory

load_dotenv()

INDEX_DIR = Path("index")


@click.group()
def cli():
    """Legal Document Analyzer - RAG over your contracts."""
    pass


@cli.command()
@click.option(
    "--docs-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data"),
    help="Directory containing .txt/.pdf documents",
)
def ingest(docs_dir: Path):
    """Build the vector index from documents."""
    click.echo(f"Loading documents from {docs_dir}/...")
    chunks = list(ingest_directory(docs_dir))
    click.echo(f"Loaded {len(chunks)} chunks from {len({c.source for c in chunks})} documents.")

    store = VectorStore()
    store.build(chunks)
    store.save(INDEX_DIR)
    click.echo(f"Index saved to {INDEX_DIR}/")


@cli.command()
@click.argument("query")
@click.option("-k", default=4, help="Number of chunks to retrieve")
def ask(query: str, k: int):
    """Ask a single question."""
    store = VectorStore()
    store.load(INDEX_DIR)
    retrieved = store.search(query, k=k)

    click.echo("\nAnswer:")
    answer = answer_query(query, retrieved)
    click.echo(answer.text)

    click.echo("\nSources:")
    for i, chunk in enumerate(answer.sources, start=1):
        click.echo(f"  [{i}] {chunk.source} (chunk {chunk.chunk_id})")


@cli.command()
@click.option("-k", default=4, help="Number of chunks to retrieve per query")
def chat(k: int):
    """Interactive Q&A session."""
    store = VectorStore()
    store.load(INDEX_DIR)
    click.echo("Type your questions. Ctrl-C or 'exit' to quit.\n")

    while True:
        try:
            query = click.prompt("Q", prompt_suffix="> ").strip()
        except (KeyboardInterrupt, EOFError):
            click.echo("\nGoodbye.")
            break

        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        retrieved = store.search(query, k=k)
        answer = answer_query(query, retrieved)

        click.echo(f"\nA: {answer.text}\n")
        click.echo("Sources:")
        for i, chunk in enumerate(answer.sources, start=1):
            click.echo(f"  [{i}] {chunk.source} (chunk {chunk.chunk_id})")
        click.echo()


if __name__ == "__main__":
    cli()
