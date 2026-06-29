# Legal Document Analyzer

A Retrieval-Augmented Generation (RAG) system for querying legal contracts in natural language. Ask questions about your contracts and get answers with citations to the source text.

## Features

- Ingests `.txt` and `.pdf` legal documents from a folder
- Chunks documents with overlap to preserve clause context
- Embeds chunks using `sentence-transformers` (runs locally, no API cost)
- Stores embeddings in a FAISS index for fast semantic search
- Answers queries using Anthropic's Claude API with retrieved context
- Returns answers with source citations (document name + chunk number)

## Architecture

```
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │  Documents   │───▶│   Chunker    │───▶│   Embedder   │
   │  (.txt/.pdf) │    │  (overlap)   │    │   (MiniLM)   │
   └──────────────┘    └──────────────┘    └──────┬───────┘
                                                  │
                                                  ▼
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │    Answer    │◀───│  Claude API  │◀───│  FAISS Index │
   │ + citations  │    │  (with ctx)  │    │  (retrieval) │
   └──────────────┘    └──────────────┘    └──────────────┘
```

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/yourusername/legal-doc-analyzer.git
cd legal-doc-analyzer
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Or copy `.env.example` to `.env` and fill it in.

## Usage

**Step 1: Build the index from your documents**

```bash
python -m src.cli ingest --docs-dir data/
```

This chunks every `.txt` and `.pdf` in `data/`, embeds the chunks, and saves a FAISS index to `index/`.

**Step 2: Ask questions**

```bash
python -m src.cli ask "What is the termination notice period?"
```

Example output:
```
Answer:
The termination notice period is 30 days written notice from either party.

Sources:
  [1] sample_nda.txt (chunk 4)
  [2] sample_service_agreement.txt (chunk 7)
```

**Interactive mode:**

```bash
python -m src.cli chat
```

## Project Structure

```
legal-doc-analyzer/
├── src/
│   ├── __init__.py
│   ├── cli.py           # CLI entry point
│   ├── ingest.py        # Document loading + chunking
│   ├── embed.py         # Embedding + FAISS index
│   ├── retrieve.py      # Semantic search
│   └── answer.py        # Claude API + prompt
├── data/                # Your legal documents
├── tests/               # Unit tests
├── requirements.txt
└── README.md
```

## Design Decisions

**Why FAISS over Chroma/Pinecone?** FAISS runs in-process with no server, making the project trivial to set up for a demo. For a production system with concurrent users or persistence guarantees, I would use a managed vector DB.

**Why sentence-transformers (MiniLM) over OpenAI embeddings?** MiniLM runs locally and is free, which keeps the project self-contained. Quality is good enough for legal text in English. For higher accuracy on domain-specific legal language, a fine-tuned legal embedding model (e.g., `nlpaueb/legal-bert-base-uncased`) would be the upgrade path.

**Why fixed-size chunking with overlap?** Legal text has structural cues (clauses, sections), but clause-aware chunking adds complexity. Fixed chunks with 50-token overlap retain enough context across boundaries for typical queries. A future improvement would be section-aware chunking using document headings.

**Why return citations?** Legal queries demand traceability. Returning the source document + chunk lets users verify the answer against the original text — non-negotiable for any tool used in legal work.

## Limitations

- No support for scanned PDFs (OCR not included)
- Chunks are fixed-size, not clause-aware
- No conversation memory in `ask` mode (use `chat` mode for multi-turn)
- Index is rebuilt from scratch on each `ingest` call

## Sample Data

The `data/` folder includes two simplified, public-domain-style sample contracts (an NDA and a Service Agreement) for demo purposes. Replace with your own documents for real use.

## License

MIT
