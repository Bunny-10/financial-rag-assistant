"""
Ingest SEC filing PDFs into per-company ChromaDB collections.
Each (ticker, form_type) pair gets its own named collection so companies
can be switched without re-embedding.

Registry: chroma_db/registry.json tracks what has been ingested.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from edgar import download_filing

load_dotenv(override=True)

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
REGISTRY_PATH = CHROMA_DIR / "registry.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------------
# Collection naming
# ChromaDB names: 3-63 chars, lowercase alphanumeric + hyphens/underscores,
# must start and end with alphanumeric.
# ---------------------------------------------------------------------------

def collection_name(ticker: str, form_type: str) -> str:
    t = ticker.lower().replace("-", "_").replace(".", "_")
    f = form_type.lower().replace("-", "")
    return f"{t}_{f}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def _save_registry(reg: dict) -> None:
    CHROMA_DIR.mkdir(exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))


def is_ingested(ticker: str, form_type: str) -> bool:
    return collection_name(ticker, form_type) in _load_registry()


def get_registry() -> dict:
    return _load_registry()


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest_ticker(
    ticker: str,
    form_type: str = "10-Q",
    progress_cb=None,
) -> dict:
    """
    Download and embed a filing for ticker+form_type into its own ChromaDB collection.
    Returns the registry entry for this company.
    Skips embedding if already ingested; always returns the existing entry.
    """
    col_name = collection_name(ticker, form_type)
    registry = _load_registry()

    if col_name in registry:
        return registry[col_name]

    def _step(msg: str):
        if progress_cb:
            progress_cb(msg)

    # Download
    pdf_path, metadata = download_filing(ticker, form_type, progress_cb=_step)

    # Load + split
    _step("Loading and splitting PDF...")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    _step(f"Split into {len(chunks)} chunks.")

    # Embed into named collection
    _step("Embedding and storing in ChromaDB...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=col_name,
        persist_directory=str(CHROMA_DIR),
    )

    # Update registry
    entry = {**metadata, "collection": col_name, "chunks": len(chunks)}
    registry[col_name] = entry
    _save_registry(registry)
    _step(f"Done. {len(chunks)} chunks stored in collection '{col_name}'.")
    return entry


# ---------------------------------------------------------------------------
# Load existing vectorstore
# ---------------------------------------------------------------------------

def load_vectorstore(ticker: str, form_type: str) -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=collection_name(ticker, form_type),
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint (backward compat: ingest AAPL 10-Q by default)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    form_arg = sys.argv[2] if len(sys.argv) > 2 else "10-Q"
    ingest_ticker(ticker_arg, form_arg, progress_cb=print)
