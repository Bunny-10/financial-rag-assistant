import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_pdfs(data_dir: Path) -> list:
    docs = []
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")
    for pdf_path in pdf_files:
        print(f"Loading {pdf_path.name}...")
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    print(f"Loaded {len(docs)} pages from {len(pdf_files)} file(s).")
    return docs


def split_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def build_vectorstore(chunks: list) -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("Embedding chunks and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Vectorstore saved to {CHROMA_DIR}")
    return vectorstore


def ingest():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in .env")

    docs = load_pdfs(DATA_DIR)
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    print("Ingestion complete.")


if __name__ == "__main__":
    ingest()
