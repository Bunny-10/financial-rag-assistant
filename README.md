---
title: Financial RAG Assistant
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Financial RAG Assistant

A retrieval-augmented generation (RAG) application for querying SEC financial filings using natural language. Ask questions about earnings, balance sheets, or risk factors and get grounded answers with source citations.

**Live demo:** [rbhuma-financial-rag-assistant.hf.space](https://rbhuma-financial-rag-assistant.hf.space)

---

## Features

- **Natural language Q&A** over SEC 10-Q filings
- **Source citations** — every answer links back to the page and chunk it came from
- **Persistent vectorstore** — ChromaDB index built from Apple's Q2 FY2026 10-Q
- **Streamlit chat UI** with conversation history

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB (persisted) |
| Orchestration | LangChain (LCEL) |
| PDF parsing | PyPDF |
| SEC data | `sec-downloader` + EDGAR API |
| UI | Streamlit |

## Project Structure

```
financial-rag-assistant/
├── src/
│   ├── app.py            # Streamlit chat interface
│   ├── ingest.py         # PDF → chunks → embeddings → ChromaDB
│   └── download_sec.py   # Download 10-Q filings from SEC EDGAR
├── data/                 # PDF filings
├── chroma_db/            # Persisted vector index
├── Dockerfile            # For Hugging Face Spaces deployment
└── requirements.txt
```

## Getting Started

### 1. Clone and set up environment

```bash
git clone https://github.com/Bunny-10/financial-rag-assistant.git
cd financial-rag-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

### 3. Download a filing and ingest it

```bash
# Download Apple's latest 10-Q from SEC EDGAR
python src/download_sec.py

# Embed and store in ChromaDB
python src/ingest.py
```

### 4. Run the app

```bash
streamlit run src/app.py
```

## Deployment

The app is deployed to [Hugging Face Spaces](https://huggingface.co/spaces/Rbhuma/financial-rag-assistant) via Docker. The `OPENAI_API_KEY` is stored as a Space Secret and never committed to the repository.

## Security

- `.env` is listed in `.gitignore` and never committed
- API keys are injected at runtime via environment variables
