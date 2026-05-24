---
title: Financial RAG Assistant
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Financial RAG Assistant

A retrieval-augmented generation (RAG) application for querying SEC financial filings using natural language. Select any S&P 500 company, auto-ingest their latest 10-Q or 10-K from EDGAR, and ask questions about revenue, earnings, risk factors, and more — with source citations back to the filing.

**Live demo:** [rbhuma-financial-rag-assistant.hf.space](https://rbhuma-financial-rag-assistant.hf.space)

---

## Features

- **All 503 S&P 500 companies** — searchable dropdown, live list from Wikipedia
- **Auto-ingest on demand** — select any company, click Ingest, and the app downloads the latest filing from SEC EDGAR, extracts text (including financial tables), embeds it, and stores it in ChromaDB
- **Per-company collections** — each company gets its own named ChromaDB collection; switching companies is instant with no re-embedding
- **10-Q and 10-K support** — toggle between quarterly and annual reports per company
- **Table-aware extraction** — financial statement rows are extracted with label and values on the same line, so revenue and EPS figures are actually retrievable
- **Source citations** — every answer shows the page number and chunk it came from
- **Streamlit chat UI** with conversation history; resets cleanly on company switch

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB (named collections, persisted) |
| Orchestration | LangChain LCEL |
| PDF parsing | PyPDF |
| SEC data | `sec-downloader` + EDGAR API |
| S&P 500 list | Wikipedia (live) with top-100 fallback |
| UI | Streamlit |

## Project Structure

```
financial-rag-assistant/
├── src/
│   ├── app.py            # Streamlit UI — company selector, auto-ingest, chat
│   ├── sp500.py          # S&P 500 ticker/company directory (Wikipedia + fallback)
│   ├── edgar.py          # Download any ticker's 10-Q/10-K from EDGAR → PDF
│   ├── ingest.py         # PDF → chunks → embeddings → named ChromaDB collection
│   └── download_sec.py   # Legacy single-ticker download script
├── data/                 # Downloaded PDF filings
├── chroma_db/            # Persisted vector index (one collection per company)
├── Dockerfile            # Hugging Face Spaces deployment
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

### 3. Run the app

```bash
streamlit run src/app.py
```

Select a company in the sidebar, choose 10-Q or 10-K, and click **Ingest** — the app handles the rest.

### 4. Ingest via CLI (optional)

```bash
python src/ingest.py MSFT 10-Q
python src/ingest.py NVDA 10-K
```

## Deployment

Deployed to [Hugging Face Spaces](https://huggingface.co/spaces/Rbhuma/financial-rag-assistant) via Docker. Pre-loaded with Apple, Microsoft, and Meta 10-Qs. The `OPENAI_API_KEY` is stored as a Space Secret and never committed to the repository.

> **Note:** The HF Spaces filesystem is ephemeral — companies ingested during a session are available until the Space restarts. For persistent multi-user ingestion, use HF Spaces Persistent Storage or an external vector database.

## Security

- `.env` is in `.gitignore` and never committed
- API keys are injected at runtime via environment variables
