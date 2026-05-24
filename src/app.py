import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

sys.path.insert(0, str(Path(__file__).parent))
from ingest import get_registry, ingest_ticker, is_ingested, load_vectorstore
from sp500 import display_options, get_sp500, parse_selection

load_dotenv(override=True)

SYSTEM_PROMPT = """You are a financial analyst assistant. Use the retrieved context \
from SEC filings to answer questions accurately and concisely. Cite specific figures \
when available. If the answer is not in the context, say so — do not fabricate data.

Context:
{context}"""


def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource
def build_chain(collection_name: str):
    """Build and cache a RAG chain for a given ChromaDB collection."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=str(Path(__file__).resolve().parent.parent / "chroma_db"),
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    answer_chain = (
        {"context": lambda x: format_docs(x["context"]), "input": lambda x: x["input"]}
        | prompt | llm | StrOutputParser()
    )
    return RunnableParallel(
        {"context": retriever, "input": RunnablePassthrough()}
    ).assign(answer=answer_chain)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — company selector
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📈 Financial RAG Assistant")
    st.caption("Powered by SEC EDGAR · LangChain · ChromaDB · OpenAI")
    st.divider()

    with st.spinner("Loading S&P 500 list..."):
        companies = get_sp500()

    options = display_options(companies)

    # Default to Apple if it's in the list
    default_label = next(
        (o for o in options if "(AAPL)" in o),
        options[0],
    )
    default_idx = options.index(default_label)

    selected = st.selectbox(
        "Company",
        options,
        index=default_idx,
        help="Search by company name or ticker",
    )
    ticker, company_name = parse_selection(selected)

    form_type = st.radio(
        "Filing type",
        ["10-Q", "10-K"],
        horizontal=True,
        help="10-Q = latest quarterly report · 10-K = latest annual report",
    )

    st.divider()

    # Registry status
    registry = get_registry()
    from ingest import collection_name as col_name_fn
    col = col_name_fn(ticker, form_type)
    already_ingested = col in registry

    if already_ingested:
        entry = registry[col]
        st.success(f"**Ingested**")
        st.caption(
            f"Period: {entry.get('report_date', '?')}  \n"
            f"Filed: {entry.get('filing_date', '?')}  \n"
            f"Chunks: {entry.get('chunks', '?')}"
        )
    else:
        st.info(f"Not yet ingested")

    st.divider()
    st.caption("All ingested companies")
    if registry:
        for k, v in registry.items():
            st.caption(f"• {v.get('company_name', k)} ({v.get('form_type', '?')})")
    else:
        st.caption("None yet")

# ---------------------------------------------------------------------------
# Detect company switch → reset chat
# ---------------------------------------------------------------------------

company_key = f"{ticker}_{form_type}"
if st.session_state.get("company_key") != company_key:
    st.session_state.company_key = company_key
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.header(f"{company_name} — {form_type}")

if not already_ingested:
    st.info(
        f"**{company_name} ({ticker})** has not been ingested yet.\n\n"
        f"Click below to download the latest {form_type} from SEC EDGAR and build the index. "
        f"This takes about 30–60 seconds."
    )
    if st.button(f"⬇️ Ingest {ticker} {form_type}", type="primary"):
        progress_lines = st.empty()
        log: list[str] = []

        def update_progress(msg: str):
            log.append(msg)
            progress_lines.markdown("\n\n".join(f"• {l}" for l in log))

        try:
            with st.spinner(f"Ingesting {ticker} {form_type}..."):
                ingest_ticker(ticker, form_type, progress_cb=update_progress)
            st.success("Ingestion complete! You can now ask questions.")
            st.rerun()
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
else:
    # Chat interface
    entry = registry[col]
    st.caption(
        f"Filing period: **{entry.get('report_date', '?')}**  ·  "
        f"Filed: **{entry.get('filing_date', '?')}**  ·  "
        f"{entry.get('chunks', '?')} indexed chunks"
    )

    if st.session_state.messages:
        if st.button("Clear chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for doc in msg["sources"]:
                        page = doc.metadata.get("page", "?")
                        source = doc.metadata.get("source", "")
                        st.markdown(f"**Page {page + 1}** — `{Path(source).name}`")
                        st.caption(doc.page_content[:300] + "...")

    query = st.chat_input(f"Ask about {company_name}'s {form_type}...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        chain = build_chain(col)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = chain.invoke(query)
            answer = result["answer"]
            sources = result.get("context", [])
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for doc in sources:
                        page = doc.metadata.get("page", "?")
                        source = doc.metadata.get("source", "")
                        st.markdown(f"**Page {page + 1}** — `{Path(source).name}`")
                        st.caption(doc.page_content[:300] + "...")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
