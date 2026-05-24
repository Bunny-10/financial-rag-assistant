import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(override=True)

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

SYSTEM_PROMPT = """You are a financial analyst assistant. Use the retrieved context \
from SEC filings to answer questions accurately and concisely. If the answer is not \
in the context, say so — do not fabricate figures or facts.

Context:
{context}"""


def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource(show_spinner="Loading vectorstore...")
def load_chain():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    answer_chain = (
        {"context": lambda x: format_docs(x["context"]), "input": lambda x: x["input"]}
        | prompt
        | llm
        | StrOutputParser()
    )

    return (
        RunnableParallel({"context": retriever, "input": RunnablePassthrough()})
        .assign(answer=answer_chain)
    )


st.set_page_config(page_title="Financial RAG Assistant", page_icon="📈", layout="centered")
st.title("📈 Financial RAG Assistant")
st.caption("Ask questions about Apple's 10-Q (Q2 FY2026, period ended March 28, 2026)")

if "messages" in st.session_state and st.session_state.messages:
    if st.button("Clear chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for doc in msg["sources"]:
                    page = doc.metadata.get("page", "?")
                    source = doc.metadata.get("source", "unknown")
                    st.markdown(f"**Page {page + 1}** — `{Path(source).name}`")
                    st.caption(doc.page_content[:300] + "...")

query = st.chat_input("Ask a question about the filing...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    chain = load_chain()

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
                    source = doc.metadata.get("source", "unknown")
                    st.markdown(f"**Page {page + 1}** — `{Path(source).name}`")
                    st.caption(doc.page_content[:300] + "...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
