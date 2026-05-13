import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from rag_engine import load_document, build_vectorstore, load_existing_vectorstore, build_rag_chain
from utils import save_uploaded_file, get_file_icon, format_source_docs

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background-color: #0d1117;
    color: #c9d1d9;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}

[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}

/* Hide default streamlit header */
[data-testid="stHeader"] { background: transparent; }

/* Title */
.app-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #58a6ff;
    letter-spacing: -0.5px;
    margin-bottom: 0;
    padding: 1rem 0 0.2rem 0;
}
.app-subtitle {
    font-size: 0.8rem;
    color: #6e7681;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 1.5rem;
}

/* Chat container */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 1rem 0;
}

/* User message */
.msg-user {
    display: flex;
    justify-content: flex-end;
}
.msg-user-bubble {
    background: #1f6feb;
    color: #ffffff;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 72%;
    font-size: 0.92rem;
    line-height: 1.55;
    box-shadow: 0 2px 8px rgba(31,111,235,0.3);
}

/* AI message */
.msg-ai {
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
    gap: 10px;
}
.msg-ai-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #21262d;
    border: 1px solid #30363d;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.msg-ai-bubble {
    background: #161b22;
    border: 1px solid #21262d;
    color: #c9d1d9;
    padding: 12px 16px;
    border-radius: 4px 18px 18px 18px;
    max-width: 75%;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* Source chips */
.source-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
    margin-left: 42px;
}
.source-chip {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
}

/* Doc card in sidebar */
.doc-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.82rem;
    color: #8b949e;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Status badge */
.badge-ok {
    background: #1a4731;
    color: #3fb950;
    border: 1px solid #2ea043;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    display: inline-block;
}
.badge-err {
    background: #3d1e1e;
    color: #f85149;
    border: 1px solid #da3633;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    display: inline-block;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #6e7681;
}
.empty-state h2 {
    font-family: 'IBM Plex Mono', monospace;
    color: #30363d;
    font-size: 2rem;
    margin-bottom: 0.5rem;
}
.empty-state p {
    font-size: 0.9rem;
    max-width: 360px;
    margin: 0 auto;
}

/* Buttons */
.stButton > button {
    background: #1f6feb !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    padding: 0.45rem 1rem !important;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #388bfd !important;
}

/* Clear button override */
.clear-btn > button {
    background: #21262d !important;
    color: #f85149 !important;
    border: 1px solid #30363d !important;
}
.clear-btn > button:hover {
    background: #3d1e1e !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0d1117;
    border: 1px dashed #30363d;
    border-radius: 8px;
    padding: 0.5rem;
}

/* Divider */
hr { border-color: #21262d !important; }

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 12px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_existing_vectorstore()
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="app-title">🧠 DocMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">multi-document RAG · groq + llama3</div>', unsafe_allow_html=True)

    # API status
    api_key = os.getenv("GROQ_API_KEY", "")
    if api_key and api_key != "your_groq_api_key_here":
        st.markdown('<span class="badge-ok">✓ Groq connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-err">✗ Missing GROQ_API_KEY in .env</span>', unsafe_allow_html=True)
        st.caption("Get a free key at [console.groq.com](https://console.groq.com)")

    st.markdown("---")
    st.markdown("**Upload Documents**")
    st.caption("PDF · DOCX · TXT · MD")

    uploaded_files = st.file_uploader(
        "Upload",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if st.button("⚡ Process & Index", disabled=not uploaded_files):
        with st.spinner("Embedding documents..."):
            all_docs = []
            names = []
            for uf in uploaded_files:
                tmp_path = save_uploaded_file(uf)
                docs = load_document(tmp_path, source_name=uf.name)
                all_docs.extend(docs)
                names.append(uf.name)
                os.unlink(tmp_path)

            st.session_state.vectorstore = build_vectorstore(all_docs)
            st.session_state.uploaded_docs = names
            st.session_state.rag_chain = None
            st.session_state.retriever = None
            st.session_state.messages = []
        st.success(f"Indexed {len(uploaded_files)} file(s)")

    # Indexed docs list
    if st.session_state.uploaded_docs:
        st.markdown("---")
        st.markdown("**Indexed Documents**")
        for name in st.session_state.uploaded_docs:
            st.markdown(
                f'<div class="doc-card">{get_file_icon(name)} {name}</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("🗑 Clear & Reset"):
        if os.path.exists("./chroma_store"):
            shutil.rmtree("./chroma_store")
        st.session_state.clear()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────
if not st.session_state.vectorstore:
    st.markdown("""
    <div class="empty-state">
        <h2>📂</h2>
        <h2>No documents loaded</h2>
        <p>Upload PDFs, DOCX, TXT or MD files in the sidebar, then click <strong>Process & Index</strong> to start chatting.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Build chain if needed
if st.session_state.rag_chain is None:
    with st.spinner("Initialising model..."):
        chain, retriever = build_rag_chain(st.session_state.vectorstore)
        st.session_state.rag_chain = chain
        st.session_state.retriever = retriever

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-user">
            <div class="msg-user-bubble">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-ai">
            <div class="msg-ai-avatar">🧠</div>
            <div class="msg-ai-bubble">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
        if msg.get("sources"):
            chips = "".join(
                f'<span class="source-chip">📄 {s["name"]}{" · p" + str(s["page"]) if s.get("page") is not None else ""}</span>'
                for s in msg["sources"][:4]
            )
            st.markdown(f'<div class="source-row">{chips}</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        history = []
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            else:
                history.append(AIMessage(content=msg["content"]))

        answer = st.session_state.rag_chain.invoke({
            "question": prompt,
            "chat_history": history
        })
        source_docs = st.session_state.retriever.invoke(prompt)
        sources = format_source_docs(source_docs)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    st.rerun()