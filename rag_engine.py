import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# =========================
# EMBEDDINGS
# =========================
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

# =========================
# DOCUMENT LOADER
# =========================
def load_document(file_path: str, source_name: str) -> list[Document]:
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()

    for doc in docs:
        doc.metadata["source_name"] = source_name

    return docs

# =========================
# VECTOR STORE (FIXED)
# =========================
def build_vectorstore(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)
    embeddings = get_embeddings()

    # ✅ IMPORTANT FIX:
    # No PersistentClient, no disk usage, works on Streamlit Cloud
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore

# =========================
# LOAD EXISTING (SAFE)
# =========================
def load_existing_vectorstore():
    # ❌ Disabled for Streamlit Cloud stability
    # (prevents Chroma tenant/database errors)
    return None

# =========================
# RAG CHAIN
# =========================
def build_rag_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a helpful assistant that answers questions based only on the provided documents.

If the answer is not present in the context, say: "I couldn't find this in the uploaded documents."

Be concise and accurate.

Context:
{context}
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever