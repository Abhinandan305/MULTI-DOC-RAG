import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import chromadb

load_dotenv()

CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "rag_collection"


# ─────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# ─────────────────────────────────────────────
# Document Loader
# ─────────────────────────────────────────────
def load_document(file_path: str, source_name: str) -> list[Document]:
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()

    for doc in docs:
        doc.metadata["source_name"] = source_name

    return docs


# ─────────────────────────────────────────────
# Vector Store (FIXED FOR STREAMLIT CLOUD)
# ─────────────────────────────────────────────
def build_vectorstore(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    texts = [d.page_content for d in chunks]
    metadatas = [d.metadata for d in chunks]

    embeddings = get_embeddings()

    # ✅ SAFE CHROMA CLIENT (fixes tenant error)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # embed documents
    embeddings_list = embeddings.embed_documents(texts)

    # clear old data (important for re-indexing)
    try:
        collection.delete()
    except:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    collection.add(
        documents=texts,
        embeddings=embeddings_list,
        metadatas=metadatas,
        ids=[str(i) for i in range(len(texts))]
    )

    # wrap for LangChain retrieval
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )


# ─────────────────────────────────────────────
# Load Existing Vector Store
# ─────────────────────────────────────────────
def load_existing_vectorstore():
    if not Path(CHROMA_DIR).exists():
        return None

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings()
    )


# ─────────────────────────────────────────────
# RAG Chain
# ─────────────────────────────────────────────
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
        ("system", """
You are a helpful assistant that answers strictly based on the provided context.

If the answer is not in the context, say you don't know.

Context:
{context}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever