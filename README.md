🧠 MULTI-DOC-RAG: Intelligent Multi-Document Chat Assistant

🚀 Live Project 

📌 Overview

MULTI-DOC-RAG is an AI-powered Retrieval-Augmented Generation (RAG) system that allows users to upload multiple documents (PDF, DOCX, TXT, MD) and interact with them using natural language.

It enables users to:

Upload multiple documents
Ask context-aware questions
Get grounded answers from relevant documents only
View source references for transparency

This project demonstrates real-world use of:

LLMs (Groq / LLaMA3)
Vector databases (ChromaDB)
Embeddings
LangChain RAG pipelines

Streamlit UI
⚙️ Features
📂 Multi-document upload support
🧠 Context-aware question answering (RAG)
🔎 Source tracking with document citations
💬 Chat-style interface
⚡ Fast inference using Groq API
🗂 Persistent vector storage using ChromaDB
🎨 Clean dark-themed UI
🧾 Supports PDF, DOCX, TXT, MD

🏗️ System Architecture
                ┌────────────────────┐
                │  User Upload Docs  │
                └─────────┬──────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │ Document Loader (LangChain)
             └─────────┬──────────────┘
                       │
                       ▼
             ┌──────────────────────────┐
             │ Text Chunking & Splitting│
             └─────────┬──────────────┘
                       │
                       ▼
             ┌──────────────────────────┐
             │ Embeddings (Groq / HF)   │
             └─────────┬──────────────┘
                       │
                       ▼
             ┌──────────────────────────┐
             │   Chroma Vector Store     │
             └─────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
User Query                    Retriever (Top-K)
        ▼                             ▼
        └──────────► LLM (LLaMA3) ◄───┘
                       │
                       ▼
               Final Answer + Sources
               
🖥️ Tech Stack
🧠 AI / ML
LangChain
Groq API (LLaMA3)
HuggingFace Embeddings
ChromaDB (Vector Database)

🌐 Backend / App
Python 3.10+
Streamlit
FastAPI (optional backend layer)

☁️ Deployment
AWS EC2 (Ubuntu)
Nginx (optional reverse proxy)
GitHub

📁 Project Structure
MULTI-DOC-RAG/
│
├── app.py                 # Streamlit frontend
├── rag_engine.py         # RAG pipeline logic
├── utils.py              # Helper functions
├── requirements.txt      # Dependencies
├── .env                  # API keys (not pushed)
│
├── chroma_store/         # Vector DB (ignored in git)
└── README.md

🚀 How to Run Locally

1️⃣ Clone repo
git clone https://github.com/your-username/MULTI-DOC-RAG.git
cd MULTI-DOC-RAG

2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Setup environment variables

Create .env file:

GROQ_API_KEY=your_api_key_here
5️⃣ Run application
streamlit run app.py
