import os
from pathlib import Path
import tempfile

def save_uploaded_file(uploaded_file) -> str:
    """Save a Streamlit uploaded file to a temp path and return the path."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

def get_file_icon(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    icons = {
        ".pdf": "📄",
        ".docx": "📝",
        ".txt": "📃",
        ".md": "🗒️",
    }
    return icons.get(ext, "📎")

def format_source_docs(source_docs: list) -> list[dict]:
    """Format retrieved source documents for display."""
    sources = []
    seen = set()
    for doc in source_docs:
        meta = doc.metadata
        name = meta.get("source_name", "Unknown")
        page = meta.get("page", None)
        key = f"{name}_{page}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "name": name,
                "page": page,
                "snippet": doc.page_content[:200].strip() + "..."
            })
    return sources