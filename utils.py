# utils.py
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import shutil

def load_documents(data_path: str) -> List[Document]:
    """Load PDF documents from the specified directory."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory {data_path} does not exist")
    try:
        document_loader = PyPDFDirectoryLoader(data_path)
        documents = document_loader.load()
        if not documents:
            print(f"⚠️ No documents found in {data_path}")
        return documents
    except Exception as e:
        raise RuntimeError(f"Failed to load documents: {str(e)}")

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    """Assign unique IDs to document chunks based on source and page."""
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        last_page_id = current_page_id
    return chunks

def clear_database(chroma_path: str):
    """Clear the Chroma database at the specified path."""
    try:
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
            print(f"🗑️ Database at {chroma_path} cleared")
        else:
            print(f"ℹ️ No database found at {chroma_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to clear database: {str(e)}")