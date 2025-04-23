import argparse
import os
import shutil
from uuid import uuid4
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

# Load environment variables
load_dotenv()

# Validate environment variables
DATA_PATH = os.getenv("DATA_PATH")
CHROMA_PATH = os.getenv("CHROMA_PATH")

if not DATA_PATH or not CHROMA_PATH:
    raise ValueError("DATA_PATH and CHROMA_PATH must be set in the .env file")

def main():
    parser = argparse.ArgumentParser(description="Manage RAG pipeline database")
    parser.add_argument("--reset", action="store_true", help="Reset the database")
    args = parser.parse_args()

    try:
        if args.reset:
            print("✨ Clearing Database")
            clear_database()

        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(chunks)
    except Exception as e:
        print(f"❌ Error in main pipeline: {str(e)}")
        raise

def load_documents() -> List[Document]:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data directory {DATA_PATH} does not exist")
    
    try:
        document_loader = PyPDFDirectoryLoader(DATA_PATH)
        documents = document_loader.load()
        if not documents:
            print("⚠️ No documents found in the directory")
        return documents
    except Exception as e:
        raise RuntimeError(f"Failed to load documents: {str(e)}")

def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: List[Document]):
    try:
        # Initialize Chroma database
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function(),
        )

        # Calculate chunk IDs
        chunks_with_ids = calculate_chunk_ids(chunks)

        # Get existing documents
        existing_items = db.get(include=[])  # IDs are included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Filter new chunks
        new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

        if new_chunks:
            print(f"👉 Adding new documents: {len(new_chunks)}")
            db.add_documents(new_chunks)
        else:
            print("✅ No new documents to add")

    except Exception as e:
        raise RuntimeError(f"Failed to add documents to Chroma: {str(e)}")

def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
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

def clear_database():
    try:
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            print(f"🗑️ Database at {CHROMA_PATH} cleared")
        else:
            print(f"ℹ️ No database found at {CHROMA_PATH}")
    except Exception as e:
        raise RuntimeError(f"Failed to clear database: {str(e)}")

if __name__ == "__main__":
    main()