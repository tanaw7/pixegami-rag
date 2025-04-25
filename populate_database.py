# populate_database.py
import argparse
import os
from dotenv import load_dotenv
from utils import load_documents, split_documents, calculate_chunk_ids, clear_database
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
            clear_database(CHROMA_PATH)

        documents = load_documents(DATA_PATH)
        chunks = split_documents(documents)
        add_to_chroma(chunks)
    except Exception as e:
        print(f"❌ Error in main pipeline: {str(e)}")
        raise

def add_to_chroma(chunks):
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

if __name__ == "__main__":
    main()