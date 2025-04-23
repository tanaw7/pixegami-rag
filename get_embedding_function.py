from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def get_embedding_function():
    """
    Returns an OpenAI embedding function for use with LangChain and Chroma.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    try:
        embeddings = OpenAIEmbeddings(
            api_key=api_key,
            model="text-embedding-ada-002"
        )
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI embeddings: {str(e)}")