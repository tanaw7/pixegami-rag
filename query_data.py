import argparse
import os
from typing import Tuple, List
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from get_embedding_function import get_embedding_function
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Validate environment variables
CHROMA_PATH = os.getenv("CHROMA_PATH")
if not CHROMA_PATH:
    raise ValueError("CHROMA_PATH environment variable is not set")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline")
    parser.add_argument("query_text", type=str, help="The query text")
    args = parser.parse_args()
    query_text = args.query_text

    try:
        response = query_rag(query_text)
        print(response)
    except Exception as e:
        print(f"❌ Error during query: {str(e)}")
        raise

def query_rag(query_text: str) -> str:
    try:
        # Prepare the DB
        if not os.path.exists(CHROMA_PATH):
            raise FileNotFoundError(f"Chroma database not found at {CHROMA_PATH}")

        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB
        results: List[Tuple[Document, float]] = db.similarity_search_with_score(query_text, k=5)
        if not results:
            return "No relevant documents found."

        # Prepare context for the prompt
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Initialize OpenAI model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        model = ChatOpenAI(
            api_key=api_key,
            model="gpt-3.5-turbo",
            temperature=0.0
        )
        response_text = model.invoke(prompt).content

        # Format output with chunks first, then response
        formatted_output = ""
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("id", "Unknown")
            content = doc.page_content
            formatted_output += f"Chunk {i}:\nSource: {source}\nScore: {score:.2f}\nContent: {content}\n\n"
        formatted_output += f"Response: {response_text}"

        return formatted_output

    except Exception as e:
        raise RuntimeError(f"Failed to query RAG pipeline: {str(e)}")

if __name__ == "__main__":
    main()