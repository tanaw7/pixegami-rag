import os
import shutil
import pytest
import warnings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from query_data import query_rag
from get_embedding_function import get_embedding_function
from dotenv import load_dotenv

# Suppress Pydantic deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()

# Validate environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PATH = os.getenv("CHROMA_PATH")
if not API_KEY or not CHROMA_PATH:
    raise ValueError("OPENAI_API_KEY and CHROMA_PATH must be set in .env")

# Evaluation prompt for semantic similarity
EVAL_PROMPT = """
You are an expert evaluator. Compare the expected response with the actual response to determine if they convey the same core meaning or key information, even if worded differently or if the actual response includes additional details. Ignore minor differences in phrasing, formatting, or extra information as long as the main idea is preserved.

Expected Response: {expected_response}
Actual Response: {actual_response}

Return only 'true' or 'false' to indicate if the actual response is semantically equivalent to the expected response.
"""

# Helper function to extract response from query_rag output
def extract_response(output: str) -> str:
    """Extract the response text after 'Response: ' from query_rag output."""
    if "Response: " in output:
        return output.split("Response: ", 1)[1].strip()
    return output.strip()

# Helper function to compute embedding-based similarity
def embedding_similarity(expected: str, actual: str) -> float:
    """Compute cosine similarity between two texts using OpenAI embeddings."""
    embeddings = OpenAIEmbeddings(api_key=API_KEY, model="text-embedding-ada-002")
    expected_emb = embeddings.embed_query(expected)
    actual_emb = embeddings.embed_query(actual)
    dot_product = sum(a * b for a, b in zip(expected_emb, actual_emb))
    norm_expected = (sum(a * a for a in expected_emb) ** 0.5)
    norm_actual = (sum(b * b for b in actual_emb) ** 0.5)
    return dot_product / (norm_expected * norm_actual) if norm_expected * norm_actual != 0 else 0

# Helper function to evaluate semantic similarity
def evaluate_response(question: str, expected_response: str, actual_response: str, retries: int = 5) -> bool:
    """Evaluate if actual_response is semantically equivalent to expected_response."""
    model = ChatOpenAI(api_key=API_KEY, model="gpt-4o-mini", temperature=0.0)
    
    # Try LLM evaluation
    for _ in range(retries):
        prompt = EVAL_PROMPT.format(expected_response=expected_response, actual_response=actual_response)
        result = model.invoke(prompt).content.strip().lower()
        if result in ("true", "false"):
            return result == "true"
    
    # Fallback to embedding-based similarity
    similarity = embedding_similarity(expected_response, actual_response)
    return similarity > 0.85  # Threshold for semantic equivalence

@pytest.fixture(scope="module")
def chroma_db(tmp_path_factory):
    """Set up a temporary Chroma database with sample Monopoly and Ticket to Ride documents."""
    temp_dir = tmp_path_factory.mktemp("chroma_test")
    temp_chroma_path = str(temp_dir / "chroma_db")

    # Sample documents
    monopoly_doc = Document(
        page_content="Each player starts with $1500 in Monopoly. Properties can be bought when landing on an unowned property at its printed price. If a player declines to buy, the property is auctioned to the highest bidder.",
        metadata={"source": "data/monopoly.pdf", "page": 1, "id": "data/monopoly.pdf:1:0"}
    )
    ticket_doc = Document(
        page_content="In Ticket to Ride, the longest continuous train scores 10 points.",
        metadata={"source": "data/ticket_to_ride.pdf", "page": 1, "id": "data/ticket_to_ride.pdf:1:0"}
    )

    # Initialize Chroma database
    db = Chroma(
        persist_directory=temp_chroma_path,
        embedding_function=get_embedding_function()
    )
    db.add_documents([monopoly_doc, ticket_doc])

    yield temp_chroma_path

    # Cleanup
    del db  # Release Chroma object
    if os.path.exists(temp_chroma_path):
        for _ in range(3):  # Retry cleanup
            try:
                shutil.rmtree(temp_chroma_path)
                break
            except PermissionError:
                import time
                time.sleep(0.1)

def test_monopoly_starting_money(chroma_db, monkeypatch):
    """Test querying the starting money in Monopoly."""
    monkeypatch.setenv("CHROMA_PATH", chroma_db)
    question = "How much total money does a player start with in Monopoly? (Answer with the number only)"
    expected_response = "$1500"
    response = extract_response(query_rag(question))
    assert evaluate_response(question, expected_response, response), f"Expected response similar to '{expected_response}', got '{response}'"

def test_monopoly_property_buying(chroma_db, monkeypatch):
    """Test querying property buying rules in Monopoly."""
    monkeypatch.setenv("CHROMA_PATH", chroma_db)
    question = "How can a player buy a property in Monopoly?"
    expected_response = "A player can buy a property by landing on it and paying the printed price."
    response = extract_response(query_rag(question))
    assert evaluate_response(question, expected_response, response), f"Expected response similar to '{expected_response}', got '{response}'"

def test_ticket_to_ride_longest_train(chroma_db, monkeypatch):
    """Test querying the longest train points in Ticket to Ride."""
    monkeypatch.setenv("CHROMA_PATH", chroma_db)
    question = "How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)"
    expected_response = "10"
    response = extract_response(query_rag(question))
    assert evaluate_response(question, expected_response, response), f"Expected response similar to '{expected_response}', got '{response}'"

def test_empty_query(chroma_db, monkeypatch):
    """Test handling of an empty query."""
    monkeypatch.setenv("CHROMA_PATH", chroma_db)
    with pytest.raises(RuntimeError, match="Query cannot be empty"):
        query_rag("")