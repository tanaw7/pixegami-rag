import logging
import os
import shutil
import pytest
import warnings
import re
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from query_data import query_rag
from get_embedding_function import get_embedding_function
from utils import load_documents, split_documents, calculate_chunk_ids
from dotenv import load_dotenv
import time

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # run - pytest test_rag_4.py -s --log-cli-level=INFO
handler = logging.StreamHandler()
logger.addHandler(handler)

# Suppress Pydantic warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()

# Validate environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PATH = os.getenv("CHROMA_PATH")
DATA_PATH = os.getenv("DATA_PATH")
if not API_KEY or not CHROMA_PATH or not DATA_PATH:
    raise ValueError("OPENAI_API_KEY, CHROMA_PATH, and DATA_PATH must be set in .env")

# Evaluation prompt
EVAL_PROMPT = """
Compare the expected and actual responses to check if they convey the same core meaning, even if worded differently or with extra details. Ignore minor phrasing differences.

Expected Response: {expected_response}
Actual Response: {actual_response}

Return 'true' or 'false'.
"""

# Extract response
def extract_response(output: str) -> str:
    return output.split("Response: ", 1)[1].strip() if "Response: " in output else output.strip()

# Simple similarity check
def simple_similarity(expected: str, actual: str) -> bool:
    def normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return ' '.join(text.split())
    expected_norm = normalize(expected)
    actual_norm = normalize(actual)
    return expected_norm == actual_norm or expected_norm in actual_norm

# Evaluate response
def evaluate_response(expected_response: str, actual_response: str, retries: int = 5) -> bool:
    model = ChatOpenAI(api_key=API_KEY, model="gpt-4o-mini", temperature=0.0)
    for _ in range(retries):
        prompt = EVAL_PROMPT.format(expected_response=expected_response, actual_response=actual_response)
        result = model.invoke(prompt).content.strip().lower()
        if result in ("true", "false"):
            return result == "true"
    return simple_similarity(expected_response, actual_response)

# Helper function to run the test
def run_test(chroma_db, monkeypatch, question: str, expected_response: str):
    monkeypatch.setenv("CHROMA_PATH", chroma_db)
    response = extract_response(query_rag(question))
    logger.info(f"Question: {question}")
    logger.info(f"Expected: {expected_response}")
    logger.info(f"Actual: {response}")
    assert evaluate_response(expected_response, response), f"Expected '{expected_response}', got '{response}'"

@pytest.fixture(scope="module")
def chroma_db(tmp_path_factory):
    # Create temporary directories
    temp_dir = tmp_path_factory.mktemp("chroma_test")
    temp_chroma_path = str(temp_dir / "chroma_db")
    temp_data_path = str(temp_dir / "data")

    # Copy test PDFs to temp_data_path
    os.makedirs(temp_data_path, exist_ok=True)
    source_data_path = os.getenv("DATA_PATH")  # Or wherever your test PDFs are
    for pdf in ["monopoly.pdf", "ticket_to_ride.pdf"]:
        src = os.path.join(source_data_path, pdf)
        dst = os.path.join(temp_data_path, pdf)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            raise FileNotFoundError(f"PDF {src} not found.")

    # Use helper functions
    documents = load_documents(temp_data_path)
    chunks = split_documents(documents)
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Initialize and populate Chroma
    db = Chroma(
        persist_directory=temp_chroma_path,
        embedding_function=get_embedding_function()
    )
    db.add_documents(chunks_with_ids)

    yield temp_chroma_path

    # Cleanup
    del db  # Explicitly delete the Chroma object to release resources
    if os.path.exists(temp_chroma_path):
        for _ in range(3):
            try:
                shutil.rmtree(temp_chroma_path)
                break
            except PermissionError:
                time.sleep(0.1)

# Test functions using the helper
def test_monopoly_starting_money(chroma_db, monkeypatch):
    run_test(chroma_db, monkeypatch, "How much total money does a player start with in Monopoly? (Answer with the number only)", "$1500")

def test_monopoly_property_buying(chroma_db, monkeypatch):
    run_test(chroma_db, monkeypatch, "How can a player buy a property in Monopoly?", "A player can buy a property by landing on it and paying the printed price.")

def test_ticket_to_ride_longest_train(chroma_db, monkeypatch):
    run_test(chroma_db, monkeypatch, "How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)", "10")

def test_empty_query(chroma_db, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", chroma_db)
    with pytest.raises(RuntimeError, match="Query cannot be empty"):
        query_rag("")