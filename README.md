# RAG Pipeline for PDF Querying

This project implements a Retrieval-Augmented Generation (RAG) pipeline to query PDF documents using OpenAI's embeddings and chat models, with Chroma as the vector database. It is designed to load PDFs, split them into chunks, store them in a vector database, and answer queries based on the document content. The example use case is querying Monopoly game rules, but the pipeline is adaptable to any PDF collection.

## Features

- **PDF Processing**: Loads and splits PDF documents into manageable chunks.
- **Vector Storage**: Stores document chunks in a Chroma vector database with OpenAI embeddings.
- **Querying**: Retrieves relevant chunks and generates answers using OpenAI's.
- **Customizable Output**: Displays retrieved chunks, their sources, similarity scores, and the generated response.
- **Robust Error Handling**: Validates inputs and handles errors gracefully.

## Project Structure

- `populate_database.py`: Loads PDFs, splits them into chunks, and stores them in Chroma.
- `query_data.py`: Queries the Chroma database and generates answers with detailed output.
- `get_embedding_function.py`: Defines the OpenAI embedding function for vectorization.
- `.env`: Configuration file for environment variables (e.g., API keys, paths).
- `requirements.txt`: Lists project dependencies.

## Prerequisites

- Python 3.8+
- An OpenAI API key (obtain from OpenAI)
- PDF documents to process (e.g., Monopoly rules PDF)

## Installation

1. **Clone the Repository**:

   ```bash
   git clone <your-repository-url>
   cd <repository-name>
   ```

2. **Create a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**: Create a `.env` file in the project root with the following:

   ```plaintext
   OPENAI_API_KEY=your_openai_api_key_here
   DATA_PATH=/path/to/your/pdf/folder
   CHROMA_PATH=/path/to/chroma/db
   ```

   - Replace `/path/to/your/pdf/folder` with the directory containing your PDFs.
   - Replace `/path/to/chroma/db` with the directory for the Chroma database.

## Usage

1. **Populate the Database**: Load PDFs into the Chroma database:

   ```bash
   python populate_database.py
   ```

   To reset the database and reload PDFs:

   ```bash
   python populate_database.py --reset
   ```

2. **Query the Database**: Ask questions about the PDFs:

   ```bash
   python query_data.py "How to buy a property in Monopoly?"
   ```

   Example output:

   ```
   Chunk 1:
   Source: data/monopoly.pdf:1:2
   Score: 0.85
   Content: When a player lands on an unowned property they may buy that property from the Bank at its printed price...
   
   Chunk 2:
   Source: data/monopoly.pdf:0:0
   Score: 0.78
   Content: The object of the game is to become the wealthiest player through buying, renting, and selling property...
   
   Response: To buy a property in Monopoly, a player must land on an unowned property and purchase it from the Bank at its printed price...
   ```

## Testing
To run unit tests
```
pytest
```
For more info (query, expected response, response, etc.) in test_results.log
```
pytest test_rag.py -s --log-cli-level=INFO
```

## Configuration

- **Embedding Model**: Defined in `get_embedding_function.py` (default: `text-embedding-ada-002`).
- **Chat Model**: Configured in `query_data.py` (default: `gpt-3.5-turbo`).
- **Retrieval Parameters**: Adjust `k` (number of chunks retrieved) in `query_data.py` (default: 5).
- **Chunking**: Modify `chunk_size` and `chunk_overlap` in `populate_database.py` (default: 800, 80).

See the code comments and documentation for advanced configuration options (e.g., switching to MMR retrieval or using `gpt-4o-mini`).

## Dependencies

Listed in `requirements.txt`:

```
langchain
langchain-community
langchain-chroma
langchain-openai
chromadb
pypdf
pytest
openai
python-dotenv
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Built with LangChain, Chroma, and OpenAI.
- Inspired by Pixegami's RAG tutorial.