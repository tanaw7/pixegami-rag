import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from langchain_text_splitters import TextSplitter

load_dotenv()

def load_documents():
    document_loader = PyPDFDirectoryLoader(os.getenv('DATA_PATH'))
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)



docs = load_documents()


### FUNzies

# for i, (key, value) in enumerate(TextSplitter.__init__.__annotations__.items()):
#     print(f"{key:<20} -> {value}")

# for i, item in enumerate(docs):
#     print(docs[i])
#     if i > 10:
#         break