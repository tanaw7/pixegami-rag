import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

load_dotenv()

def load_documents():
    document_loader = PyPDFDirectoryLoader(os.getenv('DATA_PATH'))
    return document_loader.load()



docs = load_documents()

# for i, item in enumerate(docs):
#     print(docs[i])
#     if i > 10:
#         break