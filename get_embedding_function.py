from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings

# Make sure to use the same embedding for vectorizing the documents and the query.
def get_embedding_function():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return embeddings

def get_bedrock_embedding_function():
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default", region_name="us-east-1"
    )
    return embeddings