# main.py
import os
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from azure_embeddings import AzureCustomEmbeddings  # Import your custom wrapper

load_dotenv()

# Initialize Azure Embeddings
azure_endpoint = "https://models.github.ai/inference"
azure_model = "openai/text-embedding-3-large"
azure_token = os.environ["GITHUB_TOKEN"]

embeddings = AzureCustomEmbeddings(
    endpoint=azure_endpoint,
    model=azure_model,
    token=azure_token
)

# Load all text files
loader = TextLoader("../sample_docs/faq.txt")
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
split_documents = text_splitter.split_documents(documents)
print(f"Split documents:", split_documents)

# Store in Pinecone
pinecone_index_name = "langchain-embeddings-demo"
vectorstore = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)
vectorstore.add_documents(documents=split_documents)

print("Embeddings generated via Azure and stored in Pinecone!")
