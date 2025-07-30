import os
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Load API keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
print("Got Keys")

# Optional: also set as env var for LangChain compatibility
os.environ["OPENAI_API_KEY"] = openai_api_key
print("os.environ['OPENAI_API_KEY'] set")

# # Initialize Pinecone
# pc = Pinecone(api_key=pinecone_api_key)
# print("pc initialized")
# index_name = "rag-basic"
# print("index_name set to rag-basic")

# # Create index if it doesn't exist
# print("Creating index if it doesn't exist")
# if index_name not in pc.list_indexes().names():
#     print(f"Creating index: {index_name}")
#     pc.create_index(
#         name=index_name,
#         dimension=1024,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )
# else:
#     print(f"Index {index_name} already exists")

# # Connect to the index
# print("Connecting to the index")
# index = pc.Index(index_name)


# Load documents
print("Loading documents")
loader = TextLoader("sample_docs/faq.txt")  # <- Make sure this file exists
docs = loader.load()

# Chunk the documents
print("Chunking documents")
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Generate embeddings
print("Generating embeddings")
embedding_model = OpenAIEmbeddings()
print("Embeddings generated", embedding_model)
# Upload chunks into Pinecone index using LangChain wrapper
print("Uploading chunks to Pinecone index")
vectorstore = LangchainPinecone.from_documents(
    documents=chunks,
    embedding=embedding_model,
    index_name=index_name
)

# Create retriever and chain
print("Creating retriever and QA chain")
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0, model_name="gpt-4"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Query
print("Querying the chain")
query = "What is the refund policy mentioned in the document?"
result = qa_chain(query)

# Output
print("Answer:", result['result'])
print("\nSources:")
for doc in result['source_documents']:
    print(doc.page_content)
