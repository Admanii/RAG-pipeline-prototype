import os
from dotenv import load_dotenv
from azure_embeddings import AzureCustomEmbeddings  # Import your custom wrapper


from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential


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

# Connect to the Pinecone index using LangChain's Pinecone wrapper
pinecone_index_name = "langchain-embeddings-demo"
vector_store = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)

# Define the retrieval mechanism
retriever = vector_store.as_retriever(search_kwargs={"k": 1})  # Retrieve top-1 relevant documents

# Initialize GPT-4 with OpenAI
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"
token = os.environ["GITHUB_TOKEN"]

llm = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

# # Retrieve documents
query = input("Enter your question: ")
docs = retriever.invoke(query)
context = "\n\n".join([doc.page_content for doc in docs])
# print("Retrieved context:", context)

response = llm.complete(
    messages=[
        SystemMessage(context),
        UserMessage(query),
    ],
    temperature=1.0,
    top_p=1.0,
    model=model
)

print('Answer: ', response.choices[0].message.content)