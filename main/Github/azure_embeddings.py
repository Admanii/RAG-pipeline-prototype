# azure_embeddings.py
import os
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from langchain_core.embeddings import Embeddings

class AzureCustomEmbeddings(Embeddings):
    def __init__(self, endpoint: str, model: str, token: str):
        self.client = EmbeddingsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(token)
        )
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embed(input=[text], model=self.model)
        return response.data[0].embedding
