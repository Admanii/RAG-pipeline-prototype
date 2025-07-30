# LangChain RAG Example

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** pipeline using [LangChain](https://www.langchain.com/), **Pinecone** for vector search, and **LLMs** like OpenAI or GitHub-hosted models.

## Components

- **Pinecone-Create-Index.py** – Creates a vector index in Pinecone.
- **Pinecone-Delete-Index.py** – Deletes the Pinecone index.
- **LangChain-Embedding-From-Dir.py** – Converts text files into embeddings and stores them in Pinecone.
- **LangChain-RAG-OpenAI.py** – Runs a RAG workflow using OpenAI's LLM.
- **LangChain-RAG-Github.py** – Runs a RAG workflow using GitHub-hosted LLMs via Azure-compatible endpoints.

## ⚙️ Setup

1. Clone the repo and create a `.env` file from the example:

```bash
cp .env.example .env
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Get API keys/Tokens for:
    - OpenAI
    - Pinecone
    - GitHub-hosted LLM (if using LangChain-RAG-Github.py)

4. Run scripts in order:
```bash
python Pinecone-Create-Index.py
python LangChain-Embedding-From-Dir.py
python LangChain-RAG-OpenAI.py  # or LangChain-RAG-Github.py
```

5. Notes: 
    - faq.txt is just used as an example, you may try adding multiple files to witness RAG retrival
    - Feel free to extend or swap the LLM backend as needed!