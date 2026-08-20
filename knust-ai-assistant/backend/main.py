from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import json
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# Global variables
chroma_client = None
collection = None
openai_client = None  # Will be configured for NVIDIA NIM

def load_openai_client():
    global openai_client
    api_key = os.getenv("META_NVIDIA_KEY")  # API key for build.nvidia.com
    base_url = os.getenv("base_url", "https://integrate.api.nvidia.com/v1")
    if not api_key:
        print("Warning: META_NVIDIA_KEY not set. Using dummy responses.")
        return None
    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )

def init_chromadb():
    global chroma_client, collection
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    # Get or create collection
    collection = chroma_client.get_or_create_collection(name="knust_docs")

def load_and_chunk_documents(data_dir: str = "../data"):
    """Load text files, split into chunks, compute embeddings, and store in ChromaDB."""
    global collection, openai_client
    # Ensure we have embedding client
    if openai_client is None:
        openai_client = load_openai_client()
        if openai_client is None:
            print("Warning: No LLM client, will not compute embeddings.")
            return

    # Get model name from environment
    model_name = os.getenv("model", "nvidia/nemotron-3-super-120b-a12b")
    
    # Clear existing collection (for simplicity in MVP)
    try:
        chroma_client.delete_collection(name="knust_docs")
    except:
        pass
    collection = chroma_client.get_or_create_collection(name="knust_docs")

    chunks = []
    metadatas = []
    ids = []
    idx = 0
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt") or filename.endswith(".md"):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                for para in paragraphs:
                    chunks.append(para)
                    metadatas.append({"source": f"{filename}#para{idx}"})
                    ids.append(f"{filename}_{idx}")
                    idx += 1
    if not chunks:
        chunks = ["Welcome to KNUST E-Learning Centre. How can I help you?"]
        metadatas = [{"source": "dummy"}]
        ids = ["dummy_0"]

    # Compute embeddings using NVIDIA model
    print(f"Computing embeddings for {len(chunks)} chunks using model {model_name}...")
    embeddings = []
    batch_size = 16
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        resp = openai_client.embeddings.create(
            model=model_name,
            input=batch
        )
        batch_emb = [item.embedding for item in resp.data]
        embeddings.extend(batch_emb)
    # Add to ChromaDB
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB.")

def retrieve_relevant_chunks(query: str, top_k: int = 3):
    global collection, openai_client
    if collection is None:
        return [], []
    if openai_client is None:
        openai_client = load_openai_client()
        if openai_client is None:
            return [], []
    # Get model name from environment
    model_name = os.getenv("model", "nvidia/nemotron-3-super-120b-a12b")
    # Compute query embedding
    resp = openai_client.embeddings.create(
        model=model_name,
        input=[query]
    )
    query_emb = resp.data[0].embedding
    results = collection.query(
        query_embeddings=[query_emb],
        n_neighbors=top_k,
        include=["documents", "metadatas"]
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    sources = [m.get("source", "") for m in metas]
    return docs, sources

@app.on_event("startup")
async def startup_event():
    load_openai_client()
    init_chromadb()
    load_and_chunk_documents()

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        relevant_chunks, sources = retrieve_relevant_chunks(request.question, request.top_k)
        if not relevant_chunks:
            return QueryResponse(answer="I don't know / that's outside what I can help with.", sources=[])
        context = "\n\n".join(relevant_chunks)
        prompt = f"""You are a helpful assistant for the KNUST E-Learning Centre.
Answer the user's question based only on the following context:

{context}

Question: {request.question}

Answer:"""
        # Use OpenAI client for completion
        if openai_client is None:
            openai_client = load_openai_client()
            if openai_client is None:
                answer = f"Based on the information: {context[:500]}..."
        else:
            model_name = os.getenv("model", "nvidia/nemotron-3-super-120b-a12b")
            resp = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for KNUST E-Learning Centre."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            answer = resp.choices[0].message.content.strip()
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)