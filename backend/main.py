from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import json
import numpy as np
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.neighbors import NearestNeighbors

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# Global variables for the vector store
embeddings = None          # Raw embeddings (before normalization)
embeddings_norm = None     # Normalized embeddings (unit length)
chunks = None
chunk_sources = None
nn_model = None            # NearestNeighbors model
openai_client = None

def load_openai_client():
    global openai_client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "dummy":
        print("Warning: OPENAI_API_KEY not set. Using dummy embeddings for testing.")
        return None
    return OpenAI(api_key=api_key)

def embed_texts(texts: List[str]) -> np.ndarray:
    global openai_client
    if openai_client is None:
        # Dummy embedding: return a fixed-size random vector for each text
        # In a real scenario, we would use a proper embedding model.
        print("Using dummy embeddings (random vectors).")
        return np.random.rand(len(texts), 1536)  # OpenAI ada-002 dimension
    # Use OpenAI's embedding model
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)

def load_and_chunk_documents(data_dir: str = "../data"):
    """Load text files, split into chunks, and compute embeddings."""
    global embeddings, embeddings_norm, chunks, chunk_sources, nn_model
    chunks = []
    chunk_sources = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt") or filename.endswith(".md"):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple chunking by paragraphs
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                for i, para in enumerate(paragraphs):
                    chunks.append(para)
                    chunk_sources.append(f"{filename}#para{i}")
    if not chunks:
        # If no data, create a dummy chunk
        chunks = ["Welcome to KNUST E-Learning Centre. How can I help you?"]
        chunk_sources = ["dummy"]
    # Compute embeddings
    print(f"Computing embeddings for {len(chunks)} chunks...")
    embeddings = embed_texts(chunks)
    # Normalize embeddings to unit length for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)  # avoid division by zero
    embeddings_norm = embeddings / norms
    # Fit NearestNeighbors model
    print("Fitting NearestNeighbors model...")
    nn_model = NearestNeighbors(n_neighbors=min(5, len(chunks)), metric='cosine')
    nn_model.fit(embeddings_norm)
    print("Embeddings computed and model fitted.")

def retrieve_relevant_chunks(query: str, top_k: int = 3):
    global embeddings_norm, chunks, chunk_sources, nn_model
    if embeddings_norm is None or chunks is None or nn_model is None:
        return [], []
    query_embedding = embed_texts([query])[0]
    # Normalize query embedding
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        query_norm = 1e-10
    query_embedding_norm = query_embedding / query_norm
    # Search for nearest neighbors
    distances, indices = nn_model.kneighbors([query_embedding_norm], n_neighbors=top_k)
    # Return the chunks and sources for the indices
    return [chunks[i] for i in indices[0]], [chunk_sources[i] for i in indices[0]]

@app.on_event("startup")
async def startup_event():
    load_and_chunk_documents()

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        relevant_chunks, sources = retrieve_relevant_chunks(request.question, request.top_k)
        if not relevant_chunks:
            return QueryResponse(answer="I don't know / that's outside what I can help with.", sources=[])
        # Construct prompt
        context = "\n\n".join(relevant_chunks)
        prompt = f"""You are a helpful assistant for the KNUST E-Learning Centre.
Answer the user's question based only on the following context:

{context}

Question: {request.question}

Answer:"""
        # Use OpenAI to generate answer
        global openai_client
        if openai_client is None:
            openai_client = load_openai_client()
        if openai_client is None:
            # Fallback: just return the context
            answer = f"Based on the information: {context[:500]}..."
        else:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for KNUST E-Learning Centre."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            answer = response.choices[0].message.content.strip()
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)