from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import json
import numpy as np
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.neighbors import NearestNeighbors
import logging
from loguru import logger
from dotenv import load_dotenv
import hashlib
import time
from functools import lru_cache

load_dotenv(dotenv_path='../.env')  # Load environment variables from .env file

app = FastAPI()

# Security
API_KEY = os.getenv("API_KEY")
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return credentials.credentials

# Add CORS middleware - restrict to specific origins for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Restrict to necessary methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# Embedding cache with TTL
class TTLCache:
    def __init__(self, ttl_seconds=3600):
        self.cache = {}
        self.ttl = ttl_seconds

    def _is_expired(self, timestamp):
        return time.time() - timestamp > self.ttl

    def get(self, key):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if not self._is_expired(timestamp):
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key, value):
        self.cache[key] = (value, time.time())

    def clear(self):
        self.cache.clear()

# Initialize embedding cache (1 hour TTL)
embedding_cache = TTLCache(ttl_seconds=3600)

def load_openai_client():
    # Try NVIDIA variables first
    api_key = os.getenv("NVIDIA_KEY")
    base_url = os.getenv("NVIDIA_KEY_BASE_URL")

    # Fall back to OpenAI variables if NVIDIA not set
    if not api_key or api_key == "dummy":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key or api_key == "dummy":
        logger.warning("No API key found. Using dummy embeddings for testing.")
        return None
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)

def embed_texts(texts: List[str]) -> np.ndarray:
    openai_client = app.state.openai_client
    if openai_client is None:
        # Dummy embedding: return a fixed-size random vector for each text
        # In a real scenario, we would use a proper embedding model.
        logger.warning("Using dummy embeddings (random vectors).")
        return np.random.rand(len(texts), 1536)  # OpenAI ada-002 dimension

    # Determine the embedding model to use
    # For NVIDIA API, use their embedding model; for OpenAPI, use text-embedding-3-small
    base_url = os.getenv("NVIDIA_KEY_BASE_URL", "")
    if "nvidia" in base_url.lower():
        # Use NVIDIA embedding model
        model_name = "nvidia/embed-qa-4"
    else:
        # Use OpenAI embedding model
        model_name = "text-embedding-3-small"

    # Check cache for each text
    embeddings = []
    uncached_texts = []
    uncached_indices = []

    for i, text in enumerate(texts):
        # Create a cache key based on the text and model
        cache_key = hashlib.md5(f"{text}:{model_name}".encode()).hexdigest()
        cached_embedding = embedding_cache.get(cache_key)
        if cached_embedding is not None:
            embeddings.append(cached_embedding)
        else:
            embeddings.append(None)  # Placeholder
            uncached_texts.append(text)
            uncached_indices.append(i)

    # If we have uncached texts, compute embeddings for them
    if uncached_texts:
        logger.info(f"Computing embeddings for {len(uncached_texts)} uncached texts using model {model_name}")
        response = openai_client.embeddings.create(
            model=model_name,
            input=uncached_texts
        )
        new_embeddings = [item.embedding for item in response.data]

        # Store in cache and fill in the embeddings array
        for idx, embedding in zip(uncached_indices, new_embeddings):
            cache_key = hashlib.md5(f"{texts[idx]}:{model_name}".encode()).hexdigest()
            embedding_cache.set(cache_key, embedding)
            embeddings[idx] = embedding

    return np.array(embeddings)

def load_and_chunk_documents(data_dir: str = "../data"):
    """Load text files, split into chunks, and compute embeddings."""
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
    logger.info(f"Computing embeddings for {len(chunks)} chunks...")
    embeddings = embed_texts(chunks)
    # Normalize embeddings to unit length for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)  # avoid division by zero
    embeddings_norm = embeddings / norms
    # Fit NearestNeighbors model
    logger.info("Fitting NearestNeighbors model...")
    nn_model = NearestNeighbors(n_neighbors=min(5, len(chunks)), metric='cosine')
    nn_model.fit(embeddings_norm)
    # Store in app state
    app.state.embeddings = embeddings
    app.state.embeddings_norm = embeddings_norm
    app.state.chunks = chunks
    app.state.chunk_sources = chunk_sources
    app.state.nn_model = nn_model
    print("Embeddings computed and model fitted.")

def retrieve_relevant_chunks(query: str, top_k: int = 3):
    if not hasattr(app.state, 'embeddings_norm') or app.state.embeddings_norm is None or \
       not hasattr(app.state, 'chunks') or app.state.chunks is None or \
       not hasattr(app.state, 'nn_model') or app.state.nn_model is None:
        return [], []
    query_embedding = embed_texts([query])[0]
    # Normalize query embedding
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        query_norm = 1e-10
    query_embedding_norm = query_embedding / query_norm
    # Search for nearest neighbors
    distances, indices = app.state.nn_model.kneighbors([query_embedding_norm], n_neighbors=top_k)
    # Return the chunks and sources for the indices
    return [app.state.chunks[i] for i in indices[0]], [app.state.chunk_sources[i] for i in indices[0]]

@app.on_event("startup")
async def startup_event():
    # Initialize OpenAI client in app state
    app.state.openai_client = load_openai_client()
    load_and_chunk_documents()

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, api_key: str = Depends(verify_api_key)):
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
        openai_client = app.state.openai_client
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
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)