from pathlib import Path
import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

app = FastAPI(title="KNUST RAG-GAR API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

def load_chunks():
    chunks = []
    for path in sorted(DATA.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for i, part in enumerate(re.split(r"\n\s*\n", text)):
            part = part.strip()
            if part:
                chunks.append((path.name, i, part))
    return chunks

CHUNKS = load_chunks()

def retrieve(query, top_k=5):
    terms = {x.lower() for x in re.findall(r"[A-Za-z0-9']+", query) if len(x) > 2}
    scored = []
    for source, idx, text in CHUNKS:
        low = text.lower()
        score = sum(1 for term in terms if term in low)
        if score:
            scored.append((score, source, idx, text))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:max(1, min(top_k, 10))]

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "knust-rag-gar", "chunks": len(CHUNKS)}

@app.post("/api/chat")
def chat(request: QueryRequest):
    matches = retrieve(request.question, request.top_k)
    context = "\n\n---\n\n".join(x[3] for x in matches)
    sources = [f"{x[1]}#para{x[2]}" for x in matches]

    if not context:
        return {
            "answer": "I couldn't find that in the KNUST knowledge base. Please check the official KNUST source for current information.",
            "sources": [],
        }

    token = os.getenv("HF_TOKEN")
    if not token:
        return {"answer": context, "sources": sources, "mode": "retrieval-only"}

    client = InferenceClient(token=token, model=os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    prompt = f"""You are the KNUST E-Learning Centre AI Assistant.
Answer using the supplied KNUST knowledge only. Do not invent dates, fees, requirements, policies or links.
If the context does not answer the question, say that the information is not available.

KNOWLEDGE:
{context}

QUESTION:
{request.question}

ANSWER:"""
    try:
        response = client.text_generation(prompt, max_new_tokens=500, temperature=0.2)
        return {"answer": response.strip(), "sources": sources, "mode": "hf-inference"}
    except Exception as exc:
        return {"answer": context, "sources": sources, "mode": "retrieval-fallback", "error": type(exc).__name__}
