"""KNUST RAG-GAR production API for Modal.

Retrieval runs locally in the Modal container and generation uses the
Hugging Face OpenAI-compatible router. The repository's ./data directory is
copied into the image, so serving does not depend on GitHub at request time.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import modal

APP_NAME = "knust-rag-gar"
DATA_DIR = "/root/data"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi>=0.115,<1",
        "httpx>=0.27,<1",
        "scikit-learn>=1.5,<2",
    )
    .add_local_dir("data", DATA_DIR, copy=True)
)

app = modal.App(APP_NAME)


def _load_documents() -> tuple[list[str], list[str]]:
    texts: list[str] = []
    sources: list[str] = []
    root = Path(DATA_DIR)
    if not root.exists():
        return texts, sources

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
        for i, paragraph in enumerate(paragraphs):
            words = paragraph.split()
            if len(words) <= 500:
                texts.append(paragraph)
                sources.append(f"{path.name}#para{i}")
                continue
            for start in range(0, len(words), 400):
                texts.append(" ".join(words[start : start + 500]))
                sources.append(f"{path.name}#chunk{start // 400}")

    return texts, sources


class Retriever:
    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel

        self._linear_kernel = linear_kernel
        self.texts, self.sources = _load_documents()
        if not self.texts:
            self.texts = [
                "KNUST E-Learning Centre AI Assistant. The knowledge base is currently empty."
            ]
            self.sources = ["knowledge-base-empty"]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, top_k: int) -> tuple[list[str], list[str]]:
        top_k = max(1, min(int(top_k), 8))
        query_vector = self.vectorizer.transform([query])
        scores = self._linear_kernel(query_vector, self.matrix).ravel()
        ranked = scores.argsort()[::-1]
        selected = [int(i) for i in ranked[:top_k] if scores[i] > 0]
        return [self.texts[i] for i in selected], [self.sources[i] for i in selected]


retriever: Retriever | None = None


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    max_containers=3,
    timeout=90,
)
@modal.asgi_app()
def web():
    import httpx
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    global retriever
    if retriever is None:
        retriever = Retriever()

    api = FastAPI(title="KNUST RAG-GAR API", version="1.0.0")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    class QueryRequest(BaseModel):
        question: str = Field(min_length=1, max_length=2000)
        top_k: int = Field(default=5, ge=1, le=8)

    class QueryResponse(BaseModel):
        answer: str
        sources: list[str]
        mode: str

    @api.get("/")
    async def root():
        return {"service": APP_NAME, "status": "ok", "docs": "/docs"}

    @api.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": APP_NAME,
            "documents": len(retriever.texts if retriever else []),
            "model": os.getenv("HF_MODEL", DEFAULT_MODEL),
        }

    @api.post("/chat", response_model=QueryResponse)
    @api.post("/query", response_model=QueryResponse)
    async def chat(request: QueryRequest) -> QueryResponse:
        chunks, sources = retriever.search(request.question.strip(), request.top_k)
        if not chunks:
            return QueryResponse(
                answer=(
                    "I couldn't find that in the KNUST knowledge base. "
                    "Please check the official KNUST source for current information."
                ),
                sources=[],
                mode="no-match",
            )

        context = "\n\n---\n\n".join(chunks)
        token = os.environ.get("HF_TOKEN")
        if not token:
            return QueryResponse(answer=context, sources=sources, mode="retrieval-only")

        model = os.getenv("HF_MODEL", DEFAULT_MODEL)
        prompt = (
            "You are the KNUST E-Learning Centre AI Assistant. Answer the user's "
            "question using ONLY the supplied KNUST knowledge. Do not invent dates, "
            "fees, admission requirements, policies, locations, contact details, or "
            "links. If the context does not contain the answer, clearly say that the "
            "information is not available in the knowledge base. Keep the answer "
            "concise and useful.\n\n"
            f"KNOWLEDGE:\n{context}\n\nQUESTION:\n{request.question.strip()}\n\nANSWER:"
        )

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    "https://router.huggingface.co/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 500,
                    },
                )

            if response.status_code >= 400:
                return QueryResponse(
                    answer=context,
                    sources=sources,
                    mode="retrieval-fallback",
                )

            payload = response.json()
            answer = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not isinstance(answer, str) or not answer.strip():
                return QueryResponse(
                    answer=context,
                    sources=sources,
                    mode="retrieval-fallback",
                )
            return QueryResponse(
                answer=answer.strip(),
                sources=sources,
                mode="hf-inference",
            )
        except Exception:
            return QueryResponse(
                answer=context,
                sources=sources,
                mode="retrieval-fallback",
            )

    return api
