import re
from pathlib import Path
import gradio as gr
import os
from huggingface_hub import InferenceClient

DATA = Path(__file__).parent.parent / "data"

def load_docs():
    docs = []
    for p in sorted(DATA.glob("*.txt")):
        for part in re.split(r"\n\s*\n", p.read_text(encoding="utf-8")):
            if part.strip():
                docs.append(f"[{p.name}]\n{part.strip()}")
    return docs

DOCS = load_docs()

def knust_knowledge_search(query):
    """Search the local KNUST admissions and campus knowledge base."""
    terms = {x.lower() for x in re.findall(r"[A-Za-z0-9']+", query) if len(x) > 2}
    ranked = []
    for doc in DOCS:
        score = sum(t in doc.lower() for t in terms)
        if score:
            ranked.append((score, doc))
    ranked.sort(reverse=True)
    return "\n\n---\n\n".join(x[1] for x in ranked[:5]) or "No matching local information."

def get_hf_client():
    """Get Hugging Face InferenceClient with proper token handling."""
    token = os.getenv("HF_TOKEN")
    model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    if token:
        return InferenceClient(token=token, model=model, provider="auto")
    else:
        # Fallback for HF Space without token (uses default provider routing)
        return InferenceClient(model=model, provider="auto")

def chat(message, history):
    """Handle chat messages with KNUST RAG + HF inference."""
    # Step 1: Retrieve relevant KNUST documents
    context = knust_knowledge_search(message)
    
    # Step 2: Build prompt with context
    prompt = f"""You are the KNUST E-Learning Centre AI Assistant.
Use the provided KNUST knowledge base context to answer questions accurately.
Do not invent university policies, fees, dates, requirements, or URLs.
If the context doesn't answer the question, explicitly say the information is not available.

KNUST Knowledge Context:
{context}

Question: {message}

Answer:"""
    
    try:
        client = get_hf_client()
        response = client.text_generation(prompt, max_new_tokens=500, temperature=0.2)
        return response.strip()
    except Exception as e:
        # Fallback: return the context if HF inference fails
        if context and context != "No matching local information.":
            return f"Based on available information: {context}"
        else:
            return "The AI assistant is temporarily unavailable. Please try again."

demo = gr.ChatInterface(
    fn=chat,
    title="KNUST E-Learning Centre AI Assistant",
    description="Ask about admissions, procedures and campus navigation."
)

if __name__ == "__main__":
    demo.launch()
